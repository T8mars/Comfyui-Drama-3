"""Native prompt/source-map regressions for opt-in creative long-line views."""
from dataclasses import replace

import pytest
import torch

from drama3_t8.engine import Sampling
from drama3_t8.state import Reference, Session, Take, save_session, load_session
from drama3_t8.dialogue import audio_frames
from drama3_t8.scene_chunks import JOIN, chunked_scene, source_history, validate_scene
from test_scene_chunks import Harness


TEXT = "A calm actor walks through the office. " * 6


def turn(marker, actor=0, scene=False):
    text = f"<|speaker:{actor}|> Actual prior acoustic turn {marker}."
    return Take(text, text, audio_frames(torch.full((10, 3), marker, dtype=torch.long)),
                metadata={"task": "scene", "script_lines": [text]} if scene else {})


@pytest.fixture
def prompt_spy(monkeypatch):
    from drama3_t8 import prompts
    calls, original = [], prompts.rows_from_segments
    def rows(tokenizer, config, segments):
        actual = original(tokenizer, config, segments)
        calls.append({"prompt": actual.clone(),
            "history": [part.values.clone() for part in segments if isinstance(part, prompts.FrameRows)],
            "references": [part.clone() for part in segments if torch.is_tensor(part)],
            "strings": [part for part in segments if isinstance(part, str)]})
        return actual
    monkeypatch.setattr(prompts, "rows_from_segments", rows)
    return calls


def run(engine, parent=None, *, actor=0, mode="together", **options):
    return chunked_scene(engine, [f"<|speaker:{actor}|> {TEXT}"], "", "", None,
        Sampling(seed=102, max_frames=80), None, parent, "", None, mode, 80, **options)


def history_markers(call):
    return [int(frame[0, 1]) for frame in call["history"]]


def test_first_creative_prompt_keeps_all_three_actual_history_turns_then_bounds_continuations(prompt_spy, tmp_path):
    previous = (turn(111), turn(222), turn(333, actor=1))
    parent = Session(takes=previous)
    engine = Harness()
    result = run(engine, parent, history_limit=3)
    assert len(prompt_spy) > 3
    assert history_markers(prompt_spy[0]) == [111, 222, 333]
    assert all(torch.equal(take.frames, actual) for take, actual in zip(previous, prompt_spy[0]["history"]))
    assert history_markers(prompt_spy[1]) == [111, 10]
    assert all(len(call["history"]) <= 2 for call in prompt_spy[1:])
    assert result.takes[-1].metadata["scene_chunking"]["history_limit"] == 3
    path = tmp_path / "three-history.safetensors"
    save_session(result, path)
    restored = load_session(path)
    assert restored.takes[-1].metadata == result.takes[-1].metadata
    assert len(validate_scene(restored.takes[-1])) == len(engine.requests)


def test_explicit_default_options_match_the_unchanged_existing_scene_path(prompt_spy):
    parent = Session(takes=(turn(111), turn(222), turn(333, actor=1)))
    old, explicit = Harness(), Harness()
    legacy = old.scene([f"<|speaker:0|> {TEXT}"], session=parent,
        sampling=Sampling(seed=102, max_frames=80), split_long_text=True, chunk_bytes=80)
    assert history_markers(prompt_spy[0]) == [111, 333]
    current = run(explicit, parent, history_limit=2, fixed_reference_anchor=False)
    assert len(old.requests) == len(explicit.requests)
    assert all(torch.equal(a["prompt"], b["prompt"]) for a, b in zip(old.requests, explicit.requests))
    assert torch.equal(legacy.takes[-1].frames, current.takes[-1].frames)
    assert current.takes[-1].metadata["scene_chunking"] == {"version": 3, "enabled": True, "chunk_bytes": 80}


@pytest.mark.parametrize("explicit", [False, True])
def test_fixed_reference_anchor_uses_exact_actor_and_never_rolls_history(prompt_spy, tmp_path, explicit):
    actor = 7
    refs = (Reference("<|speaker:7|> selected voice", torch.full((10, 2), 77, dtype=torch.long)),
             Reference("<|speaker:3|> unrelated voice", torch.full((10, 2), 33, dtype=torch.long))) if explicit else ()
    parent = Session(refs, (turn(17, actor, scene=True),), "User direction")
    engine = Harness()
    result = run(engine, parent, actor=actor, mode="per_line", history_limit=3, fixed_reference_anchor=True)
    take = result.takes[-1]
    assert history_markers(prompt_spy[0]) == [17]
    assert all(not call["history"] for call in prompt_spy[1:])
    expected_marker = 77 if explicit else 10
    for call in prompt_spy[1:]:
        assert len(call["references"]) == 1
        assert torch.all(call["references"][0] == expected_marker)
        assert "<|speaker:0|>" in next(text for text in call["strings"] if text.startswith("\n\n[Reference 1]"))
    if explicit:
        assert torch.all(prompt_spy[0]["references"][0] == 77)
        assert all(33 not in call["prompt"][:, 1].tolist() for call in prompt_spy)
    first_source = take.metadata["scene_sources"][0]["take_id"]
    expected_id = None if explicit else first_source
    assert take.metadata["long_context"]["source_take_id"] == expected_id
    assert take.metadata["long_context"]["policy"] == ("explicit_references" if explicit else "first_new_take_reference")
    assert all(raw["metadata"]["long_context"]["source_take_id"] == expected_id for raw in take.metadata["scene_sources"][1:])
    assert take.metadata["scene_chunking"]["fixed_reference_anchor"] is True
    assert all(source.prompt_text.startswith("<|speaker:7|>") for source in source_history(take))
    assert all(source.prompt_text.startswith("<|speaker:0|>") for source in source_history(take, actor))
    path = tmp_path / "fixed-reference.safetensors"
    save_session(result, path)
    restored = load_session(path)
    first_audio, first_timing = engine.decode_take(take, with_timeline=True)
    engine.decoded.clear()
    actual, timing = engine.decode_take(restored.takes[-1], with_timeline=True)
    assert torch.equal(actual["waveform"], first_audio["waveform"]) and timing == first_timing
    assert len(engine.decoded) == len(take.metadata["scene_sources"])
    assert timing == [{"line": 0, "speaker": actor, "text": TEXT.strip(), "start": 0.0,
                       "end": take.codes.shape[-1] * 2048 / 44100}]
    assert actual["waveform"].shape[-1] == take.codes.shape[-1] * 2048
    assert result.references == parent.references and result.takes[:-1] == parent.takes


@pytest.mark.parametrize("options", [{"history_limit": True}, {"history_limit": 1}, {"history_limit": 4},
    {"fixed_reference_anchor": 1}, {"fixed_reference_anchor": True}])
def test_creative_chunk_options_reject_unsupported_views_before_inference(options):
    engine = Harness()
    with pytest.raises(ValueError):
        run(engine, **options)
    assert not engine.requests and not engine.events


def test_fixed_reference_cannot_relabel_multiple_actors_as_one_voice():
    engine = Harness()
    with pytest.raises(ValueError, match="one actor"):
        chunked_scene(engine, ["<|speaker:7|> first", "<|speaker:3|> second"], "", "", None,
            Sampling(max_frames=80), None, None, "", None, "per_line", 80, fixed_reference_anchor=True)
    assert not engine.requests and not engine.events


class NativeSourceEmitter(Harness):
    """Emit the fixture generator's actual source frames before Take return."""
    def __init__(self):
        super().__init__()
        self.observed_events, self.source_pending = [], False

    def _take(self, prompt, request, transcript, sampling, on_codes=None, progress=None, **kwargs):
        index = len(self.requests)
        self.source_pending = True
        self.observed_events.append(("source_start", index))
        # These are precisely Harness's source codebooks, emitted at their
        # generator boundary rather than reconstructed from the final composite.
        for segment, _ in enumerate(request.split(JOIN)):
            actual = torch.full((10, 2 + segment), 10 * (index + 1) + segment, dtype=torch.long)
            for frame in actual.T:
                self.observed_events.append(("source_frame", index))
                if on_codes is not None:
                    on_codes(frame)
        take = super()._take(prompt, request, transcript, sampling, on_codes=None, progress=progress, **kwargs)
        self.observed_events.append(("source_return", index))
        self.source_pending = False
        return take


@pytest.mark.parametrize("mode", ["together", "per_line"])
@pytest.mark.parametrize("preview_kind", ["live", "bare"])
def test_measurement_observer_sees_real_source_frames_early_without_composite_double_emission(mode, preview_kind):
    from drama3_t8.performance import PerformanceRecorder
    engine, observed, bare_frames = NativeSourceEmitter(), [], []
    recorder = PerformanceRecorder(probe=lambda: {}, sample_interval=0).start()
    def observer(frame):
        assert engine.source_pending
        assert not any(event[0] == "preview" for event in engine.observed_events)
        engine.observed_events.append(("observed", len(engine.requests)))
        observed.append(frame.clone())
        recorder.mark_first_codes()
    class Preview:
        def __init__(self): self.takes = []
        def append(self, frame): pytest.fail("legacy LiveAudio received independent source frames")
        def append_scene(self, take):
            self.takes.append(take)
            engine.observed_events.append(("preview", len(engine.requests)))
    preview = Preview()
    result = chunked_scene(engine, ["<|speaker:7|> " + TEXT], "", "", None, Sampling(max_frames=80),
        None, None, "", preview.append if preview_kind == "live" else bare_frames.append, mode, 80,
        on_observed_codes=observer)
    take = result.takes[-1]
    sources = validate_scene(take)
    assert len(engine.requests) > 1 and len(observed) == sum(source.codes.shape[-1] for source in sources)
    assert torch.equal(torch.stack(observed, dim=1), torch.cat([source.codes for source in sources], dim=1))
    assert engine.observed_events[:3] == [("source_start", 0), ("source_frame", 0), ("observed", 0)]
    assert engine.observed_events.index(("observed", 0)) < engine.observed_events.index(("source_return", 0))
    assert not bare_frames
    assert preview.takes == ([take] if preview_kind == "live" else [])
    before_decode = recorder.report()
    assert before_decode["first_codes_seconds"] is not None and before_decode["first_pcm_seconds"] is None
    audio = engine.decode_take(take)
    recorder.mark_first_pcm(audio["waveform"].shape[-1], audio["sample_rate"])
    report = recorder.finish()
    assert report["first_codes_seconds"] <= report["first_pcm_seconds"] <= report["elapsed_seconds"]


def test_source_observer_failure_propagates_without_publishing_finished_composite():
    engine, calls = NativeSourceEmitter(), []
    def observer(frame):
        calls.append(frame.clone())
        raise RuntimeError("observer cancellation")
    with pytest.raises(RuntimeError, match="observer cancellation"):
        run(engine, on_observed_codes=observer)
    assert len(calls) == 1
    assert engine.observed_events == [("source_start", 0), ("source_frame", 0)]


def test_invalid_measurement_observer_is_rejected_before_inference():
    engine = Harness()
    with pytest.raises(ValueError, match="on_observed_codes"):
        run(engine, on_observed_codes=True)
    assert not engine.requests and not engine.events
