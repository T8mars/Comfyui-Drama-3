"""Real CPU prompt-row regression for isolated creative acoustic context."""
import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8 import creative_context as cc
from drama3_t8.dialogue import audio_frames, isolated_scene, recast
from drama3_t8.engine import Sampling
from drama3_t8.prompts import FrameRows, history_segments, spoken_text, tts_prompt, streaming_prompt
from drama3_t8.state import Reference, Session, Take, load_session, save_session


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode("utf-8")))


class PromptSpy:
    """Observe actual native prompt audio rows without invoking a model."""
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(tower=SimpleNamespace(num_codebooks=10, vocab_size=4096),
                                      semantic_start_token_id=248099)
        self.tokenizer = Tokenizer()

    def speak(self, text, session, sampling=None, **kwargs):
        history = [part.values.clone() for part in history_segments(session.takes) if isinstance(part, FrameRows)]
        prompt = tts_prompt(self.tokenizer, self.config, text, session.references, session.takes, session.direction)
        self.calls.append({"text": text, "session": session, "history": history, "prompt": prompt})
        take = Take(text, text, audio_frames(torch.full((10, 3), 900, dtype=torch.long)),
                    metadata={"seed": getattr(sampling, "seed", 42), "length_limit_reached": False})
        return session.append(take)


def raw(text, marker, **metadata):
    return Take(text, text, audio_frames(torch.full((10, 3), marker, dtype=torch.long)), metadata=metadata)


def scene(lines, markers, **metadata):
    rows = []
    for index, marker in enumerate(markers):
        if index:
            separator = torch.zeros((1, 11), dtype=torch.long)
            separator[0, 0] = 252196
            rows.append(separator)
        rows.append(audio_frames(torch.full((10, 3), marker, dtype=torch.long)))
    return Take("\n<|audio_separator|>\n".join(lines), "\n".join(lines), torch.cat(rows),
                metadata={"task": "scene", "script_lines": lines, **metadata})


def descriptor(tmp_path, name, role, actor, take, *, lines=None, cast=None):
    path = tmp_path / f"{name}.safetensors"
    save_session(Session(takes=(take,)), path)
    data = path.read_bytes()
    return {"candidate_id": name, "status": "completed", "session_path": str(path),
            "input_snapshot": {"lines": lines or [{"line_id": name, "role": role, "readout_text": spoken_text(take.prompt_text)}],
                               "cast": cast or {role: {"speaker": actor}}},
            "artifacts": [{"kind": "session", "path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}


def job(role="B", actor=3, contexts=(), mode="per_line_context"):
    line = {"line_id": "current", "role": role, "readout_text": "现在轮到我了"}
    return {"mode": mode, "role": role, "line": line, "lines": [line], "cast": {role: {"speaker": actor}},
            "settings": {"scene": "office", "characters": "A is curious; B is calm"},
            "context_candidates": list(contexts), "speaker_map": {"A": 7, "B": 3, "C": 9},
            "lookahead_lines": []}


def markers(spy):
    return [int(frames[0, 1]) for frames in spy.calls[-1]["history"]]


def test_old_single_line_scene_drops_other_actor_minimal_reproduction(tmp_path):
    previous = scene(["<|speaker:7|> A said the previous line"], [11])
    refs = (Reference("<|speaker:3|> B voice", torch.full((10, 2), 2, dtype=torch.long)),)
    archived = Session(refs, (previous,))
    legacy = PromptSpy()
    isolated_scene(legacy, ["<|speaker:3|> B answers"], "", "", None, Sampling(seed=42),
                   None, archived, "", None, "per_line_context")
    # The old per-call last_source is None: previous A is absent from B's prompt.
    assert legacy.calls[-1]["history"] == []
    assert not (legacy.calls[-1]["prompt"][:, 1:] == 11).any()

    candidate = descriptor(tmp_path, "A-old", "A", 7, previous,
                           lines=[{"role": "A", "readout_text": "A said the previous line"}])
    conditioned, metadata = cc.condition_creative_context(job(contexts=[candidate]), Session(refs))
    fixed = PromptSpy()
    fixed.speak(recast("B answers", 0), conditioned, Sampling())
    assert markers(fixed) == [11]
    assert (fixed.calls[-1]["prompt"][:, 1:] == 11).all(dim=1).sum() == 3
    assert conditioned.takes[0].prompt_text.startswith("<|speaker:1|>")
    assert conditioned.references[0].transcript.startswith("<|speaker:0|>")
    assert metadata["creative_context"]["history"][0]["source_take_id"] == previous.take_id


def test_own_first_latest_and_last_other_really_enter_native_history(tmp_path, monkeypatch):
    first = descriptor(tmp_path, "first-A", "A", 7, raw("<|speaker:0|> A first", 11, creative_global_speaker=7))
    skipped = descriptor(tmp_path, "old-B", "B", 3, raw("<|speaker:0|> old B", 22, creative_global_speaker=3))
    latest = descriptor(tmp_path, "latest-A", "A", 7, raw("<|speaker:0|> A latest", 33, creative_global_speaker=7))
    previous = descriptor(tmp_path, "last-C", "C", 9, raw("<|speaker:0|> C previous", 44, creative_global_speaker=9))
    loaded = []
    original = cc._load_session
    def observed(path):
        loaded.append(str(path))
        return original(path)
    monkeypatch.setattr(cc, "_load_session", observed)
    refs = (Reference("<|speaker:7|> A voice", torch.full((10, 2), 5, dtype=torch.long)),
            Reference("<|speaker:3|> B voice", torch.full((10, 2), 6, dtype=torch.long)))
    conditioned, metadata = cc.condition_creative_context(job("A", 7, [first, skipped, latest, previous]), Session(refs, direction="comedy"))
    spy = PromptSpy()
    spy.speak("<|speaker:0|> A reply", conditioned)
    assert markers(spy) == [11, 33, 44]
    assert [take.prompt_text.split()[0] for take in conditioned.takes] == ["<|speaker:0|>", "<|speaker:0|>", "<|speaker:1|>"]
    assert len(loaded) == 3 and skipped["session_path"] not in loaded
    assert len(conditioned.references) == 1 and conditioned.references[0].codes[0, 0] == 5
    assert conditioned.direction == "comedy"
    assert metadata["creative_context"]["direction_scope"] == "utterance_v1"
    assert metadata["creative_global_speaker"] == 7 and metadata["creative_local_speaker"] == 0
    assert metadata["creative_context"]["candidate_ids"] == ["first-A", "latest-A", "last-C"]
    assert load_session(first["session_path"]).takes[-1].prompt_text.startswith("<|speaker:0|>")


def test_isolated_tts_keeps_explicit_delivery_and_scene_settings_separate():
    request = job()
    original = copy.deepcopy(request)
    reference = Reference("<|speaker:3|> 一段参考。", torch.ones((10, 3), dtype=torch.long))
    delivery = "轻声、自然，保留句尾的笑意。"
    conditioned, metadata = cc.condition_creative_context(request, Session((reference,), direction=delivery))
    assert conditioned.direction == delivery
    assert conditioned.references[0].transcript == "<|speaker:0|> 一段参考。"
    assert metadata["creative_global_speaker"] == 3 and metadata["creative_local_speaker"] == 0
    assert request == original
    assert request["settings"] == {"scene": "office", "characters": "A is curious; B is calm"}
    # With no explicit delivery, do not manufacture instructions or speaker tags
    # that the user never entered. The actual cast remains in typed audio rows.
    undirected, _ = cc.condition_creative_context(request, Session((reference,)))
    assert undirected.direction == ""


def test_same_previous_actor_does_not_borrow_an_older_foreign_turn(tmp_path):
    a = descriptor(tmp_path, "A", "A", 7, raw("<|speaker:7|> A", 11))
    b = descriptor(tmp_path, "B", "B", 3, raw("<|speaker:3|> B", 22))
    last = descriptor(tmp_path, "A-last", "A", 7, raw("<|speaker:7|> A last", 33))
    conditioned, _ = cc.condition_creative_context(job("A", 7, [a, b, last]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> again", conditioned)
    assert markers(spy) == [11, 33]


def test_mixed_scene_extracts_actual_actor_pieces_instead_of_relabeling_mix(tmp_path):
    previous = scene(["<|speaker:7|> A first", "<|speaker:3|> B last"], [11, 22],
                     scene_speaker_map={"A": 7, "B": 3})
    candidate = descriptor(tmp_path, "scene", "A", 7, previous,
        lines=[{"role": "A", "readout_text": "A first"}, {"role": "B", "readout_text": "B last"}],
        cast={"A": {"speaker": 7}, "B": {"speaker": 3}})
    conditioned, metadata = cc.condition_creative_context(job("A", 7, [candidate]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> A replies", conditioned)
    assert markers(spy) == [11, 22]
    assert all(frames.shape[0] == 3 for frames in spy.calls[-1]["history"])
    assert all(item["source"] == "scene_source" for item in metadata["creative_context"]["history"])
    assert [item["local_speaker"] for item in metadata["creative_context"]["history"]] == [0, 1]


def test_isolated_scene_source_extracts_real_archived_turn(tmp_path):
    engine = PromptSpy()
    previous = isolated_scene(engine, ["<|speaker:7|> A", "<|speaker:3|> B"], "", "", None,
                              Sampling(), None, Session(), "", None, "per_line").takes[-1]
    candidate = descriptor(tmp_path, "isolated", "A", 7, previous,
        lines=[{"role": "A", "readout_text": "A"}, {"role": "B", "readout_text": "B"}],
        cast={"A": {"speaker": 7}, "B": {"speaker": 3}})
    conditioned, _ = cc.condition_creative_context(job("A", 7, [candidate]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> next A", conditioned)
    assert markers(spy) == [900, 900]
    assert len(conditioned.takes) == 2 and all(take.frames.shape[0] == 3 for take in conditioned.takes)


def test_conditioning_session_is_preferred_and_its_real_frames_used(tmp_path, monkeypatch):
    original = raw("<|speaker:7|> old", 11)
    candidate = descriptor(tmp_path, "edited", "A", 7, original)
    actual = raw("<|speaker:7|> actual edited PCM", 77, history_actual_pcm=True, task="conditioning")
    path = tmp_path / "conditioning.safetensors"
    save_session(Session(takes=(actual,)), path)
    candidate["conditioning_session_path"] = str(path)
    candidate["artifacts"].append({"kind": "conditioning_session", "bytes": path.stat().st_size,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    loaded = []
    original_load = cc._load_session
    monkeypatch.setattr(cc, "_load_session", lambda path: (loaded.append(str(path)), original_load(path))[1])
    conditioned, metadata = cc.condition_creative_context(job(contexts=[candidate]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> reply", conditioned)
    assert markers(spy) == [77] and loaded == [str(path)]
    assert metadata["creative_context"]["history"][0]["source"] == "conditioning"
    assert metadata["creative_context"]["history"][0]["source_take_id"] == actual.take_id


def test_context_checksum_change_is_refused_before_prompt_building(tmp_path):
    candidate = descriptor(tmp_path, "A", "A", 7, raw("<|speaker:7|> A", 11))
    path = tmp_path / "A.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        cc.condition_creative_context(job(contexts=[candidate]), Session())


def test_long_form_uses_first_anchor_and_explicit_reference_wins(tmp_path, monkeypatch):
    first = descriptor(tmp_path, "first", "A", 7, raw("<|speaker:7|> first", 11))
    later = descriptor(tmp_path, "later", "A", 7, raw("<|speaker:7|> later", 33))
    request = job("A", 7, [first, later], mode="long_form")
    anchored, metadata = cc.condition_creative_context(request, Session())
    assert anchored.takes == () and torch.equal(anchored.references[0].codes, load_session(first["session_path"]).takes[-1].codes)
    assert metadata["long_context"]["policy"] == "first_new_take_reference"
    reference = Reference("<|speaker:7|> fixed", torch.full((10, 2), 5, dtype=torch.long))
    monkeypatch.setattr(cc, "_load_session", lambda path: pytest.fail("Explicit references must not load an anchor"))
    fixed, metadata = cc.condition_creative_context(request, Session((reference,)))
    assert fixed.references == (reference,) and fixed.takes == () and metadata["creative_context"]["candidate_ids"] == []


def test_text_stream_bounds_real_history_and_builds_global_lookahead(tmp_path):
    first = descriptor(tmp_path, "first", "A", 7, raw("<|speaker:7|> first", 11, task="text_stream", lookahead="<|speaker:3|> then B"))
    middle = descriptor(tmp_path, "middle", "B", 3, raw("<|speaker:3|> middle", 22))
    latest = descriptor(tmp_path, "latest", "C", 9, raw("<|speaker:9|> latest", 33))
    request = job("B", 3, [first, middle, latest], mode="text_stream")
    request["lookahead_lines"] = [{"role": "C", "readout_text": "下一句不是旁白"}]
    conditioned, metadata = cc.condition_creative_context(request, Session())
    assert [int(take.codes[0, 0]) for take in conditioned.takes] == [11, 33]
    lookahead = metadata["creative_context"]["lookahead_text"]
    assert lookahead == "<|speaker:9|> 下一句不是旁白"
    spy = PromptSpy()
    prompt = streaming_prompt(spy.tokenizer, spy.config, "<|speaker:3|> current", lookahead,
                              conditioned.references, conditioned.takes, conditioned.direction)
    assert (prompt[:, 1:] == 11).all(dim=1).sum() == 3
    assert (prompt[:, 1:] == 33).all(dim=1).sum() == 3
    assert not (prompt[:, 1:] == 22).any()
    assert conditioned.takes[0].metadata["lookahead"] == "<|speaker:3|> then B"


def test_unknown_future_actor_is_not_assigned_by_guessing():
    request = job("A", 7, mode="text_stream")
    request.pop("speaker_map")
    request["lookahead_lines"] = [{"role": "Future", "readout_text": "future"}]
    with pytest.raises(ValueError, match="Lookahead"):
        cc.condition_creative_context(request, Session())


def test_assign_speakers_preserves_explicit_and_aliases_without_mutating():
    cast = {"A": {"speaker": 7, "voice_id": "voice"}, "Narrator": {"speaker": 7}}
    before = copy.deepcopy(cast)
    assigned = cc.assign_speakers(cast, [{"role": "B", "text": "B"}, {"role": "A", "text": "A"}, {"role": "C", "text": "C"}])
    assert cast == before and assigned["A"]["speaker"] == assigned["Narrator"]["speaker"] == 7
    assert assigned["B"]["speaker"] == 0 and assigned["C"]["speaker"] == 1
    reordered = cc.assign_speakers(assigned, [{"role": "C", "text": "again"}, {"role": "B", "text": "again"}])
    assert reordered == assigned


def test_assign_speakers_supports_string_lines_and_explicit_script_ids():
    assert cc.assign_speakers({}, ["hello", "again"]) == {"旁白": {"speaker": 0}}
    assigned = cc.assign_speakers({}, [{"role": "A", "text": "auto"}, {"role": "B", "text": "<|speaker:0|> explicit"}])
    assert assigned["B"]["speaker"] == 0 and assigned["A"]["speaker"] == 1
    with pytest.raises(ValueError, match="conflicts"):
        cc.assign_speakers({"A": {"speaker": 7}}, [{"role": "A", "text": "<|speaker:3|> no"}])


@pytest.mark.parametrize("value", [True, "3", -1, 16])
def test_assign_rejects_invalid_explicit_ids(value):
    with pytest.raises(ValueError):
        cc.assign_speakers({"A": {"speaker": value}}, [{"role": "A", "text": "a"}])


@pytest.mark.parametrize("text", ["<|speaker:01|> a", "<|speaker:16|> a", "<|speaker:x|> a", "<|speaker:0|> A <|speaker:1|> B"])
def test_assign_rejects_ambiguous_or_invalid_tags(text):
    with pytest.raises(ValueError):
        cc.assign_speakers({}, [text])


def test_more_than_sixteen_unaliased_roles_is_rejected():
    with pytest.raises(ValueError, match="16"):
        cc.assign_speakers({}, [{"role": f"role-{index}", "text": "a"} for index in range(17)])


def test_raw_scene_uses_recorded_line_order_when_source_ids_are_local(tmp_path):
    previous = scene(["<|speaker:0|> A first", "<|speaker:1|> B last"], [11, 22])
    candidate = descriptor(tmp_path, "local-scene", "A", 7, previous,
        lines=[{"role": "A", "readout_text": "A first"}, {"role": "B", "readout_text": "B last"}],
        cast={"A": {"speaker": 7}, "B": {"speaker": 3}})
    conditioned, _ = cc.condition_creative_context(job("A", 7, [candidate]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> A next", conditioned)
    assert markers(spy) == [11, 22]
    assert conditioned.takes[0].prompt_text.startswith("<|speaker:0|>")
    assert conditioned.takes[1].prompt_text.startswith("<|speaker:1|>")


def test_actual_single_line_pcm_edit_requires_and_uses_reencoded_conditioning(tmp_path):
    from test_source_aware_infill import Engine
    from drama3_t8 import render_infill as ri
    class SingleEngine(Engine):
        def decode_take(self, take, *, with_timeline=False):
            if "render_infill" in take.metadata:
                return super().decode_take(take, with_timeline=with_timeline)
            audio = {"waveform": self.original[None].clone(), "sample_rate": 44100}
            timeline = [{"start": 0., "end": self.original.shape[-1] / 44100, "speaker": 7, "text": "before", "line": 0}]
            return (audio, timeline) if with_timeline else audio
    engine = SingleEngine()
    parent = raw("<|speaker:7|> before", 17, task="tts")
    rendered = ri.infill_take(engine, parent, "<|speaker:7|> actual complete modified line",
        9000/44100, 13000/44100, Sampling(seed=42, max_frames=50), target_speaker=7, playback_subtype="PCM_16")
    candidate = descriptor(tmp_path, "pcm-edited", "A", 7, rendered,
        lines=[{"role": "A", "readout_text": "actual complete modified line"}])
    with pytest.raises(ValueError, match="actual conditioning"):
        cc.condition_creative_context(job(contexts=[candidate]), Session())
    actual_pcm = ri.decode_rendered(rendered)["waveform"][0]
    actual_codes = ri._encode(engine, actual_pcm)
    conditioning = Take("<|speaker:7|> actual complete modified line", "actual complete modified line",
        audio_frames(actual_codes), metadata={"task": "conditioning", "history_actual_pcm": True, "creative_global_speaker": 7})
    path = tmp_path / "edited-conditioning.safetensors"
    save_session(Session(takes=(conditioning,)), path)
    candidate["conditioning_session_path"] = str(path)
    candidate["artifacts"].append({"kind": "conditioning", "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    conditioned, metadata = cc.condition_creative_context(job(contexts=[candidate]), Session())
    spy = PromptSpy()
    spy.speak("<|speaker:0|> next B", conditioned)
    assert len(spy.calls[-1]["history"]) == 1
    assert torch.equal(spy.calls[-1]["history"][0], audio_frames(actual_codes))
    assert not (spy.calls[-1]["history"][0][:, 1:] == 17).any()
    assert metadata["creative_context"]["history"][0]["source"] == "conditioning"
    assert torch.equal(engine.encoded[-1], actual_pcm)


def test_global_actor_snapshot_disagreement_is_not_silently_recast(tmp_path):
    candidate = descriptor(tmp_path, "wrong", "A", 7, raw("<|speaker:0|> wrong actor", 11, creative_global_speaker=8))
    with pytest.raises(ValueError, match="differs"):
        cc.condition_creative_context(job(contexts=[candidate]), Session())


