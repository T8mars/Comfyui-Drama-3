"""Actual Studio generator → native source-aware infill with CPU Codec probes."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from drama3_t8.creative_studio import CreativeStudio
from drama3_t8.creative_workspace import _file_record
from drama3_t8.dialogue import audio_frames
from drama3_t8.engine import DramaEngine, Sampling
from drama3_t8.state import Session, Take, Reference, save_session, load_session
from drama3_t8 import render_infill as ri
from test_source_aware_infill import Engine as CodecProbe


class StudioInfillProbe(CodecProbe):
    """Native infill dispatch and real typed archive; synthetic model/Codec."""
    def __init__(self):
        super().__init__()
        self.original = torch.linspace(-.45, .45, 12*2048)[None]
        self.drama = self.codec = None
        self.moe_backend = self.gdn_backend = "torch"
        self.expert_placement, self.cuda_graph = "ram", False
        self.infill_calls = []
        self.source_timeline = []

    def load_model(self):
        self.drama = super().load_model()
        return self.drama

    def load_codec(self, **kwargs):
        super().load_codec(**kwargs)
        self.codec = self
        return self

    def infill(self, *args, **kwargs):
        self.infill_calls.append({"args": args, "kwargs": kwargs})
        return DramaEngine.infill(self, *args, **kwargs)

    def decode_take(self, take, with_timeline=False):
        if "render_infill" in take.metadata:
            audio, timing = ri.decode_rendered(take), ri.render_timeline(take)
        else:
            audio, timing = {"waveform": self.original[None].clone(), "sample_rate": 44100}, self.source_timeline
        return (audio, timing) if with_timeline else audio

    def encode_reference(self, waveform, sample_rate, transcript, name="", **kwargs):
        assert sample_rate == 44100 and not kwargs
        return Reference(transcript, self.encode(waveform)[0], name)

    def expert_stats(self):
        return {"actual_backend": "torch", "placement": "ram", "fallback_reason": None}


def setup(tmp_path, *, scene=False):
    engine = StudioInfillProbe()
    source_dir, output = tmp_path / "source", tmp_path / "new-candidate"
    source_dir.mkdir(); output.mkdir()
    if scene:
        first = audio_frames(torch.ones((10, 4), dtype=torch.long))
        separator = torch.zeros((1, 11), dtype=torch.long); separator[0, 0] = 252196
        second = audio_frames(torch.ones((10, 8), dtype=torch.long))
        parent = Take("<|speaker:4|> First line\n<|audio_separator|>\n<|speaker:9|> Second line", "First line\nSecond line",
            torch.cat([first, separator, second]), metadata={"task": "scene",
                "script_lines": ["<|speaker:4|> First line", "<|speaker:9|> Second line"],
                "line_speakers": [4, 9], "creative_line_ids": ["L01", "L02"],
                "creative_global_speaker": 2, "creative_local_speaker": 0})
        engine.source_timeline = [{"line": 0, "speaker": 4, "text": "First line", "start": 0., "end": 8192/44100},
                                  {"line": 1, "speaker": 9, "text": "Second line", "start": 8192/44100, "end": 24576/44100}]
        lines = [{"line_id": "L01", "role": "A", "readout_text": "First line"},
                 {"line_id": "L02", "role": "B", "readout_text": "Complete replacement words"}]
        line_id, role, cast = "L02", "B", {"A": {"speaker": 1}, "B": {"speaker": 2}}
    else:
        parent = Take("<|speaker:0|> Original complete utterance", "Original complete utterance",
            audio_frames(torch.ones((10, 12), dtype=torch.long)), metadata={"task": "tts", "creative_line_ids": ["L01"],
                "creative_global_speaker": 2, "creative_local_speaker": 0})
        lines = [{"line_id": "L01", "role": "B", "readout_text": "Complete replacement words"}]
        line_id, role, cast = "L01", "B", {"B": {"speaker": 2}}
    path = source_dir / "selected.wav"
    sf.write(path, engine.original.T.numpy(), 44100, subtype="PCM_24")
    actual, _ = sf.read(path, dtype="float32", always_2d=True)
    engine.original = torch.from_numpy(actual.T.copy())
    archive = source_dir / "session.safetensors"
    save_session(Session(takes=(parent,)), archive)
    ids = [line["line_id"] for line in lines]
    intervals = engine.source_timeline or [{"line": 0, "speaker": 0, "start": 0., "end": 24576/44100}]
    candidate = {"candidate_id": "b"*32, "audio_path": str(path), "session_path": str(archive), "line_ids": ids,
        "artifacts": [{"kind": "audio", "path": "selected.wav", **_file_record(path)},
                      {"kind": "session", "path": "session.safetensors", **_file_record(archive)}],
        "line_audio": [{"line_id": ids[item["line"]], "start_seconds": item["start"], "end_seconds": item["end"]} for item in intervals],
        "metadata": {"reference_versions": [], "line_timings": intervals},
        "input_snapshot": {"model_identity": {"codec": {"sha256": "c"*64, "bytes": 1}}}}
    job = {"project_id": "a"*32, "line_id": line_id, "line_ids": ids, "role": role, "roles": list(cast),
        "lines": lines, "cast": cast, "speaker_map": {name: value["speaker"] for name, value in cast.items()},
        "settings": {}, "mode": "together" if scene else "per_line_context", "sampling": Sampling(seed=42).__dict__,
        "seed": 42, "take_index": 2, "source_candidate": candidate,
        "model_identity": candidate["input_snapshot"]["model_identity"],
        "operation": {"kind": "infill", "line_id": line_id, "start_seconds": 9000/44100, "end_seconds": 13000/44100}}
    manager = object.__new__(CreativeStudio)
    manager.workspace = None  # The unmarked infill never enters automatic quality retries.
    manager.studio = SimpleNamespace(engine=engine)
    manager.performance_binding = lambda models=None: {"scope": "CPU synthetic Codec probe"}
    manager.reports = tmp_path / "reports"; manager.reports.mkdir()
    return manager, engine, parent, job, output


@pytest.mark.parametrize("scene,expected_actor", [(False, 0), (True, 9)])
def test_actual_studio_generate_forces_selected_pcm24_view_and_keeps_unselected_samples(tmp_path, scene, expected_actor):
    manager, engine, parent, job, output = setup(tmp_path, scene=scene)
    old_pcm = engine.original.clone()
    result = manager._generate(job, output, lambda *_: None)
    call = engine.infill_calls[0]
    assert call["kwargs"]["target_speaker"] == expected_actor
    view = call["kwargs"]["expected_view"]
    assert view["subtype"] == "PCM_24" and view["pcm24_sha256"] == view["pcm_sha256"]
    assert call["args"][1].startswith("<|speaker:4|>" if scene else "<|speaker:0|>")
    edited = load_session(result["session_path"]).takes[-1]
    assert "render_infill" in edited.metadata
    audio, _ = sf.read(result["audio_path"], dtype="float32", always_2d=True)
    assert np.array_equal(audio[:9000].view(np.int32), old_pcm.T.numpy()[:9000].view(np.int32))
    assert np.array_equal(audio[9000+4096:].view(np.int32), old_pcm.T.numpy()[13000:].view(np.int32))
    assert len(result["line_audio"]) == len(job["lines"])
    assert len(edited.metadata["render_infill"]["timeline"]) > len(job["lines"])
    if scene:
        assert result["line_audio"][0]["end_seconds"] == 8192/44100
        assert result["line_audio"][1]["start_seconds"] == 8192/44100
    else:
        assert result["conditioning_session_path"]


def test_cross_line_selection_is_rejected_before_loading_model(tmp_path):
    manager, engine, _, job, output = setup(tmp_path, scene=True)
    job["operation"]["start_seconds"] = 7000/44100
    with pytest.raises(ValueError, match="目标台词"):
        manager._generate(job, output, lambda *_: None)
    assert engine.model_loads == 0 and not engine.infill_calls


def test_changed_selected_asset_is_rejected_before_loading_model(tmp_path):
    manager, engine, _, job, output = setup(tmp_path)
    with Path(job["source_candidate"]["audio_path"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="实际音频已改变"):
        manager._generate(job, output, lambda *_: None)
    assert engine.model_loads == 0 and not engine.infill_calls


def test_second_studio_infill_uses_actual_rendered_candidate_and_keeps_pcm24_edges(tmp_path):
    manager, engine, _, job, output = setup(tmp_path)
    first = manager._generate(job, output, lambda *_: None)
    before, _ = sf.read(first["audio_path"], dtype="float32", always_2d=True)
    job["source_candidate"] = {"candidate_id": "d"*32, "line_ids": job["line_ids"],
        "audio_path": first["audio_path"], "session_path": first["session_path"], "metadata": first["metadata"],
        "line_audio": first["line_audio"], "input_snapshot": {"model_identity": job["model_identity"]},
        "artifacts": [{"kind": "audio", **_file_record(first["audio_path"])}]}
    job["lines"][0]["readout_text"] = "A second complete replacement"
    job["operation"].update(start_seconds=15000/44100, end_seconds=19000/44100)
    second_dir = tmp_path / "second"; second_dir.mkdir()
    second = manager._generate(job, second_dir, lambda *_: None)
    after, _ = sf.read(second["audio_path"], dtype="float32", always_2d=True)
    assert np.array_equal(after[:15000].view(np.int32), before[:15000].view(np.int32))
    assert np.array_equal(after[15000+4096:].view(np.int32), before[19000:].view(np.int32))
    take = load_session(second["session_path"]).takes[-1]
    assert take.metadata["render_infill"]["subtype"] == "PCM_24"
    assert len(take.sources[0].sources) == 2 and len(second["line_audio"]) == 1


@pytest.mark.parametrize("explicit", [True, 1.5, "2", -1, 16])
def test_invalid_explicit_native_speaker_is_rejected_before_loading_model(tmp_path, explicit):
    manager, engine, _, job, output = setup(tmp_path)
    job["operation"]["target_speaker"] = explicit
    with pytest.raises(ValueError, match="speaker"):
        manager._generate(job, output, lambda *_: None)
    assert engine.model_loads == 0 and not engine.infill_calls


@pytest.mark.parametrize("kind", ["named_scene", "timeline"])
def test_scene_role_uses_actual_named_map_or_recorded_native_timing_not_project_global(tmp_path, kind):
    manager, _, parent, job, _ = setup(tmp_path, scene=True)
    metadata = {**parent.metadata, "line_speakers": None,
                "script_lines": ["A: First line", "B: Second line"]}
    if kind == "named_scene":
        metadata["scene_speaker_map"] = {"A": 4, "B": 9}
    changed = Take(parent.prompt_text, parent.transcript, parent.frames, metadata=metadata)
    selected = manager._infill_candidate_input(job, changed)
    assert selected["target_speaker"] == 9 and selected["native_speakers"] == [4, 9]


def test_plain_context_local_speaker_precedes_project_global_timing_label(tmp_path):
    manager, _, parent, job, _ = setup(tmp_path)
    job["source_candidate"]["metadata"]["line_timings"][0]["speaker"] = 2
    selected = manager._infill_candidate_input(job, parent)
    assert selected["target_speaker"] == 0 and selected["native_speakers"] == [0]


def test_actual_chunked_source_keeps_native_role_and_pcm24_outside_selection(tmp_path):
    from dataclasses import replace
    from test_scene_chunks import Harness
    from drama3_t8.scene_chunks import decode_scene
    manager, engine, _, job, output = setup(tmp_path)
    built = Harness().scene(["<|speaker:7|> " + "A quiet narrator prepares a long speech. "*8],
        generation_mode="per_line", split_long_text=True, chunk_bytes=80)
    parent = replace(built.takes[-1], metadata={**built.takes[-1].metadata, "creative_line_ids": ["L01"],
        "creative_global_speaker": 2, "creative_local_speaker": 0})
    actual, source_timing = decode_scene(engine, parent)
    path = job["source_candidate"]["audio_path"]
    sf.write(path, actual["waveform"][0].T.numpy(), 44100, subtype="PCM_24")
    before, _ = sf.read(path, dtype="float32", always_2d=True)
    engine.original, engine.source_timeline = torch.from_numpy(before.T.copy()), source_timing
    save_session(Session(takes=(parent,)), job["source_candidate"]["session_path"])
    candidate = job["source_candidate"]
    candidate["artifacts"] = [{"kind": "audio", **_file_record(path)}]
    candidate["metadata"]["line_timings"] = source_timing
    candidate["line_audio"] = [{"line_id": "L01", "start_seconds": 0., "end_seconds": before.shape[0]/44100}]
    result = manager._generate(job, output, lambda *_: None)
    assert engine.infill_calls[0]["kwargs"]["target_speaker"] == 7
    assert engine.infill_calls[0]["args"][1].startswith("<|speaker:7|>")
    after, _ = sf.read(result["audio_path"], dtype="float32", always_2d=True)
    assert np.array_equal(after[:9000].view(np.int32), before[:9000].view(np.int32))
    assert np.array_equal(after[9000+4096:].view(np.int32), before[13000:].view(np.int32))
    restored = load_session(result["session_path"]).takes[-1]
    assert "scene_chunking" in restored.sources[0].metadata and len(result["line_audio"]) == 1
