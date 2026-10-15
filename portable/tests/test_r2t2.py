"""R2T2 boundary, host coexistence and resource contracts without model allocation."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from drama3_t8.r2t2 import native
from drama3_t8.r2t2.segmented import PresetBoundaryVAD, SegmentedStream


@pytest.mark.parametrize("seconds", [0, -1, 31, float("nan"), float("inf")])
def test_segment_limits_reject_nonprogressing_or_unbounded_duration(seconds):
    with pytest.raises(ValueError):
        SegmentedStream(None, vad=PresetBoundaryVAD([]), segment_seconds=seconds)


def test_asr_registration_preserves_other_node_extensions(monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import folder_paths
    import drama3_t8.nodes as nodes
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "asr", (["existing-asr"], {".gguf", ".bin"}))
    importlib.reload(nodes)
    paths, extensions = folder_paths.folder_names_and_paths["asr"]
    assert "existing-asr" in paths
    assert {".gguf", ".bin", ".safetensors"} <= extensions


def test_qwen_dropdown_only_offers_its_supported_weight_format(monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import drama3_t8.nodes as nodes
    monkeypatch.setattr(nodes.folder_paths, "get_filename_list", lambda kind: ["r2t2.gguf", "qwen.safetensors", "other.bin"])
    assert nodes.T8DramaTranscribe.define_schema().inputs[0].options == ["qwen.safetensors"]


@pytest.mark.parametrize("strong_cache", [False, True])
@pytest.mark.parametrize("extensions", [{".gguf"}, set()])
def test_registering_asr_keeps_wildcard_semantics_and_refreshes_primed_cache(tmp_path, monkeypatch, strong_cache, extensions):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import folder_paths
    import drama3_t8.nodes as nodes
    directory = tmp_path / "models/asr"
    directory.mkdir(parents=True)
    (directory / "other.gguf").touch()
    (directory / "qwen.safetensors").touch()
    monkeypatch.setattr(folder_paths, "models_dir", str(directory.parent))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "asr", ([str(directory)], set(extensions)))
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    helper = folder_paths.CacheHelper()
    helper.active = strong_cache
    monkeypatch.setattr(folder_paths, "cache_helper", helper)
    folder_paths.get_filename_list("asr")
    importlib.reload(nodes)
    assert set(folder_paths.get_filename_list("asr")) == {"other.gguf", "qwen.safetensors"}
    if not extensions:
        assert folder_paths.folder_names_and_paths["asr"][1] == set()
    assert nodes.T8DramaTranscribe.define_schema().inputs[0].options == ["qwen.safetensors"]


class FakeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def test_native_loader_adds_cuda_when_host_already_registered_cpu(tmp_path, monkeypatch):
    for folder in (tmp_path / "lib", tmp_path / "backends"):
        folder.mkdir()
    for name in ("ggml-cpu.dll", "ggml-cuda.dll"):
        (tmp_path / "backends" / name).touch()
    loaded = []
    registry = SimpleNamespace(
        ggml_backend_dev_count=FakeFunction(lambda: 1 + len(loaded)),
        ggml_backend_dev_by_type=FakeFunction(lambda kind: 1 if kind == 0 or loaded else 0),
        ggml_backend_load=FakeFunction(lambda path: loaded.append(path.decode("utf-8")) or 1))
    api = SimpleNamespace(**{name: FakeFunction(lambda *args: None) for name in (
        "r2t2_create", "r2t2_destroy", "r2t2_generate", "r2t2_last_error", "r2t2_last_truncated")})
    api.r2t2_api_version = FakeFunction(lambda: 1)
    monkeypatch.setattr(native, "runtime_directory", lambda: tmp_path)
    monkeypatch.setattr(native, "find_spec", lambda name: None)
    monkeypatch.setattr(native, "_HANDLES", [])
    monkeypatch.setattr(native, "_LIBRARIES", {})
    monkeypatch.setattr(native.os, "add_dll_directory", lambda path: object())
    monkeypatch.setattr(native.C, "CDLL", lambda path: registry if str(path).endswith("ggml.dll") else api)
    assert native._load_library() is api
    assert [str(tmp_path / "backends" / "ggml-cuda.dll")] == loaded


def test_auto_language_malformed_output_is_not_a_successful_empty_transcript():
    with pytest.raises(RuntimeError, match="ASR_PARSE_ERROR"):
        native.parse_raw("language Chinese 你好")
    assert native.parse_raw("language None<asr_text>") == ("", "")
    assert native.parse_raw("") == ("", "")
    assert native.parse_raw("language Chinese<asr_text>你好<|im_end|>") == ("Chinese", "你好")
    assert native.parse_raw("你好<|im_end|>", "Chinese") == ("Chinese", "你好")


class SequenceEngine:
    def __init__(self):
        self.lengths = []

    def transcribe(self, audio, **kwargs):
        self.lengths.append(len(audio))
        first = len(self.lengths) == 1
        return {"text": "hello" if first else "world", "language": "English",
                "finish_reason": "length" if first else "stop", "truncated": first}


def test_segment_boundaries_preserve_samples_join_words_and_keep_truncation():
    engine = SequenceEngine()
    stream = SegmentedStream(engine, vad=PresetBoundaryVAD([]), segment_seconds=1)
    stream.feed(np.ones(24000, dtype=np.float32))
    result = stream.finish()
    assert engine.lengths == [16000, 8000]
    assert result["stable_text"] == "hello world"
    assert result["truncated"] and result["finish_reason"] == "length"
    assert [(s["start_sample"], s["end_sample"]) for s in result["segments"]] == [(0, 16000), (16000, 24000)]
    assert stream.finish()["idempotent"] and len(engine.lengths) == 2


def test_digital_silence_skips_native_model_but_quiet_audio_is_retained(monkeypatch):
    from drama3_t8.r2t2.audio import limited_quiet_gain
    import drama3_t8.r2t2.transcribe as transcription
    monkeypatch.setattr(transcription, "NativeQ8Engine", lambda *args, **kwargs: pytest.fail("silence must not load a model"))
    result = transcription.transcribe_audio({"waveform": np.zeros((1, 1, 16000), dtype=np.float32),
                                             "sample_rate": 16000}, "model", "projector", "vad")
    assert result["text"] == "" and result["mode_executed"] == "digital_silence"
    quiet, gain = limited_quiet_gain(np.full(16000, 1e-7, dtype=np.float32))
    assert np.all(quiet != 0) and gain == 10


def test_audio_contract_rejects_batch_nonfinite_and_bad_rates():
    from drama3_t8.r2t2.audio import comfy_audio_to_16k
    for waveform, rate in ((np.zeros((2, 1, 1000)), 16000),
                           (np.full((1, 1, 1000), np.nan), 16000),
                           (np.zeros((1, 1, 1000)), 0),
                           (np.zeros((1, 1, 1000)), "16000")):
        with pytest.raises(ValueError):
            comfy_audio_to_16k({"waveform": waveform, "sample_rate": rate})
    assert comfy_audio_to_16k({"waveform": np.ones((1, 2, 44100)), "sample_rate": 44100}).shape == (16000,)


def test_loader_forwards_independent_custom_paths_without_using_dropdowns(tmp_path, monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import drama3_t8.nodes as nodes
    outside = tmp_path / "独立 模型"
    outside.mkdir()
    model, codec, head = (outside / name for name in ("renamed.safetensors", "codec.safetensors", "head.safetensors"))
    model.touch()
    codec.touch()
    head.touch()
    monkeypatch.setattr(nodes, "DramaEngine", lambda *args, **kwargs: (args, kwargs))
    result = nodes.T8DramaLoader.execute("unused", "unused", model_path=str(model),
                                       codec_path=str(codec), text_head_path=str(head))[0]
    assert result[0][:2] == (model.resolve(), codec.resolve())
    assert result[1]["text_head_path"] == head.resolve()


def test_r2t2_node_forwards_custom_paths_and_returns_standard_strings(tmp_path, monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import drama3_t8.nodes as nodes
    import drama3_t8.r2t2.transcribe as transcription
    audio = {"waveform": np.ones((1, 1, 16000), dtype=np.float32), "sample_rate": 16000}
    base = tmp_path / "models"
    monkeypatch.setattr(nodes.folder_paths, "models_dir", str(base))
    captured = {}
    def transcribe(value, model, projector, vad, **kwargs):
        captured.update(audio=value, paths=(model, projector, vad), options=kwargs)
        return {"text": "你好", "language": "Chinese", "execution": "in_process_native"}
    monkeypatch.setattr(transcription, "transcribe_audio", transcribe)
    result = nodes.T8DramaR2T2Transcribe.execute(audio, "elsewhere/main.gguf", str(tmp_path / "projector.gguf"),
                                              "elsewhere/vad", "Chinese", "热词", "场景")
    assert result[0:2] == ("你好", "Chinese")
    assert json.loads(result[2])["execution"] == "in_process_native"
    assert captured["audio"] is audio
    assert captured["paths"] == (base / "elsewhere/main.gguf", tmp_path / "projector.gguf", base / "elsewhere/vad")
    assert captured["options"]["hotwords"] == "热词"
    assert callable(captured["options"]["check_cancel"])


def test_hotwords_preserve_phrases_deduplicate_and_bound_combined_prompt():
    from drama3_t8.r2t2.hotwords import hotword_context, normalize_hotwords
    assert normalize_hotwords("\ufeff台词，Fish Audio;台词\n角色") == ("台词, Fish Audio, 角色", 3)
    assert hotword_context("场景", "台词") == "场景\nHotwords: 台词"
    with pytest.raises(ValueError, match="CONTEXT_LIMIT"):
        hotword_context("x" * 8190, "台词")
    with pytest.raises(ValueError, match="64 KiB"):
        normalize_hotwords("汉" * (64 * 1024 // 3 + 1))


def test_native_model_closes_when_transcription_raises_base_exception(monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import comfy.model_management as mm
    import drama3_t8.r2t2.transcribe as transcription
    calls = []
    class Stop(BaseException):
        pass
    class Engine:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            calls.append("loaded")
            return self
        def transcribe(self, *args, **kwargs):
            raise Stop()
        def __exit__(self, *args):
            calls.append("closed")
    monkeypatch.setattr(transcription, "NativeQ8Engine", Engine)
    monkeypatch.setattr(mm, "unload_all_models", lambda: None)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    with pytest.raises(Stop):
        transcription.transcribe_audio({"waveform": np.ones((1, 1, 16000), dtype=np.float32),
                                       "sample_rate": 16000}, "model", "projector", "vad")
    assert calls == ["loaded", "closed"]
