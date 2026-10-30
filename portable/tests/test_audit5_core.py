"""Fifth audit: fresh CPU contracts, tokenizer layouts and archive snapshots."""
import contextlib
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.engine import DramaEngine, Sampling
from drama3_t8.state import Reference, Session, Take, load_session, save_session


@pytest.fixture
def tokenizer():
    return Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "drama3_t8/native/fish/tokenizer.json"))


@pytest.fixture
def config():
    return SimpleNamespace(tower=SimpleNamespace(num_codebooks=10, vocab_size=4096),
                           semantic_start_token_id=248099, semantic_end_token_id=252194,
                           output_token_ids=tuple([252196, *range(248099, 252195)]))


def take(value=1, count=3, **kwargs):
    codes = torch.full((10, count), value, dtype=torch.long)
    return Take("Hello.", "Hello.", torch.cat((codes[:1] + 248099, codes)).T, **kwargs)


def test_streaming_session_keeps_global_direction_in_current_user_turn(tokenizer, config):
    engine = DramaEngine("model", "codec")
    engine.load_model = lambda: SimpleNamespace(tokenizer=tokenizer, config=config)
    seen = []
    def generated(prompt, request, transcript, *args, **kwargs):
        seen.append(prompt)
        return replace(take(), prompt_text=request, transcript=transcript)
    engine._take = generated
    source = Session(direction="请始终轻声，语气像讲睡前故事。")
    result = engine.stream_text("第一句。", "下一句。", source)
    decoded = tokenizer.decode(seen[0][:, 0].tolist(), skip_special_tokens=False)
    system = decoded.split("<|im_end|>", 1)[0]
    assert source.direction not in system
    user = decoded.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    styled = "<|speaker:0|> [" + source.direction + "] 第一句。"
    assert user == styled + "<|endofturn|>下一句。"
    assert result.takes[-1].prompt_text == styled
    assert result.takes[-1].transcript == "第一句。"
    assert result.direction == source.direction and result.takes[-1].metadata["lookahead"] == "下一句。"
    assert source.takes == ()


def test_reference_leading_whitespace_matches_canonical_training_layout(tokenizer, config):
    from drama3_t8.native.prompt import build_prompt
    from drama3_t8.prompts import tts_prompt
    codes = torch.full((10, 2), 7, dtype=torch.long)
    transcript = "\t\n <|speaker:7|> Hello, reference."
    canonical = build_prompt(tokenizer, config, "Next.", [(transcript, codes)])
    current = tts_prompt(tokenizer, config, "Next.", [Reference(transcript, codes)])
    assert torch.equal(current, canonical)


@pytest.mark.parametrize("kind", ["reference", "take"])
def test_session_save_supports_distinct_records_sharing_tensor_storage(tmp_path, kind):
    first = take()
    first = replace(first, frames=first.frames.contiguous())
    second = replace(first, take_id="b" * 32, prompt_text="Another prompt.")
    shared_codes = torch.ones(10, 4, dtype=torch.int16)
    references = (Reference("A", shared_codes), Reference("B", shared_codes)) if kind == "reference" else ()
    takes = (first, second) if kind == "take" else (first,)
    session = Session(references, takes)
    target = tmp_path / "共享数据.safetensors"
    save_session(session, target)
    restored = load_session(target)
    assert [t.prompt_text for t in restored.takes] == [t.prompt_text for t in takes]
    assert torch.equal(restored.takes[0].frames, first.frames)
    if kind == "take":
        assert torch.equal(restored.takes[1].frames, second.frames)
    assert all(torch.equal(ref.codes, shared_codes.long()) for ref in restored.references)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_residual_code_sampler_rejects_nonfinite_logits(invalid):
    from drama3_t8.native.generate import sample_top_k_top_p
    # Residual tower heads call this sampler directly, unlike the main audio head.
    with pytest.raises(FloatingPointError, match="non-finite"):
        sample_top_k_top_p(torch.tensor([[1., invalid, .3]]), .7, 3, .8,
                           torch.Generator().manual_seed(42))


def test_session_load_reads_header_and_tensors_from_one_open_snapshot(tmp_path, monkeypatch):
    import safetensors
    from safetensors.torch import load_file as real_load_file
    path = tmp_path / "current.safetensors"
    first = Session(takes=(take(value=3),), direction="archive A")
    second = Session(takes=(take(value=9),), direction="archive B")
    save_session(first, path)
    alternate = tmp_path / "next.safetensors"
    save_session(second, alternate)
    real_open = safetensors.safe_open
    replaced = []
    class SwapAtClose:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self.handle
        def __exit__(self, *args):
            result = self.handle.__exit__(*args)
            if not replaced:
                alternate.replace(path)
                replaced.append(True)
            return result
    monkeypatch.setattr(safetensors, "safe_open", SwapAtClose)
    restored = load_session(path)
    assert replaced and restored.direction == "archive A"
    assert torch.equal(restored.takes[0].frames, first.takes[0].frames), "Header from A was paired with tensors from replacement B"
    assert torch.equal(real_load_file(str(path))["take_0"], second.takes[0].frames)


def fake_drama_with_residual_heads(monkeypatch, *, bad=False, context_limit=32):
    import drama3_t8.native.generate as generator
    monkeypatch.setattr(generator.comfy.model_management, "cuda_device_context", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(generator.comfy.model_management, "throw_exception_if_processing_interrupted", lambda: None)
    mode = {"bad": bad}
    backbone = SimpleNamespace(embed_tokens=lambda ids, out_dtype: torch.zeros(len(ids), 4),
        init_caches=lambda *a: [], run_layers=lambda x, *a: x)
    def step(main, hidden, frame, context, caches, sample):
        first = sample(torch.tensor([[.2, .8]]))
        next_logits = torch.tensor([[float("nan"), .8]]) if mode["bad"] else torch.tensor([[.5, .8]])
        second = sample(next_logits)
        return torch.tensor([[main.item(), 1, first.item(), second.item(), *([1] * 7)]])
    tower = SimpleNamespace(init_caches=lambda *a: [], embed_backbone=lambda frames, text: text,
        run=lambda *a: None, context=lambda *a: None,
        is_semantic=lambda values: (values >= 248099) & (values <= 252194), step=step)
    model = SimpleNamespace(model=backbone, dtype=torch.float32)
    config = SimpleNamespace(output_token_ids=(248100,), eos_token_id=252197,
        semantic_start_token_id=248099, semantic_end_token_id=252194,
        text=SimpleNamespace(max_position_embeddings=context_limit))
    drama = SimpleNamespace(config=config, model=model, experts=None,
        tokenizer=SimpleNamespace(token_to_id=lambda s: 252196),
        parts=lambda: ([None], [torch.device("cpu")], lambda x: x, lambda x: torch.ones(1, 1), tower))
    return drama, mode


def test_generate_rejects_bad_residual_before_publishing_and_recovers(monkeypatch):
    from drama3_t8.native.generate import generate
    drama, mode = fake_drama_with_residual_heads(monkeypatch, bad=True)
    delivered = []
    prompt = torch.zeros(2, 11, dtype=torch.long)
    with pytest.raises(FloatingPointError, match="residual"):
        generate(drama, prompt, 1, .7, 2, 1., 42, on_audio=delivered.append)
    assert delivered == []
    mode["bad"] = False
    frames = generate(drama, prompt, 1, .7, 2, 1., 42, on_audio=delivered.append)
    assert frames.shape == (1, 11) and len(delivered) == 1


def test_pending_early_silence_is_superseded_by_new_speech():
    from drama3_t8.r2t2.segmented import SegmentedStream
    calls = []
    engine = SimpleNamespace(transcribe=lambda pcm, **kw: calls.append(pcm.copy()) or {
        "text": "word", "language": "English", "truncated": False, "finish_reason": "stop"})
    class VAD:
        active = False
        speech_candidate = 0
        count = 0
        def feed(self, pcm):
            self.count += 1
            self.active = self.count == 2
            return ([{"sample": 800, "reason": "silence"}] if self.count == 1 else
                    [{"sample": 3000, "reason": "silence"}] if self.count == 3 else [])
    stream = SegmentedStream(engine, vad=VAD(), segment_seconds=1, min_segment_seconds=.1)
    stream.feed(np.ones(1000, np.float32))
    assert stream._pending_silence
    stream.feed(np.ones(500, np.float32))
    assert not stream._pending_silence and not calls
    stream.feed(np.ones(2000, np.float32))
    final = stream.finish()
    assert [len(a) for a in calls] == [3000, 500]
    assert final["segments"][0]["end_sample"] == 3000


def test_vad_acoustic_bounds_clip_completed_and_open_intervals():
    from drama3_t8.r2t2.vad import FireRedOnnxVAD
    vad = object.__new__(FireRedOnnxVAD)
    vad.speech_intervals = deque([(0, 800), (1200, 2000)])
    vad.speech_start_sample = 2600
    vad.samples_seen, vad.next_frame, vad.silence_candidate = 4000, 22, 2
    assert vad.speech_bounds(500, 3000) == (500, 3000)
    assert vad.speech_bounds(800, 1200) is None
    assert vad.speech_bounds(2800, 4000) == (2800, 3440)


def test_segment_timestamp_keeps_decoder_ownership_and_leading_silence_offset():
    from drama3_t8.r2t2.segmented import PresetBoundaryVAD, SegmentedStream
    seen = []
    engine = SimpleNamespace(transcribe=lambda pcm, **kw: seen.append(pcm.copy()) or {
        "text": "hello", "language": "English", "truncated": False, "finish_reason": "stop"})
    stream = SegmentedStream(engine, vad=PresetBoundaryVAD([]), segment_seconds=1)
    stream.feed(np.zeros(2000, np.float32))
    stream.feed(np.ones(8000, np.float32))
    final = stream.finish()
    segment = final["segments"][0]
    assert len(seen) == 1 and len(seen[0]) == 8000
    assert (segment["start_sample"], segment["end_sample"], segment["skipped_zero_samples"]) == (0, 10000, 2000)
    assert (segment["speech_start_sample"], segment["speech_end_sample"]) == (2000, 10000)
    assert final["audio_end_sample"] == 10000


def test_native_library_cache_keeps_runtime_directories_independent(tmp_path, monkeypatch):
    import drama3_t8.r2t2.native as native
    directories = [tmp_path / "运行库 A", tmp_path / "运行库 B"]
    for directory in directories:
        (directory / "lib").mkdir(parents=True)
        (directory / "backends").mkdir()
    selected = [directories[0]]
    opened = []
    def function(value=None):
        return lambda *a: value
    def library(path):
        opened.append(str(path))
        if str(path).endswith("ggml.dll"):
            return SimpleNamespace(ggml_backend_load=function(1), ggml_backend_dev_count=function(2),
                ggml_backend_dev_by_type=function(1))
        return SimpleNamespace(r2t2_api_version=function(1), **{name: function() for name in
            ("r2t2_create", "r2t2_destroy", "r2t2_generate", "r2t2_last_error", "r2t2_last_truncated")})
    monkeypatch.setattr(native, "runtime_directory", lambda: selected[0])
    monkeypatch.setattr(native, "find_spec", lambda name: None)
    monkeypatch.setattr(native, "_HANDLES", [])
    monkeypatch.setattr(native, "_LIBRARIES", {})
    monkeypatch.setattr(native.os, "add_dll_directory", lambda path: object())
    monkeypatch.setattr(native.C, "CDLL", library)
    first = native._load_library()
    assert native._load_library() is first
    selected[0] = directories[1]
    second = native._load_library()
    assert second is not first and native._load_library() is second
    assert len(opened) == 4 and set(native._LIBRARIES) == set(directories)


def test_qwen_real_tokenizer_audio_embeddings_replace_only_audio_pad(monkeypatch):
    import drama3_t8.qwen3_asr.asr as qwen
    captured = []
    def embed(ids, out_dtype):
        return ids.float()[:, None].expand(-1, 4).clone()
    patcher = SimpleNamespace(load_device=torch.device("cpu"))
    decoder = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed),
        generate=lambda **kw: captured.append(kw["embeds"]) or [qwen.STOP_TOKENS[0]])
    asr = qwen.Qwen3ASR(SimpleNamespace(decoder=decoder, audio_tower=lambda f: torch.full((3, 4), -77.)),
                        patcher, torch.float32)
    monkeypatch.setattr(qwen, "audio_tokens", lambda n: 3)
    monkeypatch.setattr(qwen.comfy.model_management, "load_models_gpu", lambda *a: None)
    monkeypatch.setattr(qwen.comfy.model_management, "cuda_device_context", lambda *a: contextlib.nullcontext())
    context, language = "人物和地名：鱼声", "Chinese"
    qwen.transcribe(asr, torch.ones(1, 8000), 16000, language, context)
    ids = torch.tensor(qwen.prompt_ids(asr, context, language, 3))
    assert (ids == qwen.AUDIO_PAD).sum() == 3
    expected = embed(ids, torch.float32)
    expected[ids == qwen.AUDIO_PAD] = -77.
    assert torch.equal(captured[0][0], expected)
    assert asr.decode(ids[ids != qwen.AUDIO_PAD].tolist()).endswith("language Chinese<asr_text>")


def test_qwen_repetition_repair_matches_upstream_and_preserves_unique_suffix():
    from drama3_t8.qwen3_asr.asr import fix_repetitions
    from drama3_t8.vendor_qwen.inference.utils import detect_and_fix_repetitions
    text = "开场" + "哈" * 25 + "abcde" * 23 + "一次性结尾。"
    assert fix_repetitions(text) == detect_and_fix_repetitions(text)
    assert fix_repetitions(text) == "开场哈abcde一次性结尾。"


def test_qwen_rejects_unconverted_real_safetensor_before_model_allocation(tmp_path, monkeypatch):
    import drama3_t8.qwen3_asr.asr as qwen
    path = tmp_path / "普通权重.safetensors"
    save_file({"weight": torch.ones(2, 2)}, path, metadata={"unrelated": "model"})
    monkeypatch.setattr(qwen.comfy.model_management, "get_torch_device",
                        lambda: pytest.fail("wrong checkpoint allocated an ASR model"))
    with pytest.raises(ValueError, match="converted Qwen3-ASR"):
        qwen.load_qwen3_asr(str(path))


def test_generation_context_exact_fit_and_overflow_are_checked_before_cache_allocation(monkeypatch):
    from drama3_t8.native.generate import generate
    drama, _ = fake_drama_with_residual_heads(monkeypatch, context_limit=4)
    prompt = torch.zeros(2, 11, dtype=torch.long)
    assert generate(drama, prompt, 1, .7, 2, 1., 42).shape == (1, 11)
    drama.config.text.max_position_embeddings = 3
    drama.model.model.init_caches = lambda *a: pytest.fail("overflow allocated KV state")
    with pytest.raises(ValueError, match="context"):
        generate(drama, prompt, 1, .7, 2, 1., 42)


def test_two_device_partition_balances_measured_layer_and_tail_storage(monkeypatch):
    import drama3_t8.native.loader as loader
    def module(size):
        return SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(size)})
    sizes = [13, 19, 7, 23, 11]
    model = SimpleNamespace(model=SimpleNamespace(layers=[module(n) for n in sizes],
        embed_tokens=module(5), norm=module(3)), audio_head=module(7), tower=module(11))
    monkeypatch.setattr(loader, "FIRST_RESERVE", 0)
    monkeypatch.setattr(loader, "SECOND_RESERVE", 0)
    memory = {"card-a": 240, "card-b": 256}
    monkeypatch.setattr(loader.comfy.model_management, "get_total_memory", lambda device: memory[device])
    chosen = loader.auto_split(model, "card-a", "card-b")
    scores = {index: min(240 - 5 * 4 - sum(sizes[:index]) * 4,
        256 - (3 + 7 + 11) * 4 - sum(sizes[index:]) * 4) for index in range(1, len(sizes))}
    assert chosen == max(scores, key=scores.get)


def test_tower_backbone_embedding_ignores_codec_values_on_text_rows():
    import comfy.ops
    from drama3_t8.native.model import Drama3Tower
    config = SimpleNamespace(tower=SimpleNamespace(dim=8, vocab_size=4096, num_codebooks=10,
        backbone_codebooks=2, shared_layers=0), semantic_start_token_id=248099, semantic_end_token_id=252194)
    # Only the real embedding-mix operation is needed, with no attention layers/heads.
    tower = object.__new__(Drama3Tower)
    torch.nn.Module.__init__(tower)
    tower.cfg, tower.semantic_start, tower.semantic_end = config.tower, 248099, 252194
    tower.codebook_embeddings = comfy.ops.manual_cast.Embedding(40960, 8, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        tower.codebook_embeddings.weight.copy_(torch.arange(40960)[:, None].expand(-1, 8) / 100)
    frames = torch.tensor([[42, *([9] * 10)], [248106, *([7] * 10)]])
    text = torch.ones(2, 8)
    output = tower.embed_backbone(frames, text)
    expected_audio = (text[1] + tower.codebook_embeddings.weight[4096 + 7] +
                      tower.codebook_embeddings.weight[8192 + 7]) / np.sqrt(3)
    assert torch.equal(output[0], text[0])
    assert torch.allclose(output[1], expected_audio)


def test_backbone_device_groups_share_the_original_continuation_position(monkeypatch):
    from drama3_t8.native.model import Drama3Backbone
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "cuda_device_context", lambda *a: contextlib.nullcontext())
    observed = []
    caches = [SimpleNamespace(index=7), SimpleNamespace(index=7)]
    def segment(x, layers, local, device, past, graphs=None, after_layer=None):
        observed.append((str(device), past))
        for cache in local:
            cache.index += x.shape[1]
        return x + 1
    backbone = SimpleNamespace(_segment=segment)
    result = Drama3Backbone.run_layers(backbone, torch.zeros(1, 3, 2), caches, [None, None],
                                      [torch.device("cpu"), torch.device("cpu:0")])
    assert observed == [("cpu", 7), ("cpu:0", 7)] and torch.equal(result, torch.full((1, 3, 2), 2.))


def test_tower_multirow_continuation_mask_preserves_past_and_hides_future(monkeypatch):
    import drama3_t8.native.model as model
    monkeypatch.setattr(model, "optimized_attention_for_device", lambda *a, **kw: None)
    tower = SimpleNamespace(cfg=SimpleNamespace(rotary_dim=4, rope_base=10000))
    freqs, mask, _ = model.Drama3Tower.context(tower, torch.tensor([7, 8, 9]), 3, "cpu", torch.float32)
    assert freqs.shape[-1] == 2 and mask.shape == (3, 10)
    assert torch.all(mask[:, :8] == 0)
    assert mask[0, 8] < -1e30 and mask[0, 9] < -1e30 and mask[1, 9] < -1e30
    assert mask[1, 8] == 0 and torch.all(mask[2] == 0)


def test_mixed_scene_stream_archive_can_build_next_scene_prompt(tmp_path, tokenizer, config):
    from drama3_t8.prompts import SEPARATOR
    streaming = replace(take(), metadata={"task": "text_stream", "lookahead": "Next."})
    scene_frames = torch.cat((take(2).frames, torch.tensor([[252196, *([0] * 10)]]), take(3).frames))
    scene = Take("A: Hello\n" + SEPARATOR + "\nB: Hi", "Hello\nHi", scene_frames,
                metadata={"task": "scene", "script_lines": ["A: Hello", "B: Hi"]})
    source = Session(takes=(streaming, scene))
    path = tmp_path / "混合历史.safetensors"
    save_session(source, path)
    restored = load_session(path)
    engine = DramaEngine("model", "codec")
    engine.load_model = lambda: SimpleNamespace(tokenizer=tokenizer, config=config)
    captured = []
    engine._take = lambda prompt, *a, **kw: captured.append((prompt, kw)) or replace(scene, take_id="e" * 32)
    result = engine.scene(["A: Again", "B: Reply"], session=restored, note="more lively")
    decoded = tokenizer.decode(captured[0][0][:, 0].tolist(), skip_special_tokens=False)
    assert "<|endofturn|>" in decoded and "more lively" in decoded
    assert result.takes[:-1] == restored.takes and captured[0][1]["full_head"]
    assert result.takes[-1].metadata["actual_segments"] == 2


def test_r2t2_channel_selection_and_real_resampling_keep_orientation():
    from drama3_t8.r2t2.audio import to_mono_16k
    import soxr
    t = np.arange(4410, dtype=np.float32) / 44100
    stereo = np.stack((.2 * np.sin(2 * np.pi * 220 * t), .6 * np.cos(2 * np.pi * 440 * t)), axis=1)
    for channel, selected in (("left", stereo[:, 0]), ("right", stereo[:, 1]), ("mean", stereo.mean(axis=1))):
        result = to_mono_16k(stereo, 44100, channel)
        expected = np.ascontiguousarray(soxr.resample(selected, 44100, 16000, quality="HQ"), dtype=np.float32)
        assert result.flags.c_contiguous and result.shape == (1600,)
        assert np.array_equal(result, expected)


def test_infill_empty_eos_middle_can_delete_an_interval_without_losing_edges(tokenizer, config):
    engine = DramaEngine("model", "codec")
    engine.load_model = lambda: SimpleNamespace(tokenizer=tokenizer, config=config)
    empty = take(count=0, metadata={"finish_reason": "eos", "duration_seconds": 0})
    captured = []
    engine._take = lambda *a, **kw: captured.append(kw) or empty
    source = take(value=3, count=10)
    result = engine.infill(source, "[pause]", 3 * 2048 / 44100, 7 * 2048 / 44100)
    assert captured[0]["allow_empty"]
    assert result.codes.shape[-1] == 6 and torch.equal(result.codes, torch.cat((source.codes[:, :3], source.codes[:, 7:]), 1))
    assert result.metadata["inserted_frames"] == 0 and result.metadata["preserved_prefix_suffix_codes"]
