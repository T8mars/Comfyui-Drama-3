"""E-only proposal: actual libsndfile formats, native graph and exact PCM cuts."""
import copy
import hashlib
import io
from pathlib import Path
import wave
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from drama3_t8 import render_infill as ri
from drama3_t8.editor import fingerprint
from drama3_t8.state import Session, save_session, load_session, validate_session
from drama3_t8.state import Take
from drama3_t8.dialogue import audio_frames
from drama3_t8.engine import DramaEngine
from test_source_aware_infill import Engine, source


def bits(tensor):
    return tensor.contiguous().view(torch.int32)


def test_actual_pcm24_writer_high_bits_and_packed_integer_hash_are_exact():
    values = torch.tensor([[-2., -1., -1+2**-23, -.5, -2**-23, -2**-24, 0., 2**-24, 2**-23, .5, 1-2**-23, 1., 2.]])
    raw = io.BytesIO()
    sf.write(raw, values.T.numpy(), 44100, format="WAV", subtype="PCM_24")
    encoded = raw.getvalue()
    raw.seek(0)
    native, rate = sf.read(raw, dtype="int32", always_2d=True)
    assert rate == 44100 and not (native & 255).any()
    expected = torch.from_numpy((native.T.copy() >> 8).astype(np.int32))
    canonical, integer = ri.quantize_native(values, "PCM_24")
    assert integer.dtype == torch.int32 and torch.equal(integer, expected)
    assert torch.equal(bits(canonical), bits(expected.float()/8388608))
    assert torch.equal(bits(ri.canonical_pcm(canonical, "PCM_24")), bits(canonical))
    with wave.open(io.BytesIO(encoded), "rb") as actual:
        assert actual.getsampwidth() == 3
        payload = actual.readframes(actual.getnframes())
    digest = hashlib.sha256(f"44100:1:{values.shape[-1]}:PCM_24:".encode("ascii") + payload).hexdigest()
    identity = ri.playback_identity(values, "PCM_24")
    assert identity["pcm_sha256"] == identity["pcm24_sha256"] == digest
    assert "pcm16_sha256" not in identity
    assert identity["pcm_sha256"] != ri.playback_identity(values, "FLOAT")["pcm_sha256"]
    assert identity["pcm_sha256"] != ri.playback_identity(values, "PCM_16")["pcm_sha256"]


@pytest.mark.parametrize("subtype,dtype", [("FLOAT", "float32"), ("PCM_16", "int16"), ("PCM_24", "int32")])
def test_three_real_asset_formats_keep_every_unselected_sample_and_roundtrip_source_graph(tmp_path, subtype, dtype):
    engine, parent = Engine(), source()
    before_path, after_path = tmp_path / "selected.wav", tmp_path / "edited.wav"
    sf.write(before_path, engine.original.T.numpy(), 44100, subtype=subtype)
    selected, rate = sf.read(before_path, dtype="float32", always_2d=True)
    engine.original = torch.from_numpy(selected.T.copy())
    expected = ri.playback_identity(engine.original, subtype, source=fingerprint(parent))
    result = ri.infill_take(engine, parent, "A: First\nB: Completely new selected sentence", 9000/44100, 13000/44100,
                           target_speaker=1, expected_view=expected)
    policy = result.metadata["render_infill"]
    output = ri.decode_rendered(result)["waveform"][0]
    assert policy["subtype"] == subtype and policy["source_view"] == expected
    assert policy["target_speaker"] == 1 and policy["rendered_pcm_encoding"] is True
    assert torch.equal(bits(output[:, :9000]), bits(engine.original[:, :9000]))
    assert torch.equal(bits(output[:, 9000+4096:]), bits(engine.original[:, 13000:]))
    sf.write(after_path, output.T.numpy(), 44100, subtype=subtype)
    before_values, _ = sf.read(before_path, dtype=dtype, always_2d=True)
    after_values, _ = sf.read(after_path, dtype=dtype, always_2d=True)
    if subtype == "FLOAT":
        before_values, after_values = before_values.view(np.int32), after_values.view(np.int32)
    assert np.array_equal(after_values[:9000], before_values[:9000])
    assert np.array_equal(after_values[9000+4096:], before_values[13000:])
    archive = tmp_path / "self-contained.safetensors"
    save_session(Session(takes=(result,)), archive)
    restored = load_session(archive).takes[0]
    assert restored.metadata == result.metadata
    assert torch.equal(bits(ri.decode_rendered(restored)["waveform"]), bits(output[None]))
    assert restored.sources[0].take_id == parent.take_id
    assert restored.sources[1].metadata["history_eligible"] is False
    assert torch.equal(engine.encoded[-1], output)
    if subtype == "PCM_24":
        assert policy["pcm24_sha256"] == policy["pcm_sha256"] and "pcm16_sha256" not in policy


def test_second_pcm24_infill_preserves_nested_pcm_and_original_sources(tmp_path):
    engine, parent = Engine(), source()
    first = ri.infill_take(engine, parent, "A: First\nB: Second edited sentence", 9000/44100, 13000/44100,
                           target_speaker=1, playback_subtype="PCM_24")
    path = tmp_path / "first.safetensors"
    save_session(Session(takes=(first,)), path)
    restored = load_session(path).takes[0]
    before = ri.decode_rendered(restored)["waveform"][0]
    expected = ri.playback_identity(before, "PCM_24", source=fingerprint(restored))
    second = ri.infill_take(engine, restored, "A: First\nB: Third edited sentence", 11000/44100, 16000/44100,
                            target_speaker=1, expected_view=expected)
    after = ri.decode_rendered(second)["waveform"][0]
    assert torch.equal(bits(after[:, :11000]), bits(before[:, :11000]))
    assert torch.equal(bits(after[:, 11000+4096:]), bits(before[:, 16000:]))
    save_session(Session(takes=(second,)), tmp_path / "second.safetensors")
    final = load_session(tmp_path / "second.safetensors").takes[0]
    assert final.sources[0].sources[0].take_id == parent.take_id
    assert torch.equal(bits(ri.decode_rendered(final)["waveform"]), bits(after[None]))


@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_16", "PCM_24"])
def test_actual_engine_raw_take_expected_view_forces_pcm_source_aware_route(tmp_path, subtype):
    engine = Engine()
    engine.original = torch.linspace(-.45, .45, 12*2048)[None]
    parent = Take("<|speaker:0|> Original complete utterance", "Original complete utterance",
                  audio_frames(torch.ones((10, 12), dtype=torch.long)),
                  metadata={"task": "tts", "creative_global_speaker": 7, "creative_local_speaker": 0})
    asset = tmp_path / "selected.wav"
    sf.write(asset, engine.original.T.numpy(), 44100, subtype=subtype)
    actual, _ = sf.read(asset, dtype="float32", always_2d=True)
    engine.original = torch.from_numpy(actual.T.copy())
    def decode_take(take, with_timeline=False):
        audio = ri.decode_rendered(take) if "render_infill" in take.metadata else {
            "waveform": engine.original[None].clone(), "sample_rate": 44100}
        return (audio, []) if with_timeline else audio
    engine.decode_take = decode_take
    expected = ri.playback_identity(engine.original, sf.info(asset).subtype, source=fingerprint(parent))
    result = DramaEngine.infill(engine, parent, "Complete revised utterance", 9000/44100, 13000/44100,
                                target_speaker=0, expected_view=expected)
    assert "render_infill" in result.metadata and result.sources[0].take_id == parent.take_id
    assert result.metadata["render_infill"]["target_speaker"] == 0
    output = ri.decode_rendered(result)["waveform"][0]
    assert torch.equal(bits(output[:, :9000]), bits(engine.original[:, :9000]))
    assert torch.equal(bits(output[:, 9000+4096:]), bits(engine.original[:, 13000:]))


@pytest.mark.parametrize("corruption", ["hash", "integer_alias", "subtype", "off_grid", "parent_pcm"])
def test_pcm24_archive_rejects_wrong_hash_quantization_and_false_format(corruption):
    result = ri.infill_take(Engine(), source(), "A: First\nB: New words", 9000/44100, 13000/44100,
                           target_speaker=1, playback_subtype="PCM_24")
    metadata, blobs = copy.deepcopy(result.metadata), {name: tensor.clone() for name, tensor in result.blobs.items()}
    policy = metadata["render_infill"]
    if corruption == "hash": policy["pcm24_sha256"] = "0" * 64
    elif corruption == "integer_alias": policy["pcm16_sha256"] = policy["pcm_sha256"]
    elif corruption == "subtype": policy["subtype"] = "FLOAT"
    elif corruption == "off_grid": blobs["render_pcm"][0, 0] = 2**-24
    else: blobs["parent_pcm"][0, 0] += 2**-23
    with pytest.raises(ValueError):
        validate_session(Session(takes=(replace(result, metadata=metadata, blobs=blobs),)))


def test_pcm24_hash_rejects_left_aligned_or_out_of_range_integers():
    with pytest.raises(ValueError, match="24-bit integers"):
        ri.pcm_hash(torch.tensor([[8388608]], dtype=torch.int32), "PCM_24")
    with pytest.raises(ValueError, match="24-bit integers"):
        ri.pcm_hash(torch.tensor([[1]], dtype=torch.int64), "PCM_24")
