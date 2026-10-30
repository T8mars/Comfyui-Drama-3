from types import SimpleNamespace
import hashlib
import json

import pytest
import torch

from drama3_t8.reference_cache import encode_reference_cached
from drama3_t8.state import Reference


def encoder_at(tmp_path):
    codec = tmp_path / "codec.safetensors"
    codec.write_bytes(b"codec-version-one")
    engine = SimpleNamespace(codec_path=codec, calls=0)
    def encode(wave, rate, transcript, name, **options):
        engine.calls += 1
        return Reference(transcript, torch.ones((10, 4), dtype=torch.long), name,
                         processing={"trim_silence": options["trim_silence"], "normalize": options["normalize"]})
    engine.encode_reference = encode
    return engine


def test_reference_cache_reuses_codes_without_copying_transcript_and_tracks_codec(tmp_path):
    engine = encoder_at(tmp_path)
    wave = torch.ones((1, 1000), dtype=torch.float32) * .1
    first = encode_reference_cached(engine, wave, 16000, "第一稿", "甲", tmp_path / "cache")
    second = encode_reference_cached(engine, wave, 16000, "准确新稿", "乙", tmp_path / "cache")
    assert engine.calls == 1
    assert second.transcript == "准确新稿" and second.name == "乙"
    assert torch.equal(first.codes, second.codes)
    assert first.processing["encoding_cache"]["status"] == "miss"
    assert second.processing["encoding_cache"]["status"] == "hit"
    engine.codec_path.write_bytes(b"codec-version-two-changed")
    third = encode_reference_cached(engine, wave, 16000, "稿", "名", tmp_path / "cache")
    assert engine.calls == 2
    assert third.processing["codec_identity"] != first.processing["codec_identity"]


def test_corrupt_cache_rebuilds_and_preserves_evidence(tmp_path):
    engine = encoder_at(tmp_path)
    wave = torch.ones((1, 500), dtype=torch.float32)
    first = encode_reference_cached(engine, wave, 16000, "稿", "名", tmp_path / "cache")
    key = first.processing["encoding_cache"]["key"]
    receipt = tmp_path / "cache" / f"{key}.json"
    receipt.write_text("{broken", encoding="utf-8")
    rebuilt = encode_reference_cached(engine, wave, 16000, "新稿", "名", tmp_path / "cache")
    assert engine.calls == 2 and rebuilt.transcript == "新稿"
    assert list((tmp_path / "cache").glob("*.retired-*"))
    bad = wave.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        encode_reference_cached(engine, bad, 16000, "稿", "名", tmp_path / "cache")


@pytest.mark.parametrize("entry", [{"unexpected": 1}, [], {"signature": [1, 2, 3], "sha256": "invalid", "bytes": 1}])
def test_malformed_codec_identity_entry_is_rebuilt_without_losing_evidence(tmp_path, entry):
    from drama3_t8.reference_cache import codec_identity
    engine = encoder_at(tmp_path)
    root = tmp_path / "cache"
    root.mkdir()
    index = root / "codec-identities.json"
    damaged = json.dumps({str(engine.codec_path.resolve()): entry})
    index.write_text(damaged, encoding="utf-8")
    actual = codec_identity(engine.codec_path, root)
    assert actual == {"sha256": hashlib.sha256(engine.codec_path.read_bytes()).hexdigest(), "bytes": engine.codec_path.stat().st_size}
    assert next(root.glob("codec-identities.json.retired-*")).read_text(encoding="utf-8") == damaged
