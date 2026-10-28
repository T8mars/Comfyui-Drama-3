"""Qwen upstream-compatible quiet-point tolerance without checkpoint inference."""
import torch
import pytest

from drama3_t8.qwen3_asr import asr


@pytest.fixture
def small_limit(monkeypatch):
    # Scale only time constants; execute the real partitioning algorithm.
    monkeypatch.setattr(asr,'SAMPLE_RATE',100)
    monkeypatch.setattr(asr,'MAX_SECONDS',10)
    monkeypatch.setattr(asr,'MIN_SECONDS',.01)


@pytest.mark.parametrize('quiet_start', [850,1000,1200,None])
def test_qwen_chunks_keep_upstream_tolerance_and_all_samples(small_limit,quiet_start):
    from drama3_t8.vendor_qwen.inference.utils import split_audio_into_chunks
    raw=torch.ones(2501)
    if quiet_start is not None:
        raw[quiet_start:quiet_start+10]=0
    before=raw.clone()
    pieces=asr.split_long(raw)
    expected = split_audio_into_chunks(raw.numpy(), 100, 10)
    assert len(pieces) == len(expected)
    assert all(torch.equal(piece, torch.from_numpy(other)) for piece, (other, _) in zip(pieces, expected))
    assert all(0 < len(piece) <= 1500 for piece in pieces)
    assert torch.equal(torch.cat(pieces),raw)
    assert torch.equal(raw,before)


def test_qwen_uses_quieter_boundary_after_target_within_search_window(small_limit):
    raw=torch.ones(1500)
    raw[850:860]=.1
    raw[1200:1210]=0
    pieces=asr.split_long(raw)
    assert 1200 <= len(pieces[0]) < 1210
    assert torch.equal(torch.cat(pieces),raw)


def test_qwen_short_tail_padding_is_not_counted_as_input_loss(monkeypatch):
    monkeypatch.setattr(asr,'SAMPLE_RATE',100)
    monkeypatch.setattr(asr,'MAX_SECONDS',10)
    monkeypatch.setattr(asr,'MIN_SECONDS',.5)
    raw=torch.arange(10,dtype=torch.float32)+1
    pieces=asr.split_long(raw)
    assert len(pieces)==1 and len(pieces[0])==50
    assert torch.equal(pieces[0][:len(raw)],raw)
    assert torch.count_nonzero(pieces[0][len(raw):])==0
