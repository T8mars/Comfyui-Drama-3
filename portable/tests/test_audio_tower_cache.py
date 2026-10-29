"""Audio tower cache against an independent FP64 causal-attention oracle."""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native.cache import AudioTowerKV, AudioTowerState, TowerKV
from drama3_t8.native.model import Drama3Tower


def attention_oracle(query, key, value, positions):
    groups = query.shape[1] // key.shape[1]
    keys = key.double().repeat_interleave(groups, dim=1)
    values = value.double().repeat_interleave(groups, dim=1)
    scores = query.double() @ keys.transpose(-1, -2) * query.shape[-1] ** -.5
    visible = torch.arange(key.shape[2], device=query.device)[None, :] <= positions[:, None]
    return scores.masked_fill(~visible, -torch.inf).softmax(-1) @ values


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('groups', [1, 4])
def test_audio_cache_preserves_causal_prefix_through_mixed_decode_widths(dtype, groups):
    gen = torch.Generator().manual_seed(219)
    q = torch.randn(2, 2 * groups, 13, 16, generator=gen, dtype=dtype)
    k = torch.randn(2, 2, 13, 16, generator=gen, dtype=dtype)
    v = torch.randn(2, 2, 13, 16, generator=gen, dtype=dtype)
    state = AudioTowerState.zeros(2, 2, 32, 16, 'cpu', dtype,
        AudioTowerState.shared(32, 'cpu', dtype))
    state.prepare(8)
    state.append(k[:, :, :8], v[:, :, :8])
    state.advance(8)
    for count in (1, 3, 1):
        start, end = state.index, state.index + count
        state.prepare(count)
        actual = state.decode(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], 2)
        expected = attention_oracle(q[:, :, start:end], k[:, :, :end], v[:, :, :end], torch.arange(start, end))
        expected = expected.transpose(1, 2).reshape_as(actual).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        assert actual.dtype == dtype and actual.shape == (2, count, 2 * groups * 16)
        state.advance(count)
    assert state.index == 13
    torch.testing.assert_close(state.key[:, :, :13], k, rtol=0, atol=0)
    torch.testing.assert_close(state.value[:, :, :13], v, rtol=0, atol=0)
    with pytest.raises(ValueError, match='capacity'):
        state.prepare(20)


def test_folded_decode_masks_finite_stale_tail_slots():
    gen = torch.Generator().manual_seed(33)
    q = torch.randn(1, 8, 1, 16, generator=gen)
    k = torch.randn(1, 2, 6, 16, generator=gen)
    v = torch.randn(1, 2, 6, 16, generator=gen)
    state = AudioTowerState.zeros(1, 2, 24, 16, 'cpu', torch.float32,
        AudioTowerState.shared(24, 'cpu', torch.float32))
    state.key.fill_(1000)
    state.value.fill_(-1000)
    state.prepare(5)
    state.append(k[:, :, :5], v[:, :, :5])
    state.advance(5)
    state.prepare(1)
    actual = state.decode(q, k[:, :, 5:], v[:, :, 5:], 2)
    expected = attention_oracle(q, k, v, torch.tensor([5])).transpose(1, 2).reshape_as(actual).float()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert state.key[:, :, 6:].eq(1000).all()
    assert state.value[:, :, 6:].eq(-1000).all()


def test_audio_factory_uses_grouped_path_on_all_hosts_and_keeps_backbone_separate():
    cfg = SimpleNamespace(n_local_heads=2, head_dim=16)
    stub = SimpleNamespace(cfg=cfg, shared_layers=[None] * 4, residual_layers=[None] * 9)
    states = Drama3Tower.init_caches(stub, 32, 'cpu', torch.float32)
    assert AudioTowerKV is AudioTowerState
    assert len(states) == 13 and all(type(state) is AudioTowerState for state in states)
    assert states[0].bias is states[1].bias
    assert states[0].key is not states[1].key
    # The backbone continues to use the existing host/portable cache choice.
    from drama3_t8.native.model import FixedKVBias
    assert FixedKVBias is TowerKV

