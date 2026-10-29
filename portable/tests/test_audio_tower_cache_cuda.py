"""Real BF16 masked GQA and CUDA graph regressions, no model weights.

CPU suites exclude this CUDA-only file before collection.
"""
import pytest
import torch

from test_audio_tower_cache import attention_oracle
from drama3_t8.native.cache import AudioTowerState
from drama3_t8.native.graphs import StaticReplay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real audio tower CUDA contract')


def precision_measurement(cache_type, seed):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 16, 1, 256, generator=gen, dtype=torch.bfloat16).cuda()
    k = torch.randn(1, 2, 29, 256, generator=gen, dtype=torch.bfloat16).cuda()
    v = torch.randn(1, 2, 29, 256, generator=gen, dtype=torch.bfloat16).cuda()
    state = cache_type.zeros(1, 2, 210, 256, 'cuda', torch.bfloat16,
        cache_type.shared(210, 'cuda', torch.bfloat16))
    state.prepare(28)
    state.append(k[:, :, :28], v[:, :, :28])
    state.advance(28)
    state.prepare(1)
    out = state.decode(q, k[:, :, 28:], v[:, :, 28:], 2)
    exact = attention_oracle(q, k, v, torch.tensor([28], device='cuda')).transpose(1, 2).reshape_as(out)
    rounding = (exact.to(out.dtype).double() - exact).square().mean().sqrt().item()
    error = (out.double() - exact).square().mean().sqrt().item()
    return error, rounding


@pytest.mark.parametrize('seed', [23, 42, 219])
def test_BF16_masked_single_row_GQA_stays_near_output_rounding_floor(seed):
    before = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        # Reproduce the host setting that exposes the unequal-head math fallback.
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        with torch.inference_mode():
            error, rounding = precision_measurement(AudioTowerState, seed)
        assert torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        assert error <= max(rounding * 1.35, 2e-6), (error, rounding)
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(before)


def test_folded_audio_cache_graph_advances_mask_and_keys_like_eager():
    gen = torch.Generator().manual_seed(27)
    q = torch.randn(1, 16, 7, 256, generator=gen, dtype=torch.bfloat16).cuda()
    k = torch.randn(1, 2, 15, 256, generator=gen, dtype=torch.bfloat16).cuda()
    v = torch.randn(1, 2, 15, 256, generator=gen, dtype=torch.bfloat16).cuda()
    def cache():
        state = AudioTowerState.zeros(1, 2, 32, 256, 'cuda', torch.bfloat16,
            AudioTowerState.shared(32, 'cuda', torch.bfloat16))
        state.prepare(8)
        state.append(k[:, :, :8], v[:, :, :8])
        state.advance(8)
        return state
    with torch.inference_mode():
        eager, captured = cache(), cache()
        runner = StaticReplay(torch.device('cuda:0'))
        for step in range(7):
            end = 8 + step
            eager.prepare(1)
            captured.prepare(1)
            inputs = q[:, :, step:step+1], k[:, :, end:end+1], v[:, :, end:end+1]
            expected = eager.decode(*inputs, 2)
            actual = runner(lambda query, key, value: captured.decode(query, key, value, 2), *inputs).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            eager.advance(1)
            captured.advance(1)
        assert runner.captures == 1 and runner.replays >= 4
        assert captured.index == eager.index == 15
        torch.testing.assert_close(captured.key, eager.key, rtol=0, atol=0)
        torch.testing.assert_close(captured.value, eager.value, rtol=0, atol=0)
