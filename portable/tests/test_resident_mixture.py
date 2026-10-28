"""Stable routed mixture uses bounded temporaries and one final dtype cast."""
import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import offload_accelerated as module


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_chunked_resident_mixture_matches_full_fp64_math_and_bounds_each_conversion(monkeypatch, dtype):
    generator = torch.Generator().manual_seed(608)
    values = torch.randn(23, 3, 7, generator=generator)
    weights = torch.rand(23, 3, generator=generator)
    expected = (values.double() * weights.double().unsqueeze(-1)).sum(1).to(dtype)
    monkeypatch.setattr(module, '_RESIDENT_MIXTURE_WORK_BYTES', 512)
    conversions = []
    original = torch.Tensor.double
    def observed(tensor, *args, **kwargs):
        conversions.append(tensor.numel() * 8)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'double', observed)
    actual = module._resident_weighted_sum(values, weights, dtype)
    assert torch.equal(actual, expected) and actual.dtype == dtype
    assert max(conversions) <= 512


def test_resident_mixture_preserves_small_result_between_cancelling_large_terms():
    values = torch.tensor([[[2.**24], [1.], [-2.**24]]])
    weights = torch.ones(1, 3)
    actual = module._resident_weighted_sum(values, weights, torch.float32)
    assert actual.item() == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Real resident mixture peak requires CUDA')
def test_8192_token_resident_mixture_bounds_gpu_temporary_peak_without_full_fp64_prompt():
    device = torch.device('cuda:0')
    with torch.inference_mode():
        values = torch.randn(8192, 8, 2048, device=device)
        weights = torch.rand(8192, 8, device=device).softmax(-1)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        actual = module._resident_weighted_sum(values, weights, torch.bfloat16)
        torch.cuda.synchronize(device)
        peak_extra = torch.cuda.max_memory_allocated(device) - before
        output_bytes = actual.numel() * actual.element_size()
        assert peak_extra <= output_bytes + 2 * module._RESIDENT_MIXTURE_WORK_BYTES + 4 * 1024**2
        rows = torch.tensor([0, 4096, 8191], device=device)
        expected = (values[rows].double() * weights[rows].double().unsqueeze(-1)).sum(1).to(torch.bfloat16)
        assert torch.equal(actual[rows], expected)
        from test_cuda_quantized_experts import _record
        _record('resident_8192_mixture_peak', tokens=8192, experts_per_token=8, hidden=2048,
                output_dtype=str(actual.dtype), output_bytes=output_bytes, peak_increment_bytes=peak_extra,
                f64_values_or_product_cap_bytes=module._RESIDENT_MIXTURE_WORK_BYTES)
