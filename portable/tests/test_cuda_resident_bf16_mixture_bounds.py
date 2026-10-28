"""Real-CUDA long-prefill mixture peak; no model weights or audio inference.

Only grouped expert matrix computation is replaced by a preallocated CUDA tensor.
Production sorting, gather, reorder and BF16 weighted reduction remain active.
CPU-only suites should exclude this CUDA-specific file.
"""
import gc

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real CUDA allocation measurement required"
)


def test_resident_bf16_long_prefill_caps_mixture_peak_without_mutating_inputs():
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    from drama3_t8.native import offload_accelerated as accelerated

    device = torch.device("cuda:0")
    tokens, experts, hidden = 8192, 8, 1024
    work_budget = 4 * 1024 ** 2
    assert accelerated._RESIDENT_MIXTURE_WORK_BYTES == work_budget
    x = ids = weights = values = expected = actual = warm = None
    calls = []
    try:
        with torch.inference_mode():
            values = torch.ones((tokens * experts, hidden), device=device, dtype=torch.float32)
            x = torch.zeros((tokens, hidden), device=device, dtype=torch.bfloat16)
            ids = torch.arange(experts, device=device).expand(tokens, -1).contiguous()
            weights = torch.full((tokens, experts), 1 / experts, device=device, dtype=torch.float32)
            expected = torch.ones_like(x)
            input_identity = (values.data_ptr(), weights.data_ptr(), tuple(values.shape), tuple(weights.shape))

            def expert_tensor_fixture(sorted_x, groups, *unused):
                assert sorted_x.device == device and sorted_x.dtype == torch.bfloat16
                assert sorted_x.shape[1] == hidden and sorted_x.shape[0] % experts == 0
                calls.append(tuple(sorted_x.shape))
                return values[:sorted_x.shape[0]]

            with pytest.MonkeyPatch.context() as patches:
                patches.setattr(accelerated, "grouped_experts", expert_tensor_fixture)
                # Same production path for the two-token operator warmup.
                warm = accelerated.resident_prefill(
                    x[:2], ids[:2], weights[:2], None, None, "w4", (None, None), experts
                )
                assert torch.equal(warm, expected[:2])
                warm = None
                gc.collect()
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                actual = accelerated.resident_prefill(
                    x, ids, weights, None, None, "w4", (None, None), experts
                )
                torch.cuda.synchronize(device)
                increment = torch.cuda.max_memory_allocated(device) - before

                # Fixed contract from the actual 8192-token RED measurement:
                # required FP32 reorder + BF16 output + two4MiB work blocks
                # +4MiB allocator/operator allowance. Do not loosen this limit.
                reorder_bytes = values.numel() * values.element_size()
                output_bytes = expected.numel() * expected.element_size()
                limit = reorder_bytes + output_bytes + 2 * work_budget + 4 * 1024 ** 2
                assert limit == 297795584
                assert actual.dtype == torch.bfloat16
                assert actual.shape == (tokens, hidden)
                assert torch.equal(actual, expected)
                assert calls == [(16, hidden), (tokens * experts, hidden)]
                assert input_identity == (
                    values.data_ptr(), weights.data_ptr(), tuple(values.shape), tuple(weights.shape)
                )
                assert torch.all(weights == 1 / experts).item()
                assert torch.all(values == 1).item()
                assert increment <= limit, (
                    f"production resident_prefill peak increment {increment} exceeds fixed {limit} bytes"
                )
    finally:
        # The private MonkeyPatch context restores grouped_experts first.
        x = ids = weights = values = expected = actual = warm = None
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
