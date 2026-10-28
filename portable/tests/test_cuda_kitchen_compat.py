"""Real CUDA binding error cleanup; arithmetic/Graph lives in host attention tests."""
import importlib

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import kitchen_compat as compat


def test_real_cuda_binding_type_error_releases_all_owned_descriptors():
    if not torch.cuda.is_available():
        pytest.skip('CUDA only')
    try:
        flash = importlib.import_module('comfy_kitchen.flash_attention')
    except ImportError:
        pytest.skip('Host does not provide the GQA CUDA attention entry point')
    if not hasattr(flash, 'flash_attention_decode_gqa_is_available') or not flash.flash_attention_decode_gqa_is_available(torch.device('cuda:0')):
        pytest.skip('Host does not provide the GQA CUDA attention entry point')
    before = len(compat._exports)
    q = torch.zeros(1, 16, 1, 256, dtype=torch.bfloat16, device='cuda')
    k = torch.zeros(1, 2, 32, 256, dtype=torch.bfloat16, device='cuda')
    # The actual nanobind entry point requires int32. No CUDA kernel is launched
    # after this type refusal; both consumed and unconsumed descriptors release.
    invalid_lengths = torch.ones(1, dtype=torch.int64, device='cuda')
    with pytest.raises(TypeError):
        compat.kitchen_attention('flash_attention_decode_gqa', q, k, k, invalid_lengths)
    assert len(compat._exports) == before
