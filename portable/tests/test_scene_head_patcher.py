"""Actual ComfyUI ModelPatcher loading of the compact scene projection on CPU."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
from drama3_t8.engine import DramaEngine


@pytest.fixture
def cpu_model_management():
    # Collection can import ComfyUI before args.cpu is configured. Select its
    # official CPU backend locally, then restore the host for subsequent tests.
    previous = mm.cpu_state
    mm.cpu_state = mm.CPUState.CPU
    try:
        yield
    finally:
        mm.cpu_state = previous


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('source', ['compact', 'full_fallback'])
def test_scene_head_real_cpu_patcher_preserves_rows_logits_and_output_ids(tmp_path, source, dtype, cpu_model_management):
    # No patched ModelPatcher or mm.load_models_gpu: exercise the actual host
    # loader which used to resolve a root Linear parameter as '.weight'.
    assert mm.get_torch_device().type == 'cpu', 'The scoped official CPU backend must be active'
    weights = (torch.arange(15 * 8, dtype=torch.float32).reshape(15, 8) / 128).to(dtype)
    full = tmp_path / 'full.safetensors'
    small = tmp_path / 'small.safetensors'
    save_file({'weight': weights}, str(full))
    if source == 'compact':
        save_file({'weight': weights[[14, 13]], 'ids': torch.tensor([14, 13])}, str(small))
    engine = DramaEngine(tmp_path / 'unused-model.safetensors', tmp_path / 'unused-codec.safetensors',
                         line_head_path=small, text_head_path=full, scene_head_mode='compact',
                         moe_backend='torch', cuda_graph=False)
    audio_head = torch.nn.Linear(8, 3, bias=False, dtype=dtype)
    with torch.inference_mode():
        audio_head.weight.copy_(weights[[10, 11, 12]])
    engine.drama = SimpleNamespace(
        config=SimpleNamespace(output_token_ids=(10, 11, 12), text=SimpleNamespace(hidden_size=8)),
        tokenizer=SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[13]), token_to_id=lambda token: 14),
        patchers=lambda: [], parts=lambda: (None, None, None, audio_head, None))
    try:
        with torch.inference_mode():
            head, ids = engine.load_scene_head()
            assert ids == (10, 11, 12, 13, 14)
            patcher = engine.line_head_patcher
            assert isinstance(patcher, ModelPatcher)
            assert patcher.model is engine.line_head
            assert dict(engine.line_head.named_parameters()).keys() == {'0.weight'}
            assert torch.equal(engine.line_head[0].weight, weights[[13, 14]])
            # _load_list is real on both hosts and validates the named leaf.
            assert patcher._load_list()
            for hidden in (torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype) / 16,
                           torch.full((1, 8), -.125, dtype=dtype)):
                expected = F.linear(hidden, weights[[10, 11, 12, 13, 14]])
                assert torch.equal(head(hidden), expected)
            repeated, repeated_ids = engine.load_scene_head()
            assert repeated_ids == ids and engine.line_head_patcher is patcher
            assert torch.equal(repeated(hidden), expected)
    finally:
        mm.unload_all_models()
