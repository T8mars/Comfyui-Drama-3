"""Production norm construction must preserve Fish Audio's FP32 formulas.

Reference: Fish Audio fish-speech commit
541566932abbaebcb0a40e562a1f846738383f38,
fish_speech/models/text2semantic/fish_omni/layers.py (lines 29-32, 50-55),
source SHA256 9ebf14c5fc58c8ab8dbbb00dca26c90ba55b7a6e0717166251a81334f0ff2b6d.
The two independent reference formulas below retain Fish Audio attribution;
license: ../licenses/LICENSE-FISH-AUDIO (Fish Audio Community License).
No .upstream checkout, model weights, CUDA, or candidate class name is needed.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine

standalone_engine()
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35Config
from drama3_t8.native.config import TowerConfig
from drama3_t8.native.loader import _OpsConfig
from drama3_t8.native.model import Drama3Model


@pytest.mark.parametrize('kind', ['zero_centered', 'gated'])
@pytest.mark.parametrize('location', ['backbone', 'tower'])
def test_graph_rejects_offloaded_norm_before_capture_without_moving_it(monkeypatch, kind, location):
    from drama3_t8.native.graphs import norm_residency_reason
    from drama3_t8.native.norms import GatedRMSNorm, ZeroCenteredRMSNorm
    def forbidden():
        raise AssertionError('Residency inspection must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    cls = ZeroCenteredRMSNorm if kind == 'zero_centered' else GatedRMSNorm
    norm = cls(8, device='cpu', dtype=torch.bfloat16)
    weight = norm.weight
    before = weight.detach().clone()
    layer = torch.nn.Sequential(norm) if location == 'backbone' else torch.nn.Identity()
    tower = norm if location == 'tower' else torch.nn.Identity()
    reason = norm_residency_reason([layer], [torch.device('cuda:0')], tower, torch.device('cuda:0'))
    assert location in reason and 'cpu' in reason and 'cuda:0' in reason
    assert norm.weight is weight and weight.device.type == 'cpu'
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_norm_residency_check_preserves_ram_expert_bank():
    from drama3_t8.native.graphs import norm_residency_reason
    layer = torch.nn.Module()
    layer.register_parameter('expert_bank', torch.nn.Parameter(torch.ones(2, 8)))
    original = layer.expert_bank
    assert norm_residency_reason([layer], [torch.device('cuda:0')], torch.nn.Identity(), 'cuda:0') is None
    assert layer.expert_bank is original and original.device.type == 'cpu'


def test_norm_residency_rejects_incomplete_layer_device_plan():
    from drama3_t8.native.graphs import norm_residency_reason
    assert 'plan' in norm_residency_reason([torch.nn.Identity()], [], torch.nn.Identity(), 'cuda:0')


def test_norm_residency_accepts_matching_cpu_parameters():
    from drama3_t8.native.graphs import norm_residency_reason
    from drama3_t8.native.norms import ZeroCenteredRMSNorm
    norm = ZeroCenteredRMSNorm(8, device='cpu')
    assert norm_residency_reason([norm], ['cpu'], norm, 'cpu') is None


def _official_zero_centered(x, weight, eps):
    # Fish Audio ZeroCenteredRMSNorm.forward: shift the weight in FP32.
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def _official_gated(x, gate, weight, eps):
    # Fish Audio GatedRMSNorm.forward: round the weighted norm before the
    # FP32 SiLU product, then cast the final output back to x.dtype.
    dtype = x.dtype
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = weight * out.to(dtype)
    return (out * F.silu(gate.float())).to(dtype)


def _production_model(dtype):
    text = Qwen35Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, partial_rotary_factor=.5, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, conv_kernel_size=4,
        layer_types=['linear_attention', 'full_attention'], rms_norm_eps=1e-6,
    )
    config = SimpleNamespace(
        text=text, fp8_experts=False, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        tower=TowerConfig(
            dim=32, n_head=4, n_local_heads=2, head_dim=8,
            intermediate_size=64, norm_eps=1e-6, num_codebooks=3,
            vocab_size=16, shared_layers=1, backbone_codebooks=1,
            rope_base=10000., rotary_dim=4,
        ),
        eos_token_id=0, semantic_start_token_id=1,
        semantic_end_token_id=16, output_token_ids=tuple(range(17)),
    )
    device = torch.device('cpu')
    # This is the actual detect_layer_quantization result for converted files
    # containing .comfy_quant entries. Follow load_drama3's pick_operations
    # branch, which supplies the real host MoEExperts unavailable on manual_cast.
    # Only norms are evaluated: no expert/dense weights or checkpoint are loaded.
    operations = comfy.ops.pick_operations(
        dtype, dtype, load_device=device, model_config=_OpsConfig({'mixed_ops': True}))
    return Drama3Model(config, device=device, dtype=dtype, operations=operations)


def _zero_centered_norms(model, component):
    if component == 'backbone':
        yield 'model.norm', model.model.norm
        for index, layer in enumerate(model.model.layers):
            prefix = f'model.layers.{index}'
            for name in ('input_layernorm', 'post_attention_layernorm'):
                yield f'{prefix}.{name}', getattr(layer, name)
            if layer.layer_type == 'full_attention':
                for name in ('q_norm', 'k_norm'):
                    yield f'{prefix}.self_attn.{name}', getattr(layer.self_attn, name)
    else:
        for group in ('shared_layers', 'residual_layers'):
            for index, layer in enumerate(getattr(model.tower, group)):
                prefix = f'tower.{group}.{index}'
                for name in ('enorm', 'hnorm', 'attention_norm', 'ffn_norm', 'final_layernorm'):
                    yield f'{prefix}.{name}', getattr(layer, name)
                for name in ('q_norm', 'k_norm'):
                    yield f'{prefix}.attention.{name}', getattr(layer.attention, name)


def _parameter_contract(model, path, norm, dtype):
    assert dict(model.named_modules())[path] is norm
    assert set(dict(norm.named_parameters(recurse=False))) == {'weight'}
    assert set(norm.state_dict()) == {'weight'}
    assert f'{path}.weight' in model.state_dict()
    assert norm.weight.dtype == dtype
    assert norm.weight.device == torch.device('cpu')
    return norm.weight.data_ptr(), norm.weight.detach().clone()


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('component', ['backbone', 'tower'])
def test_production_zero_centered_norms_match_official_fp32_formula(component, dtype):
    model = _production_model(dtype)
    with torch.inference_mode():
        for path, norm in _zero_centered_norms(model, component):
            # In BF16, 1 + 1/256 is a ties-to-even boundary and becomes 1 if
            # added before promotion. The low activation must retain the scale.
            norm.weight.fill_(1 / 256)
            pointer, weight = _parameter_contract(model, path, norm, dtype)
            width = norm.weight.numel()
            x = torch.tensor([1.5, .5, -1.5, -.5], dtype=dtype).repeat(width // 4)
            x = x.reshape(1, 1, width).repeat(2, 3, 1)
            expected = _official_zero_centered(x, weight, norm.eps)
            if dtype == torch.bfloat16:
                assert expected[0, 0, 1].item() == .44921875
            actual = norm(x)
            assert actual.dtype == dtype and actual.device == x.device
            torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                       msg=f'{path}: differs from official FP32 zero-centered norm')
            assert norm.weight.data_ptr() == pointer
            torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
            _parameter_contract(model, path, norm, dtype)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_production_deltanet_gate_matches_official_fp32_silu_order(dtype):
    model = _production_model(dtype)
    # Obtain the actual norm made by DramaDeltaNet, rather than importing a
    # proposed replacement class or independently constructing the host norm.
    norm = model.model.layers[0].linear_attn.norm
    path = 'model.layers.0.linear_attn.norm'
    with torch.inference_mode():
        norm.weight.fill_(1.5)
        pointer, weight = _parameter_contract(model, path, norm, dtype)
        x = torch.ones(2, norm.weight.numel(), dtype=dtype)
        gate = torch.full_like(x, .5)
        expected = _official_gated(x, gate, weight, norm.eps)
        if dtype == torch.bfloat16:
            # Rounding SiLU(.5) to BF16 before multiplying 1.5 instead gives
            # .46484375. Official FP32 gating gives the distinct .466796875.
            assert expected[0, 0].item() == .466796875
        actual = norm(x, gate)
        assert actual.dtype == dtype and actual.device == x.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg='DramaDeltaNet.norm: differs from official FP32 gate order')
        assert norm.weight.data_ptr() == pointer
        torch.testing.assert_close(norm.weight, weight, rtol=0, atol=0)
        _parameter_contract(model, path, norm, dtype)
