"""Small real resident CUDA banks and hybrid backbone graphs; no full model load."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35Config
from drama3_t8.native import kernels
from drama3_t8.native.loader import load_checkpoint_storage
from drama3_t8.native.model import Experts, Drama3Backbone
from drama3_t8.native.offload import dequantize_fp8
from drama3_t8.native.resident import ResidentExperts, ResidentFP8Experts, ResidentMemoryError
from drama3_t8.native.graphs import StaticReplay, BackboneGraphs
from test_resident_experts import holder
from test_cuda_quantized_experts import _make_case, _reference, _cpu_threads, _record, _errors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real full-device residency needs CUDA')


def fp8():
    expert = ResidentFP8Experts(16, 256, 256, device='cpu', dtype=torch.float32)
    generator = torch.Generator().manual_seed(602)
    with torch.inference_mode():
        expert.gate_up_weight.copy_(torch.randn(expert.gate_up_weight.shape, generator=generator) * .4)
        expert.down_weight.copy_(torch.randn(expert.down_weight.shape, generator=generator) * .4)
        expert.gate_up_scale.fill_(.08)
        expert.down_scale.fill_(.08)
    gate = dequantize_fp8(expert.gate_up_weight.flatten(0, 1), expert.gate_up_scale.flatten(0, 1), torch.float32).view(16, 512, 256)
    down = dequantize_fp8(expert.down_weight.flatten(0, 1), expert.down_scale.flatten(0, 1), torch.float32).view(16, 256, 256)
    def reference(x, ids, weights):
        rows = []
        for row in range(len(x)):
            pieces = []
            for e, w in zip(ids[row].tolist(), weights[row]):
                g, u = F.linear(x[row:row + 1], gate[e]).chunk(2, -1)
                pieces.append(F.linear(F.silu(g) * u, down[e]) * w)
            rows.append(torch.stack(pieces).sum(0))
        return torch.cat(rows)
    return expert, reference


def quantized(fmt):
    case = _make_case(fmt)
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    expert = Experts(16, 256, 256, device='meta', dtype=torch.bfloat16, ops=ops)
    load_checkpoint_storage(expert.gate_up_proj, dict(case.gate))
    load_checkpoint_storage(expert.down_proj, dict(case.down))
    return expert, lambda x, ids, weights: _reference(case, x, ids, weights)


def managed(model, requested='cuda'):
    mm.archive_model_dtypes(model)
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch.device('cuda:0'), offload_device=torch.device('cpu'))
    controller = ResidentExperts(model, requested)
    return patcher, controller


def cleanup(patcher, controller):
    from drama3_t8.native.resident import _rollback_owned_patchers
    controller.close()
    _rollback_owned_patchers([patcher])
    mm.soft_empty_cache()


@pytest.mark.parametrize('fmt', ['fp8', 'int8_tensorwise', 'asym_w4a8_int8', 'w6a8_int8'])
def test_resident_cuda_prefill_and_graph_replay_match_independent_official_oracle(fmt):
    device = torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('NVRTC expert kernels unavailable')
    expert, reference = fp8() if fmt == 'fp8' else quantized(fmt)
    model = holder([expert])
    patcher, controller = managed(model)
    generator = torch.Generator().manual_seed(603)
    metrics = []
    def counted():
        controller.backend_calls['cuda'] += 1
    runtime = StaticReplay(device, on_replay=counted)
    try:
        with _cpu_threads(), torch.inference_mode(), controller.session(device, [patcher]):
            assert controller.sync_free and expert.sync_free(device)
            assert controller.stats()['actual_backend'] == 'not_run'
            for count in (5, 3):
                x = torch.randn(count, 256, generator=generator)
                ids = torch.tensor([[(row * 3 + j) % 16 for j in range(8)] for row in range(count)])
                weights = torch.rand(count, 8, generator=generator).softmax(-1)
                expected = reference(x, ids, weights)
                actual = expert(x.to(device), ids.to(device), weights.to(device)).cpu()
                metrics.append(dict(stage='prefill', **_errors(actual, expected)))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            for start in (0, 0, 0, 8, 8, 0, 8):
                x = torch.randn(1, 256, generator=generator)
                ids = torch.arange(start, start + 8)[None]
                weights = torch.rand(1, 8, generator=generator).softmax(-1)
                expected = reference(x, ids, weights)
                actual = runtime(expert, x.to(device), ids.to(device), weights.to(device)).clone().cpu()
                metrics.append(dict(stage='decode', start=start, **_errors(actual, expected)))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            assert runtime.captures == 1 and runtime.replays == 4 and runtime.fallback_reason is None
            assert controller.stats()['backend_calls'] == {'cuda': 9, 'torch': 0}
            assert controller.stats()['actual_backend'] == 'cuda'
            assert controller.cpu_experts_run == controller.cpu_decode_calls == 0 and controller.cpu_side is None
            assert not hasattr(controller, 'cache') and not hasattr(controller, 'banks')
        assert not controller.sync_free and not controller.active
        _record('resident_' + fmt, samples=metrics, stats=controller.stats(), captures=runtime.captures, replays=runtime.replays)
    finally:
        cleanup(patcher, controller)


@pytest.mark.parametrize('requested', ['torch', 'auto', 'cuda'])
def test_resident_torch_choice_or_unavailable_nvrtc_records_actual_fallback(requested, monkeypatch):
    expert, reference = fp8()
    patcher, controller = managed(holder([expert]), requested)
    monkeypatch.setattr(kernels, 'available', lambda device: False)
    try:
        with torch.inference_mode(), controller.session(patcher.load_device, [patcher]):
            assert not controller.sync_free
            x = torch.randn(1, 256)
            ids = torch.arange(8)[None]
            weights = torch.full((1, 8), .125)
            actual = expert(x.cuda(), ids.cuda(), weights.cuda()).cpu()
            torch.testing.assert_close(actual, reference(x, ids, weights), atol=2e-6, rtol=2e-4)
            stats = controller.stats()
            assert stats['actual_backend'] == 'torch' and stats['backend_calls'] == {'torch': 1, 'cuda': 0}
            assert bool(stats['fallback_reason']) == (requested != 'torch')
    finally:
        cleanup(patcher, controller)


def test_real_device_capacity_rejects_impossible_resident_reservation_before_loading():
    expert, _ = fp8()
    patcher, controller = managed(holder([expert]))
    try:
        with pytest.raises(ResidentMemoryError, match='streaming is not used'):
            controller.load(patcher.load_device, [patcher], memory_required=mm.get_total_memory(patcher.load_device) + 1)
        assert controller.last_backend == 'not_run'
        assert all(p.device.type == 'cpu' for p in patcher.model.parameters())
        assert not controller.sync_free
    finally:
        cleanup(patcher, controller)


def test_resident_hybrid_backbone_graphs_preserve_cache_and_backend_counts():
    device = torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('NVRTC expert kernels unavailable')
    text = Qwen35Config(vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=.5,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        conv_kernel_size=4, layer_types=['linear_attention'] * 3 + ['full_attention'])
    config = SimpleNamespace(text=text, fp8_experts=True, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, shared_expert_intermediate_size=128)
    torch.manual_seed(604)
    backbone = Drama3Backbone(config, device='cpu', dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    model = torch.nn.Module()
    model.dtype = torch.bfloat16
    model.model = backbone
    with torch.inference_mode():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape) * .05)
        for layer in backbone.layers:
            layer.mlp.experts.gate_up_scale.fill_(.1)
            layer.mlp.experts.down_scale.fill_(.1)
    patcher, controller = managed(model)
    outputs = []
    source = torch.randn(1, 16, 128, device=device, dtype=torch.bfloat16)
    try:
        with torch.inference_mode():
            for graph_mode in (False, True):
                with controller.session(device, [patcher]):
                    layers, devices = list(backbone.layers), [device] * 4
                    caches = backbone.init_layer_caches(64, layers, devices, torch.bfloat16)
                    graphs = BackboneGraphs(controller) if graph_mode else None
                    parts = [backbone.run_layers(source[:, :8].clone(), caches, layers, devices)]
                    for index in range(8, 16):
                        parts.append(backbone.run_layers(source[:, index:index + 1].clone(), caches, layers, devices, graphs=graphs).clone())
                    outputs.append(torch.cat(parts, dim=1))
                    assert all(cache.index == 16 for cache in caches)
                    if graphs:
                        assert graphs.report() == dict(captures=4, replays=20, fallback_reasons=[])
            torch.testing.assert_close(outputs[0], outputs[1], atol=.002, rtol=.002)
            assert controller.backend_calls == {'cuda': 72, 'torch': 0}
            _record('resident_hybrid_backbone', error=_errors(outputs[1], outputs[0]), stats=controller.stats(),
                    captures=4, replays=20, cache_positions=[cache.index for cache in caches])
    finally:
        cleanup(patcher, controller)


def test_resident_real_generator_cancellation_releases_session_and_recovery_matches_eager():
    from drama3_t8.native.config import TowerConfig
    from drama3_t8.native.loader import Drama3
    from drama3_t8.native.model import Drama3Model
    from drama3_t8.native.generate import generate
    device = torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('NVRTC expert kernels unavailable')
    text = Qwen35Config(vocab_size=252288, hidden_size=128, intermediate_size=128,
        max_position_embeddings=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, partial_rotary_factor=.5, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, conv_kernel_size=4,
        layer_types=['linear_attention'] * 3 + ['full_attention'])
    config = SimpleNamespace(text=text, fp8_experts=True, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, shared_expert_intermediate_size=128,
        tower=TowerConfig(128, 4, 2, 32, 256, 1e-5, 10, 16, 2, 4, 10000., 16),
        eos_token_id=248046, semantic_start_token_id=248099, semantic_end_token_id=248114,
        output_token_ids=(248046, *range(248099, 248115)))
    torch.manual_seed(606)
    model = Drama3Model(config, device='cpu', dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    with torch.inference_mode():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape) * .03)
        for layer in model.model.layers:
            layer.mlp.experts.gate_up_scale.fill_(.1)
            layer.mlp.experts.down_scale.fill_(.1)
    patcher, controller = managed(model)
    drama = Drama3(model, patcher, experts=controller)
    prompt = torch.zeros(2, 11, dtype=torch.long)
    prompt[:, 0] = torch.tensor([198, 100])
    def fixed(step, hidden, logits):
        logits.fill_(-100)
        logits[..., step + 1 if step < 9 else 0] = 100
    cancelled_frames = []
    def cancel_after_frame(step):
        cancelled_frames.append(step)
        if step == 4:
            mm.interrupt_current_processing(True)
    try:
        with _cpu_threads(), torch.inference_mode():
            drama.cuda_graph = False
            eager = generate(drama, prompt, 12, 1., 0, 1., 607, inspect_logits=fixed)
            drama.cuda_graph = True
            try:
                with pytest.raises(mm.InterruptProcessingException):
                    generate(drama, prompt, 12, 1., 0, 1., 607, inspect_logits=fixed, on_frame=cancel_after_frame)
            finally:
                mm.interrupt_current_processing(False)
            assert cancelled_frames == [0, 1, 2, 3, 4]
            assert not controller.active and not controller.sync_free
            recovered = generate(drama, prompt, 12, 1., 0, 1., 607, inspect_logits=fixed)
            assert torch.equal(recovered, eager) and eager.shape == (9, 11)
            assert drama.last_graph_stats['captures'] == 4
            assert drama.last_graph_stats['tower_captures'] == 1
            assert drama.last_graph_stats['tower_replays'] == 6
            assert not drama.last_graph_stats['fallback_reasons']
            assert drama.expert_stats()['actual_backend'] == 'cuda'
            assert drama.expert_stats()['cpu_experts_run'] == drama.expert_stats()['cpu_decode_calls'] == 0
            _record('resident_generator_cancel_recovery', cancelled_after_frames=len(cancelled_frames),
                    recovery_identical_to_eager=True, recovered_frames=list(recovered.shape),
                    graph=drama.last_graph_stats, stats=drama.expert_stats(), scope='small real generator, synthetic weights')
    finally:
        mm.interrupt_current_processing(False)
        cleanup(patcher, controller)
