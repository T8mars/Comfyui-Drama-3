"""Resident policy: physical budgets, storage ownership and failure cleanup."""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import resident
from drama3_t8.native.acceleration import physical_bytes, physical_tensors
from drama3_t8.native.loader import Drama3


def holder(experts):
    model = torch.nn.Module()
    model.dtype = torch.float32
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for module in experts:
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = module
        model.model.layers.append(layer)
    return model


def fp8_model():
    return holder([resident.ResidentFP8Experts(8, 128, 128, device='cpu', dtype=torch.float32)])


class Patcher:
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.load_device = torch.device(device)
        self.detached = 0

    def is_dynamic(self):
        return False

    def detach(self):
        self.detached += 1


def fake_memory(monkeypatch):
    monkeypatch.setattr(resident.mm, 'get_total_memory', lambda _: 8 * 1024**3)
    monkeypatch.setattr(resident.mm, 'get_free_memory', lambda _: 8 * 1024**3)
    monkeypatch.setattr(resident.mm, 'free_memory', lambda *args, **kwargs: [])
    monkeypatch.setattr(resident.mm, 'throw_exception_if_processing_interrupted', lambda: None)
    monkeypatch.setattr(resident.mm, 'current_loaded_models', [])


def test_controller_keeps_original_checkpoint_parameters_and_has_no_host_store():
    model = fp8_model()
    before = {name: tensor.data_ptr() for name, tensor in physical_tensors(model)}
    controller = resident.ResidentExperts(model, 'auto')
    after = {name: tensor.data_ptr() for name, tensor in physical_tensors(model)}
    assert before == after
    assert not hasattr(controller, 'cache') and not hasattr(controller, 'banks')
    assert controller.cpu_side is None and controller.stats()['cpu_experts_run'] == 0
    assert controller.stats()['actual_backend'] == 'not_run' and not controller.sync_free
    assert model.model.layers[0].mlp.experts._resident_controller is controller


def test_physical_budget_counts_packed_fields_buffers_and_shared_aliases_once():
    from test_cuda_quantized_experts import _make_case, _modules
    case = _make_case('asym_w4a8_int8')
    packed = _modules(case)
    model = torch.nn.Module()
    model.gate = packed.gate_up_proj
    model.down = packed.down_proj
    model.alias = packed.gate_up_proj
    model.register_buffer('extra', torch.ones(13))
    fields = list(physical_tensors(model))
    expected = sum(t.numel() * t.element_size() for _, t in fields)
    assert physical_bytes([model, model]) == expected
    assert physical_bytes([model], outside_device=torch.device('cpu')) == 0
    assert any('codebook' in name for name, _ in fields)
    assert any('s_channel' in name for name, _ in fields)
    assert any(name == 'extra' for name, _ in fields)
    assert physical_bytes([model]) < sum(p.numel() * p.element_size() for p in model.parameters())


def test_insufficient_total_vram_rejects_before_model_load_or_streaming(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model)
    fake_memory(monkeypatch)
    monkeypatch.setattr(resident.mm, 'get_total_memory', lambda _: 1024)
    monkeypatch.setattr(resident.mm, 'load_models_gpu', lambda *args, **kwargs: pytest.fail('Must reject before loading'))
    with pytest.raises(resident.ResidentMemoryError, match='streaming is not used'):
        controller.load(patcher.load_device, [patcher])
    assert patcher.detached == 0 and controller.last_backend == 'not_run' and not controller.sync_free


def test_insufficient_free_vram_preserves_preexisting_models(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model)
    fake_memory(monkeypatch)
    monkeypatch.setattr(resident.mm, 'get_free_memory', lambda _: 1024)
    monkeypatch.setattr(resident.mm, 'load_models_gpu', lambda *args, **kwargs: pytest.fail('Must reject before loading'))
    with pytest.raises(resident.ResidentMemoryError, match='not silently streamed'):
        controller.load(patcher.load_device, [patcher])
    assert patcher.detached == 0 and not controller.active


def test_force_full_load_is_verified_and_failure_only_cleans_owned_patchers(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model)
    fake_memory(monkeypatch)
    calls = []
    own = SimpleNamespace(model=patcher, model_unload=lambda: calls.append('owned_unload'))
    other = SimpleNamespace(model=Patcher(fp8_model()), model_unload=lambda: pytest.fail('Unrelated model'))
    resident.mm.current_loaded_models[:] = [other, own]
    def load(models, **options):
        calls.append(options)
        # Simulate a host ignoring force_full_load and keeping CPU fields.
    monkeypatch.setattr(resident.mm, 'load_models_gpu', load)
    with pytest.raises(resident.ResidentMemoryError, match='not achieved'):
        controller.load(patcher.load_device, [patcher])
    assert calls[0]['force_full_load'] is True
    assert calls[-1] == 'owned_unload'
    assert resident.mm.current_loaded_models == [other]
    assert patcher.detached == 1 and not controller.sync_free and controller.device is None


def test_load_cancellation_preserves_original_exception_and_detaches_partial_model(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model)
    fake_memory(monkeypatch)
    cancellation = KeyboardInterrupt('cancelled load')
    def cancel(*args, **kwargs):
        raise cancellation
    monkeypatch.setattr(resident.mm, 'load_models_gpu', cancel)
    with pytest.raises(KeyboardInterrupt) as caught:
        controller.load(patcher.load_device, [patcher])
    assert caught.value is cancellation and patcher.detached == 1
    assert not controller.active and not controller.sync_free


def test_allocator_oom_is_explained_without_streaming_and_partial_model_is_detached(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model)
    fake_memory(monkeypatch)
    failure = torch.cuda.OutOfMemoryError('simulated fragmented allocator')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(resident.mm, 'load_models_gpu', fail)
    with pytest.raises(resident.ResidentMemoryError, match='No streaming fallback') as caught:
        controller.load(patcher.load_device, [patcher])
    assert caught.value.__cause__ is failure and patcher.detached == 1
    assert not controller.sync_free and controller.last_backend == 'not_run'


def test_full_vram_loader_rejects_non_cuda_before_reading_a_large_checkpoint(monkeypatch):
    from drama3_t8.native import loader
    monkeypatch.setattr(loader.comfy.model_management, 'get_torch_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(loader.comfy.utils, 'load_torch_file', lambda *args, **kwargs: pytest.fail('No checkpoint allocation'))
    with pytest.raises(resident.ResidentMemoryError, match='CUDA GPU'):
        loader.load_drama3('unused-large-checkpoint.safetensors', experts_in_ram=False)


def test_checkpoint_capacity_preflight_rejects_before_tensor_loader_and_uses_bounded_header(monkeypatch, tmp_path):
    from drama3_t8.native import loader
    from safetensors.torch import save_file
    path = tmp_path / 'small.safetensors'
    save_file({'weight': torch.ones(32, 32)}, path, metadata={'drama3_config': '{}'})
    monkeypatch.setattr(loader.comfy.model_management, 'get_torch_device', lambda: torch.device('cuda:0'))
    monkeypatch.setattr(resident.mm, 'get_total_memory', lambda _: resident.RESIDENT_RESERVE + 4000)
    monkeypatch.setattr(loader.comfy.utils, 'load_torch_file', lambda *args, **kwargs: pytest.fail('No tensor loader call'))
    with pytest.raises(resident.ResidentMemoryError, match='no model tensors were loaded'):
        loader.load_drama3(path, experts_in_ram=False)
    path.write_bytes(b'\xff' * 8)
    with pytest.raises(ValueError, match='header size'):
        resident.preflight_resident_checkpoint(path, torch.device('cuda:0'))


def test_nested_session_cancellation_resets_flags_and_keeps_close_guard(monkeypatch):
    controller = resident.ResidentExperts(fp8_model())
    count = []
    def fake_load(device, patchers, memory_required=0):
        count.append('load')
        controller.device = device
        controller.selected_backend = 'cuda'
    monkeypatch.setattr(controller, 'load', fake_load)
    with pytest.raises(KeyboardInterrupt, match='cancel generation'):
        with controller.session(torch.device('cuda:0'), []):
            assert controller.sync_free
            with pytest.raises(RuntimeError, match='active'):
                controller.close()
            with controller.session(torch.device('cuda:0'), []):
                raise KeyboardInterrupt('cancel generation')
    assert count == ['load'] and not controller.active and not controller.sync_free
    controller.close()
    assert controller.closed


def test_non_cuda_or_dynamic_patcher_cannot_claim_full_vram():
    model = fp8_model()
    controller = resident.ResidentExperts(model)
    patcher = Patcher(model, 'cpu')
    with pytest.raises(resident.ResidentMemoryError, match='CUDA GPU'):
        controller.load(torch.device('cpu'), [patcher])
    patcher.load_device = torch.device('cuda:0')
    patcher.is_dynamic = lambda: True
    with pytest.raises(resident.ResidentMemoryError, match='non-dynamic'):
        controller.load(patcher.load_device, [patcher])


def test_drama_resident_api_includes_auxiliary_heads_and_unified_stats(monkeypatch):
    model = fp8_model()
    controller = resident.ResidentExperts(model, 'cuda')
    drama = object.__new__(Drama3)
    drama.model = model
    drama.patcher = Patcher(model)
    drama.second_patcher = None
    drama.experts = controller
    extra = Patcher(torch.nn.Linear(2, 2))
    calls = []
    monkeypatch.setattr(controller, 'load', lambda device, patchers, memory: calls.append((device, patchers, memory)))
    assert drama.load_resident([extra], memory_required=123) is drama
    assert calls == [(drama.patcher.load_device, [drama.patcher, extra], 123)]
    stats = drama.expert_stats()
    assert stats['placement'] == 'vram' and stats['requested_backend'] == 'cuda'
    assert stats['cpu_experts_run'] == stats['cpu_decode_calls'] == 0
    drama.experts = SimpleNamespace(requested_backend='auto', last_backend='torch', fallback_reason='test',
                                    backend_calls={'torch': 2}, cpu_experts_run=3, cpu_decode_calls=4)
    assert drama.expert_stats() == dict(placement='ram', requested_backend='auto', actual_backend='torch',
        fallback_reason='test', backend_calls={'torch': 2}, cpu_experts_run=3, cpu_decode_calls=4, sync_free=False)
    with pytest.raises(ValueError, match='VRAM'):
        drama.load_resident()
