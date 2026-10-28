"""CPU regression for legacy expert pin-copy admission under low RAM.

Exercise the real legacy selector + tiny FP8Experts + ExpertStore.add path.
Only capability/memory probes and the legacy pin-copy boundary are replaced.
The pin spy returns ordinary CPU clones: no page locking/CUDA/inference claimed.
Expected original source: two low-RAM cases fail on real pin-copy attempts;
the high-RAM torch control passes. No new API/import is required to obtain RED.
"""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import acceleration, kernels, offload

GiB = 1024 ** 3


@pytest.fixture
def legacy_case(monkeypatch):
    experts = offload.FP8Experts(2, 128, 128, device='cpu', dtype=torch.bfloat16)
    with torch.no_grad():
        experts.gate_up_weight.view(torch.uint8).fill_(7)
        experts.down_weight.view(torch.uint8).fill_(11)
        experts.gate_up_scale.fill_(1.)
        experts.down_scale.fill_(2.)
    fields = [
        {'weight': experts.gate_up_weight.view(torch.uint8).flatten(0, 1),
         'scale': experts.gate_up_scale.flatten(0, 1)},
        {'weight': experts.down_weight.view(torch.uint8).flatten(0, 1),
         'scale': experts.down_scale.flatten(0, 1)},
    ]
    snapshots = [{name: tensor.clone() for name, tensor in bank.items()} for bank in fields]
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))]))
    pin_calls = []

    def record_pin_copy(tensor):
        pin_calls.append(dict(data_ptr=tensor.data_ptr(), bytes=tensor.numel() * tensor.element_size()))
        return tensor.clone()

    def forbidden_kernel_probe(*args, **kwargs):
        pytest.fail('Low-memory CUDA rejection must happen before probing/using expert kernels')

    monkeypatch.setattr(offload, 'pinned', record_pin_copy)
    # Constructing a device descriptor does not initialize or allocate CUDA.
    monkeypatch.setattr(acceleration.mm, 'get_torch_device', lambda: torch.device('cuda'))
    monkeypatch.setattr(acceleration.mm, 'DISABLE_PINNED_MEMORY', False)
    monkeypatch.setattr(kernels, 'available', forbidden_kernel_probe)
    physical = sum(t.numel() * t.element_size() for bank in fields for t in bank.values())
    assert physical == acceleration.resident_bytes(experts)
    yield SimpleNamespace(experts=experts, model=model, fields=fields, snapshots=snapshots,
                          pin_calls=pin_calls, physical=physical)
    assert not torch.cuda.is_initialized()


def check_stored_values(case, store, *, same_storage):
    assert len(store.banks) == 1 and len(store.banks[0]) == 2
    for actual_bank, source_bank, snapshot in zip(store.banks[0], case.fields, case.snapshots):
        assert set(actual_bank.fields) == set(source_bank)
        for name, original in source_bank.items():
            actual = actual_bank.fields[name]
            assert actual.device.type == original.device.type == 'cpu'
            assert actual.dtype == original.dtype and actual.shape == original.shape
            assert torch.equal(actual.view(torch.uint8), snapshot[name].view(torch.uint8))
            assert torch.equal(original.view(torch.uint8), snapshot[name].view(torch.uint8))
            assert (actual.data_ptr() == original.data_ptr()) is same_storage


@pytest.mark.parametrize('requested', ['cuda', 'torch'])
def test_low_ram_legacy_fallback_add_borrows_contiguous_storage_without_pin_copy(monkeypatch, legacy_case, requested):
    case = legacy_case
    # Below even one full physical bank set, independently of reserve policy.
    available = case.physical // 2
    monkeypatch.setattr(acceleration.psutil, 'virtual_memory',
                        lambda: SimpleNamespace(available=available, total=64 * GiB))
    store = acceleration.expert_store(case.model, 1024, requested)
    assert type(store) is offload.ExpertStore
    assert store.requested_backend == requested
    if requested == 'cuda':
        assert 'available RAM' in store.fallback_reason
    try:
        store.add(case.experts)
        assert case.pin_calls == [], 'Legacy fallback copied full banks after insufficient-RAM selection'
        check_stored_values(case, store, same_storage=True)
    finally:
        store.close()


def test_high_ram_legacy_torch_selection_retains_existing_pin_copy_path(monkeypatch, legacy_case):
    case = legacy_case
    monkeypatch.setattr(acceleration.psutil, 'virtual_memory',
                        lambda: SimpleNamespace(available=64 * GiB, total=64 * GiB))
    store = acceleration.expert_store(case.model, 1024, 'torch')
    assert type(store) is offload.ExpertStore and store.requested_backend == 'torch'
    try:
        store.add(case.experts)
        assert len(case.pin_calls) == 4
        assert sum(call['bytes'] for call in case.pin_calls) == case.physical
        check_stored_values(case, store, same_storage=False)
    finally:
        store.close()
