"""CPU fault injection for expert pinning; no CUDA allocation or model inference.

The tagged tensors below simulate successful page locking while retaining normal
CPU storage. This tests allocation control, physical accounting and fallback;
actual CUDA/page-lock performance remains an integration check.
"""
import concurrent.futures
import threading
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
from drama3_t8.native import offload_accelerated as offload
from drama3_t8.native.offload import FP8Experts

GiB = 1024 ** 3


class LockedCPU(torch.Tensor):
    """A CPU-only test double for the successful pin allocator's result."""

    def is_pinned(self, *args, **kwargs):
        return True

    def copy_(self, source, *args, **kwargs):
        if getattr(self, "fail_copy", False):
            raise RuntimeError("injected pin copy failure")
        return super().copy_(source, *args, **kwargs)


@pytest.fixture
def stores():
    opened = []

    def make(**kwargs):
        store = offload.ExpertStore(**kwargs)
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


@pytest.fixture
def fp8():
    result = FP8Experts(2, 128, 128, device="cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if name.endswith("weight"):
                parameter.view(torch.uint8).fill_(7)
            else:
                parameter.fill_(1.)
    return result


def virtual_memory(available=60 * GiB, total=64 * GiB):
    return SimpleNamespace(available=available, total=total)


def fake_allocator(monkeypatch, *, fail_allocate=None, fail_copy=None):
    original = torch.empty
    calls = []

    def allocate(*args, **kwargs):
        if not kwargs.pop("pin_memory", False):
            return original(*args, **kwargs)
        store = offload._PIN_STORE.get()
        calls.append(store)
        if fail_allocate is not None and fail_allocate(store):
            raise RuntimeError("injected pin allocation failure")
        tensor = torch.Tensor._make_subclass(LockedCPU, original(*args, **kwargs), False)
        tensor.fail_copy = fail_copy is not None and fail_copy(store)
        return tensor

    monkeypatch.setattr(offload.torch, "empty", allocate)
    return calls


def all_fields(store):
    return [tensor for banks in store.banks for bank in banks for tensor in bank.fields.values()]


def test_low_memory_stops_all_later_pin_attempts_and_reports_exact_bank_bytes(monkeypatch, stores, fp8):
    checks = []
    monkeypatch.setattr(offload.psutil, "virtual_memory", lambda: checks.append(1) or virtual_memory(GiB))
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    expected = fp8.gate_up_weight.nbytes + fp8.gate_up_scale.nbytes
    assert not calls and len(checks) == 1
    assert store.pin_last_check == {"bank": "layer 0 gate/up", "physical_bytes": expected,
        "available_bytes": GiB, "required_bytes": 2 * expected + 8 * GiB, "reserve_bytes": 8 * GiB}
    assert store.pin_requested and not store.pin_memory and not store.sync_free
    assert store.last_backend == "torch" and store.pinned_bytes == 0
    assert "source + pinned copy" in store.pin_fallback_reason
    assert store.fallback_reason == store.pin_fallback_reason
    reason = store.pin_fallback_reason
    store.add(fp8)
    assert not calls and len(checks) == 1 and store.pin_fallback_reason == reason
    assert all(not tensor.is_pinned() for tensor in all_fields(store))
    assert store.banks[0][0].fields["weight"].data_ptr() == fp8.gate_up_weight.data_ptr()


@pytest.mark.parametrize("drop_at", [3, 4])
def test_ram_drop_between_fields_or_banks_retains_previously_pinned_data(monkeypatch, stores, fp8, drop_at):
    probes = []

    def memory():
        probes.append(1)
        return virtual_memory(60 * GiB if len(probes) < drop_at else GiB)

    monkeypatch.setattr(offload.psutil, "virtual_memory", memory)
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    fields = all_fields(store)
    pinned_count = drop_at - 2
    assert len(calls) == pinned_count
    assert [t.is_pinned() for t in fields] == [True] * pinned_count + [False] * (4 - pinned_count)
    assert store.pinned_bytes == sum(t.nbytes for t in fields[:pinned_count])
    assert torch.equal(fields[0].as_subclass(torch.Tensor), fp8.gate_up_weight.view(torch.uint8).flatten(0, 1))
    reason = store.pin_fallback_reason
    store.add(fp8)
    assert len(calls) == pinned_count and len(probes) == drop_at
    assert store.fallback_reason == reason
    if drop_at == 3:
        assert store.pin_last_check["physical_bytes"] == fp8.gate_up_scale.nbytes
    else:
        assert store.pin_last_check["bank"] == "layer 0 down"


@pytest.mark.parametrize("failure", ["allocate", "copy"])
def test_pin_failure_disables_store_without_dropping_or_corrupting_source(monkeypatch, stores, fp8, failure):
    monkeypatch.setattr(offload.psutil, "virtual_memory", virtual_memory)
    calls = fake_allocator(monkeypatch,
        fail_allocate=(lambda store: True) if failure == "allocate" else None,
        fail_copy=(lambda store: True) if failure == "copy" else None)
    store = stores()
    store.add(fp8)
    store.add(fp8)
    assert len(calls) == 1 and not store.pin_memory and store.pinned_bytes == 0
    assert f"injected pin {('allocation' if failure == 'allocate' else 'copy')} failure" in store.fallback_reason
    assert store.fallback_reason == store.pin_fallback_reason
    for layer in store.banks:
        assert torch.equal(layer[0].fields["weight"], fp8.gate_up_weight.view(torch.uint8).flatten(0, 1))
        assert all(not t.is_pinned() for bank in layer for t in bank.fields.values())


def test_normal_pin_branch_copies_every_field_and_counts_physical_bytes(monkeypatch, stores, fp8):
    monkeypatch.setattr(offload.psutil, "virtual_memory", virtual_memory)
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    assert len(calls) == 4 and store.pin_checks == 6
    assert store.pin_memory and store.pin_fallback_reason is None and store.fallback_reason is None
    assert all(t.is_pinned() for t in all_fields(store))
    assert store.pinned_bytes == sum(p.nbytes for p in fp8.parameters())


@pytest.mark.parametrize("total,reserve", [(16, 4), (32, 4), (64, 8), (128, 8)])
def test_system_reserve_is_bounded(monkeypatch, stores, total, reserve):
    store = stores()
    monkeypatch.setattr(offload.psutil, "virtual_memory", lambda: virtual_memory(60 * GiB, total * GiB))
    assert offload._pin_headroom(store, 123, "fixture")
    assert store.pin_last_check["reserve_bytes"] == reserve * GiB
    assert store.pin_last_check["required_bytes"] == reserve * GiB + 246


def test_failed_memory_probe_prevents_pin_allocation(monkeypatch, stores, fp8):
    def unavailable():
        raise OSError("injected RAM probe failure")

    monkeypatch.setattr(offload.psutil, "virtual_memory", unavailable)
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    assert not calls and not store.pin_memory
    assert "cannot verify available RAM" in store.fallback_reason
    assert "injected RAM probe failure" in store.fallback_reason


@pytest.mark.parametrize("kind,bits", [("int8", 8), ("w4", 4), ("w6", 6)])
def test_quantized_bank_budget_uses_packed_data_and_scales_not_logical_bf16(monkeypatch, stores, kind, bits):
    def module(rows):
        physical_rows, width = rows * 2, 256
        if kind == "int8":
            layout = "TensorWiseINT8Layout"
            params = TensorWiseINT8Layout.Params(scale=torch.ones(physical_rows, 1),
                orig_shape=(physical_rows, width), orig_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256)
        else:
            layout = "AsymW4A8Int8Layout"
            params = AsymW4A8Int8Layout.Params(
                scale=torch.ones(physical_rows, width // 16).to(torch.float8_e4m3fn),
                s_channel=torch.ones(physical_rows), orig_shape=(physical_rows, width), orig_dtype=torch.bfloat16,
                codebook=torch.linspace(-7, 7, 16) if bits == 4 else None, group_size=16, convrot_groupsize=256)
        weight = QuantizedTensor(torch.zeros(physical_rows, width * bits // 8, dtype=torch.int8), layout, params)
        return SimpleNamespace(weight=weight, num_experts=2)

    experts = SimpleNamespace(gate_up_proj=module(512), down_proj=module(256))
    physical = sum(t.nbytes for t in offload.per_row_fields(experts.gate_up_proj.weight).values())
    logical = experts.gate_up_proj.weight.numel() * experts.gate_up_proj.weight.element_size()
    assert physical < logical
    monkeypatch.setattr(offload.psutil, "virtual_memory", lambda: virtual_memory(8 * GiB + 2 * physical + 1))
    calls = fake_allocator(monkeypatch)
    records = []
    original = offload._pin_headroom

    def check(store, amount, label):
        records.append((amount, label))
        return original(store, amount, label)

    monkeypatch.setattr(offload, "_pin_headroom", check)
    store = stores()
    store.add(experts)
    assert records[0] == (physical, "layer 0 gate/up")
    assert store.pin_memory and store.pin_fallback_reason is None and len(calls) == (4 if kind == "int8" else 6)
    assert store.pinned_bytes == sum(t.nbytes for t in all_fields(store))
    assert all(bank.kernel == kind for bank in store.banks[0])


def test_already_pinned_and_explicitly_unrequested_tensors_skip_checks(monkeypatch, stores):
    def forbidden():
        raise AssertionError("already pinned or unrequested tensor must not query RAM")

    monkeypatch.setattr(offload.psutil, "virtual_memory", forbidden)
    calls = fake_allocator(monkeypatch)
    source = torch.arange(10)
    locked = torch.Tensor._make_subclass(LockedCPU, source, False)
    assert offload.pinned(source) is source
    store = stores(pin_memory=False)
    owner, requested = offload._PIN_STORE.set(store), offload._PIN_REQUESTED.set(True)
    try:
        assert offload.pinned(locked) is locked
        assert offload.pinned(source) is source
    finally:
        offload._PIN_STORE.reset(owner)
        offload._PIN_REQUESTED.reset(requested)
    assert not calls and store.pin_checks == 0


def test_existing_locked_bank_is_preserved_when_ram_is_low(monkeypatch, stores, fp8):
    for name, parameter in list(fp8.named_parameters()):
        tagged = torch.Tensor._make_subclass(LockedCPU, parameter.detach(), False)
        setattr(fp8, name, torch.nn.Parameter(tagged, requires_grad=False))
    assert all(p.is_pinned() for p in fp8.parameters())
    monkeypatch.setattr(offload.psutil, "virtual_memory", lambda: pytest.fail("no new physical bytes to pin"))
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    assert not calls and store.pin_checks == 0 and store.pin_memory
    assert all(t.is_pinned() for t in all_fields(store))


def test_store_failure_is_isolated_across_threads_and_contexts_restore(monkeypatch, stores, fp8):
    bad, good = stores(), stores()
    monkeypatch.setattr(offload.psutil, "virtual_memory", virtual_memory)
    calls = fake_allocator(monkeypatch, fail_allocate=lambda store: store is bad)
    barrier = threading.Barrier(2)

    def load(store):
        sentinel = object()
        owner = offload._PIN_STORE.set(sentinel)
        bank = offload._PIN_BANK.set(sentinel)
        requested = offload._PIN_REQUESTED.set(False)
        try:
            barrier.wait(timeout=10)
            store.add(fp8)
            assert offload._PIN_STORE.get() is sentinel
            assert offload._PIN_BANK.get() is sentinel
            assert not offload._PIN_REQUESTED.get()
        finally:
            offload._PIN_STORE.reset(owner)
            offload._PIN_BANK.reset(bank)
            offload._PIN_REQUESTED.reset(requested)

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(load, store) for store in (bad, good)]
        for result in futures:
            result.result(timeout=15)
    assert calls.count(bad) == 1 and calls.count(good) == 4
    assert not bad.pin_memory and good.pin_memory
    assert bad.pin_fallback_reason and good.pin_fallback_reason is None
    assert all(t.is_pinned() for t in all_fields(good))
    assert offload._PIN_STORE.get() is None and offload._PIN_BANK.get() is None
    assert not offload._PIN_REQUESTED.get()


def test_add_exception_restores_context(monkeypatch, stores, fp8):
    def broken(*args):
        raise ValueError("invalid bank")

    monkeypatch.setattr(offload.psutil, "virtual_memory", virtual_memory)
    monkeypatch.setattr(offload, "FP8HostBank", broken)
    with pytest.raises(ValueError, match="invalid bank"):
        stores().add(fp8)
    assert offload._PIN_STORE.get() is None and offload._PIN_BANK.get() is None
    assert not offload._PIN_REQUESTED.get()


def test_generation_session_keeps_concrete_pinning_reason_and_reports_torch(monkeypatch, stores, fp8):
    monkeypatch.setattr(offload.psutil, "virtual_memory", lambda: virtual_memory(GiB))
    calls = fake_allocator(monkeypatch)
    store = stores()
    store.add(fp8)
    reason = store.pin_fallback_reason
    mm = offload.comfy.model_management
    monkeypatch.setattr(mm, "free_memory", lambda *args: None)
    monkeypatch.setattr(mm, "get_free_memory", lambda device: 4 * GiB)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(mm, "current_loaded_models", [])
    monkeypatch.setattr(offload, "SlotPool", lambda *args: SimpleNamespace(report=lambda: None))
    monkeypatch.setattr(offload.kernels, "available", lambda device: pytest.fail("unpinned banks cannot use CUDA"))
    with store.session(torch.device("cpu"), [], cpu_decode=True):
        assert not store.sync_free and store.last_backend == "torch"
        assert store.fallback_reason == reason
    assert not calls and store.cache is None
