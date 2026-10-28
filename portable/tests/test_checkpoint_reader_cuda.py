"""Real file-pread -> assign -> pinned expert ownership, scheduled explicitly.

These tests add checkpoint ownership and failure coverage to the existing CUDA
quantized operator suite. They do not repeat its Graph tests. RAM snapshots,
pin allocation/admission and numerical thresholds are never mocked. Collect
and run only when the GPU owner schedules them with sufficient actual RAM.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from safetensors.torch import save_file

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from drama3_t8.native import acceleration, checkpoint_reader, loader
from drama3_t8.native import offload_accelerated as offload
from test_checkpoint_reader import _expert_file, _expert_model, _region
from test_cuda_quantized_experts import _make_case, _reference, _cpu_threads, _errors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real page locking requires the scheduled CUDA host')


def _record(name, *, reset=False, **values):
    path = os.environ.get('DRAMA_CHECKPOINT_PIN_REPORT')
    if path:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if reset or name not in data:
            data[name] = {}
        data[name].update(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _ready():
    memory = offload.psutil.virtual_memory()
    reserve = max(offload.PIN_MIN_RESERVE, min(offload.PIN_MAX_RESERVE, int(memory.total * .125)))
    # Bound fixture construction/read/pin temporaries before allocating them.
    # Production reader and per-bank pin guards still make their own real checks.
    required = reserve + 512 * 1024 ** 2
    if memory.available < required:
        pytest.skip(f'REAL_RESOURCE_BLOCKED available={memory.available} required={required}; no mocked admission')
    torch.cuda.init()
    return dict(total=memory.total, available=memory.available, reserve=reserve, required=required)


def _host_stats():
    query = getattr(torch.cuda.memory, 'host_memory_stats', None)
    if query is None:
        return dict(available=False)
    values = query()
    names = ('allocated_bytes.current', 'allocated_bytes.peak',
             'reserved_bytes.current', 'reserved_bytes.peak',
             'allocation.current', 'allocation.allocated', 'allocation.freed',
             'num_host_alloc', 'num_host_free')
    return dict(available=True, **{name: values.get(name) for name in names})


def _assert_allocated_restored(before, after):
    # Reserved cache is intentionally reported separately. Never flush a host
    # cache, reset global counters, or count its reserved bytes as live tensors.
    if before.get('available') and after.get('available'):
        for name in ('allocated_bytes.current', 'allocation.current'):
            if before.get(name) is not None and after.get(name) is not None:
                assert after[name] == before[name], (name, before, after)


def _digest(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _bank_source(experts):
    """No strong Tensor refs escape this helper; shared codebooks stay shared."""
    entries = []
    for module in (experts.gate_up_proj, experts.down_proj):
        fields = offload.per_row_fields(module.weight)
        entries.append(dict(
            fields={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                               bytes=value.numel() * value.element_size(), sha256=_digest(value))
                    for name, value in fields.items()},
            qdata_pointer=module.weight._qdata.data_ptr(),
            row_refs=[weakref.ref(value) for value in fields.values()]))
    return entries


def _all_source_refs(model):
    return [weakref.ref(value) for _, value in acceleration.physical_tensors(model)]


def _destination_refs(store):
    return [weakref.ref(value) for banks in store.banks for bank in banks for value in bank.fields.values()]


def _assert_pinned_exact(source, destination):
    assert destination.fields.keys() == source['fields'].keys()
    for name, tensor in destination.fields.items():
        expected = source['fields'][name]
        assert tensor.device.type == 'cpu' and tensor.is_pinned()
        assert list(tensor.shape) == expected['shape'] and str(tensor.dtype) == expected['dtype']
        assert tensor.numel() * tensor.element_size() == expected['bytes']
        assert _digest(tensor) == expected['sha256']


def test_file_pread_assign_real_pin_releases_each_layer_source(tmp_path):
    _record('pread_assign_real_pin_ownership', reset=True, status='started_not_passed')
    admission = _ready()
    # Large source buffers make independent VirtualAlloc release observable on
    # Windows, instead of assuming tiny malloc blocks immediately unmap.
    path = _expert_file(tmp_path / 'ownership.safetensors', experts=64, width=512, intermediate=512)
    reader = {}
    state, metadata = checkpoint_reader.load_torch_file(path, return_metadata=True, diagnostics=reader)
    assert reader['actual_reader'] == 'official_cpu_pread', reader
    assert metadata == {'drama3_config': '{}', 'drama3_format': 'w4a8'}
    source_pointer = state['model.layers.0.mlp.experts.gate_up_proj.weight'].data_ptr()
    model = _expert_model(experts=64, width=512, intermediate=512)
    loader.load_checkpoint_storage(model, state)
    assert model.model.layers[0].mlp.experts.gate_up_proj.weight._qdata.data_ptr() == source_pointer
    state.clear()
    del state
    before = _host_stats()
    store = offload.ExpertStore(pin_memory=True)
    copied, destination_refs, status = [], [], 'failed'
    try:
        for index, layer in enumerate(model.model.layers):
            source = _bank_source(layer.mlp.experts)
            layer.mlp.experts = store.add(layer.mlp.experts)
            assert store.pin_memory and store.pin_fallback_reason is None, store.pin_fallback_reason
            for original, destination in zip(source, store.banks[index]):
                _assert_pinned_exact(original, destination)
            destination = None
            gc.collect()
            assert all(ref() is None for bank in source for ref in bank['row_refs'])
            if os.name == 'nt':
                for bank in source:
                    region = _region(bank['qdata_pointer'])
                    assert region.State == 0x10000 and region.AllocationBase is None  # MEM_FREE
            copied.append(dict(layer=index, source_rows_released=True,
                               physical_bytes=sum(field['bytes'] for bank in source for field in bank['fields'].values())))
        destination_refs = _destination_refs(store)
        live = _host_stats()
        expected = sum(row['physical_bytes'] for row in copied)
        assert store.pinned_bytes == expected
        if (before.get('available') and live.get('available') and
                before.get('allocated_bytes.current') is not None and live.get('allocated_bytes.current') is not None):
            assert live['allocated_bytes.current'] >= before['allocated_bytes.current'] + expected
        status = 'pin_copy_checks_complete_cleanup_assertions_pending'
    finally:
        store.close()
        gc.collect()
        after = _host_stats()
        _record('pread_assign_real_pin_ownership', status=status, admission=admission,
                reader=reader, copied=copied, before=before, after=after,
                live=locals().get('live'), actual_pinned_bytes=store.pinned_bytes,
                note='Real per-row fields pinned; shared tiny W4 codebooks intentionally remain separate CPU allocations. Reserved cache is not flushed.')
    assert not store.banks and all(ref() is None for ref in destination_refs)
    _assert_allocated_restored(before, after)
    _record('pread_assign_real_pin_ownership', status='passed')


@pytest.mark.parametrize('failure', ['cancel_after_bank', 'error_next_bank'])
def test_body_real_pin_cancel_error_preserves_exception_and_owned_cleanup(tmp_path, monkeypatch, failure):
    _record('real_pin_' + failure, reset=True, status='started_not_passed')
    admission = _ready()
    path = _expert_file(tmp_path / 'failure.safetensors', experts=4, intermediate=256)
    made, opened, source_refs, destination_refs = [], [], [], []
    readers = []
    original_reader, original_store = checkpoint_reader.load_torch_file, acceleration.expert_store
    error = OSError('original next-bank failure')

    def read(*args, **kwargs):
        result = original_reader(*args, **kwargs)
        readers.append(dict(kwargs['diagnostics']))
        assert kwargs['diagnostics']['actual_reader'] == 'official_cpu_pread', kwargs['diagnostics']
        return result

    def model(*args, **kwargs):
        result = _expert_model(experts=4, intermediate=256)
        made.append(result)
        return result

    def store(model, cache_bytes, requested, **kwargs):
        source_refs.extend(_all_source_refs(model))
        result = original_store(model, cache_bytes, requested, **kwargs)
        opened.append(result)
        assert isinstance(result, offload.ExpertStore), getattr(result, 'fallback_reason', None)
        original_add = result.add

        def add(experts):
            if failure == 'error_next_bank' and result.banks:
                raise error
            module = original_add(experts)
            assert result.pin_memory and result.pin_fallback_reason is None, result.pin_fallback_reason
            assert all(t.is_pinned() for bank in result.banks[-1] for t in bank.fields.values())
            destination_refs.extend(_destination_refs(result))
            if failure == 'cancel_after_bank':
                mm.interrupt_current_processing(True)
            return module

        result.add = add
        return result

    # Only replace the enormous architecture/patcher boundary and add an owned
    # cancellation/error trigger. Real file reader, assign, store selection,
    # system-memory snapshots, pin allocation and pin guards remain untouched.
    monkeypatch.setattr(loader, 'Drama3Model', model)
    monkeypatch.setattr(loader, 'parse_config', lambda _: SimpleNamespace())
    monkeypatch.setattr(loader.comfy.model_patcher, 'CoreModelPatcher',
                        lambda model, **kwargs: SimpleNamespace(model=model, **kwargs))
    monkeypatch.setattr(checkpoint_reader, 'load_torch_file', read)
    monkeypatch.setattr(acceleration, 'expert_store', store)
    before = _host_stats()
    mm.interrupt_current_processing(False)
    status = 'failed'
    try:
        expected = mm.InterruptProcessingException if failure == 'cancel_after_bank' else OSError
        with pytest.raises(expected) as retained:
            loader.load_drama3(str(path), experts_in_ram=True, moe_backend='cuda')
        if failure == 'error_next_bank':
            assert retained.value is error
        assert retained.value is not None  # Deliberately retain the traceback.
        gc.collect()
        assert opened and all(not item.banks for item in opened)
        assert source_refs and all(ref() is None for ref in source_refs)
        assert destination_refs and all(ref() is None for ref in destination_refs)
        assert not list(made[0].parameters()) and not list(made[0].buffers())
        status = 'exception_checks_complete_allocator_assertion_pending'
    finally:
        mm.interrupt_current_processing(False)
        for item in opened:
            item.close()
        gc.collect()
        after = _host_stats()
        _record('real_pin_' + failure, status=status, admission=admission, reader=readers,
                before=before, after=after, source_refs=len(source_refs), destination_refs=len(destination_refs),
                note='Original exception retained; only this test load/store cleaned, no allocator cache flush.')
    _assert_allocated_restored(before, after)
    _record('real_pin_' + failure, status='passed')


def test_file_pread_real_pin_one_w4_cuda_moe_matches_original_official_a8_oracle(tmp_path):
    _record('file_pread_pin_w4_single_moe', reset=True, status='started_not_passed')
    admission = _ready()
    if mm.current_loaded_models:
        pytest.skip('Run in an isolated test process; do not evict unrelated loaded models')
    device = torch.device('cuda:0')
    free, _ = torch.cuda.mem_get_info(device)
    if free < offload.CACHE_RESERVE + 256 * 1024 ** 2:
        pytest.skip(f'REAL_GPU_RESOURCE_BLOCKED free={free}; production cache reserve remains unchanged')
    # Reuse the original independent official A8 oracle/fixtures and EXACT
    # established thresholds. This is one file-to-pin decode, not another Graph suite.
    with _cpu_threads(), torch.inference_mode():
        case = _make_case('asym_w4a8_int8')
        entries = {}
        for bank, fields in [('gate_up_proj', case.gate), ('down_proj', case.down)]:
            entries.update({f'model.layers.0.mlp.experts.{bank}.{name}': value for name, value in fields.items()})
        entries['anchor'] = torch.arange(16, dtype=torch.float32)
        path = tmp_path / 'official-w4.safetensors'
        save_file(entries, path, metadata={'drama3_config': '{}', 'drama3_format': 'w4a8'})
        reader = {}
        state, metadata = checkpoint_reader.load_torch_file(path, return_metadata=True, diagnostics=reader)
        assert reader['actual_reader'] == 'official_cpu_pread', reader
        assert metadata['drama3_format'] == 'w4a8'
        for name, value in entries.items():
            assert state[name].dtype == value.dtype and state[name].shape == value.shape
            assert _digest(state[name]) == _digest(value)
        model = _expert_model(experts=16, intermediate=256, layers=1)
        loader.load_checkpoint_storage(model, state)
        state.clear()
        del state, entries
        before = _host_stats()
        store = offload.ExpertStore(1, pin_memory=True)
        status, metrics = 'failed', None
        try:
            model.model.layers[0].mlp.experts = store.add(model.model.layers[0].mlp.experts)
            assert store.pin_memory and store.pin_fallback_reason is None, store.pin_fallback_reason
            assert all(t.is_pinned() for bank in store.banks[0] for t in bank.fields.values())
            with store.session(device, [], cpu_decode=False):
                assert store.sync_free and store.last_backend == 'cuda', store.fallback_reason
                generator = torch.Generator().manual_seed(161803)
                x = torch.randn(1, 256, generator=generator)
                ids = torch.arange(8)[None]
                weights = torch.rand(1, 8, generator=generator).softmax(-1)
                expected = _reference(case, x, ids, weights)
                actual = model.model.layers[0].mlp.experts(x.to(device), ids.to(device), weights.to(device)).cpu()
                metrics = _errors(actual, expected)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            assert store.backend_calls['cuda'] == 1 and store.cpu_experts_run == 0
            status = 'numeric_checks_complete_allocator_assertion_pending'
        finally:
            store.close()
            torch.cuda.synchronize(device)
            gc.collect()
            after = _host_stats()
            _record('file_pread_pin_w4_single_moe', status=status, admission=admission,
                    reader=reader, before=before, after=after, errors=metrics,
                    tolerance=dict(atol=2e-6, rtol=2e-4), backend_calls=dict(store.backend_calls))
        _assert_allocated_restored(before, after)
        _record('file_pread_pin_w4_single_moe', status='passed')
