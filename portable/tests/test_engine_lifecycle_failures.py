"""CPU fault contracts for head publication and best-effort idle unload.

Tiny real CPU ModelPatcher retries verify usable heads after a constructor
failure. Cleanup spies verify ownership/error semantics, never physical RAM.
"""
from types import SimpleNamespace
import logging

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
from drama3_t8 import engine as engine_module
from drama3_t8.engine import DramaEngine
from drama3_t8.native import acceleration


ENGINE_FIELDS = ('drama', 'codec', 'asr', 'text_head', 'text_head_patcher',
                 'line_head', 'line_head_patcher')


@pytest.fixture(autouse=True)
def cpu_model_management():
    # Same official host CPU selection as test_scene_head_patcher.py. In
    # particular, never select a CUDA device merely to construct a head.
    previous = mm.cpu_state
    cuda_before = torch.cuda.is_initialized()
    mm.cpu_state = mm.CPUState.CPU
    try:
        assert mm.get_torch_device().type == 'cpu'
        yield
        assert torch.cuda.is_initialized() is cuda_before
    finally:
        mm.cpu_state = previous


def tiny_engine(tmp_path):
    return DramaEngine(tmp_path / 'unused-model.safetensors',
                       tmp_path / 'unused-codec.safetensors',
                       moe_backend='torch', cuda_graph=False)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['full', 'compact', 'compact_full_fallback'])
def test_head_constructor_failure_can_retry_same_engine_with_real_cpu_patcher(
        tmp_path, monkeypatch, dtype, kind):
    weights = (torch.arange(15 * 8, dtype=torch.float32).reshape(15, 8) / 128).to(dtype)
    full, compact = tmp_path / 'full.safetensors', tmp_path / 'compact.safetensors'
    save_file({'weight': weights}, str(full))
    if kind == 'compact':
        # The source deliberately stores controls in a different order.
        save_file({'weight': weights[[14, 13]], 'ids': torch.tensor([14, 13])}, str(compact))
    engine = DramaEngine(tmp_path / 'unused-model.safetensors',
                         tmp_path / 'unused-codec.safetensors',
                         text_head_path=full, line_head_path=compact,
                         scene_head_mode='full' if kind == 'full' else 'compact',
                         moe_backend='torch', cuda_graph=False)
    audio_head = torch.nn.Linear(8, 3, bias=False, dtype=dtype)
    with torch.inference_mode():
        audio_head.weight.copy_(weights[[10, 11, 12]])
    engine.drama = SimpleNamespace(
        config=SimpleNamespace(output_token_ids=(10, 11, 12), text=SimpleNamespace(hidden_size=8)),
        tokenizer=SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[13]), token_to_id=lambda text: 14),
        patchers=lambda: [], parts=lambda: (None, None, None, audio_head, None))
    head_attr, patcher_attr = (('text_head', 'text_head_patcher') if kind == 'full'
                               else ('line_head', 'line_head_patcher'))
    load = engine.load_text_head if kind == 'full' else engine.load_scene_head
    fault = MemoryError('test-only head patcher construction failure')
    constructors, managed_loads = [], []
    real_load_models_gpu = mm.load_models_gpu

    def failed_constructor(*args, **kwargs):
        constructors.append((args, kwargs))
        raise fault

    def observed_real_load(patchers, *args, **kwargs):
        managed_loads.append(tuple(patchers))
        return real_load_models_gpu(patchers, *args, **kwargs)

    monkeypatch.setattr(mm, 'load_models_gpu', observed_real_load)
    monkeypatch.setattr(engine_module.comfy.model_patcher, 'ModelPatcher', failed_constructor)
    try:
        with torch.inference_mode():
            with pytest.raises(MemoryError) as raised:
                load()
            assert raised.value is fault
            assert len(constructors) == 1
            assert managed_loads == [], 'An incomplete head must not reach the host loader'
            assert getattr(engine, head_attr) is None
            assert getattr(engine, patcher_attr) is None

            # Restore the real class before calling any real host loading code;
            # replacing a public class with a function during that load could
            # otherwise interfere with host isinstance/clone contracts.
            monkeypatch.setattr(engine_module.comfy.model_patcher, 'ModelPatcher', ModelPatcher)
            head, ids = load()
            patcher = getattr(engine, patcher_attr)
            assert isinstance(patcher, ModelPatcher)
            assert patcher.model is getattr(engine, head_attr)
            assert patcher._load_list()
            assert managed_loads == [(patcher,)]
            expected_ids = tuple(range(15)) if kind == 'full' else (10, 11, 12, 13, 14)
            assert ids == expected_ids
            hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype) / 16
            expected = F.linear(hidden, weights[list(expected_ids)])
            assert torch.equal(head(hidden), expected)
            again, again_ids = load()
            assert getattr(engine, patcher_attr) is patcher
            assert again_ids == expected_ids
            assert torch.equal(again(hidden), expected)
    finally:
        monkeypatch.setattr(engine_module.comfy.model_patcher, 'ModelPatcher', ModelPatcher)
        # Real CPU host cleanup, not the engine.unload behavior under test below.
        mm.unload_all_models()


def populated_engine(tmp_path, store):
    engine = tiny_engine(tmp_path)
    fields = {name: object() for name in ENGINE_FIELDS}
    fields['drama'] = SimpleNamespace(experts=store)
    for name, value in fields.items():
        setattr(engine, name, value)
    return engine, fields


def cleanup_spies(monkeypatch, calls, faults):
    def operation(name):
        def run():
            calls.append(name)
            fault = faults.get(name)
            if fault is not None:
                raise fault
        return run
    monkeypatch.setattr(mm, 'unload_all_models', operation('host'))
    monkeypatch.setattr(engine_module.gc, 'collect', operation('gc'))
    monkeypatch.setattr(acceleration, 'release_unused_pinned_memory', operation('pins'))
    monkeypatch.setattr(mm, 'soft_empty_cache', operation('device_cache'))
    return operation


@pytest.mark.parametrize('failed_step', ['experts', 'host', 'gc', 'pins', 'device_cache'])
def test_idle_unload_attempts_all_cleanup_preserves_first_error_and_can_repeat(
        tmp_path, monkeypatch, failed_step):
    calls = []
    fault = OSError('test-only idle cleanup failure at ' + failed_step)
    faults = {failed_step: fault}
    operation = cleanup_spies(monkeypatch, calls, faults)
    store = SimpleNamespace(cache=None, active=False, close=operation('experts'))
    engine, _ = populated_engine(tmp_path, store)
    with pytest.raises(OSError) as raised:
        engine.unload()
    assert raised.value is fault
    assert calls == ['experts', 'host', 'gc', 'pins', 'device_cache']
    assert all(getattr(engine, name) is None for name in ENGINE_FIELDS)
    # This is a reference/attempt contract; it makes no process-RSS assertion.
    faults.clear()
    calls.clear()
    engine.unload()
    assert calls == ['host', 'gc', 'pins', 'device_cache']
    assert all(getattr(engine, name) is None for name in ENGINE_FIELDS)


@pytest.mark.parametrize('first', ['experts', 'host'])
def test_idle_unload_secondary_and_logging_failures_do_not_replace_primary(
        tmp_path, monkeypatch, first):
    class CleanupAbort(BaseException):
        pass
    calls, logs = [], []
    primary = MemoryError('test-only primary cleanup fault')
    faults = {first: primary, 'gc': CleanupAbort('test-only secondary GC fault'),
              'pins': OSError('test-only secondary pinned-cache fault'),
              'device_cache': RuntimeError('test-only secondary device-cache fault')}
    if first == 'experts':
        faults['host'] = RuntimeError('test-only secondary host-unload fault')
    operation = cleanup_spies(monkeypatch, calls, faults)

    def broken_logger(*args, **kwargs):
        logs.append(args)
        raise PermissionError('test-only log sink fault')

    monkeypatch.setattr(logging, 'warning', broken_logger)
    engine, _ = populated_engine(tmp_path, SimpleNamespace(cache=None, active=False,
                                                           close=operation('experts')))
    with pytest.raises(MemoryError) as raised:
        engine.unload()
    assert raised.value is primary
    assert calls == ['experts', 'host', 'gc', 'pins', 'device_cache']
    assert len(logs) == (4 if first == 'experts' else 3)
    assert all(getattr(engine, name) is None for name in ENGINE_FIELDS)


@pytest.mark.parametrize('kind', ['torch_cache', 'accelerated_cache', 'resident_active'])
def test_actual_active_store_close_refusal_keeps_engine_and_skips_host_cleanup(
        tmp_path, monkeypatch, kind):
    # Call each real close implementation without constructing weights, CUDA
    # allocations or the accelerated store's ThreadPoolExecutor.
    if kind == 'torch_cache':
        from drama3_t8.native.offload import ExpertStore
    elif kind == 'accelerated_cache':
        from drama3_t8.native.offload_accelerated import ExpertStore
    else:
        from drama3_t8.native.resident import ResidentExperts as ExpertStore
    store = object.__new__(ExpertStore)
    if kind == 'resident_active':
        store.active = True
        store.closed = False
        store._members = [object()]
    else:
        store.cache = object()
        store.banks = [object()]
    actual_close = store.close
    refusals, closes, cleanup_calls = [], [], []

    def observed_actual_close():
        closes.append(True)
        try:
            actual_close()
        except RuntimeError as error:
            refusals.append(error)
            raise

    store.close = observed_actual_close
    cleanup_spies(monkeypatch, cleanup_calls, {})
    engine, original = populated_engine(tmp_path, store)
    with pytest.raises(RuntimeError, match='active') as raised:
        engine.unload()
    assert closes == [True]
    assert raised.value is refusals[0]
    assert cleanup_calls == []
    assert all(getattr(engine, name) is value for name, value in original.items())
    if kind == 'resident_active':
        assert store.active is True and store.closed is False and len(store._members) == 1
    else:
        assert store.cache is not None and len(store.banks) == 1
