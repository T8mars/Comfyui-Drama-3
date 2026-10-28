"""Loader publication/rollback with tiny real CPU assign=True parameters.

Reader, admission and pin outcomes are controlled boundaries; these tests do
not claim hardware page locking or a real low-RAM model run. Store-level
fallback compatibility remains covered by test_pinned_expert_memory.py.
"""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import acceleration, checkpoint_ownership as owned, checkpoint_reader as reader, loader


class _Token:
    pass


def _harness(monkeypatch, *, pin_requested=True, drop_layer=1, cleanup_failure=None, original_error=None):
    events = []
    body = torch.nn.Module()
    body.model = torch.nn.Module()
    body.model.layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Linear(2, 2, bias=False, device='meta', dtype=torch.bfloat16)
        body.model.layers.append(layer)
    body.register_buffer('anchor', torch.empty(2, device='meta'))
    state = {name: torch.full(value.shape, index + 1, device='cpu', dtype=value.dtype)
             for index, (name, value) in enumerate(body.state_dict().items())}
    tokens, token_refs, tensor_refs = {}, [], []
    published = []

    def token():
        value = _Token()
        tokens[id(value)] = value
        token_refs.append(weakref.ref(value))
        return value

    def read(*args, **kwargs):
        events.append('read')
        kwargs['diagnostics'].update(actual_reader='host', fallback_reason='controlled CPU fixture')
        return state, {'drama3_config': '{}', 'drama3_format': 'dense'}

    def assign(model, actual_state, **kwargs):
        assert model is body and actual_state is state
        loader.load_checkpoint_storage(model, actual_state)
        assert all(value.device.type == 'cpu' for value in model.parameters())
        tensor_refs.extend(weakref.ref(value) for value in model.parameters())
        tensor_refs.extend(weakref.ref(value) for value in model.buffers())
        events.append('assign')
        return token()

    def finish(witness, model, **kwargs):
        assert not state and model is body
        tokens.pop(id(witness))
        events.append('finish_assignment')
        return token()

    def close_proof(proof):
        if proof is None:
            events.append('ownership.close.none')
            return
        tokens.pop(id(proof))
        events.append('ownership.close')
        if store.closed:
            assert not list(body.parameters()) and not list(body.buffers())
            assert not store.banks
        if cleanup_failure == 'ownership':
            raise RuntimeError('cleanup ownership error')

    class Store:
        def __init__(self):
            self.pin_requested = pin_requested
            self.pin_fallback_reason = None
            self.banks = []
            self.admission = {'policy': 'legacy_full_expert'}
            self.closed = False
            self.added = 0

        def add(self, experts):
            copied = None
            try:
                index = self.added
                self.added += 1
                events.append('add.' + str(index))
                copied = experts.weight.detach().clone()
                tensor_refs.append(weakref.ref(copied))
                self.banks.append({'weight': copied})
                if index == drop_layer:
                    self.pin_fallback_reason = 'controlled available RAM fell below unchanged reserve'
                if original_error is not None and index == 1:
                    raise original_error
                return torch.nn.Identity()
            finally:
                # External test callbacks must not retain source parameters in
                # their own traceback; production may only clear owned frames.
                experts = copied = None

        def close(self):
            events.append('store.close')
            self.banks.clear()
            self.closed = True
            if cleanup_failure == 'close':
                raise KeyboardInterrupt('cleanup close interruption')

    store = Store()

    def assert_owned_gone():
        assert store.closed and not store.banks
        assert not state and not list(body.parameters()) and not list(body.buffers())
        assert not tokens and all(ref() is None for ref in token_refs)
        assert all(ref() is None for ref in tensor_refs)

    def collect():
        events.append('gc')
        assert_owned_gone()
        if cleanup_failure == 'gc':
            raise RuntimeError('cleanup gc error')
        return gc.collect()

    def release_host_cache():
        events.append('release_host_cache')
        assert_owned_gone()
        if cleanup_failure in {'host_cache', 'logging'}:
            raise RuntimeError('cleanup host allocator error')

    def publish(model, patcher, **kwargs):
        events.append('publish')
        result = SimpleNamespace(model=model, experts=kwargs['experts'], checkpoint_reader=kwargs['checkpoint_reader'])
        published.append(result)
        return result

    monkeypatch.setattr(reader, 'load_torch_file', read)
    monkeypatch.setattr(loader, 'Drama3Model', lambda *args, **kwargs: body)
    monkeypatch.setattr(loader, 'parse_config', lambda value: SimpleNamespace())
    monkeypatch.setattr(loader, 'Drama3', publish)
    monkeypatch.setattr(loader.comfy.model_management, 'get_torch_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(loader.comfy.model_management, 'unet_offload_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(loader.comfy.model_management, 'throw_exception_if_processing_interrupted', lambda: None)
    monkeypatch.setattr(loader.comfy.utils, 'detect_layer_quantization', lambda *args: {})
    monkeypatch.setattr(loader.comfy.ops, 'pick_operations', lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.comfy.model_patcher, 'CoreModelPatcher', lambda model, **kwargs: SimpleNamespace(model=model))
    monkeypatch.setattr(owned, 'assign_owned', assign)
    monkeypatch.setattr(owned, 'finish_assignment', finish)
    monkeypatch.setattr(owned, 'is_incremental', lambda *args: False)
    monkeypatch.setattr(owned, 'close', close_proof)
    monkeypatch.setattr(acceleration, 'expert_store', lambda *args, **kwargs: store)
    # These module-private boundaries avoid changing process-wide gc or any
    # actual CUDA/host allocator. They assert the complete disposal order.
    monkeypatch.setattr(loader, 'gc', SimpleNamespace(collect=collect))
    monkeypatch.setattr(acceleration, 'release_unused_pinned_memory', release_host_cache)
    if cleanup_failure == 'logging':
        def broken_log(*args, **kwargs): raise RuntimeError('cleanup diagnostic error')
        monkeypatch.setattr(loader, 'logging', SimpleNamespace(info=lambda *args: None, warning=broken_log))
    return SimpleNamespace(body=body, store=store, events=events, published=published, assert_owned_gone=assert_owned_gone)


@pytest.mark.parametrize('cleanup_failure', [None, 'close', 'ownership', 'gc', 'host_cache', 'logging'])
def test_resource_drop_aborts_before_next_layer_and_releases_owned_storage(monkeypatch, cleanup_failure):
    fixture = _harness(monkeypatch, cleanup_failure=cleanup_failure)
    with pytest.raises(reader.ReaderResourceError, match='pinning stopped at layer 1') as caught:
        loader.load_drama3('controlled.safetensors', experts_in_ram=True, moe_backend='cuda')
    assert 'unchanged reserve' in str(caught.value)
    assert fixture.store.added == 2 and 'add.2' not in fixture.events
    assert not fixture.published and 'publish' not in fixture.events
    fixture.assert_owned_gone()
    assert fixture.events.index('store.close') < fixture.events.index('ownership.close')
    assert fixture.events.index('ownership.close') < fixture.events.index('gc') < fixture.events.index('release_host_cache')


@pytest.mark.parametrize('cleanup_failure', ['close', 'ownership', 'gc', 'host_cache', 'logging'])
def test_rollback_cleanup_errors_preserve_original_load_exception_identity(monkeypatch, cleanup_failure):
    original = OSError('exact original bank failure')
    fixture = _harness(monkeypatch, drop_layer=None, cleanup_failure=cleanup_failure, original_error=original)
    with pytest.raises(OSError) as caught:
        loader.load_drama3('controlled.safetensors', experts_in_ram=True, moe_backend='auto')
    assert caught.value is original
    assert fixture.store.added == 2 and not fixture.published
    fixture.assert_owned_gone()
    assert fixture.events[-2:] == ['gc', 'release_host_cache']


@pytest.mark.parametrize('pin_requested,drop_layer', [(False, 1), (True, None)])
def test_unrequested_or_successful_pinning_keeps_existing_publication_path(monkeypatch, pin_requested, drop_layer):
    fixture = _harness(monkeypatch, pin_requested=pin_requested, drop_layer=drop_layer)
    # This is not an exception scenario even if an unrequested store carries a
    # compatibility reason. No hardware allocator behavior is asserted here.
    result = loader.load_drama3('controlled.safetensors', experts_in_ram=True, moe_backend='auto')
    assert result is fixture.published[0] and fixture.store.added == 3
    assert 'gc' not in fixture.events and 'release_host_cache' not in fixture.events
    assert len(fixture.store.banks) == 3
    fixture.store.close()


@pytest.mark.parametrize('close_error', [RuntimeError('ownership close failed'),
                                      KeyboardInterrupt('ownership close cancelled')])
def test_successful_load_inside_caller_except_propagates_ownership_close_error(monkeypatch, close_error):
    fixture = _harness(monkeypatch, drop_layer=None)
    original_close = owned.close

    def fail_close(proof):
        original_close(proof)
        if proof is not None:
            raise close_error

    monkeypatch.setattr(owned, 'close', fail_close)
    with pytest.raises(type(close_error)) as caught:
        try:
            raise ValueError('unrelated exception handled by the caller')
        except ValueError:
            loader.load_drama3('controlled.safetensors', experts_in_ram=True, moe_backend='auto')
    assert caught.value is close_error
    assert fixture.store.added == 3
    assert 'gc' not in fixture.events and 'release_host_cache' not in fixture.events
    fixture.store.close()
