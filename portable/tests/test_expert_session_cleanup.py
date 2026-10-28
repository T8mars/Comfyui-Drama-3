"""Real store context managers, CPU-only fault injection for resource lifetime."""
from types import SimpleNamespace as NS

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import offload,offload_accelerated


class Cancellation(BaseException):
    pass


def harness(monkeypatch,module,*,kind='fp8',fail_setup=None,report_error=None,empty_error=None):
    events=[]
    class Pool:
        def __init__(self,*args):
            events.append('allocate')
            if fail_setup=='pool':raise RuntimeError('pool allocation failed')
        def report(self):
            events.append('report')
            if report_error:raise report_error
    def empty():
        events.append('empty')
        if empty_error:raise empty_error
    monkeypatch.setattr(module,'SlotPool',Pool)
    mm=module.comfy.model_management
    monkeypatch.setattr(mm,'get_free_memory',lambda *_:2**30)
    monkeypatch.setattr(mm,'soft_empty_cache',empty)
    if module is offload_accelerated:
        monkeypatch.setattr(mm,'free_memory',lambda *a,**k:None)
        monkeypatch.setattr(mm,'current_loaded_models',[])
        def available(_):
            if fail_setup=='kernel':raise RuntimeError('kernel preparation failed')
            return True
        monkeypatch.setattr(module.kernels,'available',available)
        def side(*a):
            events.append('cpu_side')
            if fail_setup=='cpu':raise RuntimeError('CPU handoff allocation failed')
            return NS()
        monkeypatch.setattr(module,'CpuSide',side)
    def codebook(_):
        events.append('codebook')
        if fail_setup=='codebook':raise RuntimeError('codebook allocation failed')
    bank=NS(num_experts=1,in_features=256,kernel=kind,codebook=codebook,
        fields={'weight':NS(nbytes=1,is_pinned=lambda:True)})
    store=object.__new__(module.ExpertStore)
    store.cache=None;store.cache_bytes=1;store.banks=[(bank,)];store.cpu_side=None;store.sync_free=False
    return store,events


@pytest.mark.parametrize('stage,kind',[('pool','fp8'),('kernel','fp8'),('codebook','w4'),('cpu','int8')])
def test_accelerated_setup_failure_always_clears_cache_and_auxiliary_state(monkeypatch,stage,kind):
    store,events=harness(monkeypatch,offload_accelerated,kind=kind,fail_setup=stage)
    with pytest.raises(RuntimeError,match='failed'):
        with store.session(torch.device('cpu'),[],cpu_decode=True):pytest.fail('setup must fail first')
    assert store.cache is None and store.cpu_side is None and store.sync_free is False
    assert events[-1]=='empty'


@pytest.mark.parametrize('module',[offload,offload_accelerated])
@pytest.mark.parametrize('report_failure',[False,True])
@pytest.mark.parametrize('empty_failure',[False,True])
def test_secondary_cleanup_failures_never_replace_original_baseexception_cancel(monkeypatch,module,report_failure,empty_failure):
    store,_=harness(monkeypatch,module,report_error=RuntimeError('report failed') if report_failure else None,
        empty_error=RuntimeError('empty failed') if empty_failure else None)
    original=Cancellation('original cancellation')
    with pytest.raises(Cancellation) as caught:
        with store.session(torch.device('cpu'),[]):raise original
    assert caught.value is original and store.cache is None
    if module is offload_accelerated:assert store.cpu_side is None and store.sync_free is False


@pytest.mark.parametrize('module',[offload,offload_accelerated])
@pytest.mark.parametrize('report_failure,empty_failure',[(True,False),(False,True),(True,True)])
def test_diagnostic_failures_do_not_turn_completed_generation_into_failure(monkeypatch,module,report_failure,empty_failure):
    store,_=harness(monkeypatch,module,report_error=RuntimeError('report failed') if report_failure else None,
        empty_error=RuntimeError('empty failed') if empty_failure else None)
    with store.session(torch.device('cpu'),[]):pass
    assert store.cache is None
    # Retry owns a fresh pool rather than incorrectly reusing a leaked cache.
    with store.session(torch.device('cpu'),[]):assert store.cache is not None
    assert store.cache is None


@pytest.mark.parametrize('module',[offload,offload_accelerated])
def test_nested_failure_keeps_outer_pool_owned_until_outer_exit(monkeypatch,module):
    store,events=harness(monkeypatch,module)
    with store.session(torch.device('cpu'),[]):
        pool=store.cache
        with pytest.raises(Cancellation):
            with store.session(torch.device('cpu'),[]):raise Cancellation()
        assert store.cache is pool and events==['allocate']
    assert store.cache is None and events==['allocate','report','empty']


@pytest.mark.parametrize('module',[offload,offload_accelerated])
def test_real_control_interrupt_without_primary_is_not_swallowed(monkeypatch,module):
    store,_=harness(monkeypatch,module,report_error=KeyboardInterrupt('manual stop'))
    with pytest.raises(KeyboardInterrupt):
        with store.session(torch.device('cpu'),[]):pass
    assert store.cache is None


@pytest.mark.parametrize('module',[offload,offload_accelerated])
def test_secondary_baseexception_cleanup_cannot_replace_existing_cancel(monkeypatch,module):
    store,_=harness(monkeypatch,module,report_error=Cancellation('secondary report'),empty_error=Cancellation('secondary cache'))
    original=Cancellation('original')
    with pytest.raises(Cancellation) as caught:
        with store.session(torch.device('cpu'),[]):raise original
    assert caught.value is original and store.cache is None
