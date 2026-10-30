"""Auto recovery is one independent read after exact typed pin-RAM rollback."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import loader,checkpoint_ownership as owned,checkpoint_reader as reader,acceleration,offload_accelerated as offload


def refusal(*,clean=True):
    error=owned.IncrementalPinRAMAdmissionError('controlled layer RAM admission',{'available_bytes':1,'required_bytes':2})
    error.partial_cleanup_complete=clean
    return error


@pytest.mark.parametrize('requested,experts,clean,expected_reads',[
    ('auto',True,True,2),('cuda',True,True,1),('torch',True,True,1),
    ('auto',False,True,1),('auto',True,False,1)])
def test_public_wrapper_only_recovers_exact_auto_incremental_ram(requested,experts,clean,expected_reads,monkeypatch):
    error=refusal(clean=clean);calls=[]
    model=SimpleNamespace(experts=SimpleNamespace(),checkpoint_reader={})
    def once(*args,**kwargs):
        calls.append((args,kwargs))
        if len(calls)==1:raise error
        return model
    monkeypatch.setattr(loader,'_load_drama3_once',once)
    monkeypatch.setattr(loader.comfy.model_management,'throw_exception_if_processing_interrupted',lambda:None)
    if expected_reads==2:
        result=loader.load_drama3('checkpoint',experts_in_ram=experts,moe_backend=requested)
        assert result is model and model.experts.requested_backend=='auto'
        assert calls[1][0][-1]=='torch' and calls[1][1]['_disable_expert_pinning'] is True
        assert model.checkpoint_reader['automatic_backend_fallback']['actual_backend']=='torch'
        assert 'controlled layer' in model.experts.fallback_reason
    else:
        with pytest.raises(owned.IncrementalPinRAMAdmissionError) as caught:
            loader.load_drama3('checkpoint',experts_in_ram=experts,moe_backend=requested)
        assert caught.value is error
    assert len(calls)==expected_reads


@pytest.mark.parametrize('error',[MemoryError('allocation OOM'),OSError('source I/O'),
    reader.ReaderResourceError('generic RAM shortage'),owned.OwnershipViolation('proof changed'),
    reader.ReaderCancelled('cancel'),KeyboardInterrupt('interrupt'),ValueError('corrupt checkpoint')])
def test_other_errors_never_trigger_fallback(error,monkeypatch):
    calls=[]
    def once(*args,**kwargs):calls.append(True);raise error
    monkeypatch.setattr(loader,'_load_drama3_once',once)
    with pytest.raises(type(error)) as caught:
        loader.load_drama3('checkpoint',experts_in_ram=True,moe_backend='auto')
    assert caught.value is error and calls==[True]


def test_cancellation_between_completed_cleanup_and_retry_is_not_caught(monkeypatch):
    calls=[];cancel=KeyboardInterrupt('cancel before independent retry')
    def once(*args,**kwargs):calls.append(True);raise refusal()
    def interrupted():raise cancel
    monkeypatch.setattr(loader,'_load_drama3_once',once)
    monkeypatch.setattr(loader.comfy.model_management,'throw_exception_if_processing_interrupted',interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:loader.load_drama3('checkpoint',experts_in_ram=True,moe_backend='auto')
    assert caught.value is cancel and calls==[True]


def test_retry_error_does_not_create_a_third_attempt(monkeypatch):
    calls=[];second=MemoryError('independent Torch load failed')
    def once(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:raise refusal()
        raise second
    monkeypatch.setattr(loader,'_load_drama3_once',once)
    monkeypatch.setattr(loader.comfy.model_management,'throw_exception_if_processing_interrupted',lambda:None)
    with pytest.raises(MemoryError) as caught:loader.load_drama3('checkpoint',experts_in_ram=True,moe_backend='auto')
    assert caught.value is second and len(calls)==2


def _actual_loader_fixture(monkeypatch,*,cleanup_failure=None,incremental_mode=True):
    events=[];models=[];stores=[];source_refs=[];published=[]
    def make_body(*args,**kwargs):
        body=torch.nn.Module();body.model=torch.nn.Module();body.model.layers=torch.nn.ModuleList()
        for _ in range(3):
            layer=torch.nn.Module();layer.mlp=torch.nn.Module()
            layer.mlp.experts=torch.nn.Linear(2,2,bias=False,device='meta',dtype=torch.bfloat16)
            body.model.layers.append(layer)
        body.register_buffer('anchor',torch.empty(2,device='meta'))
        models.append(weakref.ref(body));return body
    def read(*args,**kwargs):
        events.append('read')
        if len(events)>1:
            assert stores[0].closed and stores[0].banks==[]
            assert models[0]() is None
            assert 'host_cache' in events
        state={'anchor':torch.ones(2)}
        state.update({f'model.layers.{index}.mlp.experts.weight':torch.ones(2,2,dtype=torch.bfloat16) for index in range(3)})
        source_refs.extend(weakref.ref(value) for value in state.values())
        kwargs['diagnostics'].update(actual_reader='controlled tiny CPU reader')
        if not incremental_mode:
            kwargs['diagnostics']['expert_ownership']={'policy':'legacy_full_expert','fallback_reason':'controlled proof not issued'}
        return state,{'drama3_config':'{}','drama3_format':'dense'}
    class Store:
        def __init__(self,requested):
            self.requested_backend=requested;self.pin_requested=requested in ('auto','cuda');self.pin_memory=True
            self.pin_fallback_reason=self.pin_ram_admission=None;self.banks=[];self.closed=False;self.cache=None
            self.pin_checks=0;self.pin_last_check=None;self._pin_ram_refusal=None
            self.admission={'policy':'verified_pread_incremental' if requested=='auto' and incremental_mode else 'legacy_full_expert'}
        def add(self,experts):
            self.banks.append(experts.weight.detach().clone());events.append('add.'+self.requested_backend)
            if self.requested_backend in ('auto','cuda') and len(self.banks)==2:
                with monkeypatch.context() as patch:
                    patch.setattr(offload.psutil,'virtual_memory',lambda:SimpleNamespace(total=64*1024**3,available=8*1024**3))
                    assert offload._pin_headroom(self,self.banks[-1].numel()*self.banks[-1].element_size(),'controlled actual bank') is False
            if self.requested_backend=='torch':assert self.pin_memory is False
            return torch.nn.Identity()
        def close(self):
            events.append('close');self.banks.clear();self.closed=True
            if cleanup_failure=='close':raise OSError('controlled close failure')
    def store(*args,**kwargs):
        value=Store(args[2]);stores.append(value);return value
    def assign(model,state,**kwargs):loader.load_checkpoint_storage(model,state);return object()
    def finish(*args,**kwargs):return object() if incremental_mode else None
    def close(proof):
        events.append('proof.close')
        if cleanup_failure=='ownership':raise OSError('controlled proof close failure')
    def collect():
        events.append('gc');gc.collect()
        if cleanup_failure=='gc':raise OSError('controlled GC failure')
    def host():
        events.append('host_cache')
        if cleanup_failure=='host_cache':raise OSError('controlled host-cache failure')
    def publish(model,patcher,**kwargs):
        result=SimpleNamespace(model=model,experts=kwargs['experts'],checkpoint_reader=kwargs['checkpoint_reader'])
        published.append(result);events.append('publish');return result
    monkeypatch.setattr(loader,'Drama3Model',make_body)
    monkeypatch.setattr(loader,'Drama3',publish)
    monkeypatch.setattr(loader,'parse_config',lambda _:SimpleNamespace())
    monkeypatch.setattr(reader,'load_torch_file',read)
    monkeypatch.setattr(owned,'assign_owned',assign)
    monkeypatch.setattr(owned,'finish_assignment',finish)
    monkeypatch.setattr(owned,'is_incremental',lambda proof,model,store:store.requested_backend=='auto' and incremental_mode)
    monkeypatch.setattr(owned,'begin_layer',lambda *args:{})
    monkeypatch.setattr(owned,'complete_layer',lambda *args:{'completed_layers':args[-1]+1})
    monkeypatch.setattr(owned,'close',close)
    monkeypatch.setattr(acceleration,'expert_store',store)
    monkeypatch.setattr(offload,'ExpertStore',Store)
    monkeypatch.setattr(acceleration,'release_unused_pinned_memory',host)
    monkeypatch.setattr(loader,'gc',SimpleNamespace(collect=collect))
    monkeypatch.setattr(loader.comfy.model_management,'get_torch_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(loader.comfy.model_management,'unet_offload_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(loader.comfy.model_management,'throw_exception_if_processing_interrupted',lambda:None)
    monkeypatch.setattr(loader.comfy.model_patcher,'CoreModelPatcher',lambda model,**kwargs:SimpleNamespace(model=model))
    monkeypatch.setattr(loader.comfy.utils,'detect_layer_quantization',lambda *args:{})
    monkeypatch.setattr(loader.comfy.ops,'pick_operations',lambda *args,**kwargs:None)
    return SimpleNamespace(events=events,models=models,stores=stores,published=published,source_refs=source_refs)


def test_real_cpu_loader_rolls_back_every_partial_before_new_torch_read(monkeypatch):
    fixture=_actual_loader_fixture(monkeypatch)
    result=loader.load_drama3('controlled',experts_in_ram=True,moe_backend='auto')
    assert fixture.events.count('read')==2 and len(fixture.stores)==2 and len(fixture.published)==1
    assert fixture.models[0]() is None and result.model is fixture.models[1]()
    assert result.experts is fixture.stores[1] and result.experts.requested_backend=='auto'
    assert result.checkpoint_reader['automatic_backend_fallback']['partial_cleanup_complete'] is True


@pytest.mark.parametrize('failure',['close','ownership','gc','host_cache'])
def test_partial_cleanup_failure_prevents_independent_retry(monkeypatch,failure):
    fixture=_actual_loader_fixture(monkeypatch,cleanup_failure=failure)
    with pytest.raises(owned.IncrementalPinRAMAdmissionError) as caught:
        loader.load_drama3('controlled',experts_in_ram=True,moe_backend='auto')
    assert caught.value.partial_cleanup_complete is False
    assert fixture.events.count('read')==1 and fixture.published==[]
    assert fixture.stores[0].closed and fixture.stores[0].banks==[]


def test_bank_headroom_sets_typed_refusal_only_for_measured_ram_budget(monkeypatch):
    store=SimpleNamespace(pin_checks=0,pin_last_check=None,pin_fallback_reason=None,pin_ram_admission=None)
    monkeypatch.setattr(offload.psutil,'virtual_memory',lambda:SimpleNamespace(total=64*1024**3,available=8*1024**3))
    assert offload._pin_headroom(store,1024**2,'test bank') is False
    assert store.pin_ram_admission['required_bytes']==8*1024**3+2*1024**2
    assert store.pin_ram_admission['available_bytes']==8*1024**3
    other=SimpleNamespace(pin_checks=0,pin_last_check=None,pin_fallback_reason=None,pin_ram_admission=None)
    def unavailable():raise MemoryError('cannot inspect actual RAM')
    monkeypatch.setattr(offload.psutil,'virtual_memory',unavailable)
    assert offload._pin_headroom(other,1024**2,'test bank') is False
    assert other.pin_ram_admission is None
