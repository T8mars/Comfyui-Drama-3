"""Actual bank admission receipts and bounded read/ownership observations (CPU)."""
import dataclasses
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from test_auto_pin_ram_recovery import _actual_loader_fixture
from drama3_t8.native import loader, offload_accelerated as offload, checkpoint_ownership as owned
from drama3_t8.native import checkpoint_reader as reader


GIB=1024**3


def blank_store():
    # Do not invoke the CUDA constructor: test the real admission helper/marker.
    value=object.__new__(offload.ExpertStore)
    value.pin_checks=0;value.pin_last_check=None;value.pin_fallback_reason=None
    value.pin_ram_admission=None;value._pin_ram_refusal=None
    return value


def issue(monkeypatch,value,available=8*GIB):
    monkeypatch.setattr(offload.psutil,'virtual_memory',lambda:SimpleNamespace(total=64*GIB,available=available))
    assert offload._pin_headroom(value,1024**2,'actual controlled CPU bank') is False
    return offload.pin_ram_refusal(value)


def test_actual_nonincremental_bank_refusal_cleans_then_reads_once_with_unpinned_torch(monkeypatch):
    fixture=_actual_loader_fixture(monkeypatch,incremental_mode=False)
    result=loader.load_drama3('controlled',experts_in_ram=True,moe_backend='auto')
    assert fixture.events.count('read')==2 and len(fixture.published)==1
    assert fixture.models[0]() is None and fixture.stores[0].closed
    assert fixture.stores[0].banks==[] and result.experts.pin_memory is False
    assert result.experts.requested_backend=='auto'
    recovery=result.checkpoint_reader['automatic_backend_fallback']
    assert recovery['scope']=='accelerated_bank_pin_ram_admission'
    assert recovery['admission']['incremental'] is False
    first=recovery['first_attempt']
    assert first['ownership_incremental'] is False and first['ownership_proof_present'] is False
    assert first['ownership']['fallback_reason']=='controlled proof not issued'
    assert first['bank_refusal']['available_bytes']<first['bank_refusal']['required_bytes']
    assert first['partial_cleanup_complete'] is True
    summary=result.checkpoint_reader['load_summary']
    assert summary['requested_backend']=='auto' and summary['recovery']['first_attempt']==first
    assert len(json.dumps(summary))<8192


def test_explicit_cuda_actual_nonincremental_refusal_is_not_retried(monkeypatch):
    fixture=_actual_loader_fixture(monkeypatch,incremental_mode=False)
    with pytest.raises(owned.IncrementalPinRAMAdmissionError) as caught:
        loader.load_drama3('controlled',experts_in_ram=True,moe_backend='cuda')
    assert caught.value.admission['scope']=='accelerated_bank_pin_ram_admission'
    assert caught.value.partial_cleanup_complete is True
    assert fixture.events.count('read')==1 and fixture.published==[]


def test_mutable_public_admission_is_not_the_issued_frozen_evidence(monkeypatch):
    value=blank_store();receipt=issue(monkeypatch,value)
    assert receipt['required_bytes']==8*GIB+2*1024**2
    value.pin_ram_admission.clear();value.pin_ram_admission['available_bytes']=999999
    assert offload.pin_ram_refusal(value)==receipt
    returned=offload.pin_ram_refusal(value);returned.clear()
    assert offload.pin_ram_refusal(value)==receipt
    with pytest.raises(dataclasses.FrozenInstanceError):value._pin_ram_refusal.receipt=()


def test_foreign_and_plain_stores_cannot_recover_from_public_dict_or_message(monkeypatch):
    actual=blank_store();receipt=issue(monkeypatch,actual)
    value=SimpleNamespace(pin_ram_admission=receipt,pin_fallback_reason='available RAM is below reserve',
                          _pin_ram_refusal=actual._pin_ram_refusal)
    assert offload.pin_ram_refusal(value) is None
    plain=blank_store();plain.pin_ram_admission=receipt
    assert offload.pin_ram_refusal(plain) is None


def test_receipt_weak_owner_identity_cannot_be_copied_to_another_store(monkeypatch):
    value=blank_store();issue(monkeypatch,value)
    other=blank_store();other._pin_ram_refusal=value._pin_ram_refusal
    assert offload.pin_ram_refusal(other) is None
    marker=value._pin_ram_refusal;ref=weakref.ref(value)
    del value
    assert ref() is None and marker.owner() is None
    assert offload.pin_ram_refusal(other) is None


@pytest.mark.parametrize('change',[
    {'issuer':object()}, {'receipt':()}, {'receipt':(('bank','x'),)*6},
    {'owner':lambda:None}])
def test_malformed_or_foreign_receipt_cannot_authorize_retry(monkeypatch,change):
    value=blank_store();issue(monkeypatch,value)
    value._pin_ram_refusal=dataclasses.replace(value._pin_ram_refusal,**change)
    assert offload.pin_ram_refusal(value) is None


@pytest.mark.parametrize('error',[MemoryError('RAM query failed'),OSError('RAM query I/O'),RuntimeError('unsupported')])
def test_failed_memory_query_is_not_measured_ram_refusal(monkeypatch,error):
    value=blank_store();issue(monkeypatch,value)
    value.pin_fallback_reason=None
    def fail():raise error
    monkeypatch.setattr(offload.psutil,'virtual_memory',fail)
    assert offload._pin_headroom(value,1024**2,'query failure') is False
    assert value._pin_ram_refusal is None and value.pin_ram_admission is None
    assert offload.pin_ram_refusal(value) is None


def test_allocation_failure_after_successful_admission_has_no_refusal_receipt(monkeypatch):
    value=blank_store()
    monkeypatch.setattr(offload.psutil,'virtual_memory',lambda:SimpleNamespace(total=64*GIB,available=40*GIB))
    assert offload._pin_headroom(value,1024**2,'healthy admission') is True
    offload._disable_pinning(value,'pin allocation failed: controlled OOM')
    assert offload.pin_ram_refusal(value) is None


def test_actual_loader_with_opaque_store_refusal_remains_an_ordinary_error(monkeypatch):
    fixture=_actual_loader_fixture(monkeypatch,incremental_mode=False)
    # Store still receives a true measured receipt but is not the selected real
    # accelerated type. Matching text/dict cannot grant the typed fallback.
    monkeypatch.setattr(offload,'ExpertStore',type('OpaqueForeignStore',(),{}))
    with pytest.raises(reader.ReaderResourceError) as caught:
        loader.load_drama3('controlled',experts_in_ram=True,moe_backend='auto')
    assert type(caught.value) is reader.ReaderResourceError
    assert fixture.events.count('read')==1 and fixture.published==[]


def test_subclass_of_typed_failure_cannot_broaden_auto_recovery(monkeypatch):
    class OtherResourceFailure(owned.IncrementalPinRAMAdmissionError):pass
    error=OtherResourceFailure('other resource',{'available_bytes':1,'required_bytes':2})
    error.partial_cleanup_complete=True;calls=[]
    def once(*args,**kwargs):calls.append(True);raise error
    monkeypatch.setattr(loader,'_load_drama3_once',once)
    with pytest.raises(OtherResourceFailure) as caught:
        loader.load_drama3('controlled',experts_in_ram=True,moe_backend='auto')
    assert caught.value is error and calls==[True]


def test_summary_retains_finite_legacy_required_bytes_without_changing_math():
    required=12345*1.02+1024**3
    store=SimpleNamespace(admission={'required_bytes':required,'available_bytes':float('nan'),
        'reserve_bytes':float('inf'),'layer_destination_bytes':-2.0})
    summary=loader.checkpoint_load_summary({},store,False,'auto')
    assert summary['admission']['required_bytes']==required
    assert type(summary['admission']['required_bytes']) is float
    assert 'available_bytes' not in summary['admission'] and 'reserve_bytes' not in summary['admission']
    assert 'layer_destination_bytes' not in summary['admission']


def test_summary_is_bounded_fixed_json_and_never_formats_model_or_tensor():
    class Bomb:
        def __str__(self):raise AssertionError('must not stringify model/tensor')
    huge='x'*1024**2
    source={'actual_reader':huge,'fallback_reason':huge,'source_cpu_bytes':123,
        'state':Bomb(),'header_sha256':huge,'memory_checks':[{'available_bytes':3,'required_bytes':4,'tensor':Bomb()}]*10000,
        'host_context':{'aimdo_enabled':False,'disable_mmap':True,'path':huge},
        'expert_ownership':{'policy':huge,'fallback_reason':huge,'layers':[Bomb()]*10000,'completed_layers':7,'proof_issued':False}}
    store=SimpleNamespace(admission={'policy':huge,'ownership_refusal':huge,'available_bytes':torch.tensor(12),
        'required_bytes':True,'reserve_bytes':-1,'selected_accelerated_store':True},
        pin_ram_admission={'bank':huge,'physical_bytes':2,'available_bytes':1,'required_bytes':3,'reserve_bytes':1,'total_bytes':8})
    result=loader.checkpoint_load_summary(source,store,False,'auto',proof_present=False)
    encoded=json.dumps(result)
    assert len(encoded)<4096 and result['memory_check_count']==10000
    assert 'memory_checks' not in result and 'layers' not in result['ownership']
    assert result['memory_check_edges']['last']=={'available_bytes':3,'required_bytes':4}
    assert 'available_bytes' not in result['admission'] and 'required_bytes' not in result['admission']
    assert 'reserve_bytes' not in result['admission'] and result['ownership_incremental'] is False


@pytest.mark.parametrize('failure',[False,True])
def test_actual_engine_caches_load_facts_without_replacing_native_error(monkeypatch,failure):
    from drama3_t8 import engine as engine_module
    engine=engine_module.DramaEngine('controlled-model','controlled-codec')
    facts={'version':1,'actual_reader':'controlled','ownership_incremental':False}
    error=MemoryError('original');error.load_diagnostics=facts
    drama=SimpleNamespace(checkpoint_reader={'load_summary':facts})
    calls=[]
    def native(*args,**kwargs):
        calls.append((args,kwargs))
        if failure:raise error
        return drama
    monkeypatch.setattr(engine_module,'load_drama3',native)
    monkeypatch.setattr(engine,'set_backend',lambda *a:None)
    monkeypatch.setattr(engine,'_load_drama',lambda:None)
    if failure:
        with pytest.raises(MemoryError) as caught:engine.load_model()
        assert caught.value is error and engine.drama is None
    else:
        assert engine.load_model() is drama
    assert engine.last_load_diagnostics is facts and len(calls)==1


def test_take_metadata_has_cached_load_facts_without_resampling(rig):
    facts={'version':1,'actual_reader':'controlled','ownership_incremental':False}
    rig.engine.last_load_diagnostics=facts
    result=rig.engine.speak('Metadata only.',sampling=rig.module.Sampling(seed=12,max_frames=300))
    assert result.takes[-1].metadata['checkpoint_load']==facts
    assert rig.calls[-1]['seed']==12


# Reuse the real CPU prompt/head/method rig, not a replacement _take result.
from test_speech_output_policy import rig, module, tokenizer


def test_status_returns_cached_load_facts_without_live_sampling(tmp_path,monkeypatch):
    from studio_app_loader import application
    from fastapi.testclient import TestClient
    (tmp_path/'web').mkdir()
    app=application.create_app(tmp_path);facts={'version':1,'actual_reader':'controlled'}
    with TestClient(app) as client:
        app.state.studio.engine.last_load_diagnostics=facts
        for _ in range(3):assert client.get('/api/status').json()['last_checkpoint_load']==facts
