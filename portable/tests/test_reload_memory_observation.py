"""Unload observations are JSON/weak only, bounded and CUDA-initialization safe."""
import gc
import json
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8 import memory_diagnostics as memory
from drama3_t8.engine import DramaEngine
from drama3_t8.native import acceleration
import comfy.model_management as mm


def engine():
    return DramaEngine('unused','unused',moe_backend='torch',cuda_graph=False)


def test_uninitialized_snapshot_never_queries_host_allocator_or_initializes_cuda(monkeypatch):
    calls=[]
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,
        memory=SimpleNamespace(host_memory_stats=lambda:calls.append('wrong'))))
    monkeypatch.setitem(sys.modules,'torch',fake)
    sample=memory.snapshot(engine())
    assert calls==[] and sample['cuda_initialized'] is False
    assert sample['host_allocator']['status']=='CUDA not initialized; not queried'
    json.dumps(sample,allow_nan=False)


def test_initialized_allocator_statistics_are_observations_not_release_calls(monkeypatch):
    calls=[]
    def stats():
        calls.append('stats')
        return {'allocated_bytes.current':123,'reserved_bytes.current':456,'num_host_alloc':3,'num_host_free':1}
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,memory=SimpleNamespace(host_memory_stats=stats)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    sample=memory.snapshot(engine())
    assert calls==['stats'] and sample['host_allocator']['allocated_bytes.current']==123
    assert sample['host_allocator']['reserved_bytes.current']==456
    assert 'leak' not in json.dumps(sample).lower()


def test_unsupported_or_faulted_counters_remain_explicitly_unknown(monkeypatch):
    def error():raise RuntimeError('counter unavailable')
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,memory=SimpleNamespace(host_memory_stats=error)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    sample=memory.snapshot(engine())
    assert sample['host_allocator']['status']=='unavailable'
    assert 'allocated_bytes.current' not in sample['host_allocator']


def test_weak_observation_does_not_keep_retired_model_or_storage_alive(monkeypatch):
    item=engine();module=torch.nn.Linear(4,4)
    item.drama=SimpleNamespace(model=module,experts=None)
    reference=weakref.ref(module)
    before,probes=memory.begin_unload(item)
    item.drama=module=None
    gc.collect()
    assert reference() is None
    result=memory.finish_unload(item,before,probes)
    assert result['after']['owned_objects']['drama.model']['alive'] is False
    assert result is item.last_unload_diagnostics
    json.dumps(result,allow_nan=False)


def test_external_holder_is_reported_as_live_without_guessing_leak(monkeypatch):
    item=engine();holder=torch.nn.Linear(2,2)
    item.drama=SimpleNamespace(model=holder,experts=None)
    before,probes=memory.begin_unload(item);item.drama=None
    gc.collect()
    result=memory.finish_unload(item,before,probes)
    assert result['after']['owned_objects']['drama.model']['alive'] is True
    assert 'leak' not in json.dumps(result).lower()
    assert holder.weight.shape==(2,2)


def test_actual_unload_publishes_cached_diagnostics_after_original_cleanup(monkeypatch):
    item=engine();item.drama=SimpleNamespace(model=torch.nn.Linear(2,2),experts=None)
    events=[]
    monkeypatch.setattr(mm,'unload_all_models',lambda:events.append('models'))
    monkeypatch.setattr(acceleration,'release_unused_pinned_memory',lambda:events.append('pins'))
    monkeypatch.setattr(mm,'soft_empty_cache',lambda:events.append('device'))
    result=item.unload()
    assert events==['models','pins','device']
    assert result['status']=='ok' and result is item.last_unload_diagnostics
    assert result['before']['engine_fields']['drama'] is True
    assert result['after']['engine_fields']['drama'] is False
    assert result['after']['owned_objects']['drama.model']['alive'] is False


def test_diagnostic_fault_cannot_skip_cleanup_or_mask_actual_unload_error(monkeypatch):
    item=engine();events=[];fault=OSError('actual cleanup fault')
    def diagnostic(*args,**kwargs):raise MemoryError('observation unavailable')
    def unload():events.append('models');raise fault
    monkeypatch.setattr(memory,'begin_unload',diagnostic)
    monkeypatch.setattr(mm,'unload_all_models',unload)
    monkeypatch.setattr(acceleration,'release_unused_pinned_memory',lambda:events.append('pins'))
    monkeypatch.setattr(mm,'soft_empty_cache',lambda:events.append('device'))
    with pytest.raises(OSError) as caught:item.unload()
    assert caught.value is fault and events==['models','pins','device']


def test_status_returns_cached_snapshot_without_sampling_every_poll(tmp_path,monkeypatch):
    from studio_app_loader import application
    from fastapi.testclient import TestClient
    (tmp_path/'web').mkdir()
    app=application.create_app(tmp_path)
    cached={'version':1,'status':'ok','before':{},'after':{}}
    with TestClient(app) as client:
        app.state.studio.engine.last_unload_diagnostics=cached
        with monkeypatch.context() as patch:
            patch.setattr(memory,'snapshot',lambda *a,**k:pytest.fail('status resampled allocator'))
            for _ in range(3):assert client.get('/api/status').json()['last_unload_memory']==cached


def test_managed_registry_samples_have_a_fixed_upper_bound(monkeypatch):
    monkeypatch.setattr(mm,'current_loaded_models',[SimpleNamespace(model=None) for _ in range(30)])
    result=memory.snapshot(engine())['managed_models']
    assert result['registered_count']==30 and len(result['samples'])==16


def test_diagnostic_error_and_external_probe_strings_are_bounded():
    item=engine();report,probes=memory.begin_unload(item)
    result=memory.finish_unload(item,report,probes,status='error',error=RuntimeError('x'*(1024*1024)))
    assert len(result['error']['message'])==1024 and result['error']['message'].endswith('...')
    assert len(json.dumps(result))<16384
    references={str(index)+'x'*1000:(None,'kind'*1000) for index in range(1000)}
    sample=memory.snapshot(item,references)
    assert len(sample['owned_objects'])==42 and sample['owned_probe_truncated'] is True
    assert all(len(name)<=128 and len(value['type'])<=256 for name,value in sample['owned_objects'].items())
    assert len(json.dumps(sample))<32768
