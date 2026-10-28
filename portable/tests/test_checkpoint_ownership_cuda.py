"""Real page locking through the production owned reader/assign/selection.

Prepared for isolated root-scheduled GPU execution. No generation/Graph claim.
Original four reader CUDA goldens and their numerical thresholds remain intact.
"""
import gc
import os

import pytest
from safetensors.torch import save_file
import torch

from drama3_t8.native import checkpoint_reader as reader
from drama3_t8.native import checkpoint_ownership as own,acceleration
from test_checkpoint_reader import _expert_model,_region
from test_quantized_loading import quantized_bank
from test_checkpoint_reader_cuda import (_ready,_bank_source,_assert_pinned_exact,
                                        _host_stats,_assert_allocated_restored,_record)

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA/page-locking check')


@pytest.mark.parametrize('fmt',['int8_tensorwise','asym_w4a8_int8','w6a8_int8'])
def test_official_owned_assign_select_pin_and_release_each_layer(tmp_path,fmt):
    name='owned_reader_pin_'+fmt
    _record(name,reset=True,status='started_not_passed')
    ready=_ready()  # Existing real OS RAM reserve and CUDA initialization guard.
    values={}
    for index in range(2):
        for bank,rows,columns in [('gate_up_proj',1024,512),('down_proj',512,512)]:
            fields=quantized_bank(fmt,experts=64,rows=rows,width=columns)
            values.update({f'model.layers.{index}.mlp.experts.{bank}.{key}':value for key,value in fields.items()})
    values['anchor']=torch.arange(16,dtype=torch.float32)
    path=tmp_path/(fmt+'.safetensors')
    save_file(values,path,metadata={'drama3_config':'{}','drama3_format':fmt})
    values.clear();fields=None
    diagnostic={}
    state=reader.load_torch_file(path,diagnostics=diagnostic,_with_ownership=True)
    assert diagnostic['actual_reader']=='official_cpu_pread'
    assert diagnostic['expert_ownership']['read_certificate_issued']
    model=_expert_model(experts=64,width=512,intermediate=512)
    witness=own.assign_owned(model,state,diagnostics=diagnostic)
    state.clear();del state
    proof=own.finish_assignment(witness,model,diagnostics=diagnostic)
    assert proof is not None,diagnostic
    memory=diagnostic['expert_ownership']['assign_memory']
    assert memory['status']=='passed' and memory['cast_workspace_bytes']==0
    assert memory['available_bytes']>=memory['required_bytes']
    before=_host_stats()
    store=acceleration.expert_store(model,0,'cuda',ownership=proof)
    assert own.is_incremental(proof,model,store),store.admission
    status='failed';releases=[]
    try:
        for index in range(2):
            admit=own.begin_layer(proof,model,index)
            assert admit['eligible']
            original=_bank_source(model.model.layers[index].mlp.experts)
            model.model.layers[index].mlp.experts=store.add(model.model.layers[index].mlp.experts)
            assert store.pin_memory and store.pin_fallback_reason is None,store.pin_fallback_reason
            for source,destination in zip(original,store.banks[index]):_assert_pinned_exact(source,destination)
            source=destination=None
            released=own.complete_layer(proof,model,index)
            assert released['source_row_storage_impl_expired']
            if os.name=='nt':
                for source in original:
                    region=_region(source['qdata_pointer'])
                    assert region.State==0x10000 and region.AllocationBase is None
            source=None
            releases.append(released)
        assert own.describe(proof,model)['completed_layers']==2
        assert all(tensor.is_pinned() for banks in store.banks for bank in banks for tensor in bank.fields.values())
        assert store.pinned_bytes>0 and store.backend_calls=={'cuda':0,'torch':0}
        status='pin_and_lifetime_checks_complete_no_generation'
    finally:
        store.close();own.close(proof);gc.collect()
        after=_host_stats()
        _record(name,status=status,readiness=ready,reader=diagnostic,admission=store.admission,
                releases=releases,before=before,after=after,generation_calls=0,
                actual_pinned_bytes=store.pinned_bytes,
                note='Real owned reader/assign/selection/pin/release only; CUDA inference and Graph are not executed.')
    _assert_allocated_restored(before,after)
    _record(name,status='passed_pin_ownership_not_inference')
