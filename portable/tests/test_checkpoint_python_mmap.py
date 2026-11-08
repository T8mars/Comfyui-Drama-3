"""Windows lazy fallback: exact bytes, live aliases and deterministic cleanup."""
from dataclasses import replace
import gc
import weakref

import pytest
from safetensors.torch import save_file
import torch

from drama3_t8.native import checkpoint_reader as reader


@pytest.mark.parametrize('metadata',[None,{}, {'source':'new official weights'}])
def test_lazy_fallback_keeps_all_native_bytes_and_mapping_until_last_alias(tmp_path,monkeypatch,metadata):
    values={'fp8':torch.tensor([0,127,128,255],dtype=torch.uint8).view(torch.float8_e4m3fn),
        'bf16':torch.tensor([-0.,float('nan'),1.25],dtype=torch.bfloat16),
        'empty':torch.empty(0,3),'scalar':torch.tensor(42,dtype=torch.int64)}
    path=tmp_path/'weights.safetensors';save_file(values,path,metadata=metadata)
    real=reader._host_context()
    def unsafe(*args,**kwargs):raise AssertionError('Rust storage slicing must not be used')
    host=replace(real,aimdo_enabled=False,disable_mmap=False,load_host=unsafe)
    monkeypatch.setattr(reader,'_WINDOWS_MMAP',True)
    monkeypatch.setattr(reader,'_WINDOWS_MMAP_MIN_BYTES',1)
    mappings=[];original=reader.mmap.mmap
    def mapped(*args,**kwargs):
        result=original(*args,**kwargs);mappings.append(weakref.ref(result));return result
    monkeypatch.setattr(reader.mmap,'mmap',mapped)
    audit={}
    state,actual=reader.load_torch_file(path,prefer_pread=False,return_metadata=True,_host=host,diagnostics=audit)
    assert actual==metadata
    assert audit['actual_reader']=='python_owned_mmap' and audit['host_storage_mode']=='python_mmap'
    for name in values:
        expected=values[name].reshape(-1).view(torch.uint8)
        assert torch.equal(state[name].reshape(-1).view(torch.uint8),expected)
        assert state[name].dtype==values[name].dtype and state[name].shape==values[name].shape
        assert state[name].untyped_storage()._comfy_source_path==str(path.resolve())
    alias=state['fp8'][1:];state.clear();gc.collect()
    assert mappings[0]() is not None
    assert alias.view(torch.uint8).tolist()==[127,128,255]
    del alias;gc.collect()
    assert mappings[0]() is None


@pytest.mark.parametrize('failure',['cancel','annotate'])
def test_lazy_fallback_failure_drops_partial_mapping_even_with_retained_exception(tmp_path,monkeypatch,failure):
    path=tmp_path/'weights.safetensors';save_file({'a':torch.ones(4),'b':torch.ones(8)},path)
    real=reader._host_context();checks=[0];mappings=[];original=reader.mmap.mmap
    def mapped(*args,**kwargs):
        result=original(*args,**kwargs);mappings.append(weakref.ref(result));return result
    def cancelled():
        checks[0]+=1
        return checks[0]>=5
    def annotate(*args):raise RuntimeError('annotation failed')
    monkeypatch.setattr(reader.mmap,'mmap',mapped)
    monkeypatch.setattr(reader,'_WINDOWS_MMAP',True)
    monkeypatch.setattr(reader,'_WINDOWS_MMAP_MIN_BYTES',1)
    host=replace(real,aimdo_enabled=False,disable_mmap=False,
        annotate_state_dict=annotate if failure=='annotate' else real.annotate_state_dict)
    policy=reader.ReaderPolicy(cancelled=cancelled if failure=='cancel' else None)
    error=reader.ReaderCancelled if failure=='cancel' else RuntimeError
    with pytest.raises(error) as retained:
        reader.load_torch_file(path,prefer_pread=False,_host=host,policy=policy)
    assert retained.value is not None
    gc.collect()
    assert mappings and all(reference() is None for reference in mappings)
