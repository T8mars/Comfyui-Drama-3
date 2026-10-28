"""Real tiny CPU reader/assign provenance; CPU clones are not page locking."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import safetensors
from safetensors.torch import save_file
import torch

from drama3_t8.native import checkpoint_reader as reader
from test_checkpoint_reader import _expert_model, _require_real_headroom
from test_quantized_loading import quantized_bank


def fixture_file(path, fmt='w6a8_int8', *, unsigned=False):
    values={}
    for index in range(2):
        for bank,rows in [('gate_up_proj',512),('down_proj',256)]:
            fields=quantized_bank(fmt,experts=2,rows=rows,width=256)
            if unsigned:fields['weight']=fields['weight'].view(torch.uint8)
            values.update({f'model.layers.{index}.mlp.experts.{bank}.{name}':value for name,value in fields.items()})
    values['anchor']=torch.arange(16,dtype=torch.float32)
    save_file(values,path,metadata={'drama3_config':'{}','drama3_format':fmt})
    return path


def owned_state(tmp_path,fmt='w6a8_int8',**kwargs):
    path=fixture_file(tmp_path/(fmt+'.safetensors'),fmt,**kwargs)
    _require_real_headroom(path)
    audit={}
    state=reader.load_torch_file(path,diagnostics=audit,_with_ownership=True)
    return path,state,audit


def assigned(tmp_path,fmt='w6a8_int8'):
    from drama3_t8.native import checkpoint_ownership as own
    path,state,audit=owned_state(tmp_path,fmt)
    model=_expert_model(width=256,intermediate=256)
    witness=own.assign_owned(model,state,diagnostics=audit)
    state.clear();state=None
    proof=own.finish_assignment(witness,model,diagnostics=audit)
    assert proof is not None,audit
    return path,model,proof,audit


def test_public_reader_exact_dict_and_private_owned_return_preserves_metadata(tmp_path):
    path=fixture_file(tmp_path/'public.safetensors')
    _require_real_headroom(path)
    public,metadata=reader.load_torch_file(path,return_metadata=True)
    assert type(public) is dict and metadata=={'drama3_config':'{}','drama3_format':'w6a8_int8'}
    public.clear()
    state,actual=reader.load_torch_file(path,return_metadata=True,_with_ownership=True)
    assert isinstance(state,dict) and type(state) is not dict and actual==metadata
    assert type(dict(state)) is dict and type(state.copy()) is dict
    state.clear()


@pytest.mark.parametrize('mode',['mmap','custom_opener','custom_host','memory_provider'])
def test_compatibility_paths_never_get_internal_authority(tmp_path,mode):
    from drama3_t8.native import checkpoint_ownership as own
    path=fixture_file(tmp_path/'compat.safetensors')
    _require_real_headroom(path)
    options=dict(_with_ownership=True)
    if mode=='mmap':options['prefer_pread']=False
    elif mode=='custom_opener':options['_safe_open']=safetensors.safe_open
    elif mode=='custom_host':options['_host']=reader._host_context()
    else:options['policy']=reader.ReaderPolicy(memory_snapshot=reader.psutil.virtual_memory)
    state=reader.load_torch_file(path,**options)
    assert type(state) is dict
    model=_expert_model(width=256,intermediate=256)
    assert own.assign_owned(model,state) is None
    state.clear()
    assert own.finish_assignment(None,model) is None


@pytest.mark.parametrize('fmt',['int8_tensorwise','asym_w4a8_int8','w6a8_int8'])
def test_actual_assign_clear_issues_weak_typed_proof_with_real_workspace_guard(tmp_path,fmt):
    from drama3_t8.native import checkpoint_ownership as own
    _,model,proof,audit=assigned(tmp_path,fmt)
    description=own.describe(proof,model)
    assert description['source_tensor_count']>0 and description['derived_parameter_count']>0
    guard=audit['expert_ownership']['assign_memory']
    assert guard['clone_workspace_bytes']>0 and guard['extra_bank_parameter_bytes']>0
    assert guard['required_bytes']==guard['clone_workspace_bytes']+guard['cast_workspace_bytes']+guard['extra_bank_parameter_bytes']+guard['reserve_bytes']
    weak=weakref.ref(model)
    del model;gc.collect()
    assert weak() is None
    own.close(proof)


def test_assign_low_ram_refuses_before_real_parameter_clones(tmp_path,monkeypatch):
    from drama3_t8.native import checkpoint_ownership as own
    from comfy_kitchen.tensor.base import BaseLayoutParams
    _,state,audit=owned_state(tmp_path)
    model=_expert_model(width=256,intermediate=256)
    memory=reader.psutil.virtual_memory();calls=[]
    original=BaseLayoutParams.clone
    def observe(params):calls.append(1);return original(params)
    monkeypatch.setattr(BaseLayoutParams,'clone',observe)
    monkeypatch.setattr(reader.psutil,'virtual_memory',lambda:SimpleNamespace(total=memory.total,available=0))
    with pytest.raises(reader.ReaderResourceError,match='assign'):
        own.assign_owned(model,state,diagnostics=audit)
    assert not calls
    assert all(layer.mlp.experts.gate_up_proj.weight is None for layer in model.model.layers)
    state.clear()


def test_unsigned_qdata_is_preflighted_as_protected_cast_and_never_incremental(tmp_path):
    from drama3_t8.native import checkpoint_ownership as own
    _,state,audit=owned_state(tmp_path,unsigned=True)
    model=_expert_model(width=256,intermediate=256)
    witness=own.assign_owned(model,state,diagnostics=audit)
    assert witness is None
    assert audit['expert_ownership']['assign_memory']['cast_workspace_bytes']>0
    assert 'cast' in audit['expert_ownership']['fallback_reason'].lower()
    assert model.model.layers[0].mlp.experts.gate_up_proj.weight._qdata.dtype==torch.int8
    state.clear()
    assert own.finish_assignment(witness,model) is None


def test_unsigned_cast_low_ram_also_refuses_before_any_clone(tmp_path,monkeypatch):
    from drama3_t8.native import checkpoint_ownership as own
    from comfy_kitchen.tensor.base import BaseLayoutParams
    _,state,audit=owned_state(tmp_path,unsigned=True)
    model=_expert_model(width=256,intermediate=256)
    memory=reader.psutil.virtual_memory();calls=[]
    monkeypatch.setattr(BaseLayoutParams,'clone',lambda params:calls.append(1))
    monkeypatch.setattr(reader.psutil,'virtual_memory',lambda:SimpleNamespace(total=memory.total,available=0))
    with pytest.raises(reader.ReaderResourceError):own.assign_owned(model,state,diagnostics=audit)
    assert not calls
    state.clear()


@pytest.mark.parametrize('forgery',[None,{}, {'actual_reader':'official_cpu_pread'},object()])
def test_json_and_unissued_tokens_cannot_authorize(forgery):
    from drama3_t8.native import checkpoint_ownership as own
    model=_expert_model(width=256,intermediate=256)
    assert own.initial_admission(forgery,model) is None


def test_state_copy_loses_authority_even_when_tensor_values_are_real(tmp_path):
    from drama3_t8.native import checkpoint_ownership as own
    _,state,_=owned_state(tmp_path)
    copied=dict(state);state.clear()
    model=_expert_model(width=256,intermediate=256)
    assert own.assign_owned(model,copied) is None
    copied.clear()


def test_unreviewed_kitchen_build_retains_legacy_assign_without_incremental_authority(tmp_path,monkeypatch):
    import comfy_kitchen.tensor.base as base
    from drama3_t8.native import checkpoint_ownership as own
    _,state,audit=owned_state(tmp_path)
    model=_expert_model(width=256,intermediate=256)
    altered=tmp_path/'unreviewed-base.py'
    altered.write_bytes(Path(base.__file__).read_bytes()+b'\n# unreviewed local build\n')
    monkeypatch.setattr(base,'__file__',str(altered))
    witness=own.assign_owned(model,state,diagnostics=audit)
    assert witness is None
    assert audit['expert_ownership']['policy']=='legacy_full_expert'
    assert audit['expert_ownership']['fallback_reason']=='Unreviewed official assign/Kitchen implementation: base'
    assert model.model.layers[0].mlp.experts.gate_up_proj.weight._qdata.device.type=='cpu'
    state.clear()
    assert own.finish_assignment(witness,model) is None


def test_late_model_alias_invalidates_before_next_layer_without_tensor_traceback(tmp_path,monkeypatch):
    from drama3_t8.native import checkpoint_ownership as own
    from drama3_t8.native import acceleration,offload_accelerated as offload,kernels
    _,model,proof,audit=assigned(tmp_path)
    # Unit selection only: fake capability discovery never initializes CUDA.
    monkeypatch.setattr(acceleration.mm,'get_torch_device',lambda:torch.device('cuda'))
    monkeypatch.setattr(kernels,'available',lambda device:True)
    monkeypatch.setattr(offload,'pinned',lambda value:value.clone())
    store=acceleration.expert_store(model,0,'cuda',ownership=proof)
    assert own.is_incremental(proof,model,store)
    try:
        own.begin_layer(proof,model,0)
        model.model.layers[0].mlp.experts=store.add(model.model.layers[0].mlp.experts)
        own.complete_layer(proof,model,0)
        model.model.layers[1].mlp.experts.gate_up_proj.weight._qdata=model.model.layers[1].mlp.experts.gate_up_proj.weight._qdata.clone()
        with pytest.raises(own.OwnershipViolation) as caught:own.begin_layer(proof,model,1)
        assert caught.value is not None
        assert len(store.banks)==1
    finally:
        store.close();own.close(proof)
    assert not torch.cuda.is_initialized()


def test_dropping_unassigned_owned_state_releases_actual_storage_without_explicit_close(tmp_path):
    from drama3_t8.native import checkpoint_ownership as own
    before=len(own._READS)
    _,state,_=owned_state(tmp_path)
    handles=[own._StorageWeak(value.untyped_storage()) for value in state.values()]
    state_weak=weakref.ref(state);token_weak=weakref.ref(state._source_token)
    del state;gc.collect()
    assert state_weak() is None and token_weak() is None
    assert all(handle.expired() for handle in handles)
    assert len(own._READS)==before


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_unexpected_derive_error_preserves_original_exception_and_owned_cleanup(tmp_path,monkeypatch,failure):
    from drama3_t8.native import checkpoint_ownership as own,loader,acceleration
    path=fixture_file(tmp_path/'unexpected.safetensors')
    _require_real_headroom(path)
    original_read=reader.load_torch_file
    original_derive=own._derive_recognized_params
    source_handles=[];assigned_handles=[];made=[]
    def observe_read(*args,**kwargs):
        result=original_read(*args,**kwargs)
        source_handles.extend(own._StorageWeak(value.untyped_storage()) for value in result[0].values())
        return result
    def make(*args,**kwargs):
        result=_expert_model(width=256,intermediate=256);made.append(result);return result
    def observe_derive(model,*args,**kwargs):
        value=None
        try:
            for _,value in acceleration.physical_tensors(model):
                assigned_handles.append(own._StorageWeak(value.untyped_storage()))
            return original_derive(model,*args,**kwargs)
        finally:value=None
    error=failure('original unexpected comparison failure')
    def fail_equal(*args,**kwargs):
        try:raise error
        finally:args=();kwargs.clear()  # Simulate a native error, not a test-owned Tensor frame.
    monkeypatch.setattr(reader,'load_torch_file',observe_read)
    monkeypatch.setattr(loader,'Drama3Model',make)
    monkeypatch.setattr(loader,'parse_config',lambda value:SimpleNamespace())
    monkeypatch.setattr(loader.comfy.model_patcher,'CoreModelPatcher',lambda model,**kwargs:SimpleNamespace(model=model,**kwargs))
    monkeypatch.setattr(own,'_derive_recognized_params',observe_derive)
    monkeypatch.setattr(torch,'equal',fail_equal)
    with pytest.raises(failure) as caught:loader.load_drama3(str(path),experts_in_ram=True,moe_backend='auto')
    assert caught.value is error
    gc.collect()
    assert made and not list(made[0].parameters()) and not list(made[0].buffers())
    assert source_handles and all(handle.expired() for handle in source_handles)
    assert assigned_handles and all(handle.expired() for handle in assigned_handles)


def test_owned_proof_is_bound_to_model_and_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from drama3_t8.native import checkpoint_ownership as own
    _,model,proof,_=assigned(tmp_path)
    try:
        assert own.initial_admission(proof,_expert_model(width=256,intermediate=256)) is None
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(own.initial_admission,proof,model).result() is None
        assert own.initial_admission(proof,model) is not None
    finally:own.close(proof)


def test_unexpected_torch_load_state_dict_retained_cause_releases_source_and_keeps_frames(tmp_path,monkeypatch):
    import traceback
    from drama3_t8.native import checkpoint_ownership as own,loader
    path=fixture_file(tmp_path/'torch-failure.safetensors')
    _require_real_headroom(path)
    source_handles=[];made=[]
    original_read=reader.load_torch_file
    def observe_read(*args,**kwargs):
        result=original_read(*args,**kwargs)
        source_handles.extend(own._StorageWeak(value.untyped_storage()) for value in result[0].values())
        return result
    caller_marker=['active caller remains intact']
    try:raise ValueError('original caller cause')
    except ValueError as previous:cause=previous
    error=RuntimeError('real load_state_dict inner module failure')
    def make(*args,**kwargs):
        model=_expert_model(width=256,intermediate=256)
        module=model.model.layers[0].mlp.experts.gate_up_proj
        original=module._load_from_state_dict
        def fail(*args,**kwargs):
            try:
                original(*args,**kwargs)
                raise error from cause
            finally:args=();kwargs.clear()  # External callback keeps no user-owned source.
        module._load_from_state_dict=fail
        made.append(model)
        return model
    monkeypatch.setattr(reader,'load_torch_file',observe_read)
    monkeypatch.setattr(loader,'Drama3Model',make)
    monkeypatch.setattr(loader,'parse_config',lambda value:SimpleNamespace())
    monkeypatch.setattr(loader.comfy.model_patcher,'CoreModelPatcher',lambda model,**kwargs:SimpleNamespace(model=model,**kwargs))
    with pytest.raises(RuntimeError) as caught:loader.load_drama3(str(path),experts_in_ram=True,moe_backend='auto')
    assert caught.value is error and caught.value.__cause__ is cause
    assert caller_marker==['active caller remains intact']
    frames=traceback.extract_tb(caught.value.__traceback__)
    assert any(frame.name=='load_state_dict' for frame in frames)
    assert any(frame.name=='fail' for frame in frames)
    gc.collect()
    assert source_handles and all(handle.expired() for handle in source_handles)
    assert made and not list(made[0].parameters()) and not list(made[0].buffers())


def test_failed_loader_preserves_unrelated_cause_and_external_callback_locals(tmp_path,monkeypatch):
    from drama3_t8.native import loader
    path=fixture_file(tmp_path/'foreign-cause.safetensors')
    _require_real_headroom(path)
    def past_user_function():
        unrelated_user_sentinel=object()
        try:raise ValueError('past user cause')
        except ValueError as error:return error
    cause=past_user_function()
    cause_frame=cause.__traceback__.tb_frame
    sentinel=cause_frame.f_locals['unrelated_user_sentinel']
    error=RuntimeError('current loader failure')
    def make(*args,**kwargs):
        model=_expert_model(width=256,intermediate=256)
        module=model.model.layers[0].mlp.experts.gate_up_proj
        def fail(*args,**kwargs):
            external_callback_sentinel='external callback remains inspectable'
            try:raise error from cause
            finally:args=();kwargs.clear()
        module._load_from_state_dict=fail
        return model
    monkeypatch.setattr(loader,'Drama3Model',make)
    monkeypatch.setattr(loader,'parse_config',lambda value:SimpleNamespace())
    monkeypatch.setattr(loader.comfy.model_patcher,'CoreModelPatcher',lambda model,**kwargs:SimpleNamespace(model=model,**kwargs))
    with pytest.raises(RuntimeError) as caught:loader.load_drama3(str(path),experts_in_ram=True,moe_backend='auto')
    assert caught.value is error and caught.value.__cause__ is cause
    assert cause_frame.f_locals.get('unrelated_user_sentinel') is sentinel
    current=caught.value.__traceback__
    callback=None
    while current:
        if current.tb_frame.f_code.co_name=='fail':callback=current.tb_frame
        current=current.tb_next
    assert callback is not None
    assert callback.f_locals.get('external_callback_sentinel')=='external callback remains inspectable'


def test_failed_load_frame_cleanup_requires_exact_active_owner_not_only_source_path():
    import inspect
    from drama3_t8.native import loader
    namespace={}
    exec(compile('def internal_failure():\n    internal_sentinel = object()\n    try: raise RuntimeError("internal failure")\n    except RuntimeError as error: return error\n',loader.__file__,'exec'),namespace)
    previous=namespace['internal_failure']()
    previous_frame=previous.__traceback__.tb_frame
    previous_sentinel=previous_frame.f_locals['internal_sentinel']
    def active_owner():
        current=namespace['internal_failure']()
        current.__cause__=previous
        frame=current.__traceback__.tb_frame
        loader._clear_failed_load_frames(current,inspect.currentframe())
        assert current.__cause__ is previous
        assert frame.f_locals=={}
        return current
    result=active_owner()
    assert result.__cause__ is previous
    assert previous_frame.f_locals.get('internal_sentinel') is previous_sentinel
