"""Native CPU checkpoint selection, packed storage and failure ownership.

CPU replacement tests exercise real assign=True and HostBank replacement with
ordinary CPU clones. They do not claim real page locking or CUDA acceptance.
"""
import ctypes
from ctypes import wintypes
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import weakref

import pytest
import safetensors
from safetensors.torch import save_file
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native import checkpoint_reader as candidate


def raw(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _write(path, header, data=b''):
    text = json.dumps(header).encode() if not isinstance(header, bytes) else header
    text += b' ' * ((-len(text)) % 8)
    path.write_bytes(struct.pack('<Q', len(text)) + text + data)
    return path


def _require_real_headroom(path):
    plan = candidate.inspect_header(path, candidate.ReaderPolicy())
    memory = candidate.psutil.virtual_memory()
    reserve = max(candidate.MIN_RESERVE, min(candidate.MAX_RESERVE, int(memory.total * .125)))
    need = plan['source_cpu_bytes'] + plan['temporary_margin_bytes'] + reserve
    if memory.available < need:
        pytest.skip(f'REAL_RESOURCE_BLOCKED available={memory.available} required={need} reserve={reserve}; no fabricated availability')


@pytest.fixture
def fixture_path(tmp_path):
    # Small real safetensors with non-finite float values, signed zero, empty
    # and scalar tensors, and dtypes that must never silently cast to BF16.
    state = dict(
        scalar=torch.tensor(-0., dtype=torch.float64),
        empty=torch.empty(0, 3, dtype=torch.float32),
        flag=torch.tensor([True, False]),
        int8=torch.tensor([-128, -1, 0, 127], dtype=torch.int8),
        uint8=torch.tensor([0, 255], dtype=torch.uint8),
        int16=torch.tensor([-32768, 32767], dtype=torch.int16),
        int32=torch.tensor([-2147483648, 2147483647], dtype=torch.int32),
        int64=torch.tensor([-2**63, 2**63-1], dtype=torch.int64),
        fp16=torch.tensor([-0., float('inf'), -1.25], dtype=torch.float16),
        bf16=torch.tensor([-0., float('nan'), 1.25], dtype=torch.bfloat16),
        fp32=torch.tensor([float('-inf'), float('nan'), -0., .5], dtype=torch.float32),
        fp64=torch.tensor([float('nan'), -0., 1e-200], dtype=torch.float64),
        fp8=torch.tensor([0, 128, 127, 255], dtype=torch.uint8).view(torch.float8_e4m3fn))
    for name, dtype in [('u16',torch.uint16),('u32',torch.uint32),('u64',torch.uint64)]:
        state[name]=torch.tensor([0,1],dtype=dtype)
    path = tmp_path/'mixed.safetensors'
    save_file(state,path,metadata={'unicode':'模型—测试','json':'{"kind":"quant", "version":1}','empty':''})
    return path


def test_bounded_header_matches_physical_fixture(fixture_path):
    plan = candidate.inspect_header(fixture_path,candidate.ReaderPolicy())
    assert plan['source_cpu_bytes'] == sum(v['data_offsets'][1]-v['data_offsets'][0] for v in plan['tensors'].values())
    assert plan['source_cpu_bytes'] == fixture_path.stat().st_size-8-plan['header_bytes']
    assert plan['temporary_margin_bytes'] >= 2*plan['maximum_physical_bank_bytes']


@pytest.mark.parametrize('kind', ['huge_length','truncated','duplicate','shape_bytes','overlap','hole','unknown_dtype','metadata','negative_shape'])
def test_invalid_headers_reject_without_tensor_read(tmp_path,kind):
    path=tmp_path/'invalid.safetensors'
    if kind=='huge_length':path.write_bytes(struct.pack('<Q',2**63))
    elif kind=='truncated':path.write_bytes(struct.pack('<Q',128)+b'{}')
    elif kind=='duplicate':_write(path,b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',b'a')
    else:
        h={'x':dict(dtype='U8',shape=[2],data_offsets=[0,2])};data=b'ab'
        if kind=='shape_bytes':h['x']['shape']=[1]
        if kind=='overlap':h['y']=dict(h['x'])
        if kind=='hole':h['x']=dict(dtype='U8',shape=[1],data_offsets=[1,2])
        if kind=='unknown_dtype':h['x']['dtype']='UNKNOWN128'
        if kind=='metadata':h['__metadata__']={'value':3}
        if kind=='negative_shape':h['x']['shape']=[-2]
        _write(path,h,data)
    with pytest.raises((candidate.ReaderError,ValueError)):
        candidate.inspect_header(path,candidate.ReaderPolicy())


def test_max_tensor_count_is_bounded(fixture_path):
    with pytest.raises(candidate.UnsupportedReader,match='Tensor count'):
        candidate.inspect_header(fixture_path,candidate.ReaderPolicy(max_tensors=1))


def test_pread_resource_guard_preserves_compatible_host_mmap(fixture_path):
    called=[]
    def opener(filename,framework,device,*,backend):called.append(filename);raise AssertionError
    policy=candidate.ReaderPolicy(memory_snapshot=lambda:SimpleNamespace(total=64*candidate.GiB,available=0))
    audit={}
    actual=candidate.load_torch_file(fixture_path,policy=policy,diagnostics=audit,_safe_open=opener)
    assert called==[]
    assert audit['memory_checks'][0]['reserve_bytes']==8*candidate.GiB
    assert actual['int8'].dtype==torch.int8
    assert audit['actual_reader']=='host' and audit['host_storage_mode']=='mmap'
    assert 'available RAM' in audit['fallback_reason']


def test_real_resource_guard_preserves_actual_snapshot(fixture_path):
    before=candidate.psutil.virtual_memory();audit={}
    reserve=max(candidate.MIN_RESERVE,min(candidate.MAX_RESERVE,int(before.total*.125)))
    if before.available>=reserve+1024**2:
        pytest.skip('Actual RAM has returned; refusal specifically belongs to the constrained interval')
    actual=candidate.load_torch_file(fixture_path,diagnostics=audit)
    assert audit['memory_checks'][0]['available_bytes']<=reserve+1024**2
    assert audit['actual_reader']=='host' and actual['bf16'].dtype==torch.bfloat16


def test_memory_query_failure_selects_lazy_host_compatibility(fixture_path):
    def broken():raise OSError('query failed')
    audit={}
    actual=candidate.load_torch_file(fixture_path,policy=candidate.ReaderPolicy(memory_snapshot=broken),diagnostics=audit)
    assert audit['actual_reader']=='host' and 'Cannot verify RAM' in audit['fallback_reason']
    assert actual['int8'].dtype==torch.int8


def test_precancel_returns_no_partial_state(fixture_path):
    with pytest.raises(candidate.ReaderCancelled):
        candidate.load_torch_file(fixture_path,policy=candidate.ReaderPolicy(cancelled=lambda:True))


@pytest.mark.parametrize('device',['cuda:0','meta'])
def test_non_cpu_storage_explicitly_refused(fixture_path,device):
    with pytest.raises(candidate.UnsupportedReader,match='CPU checkpoint storage only'):
        candidate.load_torch_file(fixture_path,device=device)


@pytest.mark.parametrize('return_metadata',[False,True])
def test_actual_host_mmap_equivalence_and_storage_annotations(fixture_path,return_metadata):
    _require_real_headroom(fixture_path)
    import comfy.utils,comfy.memory_management,comfy.storage
    before=(comfy.memory_management.aimdo_enabled,comfy.utils.DISABLE_MMAP,comfy.utils.load_torch_file)
    expected=comfy.utils.load_torch_file(str(fixture_path),return_metadata=return_metadata)
    audit={};actual=candidate.load_torch_file(fixture_path,return_metadata=return_metadata,diagnostics=audit)
    if return_metadata:
        actual,am=actual;expected,em=expected;assert am==em
    assert actual.keys()==expected.keys()
    for key in actual:
        assert actual[key].dtype==expected[key].dtype and actual[key].shape==expected[key].shape
        assert raw(actual[key])==raw(expected[key])
        assert actual[key].untyped_storage()._comfy_source_path==str(fixture_path.resolve())
        assert not hasattr(actual[key].untyped_storage(),'_comfy_tensor_mmap_refs')
        assert not hasattr(actual[key].untyped_storage(),'_comfy_tensor_file_slice')
    assert before==(comfy.memory_management.aimdo_enabled,comfy.utils.DISABLE_MMAP,comfy.utils.load_torch_file)
    assert audit['actual_reader']=='official_cpu_pread' and audit['status']=='loaded'


def test_missing_metadata_is_none(tmp_path):
    path=tmp_path/'no-metadata.safetensors';save_file({'x':torch.ones(1)},path)
    _require_real_headroom(path)
    state,metadata=candidate.load_torch_file(path,return_metadata=True)
    assert metadata is None and raw(state['x'])==raw(torch.ones(1))


@pytest.mark.parametrize('fmt',['int8_tensorwise','asym_w4a8_int8','w6a8_int8'])
def test_official_cpu_quantized_fixture_fields_unchanged(tmp_path,fmt):
    from comfy_kitchen.backends.eager import quantization,w4a8_int8
    generator=torch.Generator().manual_seed(909)
    weight=torch.randn(256,256,generator=generator).to(torch.bfloat16)*.025
    prefix='model.layers.0.mlp.experts.down_proj.'
    conf=dict(format=fmt,num_experts=1,convrot=True,convrot_groupsize=256)
    if fmt=='int8_tensorwise':
        q,s=quantization.quantize_int8_convrot_weight(weight,256)
        state={prefix+'weight':q.unsqueeze(0),prefix+'weight_scale':s.unsqueeze(0)}
    else:
        q,s,ch,corr,codebook=w4a8_int8.quantize_w4a8_int8_weight(weight,group_size=16,convrot_groupsize=256,bits=4 if fmt=='asym_w4a8_int8' else 6)
        assert corr is None
        conf['group_size']=16
        state={prefix+'weight':q.unsqueeze(0),prefix+'weight_s_rel':s.view(torch.uint8).unsqueeze(0),prefix+'weight_s_channel':ch.unsqueeze(0)}
        if codebook is not None:state[prefix+'weight_codebook']=codebook
    state[prefix+'comfy_quant']=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)
    path=tmp_path/(fmt+'.safetensors');save_file(state,path,metadata={'format':fmt})
    _require_real_headroom(path)
    actual,metadata=candidate.load_torch_file(path,return_metadata=True)
    assert metadata=={'format':fmt}
    assert actual.keys()==state.keys()
    for name in state:
        assert actual[name].dtype==state[name].dtype and actual[name].shape==state[name].shape
        assert raw(actual[name])==raw(state[name])


class OpenerSpy:
    def __init__(self,fail_at=None):self.refs=[];self.closed=0;self.gets=0;self.fail_at=fail_at
    def open(self,filename,framework,device,*,backend):
        spy=self
        class Opened:
            def __init__(self):self.inner=safetensors.safe_open(filename,framework=framework,device=device,backend=backend)
            def __enter__(self):self.inner.__enter__();return self
            def __exit__(self,*exc):spy.closed+=1;return self.inner.__exit__(*exc)
            def keys(self):return self.inner.keys()
            def metadata(self):return self.inner.metadata()
            def get_tensor(self,key):
                spy.gets+=1
                if spy.gets==spy.fail_at:raise OSError('injected next-tensor read failure')
                value=self.inner.get_tensor(key);spy.refs.append(weakref.ref(value));return value
        return Opened()


@pytest.mark.parametrize('failure',['cancel_after_one','read_after_one','budget_after_one','annotation'])
def test_error_cancel_partial_cleanup_even_retained_exception(fixture_path,failure):
    _require_real_headroom(fixture_path)
    spy=OpenerSpy(fail_at=2 if failure=='read_after_one' else None)
    policy=candidate.ReaderPolicy(cancelled=(lambda:spy.gets>=1) if failure=='cancel_after_one' else None)
    if failure=='budget_after_one':
        def memory():
            actual=candidate.psutil.virtual_memory()
            return SimpleNamespace(total=actual.total,available=0 if spy.gets>=1 else actual.available)
        policy=replace(policy,memory_snapshot=memory)
    host=None
    if failure=='annotation':
        real=candidate._host_context()
        def broken_annotation(state,path):raise OSError('injected annotation failure')
        host=replace(real,annotate_state_dict=broken_annotation)
    audit={}
    expected=(candidate.ReaderCancelled if failure=='cancel_after_one' else
              candidate.ReaderResourceError if failure=='budget_after_one' else OSError)
    with pytest.raises(expected) as retained:
        candidate.load_torch_file(fixture_path,policy=policy,diagnostics=audit,_safe_open=spy.open,_host=host)
    assert retained.value is not None  # Keep traceback alive while checking ours.
    gc.collect()
    assert spy.closed==1 and spy.gets>=1
    assert all(ref() is None for ref in spy.refs)
    assert audit['status']=='failed_or_cancelled'
    assert audit['cleared_partial_tensors']>=1


@pytest.mark.parametrize('mode',['aimdo','legacy_api','rejects_pread'])
def test_unknown_api_aimdo_use_official_compatibility_without_unrelated_clone_budget(fixture_path,mode):
    _require_real_headroom(fixture_path)
    real=candidate._host_context();calls=[]
    def host_load(*args,**kwargs):calls.append((args,kwargs));return real.load_host(*args,**kwargs)
    host=replace(real,aimdo_enabled=mode=='aimdo',load_host=host_load)
    def legacy(filename,framework,device):raise AssertionError('legacy opener must never be called')
    def rejected(filename,framework,device,*,backend):raise ValueError('pread unsupported')
    opener=legacy if mode=='legacy_api' else rejected if mode=='rejects_pread' else None
    audit={}
    actual,metadata=candidate.load_torch_file(fixture_path,return_metadata=True,_host=host,_safe_open=opener,
                        diagnostics=audit)
    assert len(calls)==1 and metadata['unicode']=='模型—测试'
    assert audit['actual_reader']=='host' and audit['fallback_reason']
    assert audit['status']=='loaded'
    assert all(check['stage']!='host_eager_copy' for check in audit.get('memory_checks',[]))
    assert actual['int8'].dtype==torch.int8


def test_current_host_annotation_hook_is_called_not_reimplemented(fixture_path):
    _require_real_headroom(fixture_path)
    real=candidate._host_context();calls=[]
    def extended(state,path):
        real.annotate_state_dict(state,path);calls.append(path)
        for value in state.values():value.untyped_storage()._prototype_future_annotation='preserved'
    actual=candidate.load_torch_file(fixture_path,_host=replace(real,annotate_state_dict=extended))
    assert calls==[str(fixture_path.resolve())]
    assert all(value.untyped_storage()._prototype_future_annotation=='preserved' for value in actual.values())


def test_explicit_host_fallback_cancel_after_opaque_read_clears_return(fixture_path):
    _require_real_headroom(fixture_path)
    real=candidate._host_context();cancelled=[False];refs=[]
    def host_load(*args,**kwargs):
        result=real.load_host(*args,**kwargs)
        refs.extend(weakref.ref(value) for value in result[0].values())
        cancelled[0]=True
        return result
    host=replace(real,aimdo_enabled=True,load_host=host_load);audit={}
    with pytest.raises(candidate.ReaderCancelled) as retained:
        candidate.load_torch_file(fixture_path,return_metadata=True,_host=host,diagnostics=audit,
             policy=candidate.ReaderPolicy(cancelled=lambda:cancelled[0]))
    assert retained.value is not None
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert audit['status']=='failed_or_cancelled_host_fallback' and audit['cleared_partial_tensors']>0


@pytest.mark.parametrize('change',['mtime','header_restore_mtime'])
def test_source_changes_reject_and_clear_partial_state(fixture_path,change):
    _require_real_headroom(fixture_path)
    import os
    spy=OpenerSpy();old_stat=fixture_path.stat()
    def changed(filename,framework,device,*,backend):
        source=spy.open(filename,framework,device,backend=backend)
        if change=='mtime':
            os.utime(filename,ns=(old_stat.st_atime_ns,old_stat.st_mtime_ns+10_000_000))
            return source
        class MutateOnExit:
            def __enter__(self):return source.__enter__()
            def __exit__(self,*args):
                result=source.__exit__(*args)
                data=bytearray(fixture_path.read_bytes());length=struct.unpack('<Q',data[:8])[0]
                header=bytes(data[8:8+length]);assert b'quant' in header
                data[8:8+length]=header.replace(b'quant',b'quANT',1)
                fixture_path.write_bytes(data)
                os.utime(filename,ns=(old_stat.st_atime_ns,old_stat.st_mtime_ns))
                return result
        return MutateOnExit()
    audit={}
    with pytest.raises(candidate.ReaderError,match='changed during CPU read') as retained:
        candidate.load_torch_file(fixture_path,_safe_open=changed,diagnostics=audit)
    assert retained.value is not None
    gc.collect()
    assert spy.closed==1 and all(ref() is None for ref in spy.refs)
    assert audit['cleared_partial_tensors']>0


def test_individual_large_source_banks_release_with_tiny_anchor(tmp_path):
    # Use a real 64-MiB file so the Python source buffers are separate VirtualAlloc
    # regions, rather than relying on suballocator recycling of tiny objects.
    memory=candidate.psutil.virtual_memory()
    reserve=max(candidate.MIN_RESERVE,min(candidate.MAX_RESERVE,int(memory.total*.125)))
    if memory.available<reserve+256*1024**2:pytest.skip('REAL_RESOURCE_BLOCKED before even creating the 64-MiB fixture')
    path=tmp_path/'ownership.safetensors'
    payload=torch.arange(256,dtype=torch.uint8).repeat(131072)
    save_file({'bank0.weight':payload,'bank1.weight':payload.clone(),'weight_codebook':torch.arange(16,dtype=torch.float32)},path)
    del payload;gc.collect();_require_real_headroom(path)
    state=candidate.load_torch_file(path)
    a=state['bank0.weight'];b=state['bank1.weight'];anchor=state['weight_codebook']
    ptrs=[a.data_ptr(),b.data_ptr()];refs=[weakref.ref(a),weakref.ref(b)]
    hashes=[hashlib.sha256(raw(a)).hexdigest(),hashlib.sha256(raw(b)).hexdigest()]
    copies=[a.clone(),b.clone()]
    del a,b,state;gc.collect()
    assert all(ref() is None for ref in refs)
    assert anchor.untyped_storage().nbytes()==64
    assert hashes==[hashlib.sha256(raw(value)).hexdigest() for value in copies]
    if __import__('os').name=='nt':
        class MBI(ctypes.Structure):
            _fields_=[('BaseAddress',ctypes.c_void_p),('AllocationBase',ctypes.c_void_p),('AllocationProtect',wintypes.DWORD),('RegionSize',ctypes.c_size_t),('State',wintypes.DWORD),('Protect',wintypes.DWORD),('Type',wintypes.DWORD)]
        virtual_query=ctypes.WinDLL('kernel32').VirtualQuery
        virtual_query.argtypes=[ctypes.c_void_p,ctypes.POINTER(MBI),ctypes.c_size_t];virtual_query.restype=ctypes.c_size_t
        for ptr in ptrs:
            result=MBI();assert virtual_query(ptr,ctypes.byref(result),ctypes.sizeof(result))
            assert result.State==0x10000 and result.AllocationBase is None  # MEM_FREE


@pytest.mark.parametrize('route',['pread','host_fallback'])
@pytest.mark.parametrize('return_metadata',[False,True])
def test_complex64_official_equivalence_both_reader_routes(tmp_path,route,return_metadata):
    path=tmp_path/'complex.sft'
    # Include exact signed zeros and a non-finite real component: compare raw
    # complex64 bits rather than floating equality that treats NaNs specially.
    original=torch.tensor([1+2j,-3-4j,complex(-0.,-0.),complex(float('nan'),.5)],dtype=torch.complex64)
    save_file({'z':original},path,metadata={'fixture':'C64','unicode':'复数'})
    _require_real_headroom(path)
    real=candidate._host_context()
    official=real.load_host(str(path),return_metadata=return_metadata)
    def legacy(filename,framework,device):raise AssertionError('legacy opener must not be called')
    audit={}
    actual=candidate.load_torch_file(path,return_metadata=return_metadata,diagnostics=audit,
              _safe_open=legacy if route=='host_fallback' else None)
    if return_metadata:
        actual,am=actual;official,om=official
        assert am==om=={'fixture':'C64','unicode':'复数'}
    assert actual['z'].dtype==official['z'].dtype==torch.complex64
    assert actual['z'].shape==official['z'].shape==original.shape
    assert raw(actual['z'])==raw(official['z'])==raw(original)
    assert actual['z'].untyped_storage()._comfy_source_path==str(path.resolve())
    assert audit['actual_reader']==('host' if route=='host_fallback' else 'official_cpu_pread')


@pytest.mark.parametrize('route',['legacy_api','aimdo','rejects_pread'])
@pytest.mark.parametrize('change',['mtime','fixed_header_restored_mtime'])
def test_fallback_common_source_guard_clears_return(tmp_path,route,change):
    import os
    path=tmp_path/'source-change.safetensors'
    save_file({'packed':torch.arange(64,dtype=torch.int8)},path,metadata={'state':'before'})
    _require_real_headroom(path)
    real=candidate._host_context();original_stat=path.stat();plan=candidate.inspect_header(path,candidate.ReaderPolicy())
    refs=[];changed=[False]
    def host_load(*args,**kwargs):
        result=real.load_host(*args,**kwargs)
        refs.extend(weakref.ref(value) for value in result[0].values())
        if change=='mtime':os.utime(path,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns+10_000_000))
        else:
            # Fixed-length in-place write is legal while the official host's
            # tensor mmap lives. Truncating write_bytes is not this probe.
            with path.open('r+b') as stream:
                size,=struct.unpack('<Q',stream.read(8));header=stream.read(size)
                rewritten=header.replace(b'before',b'after!',1)
                assert rewritten!=header and len(rewritten)==len(header)
                stream.seek(8);assert stream.write(rewritten)==len(header);stream.flush()
            os.utime(path,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns))
            assert candidate._identity(path)==plan['identity']
            assert candidate.inspect_header(path,candidate.ReaderPolicy())['header_sha256']!=plan['header_sha256']
        changed[0]=True
        return result
    host=replace(real,aimdo_enabled=route=='aimdo',load_host=host_load)
    def legacy(filename,framework,device):raise AssertionError('legacy opener must not be called')
    def rejected(filename,framework,device,*,backend):raise ValueError('pread unsupported')
    opener=legacy if route=='legacy_api' else rejected if route=='rejects_pread' else None
    audit={}
    with pytest.raises(candidate.ReaderError,match='changed during CPU read') as retained:
        candidate.load_torch_file(path,return_metadata=True,_host=host,_safe_open=opener,
             diagnostics=audit)
    assert retained.value is not None and changed[0]
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert audit['status']=='failed_or_cancelled_host_fallback' and audit['cleared_partial_tensors']==1


@pytest.mark.parametrize('route',['not_preferred','old_api','aimdo','unknown_legal_dtype'])
def test_compatibility_mmap_does_not_require_eager_ram_budget(fixture_path,route,monkeypatch):
    real=candidate._host_context();audit={}
    def no_eager_budget():raise AssertionError('Lazy host mmap has no eager all-tensor allocation')
    def legacy(filename,framework,device):raise AssertionError('Old opener must not be called')
    if route=='unknown_legal_dtype':monkeypatch.delitem(candidate.DTYPE_BITS,'F32')
    actual=candidate.load_torch_file(fixture_path,prefer_pread=route!='not_preferred',
        _safe_open=legacy if route=='old_api' else None,_host=replace(real,aimdo_enabled=route=='aimdo'),
        policy=candidate.ReaderPolicy(memory_snapshot=no_eager_budget),diagnostics=audit)
    assert audit['actual_reader']=='host' and audit['fallback_reason']
    assert actual['fp32'].dtype==torch.float32
    json.dumps(audit)


def test_actual_host_eager_copy_is_guarded_before_unsafe_fallback(fixture_path):
    real=candidate._host_context();called=[];audit={}
    def host(*args,**kwargs):called.append(1);raise AssertionError('Eager host allocation must be refused')
    context=replace(real,disable_mmap=True,load_host=host)
    policy=candidate.ReaderPolicy(memory_snapshot=lambda:SimpleNamespace(total=64*candidate.GiB,available=0))
    with pytest.raises(candidate.ReaderResourceError):
        candidate.load_torch_file(fixture_path,_host=context,policy=policy,diagnostics=audit)
    assert called==[] and audit['memory_checks'][-1]['stage']=='host_eager_copy'
    assert audit['memory_checks'][-1]['remaining_source_bytes']==2*audit['source_cpu_bytes']


def test_reused_reader_diagnostics_cannot_publish_a_stale_fallback(fixture_path):
    _require_real_headroom(fixture_path)
    audit={}
    first=candidate.load_torch_file(fixture_path,prefer_pread=False,diagnostics=audit)
    assert audit['actual_reader']=='host' and audit['fallback_reason']
    del first
    audit['stale_non_json']=object()
    second=candidate.load_torch_file(fixture_path,diagnostics=audit)
    assert audit['actual_reader']=='official_cpu_pread' and not audit.get('fallback_reason')
    assert not audit.get('host_storage_mode') and 'stale_non_json' not in audit
    assert second['int8'].dtype==torch.int8
    json.dumps(audit)


def test_corrupt_eager_input_is_not_disguised_as_compatibility_fallback(tmp_path):
    path=_write(tmp_path/'broken.safetensors',{'x':dict(dtype='U8',shape=[99],data_offsets=[0,2])},b'ab')
    real=candidate._host_context()
    def no_host(*a,**k):raise AssertionError('Corruption must not trigger fallback')
    with pytest.raises(candidate.ReaderError,match='disagrees'):
        candidate.load_torch_file(path,_host=replace(real,load_host=no_host))


def test_actual_aimdo_annotations_remain_owned_by_host(fixture_path,monkeypatch):
    import comfy.utils,comfy.memory_management
    if not hasattr(comfy.utils,'load_safetensors'):pytest.skip('This host has no aimdo source loader')
    import comfy_aimdo.model_mmap
    if comfy_aimdo.model_mmap.lib is None:
        pytest.skip('Actual aimdo native runtime is not initialized in this CPU-only process; do not initialize a GPU driver here')
    monkeypatch.setattr(comfy.memory_management,'aimdo_enabled',True)
    expected,em=comfy.utils.load_torch_file(str(fixture_path),return_metadata=True)
    audit={};actual,am=candidate.load_torch_file(fixture_path,return_metadata=True,diagnostics=audit)
    assert am==em and audit['actual_reader']=='host' and audit['host_storage_mode']=='aimdo'
    for name in actual:
        assert raw(actual[name])==raw(expected[name])
        for attr in ['_comfy_source_path','_comfy_tensor_file_slice','_comfy_tensor_mmap_refs']:
            assert hasattr(actual[name].untyped_storage(),attr)==hasattr(expected[name].untyped_storage(),attr)


def test_aimdo_original_host_error_is_not_hidden_by_pread_restart(fixture_path):
    real=candidate._host_context();error=OSError('original aimdo host failure');audit={}
    def fail(*a,**k):raise error
    with pytest.raises(OSError) as caught:
        candidate.load_torch_file(fixture_path,_host=replace(real,aimdo_enabled=True,load_host=fail),diagnostics=audit)
    assert caught.value is error and audit['actual_reader']=='host'


def _expert_model(experts=2,width=256,intermediate=16,layers=2):
    import comfy.ops
    from drama3_t8.native.model import Experts
    ops=comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    model=torch.nn.Module();model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList()
    for _ in range(layers):
        layer=torch.nn.Module();layer.mlp=torch.nn.Module()
        layer.mlp.experts=Experts(experts,width,intermediate,device='meta',dtype=torch.bfloat16,ops=ops)
        model.model.layers.append(layer)
    model.register_buffer('anchor',torch.empty(16,device='meta'))
    return model


def _expert_file(path,experts=2,width=256,intermediate=16,layers=2):
    from test_quantized_loading import quantized_bank
    state={}
    for layer in range(layers):
        for bank,rows in [('gate_up_proj',2*intermediate),('down_proj',width)]:
            cols=width if bank=='gate_up_proj' else intermediate
            values=quantized_bank('asym_w4a8_int8',experts=experts,rows=rows,width=cols)
            state.update({f'model.layers.{layer}.mlp.experts.{bank}.{name}':value for name,value in values.items()})
    state['anchor']=torch.arange(16,dtype=torch.float32)
    save_file(state,path,metadata={'drama3_config':'{}','drama3_format':'w4a8'})
    return path


def _region(ptr):
    class MBI(ctypes.Structure):
        _fields_=[('BaseAddress',ctypes.c_void_p),('AllocationBase',ctypes.c_void_p),('AllocationProtect',wintypes.DWORD),('RegionSize',ctypes.c_size_t),('State',wintypes.DWORD),('Protect',wintypes.DWORD),('Type',wintypes.DWORD)]
    query=ctypes.WinDLL('kernel32').VirtualQuery
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(MBI),ctypes.c_size_t];query.restype=ctypes.c_size_t
    out=MBI();assert query(ptr,ctypes.byref(out),ctypes.sizeof(out))
    return out


@pytest.mark.parametrize('pread',[False,True])
def test_real_assign_and_hostbank_cpu_replacement_release_source_by_layer(tmp_path,monkeypatch,pread):
    from drama3_t8.native.loader import load_checkpoint_storage
    from drama3_t8.native import offload_accelerated as offload
    memory=candidate.psutil.virtual_memory();reserve=max(candidate.MIN_RESERVE,min(candidate.MAX_RESERVE,int(memory.total*.125)))
    if memory.available<reserve+256*1024**2:pytest.skip('REAL_RESOURCE_BLOCKED before fixture creation')
    path=_expert_file(tmp_path/'assign.safetensors',experts=64,width=512,intermediate=512)
    _require_real_headroom(path)
    audit={};state=candidate.load_torch_file(path,prefer_pread=pread,diagnostics=audit)
    assert audit['actual_reader']==('official_cpu_pread' if pread else 'host')
    source=state['model.layers.0.mlp.experts.gate_up_proj.weight'].data_ptr()
    model=_expert_model(experts=64,width=512,intermediate=512)
    load_checkpoint_storage(model,state)
    assert model.model.layers[0].mlp.experts.gate_up_proj.weight._qdata.data_ptr()==source
    state.clear();del state
    copies=[]
    def cpu_replacement(value):
        result=value.clone();assert raw(result)==raw(value)
        copies.append(result.numel()*result.element_size())
        return result
    # Explicit CPU-only model of a pin destination. The actual HostBank,
    # QuantHostBank and ExpertStore.add path runs; no page locking is claimed.
    monkeypatch.setattr(offload,'pinned',cpu_replacement)
    store=offload.ExpertStore(pin_memory=False)
    try:
        for layer in model.model.layers:
            layer.mlp.experts=store.add(layer.mlp.experts)
            gc.collect()
            if __import__('os').name=='nt':
                region=_region(source)
                assert region.State==0x10000 if pread else region.Type==0x40000
        assert copies and store.pinned_bytes==0
        assert all(not value.is_pinned() for banks in store.banks for bank in banks for value in bank.fields.values())
    finally:store.close()
    assert not store.banks


@pytest.mark.parametrize('failure',[None,'cancel_after_bank','error_next_bank'])
def test_load_drama_body_routes_reader_assigns_replaces_and_rolls_back_owned_state(tmp_path,monkeypatch,failure):
    from drama3_t8.native import loader,acceleration,offload_accelerated as offload
    import comfy.model_management as mm
    path=_expert_file(tmp_path/'body.safetensors',intermediate=256)
    _require_real_headroom(path)
    made=[];stores=[];refs=[]
    def model(*args,**kwargs):
        result=_expert_model(intermediate=256);made.append(result);return result
    monkeypatch.setattr(loader,'Drama3Model',model)
    monkeypatch.setattr(loader,'parse_config',lambda c:SimpleNamespace())
    monkeypatch.setattr(loader.comfy.model_patcher,'CoreModelPatcher',lambda model,**kw:SimpleNamespace(model=model,**kw))
    monkeypatch.setattr(offload,'pinned',lambda value:value.clone())
    error=OSError('exact injected bank error')
    class Store(offload.ExpertStore):
        def add(self,experts):
            if failure=='error_next_bank' and self.banks:raise error
            result=super().add(experts)
            refs.extend(weakref.ref(value) for bank in self.banks[-1] for value in bank.fields.values())
            if failure=='cancel_after_bank':mm.interrupt_current_processing(True)
            return result
    def store(*args,**kwargs):
        result=Store(pin_memory=False);stores.append(result);return result
    monkeypatch.setattr(acceleration,'expert_store',store)
    mm.interrupt_current_processing(False)
    try:
        if failure:
            expected=OSError if failure=='error_next_bank' else mm.InterruptProcessingException
            with pytest.raises(expected) as caught:loader.load_drama3(str(path),experts_in_ram=True,moe_backend='auto')
            if failure=='error_next_bank':assert caught.value is error
            gc.collect()
            assert stores and not stores[0].banks and not list(made[0].parameters())
            assert all(ref() is None for ref in refs)
        else:
            drama=loader.load_drama3(str(path),experts_in_ram=True,moe_backend='auto')
            assert drama.checkpoint_reader['actual_reader']=='official_cpu_pread'
            assert len(drama.experts.banks)==2
            json.dumps(drama.checkpoint_reader)
    finally:
        mm.interrupt_current_processing(False)
        for store in stores:store.close()
