"""Private provenance for bounded pread expert admission; never JSON authority."""
from dataclasses import dataclass
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import threading
import weakref

import psutil
import safetensors
import safetensors._safetensors_rust as _rust
import torch
from . import checkpoint_reader as reader

class OwnershipUnavailable(RuntimeError):
    pass

class OwnershipViolation(RuntimeError):
    """An active incremental load lost its proof; no next layer is allowed."""

_KNOWN_CLONE_FILES = {
    # Reviewed Kitchen 0.2.37 and 0.2.40 clone/physical-field contracts.
    'base': {'badc436f9f0b9c3736551fdd471257fae85fcf88533461494d36b293a5bcf2ee',
             'ef70849450b63efa0b6305eaa65c89443451df637957c570266bb15ea717c8a4'},
    'int8': {'57c592711013e6cb33c8e8c22e0c1b7eb94ea71d192a08c770349e80127cabf9',
             '307a4133908c07b4ffa04d3cb94991140afcef9b66da126809620f6184549614'},
    'w4a8_int8': {'251f1231b48a6748cfa33a39323dbfd77081e27344cf81c69df4c897c5131636',
                 '18b4e17a89196f10986dbd0739a486671da564c831a1f0c3cbc22bece4e2b86b'},
    'ops': {'0167aef2929b92ba3623376f7656e58ead5869485b926b09bead525f32e4dd06',
            'dc2c13bdbc5e6073f0b869ee6260f925025b4c6867fe8c097dd24ce47837e16c'},
}
_ASSIGN_AST_SHA = '7a31d6d43eb86c1e839cda5414819b1db84e5845c38fa7571928a6c36c9bf53c'

_ANNOTATOR = '''def annotate_state_dict(state_dict, path):
    path = os.path.realpath(path)
    for value in state_dict.values():
        untyped_storage = getattr(value, "untyped_storage", None)
        if untyped_storage is not None:
            untyped_storage()._comfy_source_path = path
'''

class _StorageWeak:
    """Weak StorageImpl handle, unlike a weak Python storage/tensor wrapper."""
    __slots__ = ('cdata', '_free', '_expired')

    def __init__(self, storage):
        try:
            required = ('_weak_ref', '_expired', '_free_weak_ref')
            if not all(hasattr(storage, name) for name in required):
                raise OwnershipUnavailable('This Torch storage has no reliable StorageImpl weak API')
            self._free = type(storage)._free_weak_ref
            self._expired = type(storage)._expired
            self.cdata = storage._weak_ref()
        finally:
            storage = None

    def expired(self):
        return self._expired(self.cdata)

    def __del__(self):
        if hasattr(self, 'cdata'):
            self._free(self.cdata)

@dataclass(frozen=True)
class Span:
    storage_id: int
    address: int
    storage_bytes: int
    tensor_bytes: int
    offset_bytes: int
    shape: tuple
    stride: tuple
    dtype: str

def _span(tensor):
    storage = None
    try:
        if (not torch.is_tensor(tensor) or tensor.is_meta or tensor.device.type != 'cpu'
                or tensor.layout != torch.strided or not tensor.is_contiguous()):
            raise OwnershipUnavailable('Only real contiguous strided CPU physical fields are understood')
        storage = tensor.untyped_storage()
        size = tensor.numel() * tensor.element_size()
        offset = tensor.storage_offset() * tensor.element_size()
        if not size or offset or storage.nbytes() != size:
            raise OwnershipUnavailable('A physical field must occupy one entire nonempty source allocation')
        return Span(storage._cdata, storage.data_ptr(), storage.nbytes(), size, offset,
                    tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
    finally:
        # A retained refusal traceback must not become another source owner.
        tensor = storage = None

def _same_allocation(left, right):
    return (left.storage_id, left.address, left.storage_bytes, left.tensor_bytes, left.offset_bytes) == (
        right.storage_id, right.address, right.storage_bytes, right.tensor_bytes, right.offset_bytes)

def _no_overlap(spans):
    seen, ranges = set(), []
    for name, span in spans:
        if span.storage_id in seen:
            raise OwnershipUnavailable(f'Source allocation is aliased: {name}')
        seen.add(span.storage_id)
        ranges.append((span.address, span.address + span.storage_bytes, name))
    last = 0
    for begin, end, name in sorted(ranges):
        if begin < last:
            raise OwnershipUnavailable(f'Source storage ranges overlap: {name}')
        last = end

def _official_contract():
    """No caller opener/context/annotation override can enter this experiment."""
    import comfy.storage
    host = reader._host_context()
    if host.aimdo_enabled:
        raise OwnershipUnavailable('aimdo retains the unchanged legacy policy')
    opener = safetensors.safe_open
    if (opener is not _rust.safe_open or safetensors.__version__ != '0.8.0'
            or 'backend' not in inspect.signature(opener).parameters):
        raise OwnershipUnavailable('No verifiable official safe_open backend contract')
    annotation = host.annotate_state_dict
    try:
        actual = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(annotation))), include_attributes=False)
    except (TypeError, OSError, SyntaxError) as exc:
        raise OwnershipUnavailable('Cannot identify the host annotation contract') from exc
    expected = ast.dump(ast.parse(_ANNOTATOR), include_attributes=False)
    if (annotation is not comfy.storage.annotate_state_dict or annotation.__module__ != 'comfy.storage'
            or annotation.__closure__ or actual != expected
            or Path(annotation.__code__.co_filename).resolve() != Path(comfy.storage.__file__).resolve()):
        raise OwnershipUnavailable('Unknown host annotation retains the unchanged legacy policy')
    return host, opener

def _control_config(value):
    import json
    try:
        if value.numel() > 65536:
            raise OwnershipUnavailable('Unknown oversized quant control record')
        return json.loads(bytes(value.tolist()))
    finally:
        value = None

def _allocation_record(value):
    try:
        return _span(value), _StorageWeak(value.untyped_storage())
    finally:
        value = None

def _inventory(model, manifest, configs, first=0):
    """Recognize only current INT8/W4/W6 physical expert layouts, fail closed."""
    # Explicit Tensor temporary cleanup.
    fields={}
    rows={}
    weight=value=module=layer=physical_iterator=None
    try:
        from .offload import tensor_fields, per_row_fields
        from comfy.quant_ops import QuantizedTensor
        layers, all_spans = [], []
        expert_names = set()
        for index, layer in enumerate(model.model.layers):
            if index < first:
                continue
            banks = []
            for bank_name in ('gate_up_proj', 'down_proj'):
                prefix = f'model.layers.{index}.mlp.experts.{bank_name}'
                module = getattr(layer.mlp.experts, bank_name)
                weight, config = module.weight, configs.get(prefix, {})
                fmt = config.get('format')
                expected_layout = 'TensorWiseINT8Layout' if fmt == 'int8_tensorwise' else 'AsymW4A8Int8Layout'
                if (fmt not in ('int8_tensorwise', 'asym_w4a8_int8', 'w6a8_int8')
                        or not isinstance(weight, QuantizedTensor) or weight._layout_cls != expected_layout
                        or module.quant_format != fmt or weight._qdata.ndim != 2):
                    raise OwnershipUnavailable(f'Unrecognized expert layout: {prefix}')
                if type(module.num_experts) is not int or module.num_experts <= 0:
                    raise OwnershipUnavailable(f'Invalid expert count: {prefix}')
                source_weight = manifest.get(prefix + '.weight')
                source_shape = source_weight[0].shape if source_weight is not None else ()
                bits = {'int8_tensorwise':8, 'asym_w4a8_int8':4, 'w6a8_int8':6}[fmt]
                if (len(source_shape) != 3 or source_shape[2] * 8 % bits
                        or config.get('num_experts') != source_shape[0]
                        or module.num_experts != source_shape[0]
                        or module.out_features != source_shape[1]
                        or module.in_features != source_shape[2] * 8 // bits
                        or tuple(weight._params.orig_shape) != (source_shape[0] * source_shape[1], source_shape[2] * 8 // bits)
                        or weight._params.orig_dtype != torch.bfloat16
                        or getattr(weight._params, 'transposed', False)
                        or getattr(weight._params, 'mma_packed', False)):
                    raise OwnershipUnavailable(f'Unrecognized expert logical/physical layout relation: {prefix}')
                fields = {'_qdata': weight._qdata, **tensor_fields(weight)}
                expected = {'_qdata', 'scale'} if fmt == 'int8_tensorwise' else {'_qdata', 'scale', 's_channel'}
                if fmt == 'asym_w4a8_int8' and prefix + '.weight_codebook' in manifest:
                    expected.add('codebook')
                if set(fields) != expected:
                    raise OwnershipUnavailable(f'Unrecognized extra physical fields: {prefix}')
                if fmt == 'int8_tensorwise':
                    if not weight._params.convrot or weight._params.convrot_groupsize != 256:
                        raise OwnershipUnavailable('Unrecognized INT8 activation/workspace layout')
                elif weight._params.group_size != 16 or weight._params.convrot_groupsize != 256:
                    raise OwnershipUnavailable('Unrecognized grouped activation/workspace layout')
                rows = per_row_fields(weight)
                expected_rows = expected - {'codebook'}
                if set(rows) != expected_rows:
                    raise OwnershipUnavailable('Unrecognized row/shared field partition')
                field_records, row_bytes = [], 0
                for field_name, value in fields.items():
                    suffix = {'_qdata':'weight', 'scale':'weight_scale' if fmt == 'int8_tensorwise' else 'weight_s_rel',
                              's_channel':'weight_s_channel', 'codebook':'weight_codebook'}[field_name]
                    key = prefix + '.' + suffix
                    original = manifest.get(key)
                    span = _span(value)
                    if original is None or not _same_allocation(span, original[0]) or original[1].expired():
                        raise OwnershipUnavailable(f'Assigned physical field is not the original independent allocation: {key}')
                    source_shape = original[2] if len(original) > 2 else original[0].shape
                    shape = source_shape if field_name == 'codebook' else (source_shape[0] * source_shape[1], *source_shape[2:])
                    source_dtype = original[3] if len(original) > 3 else original[0].dtype
                    dtype = 'torch.float8_e4m3fn' if field_name == 'scale' and fmt != 'int8_tensorwise' else source_dtype
                    if span.shape != shape or span.dtype != dtype:
                        raise OwnershipUnavailable(f'Unrecognized physical flatten/dtype view: {key}')
                    if field_name in rows:
                        if span.shape[0] % module.num_experts:
                            raise OwnershipUnavailable('Physical rows are not divisible by the expert count')
                        row_bytes += span.tensor_bytes
                    field_records.append((field_name, key, span, field_name in rows))
                    all_spans.append((key, span))
                    expert_names.add(prefix + '.weight.' + field_name)
                banks.append(dict(name=bank_name, fields=tuple(field_records), physical_bytes=row_bytes))
            layers.append(tuple(banks))
        # Catch a hidden dense/model alias into expert storage, not just bank aliases.
        def physical_occurrences():
            # named_parameters' default de-duplicates shared Parameter objects. A
            # late dense alias must not disappear from this ownership audit.
            # Explicit Tensor temporary cleanup.
            values=[]
            value=field=None
            try:
                try:
                    values = [*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)]
                except TypeError as exc:
                    raise OwnershipUnavailable('Cannot enumerate duplicate physical model references') from exc
                for name, value in values:
                    if hasattr(value, '_qdata'):
                        yield name + '._qdata', value._qdata
                        for field_name, field in tensor_fields(value).items():
                            yield name + '.' + field_name, field
                    else:
                        yield name, value
            finally:
                values.clear()
                value=field=None
        physical_iterator=physical_occurrences()
        for name, value in physical_iterator:
            if '.mlp.experts.' in name:
                if name not in expert_names:
                    raise OwnershipUnavailable(f'Unrecognized model expert physical field: {name}')
                continue
            span = _span(value)
            original = manifest.get(name)
            if original is None or not _same_allocation(span, original[0]):
                raise OwnershipUnavailable(f'Non-expert assigned storage changed: {name}')
            all_spans.append((name, span))
        _no_overlap(all_spans)
        return tuple(layers)
    finally:
        if physical_iterator is not None:physical_iterator.close()
        fields.clear()
        rows.clear()
        weight=value=module=layer=physical_iterator=None

def _clone_contract():
    import comfy.ops
    import comfy_kitchen.tensor.base as base
    import comfy_kitchen.tensor.int8 as int8
    import comfy_kitchen.tensor.w4a8_int8 as w4a8_int8
    from . import loader
    if torch.__version__ != '2.8.0+cu128':
        raise OwnershipUnavailable('Unreviewed Torch Parameter/detach implementation')
    function=loader.load_checkpoint_storage
    try:
        actual=ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))).body[0],include_attributes=False)
    except (TypeError,OSError,SyntaxError) as exc:
        raise OwnershipUnavailable('Cannot identify actual assign function') from exc
    if (function.__module__ != loader.__name__ or function.__closure__
            or Path(function.__code__.co_filename).resolve()!=Path(loader.__file__).resolve()
            or hashlib.sha256(actual.encode()).hexdigest()!=_ASSIGN_AST_SHA):
        raise OwnershipUnavailable('Unreviewed actual assign function contract')
    modules = dict(base=base, int8=int8, w4a8_int8=w4a8_int8, ops=comfy.ops)
    origins = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in _KNOWN_CLONE_FILES[name]:
            raise OwnershipUnavailable(f'Unreviewed official assign/Kitchen implementation: {name}')
        origins[name] = dict(path=str(path), sha256=digest)
    origins['assign_function']=dict(path=str(Path(loader.__file__).resolve()),ast_sha256=_ASSIGN_AST_SHA)
    return origins

def _derive_recognized_params(model, state, manifest, configs):
    """Only the controlled assign boundary may establish known scale clones.

    Original qdata is compulsory. The original tensors are still held by the
    private state here so each recognized copied parameter can be compared
    byte-for-byte before state.clear. Old copied allocations must then actually
    expire before the capability is issued. This cannot adopt a later clone.
    """
    # Explicit Tensor temporary cleanup.
    fields={}
    value=original=module=layer=None
    try:
        from .offload import tensor_fields
        origins = _clone_contract()
        derived, result = [], dict(manifest)
        for index, layer in enumerate(model.model.layers):
            for bank in ('gate_up_proj', 'down_proj'):
                prefix = f'model.layers.{index}.mlp.experts.{bank}'
                module = getattr(layer.mlp.experts, bank)
                config = configs.get(prefix, {})
                fmt = config.get('format')
                if fmt not in ('int8_tensorwise', 'asym_w4a8_int8', 'w6a8_int8'):
                    raise OwnershipUnavailable('Unknown format cannot authorize derived parameter storage')
                # The full known-layout/row/shape/alias validation is performed by
                # _inventory immediately after this narrowly controlled derivation.
                fields = {'_qdata':module.weight._qdata, **tensor_fields(module.weight)}
                for field_name, value in fields.items():
                    suffix = {'_qdata':'weight', 'scale':'weight_scale' if fmt == 'int8_tensorwise' else 'weight_s_rel',
                              's_channel':'weight_s_channel', 'codebook':'weight_codebook'}.get(field_name)
                    if suffix is None:
                        raise OwnershipUnavailable('Unexpected derived tensor parameter')
                    key = prefix + '.' + suffix
                    source = manifest.get(key)
                    assigned = _span(value)
                    if source is None or source[1].expired():
                        raise OwnershipUnavailable('Derived parameter has no live official pread source')
                    if _same_allocation(assigned, source[0]):
                        continue
                    if field_name == '_qdata':
                        raise OwnershipUnavailable('A cloned qdata cannot acquire independent-pread provenance')
                    original = None
                    try:
                        original = state[key]
                        # Exact physical bytes; dtype-view changes are validated by
                        # _inventory. Do not retain this original in an exception.
                        equal = assigned.tensor_bytes == source[0].tensor_bytes and torch.equal(
                            original.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
                    finally:
                        original = None
                    if not equal:
                        raise OwnershipUnavailable('Derived quantization parameter differs from the official source bytes')
                    result[key] = (assigned, _StorageWeak(value.untyped_storage()), source[0].shape, source[0].dtype)
                    derived.append(dict(key=key, field=field_name, bytes=assigned.tensor_bytes,
                                        original_span=source[0], assigned_span=assigned,
                                        original_weak=source[1], bytes_equal=True))
        return result, derived, origins
    finally:
        fields.clear()
        value=original=module=layer=None


_ISSUER=object()
_READS=weakref.WeakKeyDictionary()
_ASSIGNMENTS=weakref.WeakKeyDictionary()
_PROOFS=weakref.WeakKeyDictionary()


class _Token:
    __slots__=('__weakref__',)
    def __init__(self,issuer):
        if issuer is not _ISSUER:raise OwnershipUnavailable('No internal ownership issuer')


class _OwnedState(dict):
    __slots__=('_source_token','__weakref__')
    def __init__(self,state,issuer):
        if issuer is not _ISSUER:raise OwnershipUnavailable('No internal owned-state issuer')
        super().__init__(state)


def _diagnostic(target,**values):
    if target is not None:target.setdefault('expert_ownership',{}).update(values)


def _seal_official_read(state,path,plan,host,opener):
    """Reader-only mint point, after real pread and its final source guards.

    The caller must be the native reader's non-overridden official path. This
    private issuer is not a sandbox against malicious same-interpreter Python.
    Registry values deliberately never strongly own the returned state.
    """
    try:
        current,official=_official_contract()
        if opener is not official or host.annotate_state_dict is not current.annotate_state_dict or host.aimdo_enabled:
            raise OwnershipUnavailable('The actual reader context is not the reviewed official pread path')
        manifest={name:_allocation_record(state[name]) for name in state}
        _no_overlap((name,pair[0]) for name,pair in manifest.items())
        configs={name[:-len('.comfy_quant')]:_control_config(state[name])
                 for name in state if name.endswith('.comfy_quant')}
        if not configs:raise OwnershipUnavailable('No recognized quantization controls for incremental admission')
    except (OwnershipUnavailable,ValueError,TypeError,KeyError,AttributeError) as exc:
        return state,dict(read_certificate_issued=False,fallback_reason=str(exc))
    owned=_OwnedState(state,_ISSUER)
    token=_Token(_ISSUER)
    owned._source_token=token
    _READS[token]=dict(state=weakref.ref(owned),manifest=manifest,configs=configs,path=str(path),
                       plan=plan,thread=threading.get_ident(),safetensors_version=safetensors.__version__,
                       opener_module=str(_rust.__file__))
    return owned,dict(read_certificate_issued=True,source_tensor_count=len(manifest),
                      safetensors_version=safetensors.__version__)


def _consume_read(state):
    if type(state) is not _OwnedState:return None
    token=getattr(state,'_source_token',None)
    if type(token) is not _Token:return None
    record=_READS.pop(token,None)
    if record is None or record['state']() is not state or record['thread']!=threading.get_ident():return None
    return record


def _preassign(record,model,state):
    """Prove no hidden full-bank cast before using a scale-only copy budget."""
    origins=_clone_contract()
    manifest,configs=record['manifest'],record['configs']
    cloned=cast=0
    by_bank={}
    eligible=True
    reasons=[]
    value=None
    try:
        for prefix,config in configs.items():
            if not isinstance(config,dict) or '.mlp.experts.' not in prefix:
                raise OwnershipUnavailable('Unknown dense quantization/layout keeps legacy admission')
            fmt=config.get('format')
            if fmt not in ('int8_tensorwise','asym_w4a8_int8','w6a8_int8'):
                raise OwnershipUnavailable('Unknown expert format keeps legacy admission')
            params=config.get('params',{})
            if not isinstance(params,dict):params={}
            if (int(config.get('convrot_groupsize',params.get('convrot_groupsize',256)))!=256
                    or (fmt=='int8_tensorwise' and not config.get('convrot',False))
                    or (fmt!='int8_tensorwise' and int(config.get('group_size',params.get('group_size',16)))!=16)):
                eligible=False;reasons.append('Expert layout is outside the reviewed incremental contract')
            expected=['weight_scale'] if fmt=='int8_tensorwise' else ['weight_s_rel','weight_s_channel']
            if prefix+'.weight_codebook' in manifest:expected.append('weight_codebook')
            size=0
            for suffix in expected:
                key=prefix+'.'+suffix
                if key not in manifest:raise OwnershipUnavailable('Missing known scale/codebook source')
                size+=manifest[key][0].tensor_bytes
                known_dtypes=({'torch.uint8','torch.float8_e4m3fn'} if suffix=='weight_s_rel' else {'torch.float32'})
                if manifest[key][0].dtype not in known_dtypes:
                    eligible=False;reasons.append('Scale/codebook dtype is outside the reviewed incremental contract')
            if fmt=='w6a8_int8' and prefix+'.weight_codebook' in manifest:
                eligible=False;reasons.append('W6 codebook layout is outside the reviewed incremental contract')
            by_bank[prefix]=size
            cloned+=size
            key=prefix+'.weight'
            if key not in manifest:raise OwnershipUnavailable('Missing original expert qdata source')
            value=state[key]
            if value.dtype!=torch.int8:
                cast+=value.numel()  # target int8 is exactly one physical byte
                eligible=False;reasons.append('Expert qdata requires a dtype cast; legacy admission only')
        # Validate every original source still has its full independent span.
        # Direct Parameter/buffer expected dtypes cover e.g. F32 norm buffers;
        # no global BF16 assumption is made for those schema-defined objects.
        expected_parameters=dict(model.named_parameters(remove_duplicate=False))
        expected_buffers=dict(model.named_buffers(remove_duplicate=False))
        for name,pair in manifest.items():
            value=state.get(name)
            if value is None or _span(value)!=pair[0] or pair[1].expired():
                raise OwnershipUnavailable('Original read storage changed before assign')
            if '.mlp.experts.' in name or name.endswith('.comfy_quant'):continue
            if name in expected_buffers:
                if value.dtype!=expected_buffers[name].dtype:
                    eligible=False;reasons.append('A buffer dtype is outside the reviewed no-copy schema')
                continue
            if name in expected_parameters:
                if value.dtype!=expected_parameters[name].dtype:
                    eligible=False;reasons.append('A parameter dtype is outside the reviewed no-copy schema')
                    cast+=value.numel()*expected_parameters[name].element_size()
                continue
            if name.endswith('.weight'):
                # MixedPrecisionOps.Linear has weight=None until actual load;
                # its constructor still declares the precise compute dtype.
                owner_name=name[:-len('.weight')]
                module=model.get_submodule(owner_name)
                factory=getattr(module,'factory_kwargs',None)
                target=factory.get('dtype') if isinstance(factory,dict) else None
                if target is None:raise OwnershipUnavailable('Unknown dense weight assignment contract')
                if value.dtype!=target:
                    eligible=False;reasons.append('Dense weight requires a dtype cast; legacy admission only')
                    cast+=value.numel()*torch.empty((),device='meta',dtype=target).element_size()
            else:
                raise OwnershipUnavailable('Unknown non-expert field assignment contract')
        if not by_bank:raise OwnershipUnavailable('No known expert banks')
        return dict(clone_workspace_bytes=cloned,cast_workspace_bytes=cast,
                    extra_bank_parameter_bytes=max(by_bank.values()),incremental_eligible=eligible,
                    fallback_reason='; '.join(dict.fromkeys(reasons)) or None),origins
    finally:
        value=None


def _assign_memory(plan,diagnostics=None):
    memory=psutil.virtual_memory()
    total,available=int(memory.total),int(memory.available)
    if total<=0 or available<0:raise reader.ReaderResourceError('Cannot verify real RAM before assign')
    reserve=max(reader.MIN_RESERVE,min(reader.MAX_RESERVE,int(total*.125)))
    required=plan['clone_workspace_bytes']+plan['cast_workspace_bytes']+plan['extra_bank_parameter_bytes']+reserve
    result=dict(plan,total_bytes=total,available_bytes=available,reserve_bytes=reserve,required_bytes=required,
                status='passed' if available>=required else 'refused')
    _diagnostic(diagnostics,assign_memory=result)
    if available<required:
        reason=(f'before assign: available RAM {available} < required {required} '
                f'(known parameter clones + casts + one-bank parameter margin + unchanged reserve {reserve})')
        _diagnostic(diagnostics,policy='owned_assign_resource_refused',fallback_reason=reason)
        raise reader.ReaderResourceError(reason)
    return result


def assign_owned(model,state,*,diagnostics=None):
    """Call the real assign exactly once; stage optional proof before clear."""
    from .loader import load_checkpoint_storage
    record=_consume_read(state)
    staged=None
    if record is not None:
        try:
            plan,origins=_preassign(record,model,state)
        except (OwnershipUnavailable,KeyError,AttributeError,ValueError,TypeError) as exc:
            _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason=str(exc))
            record=None
        else:
            # Guard executes before any actual model assignment/parameter clone.
            # Known U8/F32 cast compatibility includes the complete target copy
            # bytes and never obtains an incremental proof.
            memory=_assign_memory(plan,diagnostics)
            if not plan['incremental_eligible']:
                _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason=plan['fallback_reason'])
                record=None
    load_checkpoint_storage(model,state)
    if record is None:return None
    try:
        manifest,derived,origins=_derive_recognized_params(model,state,record['manifest'],record['configs'])
    except OwnershipUnavailable as exc:
        _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason=str(exc))
        return None
    record.update(manifest=manifest,derived_params=derived,assign_implementation=origins,model=weakref.ref(model))
    staged=_Token(_ISSUER)
    _ASSIGNMENTS[staged]=record
    return staged


def _inventory_without_traceback(model,manifest,configs,first=0):
    # Expected compatibility refusals are descriptions, not retained private
    # traceback frames containing every model Tensor. Real read/cancel failures
    # are not converted here.
    try:return _inventory(model,manifest,configs,first=first),None
    except (OwnershipUnavailable,AttributeError,KeyError,IndexError) as exc:
        reason=str(exc)
        exc.__traceback__=None
    return None,reason


def finish_assignment(witness,model,*,diagnostics=None):
    if type(witness) is not _Token:return None
    record=_ASSIGNMENTS.pop(witness,None)
    if record is None or record['model']() is not model:return None
    source=record['state']()
    if source is not None and len(source):
        _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason='Source state was not cleared after assign')
        return None
    source=None
    reader._cancel(reader.ReaderPolicy())
    reader._verify_source(record['path'],record['plan'],reader.ReaderPolicy(),'after owned assign/state.clear')
    for field in record['derived_params']:
        if not field.pop('original_weak').expired():
            _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason='Original copied parameter StorageImpl survived source clear')
            return None
    layers,reason=_inventory_without_traceback(model,record['manifest'],record['configs'])
    if reason or not layers:
        _diagnostic(diagnostics,policy='legacy_full_expert',fallback_reason=reason or 'No recognized expert layers')
        return None
    record.update(layers=layers,completed=0,active=None,invalid=None,store=None)
    proof=_Token(_ISSUER)
    _PROOFS[proof]=record
    _diagnostic(diagnostics,proof_issued=True,fallback_reason=None,policy='verified_pread_pending_selection',**describe(proof,model))
    return proof


def _record(proof,model):
    if type(proof) is not _Token or proof not in _PROOFS:raise OwnershipUnavailable('No typed internal ownership proof')
    record=_PROOFS[proof]
    if record['model']() is not model or record['thread']!=threading.get_ident():
        raise OwnershipUnavailable('Ownership belongs to another model/thread')
    if record['invalid']:raise OwnershipUnavailable(record['invalid'])
    return record


def _admission(record,model,index):
    reader._cancel(reader.ReaderPolicy())
    reader._verify_source(record['path'],record['plan'],reader.ReaderPolicy(),'before expert admission')
    remaining,reason=_inventory_without_traceback(model,record['manifest'],record['configs'],first=index)
    if reason or not remaining or remaining!=record['layers'][index:]:
        raise OwnershipUnavailable(reason or 'Remaining source layout changed')
    largest=max(bank['physical_bytes'] for layer in remaining for bank in layer)
    layer_bytes=sum(bank['physical_bytes'] for bank in remaining[0])
    if layer_bytes>2*largest:raise OwnershipUnavailable('Known layer exceeds two-bank bound')
    memory=psutil.virtual_memory()
    reserve=max(reader.MIN_RESERVE,min(reader.MAX_RESERVE,int(memory.total*.125)))
    required=2*largest+reserve
    return dict(policy='verified_pread_incremental',layer=index,maximum_remaining_bank_bytes=largest,
                layer_destination_bytes=layer_bytes,total_bytes=int(memory.total),available_bytes=int(memory.available),
                reserve_bytes=reserve,required_bytes=required,eligible=int(memory.available)>=required)


def initial_admission(proof,model):
    try:
        record=_record(proof,model)
        return _admission(record,model,0)
    except OwnershipUnavailable as exc:
        if type(proof) is _Token and proof in _PROOFS:
            _PROOFS[proof]['selection_refusal']=str(exc)
        return None


def selection_refusal(proof):
    if type(proof) is _Token and proof in _PROOFS:return _PROOFS[proof].get('selection_refusal')
    return None


def bind_store(proof,model,store):
    from .offload_accelerated import ExpertStore
    record=_record(proof,model)
    if type(store) is not ExpertStore or not store.pin_memory:
        raise OwnershipUnavailable('Only the actual supported pinned ExpertStore may use incremental admission')
    record['store']=weakref.ref(store)


def is_incremental(proof,model,store):
    try:
        record=_record(proof,model)
        return record['store'] is not None and record['store']() is store
    except OwnershipUnavailable:return False


def _invalid(record,reason):
    record['invalid']=reason
    raise OwnershipViolation(reason)


def begin_layer(proof,model,index):
    try:record=_record(proof,model)
    except OwnershipUnavailable as exc:raise OwnershipViolation(str(exc)) from None
    if (type(index) is not int or not 0<=index<len(record['layers']) or index!=record['completed']
            or record['active'] is not None or record['store'] is None or record['store']() is None):
        _invalid(record,'Incremental layer order/store is invalid')
    try:receipt=_admission(record,model,index)
    except OwnershipUnavailable as exc:
        reason=str(exc)
    else:
        if not receipt['eligible']:_invalid(record,f'Incremental layer {index} lacks real RAM: {receipt}')
        record['active']=index
        return receipt
    _invalid(record,reason)


def complete_layer(proof,model,index):
    record=_record(proof,model)
    store=record['store']() if record['store'] is not None else None
    if record['active']!=index or index!=record['completed'] or store is None:
        _invalid(record,'No matching complete layer replacement')
    from .offload_accelerated import OffloadedExperts
    replacement=model.model.layers[index].mlp.experts
    if type(replacement) is not OffloadedExperts or replacement.store is not store or replacement.layer!=index:
        _invalid(record,'The complete source expert module was not replaced')
    for bank in record['layers'][index]:
        for _,name,_,row in bank['fields']:
            if row and not record['manifest'][name][1].expired():
                _invalid(record,f'Source StorageImpl still live after layer {index}: {name}')
    record['completed']+=1;record['active']=None
    return dict(layer=index,source_row_storage_impl_expired=True,completed_layers=record['completed'],
                expected_layers=len(record['layers']),os_available_ram_increase_proven=False)


def describe(proof,model):
    record=_record(proof,model)
    return dict(source_tensor_count=len(record['manifest']),derived_parameter_count=len(record['derived_params']),
                derived_parameter_bytes=sum(field['bytes'] for field in record['derived_params']),
                completed_layers=record['completed'],expected_layers=len(record['layers']),
                assign_implementation=record['assign_implementation'],tensor_strong_references_in_proof=0,
                source_header_sha256=record['plan']['header_sha256'])


def close(token):
    if type(token) is _Token:
        _READS.pop(token,None);_ASSIGNMENTS.pop(token,None);_PROOFS.pop(token,None)
