"""DRAMA-local CPU checkpoint loading; no Comfy/Safetensors global changes.

RAM-offloaded bodies prefer official per-tensor pread storage when its eager
source allocation fits the existing RAM reserve. Lazy host mmap remains the
compatibility path for small-memory hosts, aimdo and older Safetensors APIs.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct

import psutil
import safetensors
import torch

GiB = 1024 ** 3
MIN_RESERVE, MAX_RESERVE = 4 * GiB, 8 * GiB
MAX_FINGERPRINT_HEADER_BYTES = 100_000_000  # Official Safetensors header limit.
DTYPE_BITS = {
    'BOOL': 8, 'U8': 8, 'I8': 8, 'U16': 16, 'I16': 16,
    'U32': 32, 'I32': 32, 'U64': 64, 'I64': 64,
    'F16': 16, 'BF16': 16, 'F32': 32, 'F64': 64, 'C64': 64,
    'F8_E4M3': 8, 'F8_E4M3FN': 8, 'F8_E5M2': 8,
    'F8_E4M3FNUZ': 8, 'F8_E5M2FNUZ': 8, 'F8_E8M0': 8,
    'F4': 4,
}
TORCH_DTYPE_NAMES = {
    'BOOL':'bool','U8':'uint8','I8':'int8','U16':'uint16','I16':'int16',
    'U32':'uint32','I32':'int32','U64':'uint64','I64':'int64',
    'F16':'float16','BF16':'bfloat16','F32':'float32','F64':'float64','C64':'complex64',
    'F8_E4M3':'float8_e4m3fn','F8_E4M3FN':'float8_e4m3fn',
    'F8_E5M2':'float8_e5m2','F8_E4M3FNUZ':'float8_e4m3fnuz',
    'F8_E5M2FNUZ':'float8_e5m2fnuz','F8_E8M0':'float8_e8m0fnu',
}


class ReaderError(RuntimeError):
    pass


class UnsupportedReader(ReaderError):
    pass


class ReaderCancelled(ReaderError):
    pass


class ReaderResourceError(ReaderError):
    pass


@dataclass(frozen=True)
class HostContext:
    aimdo_enabled: bool
    annotate_state_dict: object
    load_host: object
    host_utils_path: str
    host_storage_path: str
    disable_mmap: bool = False


@dataclass(frozen=True)
class ReaderPolicy:
    max_header_bytes: int = 8 * 1024 ** 2
    max_tensors: int = 100_000
    cancelled: object = None  # Zero-argument bool; evaluated at tensor boundaries.
    memory_snapshot: object = None


def _host_context():
    import comfy.memory_management
    import comfy.storage
    import comfy.utils
    return HostContext(bool(comfy.memory_management.aimdo_enabled),
                       comfy.storage.annotate_state_dict, comfy.utils.load_torch_file,
                       comfy.utils.__file__, comfy.storage.__file__,
                       bool(getattr(comfy.utils, 'DISABLE_MMAP', False)))


def _identity(path):
    value = os.stat(path)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ReaderError(f'Duplicate header key: {key}')
        result[key] = value
    return result


def inspect_header(path, policy):
    """Bounded metadata read before any get_tensor/eager source allocation."""
    identity = _identity(path)
    with open(path, 'rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReaderError('Checkpoint is shorter than the safetensors header length')
        size, = struct.unpack('<Q', prefix)
        if size < 2:
            raise ReaderError(f'Invalid safetensors header length: {size}')
        if size > identity[2] - 8:
            raise ReaderError('Truncated safetensors header')
        if size > policy.max_header_bytes:
            raise UnsupportedReader(f'Header length {size} exceeds eager-reader JSON bound {policy.max_header_bytes}')
        raw = stream.read(size)
    header = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(header, dict):
        raise ReaderError('Safetensors header must be an object')
    metadata = header.get('__metadata__')
    if metadata is not None and (not isinstance(metadata, dict) or
                                not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
        raise ReaderError('Safetensors metadata must map strings to strings')
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if len(tensors) > policy.max_tensors:
        raise UnsupportedReader('Tensor count exceeds eager-reader policy')
    data_bytes = identity[2] - 8 - size
    ranges, groups, total, largest = [], {}, 0, 0
    expert_groups = {}
    for name, value in tensors.items():
        if not isinstance(value, dict):
            raise ReaderError(f'Invalid tensor descriptor: {name}')
        dtype, shape, offsets = value.get('dtype'), value.get('shape'), value.get('data_offsets')
        if dtype not in DTYPE_BITS:
            raise UnsupportedReader(f'Unknown storage dtype {dtype!r}: {name}; no guessed byte estimate')
        native_dtype = TORCH_DTYPE_NAMES.get(dtype)
        if native_dtype is None or not hasattr(torch, native_dtype):
            raise UnsupportedReader(f'Native CPU dtype contract is unavailable for {dtype}: {name}')
        if (not isinstance(shape, list) or len(shape) > 64 or
                not all(type(n) is int and n >= 0 for n in shape)):
            raise ReaderError(f'Invalid shape: {name}')
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                not all(type(n) is int for n in offsets) or
                not 0 <= offsets[0] <= offsets[1] <= data_bytes):
            raise ReaderError(f'Invalid data offsets: {name}')
        physical = offsets[1] - offsets[0]
        expected = (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8
        if physical != expected:
            raise ReaderError(f'Physical tensor byte count disagrees with shape/dtype: {name}')
        ranges.append((offsets[0], offsets[1], name))
        total += physical
        largest = max(largest, physical)
        field = name.rsplit('.', 1)[-1]
        if field not in ('comfy_quant', 'weight_codebook'):
            group = name.rsplit('.', 1)[0] if '.' in name else name
            groups[group] = groups.get(group, 0) + physical
            if '.experts.' in name:
                expert_groups[group] = expert_groups.get(group, 0) + physical
    end = 0
    for begin, finish, name in sorted(ranges):
        if begin != end:
            raise ReaderError(f'Overlapping or non-contiguous tensor data: {name}')
        end = finish
    if end != data_bytes:
        raise ReaderError('Unreferenced or truncated checkpoint data')
    max_bank = max((expert_groups or groups).values(), default=0)
    # All eager source bytes plus the existing two-bank margin, and at least
    # one largest-tensor read workspace. This is deliberately conservative.
    transient = max(2 * max_bank, largest)
    return dict(identity=identity, header_bytes=size,
                header_sha256=hashlib.sha256(raw).hexdigest(), tensors=tensors,
                source_cpu_bytes=total, largest_tensor_bytes=largest,
                maximum_physical_bank_bytes=max_bank, temporary_margin_bytes=transient)


def _cancel(policy):
    if policy.cancelled is not None:
        if policy.cancelled():
            raise ReaderCancelled('CPU checkpoint read cancelled; no partial state is returned')
    else:
        import comfy.model_management
        check = getattr(comfy.model_management, 'throw_exception_if_processing_interrupted', None)
        if check is not None:
            check()


def _fingerprint(path):
    """Bounded-memory header guard also works for unknown future legal dtypes."""
    identity = _identity(path)
    with open(path, 'rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReaderError('Checkpoint is shorter than the safetensors header length')
        size, = struct.unpack('<Q', prefix)
        if not 2 <= size <= min(identity[2] - 8, MAX_FINGERPRINT_HEADER_BYTES):
            raise ReaderError('Invalid or truncated safetensors header')
        # Safetensors enforces its own header validity. Hash in bounded chunks
        # rather than parsing a potentially large legal header into Python.
        digest, remaining = hashlib.sha256(), size
        while remaining:
            block = stream.read(min(remaining, 65536))
            if not block:
                raise ReaderError('Truncated safetensors header')
            digest.update(block)
            remaining -= len(block)
    return dict(identity=identity, header_bytes=size, header_sha256=digest.hexdigest(),
                source_cpu_bytes=identity[2]-size-8, temporary_margin_bytes=0)


def _verify_source(path, plan, policy, phase):
    """Common pre-publication contract for pread and explicit host fallback.

    Stat identity alone misses an in-place fixed-length header rewrite followed
    by restoring its mtime. Keep a bounded header digest check in both routes.
    This still does not replace a caller's full tensor-file SHA guard.
    """
    if _identity(path) != plan['identity']:
        raise ReaderError(f'Checkpoint changed {phase}')
    current = _fingerprint(path)
    if current['identity'] != plan['identity'] or _identity(path) != plan['identity']:
        raise ReaderError(f'Checkpoint changed {phase}')
    if current['header_sha256'] != plan['header_sha256']:
        raise ReaderError(f'Checkpoint header changed {phase}')


def _budget(policy, remaining, margin, diagnostics, stage):
    try:
        memory = (policy.memory_snapshot or psutil.virtual_memory)()
        total, available = int(memory.total), int(memory.available)
        if total <= 0 or available < 0:
            raise ValueError('invalid memory snapshot')
    except (OSError, RuntimeError, MemoryError, ValueError, TypeError, AttributeError) as exc:
        raise ReaderResourceError(f'Cannot verify RAM before checkpoint allocation: {type(exc).__name__}: {exc}') from exc
    reserve = max(MIN_RESERVE, min(MAX_RESERVE, int(total * .125)))
    required = remaining + margin + reserve
    check = dict(stage=stage, available_bytes=available, remaining_source_bytes=remaining,
                 temporary_margin_bytes=margin, reserve_bytes=reserve, required_bytes=required)
    diagnostics.setdefault('memory_checks', []).append(check)
    if available < required:
        raise ReaderResourceError(f'{stage}: available RAM {available} < required {required} '
                                  f'(remaining source + temporary bank margin + unchanged reserve {reserve})')


def _fallback(reason, ckpt, safe_load, device, return_metadata, host, policy, plan, audit):
    audit['fallback_reason'] = reason
    # Default host mmap/aimdo has no eager CPU source allocation. Applying a
    # full-copy budget here would reject working small-memory mmap users.
    if host.disable_mmap and not host.aimdo_enabled:
        _budget(policy, 2 * plan['source_cpu_bytes'], 0, audit, 'host_eager_copy')
    _cancel(policy)
    _verify_source(ckpt, plan, policy, 'after header preflight')
    audit['actual_reader'] = 'host'
    audit['status'] = 'delegating_to_host'
    audit['host_storage_mode'] = 'aimdo' if host.aimdo_enabled else 'eager_copy' if host.disable_mmap else 'mmap'
    result, success = None, False
    try:
        result = host.load_host(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)
        _cancel(policy)
        _verify_source(ckpt, plan, policy, 'during CPU read')
        _cancel(policy)
        success = True
    finally:
        if not success:
            # The official loader owns cleanup if it raises before returning.
            # Once it returns, this boundary owns the not-yet-published state.
            partial = result[0] if return_metadata and result is not None else result
            if isinstance(partial, dict):
                audit['cleared_partial_tensors'] = len(partial)
                partial.clear()
            result = None
            audit['status'] = 'failed_or_cancelled_host_fallback'
    audit['status'] = 'loaded'
    return result


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False, *,
                    prefer_pread=True, policy=None, diagnostics=None, _host=None, _safe_open=None,
                    _with_ownership=False):
    """Standard CPU loader return contract plus JSON reader diagnostics.

    Unsupported eager capabilities and initial RAM shortage select the official
    host loader. Corrupt input, changed files, cancellation and failures after a
    pread begins remain errors; they never silently restart a different reader.
    """
    policy = policy or ReaderPolicy()
    audit = diagnostics if diagnostics is not None else {}
    # This output mapping describes one load. Never publish a stale fallback
    # reason (or non-JSON caller data) when the same diagnostics dict is reused.
    audit.clear()
    if device is not None and torch.device(device).type != 'cpu':
        raise UnsupportedReader('DRAMA checkpoint reader supports CPU checkpoint storage only')
    path = str(Path(ckpt).resolve())
    _cancel(policy)
    host = _host or _host_context()
    if not path.lower().endswith(('.safetensors', '.sft')):
        audit.update(actual_reader='host', fallback_reason='Non-safetensors checkpoint uses the host loader')
        return host.load_host(path, safe_load=safe_load, device=device, return_metadata=return_metadata)
    plan = _fingerprint(path)
    audit.update(status='prepared', requested_reader='official_cpu_pread' if prefer_pread else 'host', actual_reader=None,
                 path=path, source_cpu_bytes=plan['source_cpu_bytes'],
                 header_sha256=plan['header_sha256'], header_bytes=plan['header_bytes'],
                 maximum_physical_bank_bytes=0,
                 temporary_margin_bytes=plan['temporary_margin_bytes'])
    audit['host_context'] = dict(aimdo_enabled=host.aimdo_enabled,
                                utils=host.host_utils_path, storage=host.host_storage_path,
                                disable_mmap=host.disable_mmap)
    if not prefer_pread:
        return _fallback('Pread is only preferred for RAM-offloaded DRAMA body storage', path,
                         safe_load, device, return_metadata, host, policy, plan, audit)
    if host.aimdo_enabled:
        return _fallback('Host aimdo requires its own mmap/TensorFileSlice contract', path,
                         safe_load, device, return_metadata, host, policy, plan, audit)
    opener = _safe_open or safetensors.safe_open
    try:
        signature = inspect.signature(opener)
    except (TypeError, ValueError):
        signature = None
    if signature is None or 'backend' not in signature.parameters:
        return _fallback('Installed safe_open has no verifiable official backend parameter', path,
                         safe_load, device, return_metadata, host, policy, plan, audit)
    try:
        eager_plan = inspect_header(path, policy)
    except UnsupportedReader as exc:
        return _fallback(str(exc), path, safe_load, device, return_metadata, host, policy, plan, audit)
    if eager_plan['identity'] != plan['identity'] or eager_plan['header_sha256'] != plan['header_sha256']:
        raise ReaderError('Checkpoint changed after header preflight')
    plan = eager_plan
    audit.update(maximum_physical_bank_bytes=plan['maximum_physical_bank_bytes'],
                 temporary_margin_bytes=plan['temporary_margin_bytes'])
    try:
        _budget(policy, plan['source_cpu_bytes'], plan['temporary_margin_bytes'], audit, 'before_open')
    except ReaderResourceError as exc:
        return _fallback(str(exc), path, safe_load, device, return_metadata, host, policy, plan, audit)
    _cancel(policy)
    _verify_source(path, plan, policy, 'after header preflight')
    try:
        opened = opener(path, framework='pt', device='cpu', backend='pread')
    except (TypeError, ValueError) as exc:
        return _fallback(f'Official safe_open rejected backend=pread: {type(exc).__name__}: {exc}',
                         path, safe_load, device, return_metadata, host, policy, plan, audit)
    state, tensor = {}, None
    success = False
    try:
        with opened as source:
            keys = source.keys()
            if set(keys) != set(plan['tensors']):
                raise ReaderError('Checkpoint key set changed after header preflight')
            remaining = plan['source_cpu_bytes']
            for name in keys:
                _cancel(policy)
                _budget(policy, remaining, plan['temporary_margin_bytes'], audit, f'before_tensor:{name}')
                tensor = source.get_tensor(name)
                descriptor = plan['tensors'][name]
                physical = descriptor['data_offsets'][1] - descriptor['data_offsets'][0]
                if tensor.device.type != 'cpu' or list(tensor.shape) != descriptor['shape']:
                    raise ReaderError(f'Loaded tensor device/shape disagrees with preflight: {name}')
                if tensor.numel() * tensor.element_size() != physical:
                    raise ReaderError(f'Loaded physical tensor bytes disagree with preflight: {name}')
                expected_dtype = getattr(torch, TORCH_DTYPE_NAMES.get(descriptor['dtype'], ''), None)
                if expected_dtype is None or tensor.dtype != expected_dtype:
                    raise UnsupportedReader(f'Unsupported or mismatched native CPU dtype: {name} {descriptor["dtype"]}')
                state[name] = tensor
                tensor = None
                remaining -= physical
                _cancel(policy)
            metadata = source.metadata() if return_metadata else None
        _cancel(policy)
        # Call the actual current host hook rather than copying its metadata
        # implementation. Pread allocations are not mmap/TensorFileSlice views.
        host.annotate_state_dict(state, path)
        _cancel(policy)
        _verify_source(path, plan, policy, 'during CPU read')
        _cancel(policy)
        if _with_ownership:
            if _host is None and _safe_open is None and policy.memory_snapshot is None:
                from .checkpoint_ownership import _seal_official_read
                state, ownership = _seal_official_read(state, path, plan, host, opener)
                audit['expert_ownership'] = ownership
            else:
                audit['expert_ownership'] = dict(read_certificate_issued=False,
                    fallback_reason='Custom reader/host/memory-provider paths cannot issue internal ownership')
            _cancel(policy)
        audit.update(status='loaded', actual_reader='official_cpu_pread',
                     loaded_tensors=len(state), remaining_source_bytes=0,
                     ownership='independent per-tensor CPU allocations; caller owns returned state')
        success = True
        return (state, metadata) if return_metadata else state
    finally:
        tensor = None
        opened = None
        if not success:
            released = len(state)
            state.clear()
            audit.update(status='failed_or_cancelled', cleared_partial_tensors=released)
