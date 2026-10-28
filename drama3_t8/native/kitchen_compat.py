"""Private DLPack boundary for ComfyKitchen's fixed-stride CUDA attention.

PyTorch 2.8 exports a singleton dimension with stride 1 even when the Tensor's
actual stride is different. Kitchen 0.2.40 requires the *output* B/S strides to
be canonical, including B/S == 1. A contiguous Tensor therefore still fails its
binding check. We preserve the Tensor's strides in an owned DLPack descriptor;
neither tensor storage nor the source exporter is modified.

The official Python function/code and loaded CUDA extension are retained. Only
its private export dependency changes; no attention arithmetic is copied here.
"""
import ctypes
import contextvars
import functools
import importlib
import inspect
import types


class _DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int), ('device_id', ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', _DLDevice), ('ndim', ctypes.c_int),
                ('dtype', _DLDataType), ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('strides', ctypes.POINTER(ctypes.c_int64)), ('byte_offset', ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [('dl_tensor', _DLTensor), ('manager_ctx', ctypes.c_void_p),
                ('deleter', ctypes.c_void_p)]


# Separate callable wrappers avoid changing ctypes.pythonapi's shared argtypes.
_get_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(
    ('PyCapsule_GetPointer', ctypes.pythonapi))
_get_raw_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ('PyCapsule_GetPointer', ctypes.pythonapi))
_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ('PyCapsule_IsValid', ctypes.pythonapi))
_new_capsule = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)(
    ('PyCapsule_New', ctypes.pythonapi))
_exports = {}
_export_scope = contextvars.ContextVar('drama_kitchen_export_scope', default=None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _release(pointer):
    # The consumer releases our descriptor. Dropping source_capsule invokes the
    # original producer's deleter, which owns the tensor/storage lifetime.
    _exports.pop(pointer, None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _destroy_unconsumed(capsule):
    # A consumer renames the capsule to "used_dltensor"; then only its managed
    # deleter releases the descriptor. Unconsumed capsules need this destructor.
    if _is_valid(capsule, b'dltensor'):
        _release(_get_raw_pointer(capsule, b'dltensor'))


class _Export:
    def __init__(self, tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        self.source_capsule = tensor.__dlpack__(stream=-1)
        source = ctypes.cast(_get_pointer(self.source_capsule, b'dltensor'),
                             ctypes.POINTER(_DLManagedTensor)).contents.dl_tensor
        if source.ndim != tensor.ndim:
            raise RuntimeError('Unexpected DLPack rank at DRAMA attention boundary')
        self.shape = (ctypes.c_int64 * tensor.ndim)(*tensor.shape)
        self.strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
        descriptor = _DLTensor(source.data, source.device, source.ndim, source.dtype,
                               self.shape, self.strides, source.byte_offset)
        self.managed = _DLManagedTensor(descriptor, None, ctypes.cast(_release, ctypes.c_void_p))


def stride_preserving_dlpack(tensor):
    """Zero-copy owned legacy DLPack capsule, with the producer's true strides.

    The legacy DLPack ABI is the protocol Kitchen's nanobind binding consumes.
    All shape/stride memory is owned by this module until consumer release. The
    producer capsule stays alive for exactly that period. No CUDA allocation,
    copy, synchronization, or alteration of the producer capsule occurs here.
    """
    export = _Export(tensor)
    pointer = ctypes.addressof(export.managed)
    _exports[pointer] = export
    try:
        capsule = _new_capsule(pointer, b'dltensor', ctypes.cast(_destroy_unconsumed, ctypes.c_void_p))
        scope = _export_scope.get()
        if scope is not None:
            # Keep even unconsumed capsules alive across nanobind conversion.
            # On a type error, it may decref argument temporaries while a C-level
            # exception is pending. Python C-API calls in ctypes destructors
            # cannot safely inspect a capsule in that state. Our finally block
            # drops these references after the exception reaches Python.
            scope.append(capsule)
        return capsule
    except BaseException:
        _exports.pop(pointer, None)
        raise


class _BackendExports:
    def __init__(self, backend):
        self._backend = backend

    _wrap_for_dlpack = staticmethod(stride_preserving_dlpack)

    def __getattr__(self, name):
        return getattr(self._backend, name)


@functools.lru_cache(maxsize=8)
def _with_exact_export(function):
    if not isinstance(function, types.FunctionType):
        raise RuntimeError('Unsupported ComfyKitchen attention wrapper: expected a Python function')
    namespace = function.__globals__
    backend = namespace.get('_cuda_backend')
    if backend is None or namespace.get('_hip_backend') is not None:
        return function
    expected = {
        'flash_attention_decode_gqa': ('q', 'k', 'v', 'kv_lengths', 'return_lse', 'causal'),
        'flash_attention_decode_tree_merge': ('out', 'lse', 'q', 'k', 'v', 'mask', 'merged'),
    }.get(function.__name__)
    if (expected is None or tuple(inspect.signature(function).parameters) != expected
            or function.__closure__ or '_wrap_for_dlpack' not in function.__code__.co_names
            or not hasattr(getattr(backend, '_C', None), function.__name__)):
        raise RuntimeError('Unsupported ComfyKitchen CUDA attention export contract; '
                           'DRAMA requires the official GQA/tree Python wrapper and CUDA entry point')
    private = dict(namespace, _cuda_backend=_BackendExports(backend))
    result = types.FunctionType(function.__code__, private, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__doc__ = function.__doc__
    return result


def kitchen_attention(name, *args, **kwargs):
    """Run the official Kitchen function with a DRAMA-private export boundary."""
    module = importlib.import_module('comfy_kitchen.flash_attention')
    function = getattr(module, name)
    query_index = {'flash_attention_decode_gqa': 0,
                   'flash_attention_decode_tree_merge': 2}.get(name)
    if query_index is None:
        raise RuntimeError('Unknown DRAMA ComfyKitchen attention export entry point')
    query = args[query_index] if len(args) > query_index else kwargs['q']
    # Preserve the official CPU/unsupported-dtype/head-size path verbatim. An
    # unavailable CUDA extension must not prevent Kitchen's existing fallback.
    if (query.shape[-1] != 256
            or not module.flash_attention_decode_gqa_is_available(query.device, query.dtype)):
        return function(*args, **kwargs)
    capsules = []
    token = _export_scope.set(capsules)
    try:
        return _with_exact_export(function)(*args, **kwargs)
    finally:
        _export_scope.reset(token)
        capsules.clear()


def compatible_cache(cache):
    """Adapt only the host's BF16 FixedKVCache; other cache policies stay intact."""
    from comfy.text_encoders.llama import FixedKVCache
    if type(cache) is FixedKVCache:
        cache.__class__ = _compatible_cache_class(FixedKVCache)
    return cache


@functools.lru_cache(maxsize=1)
def _compatible_cache_class(base):
    class DramaFixedKVCache(base):
        def attend(self, xq, xk, xv):
            tree = self.tracker['tree']
            if tree is not None:
                import torch
                out, lse = kitchen_attention('flash_attention_decode_gqa',
                    xq, self.key, self.value, self.seqlen, return_lse=True, causal=False)
                return kitchen_attention('flash_attention_decode_tree_merge',
                    out, lse, xq, xk, xv, tree.mask, torch.empty_like(out))
            return kitchen_attention('flash_attention_decode_gqa',
                                     xq, self.key, self.value, self.seqlen)
    return DramaFixedKVCache
