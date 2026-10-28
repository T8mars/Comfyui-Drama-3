"""DLPack descriptor ownership and official-wrapper isolation (CPU only)."""
import ctypes
import gc
import types
import weakref

import pytest
import torch

from drama3_t8.native import kitchen_compat as compat


def descriptor(capsule):
    pointer = compat._get_pointer(capsule, b'dltensor')
    return ctypes.cast(pointer, ctypes.POINTER(compat._DLManagedTensor)).contents.dl_tensor


@pytest.mark.parametrize('shape', [(1, 1, 4096), (1, 16, 1, 256), (1, 2, 128, 256), (3, 1, 9)])
def test_owned_descriptor_preserves_true_strides_and_shares_storage(shape):
    before = len(compat._exports)
    tensor = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    capsule = compat.stride_preserving_dlpack(tensor)
    dl = descriptor(capsule)
    assert [dl.strides[i] for i in range(dl.ndim)] == list(tensor.stride())
    imported = torch.from_dlpack(capsule)
    assert imported.data_ptr() == tensor.data_ptr()
    assert torch.equal(imported, tensor)
    with pytest.raises(RuntimeError, match='invalid capsule'):
        torch.from_dlpack(capsule)
    del imported, capsule
    gc.collect()
    assert len(compat._exports) == before


def test_export_keeps_source_alive_until_consumer_release():
    before = len(compat._exports)
    tensor = torch.arange(30.).reshape(1, 3, 10).clone()
    alive = weakref.ref(tensor)
    capsule = compat.stride_preserving_dlpack(tensor)
    del tensor
    gc.collect()
    assert alive() is not None
    imported = torch.from_dlpack(capsule)
    del capsule
    gc.collect()
    assert alive() is not None
    assert imported.sum().item() == 435
    del imported
    gc.collect()
    assert alive() is None
    assert len(compat._exports) == before


def test_noncontiguous_offset_view_and_producer_descriptor_stay_unchanged():
    tensor = torch.arange(120.).reshape(3, 4, 10)[1:2, :, 1:9:2]
    original = tensor.__dlpack__(stream=-1)
    original_dl = descriptor(original)
    before = [original_dl.strides[i] for i in range(original_dl.ndim)]
    capsule = compat.stride_preserving_dlpack(tensor)
    imported = torch.from_dlpack(capsule)
    assert imported.data_ptr() == tensor.data_ptr()
    assert torch.equal(imported, tensor)
    assert imported.stride() == tensor.stride()
    assert [original_dl.strides[i] for i in range(original_dl.ndim)] == before
    imported.add_(3)
    assert torch.equal(imported, tensor)
    del imported, capsule, original
    assert not compat._exports


def test_unconsumed_capsule_and_error_path_release_descriptor():
    before = len(compat._exports)
    for index in range(100):
        tensor = torch.ones(1, 1, 10)
        capsule = compat.stride_preserving_dlpack(tensor)
        assert len(compat._exports) == before + 1
        del capsule, tensor
    gc.collect()
    assert len(compat._exports) == before
    with pytest.raises(ValueError):
        capsule = compat.stride_preserving_dlpack(torch.ones(1, 1, 10))
        try:
            raise ValueError('simulated C++ binding refusal')
        finally:
            del capsule
    gc.collect()
    assert len(compat._exports) == before


def test_private_wrapper_retains_official_code_backend_and_globals():
    source = '''def flash_attention_decode_gqa(q, k, v, kv_lengths, return_lse=False, causal=True):
    return _cuda_backend._C.flash_attention_decode_gqa(_cuda_backend._wrap_for_dlpack(q))
'''
    original_export = lambda tensor: 'original'
    seen = []
    def kernel(capsule):
        imported = torch.from_dlpack(capsule)
        seen.append(imported.clone())
    extension = types.SimpleNamespace(flash_attention_decode_gqa=kernel)
    backend = types.SimpleNamespace(_C=extension, _wrap_for_dlpack=original_export)
    namespace = dict(_cuda_backend=backend, _hip_backend=None)
    exec(source, namespace)
    official = namespace['flash_attention_decode_gqa']
    local = compat._with_exact_export(official)
    assert local.__code__ is official.__code__
    assert local.__globals__['_cuda_backend']._C is extension
    assert official.__globals__['_cuda_backend'] is backend
    assert backend._wrap_for_dlpack is original_export
    tensor = torch.ones(1, 1, 10)
    local(tensor, None, None, None)
    assert torch.equal(seen[0], tensor)
    assert not compat._exports


def test_unknown_cuda_wrapper_contract_is_explicitly_rejected():
    def changed(q, k, v, kv_lengths):
        return q
    changed.__globals__['_cuda_backend'] = types.SimpleNamespace(_C=object())
    try:
        with pytest.raises(RuntimeError, match='Unsupported ComfyKitchen CUDA attention export contract'):
            compat._with_exact_export(changed)
    finally:
        del changed.__globals__['_cuda_backend']


@pytest.mark.parametrize('dimension', [32, 256])
def test_unavailable_extension_keeps_official_torch_fallback(dimension, monkeypatch):
    calls = []
    def official(*args, **kwargs):
        calls.append((args, kwargs))
        return args[0] + 1
    module = types.SimpleNamespace(flash_attention_decode_gqa=official,
        flash_attention_decode_gqa_is_available=lambda device, dtype: False)
    monkeypatch.setattr(compat.importlib, 'import_module', lambda name: module)
    tensor = torch.zeros(1, 2, 1, dimension)
    result = compat.kitchen_attention('flash_attention_decode_gqa', tensor, None, None, None)
    assert torch.equal(result, tensor + 1)
    assert len(calls) == 1 and not compat._exports
