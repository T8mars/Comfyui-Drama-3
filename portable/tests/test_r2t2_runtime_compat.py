import hashlib
import json

import pytest

from drama3_t8.r2t2 import runtime_compat as module


def runtime(tmp_path,real,virtual):
    (tmp_path/'backends').mkdir()
    backend=tmp_path/'backends/ggml-cuda.dll';backend.write_bytes(b'fixture CUDA backend')
    record={'schema':1,'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
        'cuda_real_arches':real,'cuda_virtual_arches':virtual}
    (tmp_path/'runtime-compat.json').write_text(json.dumps(record),encoding='utf-8')
    return tmp_path


@pytest.mark.parametrize('capability',[(6,1),(7,5),(8,0),(8,6),(8,9),(9,0),(12,0)])
def test_multiarch_images_cover_explicit_ptx_and_sass_targets(tmp_path,capability):
    directory=runtime(tmp_path,[86,89,120],[50,61,70,75,80,90])
    assert module.assert_runtime_compatible(directory,capability)['device_capability']==list(capability)


@pytest.mark.parametrize('capability',[(6,1),(7,5),(8,6),(8,9),(12,1)])
def test_120a_only_backend_fails_before_any_native_call(tmp_path,capability):
    directory=runtime(tmp_path,[120],[])
    with pytest.raises(RuntimeError,match='不支持当前显卡'):
        module.assert_runtime_compatible(directory,capability)


def test_replaced_binary_cannot_reuse_an_unrelated_architecture_manifest(tmp_path):
    directory=runtime(tmp_path,[120],[])
    (directory/'backends/ggml-cuda.dll').write_bytes(b'different unsupported image')
    with pytest.raises(RuntimeError,match='不匹配'):module.assert_runtime_compatible(directory,(12,0))


def test_known_legacy_binary_is_classified_without_manifest(tmp_path,monkeypatch):
    directory=runtime(tmp_path,[120],[])
    (directory/'runtime-compat.json').unlink()
    backend=directory/'backends/ggml-cuda.dll'
    monkeypatch.setattr(module,'LEGACY_BACKENDS',{hashlib.sha256(backend.read_bytes()).hexdigest():([120],[])})
    with pytest.raises(RuntimeError,match='不支持当前显卡'):module.assert_runtime_compatible(directory,(8,6))
    assert module.assert_runtime_compatible(directory,(12,0))['cuda_real_arches']==[120]


def test_unknown_unmanifested_binary_is_not_assumed_compatible(tmp_path):
    directory=runtime(tmp_path,[120],[])
    (directory/'runtime-compat.json').unlink()
    with pytest.raises(RuntimeError,match='架构清单'):module.assert_runtime_compatible(directory,(12,0))


def test_unsupported_runtime_is_rejected_before_loading_any_ggml_dll(tmp_path,monkeypatch):
    import os
    if os.name!='nt':pytest.skip('Windows loader integration')
    from drama3_t8.r2t2 import native
    directory=runtime(tmp_path,[120],[])
    check=module.assert_runtime_compatible
    monkeypatch.setattr(module,'assert_runtime_compatible',lambda path:check(path,(8,6)))
    monkeypatch.setattr(native,'runtime_directory',lambda:directory)
    monkeypatch.setattr(native,'_LIBRARIES',{})
    monkeypatch.setattr(native.C,'CDLL',lambda *_:pytest.fail('An incompatible DLL reached native loading'))
    with pytest.raises(RuntimeError,match='不支持当前显卡'):native._load_library()
