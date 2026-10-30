"""Real CSC products and tiny packages only; no model/service execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tools import build_portable as builder
from tools import build_launcher as launcher
from drama3_t8 import updater
from test_packaging import miniature_source,put
from test_portable_final_guard import final_bundle


def test_direct_full_zip_refuses_missing_launcher_without_touching_existing_zip(final_bundle):
    _,target,manifest,_=final_bundle
    del manifest['launcher']
    previous=put(target.parent,f'Drama3-T8-Portable-{builder.__version__}-win64.zip',b'old ZIP')
    with pytest.raises(ValueError,match='build record is required'):builder.write_archive(manifest)
    assert previous.read_bytes()==b'old ZIP'


def test_zip_only_legacy_layout_refuses_missing_launcher_before_writing(final_bundle,monkeypatch):
    _,target,manifest,_=final_bundle
    del manifest['launcher']
    (target/'package-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['build_portable.py','--target',str(target),'--zip-only'])
    with pytest.raises(ValueError,match='build record is required'):builder.main()
    assert not list(target.parent.glob('*.zip'))


@pytest.mark.parametrize('field,value',[('path','other.exe'),('bytes',True),('sha256','0'*64)])
def test_record_tampering_refused_before_zip_replacement(final_bundle,field,value):
    _,target,manifest,_=final_bundle
    manifest['launcher'][field]=value
    previous=put(target.parent,f'Drama3-T8-Portable-{builder.__version__}-win64.zip',b'old ZIP')
    with pytest.raises(ValueError):builder.write_archive(manifest)
    assert previous.read_bytes()==b'old ZIP'


def test_same_size_changed_exe_is_rejected(final_bundle):
    _,target,manifest,_=final_bundle
    product=target/'Start-Drama3.exe';original=product.read_bytes()
    product.write_bytes(original[:-1]+bytes([original[-1]^1]))
    with pytest.raises(ValueError,match='bytes differ'):builder.write_archive(manifest)


def test_matching_hash_is_not_enough_for_non_PE_artifact(final_bundle):
    _,target,manifest,_=final_bundle
    product=target/'Start-Drama3.exe';product.write_bytes(b'not a Windows executable')
    manifest['launcher'].update(bytes=product.stat().st_size,sha256=hashlib.sha256(product.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='Windows x64 console'):builder.write_archive(manifest)


@pytest.mark.parametrize('name',['tools/build_launcher.py','tools/StartDrama3.cs'])
def test_changed_build_input_requires_new_launcher(final_bundle,name):
    source,_,manifest,_=final_bundle
    with (source/name).open('ab') as handle:handle.write(b'\nchanged build input\n')
    with pytest.raises(ValueError,match='source changes'):builder.write_archive(manifest)


def test_valid_recorded_exe_is_never_recompiled_during_zip(final_bundle,monkeypatch):
    _,target,manifest,_=final_bundle
    before=(target/'Start-Drama3.exe').read_bytes()
    def forbidden(*a,**k):raise AssertionError('ZIP must validate, never recompile its EXE')
    monkeypatch.setattr(launcher,'build_launcher',forbidden)
    builder.write_archive(manifest)
    assert (target/'Start-Drama3.exe').read_bytes()==before


def test_full_layout_refresh_reuses_recorded_identical_launcher(tmp_path,monkeypatch):
    if not launcher._compiler().is_file():pytest.skip('Windows .NET Framework C# compiler required')
    source,target=tmp_path/'source',tmp_path/'full-layout'
    miniature_source(source)
    monkeypatch.setattr(builder,'ROOT',source)
    monkeypatch.setattr(builder.importlib.metadata,'distributions',lambda **k:[SimpleNamespace(metadata={'Name':'transformers'},version='4.57.1')])
    monkeypatch.setattr(sys,'argv',['build_portable.py','--target',str(target)])
    builder.main()
    product=target/'Start-Drama3.exe';first=(product.read_bytes(),product.stat().st_mtime_ns,product.stat().st_ino)
    def forbidden(*a,**k):raise AssertionError('An unchanged recorded EXE must be reused')
    monkeypatch.setattr(launcher,'build_launcher',forbidden)
    builder.main()
    assert first==(product.read_bytes(),product.stat().st_mtime_ns,product.stat().st_ino)
    assert 'Start-Drama3.exe' in json.loads((target/'package-manifest.json').read_text())['files']


def test_compiler_failure_preserves_old_exe_manifest_and_user_data(tmp_path,monkeypatch):
    if not launcher._compiler().is_file():pytest.skip('Windows .NET Framework C# compiler required')
    source,target=tmp_path/'source',tmp_path/'full-layout'
    miniature_source(source)
    (source/'tools/StartDrama3.cs').write_text('this is invalid C#',encoding='utf-8')
    old=put(target,'Start-Drama3.exe',b'previous executable remains')
    manifest=put(target,'package-manifest.json',b'{"previous":true}')
    data=put(target,'userdata/keep.txt',b'user recording remains')
    monkeypatch.setattr(builder,'ROOT',source)
    monkeypatch.setattr(builder.importlib.metadata,'distributions',lambda **k:[SimpleNamespace(metadata={'Name':'transformers'},version='4.57.1')])
    monkeypatch.setattr(sys,'argv',['build_portable.py','--target',str(target)])
    with pytest.raises(RuntimeError,match='compilation failed'):builder.main()
    assert old.read_bytes()==b'previous executable remains'
    assert manifest.read_bytes()==b'{"previous":true}'
    assert data.read_bytes()==b'user recording remains'


def test_legacy_updater_allowlist_is_unchanged_and_excludes_launcher():
    assert updater.TOOLS==frozenset({'check_runtime.py','download_models.py','convert_model.py','safetensors_stream.py','extract_text_head.py','update_portable.py'})
    for path in ['Start-Drama3.exe','tools/build_launcher.py','tools/StartDrama3.cs']:
        assert not updater.is_update_path(path)
    assert 'torch' not in sys.modules
