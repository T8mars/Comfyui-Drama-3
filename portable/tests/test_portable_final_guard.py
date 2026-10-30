"""Actual final acceptance validator + tiny ZIPs; no Torch/model/service imports."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pytest
from tools import build_portable as builder
from tools import finalize_manifest as finalizer
from tools import build_launcher as launcher


def put(root,name,content=b'fixture'):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def final_bundle(tmp_path,monkeypatch):
    if not launcher._compiler().is_file():
        pytest.skip('Windows .NET Framework C# compiler is required for actual EXE archive fixtures')
    source,target=tmp_path/'source',tmp_path/'distribution/Drama3-T8-Portable'
    entries={'app.py':b'# app\n','README.md':b'English','README.zh-CN.md':'中文'.encode(),
        'requirements.txt':b'transformers==4.57.1\n','upstream-lock.json':b'{}',
        'LICENSE':b'license','Start-Drama3.cmd':b'current launcher\n',
        'Update-Drama3.cmd':b'current update launcher\n',
        'acceleration-manifest.json':b'{}','acceleration-requirements.txt':b'# optional\n',
        'drama3_t8/__init__.py':f'__version__ = "{builder.__version__}"\n'.encode(),
        'drama3_t8/core.py':b'current code\n','web/creative.js':b'current UI\n'}
    entries.update({'tools/'+name:b'# current tool\n' for name in
        ('check_runtime.py','download_models.py','convert_model.py','safetensors_stream.py',
         'extract_text_head.py','update_portable.py')})
    for name,data in entries.items():
        put(source,name,data)
        put(target,name,data)
    # Build inputs remain development source, outside the full environment.
    for name in ('build_launcher.py','StartDrama3.cs'):
        put(source,'tools/'+name,(Path(launcher.__file__).parent/name).read_bytes())
    executable=launcher.build_launcher(target/'Start-Drama3.exe',source=source/'tools/StartDrama3.cs')
    launcher_record={'path':'Start-Drama3.exe','bytes':executable.stat().st_size,
        'sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
        'inputs':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ('tools/build_launcher.py','tools/StartDrama3.cs')}}
    evidence=put(tmp_path,'actual-evidence.json',b'{"test_fixture":true}')
    report={'schema':'drama3.release-acceptance.v1','version':builder.__version__,
        'status':'passed','validation_scope':'current_release',
        'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in entries.items()},
        'gates':{name:{'status':'passed','method':'visible_ui',
            'evidence':[{'path':str(evidence),'sha256':finalizer.sha(evidence)}]}
            for name in finalizer.GATES}}
    # Synthetic receipts only qualify this tiny structural fixture, never DRAMA effects.
    put(target,'acceptance-report.json',json.dumps(report).encode())
    manifest={'version':builder.__version__,'launcher':launcher_record,'files':[*entries,'acceptance-report.json','Start-Drama3.exe']}
    put(target,'package-manifest.json',json.dumps(manifest).encode())
    monkeypatch.setattr(builder,'ROOT',source)
    monkeypatch.setattr(builder,'TARGET',target)
    return source,target,manifest,report


def write_report(target,report):
    (target/'acceptance-report.json').write_text(json.dumps(report),encoding='utf-8')


def test_current_source_current_acceptance_writes_real_tiny_zip(final_bundle):
    source,target,manifest,_=final_bundle
    archive=builder.write_archive(manifest)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read(target.name+'/drama3_t8/core.py')==(source/'drama3_t8/core.py').read_bytes()
        assert len(z.namelist())==len(manifest['files'])+1


@pytest.mark.parametrize('field,value,message',[
    ('version','0.2.4','version differs'),
    ('status','in_progress','completed current-release'),
    ('validation_scope','development','completed current-release')])
def test_old_or_pending_acceptance_is_refused_before_zip_mutation(final_bundle,field,value,message):
    _,target,manifest,report=final_bundle
    report[field]=value
    write_report(target,report)
    archive=put(target.parent,f'Drama3-T8-Portable-{builder.__version__}-win64.zip',b'previous ZIP')
    with pytest.raises(ValueError,match=message):builder.write_archive(manifest)
    assert archive.read_bytes()==b'previous ZIP'
    assert not list(target.parent.glob('*.part'))


@pytest.mark.parametrize('name',['drama3_t8/core.py','Start-Drama3.cmd','tools/update_portable.py'])
def test_same_version_stale_packaged_bytes_are_refused(final_bundle,name):
    _,target,manifest,_=final_bundle
    (target/name).write_bytes(b'outdated code\n')
    with pytest.raises(ValueError,match='packaged source differs'):builder.write_archive(manifest)


def test_same_version_stale_acceptance_is_refused_when_source_changed(final_bundle):
    source,_,manifest,_=final_bundle
    (source/'drama3_t8/core.py').write_bytes(b'newer source\n')
    with pytest.raises(ValueError,match='Source changed since acceptance'):builder.write_archive(manifest)


def test_new_source_must_be_accepted_and_shipped(final_bundle):
    source,target,manifest,report=final_bundle
    extra=put(source,'drama3_t8/creative_progress.py',b'new module\n')
    report['source_sha256']['drama3_t8/creative_progress.py']=finalizer.sha(extra)
    write_report(target,report)
    with pytest.raises(ValueError,match='source members are missing'):builder.write_archive(manifest)


def test_report_missing_new_source_hash_is_refused(final_bundle):
    source,_,manifest,_=final_bundle
    put(source,'drama3_t8/creative_progress.py',b'new module\n')
    with pytest.raises(ValueError,match='does not cover current'):builder.write_archive(manifest)


def test_omitted_acceptance_member_is_refused(final_bundle):
    _,_,manifest,_=final_bundle
    manifest['files'].remove('acceptance-report.json')
    with pytest.raises(ValueError,match='requires acceptance-report'):builder.write_archive(manifest)


def test_gate_and_evidence_are_actually_validated(final_bundle):
    _,target,manifest,report=final_bundle
    report['gates']['portable_update']['evidence'][0]['sha256']='0'*64
    write_report(target,report)
    with pytest.raises(ValueError,match='Missing or changed evidence'):builder.write_archive(manifest)


def test_incomplete_gate_is_refused(final_bundle):
    _,target,manifest,report=final_bundle
    report['gates']['studio_ui']['status']='pending'
    write_report(target,report)
    with pytest.raises(ValueError,match='studio_ui has not passed'):builder.write_archive(manifest)


def test_missing_launcher_acceptance_hash_gives_explicit_value_error(final_bundle):
    _,target,manifest,report=final_bundle
    del report['source_sha256']['Start-Drama3.cmd']
    write_report(target,report)
    with pytest.raises(ValueError,match='does not cover portable launch/update sources'):
        builder.write_archive(manifest)
