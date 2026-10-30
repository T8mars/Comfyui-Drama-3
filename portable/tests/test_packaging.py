import pytest
import hashlib
import io
import tarfile
import json
import os
import subprocess
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile
from tools import build_portable
from tools import build_public_release as public
from tools import build_launcher as launcher


def put(root, name, content=b"fixture"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def compiled_launcher_manifest(target, manifest, source=None):
    """Use the real C# compiler, never a fabricated EXE header or hash."""
    if not launcher._compiler().is_file():
        pytest.skip('Windows .NET Framework C# compiler is required for full Windows EXE fixtures')
    source = Path(source or build_portable.ROOT)
    product = launcher.build_launcher(target / 'Start-Drama3.exe', source=source/'tools/StartDrama3.cs')
    manifest['files'].append('Start-Drama3.exe')
    manifest['launcher'] = {'path':'Start-Drama3.exe','bytes':product.stat().st_size,
        'sha256':hashlib.sha256(product.read_bytes()).hexdigest(),
        'inputs':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                  for name in ('tools/build_launcher.py','tools/StartDrama3.cs')}}
    return manifest


@pytest.mark.parametrize("suffix", [".dll", ".safetensors", ".json"])
def test_same_size_replacement_is_included_in_next_package(tmp_path, monkeypatch, suffix):
    target = tmp_path / "package"
    monkeypatch.setattr(build_portable, "TARGET", target)
    monkeypatch.setattr(build_portable, "COPIED", set())
    source = tmp_path / ("source" + suffix)
    destination = target / source.name
    source.write_bytes(b"old")
    build_portable.copy_file(source, destination)
    source.write_bytes(b"new")
    build_portable.copy_file(source, destination)
    assert destination.read_bytes() == b"new"


def test_replacing_an_existing_hardlink_does_not_modify_an_old_source_or_package(tmp_path, monkeypatch):
    target = tmp_path / "package"
    monkeypatch.setattr(build_portable, "TARGET", target)
    monkeypatch.setattr(build_portable, "COPIED", set())
    old = put(tmp_path, "old-source.dat", b"a" * (4 * 1024 * 1024 + 1))
    destination = target / "large.dat"
    build_portable.copy_file(old, destination)
    assert os.path.samefile(old, destination)
    replacement = put(tmp_path, "replacement.dat", b"b" * old.stat().st_size)
    build_portable.copy_file(replacement, destination)
    assert old.read_bytes() == b"a" * old.stat().st_size
    assert destination.read_bytes() == replacement.read_bytes()
    assert not os.path.samefile(old, destination)


def test_failed_copy_retains_previous_file_and_removes_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "package"
    monkeypatch.setattr(build_portable, "TARGET", target)
    monkeypatch.setattr(build_portable, "COPIED", set())
    source = put(tmp_path, "source.dat", b"new")
    destination = put(target, "item.dat", b"old")
    def fail(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("copy interrupted")
    monkeypatch.setattr(build_portable.shutil, "copy2", fail)
    with pytest.raises(OSError, match="interrupted"):
        build_portable.copy_file(source, destination)
    assert destination.read_bytes() == b"old"
    assert [path.name for path in target.iterdir()] == ["item.dat"]


def _windows_replace_error(code):
    error = PermissionError(13, 'file temporarily unavailable')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
@pytest.mark.parametrize('existing', [False, True])
def test_copy_file_retries_transient_windows_replace_without_touching_old_file(tmp_path, monkeypatch, code, existing):
    target = tmp_path / 'package'
    source = put(tmp_path, 'runtime.dll', b'new')
    destination = target / 'runtime.dll'
    if existing:
        put(target, 'runtime.dll', b'old')
    monkeypatch.setattr(build_portable, 'TARGET', target)
    monkeypatch.setattr(build_portable, 'COPIED', set())
    monkeypatch.setattr(build_portable, '_WINDOWS', True, raising=False)
    delays = []
    def wait(seconds):
        delays.append(seconds)
        time.sleep(seconds)
    monkeypatch.setattr(build_portable, 'time', SimpleNamespace(sleep=wait), raising=False)
    original = build_portable.os.replace
    attempts = []

    def briefly_busy(temporary, final):
        attempts.append((temporary, final))
        assert destination.exists() == existing
        if existing:
            assert destination.read_bytes() == b'old'
        if len(attempts) < 3:
            raise _windows_replace_error(code)
        original(temporary, final)

    monkeypatch.setattr(build_portable.os, 'replace', briefly_busy)
    build_portable.copy_file(source, destination)
    assert destination.read_bytes() == b'new'
    assert source.read_bytes() == b'new'
    assert len(attempts) == 3
    assert delays == [.05, .1]
    assert not list(target.glob('*.part'))


@pytest.mark.parametrize('existing', [False, True])
def test_copy_file_permanent_windows_replace_failure_is_bounded_and_retains_previous_data(tmp_path, monkeypatch, existing):
    target = tmp_path / 'package'
    source = put(tmp_path, 'runtime.dll', b'new')
    destination = target / 'runtime.dll'
    if existing:
        put(target, 'runtime.dll', b'old')
    monkeypatch.setattr(build_portable, 'TARGET', target)
    monkeypatch.setattr(build_portable, 'COPIED', set())
    monkeypatch.setattr(build_portable, '_WINDOWS', True, raising=False)
    delays = []
    monkeypatch.setattr(build_portable, 'time', SimpleNamespace(sleep=delays.append), raising=False)
    error = _windows_replace_error(5)
    attempts = []

    def locked(temporary, final):
        attempts.append((temporary, final))
        raise error

    monkeypatch.setattr(build_portable.os, 'replace', locked)
    with pytest.raises(PermissionError) as raised:
        build_portable.copy_file(source, destination)
    assert raised.value is error
    assert len(attempts) == 6
    assert delays == pytest.approx([.05, .1, .15, .2, .25])
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == b'old'
    assert source.read_bytes() == b'new'
    assert not list(target.glob('*.part'))


@pytest.mark.parametrize('windows,code', [(False, 5), (True, None), (True, 2), (True, 87)])
def test_copy_file_does_not_retry_non_windows_or_unrelated_replace_errors(tmp_path, monkeypatch, windows, code):
    target = tmp_path / 'package'
    source = put(tmp_path, 'runtime.dll', b'new')
    destination = put(target, 'runtime.dll', b'old')
    monkeypatch.setattr(build_portable, 'TARGET', target)
    monkeypatch.setattr(build_portable, 'COPIED', set())
    monkeypatch.setattr(build_portable, '_WINDOWS', windows, raising=False)
    delays = []
    monkeypatch.setattr(build_portable, 'time', SimpleNamespace(sleep=delays.append), raising=False)
    error = PermissionError(13, 'permanent') if code is None else _windows_replace_error(code)
    attempts = []

    def permanent(temporary, final):
        attempts.append((temporary, final))
        raise error

    monkeypatch.setattr(build_portable.os, 'replace', permanent)
    with pytest.raises(PermissionError) as raised:
        build_portable.copy_file(source, destination)
    assert raised.value is error
    assert len(attempts) == 1 and delays == []
    assert destination.read_bytes() == b'old'
    assert not list(target.glob('*.part'))


def test_full_zip_commit_retries_windows_busy_error_without_unlinking_old_archive(tmp_path, monkeypatch):
    # This fixture exercises archive/layout mechanics, not release qualification.
    monkeypatch.setattr(build_portable, 'validate_final_archive', lambda *args: None)
    target = tmp_path / 'package'
    monkeypatch.setattr(build_portable, 'TARGET', target)
    monkeypatch.setattr(build_portable, '_WINDOWS', True, raising=False)
    delays = []
    def wait(seconds):
        delays.append(seconds)
        time.sleep(seconds)
    monkeypatch.setattr(build_portable, 'time', SimpleNamespace(sleep=wait), raising=False)
    put(target, 'app.py', b'current code')
    put(target, 'package-manifest.json', b'{}')
    manifest=compiled_launcher_manifest(target, {'version':build_portable.__version__,'files':['app.py']})
    archive = put(target.parent, f'Drama3-T8-Portable-{build_portable.__version__}-win64.zip', b'old archive')
    original = build_portable.os.replace
    attempts = []

    def briefly_busy(temporary, final):
        attempts.append((temporary, final))
        assert archive.read_bytes() == b'old archive'
        if len(attempts) == 1:
            raise _windows_replace_error(32)
        original(temporary, final)

    monkeypatch.setattr(build_portable.os, 'replace', briefly_busy)
    build_portable.write_archive(manifest)
    with zipfile.ZipFile(archive) as z:
        assert z.read(target.name + '/app.py') == b'current code'
    assert len(attempts) == 2 and delays == [.05]
    assert not list(target.parent.glob('*.part'))


@pytest.mark.parametrize("relative", [".", "..", "runtime/new", "models/new", "web/new", "tools/new",
    "licenses/new", "comfyui-drama3-T8/new", "vendor/new", "drama3_t8/new", "userdata/new", "outputs/new"])
def test_package_target_cannot_overlap_source_trees(tmp_path, monkeypatch, relative):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(build_portable, "ROOT", root)
    with pytest.raises(ValueError, match="separate"):
        build_portable.validate_target(root / relative)
    assert build_portable.validate_target(root / "release_packages/new") == root / "release_packages/new"
    assert build_portable.validate_target(tmp_path / "other drive package") == tmp_path / "other drive package"


@pytest.mark.parametrize("base_collision", [True, False])
def test_duplicate_extra_model_filenames_are_rejected_before_copying(tmp_path, monkeypatch, base_collision):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(build_portable, "ROOT", root)
    target = tmp_path / "package"
    monkeypatch.setattr(build_portable, "TARGET", target)
    first = put(tmp_path, "a/DRAMA3-T8-FP8.SAFETENSORS" if base_collision else "a/quant.safetensors", b"fixture")
    args = ["build_portable.py", "--target", str(target), "--extra-model", str(first)]
    if not base_collision:
        second = put(tmp_path, "b/QUANT.SAFETENSORS", b"different fixture")
        args += ["--extra-model", str(second)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="duplicates"):
        build_portable.main()
    assert not target.exists()


@pytest.mark.parametrize("source_kind", ["runtime", "models"])
def test_selected_external_sources_cannot_contain_the_package(tmp_path, monkeypatch, source_kind):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(build_portable, "ROOT", root)
    source = tmp_path / "verified external environment" / source_kind
    target = source / "recursive-package"
    monkeypatch.setattr(sys, "argv", ["build_portable.py", "--target", str(target),
        "--runtime-source" if source_kind == "runtime" else "--models-source", str(source)])
    with pytest.raises(ValueError, match="separate"):
        build_portable.main()
    assert not target.exists()


@pytest.mark.parametrize("name", ["../private.txt", "web/../../private.txt", "web\\private.txt", "/private.txt",
    "C:/private.txt", "RoadMap.MD", "web/RoadMap.MD", "userdata/voices.json", "outputs/private.wav", "package-manifest.json"])
def test_zip_only_rejects_invalid_or_private_manifest_paths(tmp_path, monkeypatch, name):
    target = tmp_path / "package"
    target.mkdir()
    monkeypatch.setattr(build_portable, "TARGET", target)
    manifest = {"version": build_portable.__version__, "files": [name]}
    with pytest.raises(ValueError):
        build_portable.archive_members(manifest)


def test_zip_only_rejects_stale_versions_and_case_duplicate_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(build_portable, "TARGET", tmp_path)
    put(tmp_path, "app.py")
    put(tmp_path, "package-manifest.json")
    with pytest.raises(ValueError, match="Refresh"):
        build_portable.archive_members({"version": "0.0.0", "files": ["app.py"]})
    with pytest.raises(ValueError, match="Invalid"):
        build_portable.archive_members({"version": build_portable.__version__, "files": ["app.py", "APP.PY"]})


def miniature_source(root):
    """Text placeholders only: no real weights, Python installation or model execution."""
    root_files = ['app.py','README.md','README.zh-CN.md','LICENSE','Start-Drama3.cmd','Update-Drama3.cmd',
        'requirements.txt','acceleration-manifest.json','acceleration-requirements.txt','acceptance-report.json']
    for name in root_files:
        put(root, name)
    put(root, 'upstream-lock.json', b'{}')
    put(root, '.gitattributes', b'* -text\n')
    put(root, 'comfyui-drama3-T8/.gitattributes', b'* -text\n')
    for name in ['README.md', 'README.zh-CN.md', 'LICENSE', 'upstream-lock.json']:
        put(root, 'comfyui-drama3-T8/' + name, b'stale attached node metadata')
    for name in ['check_runtime.py','download_models.py','convert_model.py','quantize_converted.py',
                 'safetensors_stream.py','extract_text_head.py','update_portable.py','make_workflows.py','build_portable.py','build_public_release.py','finalize_manifest.py']:
        put(root, 'tools/' + name)
    for name in ['build_launcher.py','StartDrama3.cs']:
        put(root, 'tools/'+name, (Path(launcher.__file__).parent/name).read_bytes())
    for name in ['editor_core.js','drama3_editor.js','seed-compat.js']:
        put(root, 'web/' + name, b'current web source')
        put(root, 'comfyui-drama3-T8/web/' + name, b'stale web source')
    put(root, 'drama3_t8/engine.py', b'current engine source')
    put(root, 'drama3_t8/editor.py', b'current editor source')
    put(root, 'drama3_t8/studio_editor.py', b'studio only')
    put(root, 'drama3_t8/updater.py', b'current updater source')
    put(root, 'drama3_t8/native/fish/tokenizer.json', b'{}')
    put(root, 'comfyui-drama3-T8/drama3_t8/engine.py', b'stale engine source')
    put(root, 'comfyui-drama3-T8/__init__.py')
    put(root, 'comfyui-drama3-T8/workflows/workflow_basic.json', b'{"version":0.4,"nodes":[],"links":[]}')
    put(root, 'comfyui-drama3-T8/.github/workflows/publish.yml')
    put(root, 'licenses/NOTICE-DRAMA3-EDITOR', b'editor notice')
    put(root, 'runtime/python/python.exe', b'fake runtime placeholder')
    put(root, 'vendor/ComfyUI/comfy/model_management.py')
    put(root, 'vendor/ComfyUI/comfy_api/input/__init__.py')
    put(root, 'vendor/ComfyUI/comfy/ldm/models/autoencoder.py')
    for name in ['custom_nodes','models','outputs','output','userdata','user','input','temp']:
        put(root, 'vendor/ComfyUI/' + name + '/private.py', b'top-level runtime data')
        put(root, 'vendor/ComfyUI/comfy/source_packages/' + name + '/__init__.py', b'nested source package')
    put(root, 'vendor/ComfyUI/models/private.safetensors')
    put(root, 'vendor/ComfyUI/user/private.json')
    put(root, 'vendor/r2t2_native/native_ext.cpp')
    for name in ['drama3-t8-fp8.safetensors','drama3-codec.safetensors','drama3-text-head.safetensors',
                 'qwen3-asr-06b.safetensors','drama3-line-head.safetensors']:
        put(root, 'models/' + name, b'not model weights')
    put(root, 'models/qwen-aligner/shards/part.safetensors', b'not model weights')
    put(root, 'models/Confucius4-R2T2-GGUF/nested/projector.gguf', b'not model weights')
    put(root, 'models/FireRedVAD-ONNX/model.onnx', b'not model weights')
    put(root, 'web/RoadMap.MD', b'private research')
    put(root, 'web/__pycache__/ignored.pyc')


def test_full_layout_uses_requested_target_current_nodes_and_complete_model_inventory(tmp_path, monkeypatch):
    if not launcher._compiler().is_file():
        pytest.skip('Windows .NET Framework C# compiler is required for the full Windows layout')
    # This fixture exercises archive/layout mechanics, not release qualification.
    monkeypatch.setattr(build_portable, 'validate_final_archive', lambda *args: None)
    root, target = tmp_path / 'source', tmp_path / 'distribution/Chinese 空格 package'
    miniature_source(root)
    monkeypatch.setattr(build_portable, 'ROOT', root)
    monkeypatch.setattr(build_portable, 'TARGET', target)
    monkeypatch.setattr(build_portable, 'COPIED', {'web/stale-file.js'})
    metadata_paths=[]
    def distributions(*,path):
        metadata_paths.extend(path)
        return [SimpleNamespace(metadata={"Name": "transformers"}, version="4.57.1")]
    monkeypatch.setattr(build_portable.importlib.metadata, 'distributions', distributions)
    extra = put(tmp_path, 'external/quant.SAFETENSORS', b'not model weights')
    old_archive = put(target.parent, 'Drama3-T8-Portable-0.0.1-win64.zip', b'old release remains')
    private = put(target, 'userdata/voices.json', b'user voices remain')
    output = put(target, 'outputs/prior.wav', b'user output remains')
    monkeypatch.setattr(sys, 'argv', ['build_portable.py','--target',str(target),'--extra-model',str(extra),'--hash-models','--zip'])
    build_portable.main()
    manifest = json.loads((target / 'package-manifest.json').read_text(encoding='utf-8'))
    assert metadata_paths == [str(root / 'runtime/python/Lib/site-packages')]
    assert 'web/stale-file.js' not in manifest['files']
    assert 'Start-Drama3.exe' in manifest['files']
    assert (target/'Start-Drama3.exe').read_bytes()[:2]==b'MZ'
    assert manifest['launcher']['sha256']==hashlib.sha256((target/'Start-Drama3.exe').read_bytes()).hexdigest()
    assert manifest['launcher']['inputs']['tools/StartDrama3.cs']==hashlib.sha256((root/'tools/StartDrama3.cs').read_bytes()).hexdigest()
    for name in ['build_launcher.py','StartDrama3.cs']:
        assert 'tools/'+name not in manifest['files']
        assert not (target/'tools'/name).exists()
    assert private.read_bytes() == b'user voices remain' and output.read_bytes() == b'user output remains'
    assert old_archive.read_bytes() == b'old release remains'
    # The public v0.1.11 updater only accepts the original six tool paths.
    # Quantization remains available through convert_model.py quantize.
    assert 'tools/convert_model.py' in manifest['files']
    assert 'tools/quantize_converted.py' not in manifest['files']
    assert not (target / 'tools/quantize_converted.py').exists()
    assert 'tools/finalize_manifest.py' not in manifest['files']
    assert not (target / 'tools/finalize_manifest.py').exists()
    assert not (target / '.gitattributes').exists()
    assert (target / 'comfyui-drama3-T8/.gitattributes').read_bytes() == b'* -text\n'
    for name in ['editor_core.js','drama3_editor.js','seed-compat.js']:
        assert (target / 'comfyui-drama3-T8/web' / name).read_bytes() == b'current web source'
    assert (target / 'comfyui-drama3-T8/drama3_t8/engine.py').read_bytes() == b'current engine source'
    assert not (target / 'comfyui-drama3-T8/drama3_t8/studio_editor.py').exists()
    vendor_sources = ['comfy_api/input/__init__.py','comfy/ldm/models/autoencoder.py']
    for name in ['custom_nodes','models','outputs','output','userdata','user','input','temp']:
        vendor_sources.append('comfy/source_packages/' + name + '/__init__.py')
        assert not (target / 'vendor/ComfyUI' / name / 'private.py').exists()
        assert 'vendor/ComfyUI/' + name + '/private.py' not in manifest['files']
    for name in vendor_sources:
        assert (target / 'vendor/ComfyUI' / name).read_bytes() == (root / 'vendor/ComfyUI' / name).read_bytes()
        assert 'vendor/ComfyUI/' + name in manifest['files']
    shared_metadata = ['README.md', 'README.zh-CN.md', 'LICENSE', 'upstream-lock.json']
    for name in shared_metadata:
        assert (target / 'comfyui-drama3-T8' / name).read_bytes() == (root / name).read_bytes()
        assert 'comfyui-drama3-T8/' + name in manifest['files']
        # Standalone full builds refresh their output without mutating the
        # older node staging folder or requiring public.sync_node_source().
        assert (root / 'comfyui-drama3-T8' / name).read_bytes() == b'stale attached node metadata'
    modeled = {record['path']: record for record in manifest['models']}
    assert set(modeled) == {name for name in manifest['files'] if name.startswith('models/')}
    assert 'models/qwen-aligner/shards/part.safetensors' in modeled and 'models/quant.SAFETENSORS' in modeled
    for name, record in modeled.items():
        assert record['sha256'] == hashlib.sha256((target / name).read_bytes()).hexdigest()
    archive = target.parent / f'Drama3-T8-Portable-{build_portable.__version__}-win64.zip'
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        names = {name.removeprefix(target.name + '/') for name in handle.namelist()}
        assert names == set(manifest['files']) | {'package-manifest.json'}
        assert not any('roadmap.md' in name.casefold() or name.startswith(('userdata/','outputs/')) for name in names)
        assert 'vendor/ComfyUI/models/private.safetensors' not in names
        assert 'vendor/ComfyUI/user/private.json' not in names
        assert handle.getinfo(target.name + '/models/quant.SAFETENSORS').compress_type == zipfile.ZIP_STORED
        for name in shared_metadata:
            assert handle.read(target.name + '/comfyui-drama3-T8/' + name) == (root / name).read_bytes()
        for name in vendor_sources:
            assert handle.read(target.name + '/vendor/ComfyUI/' + name) == (root / 'vendor/ComfyUI' / name).read_bytes()


@pytest.fixture
def public_source(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    miniature_source(root)
    node, output = root / 'comfyui-drama3-T8', tmp_path / 'public-assets'
    output.mkdir()
    monkeypatch.setattr(public, 'ROOT', root)
    monkeypatch.setattr(public, 'NODE', node)
    monkeypatch.setattr(public, 'OUT', output)
    return root, node, output


def test_public_code_and_node_assets_exclude_weights_runtime_and_research_and_keep_notices(public_source):
    root, node, output = public_source
    for name in ['models/leak.safetensors','runtime/python/python.exe','drama3_t8/bank.GGUF',
                 'web/library.DLL','native-source/extension.PYD','RoadMap.MD','web/RoadMap.MD']:
        put(node, name, b'must not publish')
    public.sync_node_source()
    public.code_asset()
    public.node_asset()
    for name in [f'Drama3-T8-Portable-{public.__version__}-code.zip',f'ComfyUI-Drama3-T8-{public.__version__}.zip']:
        with zipfile.ZipFile(output / name) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert not any(path.casefold().endswith(('.safetensors','.gguf','.onnx','.dll','.pyd','.exe','.pth','.bin'))
                           or 'roadmap.md' in path.casefold() for path in names)
            assert not any(path.endswith('tools/quantize_converted.py') for path in names)
            assert any(path.endswith('web/seed-compat.js') for path in names)
            assert any(path.endswith('licenses/NOTICE-DRAMA3-EDITOR') for path in names)
    manifest = json.loads((output / 'portable-update.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(output / manifest['asset']['name']) as archive:
        assert set(archive.namelist()) == set(manifest['files'])
        for name, record in manifest['files'].items():
            assert record['sha256'] == hashlib.sha256(archive.read(name)).hexdigest()


@pytest.mark.parametrize('autocrlf', ['true', 'false'])
def test_source_sync_refreshes_editor_and_seed_without_publishing_runtime(public_source, tmp_path, autocrlf):
    if not launcher._compiler().is_file():
        pytest.skip('Windows .NET Framework C# compiler is required for the isolated copied-helper compile')
    root, node, output = public_source
    # This is the real development alias, not an invented wrapper fixture.
    wrapper = (Path(__file__).resolve().parents[1] / 'tools/quantize_converted.py').read_bytes()
    publisher = (Path(__file__).resolve().parents[1] / 'tools/build_public_release.py').read_bytes()
    put(root, 'tools/quantize_converted.py', wrapper)
    put(root, 'tools/build_public_release.py', publisher)
    source = Path(__file__).resolve().parents[1]
    builder = (source / 'tools/build_portable.py').read_bytes()
    finalizer = (source / 'tools/finalize_manifest.py').read_bytes()
    put(root, 'tools/build_portable.py', builder)
    put(root, 'tools/finalize_manifest.py', finalizer)
    put(root, 'drama3_t8/__init__.py', (source / 'drama3_t8/__init__.py').read_bytes())
    destination = tmp_path / 'checkout'
    destination.mkdir()
    empty = put(tmp_path, 'empty-git-config-and-attributes', b'')
    template = tmp_path / 'empty-git-template'
    template.mkdir()
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=str(empty),
        GIT_ATTR_NOSYSTEM='1', GIT_TEMPLATE_DIR=str(template))
    def git(*arguments, cwd=destination):
        return subprocess.run(['git', '-c', 'core.attributesFile=' + str(empty),
            '-c', 'core.hooksPath=' + str(template), '-c', 'core.safecrlf=false',
            '-c', 'commit.gpgsign=false', *arguments], cwd=cwd, env=environment,
            check=True, capture_output=True).stdout
    git('init', str(destination), cwd=tmp_path)
    git('config', 'user.name', 'Byte Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'core.autocrlf', autocrlf)
    put(root, 'tests/line-endings-fixture.py', b'VALUE = 1\r\n')
    put(node, '.gitattributes', b'stale node attributes\n')
    put(node, 'models/leak.safetensors')
    put(node, 'runtime/python/python.exe')
    put(node, 'RoadMap.MD')
    put(root, 'vendor/r2t2_native/build/leak.dll')
    put(root, 'tests/fixture.safetensors')
    put(root, 'tests/runtime/python.exe')
    put(root, 'tests/frontend_canvas_oracle.mjs', b'actual test source')
    public.sync_source(destination)
    assert (destination / '.gitattributes').read_bytes() == b'* -text\n'
    assert (node / '.gitattributes').read_bytes() == b'* -text\n'
    assert (destination / 'web/seed-compat.js').read_bytes() == b'current web source'
    assert (destination / 'drama3_t8/editor.py').read_bytes() == b'current editor source'
    assert (destination / 'licenses/NOTICE-DRAMA3-EDITOR').read_bytes() == b'editor notice'
    assert (destination / '.github/workflows/publish.yml').exists()
    assert (destination / 'portable/tests/frontend_canvas_oracle.mjs').read_bytes() == b'actual test source'
    assert (destination / 'portable/tools/quantize_converted.py').read_bytes() == wrapper
    assert (destination / 'portable/tools/build_public_release.py').read_bytes() == publisher
    assert not (destination / 'tools/quantize_converted.py').exists()
    assert (destination / 'portable/tools/finalize_manifest.py').read_bytes() == finalizer
    for name in ['build_launcher.py','StartDrama3.cs']:
        assert (destination/'portable/tools'/name).read_bytes()==(root/'tools'/name).read_bytes()
    # A fresh isolated interpreter cannot borrow the original source's tools.
    probe = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
from tools import build_portable as builder
assert Path(builder.__file__).resolve()==root/'tools/build_portable.py'
try:builder.validate_final_archive({},[])
except ValueError as error:assert 'requires acceptance-report.json' in str(error)
else:raise AssertionError('Empty archive qualification was accepted')
from tools import finalize_manifest as finalizer
assert Path(finalizer.__file__).resolve()==root/'tools/finalize_manifest.py'
from tools import build_launcher as executable_builder
assert Path(executable_builder.__file__).resolve()==root/'tools/build_launcher.py'
product=executable_builder.build_launcher(Path(sys.argv[2]))
assert product.read_bytes()[:2]==b'MZ'
assert 'torch' not in sys.modules
print(json.dumps({'exe_bytes':product.stat().st_size,'launcher_helper':str(Path(executable_builder.__file__).resolve()),'builder':str(Path(builder.__file__).resolve()),'finalizer':str(Path(finalizer.__file__).resolve()),'isolated':sys.flags.isolated}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', probe, str(destination / 'portable'),str(tmp_path/'isolated Chinese 空格'/'Start-Drama3.exe')],
        cwd=tmp_path, check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    assert receipt['isolated'] == 1
    assert receipt['exe_bytes']>0
    assert receipt['launcher_helper']==str((destination/'portable/tools/build_launcher.py').resolve())
    assert receipt['builder'] == str((destination / 'portable/tools/build_portable.py').resolve())
    assert receipt['finalizer'] == str((destination / 'portable/tools/finalize_manifest.py').resolve())
    expected = {path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob('*') if path.is_file() and '.git' not in path.relative_to(destination).parts}
    git('add', '--all', '--', '.')
    git('commit', '-m', 'source sync bytes')
    with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', 'HEAD'))) as committed:
        actual = {member.name: committed.extractfile(member).read()
            for member in committed.getmembers() if member.isfile()}
    assert actual == expected
    sample = 'portable/tests/line-endings-fixture.py'
    assert git('hash-object', '--no-filters', sample) == git('hash-object', '--path', sample, sample)
    assert git('check-attr', 'text', '--', sample).decode().strip().endswith(': unset')
    for checkout_mode in ('true', 'false'):
        clone = tmp_path / ('checkout-' + checkout_mode)
        git('clone', '--no-checkout', '--no-hardlinks', str(destination), str(clone), cwd=tmp_path)
        git('config', 'core.autocrlf', checkout_mode, cwd=clone)
        git('checkout', '--force', 'HEAD', cwd=clone)
        assert {name: (clone / name).read_bytes() for name in expected} == expected
    # The source-only alias remains absent from every distributed tool list.
    public.code_asset()
    public.node_asset()
    with zipfile.ZipFile(output / f'Drama3-T8-Portable-{public.__version__}-code.zip') as archive:
        assert 'tools/quantize_converted.py' not in archive.namelist()
        assert 'tools/build_public_release.py' not in archive.namelist()
        assert 'tools/finalize_manifest.py' not in archive.namelist()
        assert not any(name.endswith(('build_launcher.py','StartDrama3.cs','.exe')) for name in archive.namelist())
        assert 'tools/convert_model.py' in archive.namelist()
        assert '.gitattributes' not in archive.namelist()
        assert all(archive.read(name) == expected['portable/' + name] for name in archive.namelist())
    with zipfile.ZipFile(output / f'ComfyUI-Drama3-T8-{public.__version__}.zip') as archive:
        assert not any(name.endswith('tools/quantize_converted.py') for name in archive.namelist())
        assert not any(name.endswith('tools/build_public_release.py') for name in archive.namelist())
        assert not any(name.endswith('tools/finalize_manifest.py') for name in archive.namelist())
        assert not any(name.endswith(('build_launcher.py','StartDrama3.cs','.exe')) for name in archive.namelist())
        assert 'Comfyui-Drama-3/.gitattributes' in archive.namelist()
        assert all(archive.read(name) == expected[name.split('/', 1)[1]] for name in archive.namelist())
    assert not any(path.suffix.lower() in {'.safetensors','.exe','.dll'} or path.name.casefold() == 'roadmap.md'
                   for path in destination.rglob('*') if path.is_file())


def test_native_asset_rebuilds_changed_contents_and_keeps_an_unchanged_release_pin(public_source):
    root, node, output = public_source
    library = put(root, 'runtime/r2t2-native/lib/drama_r2t2.dll', b'old')
    put(root, 'runtime/r2t2-native/lib/python.exe', b'must not publish')
    put(root, 'runtime/r2t2-native/lib/model.safetensors', b'must not publish')
    public.native_asset()
    descriptor = node / 'r2t2-runtime.json'
    previous = json.loads(descriptor.read_text(encoding='utf-8'))
    previous['url'] = previous['url'].replace('/v' + public.__version__ + '/', '/v0.1.3/')
    descriptor.write_text(json.dumps(previous), encoding='utf-8')
    public.native_asset()
    assert json.loads(descriptor.read_text(encoding='utf-8'))['url'] == previous['url']
    library.write_bytes(b'new')
    public.native_asset()
    with zipfile.ZipFile(output / 'Drama3-R2T2-native-win64-cu128.zip') as archive:
        assert archive.read('lib/drama_r2t2.dll') == b'new'
        assert not any(name.endswith(('.exe','.safetensors')) for name in archive.namelist())
    assert '/v' + public.__version__ + '/' in json.loads(descriptor.read_text(encoding='utf-8'))['url']


def test_interrupted_archive_write_retains_old_release_and_cleans_partial(tmp_path, monkeypatch):
    archive = put(tmp_path, 'old-release.zip', b'old release bytes')
    source = put(tmp_path, 'source.py')
    def fail(*args, **kwargs):
        raise OSError('archive interrupted')
    monkeypatch.setattr(public.zipfile.ZipFile, 'write', fail)
    with pytest.raises(OSError, match='interrupted'):
        public.write_archive(archive, {'source.py': source})
    assert archive.read_bytes() == b'old release bytes'
    assert not list(tmp_path.glob('*.part'))


def test_standalone_public_build_refreshes_nodes_without_a_publish_checkout(public_source, monkeypatch):
    root, node, output = public_source
    put(root, 'runtime/r2t2-native/lib/drama_r2t2.dll', b'fake native fixture')
    monkeypatch.setattr(sys, 'argv', ['build_public_release.py'])
    public.main()
    with zipfile.ZipFile(output / f'ComfyUI-Drama3-T8-{public.__version__}.zip') as archive:
        assert archive.read('Comfyui-Drama-3/web/seed-compat.js') == b'current web source'
        assert archive.read('Comfyui-Drama-3/drama3_t8/engine.py') == b'current engine source'
    assert (output / 'SHA256SUMS.txt').is_file()


def test_source_checkout_cannot_be_its_own_publish_destination(public_source):
    root, node, output = public_source
    (root / '.git').mkdir()
    with pytest.raises(ValueError, match='separate'):
        public.sync_source(root)
    assert (node / 'web/seed-compat.js').read_bytes() == b'stale web source'


@pytest.mark.parametrize('builder', [build_portable, public])
def test_source_walk_excludes_links_and_case_variants_of_local_research(tmp_path, monkeypatch, builder):
    source, destination = tmp_path / 'source', tmp_path / 'package'
    put(source, 'module.py')
    put(source, 'nested/RoadMap.MD', b'private research')
    outside = put(tmp_path, 'private.py', b'private outside source')
    link = source / 'linked.py'
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip('This filesystem does not permit symlink creation')
    if builder is build_portable:
        monkeypatch.setattr(builder, 'TARGET', destination)
        monkeypatch.setattr(builder, 'COPIED', set())
        builder.copy_tree(source, destination)
        names = set(builder.COPIED)
    else:
        names = {path.relative_to(source).as_posix() for path in builder.files(source)}
    assert names == {'module.py'}


def test_interrupted_full_zip_keeps_existing_archive(tmp_path, monkeypatch):
    # This fixture exercises archive/layout mechanics, not release qualification.
    monkeypatch.setattr(build_portable, 'validate_final_archive', lambda *args: None)
    target = tmp_path / 'package'
    monkeypatch.setattr(build_portable, 'TARGET', target)
    put(target, 'app.py')
    put(target, 'package-manifest.json', b'{}')
    manifest=compiled_launcher_manifest(target, {'version':build_portable.__version__,'files':['app.py']})
    archive = put(target.parent, f'Drama3-T8-Portable-{build_portable.__version__}-win64.zip', b'previous ZIP bytes')
    def fail(*args, **kwargs):
        raise OSError('zip interrupted')
    monkeypatch.setattr(build_portable.zipfile.ZipFile, 'write', fail)
    with pytest.raises(OSError, match='interrupted'):
        build_portable.write_archive(manifest)
    assert archive.read_bytes() == b'previous ZIP bytes'
    assert not list(target.parent.glob('*.part'))



def test_native_license_scope_keeps_existing_archive_when_only_nested_model_notices_are_added(public_source, monkeypatch):
    root, node, output = public_source
    put(root, 'runtime/r2t2-native/lib/drama_r2t2.dll', b'unchanged native DLL')
    put(root, 'licenses/LICENSE-R2T2', b'top-level R2T2 license')
    public.native_asset()
    archive = output / 'Drama3-R2T2-native-win64-cu128.zip'
    descriptor = node / 'r2t2-runtime.json'
    pinned = json.loads(descriptor.read_text(encoding='utf-8'))
    pinned['url'] = pinned['url'].replace('/v' + public.__version__ + '/', '/v0.1.11/')
    descriptor.write_text(json.dumps(pinned, indent=2), encoding='utf-8')
    original_archive = archive.read_bytes()
    original_mtime = archive.stat().st_mtime_ns
    original_descriptor = descriptor.read_bytes()
    # Two nested models use the same LICENSE/NOTICE basenames. They belong
    # in code/node packages at their own paths, never in this native asset.
    nested = {}
    for model in ('QWEN3-ASR-1.7B', 'another-model'):
        for name in ('LICENSE', 'NOTICE'):
            relative = f'licenses/{model}/{name}'
            nested[relative] = (model + ':' + name).encode('utf-8')
            put(root, relative, nested[relative])
    real_write = public.write_archive
    attempts = []

    def forbid_native_rebuild(path, members):
        if Path(path) == archive:
            attempts.append(path)
            raise AssertionError('Nested model licenses must not rebuild the native runtime')
        return real_write(path, members)

    monkeypatch.setattr(public, 'write_archive', forbid_native_rebuild)
    public.native_asset()
    assert attempts == []
    assert archive.read_bytes() == original_archive
    assert archive.stat().st_mtime_ns == original_mtime
    assert descriptor.read_bytes() == original_descriptor
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert set(handle.namelist()) == {
            'lib/drama_r2t2.dll', 'licenses/NOTICE-DRAMA3-EDITOR', 'licenses/LICENSE-R2T2'}
        assert 'licenses/LICENSE' not in handle.namelist()
        assert 'licenses/NOTICE' not in handle.namelist()
    assert all((root / relative).read_bytes() == content for relative, content in nested.items())


def test_native_license_scope_never_overwrites_a_real_top_level_license_with_a_nested_basename(public_source):
    root, node, output = public_source
    put(root, 'runtime/r2t2-native/lib/drama_r2t2.dll', b'unchanged native DLL')
    put(root, 'licenses/LICENSE', b'actual top-level native license')
    put(root, 'licenses/NOTICE', b'actual top-level native notice')
    for model in ('QWEN3-ASR-1.7B', 'another-model'):
        put(root, f'licenses/{model}/LICENSE', (model + ' model-only license').encode('utf-8'))
        put(root, f'licenses/{model}/NOTICE', (model + ' model-only notice').encode('utf-8'))
    public.native_asset()
    archive = output / 'Drama3-R2T2-native-win64-cu128.zip'
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert set(handle.namelist()) == {
            'lib/drama_r2t2.dll', 'licenses/NOTICE-DRAMA3-EDITOR', 'licenses/LICENSE', 'licenses/NOTICE'}
        assert handle.read('licenses/LICENSE') == b'actual top-level native license'
        assert handle.read('licenses/NOTICE') == b'actual top-level native notice'
    descriptor = json.loads((node / 'r2t2-runtime.json').read_text(encoding='utf-8'))
    assert descriptor['files']['licenses/LICENSE']['sha256'] == hashlib.sha256(b'actual top-level native license').hexdigest()
    assert descriptor['files']['licenses/NOTICE']['sha256'] == hashlib.sha256(b'actual top-level native notice').hexdigest()


def test_native_license_scope_keeps_nested_model_licenses_in_code_and_node_assets(public_source):
    root, node, output = public_source
    put(root, 'runtime/r2t2-native/lib/drama_r2t2.dll', b'unchanged native DLL')
    put(root, 'licenses/LICENSE-R2T2', b'top-level R2T2 license')
    nested = {}
    for model in ('QWEN3-ASR-1.7B', 'another-model'):
        for name in ('LICENSE', 'NOTICE'):
            relative = f'licenses/{model}/{name}'
            nested[relative] = (model + ':' + name).encode('utf-8')
            put(root, relative, nested[relative])
    public.native_asset()
    public.sync_node_source()
    public.code_asset()
    public.node_asset()
    for filename, prefix in (
        (f'Drama3-T8-Portable-{public.__version__}-code.zip', ''),
        (f'ComfyUI-Drama3-T8-{public.__version__}.zip', 'Comfyui-Drama-3/'),
    ):
        with zipfile.ZipFile(output / filename) as handle:
            assert handle.testzip() is None
            for relative, content in nested.items():
                assert handle.read(prefix + relative) == content
            assert prefix + 'licenses/LICENSE' not in handle.namelist()
            assert prefix + 'licenses/NOTICE' not in handle.namelist()
            assert handle.read(prefix + 'licenses/LICENSE-R2T2') == b'top-level R2T2 license'
    with zipfile.ZipFile(output / 'Drama3-R2T2-native-win64-cu128.zip') as handle:
        assert not any(name in handle.namelist() for name in nested)
        assert 'licenses/LICENSE' not in handle.namelist()
        assert 'licenses/NOTICE' not in handle.namelist()
