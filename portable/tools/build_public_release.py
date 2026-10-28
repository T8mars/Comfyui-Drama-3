"""Build GitHub/Registry sources and code-only portable release assets.

Full model/Python environments are distributed separately by the author.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drama3_t8 import __version__

OUT = ROOT / 'release_packages/public'
NODE = ROOT / 'comfyui-drama3-T8'
SKIP = {'.git', '.cache', '__pycache__', '.pytest_cache', 'node_modules'}
NODE_ROOT_FILES = {'.gitignore', '.comfyignore', '__init__.py', 'install.py', 'LICENSE',
    'pyproject.toml', 'r2t2-runtime.json', 'README.md', 'README.zh-CN.md',
    'requirements.txt', 'upstream-lock.json'}
SOURCE_SUFFIXES = {'.py', '.json', '.txt', '.jinja', '.jinja2', '.dict', '.js', '.css',
    '.html', '.md', '.toml', '.yaml', '.yml', '.c', '.cpp', '.h', '.hpp', '.cu', '.cuh',
    '.svg', '.png', '.ico', '.webp', '.jpg', '.jpeg', '.gif', '.woff', '.woff2', '.ttf', '.cmake', '.mjs', '.cjs'}


def reparse(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def member_sha(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def files(root):
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d.casefold() not in SKIP and not reparse(Path(directory) / d)]
        for name in sorted(names):
            p = Path(directory) / name
            if not reparse(p) and p.suffix.lower() not in {'.pyc', '.pyo'} and p.name.casefold() != 'roadmap.md':
                yield p


def node_files(include_github=False):
    """Only node source, workflows and notices may enter GitHub code assets."""
    for path in files(NODE):
        relative = path.relative_to(NODE)
        if len(relative.parts) == 1:
            allowed = relative.name in NODE_ROOT_FILES
        elif relative.parts[0] == '.github':
            allowed = include_github and path.suffix.lower() in {'.yml', '.yaml', '.json', '.md'}
        elif relative.parts[0] in {'drama3_t8', 'web', 'workflows', 'licenses', 'native-source'}:
            allowed = path.suffix.lower() in SOURCE_SUFFIXES or path.name in {'LICENSE', 'COPYING'}
            if relative.parts[0] == 'licenses':
                from drama3_t8.updater import is_update_path
                allowed = is_update_path(relative.as_posix())
            if relative.parts[0] == 'workflows':
                allowed = path.suffix.lower() == '.json' and not path.name.lower().endswith('.api.json')
            if relative.parts[:2] in {('drama3_t8', name) for name in ('studio.py', 'studio_editor.py', 'bootstrap.py', 'updater.py')}:
                allowed = False
        else:
            allowed = False
        if allowed:
            yield path


def write_archive(archive, members):
    """Replace only a completely written ZIP; retain previous release artifacts on failure."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix='.' + archive.name + '.', suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, path in sorted(members.items()):
                z.write(path, name)
        os.replace(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and os.path.samefile(src, dst):
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dst.parent, prefix='.' + dst.name + '.', suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def native_asset():
    from drama3_t8.updater import is_update_path
    source = ROOT / 'runtime/r2t2-native'
    archive = OUT / 'Drama3-R2T2-native-win64-cu128.zip'
    paths = {p.relative_to(source).as_posix(): p for p in sorted(files(source))
        if p.suffix.lower() == '.dll' and p.relative_to(source).parts[0] in {'lib', 'backends'}}
    if 'lib/drama_r2t2.dll' not in paths:
        raise ValueError('Native runtime asset is missing lib/drama_r2t2.dll')
    paths.update({f'licenses/{p.name}': p for p in files(ROOT / 'licenses')
        if p.parent == ROOT / 'licenses' and is_update_path(f'licenses/{p.name}')})
    rebuild = True
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as z:
                rebuild = set(z.namelist()) != set(paths) or z.testzip() is not None
                if not rebuild:
                    rebuild = any(member_sha(z, name) != sha(path) for name, path in paths.items())
        except zipfile.BadZipFile:
            rebuild = True
    if rebuild:
        write_archive(archive, paths)
    # Check the member list and CRC even when a previously built asset is reused.
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(paths)
        assert z.testzip() is None
        for name, p in paths.items():
            assert member_sha(z, name) == sha(p), f'Stale native asset: {name}'
    data = {'version': '1', 'url': f'https://github.com/T8mars/Comfyui-Drama-3/releases/download/v{__version__}/{archive.name}',
            **record(archive), 'files': {name: record(p) for name, p in sorted(paths.items())}}
    previous_path = NODE / 'r2t2-runtime.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if (previous.get('sha256') == data['sha256'] and previous.get('files') == data['files']
                and previous.get('url', '').startswith('https://github.com/T8mars/Comfyui-Drama-3/releases/download/')):
            # A code fix must not make a verified native dependency point at an
            # unpublished tag. Keep its original immutable release pin.
            data['url'] = previous['url']
    (NODE / 'r2t2-runtime.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'NATIVE_ASSET {archive} ({archive.stat().st_size:,} bytes)', flush=True)


def validate_canvas_workflows(folder):
    examples = sorted(folder.glob('*.json'))
    if not examples:
        raise ValueError(f'No canvas workflows in {folder}')
    for path in examples:
        data = json.loads(path.read_text(encoding='utf-8'))
        if path.name.lower().endswith('.api.json') or not isinstance(data, dict) or not isinstance(data.get('nodes'), list) or not isinstance(data.get('links'), list):
            raise ValueError(f'Only canvas workflows may be published: {path}')
    return examples


def portable_files():
    from drama3_t8.updater import is_update_path
    candidates = [ROOT / p for p in ['app.py', 'Start-Drama3.cmd', 'Update-Drama3.cmd',
                  'README.md', 'README.zh-CN.md', 'LICENSE', 'requirements.txt', 'upstream-lock.json',
                  'acceleration-manifest.json', 'acceleration-requirements.txt', 'acceptance-report.json']]
    for name in ['drama3_t8', 'web', 'tools', 'vendor/ComfyUI', 'licenses']:
        candidates.extend(files(ROOT / name))
    result = {}
    for path in candidates:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file() and is_update_path(relative):
            result[relative] = path
    assert all(name in result for name in ['app.py', 'Start-Drama3.cmd', 'Update-Drama3.cmd',
                'drama3_t8/updater.py', 'tools/update_portable.py', 'drama3_t8/native/fish/tokenizer.json',
                'vendor/ComfyUI/comfy/model_management.py',
                'vendor/ComfyUI/comfy_api/input/__init__.py',
                'vendor/ComfyUI/comfy/ldm/models/autoencoder.py'])
    assert not any(p.lower().endswith(('.safetensors', '.gguf', '.onnx', '.pth', '.dll', '.pyd', '.exe'))
                   or p.startswith(('models/', 'runtime/', 'userdata/', 'outputs/')) for p in result)
    return dict(sorted(result.items()))


def code_asset():
    members = portable_files()
    archive = OUT / f'Drama3-T8-Portable-{__version__}-code.zip'
    write_archive(archive, members)
    data = {'schema': 1, 'version': __version__, 'min_version': '0.1.3',
            'asset': {'name': archive.name, **record(archive)},
            'files': {name: record(path) for name, path in members.items()}}
    (OUT / 'portable-update.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'CODE_ASSET {archive} ({len(members)} files; {archive.stat().st_size:,} bytes)', flush=True)


def sync_node_source():
    """Refresh independent node sources for every build, even without --source."""
    validate_canvas_workflows(NODE / 'workflows')
    # Shared inference source is copied into the node independently of the studio.
    for p in files(ROOT / 'drama3_t8'):
        if p.name in {'studio.py', 'studio_editor.py', 'bootstrap.py', 'updater.py'} or p.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        copy(p, NODE / 'drama3_t8' / p.relative_to(ROOT / 'drama3_t8'))
    for name in ['editor_core.js','drama3_editor.js','seed-compat.js']:
        copy(ROOT / 'web' / name, NODE / 'web' / name)
    for name in ['licenses']:
        for p in files(ROOT / name):
            copy(p, NODE / name / p.relative_to(ROOT / name))
    for name in ['LICENSE', 'README.md', 'README.zh-CN.md', 'upstream-lock.json']:
        copy(ROOT / name, NODE / name)
    for p in files(ROOT / 'vendor/r2t2_native'):
        if p.suffix.lower() in SOURCE_SUFFIXES or p.name in {'LICENSE', 'COPYING'}:
            copy(p, NODE / 'native-source' / p.relative_to(ROOT / 'vendor/r2t2_native'))


def sync_source(destination):
    destination = Path(destination).resolve()
    if not (destination / '.git').is_dir():
        raise ValueError('Destination must be a pre-cloned Git checkout')
    protected = ('drama3_t8', 'web', 'tools', 'licenses', 'vendor', 'runtime', 'models', 'userdata', 'outputs')
    if (destination == ROOT.resolve() or destination in ROOT.resolve().parents
            or destination == NODE.resolve() or destination.is_relative_to(NODE.resolve())
            or any(destination == ROOT/name or destination.is_relative_to(ROOT/name) for name in protected)):
        raise ValueError('The publish checkout must be separate from source and node staging directories')
    sync_node_source()
    for p in node_files(include_github=True):
        relative = p.relative_to(NODE)
        copy(p, destination / relative)
    for obsolete in (destination / 'workflows').glob('*.api.json'):
        obsolete.unlink()
    validate_canvas_workflows(destination / 'workflows')
    copy(ROOT / 'tools/make_workflows.py', destination / 'tools/make_workflows.py')
    for name, p in portable_files().items():
        copy(p, destination / 'portable' / name)
    for p in files(ROOT / 'tests'):
        if p.suffix.lower() in SOURCE_SUFFIXES:
            copy(p, destination / 'portable/tests' / p.relative_to(ROOT / 'tests'))
    # Development-only helpers are available in the source checkout. They do
    # not enter full environments or the legacy-compatible code-update ZIP.
    for name in ['build_portable.py', 'build_public_release.py', 'quantize_converted.py']:
        copy(ROOT / 'tools' / name, destination / 'portable/tools' / name)
    # Native library source retains complete notices and build instructions.
    for p in files(ROOT / 'vendor/r2t2_native'):
        if p.suffix.lower() in SOURCE_SUFFIXES or p.name in {'LICENSE', 'COPYING'}:
            copy(p, destination / 'native-source' / p.relative_to(ROOT / 'vendor/r2t2_native'))
    print(f'PUBLISH_SOURCE {destination}', flush=True)


def node_asset():
    validate_canvas_workflows(NODE / 'workflows')
    archive = OUT / f'ComfyUI-Drama3-T8-{__version__}.zip'
    write_archive(archive, {(Path('Comfyui-Drama-3') / p.relative_to(NODE)).as_posix(): p for p in node_files()})
    print(f'NODE_ASSET {archive} ({archive.stat().st_size:,} bytes)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    native_asset()
    if args.native_only:
        return
    if args.source:
        sync_source(args.source)
    else:
        sync_node_source()
    code_asset()
    node_asset()
    assets = [OUT / f'ComfyUI-Drama3-T8-{__version__}.zip',
              OUT / f'Drama3-T8-Portable-{__version__}-code.zip', OUT / 'portable-update.json']
    native = json.loads((NODE / 'r2t2-runtime.json').read_text(encoding='utf-8'))
    if f'/v{__version__}/' in native['url']:
        assets.append(OUT / 'Drama3-R2T2-native-win64-cu128.zip')
    assets.sort()
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in assets), encoding='utf-8')


if __name__ == '__main__':
    main()
