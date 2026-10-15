"""Install the optional same-process Windows R2T2 runtime; never download models."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install(root=ROOT):
    root = Path(root).resolve()
    if os.environ.get('DRAMA_SKIP_R2T2_INSTALL') == '1':
        print('R2T2 native runtime skipped. Set DRAMA_R2T2_NATIVE_DIR to use an existing runtime.')
        return
    if os.name != 'nt' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        print('R2T2 bundled native runtime requires Windows x64 / CUDA 12.8. Qwen ASR remains available.')
        return
    manifest = json.loads((root / 'r2t2-runtime.json').read_text(encoding='utf-8'))
    target = root / 'r2t2-runtime'
    def safe(path):
        p = PurePosixPath(path)
        if (not path or '\\' in path or ':' in path or p.is_absolute() or p.as_posix() != path
                or len(p.parts) < 2 or p.parts[0] not in {'lib', 'backends', 'licenses'}
                or any(part in {'.', '..'} or part.endswith(('.', ' '))
                       or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                       or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])', part.split('.')[0])
                       for part in p.parts)):
            raise ValueError(f'Invalid native runtime member: {path}')
        return p
    folded = set()
    for name in manifest['files']:
        safe(name)
        if name.casefold() in folded:
            raise ValueError('Native runtime has duplicate Windows paths')
        folded.add(name.casefold())
    def existing_file(name):
        # Older verified runtime bundles keep license texts at the node root.
        if name.startswith('licenses/') and not (target / name).exists():
            return root / name
        return target / name
    if target.is_dir() and not target.is_symlink() and all(
        existing_file(n).is_file() and existing_file(n).stat().st_size == r['bytes'] and
        digest(existing_file(n)) == r['sha256'] for n, r in manifest['files'].items()
    ):
        print('R2T2 native runtime verified.')
        return
    if target.exists():
        raise RuntimeError('Existing r2t2-runtime is incomplete or modified. Move it aside before reinstalling.')
    url = manifest['url']
    if not url.startswith('https://github.com/T8mars/Comfyui-Drama-3/releases/download/'):
        raise ValueError('Unexpected runtime download source')
    print('Downloading R2T2 CUDA native runtime (no Python or model weights)...', flush=True)
    with tempfile.TemporaryDirectory(prefix='.r2t2-install-', dir=root) as temp:
        temp = Path(temp)
        archive = temp / 'runtime.zip'
        with urlopen(Request(url, headers={'User-Agent': 'Drama3-T8-installer'}), timeout=30) as response, archive.open('wb') as out:
            total = 0
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                total += len(block)
                if total > manifest['bytes']:
                    raise ValueError('Runtime archive exceeded declared size')
                out.write(block)
        if total != manifest['bytes'] or digest(archive) != manifest['sha256']:
            raise ValueError('Runtime archive integrity check failed; installation unchanged')
        unpacked = temp / 'unpacked'
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len({n.casefold() for n in z.namelist()}) or set(z.namelist()) != set(manifest['files']):
                raise ValueError('Runtime archive file list mismatch')
            for member in z.infolist():
                safe(member.filename)
                record = manifest['files'][member.filename]
                if stat.S_ISLNK(member.external_attr >> 16) or member.file_size != record['bytes']:
                    raise ValueError('Invalid runtime archive entry')
                path = unpacked / member.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                if digest(path) != record['sha256']:
                    raise ValueError(f'Runtime file integrity check failed: {member.filename}')
        os.replace(unpacked, target)
    print('R2T2 native runtime installed. NVIDIA driver supporting CUDA 12.8 is required.')


if __name__ == '__main__':
    install()
