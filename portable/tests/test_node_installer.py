"""Exercise first install and rejection without a large download or inference."""
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'comfyui-drama3-T8/install.py'
if not SOURCE.is_file():
    SOURCE = ROOT.parent / 'install.py'
spec = importlib.util.spec_from_file_location('drama_test_installer', SOURCE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows native runtime installer')


def prepare(root, entries=None):
    entries = entries or {'lib/drama_r2t2.dll': b'test-native-library'}
    data = BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        for name, value in entries.items():
            z.writestr(name, value)
    payload = data.getvalue()
    record = lambda value: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    manifest = {'url': 'https://github.com/T8mars/Comfyui-Drama-3/releases/download/v0.1.7/runtime.zip',
                **record(payload), 'files': {n: record(v) for n, v in entries.items()}}
    (root / 'r2t2-runtime.json').write_text(json.dumps(manifest), encoding='utf-8')
    return payload, manifest


def test_native_install_and_verified_reuse(tmp_path, monkeypatch):
    payload, _ = prepare(tmp_path)
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: BytesIO(payload))
    installer.install(tmp_path)
    assert (tmp_path / 'r2t2-runtime/lib/drama_r2t2.dll').read_bytes() == b'test-native-library'
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: pytest.fail('Verified runtime downloaded again'))
    installer.install(tmp_path)


def test_native_bad_download_does_not_install(tmp_path, monkeypatch):
    payload, _ = prepare(tmp_path)
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: BytesIO(payload[:-1]))
    with pytest.raises(ValueError, match='integrity'):
        installer.install(tmp_path)
    assert not (tmp_path / 'r2t2-runtime').exists()
    assert not list(tmp_path.glob('.r2t2-install-*'))


def test_native_existing_modified_directory_is_retained(tmp_path, monkeypatch):
    prepare(tmp_path)
    target = tmp_path / 'r2t2-runtime/lib'
    target.mkdir(parents=True)
    (target / 'drama_r2t2.dll').write_bytes(b'user-build')
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: pytest.fail('Unexpected network request'))
    with pytest.raises(RuntimeError, match='Existing'):
        installer.install(tmp_path)
    assert (target / 'drama_r2t2.dll').read_bytes() == b'user-build'


def test_native_rejects_traversal_before_download(tmp_path, monkeypatch):
    prepare(tmp_path, {'../escape.dll': b'invalid'})
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: pytest.fail('Unexpected network request'))
    with pytest.raises(ValueError, match='Invalid'):
        installer.install(tmp_path)
    assert not (tmp_path / 'r2t2-runtime').exists()


def test_native_legacy_runtime_uses_existing_license_texts(tmp_path, monkeypatch):
    _, manifest = prepare(tmp_path, {'lib/drama_r2t2.dll': b'native', 'licenses/NOTICE': b'notice'})
    (tmp_path / 'r2t2-runtime/lib').mkdir(parents=True)
    (tmp_path / 'r2t2-runtime/lib/drama_r2t2.dll').write_bytes(b'native')
    (tmp_path / 'licenses').mkdir()
    (tmp_path / 'licenses/NOTICE').write_bytes(b'notice')
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: pytest.fail('Legacy runtime downloaded again'))
    installer.install(tmp_path)


def test_native_skip_does_not_require_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv('DRAMA_SKIP_R2T2_INSTALL', '1')
    installer.install(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('entries', [
    {'lib/Same.dll': b'a', 'lib/same.dll': b'b'},
    {'lib/escape.dll.': b'invalid'},
    {'lib/escape.dll ': b'invalid'},
    {'lib/CON.dll': b'invalid'},
    {'lib/safe.dll:stream': b'invalid'},
    {'models/main.gguf': b'invalid'},
])
def test_native_rejects_windows_aliases_and_unexpected_layout(tmp_path, monkeypatch, entries):
    prepare(tmp_path, entries)
    monkeypatch.setattr(installer, 'urlopen', lambda *a, **kw: pytest.fail('Unexpected network request'))
    with pytest.raises(ValueError):
        installer.install(tmp_path)
    assert not (tmp_path / 'r2t2-runtime').exists()
