"""Keep code-only updates installable by the actual published v0.1.11 updater."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from drama3_t8 import updater
from tools import build_public_release


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / 'fixtures/public-v0.1.11-updater'


def published_updater():
    source = (FIXTURE / 'updater.py').read_bytes()
    provenance = json.loads((FIXTURE / 'SOURCE.json').read_text('utf-8'))
    assert hashlib.sha256(source).hexdigest() == provenance['member_sha256']
    namespace = {'__name__': 'published_v0111_updater'}
    exec(compile(source, str(FIXTURE / 'updater.py'), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def converter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    return importlib.import_module('convert_model')


def test_every_current_code_asset_member_is_accepted_by_exact_public_v0111():
    old = published_updater()
    files = build_public_release.portable_files()
    manifest = dict(schema=1, version='0.2.0', min_version='0.1.3',
        asset=dict(name='Drama3-T8-Portable-0.2.0-code.zip', bytes=1, sha256='0' * 64),
        files={name: dict(bytes=path.stat().st_size, sha256='0' * 64)
               for name, path in files.items()})
    old.validate_manifest(manifest, '0.1.11')
    assert {name.removeprefix('tools/') for name in files if name.startswith('tools/')} == old.TOOLS
    assert updater.TOOLS == old.TOOLS
    assert 'tools/quantize_converted.py' not in files


def test_actual_public_v0111_installer_stages_all_current_asset_paths_and_preserves_data(tmp_path):
    """Real old validation/staging/transaction; small text payloads, no fake version overlay."""
    from drama3_t8.workspace_lock import WorkspaceLock
    old = published_updater()
    root = tmp_path / 'old portable'
    protected = {
        'package-manifest.json': b'{"version":"0.1.11"}',
        'runtime/python/python.exe': b'test-only interpreter placeholder',
        'models/user.safetensors': b'test-only protected model placeholder',
        'userdata/custom-paths.json': b'{"model":"E:/external/model.safetensors"}',
        'outputs/prior.wav': b'test-only protected audio placeholder',
    }
    for name, data in {**protected, 'app.py': b'# actual old entry',
                       'drama3_t8/__init__.py': b'__version__ = "0.1.11"\n'}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    owner = WorkspaceLock(root / 'userdata/studio.lock')
    owner.close()
    payloads = {name: b'# test current member\n' for name in build_public_release.portable_files()}
    payloads['drama3_t8/__init__.py'] = b'__version__ = "0.2.0"\n'
    archive = tmp_path / 'Drama3-T8-Portable-0.2.0-code.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in payloads.items():
            output.writestr(name, data)
    manifest = dict(schema=1, version='0.2.0', min_version='0.1.3',
        asset=dict(name=archive.name, bytes=archive.stat().st_size,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
        files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
               for name, data in payloads.items()})
    result = old.install_archive(root, archive, manifest, progress=lambda *_: None)
    assert result == {'status': 'updated', 'version': '0.2.0'}
    assert old._read_version(root) == '0.2.0'
    for name, data in payloads.items():
        assert (root / name).read_bytes() == data
    for name, data in protected.items():
        assert (root / name).read_bytes() == data
    transaction = json.loads((root / '.updates/transaction.json').read_text('utf-8'))
    assert transaction['state'] == 'committed'
    assert transaction['from_version'] == '0.1.11' and transaction['to_version'] == '0.2.0'


def test_quantize_cli_routes_without_loading_a_checkpoint(tmp_path, monkeypatch):
    module = converter(monkeypatch)
    calls = []
    monkeypatch.setattr(module, 'convert_quantized', lambda args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', ['convert_model.py', 'quantize', str(tmp_path / 'source'),
        str(tmp_path / 'out'), '--format', 'w6a8', '--device', 'cpu'])
    module.main()
    assert len(calls) == 1
    assert calls[0].src == tmp_path / 'source' and calls[0].out == tmp_path / 'out'
    assert calls[0].format == 'w6a8' and calls[0].device == 'cpu'


def test_original_source_wrapper_forwards_to_the_same_cli(tmp_path, monkeypatch):
    module = converter(monkeypatch)
    calls = []
    monkeypatch.setattr(module, 'convert_quantized', lambda args: calls.append(args))
    wrapper = importlib.import_module('quantize_converted')
    monkeypatch.setattr(sys, 'argv', ['quantize_converted.py', str(tmp_path / 'source'),
        str(tmp_path / 'out'), '--format', 'int8_convrot', '--device', 'cpu'])
    wrapper.main()
    assert len(calls) == 1 and calls[0].kind == 'quantize'
    assert calls[0].format == 'int8_convrot'


@pytest.mark.parametrize('extra', [[], ['--format', 'fp8'], ['--format', 'unknown']])
def test_quantize_requires_an_explicit_supported_format_before_filesystem_mutation(tmp_path, monkeypatch, extra):
    module = converter(monkeypatch)
    target = tmp_path / 'new-folder/out.safetensors'
    monkeypatch.setattr(sys, 'argv', ['convert_model.py', 'quantize', str(tmp_path / 'source'), str(target), *extra])
    with pytest.raises(SystemExit):
        module.main()
    assert not target.parent.exists()


@pytest.mark.parametrize('which', ['same-path', 'existing-output', 'wrong-format', 'no-banks'])
def test_quantize_rejects_invalid_source_or_output_without_overwriting(tmp_path, monkeypatch, which):
    module = converter(monkeypatch)
    source, target = tmp_path / 'source.safetensors', tmp_path / 'out.safetensors'
    save_file({'dense.weight': torch.arange(8).to(torch.bfloat16)}, str(source),
              metadata={'drama3_format': 'w4a8' if which == 'wrong-format' else 'fp8'})
    before = source.read_bytes()
    if which == 'same-path':
        target = source
    if which == 'existing-output':
        target.write_bytes(b'original user output')
    monkeypatch.setattr(sys, 'argv', ['convert_model.py', 'quantize', str(source), str(target),
        '--format', 'w4a8', '--device', 'cpu'])
    error = FileExistsError if which == 'existing-output' else ValueError
    with pytest.raises(error):
        module.main()
    assert source.read_bytes() == before
    if which == 'existing-output':
        assert target.read_bytes() == b'original user output'
    elif target != source:
        assert not target.exists()
    assert not target.with_suffix(target.suffix + '.part').exists()


@pytest.mark.parametrize('fmt', ['int8_convrot', 'w6a8', 'w4a8'])
def test_real_cpu_tiny_fp8_quantization_preserves_dense_and_controls(tmp_path, monkeypatch, fmt):
    """Two real FP8 banks and original metadata go through the streaming writer."""
    import comfy_kitchen
    module = converter(monkeypatch)
    generator = torch.Generator().manual_seed(606)
    values = {
        'dense.weight': torch.randn(12, 16, generator=generator).to(torch.bfloat16),
        'audio_head.weight': torch.randn(5, 16, generator=generator).to(torch.bfloat16),
    }
    for bank in ('gate_up', 'down'):
        prefix = 'model.layers.0.mlp.experts.' + bank
        values[prefix + '_weight'] = torch.randn(2, 256, 256, generator=generator).to(torch.float8_e4m3fn)
        values[prefix + '_scale'] = torch.full((2, 2, 2), .03125)
    source, target = tmp_path / 'fp8.safetensors', tmp_path / (fmt + '.safetensors')
    config = json.dumps({'drama3_output_token_ids': [248046, 248099, 252194, 252196, 252195]})
    metadata = dict(drama3_format='fp8', drama3_dense='bf16', drama3_config=config,
                    source_revision='fixed-source-proof')
    save_file(values, str(source), metadata=metadata)
    source_before = source.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['convert_model.py', 'quantize', str(source), str(target),
        '--format', fmt, '--device', 'cpu'])
    with comfy_kitchen.use_backend('eager'):
        module.main()
    assert source.read_bytes() == source_before
    assert not target.with_suffix('.safetensors.part').exists()
    with safe_open(str(target), framework='pt', device='cpu') as archive:
        actual_metadata = archive.metadata()
        assert actual_metadata == {**metadata, 'drama3_format': fmt, 'quantized_from': source.name}
        for name in ('dense.weight', 'audio_head.weight'):
            assert torch.equal(archive.get_tensor(name), values[name])
        original_fp8_keys = {name for name in values if '.experts.' in name}
        assert not original_fp8_keys.intersection(archive.keys())
        for bank in ('gate_up_proj', 'down_proj'):
            prefix = 'model.layers.0.mlp.experts.' + bank
            qdata = archive.get_tensor(prefix + '.weight')
            assert qdata.dtype == torch.int8 and qdata.shape[0] == 2
            conf = json.loads(bytes(archive.get_tensor(prefix + '.comfy_quant').tolist()))
            assert conf['num_experts'] == 2
            assert conf['format'] == {'int8_convrot': 'int8_tensorwise', 'w6a8': 'w6a8_int8',
                                       'w4a8': 'asym_w4a8_int8'}[fmt]
            if fmt != 'int8_convrot':
                assert archive.get_tensor(prefix + '.weight_s_channel').shape[0] == 2
                assert archive.get_tensor(prefix + '.weight_s_rel').shape[0] == 2
    report = json.loads(target.with_suffix('.conversion.json').read_text('utf-8'))
    assert report['banks'] == 2 and report['format'] == fmt
    assert report['bytes'] == target.stat().st_size
