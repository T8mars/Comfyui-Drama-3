"""Validate reviewed, source-bound release acceptance before recording it.

No benchmarks or pass statuses are invented. Dedicated build/update commands
handle packages; this command only verifies and records the reviewed evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drama3_t8 import __version__
GATES=frozenset({'cpu_regressions','native_numerics','real_model_checks',
    'canvas_old_host','canvas_new_host','studio_ui','portable_update','requirements_audit'})

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def validate(report,root=ROOT,version=__version__):
    require(isinstance(report,dict),'Acceptance must be an object')
    json.dumps(report,allow_nan=False)
    require(report.get('schema')=='drama3.release-acceptance.v1','A reviewed release-acceptance schema is required')
    require(report.get('version')==version,'Acceptance version differs from current source')
    require(report.get('status')=='passed' and report.get('validation_scope')=='current_release',
        'Only completed current-release acceptance can be recorded')
    sources=report.get('source_sha256')
    require(isinstance(sources,dict) and sources,'Source hashes are required')
    required={'app.py','README.md','README.zh-CN.md','LICENSE','Start-Drama3.cmd','Update-Drama3.cmd',
        'requirements.txt','upstream-lock.json','acceleration-manifest.json','acceleration-requirements.txt'}
    required.update('tools/'+name for name in ('check_runtime.py','download_models.py','convert_model.py',
        'safetensors_stream.py','extract_text_head.py','update_portable.py'))
    portable_required=set(required)
    required.update(p.relative_to(root).as_posix() for p in (root/'drama3_t8').rglob('*.py') if '__pycache__' not in p.parts)
    required.update(p.relative_to(root).as_posix() for p in (root/'web').rglob('*')
        if p.is_file() and p.suffix.lower() in {'.js','.css','.html'})
    require(portable_required<=set(sources),'Acceptance does not cover portable launch/update sources')
    require(required<=set(sources),'Acceptance does not cover current inference, node, Studio and web sources')
    for name,digest in sources.items():
        require(isinstance(name,str) and isinstance(digest,str),'Invalid source hash entry')
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'Source must be a workspace-relative path')
        path=(root/relative).resolve(strict=True)
        require(path.is_relative_to(root.resolve()) and path.is_file(),'Source path leaves the workspace')
        require(path!=(root/'acceptance-report.json').resolve(),'Acceptance output cannot bind itself as a source')
        require(sha(path)==digest,'Source changed since acceptance: '+name)
    gates=report.get('gates')
    require(isinstance(gates,dict) and GATES<=set(gates),'Required acceptance gates are missing')
    for name in GATES:
        gate=gates[name]
        require(isinstance(gate,dict) and gate.get('status')=='passed',name+' has not passed')
        require(isinstance(gate.get('evidence'),list) and gate['evidence'],name+' has no evidence')
        if name in {'canvas_old_host','canvas_new_host','studio_ui'}:
            require(gate.get('method')=='visible_ui',name+' requires actual visible UI operation')
        for item in gate['evidence']:
            require(isinstance(item,dict) and isinstance(item.get('path'),str)
                and isinstance(item.get('sha256'),str),'Invalid evidence entry for '+name)
            path=Path(item['path'])
            if not path.is_absolute():path=root/path
            require(path.resolve()!=(root/'acceptance-report.json').resolve(),'Acceptance output cannot be its own evidence')
            require(path.is_file() and sha(path)==item['sha256'],'Missing or changed evidence for '+name)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,help='Reviewed current-release JSON with source and gate evidence hashes')
    parser.add_argument('--write',action='store_true',help='Atomically record acceptance-report.json after validation')
    args=parser.parse_args()
    try:
        report=validate(json.loads(args.report.read_text(encoding='utf-8')))
    except (ValueError,TypeError,OSError) as exc:
        parser.exit(1,f'Acceptance not recorded: {exc}\n')
    if args.write:
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=ROOT,
                    prefix='.acceptance-',suffix='.json',delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.write('\n')
            os.replace(temporary,ROOT/'acceptance-report.json')
        except (ValueError,TypeError,OSError) as exc:
            parser.exit(1,f'Acceptance not recorded: {exc}\n')
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    print('Current acceptance validated'+(' and recorded' if args.write else '; no files changed'))
    return 0

if __name__=='__main__':raise SystemExit(main())
