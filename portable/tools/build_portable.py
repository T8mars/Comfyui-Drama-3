"""Create a self-contained portable layout from an explicit distribution allowlist."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drama3_t8 import __version__
TARGET=ROOT/'release_packages/Drama3-T8-Portable'
SKIP={'.git','.cache','__pycache__','.pytest_cache','node_modules'}
COPIED=set()


def copy_file(source,target):
    COPIED.add(target.relative_to(TARGET).as_posix())
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and os.path.samefile(source,target):
        return
    if target.exists() and target.stat().st_size==source.stat().st_size:
        # A replacement DLL or model can have the same length as its previous build.
        with source.open('rb') as left,target.open('rb') as right:
            while True:
                block=left.read(4*1024*1024)
                if block!=right.read(4*1024*1024):
                    break
                if not block:
                    return
    if source.stat().st_size>4*1024*1024 and not target.exists():
        try:
            os.link(source,target)
            return
        except OSError:
            pass
    shutil.copy2(source,target)


def copy_tree(source,target,extra_skip=()):
    for folder,dirs,files in os.walk(source):
        dirs[:]=[d for d in dirs if d not in SKIP and d not in extra_skip]
        for name in files:
            if name=='roadmap.md' or name.endswith(('.pyc','.pyo')):
                continue
            path=Path(folder)/name
            copy_file(path,target/path.relative_to(source))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',action='store_true')
    parser.add_argument('--hash-models',action='store_true')
    parser.add_argument('--zip-only',action='store_true')
    args=parser.parse_args()
    if args.zip_only:
        manifest=json.loads((TARGET/'package-manifest.json').read_text(encoding='utf-8'))
        archive=TARGET.parent/f'Drama3-T8-Portable-{__version__}-win64.zip'
        with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
            for name in sorted([*manifest['files'],'package-manifest.json']):
                path=TARGET/name
                stored=path.suffix in {'.safetensors','.gguf','.pth','.pt','.dll','.pyd','.lib','.exe'}
                z.write(path,path.relative_to(TARGET.parent),compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        print(f'PORTABLE ZIP: {archive} ({archive.stat().st_size} bytes)',flush=True)
        return
    TARGET.mkdir(parents=True,exist_ok=True)
    for name in ['app.py','README.md','README.zh-CN.md','LICENSE','Start-Drama3.cmd','Update-Drama3.cmd','requirements.txt','upstream-lock.json','acceleration-manifest.json','acceleration-requirements.txt','acceptance-report.json']:
        copy_file(ROOT/name,TARGET/name)
    for name in ['drama3_t8','web','licenses','runtime']:
        copy_tree(ROOT/name,TARGET/name)
    copy_tree(ROOT/'vendor/ComfyUI',TARGET/'vendor/ComfyUI',extra_skip=('custom_nodes','models','outputs','output','userdata','user','input','temp'))
    copy_tree(ROOT/'vendor/r2t2_native',TARGET/'vendor/r2t2_native')
    (TARGET/'vendor/ComfyUI/custom_nodes').mkdir(parents=True,exist_ok=True)
    copy_tree(ROOT/'comfyui-drama3-T8',TARGET/'comfyui-drama3-T8')
    for name in ['check_runtime.py','download_models.py','convert_model.py','safetensors_stream.py','extract_text_head.py','update_portable.py']:
        copy_file(ROOT/'tools'/name,TARGET/'tools'/name)
    models=['drama3-t8-fp8.safetensors','drama3-codec.safetensors','drama3-text-head.safetensors','qwen3-asr-06b.safetensors']
    records=[]
    for name in models:
        source=ROOT/'models'/name
        copy_file(source,TARGET/'models'/name)
        record={'path':f'models/{name}','bytes':source.stat().st_size}
        if args.hash_models:
            with source.open('rb') as handle:
                digest=hashlib.sha256()
                for block in iter(lambda:handle.read(32*1024*1024),b''):digest.update(block)
            record['sha256']=digest.hexdigest()
        records.append(record)
    copy_tree(ROOT/'models/qwen-aligner',TARGET/'models/qwen-aligner')
    for name in ['Confucius4-R2T2-GGUF','FireRedVAD-ONNX']:
        copy_tree(ROOT/'models'/name,TARGET/'models'/name)
        for path in sorted((ROOT/'models'/name).iterdir()):
            if path.is_file():
                record={'path':f'models/{name}/{path.name}','bytes':path.stat().st_size}
                if args.hash_models:
                    with path.open('rb') as handle:
                        digest=hashlib.sha256()
                        for block in iter(lambda:handle.read(32*1024*1024),b''):digest.update(block)
                    record['sha256']=digest.hexdigest()
                records.append(record)
    copied=sorted(COPIED)
    assert not any('roadmap.md' in p or '.tmpaEQMC2' in p or '.upstream/' in p for p in copied)
    installed={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()}
    manifest={'version':__version__,'models':records,'model_source':json.loads((ROOT/'upstream-lock.json').read_text()),
              'runtime_packages':installed,'files':copied,'excluded':['roadmap.md','official source weights','failed conversion temp','userdata','outputs','development validation']}
    (TARGET/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PORTABLE LAYOUT: {TARGET}, {len(copied)} files',flush=True)
    if args.zip:
        archive=TARGET.parent/f'Drama3-T8-Portable-{__version__}-win64.zip'
        with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
            for name in sorted([*copied,'package-manifest.json']):
                path=TARGET/name
                if path.is_file():
                    stored=path.suffix in {'.safetensors','.gguf','.pth','.pt','.dll','.pyd','.lib','.exe'}
                    z.write(path,path.relative_to(TARGET.parent),compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        print(f'PORTABLE ZIP: {archive} ({archive.stat().st_size} bytes)',flush=True)


if __name__=='__main__':main()
