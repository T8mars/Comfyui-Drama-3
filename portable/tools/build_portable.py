"""Create a self-contained portable layout from an explicit distribution allowlist."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from drama3_t8 import __version__
TARGET=ROOT/'release_packages/Drama3-T8-Portable'
SKIP={'.git','.cache','__pycache__','.pytest_cache','node_modules'}
COPIED=set()
_WINDOWS=os.name=='nt'


def reparse(path):
    info=path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)


def validate_target(target):
    target=Path(target).resolve()
    root=ROOT.resolve()
    sources=('runtime','models','userdata','outputs','vendor','drama3_t8','web','licenses','tools','comfyui-drama3-T8')
    if target==root or target in root.parents or any(target==root/name or root/name in target.parents for name in sources):
        raise ValueError('The package destination must be separate from source/runtime/model/user folders')
    return target


def extra_models(sources,base_names):
    seen={name.casefold() for name in base_names}
    result=[]
    for source in sources:
        source=Path(source).resolve()
        if not source.is_file() or source.suffix.lower()!='.safetensors':
            raise ValueError(f'Extra model must be an existing .safetensors file: {source}')
        if source.name.casefold() in seen:
            raise ValueError(f'Extra model duplicates a model filename: {source.name}')
        seen.add(source.name.casefold())
        result.append(source)
    return result


def replace_file(source,target):
    """Keep atomic commits while tolerating bounded Windows sharing delays."""
    for attempt in range(6):
        try:
            os.replace(source,target)
            return
        except OSError as error:
            if not _WINDOWS or getattr(error,'winerror',None) not in {5,32,33} or attempt==5:
                raise
            time.sleep(.05*(attempt+1))


def copy_file(source,target):
    if not target.resolve().is_relative_to(TARGET.resolve()):
        raise ValueError(f'Package file escapes the destination: {target}')
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
    # Existing files may be hard links to a previous source or package. Replace
    # the directory entry atomically instead of modifying that shared inode.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.'+target.name+'.',suffix='.part',delete=False) as handle:
            temporary=Path(handle.name)
        shutil.copy2(source,temporary)
        replace_file(temporary,target)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def copy_tree(source,target,extra_skip=(),exclude_files=(),root_skip=()):
    for folder,dirs,files in os.walk(source):
        excluded={name.casefold() for name in (*SKIP,*extra_skip)}
        if Path(folder)==source:
            excluded.update(name.casefold() for name in root_skip)
        dirs[:]=[d for d in dirs if d.casefold() not in excluded and not reparse(Path(folder)/d)]
        for name in files:
            if name.casefold()=='roadmap.md' or name in exclude_files or name.lower().endswith(('.pyc','.pyo')):
                continue
            path=Path(folder)/name
            if reparse(path):continue
            copy_file(path,target/path.relative_to(source))


def archive_members(manifest):
    if manifest.get('version')!=__version__:
        raise ValueError('Refresh the portable directory before building this version of its ZIP')
    names=manifest.get('files')
    if not isinstance(names,list) or any(not isinstance(name,str) for name in names):
        raise ValueError('Package manifest files must be a list of relative paths')
    seen=set()
    result=[]
    for name in [*names,'package-manifest.json']:
        parts=name.split('/')
        if ('\\' in name or ':' in name or any(part in {'','.','..'} for part in parts)
                or any(part.casefold() in {'.git','roadmap.md','__pycache__','.tmpaeqmc2'} for part in parts)
                or parts[0].casefold() in {'userdata','outputs','validation','.upstream','release_packages'}
                or name.casefold() in seen):
            raise ValueError(f'Invalid package manifest path: {name}')
        seen.add(name.casefold())
        path=TARGET/name
        if not path.resolve().is_relative_to(TARGET) or not path.is_file() or any(
                reparse(parent) for parent in [path,*path.parents] if parent!=TARGET and TARGET in parent.parents):
            raise ValueError(f'Package manifest file is missing or escapes the destination: {name}')
        result.append(path)
    return sorted(result)


def write_archive(manifest):
    paths=archive_members(manifest)
    archive=TARGET.parent/f'Drama3-T8-Portable-{__version__}-win64.zip'
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent,prefix='.'+archive.name+'.',suffix='.part',delete=False) as handle:
            temporary=Path(handle.name)
        with zipfile.ZipFile(temporary,'w',allowZip64=True) as z:
            for path in paths:
                stored=path.suffix.lower() in {'.safetensors','.gguf','.pth','.pt','.dll','.pyd','.lib','.exe'}
                z.write(path,path.relative_to(TARGET.parent),compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        replace_file(temporary,archive)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    print(f'PORTABLE ZIP: {archive} ({archive.stat().st_size} bytes)',flush=True)
    return archive


def main():
    global TARGET
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,default=TARGET,help='Full environment folder (may be on another drive)')
    parser.add_argument('--extra-model',action='append',type=Path,default=[],help='Additional safetensors copied into models/')
    parser.add_argument('--zip',action='store_true')
    parser.add_argument('--hash-models',action='store_true')
    parser.add_argument('--zip-only',action='store_true')
    args=parser.parse_args()
    TARGET=validate_target(args.target)
    models=['drama3-t8-fp8.safetensors','drama3-codec.safetensors','drama3-text-head.safetensors','qwen3-asr-06b.safetensors']
    if (ROOT/'models/drama3-line-head.safetensors').is_file():models.append('drama3-line-head.safetensors')
    additional=extra_models(args.extra_model,models)
    if args.zip_only:
        manifest=json.loads((TARGET/'package-manifest.json').read_text(encoding='utf-8'))
        write_archive(manifest)
        return
    if not (ROOT/'runtime/python/python.exe').is_file():
        raise ValueError('The full environment requires runtime/python/python.exe')
    COPIED.clear()
    TARGET.mkdir(parents=True,exist_ok=True)
    for name in ['app.py','README.md','README.zh-CN.md','LICENSE','Start-Drama3.cmd','Update-Drama3.cmd','requirements.txt','upstream-lock.json','acceleration-manifest.json','acceleration-requirements.txt','acceptance-report.json']:
        copy_file(ROOT/name,TARGET/name)
    for name in ['drama3_t8','web','licenses','runtime']:
        copy_tree(ROOT/name,TARGET/name)
    # These are ComfyUI's top-level runtime data folders. Their names can also
    # occur in required source packages (comfy_api/input, comfy/ldm/models).
    copy_tree(ROOT/'vendor/ComfyUI',TARGET/'vendor/ComfyUI',root_skip=('custom_nodes','models','outputs','output','userdata','user','input','temp'))
    copy_tree(ROOT/'vendor/r2t2_native',TARGET/'vendor/r2t2_native')
    (TARGET/'vendor/ComfyUI/custom_nodes').mkdir(parents=True,exist_ok=True)
    node_target=TARGET/'comfyui-drama3-T8'
    copy_tree(ROOT/'comfyui-drama3-T8',node_target,extra_skip=('drama3_t8',))
    # A full build must contain the current shared engine/editor even when the
    # independent node staging folder has not been refreshed separately.
    for name in ['README.md','README.zh-CN.md','LICENSE','upstream-lock.json']:
        copy_file(ROOT/name,node_target/name)
    copy_tree(ROOT/'drama3_t8',node_target/'drama3_t8',
        exclude_files=('studio.py','studio_editor.py','bootstrap.py','updater.py'))
    copy_tree(ROOT/'licenses',node_target/'licenses')
    copy_tree(ROOT/'vendor/r2t2_native',node_target/'native-source')
    for name in ['editor_core.js','drama3_editor.js','seed-compat.js']:
        copy_file(ROOT/'web'/name,node_target/'web'/name)
    for name in ['check_runtime.py','download_models.py','convert_model.py','safetensors_stream.py','extract_text_head.py','update_portable.py']:
        copy_file(ROOT/'tools'/name,TARGET/'tools'/name)
    for name in models:
        source=ROOT/'models'/name
        copy_file(source,TARGET/'models'/name)
    for source in additional:
        copy_file(source,TARGET/'models'/source.name)
    copy_tree(ROOT/'models/qwen-aligner',TARGET/'models/qwen-aligner')
    for name in ['Confucius4-R2T2-GGUF','FireRedVAD-ONNX']:
        copy_tree(ROOT/'models'/name,TARGET/'models'/name)
    copied=sorted(COPIED)
    assert not any('roadmap.md' in p.casefold() or '.tmpaeqmc2' in p.casefold() or '.upstream/' in p.casefold() for p in copied)
    records=[]
    for name in copied:
        if not name.startswith('models/'):continue
        path=TARGET/name
        record={'path':name,'bytes':path.stat().st_size}
        if args.hash_models:
            with path.open('rb') as handle:
                digest=hashlib.sha256()
                for block in iter(lambda:handle.read(32*1024*1024),b''):digest.update(block)
            record['sha256']=digest.hexdigest()
        records.append(record)
    installed={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions(
        path=[str(ROOT/'runtime/python/Lib/site-packages')])}
    manifest={'version':__version__,'models':records,'model_source':json.loads((ROOT/'upstream-lock.json').read_text()),
              'runtime_packages':installed,'files':copied,'excluded':['roadmap.md','official source weights','failed conversion temp','userdata','outputs','development validation']}
    (TARGET/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PORTABLE LAYOUT: {TARGET}, {len(copied)} files',flush=True)
    if args.zip:
        write_archive(manifest)


if __name__=='__main__':main()
