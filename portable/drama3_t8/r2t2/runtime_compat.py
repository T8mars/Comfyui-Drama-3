"""Reject unsupported CUDA images before ggml can abort the host process."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

# The originally distributed DLL was built for Blackwell 12.0 only.
LEGACY_BACKENDS={'8a8474fa53f9b764570bde609963c1185cf80c0593be14db90f1f2f02e94d439':([120],[])}


@lru_cache(maxsize=16)
def _digest(path,size,mtime):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def supports_device(real,virtual,capability):
    major,minor=capability
    target=major*10+minor
    return target in real or any(arch<=target for arch in virtual)


def assert_runtime_compatible(directory,capability=None):
    directory=Path(directory)
    backend=directory/'backends/ggml-cuda.dll'
    stat=backend.stat()
    sha=_digest(str(backend.resolve()),stat.st_size,stat.st_mtime_ns)
    manifest=directory/'lib/runtime-compat.json'
    if not manifest.is_file():manifest=directory/'runtime-compat.json'
    if manifest.is_file():
        data=json.loads(manifest.read_text('utf-8'))
        if data.get('schema')!=1 or data.get('backend_sha256')!=sha:
            raise RuntimeError('Confucius CUDA 运行库与兼容清单不匹配，请重新安装同一套 R2T2 运行库')
        real,virtual=data.get('cuda_real_arches'),data.get('cuda_virtual_arches')
        if (not isinstance(real,list) or not isinstance(virtual,list) or not real+virtual
                or any(type(arch) is not int or not 50<=arch<=999 for arch in real+virtual)):
            raise RuntimeError('Confucius CUDA 运行库兼容清单无效')
    elif sha in LEGACY_BACKENDS:
        real,virtual=LEGACY_BACKENDS[sha]
    else:
        raise RuntimeError('Confucius CUDA 运行库缺少已验证的架构清单。请更新附带 R2T2 运行库；自定义路径需使用完整、匹配的运行库目录')
    if capability is None:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Confucius CUDA 后端需要可用的 NVIDIA GPU')
        capability=torch.cuda.get_device_capability(0)
    if not supports_device(real,virtual,capability):
        current=f'{capability[0]}.{capability[1]}'
        raise RuntimeError(f'Confucius CUDA 运行库不支持当前显卡计算能力 {current}。'
            '此旧库仅包含其他架构内核，请更新为多架构 R2T2 运行库；'
            '也可在环境设置中临时选择 Qwen ASR。未调用不兼容内核，Studio 可继续使用')
    return {'backend_sha256':sha,'cuda_real_arches':real,'cuda_virtual_arches':virtual,
        'device_capability':list(capability)}
