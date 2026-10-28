"""Model locations are user configuration, independent of the application directory."""
import os
from pathlib import Path


DEFAULT_PATHS = {
    "model": "drama3-t8-fp8.safetensors",
    "codec": "drama3-codec.safetensors",
    "text_head": "drama3-text-head.safetensors",
    "line_head": "drama3-line-head.safetensors",
    "asr_model": "qwen3-asr-06b.safetensors",
    "aligner": "qwen-aligner",
    "r2t2_model": "Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf",
    "r2t2_projector": "Confucius4-R2T2-GGUF/mmproj-Confucius4-R2T2-Q8_0.gguf",
    "r2t2_vad": "FireRedVAD-ONNX",
}
DIRECTORIES = {"aligner", "r2t2_vad"}


def model_path(value, base):
    """Absolute/UNC paths stay absolute; relative paths use the given models root."""
    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    if not text:
        raise ValueError("模型路径不能为空")
    path = Path(text)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def resolve_models(settings, base, *, check=False):
    paths = {}
    for key, default in DEFAULT_PATHS.items():
        value = settings.get(key, default)
        # Unused auxiliary inputs may be cleared in Studio. Keep a resolvable
        # default location and validate existence only when that feature runs.
        if key not in {'model', 'codec'} and isinstance(value, str) and not value.strip():
            value = default
        path = model_path(value, base)
        if key not in DIRECTORIES:
            expected = ".gguf" if key.startswith("r2t2_") else ".safetensors"
            if path.suffix.lower() != expected:
                raise ValueError(f"{key} 应选择 {expected} 模型文件：{path}")
        paths[key] = path
    if check:
        required = {"model", "codec"}
        head="text_head" if settings.get("scene_head_mode","compact")=="full" else (
            "line_head" if paths["line_head"].is_file() else "text_head")
        required.add(head)
        require_models(paths, *required)
    return paths


def require_models(paths, *keys):
    """Require only assets used by the current operation, without loading any."""
    for key in keys:
        if key not in paths:
            raise FileNotFoundError(f"缺少 {key} 模型路径，请在运行设置中配置")
        path = Path(paths[key])
        if key == 'aligner':
            if not (path.is_dir() or path.is_file() and path.suffix.lower() == '.safetensors'):
                raise FileNotFoundError(f"aligner 应为完整模型目录或转换后的 .safetensors：{path}")
            if path.is_dir() and not (path / 'config.json').is_file():
                raise FileNotFoundError(f"aligner 目录缺少 config.json：{path}")
        elif not (path.is_dir() if key in DIRECTORIES else path.is_file()):
            raise FileNotFoundError(f"{key} 路径不存在：{path}")
        if key == 'r2t2_vad':
            for name in ('fireredvad_stream_vad_with_cache.onnx', 'cmvn.ark'):
                if not (path / name).is_file():
                    raise FileNotFoundError(f"r2t2_vad 目录缺少 {name}：{path}")
