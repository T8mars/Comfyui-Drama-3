"""Model locations are user configuration, independent of the application directory."""
import os
from pathlib import Path


DEFAULT_PATHS = {
    "model": "drama3-t8-fp8.safetensors",
    "codec": "drama3-codec.safetensors",
    "text_head": "drama3-text-head.safetensors",
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
        path = model_path(settings.get(key, default), base)
        if key not in DIRECTORIES:
            expected = ".gguf" if key.startswith("r2t2_") else ".safetensors"
            if path.suffix.lower() != expected:
                raise ValueError(f"{key} 应选择 {expected} 模型文件：{path}")
        paths[key] = path
    if check:
        required = {"model", "codec", "text_head", "aligner"}
        required |= ({"r2t2_model", "r2t2_projector", "r2t2_vad"}
                     if settings.get("asr_backend") == "r2t2" else {"asr_model"})
        for key in required:
            path = paths[key]
            if not (path.is_dir() if key in DIRECTORIES else path.is_file()):
                raise FileNotFoundError(f"{key} 路径不存在：{path}")
        if not (paths["aligner"] / "config.json").is_file():
            raise FileNotFoundError("词级对齐路径应为包含 config.json 的完整模型目录")
        if "r2t2_vad" in required:
            for name in ("fireredvad_stream_vad_with_cache.onnx", "cmvn.ark"):
                if not (paths["r2t2_vad"] / name).is_file():
                    raise FileNotFoundError(f"VAD 目录缺少 {name}")
    return paths
