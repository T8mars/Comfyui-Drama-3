"""Use ComfyUI's native model engine without starting a ComfyUI server."""
import os
import sys
from pathlib import Path


def standalone_engine(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    engine = root / "vendor" / "ComfyUI"
    if not (engine / "comfy" / "model_management.py").is_file():
        raise FileNotFoundError(f"Missing bundled model engine: {engine}")
    sys.path.insert(0, str(engine))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "userdata" / "cache" / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "userdata" / "cache" / "inductor"))
    from comfy.cli_args import args
    args.use_pytorch_cross_attention = True
    args.disable_pinned_memory = True
    args.disable_dynamic_vram = True
    return engine
