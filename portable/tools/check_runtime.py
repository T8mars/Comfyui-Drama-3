import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drama3_t8.bootstrap import standalone_engine

standalone_engine()
import torch
from drama3_t8.native.loader import load_drama3, load_codec
from drama3_t8.native.config import parse_config

print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "engine_root": str(ROOT / "vendor/ComfyUI"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "native_loader": load_drama3.__module__,
    "codec_loader": load_codec.__module__,
}, indent=2))
