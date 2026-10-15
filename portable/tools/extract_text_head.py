"""Keep the official full output head required for interleaved scene control text."""
from pathlib import Path
import json
import sys
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "models/official"
index = json.loads((source / "model.safetensors.index.json").read_text())
with safe_open(str(source / index["weight_map"]["lm_head.weight"]), framework="pt") as f:
    weight = f.get_tensor("lm_head.weight")
save_file({"weight":weight}, str(ROOT / "models/drama3-text-head.safetensors"),
          metadata={"source_revision":"e0a9218aa48888270f06ef1ed9ea67622c0e17d2"})
print(weight.shape)
