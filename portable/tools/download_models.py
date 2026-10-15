"""Download the pinned model artifacts. This command is explicitly user initiated."""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["drama", "asr", "aligner", "r2t2"], default="drama")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream-lock.json").read_text(encoding="utf-8"))
    if args.model == "r2t2":
        from huggingface_hub import hf_hub_download
        names = ["Confucius4-R2T2-Q8_0.gguf", "mmproj-Confucius4-R2T2-Q8_0.gguf", "MODEL_LICENSE", "MODEL_LICENSE_zh",
                 "FireRedVAD-ONNX/fireredvad_stream_vad_with_cache.onnx", "FireRedVAD-ONNX/cmvn.ark", "FireRedVAD-ONNX/LICENSE"]
        for name in names:
            target = ROOT / "models" if name.startswith("FireRedVAD-ONNX/") else ROOT / "models/Confucius4-R2T2-GGUF"
            hf_hub_download(repo_id=lock["r2t2_model_repo"], revision=lock["r2t2_model_revision"],
                             filename=name, local_dir=target)
        print("Confucius Q8 pair and FireRedVAD download complete", flush=True)
        return
    choices = {
        "drama": (lock["model_repo"], lock["model_revision"], ROOT / "models" / "official"),
        "asr": (lock["asr_repo"], lock["asr_revision"], ROOT / "models" / "qwen-asr"),
        "aligner": (lock["aligner_repo"], lock["aligner_revision"], ROOT / "models" / "qwen-aligner"),
    }
    repo, revision, target = choices[args.model]
    if revision == "main":
        revision = HfApi().model_info(repo, revision=revision).sha
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(f"Downloading {repo}@{revision} into {target}", flush=True)
    result = snapshot_download(repo, revision=revision, local_dir=target, max_workers=4)
    (target / "download-revision.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2), encoding="utf-8")
    print(f"DOWNLOAD COMPLETE: {result}", flush=True)


if __name__ == "__main__":
    main()
