# ComfyUI-Drama3-T8

**English** | [简体中文](README.zh-CN.md)

Native [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3) nodes and a local voice studio by T8star. Built with Fish Audio.

[Models](https://huggingface.co/t8star/Drama-3-Comfy) · [Releases](https://github.com/T8mars/Comfyui-Drama-3/releases) · [Example workflows](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [Issues](https://github.com/T8mars/Comfyui-Drama-3/issues)

## Features

- Voice cloning with multiple references and up to 16 speaker IDs; inline directions and multi-character scenes.
- Director revisions using actual audio history; infill editing; long text, SRT and streamed text with lookahead.
- Codec encoding/decoding, reusable references, session save/load/branch and audio chunk previews.
- Local Qwen3-ASR, Qwen ForcedAligner and [Confucius R2T2](https://github.com/T8mars/comfyui-confucius-r2t2-t8), with editable model paths.
- 17 native ComfyUI V3 nodes using standard `AUDIO`. Inference, codec and R2T2 run inside the host process. The separate white-and-pink studio shares the inference core.

## Install in ComfyUI

Install **ComfyUI-Drama3-T8**, publisher **t8star**, through ComfyUI-Manager, or install manually:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

Use your **ComfyUI Python**, then restart ComfyUI. Import a workflow from `workflows/` and select your models and reference audio. Requires ComfyUI 0.40.0+ and Python 3.10+. `install.py` downloads and verifies the separately released Windows x64 / CUDA 12.8 R2T2 native runtime; it never downloads models. Set `DRAMA_SKIP_R2T2_INSTALL=1` to skip this optional backend, or `DRAMA_R2T2_NATIVE_DIR` to use an existing runtime. Qwen ASR works independently.

## Models

Download from [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy). Default paths are relative to `ComfyUI/`.

| Model | Default location |
|---|---|
| DRAMA FP8, 38.58 GB | `models/diffusion_models/drama3-t8-fp8.safetensors` |
| Complete text/control head, 1.03 GB | `models/diffusion_models/drama3-text-head.safetensors` |
| Codec, 1.57 GB | `models/vae/drama3-codec.safetensors` |
| Qwen ASR, optional | `models/asr/qwen3-asr-06b.safetensors` |
| ForcedAligner, optional full directory | `models/asr/qwen-aligner/` |
| R2T2 Q8 model + projector, optional | `models/asr/Confucius4-R2T2-GGUF/` |
| FireRedVAD, for segmented R2T2 | `models/asr/FireRedVAD-ONNX/` |

Keep all three DRAMA files for complete functionality. Each loader supports custom file/directory paths, including another drive. The studio stores these settings under `userdata/`.

## Local studio and updates

The author distributes the complete Windows environment separately. GitHub Releases contain **program files and native libraries only, without model weights or a Python runtime**. Studio source is in [portable/](https://github.com/T8mars/Comfyui-Drama-3/tree/main/portable).

For an existing complete bundle based on 0.1.3 or later, close the studio and extract `Drama3-T8-Portable-0.1.7-code.zip` into the directory containing `Start-Drama3.cmd`. Double-click `Start-Drama3.cmd` to launch. Subsequent launches check stable GitHub Releases and apply compatible code updates; `Update-Drama3.cmd` checks manually while the studio is closed. Updates verify sizes and SHA-256, back up old files and roll back on failure. Models, Python, custom paths, projects, voice libraries and outputs are retained. Offline checks allow startup; dependency/runtime changes require a new complete environment from the author.

## Hardware and acceleration

Validated: Windows x64, RTX 5090 Laptop **24 GB VRAM / 64 GB RAM**, Torch 2.8 + CUDA 12.8, 12 GiB expert cache. FP8 MoE experts stay in system RAM; cache `0` selects an automatic budget. Leave VRAM for resident weights, codec and context. Other hardware configurations have not been validated.

Default: Torch linear attention + SDPA. Optional FLA is integrated but showed no clear speed gain on this setup. Installing FlashAttention alone does not enable it in the model; CUDA Graph is disabled. CPU previews arrive during generation, but the tested offload setup generates below real time. Transcription may require manual correction.

## Credits and licenses

Based on [Fish Speech](https://github.com/fishaudio/fish-speech), [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3), [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and the linked R2T2 project. T8 additions include complete control output, stateful editing, streaming, native ASR integration and the local studio.

Project code: [GPL-3.0](LICENSE), with upstream components retaining their licenses. DRAMA materials and weights: **Fish Audio Community License v1.0**; Qwen and FireRedVAD: Apache-2.0; Confucius models: NetEase Youdao Model Use License Agreement; llama.cpp: MIT. See [licenses/](licenses/) and [upstream-lock.json](upstream-lock.json). This project is not affiliated with or endorsed by upstream authors.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
