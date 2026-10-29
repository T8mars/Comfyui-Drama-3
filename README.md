# ComfyUI-Drama3-T8

**English** | [简体中文](README.zh-CN.md)

Native [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3) nodes and a local voice studio by T8star. Built with Fish Audio.

[Models](https://huggingface.co/t8star/Drama-3-Comfy) · [Releases](https://github.com/T8mars/Comfyui-Drama-3/releases) · [16 canvas examples](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [Issues](https://github.com/T8mars/Comfyui-Drama-3/issues)

## Features

- **18 native ComfyUI V3 nodes**, standard `AUDIO` and 16 canvas workflows. Inference runs inside ComfyUI; no Studio server bridge is needed.
- Native reference-voice conditioning without training: encode reference audio with an accurate transcript. Multiple references, up to 16 speaker IDs, delivery cues and four dialogue modes.
- Director revisions using real audio history; Session save/load/branch and reusable reference-only voice libraries.
- Long text, text streaming with lookahead, generation audio previews, SRT-driven speech and subtitle export.
- A separate white-and-pink Studio with the same inference core and multitrack editor: role gain/mute, cuts, inserted silence, pause presets, undo and captions.
- Source-aware model infill, local Qwen3-ASR 0.6B/1.7B, word alignment and [Confucius R2T2](https://github.com/T8mars/comfyui-confucius-r2t2-t8). All model paths are configurable.

## Changes in 0.2.4

- Use Fish Audio's grouped attention path for audio-tower decode on both tested ComfyUI hosts. The reproduced W4 seed-42 repetition now completes with the original prompt and no retries.
- Check normalization-weight residency before CUDA Graph capture. If those weights are offloaded, generate eagerly and record the reason. Backbone selection, normalization arithmetic and model weights are unchanged.
- This is a code update; existing model files remain usable.

## Changes in 0.2.3

- Release decoder objects held by nested streaming errors and scene PCM held after preview callbacks fail. Preserve error identities, stack locations and unrelated caller diagnostics.
- Clarify Qwen ASR chunking: the target duration has an upstream-compatible +/-5-second quiet-point search. The partitioning algorithm is unchanged.
- Targeted Transformers 4.57.1 regression: **242 passed, 1 skipped** (Windows symlink privilege).

## Changes in 0.2.2

- Close workspace file handles even when initialization or explicit unlocking fails, preserving the original error and allowing the workspace to reopen. With Transformers 4.57.1, 494 targeted CPU regressions passed, including five new fault/ownership checks. Release 0.2.2 did not change generation code.

## Changes in 0.2.1

- Long-text continuation keeps a fixed voice reference. The first new chunk uses the existing conversation; later chunks use the Session's explicit references in their original order, or the first newly generated Take when no references were supplied. Saved references, prior Takes and direction remain in the Session. Ordinary speech, Scene, streaming and revision keep their existing history policies.
- Accelerated quantized MoE paths now preserve the intended BF16 intermediate rounding. BF16 Hadamard accumulation uses higher precision before a single BF16 rounding. This is an arithmetic correction, not a promise of identical output across all backends or higher speed.
- Quantized BF16 full-resident prefill computes expert mixtures in chunks to bound extra workspace. Model weights and full-prefill tensors remain part of the VRAM budget.
- Unloading waits for outstanding GPU copies before clearing unused pinned RAM. Load/cleanup errors preserve their original failure; Studio rejects new jobs with HTTP 503 if job-state storage fails.

## Installation

Registry node ID: **`comfyui-drama3-t8`** · Publisher: **`t8star`**. For manual installation, use **ComfyUI's own Python**:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

Restart ComfyUI, import a JSON from `workflows/`, select models/reference audio and click **Run**. Requires ComfyUI 0.40.0+ and Python 3.10+.

`install.py` installs the separately verified **Windows x64 / CUDA 12.8** R2T2 runtime, without model weights. Set `DRAMA_SKIP_R2T2_INSTALL=1` to skip it or `DRAMA_R2T2_NATIVE_DIR` to use an existing runtime. Qwen ASR works independently.

Use the supported dependency set: **Transformers 4.57.1 / tokenizers 0.22.2 / huggingface-hub 0.36.2**. Validation used Torch 2.8 + CUDA 12.8:

```sh
python -m pip install "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
```

Only **Transformers 4.57.1** is supported and tested. Transformers 5.x/6.x and other dependency combinations are not covered. Install into a compatible ComfyUI environment; the portable updater does not install Python dependencies.

## Voices and dialogue

Use matching `<|speaker:N|>` tags in reference transcripts and scripts to bind voices; names are display labels. Named roles keep stable IDs across revisions. Correct inaccurate automatic transcripts before generating.

`T8DramaLoadSession` output **0** is the complete Session; output **1** is References. Use output 1 with a fresh Session to reuse voices without old takes. Save, refresh the selector, then load. Scene's optional `session` preserves direction/history; connecting `references` explicitly replaces its references.

| Scene mode | Behavior |
|---|---|
| `together` | Generates dialogue with shared context. |
| `per_line` | Each line uses its role's references and history. |
| `per_line_context` | Also uses the preceding other role's actual line. |
| `per_speaker` | Groups by role, then restores script order. |

All modes expose `temperature`, `top_p`, `top_k`, `sampling_preset` and `opening_retries`. **stable** uses 0.7 / 0.7 / 30; **expressive** uses 1.0 / 0.9 / 0; **custom** uses entered values. Retries default to 0. Old workflows retain saved settings. For seeds above JavaScript's safe integer limit `9007199254740991`, use decimal `seed_text` up to `18446744073709551615`.

All four modes support `split_long_text`. `chunk_bytes` is **32–3000 UTF-8 bytes**, default 300, including new-script tags/separators; references, direction and history also consume context. Same-line chunks add no gaps. Isolated Scene previews play in script order after generation; ordinary speech can preview live audio.

Chunk policy v3 preserves spoken content and protected tags. Old v1/v2 boundaries remain readable; revisions retain their budget. Archive v2 stores source groups, roles and timing; incompatible readers reject it explicitly.

## Editing, transcription and alignment

Editor outputs a new Session, audio, SRT and source-bound edit metadata. Defaults preserve the original waveform. Set `align_words` / `aligner_path` for word timing; otherwise captions use line timing. Rhythm warnings are editing aids.

Model infill uses the actual playback selection and **complete replacement script**; set `target_speaker` for cross-role or silent selections. Native Session/AUDIO preserve Float32. Studio preserves unselected FLOAT/PCM16 samples, archives source/model output and encodes the complete result for history. FLAC examples are playback exports, not lossless Float32 exports.

Qwen ASR accepts ordered `AUDIO` batches and a per-clip `max_new_tokens` budget. Reference encoding and alignment take one clip at a time. A manual reference transcript needs no ASR. Auxiliary models load only when their features are used; short R2T2 clips need no VAD.

## Models and custom paths

Download converted weights from [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy). Node locations below are relative to `ComfyUI/`. Studio uses the same filenames/directories directly under `<bundle>/models/`.

| Asset | Node location |
|---|---|
| DRAMA FP8, 38.58 GB | `models/diffusion_models/drama3-t8-fp8.safetensors` |
| **Optional W4A8, 24.60 GB** | `models/diffusion_models/drama3-t8-w4a8.safetensors` |
| Codec, 1.57 GB | `models/vae/drama3-codec.safetensors` |
| Compact Scene head | `models/diffusion_models/drama3-line-head.safetensors` |
| Full text/control head, fallback | `models/diffusion_models/drama3-text-head.safetensors` |
| Qwen3-ASR 0.6B / optional 1.7B | `models/asr/qwen3-asr-06b.safetensors` / `models/asr/qwen3_asr_1.7b_bf16.safetensors` |
| ForcedAligner | `models/asr/qwen3_forced_aligner_0.6b_bf16.safetensors` or complete `models/asr/qwen-aligner/` |
| R2T2 model + projector | `models/asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf` + `mmproj-Confucius4-R2T2-Q8_0.gguf` |
| FireRedVAD for segmented R2T2 | `models/asr/FireRedVAD-ONNX/` |

Absolute paths, other drives and Windows UNC shares are supported. Relative paths use `ComfyUI/models/` in nodes or `<bundle>/models/` in Studio. Studio saves all nine paths under `userdata/`. Loader overrides: `model_path`, `codec_path`, `text_head_path`, `line_head_path`; ASR/alignment: `model_path`; R2T2: `model_path` / `projector_path` / `vad_path`.

`scene_head_mode=compact` uses the small head, falling back to required rows from the full head. `full` uses the legacy full projection. Saving Studio settings requires only the main model, Codec and selected head; unused auxiliary files are checked only when requested.

Converted **INT8 convrot / W6A8 / W4A8** load through the same main-model selector. Convert an existing FP8 file in a complete environment with a new destination filename:

```bat
runtime\python\python.exe tools\convert_model.py quantize "models\drama3-t8-fp8.safetensors" "models\drama3-t8-w4a8.safetensors" --format w4a8
```

Formats: `int8_convrot`, `w6a8`, `w4a8`. Smaller storage does not establish quality or memory feasibility for every model format.

## Studio and automatic updates

The author distributes the complete Windows environment separately. GitHub's `Drama3-T8-Portable-<version>-code.zip` contains **code only, without models or Python**. R2T2 native libraries are a separate verified asset. [Portable source](https://github.com/T8mars/Comfyui-Drama-3/tree/main/portable) is included in this repository.

In a compatible complete bundle based on 0.1.3+, run `Start-Drama3.cmd` to start Studio and check stable GitHub Releases. Close Studio before running `Update-Drama3.cmd` manually. For an offline code patch, close Studio and extract the code ZIP beside `Start-Drama3.cmd`.

The updater verifies size/SHA-256, backs up replaced files and rolls back failed transactions. It preserves models, Python, paths, projects, voices and outputs. Offline checks allow startup; updates do not run pip. Dependency changes require a compatible complete environment.

## Hardware and acceleration

Local validation used Windows x64, **24 GB VRAM / 64 GB RAM**, RTX 5090 Laptop and Torch 2.8 + CUDA 12.8. The W4 fixed cases used an **8 GiB expert VRAM cache**, which is not total memory consumption. Host RAM is checked before loading/pinning.

| Control | Meaning |
|---|---|
| `gdn_backend` | `torch` default, optional `fla`. Attention follows available ComfyUI kernels; installing FlashAttention does not force its use. |
| `moe_backend` | `auto` default, or `torch` / `cuda`. Actual backend and fallback reasons are recorded. |
| `cuda_graph` | Requested by default. Capture requires a compatible CUDA expert/host lifecycle and resident normalization weights; otherwise eager execution records the reason. |
| `expert_placement` | `ram` default: host experts plus cache. `vram` requires the complete model and heads to fit one GPU. |
| `expert_cache_gb` | RAM-mode cache budget in GiB; `0` is automatic. Related takes share the cache; supported single-GPU INT8 can use CPU experts. |

Codec uses bounded windows. Supported quantized pread loads check ownership and memory budgets; other layouts retain conservative loading. Whole-model VRAM residency has not been established on this 24 GB device. Prior W6/INT8 fixed speech checks do not establish general quality, memory feasibility or current-version coverage. Multi-GPU is outside scope. If CUDA capture is invalidated, restart the host and disable Graph for that configuration. Peer Voice/Codes files are not interchangeable with this project's Session archives.

## Validation scope

With Transformers 4.57.1, two CPU dependency stacks (Comfy Kitchen 0.2.37 / 0.2.40) each completed **1413 passed, 5 skipped, 1 GPU test deselected**, plus five passing subtests; CUDA remained uninitialized. **16 targeted GPU tests** cover numerical, cache and Graph contracts.

Fresh testing used **four actual ComfyUI canvases** across hosts 0.40.0 / 0.42.0 and **two Studio generation tasks**. Cases include the original no-reference W4 seed 42, long text with a voice reference, and a two-reference scene followed by two revisions. The original short case completed after the long-text task with seed 42, a 180-frame budget and zero retries. Each scene canvas includes generation and two revisions.

Blind checks of the newly exported audio: **R2T2 31/31; Qwen 29/31**. The 31 whole clips and crops contain 18 unique PCM inputs. All 24 short, chunk and line checks matched both recognizers. The two Qwen disagreements are the same whole-scene PCM exported by both hosts: Qwen heard “政府里” for “这福利”; isolated-line checks and a third blind recognizer support “这福利”. The third recognizer also wrote “写休” for “邪修”; all disagreements are retained. In each of the two scene/revision chains, reference-identity checks selected the intended reference for **9/9 line clips**; this is not a naturalness or MOS score.

These are fixed-case results, not a guarantee for arbitrary scripts, dialects, emotions or voices. Cross-backend output identity, speed improvements and full-model VRAM residency are not established. See the version, source scope and limitations in the [acceptance report](https://github.com/T8mars/Comfyui-Drama-3/blob/main/portable/acceptance-report.json).

## Credits and licenses

Based on [Fish Speech](https://github.com/fishaudio/fish-speech), [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3), [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and R2T2. Project code: [GPL-3.0](LICENSE); upstream licenses remain applicable. DRAMA weights/materials: Fish Audio Community License v1.0; Qwen/FireRedVAD: Apache-2.0; Confucius: NetEase Youdao Model Use License Agreement; llama.cpp: MIT. See [licenses/](licenses/) and [upstream-lock.json](upstream-lock.json). No upstream endorsement is implied.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
