# Drama-3 T8 Studio & ComfyUI

**English** | [简体中文](README.zh-CN.md)

A local voice creation studio and native ComfyUI nodes for [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3), by T8star. Built with Fish Audio.

[Models](https://huggingface.co/t8star/Drama-3-Comfy) · [Releases](https://github.com/T8mars/Comfyui-Drama-3/releases) · [19 canvas workflows](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [Issues](https://github.com/T8mars/Comfyui-Drama-3/issues)

## Local Studio

Director revisions offer **Replace previous emotion/speed tags**, enabled by default in Studio. The new direction uses the [official Direction/Line format](https://huggingface.co/spaces/fishaudio/Drama-3/blob/c368729ab9347dda6635c5272ad31e6521114b79/server.py#L799); original takes remain unchanged. Source auditions have their own player, and the output identifies the generated version. Edit regions initially cover the whole recording. For scenes, write `Director: dialogue` (Chinese `：` also works) so character descriptions map to the intended speaker. Emotion and pace remain model-generated controls rather than strict timing guarantees.

The white-and-pink Studio brings voices, scripts, takes and delivery into one local workspace. The complete Windows bundle is distributed by the author: [download](https://pan.quark.cn/s/264edb7e36bd). The complete bundle includes its matched Python environment and dependencies; no system Python or separate ComfyUI server is required. Run `Start-Drama3.exe` (or `Start-Drama3.cmd`), confirm model paths, then open **Creative Desk**.

1. **Build your cast.** Upload a clean reference, preview it immediately and edit its automatic local transcript, or select a short region from a recording. Name, tag, preview and organize voices; save explicit voice versions and reference groups for each character.
2. **Direct the performance.** Import lines, long text or SRT. Keep the original script, spoken text and subtitles separate. Generate several takes, compare individual lines at equal audition volume, then select and lock the performances to keep.
3. **Deliver or continue later.** Export a WAV, role stems, SRT and manifest, or export a complete project ZIP to move the project and continue on another machine.

DRAMA supports native reference-voice conditioning without training. Reference quality, transcription and delivery affect the result. For a new unreferenced voice, audition a take, save it as a voice reference and bind that version before continuing the character.

## New in 0.3.2

- **Clear voice casting:** the middle panel offers mutually exclusive individual casting and saved cast groups. Select each voice/version from the library or use one fixed group, with reference playback. Named roles map to the actual script speaker IDs; conflicting voice assignments are rejected.
- **Director revisions:** replace previous opening emotion/speed tags using the official Direction/Line format. Original takes remain unchanged; emotion and pace are model controls, with no strict speed guarantee.
- **Source-aware editing:** the selected source has its own player. Infill automatically prepares that source and checks its exact version/audio identity, even when the right player shows another result. Full-recording selection and nearest valid edit boundaries remain available.
- **Visible service console:** keep the EXE/CMD window open while using Studio. Closing it stops the owned service and helper processes. The launcher also cleans up helpers on normal exit.

Models and Python dependencies are unchanged. [Validation scope](https://github.com/T8mars/Comfyui-Drama-3/blob/v0.3.2/portable/acceptance-report.json) separates actual Studio generation, browser-only checks and earlier canvas validation. Updating an older EXE requires the separate launcher attachment below.

## New in 0.3.1

- **Floating generation controls:** a centered panel stays visible while scrolling, with real task progress, cancellation, model-path access and adjustable take counts (default: 3).
- **Reference preview and transcription:** uploaded audio plays immediately. Reference forms automatically use the selected local ASR; results remain editable, and delayed results preserve newer files and manual corrections.
- **Faster organization:** newest items first, cast-aware dialogue insertion, visible selected takes and recoverable deletion for voices, versions, cast snapshots, projects, candidates and history. Locked takes and active tasks are protected.
- **Editing fixes:** director revision validates only active fields and shows submission/errors beside its button. Word boundaries, waveform selections and entered edit times automatically snap to the nearest valid 0.05-second value; the waveform and fields stay synchronized.
- **Launcher and layout:** reopening an already running bundle opens its page; failures retain a log. Checkbox alignment, branding and social links are corrected.

This patch uses the existing models and dependencies. Its [validation report](https://github.com/T8mars/Comfyui-Drama-3/blob/v0.3.1/portable/acceptance-report.json) distinguishes the new UI/API/update checks from the earlier 0.3.0 canvas and model validation; it adds no acceleration recommendation or acting-quality score.

## Creative features introduced in 0.3.0

- **Versioned voice library:** search, tags, favorites, notes, archiving, original/processed audio playback, language/delivery variants and portable voice packs. Cast snapshots pin immutable versions and named reference groups; adding a version does not change older projects. Codec compatibility is visible, with explicit reencoding into a new version.
- **Creative projects:** stable line IDs, multiple takes, exact seeds, ratings, favorites and locked selections. Per-line delivery, sampling, reference group, target duration and pauses are editable. Voice defaults apply only when explicitly requested. In Creative Desk, scene/character descriptions apply to together/per-speaker synthesis; per-line, long-form and text-stream modes use explicit overall/per-line delivery directions, with voices bound by references and native speaker tags. Delivery directions use the model's current-line style syntax; reference transcripts and earlier takes remain unchanged.
- **Content-bound resume:** reuse completed work only when input fingerprints and media hashes match. Line modes resume by line; together/per-speaker modes resume by scene/role unit. Changed text, references, models or sampling invalidate affected work while preserving older candidates.
- **Real continuation and revisions:** per-line context uses actual preceding character audio; long text keeps a fixed voice anchor; text streaming uses bounded history and lookahead. Source-bound revise/infill produces alternative takes. Creative Desk infill uses the full intended line and a real audio region within that line, preserving PCM outside the region. Director notes apply to Revise. Complete project bundles carry the necessary Sessions, context and referenced voice versions in a project-private library.
- **Recording preparation:** file/microphone input, VAD segments, waveform and editable boundaries, roles and transcripts. Optional CAM++ provides speaker-group suggestions; a chosen short region can become a voice version. It does not separate overlapping voices.
- **Readout and review:** pronunciation aliases, digit/cardinal number reading and optional valid date/time expansion. Preview changes before generation; original text and subtitles remain separate. Optional independent ASR highlights textual differences; up to two explicit retry rounds suggest new takes for listening.
- **Delivery:** 44.1/48 kHz, duration reporting/padding/stretching, RMS/peak/LUFS options, role stems and SRT. The complete bundle includes FFmpeg for LUFS processing. A code update does not install imageio-ffmpeg into an older environment; existing system FFmpeg or `DRAMA_FFMPEG_PATH` can be used. Target duration does not guarantee a natural performance at every speed.
- **Measured performance:** actual backend/fallback reasons, resources, stage timings and comparable history. A short benchmark does not create a recommendation. A complete benchmark must pass its quality gate, and saved profiles are rechecked against the current machine/model binding before application. Idle unloading is configurable.

Automatic transcription and speaker similarity assist review; they are not acting, naturalness, identity-probability or MOS scores. Listen to final dialogue, edits and timing before delivery. EXE startup, actual canvas generation and resume have been checked; see the [versioned acceptance report](https://github.com/T8mars/Comfyui-Drama-3/blob/main/portable/acceptance-report.json) for scope and limits. The optional full acceleration comparison stopped at a RAM reader check; no speed recommendation was saved.

## Native ComfyUI

**30 native V3 nodes**, standard `AUDIO`, and **19 canvas workflow JSONs**. Inference runs directly in ComfyUI with the shared engine; no Studio HTTP bridge is required.

Registry ID: **`comfyui-drama3-t8`** · Publisher: **`t8star`**. Requires ComfyUI 0.40.0+ and Python 3.10+. Install using **ComfyUI's own Python**:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

Restart ComfyUI, import a JSON from `workflows/`, configure models/reference audio and click **Run**. New creative nodes are under `T8/Drama3/Creative`. Start with `workflow_creative_voice_project.json`; use `workflow_creative_project_resume.json` to open an existing project, or `workflow_creative_bundle_resume.json` to import a complete project copy. Repeated Create/Import operations create new projects; use Load to continue the same one. `takes_per_line` is the total desired take count: changing 3 to 6 adds the missing takes.

Existing nodes cover speech, reference encoding, four Scene modes (`together`, `per_line`, `per_line_context`, `per_speaker`), long text, streaming, Session save/load/branch, director revisions, infill, multitrack edits, SRT, transcription and word alignment. Reference transcripts and scripts use matching `<|speaker:N|>` tags, with up to 16 IDs. `T8DramaLoadSession` output 0 is the full Session; output 1 is References for reuse without old takes. For exact uint64 seeds, use decimal `seed_text` up to `18446744073709551615`.

### Dependencies

The supported set is **Transformers 4.57.1 / tokenizers 0.22.2 / huggingface-hub 0.36.2**, with Torch 2.8 + CUDA 12.8 as the target Windows environment. Other Transformers versions are not covered.

```sh
python -m pip install "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
```

`install.py` installs the separately verified Windows x64 / CUDA 12.8 R2T2 runtime, without weights. Set `DRAMA_SKIP_R2T2_INSTALL=1` to skip, or `DRAMA_R2T2_NATIVE_DIR` to use an existing runtime. Qwen ASR operates independently. Reference encoding with a manual transcript needs no ASR model.

## Models and custom paths

Download converted weights from [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy). These node paths are relative to `ComfyUI/`. Studio uses the same filenames/directories directly inside `<bundle>/models/`.

| Asset | ComfyUI location |
|---|---|
| Main model, FP8 or optional W4A8 | `models/diffusion_models/drama3-t8-fp8.safetensors` / `drama3-t8-w4a8.safetensors` |
| Codec | `models/vae/drama3-codec.safetensors` |
| Compact Scene head | `models/diffusion_models/drama3-line-head.safetensors` |
| Full text/control head, fallback | `models/diffusion_models/drama3-text-head.safetensors` |
| Qwen3-ASR 0.6B / optional 1.7B | `models/asr/qwen3-asr-06b.safetensors` / `qwen3_asr_1.7b_bf16.safetensors` |
| ForcedAligner | `models/asr/qwen3_forced_aligner_0.6b_bf16.safetensors` or complete `models/asr/qwen-aligner/` |
| Confucius R2T2 model + projector | `models/asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf` + `mmproj-Confucius4-R2T2-Q8_0.gguf` |
| FireRedVAD for segmented R2T2 | `models/asr/FireRedVAD-ONNX/` |
| Optional CAM++ speaker model | Studio: `models/campplus-zh-en.onnx`; custom path supported |

Main model, Codec and the selected/fallback head are needed for speech. ASR, aligner, VAD and speaker models are checked when their features run. Existing DRAMA weights remain usable for this code update; CAM++ is an optional addition.

- **Studio:** all nine speech/ASR model paths can be saved in the environment page. Creative settings independently configure the voice library, workspace and optional speaker-model file. Relative model paths use `<bundle>/models/`; relative library/workspace paths use the bundle root.
- **ComfyUI:** model loaders accept `model_path`, `codec_path`, `text_head_path` and `line_head_path`; ASR/alignment have their own paths, and R2T2 accepts model/projector/VAD paths. Creative nodes accept library/workspace paths, defaulting to ComfyUI's user directory outside the plugin. `DRAMA3_VOICE_LIBRARY` can set the native library default.
- Absolute paths, other drives and Windows UNC shares are supported. Keep custom libraries/workspaces outside the plugin source directory so plugin updates preserve them.

## Acceleration and memory

Available controls are `gdn_backend=torch/fla`, `moe_backend=auto/torch/cuda`, CUDA Graph, RAM/VRAM expert placement and `expert_cache_gb` (`0` means automatic). The expert cache is part of total memory use. Whole-model VRAM placement requires all relevant weights and heads to fit.

Installing FlashAttention or a wheel alone does not establish that it is selected or faster. Use the actual backend/fallback report and the local benchmark for your model and hardware. Different backends may produce different audio; benchmark recommendations do not replace listening. The full model and host expert cache need substantial RAM/VRAM, and loading checks available resources. Auto can retry once with Torch after a verified pinned-RAM refusal and complete cleanup; explicit CUDA retains the resource error.

## Automatic updates

GitHub's `Drama3-T8-Portable-<version>-code.zip` contains **code only: no Python runtime and no models**. The author supplies the complete bundle separately. R2T2 native libraries are a separate verified asset.

The full bundle provides `Start-Drama3.exe`; `Start-Drama3.cmd` remains available. Both check stable GitHub Releases before starting. The code-only update ZIP does not install or replace the EXE launcher. Close Studio before running `Update-Drama3.cmd` manually. The updater verifies size/SHA-256, backs up replaced files and rolls back failed transactions. Models, Python, `userdata`, outputs, libraries/workspaces and custom paths are preserved. Updates do not run pip; a dependency change requires a compatible full environment. Offline checks allow the installed version to start. Manually extracting a code ZIP directly overwrites files and does not run the updater's SHA verification or transaction rollback; prefer `Update-Drama3.cmd`.

For the console/cleanup fix, close all Studio launcher windows and replace the bundle's `Start-Drama3.exe` with the [0.3.2 launcher](https://github.com/T8mars/Comfyui-Drama-3/releases/download/v0.3.2/Start-Drama3.exe). Keep its console open; closing it stops that bundle's service and helpers. It requires the existing complete bundle and is not a standalone Studio installer.

## Sources and licenses

Based on [Fish Speech](https://github.com/fishaudio/fish-speech), [PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3), [ComfyUI](https://github.com/Comfy-Org/ComfyUI) and [Confucius R2T2 T8](https://github.com/T8mars/comfyui-confucius-r2t2-t8). Project code: [GPL-3.0](LICENSE). DRAMA weights/materials: Fish Audio Community License v1.0; Qwen, FireRedVAD and optional CAM++: Apache-2.0; Confucius weights: NetEase Youdao Model Use License Agreement; llama.cpp: MIT. See [licenses/](licenses/), the [CAM++ notice](licenses/NOTICE-CAMPPLUS.md) and [upstream-lock.json](upstream-lock.json). Upstream licenses continue to apply; no upstream endorsement is implied.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
