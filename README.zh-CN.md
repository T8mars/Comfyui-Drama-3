# ComfyUI-Drama3-T8

[English](README.md) | **简体中文**

T8star 制作的 [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3) 原生节点与本地配音工作台。Built with Fish Audio.

[模型下载](https://huggingface.co/t8star/Drama-3-Comfy) · [版本下载](https://github.com/T8mars/Comfyui-Drama-3/releases) · [16 张画布示例](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [问题反馈](https://github.com/T8mars/Comfyui-Drama-3/issues)

## 功能

- **18 个 ComfyUI V3 原生节点**、标准 `AUDIO`、16 张画布工作流。推理直接在 ComfyUI 内运行，无需 Studio 服务桥接。
- 无需训练的原生参考音色条件：编码参考音频并配准确台词。支持多参考、最多 16 个角色编号、演出提示与四种对话模式。
- 使用真实音频历史的导演改稿；Session 保存、回读、分支与仅含参考的可复用音色库。
- 长文、带 lookahead 的流式文本、生成中音频试听、SRT 驱动配音与字幕导出。
- 独立白粉 Studio，共用推理核心与多轨编辑器：角色音量/静音、剪切、插入静音、停顿预设、撤销与字幕。
- 基于真实音频来源的模型局部编辑、本地 Qwen3-ASR 0.6B/1.7B、词级对齐及 [Confucius R2T2](https://github.com/T8mars/comfyui-confucius-r2t2-t8)。模型路径均可自定义。

## 0.2.4 更新

- 两个已测 ComfyUI 宿主统一使用 Fish Audio 的音频塔 grouped attention 路径。原先复现静音重复的 W4、seed 42 案例现可用原稿完成，无需增加重试。
- CUDA Graph 捕获前检查归一化权重驻留；权重已卸载时提前使用 eager 生成并记录原因。主干选择、归一化算术与模型权重保持原样。
- 本次为代码更新，现有模型文件可继续使用。

## 0.2.3 更新

- 修复流式解码嵌套异常留住解码器对象、预览回调失败留住场景 PCM 的问题；保留原始异常、堆栈位置和无关调用方的诊断信息。
- 澄清 Qwen 转写分段：目标时长允许按上游策略在前后 5 秒内寻找静音点，切分算法未改。
- Transformers 4.57.1 定向回归 **242 项通过、1 项跳过**（Windows 符号链接权限）。

## 0.2.2 更新

- 修复工作区锁初始化或显式解锁失败时未关闭文件句柄的问题，保留原始异常，允许重新打开工作区。Transformers 4.57.1 环境下共 494 项定向 CPU 回归通过，包含新增的 5 项故障/锁归属检查。0.2.2 未修改生成代码。

## 0.2.1 更新

- 长文续读使用固定音色参考：首个新分段沿用原会话，后续优先按原顺序使用 Session 显式参考；未提供参考时，固定使用本次首个新 Take。原参考、旧 Take 和方向指令仍完整保存在 Session 中。普通语音、Scene、流式和改稿的历史策略保持原样。
- 修正量化 MoE 加速路径中的 BF16 中间结果舍入。BF16 Hadamard 累加先用更高精度计算，再进行一次 BF16 舍入。 这是算术修正，不表示所有后端输出完全一致或速度提高。
- BF16 量化全驻留预填充按块计算专家混合，限制额外工作区；模型权重与整段预填充张量仍计入显存需求。

- 模型加载/清理保留原始异常；Studio 任务状态写入失败时暂停接单并返回可读的 503。

卸载会先等待 GPU 复制完成，再清理闲置锁页内存缓存。

## 安装

Registry 节点 ID：**`comfyui-drama3-t8`** · 发布者：**`t8star`**。手动安装须使用 **ComfyUI 自己的 Python**：

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

重启 ComfyUI，导入 `workflows/` 的 JSON，选择模型、参考音频后点击 **Run**。要求 ComfyUI 0.40.0+、Python 3.10+。

`install.py` 安装单独校验的 **Windows x64 / CUDA 12.8** R2T2 运行库，不下载模型。设 `DRAMA_SKIP_R2T2_INSTALL=1` 可跳过，已有运行库用 `DRAMA_R2T2_NATIVE_DIR` 指定。Qwen 转写独立可用。

支持的依赖组合为 **Transformers 4.57.1 / tokenizers 0.22.2 / huggingface-hub 0.36.2**。实际验证使用 Torch 2.8 + CUDA 12.8：

```sh
python -m pip install "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
```

仅支持并验证 **Transformers 4.57.1**，不声明兼容 Transformers 5.x/6.x 或其他依赖组合。请使用兼容的 ComfyUI 环境；整合包更新器不会安装 Python 依赖。

## 音色与对话

用参考台词与脚本中对应的 `<|speaker:N|>` 标签绑定音色，名称仅用于显示。命名角色改稿时保留稳定编号；自动转写不准确时请先校正。

`T8DramaLoadSession` 输出 **0** 为完整 Session，输出 **1** 为 References。不带旧 Take 复用音色时，将输出 1 接到新 Session。先保存、刷新列表，再加载。Scene 的可选 `session` 保留方向与历史；显式连接 `references` 会替换其参考。

| Scene 模式 | 行为 |
|---|---|
| `together` | 共享上下文生成对话。 |
| `per_line` | 每行使用本角色参考和历史。 |
| `per_line_context` | 另带前一位其他角色的真实台词。 |
| `per_speaker` | 分角色生成，再还原脚本顺序。 |

四种模式均提供 `temperature`、`top_p`、`top_k`、`sampling_preset` 和 `opening_retries`。**stable** 使用 0.7 / 0.7 / 30；**expressive** 使用 1.0 / 0.9 / 0；**custom** 使用手填数值。重试默认 0，旧工作流保留已存设置。超过 JavaScript 安全整数上限 `9007199254740991` 的种子，请使用十进制 `seed_text`，最大 `18446744073709551615`。

四种模式均支持 `split_long_text`。`chunk_bytes` 为 **32–3000 UTF-8 字节**，默认 300，含新脚本标签/分隔符；参考、方向与历史也占上下文。同一行分块不加停顿。隔离 Scene 完成后按脚本顺序试听，普通配音支持生成中试听。

分块策略 v3 保留正文与受保护标签。旧 v1/v2 边界仍可读取，改稿保留预算。v2 归档保存来源分组、角色与时间线；不兼容客户端会明确拒绝。

## 编辑、转写与对齐

Editor 输出新 Session、音频、SRT 与来源绑定的编辑信息，默认保留原波形。设置 `align_words` / `aligner_path` 可启用词时间，否则按行计时。节奏提示仅作编辑辅助。

模型 infill 使用真实播放选区和**完整替换脚本**，跨角色或静音选区设置 `target_speaker`。原生 Session/AUDIO 保留 Float32；Studio 保留未选 FLOAT/PCM16 样本，归档来源与模型输出，并编码完整成品供后续历史使用。FLAC 示例是播放导出，不是无损 Float32 导出。

Qwen ASR 支持顺序 `AUDIO` batch 和每条音频的 `max_new_tokens` 预算。参考编码、对齐每次接收一个音频；手填参考台词无需 ASR。辅助模型仅在对应功能运行时加载，短 R2T2 音频无需 VAD。

## 模型与自定义路径

从 [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy) 下载转换后的权重。下表节点位置相对 `ComfyUI/`；Studio 将对应文件名/目录直接放在 `<整合包>/models/` 下。

| 模型 | 节点位置 |
|---|---|
| DRAMA FP8，38.58 GB | `models/diffusion_models/drama3-t8-fp8.safetensors` |
| **可选 W4A8，24.60 GB** | `models/diffusion_models/drama3-t8-w4a8.safetensors` |
| Codec，1.57 GB | `models/vae/drama3-codec.safetensors` |
| 紧凑 Scene 控制头 | `models/diffusion_models/drama3-line-head.safetensors` |
| 完整文本/控制头，可回退 | `models/diffusion_models/drama3-text-head.safetensors` |
| Qwen3-ASR 0.6B / 可选 1.7B | `models/asr/qwen3-asr-06b.safetensors` / `models/asr/qwen3_asr_1.7b_bf16.safetensors` |
| ForcedAligner | `models/asr/qwen3_forced_aligner_0.6b_bf16.safetensors` 或完整 `models/asr/qwen-aligner/` |
| R2T2 模型与 projector | `models/asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf` + `mmproj-Confucius4-R2T2-Q8_0.gguf` |
| 分段 R2T2 的 FireRedVAD | `models/asr/FireRedVAD-ONNX/` |

支持绝对路径、其他盘与 Windows UNC 共享。相对路径在节点以 `ComfyUI/models/` 为根，Studio 以 `<整合包>/models/` 为根，九项路径保存于 `userdata/`。Loader 覆盖参数：`model_path`、`codec_path`、`text_head_path`、`line_head_path`；转写/对齐：`model_path`；R2T2：`model_path` / `projector_path` / `vad_path`。

`scene_head_mode=compact` 使用小控制头，缺少时从完整头读取所需行；`full` 使用旧完整投影。保存 Studio 设置只需主模型、Codec 与当前控制头；未使用的辅助文件仅在调用时检查。

转换后的 **INT8 convrot / W6A8 / W4A8** 直接在同一主模型选择器加载。完整环境可将已有 FP8 转换到新的输出文件：

```bat
runtime\python\python.exe tools\convert_model.py quantize "models\drama3-t8-fp8.safetensors" "models\drama3-t8-w4a8.safetensors" --format w4a8
```

格式可选 `int8_convrot`、`w6a8`、`w4a8`。存储更小不代表所有格式的质量或内存适用性均已验证。

## Studio 与自动更新

作者单独分发完整 Windows 环境。GitHub 的 `Drama3-T8-Portable-<版本>-code.zip` **仅含代码，不含模型和 Python**；R2T2 原生库为单独校验的资产。仓库同时提供[工作台源码](https://github.com/T8mars/Comfyui-Drama-3/tree/main/portable)。

已有兼容的 0.1.3+ 完整环境，运行 `Start-Drama3.cmd` 启动 Studio 并检查 GitHub 稳定 Release。手动更新前关闭 Studio，再运行 `Update-Drama3.cmd`。离线代码补丁可在关闭 Studio 后，解压到 `Start-Drama3.cmd` 所在目录。

更新器校验大小/SHA-256、备份文件并在事务失败时回滚，保留模型、Python、路径、工程、音色库与输出。离线检查不阻止启动，更新不自动 pip；依赖变更需兼容的完整环境。

## 硬件与加速

本地验证设备为 Windows x64、**24 GB 显存 / 64 GB 内存**、RTX 5090 Laptop、Torch 2.8 + CUDA 12.8。固定 W4 案例使用 **8 GiB 专家显存缓存**，它不是总内存占用；加载和锁页前会检查主机 RAM。

| 控件 | 含义 |
|---|---|
| `gdn_backend` | 默认 `torch`，可选 `fla`。注意力使用 ComfyUI 可用内核；安装 FlashAttention 不会强制启用。 |
| `moe_backend` | 默认 `auto`，可选 `torch` / `cuda`；记录实际后端和回退原因。 |
| `cuda_graph` | 默认请求开启；捕获需要兼容的 CUDA 专家/宿主生命周期和已驻留的归一化权重，否则使用 eager 并记录原因。 |
| `expert_placement` | 默认 `ram`：主存专家加显存缓存；`vram` 要求整模型和控制头装入同一 GPU。 |
| `expert_cache_gb` | RAM 模式缓存 GiB，`0` 自动；相关 Take 共用缓存，支持的单卡 INT8 可协同 CPU 专家。 |

Codec 使用有界窗口；支持的量化 pread 加载会校验所有权与内存预算，其他布局保留保守加载。本机 24 GB 显存尚未验证整模型全显存驻留。此前 W6/INT8 固定语音检查不代表通用质量、内存可行性或当前版本覆盖，多卡不在范围内。CUDA capture 失效后，须重启宿主并对该配置关闭 Graph。对方项目的 Voice/Codes 文件不能直接替换本项目 Session 归档。

## 验证范围

Transformers 4.57.1 下，两套 CPU 依赖环境（Comfy Kitchen 0.2.37 / 0.2.40）各完成 **1413 项通过、5 项跳过、1 项 GPU 测试未选入**，另有 5 个子测试通过，CUDA 未初始化。**16 项定向 GPU 测试**覆盖数值、缓存与 Graph 契约。

本轮通过 ComfyUI 0.40.0 / 0.42.0 的 **4 张真实画布**及 **2 个 Studio 生成任务**复测，包含原无参考 W4 seed 42、带音色参考的长文，以及双参考场景连续两次改稿。原短句在长文任务后，以 seed 42、180 帧预算、0 重试完成。每张场景画布包含原始生成及连续两次改稿。

新导出音频盲转写：**R2T2 31/31；Qwen 29/31**。这 31 份整段与切片含 18 份独立 PCM；短句、分段和逐句共 24 项均通过两路核对。两处 Qwen 分歧来自两个宿主导出的同一份整场 PCM：将“这福利”识为“政府里”；逐句核对与第三路盲转写支持“这福利”。第三路另有“邪修”识为“写休”的分歧，均如实保留。两条场景/改稿链的参考身份检查各有 **9/9 条分句**选中预期参考；这不是自然度或 MOS 评分。

以上只覆盖固定案例，不代表任意脚本、方言、情绪或音色均已验证，也不承诺不同后端输出一致、速度提高或整模型全显存驻留。具体版本、源码范围与限制见[验收报告](https://github.com/T8mars/Comfyui-Drama-3/blob/main/portable/acceptance-report.json)。

## 来源与许可

基于 [Fish Speech](https://github.com/fishaudio/fish-speech)、[PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI) 与 R2T2。项目代码 [GPL-3.0](LICENSE)，各上游保留原许可。DRAMA 权重/材料：Fish Audio Community License v1.0；Qwen/FireRedVAD：Apache-2.0；Confucius：网易有道模型使用许可；llama.cpp：MIT。详见 [licenses/](licenses/) 与 [upstream-lock.json](upstream-lock.json)，不代表上游认可或背书。

## T8star

[哔哩哔哩](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费图库](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
