# Drama-3 T8 Studio & ComfyUI

[English](README.md) | **简体中文**

T8star 制作的 [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3) 本地配音工作台与 ComfyUI 原生节点。Built with Fish Audio.

[模型下载](https://huggingface.co/t8star/Drama-3-Comfy) · [版本下载](https://github.com/T8mars/Comfyui-Drama-3/releases) · [19 张画布工作流](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [问题反馈](https://github.com/T8mars/Comfyui-Drama-3/issues)

## 本地 Studio

白粉 Studio 将音色、剧本、候选与交付放在同一个本地工作区。完整 Windows 整合包由作者提供：[下载地址](https://pan.quark.cn/s/264edb7e36bd)。完整包自带匹配的 Python 与依赖，无需系统 Python 或另开 ComfyUI 服务。双击 `Start-Drama3.exe`（也可使用 `Start-Drama3.cmd`），先确认模型路径，再进入**创作工程**。

1. **建立选角。** 上传干净参考录音即可试听、自动本地转写并手动校稿，也可从长录音中选择短片段。命名、标签、试听与整理音色，为每个角色固定声音版本和参考组。
2. **安排演出。** 导入逐行剧本、长文或 SRT，分别保存原稿、朗读稿与字幕。每句生成多个候选，按单句同音量试听，选片并锁定满意的演出。
3. **交付或续做。** 导出 WAV、角色分轨、SRT 和清单；需要迁移设备时，导出完整工程 ZIP 继续制作。

DRAMA 原生支持无需训练的参考音色条件生成，参考音质、准确台词与演出方式都会影响结果。无参考试音得到满意声音后，可将候选保存为音色并绑定固定版本，再继续这个角色。

## 0.3.1 更新

- **悬浮生成面板：** 页面滚动时始终居中可见，显示真实任务进度、取消、模型路径入口与候选数量设置，默认仍为 3 遍。
- **参考试听与自动转写：** 上传音频立即试听，参考表单自动调用所选本地 ASR，结果可直接校稿；过期结果不会覆盖新文件或人工修改。
- **整理与选片：** 新项目靠前，按选角插入对白，清楚显示已选入成片的候选。音色、版本、选角、工程、候选与历史支持删除和回收站恢复；保护锁定选片与活动任务。
- **编辑修复：** 导演修改仅校验当前字段，并在按钮附近显示提交或错误信息。获取词边界、拖动波形和输入时间自动取最近有效的 0.05 秒值，波形与输入同步。
- **启动与布局：** 重复启动会打开已运行页面，启动失败保留日志；修复复选框对齐，完善 By T8star 和社媒链接。

本次沿用现有模型与依赖。[验证报告](https://github.com/T8mars/Comfyui-Drama-3/blob/v0.3.1/portable/acceptance-report.json) 区分本轮界面、API、更新检查与 0.3.0 原有画布和模型验证，不新增加速推荐或演技评分。

## 0.3.0 创作功能

- **版本化音色库：** 搜索、标签、收藏、备注、归档、原音/处理片段试听、多语言/多演绎与便携音色包。选角快照固定不可变版本和命名参考组，新增版本不改变旧工程。显示 Codec 兼容状态，可显式重编码为新版本。
- **创作工程：** 稳定台词编号、多 Take、精确种子、评分、收藏与选片锁定。逐句设置演出、采样、参考组、目标时长和停顿。音色默认参数仅在明确点击应用后生效。创作工程的场景/人物描述用于同场对话与按角色合成；逐句、长文和文本流使用显式整体/句级演出方向，参考音色与原生角色标签负责声音绑定。演出方向按模型当前台词的样式语法传入，参考稿与旧候选保持原样。
- **按内容断点续做：** 完成项只有在输入指纹和媒体校验一致时才复用；逐句模式按句恢复，一起生成/按角色模式以场景或角色为单位恢复。修改文字、参考、模型或采样会使相关项失效，旧候选仍保留。
- **真实上下文与改稿：** 逐句上下文使用前一角色的实际声音；长文保持固定音色锚点，文本流使用有界历史和 lookahead。基于原始来源的改稿/局部重配生成替代候选。创作工程的局部重配填写完整目标台词，选择该句内的真实音频区间，选区外 PCM 保持不变；导演意见用于改稿。完整工程包携带必要 Session、上下文和实际引用音色版本，导入使用工程私有音色库。
- **录音整理：** 文件/麦克风输入、VAD 分段、真实波形，可修改边界、角色与校稿。可选 CAM++ 提供说话人分组建议，选中短片段可存成音色版本；不分离重叠人声。
- **朗读稿与复核：** 别名词典、数字逐位/基数读法，可选有效日期与时间展开，生成前预览；原稿和字幕独立保存。可选独立 ASR 标出文字差异，最多两轮显式补录产生建议候选，仍需试听选择。
- **交付：** 44.1/48 kHz、时长报告/补静音/拉伸、RMS/峰值/LUFS、角色分轨与 SRT。完整包已内置 FFmpeg，供 LUFS 使用；旧包代码更新不会安装 imageio-ffmpeg，可使用已有系统 FFmpeg 或 `DRAMA_FFMPEG_PATH`。目标时长不保证任意语速都自然。
- **实际性能记录：** 显示真正使用的后端、回退原因、资源、阶段耗时和可比历史。短句对照不生成推荐档案；完整对照须通过质量门禁，保存的档案应用前重新核验当前设备和模型绑定。可配置空闲卸载。

自动转写与音色相似度用于辅助复核，不是演技、自然度、身份概率或 MOS 分数。交付前请试听对白、改稿与节奏。EXE 启动、真实画布生成与续作已验证，检查范围与限制见[对应版本验收报告](https://github.com/T8mars/Comfyui-Drama-3/blob/main/portable/acceptance-report.json)。完整加速对照触发 RAM 读取检查后中止，未保存加速推荐。

## ComfyUI 原生节点

**30 个 V3 原生节点**、标准 `AUDIO` 与 **19 张画布 JSON**。节点直接在 ComfyUI 内使用共用推理核心，无需 Studio HTTP 桥接。

节点管理器 ID：**`comfyui-drama3-t8`**；Publisher：**`t8star`**。需要 ComfyUI 0.40.0+、Python 3.10+。手动安装请使用 **ComfyUI 自带的 Python**：

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

重启 ComfyUI，导入 `workflows/` 的 JSON，选择模型和参考音频后点击**运行**。新增创作节点位于 `T8/Drama3/Creative`。先用 `workflow_creative_voice_project.json` 建立工程；续做现有工程用 `workflow_creative_project_resume.json`，导入完整工程副本用 `workflow_creative_bundle_resume.json`。重复 Create/Import 会新建工程，继续同一工程请用 Load。`takes_per_line` 是候选目标总数，从 3 改为 6 只补缺少的候选。

原有节点覆盖语音、参考编码、四种 Scene 模式（`together`、`per_line`、`per_line_context`、`per_speaker`）、长文、流式文本、Session 保存/回读/分支、导演改稿、局部重配、多轨编辑、SRT、转写及词对齐。参考台词与剧本使用一致的 `<|speaker:N|>`，最多 16 个编号。`T8DramaLoadSession` 输出 0 是完整 Session，输出 1 是可单独复用的 References。精确 64 位种子请用十进制 `seed_text`，最大 `18446744073709551615`。

### 依赖版本

支持 **Transformers 4.57.1 / tokenizers 0.22.2 / huggingface-hub 0.36.2**，Windows 目标环境为 Torch 2.8 + CUDA 12.8。其他 Transformers 版本不在覆盖范围。

```sh
python -m pip install "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
```

`install.py` 安装单独校验的 Windows x64 / CUDA 12.8 R2T2 运行库，不下载权重。设置 `DRAMA_SKIP_R2T2_INSTALL=1` 跳过，或用 `DRAMA_R2T2_NATIVE_DIR` 指向已有运行库。Qwen ASR 可独立使用；手动提供准确稿件的参考编码不需要 ASR 模型。

## 模型与自定义路径

转换权重见 [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy)。下表节点路径相对 `ComfyUI/`；Studio 将相同文件名/目录直接放在 `<整合包>/models/` 中。

| 文件 | ComfyUI 位置 |
|---|---|
| 主模型 FP8 / 可选 W4A8 | `models/diffusion_models/drama3-t8-fp8.safetensors` / `drama3-t8-w4a8.safetensors` |
| Codec | `models/vae/drama3-codec.safetensors` |
| 精简 Scene head | `models/diffusion_models/drama3-line-head.safetensors` |
| 完整文本/控制 head，兼作回退 | `models/diffusion_models/drama3-text-head.safetensors` |
| Qwen3-ASR 0.6B / 可选 1.7B | `models/asr/qwen3-asr-06b.safetensors` / `qwen3_asr_1.7b_bf16.safetensors` |
| ForcedAligner | `models/asr/qwen3_forced_aligner_0.6b_bf16.safetensors`，或完整 `models/asr/qwen-aligner/` |
| Confucius R2T2 模型 + 投影 | `models/asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf` + `mmproj-Confucius4-R2T2-Q8_0.gguf` |
| R2T2 长音频 FireRedVAD | `models/asr/FireRedVAD-ONNX/` |
| 可选 CAM++ 说话人模型 | Studio：`models/campplus-zh-en.onnx`，支持自定义路径 |

语音生成需要主模型、Codec 和选定/回退 head。ASR、对齐、VAD 与说话人模型在使用对应功能时检查。本次代码更新可继续使用已有 DRAMA 权重，CAM++ 为可选新增。

- **Studio：** 环境页面保存全部 9 项语音/转写模型路径；创作设置独立指定音色库、工作区和可选说话人模型。相对模型路径以整合包 `models/` 为基准，相对音色库/工作区路径以整合包根目录为基准。
- **ComfyUI：** 加载节点支持 `model_path`、`codec_path`、`text_head_path`、`line_head_path`；ASR/对齐单独指定路径，R2T2 可指定模型、投影和 VAD。创作节点支持音色库/工作区路径，默认位于插件之外的 ComfyUI 用户目录。`DRAMA3_VOICE_LIBRARY` 可指定原生库默认位置。
- 支持绝对路径、其他盘符与 Windows UNC 共享。自定义库和工作区请放在插件源码目录之外，以便保留用户资料。

## 加速与内存

现有设置包括 `gdn_backend=torch/fla`、`moe_backend=auto/torch/cuda`、CUDA Graph、RAM/VRAM 专家放置及 `expert_cache_gb`（`0` 自动）。专家缓存只是总占用的一部分；整模型显存驻留需要相关权重和 head 全部放得下。

安装 FlashAttention 或某个轮子本身不代表已经启用或更快。请查看实际后端/回退报告，在本机、当前模型上运行对照。后端不同可能改变音频，性能推荐仍需配合试听。大模型与主存专家缓存需要充足内存/显存，加载前会检查可用资源。Auto 遇到已核实的页锁 RAM 拒绝并完整清理后，最多独立用 Torch 重载一次；显式 CUDA 保留资源拒绝信息。

## 自动更新

GitHub 的 `Drama3-T8-Portable-<版本>-code.zip` **只含代码，不含 Python 运行时和模型**。完整整合包由作者单独提供；R2T2 原生运行库是单独校验的附件。

完整包提供 `Start-Drama3.exe`，`Start-Drama3.cmd` 继续可用，两者均在启动前检查 GitHub 稳定版 Release。代码更新 ZIP 不安装或替换 EXE 启动器。手动更新请先关闭 Studio，再运行 `Update-Drama3.cmd`。更新器核验大小/SHA-256，备份替换文件，事务失败自动回滚；保留模型、Python、`userdata`、输出、音色库/工作区与自定义路径。离线检查失败仍可启动，不会运行 pip；依赖变动需使用匹配的完整环境。离线时可继续使用原版。手动解压代码 ZIP 属于直接覆盖，不经过更新器的 SHA 校验和事务回滚；优先使用 `Update-Drama3.cmd`。

需要修复旧启动器时，先退出 Studio，再用[单独提供的 Start-Drama3.exe](https://github.com/T8mars/Comfyui-Drama-3/releases/download/v0.3.1/Start-Drama3.exe) 替换整合包根目录的同名文件。它依赖已有完整整合包，不是独立安装程序。

## 来源与许可证

基于 [Fish Speech](https://github.com/fishaudio/fish-speech)、[PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI) 及 [Confucius R2T2 T8](https://github.com/T8mars/comfyui-confucius-r2t2-t8)。本项目代码为 [GPL-3.0](LICENSE)。DRAMA 权重/材料适用 Fish Audio Community License v1.0；Qwen、FireRedVAD 与可选 CAM++ 为 Apache-2.0；Confucius 权重适用网易有道模型使用许可协议；llama.cpp 为 MIT。详见 [licenses/](licenses/)、[CAM++ 说明](licenses/NOTICE-CAMPPLUS.md) 和 [upstream-lock.json](upstream-lock.json)。上游许可继续适用，不表示获得上游背书。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
