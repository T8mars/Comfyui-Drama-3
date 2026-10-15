# ComfyUI-Drama3-T8

[English](README.md) | **简体中文**

T8star 制作的 [Fish Audio DRAMA-3](https://huggingface.co/fishaudio/drama-3) 原生节点与本地演出工作台。Built with Fish Audio.

[模型下载](https://huggingface.co/t8star/Drama-3-Comfy) · [版本下载](https://github.com/T8mars/Comfyui-Drama-3/releases) · [示例工作流](https://github.com/T8mars/Comfyui-Drama-3/tree/main/workflows) · [问题反馈](https://github.com/T8mars/Comfyui-Drama-3/issues)

## 功能

- 多参考音色克隆、最多 16 个角色编号、行内演出指令与多角色场景。
- 基于真实音频历史的导演修改、局部编辑、长文、SRT 与带下段预览的流式文本输入。
- Codec 编解码、参考复用、工程保存/回读/分支，以及生成中的音频分包试听。
- 本地 Qwen3-ASR、Qwen ForcedAligner 和 [Confucius R2T2 转写](https://github.com/T8mars/comfyui-confucius-r2t2-t8)，均支持自定义模型路径。
- 17 个 ComfyUI V3 原生节点，输出标准 `AUDIO`；模型、Codec 与 R2T2 均在宿主进程内执行。独立白粉工作台共用推理核心。

## 安装节点

在 ComfyUI-Manager 中搜索 **ComfyUI-Drama3-T8**，发布者 **t8star**。也可手动安装：

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Drama-3.git
cd Comfyui-Drama-3
python -m pip install -r requirements.txt
python install.py
```

请使用 **ComfyUI 自己的 Python** 执行，然后重启。导入 `workflows/` 中的示例，选择本地模型与参考音频。要求 ComfyUI 0.40.0+、Python 3.10+。`install.py` 下载并校验单独分发的 Windows x64 / CUDA 12.8 R2T2 原生运行库，不下载模型。设置 `DRAMA_SKIP_R2T2_INSTALL=1` 可跳过；已有运行库可设置 `DRAMA_R2T2_NATIVE_DIR`。Qwen 转写独立可用。

## Transformers 兼容性

支持 **Transformers 4.57.1**，已有兼容的 4.x 环境可继续使用。节点依赖范围为 `transformers>=4.57.1,<6`、`tokenizers>=0.22,<0.24`。

| 已验证的 Transformers | Tokenizers | Hugging Face Hub |
|---|---|---|
| **4.57.1** | **0.22.2** | **0.36.2** |
| 5.19.0 | 0.23.2 | 1.33.0 |

整合包的 `requirements.txt` 默认安装 4.57.1 这组依赖；已有 5.19 环境也可继续使用。两组均通过完整 CPU 回归，包含真实 Qwen 模型前向与缓存续算；4.57.1 另经过浏览器画布生成、转写和词级对齐实测。允许范围内的其他版本尚未逐版验证。

需要切换到已测 4.57.1 组合时，使用 ComfyUI 或工作台自己的 Python 执行：

```sh
python -m pip install "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
```

程序自动更新会保留已安装的 Python 包，不运行 pip，也不自动切换 Transformers 版本；本次兼容更新无需重新下载模型。

## 画布工作流

`workflows/` 提供 15 张画布：普通生成、参考音色、导演修改、局部编辑、长文、流式文本、场景、SRT、Codec、Qwen 转写、R2T2、对齐、实时试听、工程保存与加载。将 JSON 导入 ComfyUI，点击 **运行**。示例均经过浏览器画布实跑，核对了文字显示和音频播放。

在 `LoadAudio` 中选择自己的音频；参考与对齐节点的台词须匹配录音。加载工程前先运行 `workflow_save_session.json`，刷新 ComfyUI，再选择已保存的文件。长文默认输出全部段落。数字种子范围为 `0–9007199254740991`；更大的种子请在 `seed_text` 填写完整十进制数，最大 `18446744073709551615`。

带参考的场景会按首次出现顺序为角色分配稳定的 `<|speaker:N|>` 编号，连续改稿沿用原映射。参考名称仅作显示；需要指定角色音色时，在参考转写与脚本中使用相同的显式 speaker 标签。

## 模型

从 [t8star/Drama-3-Comfy](https://huggingface.co/t8star/Drama-3-Comfy) 下载。默认路径相对于 `ComfyUI/`：

| 模型 | 默认位置 |
|---|---|
| DRAMA FP8，38.58 GB | `models/diffusion_models/drama3-t8-fp8.safetensors` |
| 完整文本/控制输出头，1.03 GB | `models/diffusion_models/drama3-text-head.safetensors` |
| Codec，1.57 GB | `models/vae/drama3-codec.safetensors` |
| Qwen 转写，可选 | `models/asr/qwen3-asr-06b.safetensors` |
| 词级对齐，可选，下载完整目录 | `models/asr/qwen-aligner/` |
| R2T2 Q8 主模型与投影器，可选 | `models/asr/Confucius4-R2T2-GGUF/` |
| R2T2 分段使用的 FireRedVAD | `models/asr/FireRedVAD-ONNX/` |

完整 DRAMA 功能需保留前三个文件。各加载器均可填写文件或目录的自定义路径，支持跨磁盘存放；工作台设置保存在 `userdata/` 中。

## 整合包与自动更新

完整 Windows 运行环境由作者单独提供。GitHub Release **只发布程序文件及原生运行库，不含模型和 Python 运行环境**；工作台源码位于 [portable/](https://github.com/T8mars/Comfyui-Drama-3/tree/main/portable)。

已有基于 0.1.3 或后续版本的完整包：先关闭工作台，将 `Drama3-T8-Portable-0.1.11-code.zip` 解压到包含 `Start-Drama3.cmd` 的目录，再双击启动。此后启动时自动检查 GitHub 正式版并安装兼容的程序更新；关闭工作台后也可双击 `Update-Drama3.cmd` 手动检查。更新核对文件大小与 SHA-256，备份旧文件，失败时回滚。模型、Python、自定义路径、工程、音色库和输出均保留。断网仍可正常启动；依赖或运行库变更需使用作者提供的新完整环境。

## 硬件与加速

实测 Windows x64、RTX 5090 Laptop **24 GB 显存 / 64 GB 内存**、Torch 2.8 / CUDA 12.8，专家缓存 12 GiB。FP8 专家驻系统内存，缓存 `0` 自动预算；显存需为驻留权重、Codec 与上下文留出空间。其他硬件组合尚未实测。

默认 Torch 线性注意力与 SDPA。FLA 已接入，本机测试未见明显提速；安装 FlashAttention 不代表模型已启用，CUDA Graph 当前关闭。CPU 试听分包会在生成过程中送达，本机卸载配置低于音频实时生成速度。转写结果可能需要人工校对。场景改稿保留完整工程历史，并对提前结束的台词进行有限边界恢复；空段、多段或未完成的输出仍会被拒绝。

## 来源与许可

基于 [Fish Speech](https://github.com/fishaudio/fish-speech)、[PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI) 及上述 R2T2 项目。T8 补充完整控制输出、有状态编辑、流式、原生转写集成与本地工作台。

项目代码遵循 [GPL-3.0](LICENSE)，上游组件保留原许可。DRAMA 材料及权重遵循 **Fish Audio Community License v1.0**；Qwen、FireRedVAD 为 Apache-2.0，Confucius 模型遵循网易有道模型使用许可，llama.cpp 为 MIT。完整文本见 [licenses/](licenses/)，固定来源版本见 [upstream-lock.json](upstream-lock.json)。本项目未获上游作者背书，亦非其官方产品。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线AI应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
