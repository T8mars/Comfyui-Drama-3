"""Generate importable ComfyUI canvas examples; never export API prompts."""
import json
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = (ROOT / "comfyui-drama3-T8/workflows" if (ROOT / "comfyui-drama3-T8").is_dir()
          else ROOT / "workflows")
ENGINE = "DRAMA3_T8_ENGINE"
SESSION = "DRAMA3_T8_SESSION"
REFERENCES = "DRAMA3_T8_REFERENCES"
CODES = "DRAMA3_T8_CODES"

# Values are addressed by input name, never by their position in define_schema.
# Required/optional groups returned by GET_NODE_INFO_V1 determine frontend order.
EXAMPLES = {
    "T8DramaLoader": {"model": "drama3-t8-fp8.safetensors", "codec": "drama3-codec.safetensors", "expert_placement": "ram"},
    "T8DramaGenerate": {"mode": "speech", "text": "你好，这是本地语音测试。", "max_frames": 180},
    "T8DramaScene": {"script": "Anna: Ready?\nBen: Always.", "scene": "Two friends begin a journey.",
                     "characters": "- Anna: a calm young woman.\n- Ben: a cheerful young man.", "seed": 44, "max_frames": 180},
    "T8DramaInfill": {"target_transcript": "您好，这是本地语音测试。", "end_seconds": .5, "seed": 45, "max_frames": 180},
    "T8DramaSaveSession": {"name": "drama3_example"},
    "T8DramaLoadSession": {"session_file": ""},
    "T8DramaTranscribe": {"asr_model": "qwen3-asr-06b.safetensors", "language": "Chinese"},
    "T8DramaSRT": {"srt": "1\n00:00:00,000 --> 00:00:02,000\n你好。\n\n2\n00:00:03,000 --> 00:00:05,000\n再见。\n", "seed": 55, "max_frames": 180},
    "T8DramaAlign": {"aligner": "qwen3_forced_aligner_0.6b_bf16.safetensors", "transcript": "你好，这是本地语音测试。", "language": "Chinese",
                     "model_path": "asr/qwen3_forced_aligner_0.6b_bf16.safetensors"},
    "T8DramaR2T2Transcribe": {"language": "Chinese"},
    "T8DramaLive": {"text": "你好，现在是浏览器实时试听测试。", "preview_audio": True, "seed": 67, "max_frames": 180},
    "T8DramaEditor": {"aligner_path": "asr/qwen3_forced_aligner_0.6b_bf16.safetensors", "align_words": False},
    "LoadAudio": {"audio": ""},
}


# Public example asset selectors are stable even before those files are installed.
# User-selected audio/session inputs stay empty rather than embedding local files.
PUBLIC_FILE_SELECTORS = {
    ("T8DramaLoader", "model"), ("T8DramaLoader", "codec"),
    ("T8DramaTranscribe", "asr_model"), ("T8DramaAlign", "aligner"),
}
EMPTY_FILE_SELECTORS = {("LoadAudio", "audio"), ("T8DramaLoadSession", "session_file")}


@lru_cache(maxsize=1)
def node_infos():
    """Read actual host-adapted metadata without starting a server or loading models."""
    # Keep the host before this project's app.py: recent ComfyUI imports its
    # own app.governance package while defining built-in audio/text nodes.
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    try:
        import folder_paths  # noqa: F401
    except ImportError:
        from drama3_t8.bootstrap import standalone_engine
        standalone_engine(ROOT)
    from drama3_t8.nodes import NODES
    from comfy_extras.nodes_audio import LoadAudio, SaveAudio
    from comfy_extras.nodes_text import SaveTextNode
    from comfy_extras.nodes_preview_any import PreviewAny
    infos = {cls.__name__: cls.GET_NODE_INFO_V1() for cls in NODES}
    for cls in (LoadAudio, SaveAudio, SaveTextNode):
        info = cls.GET_NODE_INFO_V1()
        infos[info["name"]] = info
    infos["PreviewAny"] = {"input": PreviewAny.INPUT_TYPES(), "output": list(PreviewAny.RETURN_TYPES)}
    # Note is a frontend-only built-in, with one serializable multiline widget.
    infos["Note"] = {"input": {"required": {"text": ["STRING", {"default": "", "multiline": True}]}}, "output": []}
    return infos


def input_specs(info):
    for group in ("required", "optional"):
        inputs = info["input"].get(group, {})
        order = info.get("input_order", {}).get(group, list(inputs))
        for name in order:
            yield name, inputs[name]


def is_widget(spec):
    kind, options = spec[0], spec[1] if len(spec) > 1 else {}
    return not options.get("forceInput", False) and (isinstance(kind, list) or kind in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"})


def widget_names(info):
    names = []
    for name, spec in input_specs(info):
        if not is_widget(spec):
            continue
        names.append(name)
        options = spec[1] if len(spec) > 1 else {}
        # Legacy canvas arrays include the frontend's linked seed control.
        # seed-compat.js also handles hosts that mark it nonserializable.
        if spec[0] == "INT" and options.get("control_after_generate", name in {"seed", "noise_seed"}):
            names.append("control_after_generate")
    return names


def widget_values(node_type, values=None):
    info = node_infos()[node_type]
    wanted = {**EXAMPLES.get(node_type, {}), **(values or {})}
    allowed = set(widget_names(info))
    if unknown := set(wanted) - allowed:
        raise ValueError(f"Unknown widget input for {node_type}: {sorted(unknown)}")
    result = {}
    for name, spec in input_specs(info):
        if not is_widget(spec):
            continue
        kind, options = spec[0], spec[1] if len(spec) > 1 else {}
        choices = kind if isinstance(kind, list) else options.get("options", [])
        default = options.get("default", choices[0] if choices else False if kind == "BOOLEAN" else 0 if kind in {"INT", "FLOAT"} else "")
        value = wanted.get(name, default)
        # Do not make a distributable canvas depend on the author's file list.
        # Only known public example files and intentionally empty selectors may
        # be absent; ordinary enums and explicit unknown files remain validated.
        selector = (node_type, name)
        public_file = selector in PUBLIC_FILE_SELECTORS and value == EXAMPLES[node_type][name]
        requires_selection = selector in EMPTY_FILE_SELECTORS and value == ""
        if choices and value not in choices and not (public_file or requires_selection):
            if name in (values or {}):
                raise ValueError(f"Invalid combo value for {node_type}.{name}: {value}")
            value = default
        result[name] = value
    result["control_after_generate"] = wanted.get("control_after_generate", "fixed")
    return [deepcopy(result[name]) for name in widget_names(info)]


class Canvas:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []

    def add(self, node_type, column, row=0, widgets=None, title=None):
        info = node_infos()[node_type]
        inputs = [(name, spec[0]) for name, spec in input_specs(info) if not is_widget(spec)]
        outputs = info.get("output", [])
        defaults = widget_values(node_type, widgets)
        size = {"T8DramaLoader": [370, 430], "T8DramaGenerate": [370, 540],
                "T8DramaScene": [410, 600], "T8DramaEditor": [540, 720]}.get(
                    node_type, [370, 380 if len(defaults) > 4 else 260])
        node = {
            "id": len(self.nodes) + 1, "type": node_type,
            "pos": [60 + column * 420, 60 + row * 460],
            "size": size,
            "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": name, "type": kind, "link": None} for name, kind in inputs],
            "outputs": [{"name": info.get("output_name", outputs)[index], "type": kind, "links": []} for index, kind in enumerate(outputs)],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": defaults,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def connect(self, source, target, input_name, output=0):
        slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == input_name)
        socket = target["inputs"][slot]
        if socket["link"] is not None:
            raise ValueError(f"Input already connected: {target['type']}.{input_name}")
        if socket["type"] != "*" and source["outputs"][output]["type"] != socket["type"]:
            raise ValueError(f"Socket type mismatch: {source['type']} -> {target['type']}.{input_name}")
        link_id = len(self.links) + 1
        self.links.append([link_id, source["id"], output, target["id"], slot, source["outputs"][output]["type"]])
        source["outputs"][output]["links"].append(link_id)
        socket["link"] = link_id

    def note(self, text, column=0, row=2):
        self.add("Note", column, row, {"text": text}, "使用说明 / Instructions")

    def audio_output(self, engine, session, column=3, row=0):
        decode = self.add("T8DramaDecode", column, row)
        self.connect(engine, decode, "engine")
        self.connect(session, decode, "session")
        self.save_audio(decode, column + 1, row)
        preview = self.add("PreviewAny", column, row + 1, title="生成台词 / Generated transcript")
        self.connect(decode, preview, "source", 1)
        return decode

    def save_audio(self, source, column, row=0, output=0):
        save = self.add("SaveAudio", column, row, {"filename_prefix": f"Drama3_T8/{self.name}"})
        self.connect(source, save, "audio", output)
        return save

    def export(self):
        seeded = {"T8DramaGenerate", "T8DramaScene", "T8DramaInfill", "T8DramaSRT", "T8DramaLive"}
        if any(node["type"] in seeded for node in self.nodes):
            self.note("seed 数字控件支持 JavaScript 安全整数；更大的 uint64 种子请填 seed_text 十进制文本。\nUse decimal seed_text for larger uint64 seeds; leave it blank to use the numeric seed widget.", 3, 2)
        return {"last_node_id": len(self.nodes), "last_link_id": len(self.links),
                "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
                "extra": {"ds": {"scale": .75, "offset": [40, 40]}}, "version": .4}


def speech_graph(name, mode="speech"):
    graph = Canvas(name)
    engine = graph.add("T8DramaLoader", 0)
    session = graph.add("T8DramaSession", 0, 1)
    widgets = {"mode": mode}
    if mode == "long":
        widgets["text"] = "你好，这是长文分段测试。\n第二段会沿用前一段的语音历史。"
        widgets["chunk_bytes"] = 48
    elif mode == "stream":
        widgets["text"] = "你好，这是流式文本测试。"
        widgets["lookahead"] = "接下来会继续介绍本地演出工作台。"
    generate = graph.add("T8DramaGenerate", 1, widgets=widgets)
    graph.connect(engine, generate, "engine")
    graph.connect(session, generate, "session")
    return graph, engine, session, generate


def make_workflows():
    workflows = {}
    for name, mode in (("basic", "speech"), ("long", "long"), ("stream", "stream")):
        graph, engine, _, generate = speech_graph(name, mode)
        decode = graph.audio_output(engine, generate, 2)
        if mode == "long":
            decode["widgets_values"] = widget_values("T8DramaDecode", {"all_takes": True})
        workflows[name] = graph.export()

    graph, engine, _, generate = speech_graph("revision")
    widgets = {"mode": "revision", "text": "Please say the previous sentence in a cheerful, warm tone.", "seed": 43}
    revise = graph.add("T8DramaGenerate", 2, widgets=widgets)
    graph.connect(engine, revise, "engine")
    graph.connect(generate, revise, "session")
    graph.audio_output(engine, revise, 3)
    graph.note("先生成原始台词，再基于真实音频历史修改表演。\nThe revision node changes the preceding performance using its audio history.")
    workflows[graph.name] = graph.export()

    graph = Canvas("reference")
    engine = graph.add("T8DramaLoader", 0)
    audio = graph.add("LoadAudio", 0, 1, title="选择参考音频 / Choose reference audio")
    reference = graph.add("T8DramaReference", 1, widgets={"transcript": "请填写参考音频中实际说出的文字。", "name": "Narrator"})
    session = graph.add("T8DramaSession", 2)
    generate = graph.add("T8DramaGenerate", 3)
    graph.connect(engine, reference, "engine")
    graph.connect(audio, reference, "audio")
    graph.connect(reference, session, "references")
    save_voice = graph.add("T8DramaSaveSession", 2, 1, {"name": "drama3_voice_library"},
                           title="保存参考音色 / Save reusable references")
    graph.connect(session, save_voice, "session")
    graph.connect(engine, generate, "engine")
    graph.connect(session, generate, "session")
    graph.audio_output(engine, generate, 4)
    graph.note("在 LoadAudio 上传参考音频，并把 transcript 改为真实台词；角色绑定用转写前的 <|speaker:0|> / <|speaker:1|>，name 仅显示。\nUpload audio and enter its exact transcript with matching speaker tags; name is display-only.\n多参考：将前一个 references 输出接下一个 previous。保存音色分支写入仅含 references 的会话，再用 workflow_load_session.json 复用。\nChain references into previous for multiple voices. Save reusable references, then open workflow_load_session.json.", 0, 2)
    workflows[graph.name] = graph.export()

    graph = Canvas("scene")
    engine = graph.add("T8DramaLoader", 0)
    scene = graph.add("T8DramaScene", 1, widgets={"split_long_text":True,"chunk_bytes":300})
    graph.connect(engine, scene, "engine")
    graph.audio_output(engine, scene, 2)
    graph.note("可选 session 输入保留导演指令、参考音色与历史；references 未连接时继承会话参考，连接后替换。\nConnect session for direction, voices and history; leave references unconnected to inherit the Session voices.\ngeneration_mode：together 同场生成；per_line 逐句；per_line_context 逐句听前句；per_speaker 按角色生成后还原剧本顺序。\nSelect any of the four generation modes in the scene node.\nsplit_long_text 可用于全部四模式；chunk_bytes 限每次新剧本 UTF-8 字节（含标签/分隔符），历史另受模型上下文限制。同原台词分块间不加静音。\nEnable split_long_text for all modes; chunk_bytes includes new-script tags/separators. No extra gap between chunks of one original line.\n参考转写前加 <|speaker:0|> / <|speaker:1|>；剧本使用相同标签，或让 Anna / Ben 按首次出现顺序映射到 0 / 1。name 只作显示，不绑定音色。\nPrefix reference transcripts with matching speaker:0 / speaker:1 tags. Use the same script tags, or first-seen Anna / Ben mapping to 0 / 1. Reference name is display-only.\n加载器支持 FP8、INT8、W6、W4；可填写 model_path / codec_path 自定义位置。expert_placement 默认 ram；vram 要求完整单卡驻留，显存不足明确拒绝。moe_backend / cuda_graph 记录实际后端与回退。\nChoose your checkpoint and custom paths. expert_placement=ram uses a cache; vram requires full single-GPU residency and rejects insufficient memory. Kernel and Graph fallbacks are recorded separately.", 0, 2)
    workflows[graph.name] = graph.export()

    graph, engine, _, generate = speech_graph("infill")
    infill = graph.add("T8DramaInfill", 2)
    graph.connect(engine, infill, "engine")
    graph.connect(generate, infill, "session")
    graph.audio_output(engine, infill, 3)
    graph.note("先试听原句，再按实际音频调整 start_seconds/end_seconds，仅框选要替换的词；0–0.5 秒仅为示例。\nListen to the original and select the actual word interval; 0–0.5 seconds is only an example.\ntarget_transcript 为编辑后整句；本例仅将开头“你好”改为“您好”，保留后文。\nThe complete edited transcript changes only the opening greeting and keeps the remainder.\nduration_frames=0 使用模型生成长度；正值指定替换帧数。\nUse 0 for generated duration, or a positive frame count.\n源感知场景按实际播放时间补写；跨角色/静音选区设置 target_speaker。expected_view 可填已试听资产身份JSON，防止误用旧选区。\nFor source-aware scenes, choose target_speaker for cross-role/silence and optionally bind expected_view. Session/AUDIO retain FLOAT; the standard FLAC output is a playback export.")
    workflows[graph.name] = graph.export()

    graph = Canvas("srt")
    engine = graph.add("T8DramaLoader", 0)
    srt = graph.add("T8DramaSRT", 1)
    graph.connect(engine, srt, "engine")
    graph.save_audio(srt, 2, output=1)
    preview = graph.add("PreviewAny", 2, 1, title="实际字幕时间 / Actual subtitle timing")
    graph.connect(srt, preview, "source", 2)
    workflows[graph.name] = graph.export()

    graph = Canvas("codec")
    engine = graph.add("T8DramaLoader", 0)
    audio = graph.add("LoadAudio", 0, 1, title="选择音频 / Choose audio")
    encode = graph.add("T8DramaCodecEncode", 1)
    decode = graph.add("T8DramaCodecDecode", 2)
    graph.connect(engine, encode, "engine")
    graph.connect(audio, encode, "audio")
    graph.connect(engine, decode, "engine")
    graph.connect(encode, decode, "codes")
    graph.save_audio(decode, 3)
    workflows[graph.name] = graph.export()

    for name, node_type in (("asr", "T8DramaTranscribe"), ("align", "T8DramaAlign"), ("r2t2", "T8DramaR2T2Transcribe")):
        graph = Canvas(name)
        audio = graph.add("LoadAudio", 0, title="上传音频 / Upload audio")
        transcribe = graph.add(node_type, 1)
        graph.connect(audio, transcribe, "audio")
        labels = ["对齐词与时间 / Aligned words and timestamps"] if name == "align" else ["转写文字 / Transcript", "识别语言 / Language"]
        if name == "r2t2":
            labels.append("分段与元数据 / Segments and metadata")
        for output, label in enumerate(labels):
            preview = graph.add("PreviewAny", 2, output, title=label)
            graph.connect(transcribe, preview, "source", output)
        text = "在 LoadAudio 上传或选择本地音频；运行后，在文字预览节点中查看结果。\nUpload or select audio in LoadAudio; after Run, read the results in the Preview as Text nodes."
        if name == "align":
            text += "\ntranscript 必须与音频中实际说出的文字对应。\nReplace transcript with the words actually spoken in the audio."
            text += "\nmodel_path 默认演示单文件 asr/qwen3_forced_aligner_0.6b_bf16.safetensors（相对 models）；也可填写原版模型目录的绝对路径。\nThe custom path accepts a native single-file aligner or the original model directory."
        elif name == "r2t2":
            text += "\n模型、投影器和 VAD 路径相对于 ComfyUI/models，也可填写绝对路径。\nModel, projector and VAD paths accept absolute paths or paths relative to ComfyUI/models."
        graph.note(text)
        workflows[graph.name] = graph.export()

    graph = Canvas("live")
    engine = graph.add("T8DramaLoader", 0)
    session = graph.add("T8DramaSession", 0, 1)
    live = graph.add("T8DramaLive", 1)
    graph.connect(engine, live, "engine")
    graph.connect(session, live, "session")
    graph.save_audio(live, 2, output=1)
    graph.note("preview_audio=true 时，先在画布上点击一次，再点击运行；浏览器需要鼠标操作来启用声音。\nWith preview_audio enabled, click the canvas before Run to unlock browser audio.\n分片在生成过程中播放；完整音频在结束后保存。\nChunks play during generation; the final audio is saved at completion.", 1, 1)
    workflows[graph.name] = graph.export()

    graph, engine, _, generate = speech_graph("save_session")
    branch = graph.add("T8DramaBranch", 2)
    graph.connect(generate, branch, "session")
    graph.audio_output(engine, branch, 3)
    save = graph.add("T8DramaSaveSession", 2, 1)
    graph.connect(branch, save, "session")
    graph.note("运行后在 output/drama3_t8_sessions/ 保存会话；take_count 选择保留前几个 take。恢复音频时，将 LoadSession 的第 0 个 Session 输出接 Decode。\nSave under output/drama3_t8_sessions/. take_count retains the first N takes. Reconnect LoadSession output 0 (Session) to Decode to restore audio.\nworkflow_load_session.json 演示第 1 个 References 输出复用已存音色；请先在 workflow_reference.json 保存参考。\nworkflow_load_session.json uses output 1 (References) for reusable voices saved by workflow_reference.json.", 1, 2)
    workflows[graph.name] = graph.export()

    graph = Canvas("load_session")
    engine = graph.add("T8DramaLoader", 0)
    session = graph.add("T8DramaLoadSession", 0, 1, title="选择已存音色 / Select saved voice references")
    scene = graph.add("T8DramaScene", 1)
    graph.connect(engine, scene, "engine")
    graph.connect(session, scene, "references", 1)
    graph.audio_output(engine, scene, 2)
    graph.note("先在 workflow_reference.json 编码并保存参考音色，刷新后选择含 references 的会话。第 1 个输出 References 接 Scene，按转写 speaker 标签复用；name 不参与绑定。\nSave references with workflow_reference.json, refresh, then select that voice-library session. Output 1 (References) feeds Scene; transcript speaker tags bind voices, not names.\n输出 0 仍是完整 Session，可接旧 Generate / Decode / Editor 连线。纯音色会话没有 take，不应直接接 Decode。\nOutput 0 remains Session for existing Generate/Decode/Editor links. Reference-only archives have no take to decode.\n本画布不附带私人音频或会话。\nNo private audio or sessions are distributed.", 1, 2)
    workflows[graph.name] = graph.export()

    graph = Canvas("editor")
    engine = graph.add("T8DramaLoader", 0)
    session = graph.add("T8DramaLoadSession", 0, 1, title="选择已保存的会话 / Select a saved session")
    editor = graph.add("T8DramaEditor", 1, title="多轨剪辑与字幕 / Multitrack editor and captions")
    graph.connect(engine, editor, "engine")
    graph.connect(session, editor, "session")
    graph.save_audio(editor, 3, output=1)
    save = graph.add("T8DramaSaveSession", 3, 1, {"name": "drama3_edited"})
    graph.connect(editor, save, "session", 0)
    subtitle = graph.add("PreviewAny", 4, title="剪辑后的字幕 / Edited subtitle timing")
    graph.connect(editor, subtitle, "source", 2)
    text_save = graph.add("SaveText", 4, 1, {"filename_prefix": "Drama3_T8/editor_subtitles", "format": "txt"},
                          title="保存 SRT 文本 / Save SRT text")
    graph.connect(editor, text_save, "text", 2)
    graph.note("先运行 workflow_save_session.json；刷新页面并选择会话，再运行本画布加载波形。点击编辑器展开，剪切/插入停顿/调整角色音量与静音，应用后重新运行以保存。\nSave a session first, select it after refresh, and Run to load the waveform. Open the editor, adjust cuts/pauses/voice gain or mute, then apply and Run to save.\n默认保留原始音频；align_words=false 使用句级字幕。逐词字幕须填好单文件或目录 aligner_path，再启用 align_words。\nAudio is unchanged by default. Set your aligner path and enable align_words for word timing.\nSaveSession 保留非破坏剪辑；字幕保存为 .txt（内容为 SRT，可改为 .srt）。所有模型与用户音频均不随工作流分发。\nSaveSession preserves nondestructive edits; the saved .txt contains SRT and can be renamed .srt.", 0, 2)
    workflows[graph.name] = graph.export()
    workflows.update(make_creative_workflows())
    return workflows


def _creative_connect_widget(graph, source, target, name, output=0):
    spec = dict(input_specs(node_infos()[target["type"]]))[name]
    assert is_widget(spec) and spec[0] == source["outputs"][output]["type"] == "STRING"
    target["inputs"].append({"name": name, "type": "STRING", "link": None, "widget": {"name": name}})
    graph.connect(source, target, name, output)


def _creative_finishing(graph, batch, column, row=0):
    delivery = graph.add("T8DramaCreativeDelivery", column, row,
        {"sample_rate": "48000", "duration_mode": "report", "loudness_mode": "none", "gap_seconds": .15},
        "交付 ZIP：WAV / SRT / 清单 · Delivery package")
    graph.connect(batch, delivery, "project")
    exported = graph.add("T8DramaCreativeProjectExport", column, row + 1,
        {"include_all": True, "include_sources": True}, "完整工程 + 音色素材 · Portable project")
    graph.connect(batch, exported, "project")
    return delivery, exported


def make_creative_workflows():
    graphs = {}
    graph = Canvas("creative_voice_project")
    engine = graph.add("T8DramaLoader", 0)
    audio = graph.add("LoadAudio", 0, 1)
    reference = graph.add("T8DramaReference", 1, 0,
        {"transcript": "请替换为参考录音中实际说出的原文。", "name": "演示主角", "trim_silence": False, "normalize": False})
    graph.connect(engine, reference, "engine")
    graph.connect(audio, reference, "audio")
    session = graph.add("T8DramaSession", 1, 1)
    graph.connect(reference, session, "references")
    saved = graph.add("T8DramaVoiceLibrarySave", 2, 0,
        {"mode": "new_voice", "name": "演示主角", "language": "zh", "delivery": "自然"})
    graph.connect(session, saved, "session")
    loaded = graph.add("T8DramaVoiceLibraryLoad", 2, 1, {"speaker": 0}, "回读刚保存的不可变版本 · Load exact version")
    _creative_connect_widget(graph, saved, loaded, "voice", 0)
    _creative_connect_widget(graph, saved, loaded, "version_id", 1)
    archive = graph.add("T8DramaSaveSession", 3, 1, {"name": "Drama3_T8/creative_reference"})
    graph.connect(loaded, archive, "session", 1)
    cast = graph.add("T8DramaVoiceCastBuild", 3, 0, {"role": "主角", "speaker": 0})
    _creative_connect_widget(graph, saved, cast, "voice", 0)
    _creative_connect_widget(graph, saved, cast, "version_id", 1)
    created = graph.add("T8DramaCreativeProjectCreate", 4, 0,
        {"name": "下班研究所", "lines_json": json.dumps([
            {"line_id": "L01", "role": "主角", "original_text": "老板说，年轻人要有梦想。我说，我的梦想是准点下班。"},
            {"line_id": "L02", "role": "主角", "original_text": "今天不是摸鱼，是给大脑做液冷散热。"}], ensure_ascii=False),
         "dictionary_json": json.dumps({"number_mode": "none", "expand_dates": False})})
    _creative_connect_widget(graph, cast, created, "cast_json")
    batch = graph.add("T8DramaCreativeBatch", 5, 0,
        {"takes_per_line": 3, "seed_text": "42", "generation_mode": "per_line_context",
         "sampling_json": '{"temperature":0.7,"top_p":0.7,"top_k":30,"max_frames":600}'})
    graph.connect(engine, batch, "engine")
    graph.connect(created, batch, "project")
    _creative_finishing(graph, batch, 6)
    graph.note("先选参考录音并填写准确原文，再运行。新建音色会保存独立UUID和版本，连线使用该准确版本。\n"
               "默认用户目录保存工程/音色，资料不随插件更新覆盖；自定义时请统一修改所有library_path。\n"
               "Batch每句3候选，默认先选首个有效Take；试听/评分/选片可在Studio继续。\n"
               "右侧交付ZIP含成片/分句/字幕；完整工程ZIP另含Session和实际音色素材。\n"
               "续做请打开workflow_creative_bundle_resume / workflow_creative_project_resume，保留工程ID、模型、参数和seed_text，不要反复新建。\n"
               "Choose reference audio and enter its exact transcript. Keep model/seed/options unchanged to resume.\n"
               "Voice and project data live outside the plugin; delivery and complete project are separate ZIPs.", 1, 2)
    graphs[graph.name] = graph.export()

    graph = Canvas("creative_bundle_resume")
    engine = graph.add("T8DramaLoader", 0)
    imported = graph.add("T8DramaCreativeProjectImport", 0, 1,
        {"bundle_path": ""}, "选择完整工程ZIP · Import portable project")
    batch = graph.add("T8DramaCreativeBatch", 1, 0,
        {"takes_per_line": 3, "seed_text": "42", "generation_mode": "per_line_context",
         "sampling_json": '{"temperature":0.7,"top_p":0.7,"top_k":30,"max_frames":600}'})
    graph.connect(engine, batch, "engine")
    graph.connect(imported, batch, "project")
    _creative_finishing(graph, batch, 2)
    graph.note("bundle_path填写完整工程ZIP的本机路径（Studio或本组其他画布导出），不是普通交付ZIP。\n"
               "导入新工程并保留原音色UUID/版本；素材放项目私有库，不覆盖全局同名音色。\n"
               "续做时保持模型与生成参数、模式、seed_text一致；只有指纹匹配且媒体SHA完整的候选会跳过。\n"
               "若原工程不是per_line_context，请把Batch模式/参数改成原工程值；takes_per_line是目标总数。\n"
               "导入只做一次。再次打开已有工程请用workflow_creative_project_resume，避免重新导入出新副本。\n"
               "Import a complete project ZIP, then match original model/options/seeds for resume.\n"
               "The imported voice assets stay in this project's private catalog. Use workflow_creative_project_resume to reopen it later.", 0, 2)
    graphs[graph.name] = graph.export()

    graph = Canvas("creative_project_resume")
    engine = graph.add("T8DramaLoader", 0)
    loaded = graph.add("T8DramaCreativeProjectLoad", 0, 1,
        {"project_id": ""}, "填写已有工程ID · Open existing project")
    batch = graph.add("T8DramaCreativeBatch", 1, 0,
        {"takes_per_line": 3, "seed_text": "42", "generation_mode": "per_line_context",
         "sampling_json": '{"temperature":0.7,"top_p":0.7,"top_k":30,"max_frames":600}'})
    graph.connect(engine, batch, "engine")
    graph.connect(loaded, batch, "project")
    _creative_finishing(graph, batch, 2)
    graph.note("从Studio工程信息或Create/Import返回JSON复制32位project_id；workspace_path与原工程目录一致。\n"
               "Library路径为空使用ComfyUI用户目录；导入工程自动使用其私有库。\n"
               "保持原模型/参数/seed_text即可跳过匹配完成项；takes_per_line从3改6只补新增候选。\n"
               "可在line_ids_json填[\"L02\"]只续某句；复杂场景/按角色模式仍以场景/角色为原子检查点。\n"
               "出现失败后保留旧候选，修正资源或输入再运行。不会用文件存在冒充生成完成。\n"
               "Paste the saved project_id and use its workspace_path. Match original model/options to reuse intact takes.\n"
               "Raise the target take count to add candidates; line_ids_json limits work to selected lines.", 0, 2)
    graphs[graph.name] = graph.export()

    return graphs


def write_workflows(folder=FOLDER):
    folder.mkdir(parents=True, exist_ok=True)
    workflows = make_workflows()
    for name, workflow in workflows.items():
        (folder / f"workflow_{name}.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return workflows


if __name__ == "__main__":
    print(f"Wrote {len(write_workflows())} canvas workflows; no API workflows.")
