"""Generate importable ComfyUI canvas examples; never export API prompts."""
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = (ROOT / "comfyui-drama3-T8/workflows" if (ROOT / "comfyui-drama3-T8").is_dir()
          else ROOT / "workflows")
ENGINE = "DRAMA3_T8_ENGINE"
SESSION = "DRAMA3_T8_SESSION"
REFERENCES = "DRAMA3_T8_REFERENCES"
CODES = "DRAMA3_T8_CODES"

# Socket order and widget order follow drama3_t8/nodes.py and the ComfyUI frontend.
# Seed widgets have an additional, frontend-only control_after_generate value.
SPECS = {
    "T8DramaLoader": ([], [ENGINE], ["drama3-t8-fp8.safetensors", "drama3-codec.safetensors", 12, "torch", "", "", ""]),
    "T8DramaReference": ([("engine", ENGINE), ("audio", "AUDIO"), ("previous", REFERENCES)], [REFERENCES], ["", ""]),
    "T8DramaSession": ([("references", REFERENCES)], [SESSION], [""]),
    "T8DramaGenerate": ([("engine", ENGINE), ("session", SESSION)], [SESSION], ["speech", "你好，这是本地语音测试。", "", 300, 42, "fixed", .7, .7, 30, 180, ""]),
    "T8DramaScene": ([("engine", ENGINE), ("references", REFERENCES)], [SESSION], ["Anna: Ready?\nBen: Always.", "Two friends begin a journey.", "- Anna: a calm young woman.\n- Ben: a cheerful young man.", 44, "fixed", 180, ""]),
    "T8DramaInfill": ([("engine", ENGINE), ("session", SESSION)], [SESSION], ["您好，这是本地语音测试。", 0, .5, 0, 45, "fixed", 180, ""]),
    "T8DramaDecode": ([("engine", ENGINE), ("session", SESSION)], ["AUDIO", "STRING"], [-1, False]),
    "T8DramaSaveSession": ([("session", SESSION)], [], ["drama3_example"]),
    "T8DramaLoadSession": ([], [SESSION], [""]),
    "T8DramaBranch": ([("session", SESSION)], [SESSION], [1]),
    "T8DramaTranscribe": ([("audio", "AUDIO")], ["STRING", "STRING"], ["qwen3-asr-06b.safetensors", "Chinese", "", ""]),
    "T8DramaCodecEncode": ([("engine", ENGINE), ("audio", "AUDIO")], [CODES], []),
    "T8DramaCodecDecode": ([("engine", ENGINE), ("codes", CODES)], ["AUDIO"], []),
    "T8DramaSRT": ([("engine", ENGINE), ("session", SESSION)], [SESSION, "AUDIO", "STRING"], ["1\n00:00:00,000 --> 00:00:02,000\n你好。\n\n2\n00:00:03,000 --> 00:00:05,000\n再见。\n", 55, "fixed", 180, ""]),
    "T8DramaAlign": ([("audio", "AUDIO")], ["STRING"], ["qwen-aligner", "你好，这是本地语音测试。", "Chinese", ""]),
    "T8DramaR2T2Transcribe": ([("audio", "AUDIO")], ["STRING", "STRING", "STRING"], ["asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf", "asr/Confucius4-R2T2-GGUF/mmproj-Confucius4-R2T2-Q8_0.gguf", "asr/FireRedVAD-ONNX", "Chinese", "", ""]),
    "T8DramaLive": ([("engine", ENGINE), ("session", SESSION)], [SESSION, "AUDIO", "STRING"], ["你好，现在是浏览器实时试听测试。", True, 67, "fixed", 180, ""]),
    "LoadAudio": ([], ["AUDIO"], [""]),
    "SaveAudio": ([("audio", "AUDIO")], ["AUDIO"], ["Drama3_T8/output"]),
    "PreviewAny": ([("source", "*")], ["STRING"], []),
    "Note": ([], [], [""]),
}


class Canvas:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []

    def add(self, node_type, column, row=0, widgets=None, title=None):
        inputs, outputs, defaults = SPECS[node_type]
        node = {
            "id": len(self.nodes) + 1, "type": node_type,
            "pos": [60 + column * 420, 60 + row * 460],
            "size": [370, 380 if len(defaults) > 4 else 260],
            "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": name, "type": kind, "link": None} for name, kind in inputs],
            "outputs": [{"name": kind, "type": kind, "links": []} for kind in outputs],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": deepcopy(defaults if widgets is None else widgets),
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
        self.add("Note", column, row, [text], "使用说明 / Instructions")

    def audio_output(self, engine, session, column=3, row=0):
        decode = self.add("T8DramaDecode", column, row)
        self.connect(engine, decode, "engine")
        self.connect(session, decode, "session")
        self.save_audio(decode, column + 1, row)
        preview = self.add("PreviewAny", column, row + 1, title="生成台词 / Generated transcript")
        self.connect(decode, preview, "source", 1)
        return decode

    def save_audio(self, source, column, row=0, output=0):
        save = self.add("SaveAudio", column, row, [f"Drama3_T8/{self.name}"])
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
    widgets = deepcopy(SPECS["T8DramaGenerate"][2])
    widgets[0] = mode
    if mode == "long":
        widgets[1] = "你好，这是长文分段测试。\n第二段会沿用前一段的语音历史。"
        widgets[3] = 48
    elif mode == "stream":
        widgets[1] = "你好，这是流式文本测试。"
        widgets[2] = "接下来会继续介绍本地演出工作台。"
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
            decode["widgets_values"] = [-1, True]
        workflows[name] = graph.export()

    graph, engine, _, generate = speech_graph("revision")
    widgets = deepcopy(SPECS["T8DramaGenerate"][2])
    widgets[:3] = ["revision", "Please say the previous sentence in a cheerful, warm tone.", ""]
    widgets[4] = 43
    revise = graph.add("T8DramaGenerate", 2, widgets=widgets)
    graph.connect(engine, revise, "engine")
    graph.connect(generate, revise, "session")
    graph.audio_output(engine, revise, 3)
    graph.note("先生成原始台词，再基于真实音频历史修改表演。\nThe revision node changes the preceding performance using its audio history.")
    workflows[graph.name] = graph.export()

    graph = Canvas("reference")
    engine = graph.add("T8DramaLoader", 0)
    audio = graph.add("LoadAudio", 0, 1, title="选择参考音频 / Choose reference audio")
    reference = graph.add("T8DramaReference", 1, widgets=["请填写参考音频中实际说出的文字。", "Narrator"])
    session = graph.add("T8DramaSession", 2)
    generate = graph.add("T8DramaGenerate", 3)
    graph.connect(engine, reference, "engine")
    graph.connect(audio, reference, "audio")
    graph.connect(reference, session, "references")
    graph.connect(engine, generate, "engine")
    graph.connect(session, generate, "session")
    graph.audio_output(engine, generate, 4)
    graph.note("在 LoadAudio 上传参考音频，并把编码参考节点的 transcript 改为该音频的真实台词。\nUpload your audio and replace transcript with the words actually spoken.\n多参考：复制编码参考节点，将前一个 references 输出接入下一个 previous。\nFor multiple references, chain each references output into the next previous input.", 1, 1)
    workflows[graph.name] = graph.export()

    graph = Canvas("scene")
    engine = graph.add("T8DramaLoader", 0)
    scene = graph.add("T8DramaScene", 1)
    graph.connect(engine, scene, "engine")
    graph.audio_output(engine, scene, 2)
    workflows[graph.name] = graph.export()

    graph, engine, _, generate = speech_graph("infill")
    infill = graph.add("T8DramaInfill", 2)
    graph.connect(engine, infill, "engine")
    graph.connect(generate, infill, "session")
    graph.audio_output(engine, infill, 3)
    graph.note("先试听原句，再按实际音频调整 start_seconds/end_seconds，仅框选要替换的词；0–0.5 秒仅为示例。\nListen to the original and select the actual word interval; 0–0.5 seconds is only an example.\ntarget_transcript 为编辑后整句；本例仅将开头“你好”改为“您好”，保留后文。\nThe complete edited transcript changes only the opening greeting and keeps the remainder.\nduration_frames=0 使用模型生成长度；正值指定替换帧数。\nUse 0 for generated duration, or a positive frame count.")
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
    graph.note("运行后在 output/drama3_t8_sessions/ 保存会话。take_count 可选择保留前几个生成片段。\nRun to save a session under output/drama3_t8_sessions/. take_count keeps the first N takes.\n随后打开 workflow_load_session.json，刷新页面并选择刚保存的会话。\nNext open workflow_load_session.json, refresh the page and select the saved session.", 1, 1)
    workflows[graph.name] = graph.export()

    graph = Canvas("load_session")
    engine = graph.add("T8DramaLoader", 0)
    session = graph.add("T8DramaLoadSession", 0, 1, title="选择已保存的会话 / Select a saved session")
    graph.audio_output(engine, session, 1)
    graph.note("先运行 workflow_save_session.json；刷新 ComfyUI 后，在加载会话节点选择已保存的文件。\nRun workflow_save_session.json first, refresh ComfyUI, then select the saved session file.\n本示例不附带用户会话或参考音频。\nNo user sessions or reference audio are distributed.", 2, 1)
    workflows[graph.name] = graph.export()
    return workflows


def write_workflows(folder=FOLDER):
    folder.mkdir(parents=True, exist_ok=True)
    # These obsolete files were previously distributed alongside canvas examples.
    for path in folder.glob("*.api.json"):
        path.unlink()
    workflows = make_workflows()
    for name, workflow in workflows.items():
        (folder / f"workflow_{name}.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return workflows


if __name__ == "__main__":
    print(f"Wrote {len(write_workflows())} canvas workflows; no API workflows.")
