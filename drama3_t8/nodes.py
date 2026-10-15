"""Native ComfyUI nodes. No HTTP client and no external inference process."""
import os
import json
import uuid
from pathlib import Path

import torch
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui

from .engine import DramaEngine, Sampling
from .state import Session, load_session, save_session
from .batch import generate_srt
from .model_paths import model_path as resolve_model_path
from .audio import audio_waveform

EngineType = io.Custom("DRAMA3_T8_ENGINE")
ReferencesType = io.Custom("DRAMA3_T8_REFERENCES")
SessionType = io.Custom("DRAMA3_T8_SESSION")
CodesType = io.Custom("DRAMA3_T8_CODES")
CANVAS_SEED_MAX = 2**53 - 1


def sampling_seed(seed, seed_text=""):
    """A decimal text override survives the browser's JSON number precision."""
    if not isinstance(seed_text, str):
        raise ValueError("seed_text must be a decimal string")
    text = seed_text.strip()
    if text:
        if not text.isascii() or not text.isdecimal() or len(text) > 20:
            raise ValueError("seed_text must contain 0 to 18446744073709551615")
        seed = int(text)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an integer from 0 to 18446744073709551615")
    return seed


def exact_seed_input():
    return io.String.Input("seed_text", default="", optional=True,
        tooltip="Optional exact 64-bit decimal seed (0–18446744073709551615). Overrides seed; leave blank for its normal controls.")

SESSIONS = Path(folder_paths.get_output_directory()) / "drama3_t8_sessions"
folder_paths.add_model_folder_path("drama3_t8_sessions", str(SESSIONS))
folder_paths.folder_names_and_paths["drama3_t8_sessions"] = (
    folder_paths.folder_names_and_paths["drama3_t8_sessions"][0], {".safetensors"})
folder_paths.add_model_folder_path("asr", os.path.join(folder_paths.models_dir, "asr"))
asr_paths, asr_extensions = folder_paths.folder_names_and_paths["asr"]
# An empty extension set means "all files" in ComfyUI; keep that shared contract.
asr_updated_extensions = set(asr_extensions) | {".safetensors"} if asr_extensions else set()
folder_paths.folder_names_and_paths["asr"] = (asr_paths, asr_updated_extensions)
if asr_updated_extensions != set(asr_extensions):
    folder_paths.filename_list_cache.pop("asr", None)
    folder_paths.cache_helper.cache.pop("asr", None)


class T8DramaLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaLoader", display_name="T8 Drama 3 · 原生模型",
            category="T8/Drama3", inputs=[
                io.Combo.Input("model", options=folder_paths.get_filename_list("diffusion_models") or ["<custom path>"]),
                io.Combo.Input("codec", options=folder_paths.get_filename_list("vae") or ["<custom path>"]),
                io.Float.Input("expert_cache_gb", default=12, min=0, max=128, step=.5),
                io.Combo.Input("gdn_backend", options=["torch", "fla"], default="torch"),
                io.String.Input("model_path", default="", optional=True),
                io.String.Input("codec_path", default="", optional=True),
                io.String.Input("text_head_path", default="", optional=True),
            ], outputs=[EngineType.Output()])

    @classmethod
    def execute(cls, model, codec, expert_cache_gb=12, gdn_backend="torch", model_path="", codec_path="", text_head_path=""):
        base = Path(folder_paths.models_dir)
        model = (resolve_model_path(model_path, base) if model_path.strip()
                 else folder_paths.get_full_path_or_raise("diffusion_models", model))
        codec = (resolve_model_path(codec_path, base) if codec_path.strip()
                 else folder_paths.get_full_path_or_raise("vae", codec))
        for path in (model, codec):
            if not Path(path).is_file():
                raise FileNotFoundError(f"模型路径不存在：{path}")
        engine = DramaEngine(model, codec, expert_cache_gb, gdn_backend,
                             text_head_path=resolve_model_path(text_head_path, base) if text_head_path.strip() else None)
        return io.NodeOutput(engine)


class T8DramaReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaReference", display_name="T8 Drama 3 · 编码参考",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), io.Audio.Input("audio"),
                io.String.Input("transcript", multiline=True), io.String.Input("name", default=""),
                ReferencesType.Input("previous", optional=True)], outputs=[ReferencesType.Output()])

    @classmethod
    def execute(cls, engine, audio, transcript, name="", previous=None):
        reference = engine.encode_reference(audio["waveform"], audio["sample_rate"], transcript, name)
        return io.NodeOutput((*tuple(previous or ()), reference))


class T8DramaSession(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaSession", display_name="T8 Drama 3 · 新会话",
            category="T8/Drama3", inputs=[ReferencesType.Input("references", optional=True),
                io.String.Input("direction", default="", multiline=True)], outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, references=None, direction=""):
        return io.NodeOutput(Session(tuple(references or ()), direction=direction))


class T8DramaGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaGenerate", display_name="T8 Drama 3 · 生成 / 导演修改 / 流式",
            category="T8/Drama3", inputs=[EngineType.Input("engine"),
                io.Combo.Input("mode", options=["speech", "revision", "long", "stream"]),
                io.String.Input("text", multiline=True), SessionType.Input("session", optional=True),
                io.String.Input("lookahead", default="", multiline=True),
                io.Int.Input("chunk_bytes", default=300, min=32, max=3000),
                io.Int.Input("seed", default=42, min=0, max=CANVAS_SEED_MAX),
                io.Float.Input("temperature", default=.7, min=.01, max=2, step=.05),
                io.Float.Input("top_p", default=.7, min=.01, max=1, step=.05),
                io.Int.Input("top_k", default=30, min=1),
                io.Int.Input("max_frames", default=600, min=1, max=8192), exact_seed_input()],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, mode, text, session=None, lookahead="", seed=42, temperature=.7,
                top_p=.7, top_k=30, max_frames=600, chunk_bytes=300, seed_text=""):
        sampling = Sampling(sampling_seed(seed, seed_text), temperature, top_k, top_p, max_frames)
        session = session or Session()
        if mode == "speech":
            result = engine.speak(text, session, sampling)
        elif mode == "revision":
            result = engine.revise(session, text, sampling)
        elif mode == "long":
            result = engine.long_form(text, session, sampling, chunk_bytes=chunk_bytes)
        elif mode == "stream":
            result = engine.stream_text(text, lookahead, session, sampling)
        else:
            raise ValueError(f"Unknown generation mode {mode}")
        return io.NodeOutput(result)


class T8DramaScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaScene", display_name="T8 Drama 3 · 多角色场景",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), io.String.Input("script", multiline=True),
                io.String.Input("scene", default="", multiline=True),
                io.String.Input("characters", default="", multiline=True),
                ReferencesType.Input("references", optional=True),
                io.Int.Input("seed", default=42, min=0, max=CANVAS_SEED_MAX),
                io.Int.Input("max_frames", default=1200, min=1, max=8192), exact_seed_input()],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, script, scene="", characters="", references=None, seed=42, max_frames=1200, seed_text=""):
        lines = [line for line in script.splitlines() if line.strip()]
        return io.NodeOutput(engine.scene(lines, scene, characters, tuple(references or ()),
                                         Sampling(seed=sampling_seed(seed, seed_text), max_frames=max_frames)))


class T8DramaInfill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaInfill", display_name="T8 Drama 3 · 局部音频编辑",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), SessionType.Input("session"),
                io.String.Input("target_transcript", multiline=True),
                io.Float.Input("start_seconds", default=0, min=0, step=.05),
                io.Float.Input("end_seconds", default=1, min=0, step=.05),
                io.Int.Input("duration_frames", default=0, min=0),
                io.Int.Input("seed", default=42, min=0, max=CANVAS_SEED_MAX),
                io.Int.Input("max_frames", default=300, min=1, max=8192), exact_seed_input()],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, session, target_transcript, start_seconds=0, end_seconds=1,
                duration_frames=0, seed=42, max_frames=300, seed_text=""):
        if not session.takes:
            raise ValueError("Connect a session containing the source take")
        take = engine.infill(session.takes[-1], target_transcript, start_seconds, end_seconds,
                             Sampling(seed=sampling_seed(seed, seed_text), max_frames=max_frames), duration_frames or None)
        return io.NodeOutput(session.append(take))


class T8DramaDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaDecode", display_name="T8 Drama 3 · 输出音频",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), SessionType.Input("session"),
                io.Int.Input("take_index", default=-1, min=-1),
                io.Boolean.Input("all_takes", default=False)], outputs=[io.Audio.Output(), io.String.Output()])

    @classmethod
    def execute(cls, engine, session, take_index=-1, all_takes=False):
        if not session.takes:
            raise ValueError("The session contains no audio take")
        chosen = session.takes if all_takes else (session.takes[take_index],)
        codes = torch.cat([take.codes for take in chosen], dim=1)
        return io.NodeOutput(engine.decode(codes), "\n".join(take.transcript for take in chosen))


class T8DramaSaveSession(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaSaveSession", display_name="T8 Drama 3 · 保存会话 / 音色",
            category="T8/Drama3", inputs=[SessionType.Input("session"), io.String.Input("name", default="session")],
            is_output_node=True)

    @classmethod
    def execute(cls, session, name="session"):
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(name, str(SESSIONS))
        target = Path(folder) / f"{filename}_{counter:05}.safetensors"
        if not target.resolve().is_relative_to(SESSIONS.resolve()):
            raise ValueError("Session output must stay inside the session directory")
        save_session(session, target)
        return io.NodeOutput()


class T8DramaLoadSession(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaLoadSession", display_name="T8 Drama 3 · 加载会话 / 音色",
            category="T8/Drama3", inputs=[
                io.Combo.Input("session_file", options=folder_paths.get_filename_list("drama3_t8_sessions"))],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, session_file):
        path = folder_paths.get_full_path_or_raise("drama3_t8_sessions", session_file)
        return io.NodeOutput(load_session(path))


class T8DramaBranch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaBranch", display_name="T8 Drama 3 · 会话分支",
            category="T8/Drama3", inputs=[SessionType.Input("session"),
                io.Int.Input("take_count", default=1, min=0)], outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, session, take_count=1):
        return io.NodeOutput(session.branch(take_count))


class T8DramaTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .qwen3_asr.asr import LANGUAGES
        return io.Schema(node_id="T8DramaTranscribe", display_name="T8 Drama 3 · Qwen3-ASR 转录",
            category="T8/Drama3", inputs=[
                io.Combo.Input("asr_model", options=[name for name in folder_paths.get_filename_list("asr")
                    if name.lower().endswith(".safetensors")] or ["<custom path>"]),
                io.Audio.Input("audio"), io.Combo.Input("language", options=["auto", *LANGUAGES]),
                io.String.Input("context", default="", multiline=True),
                io.String.Input("model_path", default="", optional=True)],
            outputs=[io.String.Output(), io.String.Output()], is_output_node=True)

    @classmethod
    def execute(cls, asr_model, audio, language="auto", context="", model_path=""):
        from .qwen3_asr.asr import load_qwen3_asr, transcribe
        waveform = audio_waveform(audio["waveform"], audio["sample_rate"])
        path = (resolve_model_path(model_path, folder_paths.models_dir) if model_path.strip()
                else folder_paths.get_full_path_or_raise("asr", asr_model))
        model = load_qwen3_asr(str(path))
        text, found = transcribe(model, waveform, audio["sample_rate"],
                                 None if language == "auto" else language, context)
        return io.NodeOutput(text, found,ui=ui.PreviewText(text))


class T8DramaCodecEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaCodecEncode", display_name="T8 Drama 3 · Codec 编码",category="T8/Drama3",
                         inputs=[EngineType.Input("engine"),io.Audio.Input("audio")],outputs=[CodesType.Output()])

    @classmethod
    def execute(cls, engine, audio):
        reference=engine.encode_reference(audio["waveform"],audio["sample_rate"],"")
        return io.NodeOutput(reference.codes)


class T8DramaCodecDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaCodecDecode", display_name="T8 Drama 3 · Codec 解码",category="T8/Drama3",
                         inputs=[EngineType.Input("engine"),CodesType.Input("codes")],outputs=[io.Audio.Output()])

    @classmethod
    def execute(cls, engine, codes):
        return io.NodeOutput(engine.decode(codes))


class T8DramaSRT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaSRT", display_name="T8 Drama 3 · SRT 批量台词",category="T8/Drama3",
                         inputs=[EngineType.Input("engine"),io.String.Input("srt",multiline=True),SessionType.Input("session",optional=True),
                                 io.Int.Input("seed",default=42,min=0,max=CANVAS_SEED_MAX),io.Int.Input("max_frames",default=600,min=1,max=8192),exact_seed_input()],
                         outputs=[SessionType.Output(),io.Audio.Output(),io.String.Output()])

    @classmethod
    def execute(cls,engine,srt,session=None,seed=42,max_frames=600,seed_text=""):
        result,audio,actual=generate_srt(engine,srt,session or Session(),Sampling(seed=sampling_seed(seed,seed_text),max_frames=max_frames))
        return io.NodeOutput(result,audio,actual)


class T8DramaAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        base=Path(folder_paths.models_dir)/"asr"
        options=[p.name for p in base.iterdir() if p.is_dir() and (p/"config.json").exists()] if base.exists() else []
        return io.Schema(node_id="T8DramaAlign",display_name="T8 Drama 3 · 词级时间对齐",category="T8/Drama3",
                         inputs=[io.Combo.Input("aligner",options=options or ["<custom path>"]),io.Audio.Input("audio"),io.String.Input("transcript",multiline=True),
                                 io.Combo.Input("language",options=["Chinese","English","Cantonese","Japanese","Korean","German","French","Spanish","Italian","Portuguese","Russian"]),
                                 io.String.Input("model_path",default="",optional=True)],
                         outputs=[io.String.Output()],is_output_node=True)

    @classmethod
    def execute(cls,aligner,audio,transcript,language="Chinese",model_path=""):
        from .alignment import align_audio
        base=(Path(folder_paths.models_dir)/"asr").resolve()
        path=resolve_model_path(model_path,folder_paths.models_dir) if model_path.strip() else (base/aligner).resolve()
        if not model_path.strip() and path.parent!=base:
            raise ValueError("Invalid aligner directory")
        words=align_audio(path,audio["waveform"],audio["sample_rate"],transcript,language)
        rendered=json.dumps(words,ensure_ascii=False,indent=2)
        return io.NodeOutput(rendered,ui=ui.PreviewText(rendered))


class T8DramaR2T2Transcribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .r2t2.native import SUPPORTED_LANGUAGES
        return io.Schema(node_id="T8DramaR2T2Transcribe", display_name="T8 Drama 3 · Confucius R2T2 Q8 转写",
            category="T8/Drama3", inputs=[io.Audio.Input("audio"),
                io.String.Input("model_path", default="asr/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf"),
                io.String.Input("projector_path", default="asr/Confucius4-R2T2-GGUF/mmproj-Confucius4-R2T2-Q8_0.gguf"),
                io.String.Input("vad_path", default="asr/FireRedVAD-ONNX"),
                io.Combo.Input("language", options=["auto", *SUPPORTED_LANGUAGES]),
                io.String.Input("hotwords", default="", multiline=True),
                io.String.Input("context", default="", multiline=True)],
            outputs=[io.String.Output(display_name="transcript"),io.String.Output(display_name="language"),
                     io.String.Output(display_name="metadata")], is_output_node=True)

    @classmethod
    def execute(cls, audio, model_path, projector_path, vad_path, language="auto", hotwords="", context=""):
        import comfy.model_management as mm
        from .r2t2.transcribe import transcribe_audio
        base = folder_paths.models_dir
        result = transcribe_audio(audio, resolve_model_path(model_path, base),
            resolve_model_path(projector_path, base), resolve_model_path(vad_path, base),
            language=language, hotwords=hotwords, context=context, check_cancel=mm.throw_exception_if_processing_interrupted)
        rendered=json.dumps(result,ensure_ascii=False,indent=2)
        return io.NodeOutput(result["text"],result["language"],rendered,ui=ui.PreviewText(result["text"]))


class T8DramaLive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaLive",display_name="T8 Drama 3 · 实时语音输出",category="T8/Drama3",
                         inputs=[EngineType.Input("engine"),io.String.Input("text",multiline=True),SessionType.Input("session",optional=True),
                                 io.Boolean.Input("preview_audio",default=False),io.Int.Input("seed",default=42,min=0,max=CANVAS_SEED_MAX),
                                 io.Int.Input("max_frames",default=600,min=1,max=8192),exact_seed_input()],
                         outputs=[SessionType.Output(),io.Audio.Output(),io.String.Output()],hidden=[io.Hidden.unique_id],is_output_node=True)

    @classmethod
    def execute(cls,engine,text,session=None,preview_audio=False,seed=42,max_frames=600,seed_text=""):
        from server import PromptServer
        from .live_audio import LiveAudio
        folder=Path(folder_paths.get_output_directory())/"drama3_live"/uuid.uuid4().hex
        client=PromptServer.instance.client_id
        node_id=cls.hidden.unique_id
        chunk_paths=[]
        def emitted(path):
            relative=path.relative_to(Path(folder_paths.get_output_directory()))
            chunk_paths.append(relative.as_posix())
            PromptServer.instance.send_sync("drama3_t8_live_audio",{"node_id":node_id,"filename":path.name,
                "subfolder":relative.parent.as_posix(),"type":"output","preview":preview_audio},sid=client)
        live=LiveAudio(engine.codec_path,folder,on_chunk=emitted)
        try:
            result=engine.speak(text,session or Session(),Sampling(seed=sampling_seed(seed,seed_text),max_frames=max_frames),on_codes=live.append)
            live.finish()
            audio=engine.decode(result.takes[-1].codes)
        except BaseException:
            live.finish(abort=True)
            PromptServer.instance.send_sync("drama3_t8_live_stop",{"node_id":node_id},sid=client)
            raise
        return io.NodeOutput(result,audio,json.dumps(chunk_paths,ensure_ascii=False))


NODES = [T8DramaLoader, T8DramaReference, T8DramaSession, T8DramaGenerate, T8DramaScene,
         T8DramaInfill, T8DramaDecode, T8DramaSaveSession, T8DramaLoadSession, T8DramaBranch, T8DramaTranscribe,
         T8DramaCodecEncode,T8DramaCodecDecode,T8DramaSRT,T8DramaAlign,T8DramaLive,T8DramaR2T2Transcribe]


class Drama3Extension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    return Drama3Extension()
