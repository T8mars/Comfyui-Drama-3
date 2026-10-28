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
from .audio import audio_waveform, asr_waveforms

EngineType = io.Custom("DRAMA3_T8_ENGINE")
ReferencesType = io.Custom("DRAMA3_T8_REFERENCES")
SessionType = io.Custom("DRAMA3_T8_SESSION")
CodesType = io.Custom("DRAMA3_T8_CODES")
CANVAS_SEED_MAX = 2**53 - 1


class _NodeProgress:
    """Expose Engine progress, committing its endpoint only after node success."""

    def __init__(self, total):
        from comfy.utils import ProgressBar
        self.bar = ProgressBar(total)
        self.total = total

    def __call__(self, value, total):
        self.total = total
        # Engine may publish its endpoint before its last cancellation check.
        # Keep that endpoint pending until result/NodeOutput construction works.
        if value < total:
            self.bar.update_absolute(value, total)

    def complete(self):
        self.bar.update_absolute(self.total, self.total)


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
                io.String.Input("line_head_path", default="", optional=True),
                io.Combo.Input("scene_head_mode", options=["compact", "full"], default="compact", optional=True),
                io.Combo.Input("moe_backend",options=["auto","torch","cuda"],default="auto",optional=True),
                io.Boolean.Input("cuda_graph",default=True,optional=True),
                io.Combo.Input("expert_placement",options=["ram","vram"],default="ram",optional=True,
                    tooltip="ram: host experts with VRAM cache. vram: require all experts/body on one GPU; insufficient VRAM is an error."),
            ], outputs=[EngineType.Output()])

    @classmethod
    def execute(cls, model, codec, expert_cache_gb=12, gdn_backend="torch", model_path="", codec_path="", text_head_path="",
                line_head_path="", scene_head_mode="compact",moe_backend="auto",cuda_graph=True,expert_placement="ram"):
        base = Path(folder_paths.models_dir)
        model = (resolve_model_path(model_path, base) if model_path.strip()
                 else folder_paths.get_full_path_or_raise("diffusion_models", model))
        codec = (resolve_model_path(codec_path, base) if codec_path.strip()
                 else folder_paths.get_full_path_or_raise("vae", codec))
        for path in (model, codec):
            if not Path(path).is_file():
                raise FileNotFoundError(f"模型路径不存在：{path}")
        engine = DramaEngine(model, codec, expert_cache_gb, gdn_backend,
                             text_head_path=resolve_model_path(text_head_path, base) if text_head_path.strip() else None,
                             line_head_path=resolve_model_path(line_head_path, base) if line_head_path.strip() else None,
                             scene_head_mode=scene_head_mode,moe_backend=moe_backend,cuda_graph=cuda_graph,
                             expert_placement=expert_placement)
        return io.NodeOutput(engine)


class T8DramaReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaReference", display_name="T8 Drama 3 · 编码参考",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), io.Audio.Input("audio"),
                io.String.Input("transcript", multiline=True), io.String.Input("name", default=""),
                ReferencesType.Input("previous", optional=True),
                io.Boolean.Input("trim_silence",default=False,optional=True),
                io.Boolean.Input("normalize",default=False,optional=True)], outputs=[ReferencesType.Output()])

    @classmethod
    def execute(cls, engine, audio, transcript, name="", previous=None,trim_silence=False,normalize=False):
        options={key:value for key,value in {"trim_silence":trim_silence,"normalize":normalize}.items() if value}
        reference = engine.encode_reference(audio["waveform"], audio["sample_rate"], transcript, name,**options)
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
                io.Int.Input("top_k", default=30, min=0, tooltip="0 保留全部候选，其余值限制候选数量"),
                io.Int.Input("max_frames", default=600, min=1, max=8192), exact_seed_input(),
                io.Int.Input("opening_retries",default=0,min=0,max=1,optional=True),
                io.Combo.Input("sampling_preset",options=["custom","stable","expressive"],default="custom",optional=True)],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, mode, text, session=None, lookahead="", seed=42, temperature=.7,
                top_p=.7, top_k=30, max_frames=600, chunk_bytes=300, seed_text="",opening_retries=0,sampling_preset="custom"):
        sampling = Sampling(sampling_seed(seed, seed_text), temperature, top_k, top_p, max_frames,opening_retries).with_preset(sampling_preset)
        session = session or Session()
        progress = _NodeProgress(max_frames)
        if mode == "speech":
            result = engine.speak(text, session, sampling, progress=progress)
        elif mode == "revision":
            result = engine.revise(session, text, sampling, progress=progress)
        elif mode == "long":
            result = engine.long_form(text, session, sampling, chunk_bytes=chunk_bytes, progress=progress)
        elif mode == "stream":
            result = engine.stream_text(text, lookahead, session, sampling, progress=progress)
        else:
            raise ValueError(f"Unknown generation mode {mode}")
        output = io.NodeOutput(result)
        progress.complete()
        return output


class T8DramaScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaScene", display_name="T8 Drama 3 · 多角色场景",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), io.String.Input("script", multiline=True),
                io.String.Input("scene", default="", multiline=True),
                io.String.Input("characters", default="", multiline=True),
                ReferencesType.Input("references", optional=True),
                io.Int.Input("seed", default=42, min=0, max=CANVAS_SEED_MAX),
                io.Int.Input("max_frames", default=1200, min=1, max=8192), exact_seed_input(),
                io.Combo.Input('generation_mode',options=['together','per_line','per_line_context','per_speaker'],default='together',optional=True),
                io.Boolean.Input('split_long_text',default=False,optional=True),
                io.Int.Input('chunk_bytes',default=300,min=32,max=3000,optional=True),
                io.Float.Input('temperature',default=.7,min=.01,max=2,step=.05,optional=True),
                io.Float.Input('top_p',default=.7,min=.01,max=1,step=.05,optional=True),
                io.Int.Input('top_k',default=30,min=0,optional=True,tooltip="0 保留全部候选，其余值限制候选数量"),
                io.Int.Input('opening_retries',default=0,min=0,max=1,optional=True),
                io.Combo.Input('sampling_preset',options=['custom','stable','expressive'],default='custom',optional=True),
                SessionType.Input('session',optional=True)],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, script, scene="", characters="", references=None, seed=42, max_frames=1200, seed_text="",generation_mode='together',split_long_text=False,chunk_bytes=300,
                temperature=.7,top_p=.7,top_k=30,opening_retries=0,sampling_preset='custom',session=None):
        lines = [line for line in script.splitlines() if line.strip()]
        sampling=Sampling(sampling_seed(seed,seed_text),temperature,top_k,top_p,max_frames,opening_retries).with_preset(sampling_preset)
        sampling.validate()
        options={} if session is None else {'session':session}
        scene_references=(tuple(references or ()) if session is None else
                          None if references is None else tuple(references))
        progress = _NodeProgress(max_frames)
        result = engine.scene(lines, scene, characters, scene_references,
                              sampling,generation_mode=generation_mode,
                              split_long_text=split_long_text,chunk_bytes=chunk_bytes,progress=progress,**options)
        output = io.NodeOutput(result)
        progress.complete()
        return output


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
                io.Int.Input("max_frames", default=300, min=1, max=8192), exact_seed_input(),
                io.Combo.Input('target_speaker',options=['auto',*[str(i) for i in range(16)]],default='auto',optional=True),
                io.String.Input('expected_view',default='',multiline=True,optional=True)],
            outputs=[SessionType.Output()])

    @classmethod
    def execute(cls, engine, session, target_transcript, start_seconds=0, end_seconds=1,
                duration_frames=0, seed=42, max_frames=300, seed_text="",target_speaker='auto',expected_view=''):
        if not session.takes:
            raise ValueError("Connect a session containing the source take")
        from .editor import json_object
        if target_speaker not in ['auto',*[str(i) for i in range(16)]]:
            raise ValueError('target_speaker must be auto or 0..15')
        role=None if target_speaker=='auto' else int(target_speaker)
        view=None if expected_view=='' else json_object(expected_view,'expected_view',maximum=10000)
        options={}
        if role is not None:options['target_speaker']=role
        if view is not None:options['expected_view']=view
        progress = _NodeProgress(max_frames)
        take = engine.infill(session.takes[-1], target_transcript, start_seconds, end_seconds,
                             Sampling(seed=sampling_seed(seed, seed_text), max_frames=max_frames), duration_frames or None,
                             progress=progress,**options)
        output = io.NodeOutput(session.append(take))
        progress.complete()
        return output


class T8DramaDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="T8DramaDecode", display_name="T8 Drama 3 · 输出音频",
            category="T8/Drama3", inputs=[EngineType.Input("engine"), SessionType.Input("session"),
                io.Int.Input("take_index", default=-1, min=-1),
                io.Boolean.Input("all_takes", default=False),
                io.Boolean.Input("shorten_long_pauses",default=False,optional=True),
                io.Float.Input("max_pause_seconds",default=.75,min=0,max=60,step=.05,optional=True)], outputs=[io.Audio.Output(), io.String.Output()])

    @classmethod
    def execute(cls, engine, session, take_index=-1, all_takes=False,shorten_long_pauses=False,max_pause_seconds=.75):
        if not session.takes:
            raise ValueError("The session contains no audio take")
        chosen = session.takes if all_takes else (session.takes[take_index],)
        codes = torch.cat([take.codes for take in chosen], dim=1)
        if any(t.metadata.get('audio_editor') or t.metadata.get('decode_sources') or 'scene_chunking' in t.metadata or 'render_infill' in t.metadata for t in chosen):
            from .editor import decode_output
            pieces=[decode_output(engine,t) for t in chosen]
            audio={**pieces[0],'waveform':torch.cat([p['waveform'] for p in pieces],dim=-1)}
        else:audio=engine.decode(codes)
        if shorten_long_pauses:
            from .audio_processing import shorten_pauses
            wave,mapping=shorten_pauses(audio["waveform"][0],audio["sample_rate"],maximum=max_pause_seconds)
            audio={**audio,"waveform":wave[None]}
        return io.NodeOutput(audio, "\n".join(take.transcript for take in chosen))


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
            outputs=[SessionType.Output(),ReferencesType.Output(display_name='references')])

    @classmethod
    def execute(cls, session_file):
        path = folder_paths.get_full_path_or_raise("drama3_t8_sessions", session_file)
        session=load_session(path)
        return io.NodeOutput(session,session.references)


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
                io.String.Input("model_path", default="", optional=True),
                io.Int.Input("max_new_tokens", default=4096, min=1, max=32768, optional=True,
                    tooltip="Maximum output tokens per ASR segment; batch clips are transcribed in order.")],
            outputs=[io.String.Output(), io.String.Output()], is_output_node=True)

    @classmethod
    def execute(cls, asr_model, audio, language="auto", context="", model_path="", max_new_tokens=4096):
        from .qwen3_asr.asr import load_qwen3_asr, transcribe, validate_max_new_tokens
        validate_max_new_tokens(max_new_tokens)
        waveforms = asr_waveforms(audio["waveform"], audio["sample_rate"])
        path = (resolve_model_path(model_path, folder_paths.models_dir) if model_path.strip()
                else folder_paths.get_full_path_or_raise("asr", asr_model))
        model = load_qwen3_asr(str(path))
        results = [transcribe(model, waveform, audio["sample_rate"],
                             None if language == "auto" else language, context,
                             max_new_tokens=max_new_tokens) for waveform in waveforms]
        if len(results) == 1:
            text, found = results[0]
        else:
            text = "\n".join(item[0] for item in results)
            languages = []
            for _, detected in results:
                for value in detected.split(","):
                    value = value.strip()
                    if value and value not in languages:
                        languages.append(value)
            found = ",".join(languages)
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
        options=[p.name for p in base.iterdir() if (p.is_dir() and (p/"config.json").exists())
                 or p.is_file() and p.suffix.lower()==".safetensors"] if base.exists() else []
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


class T8DramaEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='T8DramaEditor',display_name='T8 Drama 3 · 多轨后期剪辑',category='T8/Drama3',
            inputs=[EngineType.Input('engine'),SessionType.Input('session'),
                io.Int.Input('take_index',default=-1,min=-1,max=99999),
                io.Boolean.Input('align_words',default=False),
                io.String.Input('aligner_path',default='',optional=True,
                    tooltip='可自定义单文件 .safetensors 或 Qwen3-ForcedAligner 目录'),
                io.Combo.Input('language',options=['Chinese','English','Cantonese','Japanese','Korean','German','French','Spanish','Italian','Portuguese','Russian'],default='Chinese'),
                io.String.Input('edits',default='{}',multiline=True,optional=True)],
            outputs=[SessionType.Output(display_name='edited_session'),io.Audio.Output(),
                io.String.Output(display_name='subtitle_srt'),io.String.Output(display_name='edit_metadata')],is_output_node=True)

    @classmethod
    def execute(cls,engine,session,take_index=-1,align_words=False,aligner_path='',language='Chinese',edits='{}'):
        import soundfile as sf
        from . import editor
        if type(take_index) is not int or not session.takes or not -1<=take_index<len(session.takes):
            raise ValueError('The requested take does not exist')
        if type(align_words) is not bool:
            raise ValueError('align_words must be a boolean')
        index=take_index%len(session.takes)
        aligner=resolve_model_path(aligner_path,folder_paths.models_dir) if align_words and aligner_path.strip() else None
        if align_words and aligner is None:
            raise ValueError('请选择对齐模型路径，再启用逐词字幕')
        source,analysis,saved=editor.prepare(engine,session.takes[index],aligner=aligner,language=language)
        request=editor.json_object(edits or '{}','edits')
        # Validate even when this belongs to an earlier source; a malformed
        # fingerprint is not a legitimate request to reset prior edits.
        editor.parse_edits(request)
        reset=False
        if request.get('source') and request['source']!=analysis['source']:
            # A new seed/input must never inherit a previous take's cuts.
            request={};reset=True
        chosen=editor.parse_edits(request,source=analysis['source'],duration=analysis['duration']) if request else saved
        planned=editor.plan(analysis,chosen)
        waveform=editor.render(source['waveform'][0],analysis,planned['pieces'],chosen)
        result=editor.edited_session(session,index,analysis,chosen)
        info=result.takes[index].metadata['audio_editor']
        folder=Path(folder_paths.get_temp_directory())/'drama3_t8_editor'
        folder.mkdir(parents=True,exist_ok=True)
        identity=uuid.uuid4().hex
        audio_name,analysis_name=f'{identity}.wav',f'{identity}.json'
        sf.write(folder/audio_name,source['waveform'][0].T.numpy(),source['sample_rate'],subtype='FLOAT')
        (folder/analysis_name).write_text(json.dumps(analysis,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        entry=dict(audio=audio_name,analysis=analysis_name,subfolder='drama3_t8_editor',type='temp',
            edits=chosen,source=analysis['source'],edits_reset=reset)
        return io.NodeOutput(result,{'waveform':waveform[None],'sample_rate':source['sample_rate']},
            editor.subtitle_srt(info['captions']),json.dumps(info,ensure_ascii=False,allow_nan=False),
            ui={'drama3_t8_editor':[entry]})


NODES = [T8DramaLoader, T8DramaReference, T8DramaSession, T8DramaGenerate, T8DramaScene,
         T8DramaInfill, T8DramaDecode, T8DramaSaveSession, T8DramaLoadSession, T8DramaBranch, T8DramaTranscribe,
         T8DramaCodecEncode,T8DramaCodecDecode,T8DramaSRT,T8DramaAlign,T8DramaLive,T8DramaR2T2Transcribe,T8DramaEditor]


class Drama3Extension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint():
    from .editor_routes import register_routes
    register_routes()
    return Drama3Extension()
