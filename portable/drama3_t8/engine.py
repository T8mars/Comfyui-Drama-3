"""Direct tensor inference shared by the local app and native ComfyUI nodes."""
from dataclasses import dataclass, replace
import contextlib
import gc
import time
import math
from pathlib import Path

import torch
import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .native.codec import SAMPLE_RATE, FRAME_LENGTH, DECODE_WINDOW, DECODE_CONTEXT, DECODE_FRAME_BYTES
from .native.generate import generate
from .native.loader import load_drama3, load_codec
from .native.chunking import split_text
from .prompts import (
    SEPARATOR, infill_prompt, scene_prompt, streaming_prompt, tts_prompt, spoken_text,
    directed_text, directed_scene_lines, direction_tags, with_speaker_tag,
)
from .state import Reference, Session, Take, validate_codes
from .scene_cast import prepare_scene_cast
from .audio import audio_waveform


class SpeechGenerationError(RuntimeError):
    def __init__(self,message,attempts):
        seeds=", ".join(str(item["seed"]) for item in attempts)
        super().__init__(f"{message} (attempted seeds: {seeds})")
        self.generation_attempts=attempts


@dataclass(frozen=True)
class Sampling:
    seed: int = 42
    temperature: float = 0.7
    top_k: int = 30
    top_p: float = 0.7
    max_frames: int = 600
    opening_retries: int = 0

    def with_preset(self, name):
        if name=="custom":return self
        presets={"stable":(.7,30,.7),"expressive":(1.,0,.9)}
        if name not in presets:raise ValueError("Unknown sampling preset")
        temperature,top_k,top_p=presets[name]
        return replace(self,temperature=temperature,top_k=top_k,top_p=top_p)

    def validate(self):
        if type(self.opening_retries) is not int or self.opening_retries not in (0,1):
            raise ValueError("At most one opening retry is supported")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("Seed must be an integer from 0 to 2**64-1")
        if (type(self.max_frames) is not int or type(self.top_k) is not int
                or self.max_frames < 1 or self.top_k < 0
                or not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1):
            raise ValueError("Invalid sampling limits")


class _SpeechHeadSelection:
    """Select existing ordinary-TTS columns; own only the small index tensor."""
    def __init__(self, head, positions, width):
        self.head, self.positions, self.width = head, tuple(positions), width
        self.index = None

    def __call__(self, hidden):
        if self.head is None:
            raise RuntimeError("Speech head selection is already closed")
        logits = self.head(hidden)
        if logits.shape[-1] != self.width:
            raise ValueError("Speech head logits disagree with output-token metadata")
        if self.index is None or self.index.device != logits.device:
            self.index = torch.tensor(self.positions, device=logits.device, dtype=torch.long)
        return logits.index_select(-1, self.index)

    def close(self):
        self.index = None
        self.head = None


def _speech_head_override(drama):
    config = drama.config
    original = tuple(config.output_token_ids)
    eos, start, end = config.eos_token_id, config.semantic_start_token_id, config.semantic_end_token_id
    if any(type(value) is not int for value in (eos, start, end, *original)) or start > end:
        raise ValueError("Invalid ordinary-TTS token metadata")
    allowed = {eos, *range(start, end + 1)}
    if eos in range(start, end + 1) or len(original) != len(set(original)) or not allowed.issubset(original):
        raise ValueError("Ordinary TTS requires unique EOS and every semantic token")
    extras = {drama.tokenizer.token_to_id(token) for token in (SEPARATOR, "<|endofturn|>")}
    if set(original) - allowed - extras:
        raise ValueError("Unrecognized extra ordinary-TTS head tokens")
    positions = tuple(index for index, token in enumerate(original) if token in allowed)
    if len(positions) == len(original):
        return None
    head = drama.parts()[3]
    selected = _SpeechHeadSelection(head, positions, len(original))
    return selected, tuple(original[index] for index in positions)


class DramaEngine:
    def __init__(self, model_path, codec_path, expert_cache_gb=12.0, gdn_backend="torch", *,
                 text_head_path=None, asr_path=None, asr_backend="qwen", r2t2_paths=None,
                 line_head_path=None, scene_head_mode="compact",moe_backend="auto",cuda_graph=True,expert_placement="ram"):
        self.model_path = Path(model_path)
        self.codec_path = Path(codec_path)
        self.text_head_path = Path(text_head_path) if text_head_path else self.model_path.parent / "drama3-text-head.safetensors"
        self.line_head_path = Path(line_head_path) if line_head_path else self.model_path.parent / "drama3-line-head.safetensors"
        if scene_head_mode not in {"compact", "full"}:
            raise ValueError("Scene head mode must be compact or full")
        self.scene_head_mode = scene_head_mode
        if moe_backend not in {"auto","torch","cuda"}:raise ValueError("Invalid MoE backend")
        self.moe_backend=moe_backend
        if expert_placement not in {'ram','vram'}:raise ValueError('Expert placement must be ram or vram')
        self.expert_placement=expert_placement
        if type(cuda_graph) is not bool:raise ValueError("CUDA graph setting must be a boolean")
        self.cuda_graph=cuda_graph
        self.line_head = None
        self.line_head_patcher = None
        self.asr_path = Path(asr_path) if asr_path else self.model_path.parent / "qwen3-asr-06b.safetensors"
        self.asr_backend = asr_backend
        self.r2t2_paths = r2t2_paths or {}
        self.expert_cache_gb = expert_cache_gb
        self.gdn_backend = gdn_backend
        self.requested_backend = gdn_backend
        self.acceleration_fallback = None
        self.drama = None
        self.codec = None
        self.asr = None
        self.text_head = None
        self.text_head_patcher = None

    def load_text_head(self):
        if self.text_head is None:
            state = comfy.utils.load_torch_file(str(self.text_head_path))
            width, hidden = state["weight"].shape
            projection = comfy.ops.manual_cast.Linear(hidden, width, bias=False, dtype=torch.bfloat16, device="meta")
            projection.load_state_dict(state, assign=True)
            head = torch.nn.Sequential(projection)
            patcher = comfy.model_patcher.ModelPatcher(head, load_device=mm.get_torch_device(),
                                                       offload_device=mm.unet_offload_device())
            self.text_head, self.text_head_patcher = head, patcher
        self._load_drama((self.text_head_patcher,))
        return self.text_head, tuple(range(self.text_head[0].out_features))

    def load_model(self):
        from .native.graphs import ensure_healthy
        ensure_healthy()
        if self.drama is None:
            try:
                self.drama = load_drama3(str(self.model_path), experts_in_ram=self.expert_placement=='ram',
                                        expert_cache_gb=self.expert_cache_gb,moe_backend=self.moe_backend)
            except BaseException as error:
                self.last_load_diagnostics=getattr(error,'load_diagnostics',None)
                raise
            loaded_reader=getattr(self.drama,'checkpoint_reader',None)
            self.last_load_diagnostics=loaded_reader.get('load_summary') if type(loaded_reader) is dict else None
            self.drama.cuda_graph=self.cuda_graph
            self.set_backend(self.gdn_backend)
        self._load_drama()
        return self.drama

    def _load_drama(self, extra_patchers=()):
        if self.expert_placement == 'vram':
            # Native loading checks the complete single-GPU body and heads.
            # It must reject insufficient VRAM rather than silently stream.
            self.drama.load_resident(extra_patchers=tuple(extra_patchers))
        else:
            mm.load_models_gpu([*self.drama.patchers(), *extra_patchers])

    def expert_stats(self):
        if self.drama is not None and hasattr(self.drama,'expert_stats'):
            return self.drama.expert_stats()
        experts=getattr(self.drama,'experts',None)
        return dict(placement=self.expert_placement,requested_backend=self.moe_backend,
            actual_backend=getattr(experts,'last_backend','not_run'),
            fallback_reason=getattr(experts,'fallback_reason',None),
            backend_calls=dict(getattr(experts,'backend_calls',{})),
            cpu_experts_run=getattr(experts,'cpu_experts_run',0),cpu_decode_calls=0)

    @contextlib.contextmanager
    def generation_session(self):
        """Keep expert slots across the independently prompted takes of one script."""
        drama=self.load_model()
        with drama.session(cpu_decode=True):
            yield

    def load_scene_head(self):
        if self.scene_head_mode=="full":
            return self.load_text_head()
        drama=self.drama
        base_ids=tuple(drama.config.output_token_ids)
        newline=drama.tokenizer.encode("\n").ids
        if len(newline)!=1:
            raise ValueError("The model tokenizer must encode a newline as one control")
        controls=(newline[0],drama.tokenizer.token_to_id(SEPARATOR))
        extra=tuple(i for i in controls if i not in base_ids)
        if not extra:
            return drama.parts()[3],base_ids
        if self.line_head is None:
            from .native.control_head import control_rows
            source=self.line_head_path if self.line_head_path.is_file() else self.text_head_path
            weight=control_rows(source,extra,drama.config.text.hidden_size)
            projection=comfy.ops.manual_cast.Linear(weight.shape[1],len(extra),bias=False,
                dtype=weight.dtype,device="meta")
            projection.load_state_dict({"weight":weight},assign=True)
            # ModelPatcher resolves leaf parameters by a nonempty module path;
            # match the full-head wrapper so this weight is named '0.weight'.
            head=torch.nn.Sequential(projection)
            patcher=comfy.model_patcher.ModelPatcher(head,
                load_device=mm.get_torch_device(),offload_device=mm.unet_offload_device())
            self.line_head,self.line_head_patcher=head,patcher
        self._load_drama((self.line_head_patcher,))
        audio_head=drama.parts()[3]
        def combined(hidden):
            return torch.cat((audio_head(hidden),self.line_head(hidden)),dim=-1)
        return combined,(*base_ids,*extra)

    def set_backend(self, backend):
        self.requested_backend = backend
        self.acceleration_fallback = None
        if backend not in {"torch", "fla"}:
            raise ValueError(f"Unsupported DeltaNet backend {backend}")
        if backend == "fla":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except (ImportError, OSError) as exc:
                self.acceleration_fallback = str(exc)
                backend = "torch"
        else:
            self.acceleration_fallback = None
        self.gdn_backend = backend
        if self.drama is not None:
            for layer in self.drama.model.model.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.backend = backend
                    layer.linear_attn.fallback_reason = None

    def load_codec(self, *, memory_required=0):
        from .native.graphs import ensure_healthy
        ensure_healthy()
        if self.codec is None:
            self.codec = load_codec(str(self.codec_path))
        mm.load_models_gpu([self.codec.patcher], memory_required=memory_required)
        return self.codec

    def encode_reference(self, waveform, sample_rate, transcript, name="", *, trim_silence=False, normalize=False):
        if type(trim_silence) is not bool or type(normalize) is not bool:
            raise ValueError("Reference trim and normalization options must be booleans")
        waveform = audio_waveform(waveform, sample_rate)
        waveform = waveform.mean(0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = comfy.audio.resample(waveform, sample_rate, SAMPLE_RATE)
        from .audio_processing import prepare_reference
        waveform,processing=prepare_reference(waveform,SAMPLE_RATE,trim_silence=trim_silence,normalize=normalize)
        codec = self.load_codec()
        with mm.cuda_device_context(codec.patcher.load_device):
            codes = codec.model.encode(waveform.to(codec.patcher.load_device))[0].cpu()
        return Reference(transcript, codes, name,processing=processing)

    def transcribe(self, waveform, sample_rate, language=None, context=""):
        result = self.transcribe_detailed(waveform, sample_rate, language=language, context=context)
        return result["text"], result["language"]

    def transcribe_detailed(self, waveform, sample_rate, language=None, context="", hotwords="", check_cancel=None):
        from .asr_language import normalize_asr_language
        if self.asr_backend == "r2t2":
            from .r2t2.native import SUPPORTED_LANGUAGES as supported_languages
        else:
            from .qwen3_asr.asr import LANGUAGES as supported_languages
        language = normalize_asr_language(language, supported_languages,
            backend="Confucius" if self.asr_backend == "r2t2" else "Qwen3-ASR")
        waveform = audio_waveform(waveform, sample_rate)[None]
        from .model_paths import require_models
        if self.asr_backend == "r2t2":
            require_models(self.r2t2_paths,'r2t2_model','r2t2_projector')
            from .r2t2.transcribe import transcribe_audio
            return transcribe_audio({"waveform": waveform, "sample_rate": sample_rate},
                self.r2t2_paths["r2t2_model"], self.r2t2_paths["r2t2_projector"], self.r2t2_paths["r2t2_vad"],
                language=language, context=context, hotwords=hotwords, check_cancel=check_cancel)
        from .r2t2.hotwords import hotword_context
        require_models({'asr_model':self.asr_path},'asr_model')
        from .qwen3_asr.asr import load_qwen3_asr, transcribe
        start = time.perf_counter()
        if self.asr is None:
            self.asr = load_qwen3_asr(str(self.asr_path))
        text, found = transcribe(self.asr, waveform[0], sample_rate,
                                 language=None if language in ("auto", "Auto") else language,
                                 context=hotword_context(context, hotwords))
        return {"text": text, "language": found, "backend": "qwen", "model_path": str(self.asr_path),
                "duration_seconds": waveform.shape[-1] / sample_rate,
                "elapsed_seconds": time.perf_counter() - start}

    def decode(self, codes):
        validate_codes(codes)
        if codes.shape[-1] == 0:
            raise ValueError("The model returned no audio frames")
        codec = self.load_codec(memory_required=min(codes.shape[-1], DECODE_WINDOW + DECODE_CONTEXT) * DECODE_FRAME_BYTES)
        device = codec.patcher.load_device
        with mm.cuda_device_context(device):
            waveform = codec.model.decode(codes[None].to(device=device, dtype=torch.long), torch.float32).float().cpu()
        return {"waveform": waveform[:, None], "sample_rate": SAMPLE_RATE}

    def _take(self, prompt, prompt_text, transcript, sampling, on_codes=None, progress=None,
              stop_token_ids=(), full_head=False, allow_empty=False, expected_segments=None, output_policy="legacy"):
        sampling.validate()
        drama = self.load_model()
        if output_policy not in {"legacy", "speech"}:
            raise ValueError("Unknown generation output policy")
        if output_policy == "speech" and full_head:
            raise ValueError("Ordinary-TTS policy cannot replace a scene/full head")
        head = self.load_scene_head() if full_head else None
        if output_policy == "speech":
            head = _speech_head_override(drama)
        calls_before = {backend:sum(layer.linear_attn.backend_calls[backend]
                           for layer in drama.model.model.layers if layer.layer_type == "linear_attention")
                        for backend in ("torch","fla")}
        experts_before=self.expert_stats()
        cpu_before=experts_before['cpu_experts_run']
        start = time.perf_counter()
        first_frame = None
        finish_reason = None
        boundary_corrections=[]
        attempts=[]
        pending=[]
        stream_committed=False
        seen=set()

        def finished(reason):
            nonlocal finish_reason
            finish_reason = reason

        def audio_callback(frame):
            nonlocal first_frame,stream_committed
            if first_frame is None:
                first_frame = time.perf_counter() - start
            # Only delivered audio prevents an opening retry. Native callbacks
            # also run for non-streaming jobs, including frames later trimmed.
            if on_codes is None:
                return
            if sampling.opening_retries and not stream_committed:
                pending.append(frame)
                seen.add(int(frame[0]))
                if len(pending)<54 or len(seen)<24:
                    return
                stream_committed=True
                if on_codes is not None:
                    for item in pending:on_codes(item)
                pending.clear()
            elif on_codes is not None:
                on_codes(frame)

        effective_seed=sampling.seed
        try:
            for attempt in range(sampling.opening_retries+1):
                effective_seed=(sampling.seed+attempt)%2**64
                pending.clear();seen.clear();stream_committed=False;first_frame=None
                boundary_corrections.clear()
                finish_reason=None
                frames = generate(drama, prompt, sampling.max_frames, sampling.temperature,
                                  sampling.top_k, sampling.top_p, effective_seed,
                                  on_audio=audio_callback, progress=progress, stop_token_ids=stop_token_ids,
                                  on_finish=finished, head_override=head,
                                  extra_patchers=tuple(p for p in (self.text_head_patcher,self.line_head_patcher) if p is not None) if full_head else (),
                                  expected_segments=expected_segments,on_boundary=boundary_corrections.append)
                count=int(((frames[:,0]>=drama.config.semantic_start_token_id)&
                           (frames[:,0]<=drama.config.semantic_end_token_id)).sum())
                retryable=(not count and not allow_empty and finish_reason=="eos" or
                           finish_reason=="stalled" and count<54) and not stream_committed
                record=dict(seed=effective_seed,finish_reason=finish_reason,audio_frames=count,
                    retry=bool(retryable and attempt<sampling.opening_retries),
                    boundary_corrections=list(boundary_corrections))
                if record["retry"]:record["discarded_frames"]=frames.tolist()
                attempts.append(record)
                if not record["retry"]:break
        finally:
            if head is not None and isinstance(head[0], _SpeechHeadSelection):
                head[0].close()
        if finish_reason == "stalled":
            raise SpeechGenerationError("The model entered a silent repetition; use another seed or a more accurate reference transcript",attempts)
        if on_codes is not None:
            for item in pending:on_codes(item)
        duration = ((frames[:, 0] >= drama.config.semantic_start_token_id)
                    & (frames[:, 0] <= drama.config.semantic_end_token_id)).sum().item() * FRAME_LENGTH / SAMPLE_RATE
        metadata = {"seed": effective_seed,"requested_seed":sampling.seed,
                    "sampling": replace(sampling,seed=effective_seed).__dict__,"generation_attempts":attempts,
                    "model": self.model_path.name, "elapsed_seconds": time.perf_counter() - start,
                    "first_frame_seconds": first_frame, "duration_seconds": duration,
                    "length_limit_reached": finish_reason == "length", "finish_reason": finish_reason}
        metadata["gdn_backend"] = self.gdn_backend
        if getattr(self,'last_load_diagnostics',None) is not None:
            metadata['checkpoint_load']=self.last_load_diagnostics
        metadata["requested_gdn_backend"] = self.requested_backend
        metadata["actual_gdn_backends"] = sorted({layer.linear_attn.backend for layer in drama.model.model.layers
                                                  if layer.layer_type == "linear_attention"})
        metadata["acceleration_fallback"] = self.acceleration_fallback or next((layer.linear_attn.fallback_reason
            for layer in drama.model.model.layers if layer.layer_type == "linear_attention" and layer.linear_attn.fallback_reason),None)
        metadata["gdn_calls_by_backend"] = {backend:sum(layer.linear_attn.backend_calls[backend]
            for layer in drama.model.model.layers if layer.layer_type == "linear_attention") - calls_before[backend]
            for backend in ("torch","fla")}
        metadata["gdn_calls"] = sum(metadata["gdn_calls_by_backend"].values())
        stats=self.expert_stats()
        metadata['expert_placement']=stats['placement']
        metadata['expert_requested_placement']=self.expert_placement
        metadata["moe_requested_backend"]=self.moe_backend
        metadata["moe_backend"]=stats['actual_backend']
        metadata["moe_fallback"]=stats['fallback_reason']
        metadata['expert_backend_calls']={key:count-experts_before['backend_calls'].get(key,0)
            for key,count in stats['backend_calls'].items()}
        metadata['cpu_experts_run']=stats['cpu_experts_run']-cpu_before
        metadata['cpu_decode_calls']=stats['cpu_decode_calls']-experts_before['cpu_decode_calls']
        metadata["cuda_graph"]=getattr(drama,"last_graph_stats",dict(captures=0,replays=0))
        if expected_segments is not None:
            metadata["scene_boundary_mode"]="eos_to_newline_once"
            metadata["scene_boundary_corrections"]=boundary_corrections
        if duration <= 0 and not (allow_empty and finish_reason == "eos"):
            raise SpeechGenerationError("The model stopped without producing audio; retry with another seed or reference",attempts)
        return Take(prompt_text, transcript, frames, drama.config.semantic_start_token_id,
                    drama.config.semantic_end_token_id,
                    drama.tokenizer.token_to_id(SEPARATOR), metadata)

    def speak(self, text, session=None, sampling=None, on_codes=None, progress=None, *, note="", system_context=""):
        session = session or Session()
        sampling = sampling or Sampling()
        request = with_speaker_tag(directed_text(text, session.direction, note))
        drama = self.load_model()
        prompt = tts_prompt(drama.tokenizer, drama.config, request, session.references, session.takes, system_context=system_context)
        take = self._take(prompt, request, spoken_text(text), sampling, on_codes, progress, output_policy="speech")
        take.metadata['raw_request_text'] = text
        if system_context:
            take.metadata['system_context'] = system_context
        return session.append(take)

    def revise(self, session, note, sampling=None, on_codes=None, progress=None):
        if not session.takes:
            raise ValueError("Generate a take before giving a director revision")
        previous=session.takes[-1]
        if 'render_infill' in previous.metadata:
            from .state import validate_session
            validate_session(session)
            policy=previous.metadata['render_infill']
            if policy.get('scene_text'):
                base=previous
                while 'render_infill' in base.metadata:base=base.sources[0]
                lines=[line for line in policy['full_target_transcript'].splitlines() if line.strip()]
                return self.scene(lines,base.metadata.get('scene',''),base.metadata.get('characters',''),
                    session.references,sampling,progress,session=session,note=note,on_codes=on_codes,
                    generation_mode=base.metadata.get('generation_mode','together'),
                    split_long_text=base.metadata.get('scene_chunking',{}).get('enabled',False),
                    chunk_bytes=base.metadata.get('scene_chunking',{}).get('chunk_bytes',300),
                    system_context=previous.metadata.get('system_context',''))
        if 'scene_chunking' in previous.metadata:
            from .scene_chunks import validate_scene
            validate_scene(previous)
        if previous.metadata.get("task")=="scene":
            lines=previous.metadata.get("script_lines")
            if not lines:
                script=previous.prompt_text
                # Older archives stored a global note before the first named
                # line. Remove only our exact recorded prefix on this fallback;
                # keep actor directions and all original archived data intact.
                old_note=previous.metadata.get("director_note")
                prefix=f"[{old_note}] " if old_note else ""
                if prefix and script.startswith(prefix):
                    script=script[len(prefix):]
                lines=script.replace(SEPARATOR,"\n").splitlines()
            lines=[line for line in lines if line.strip()]
            return self.scene(lines,previous.metadata.get("scene",""),previous.metadata.get("characters",""),
                              session.references,sampling,progress,session=session,note=note,on_codes=on_codes,
                              generation_mode=previous.metadata.get('generation_mode','together'),
                              split_long_text=previous.metadata.get('scene_chunking',{}).get('enabled',False),
                              chunk_bytes=previous.metadata.get('scene_chunking',{}).get('chunk_bytes',300),
                              system_context=previous.metadata.get('system_context',''))
        original = spoken_text(previous.transcript)
        request = previous.metadata.get('raw_request_text', original)
        if not isinstance(request, str):
            raise ValueError('Take raw_request_text must be text')
        if spoken_text(request) != original:
            raise ValueError('Take raw_request_text differs from its spoken transcript')
        from .native.prompt import SPEAKER_TAG, WHITE_SPACE
        tag = SPEAKER_TAG.match(previous.prompt_text.lstrip(WHITE_SPACE))
        if tag is not None:
            raw_tag = SPEAKER_TAG.match(request.lstrip(WHITE_SPACE))
            if raw_tag is not None and raw_tag.group(0) != tag.group(0):
                raise ValueError('Take raw_request_text differs from its actual speaker')
            request = tag.group(0) + ' ' + request if raw_tag is None else request
        result = self.speak(request, session, sampling, on_codes, progress, note=note,
                            system_context=previous.metadata.get('system_context',''))
        revised = result.takes[-1]
        take = replace(revised, transcript=original)
        return replace(result, takes=(*result.takes[:-1], take))

    def scene(self, lines, scene="", characters="", references=None, sampling=None, progress=None, session=None, note="",on_codes=None,generation_mode='together',split_long_text=False,chunk_bytes=300,*,system_context=""):
        lines=list(lines)
        from .scene_chunks import options,chunked_scene
        options(split_long_text,chunk_bytes)
        from .dialogue import MODES,isolated_scene
        if generation_mode not in MODES:raise ValueError('Unknown dialogue generation mode')
        if not lines:raise ValueError('The scene contains no lines')
        if split_long_text:
            sampling=sampling or Sampling();sampling.validate()
            return chunked_scene(self,lines,scene,characters,references,sampling,progress,session,note,on_codes,generation_mode,chunk_bytes,system_context=system_context)
        if generation_mode!='together':
            sampling=sampling or Sampling();sampling.validate()
            return isolated_scene(self,lines,scene,characters,references,sampling,progress,session,note,on_codes,generation_mode,system_context=system_context)
        drama = self.load_model()
        controls = {drama.tokenizer.token_to_id(SEPARATOR)}
        if not controls.issubset(drama.config.output_token_ids):
            raise ValueError("This checkpoint lacks scene control weights; use the T8 format-v2 converter")
        if references is not None:
            references = tuple(references)
        session=session or Session(references or ())
        if references is None:
            references = session.references
        else:
            session = replace(session, references=references)
        cast_lines,_,speaker_map=directed_scene_lines(lines,references,session.takes,session.direction,note)
        script=("\n" + SEPARATOR + "\n").join(cast_lines)
        request=script
        prompt = scene_prompt(drama.tokenizer, drama.config, lines, scene, characters, references,
                              session.takes, note, session.direction, system_context=system_context)
        take = self._take(prompt, request, spoken_text("\n".join(lines),scene=True), sampling or Sampling(), on_codes=on_codes,progress=progress,
                          full_head=True,expected_segments=len(lines))
        take.metadata["expected_segments"] = len(lines)
        take.metadata["actual_segments"] = len(take.segments())
        take.metadata["task"] = "scene"
        take.metadata['generation_mode']='together'
        take.metadata.update(scene=scene,characters=characters,script_lines=lines,director_note=note)
        if system_context:
            take.metadata['system_context'] = system_context
        if references:
            take.metadata["scene_speaker_map"]=speaker_map
            take.metadata["scene_casting"]="reference_speaker_tags_v1"
        controls = take.frames[(take.frames[:,0] < drama.config.semantic_start_token_id)
                               | (take.frames[:,0] > drama.config.semantic_end_token_id),0].tolist()
        take.metadata["output_controls"] = [drama.tokenizer.id_to_token(t) for t in controls]
        if take.metadata.get("length_limit_reached"):
            raise RuntimeError("Scene generation reached max_frames before finishing the script")
        if len(take.segments()) != len(lines) or any(s.shape[-1] == 0 for s in take.segments()):
            raise RuntimeError(f"Scene generation returned {len(take.segments())}/{len(lines)} segments; controls: {take.metadata['output_controls'][:40]}")
        return session.append(take)

    def decode_take(self,take,*,with_timeline=False):
        if 'render_infill' in take.metadata:
            from .state import validate_session
            validate_session(Session(takes=(take,)))
            from .render_infill import decode_rendered,render_timeline
            audio,timeline=decode_rendered(take),render_timeline(take)
        elif 'scene_chunking' in take.metadata:
            from .scene_chunks import decode_scene
            audio,timeline=decode_scene(self,take)
        elif take.metadata.get('decode_sources'):
            from .dialogue import decode_isolated
            audio,timeline=decode_isolated(self,take)
        else:
            audio=self.decode(take.codes)
            timeline=[]
        return (audio,timeline) if with_timeline else audio

    def infill(self, source_take, target_text, start_seconds, end_seconds, sampling=None,
               duration_frames=None, progress=None,target_speaker=None,expected_view=None):
        if (source_take.metadata.get('decode_sources') or 'scene_chunking' in source_take.metadata or
                'render_infill' in source_take.metadata or source_take.metadata.get('audio_editor') or
                target_speaker is not None or expected_view is not None):
            from .render_infill import infill_take
            if expected_view is not None:
                from .editor import json_object
                expected_view=json_object(expected_view,'expected_view',maximum=10000)
            return infill_take(self,source_take,target_text,start_seconds,end_seconds,sampling,
                duration_frames,progress,target_speaker=target_speaker,expected_view=expected_view)
        codes = source_take.codes
        duration = codes.shape[-1] * FRAME_LENGTH / SAMPLE_RATE
        if not 0 <= start_seconds <= end_seconds <= duration:
            raise ValueError(f"Edit interval must be within 0–{duration:.3f} seconds")
        start = min(codes.shape[-1], round(start_seconds * SAMPLE_RATE / FRAME_LENGTH))
        end = min(codes.shape[-1], round(end_seconds * SAMPLE_RATE / FRAME_LENGTH))
        drama = self.load_model()
        prefix, suffix = codes[:, :start], codes[:, end:]
        prompt = infill_prompt(drama.tokenizer, drama.config, target_text, prefix, suffix, duration_frames)
        middle = self._take(prompt, target_text, target_text, sampling or Sampling(), progress=progress, allow_empty=True)
        edited = torch.cat((prefix, middle.codes, suffix), dim=1)
        frames = torch.cat((edited[:1] + drama.config.semantic_start_token_id, edited), dim=0).T
        return Take(target_text, target_text, frames, metadata={**middle.metadata,
                    "duration_seconds": edited.shape[-1] * FRAME_LENGTH / SAMPLE_RATE,
                    "inserted_duration_seconds": middle.codes.shape[-1] * FRAME_LENGTH / SAMPLE_RATE,
                    "source_take_id": source_take.take_id, "edit_start_frame": start,
                    "edit_end_frame": end, "original_frames": codes.shape[-1],
                    "inserted_frames": middle.codes.shape[-1], "preserved_prefix_suffix_codes": True})

    def long_form(self, text, session=None, sampling=None, progress=None, on_codes=None, chunk_bytes=300):
        session = session or Session()
        sampling = sampling or Sampling()
        sampling.validate()
        if not 32 <= int(chunk_bytes) <= 3000:
            raise ValueError("Chunk byte budget must be between 32 and 3000")
        style = direction_tags(session.direction)
        reserve = len(style.encode('utf-8')) + 1 + len('<|speaker:15|> '.encode('utf-8')) if style else 0
        if reserve >= int(chunk_bytes):
            raise ValueError('Direction controls cannot fit chunk_bytes including speaker overhead; increase chunk_bytes')
        chunks = split_text(text, max_bytes=int(chunk_bytes) - reserve)
        if style and any(len(with_speaker_tag(directed_text(chunk, style)).encode('utf-8')) > int(chunk_bytes) for chunk in chunks):
            raise ValueError('Directed long-form request exceeds chunk_bytes; increase chunk_bytes for protected spans')
        if not chunks:
            raise ValueError("Long-form text is empty")
        voice_references = session.references
        anchor = None
        with self.generation_session():
            for index, chunk in enumerate(chunks):
                if index == 0:
                    # The first new chunk retains the original bounded conversation.
                    history = session.takes if len(session.takes) < 3 else (session.takes[0], session.takes[-1])
                    context = replace(session, takes=history)
                else:
                    # Inference view only: explicit user references take priority;
                    # otherwise anchor once to the first new chunk, never the latest.
                    if not voice_references:
                        voice_references = (Reference(anchor.prompt_text, anchor.codes),)
                    context = replace(session, references=voice_references, takes=())
                generated = self.speak(chunk, context, replace(sampling, seed=(sampling.seed + index) % 2**64), on_codes, progress)
                take = generated.takes[-1]
                if index == 0:
                    anchor = take
                else:
                    take = replace(take, metadata={**take.metadata, "long_context": {
                        "version": 1,
                        "policy": "explicit_references" if session.references else "first_new_take_reference",
                        "source_take_id": None if session.references else anchor.take_id,
                        "explicit_reference_count": len(session.references),
                        "completed_history_turns": 0,
                    }})
                # Keep the full persistent archive and its original references.
                session = session.append(take)
        return session

    def stream_text(self, text, lookahead="", session=None, sampling=None, on_codes=None, progress=None,*,system_context=""):
        session = session or Session()
        request = with_speaker_tag(directed_text(text, session.direction))
        drama = self.load_model()
        end = drama.tokenizer.token_to_id("<|endofturn|>")
        prompt = streaming_prompt(drama.tokenizer, drama.config, request, lookahead,
                                  session.references, session.takes, system_context=system_context)
        take = self._take(prompt, request, spoken_text(text), sampling or Sampling(), on_codes, progress,
                          stop_token_ids=(end,))
        take = replace(take, metadata={**take.metadata, "task": "text_stream", "lookahead": lookahead})
        take.metadata['raw_request_text'] = text
        if system_context:
            take.metadata['system_context'] = system_context
        return session.append(take)

    def unload(self):
        unload_report=unload_references=None
        try:
            from .memory_diagnostics import begin_unload,finish_unload
            unload_report,unload_references=begin_unload(self)
        except Exception:
            pass  # Optional observations cannot prevent resource cleanup.

        def observed(status='ok',error=None):
            if unload_report is None:return None
            try:return finish_unload(self,unload_report,unload_references,status=status,error=error)
            except Exception:return None

        experts=getattr(self.drama,'experts',None)
        close=getattr(experts,'close',None)
        failure=None
        if close is not None:
            try:
                close()
            except BaseException as exc:
                # An active store intentionally rejects destruction. Keep its
                # owning engine and every resource intact in that case.
                if getattr(experts,'cache',None) is not None or getattr(experts,'active',False):
                    observed('refused_active',exc)
                    raise
                failure=exc
        experts=close=None

        def cleanup(operation):
            nonlocal failure
            try:
                operation()
            except BaseException as exc:
                if failure is None:
                    failure=exc
                else:
                    # Preserve the primary error; diagnostics cannot replace it.
                    try:
                        import logging
                        logging.warning('Drama 3 secondary unload cleanup failed: %s: %s',
                                        type(exc).__name__,str(exc))
                    except BaseException:
                        pass
            finally:
                operation=None

        cleanup(mm.unload_all_models)
        self.drama = None
        self.codec = None
        self.asr = None
        self.text_head = None
        self.text_head_patcher = None
        self.line_head = None
        self.line_head_patcher = None
        cleanup(gc.collect)

        def release_pins():
            from .native.acceleration import release_unused_pinned_memory
            release_unused_pinned_memory()

        cleanup(release_pins)
        cleanup(mm.soft_empty_cache)
        report=observed('error' if failure is not None else 'ok',failure)
        if failure is not None:
            raise failure
        return report
