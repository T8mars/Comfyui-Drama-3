"""Direct tensor inference shared by the local app and native ComfyUI nodes."""
from dataclasses import dataclass, replace
import time
import math
from pathlib import Path

import torch
import comfy.audio
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .native.codec import SAMPLE_RATE, FRAME_LENGTH
from .native.generate import generate
from .native.loader import load_drama3, load_codec
from .native.chunking import split_text
from .prompts import (
    SEPARATOR, infill_prompt, scene_prompt, streaming_prompt, tts_prompt, spoken_text,
)
from .state import Reference, Session, Take, validate_codes
from .audio import audio_waveform


@dataclass(frozen=True)
class Sampling:
    seed: int = 42
    temperature: float = 0.7
    top_k: int = 30
    top_p: float = 0.7
    max_frames: int = 600

    def validate(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("Seed must be an integer from 0 to 2**64-1")
        if (type(self.max_frames) is not int or type(self.top_k) is not int
                or self.max_frames < 1 or self.top_k < 1
                or not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1):
            raise ValueError("Invalid sampling limits")


class DramaEngine:
    def __init__(self, model_path, codec_path, expert_cache_gb=12.0, gdn_backend="torch", *,
                 text_head_path=None, asr_path=None, asr_backend="qwen", r2t2_paths=None):
        self.model_path = Path(model_path)
        self.codec_path = Path(codec_path)
        self.text_head_path = Path(text_head_path) if text_head_path else self.model_path.parent / "drama3-text-head.safetensors"
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
            self.text_head = torch.nn.Sequential(projection)
            self.text_head_patcher = comfy.model_patcher.ModelPatcher(self.text_head, load_device=mm.get_torch_device(),
                                                                   offload_device=mm.unet_offload_device())
        mm.load_models_gpu([*self.drama.patchers(), self.text_head_patcher])
        return self.text_head, tuple(range(self.text_head[0].out_features))

    def load_model(self):
        if self.drama is None:
            self.drama = load_drama3(str(self.model_path), experts_in_ram=True,
                                    expert_cache_gb=self.expert_cache_gb)
            self.set_backend(self.gdn_backend)
        mm.load_models_gpu(self.drama.patchers())
        return self.drama

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

    def load_codec(self):
        if self.codec is None:
            self.codec = load_codec(str(self.codec_path))
        mm.load_models_gpu([self.codec.patcher])
        return self.codec

    def encode_reference(self, waveform, sample_rate, transcript, name=""):
        waveform = audio_waveform(waveform, sample_rate)
        waveform = waveform.mean(0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = comfy.audio.resample(waveform, sample_rate, SAMPLE_RATE)
        codec = self.load_codec()
        with mm.cuda_device_context(codec.patcher.load_device):
            codes = codec.model.encode(waveform.to(codec.patcher.load_device))[0].cpu()
        return Reference(transcript, codes, name)

    def transcribe(self, waveform, sample_rate, language=None, context=""):
        result = self.transcribe_detailed(waveform, sample_rate, language=language, context=context)
        return result["text"], result["language"]

    def transcribe_detailed(self, waveform, sample_rate, language=None, context="", hotwords="", check_cancel=None):
        waveform = audio_waveform(waveform, sample_rate)[None]
        if self.asr_backend == "r2t2":
            from .r2t2.transcribe import transcribe_audio
            return transcribe_audio({"waveform": waveform, "sample_rate": sample_rate},
                self.r2t2_paths["r2t2_model"], self.r2t2_paths["r2t2_projector"], self.r2t2_paths["r2t2_vad"],
                language=language, context=context, hotwords=hotwords, check_cancel=check_cancel)
        from .r2t2.hotwords import hotword_context
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
        codec = self.load_codec()
        device = codec.patcher.load_device
        with mm.cuda_device_context(device):
            waveform = codec.model.decode(codes[None].to(device=device, dtype=torch.long), torch.float32).float().cpu()
        return {"waveform": waveform[:, None], "sample_rate": SAMPLE_RATE}

    def _take(self, prompt, prompt_text, transcript, sampling, on_codes=None, progress=None,
              stop_token_ids=(), full_head=False, allow_empty=False):
        sampling.validate()
        drama = self.load_model()
        head = self.load_text_head() if full_head else None
        calls_before = {backend:sum(layer.linear_attn.backend_calls[backend]
                           for layer in drama.model.model.layers if layer.layer_type == "linear_attention")
                        for backend in ("torch","fla")}
        start = time.perf_counter()
        first_frame = None
        finish_reason = None

        def finished(reason):
            nonlocal finish_reason
            finish_reason = reason

        def audio_callback(frame):
            nonlocal first_frame
            if first_frame is None:
                first_frame = time.perf_counter() - start
            if on_codes is not None:
                on_codes(frame)

        frames = generate(drama, prompt, sampling.max_frames, sampling.temperature,
                          sampling.top_k, sampling.top_p, sampling.seed,
                          on_audio=audio_callback, progress=progress, stop_token_ids=stop_token_ids,
                          on_finish=finished, head_override=head,
                          extra_patchers=(self.text_head_patcher,) if full_head else ())
        if finish_reason == "stalled":
            raise RuntimeError("The model entered a silent repetition; use another seed or a more accurate reference transcript")
        duration = ((frames[:, 0] >= drama.config.semantic_start_token_id)
                    & (frames[:, 0] <= drama.config.semantic_end_token_id)).sum().item() * FRAME_LENGTH / SAMPLE_RATE
        metadata = {"seed": sampling.seed, "sampling": sampling.__dict__,
                    "model": self.model_path.name, "elapsed_seconds": time.perf_counter() - start,
                    "first_frame_seconds": first_frame, "duration_seconds": duration,
                    "length_limit_reached": finish_reason == "length", "finish_reason": finish_reason}
        metadata["gdn_backend"] = self.gdn_backend
        metadata["requested_gdn_backend"] = self.requested_backend
        metadata["actual_gdn_backends"] = sorted({layer.linear_attn.backend for layer in drama.model.model.layers
                                                  if layer.layer_type == "linear_attention"})
        metadata["acceleration_fallback"] = self.acceleration_fallback or next((layer.linear_attn.fallback_reason
            for layer in drama.model.model.layers if layer.layer_type == "linear_attention" and layer.linear_attn.fallback_reason),None)
        metadata["gdn_calls_by_backend"] = {backend:sum(layer.linear_attn.backend_calls[backend]
            for layer in drama.model.model.layers if layer.layer_type == "linear_attention") - calls_before[backend]
            for backend in ("torch","fla")}
        metadata["gdn_calls"] = sum(metadata["gdn_calls_by_backend"].values())
        if duration <= 0 and not (allow_empty and finish_reason == "eos"):
            raise RuntimeError("The model stopped without producing audio; retry with another seed or reference")
        return Take(prompt_text, transcript, frames, drama.config.semantic_start_token_id,
                    drama.config.semantic_end_token_id,
                    drama.tokenizer.token_to_id(SEPARATOR), metadata)

    def speak(self, text, session=None, sampling=None, on_codes=None, progress=None):
        session = session or Session()
        sampling = sampling or Sampling()
        drama = self.load_model()
        prompt = tts_prompt(drama.tokenizer, drama.config, text, session.references,
                            session.takes, session.direction)
        take = self._take(prompt, text, spoken_text(text), sampling, on_codes, progress)
        return session.append(take)

    def revise(self, session, note, sampling=None, on_codes=None, progress=None):
        if not session.takes:
            raise ValueError("Generate a take before giving a director revision")
        previous=session.takes[-1]
        if previous.metadata.get("task")=="scene":
            lines=previous.metadata.get("script_lines") or previous.prompt_text.replace(SEPARATOR,"\n").splitlines()
            lines=[line for line in lines if line.strip()]
            return self.scene(lines,previous.metadata.get("scene",""),previous.metadata.get("characters",""),
                              session.references,sampling,progress,session=session,note=note,on_codes=on_codes)
        original = spoken_text(previous.transcript)
        request = f"[{note}] {original}"
        result = self.speak(request, session, sampling, on_codes, progress)
        revised = result.takes[-1]
        take = replace(revised, transcript=original)
        return replace(result, takes=(*result.takes[:-1], take))

    def scene(self, lines, scene="", characters="", references=None, sampling=None, progress=None, session=None, note="",on_codes=None):
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
        script=("\n" + SEPARATOR + "\n").join(lines)
        request=(f"[{note}] " if note else "")+script
        prompt = scene_prompt(drama.tokenizer, drama.config, lines, scene, characters, references,
                              session.takes, note, session.direction)
        take = self._take(prompt, request, spoken_text("\n".join(lines),scene=True), sampling or Sampling(), on_codes=on_codes,progress=progress,
                          full_head=True)
        take.metadata["expected_segments"] = len(lines)
        take.metadata["actual_segments"] = len(take.segments())
        take.metadata["task"] = "scene"
        take.metadata.update(scene=scene,characters=characters,script_lines=lines,director_note=note)
        controls = take.frames[(take.frames[:,0] < drama.config.semantic_start_token_id)
                               | (take.frames[:,0] > drama.config.semantic_end_token_id),0].tolist()
        take.metadata["output_controls"] = [drama.tokenizer.id_to_token(t) for t in controls]
        if len(take.segments()) != len(lines) or any(s.shape[-1] == 0 for s in take.segments()):
            raise RuntimeError(f"Scene generation returned {len(take.segments())}/{len(lines)} segments; controls: {take.metadata['output_controls'][:40]}")
        return session.append(take)

    def infill(self, source_take, target_text, start_seconds, end_seconds, sampling=None,
               duration_frames=None, progress=None):
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
        chunks = split_text(text, max_bytes=int(chunk_bytes))
        if not chunks:
            raise ValueError("Long-form text is empty")
        for index, chunk in enumerate(chunks):
            # Keep the complete archive while bounding inference context to the first and latest take.
            history = session.takes if len(session.takes) < 3 else (session.takes[0], session.takes[-1])
            context = replace(session, takes=history)
            generated = self.speak(chunk, context, replace(sampling, seed=(sampling.seed + index) % 2**64), on_codes, progress)
            session = session.append(generated.takes[-1])
        return session

    def stream_text(self, text, lookahead="", session=None, sampling=None, on_codes=None, progress=None):
        session = session or Session()
        drama = self.load_model()
        end = drama.tokenizer.token_to_id("<|endofturn|>")
        prompt = streaming_prompt(drama.tokenizer, drama.config, text, lookahead,
                                  session.references, session.takes, session.direction)
        take = self._take(prompt, text, spoken_text(text), sampling or Sampling(), on_codes, progress,
                          stop_token_ids=(end,))
        take = replace(take, metadata={**take.metadata, "task": "text_stream", "lookahead": lookahead})
        return session.append(take)

    def unload(self):
        mm.unload_all_models()
        self.drama = None
        self.codec = None
        self.asr = None
        self.text_head = None
        self.text_head_patcher = None
        mm.soft_empty_cache()
