"""File transcription using the T8 R2T2 Q8 engine and FireRedVAD boundaries."""
import time
from contextlib import ExitStack
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .audio import comfy_audio_to_16k, limited_quiet_gain
from .hotwords import hotword_context
from .native import NativeQ8Engine, build_prompt, SUPPORTED_LANGUAGES
from ..asr_language import normalize_asr_language


def _validate_paths(model, projector):
    for role, location in (("model", model), ("projector", projector)):
        if not Path(location).is_file():
            raise FileNotFoundError(f"Confucius R2T2 {role} file not found: {location}")


def _silent_result():
    return {"text": "", "language": "", "segments": [], "truncated": False,
            "finish_reason": "stop", "mode_executed": "digital_silence"}


def _transcribe_pcm(pcm, engine, vad_dir, language, context, check_cancel):
    """One independent recording; boundary/VAD state never crosses files."""
    from .segmented import PresetBoundaryVAD, SegmentedStream, exact_zero_boundaries
    from .vad import FireRedOnnxVAD
    cuts = exact_zero_boundaries(pcm) if len(pcm) <= 30 * 16000 else []
    if len(pcm) <= 30 * 16000 and not cuts:
        return {**engine.transcribe(pcm, context=context, language=language),
                "mode_executed": "offline", "segments": []}
    vad = PresetBoundaryVAD(cuts) if cuts else FireRedOnnxVAD(Path(vad_dir))
    session = SegmentedStream(engine, language=language, context=context, offline=True, vad=vad)
    for pos in range(0, len(pcm), 640):
        check_cancel()
        session.feed(pcm[pos:pos + 640])
    terminal = session.finish()
    return {"text": terminal["stable_text"], "language": session.detected_language or language or "",
            "finish_reason": terminal.get("finish_reason"), "truncated": terminal.get("truncated", False),
            "segments": session.segments, "forced_boundaries": session.forced_boundaries,
            "mode_executed": "offline_zero_split" if cuts else "segmented_offline"}


def _metadata(result, pcm, gain, start, model, projector, vad_dir, *, released):
    return {**result, "backend": "r2t2", "duration_seconds": len(pcm) / 16000,
            "elapsed_seconds": time.perf_counter() - start, "input_gain": gain,
            "timing_method": "segment_estimate", "model_path": str(model),
            "projector_path": str(projector), "vad_path": str(vad_dir),
            "execution": "in_process_native", "model_released": released}


def transcribe_audio(audio, model, projector, vad_dir, *, language=None, context="", hotwords="",
                     check_cancel=None):
    language = normalize_asr_language(language, SUPPORTED_LANGUAGES, backend="Confucius")
    from .segmented import PresetBoundaryVAD, SegmentedStream, exact_zero_boundaries, join_segment_text
    from .vad import FireRedOnnxVAD
    check_cancel = check_cancel or (lambda: None)
    context = hotword_context(context, hotwords)
    # Validate before releasing other models or allocating native weights.
    build_prompt(context, language)
    pcm, gain = limited_quiet_gain(comfy_audio_to_16k(audio))
    start = time.perf_counter()
    check_cancel()
    # A silent recording avoids model allocation, but it must not turn a
    # missing configuration into successful transcription. Keep this shared
    # by node and Studio callers; VAD is required only by its actual branch.
    _validate_paths(model, projector)
    result = _silent_result()
    if np.any(pcm):
        import comfy.model_management as mm
        # llama.cpp allocations are outside Torch's allocator; release managed GPU models first.
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            with NativeQ8Engine(model, projector, check_cancel=check_cancel) as engine:
                result = _transcribe_pcm(pcm, engine, vad_dir, language, context, check_cancel)
        finally:
            mm.soft_empty_cache()
    check_cancel()
    return _metadata(result, pcm, gain, start, model, projector, vad_dir, released=True)


def transcribe_audio_batch(audios, model, projector, vad_dir, *, language=None, context="", hotwords="",
                           check_cancel=None, max_items=256):
    """Transcribe ordered independent recordings with one lazy native engine.

    PCM conversion and VAD state stay per-file; NativeQ8Engine resets native KV
    for each transcribe call. All-silent batches allocate no native/Torch model.
    Results are returned only after the shared engine has closed successfully.
    Per-file elapsed time excludes shared batch close; batch_elapsed_seconds
    includes conversion, model load, inference, and cleanup. No partial results
    are returned on failure/cancellation. An empty batch still validates paths.
    """
    if isinstance(audios, Mapping):
        raise TypeError("Use transcribe_audio for one AUDIO; batch expects an iterable of AUDIO objects")
    if type(max_items) is not int or not 1 <= max_items <= 10000:
        raise ValueError("max_items must be between1 and10000")
    check_cancel = check_cancel or (lambda: None)
    language = normalize_asr_language(language, SUPPORTED_LANGUAGES, backend="Confucius")
    context = hotword_context(context, hotwords)
    build_prompt(context, language)
    started = time.perf_counter()
    check_cancel()
    _validate_paths(model, projector)
    results, engine, mm = [], None, None
    cleanup_needed = False
    try:
        with ExitStack() as stack:
            for index, audio in enumerate(audios):
                check_cancel()
                if index >= max_items:raise ValueError("R2T2 audio batch exceeds max_items")
                pcm, gain = limited_quiet_gain(comfy_audio_to_16k(audio))
                start = time.perf_counter()
                check_cancel()
                _validate_paths(model, projector)
                result = _silent_result()
                if np.any(pcm):
                    if engine is None:
                        import comfy.model_management as mm
                        mm.unload_all_models()
                        mm.soft_empty_cache()
                        cleanup_needed = True
                        engine = stack.enter_context(NativeQ8Engine(model, projector, check_cancel=check_cancel))
                    result = _transcribe_pcm(pcm, engine, vad_dir, language, context, check_cancel)
                check_cancel()
                results.append(_metadata(result, pcm, gain, start, model, projector, vad_dir, released=False))
    finally:
        if cleanup_needed:mm.soft_empty_cache()
    check_cancel()
    elapsed = time.perf_counter() - started
    for index, result in enumerate(results):
        result.update(model_released=True, batch_index=index, batch_size=len(results),
                      batch_engine_loads=int(engine is not None), batch_elapsed_seconds=elapsed,
                      timing_scope="per_clip_work_excluding_shared_batch_close")
    return results
