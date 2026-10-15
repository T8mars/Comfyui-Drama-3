"""File transcription using the T8 R2T2 Q8 engine and FireRedVAD boundaries."""
import time
from pathlib import Path

import numpy as np

from .audio import comfy_audio_to_16k, limited_quiet_gain
from .hotwords import hotword_context
from .native import NativeQ8Engine, build_prompt


def transcribe_audio(audio, model, projector, vad_dir, *, language=None, context="", hotwords="",
                     check_cancel=None):
    from .segmented import PresetBoundaryVAD, SegmentedStream, exact_zero_boundaries, join_segment_text
    from .vad import FireRedOnnxVAD
    check_cancel = check_cancel or (lambda: None)
    language = None if language in (None, "", "auto", "Auto") else language
    context = hotword_context(context, hotwords)
    # Validate before releasing other models or allocating native weights.
    build_prompt(context, language)
    pcm, gain = limited_quiet_gain(comfy_audio_to_16k(audio))
    start = time.perf_counter()
    check_cancel()
    result = {"text": "", "language": "", "segments": [], "truncated": False,
              "finish_reason": "stop", "mode_executed": "digital_silence"}
    if np.any(pcm):
        import comfy.model_management as mm
        # llama.cpp allocations are outside Torch's allocator; release managed GPU models first.
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            with NativeQ8Engine(model, projector, check_cancel=check_cancel) as engine:
                cuts = exact_zero_boundaries(pcm) if len(pcm) <= 30 * 16000 else []
                if len(pcm) <= 30 * 16000 and not cuts:
                    result = {**engine.transcribe(pcm, context=context, language=language),
                              "mode_executed": "offline", "segments": []}
                else:
                    vad = (PresetBoundaryVAD(cuts) if cuts else FireRedOnnxVAD(Path(vad_dir)))
                    session = SegmentedStream(engine, language=language, context=context, offline=True, vad=vad)
                    for pos in range(0, len(pcm), 640):
                        check_cancel()
                        session.feed(pcm[pos:pos + 640])
                    terminal = session.finish()
                    result = {"text": terminal["stable_text"],
                              "language": session.detected_language or language or "",
                              "finish_reason": terminal.get("finish_reason"),
                              "truncated": terminal.get("truncated", False),
                              "segments": session.segments,
                              "forced_boundaries": session.forced_boundaries,
                              "mode_executed": "offline_zero_split" if cuts else "segmented_offline"}
        finally:
            mm.soft_empty_cache()
    check_cancel()
    return {**result, "backend": "r2t2", "duration_seconds": len(pcm) / 16000,
            "elapsed_seconds": time.perf_counter() - start, "input_gain": gain,
            "timing_method": "segment_estimate", "model_path": str(model),
            "projector_path": str(projector), "vad_path": str(vad_dir),
            "execution": "in_process_native", "model_released": True}
