"""Reference ASR uses the same sample-rounded interval as reference encoding."""
import math


def reference_interval(start_seconds=0, end_seconds=None):
    start = float(start_seconds)
    end = None if end_seconds in (None, "") else float(end_seconds)
    if not math.isfinite(start) or start < 0 or end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("转写选区必须是有效的起止秒数")
    return start, end


def reference_asr_audio(waveform, sample_rate, start_seconds=0, end_seconds=None):
    start, end = reference_interval(start_seconds, end_seconds)
    duration = waveform.shape[-1] / sample_rate
    end = duration if end is None else end
    if not 0 <= start < end <= duration + 1 / sample_rate:
        raise ValueError("转写选区必须位于原录音内")
    selected = waveform[..., round(start * sample_rate):round(end * sample_rate)]
    if selected.shape[-1] == 0:
        raise ValueError("转写选区没有有效音频样本")
    return selected, {"start_seconds": start, "end_seconds": min(end, duration)}
