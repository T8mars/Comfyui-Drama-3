"""Shared single-clip waveform contract for codec, ASR and forced alignment."""
import math
import numbers

import torch


def audio_waveform(waveform, sample_rate):
    """Validate one clip and return floating samples [channels, time]."""
    if not isinstance(waveform, torch.Tensor) or waveform.ndim not in (1, 2, 3) or waveform.is_complex():
        raise ValueError("Audio must be a real waveform tensor [samples], [channels,samples] or [1,channels,samples]")
    if (not isinstance(sample_rate, numbers.Real) or isinstance(sample_rate, bool)
            or not math.isfinite(sample_rate) or sample_rate <= 0 or int(sample_rate) != sample_rate):
        raise ValueError("Audio sample rate must be a positive integer")
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError("Encode or transcribe one audio clip at a time; AUDIO batch must be 1")
        waveform = waveform[0]
    if not waveform.numel() or not torch.isfinite(waveform).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    if waveform.ndim == 1:
        waveform = waveform[None]
    return waveform.float()
