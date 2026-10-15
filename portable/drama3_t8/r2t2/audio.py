"""Convert ComfyUI AUDIO and WAV samples to the model's 16 kHz mono PCM."""

from __future__ import annotations

from typing import Literal

import numpy as np
import soxr

Channel = Literal["mean", "left", "right"]


def limited_quiet_gain(pcm16k: np.ndarray) -> tuple[np.ndarray, float]:
    """Raise only exceptionally low-level file audio, with at most 20 dB gain."""
    value = np.asarray(pcm16k, dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Expected finite mono PCM")
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if not 0.0 < peak < 0.001:
        return value, 1.0
    gain = min(10.0, 0.003 / peak)
    return np.ascontiguousarray(value * np.float32(gain)), gain


def to_mono_16k(samples: np.ndarray, sample_rate: int, channel: Channel = "mean") -> np.ndarray:
    """Accept [time] or [time, channels] float audio without loudness changes."""
    if not isinstance(sample_rate, int) or not 8_000 <= sample_rate <= 192_000:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    value = np.asarray(samples)
    if value.ndim == 1:
        mono = value
    elif value.ndim == 2:
        if value.shape[1] < 1 or value.shape[1] > 8:
            raise ValueError(f"Invalid channel count: {value.shape[1]}")
        if channel == "mean":
            mono = value.mean(axis=1)
        elif channel == "left":
            mono = value[:, 0]
        elif channel == "right" and value.shape[1] > 1:
            mono = value[:, 1]
        else:
            raise ValueError(f"Channel selection unavailable: {channel}")
    else:
        raise ValueError(f"Expected [time] or [time, channels], got {value.shape}")
    if mono.size == 0:
        raise ValueError("Audio is empty")
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("Audio contains NaN or infinity")
    if sample_rate != 16_000:
        mono = np.ascontiguousarray(soxr.resample(mono, sample_rate, 16_000, quality="HQ"), dtype=np.float32)
        if not mono.size:
            raise ValueError("Audio is too short to resample to 16 kHz")
    return mono


def comfy_audio_to_16k(audio: dict, channel: Channel = "mean") -> np.ndarray:
    """Convert ComfyUI AUDIO waveform [batch, channels, time], requiring B=1."""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected ComfyUI AUDIO with waveform and sample_rate")
    if type(audio["sample_rate"]) is not int:
        raise ValueError("Audio sample_rate must be an integer")
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    waveform = np.asarray(waveform)
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError(f"Expected AUDIO [1, channels, time], got {waveform.shape}")
    return to_mono_16k(waveform[0].T, audio["sample_rate"], channel)
