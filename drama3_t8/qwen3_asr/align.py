"""Word timestamps by Qwen3-ForcedAligner, as qwen_asr's `Qwen3ForcedAligner.align` makes them: after the audio, each
word of a known text is followed by two <timestamp> tokens, and one forward pass classifies each of them into a step
of `timestamp_segment_time` ms, the word's start and end. Words are white-space separated runs of letters, digits and
apostrophes; each CJK character is a word of its own (Japanese and Korean dictionaries are not used)."""
import unicodedata

import torch

import comfy.model_management
from .asr import AUDIO_END, AUDIO_PAD, AUDIO_START, log_mel, prepare_audio
from .model import audio_tokens

CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
       (0xF900, 0xFAFF))


def is_cjk(ch):
    return any(first <= ord(ch) <= last for first, last in CJK)


def words(text):
    """The aligner's units of `text`, in order."""
    units = []
    for piece in text.split():
        cleaned = "".join(ch for ch in piece if ch == "'" or unicodedata.category(ch)[0] in "LN")
        run = ""
        for ch in cleaned:
            if is_cjk(ch):
                if run:
                    units.append(run)
                    run = ""
                units.append(ch)
            else:
                run += ch
        if run:
            units.append(run)
    return units


def monotonic(values):
    """qwen_asr's fix_timestamp: keep the longest non-decreasing run, give runs of up to two stray values their nearer
    neighbour's time and spread longer ones evenly between their neighbours."""
    n = len(values)
    best, parent = [1] * n, [-1] * n
    for i in range(1, n):
        for j in range(i):
            if values[j] <= values[i] and best[j] + 1 > best[i]:
                best[i], parent[i] = best[j] + 1, j
    normal = [False] * n
    index = best.index(max(best)) if n else -1
    while index != -1:
        normal[index] = True
        index = parent[index]
    result = list(values)
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue
        j = i
        while j < n and not normal[j]:
            j += 1
        left = next((result[k] for k in range(i - 1, -1, -1) if normal[k]), None)
        right = next((values[k] for k in range(j, n) if normal[k]), None)
        for k in range(i, j):
            if j - i <= 2:
                if left is None or right is None:
                    result[k] = right if left is None else left
                else:
                    result[k] = left if k - (i - 1) <= j - k else right
            elif left is not None and right is not None:
                result[k] = left + (right - left) / (j - i + 1) * (k - i + 1)
            else:
                result[k] = right if left is None else left
        i = j
    return [int(value) for value in result]


def align(aligner, waveform, sample_rate, text):
    """[(word, start s, end s)] for the words of `text` spoken in [channels, samples] audio."""
    model = aligner.model
    if not model.aligner:
        raise ValueError("word timestamps need Qwen3-ForcedAligner (tools/convert.py asr), not a Qwen3-ASR model")
    units = words(text)
    if not units:
        return []
    device = aligner.patcher.load_device
    features = log_mel(prepare_audio(waveform.to(device), sample_rate))
    stamp = model.timestamp_token_id
    ids = [AUDIO_START, *[AUDIO_PAD] * audio_tokens(features.shape[-1]), AUDIO_END]
    for unit in units:
        ids += [*aligner.encode(unit), stamp, stamp]
    ids = torch.tensor(ids, device=device)
    comfy.model_management.load_models_gpu([aligner.patcher])
    with comfy.model_management.cuda_device_context(device):
        embeds = model.decoder.model.embed_tokens(ids, out_dtype=aligner.compute_dtype)
        embeds[ids == AUDIO_PAD] = model.audio_tower(features.to(aligner.compute_dtype))
        hidden = model.decoder.model(None, embeds=embeds[None])[0][0]
        steps = model.timestamp_head(hidden[ids == stamp]).argmax(-1).tolist()
    times = monotonic([step * model.timestamp_segment_time for step in steps])
    return [(unit, times[2 * i] / 1000, times[2 * i + 1] / 1000) for i, unit in enumerate(units)]
