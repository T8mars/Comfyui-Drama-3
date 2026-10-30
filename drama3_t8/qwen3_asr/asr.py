"""Load converted Qwen3-ASR files (tools/convert.py asr) and transcribe audio the way Qwen's qwen_asr package does:
16 kHz mono, Whisper log-mel features, the chat prompt with one audio pad token per encoder token, greedy decoding,
and its output parsing ("language X<asr_text>text"). Audio longer than 20 minutes is cut at quiet points first."""
import json
import os

import torch
from transformers import AddedToken, Qwen2Tokenizer

import comfy.audio
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.text_encoders import llama
from .model import Qwen3ASRModel, audio_tokens
from ..audio import audio_waveform
from ..transcript import join_segment_text
from ..asr_language import normalize_asr_language

LANGUAGES = ["Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian",
             "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish",
             "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"]
SAMPLE_RATE = 16000
N_FFT, HOP, MELS = 400, 160, 128
MAX_SECONDS = 1200
MIN_SECONDS = 0.5
STOP_TOKENS = [151643, 151645]
IM_START, IM_END = 151644, 151645
AUDIO_START, AUDIO_END, AUDIO_PAD = 151669, 151670, 151676
ASR_TEXT = "<asr_text>"
# Qwen3-ASR's vocabulary and merges are core's Qwen2.5 ones; its added tokens follow them in this list
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(llama.__file__)), "qwen25_tokenizer")
ADDED_TOKENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "added_tokens.json")


class Qwen3ASR:
    def __init__(self, model, patcher, compute_dtype):
        self.model = model
        self.patcher = patcher
        self.compute_dtype = compute_dtype
        self.tokenizer = Qwen2Tokenizer(os.path.join(VOCAB_PATH, "vocab.json"), os.path.join(VOCAB_PATH, "merges.txt"))
        added = json.loads(open(ADDED_TOKENS_PATH, encoding="utf-8").read())
        self.tokenizer.add_tokens([AddedToken(content, special=special, normalized=False) for content, special in added["tokens"]])
        self.special_ids = {added["first_id"] + index for index, (_, special) in enumerate(added["tokens"]) if special}
        if self.tokenizer.convert_tokens_to_ids(added["tokens"][-1][0]) != added["first_id"] + len(added["tokens"]) - 1:
            raise ValueError("Qwen3-ASR's added tokens did not land on their ids")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode([i for i in ids if i not in self.special_ids], clean_up_tokenization_spaces=False)


def load_qwen3_asr(path):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if not metadata or "qwen3_asr_config" not in metadata:
        raise ValueError(f"{os.path.basename(path)} is not a converted Qwen3-ASR file (tools/convert.py asr)")
    config = json.loads(metadata["qwen3_asr_config"])
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    compute_dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(load_device) else torch.float32
    operations = comfy.ops.pick_operations(torch.bfloat16, compute_dtype, load_device=load_device)
    model = Qwen3ASRModel(config, device=offload_device, dtype=torch.bfloat16, operations=operations)
    comfy.model_management.archive_model_dtypes(model)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    model.load_state_dict(state, strict=True, assign=patcher.is_dynamic())
    return Qwen3ASR(model, patcher, compute_dtype)


def prepare_audio(waveform, sample_rate):
    """[channels, samples] -> mono 16 kHz in [-1, 1], scaled down only if it exceeds that range."""
    audio = audio_waveform(waveform, sample_rate).mean(0)
    if sample_rate != SAMPLE_RATE:
        # librosa's "kaiser_best" filter: the closest of these to the soxr resampling of Qwen's loader
        audio = comfy.audio.resample(audio, sample_rate, SAMPLE_RATE, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                     resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    peak = float(audio.abs().max()) if audio.numel() else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.clamp(-1.0, 1.0)


def split_long(audio):
    """Target MAX_SECONDS per piece, searching for quiet points within +/-5 s.

    Like upstream, a cut may fall up to 5 s after the target. Short pieces are
    padded to MIN_SECONDS; audio within the target is returned whole.
    """
    limit = MAX_SECONDS * SAMPLE_RATE
    shortest = int(MIN_SECONDS * SAMPLE_RATE)
    if len(audio) <= limit:
        return [torch.nn.functional.pad(audio, (0, max(0, shortest - len(audio))))]
    expand, window = 5 * SAMPLE_RATE, SAMPLE_RATE // 10
    pieces, start = [], 0
    while len(audio) - start > limit:
        cut = start + limit
        left, right = max(start, cut - expand), min(len(audio), cut + expand)
        boundary = cut
        if right - left > window:
            loudness = audio[left:right].abs()
            sums = torch.nn.functional.conv1d(loudness[None, None], torch.ones(1, 1, window, device=audio.device))[0, 0]
            first = int(sums.argmin())
            boundary = left + first + int(loudness[first:first + window].argmin())
        boundary = min(max(boundary, start + 1), len(audio))
        pieces.append(audio[start:boundary])
        start = boundary
    pieces.append(audio[start:])
    return [torch.nn.functional.pad(piece, (0, shortest - len(piece))) if len(piece) < shortest else piece for piece in pieces]


def log_mel(audio):
    """Whisper's features without its 30 s padding: [MELS, samples // HOP]."""
    mel = comfy.audio.MelSpectrogram(SAMPLE_RATE, N_FFT, HOP, MELS, f_min=0, f_max=8000).to(audio.device)
    spectrum = mel(audio)[:, :-1].clamp(min=1e-10).log10()
    return (torch.maximum(spectrum, spectrum.max() - 8.0) + 4.0) / 4.0


def prompt_ids(asr, context, language, tokens):
    """The chat template's ids, each text run between special tokens tokenized alone as the tokenizer does."""
    language = normalize_asr_language(language, LANGUAGES, backend="Qwen3-ASR")
    ids = [IM_START, *asr.encode("system\n" + context), IM_END, *asr.encode("\n"), IM_START, *asr.encode("user\n"),
           AUDIO_START, *[AUDIO_PAD] * tokens, AUDIO_END, IM_END, *asr.encode("\n"), IM_START]
    if language:
        return ids + asr.encode(f"assistant\nlanguage {language}{ASR_TEXT}")
    return ids + asr.encode("assistant\n")


def fix_repetitions(text, threshold=20):
    """Collapse a character repeated more than `threshold` times, then the first pattern of up to 20 characters
    repeated `threshold` times or more (with the rest of the text after it treated the same way)."""
    out, index = [], 0
    while index < len(text):
        count = 1
        while index + count < len(text) and text[index + count] == text[index]:
            count += 1
        out.append(text[index] if count > threshold else text[index:index + count])
        index += count
    return fix_pattern_repeats("".join(out), threshold)


def fix_pattern_repeats(text, threshold, max_len=20):
    if len(text) < threshold * 2:
        return text
    out, index = [], 0
    while index <= len(text) - threshold * 2:
        for length in range(1, max_len + 1):
            if index + length * threshold > len(text):
                break
            pattern = text[index:index + length]
            if all(text[index + rep * length:index + (rep + 1) * length] == pattern for rep in range(1, threshold)):
                end = index + threshold * length
                while end + length <= len(text) and text[end:end + length] == pattern:
                    end += length
                return "".join(out) + pattern + fix_pattern_repeats(text[end:], threshold, max_len)
        out.append(text[index])
        index += 1
    return "".join(out) + text[index:]


def parse_output(raw, language):
    """(language, text) from the decoded output, as qwen_asr's parse_asr_output."""
    text = raw.strip()
    if not text:
        return "", ""
    text = fix_repetitions(text)
    if language:
        return language, text
    if ASR_TEXT not in text:
        return "", text.strip()
    meta, text = text.split(ASR_TEXT, 1)
    if "language none" in meta.lower():
        return "", text.strip()
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            name = line[len("language "):].strip()
            return (name[:1].upper() + name[1:].lower() if name else ""), text.strip()
    return "", text.strip()


def validate_max_new_tokens(value):
    if type(value) is not int or not 1 <= value <= 32768:
        raise ValueError("max_new_tokens must be an integer from 1 to 32768")
    return value


def transcribe(asr, waveform, sample_rate, language=None, context="", max_new_tokens=4096):
    """(text, language) of [channels, samples] audio; `language` None detects it."""
    validate_max_new_tokens(max_new_tokens)
    language = normalize_asr_language(language, LANGUAGES, backend="Qwen3-ASR")
    device = asr.patcher.load_device
    model = asr.model
    if getattr(model,"aligner",False):
        raise ValueError("Qwen3-ForcedAligner needs known text; select a Qwen3-ASR model for transcription")
    text_output, languages = "", []
    for piece in split_long(prepare_audio(waveform.to(device), sample_rate)):
        features = log_mel(piece.to(device))
        ids = torch.tensor(prompt_ids(asr, context, language, audio_tokens(features.shape[-1])), device=device)
        comfy.model_management.load_models_gpu([asr.patcher])
        with comfy.model_management.cuda_device_context(device):
            embeds = model.decoder.model.embed_tokens(ids, out_dtype=asr.compute_dtype)
            embeds[ids == AUDIO_PAD] = model.audio_tower(features.to(asr.compute_dtype))
            generated = model.decoder.generate(embeds=embeds[None], do_sample=False, max_length=max_new_tokens,
                                               stop_tokens=STOP_TOKENS, execution_dtype=asr.compute_dtype)
        found, text = parse_output(asr.decode([i for i in generated if i not in STOP_TOKENS]), language)
        text_output = join_segment_text(text_output, text, found or language)
        if found and (not languages or languages[-1] != found):
            languages.append(found)
    return text_output, ",".join(languages)
