"""Drama task formats from the public report, with real text/audio rows."""
from dataclasses import dataclass
import re
import torch

from .native.prompt import (
    GENERATION_START, MESSAGE_END, REFERENCE_INSTRUCTION, TTS_INSTRUCTION,
    SPEAKER_TAG, WHITE_SPACE, check_text, with_speaker_tag,
)
from .state import validate_codes, validate_frames

SEPARATOR = "<|audio_separator|>"
END_OF_TURN = "<|endofturn|>"
INFILL_INSTRUCTION = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> "
    "and the prefix follows <|fim_prefix|>"
)
LINE_INSTRUCTION = (
    "convert the provided text to speech; the lines are separated by <|audio_separator|> "
    "— synthesize each line in order and output <|audio_separator|> between adjacent audio segments"
)
STREAM_INSTRUCTION = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"


def spoken_text(text, scene=False):
    """Keep spoken words separate from the model's direction and casting syntax."""
    if scene:
        text=re.sub(r"(?m)^[^\n<>:\[\]]{1,48}:\s*", "", text)
    text=re.sub(r"<\|speaker:\d+\|>|<\|audio_separator\|>|<\|endofturn\|>", "", text)
    text=re.sub(r"\[[^\[\]]*\]", "", text)
    text=re.sub(r"</?[A-Za-z][^>]*>", "", text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


@dataclass(frozen=True)
class FrameRows:
    values: torch.Tensor


def rows_from_segments(tokenizer, config, segments):
    rows = []
    width = 1 + config.tower.num_codebooks
    for item in segments:
        if isinstance(item, str):
            ids = tokenizer.encode(item).ids
            part = torch.zeros((len(ids), width), dtype=torch.long)
            part[:, 0] = torch.tensor(ids)
        elif isinstance(item, FrameRows):
            validate_frames(item.values)
            part = item.values.to(device="cpu", dtype=torch.long)
            if part.ndim != 2 or part.shape[1] != width:
                raise ValueError(f"Expected frame rows [frames,{width}]")
        elif item.ndim == 2 and item.shape[0] == width - 1:
            validate_codes(item)
            codes = item.to(device="cpu", dtype=torch.long)
            if codes.numel() and (codes.min() < 0 or codes.max() >= config.tower.vocab_size):
                raise ValueError("Reference audio codes are outside the codec vocabulary")
            part = torch.cat((codes[:1] + config.semantic_start_token_id, codes)).T
        else:
            raise ValueError(f"Expected audio codes [{width - 1},frames] or frame rows [frames,{width}]")
        rows.append(part)
    result = torch.cat(rows)
    if len(result) > 8192:
        raise ValueError(f"Prompt has {len(result)} rows; the model supports at most 8192 prompt rows")
    return result


def system_segments(references, instruction=TTS_INSTRUCTION, direction=""):
    segments = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if references else instruction]
    for i, reference in enumerate(references, 1):
        check_text(reference.transcript, f"reference {i}")
        transcript = with_speaker_tag(reference.transcript)
        body = transcript.lstrip(WHITE_SPACE)
        tag = SPEAKER_TAG.match(body)
        transcript = tag.group(0) + transcript[:len(transcript) - len(body)] + body[tag.end():]
        segments.extend((f"\n\n[Reference {i}]\nText: {transcript}\nSpeech: ", reference.codes))
    if references and instruction != TTS_INSTRUCTION:
        segments.extend(("\n\n", instruction))
    if direction:
        check_text(direction, "direction")
        segments.extend(("\n\n", direction))
    segments.append(MESSAGE_END)
    return segments


def history_segments(takes):
    result = []
    for take in takes:
        # Scene requests contain separators inserted by this builder; all other
        # structure must still be checked when history came from an imported file.
        historical_text = take.prompt_text.replace(SEPARATOR, "\n") if take.metadata.get("task") == "scene" else take.prompt_text
        check_text(historical_text, "historical text")
        if take.metadata.get("task") == "text_stream":
            if take.metadata["lookahead"].strip():
                check_text(take.metadata["lookahead"], "historical lookahead")
            result.extend(("<|im_start|>user\n", with_speaker_tag(take.prompt_text), END_OF_TURN,
                           take.metadata["lookahead"], MESSAGE_END, GENERATION_START,
                           FrameRows(take.frames), MESSAGE_END))
        else:
            text = take.prompt_text if take.metadata.get("task") == "scene" else with_speaker_tag(take.prompt_text)
            result.extend(("<|im_start|>user\n", text, MESSAGE_END,
                           GENERATION_START, FrameRows(take.frames), MESSAGE_END))
    return result


def tts_prompt(tokenizer, config, text, references=(), takes=(), direction=""):
    check_text(text, "text")
    segments = system_segments(references, direction=direction) + history_segments(takes)
    segments.extend(("<|im_start|>user\n", with_speaker_tag(text), MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)


def scene_prompt(tokenizer, config, lines, scene="", characters="", references=(), takes=(), note=""):
    if not lines:
        raise ValueError("The scene contains no lines")
    for line in lines:
        check_text(line, "scene line")
    for description, location in ((scene, "scene description"), (characters, "characters")):
        if description.strip():
            check_text(description, location)
    description = "\n".join(p for p in (f"Scene: {scene}" if scene else "",
                           f"Characters:\n{characters}" if characters else "") if p)
    if characters and not references:
        # The report's scene task uses character descriptions before its explicit line-output instruction.
        instruction = description + "\nVoice every line of the script in order, with <|audio_separator|> between consecutive lines."
        segments = ["<|im_start|>system\n", instruction, MESSAGE_END]
    else:
        segments = system_segments(references, LINE_INSTRUCTION, description)
    segments.extend(history_segments(takes))
    if note:
        check_text(note,"director note")
    script=("\n" + SEPARATOR + "\n").join(lines)
    segments.extend(("<|im_start|>user\n", (f"[{note}] " if note else "")+script, MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)


def infill_prompt(tokenizer, config, target_text, prefix_codes, suffix_codes, duration_frames=None):
    check_text(target_text, "edited transcript")
    if duration_frames is not None and (type(duration_frames) is not int or duration_frames < 1):
        raise ValueError("duration_frames must be a positive integer or None")
    duration = f" [audio tokens: {duration_frames}]" if duration_frames is not None else ""
    segments = ["<|im_start|>system\n", INFILL_INSTRUCTION, MESSAGE_END,
                "<|im_start|>user\n", with_speaker_tag(target_text) + duration,
                "<|fim_suffix|>", suffix_codes, "<|fim_prefix|>", prefix_codes,
                MESSAGE_END, GENERATION_START]
    return rows_from_segments(tokenizer, config, segments)


def streaming_prompt(tokenizer, config, text, lookahead="", references=(), takes=(), direction=""):
    check_text(text, "stream text")
    if lookahead.strip():
        check_text(lookahead, "lookahead")
    segments = system_segments(references, STREAM_INSTRUCTION, direction) + history_segments(takes)
    segments.extend(("<|im_start|>user\n", with_speaker_tag(text), END_OF_TURN, lookahead,
                     MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)
