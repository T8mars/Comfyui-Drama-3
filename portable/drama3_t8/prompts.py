"""Drama task formats from the public report, with real text/audio rows."""
from dataclasses import dataclass
import re
import torch

from .native.prompt import (
    GENERATION_START, MESSAGE_END, REFERENCE_INSTRUCTION, TTS_INSTRUCTION,
    SPEAKER_TAG, WHITE_SPACE, check_text, with_speaker_tag,
)
from .state import validate_codes, validate_frames, validate_reference
from .scene_cast import prepare_scene_cast

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
    # Directions can precede a named role as well as follow its colon. Remove
    # them first, but keep speaker tags until after detecting a named prefix so
    # that '<|speaker:0|> Meet at 9:30.' retains its spoken colon and words.
    text=re.sub(r"\[[^\[\]]*\]", "", text)
    if scene:
        text=re.sub(r"(?m)^[^\n<>:：\[\]]{1,48}[:：]\s*", "", text)
    text=re.sub(r"<\|speaker:\d+\|>|<\|audio_separator\|>|<\|endofturn\|>", "", text)
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


def direction_tags(*values):
    """Canonical flat current-user controls; never invent a system direction.

    Preserve existing flat controls such as ``[gently]`` and wrap only plain
    descriptions. Newlines are ordinary whitespace in a control, not new turns.
    Structural role/task tokens and nested/unbalanced square brackets are not
    valid delivery descriptions.
    """
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Drama 3 direction must be text")
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            continue
        check_text(value, "direction")
        if "<|speaker" in value:
            raise ValueError("Drama 3 direction cannot contain speaker tokens")
        cursor = 0
        while cursor < len(value):
            opening = value.find("[", cursor)
            plain = value[cursor:opening if opening >= 0 else len(value)].strip()
            if "]" in plain:
                raise ValueError("Drama 3 direction has unbalanced brackets")
            if plain:
                result.append("[" + plain + "]")
            if opening < 0:
                break
            closing = value.find("]", opening + 1)
            if closing < 0 or "[" in value[opening + 1:closing]:
                raise ValueError("Drama 3 direction requires flat, balanced brackets")
            body = value[opening + 1:closing].strip()
            if not body:
                raise ValueError("Drama 3 direction controls cannot be empty")
            result.append("[" + body + "]")
            cursor = closing + 1
    return " ".join(result)


def directed_text(text, direction="", note="", *, scene=False):
    """Insert actual direction after a leading role, preserving spoken text.

    Empty controls are a byte-for-byte identity operation. Named role parsing
    is explicit for Scene only; a spoken clock time is not a character name.
    """
    prefix = direction_tags(direction, note)
    if not prefix:
        return text
    check_text(text, "directed text")
    leading = len(text) - len(text.lstrip(WHITE_SPACE))
    tag = SPEAKER_TAG.match(text, leading)
    if tag is not None:
        return text[:tag.end()] + " " + prefix + " " + text[tag.end():].lstrip(WHITE_SPACE)
    if scene:
        from .scene_cast import _named
        named = _named(text)
        if named is not None:
            match, name = named
            return (match.group("indent") + name + ": " + prefix + " " +
                    match.group("directions") + text[match.end():])
    return prefix + " " + text


def directed_scene_lines(lines, references=(), takes=(), direction="", note=""):
    cast, history, mapping = prepare_scene_cast(lines, references, takes)
    return tuple(directed_text(line, direction, note, scene=True) for line in cast), history, mapping


def system_segments(references, instruction=TTS_INSTRUCTION, context=""):
    segments = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if references else instruction]
    if context:
        check_text(context, "task context")
        segments.extend(("\n\n", context))
    for i, reference in enumerate(references, 1):
        validate_reference(reference)
        check_text(reference.transcript, f"reference {i}")
        transcript = with_speaker_tag(reference.transcript)
        body = transcript.lstrip(WHITE_SPACE)
        tag = SPEAKER_TAG.match(body)
        transcript = tag.group(0) + transcript[:len(transcript) - len(body)] + body[tag.end():]
        segments.extend((f"\n\n[Reference {i}]\nText: {transcript}\nSpeech: ", reference.codes))
    if references and instruction != TTS_INSTRUCTION:
        segments.extend(("\n\n", instruction))
    segments.append(MESSAGE_END)
    return segments


def history_segments(takes):
    from .scene_chunks import history_view
    takes = history_view(takes)
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


def tts_prompt(tokenizer, config, text, references=(), takes=(), direction="", *, system_context=""):
    check_text(text, "text")
    text = directed_text(text, direction)
    segments = system_segments(references, context=system_context) + history_segments(takes)
    segments.extend(("<|im_start|>user\n", with_speaker_tag(text), MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)


def scene_prompt(tokenizer, config, lines, scene="", characters="", references=(), takes=(), note="", direction="", *, system_context=""):
    lines=tuple(lines)
    references=tuple(references)
    if not lines:
        raise ValueError("The scene contains no lines")
    for line in lines:
        check_text(line, "scene line")
    for description, location in ((scene, "scene description"), (characters, "characters"), (system_context,"task context")):
        if description.strip():
            check_text(description, location)
    lines,takes,mapping=directed_scene_lines(lines,references,takes,direction,note)
    assignments = '\n'.join(f'{name}: <|speaker:{speaker}|>' for name,speaker in mapping.items()) if mapping and any((scene,characters,system_context)) else ''
    description = "\n".join(p for p in (system_context, f"Scene: {scene}" if scene else "",
                           f"Characters:\n{characters}" if characters else "",
                           'Speaker assignments:\n'+assignments if assignments else '') if p)
    if characters and not references:
        # The report's scene task uses character descriptions before its explicit line-output instruction.
        instruction = description + "\nVoice every line of the script in order, with <|audio_separator|> between consecutive lines."
        segments = ["<|im_start|>system\n", instruction, MESSAGE_END]
    else:
        segments = system_segments(references, LINE_INSTRUCTION, description)
    segments.extend(history_segments(takes))
    script=("\n" + SEPARATOR + "\n").join(lines)
    segments.extend(("<|im_start|>user\n", script, MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)


def infill_prompt(tokenizer, config, target_text, prefix_codes, suffix_codes, duration_frames=None,target_speaker=None):
    check_text(target_text, "edited transcript")
    if target_speaker is not None and (type(target_speaker) is not int or not 0<=target_speaker<16):
        raise ValueError('Infill target_speaker must be 0 to 15')
    if duration_frames is not None and (type(duration_frames) is not int or duration_frames < 1):
        raise ValueError("duration_frames must be a positive integer or None")
    duration = f" [audio tokens: {duration_frames}]" if duration_frames is not None else ""
    instruction=INFILL_INSTRUCTION
    if target_speaker is not None:
        instruction+=f'; synthesize the missing interval using the voice of <|speaker:{target_speaker}|> from the script and surrounding speech'
    segments = ["<|im_start|>system\n", instruction, MESSAGE_END,
                "<|im_start|>user\n", with_speaker_tag(target_text) + duration,
                "<|fim_suffix|>", suffix_codes, "<|fim_prefix|>", prefix_codes,
                MESSAGE_END, GENERATION_START]
    return rows_from_segments(tokenizer, config, segments)


def streaming_prompt(tokenizer, config, text, lookahead="", references=(), takes=(), direction="", *, system_context=""):
    check_text(text, "stream text")
    if lookahead.strip():
        check_text(lookahead, "lookahead")
    text = directed_text(text, direction)
    segments = system_segments(references, STREAM_INSTRUCTION, system_context) + history_segments(takes)
    segments.extend(("<|im_start|>user\n", with_speaker_tag(text), END_OF_TURN, lookahead,
                     MESSAGE_END, GENERATION_START))
    return rows_from_segments(tokenizer, config, segments)
