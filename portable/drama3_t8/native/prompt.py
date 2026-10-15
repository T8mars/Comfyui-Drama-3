"""Drama 3 prompt rows in the reference-bracket-v1 layout (fish-speech `build_drama3_prompt`).

A system turn carries the voice references as `[Reference N]` blocks of transcript and codes; earlier takes follow
as finished user and assistant turns; the user turn holds the text; the assistant turn opens with `<|voice|>`.
Every literal is tokenized on its own, because tokenizing the joined string changes ids at the seams.
"""
import re

import torch

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:",
                     "<|fim_prefix|>", "<|fim_suffix|>", "<|audio_separator|>", "<|endofturn|>")
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER = 15
# The Unicode White_Space property: str.strip() without arguments also strips U+001C..U+001F.
WHITE_SPACE = "".join(map(chr, (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                                0x2028, 0x2029, 0x202F, 0x205F, 0x3000)))


def check_text(text, location):
    if not text.strip(WHITE_SPACE):
        raise ValueError(f"Drama 3 {location}: the text is empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"Drama 3 {location}: {token} is written by the prompt builder, not the text")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER:
            raise ValueError(f"Drama 3 {location}: speaker tags are <|speaker:0|> to <|speaker:{MAX_SPEAKER}|>")


def with_speaker_tag(text):
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG.match(text.lstrip(WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text


def build_prompt(tokenizer, config, text, references=(), history=()):
    """CPU int64 rows [rows, 1 + N]: text rows are [token, 0, ...], audio rows [semantic token of c0, c0, ..., cN-1].

    `references` and `history` are (transcript, codes [N, frames]) pairs."""
    num_codebooks = config.tower.num_codebooks
    check_text(text, "text")
    segments = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION]
    for index, (transcript, codes) in enumerate(references, 1):
        check_text(transcript, f"reference {index}")
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(WHITE_SPACE)
        tag = SPEAKER_TAG.match(body)
        # the tag must follow "Text: " directly; leading white space moves behind it
        transcript = tag.group(0) + tagged[:len(tagged) - len(body)] + body[tag.end():]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        check_text(turn_text, f"take {index}")
        segments += ["<|im_start|>user\n", with_speaker_tag(turn_text), MESSAGE_END, GENERATION_START, codes, MESSAGE_END]
    segments += ["<|im_start|>user\n", with_speaker_tag(text), MESSAGE_END, GENERATION_START]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long)
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            if codes.ndim != 2 or codes.shape[0] != num_codebooks:
                raise ValueError(f"Drama 3 codes must be [{num_codebooks}, frames]; got {tuple(codes.shape)}")
            rows.append(torch.cat((codes[:1] + config.semantic_start_token_id, codes)).T)
    return torch.cat(rows)
