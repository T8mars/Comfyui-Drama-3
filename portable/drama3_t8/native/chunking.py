"""Long text for Drama 3, as fish-speech's `fish_omni/chunking.py` (modeled on Fish Audio's hosted server).

Text is cut into chunks of about CHUNK_BYTES UTF-8 bytes with at most CHUNK_MAX_SPEAKERS speakers, preferring
sentence ends, then clause marks, then spaces; tags such as `<|speaker:1|>`, `[laughs]` or `<whisper>...</whisper>`
are never split. Each chunk is generated on its own; the first chunk and the latest one go back into the next
prompt as finished turns, which keeps the voice consistent across chunks.
"""
import re

from .prompt import SPEAKER_TAG, WHITE_SPACE

CHUNK_BYTES = 300
CHUNK_MAX_SPEAKERS = 3
SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
PROTECTED_SPAN = re.compile(r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S)


def utf8_len(text):
    return len(text.encode("utf-8"))


def protected_pieces(text):
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last:match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def breaks_at(text, index, marks):
    # keep decimals such as 3.14 in one piece
    return text[index] in marks and not (text[index] == "." and 0 < index < len(text) - 1
                                         and text[index - 1].isdigit() and text[index + 1].isdigit())


def break_at_marks(pieces, length, marks):
    """Split the free pieces longer than `length` bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if breaks_at(text, index, marks):
                out.append((text[start:index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def break_by_length(pieces, length):
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current = ""
        for char in text:
            if current and utf8_len(current + char) > length:
                out.append((current, False))
                current = ""
            current += char
        out.append((current, False))
    return out


def split_by_length(text, length=CHUNK_BYTES):
    if not text.strip(WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = break_at_marks(protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = break_at_marks(pieces, length, marks)
    chunks, current = [], ""
    for piece, _ in break_by_length(pieces, length):
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [chunk.strip(WHITE_SPACE) for chunk in chunks if chunk.strip(WHITE_SPACE)]


def split_text(text, max_bytes=CHUNK_BYTES, max_speakers=CHUNK_MAX_SPEAKERS):
    """Generation chunks in order. Tagged text is split per speaker segment and regrouped: a chunk ends when the
    next segment would bring in one speaker too many or exceed `max_bytes`, so A, B, A can share a chunk."""
    if SPEAKER_TAG.search(text) is None:
        return split_by_length(text, max_bytes)
    parts = SPEAKER_TAG.split(text)
    segments = [("0", parts[0])] if parts[0].strip(WHITE_SPACE) else []
    segments += [(speaker, body) for speaker, body in zip(parts[1::2], parts[2::2]) if body.strip(WHITE_SPACE)]
    tagged = [(speaker, f"<|speaker:{speaker}|> {piece}") for speaker, body in segments for piece in split_by_length(body, max_bytes)]
    chunks, batch, speakers, size = [], [], [], 0
    for speaker, segment in tagged:
        new_speaker = speaker not in speakers
        if (new_speaker and len(speakers) >= max_speakers) or (batch and size + utf8_len(segment) > max_bytes):
            chunks.append("\n".join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append("\n".join(batch))
    return chunks
