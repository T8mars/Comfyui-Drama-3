"""Replace opening delivery cues without rewriting speech or point events."""
import re

# Delivery controls are distinct from point events (laughs, sighs, pauses)
# and identity controls (accents/languages), which must remain in the script.
DELIVERY = frozenset('warm warmly gentle gently calm calmly happy happily sad sadly angry angrily excited excitedly cheerful cheerfully serious seriously neutral softly loudly whispering shouting fast faster quickly slow slower slowly'.split())
DELIVERY |= frozenset(('温柔', '亲切', '平静', '开心', '悲伤', '生气', '愤怒', '激动', '轻声', '大声', '快速', '慢速'))

def replace_opening_delivery(text, *, scene=False):
    from .native.prompt import SPEAKER_TAG, WHITE_SPACE
    if scene:
        from .scene_cast import _named
        named = _named(text)
        if named is not None:
            match, _ = named
            prefix, first = replace_opening_delivery(text[:match.end()])
            suffix, second = replace_opening_delivery(text[match.end():])
            return prefix + suffix, first + second
    leading = len(text) - len(text.lstrip(WHITE_SPACE))
    tag = SPEAKER_TAG.match(text, leading)
    offset = tag.end() if tag else leading
    body = text[offset:]
    cursor = 0
    kept, removed = [], []
    while match := re.match(r'\s*\[([^\[\]]+)\]\s*', body[cursor:]):
        token = match.group(1)
        (removed if token.strip().casefold() in DELIVERY else kept).append(token)
        cursor += match.end()
    if not removed:
        return text, []
    suffix = ' '.join('[' + value + ']' for value in kept)
    if suffix:
        suffix += ' '
    return text[:offset] + (' ' if tag else '') + suffix + body[cursor:], removed
