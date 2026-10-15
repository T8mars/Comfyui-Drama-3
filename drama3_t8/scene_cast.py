"""Stable speaker tags for named, reference-conditioned scene scripts.

Reference names are display labels. Unlabelled clips keep the existing speaker-0
default; precise reference identities are expressed with explicit speaker tags.
Historical takes are rendered through immutable prompt views only.
"""
from dataclasses import replace
import re

from .native.prompt import SPEAKER_TAG, check_text


SEPARATOR = "<|audio_separator|>"
_UNICODE_LINE_BREAKS = "\u0085\u2028\u2029"
_NAMED_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<directions>(?:\[[^\[\]\r\n]*\][ \t]*)*)"
    r"(?P<name>[^<>:\r\n\u0085\u2028\u2029\[\]]+):[ \t]*"
)
_EXPLICIT_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<tag><\|speaker:[0-9]+\|>)[ \t]*")


def validate_scene_speaker_map(mapping):
    """Validate optional persisted casting metadata without changing it."""
    if not isinstance(mapping, dict):
        raise ValueError("Scene speaker map must be an object")
    for name, speaker in mapping.items():
        if (not isinstance(name, str) or not name or name != name.strip()
                or len(name) > 48 or any(char in "<>:[]" + _UNICODE_LINE_BREAKS or ord(char) < 32 or ord(char) == 127 for char in name)):
            raise ValueError("Scene speaker map names must be nonempty canonical role names, at most 48 characters")
        if type(speaker) is not int or not 0 <= speaker < 16:
            raise ValueError("Scene speaker map IDs must be integers from 0 to 15")


def _named(line):
    match = _NAMED_LINE.match(line)
    if match is None:
        return None
    name = match.group("name").strip()
    validate_scene_speaker_map({name: 0})
    return match, name


def _history_body(take):
    text = take.prompt_text
    old_note = take.metadata.get("director_note", "")
    prefix = f"[{old_note}] " if old_note else ""
    if prefix and text.startswith(prefix):
        return text[len(prefix):], prefix
    return text, ""


def _scene_lines(take):
    lines = take.metadata.get("script_lines")
    if lines:
        return tuple(lines)
    text, _ = _history_body(take)
    return tuple(line for line in text.replace(SEPARATOR, "\n").splitlines() if line.strip())


def _render_line(line, mapping):
    found = _named(line)
    if found is None:
        return line
    match, name = found
    if name not in mapping:
        raise ValueError(f"Historical scene role {name!r} is absent from its recorded script")
    return f"{match.group('indent')}<|speaker:{mapping[name]}|> {match.group('directions')}{line[match.end():]}"


def _render_history(take, mapping):
    body, old_prefix = _history_body(take)
    parts = body.splitlines(keepends=True)
    rendered = []
    for index, line in enumerate(parts):
        # splitlines(keepends=True) retains the original separator/newline rows.
        content = line.rstrip("\r\n")
        ending = line[len(content):]
        content = _render_line(content, mapping)
        if index == 0 and old_prefix:
            tag = _EXPLICIT_LINE.match(content)
            if tag is not None:
                content = f"{tag.group('indent')}{tag.group('tag')} {old_prefix}{content[tag.end():]}"
            else:
                content = old_prefix + content
        rendered.append(content + ending)
    text = "".join(rendered)
    return take if text == take.prompt_text else replace(take, prompt_text=text)


def prepare_scene_cast(lines, references=(), takes=()):
    """Return mapped current lines, prompt-only history views and a stable map.

    With references, names are allocated in historical-script/current first-use
    order. Persisted scene maps take precedence. Explicit script IDs reserve
    their slots; reference block numbers and display names never allocate IDs.
    Without references all supplied lines and take objects are preserved.
    """
    lines, takes = tuple(lines), tuple(takes)
    if not references:
        return lines, takes, {}
    for line in lines:
        check_text(line, "scene line")
    scenes = tuple(take for take in takes if take.metadata.get("task") == "scene")
    mapping = {}
    for take in scenes:
        if "scene_speaker_map" in take.metadata:
            previous = take.metadata["scene_speaker_map"]
            validate_scene_speaker_map(previous)
            for name, speaker in previous.items():
                if name in mapping and mapping[name] != speaker:
                    raise ValueError(f"Conflicting historical speaker ID for scene role {name!r}")
                mapping[name] = speaker
        check_text(take.prompt_text.replace(SEPARATOR, "\n"), "historical scene text")
    scripts = [line for take in scenes for line in _scene_lines(take)] + list(lines)
    explicit = set()
    for line in scripts:
        check_text(line, "scene script line")
        explicit.update(int(match.group(1)) for match in SPEAKER_TAG.finditer(line))
    occupied = set(mapping.values()) | explicit
    for line in scripts:
        found = _named(line)
        if found is None or found[1] in mapping:
            continue
        free = next((speaker for speaker in range(16) if speaker not in occupied), None)
        if free is None:
            raise ValueError("Scene has more automatic roles than the 16 available speaker IDs; use explicit shared IDs for aliases")
        mapping[found[1]] = free
        occupied.add(free)
    rendered_lines = tuple(_render_line(line, mapping) for line in lines)
    history = tuple(_render_history(take, mapping) if take.metadata.get("task") == "scene" else take for take in takes)
    return rendered_lines, history, dict(mapping)
