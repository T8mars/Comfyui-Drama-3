"""Shared hotword parsing without inference or ComfyUI dependencies."""

from __future__ import annotations

import re
from pathlib import Path

MAX_CONTEXT_CHARS = 8192
MAX_HOTWORD_BYTES = 64 * 1024
_SEPARATORS = re.compile(r"[\r\n,，;；、]+")


def normalize_hotwords(value: str) -> tuple[str, int]:
    """Preserve phrases and order while removing empty and duplicate entries."""
    if not isinstance(value, str):
        raise ValueError("Hotwords must be text")
    if len(value.encode("utf-8")) > MAX_HOTWORD_BYTES:
        raise ValueError("Hotword input exceeds 64 KiB; use a smaller relevant word list")
    entries = []
    for part in _SEPARATORS.split(value):
        part = part.strip()
        while part.startswith("\ufeff"):
            part = part[1:].strip()
        entries.append(part)
    words = dict.fromkeys(part for part in entries if part)
    text = ", ".join(words)
    if len(text) + (len("Hotwords: ") if text else 0) > MAX_CONTEXT_CHARS:
        raise ValueError("CONTEXT_LIMIT: hotwords exceed the 8192-character prompt limit")
    return text, len(words)


def hotword_context(context: str, hotwords: str) -> str:
    if not isinstance(context, str):
        raise ValueError("Context must be text")
    words, _ = normalize_hotwords(hotwords)
    result = context
    if words:
        result += ("\n" if result else "") + "Hotwords: " + words
    if len(result) > MAX_CONTEXT_CHARS:
        raise ValueError("CONTEXT_LIMIT: context and hotwords together exceed 8192 characters")
    return result


def read_hotword_file(input_dir: Path, filename: str) -> bytes:
    """Read only a bounded UTF-8 TXT file within ComfyUI's input directory."""
    if not isinstance(filename, str):
        raise ValueError("Hotword file must be a relative TXT path")
    if not filename.strip():
        return b""
    root = input_dir.resolve()
    relative = Path(filename.strip())
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("Use a TXT path relative to ComfyUI/input")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix.lower() != ".txt" or not path.is_file():
        raise ValueError("Hotword file must be a TXT file within ComfyUI/input")
    with path.open("rb") as handle:
        data = handle.read(MAX_HOTWORD_BYTES + 1)
    if len(data) > MAX_HOTWORD_BYTES:
        raise ValueError("Hotword file exceeds 64 KiB")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the hotword TXT file as UTF-8") from exc
    return data
