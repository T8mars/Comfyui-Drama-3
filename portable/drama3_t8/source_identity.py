"""Semantic identity shared by Studio and the independent native node package."""
import hashlib
from pathlib import Path

_STUDIO_ONLY = {"studio.py", "studio_editor.py", "bootstrap.py", "updater.py"}
_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".json", ".jinja", ".jinja2", ".dict"}


def semantic_source_identity(directory):
    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in _SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in _STUDIO_ONLY:
            continue
        text = path.read_text(encoding="utf-8")
        name = relative.encode("utf-8")
        body = text.encode("utf-8")
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(len(body).to_bytes(8, "little"))
        digest.update(body)
    return digest.hexdigest()
