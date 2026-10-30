"""Canonical local comparisons, including Windows extended-length paths."""
import os
from pathlib import Path


def canonical_path(value):
    text = str(Path(value).expanduser().resolve())
    if os.name == "nt":
        if text.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            text = text[4:]
    return Path(text)
