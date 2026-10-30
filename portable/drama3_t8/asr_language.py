"""ASR request aliases only; voice catalog labels remain caller-owned text."""
from __future__ import annotations

_ALIASES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian",
    "pt": "Portuguese", "es": "Spanish", "it": "Italian",
    "中文": "Chinese", "汉语": "Chinese", "普通话": "Chinese",
    "英语": "English", "英文": "English", "粤语": "Cantonese", "广东话": "Cantonese",
    "日语": "Japanese", "韩语": "Korean", "德语": "German", "法语": "French",
    "俄语": "Russian", "葡萄牙语": "Portuguese", "西班牙语": "Spanish", "意大利语": "Italian",
}


def normalize_asr_language(language, supported_languages, *, backend="ASR"):
    """Return the backend's exact canonical name, or None for automatic mode.

    Every non-auto request is validated against the caller's real backend list.
    This function imports no model/runtime and never rewrites a catalog record.
    """
    if language is None:
        return None
    if not isinstance(language, str):
        raise ValueError(f"{backend} 语言必须是文本、auto 或 None")
    value = language.strip()
    if not value or value.casefold() == "auto":
        return None
    if isinstance(supported_languages, str):
        raise TypeError("supported_languages must be the backend language sequence")
    canonical = {}
    for item in supported_languages:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Invalid backend language list")
        key = item.strip().casefold()
        if key in canonical and canonical[key] != item:
            raise ValueError("Ambiguous backend language list")
        canonical[key] = item
    requested = _ALIASES.get(value.casefold(), value)
    if requested.casefold() not in canonical:
        raise ValueError(f"{backend} 不支持指定语言：{language}")
    return canonical[requested.casefold()]
