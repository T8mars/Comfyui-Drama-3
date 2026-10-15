"""Join independently decoded speech chunks without merging adjacent words."""

_SPACE_SEPARATED_LANGUAGES = frozenset({
    "English", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
    "Arabic", "Indonesian", "Vietnamese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish",
    "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
})
_CLOSING_PUNCTUATION = frozenset(".,!?;:%)]}，。！？；：、）》」』")
_OPENING_OR_JOINING = frozenset("([{“‘「『-/'")


def join_segment_text(prefix: str, suffix: str, language: str | None) -> str:
    if not prefix or not suffix:
        return prefix + suffix
    spaced = language in _SPACE_SEPARATED_LANGUAGES or (
        not language and suffix[0].isascii() and suffix[0].isalnum())
    if (not spaced or prefix[-1].isspace() or suffix[0].isspace()
            or prefix[-1] in _OPENING_OR_JOINING or suffix[0] in _CLOSING_PUNCTUATION):
        return prefix + suffix
    return prefix + " " + suffix
