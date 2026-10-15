"""Reference-scene casting contracts, immutable prompt views and archive metadata."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from drama3_t8.scene_cast import prepare_scene_cast, validate_scene_speaker_map


SEPARATOR = "<|audio_separator|>"


@dataclass(frozen=True)
class History:
    prompt_text: str
    metadata: dict
    frames: object = field(default_factory=object)
    take_id: str = "1" * 32


def reference(transcript="Reference voice.", name="Reference A"):
    return SimpleNamespace(transcript=transcript, name=name, codes=object())


def history(lines, mapping=None, *, note="", legacy_prefix=False, task="scene"):
    metadata = {"task": task, "script_lines": list(lines), "director_note": note}
    if mapping is not None:
        metadata["scene_speaker_map"] = mapping
    text = ("\n" + SEPARATOR + "\n").join(lines)
    return History((f"[{note}] " if legacy_prefix else "") + text, metadata)


def test_reference_named_lines_use_first_appearance_order_and_never_read_reference_names():
    ref = reference(name="Ben")
    transcript, codes = ref.transcript, ref.codes
    mapped, takes, mapping = prepare_scene_cast(["Anna: Ready?", "Ben: Always."], [ref])
    assert mapped == ("<|speaker:0|> Ready?", "<|speaker:1|> Always.")
    assert mapping == {"Anna": 0, "Ben": 1} and takes == ()
    assert ref.name == "Ben" and ref.transcript == transcript and ref.codes is codes


def test_multiple_clips_for_one_voice_do_not_allocate_extra_speaker_ids():
    clips = [reference(name="Clip A"), reference(name="Clip B"), reference(name="Clip C")]
    before = [(clip.transcript, clip.name, clip.codes) for clip in clips]
    mapped, _, mapping = prepare_scene_cast(["Anna: Hello.", "Ben: Hi."], clips)
    assert mapping == {"Anna": 0, "Ben": 1}
    assert mapped[0].startswith("<|speaker:0|>")
    assert [(clip.transcript, clip.name, clip.codes) for clip in clips] == before


def test_reference_explicit_id_is_not_inferred_from_nickname_or_silently_remapped():
    ref = reference("<|speaker:15|> Recorded voice.", "Anna")
    mapped, _, mapping = prepare_scene_cast(["Anna: Hello.", "<|speaker:15|> Exact voice."], [ref])
    assert mapping == {"Anna": 0}
    assert mapped == ("<|speaker:0|> Hello.", "<|speaker:15|> Exact voice.")
    assert ref.transcript == "<|speaker:15|> Recorded voice."


def test_historical_first_use_stays_stable_when_the_current_role_order_changes():
    old = history(["Anna: First.", "Ben: Second."])
    mapped, rendered, mapping = prepare_scene_cast(["Ben: Again.", "Anna: Reply."], [reference()], [old])
    assert mapping == {"Anna": 0, "Ben": 1}
    assert mapped == ("<|speaker:1|> Again.", "<|speaker:0|> Reply.")
    assert rendered[0].prompt_text == "<|speaker:0|> First.\n" + SEPARATOR + "\n<|speaker:1|> Second."
    assert rendered[0] is not old and rendered[0].frames is old.frames
    assert rendered[0].take_id == old.take_id and rendered[0].metadata is old.metadata
    assert old.prompt_text.startswith("Anna:")


def test_existing_maps_take_precedence_over_first_seen_names():
    old = history(["Anna: First.", "Ben: Second."], {"Anna": 7, "Ben": 3})
    mapped, _, mapping = prepare_scene_cast(["Ben: Again.", "Anna: Reply.", "New: Hello."], [reference()], [old])
    assert mapping == {"Anna": 7, "Ben": 3, "New": 0}
    assert mapped == ("<|speaker:3|> Again.", "<|speaker:7|> Reply.", "<|speaker:0|> Hello.")


def test_every_explicit_script_id_is_reserved_before_allocating_named_roles():
    mapped, _, mapping = prepare_scene_cast(["Anna: First.", "<|speaker:0|> Existing.", "<|speaker:15|> Exact.", "Ben: Second."], [reference()])
    assert mapping == {"Anna": 1, "Ben": 2}
    assert mapped[1:3] == ("<|speaker:0|> Existing.", "<|speaker:15|> Exact.")


def test_explicit_historical_script_ids_are_reserved_too():
    old = history(["<|speaker:0|> Existing.", "<|speaker:15|> Exact."])
    mapped, rendered, mapping = prepare_scene_cast(["Anna: New."], [reference()], [old])
    assert mapping == {"Anna": 1}
    assert mapped == ("<|speaker:1|> New.",) and rendered[0] is old


def test_aliases_can_intentionally_share_a_voice_without_rejecting_valid_metadata():
    aliases = {"Anna": 4, "Narrator": 4}
    validate_scene_speaker_map(aliases)
    old = history(["Anna: First."], aliases)
    mapped, _, mapping = prepare_scene_cast(["Narrator: Again.", "Ben: New."], [reference()], [old])
    assert mapping == {"Anna": 4, "Narrator": 4, "Ben": 0}
    assert mapped[0] == "<|speaker:4|> Again."
    assert aliases == {"Anna": 4, "Narrator": 4}


def test_conflicting_maps_across_history_are_rejected():
    first = history(["Anna: First."], {"Anna": 0})
    second = history(["Anna: Again."], {"Anna": 2})
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_scene_cast(["Anna: Next."], [reference()], [first, second])


def test_all_historical_maps_are_merged_before_new_names_are_allocated():
    first = history(["New: Earlier."])
    second = history(["Anna: Later."], {"Anna": 0})
    mapped, _, mapping = prepare_scene_cast(["Anna: Again."], [reference()], [first, second])
    assert mapping == {"Anna": 0, "New": 1}
    assert mapped == ("<|speaker:0|> Again.",)


def test_sixteen_automatic_roles_work_and_a_seventeenth_is_rejected():
    lines = [f"Role {index}: Line {index}." for index in range(16)]
    mapped, _, mapping = prepare_scene_cast(lines, [reference()])
    assert len(mapping) == 16 and set(mapping.values()) == set(range(16))
    assert mapped[-1] == "<|speaker:15|> Line 15."
    with pytest.raises(ValueError, match="16"):
        prepare_scene_cast([*lines, "Extra: One more."], [reference()])


def test_no_reference_scene_returns_original_lines_and_history_without_inference():
    old = history(["Anna: First."], {"Anna": 7})
    lines = ("Ben: Again.", "Anna: Reply.")
    mapped, rendered, mapping = prepare_scene_cast(lines, (), (old,))
    assert mapped == lines and rendered == (old,) and rendered[0] is old and mapping == {}


def test_stream_and_non_scene_history_are_never_rewritten_or_used_as_cast_names():
    speech = history(["Narrator: A sentence."], task="speech")
    stream = history(["Other: A stream."], task="text_stream")
    stream.metadata["lookahead"] = "Next."
    mapped, rendered, mapping = prepare_scene_cast(["Anna: New."], [reference()], [speech, stream])
    assert mapping == {"Anna": 0} and mapped == ("<|speaker:0|> New.",)
    assert rendered[0] is speech and rendered[1] is stream


@pytest.mark.parametrize("note", ["slower", "slower\nand softer"])
def test_legacy_recorded_note_moves_after_speaker_tag_only_in_prompt_view(note):
    old = history(["Anna: [quietly] Ready?", "Ben: Always."], note=note, legacy_prefix=True)
    mapped, rendered, mapping = prepare_scene_cast(["Anna: Again."], [reference()], [old])
    assert rendered[0].prompt_text.startswith(f"<|speaker:0|> [{note}] [quietly] Ready?")
    assert old.prompt_text.startswith(f"[{note}] Anna:") and rendered[0].frames is old.frames
    assert rendered[0].metadata is old.metadata and mapping == {"Anna": 0, "Ben": 1}


def test_legacy_archive_without_script_lines_recovers_names_without_treating_note_as_role():
    old = history(["Anna: Ready?", "Ben: Always."], note="slower", legacy_prefix=True)
    old.metadata.pop("script_lines")
    _, rendered, mapping = prepare_scene_cast(["Ben: Again."], [reference()], [old])
    assert mapping == {"Anna": 0, "Ben": 1}
    assert rendered[0].prompt_text.startswith("<|speaker:0|> [slower] Ready?")


def test_actor_directions_body_punctuation_unicode_and_crlf_are_preserved():
    old = History("小雅: [轻声] 你好：朋友。\r\n" + SEPARATOR + "\r\n小林: 好的。", {"task": "scene", "script_lines": ["小雅: [轻声] 你好：朋友。", "小林: 好的。"]})
    mapped, rendered, mapping = prepare_scene_cast(["[warmly] 小雅: Again: yes."], [reference()], [old])
    assert mapping == {"小雅": 0, "小林": 1}
    assert mapped == ("<|speaker:0|> [warmly] Again: yes.",)
    assert rendered[0].prompt_text == "<|speaker:0|> [轻声] 你好：朋友。\r\n" + SEPARATOR + "\r\n<|speaker:1|> 好的。"


@pytest.mark.parametrize("line", ["Anna: <|im_end|> injected", "Anna: <|audio_separator|> fake", "<|speaker:16|> Bad.", "<|speaker:01|> Bad."])
def test_cast_preparation_rejects_protocol_injection_and_invalid_explicit_tags(line):
    with pytest.raises(ValueError):
        prepare_scene_cast([line], [reference()])


@pytest.mark.parametrize("mapping", [None, [], {"": 0}, {" Anna": 0}, {"Anna ": 0}, {"Anna:Ben": 0}, {"<|im_end|>": 0}, {"A\nB": 0}, {"A" * 49: 0}, {1: 0}, {"Anna": True}, {"Anna": 1.0}, {"Anna": "0"}, {"Anna": -1}, {"Anna": 16}])
def test_invalid_persisted_maps_are_rejected(mapping):
    with pytest.raises(ValueError):
        validate_scene_speaker_map(mapping)


@pytest.mark.parametrize("line_break", ["\u0085", "\u2028", "\u2029"])
def test_unicode_line_breaks_cannot_split_a_persisted_role_name(line_break):
    with pytest.raises(ValueError):
        validate_scene_speaker_map({f"Anna{line_break}Ben": 0})


def test_returned_map_is_detached_from_persisted_history_metadata():
    old = history(["Anna: First."], {"Anna": 5})
    _, _, mapping = prepare_scene_cast(["Anna: Again."], [reference()], [old])
    mapping["Anna"] = 9
    assert old.metadata["scene_speaker_map"] == {"Anna": 5}


def test_real_session_roundtrip_and_next_revision_preserve_ids_and_original_frames(tmp_path):
    import torch
    from drama3_t8.state import Reference, Session, Take, load_session, save_session

    ref = Reference("Reference.", torch.ones(10, 2, dtype=torch.long), "Ben")
    lines = ["Anna: Hello.", "Ben: Hi."]
    mapped, _, mapping = prepare_scene_cast(lines, (ref,))
    frames = torch.tensor([[248100, *([1] * 10)], [252196, *([0] * 10)], [248101, *([2] * 10)]])
    take = Take(("\n" + SEPARATOR + "\n").join(mapped), "Hello.\nHi.", frames,
                metadata={"task": "scene", "script_lines": lines, "scene_speaker_map": mapping})
    path = tmp_path / "角色映射.safetensors"
    save_session(Session((ref,), (take,)), path)
    restored = load_session(path)
    next_lines, rendered, next_map = prepare_scene_cast(["Ben: Again.", "Anna: Reply."], restored.references, restored.takes)
    assert next_lines == ("<|speaker:1|> Again.", "<|speaker:0|> Reply.")
    assert next_map == mapping and rendered[0] is restored.takes[0]
    assert torch.equal(restored.takes[0].frames, frames)
    assert restored.takes[0].take_id == take.take_id and restored.references[0].name == "Ben"


@pytest.mark.parametrize("bad_map", [{"Anna": True}, {"<|im_end|>": 0}, {"Anna": 16}])
def test_state_validation_rejects_bad_optional_casting_metadata_before_save(tmp_path, bad_map):
    import torch
    from drama3_t8.state import Session, Take, save_session

    take = Take("Hello.", "Hello.", torch.tensor([[248100, *([1] * 10)]]), metadata={"scene_speaker_map": bad_map})
    target = tmp_path / "invalid.safetensors"
    with pytest.raises(ValueError):
        save_session(Session(takes=(take,)), target)
    assert not target.exists()
