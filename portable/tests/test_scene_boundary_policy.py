"""Real CPU-only line-boundary policy contracts; no tensor/model stubs."""
import pytest

from drama3_t8.native.scene_boundary import SceneBoundaryPolicy


NEWLINE = 198
EOS = 248046
SEPARATOR = 252196
SEMANTIC_START = 248099
SEMANTIC_END = 252194
VOICE_A = SEMANTIC_START + 1
VOICE_B = SEMANTIC_START + 2
VOICE_C = SEMANTIC_START + 3


def policy(segments=2, events=None):
    return SceneBoundaryPolicy(segments, newline_id=NEWLINE, eos_id=EOS,
        separator_id=SEPARATOR, semantic_start=SEMANTIC_START, semantic_end=SEMANTIC_END,
        on_correction=events.append if events is not None else None)


def execute(boundaries, tokens, start_step=0):
    return [boundaries.select(token, start_step + index) for index, token in enumerate(tokens)]


def test_natural_scene_boundary_tokens_are_never_changed():
    events = []
    tokens = [VOICE_A, VOICE_A, NEWLINE, SEPARATOR, NEWLINE, VOICE_B, EOS]
    assert execute(policy(events=events), tokens) == tokens
    assert events == []


def test_premature_eos_becomes_only_newline_and_separator_stays_natural():
    events = []
    tokens = [VOICE_A, EOS, SEPARATOR, NEWLINE, VOICE_B, EOS]
    actual = execute(policy(events=events), tokens)
    assert actual == [VOICE_A, NEWLINE, SEPARATOR, NEWLINE, VOICE_B, EOS]
    assert tokens == [VOICE_A, EOS, SEPARATOR, NEWLINE, VOICE_B, EOS]
    assert len(events) == 1
    assert events[0]["step"] == 1
    assert events[0]["raw_token_id"] == EOS
    assert events[0]["emitted_token_id"] == NEWLINE
    assert events[0]["line"] == 1
    assert SEPARATOR not in [event["emitted_token_id"] for event in events]


def test_three_segments_can_each_recover_one_boundary_before_natural_separator():
    events = []
    tokens = [VOICE_A, EOS, SEPARATOR, NEWLINE, VOICE_B, EOS, SEPARATOR, NEWLINE, VOICE_C, EOS]
    expected = [VOICE_A, NEWLINE, SEPARATOR, NEWLINE, VOICE_B, NEWLINE, SEPARATOR, NEWLINE, VOICE_C, EOS]
    assert execute(policy(3, events), tokens, start_step=20) == expected
    assert [event["step"] for event in events] == [21, 25]
    assert all(event["raw_token_id"] == EOS and event["emitted_token_id"] == NEWLINE for event in events)
    assert [event["line"] for event in events] == [1, 2]
    assert events[0] is not events[1]


def test_natural_separator_resets_only_the_next_segments_recovery_budget():
    events = []
    tokens = [VOICE_A, NEWLINE, SEPARATOR, NEWLINE, VOICE_B, EOS, SEPARATOR, VOICE_C, EOS]
    expected = [VOICE_A, NEWLINE, SEPARATOR, NEWLINE, VOICE_B, NEWLINE, SEPARATOR, VOICE_C, EOS]
    assert execute(policy(3, events), tokens) == expected
    assert len(events) == 1 and events[0]["step"] == 5


@pytest.mark.parametrize("prefix", [[], [NEWLINE], [NEWLINE, NEWLINE]])
def test_empty_audio_segment_cannot_end_with_eos(prefix):
    boundaries = policy()
    execute(boundaries, prefix)
    with pytest.raises(RuntimeError):
        boundaries.select(EOS, len(prefix))


@pytest.mark.parametrize("prefix", [[], [NEWLINE], [VOICE_A, SEPARATOR], [VOICE_A, SEPARATOR, NEWLINE]])
def test_empty_audio_segment_cannot_emit_separator(prefix):
    boundaries = policy(3)
    execute(boundaries, prefix)
    with pytest.raises(RuntimeError):
        boundaries.select(SEPARATOR, len(prefix))


@pytest.mark.parametrize("tail", [[], [NEWLINE], [VOICE_A], [NEWLINE, VOICE_A]])
def test_second_eos_before_natural_separator_is_rejected_instead_of_looping(tail):
    events = []
    boundaries = policy(events=events)
    assert execute(boundaries, [VOICE_A, EOS]) == [VOICE_A, NEWLINE]
    execute(boundaries, tail, start_step=2)
    with pytest.raises(RuntimeError):
        boundaries.select(EOS, 2 + len(tail))
    assert len(events) == 1


def test_extra_separator_after_final_nonempty_segment_is_rejected():
    boundaries = policy()
    execute(boundaries, [VOICE_A, SEPARATOR, NEWLINE, VOICE_B])
    with pytest.raises(RuntimeError):
        boundaries.select(SEPARATOR, 4)


def test_eos_in_an_empty_last_segment_is_rejected():
    boundaries = policy()
    execute(boundaries, [VOICE_A, SEPARATOR, NEWLINE])
    with pytest.raises(RuntimeError):
        boundaries.select(EOS, 3)


def test_single_segment_eos_is_unchanged_and_never_records_a_recovery():
    events = []
    tokens = [VOICE_A, VOICE_A, NEWLINE, EOS]
    assert execute(policy(1, events), tokens) == tokens
    assert events == []


def test_single_segment_never_accepts_a_separator_after_real_audio():
    boundaries = policy(1)
    boundaries.select(VOICE_A, 0)
    with pytest.raises(RuntimeError):
        boundaries.select(SEPARATOR, 1)


@pytest.mark.parametrize("semantic_token", [SEMANTIC_START, SEMANTIC_END])
def test_both_ends_of_the_semantic_band_count_as_real_audio(semantic_token):
    boundaries = policy()
    assert execute(boundaries, [semantic_token, SEPARATOR, semantic_token, EOS]) == [semantic_token, SEPARATOR, semantic_token, EOS]


def test_non_audio_text_token_does_not_make_an_empty_segment_valid():
    boundaries = policy()
    boundaries.select(13, 0)
    with pytest.raises(RuntimeError):
        boundaries.select(EOS, 1)


@pytest.mark.parametrize("segments", [0, -1, True, False, 1.5, "2", None])
def test_segment_count_requires_a_positive_exact_integer(segments):
    with pytest.raises((TypeError, ValueError)):
        policy(segments)
