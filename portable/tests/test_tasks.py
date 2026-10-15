from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from pathlib import Path

from drama3_t8.prompts import infill_prompt, scene_prompt, tts_prompt, streaming_prompt
from drama3_t8.state import Reference, Session, Take, load_session, save_session

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tokenizer():
    return Tokenizer.from_file(str(ROOT / "drama3_t8" / "native" / "fish" / "tokenizer.json"))


@pytest.fixture
def config():
    return SimpleNamespace(tower=SimpleNamespace(num_codebooks=10, vocab_size=4096),
                           semantic_start_token_id=248099, semantic_end_token_id=252194)


def audio_rows(rows):
    return rows[(rows[:, 0] >= 248099) & (rows[:, 0] <= 252194)]


def test_reference_eleven_frames_is_not_mistaken_for_frame_rows(tokenizer, config):
    codes = torch.arange(110).reshape(10, 11)
    ref = Reference("Hello.", codes)
    rows = tts_prompt(tokenizer, config, "Good morning.", [ref])
    assert torch.equal(audio_rows(rows)[:, 1:].T, codes)


def test_revision_retains_audio_history_and_relative_note(tokenizer, config):
    frames = torch.zeros((10, 11), dtype=torch.long)
    frames[:, 0] = 248099
    take = Take("<|speaker:0|> Hello.", "Hello.", frames)
    rows = tts_prompt(tokenizer, config, "[warmer versus previous] Hello.", takes=[take])
    assert torch.equal(audio_rows(rows), frames)
    assert "warmer versus previous" in tokenizer.decode(rows[:, 0].tolist(), skip_special_tokens=False)


def test_infill_suffix_precedes_prefix_and_preserves_real_codes(tokenizer, config):
    prefix = torch.ones((10, 4), dtype=torch.long)
    suffix = torch.full((10, 3), 2, dtype=torch.long)
    rows = infill_prompt(tokenizer, config, "The demo is Thursday.", prefix, suffix, 14)
    audio = audio_rows(rows)
    assert torch.equal(audio[:3, 1:].T, suffix)
    assert torch.equal(audio[3:, 1:].T, prefix)
    ids = rows[:, 0].tolist()
    suffix_id, prefix_id = tokenizer.token_to_id("<|fim_suffix|>"), tokenizer.token_to_id("<|fim_prefix|>")
    # The instruction also names the markers; inspect their last occurrence in the actual user turn.
    assert max(i for i, x in enumerate(ids) if x == suffix_id) < max(i for i, x in enumerate(ids) if x == prefix_id)


def test_scene_and_stream_have_distinct_controls(tokenizer, config):
    scene = scene_prompt(tokenizer, config, ["<|speaker:0|> Hello.", "<|speaker:15|> Hi."])
    stream = streaming_prompt(tokenizer, config, "Hello.", "Next sentence.")
    assert tokenizer.token_to_id("<|audio_separator|>") in scene[:, 0].tolist()
    assert tokenizer.token_to_id("<|endofturn|>") in stream[:, 0].tolist()
    assert "Next sentence." in tokenizer.decode(stream[:, 0].tolist(), skip_special_tokens=False)


def test_session_roundtrip_and_branch_do_not_drop_controls(tmp_path):
    frames = torch.zeros((3, 11), dtype=torch.long)
    frames[:, 0] = torch.tensor([248099, 252196, 248100])
    session = Session((Reference("Hello", torch.ones((10, 11), dtype=torch.long)),))
    child = session.append(Take("Hello\nHi", "Hello\nHi", frames))
    assert not session.takes
    save_session(child, tmp_path / "中文 会话.safetensors")
    restored = load_session(tmp_path / "中文 会话.safetensors")
    assert torch.equal(restored.takes[0].frames, frames)
    assert torch.equal(restored.references[0].codes, child.references[0].codes)
    assert len(restored.takes[0].segments()) == 2
    assert restored.branch(0).takes == ()


def test_invalid_speaker_rejected(tokenizer, config):
    with pytest.raises(ValueError, match="speaker"):
        tts_prompt(tokenizer, config, "<|speaker:16|> Hello")
