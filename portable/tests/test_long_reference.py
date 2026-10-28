"""Real CPU Long continuation contracts using native tokenizer/prompt/state code.

Only model loading/generation is replaced with distinct CPU tensor results.
Run with the repository's official CPU harness, which selects CPU and isolates
host GPU-only discovery before test collection. These are structural prompt and
archive contracts, not waveform/voice-quality tests.
"""
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8 import engine as canonical_engine
from drama3_t8 import prompts
from drama3_t8.native.config import parse_config
from drama3_t8.native.chunking import split_text
from drama3_t8.native.prompt import GENERATION_START, MESSAGE_END, with_speaker_tag
from drama3_t8.state import Reference, Session, Take, load_session, save_session

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    assert not torch.cuda.is_initialized()
    def forbidden(*unused, **kwargs):
        raise AssertionError('Real CPU prompt contracts must not initialize CUDA')
    # pytest restores this per-test guard, including when an assertion fails.
    # No CPUState/CUDA availability flags or module-global GPU probes are patched.
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    assert canonical_engine.mm.get_torch_device().type == 'cpu'
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module')
def selected_module():
    return canonical_engine


@pytest.fixture(scope='module')
def actual_prompt_assets():
    package = ROOT / 'drama3_t8/native/fish'
    tokenizer_path = package / 'tokenizer.json'
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == '2da260680bf5558e8167d407d7cc92fbb243deecc4e8f0d0ccd2a09f319205ca'
    assert importlib.metadata.version('tokenizers') == '0.22.2'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = parse_config(json.loads((package / 'config.json').read_text('utf-8')))
    assert config.tower.num_codebooks == 10 and config.tower.vocab_size == 4096
    return tokenizer, config


def frames(config, number, count):
    codes = (torch.arange(10 * count, dtype=torch.int64, device='cpu').reshape(10, count) + number * 101) % 4096
    return torch.cat((codes[:1] + config.semantic_start_token_id, codes)).T.contiguous()


def harness(module, assets):
    tokenizer, config = assets
    engine = module.DramaEngine.__new__(module.DramaEngine)
    # Only model loading/generation is fake. Real generation_session, long_form,
    # speak, tts_prompt, rows_from_segments and all state objects execute.
    drama = SimpleNamespace(tokenizer=tokenizer, config=config,
                            session=lambda **kwargs: contextlib.nullcontext())
    calls = []
    def load_model(self):
        return drama
    def take(self, prompt, prompt_text, transcript, sampling, on_codes=None, progress=None, **options):
        assert prompt.device.type == 'cpu' and prompt.dtype == torch.int64
        assert options['output_policy'] == 'speech'
        index = len(calls)
        result = Take(prompt_text, transcript, frames(config, index + 1, 53 + index),
                      config.semantic_start_token_id, config.semantic_end_token_id,
                      tokenizer.token_to_id('<|audio_separator|>'),
                      metadata={'sampling': sampling.__dict__.copy(), 'finish_reason': 'eos',
                                'structural_generation_stub_not_audio': True})
        calls.append(dict(prompt=prompt.clone(), text=prompt_text, take=result,
                          sampling=sampling, on_codes=on_codes, progress=progress))
        return result
    engine.load_model = MethodType(load_model, engine)
    engine._take = MethodType(take, engine)
    return engine, calls


def semantic_rows(rows, config):
    return rows[(rows[:, 0] >= config.semantic_start_token_id) & (rows[:, 0] <= config.semantic_end_token_id)]


def assert_single_current_user(rows, text, tokenizer, config):
    # Three role starts: system, current user, open assistant. Completed history
    # would add two more; the old text is valid inside Reference Text instead.
    assert int((rows[:, 0] == tokenizer.token_to_id('<|im_start|>')).sum()) == 3
    tail = prompts.rows_from_segments(tokenizer, config,
        ('<|im_start|>user\n', with_speaker_tag(text), MESSAGE_END, GENERATION_START))
    assert torch.equal(rows[-len(tail):], tail)


def test_real_second_prompt_has_exact_first53_reference_and_current_text(selected_module, actual_prompt_assets):
    tokenizer, config = actual_prompt_assets
    engine, calls = harness(selected_module, actual_prompt_assets)
    text = '你好，这是长文分段测试。\n第二段会沿用前一段的语音历史。'
    result = engine.long_form(text, sampling=selected_module.Sampling(seed=42, max_frames=180), chunk_bytes=48)
    chunks = split_text(text, max_bytes=48)
    assert len(calls) == 2 and [c['text'] for c in calls] == chunks
    assert torch.equal(calls[0]['prompt'], prompts.tts_prompt(tokenizer, config, chunks[0]))
    anchor = result.takes[0]
    assert anchor is calls[0]['take'] and len(anchor.frames) == 53
    expected = prompts.tts_prompt(tokenizer, config, chunks[1],
                                 references=(Reference(anchor.prompt_text, anchor.codes),), takes=())
    assert torch.equal(calls[1]['prompt'], expected)
    assert torch.equal(semantic_rows(calls[1]['prompt'], config), anchor.frames)
    assert_single_current_user(calls[1]['prompt'], chunks[1], tokenizer, config)
    assert result.references == () and result.direction == '' and len(result.takes) == 2
    assert result.takes[1].frames is calls[1]['take'].frames
    assert result.takes[1].take_id == calls[1]['take'].take_id
    assert result.takes[1].metadata['long_context']['source_take_id'] == anchor.take_id
    assert 'long_context' not in calls[0]['take'].metadata and 'long_context' not in calls[1]['take'].metadata


def test_real_three_chunk_reference_stays_first_with_cues_and_seed_wrap(selected_module, actual_prompt_assets):
    tokenizer, config = actual_prompt_assets
    engine, calls = harness(selected_module, actual_prompt_assets)
    text = '[laughs] First short sentence. Second short sentence. Third short sentence.'
    chunks = split_text(text, max_bytes=32)
    assert len(chunks) >= 3
    result = engine.long_form(text, sampling=selected_module.Sampling(seed=2**64 - 1, max_frames=180), chunk_bytes=32)
    assert [c['sampling'].seed for c in calls] == [(2**64 - 1 + i) % 2**64 for i in range(len(chunks))]
    anchor = result.takes[0]
    assert anchor.prompt_text == chunks[0] and '[laughs]' in anchor.prompt_text
    for index in range(1, len(chunks)):
        expected = prompts.tts_prompt(tokenizer, config, chunks[index], references=(Reference(anchor.prompt_text, anchor.codes),), takes=())
        assert torch.equal(calls[index]['prompt'], expected)
        assert torch.equal(semantic_rows(calls[index]['prompt'], config), anchor.frames)
        assert_single_current_user(calls[index]['prompt'], chunks[index], tokenizer, config)
        assert result.takes[index].metadata['long_context']['source_take_id'] == anchor.take_id
        assert torch.equal(result.takes[index].frames, calls[index]['take'].frames)


def test_real_explicit_refs_order_direction_priority_not_generated_history(selected_module, actual_prompt_assets):
    tokenizer, config = actual_prompt_assets
    refs = (Reference('<|speaker:0|> First reference.', frames(config, 7, 2)[:, 1:].T.contiguous()),
            Reference('<|speaker:1|> Second reference.', frames(config, 8, 3)[:, 1:].T.contiguous()))
    old = tuple(Take('Prior ' + str(i), 'Prior ' + str(i), frames(config, 10 + i, 2), metadata={'old': i}) for i in range(3))
    direction = '[calm] Keep every role; <whisper>retain this direction</whisper>.'
    parent = Session(refs, old, direction)
    parent_bytes = [t.frames.numpy().tobytes() for t in old]
    engine, calls = harness(selected_module, actual_prompt_assets)
    text = 'First short sentence. Second short sentence. Third short sentence.'
    chunks = split_text(text, max_bytes=32)
    result = engine.long_form(text, parent, selected_module.Sampling(seed=42, max_frames=180), chunk_bytes=32)
    assert torch.equal(calls[0]['prompt'], prompts.tts_prompt(tokenizer, config, chunks[0], refs, (old[0], old[-1]), direction))
    ref_rows = torch.cat([torch.cat((ref.codes[:1] + config.semantic_start_token_id, ref.codes)).T for ref in refs])
    for i in range(1, len(chunks)):
        expected = prompts.tts_prompt(tokenizer, config, chunks[i], refs, (), direction)
        assert torch.equal(calls[i]['prompt'], expected)
        assert torch.equal(semantic_rows(calls[i]['prompt'], config), ref_rows)
        assert_single_current_user(calls[i]['prompt'], chunks[i], tokenizer, config)
        policy = result.takes[len(old) + i].metadata['long_context']
        assert policy['policy'] == 'explicit_references' and policy['source_take_id'] is None
    assert result.references is parent.references and result.direction == parent.direction
    for i, original in enumerate(old):
        assert result.takes[i] is original and original.frames.numpy().tobytes() == parent_bytes[i]
        assert original.metadata == {'old': i}


def test_real_branch_archive_roundtrip_keeps_old_tensor_metadata_and_refs(selected_module, actual_prompt_assets, tmp_path):
    _, config = actual_prompt_assets
    ref = Reference('<|speaker:0|> Existing voice.', frames(config, 20, 3)[:, 1:].T.contiguous(), 'voice')
    old = tuple(Take('Old ' + str(i), 'Old ' + str(i), frames(config, 21 + i, 2), metadata={'old': i}) for i in range(3))
    parent = Session((ref,), old, '[calm] Preserve original direction.')
    for count in (0, 1, 3):
        branch = parent.branch(count)
        engine, calls = harness(selected_module, actual_prompt_assets)
        result = engine.long_form('First short sentence. Second short sentence.', branch,
                                 selected_module.Sampling(seed=42, max_frames=180), chunk_bytes=32)
        assert len(result.takes) == count + len(calls)
        assert_single_current_user(calls[1]['prompt'], calls[1]['text'], *actual_prompt_assets)
        assert result.references is branch.references and result.direction == branch.direction
        for i in range(count):
            assert result.takes[i] is old[i] and old[i].metadata == {'old': i}
        path = tmp_path / ('branch-' + str(count) + '.safetensors')
        save_session(result, path)
        loaded = load_session(path)
        assert loaded.session_id == result.session_id and loaded.direction == result.direction
        assert len(loaded.takes) == len(result.takes) and len(loaded.references) == 1
        assert loaded.references[0].transcript == ref.transcript and loaded.references[0].name == ref.name
        assert torch.equal(loaded.references[0].codes, ref.codes)
        for actual, saved in zip(result.takes, loaded.takes):
            assert saved.take_id == actual.take_id and saved.prompt_text == actual.prompt_text
            assert saved.transcript == actual.transcript and saved.metadata == actual.metadata
            assert saved.frames.dtype == actual.frames.dtype and saved.frames.numpy().tobytes() == actual.frames.numpy().tobytes()
    assert parent.takes == old and parent.references[0] is ref


def test_real_single_chunk_keeps_original_conversation_prompt(selected_module, actual_prompt_assets):
    tokenizer, config = actual_prompt_assets
    old = Take('Prior speech.', 'Prior speech.', frames(config, 31, 4), metadata={'untouched': True})
    parent = Session(takes=(old,), direction='Keep this direction.')
    engine, calls = harness(selected_module, actual_prompt_assets)
    result = engine.long_form('Only one short line.', parent,
                             selected_module.Sampling(seed=42, max_frames=180), chunk_bytes=48)
    assert len(calls) == 1
    assert torch.equal(calls[0]['prompt'], prompts.tts_prompt(tokenizer, config, 'Only one short line.', (), (old,), parent.direction))
    assert result.takes[0] is old and old.metadata == {'untouched': True}
    assert 'long_context' not in result.takes[-1].metadata
