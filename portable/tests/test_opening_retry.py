"""Opening retries must follow delivered audio, including real stall trimming."""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.engine import DramaEngine, Sampling, SpeechGenerationError


def engine_and_generator(monkeypatch, *, always_stall=False):
    import drama3_t8.engine as module
    engine = DramaEngine('model', 'codec')
    config = SimpleNamespace(semantic_start_token_id=248099, semantic_end_token_id=252194)
    drama = SimpleNamespace(config=config, model=SimpleNamespace(model=SimpleNamespace(layers=[])),
                            tokenizer=SimpleNamespace(token_to_id=lambda text: 252196))
    engine.load_model = lambda: drama
    seeds = []

    def generate(model, prompt, max_frames, temperature, top_k, top_p, seed, **callbacks):
        seeds.append(seed)
        stall = always_stall or len(seeds) == 1
        # The native generator calls on_audio before checking its 96-frame
        # low-diversity guard, then returns the shortened 12-frame prefix.
        callback_frames = torch.zeros(98 if stall else 60, 11, dtype=torch.long)
        callback_frames[:, 0] = config.semantic_start_token_id
        callback_frames[:, 1] = torch.arange(len(callback_frames)) % 24 if stall else torch.arange(60)
        for frame in callback_frames:
            callbacks['on_audio'](frame[1:])
        callbacks['on_finish']('stalled' if stall else 'eos')
        return callback_frames[:12] if stall else callback_frames

    monkeypatch.setattr(module, 'generate', generate)
    return engine, seeds


def test_nonstreaming_stall_can_use_requested_retry_after_many_callbacks(monkeypatch):
    engine, seeds = engine_and_generator(monkeypatch)
    take = engine._take(torch.zeros(1, 11, dtype=torch.long), 'hello', 'hello',
                        Sampling(seed=2**64 - 1, opening_retries=1))
    assert seeds == [2**64 - 1, 0]
    assert take.metadata['requested_seed'] == 2**64 - 1
    assert take.metadata['seed'] == 0
    assert take.metadata['finish_reason'] == 'eos'
    attempts = take.metadata['generation_attempts']
    assert len(attempts) == 2 and attempts[0]['retry'] and not attempts[1]['retry']
    assert attempts[0]['audio_frames'] == 12
    assert len(attempts[0]['discarded_frames']) == 12
    assert attempts[0]['discarded_frames'][0][0] == 248099
    assert len(take.frames) == 60


def test_stall_after_actual_stream_delivery_does_not_restart_or_duplicate(monkeypatch):
    engine, seeds = engine_and_generator(monkeypatch)
    delivered = []
    with pytest.raises(SpeechGenerationError) as failure:
        engine._take(torch.zeros(1, 11, dtype=torch.long), 'hello', 'hello',
                     Sampling(seed=516, opening_retries=1), on_codes=lambda frame: delivered.append(int(frame[0])))
    assert seeds == [516]
    assert delivered == [i % 24 for i in range(98)]
    assert len(failure.value.generation_attempts) == 1
    assert not failure.value.generation_attempts[0]['retry']


@pytest.mark.parametrize('retries,expected_seeds', [(0, [516]), (1, [516, 517])])
def test_nonstreaming_retries_keep_requested_limit(monkeypatch, retries, expected_seeds):
    engine, seeds = engine_and_generator(monkeypatch, always_stall=True)
    with pytest.raises(SpeechGenerationError) as failure:
        engine._take(torch.zeros(1, 11, dtype=torch.long), 'hello', 'hello',
                     Sampling(seed=516, opening_retries=retries))
    assert seeds == expected_seeds
    attempts = failure.value.generation_attempts
    assert len(attempts) == len(expected_seeds)
    assert not attempts[-1]['retry']
