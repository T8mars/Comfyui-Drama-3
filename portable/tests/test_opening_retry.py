"""Opening retries must follow delivered audio, including real stall trimming."""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.engine import DramaEngine, Sampling, SpeechGenerationError


def engine_and_generator(monkeypatch, *, always_stall=False, retained_frames=12, committed_opening=False):
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
        callback_frames = torch.zeros(max(98,retained_frames) if stall else 60, 11, dtype=torch.long)
        callback_frames[:, 0] = config.semantic_start_token_id
        callback_frames[:, 1] = torch.arange(len(callback_frames)) % 24 if stall else torch.arange(60)
        if stall and committed_opening:
            callback_frames[:54,1] = torch.arange(54)
        for frame in callback_frames:
            callbacks['on_audio'](frame[1:])
        callbacks['on_finish']('stalled' if stall else 'eos')
        return callback_frames[:retained_frames] if stall else callback_frames

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
    engine, seeds = engine_and_generator(monkeypatch,committed_opening=True)
    delivered = []
    with pytest.raises(SpeechGenerationError) as failure:
        engine._take(torch.zeros(1, 11, dtype=torch.long), 'hello', 'hello',
                     Sampling(seed=516, opening_retries=1), on_codes=lambda frame: delivered.append(int(frame[0])))
    assert seeds == [516]
    assert delivered == [*range(54),*[i % 24 for i in range(54,98)]]
    assert len(failure.value.generation_attempts) == 1
    assert not failure.value.generation_attempts[0]['retry']


def test_exact_stall_diversity_is_buffered_and_failed_opening_is_never_played(monkeypatch):
    engine,seeds=engine_and_generator(monkeypatch)
    delivered=[]
    take=engine._take(torch.zeros(1,11,dtype=torch.long),'hello','hello',
        Sampling(seed=25,opening_retries=1),on_codes=lambda frame:delivered.append(int(frame[0])))
    assert seeds==[25,26]
    assert delivered==list(range(60))
    assert not take.metadata['generation_attempts'][0]['stream_committed']


@pytest.mark.parametrize('retained_frames',[54,80,150])
def test_late_stall_without_stream_delivery_retries_same_prompt_and_preserves_requested_seed(monkeypatch,retained_frames):
    engine,seeds=engine_and_generator(monkeypatch,retained_frames=retained_frames)
    take=engine._take(torch.zeros(1,11,dtype=torch.long),'hello','hello',Sampling(seed=25,opening_retries=1))
    assert seeds==[25,26] and take.metadata['seed']==26
    assert take.metadata['requested_seed']==25
    assert len(take.metadata['generation_attempts'][0]['discarded_frames'])==retained_frames
    assert len(take.frames)==60


@pytest.mark.parametrize('error',[KeyboardInterrupt(),FloatingPointError('non-finite logits'),MemoryError('no RAM')])
def test_cancellation_and_resource_or_numeric_errors_never_trigger_seed_retry(monkeypatch,error):
    import drama3_t8.engine as module
    engine,seeds=engine_and_generator(monkeypatch)
    def fail(*args,**kwargs):
        seeds.append(args[6])
        raise error
    monkeypatch.setattr(module,'generate',fail)
    with pytest.raises(type(error)):
        engine._take(torch.zeros(1,11,dtype=torch.long),'hello','hello',Sampling(seed=25,opening_retries=1))
    assert seeds==[25]


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
