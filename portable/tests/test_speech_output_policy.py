"""Ordinary TTS candidate policy and task-routing CPU regressions."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
EOS, START, END, SEP, TURN = 248046, 248099, 252194, 252196, 252195
BASE = (EOS, *range(START, END + 1), SEP, TURN)


@pytest.fixture(scope='module')
def module():
    return importlib.import_module('drama3_t8.engine')


@pytest.fixture(scope='module')
def tokenizer():
    return Tokenizer.from_file(str(ROOT / 'drama3_t8/native/fish/tokenizer.json'))


class RealCPUHead(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(width * 32, dtype=torch.float32).reshape(width, 32) / 32768,
                                         requires_grad=False)
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return F.linear(hidden, self.weight)


@pytest.fixture
def rig(module, tokenizer, monkeypatch):
    head = RealCPUHead(len(BASE))
    config = SimpleNamespace(output_token_ids=BASE, eos_token_id=EOS, semantic_start_token_id=START,
        semantic_end_token_id=END, tower=SimpleNamespace(num_codebooks=10, vocab_size=4096), text=SimpleNamespace(hidden_size=32))
    drama = SimpleNamespace(config=config, tokenizer=tokenizer,
        model=SimpleNamespace(model=SimpleNamespace(layers=[])),
        experts=SimpleNamespace(last_backend='torch', backend_calls={'torch': 0, 'cuda': 0}),
        parts=lambda: (None, [torch.device('cpu')], None, head, None))
    engine = module.DramaEngine('fixture-model.safetensors', 'fixture-codec.safetensors')
    engine.drama = drama
    engine.load_model = lambda: drama
    engine._load_drama = lambda extra_patchers=(): None
    # These are actual CPU projection weights; the existing compact routing is
    # exercised without loading a checkpoint or asking model_management to load.
    engine.line_head = torch.nn.Sequential(torch.nn.Linear(32, 1, bias=False))
    engine.line_head_patcher = object()
    calls = []
    state = SimpleNamespace(failure=None, stall_once=False)

    def native(actual_drama, prompt, max_frames, temperature, top_k, top_p, seed, **kwargs):
        hidden = torch.linspace(-1, 1, 64, dtype=torch.float32).reshape(2, 32)
        override = kwargs['head_override']
        ids = tuple(override[1]) if override is not None else tuple(config.output_token_ids)
        projection = override[0] if override is not None else head
        before = head.calls
        scores = projection(hidden)
        calls.append(dict(ids=ids, head=projection, override=override, logits=scores.clone(), hidden=hidden,
                          base_head_calls=head.calls-before, seed=seed, temperature=temperature, top_k=top_k, top_p=top_p,
                          max_frames=max_frames, stop=kwargs['stop_token_ids'], expected_segments=kwargs['expected_segments']))
        if state.failure is not None:
            raise state.failure
        stall = state.stall_once and len(calls) == 1
        rows = torch.zeros((2, 11), dtype=torch.long)
        rows[:, 0] = START
        if kwargs['expected_segments'] == 2:
            rows = torch.cat((rows[:1], torch.tensor([[SEP, *([0] * 10)]], dtype=torch.long), rows[1:]))
        for row in rows:
            if START <= row[0] <= END:
                kwargs['on_audio'](row[1:])
        kwargs['on_finish']('stalled' if stall else 'eos')
        return rows

    monkeypatch.setattr(module, 'generate', native)
    with torch.inference_mode():
        yield SimpleNamespace(module=module, engine=engine, drama=drama, head=head, calls=calls, state=state)
    assert not torch.cuda.is_initialized()


def test_speak_only_exposes_eos_and_every_semantic_token(rig):
    result = rig.engine.speak('A short exact line.', sampling=rig.module.Sampling(seed=521, max_frames=300))
    call = rig.calls[-1]
    assert len(call['ids']) == 4097
    assert call['ids'] == (EOS, *range(START, END + 1))
    assert SEP not in call['ids'] and TURN not in call['ids']
    assert result.takes[-1].metadata['seed'] == 521
    assert rig.drama.config.output_token_ids == BASE


def test_projection_gathers_one_real_head_output_bitexact_without_new_weights(rig):
    pointer = rig.head.weight.untyped_storage().data_ptr()
    rig.engine.speak('exact')
    call = rig.calls[-1]
    expected = F.linear(call['hidden'], rig.head.weight)[:, :4097]
    assert call['base_head_calls'] == 1
    assert torch.equal(call['logits'].view(torch.int32), expected.contiguous().view(torch.int32))
    assert rig.head.weight.untyped_storage().data_ptr() == pointer
    assert call['head'].index is None and call['head'].head is None


@pytest.mark.parametrize('reordered', [False, True])
def test_legacy_4097_is_not_wrapped_or_reordered(rig, reordered):
    ids = tuple(reversed(BASE[:-2])) if reordered else BASE[:-2]
    rig.drama.config.output_token_ids = ids
    rig.head.weight = torch.nn.Parameter(rig.head.weight[:4097].clone(), requires_grad=False)
    rig.engine.speak('legacy exact')
    call = rig.calls[-1]
    assert call['override'] is None and call['ids'] == ids and call['base_head_calls'] == 1


def test_reordered_base_vocabulary_uses_actual_column_positions(rig):
    ids = (TURN, *reversed(BASE[:-2]), SEP)
    rig.drama.config.output_token_ids = ids
    rig.engine.speak('reordered')
    call = rig.calls[-1]
    positions = [index for index, token in enumerate(ids) if token == EOS or START <= token <= END]
    expected = F.linear(call['hidden'], rig.head.weight).index_select(-1, torch.tensor(positions))
    assert call['ids'] == tuple(ids[index] for index in positions)
    assert torch.equal(call['logits'].view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize('kind', ['missing_eos', 'missing_semantic', 'duplicate', 'unknown_control', 'bool_token'])
def test_invalid_head_metadata_is_rejected_before_native_sampling(rig, kind):
    bad = list(BASE)
    if kind == 'missing_eos': bad.remove(EOS)
    if kind == 'missing_semantic': bad.remove(START)
    if kind == 'duplicate': bad[-1] = EOS
    if kind == 'unknown_control': bad[-1] = 198
    if kind == 'bool_token': bad[-1] = True
    rig.drama.config.output_token_ids = tuple(bad)
    with pytest.raises(ValueError):
        rig.engine.speak('invalid')
    assert rig.calls == [] and rig.head.calls == 0


@pytest.mark.parametrize('mode', ['compact', 'full'])
def test_speak_then_stream_then_scene_preserves_each_task_contract(rig, mode, monkeypatch):
    rig.engine.speak('first')
    rig.engine.stream_text('second', lookahead='third')
    assert rig.calls[-1]['override'] is None and rig.calls[-1]['ids'] == BASE
    assert rig.calls[-1]['stop'] == (TURN,)
    rig.engine.scene_head_mode = mode
    if mode == 'full':
        compact = rig.engine.load_scene_head
        rig.engine.scene_head_mode = 'compact'
        full_head = compact()
        rig.engine.scene_head_mode = 'full'
        monkeypatch.setattr(rig.engine, 'load_text_head', lambda: full_head)
    rig.engine.scene(['<|speaker:0|> One.', '<|speaker:1|> Two.'])
    scene = rig.calls[-1]
    assert scene['expected_segments'] == 2 and SEP in scene['ids'] and 198 in scene['ids']
    assert TURN in scene['ids']  # Existing scene behavior remains unchanged.
    assert rig.drama.config.output_token_ids == BASE
    rig.engine.speak('again')
    assert rig.calls[-1]['ids'] == BASE[:-2]


@pytest.mark.parametrize('preset', ['stable', 'expressive'])
def test_sampling_seed_budget_and_top_k_zero_are_untouched(rig, preset):
    sampling = rig.module.Sampling(seed=2**64-1, max_frames=300).with_preset(preset)
    rig.engine.speak('sampling', sampling=sampling)
    call = rig.calls[-1]
    assert (call['seed'], call['max_frames'], call['temperature'], call['top_k'], call['top_p']) == (
        sampling.seed, sampling.max_frames, sampling.temperature, sampling.top_k, sampling.top_p)


@pytest.mark.parametrize('failure', [RuntimeError('native-error'), KeyboardInterrupt('cancel')])
def test_private_selector_resources_clear_even_when_exception_traceback_is_retained(rig, failure):
    rig.state.failure = failure
    with pytest.raises(type(failure)) as caught:
        rig.engine.speak('cancelled')
    assert caught.value is failure and caught.value.__traceback__ is not None
    selector = rig.calls[-1]['head']
    assert selector.index is None and selector.head is None
    assert rig.drama.config.output_token_ids == BASE
    rig.state.failure = None
    rig.engine.speak('recovery')
    assert rig.calls[-1]['ids'] == BASE[:-2]


def test_requested_opening_retry_reuses_live_selector_then_closes(rig):
    rig.state.stall_once = True
    result = rig.engine.speak('retry', sampling=rig.module.Sampling(seed=2**64-1, opening_retries=1))
    assert [call['seed'] for call in rig.calls] == [2**64-1, 0]
    assert all(call['ids'] == BASE[:-2] for call in rig.calls)
    assert rig.calls[0]['head'] is rig.calls[1]['head']
    assert rig.calls[-1]['head'].index is None
    assert len(result.takes[-1].metadata['generation_attempts']) == 2


def test_unknown_policy_and_scene_policy_override_fail_closed(rig):
    prompt = torch.zeros(1, 11, dtype=torch.long)
    for options in ({'output_policy': 'unknown'}, {'output_policy': 'speech', 'full_head': True}):
        with pytest.raises(ValueError):
            rig.engine._take(prompt, 'exact', 'exact', rig.module.Sampling(), **options)
    assert rig.calls == []


def test_real_fim_prompt_internal_take_remains_legacy(rig):
    codes = torch.zeros(10, 3, dtype=torch.long)
    prompt = rig.module.infill_prompt(rig.drama.tokenizer, rig.drama.config, 'Changed words.', codes, codes)
    rig.engine._take(prompt, 'Changed words.', 'Changed words.', rig.module.Sampling(seed=516), allow_empty=True)
    assert rig.calls[-1]['ids'] == BASE and rig.calls[-1]['override'] is None
