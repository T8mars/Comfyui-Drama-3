"""Official FIM message framing with the real tokenizer and captured audio IDs."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

from drama3_t8.prompts import infill_prompt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/fim-system-protocol'
OFFICIAL_SYSTEM = ('fill in the missing part of the speech; the suffix follows <|fim_suffix|> '
                   'and the prefix follows <|fim_prefix|>')
TARGET = ('<|speaker:0|> [curious] 今天又整了什么邪修？\n'
          '<|speaker:1|> [deadpan] 我把会议改成了午睡。\n'
          '<|speaker:15|> [excited] 这福利建议全国推广。')


@pytest.fixture(scope='module')
def tokenizer():
    return Tokenizer.from_file(str(ROOT/'drama3_t8/native/fish/tokenizer.json'))


@pytest.fixture
def config():
    return SimpleNamespace(tower=SimpleNamespace(num_codebooks=10, vocab_size=4096),
                           semantic_start_token_id=248099, semantic_end_token_id=252194)


def text_rows(tokenizer, text):
    ids = tokenizer.encode(text).ids
    result = torch.zeros((len(ids), 11), dtype=torch.long)
    result[:, 0] = torch.tensor(ids)
    return result


def codec_rows(codes):
    # Independent protocol representation: column0 is the semantic text ID;
    # all ten original codebooks follow it. Do not reuse the product builder.
    return torch.cat((codes[:1] + 248099, codes), dim=0).T


def expected_message(tokenizer, target, prefix, suffix, duration, actor=None):
    duration_text = '' if duration is None else f' [audio tokens: {duration}]'
    instruction = OFFICIAL_SYSTEM
    if actor is not None:
        instruction += (f'; synthesize the missing interval using the voice of <|speaker:{actor}|> '
                        'from the script and surrounding speech')
    return torch.cat((
        text_rows(tokenizer, '<|im_start|>system\n' + instruction + '<|im_end|>\n'),
        text_rows(tokenizer, '<|im_start|>user\n' + target + duration_text + '<|fim_suffix|>'),
        codec_rows(suffix), text_rows(tokenizer, '<|fim_prefix|>'), codec_rows(prefix),
        text_rows(tokenizer, '<|im_end|>\n<|im_start|>assistant\n<|voice|>')))


@pytest.mark.parametrize('actor', [None, 0, 1, 15])
@pytest.mark.parametrize('duration', [None, 57])
def test_infill_system_protocol_preserves_full_cast_and_every_context_code(tokenizer, config, actor, duration):
    prefix = torch.arange(40).reshape(10, 4)
    suffix = (torch.arange(30) + 101).reshape(10, 3)
    before_prefix, before_suffix = prefix.clone(), suffix.clone()
    actual = infill_prompt(tokenizer, config, TARGET, prefix, suffix, duration, target_speaker=actor)
    assert torch.equal(actual, expected_message(tokenizer, TARGET, prefix, suffix, duration, actor))
    assert torch.equal(prefix, before_prefix) and torch.equal(suffix, before_suffix)
    spoken_rows = actual[(actual[:, 0] >= 248099) & (actual[:, 0] <= 252194)]
    assert torch.equal(spoken_rows[:, 1:].T, torch.cat((suffix, prefix), dim=1))
    # An explicit builder hint still affects the actual input. The render path
    # omits it only when the selected audible context already identifies actor.
    decoded = tokenizer.decode(actual[:, 0].tolist(), skip_special_tokens=False)
    assert '<|im_start|>user\n' + TARGET in decoded
    assert decoded.endswith('<|im_end|>\n<|im_start|>assistant\n<|voice|>')


@pytest.mark.parametrize('prefix_frames,suffix_frames', [(0, 0), (0, 3), (4, 0), (4, 3)])
def test_infill_system_protocol_keeps_empty_and_nonempty_FIM_context_order(tokenizer, config, prefix_frames, suffix_frames):
    prefix = torch.arange(10*prefix_frames).reshape(10, prefix_frames)
    suffix = (torch.arange(10*suffix_frames) + 51).reshape(10, suffix_frames)
    target = '<|speaker:15|> [softly] Continue here.'
    actual = infill_prompt(tokenizer, config, target, prefix, suffix)
    assert torch.equal(actual, expected_message(tokenizer, target, prefix, suffix, None))


@pytest.mark.parametrize('actor', [True, False, -1, 16, '1', 1.0])
def test_infill_system_protocol_still_rejects_invalid_actor(tokenizer, config, actor):
    codes = torch.ones((10, 2), dtype=torch.long)
    with pytest.raises(ValueError, match='target_speaker'):
        infill_prompt(tokenizer, config, TARGET, codes, codes, target_speaker=actor)


def test_infill_system_protocol_replays_the_real_stalled_prompt_with_only_system_extension_removed(tokenizer, config):
    details = json.loads((FIXTURE/'fixture.json').read_text(encoding='utf-8'))
    path = FIXTURE/'original-speaker1-prompt.json'
    # Universal-newline text keeps this public fixture pin stable under Git autocrlf.
    text = path.read_text(encoding='utf-8')
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == 'b16459d4c391cfd84df13bfe02a00be0c0fb49538353017115ab204bd6da20c3'
    captured = json.loads(text)
    assert captured['tensor_name'] == 'prompt' and captured['dtype'] == 'I64'
    assert captured['shape'] == [204, 11]
    assert captured['raw_encoding'] == 'little-endian signed int64 row-major'
    assert captured['original_safetensors_sha256'] == details['sha256']
    assert captured['raw_tensor_sha256'] == '505da7c61980ec7b53f2004ef86d023eb44e4f4b8239926be5a04b138856346a'
    rows = captured['rows']
    assert isinstance(rows, list) and len(rows) == 204
    assert all(isinstance(row, list) and len(row) == 11 for row in rows)
    assert all(type(value) is int and -(2**63) <= value < 2**63 for row in rows for value in row)
    original = torch.tensor(rows, dtype=torch.int64)
    assert tuple(original.shape) == (204, 11) and original.dtype == torch.int64
    assert hashlib.sha256(original.numpy().astype('<i8', copy=False).tobytes()).hexdigest() == '505da7c61980ec7b53f2004ef86d023eb44e4f4b8239926be5a04b138856346a'
    a, b = details['prefix_audio_rows']
    c, d = details['suffix_audio_rows']
    prefix, suffix = original[a:b, 1:].T.contiguous(), original[c:d, 1:].T.contiguous()
    assert prefix.shape == (10, 54) and suffix.shape == (10, 57)
    from drama3_t8.render_infill import _selected_roles
    assert _selected_roles(details['selection_intervals'], *details['selection_samples']) == {1}
    actual = infill_prompt(tokenizer, config, details['target_text'], prefix, suffix)
    start, end = details['removed_system_rows']
    expected = torch.cat((original[:start], original[end:]))
    assert original.shape == (204, 11) and actual.shape == (186, 11)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, expected_message(tokenizer, details['target_text'], prefix, suffix, None))


@pytest.fixture(scope='module')
def render_probes():
    # Reuse the existing deterministic Codec/render contract fixture unchanged;
    # only replace its character tokenizer with the actual DRAMA tokenizer.
    path = ROOT/'tests/test_source_aware_infill.py'
    spec = importlib.util.spec_from_file_location('_existing_infill_render_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('start,end,explicit,actor,hint', [
    (9000,13000,None,1,None),
    (9000,13000,1,1,None),
    (9000,13000,0,0,0),
    (4000,9000,0,0,0),
    (4000,9000,1,1,1),
    (5500,6500,1,1,1),
    (9000,9000,1,1,None),
    (5000,5000,1,1,1),
])
def test_infill_system_protocol_render_only_omits_redundant_hint(tokenizer, config, render_probes,
                                                               start, end, explicit, actor, hint):
    engine = render_probes.Engine()
    engine.fake.tokenizer = tokenizer
    kwargs = {} if explicit is None else dict(target_speaker=explicit)
    _,_,result = render_probes.operation(engine=engine,start=start,end=end,**kwargs)
    request = engine.requests[0]
    assert request['text'] == '<|speaker:0|> First\n<|speaker:1|> The complete new second line'
    assert torch.equal(request['prompt'], expected_message(tokenizer, request['text'],
        result.blobs['prefix_codes'], result.blobs['suffix_codes'], None, hint))
    assert result.metadata['target_speaker'] == actor
    assert result.metadata['render_infill']['target_speaker'] == actor
    assert result.sources[1].metadata['target_speaker'] == actor
    assert result.sources[1].metadata['task'] == 'infill_raw'
    assert any(item['speaker']==actor and item['start']==start/44100
               and item['end']==(start+4096)/44100
               for item in result.metadata['render_infill']['timeline'])


@pytest.mark.parametrize('start,end,expected', [
    (0,10,{0}), (9,10,{0}), (10,10,{1}), (10,11,{1}),
    (9,11,{0,1}), (20,20,set()), (20,25,set()), (30,31,set()),
])
def test_infill_system_protocol_uses_same_audible_half_open_roles_for_inference_and_hint(start, end, expected):
    from drama3_t8.render_infill import _selected_roles, _role
    intervals = [dict(start=0,end=10,speaker=0,audible=True),
                 dict(start=10,end=20,speaker=1,audible=True),
                 dict(start=20,end=30,speaker=2,audible=False)]
    assert _selected_roles(intervals,start,end) == expected
    if len(expected)==1:
        assert _role(intervals,start,end,None) == next(iter(expected))
    else:
        with pytest.raises(ValueError,match='explicit target speaker'):
            _role(intervals,start,end,None)
    assert _role(intervals,start,end,15)==15  # Explicit voice override remains available.
