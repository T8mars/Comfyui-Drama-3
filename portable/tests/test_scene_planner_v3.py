"""V3 tail quality with exact V1/V2 archive compatibility; CPU, no models."""
import base64
import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import pytest
import torch

from test_scene_chunks import Harness, MODES, references
from drama3_t8.scene_chunks import JOIN, body_of, tagged, plan_scene, playback_plan, validate_scene
from drama3_t8.state import load_session, save_session

SCRIPT = ['主持人: [curious] 今天又整了什么邪修？',
          '猫主管: [deadpan] 我把会议改成了午睡。',
          '主持人: [excited] 这福利建议全国推广。']


@pytest.mark.parametrize('mode', MODES)
def test_real_v2_archive_exact_plan_playback_and_roundtrip(mode, tmp_path):
    fixture = json.loads((Path(__file__).parent / 'fixtures/scene-chunks-v2.json').read_text(encoding='utf-8'))
    record = fixture['archives'][mode]
    raw = base64.b64decode(record['base64'], validate=True)
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    path = tmp_path / 'original-v2.safetensors'
    path.write_bytes(raw)
    session = load_session(path)
    take = session.takes[-1]
    assert take.metadata['scene_chunking'] == dict(version=2, enabled=True, chunk_bytes=48)
    assert [unit['text'] for unit in take.metadata['scene_units']] == [
        '[curious] 今天又整了什么邪', '修？', '[deadpan] 我把会议改成了午', '睡。', '[excited] 这福利建议全国推', '广。']
    replanned = plan_scene(take.metadata['scene_cast_lines'], mode=mode, chunk_bytes=48, chunking_version=2)
    for expected, archived in zip(replanned['units'], take.metadata['scene_units']):
        assert all(archived[key] == value for key, value in expected.items())
    assert playback_plan(take)['version'] == 2 and validate_scene(take)
    engine = Harness()
    audio, timeline = engine.decode_take(take, with_timeline=True)
    output = tmp_path / 'roundtrip-v2.safetensors'
    save_session(session, output)
    again = load_session(output)
    assert again.takes[-1].metadata == take.metadata
    assert torch.equal(again.takes[-1].frames, take.frames)
    replay, replay_timeline = Harness().decode_take(again.takes[-1], with_timeline=True)
    assert torch.equal(audio['waveform'], replay['waveform']) and timeline == replay_timeline
    assert audio['waveform'].shape[-1] == take.codes.shape[-1] * 2048 + 2 * round(.3 * 44100)


@pytest.mark.parametrize('mode', MODES)
def test_default_v3_balances_real_48_byte_tail_without_changing_text_or_roles(mode):
    from drama3_t8.scene_chunks import VERSION
    assert VERSION == 3
    plan = plan_scene(SCRIPT, references(), mode=mode, chunk_bytes=48)
    assert [unit['text'] for unit in plan['units']] == [
        '[curious] 今天又整了', '什么邪修？', '[deadpan] 我把会议改', '成了午睡。', '[excited] 这福利建议', '全国推广。']
    assert [(unit['text_start'], unit['text_end']) for unit in plan['units']] == [(0, 15), (15, 20)] * 3
    assert [(unit['byte_start'], unit['byte_end']) for unit in plan['units']] == [(0, 25), (25, 40)] * 3
    for index, cast in enumerate(plan['cast_lines']):
        assert ''.join(unit['text'] for unit in plan['units'] if unit['line_index'] == index) == body_of(cast)[1]
    assert plan['line_speakers'] == [0, 1, 0]
    for request in plan['requests']:
        text = JOIN.join(tagged(unit['text'], unit['speaker_id'] if mode == 'together' else 0) for unit in request)
        assert len(text.encode('utf-8')) <= 48


def test_original_v1_and_v2_partition_functions_and_helpers_remain_exact():
    """Freeze the original planner bodies, allowing only V2's function rename."""
    import drama3_t8.scene_chunks as module
    expected = {
        'split_body_v1': 'a1383b40bbdf34123622982dfced67760733ee3171b555ce72533438f64548f0',
        'feasible_parts': 'a3c550504a84cd562a185369a2f1e60e6479de40201a31c6654b2984c20f41fd',
        '_body_content': 'a9718b7ea110735ebf56a5db59af952a6fd7440805f1497dd6c2590d3fa67937',
        '_content_feasible': 'd16b62ba1b2311a48354f87f6990779a0cccb6848981ee92577b685447690d55',
        'split_body_v2': '59963b45e5365d603bbc6e24146b3ac6fe69dfd2c624e4e8c8f1b910d4f9494d',
    }
    for name, digest in expected.items():
        node = ast.parse(inspect.getsource(getattr(module, name))).body[0]
        if name == 'split_body_v2':node.name = 'split_body'
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest


def test_final_two_chunks_match_independent_known_atom_cut_enumeration():
    """All integer cuts of fixture tokens, independently of production regex."""
    from drama3_t8.scene_chunks import _rebalance_short_tail
    checked = improved = 0
    for size in range(1, 5):
        for tokens in itertools.product(('甲', '乙', ' ', '。', '[x]', '<i>甲乙</i>'), repeat=size):
            body = ''.join(tokens)
            forbidden, cursor = set(), 0
            for token in tokens:
                if token in ('[x]', '<i>甲乙</i>'):
                    forbidden.update(range(cursor + 1, cursor + len(token)))
                cursor += len(token)
            def count(start, end):return sum(char in '甲乙' for char in body[start:end])
            def natural(cut):return body[cut - 1] in ' 。' or body[cut] == ' '
            for cap, actor in itertools.product((32, 36, 48), (0, 15)):
                budget = cap - len(f'<|speaker:{actor}|> '.encode())
                legal = [cut for cut in range(1, len(body)) if cut not in forbidden
                    and count(0, cut) and count(cut, len(body))
                    and len(body[:cut].encode()) <= budget and len(body[cut:].encode()) <= budget]
                for old_cut in legal:
                    original = [(0, old_cut), (old_cut, len(body))]
                    actual = _rebalance_short_tail(body, original, budget)
                    expected = original
                    if count(old_cut, len(body)) <= 2 and not natural(old_cut):
                        def score(cut):
                            return (abs(count(0, cut) - count(cut, len(body))), 0 if natural(cut) else 1,
                                    -count(0, cut), abs(cut - old_cut), cut)
                        best = min(legal, key=score)
                        if score(best)[:3] < score(old_cut)[:3]:expected = [(0, best), (best, len(body))]
                    assert actual == expected, (body, cap, actor, original, actual, expected)
                    assert len(actual) == 2 and ''.join(body[a:b] for a, b in actual) == body
                    assert all(count(a, b) and len(tagged(body[a:b], actor).encode()) <= cap for a, b in actual)
                    checked += 1
                    improved += actual != original
    assert checked > 10000 and improved > 2000


@pytest.mark.parametrize('budget', [32, 33, 48, 64, 127, 300, 999, 3000])
@pytest.mark.parametrize('actor', [0, 15])
def test_v3_preserves_v2_prefix_units_feasibility_bytes_atoms_and_count(budget, actor):
    from drama3_t8.scene_chunks import split_body, split_body_v2
    body = '[x] ' + '无空格长文本' * (budget // 3) + '<i>字</i>' + '甲乙丙丁戊己庚辛壬。'
    payload = budget - len(tagged('', actor).encode())
    old, new = split_body_v2(body, payload), split_body(body, payload)
    assert len(old) == len(new) and old[:-2] == new[:-2]
    assert ''.join(body[a:b] for a, b in new) == body
    for start, end in new:
        assert len(tagged(body[start:end], actor).encode()) <= budget
        from test_scene_chunk_body import content
        assert content(body[start:end])
        for token in ('[x]', '<i>字</i>'):
            point = body.index(token)
            assert not point < start < point + len(token) and not point < end < point + len(token)


def test_rebalancing_scans_only_the_bounded_last_two_units(monkeypatch):
    import drama3_t8.scene_chunks as module
    body = '甲' * 11002
    old = module.split_body_v2(body, 34)
    seen = []
    class ObservedPattern:
        def __init__(self, original):self.original = original
        def finditer(self, value):
            seen.append((len(value), len(value.encode())))
            return self.original.finditer(value)
    monkeypatch.setattr(module, 'MARKUP', ObservedPattern(module.MARKUP))
    monkeypatch.setattr(module, 'ATOMIC_SPAN', ObservedPattern(module.ATOMIC_SPAN))
    result = module._rebalance_short_tail(body, old, 34)
    assert len(old) > 1000 and len(old) == len(result) and old[:-2] == result[:-2]
    assert [end - start for start, end in result[-2:]] == [7, 6]
    assert len(seen) == 2 and all(chars == 13 and size == 39 and size <= 2 * 34 for chars, size in seen)


@pytest.mark.parametrize('version', [0, 4, -1, True, 2., '3', None])
def test_unknown_or_noninteger_planner_versions_reject_before_generation(version):
    with pytest.raises(ValueError, match='version'):
        plan_scene(['A: exact original text'], chunking_version=version)


@pytest.mark.parametrize('body,parts,budget', [
    ('甲乙。丙', [(0, 3), (3, 4)], 12),
    ('甲乙 丙', [(0, 3), (3, 4)], 12),
    ('甲乙\u3000丙', [(0, 3), (3, 4)], 12),
    ('[laughs][sighs]', [(0, 8), (8, 15)], 20),
    ('<i>甲乙</i>丙', [(0, 9), (9, 10)], 15),
    ('abcdefg', [(0, 4), (4, 7)], 5),
])
def test_natural_long_effect_or_immovable_tail_keeps_original_v2_parts(body, parts, budget):
    from drama3_t8.scene_chunks import _rebalance_short_tail
    assert _rebalance_short_tail(body, parts, budget) == parts


@pytest.mark.parametrize('text,forbidden', [
    ('a\u0301bcde', {1}), ('甲\u200d乙丙丁', {1, 2}), ('甲\ufe0f乙丙丁', {1}),
    ('🇨🇳甲乙丙', {1}), ('甲\U000e0100乙丙丁', {1}),
])
def test_balancing_never_introduces_new_combining_zwj_variation_or_flag_cut(text, forbidden):
    from drama3_t8.scene_chunks import _rebalance_short_tail
    for cut in range(1, len(text)):
        if cut in forbidden:continue
        old = [(0, cut), (cut, len(text))]
        result = _rebalance_short_tail(text, old, len(text.encode()))
        if result != old:assert result[0][1] not in forbidden


@pytest.mark.parametrize('mode', MODES)
def test_v3_real_engine_history_seed_windows_archive_and_playback(mode, tmp_path):
    expected_audio = {
        'together': [[1, 2], [1, 2, 10], [1, 2, 10, 20], [1, 2, 10, 30], [1, 2, 10, 40], [1, 2, 10, 50]],
        'per_line': [[1], [1, 10], [2], [2, 30], [1, 10, 20], [1, 10, 50]],
        'per_line_context': [[1], [1, 10], [2, 20], [2, 30, 20], [1, 10, 20, 40], [1, 10, 50, 40]],
        'per_speaker': [[1], [1, 10], [1, 10, 20], [1, 10, 30], [2], [2, 50]],
    }
    from drama3_t8.engine import Sampling
    from drama3_t8.scene_chunks import history_view
    engine = Harness()
    result = engine.scene(SCRIPT, references=references(), sampling=Sampling(seed=516, max_frames=300, opening_retries=0),
        generation_mode=mode, split_long_text=True, chunk_bytes=48)
    take = result.takes[-1]
    assert take.metadata['scene_chunking'] == dict(version=3, enabled=True, chunk_bytes=48)
    assert len(engine.requests) == 6 and [r['sampling'].seed for r in engine.requests] == list(range(516, 522))
    for request, expected in zip(engine.requests, expected_audio[mode]):
        prompt = request['prompt']
        audio = prompt[(prompt[:, 0] >= 248099) & (prompt[:, 0] <= 252194), 1]
        assert audio.unique_consecutive().tolist() == expected
        assert request['sampling'].opening_retries == 0
    path = tmp_path / 'planner-v3.safetensors'
    save_session(result, path)
    again = load_session(path)
    assert again.takes[-1].metadata == take.metadata and playback_plan(again.takes[-1])['version'] == 3
    assert len(history_view(again.takes)) == 2  # Actual first/latest sources, never composite codes.
    first, timeline = Harness().decode_take(take, with_timeline=True)
    second, after = Harness().decode_take(again.takes[-1], with_timeline=True)
    assert torch.equal(first['waveform'], second['waveform']) and timeline == after
    assert first['waveform'].shape[-1] == take.codes.shape[-1] * 2048 + 2 * round(.3 * 44100)


@pytest.mark.parametrize('mode', MODES)
def test_v2_and_v3_source_maps_cannot_be_relabelled_as_each_other(mode, tmp_path):
    row = json.loads((Path(__file__).parent / 'fixtures/scene-chunks-v2.json').read_text(encoding='utf-8'))['archives'][mode]
    path = tmp_path / 'v2.safetensors'
    path.write_bytes(base64.b64decode(row['base64'], validate=True))
    old = load_session(path).takes[-1]
    changed = copy.deepcopy(old.metadata)
    changed['scene_chunking']['version'] = 3
    with pytest.raises(ValueError, match='planner|policy|requests'):
        validate_scene(replace(old, metadata=changed))
    current = Harness().scene(SCRIPT, references=references(), generation_mode=mode,
        split_long_text=True, chunk_bytes=48).takes[-1]
    changed = copy.deepcopy(current.metadata)
    changed['scene_chunking']['version'] = 2
    with pytest.raises(ValueError, match='planner|policy|requests'):
        validate_scene(replace(current, metadata=changed))
