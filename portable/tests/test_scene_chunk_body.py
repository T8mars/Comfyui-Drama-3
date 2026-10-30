"""Direction-only auto chunks are not speech; original effect-only lines are valid."""
import base64
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pytest
import torch

from test_scene_chunks import Harness,MODES,references
from drama3_t8.scene_chunks import JOIN,body_of,tagged,plan_scene
from drama3_t8.prompts import spoken_text
from drama3_t8.scene_chunks import split_body,split_body_v1,validate_scene,playback_plan
from drama3_t8.state import load_session,save_session

SCRIPT=['主持人: [curious] 今天又整了什么邪修？',
        '猫主管: [deadpan] 我把会议改成了午睡。',
        '主持人: [excited] 这福利建议全国推广。']


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('budget',[48,300])
def test_real_direction_prefix_never_becomes_automatic_tag_only_unit(mode,budget):
    lines=SCRIPT if budget==48 else ['A: [curious] '+'今天又整了什么邪修'*30]
    engine=Harness()
    result=engine.scene(lines,references=references(),generation_mode=mode,
        split_long_text=True,chunk_bytes=budget)
    take=result.takes[-1]
    assert all(spoken_text(unit['text']) for unit in take.metadata['scene_units'])
    for index,cast in enumerate(take.metadata['scene_cast_lines']):
        assert ''.join(unit['text'] for unit in take.metadata['scene_units'] if unit['line_index']==index)==body_of(cast)[1]
    assert all(len(request['text'].encode('utf-8'))<=budget for request in engine.requests)


@pytest.mark.parametrize('effect',['[laughs]','[sighs]','[laughs] [sighs]'])
def test_explicit_original_effect_only_line_remains_supported(effect):
    plan=plan_scene(['A: '+effect],chunk_bytes=48)
    assert ''.join(unit['text'] for unit in plan['units'])==effect
    assert all(len(JOIN.join(tagged(unit['text'],0) for unit in request).encode())<=48 for request in plan['requests'])


def content(text):
    """Independent test predicate: strip syntax, then require a graphic body."""
    plain=re.sub(r'\[[^\[\]]*\]|<\|[^|>]*\|>|</?[A-Za-z][^>]*>','',text)
    return any(not char.isspace() and unicodedata.category(char)[0] not in {'P','C','Z'} for char in plain)


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('text,budget',[
    ('[curious][deadpan] 中文中文中文中文中文中文中文中文。',48),
    ('中文中文中文中文中文中文中文 [laughs] [sighs]',48),
    ('你好。 [curious] ... [deadpan]  继续说继续说继续说。',48),
    ('[curious] <whisper>悄悄话</whisper> 中文继续说。',64),
    ('你好。 <whisper>[laughs]</whisper> 接着说接着说。',64),
    ('[curious] 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀',48),
])
def test_leading_internal_trailing_controls_and_xml_scopes_stay_with_content(mode,text,budget):
    result=plan_scene(['A: '+text],mode=mode,chunk_bytes=budget)
    units=result['units']
    assert ''.join(unit['text'] for unit in units)==text
    assert all(content(unit['text']) for unit in units)
    for unit in units:
        assert text.encode()[unit['byte_start']:unit['byte_end']].decode()==unit['text']
        assert unit['text']==text[unit['text_start']:unit['text_end']]
        for token in re.findall(r'\[[^\[\]]*\]|<whisper>.*?</whisper>',text):
            assert not any(token[:cut]==unit['text'][-cut:] for cut in range(1,len(token)))
    for request in result['requests']:
        assert len(JOIN.join(tagged(unit['text'],unit['speaker_id'] if mode=='together' else 0) for unit in request).encode())<=budget


@pytest.mark.parametrize('text',['[abcdefghijklmno] 字','[curious] 。 。字字字字'])
def test_impossible_direction_body_binding_cannot_be_faked_with_punctuation(text):
    engine=Harness()
    with pytest.raises(ValueError,match='increase chunk_bytes|containing text'):
        engine.scene(['A: '+text],split_long_text=True,chunk_bytes=32)
    assert engine.requests==[] and engine.events==[]


@pytest.mark.parametrize('mode',MODES)
def test_immutable_v1_archive_load_decode_roundtrip_and_new_v2_rejects_old_tag_only_map(mode,tmp_path):
    fixture=json.loads((Path(__file__).parent/'fixtures/scene-chunks-v1.json').read_text(encoding='utf-8'))
    row=fixture['archives'][mode];data=base64.b64decode(row['base64'],validate=True)
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    path=tmp_path/'legacy.safetensors';path.write_bytes(data)
    session=load_session(path);take=session.takes[-1]
    assert take.metadata['scene_chunking']['version']==1
    assert any(not spoken_text(unit['text']) for unit in take.metadata['scene_units'])
    engine=Harness();audio,timeline=engine.decode_take(take,with_timeline=True)
    assert len(engine.decoded)==len(take.metadata['scene_sources'])
    assert len(timeline)==3 and playback_plan(take)['version']==1
    assert audio['waveform'].shape[-1]==take.codes.shape[-1]*2048+2*round(.3*44100)
    output=tmp_path/'roundtrip.safetensors';save_session(session,output);again=load_session(output)
    assert again.takes[-1].metadata==take.metadata and torch.equal(again.takes[-1].frames,take.frames)
    assert torch.equal(Harness().decode_take(again.takes[-1])['waveform'],audio['waveform'])
    bad=copy.deepcopy(take.metadata);bad['scene_chunking']['version']=2
    with pytest.raises(ValueError,match='planner|policy|requests'):
        validate_scene(replace(take,metadata=bad))
    with pytest.raises(ValueError,match='protected scene span|Direction controls'):
        engine.revise(session,'Same words, clear delivery')
    revised=engine.revise(session,'clear')
    assert revised.takes[-1].metadata['scene_chunking']==dict(version=3,enabled=True,chunk_bytes=48,request_direction='[clear]')
    assert all(content(unit['text']) for unit in revised.takes[-1].metadata['scene_units'])
    assert take.metadata['scene_chunking']['version']==1


def test_seeded_partition_oracle_covers_protected_body_and_pure_effect_lines():
    import random
    from drama3_t8.native.chunking import PROTECTED_SPAN
    rng=random.Random(516)
    alphabet=['a',' ','。','😀','[laughs]','[curious]','<whisper>字</whisper>']
    for _ in range(1000):
        text=''.join(rng.choice(alphabet) for _ in range(rng.randrange(1,8)))
        budget=rng.randrange(1,40)
        boundaries={0,len(text)};cursor=0
        for match in PROTECTED_SPAN.finditer(text):
            boundaries.update(range(cursor,match.start()+1));boundaries.add(match.end());cursor=match.end()
        boundaries.update(range(cursor,len(text)+1));boundaries=sorted(boundaries)
        for split in (split_body_v1,split_body):
            valid=content if split is split_body and content(text) else lambda part:bool(part.strip())
            reachable={0}
            for start in boundaries:
                if start in reachable:
                    reachable.update(end for end in boundaries if end>start and len(text[start:end].encode())<=budget and valid(text[start:end]))
            possible=len(text) in reachable
            try:parts=split(text,budget)
            except ValueError:assert not possible,(text,budget,split.__name__)
            else:
                assert possible and ''.join(text[a:b] for a,b in parts)==text
                assert all(valid(text[a:b]) and len(text[a:b].encode())<=budget for a,b in parts)


def test_large_tagged_script_counts_content_once_per_atom(monkeypatch):
    import drama3_t8.scene_chunks as module
    calls=[];actual=module._body_content
    def counted(piece):calls.append(len(piece));return actual(piece)
    monkeypatch.setattr(module,'_body_content',counted)
    text='[curious] '+'无空格长文本'*10000
    parts=split_body(text,286)
    assert ''.join(text[a:b] for a,b in parts)==text and all(content(text[a:b]) for a,b in parts)
    assert sum(calls)==len(text)  # No repeated full-prefix/body substring scan.
