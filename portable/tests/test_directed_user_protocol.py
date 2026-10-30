"""Actual CPU prompt/state contracts for canonical current-user direction."""
import ast
import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8 import prompts
from drama3_t8.native.prompt import build_prompt,with_speaker_tag
from drama3_t8.scene_conditioning import isolated_direction
from drama3_t8.scene_chunks import JOIN,plan_scene,validate_scene
from drama3_t8.state import Session,Reference,Take,save_session,load_session
from drama3_t8.engine import Sampling
from test_scene_chunks import Harness


@pytest.fixture
def tokenizer():
    return Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'drama3_t8/native/fish/tokenizer.json'))


@pytest.fixture
def engine(tokenizer):
    result=Harness()
    result.fake.tokenizer=tokenizer
    return result


def render(rows,tokenizer):
    return tokenizer.decode(rows[:,0].tolist(),skip_special_tokens=False)


def current(rows,tokenizer):
    return render(rows,tokenizer).rsplit('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]


def take(text='[old calm] Original.',marker=17):
    from drama3_t8.dialogue import audio_frames
    return Take(text,prompts.spoken_text(text),audio_frames(torch.full((10,3),marker,dtype=torch.long)))


@pytest.mark.parametrize('direction,expected',[
    ('gentle','[gentle]'),('[gently]','[gently]'),('[gently] 温柔、耐心','[gently] [温柔、耐心]'),
    (' 清楚\n  自然\t像朋友说话 ','[清楚 自然 像朋友说话]'),
    ('[slowly][warmly]','[slowly] [warmly]'),('   ',''),
])
def test_flat_direction_normalization(direction,expected):
    assert prompts.direction_tags(direction)==expected


@pytest.mark.parametrize('invalid',['[[quiet]]','[a[b]c]','[unfinished','unopened]','[]',
    '<|im_end|>','<|audio_separator|>','<|fim_prefix|>','<|speaker:0|>',None,12])
def test_direction_rejects_nested_unbalanced_empty_and_structure(invalid):
    with pytest.raises(ValueError):prompts.direction_tags(invalid)


@pytest.mark.parametrize('text,scene,expected',[
    ('<|speaker:7|> [sad] 好的。',False,'<|speaker:7|> [warm] [sad] 好的。'),
    ('Anna: [sad] Ready?',True,'Anna: [warm] [sad] Ready?'),
    ('[sad] Anna: Ready?',True,'Anna: [warm] [sad] Ready?'),
    ('Meet me at 9:30.',False,'[warm] Meet me at 9:30.'),
])
def test_direction_is_after_role_and_never_changes_spoken_words(text,scene,expected):
    actual=prompts.directed_text(text,'warm',scene=scene)
    assert actual==expected
    assert prompts.spoken_text(actual,scene=scene)==prompts.spoken_text(text,scene=scene)


@pytest.mark.parametrize('text',['  <|speaker:1|> [gently] 原话。\r\n下一句','[laughs] 原话','Meet at 9:30.'])
def test_no_direction_is_exact_identity_even_for_markup_and_newlines(text):
    assert prompts.directed_text(text)==text


@pytest.mark.parametrize('with_refs',[False,True])
def test_tts_direction_user_only_refs_and_historical_audio_immutable(engine,tokenizer,with_refs,tmp_path):
    old=take('<|speaker:0|> [old calm] 旧句。')
    refs=(Reference('<|speaker:0|> 原参考。',torch.ones(10,4,dtype=torch.long)),) if with_refs else ()
    parent=Session(refs,(old,),'自然清楚，准确读完当前台词。')
    archived=tmp_path/'old.safetensors';save_session(parent,archived);old_bytes=archived.read_bytes()
    before=copy.deepcopy(old.metadata)
    result=engine.speak('<|speaker:0|> [deadpan] 新句。',parent,Sampling(seed=42))
    request=engine.requests[-1]
    system=render(request['prompt'],tokenizer).split('<|im_end|>',1)[0]
    assert parent.direction not in system
    assert current(request['prompt'],tokenizer)=='<|speaker:0|> [自然清楚，准确读完当前台词。] [deadpan] 新句。'
    assert result.takes[-1].prompt_text==current(request['prompt'],tokenizer)
    assert result.takes[-1].transcript=='新句。'
    assert result.takes[0] is old and old.metadata==before
    assert result.direction==parent.direction and archived.read_bytes()==old_bytes
    save_session(result,tmp_path/'new.safetensors')
    reloaded=load_session(tmp_path/'new.safetensors')
    assert reloaded.takes[-1].prompt_text==result.takes[-1].prompt_text
    assert reloaded.takes[0].prompt_text==old.prompt_text
    assert render(request['prompt'],tokenizer).count('[old calm]')==1


@pytest.mark.parametrize('with_refs',[False,True])
def test_empty_direction_rows_equal_native_raw_builder(engine,tokenizer,with_refs):
    refs=(Reference('参考。',torch.ones(10,2,dtype=torch.long)),) if with_refs else ()
    old=take('<|speaker:0|> [old] old words')
    text='  <|speaker:0|> [gently] 原话。\n下一句'
    expected=build_prompt(tokenizer,engine.fake.config,text,
        [(ref.transcript,ref.codes) for ref in refs],[(old.prompt_text,old.codes)])
    result=engine.speak(text,Session(refs,(old,)),Sampling())
    assert torch.equal(engine.requests[-1]['prompt'],expected)
    assert result.takes[-1].prompt_text==text


def test_revise_applies_new_note_once_and_never_rewrites_old_actual_user(engine,tokenizer,tmp_path):
    original=take('<|speaker:0|> [old sad] 原话。')
    parent=Session(takes=(original,),direction='清楚自然')
    for note in ('[warm]','[amused]'):
        parent=engine.revise(parent,note,Sampling())
        actual=current(engine.requests[-1]['prompt'],tokenizer)
        assert actual==f'<|speaker:0|> [清楚自然] {note} 原话。'
        assert parent.takes[-1].prompt_text==actual
    assert parent.takes[0] is original and original.prompt_text=='<|speaker:0|> [old sad] 原话。'
    assert '[warm] [amused]' not in parent.takes[-1].prompt_text


def test_stream_current_direction_stays_before_endofturn_lookahead_is_exact(engine,tokenizer,tmp_path):
    old=take('<|speaker:0|> [old] 原句。')
    future='<|speaker:4|> [curious] 下一句。'
    result=engine.stream_text('<|speaker:2|> 当前句。',future,Session(takes=(old,),direction='natural'),Sampling())
    request=engine.requests[-1]
    expected='<|speaker:2|> [natural] 当前句。'
    assert current(request['prompt'],tokenizer)==expected+'<|endofturn|>'+future
    assert result.takes[-1].prompt_text==expected
    assert result.takes[-1].transcript=='当前句。' and result.takes[-1].metadata['lookahead']==future
    engine.stream_text('再一句。','',result,Sampling())
    history=render(engine.requests[-1]['prompt'],tokenizer).split('<|im_start|>user\n')[-2]
    assert expected+'<|endofturn|>'+future in history
    assert result.takes[-1].prompt_text==expected


def test_native_revise_retains_real_speaker_instead_of_default_zero(engine,tokenizer):
    source=take('<|speaker:2|> [sad] 原文。')
    reference=Reference('<|speaker:2|> 不同的参考。',torch.ones(10,2,dtype=torch.long))
    result=engine.revise(Session((reference,),(source,)),'warm',Sampling())
    assert current(engine.requests[-1]['prompt'],tokenizer)=='<|speaker:2|> [warm] 原文。'
    assert result.takes[-1].prompt_text=='<|speaker:2|> [warm] 原文。'
    assert result.takes[-1].transcript=='原文。'


def test_new_native_raw_request_preserves_inline_effect_without_old_automatic_style(engine,tokenizer):
    parent=Session(direction='calm')
    parent=engine.speak('<|speaker:2|> [laughs] 原文。',parent,Sampling())
    old=parent.takes[-1]
    assert old.metadata['raw_request_text']=='<|speaker:2|> [laughs] 原文。'
    for note in ('[warm]','[amused]'):
        parent=engine.revise(parent,note,Sampling())
        assert current(engine.requests[-1]['prompt'],tokenizer)==f'<|speaker:2|> [calm] {note} [laughs] 原文。'
        assert parent.takes[-1].transcript=='原文。'
    assert old.prompt_text=='<|speaker:2|> [calm] [laughs] 原文。'


def test_revision_replaces_opening_warm_delivery_and_keeps_events_and_accent(engine,tokenizer):
    parent=engine.speak('<|speaker:2|> [四川话] [warmly] 原文。[laughs]',Session(),Sampling())
    original=parent.takes[-1]
    revised=engine.revise(parent,'angry, faster',Sampling(),replace_delivery_tags=True)
    assert current(engine.requests[-1]['prompt'],tokenizer)=='<|speaker:2|> Direction: angry, faster\nLine: [angry, faster] [四川话] 原文。[laughs]'
    assert original.prompt_text=='<|speaker:2|> [四川话] [warmly] 原文。[laughs]'
    assert revised.takes[-1].metadata['director_note']=='angry, faster'
    assert revised.takes[-1].metadata['revision_source_take_id']==original.take_id
    assert revised.takes[-1].metadata['replaced_delivery_controls']==['warmly']
    kept=engine.revise(parent,'angry, faster',Sampling(),replace_delivery_tags=False)
    assert '[warmly]' in current(engine.requests[-1]['prompt'],tokenizer)
    assert kept.takes[-1].metadata['replaced_delivery_controls']==[]


def test_director_instruction_overrides_saved_emotion_context_but_preserves_history(engine,tokenizer):
    parent=engine.speak('<|speaker:2|> 原文。',Session(direction='calm'),Sampling())
    original=parent.takes[-1]
    revised=engine.revise(parent,'angry, faster',Sampling(),replace_delivery_tags=True)
    assert current(engine.requests[-1]['prompt'],tokenizer)=='<|speaker:2|> Direction: angry, faster\nLine: [angry, faster] 原文。'
    assert original.prompt_text=='<|speaker:2|> [calm] 原文。' and parent.direction=='calm'
    assert revised.takes[-1].transcript=='原文。'
    assert revised.takes[-1].metadata['director_instruction_format']=='direction_line'
    from drama3_t8.scene_cast import _scene_lines
    assert _scene_lines(revised.takes[-1])==(revised.takes[-1].prompt_text,)


def test_chinese_colon_cast_labels_are_identical_to_ascii_colons(engine,tokenizer):
    refs=(Reference('<|speaker:0|> 参考。',torch.ones(10,2,dtype=torch.long)),)
    for delimiter in (':','：'):
        lines=[f'导演{delimiter} [angry] 现在开始。',f'女主{delimiter} [warmly] 知道啦。']
        source=Session(refs,(),direction='clear')
        result=engine.scene(lines,characters='导演：生气\n女主：亲切',session=source)
        assert result.takes[-1].metadata['scene_speaker_map']=={'导演':0,'女主':1}
        assert result.takes[-1].transcript=='现在开始。\n知道啦。'
        assert '<|speaker:1|> [clear] [warmly] 知道啦。' in current(engine.requests[-1]['prompt'],tokenizer)
        system=render(engine.requests[-1]['prompt'],tokenizer).split('<|im_end|>',1)[0]
        assert '导演: <|speaker:0|>' in system and '女主: <|speaker:1|>' in system


@pytest.mark.parametrize('raw',['different spoken words','<|speaker:3|> 原文。'])
def test_imported_raw_request_mismatch_is_not_silently_used(engine,raw):
    source=take('<|speaker:2|> 原文。')
    source=replace(source,metadata={'raw_request_text':raw})
    with pytest.raises(ValueError,match='raw_request_text differs'):
        engine.revise(Session(takes=(source,)),'warm')
    assert engine.requests==[]


@pytest.mark.parametrize('with_refs',[False,True])
@pytest.mark.parametrize('named',[False,True])
def test_scene_actual_user_archives_each_role_style_but_not_source_script(engine,tokenizer,tmp_path,with_refs,named):
    lines=['Anna: [quietly] Ready?','Ben: Always.'] if named else ['<|speaker:0|> [quietly] Ready?','<|speaker:1|> Always.']
    refs=(Reference('<|speaker:0|> source',torch.ones(10,2,dtype=torch.long)),) if with_refs else ()
    parent=Session(refs,(),direction='steady')
    result=engine.scene(lines,'A small office.','Anna: calm.\nBen: cheerful.',session=parent,note='warmer')
    request=engine.requests[-1]
    actual=current(request['prompt'],tokenizer)
    assert actual.count('[steady]')==2 and actual.count('[warmer]')==2
    assert actual.count(prompts.SEPARATOR)==1
    assert request['text']==actual==result.takes[-1].prompt_text
    assert result.takes[-1].metadata['script_lines']==lines
    assert result.takes[-1].metadata['director_note']=='warmer'
    assert result.takes[-1].transcript=='Ready?\nAlways.'
    system=render(request['prompt'],tokenizer).split('<|im_end|>',1)[0]
    assert 'A small office.' in system and 'Ben: cheerful.' in system
    assert '[steady]' not in system and '[warmer]' not in system
    save_session(result,tmp_path/'scene.safetensors')
    assert load_session(tmp_path/'scene.safetensors').takes[-1].prompt_text==actual


@pytest.mark.parametrize('mode',['together','per_line','per_line_context','per_speaker'])
def test_directed_chunk_actual_budget_archives_and_restore(engine,mode,tmp_path):
    lines=['A: '+'长文需要完整说完。'*18,'B: '+'另一位角色回复。'*8]
    result=engine.scene(lines,session=Session(direction='清楚自然'),note='gently',
        generation_mode=mode,split_long_text=True,chunk_bytes=96,sampling=Sampling(max_frames=200))
    assert all(len(item['text'].encode('utf-8'))<=96 for item in engine.requests)
    assert all('[清楚自然]' in item['text'] and '[gently]' in item['text'] for item in engine.requests)
    policy=result.takes[-1].metadata['scene_chunking']
    assert policy['request_direction']=='[清楚自然] [gently]'
    validate_scene(result.takes[-1])
    file=tmp_path/'directed.safetensors';save_session(result,file);loaded=load_session(file)
    assert loaded.takes[-1].prompt_text==result.takes[-1].prompt_text
    assert loaded.takes[-1].metadata['script_lines']==lines
    bad=copy.deepcopy(loaded.takes[-1].metadata)
    bad['scene_sources'][0]['prompt_text']=bad['scene_sources'][0]['prompt_text'].replace('[gently] ','')
    with pytest.raises(ValueError,match='prompt'):validate_scene(replace(loaded.takes[-1],metadata=bad))


def test_user_hint_larger_than_budget_is_rejected_before_generation(engine):
    with pytest.raises(ValueError,match='Direction controls cannot fit'):
        engine.scene(['A: words'],session=Session(direction='长'*80),split_long_text=True,chunk_bytes=96)
    assert engine.requests==[]
    with pytest.raises(ValueError,match='Direction controls cannot fit'):
        engine.long_form('words',Session(direction='长'*80),chunk_bytes=96)
    assert engine.requests==[]


def test_long_every_actual_chunk_has_single_style_and_stays_within_budget(engine):
    result=engine.long_form('English words for a long paragraph. '*14,Session(direction='自然清楚'),chunk_bytes=96)
    assert len(engine.requests)>1
    assert all(len(item['text'].encode('utf-8'))<=96 and item['text'].count('[自然清楚]')==1 for item in engine.requests)
    assert all(t.prompt_text==item['text'] for t,item in zip(result.takes,engine.requests))


def test_empty_native_isolated_controls_do_not_manufacture_mapping_text():
    assert isolated_direction('','','',actor=7,name='角色甲',previous_actor=3,note='')==''
    assert isolated_direction('[warm]','','',actor=7,name='角色甲')=='[warm]'


@pytest.mark.parametrize('mode',['per_line','per_line_context','per_speaker'])
@pytest.mark.parametrize('split',[False,True])
def test_isolated_real_scene_context_stays_in_system_and_actual_speaker_rows(engine,tokenizer,mode,split):
    description='User scene in the office.'
    characters='<|speaker:7|>: 甲稳重。\n<|speaker:3|>: 乙活泼。'
    lines=['<|speaker:7|> 甲说话。','<|speaker:3|> 乙回复。']
    refs=(Reference('<|speaker:7|> 甲参考',torch.ones(10,2,dtype=torch.long)),
          Reference('<|speaker:3|> 乙参考',torch.full((10,2),2,dtype=torch.long)))
    result=engine.scene(lines,description,characters,session=Session(refs,direction='自然'),
        generation_mode=mode,split_long_text=split,chunk_bytes=64)
    for request in engine.requests:
        system=render(request['prompt'],tokenizer).split('<|im_end|>',1)[0]
        body=current(request['prompt'],tokenizer)
        assert description in system and characters in system
        assert description not in body and 'Current character:' not in body
        assert body.startswith('<|speaker:0|> [自然] ')
        assert len(body.encode('utf-8'))<=64
    assert result.takes[-1].metadata['script_lines']==lines


def test_native_system_context_persists_for_revision_without_becoming_user_style(engine,tokenizer,tmp_path):
    context='Scene: quiet office.\nCharacters: actor two.'
    parent=engine.speak('<|speaker:2|> [laughs] 原话。',Session(direction='calm'),Sampling(),system_context=context)
    path=tmp_path/'system-context.safetensors';save_session(parent,path);original=path.read_bytes()
    parent=load_session(path)
    result=engine.revise(parent,'warm',Sampling())
    actual=render(engine.requests[-1]['prompt'],tokenizer)
    assert context in actual.split('<|im_end|>',1)[0]
    assert context not in current(engine.requests[-1]['prompt'],tokenizer)
    assert result.takes[-1].metadata['system_context']==context
    assert path.read_bytes()==original


def test_creative_only_overall_direction_goes_to_session():
    from drama3_t8.creative_studio import _delivery_direction
    assert _delivery_direction({'settings':{'direction':'overall'},'lines':[{'direction':'A only'},{'direction':'B only'}]})=='overall'


@pytest.mark.parametrize('mode',['together','per_speaker','per_line'])
def test_creative_line_delivery_never_leaks_to_other_roles_or_source_text(tmp_path,mode):
    from test_creative_studio_execution import client_at,wait_job,TinyEngine
    app,client=client_at(tmp_path)
    with client:
        engine=TinyEngine();app.state.studio.engine=engine
        rows=[{'line_id':'L01','role':'A','original_text':'甲的原稿。','direction':'A only'},
              {'line_id':'L02','role':'B','original_text':'乙的原稿。','direction':'B only'}]
        view=client.post('/api/creative/projects',json={'name':'方向分层','settings':{'direction':'overall'},'lines':rows}).json()
        pid=view['project_id']
        job=client.post(f'/api/creative/projects/{pid}/run',json={'generation_mode':mode,'takes_per_line':1}).json()
        done=wait_job(client,job['job_id']);assert done['status']=='done',done
        a=next(text for text in engine.spoken_texts if '甲的原稿。' in text)
        b=next(text for text in engine.spoken_texts if '乙的原稿。' in text)
        assert '[A only]' in a and '[B only]' not in a
        assert '[B only]' in b and '[A only]' not in b
        assert '[overall]' in a and '[overall]' in b
        after=client.get(f'/api/creative/projects/{pid}').json()
        for raw,line in zip(rows,after['lines']):
            assert line['original_text']==line['readout_text']==line['subtitle_text']==raw['original_text']
        assert all(direction=='overall' for direction in engine.delivery_directions)


@pytest.mark.parametrize('literal',['Meet at 9:30.','Note: read this.'])
def test_creative_accepts_correct_native_pure_spoken_literal_colon(tmp_path,literal):
    from test_creative_studio_execution import client_at,wait_job,TinyEngine
    class NativeTranscriptEngine(TinyEngine):
        def _make(self,lines,session,sampling,on_codes):
            result=super()._make(lines,session,sampling,on_codes)
            actual=replace(result.takes[-1],transcript=prompts.spoken_text('\n'.join(lines),scene=True))
            return replace(result,takes=(*result.takes[:-1],actual))
    app,client=client_at(tmp_path)
    with client:
        engine=NativeTranscriptEngine();app.state.studio.engine=engine
        project=client.post('/api/creative/projects',json={'lines':[literal]}).json()
        pid=project['project_id']
        job=client.post(f'/api/creative/projects/{pid}/run',json={'takes_per_line':1}).json()
        completed=wait_job(client,job['job_id']);assert completed['status']=='done',completed
        project=client.get(f'/api/creative/projects/{pid}').json()
        source=project['candidates'][project['lines'][0]['selected_candidate_id']]
        archived=load_session(source['session_path'])
        assert archived.takes[-1].transcript==literal
        assert project['lines'][0]['original_text']==project['lines'][0]['subtitle_text']==literal


def test_creative_still_refuses_native_transcript_with_wrong_words(tmp_path):
    from test_creative_studio_execution import client_at,wait_job,TinyEngine
    class WrongTranscriptEngine(TinyEngine):
        def _make(self,lines,session,sampling,on_codes):
            result=super()._make(lines,session,sampling,on_codes)
            return replace(result,takes=(*result.takes[:-1],replace(result.takes[-1],transcript='Different words.')))
    app,client=client_at(tmp_path)
    with client:
        engine=WrongTranscriptEngine();app.state.studio.engine=engine
        project=client.post('/api/creative/projects',json={'lines':['Meet at 9:30.']}).json()
        job=client.post(f"/api/creative/projects/{project['project_id']}/run",json={'takes_per_line':1}).json()
        completed=wait_job(client,job['job_id']);assert completed['status']=='failed'
        after=client.get(f"/api/creative/projects/{project['project_id']}").json()
        assert after['lines'][0]['selected_candidate_id'] is None
        candidate=next(iter(after['candidates'].values()))
        assert candidate['status']=='failed' and '原生演出台词记录' in candidate['error']
        assert engine.codec is None


def test_fim_signal_comparison_preserves_new_spoken_prefix_before_literal_colon(tmp_path,monkeypatch):
    from test_creative_infill_execution import setup
    from drama3_t8 import creative_infill_quality as quality
    from drama3_t8.creative_workspace import _file_record
    manager,engine,source,job,output=setup(tmp_path)
    source=replace(source,prompt_text='<|speaker:0|> Note: read this.',transcript='Note: read this.')
    archive=Path(job['source_candidate']['session_path'])
    save_session(Session(takes=(source,)),archive)
    for artifact in job['source_candidate']['artifacts']:
        if artifact['kind']=='session':artifact.update(_file_record(archive))
    job['lines'][0]['readout_text']='Alert: read this.'
    calls=[];real=quality.analyze_infill_insert
    def capture(before,after,*args):
        calls.append((before,after))
        return real(before,after,*args)
    monkeypatch.setattr(quality,'analyze_infill_insert',capture)
    result=manager._generate(job,output,lambda *_:None)
    assert calls==[('Note: read this.','Alert: read this.')]
    import json
    evidence=json.loads((output/'infill-energy.json').read_text(encoding='utf-8'))
    assert evidence['requires_new_spoken_alnum'] is True
    assert quality.needs_new_spoken_alnum('read this.','read this.') is False
    assert result['conditioning_session_path']


def test_fim_builder_function_keeps_frozen_protocol_sha256():
    def function(path):
        source=path.read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='infill_prompt')
        return ast.get_source_segment(source,node)
    shadow=Path(prompts.__file__)
    assert hashlib.sha256(function(shadow).encode('utf-8')).hexdigest()=='851768c43020f8f814b989e952b6287a6051473115838b09930732a4ce103ba9'
