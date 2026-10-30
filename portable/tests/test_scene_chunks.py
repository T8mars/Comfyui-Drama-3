"""Real Engine/prompt/node/archive CPU contracts; no checkpoint or GPU inference."""
import contextlib
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from drama3_t8.engine import DramaEngine, Sampling
from drama3_t8.state import Session, Reference, Take, save_session, load_session, validate_session
from drama3_t8.dialogue import audio_frames, own_history
from drama3_t8.scene_chunks import (JOIN, body_of, tagged, plan_scene, validate_scene,
    playback_plan, decode_scene, history_view, source_history)

MODES = ('together', 'per_line', 'per_line_context', 'per_speaker')
SCRIPT = ['A: 今天又是什么邪修？我把很长的会议通知改成午睡申请，领导居然秒批。',
          'B: 猫主管说：这就是我的睡后收入。',
          'A: 这福利建议全国推广。真的非常需要。']


class Tokenizer:
    def encode(self, text):return SimpleNamespace(ids=[100 + ord(c) % 100 for c in text])
    def token_to_id(self, text):return 252196 if text == '<|audio_separator|>' else 100
    def id_to_token(self, token):return '<|audio_separator|>' if token == 252196 else str(token)


class Harness(DramaEngine):
    """Keep the real public methods and prompt builders; replace only inference/codec."""
    def __init__(self):
        super().__init__('model','codec')
        self.requests=[];self.decoded=[];self.events=[];self.fail_at=None;self.cancel_at=None
        self.fake=SimpleNamespace(tokenizer=Tokenizer(),config=SimpleNamespace(output_token_ids=(252196,),
            semantic_start_token_id=248099,semantic_end_token_id=252194,tower=SimpleNamespace(num_codebooks=10,vocab_size=4096)))
    def load_model(self):return self.fake
    @contextlib.contextmanager
    def generation_session(self):
        self.events.append('session_enter')
        try:yield
        finally:self.events.append('session_exit')
    def _take(self,prompt,request,transcript,sampling,on_codes=None,progress=None,**kwargs):
        index=len(self.requests)
        self.requests.append(dict(prompt=prompt.clone(),text=request,sampling=sampling,kwargs=kwargs))
        self.events.append(('request',index))
        if index==self.fail_at:raise RuntimeError('Injected source failure')
        if index==self.cancel_at:mm.interrupt_current_processing(True)
        texts=request.split(JOIN);rows=[]
        for number,text in enumerate(texts):
            if number:
                control=torch.zeros((1,11),dtype=torch.long);control[0,0]=252196;rows.append(control)
            codes=torch.full((10,2+number),10*(index+1)+number,dtype=torch.long)
            rows.append(audio_frames(codes))
        return Take(request,transcript,torch.cat(rows),metadata=dict(seed=sampling.seed,sampling=sampling.__dict__,
            elapsed_seconds=.01,length_limit_reached=False,expert_placement='ram',moe_backend='torch',
            expert_backend_calls={'torch':index+1},cpu_experts_run=index+1,cpu_decode_calls=0))
    def decode(self,codes):
        self.decoded.append(codes.clone())
        # The first frame changes every later sample: detecting wrong Codec
        # source concatenation requires more than counting decode calls.
        wave=(codes[:1].float()+1000*codes[0,0]).repeat_interleave(2048,dim=-1)
        return {'waveform':wave[None],'sample_rate':44100}


def references():
    return (Reference('<|speaker:0|> one',torch.ones((10,3),dtype=torch.long),'display only'),
            Reference('<|speaker:1|> two',torch.full((10,3),2,dtype=torch.long),'not a binding'))


def make(mode='together', budget=100, engine=None, parent=None, on_codes=None):
    engine=engine or Harness();parent=parent or Session(references=references())
    result=engine.scene(SCRIPT,references=parent.references,sampling=Sampling(seed=2**64-2,max_frames=80),
        session=parent,generation_mode=mode,split_long_text=True,chunk_bytes=budget,on_codes=on_codes)
    return engine,parent,result


@pytest.mark.parametrize('mode',MODES)
def test_planner_exact_text_unicode_offsets_actual_serialized_budget_and_global_order(mode):
    lines=['A: [laughs] 中文  😀 emoji  3.14。<whisper>悄悄话</whisper> 继续说。'*2,
           'B: two','A: same actor returns']
    plan=plan_scene(lines, references(),mode=mode,chunk_bytes=100)
    assert plan['speaker_map']=={'A':0,'B':1} and plan['line_speakers']==[0,1,0]
    for line_index,cast in enumerate(plan['cast_lines']):
        body=body_of(cast)[1];parts=[u for u in plan['units'] if u['line_index']==line_index]
        assert ''.join(u['text'] for u in parts)==body
        assert parts[0]['text_start']==parts[0]['byte_start']==0
        assert parts[-1]['text_end']==len(body)
        for unit in parts:
            assert body[unit['text_start']:unit['text_end']]==unit['text']
            assert body.encode('utf-8')[unit['byte_start']:unit['byte_end']].decode('utf-8')==unit['text']
    for request in plan['requests']:
        assert len(JOIN.join(tagged(u['text'],u['speaker_id'] if mode=='together' else 0) for u in request).encode('utf-8'))<=100
    ids=[u['unit_index'] for req in plan['requests'] for u in req]
    assert sorted(ids)==list(range(len(plan['units'])))
    if mode=='per_speaker':
        assert [u['speaker_id'] for req in plan['requests'] for u in req]==sorted(u['speaker_id'] for u in plan['units'])
    else:assert ids==list(range(len(plan['units'])))


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_split_flag_is_strict_and_rejected_before_loading(value):
    engine=Harness()
    with pytest.raises(ValueError):engine.scene(['hi'],split_long_text=value)
    assert engine.requests==[] and engine.events==[]


@pytest.mark.parametrize('budget',[True,31,3001,32.,float('nan'),'300'])
def test_budget_is_strict_before_loading(budget):
    engine=Harness()
    with pytest.raises(ValueError):engine.scene(['hi'],split_long_text=True,chunk_bytes=budget)
    assert not engine.requests and not engine.events


@pytest.mark.parametrize('tail',['  ','\u3000'])
def test_trailing_whitespace_borrows_last_free_atom_without_losing_bytes(tail):
    plan=plan_scene(['A: '+'a'*17+tail],chunk_bytes=32)
    assert len(plan['units'])==2
    assert ''.join(unit['text'] for unit in plan['units'])=='a'*17+tail
    assert all(unit['text'].strip() for unit in plan['units'])
    assert all(len(tagged(unit['text'],0).encode('utf-8'))<=32 for unit in plan['units'])


def test_non_unicode_whitespace_control_before_tag_is_not_silently_dropped():
    assert body_of('\x1c<|speaker:1|> words')==(0,'\x1c<|speaker:1|> words')
    with pytest.raises(ValueError):plan_scene(['\x1c<|speaker:1|> words'],chunk_bytes=100)


def test_internal_whitespace_uses_feasible_protected_partition_instead_of_false_rejection():
    text='a。'+' '*11+'[laughs]bbb'
    plan=plan_scene(['A: '+text],chunk_bytes=32)
    chunks=[unit['text'] for unit in plan['units']]
    assert ''.join(chunks)==text and all(part.strip() for part in chunks)
    assert all(len(tagged(part,0).encode('utf-8'))<=32 for part in chunks)
    assert sum('[laughs]' in part for part in chunks)==1


def test_fallback_suffix_feasibility_matches_exhaustive_short_text_partitions():
    import itertools
    from drama3_t8.scene_chunks import split_body,split_body_v1
    for count in range(1,7):
        for chars in itertools.product(('a',' ','。'),repeat=count):
            text=''.join(chars)
            for budget in (2,3,4):
                for split in (split_body_v1,split_body):
                    valid=(lambda part:'a' in part) if split is split_body and 'a' in text else (lambda part:bool(part.strip()))
                    reachable={0}
                    for start in range(len(text)):
                        if start in reachable:
                            reachable.update(end for end in range(start+1,len(text)+1)
                                if len(text[start:end].encode('utf-8'))<=budget and valid(text[start:end]))
                    possible=len(text) in reachable
                    try:parts=split(text,budget)
                    except ValueError:assert not possible
                    else:
                        assert possible and ''.join(text[a:b] for a,b in parts)==text
                        assert all(valid(text[a:b]) and len(text[a:b].encode('utf-8'))<=budget for a,b in parts)


def test_indivisible_markup_overhead_and_multiple_roles_fail_before_inference():
    for script in (['A: <whisper>'+'x'*80+'</whisper>'],['<|speaker:0|> hi <|speaker:1|> no'],
                   ['<|speaker:0|> '+' '*80+'Hello']):
        engine=Harness()
        with pytest.raises(ValueError):engine.scene(script,split_long_text=True,chunk_bytes=32)
        assert engine.requests==[] and engine.events==[]


@pytest.mark.parametrize('mode',MODES)
def test_real_engine_source_map_seeds_history_stats_and_archive_roundtrip(mode,tmp_path):
    engine,parent,result=make(mode)
    take=result.takes[-1];meta=take.metadata
    assert parent.takes==() and meta['script_lines']==SCRIPT and len(take.segments())==3
    assert meta['scene_speaker_map']=={'A':0,'B':1} and meta['line_speakers']==[0,1,0]
    assert meta['request_count']==len(engine.requests)>3
    assert [r['sampling'].seed for r in engine.requests]==[(2**64-2+i)%2**64 for i in range(len(engine.requests))]
    assert max(len(r['text'].encode('utf-8')) for r in engine.requests)<=100
    assert engine.events[0]=='session_enter' and engine.events[-1]=='session_exit'
    assert engine.events.count('session_enter')==1
    assert len(validate_scene(take))==len(engine.requests)
    assert meta['expert_backend_calls']['torch']==sum(range(1,len(engine.requests)+1))
    path=tmp_path/'scene.safetensors';save_session(result,path);restored=load_session(path)
    assert restored.takes[-1].metadata==meta and torch.equal(restored.takes[-1].frames,take.frames)
    first=engine.decode_take(take,with_timeline=True)
    second=engine.decode_take(restored.takes[-1],with_timeline=True)
    assert torch.equal(first[0]['waveform'],second[0]['waveform']) and first[1]==second[1]
    assert [row['speaker'] for row in first[1]]==[0,1,0]
    assert [row['line'] for row in first[1]]==[0,1,2]
    assert first[0]['waveform'].shape[-1]==take.codes.shape[-1]*2048+2*round(.3*44100)
    # Per-role history uses local 0, ordinary prompt history uses global roles.
    for actor,takes in own_history(restored).items():
        assert takes and all(t.prompt_text.startswith('<|speaker:0|>') for t in takes)
    if mode!='together':
        global_sources=source_history(take)
        b=next(t for t,raw in zip(global_sources,meta['scene_sources']) if raw['speaker']==1)
        assert b.prompt_text.startswith('<|speaker:1|>')


@pytest.mark.parametrize('mode',MODES)
def test_source_context_decode_once_then_slice_original_lines_and_no_chunk_gaps(mode):
    engine,_,result=make(mode);take=result.takes[-1]
    plan=playback_plan(take);audio,timeline=decode_scene(engine,take)
    assert len(engine.decoded)==len(plan['sources'])
    for source,decoded in zip(plan['sources'],engine.decoded):
        assert torch.equal(decoded,source['take'].codes)
        for unit in source['units']:
            line_start=round(timeline[unit['line_index']]['start']*44100)
            frame=unit['line_frame_start']
            output=audio['waveform'][0,0,line_start+frame*2048]
            expected=decoded[0,unit['source_start']].float()+1000*decoded[0,0]
            assert output==expected
    for line in range(2):
        assert round((timeline[line+1]['start']-timeline[line]['end'])*44100)==round(.3*44100)


@pytest.mark.parametrize('mode',MODES)
def test_two_revisions_after_real_archive_roundtrip_keep_policy_role_ids_and_history(mode,tmp_path):
    engine,_,result=make(mode);first=result.takes[0]
    for note in ('slower','laughing'):
        path=tmp_path/'revision.safetensors';save_session(result,path);restored=load_session(path)
        result=engine.revise(restored,note,Sampling(seed=7,max_frames=80))
        last=result.takes[-1]
        assert last.metadata['generation_mode']==mode and last.metadata['scene_chunking']=={**first.metadata['scene_chunking'],'request_direction':'['+note+']'}
        assert last.metadata['script_lines']==SCRIPT and last.metadata['scene_speaker_map']=={'A':0,'B':1}
        assert last.metadata['director_note']==note
    assert len(result.takes)==3 and torch.equal(result.takes[0].frames,first.frames)


@pytest.mark.parametrize('failure',['error','cancel'])
def test_failure_and_official_cancel_never_publish_partial_scene_and_recover(failure):
    engine=Harness();parent=Session(references=references());published=[]
    class Callback:
        def append(self,frame):pytest.fail('Partial frame preview is forbidden')
        def append_scene(self,take):published.append(take)
    if failure=='error':engine.fail_at=1;error=RuntimeError
    else:engine.cancel_at=1;error=mm.InterruptProcessingException
    try:
        with pytest.raises(error):make('per_line',engine=engine,parent=parent,on_codes=Callback().append)
        assert parent.takes==() and not published and engine.events[-1]=='session_exit'
    finally:mm.interrupt_current_processing(False)
    engine.fail_at=engine.cancel_at=None
    result=engine.scene(['A: recovered'],references=references(),split_long_text=True,chunk_bytes=100)
    assert len(result.takes)==1


@pytest.mark.parametrize('change',['version','duplicate_source','source_frames','source_codes','unit_frame','missing_unit','text','offset','source_order','actor','gap','unknown_id'])
def test_corrupt_maps_rejected_by_archive_decode_history_editor_and_infill(change,tmp_path):
    from drama3_t8.editor import fingerprint
    engine,_,result=make('per_line');original=result.takes[-1]
    meta=copy.deepcopy(original.metadata)
    if change=='version':meta['scene_chunking']['version']=999
    elif change=='duplicate_source':meta['scene_sources'][1]['take_id']=meta['scene_sources'][0]['take_id']
    elif change=='source_frames':meta['scene_sources'][0]['frames'][0][0]=999999
    elif change=='source_codes':meta['scene_sources'][0]['frames'][0][1]+=1
    elif change=='unit_frame':meta['scene_units'][0]['line_frame_start']=1
    elif change=='missing_unit':meta['scene_sources'][0]['unit_indexes']=[]
    elif change=='text':meta['scene_units'][0]['text']='lost original text'
    elif change=='offset':meta['scene_units'][0]['byte_end']+=1
    elif change=='source_order':meta['scene_sources'][0]['unit_indexes']=[1]
    elif change=='actor':meta['scene_units'][0]['speaker_id']=1
    elif change=='gap':meta['gap_seconds']=float('nan')
    else:meta['scene_units'][0]['source_take_id']='f'*32
    take=replace(original,metadata=meta)
    for operation in (lambda:validate_session(replace(result,takes=(take,))),lambda:engine.decode_take(take),
            lambda:history_view((take,)),lambda:fingerprint(take),lambda:engine.infill(take,'x',0,.1)):
        with pytest.raises(ValueError):operation()
    with pytest.raises(ValueError):save_session(replace(result,takes=(take,)),tmp_path/'corrupt.safetensors')
    assert not (tmp_path/'corrupt.safetensors').exists()


def test_default_disabled_keeps_legacy_scene_and_legacy_revision_default(monkeypatch):
    engine=Harness();result=engine.scene(['A: short','B: yes'],references=references())
    assert 'scene_chunking' not in result.takes[-1].metadata and len(engine.requests)==1
    captured={}
    monkeypatch.setattr(engine,'scene',lambda *a,**kw:captured.update(kw) or result)
    engine.revise(result,'again')
    assert captured['split_long_text'] is False and captured['chunk_bytes']==300


def test_node_appends_controls_and_decode_uses_source_map_not_raw_fast_path():
    from drama3_t8.nodes import T8DramaScene,T8DramaDecode
    info=T8DramaScene.GET_NODE_INFO_V1()
    assert list(info['input']['optional'])==[
        'references','seed_text','generation_mode',
        'split_long_text','chunk_bytes','temperature','top_p','top_k',
        'opening_retries','sampling_preset','session']
    assert info['input']['optional']['session'][0]=='DRAMA3_T8_SESSION'
    engine=Harness()
    session=T8DramaScene.execute(engine,'\n'.join(SCRIPT),references=references(),split_long_text=True,chunk_bytes=100)[0]
    output=T8DramaDecode.execute(engine,session)[0]
    assert len(engine.decoded)==session.takes[-1].metadata['request_count']
    assert output['waveform'].shape[-1]==session.takes[-1].codes.shape[-1]*2048+round(.3*44100)*2


def test_infill_routes_preexisting_isolated_scene_to_visible_pcm_implementation(monkeypatch):
    from drama3_t8 import render_infill
    take=Take('a','a',audio_frames(torch.ones((10,3),dtype=torch.long)),metadata={'decode_sources':[{'line_indexes':[0]}]})
    engine=Harness();observed={}
    def infill(*args,**kwargs):observed.update(args=args,kwargs=kwargs);return take
    monkeypatch.setattr(render_infill,'infill_take',infill)
    assert engine.infill(take,'new',0,.1) is take
    assert observed['args'][:5]==(engine,take,'new',0,.1) and not engine.requests


@pytest.mark.parametrize('middle',['second','很长的第二句。'*10])
def test_previous_other_context_is_immediate_original_line_and_frozen_for_continuations(middle):
    engine=Harness()
    result=engine.scene(['A: first','B: '+middle,'B: third'],references=references(),
        generation_mode='per_line_context',split_long_text=True,chunk_bytes=80)
    metadata=result.takes[-1].metadata
    for raw,request in zip(metadata['scene_sources'],engine.requests):
        line=metadata['scene_units'][raw['unit_indexes'][0]]['line_index']
        prompt=request['prompt'];audio=prompt[(prompt[:,0]>=248099)&(prompt[:,0]<=252194),1]
        if line==1:assert 10 in audio.tolist()  # A's actual previous source.
        elif line==2:assert 10 not in audio.tolist()  # B/B cannot reuse stale A.


def test_all_prompt_consumers_use_matching_actual_sources_with_global_roles(monkeypatch):
    from drama3_t8 import prompts
    engine,_,result=make('per_speaker');take=result.takes[-1]
    sources=source_history(take)
    assert sources[-1].prompt_text.startswith('<|speaker:1|>')
    observed=[];original=prompts.rows_from_segments
    def rows(tokenizer,config,segments):
        observed.append(segments)
        return original(tokenizer,config,segments)
    monkeypatch.setattr(prompts,'rows_from_segments',rows)
    engine.speak('next ordinary speech',result,Sampling())
    engine.stream_text('stream continuation','next',result,Sampling())
    engine.scene(['A: next scene'],session=result,references=result.references)
    engine.long_form('Long text now. More text next. '*4,result,Sampling(),chunk_bytes=40)
    expected=(sources[0].frames,sources[-1].frames)
    assert len(observed) > 4  # ordinary/stream/Scene + first Long + later Long
    for segments in observed[:4]:
        history=[item.values for item in segments if isinstance(item,prompts.FrameRows)]
        assert history and not any(torch.equal(frames,take.frames) for frames in history)
    for segments in observed[:4]:
        history=[item.values for item in segments if isinstance(item,prompts.FrameRows)]
        assert len(history)==2 and all(torch.equal(a,b) for a,b in zip(history,expected))
        assert sources[-1].prompt_text in segments

    for segments in observed[4:]:
        assert not any(isinstance(item,prompts.FrameRows) for item in segments)
        refs=[item for item in segments if torch.is_tensor(item)]
        assert len(refs)==len(result.references)
        assert all(item is ref.codes for item,ref in zip(refs,result.references))
        strings=[item for item in segments if isinstance(item,str)]
        assert strings.count('<|im_start|>user\n')==1
        assert strings.count(prompts.GENERATION_START)==1
        for index,ref in enumerate(result.references,1):
            assert f'\n\n[Reference {index}]\nText: {ref.transcript}\nSpeech: ' in strings


def test_default_editor_pcm_fingerprint_subtitles_and_saved_postproduction_on_chunked_sources(tmp_path):
    from drama3_t8 import editor
    engine,_,result=make('per_speaker');take=result.takes[-1]
    raw_decode=engine.decode
    def bounded_decode(codes):
        audio=raw_decode(codes)
        return {**audio,'waveform':audio['waveform']/1000000}
    engine.decode=bounded_decode  # Editor input is finite audio, not the >80dB integer marker fixture.
    audio,analysis,edits=editor.prepare(engine,take)
    planned=editor.plan(analysis,edits)
    assert len(analysis['lines'])==3
    assert torch.equal(editor.render(audio['waveform'][0],analysis,planned['pieces'],edits),audio['waveform'][0])
    assert len(editor.captions(analysis,planned['pieces'],edits))==3
    assert editor.source_lines(take,analysis['duration'])==[dict(row,chunk=i) for i,row in enumerate(engine.decode_take(take,with_timeline=True)[1])]
    changed=copy.deepcopy(edits);changed['speakers']={'1':{'mute':True},'0':{'gain_db':-6}}
    changed['inserts']=[[.01,.02]]
    edited=editor.edited_session(result,-1,analysis,changed)
    expected=editor.decode_output(engine,edited.takes[-1])
    path=tmp_path/'edited.safetensors';save_session(edited,path);restored=load_session(path)
    assert torch.equal(editor.decode_output(engine,restored.takes[-1])['waveform'],expected['waveform'])
    assert torch.equal(restored.takes[-1].frames,take.frames)
    assert restored.takes[-1].metadata['scene_sources']==take.metadata['scene_sources']
    assert editor.fingerprint(restored.takes[-1])!=editor.fingerprint(take)
    assert restored.takes[-1].metadata['editor_source_take_id']==take.take_id
    assert restored.takes[-1].metadata['audio_editor']['source']==editor.fingerprint(restored.takes[-1])


def test_completed_live_preview_uses_same_real_sources_and_pcm(tmp_path,monkeypatch):
    import soundfile as sf
    from drama3_t8 import live_audio
    calls=[]
    def codec(codes,dtype):
        calls.append(codes.clone())
        return (codes[:,0].float()/1000+codes[:,0,:1].float()/10000).repeat_interleave(2048,dim=-1)
    monkeypatch.setattr(live_audio,'load_codec',lambda _:SimpleNamespace(model=SimpleNamespace(decode=codec)))
    files=[];worker=live_audio.LiveAudio('codec',tmp_path,chunk_frames=2,on_chunk=files.append)
    engine,_,result=make('per_speaker',on_codes=worker.append)
    worker.finish()
    assert len(calls)==result.takes[-1].metadata['request_count']
    actual=torch.cat([torch.from_numpy(sf.read(path,dtype='float32')[0]) for path in files])
    adapter=SimpleNamespace(decode=lambda codes:{'waveform':codec(codes[None],torch.float32)[:,None],'sample_rate':44100})
    expected,_=decode_scene(adapter,result.takes[-1])
    assert torch.allclose(actual,expected['waveform'][0,0],atol=1/32768)


@pytest.mark.parametrize('corruption',['version','feature','downgrade','source_map'])
def test_archive_feature_boundary_and_actual_load_rejects_bad_source_map(tmp_path,corruption):
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file
    _,_,result=make('per_line');path=tmp_path/'good.safetensors';save_session(result,path)
    with safe_open(str(path),framework='pt') as file:
        tensors={key:file.get_tensor(key) for key in file.keys()}
        metadata=json.loads(file.metadata()['drama3_t8_session'])
    assert metadata['version']==2 and metadata['features']==['scene_chunking_v1']
    if corruption=='version':metadata['version']=3
    elif corruption=='feature':metadata['features']=['unimplemented_source_infill']
    elif corruption=='downgrade':metadata['version']=1;metadata.pop('features')
    else:metadata['takes'][0]['metadata']['scene_units'][0]['source_frame_end']+=1
    bad=tmp_path/'bad.safetensors'
    save_file(tensors,str(bad),metadata={'drama3_t8_session':json.dumps(metadata)})
    with pytest.raises(ValueError):load_session(bad)
    legacy=tmp_path/'legacy.safetensors';save_session(Session(references=references()),legacy)
    with safe_open(str(legacy),framework='pt') as file:
        assert json.loads(file.metadata()['drama3_t8_session'])['version']==1
    assert load_session(legacy).references
