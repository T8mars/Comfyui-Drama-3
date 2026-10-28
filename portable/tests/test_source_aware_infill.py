"""Actual FIM/persistence/Editor contracts with deterministic Codec probes."""
import contextlib
import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from drama3_t8.dialogue import audio_frames
from drama3_t8.engine import Sampling
from drama3_t8.state import Take, Session, save_session, load_session, validate_session
from drama3_t8 import render_infill as ri


class Tokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[100+ord(char)%100 for char in text])


class Engine:
    def __init__(self):
        self.original = torch.linspace(-.45, .45, 24000)[None]
        self.encoded = []; self.decoded = []; self.requests = []; self.budgets = []
        self.model_loads = 0
        self.patcher = SimpleNamespace(load_device=torch.device('cpu'))
        self.model = SimpleNamespace(encode=self.encode)
        self.fake = SimpleNamespace(tokenizer=Tokenizer(), config=SimpleNamespace(
            semantic_start_token_id=248099, tower=SimpleNamespace(num_codebooks=10,vocab_size=4096)))
    def load_codec(self, *, memory_required=0):
        self.budgets.append(memory_required)
        return self
    def encode(self, wave):
        self.encoded.append(wave.clone())
        # Same actual input padding rule as Drama3Codec.encode. Silence gives
        # learned-like nonzero tokens; it is never represented by invented0s.
        padded=F.pad(wave,(0,-wave.shape[-1]%2048))
        values=(padded.unfold(-1,2048,2048).mean(-1)*1000).round().long().add(1000).clamp(0,4095)
        return values[:,None].expand(-1,10,-1).contiguous()
    def load_model(self):
        self.model_loads += 1
        return self.fake
    def _take(self,prompt,text,transcript,sampling,**kwargs):
        self.requests.append(dict(prompt=prompt.clone(),text=text,transcript=transcript,kwargs=kwargs))
        codes=torch.full((10,2),250,dtype=torch.long)
        return Take(text,transcript,audio_frames(codes),metadata=dict(seed=sampling.seed,length_limit_reached=False))
    def decode(self,codes):
        self.decoded.append(codes.clone())
        # Different left contexts change emitted middle PCM, catching a bare
        # middle decode instead of actual prefix+middle then prefix cropping.
        wave=(codes[:1].float()/5000 + codes[0,0]/100000).repeat_interleave(2048,-1)
        return dict(waveform=wave[None],sample_rate=44100)
    def decode_take(self,take,*,with_timeline=False):
        if 'render_infill' in take.metadata:
            audio=ri.decode_rendered(take);timeline=ri.render_timeline(take)
        else:
            audio=dict(waveform=self.original[None].clone(),sample_rate=44100)
            timeline=[dict(start=0.,end=5000/44100,speaker=0,text='First',line=0),
                dict(start=7000/44100,end=24000/44100,speaker=1,text='Second',line=1)]
        return (audio,timeline) if with_timeline else audio


def source():
    return Take('A: First\nB: Second','First\nSecond',audio_frames(torch.ones((10,4),dtype=torch.long)),
        metadata=dict(task='scene',script_lines=['A: First','B: Second'],scene_speaker_map={'A':0,'B':1},
            decode_sources=[dict(line_indexes=[0,1])]))


def operation(engine=None,parent=None,start=9000,end=13000,**kwargs):
    engine=engine or Engine();parent=parent or source()
    kwargs.setdefault('playback_subtype','PCM_16')
    return engine,parent,ri.infill_take(engine,parent,'A: First\nB: The complete new second line',
        start/44100,end/44100,Sampling(seed=2**64-1,max_frames=50),**kwargs)


def test_actual_playback_second_line_selection_preserves_integer_pcm_and_true_codec_context():
    engine,parent,result=operation()
    original,intervals,identity=ri.visible_view(engine,parent,'PCM_16')
    output=ri.decode_rendered(result)['waveform'][0]
    policy=result.metadata['render_infill']
    assert policy['target_speaker']==1 and policy['start_sample']==9000 and policy['end_sample']==13000
    assert output.shape[-1]==24000-4000+4096
    assert torch.equal(output[:,:9000],original[:,:9000])
    assert torch.equal(output[:,9000+4096:],original[:,13000:])
    assert torch.equal(engine.decoded[0],torch.cat((result.blobs['prefix_codes'],result.sources[1].codes),-1))
    assert len(engine.encoded)==3 and engine.encoded[-1].shape==output.shape
    assert all(b>0 for b in engine.budgets)
    assert engine.budgets[-1]==ri.encode_memory_budget(output.shape[-1])
    assert policy['rendered_pcm_encoding'] is True
    assert policy['prompt_rows']==len(engine.requests[0]['prompt'])
    # Full new script is FIM input; raw middle is explicitly not a TTS turn.
    assert 'The complete new second line' in engine.requests[0]['text']
    assert result.sources[1].metadata['history_eligible'] is False
    assert result.sources[1].metadata['transcript_scope']=='complete_FIM_target_not_middle'
    assert result.frames.shape[0]==math.ceil(output.shape[-1]/2048)>result.sources[1].frames.shape[0]
    assert all(not row['text'] and row['needs_alignment'] for row in policy['timeline'])


def test_short_prefix_left_padding_keeps_real_cut_boundary_at_frame_end():
    engine,parent,result=operation(start=1000,end=1600)
    visible,_,_=ri.visible_view(engine,parent,'PCM_16')
    left=engine.encoded[0]
    assert left.shape[-1]==2048 and not left[:,:1048].any()
    assert torch.equal(left[:,1048:],visible[:,:1000])
    assert left[0,-1]==visible[0,999]
    context=result.metadata['render_infill']['context']
    assert context['left_pad_samples']==1048 and context['left_real_samples']==1000


def test_real_gap_silence_is_encoded_not_fake_zero_code_duration():
    engine=Engine();engine.original[:,5000:7000]=0
    _,_,result=operation(engine=engine,start=5500,end=6500,target_speaker=1)
    assert not engine.encoded[0][0,-500:].any()
    assert (result.blobs['prefix_codes']!=0).any()
    assert result.metadata['render_infill']['inserted_samples']==4096


@pytest.mark.parametrize('start,end',[(4000,9000),(5500,6500),(5000,5000)])
def test_ambiguous_or_gap_role_requires_explicit_target_before_encoding(start,end):
    engine=Engine()
    with pytest.raises(ValueError,match='explicit target speaker'):
        operation(engine=engine,start=start,end=end)
    assert not engine.encoded and not engine.requests and engine.model_loads==0


@pytest.mark.parametrize('role',[True,-1,16,'1'])
def test_target_role_validation(role):
    with pytest.raises(ValueError,match='Target speaker'):
        operation(target_speaker=role)


def test_encoder_activation_budget_uses_actual_padded_input_length():
    # Native Codec pads all these inputs to one complete2048-sample frame
    # before its hop512 encoder; each therefore has the same activation shape.
    assert ri.encode_memory_budget(1)==ri.encode_memory_budget(2048)
    assert ri.encode_memory_budget(2049)==ri.encode_memory_budget(4096)


def test_explicit_cross_role_target_reaches_actual_fim_prompt_not_only_metadata():
    first=Engine();second=Engine();parent=source()
    _,_,a=operation(engine=first,parent=parent,start=4000,end=9000,target_speaker=0)
    _,_,b=operation(engine=second,parent=parent,start=4000,end=9000,target_speaker=1)
    assert first.requests[0]['text']==second.requests[0]['text']  # Same complete script.
    assert torch.equal(a.blobs['prefix_codes'],b.blobs['prefix_codes'])
    assert torch.equal(a.blobs['suffix_codes'],b.blobs['suffix_codes'])
    assert not torch.equal(first.requests[0]['prompt'],second.requests[0]['prompt'])
    assert a.metadata['render_infill']['target_speaker']==0 and b.metadata['render_infill']['target_speaker']==1


def test_expected_playback_view_hash_refuses_stale_selection_before_encoding():
    engine=Engine();parent=source()
    _,_,view=ri.visible_view(engine,parent,'PCM_16')
    stale={**view,'pcm16_sha256':'0'*64}
    with pytest.raises(ValueError,match='Playback source changed'):
        operation(engine=engine,parent=parent,expected_view=stale)
    assert not engine.encoded
    _,_,result=operation(engine=engine,parent=parent,expected_view=view)
    assert result.metadata['render_infill']['source_view']==view


def test_archive_self_contained_roundtrip_then_second_edit_preserves_all_raw_sources(tmp_path):
    engine,parent,first=operation()
    file=tmp_path/'render.safetensors'
    # Export just the result; all required sources/PCM must travel with it.
    save_session(Session(takes=(first,)),file)
    loaded=load_session(file);restored=loaded.takes[0]
    assert restored.metadata==first.metadata
    assert torch.equal(ri.decode_rendered(restored)['waveform'],ri.decode_rendered(first)['waveform'])
    assert restored.sources[0].take_id==parent.take_id and torch.equal(restored.sources[0].frames,parent.frames)
    before,_,_=ri.visible_view(engine,restored)
    second=ri.infill_take(engine,restored,'A: First\nB: Another complete revised script',
        11000/44100,16000/44100,target_speaker=1)
    after=ri.decode_rendered(second)['waveform'][0]
    assert torch.equal(before[:,:11000],after[:,:11000])
    assert torch.equal(before[:,16000:],after[:,11000+4096:])
    save_session(Session(takes=(second,)),file)
    last=load_session(file).takes[0]
    assert last.sources[0].sources[0].take_id==parent.take_id
    assert torch.equal(ri.decode_rendered(last)['waveform'],after[None])
    assert len(last.sources[0].sources)==2 and last.sources[1].metadata['history_eligible'] is False


def test_valid_integer_source_frames_survive_archive_storage_dtype_conversion(tmp_path,monkeypatch):
    engine=Engine();original=engine._take
    def int32_frames(*args,**kwargs):
        take=original(*args,**kwargs)
        return replace(take,frames=take.frames.to(torch.int32))
    monkeypatch.setattr(engine,'_take',int32_frames)
    _,_,result=operation(engine=engine)
    path=tmp_path/'valid-int32-source.safetensors'
    save_session(Session(takes=(result,)),path)
    loaded=load_session(path).takes[0]
    assert torch.equal(ri.decode_rendered(loaded)['waveform'],ri.decode_rendered(result)['waveform'])


def test_second_scene_infill_reordered_roles_keep_original_stable_speaker_map():
    engine,_,first=operation()
    second=ri.infill_take(engine,first,'B: New first turn\nA: New second turn',
        9000/44100,13000/44100,target_speaker=1)
    policy=second.metadata['render_infill']
    assert policy['scene_speaker_map']=={'A':0,'B':1}
    assert policy['history_text'].startswith('<|speaker:1|> New first turn')
    assert '\n<|speaker:0|> New second turn' in policy['history_text']


def test_saved_editor_cuts_insert_gain_mute_compiled_once_before_splice():
    from drama3_t8 import editor
    engine=Engine();parent=source()
    _,analysis,_=editor.prepare(engine,parent)
    edits=dict(source=analysis['source'],cuts=[[.01,.02]],inserts=[[.1,.03]],
        speakers={'0':dict(gain_db=-6),'1':dict(mute=True)})
    changed=editor.edited_session(Session(takes=(parent,)),0,analysis,edits).takes[0]
    expected,_,_=ri.visible_view(engine,changed,'PCM_16')
    _,_,result=operation(engine=engine,parent=changed,start=2000,end=3000,target_speaker=0)
    actual=ri.decode_rendered(result)['waveform'][0]
    assert torch.equal(actual[:,:2000],expected[:,:2000])
    assert torch.equal(actual[:,2000+4096:],expected[:,3000:])
    assert not result.metadata.get('audio_editor')  # No inherited gain replay.


def test_full_encoded_history_has_full_target_text_and_actual_audio_not_middle_frames():
    engine,_,result=operation()
    history=ri.history_take(result)
    assert history.metadata['history_actual_pcm'] is True
    assert 'complete new second line' in history.prompt_text
    assert torch.equal(history.frames,result.frames)
    assert not history.blobs and not history.sources and 'render_infill' not in history.metadata
    assert not torch.equal(history.frames,result.sources[1].frames)


def test_final_playback_file_preserves_original_pcm16_integer_samples(tmp_path):
    import soundfile as sf
    engine,parent,result=operation()
    original,_,_=ri.visible_view(engine,parent,'PCM_16')
    before=tmp_path/'before.wav';after=tmp_path/'after.wav'
    sf.write(before,original.T.numpy(),44100,subtype='PCM_16')
    sf.write(after,ri.decode_rendered(result)['waveform'][0].T.numpy(),44100,subtype='PCM_16')
    old,_=sf.read(before,dtype='int16');new,_=sf.read(after,dtype='int16')
    assert (new[:9000]==old[:9000]).all()
    assert (new[9000+4096:]==old[13000:]).all()


def test_native_default_float_preserves_overrange_and_signed_zero_bits():
    engine=Engine();engine.original*=10
    engine.original[0,1]=-0.
    parent=source()
    before,_,view=ri.visible_view(engine,parent)
    assert view['subtype']=='FLOAT' and before.abs().max()>1 and 'pcm16_sha256' not in view
    result=ri.infill_take(engine,parent,'A: First\nB: Full new text',9000/44100,13000/44100)
    after=ri.decode_rendered(result)['waveform'][0]
    assert result.metadata['render_infill']['subtype']=='FLOAT'
    assert torch.equal(after[:,:9000].contiguous().view(torch.int32),before[:,:9000].contiguous().view(torch.int32))
    assert torch.equal(after[:,9000+4096:].contiguous().view(torch.int32),before[:,13000:].contiguous().view(torch.int32))
    assert after[0,1].view(torch.int32)==before[0,1].view(torch.int32)  # Preserve negative zero.


def test_float_editor_gain_and_actual_float_wav_reload_then_continuous_edit(tmp_path):
    from drama3_t8 import editor
    import soundfile as sf
    engine=Engine();parent=source()
    _,analysis,_=editor.prepare(engine,parent)
    changed=editor.edited_session(Session(takes=(parent,)),0,analysis,
        dict(source=analysis['source'],speakers={'0':{'gain_db':12},'1':{'gain_db':12}},
            cuts=[[.01,.02]],inserts=[[.1,.03]])).takes[0]
    before,_,identity=ri.visible_view(engine,changed,'FLOAT')
    assert before.abs().max()>1
    first=ri.infill_take(engine,changed,'A: First\nB: Full changed text',
        9000/44100,13000/44100,target_speaker=1,expected_view=identity)
    after=ri.decode_rendered(first)['waveform'][0]
    out=tmp_path/'actual-editor-float.wav'
    sf.write(out,after.T.numpy(),44100,subtype='FLOAT')
    saved,_=sf.read(out,dtype='float32',always_2d=True)
    actual=torch.from_numpy(saved.T.copy())
    assert torch.equal(actual.view(torch.int32),after.view(torch.int32))
    assert torch.equal(actual[:,:9000].contiguous().view(torch.int32),before[:,:9000].contiguous().view(torch.int32))
    assert torch.equal(actual[:,9000+4096:].contiguous().view(torch.int32),before[:,13000:].contiguous().view(torch.int32))
    archive=tmp_path/'float-render.safetensors';save_session(Session(takes=(first,)),archive)
    loaded=load_session(archive).takes[0]
    assert loaded.metadata['render_infill']['subtype']=='FLOAT'
    second=ri.infill_take(engine,loaded,'A: Still first\nB: Another full new text',
        10000/44100,14000/44100,target_speaker=1)
    next_pcm=ri.decode_rendered(second)['waveform'][0]
    assert second.metadata['render_infill']['subtype']=='FLOAT'
    assert torch.equal(next_pcm[:,:10000].contiguous().view(torch.int32),after[:,:10000].contiguous().view(torch.int32))
    assert torch.equal(next_pcm[:,10000+4096:].contiguous().view(torch.int32),after[:,14000:].contiguous().view(torch.int32))
    save_session(Session(takes=(second,)),archive)
    assert torch.equal(ri.decode_rendered(load_session(archive).takes[0])['waveform'].view(torch.int32),next_pcm[None].view(torch.int32))


def _edited_gain(engine,parent):
    from drama3_t8 import editor
    _,analysis,_=editor.prepare(engine,parent)
    return editor.edited_session(Session(takes=(parent,)),0,analysis,
        dict(source=analysis['source'],speakers={'0':{'gain_db':6},'1':{'gain_db':6}})).takes[0]


def test_pcm16_result_followed_by_editor_uses_actual_float_default_and_continues_after_reload(tmp_path):
    engine=Engine();engine.original*=2
    _,_,prior=operation(engine=engine)
    assert prior.metadata['render_infill']['subtype']=='PCM_16'
    edited=_edited_gain(engine,prior)
    expected,_,_=ri.visible_view(engine,edited,'FLOAT')
    assert expected.abs().max()>1
    result=ri.infill_take(engine,edited,'A: First\nB: Full edited replacement',
        9000/44100,13000/44100,target_speaker=1)
    policy=result.metadata['render_infill']
    actual=ri.decode_rendered(result)['waveform'][0]
    assert policy['subtype']=='FLOAT'  # Current Editor writes FLOAT, even over an old16 take.
    assert torch.equal(actual[:,:9000].contiguous().view(torch.int32),expected[:,:9000].contiguous().view(torch.int32))
    assert torch.equal(actual[:,9000+4096:].contiguous().view(torch.int32),expected[:,13000:].contiguous().view(torch.int32))
    path=tmp_path/'pcm16-editor-native.safetensors';save_session(Session(takes=(result,)),path)
    loaded=load_session(path).takes[0]
    again=ri.infill_take(engine,loaded,'A: First\nB: Full second replacement',
        11000/44100,15000/44100,target_speaker=1)
    final=ri.decode_rendered(again)['waveform'][0]
    assert again.metadata['render_infill']['subtype']=='FLOAT'
    assert torch.equal(final[:,:11000].contiguous().view(torch.int32),actual[:,:11000].contiguous().view(torch.int32))
    assert torch.equal(final[:,11000+4096:].contiguous().view(torch.int32),actual[:,15000:].contiguous().view(torch.int32))


def test_editor_can_still_bind_an_explicit_pcm16_asset_over_a_float_view():
    engine=Engine();engine.original*=2
    _,_,prior=operation(engine=engine)
    edited=_edited_gain(engine,prior)
    selected,_,identity=ri.visible_view(engine,edited,'PCM_16')
    result=ri.infill_take(engine,edited,'A: First\nB: Full replacement',9000/44100,13000/44100,
        target_speaker=1,expected_view=identity)
    output=ri.decode_rendered(result)['waveform'][0]
    assert result.metadata['render_infill']['subtype']=='PCM_16'
    assert torch.equal(output[:,:9000],selected[:,:9000])
    assert torch.equal(output[:,9000+4096:],selected[:,13000:])


def test_unedited_render_preserves_its_original_pcm16_default():
    engine,_,prior=operation()
    view,_,identity=ri.visible_view(engine,prior)
    assert not prior.metadata.get('audio_editor') and identity['subtype']=='PCM_16'
    result=ri.infill_take(engine,prior,'A: First\nB: Another full replacement',
        9000/44100,13000/44100,target_speaker=1)
    assert result.metadata['render_infill']['subtype']=='PCM_16'


def test_editor_float_negative_zero_bits_survive_native_default_and_archive(tmp_path):
    engine=Engine();engine.original[0,1]=-0.
    _,_,prior=operation(engine=engine,playback_subtype='FLOAT')
    edited=_edited_gain(engine,prior)
    before,_,_=ri.visible_view(engine,edited,'FLOAT')
    assert before[0,1].view(torch.int32).item()==-(2**31)
    result=ri.infill_take(engine,edited,'A: First\nB: Full replacement',9000/44100,13000/44100,target_speaker=1)
    path=tmp_path/'negative-zero-editor.safetensors';save_session(Session(takes=(result,)),path)
    output=ri.decode_rendered(load_session(path).takes[0])['waveform'][0]
    assert result.metadata['render_infill']['subtype']=='FLOAT'
    assert torch.equal(output[:,:9000].contiguous().view(torch.int32),before[:,:9000].contiguous().view(torch.int32))


@pytest.mark.parametrize('subtype',['DOUBLE','PCM_24',None,True,17])
def test_unknown_playback_contract_rejected_before_codec_encoding(subtype):
    engine=Engine()
    with pytest.raises(ValueError,match='subtype'):
        ri.infill_take(engine,source(),'Full text',.1,.2,target_speaker=0,expected_view={'subtype':subtype})
    assert not engine.encoded and engine.model_loads==0


def test_explicit_subtype_cannot_conflict_with_actual_asset_and_hash_is_format_bound():
    engine=Engine();parent=source()
    wave,_,identity=ri.visible_view(engine,parent,'FLOAT')
    assert ri.playback_identity(wave,'FLOAT')['pcm_sha256']!=ri.playback_identity(wave,'PCM_16')['pcm_sha256']
    with pytest.raises(ValueError,match='conflicts'):
        ri.infill_take(engine,parent,'Full text',.1,.2,target_speaker=0,
            expected_view=identity,playback_subtype='PCM_16')
    assert not engine.encoded


@pytest.mark.parametrize('corruption',['subtype','parent_subtype','float_claims_pcm16'])
def test_wrong_float_archive_subtype_or_false_integer_identity_is_rejected(corruption,tmp_path):
    engine=Engine();engine.original*=10
    _,_,original=operation(engine=engine,playback_subtype='FLOAT')
    meta=copy.deepcopy(original.metadata)
    if corruption=='subtype':meta['render_infill']['subtype']='PCM_16'
    elif corruption=='parent_subtype':meta['render_infill']['source_view']['subtype']='PCM16'
    else:meta['render_infill']['pcm16_sha256']=meta['render_infill']['pcm_sha256']
    changed=replace(original,metadata=meta)
    for action in (lambda:ri.validate_take(changed),lambda:save_session(Session(takes=(changed,)),tmp_path/'bad-format.safetensors')):
        with pytest.raises(ValueError):action()


@pytest.mark.parametrize('corruption',['view_source','middle_parent','middle_role','middle_scope','middle_transcript','middle_prompt'])
def test_independent_review_provenance_and_full_fim_target_contract(corruption,tmp_path):
    _,_,original=operation()
    meta=copy.deepcopy(original.metadata);parent,middle=original.sources
    if corruption=='view_source':meta['render_infill']['source_view']['source']='0'*64
    elif corruption=='middle_transcript':middle=replace(middle,transcript='Wrong complete transcript')
    elif corruption=='middle_prompt':middle=replace(middle,prompt_text='<|speaker:0|> Wrong complete script')
    else:
        changes={'middle_parent':{'parent_take_id':'f'*32},'middle_role':{'target_speaker':7},
            'middle_scope':{'transcript_scope':'guessed_middle_transcript'}}[corruption]
        middle=replace(middle,metadata={**middle.metadata,**changes})
    changed=replace(original,metadata=meta,sources=(parent,middle))
    for action in (lambda:ri.validate_take(changed),lambda:ri.decode_rendered(changed),
            lambda:save_session(Session(takes=(changed,)),tmp_path/'false-provenance.safetensors')):
        with pytest.raises(ValueError):action()


def test_changed_full_target_cannot_keep_an_old_canonical_acoustic_history():
    from drama3_t8.prompts import spoken_text
    _,_,original=operation()
    target='A: Completely different words\nB: Completely different target'
    meta=copy.deepcopy(original.metadata)
    meta['render_infill']['full_target_transcript']=target
    parent,middle=original.sources
    middle=replace(middle,transcript=target)
    changed=replace(original,metadata=meta,transcript=spoken_text(target,scene=True),sources=(parent,middle))
    # All prior pairwise raw/full-transcript comparisons agree, but the stored
    # canonical prompt still contains the old words. Rebuild it from full text.
    with pytest.raises(ValueError):validate_session(Session(takes=(changed,)))


@pytest.mark.parametrize('kind,role,target',[
    ('plain',0,'Meet at 9:30. The complete new utterance.'),
    ('plain',1,'  New whole utterance.\n  More words.  '),
    ('plain',1,'  <|speaker:1|> Explicit whole utterance.  '),
    ('scene',0,'A: Whole first line\nB: Whole second line'),
    ('scene',1,'\n  B: Reordered first line  \n A: Reordered second line\n'),
    ('scene',1,'  <|speaker:1|> Explicit first\n<|speaker:0|> Explicit second  '),
    ('scene',1,'[laughs] B: Full first turn\nA: Full second turn')])
def test_reconstructed_canonical_full_target_accepts_normal_roles_whitespace_and_tags(kind,role,target,tmp_path):
    engine=Engine();parent=source()
    if kind=='plain':parent=replace(parent,metadata={'task':'tts'})
    result=ri.infill_take(engine,parent,target,9000/44100,13000/44100,target_speaker=role)
    policy=result.metadata['render_infill']
    assert policy['full_target_transcript']==target
    assert policy['history_text']==result.prompt_text==result.sources[1].prompt_text
    assert policy['scene_text'] is (kind=='scene')
    if kind=='scene':assert policy['scene_speaker_map']=={'A':0,'B':1}
    else:
        assert policy['scene_speaker_map']=={}
        assert '<|speaker:'+str(role)+'|>' in result.prompt_text
    path=tmp_path/'valid-canonical-target.safetensors'
    save_session(Session(takes=(result,)),path)
    restored=load_session(path).takes[0]
    assert restored.metadata==result.metadata
    assert torch.equal(ri.decode_rendered(restored)['waveform'].view(torch.int32),ri.decode_rendered(result)['waveform'].view(torch.int32))


@pytest.mark.parametrize('where',['policy','root'])
def test_wrong_canonical_scene_cast_map_is_not_accepted_as_matching_full_text(where):
    _,_,original=operation()
    metadata=copy.deepcopy(original.metadata)
    if where=='policy':metadata['render_infill']['scene_speaker_map']={'A':1,'B':0}
    else:metadata['scene_speaker_map']={'A':1,'B':0}
    with pytest.raises(ValueError):validate_session(Session(takes=(replace(original,metadata=metadata),)))


def test_full_script_prompt_limit_rejects_without_truncating_context_or_generating():
    engine=Engine();parent=source();before=parent.frames.clone()
    class LongTokenizer:
        def encode(self,text):return SimpleNamespace(ids=[100]*8200)
    engine.fake.tokenizer=LongTokenizer()
    with pytest.raises(ValueError,match='8192 prompt rows'):
        operation(engine=engine,parent=parent)
    assert not engine.requests and torch.equal(parent.frames,before)
    assert len(engine.encoded)==2  # Real context encoded; nothing silently dropped.


def test_history_prompt_uses_complete_rendered_frames_and_native_context_guard():
    from drama3_t8.prompts import tts_prompt
    engine,_,result=operation();view=ri.history_take(result)
    class LongTokenizer:
        def encode(self,text):return SimpleNamespace(ids=[100]*8200)
    before=view.frames.clone()
    with pytest.raises(ValueError,match='8192 prompt rows'):
        tts_prompt(LongTokenizer(),engine.fake.config,'next',takes=(view,))
    assert torch.equal(view.frames,before)


def test_empty_middle_is_a_precise_deletion_and_zero_length_selection_inserts(monkeypatch):
    engine=Engine()
    def empty(prompt,text,transcript,sampling,**kwargs):
        return Take(text,transcript,torch.empty((0,11),dtype=torch.long),metadata={'length_limit_reached':False})
    monkeypatch.setattr(engine,'_take',empty)
    _,parent,result=operation(engine=engine)
    original,_,_=ri.visible_view(engine,parent,'PCM_16')
    assert result.metadata['render_infill']['inserted_samples']==0
    assert torch.equal(ri.decode_rendered(result)['waveform'][0],torch.cat((original[:,:9000],original[:,13000:]),-1))
    assert not engine.decoded
    engine2,parent2,inserted=operation(start=9000,end=9000)
    assert inserted.metadata['render_infill']['output_samples']==24000+4096


def test_partial_generation_and_final_encoding_failure_never_modify_parent(monkeypatch):
    engine=Engine();parent=source();before=parent.frames.clone();meta=copy.deepcopy(parent.metadata)
    original=engine._take
    def truncated(*args,**kwargs):
        take=original(*args,**kwargs)
        return replace(take,metadata={**take.metadata,'length_limit_reached':True})
    monkeypatch.setattr(engine,'_take',truncated)
    with pytest.raises(RuntimeError,match='max_frames'):
        operation(engine=engine,parent=parent)
    assert not engine.decoded and parent.metadata==meta and torch.equal(parent.frames,before)
    engine2=Engine();encode=engine2.encode
    def fail_final(wave):
        if len(engine2.encoded)==2:raise RuntimeError('Injected complete-history encode failure')
        return encode(wave)
    monkeypatch.setattr(engine2.model,'encode',fail_final)
    with pytest.raises(RuntimeError,match='complete-history encode failure'):
        operation(engine=engine2,parent=parent)
    assert parent.metadata==meta and torch.equal(parent.frames,before)


def test_official_cancel_leaves_parent_unmodified_and_followup_recovers():
    engine=Engine();parent=source();before=parent.frames.clone();old_meta=copy.deepcopy(parent.metadata)
    def progress(done,total):
        if done>=3:mm.interrupt_current_processing(True)
    try:
        with pytest.raises(mm.InterruptProcessingException):operation(engine=engine,parent=parent,progress=progress)
        assert not engine.requests and torch.equal(parent.frames,before) and parent.metadata==old_meta
    finally:mm.interrupt_current_processing(False)
    _,_,recovered=operation(engine=engine,parent=parent)
    ri.validate_take(recovered)


@pytest.mark.parametrize('corruption',['version','pcm','hash','prefix','middle','duration','timeline','source','parent_pcm','context','transcript','history','view_shape','parent_frames','parent_metadata'])
def test_corrupt_render_archive_rejected(corruption,tmp_path):
    _,_,original=operation()
    meta=copy.deepcopy(original.metadata);blobs={k:v.clone() for k,v in original.blobs.items()}
    policy=meta['render_infill'];sources=original.sources
    if corruption=='version':policy['version']=2
    elif corruption=='pcm':blobs['render_pcm'][0,0]=float('nan')
    elif corruption=='hash':policy['pcm16_sha256']='0'*64
    elif corruption=='prefix':blobs['prefix_codes'][0,0]=9999
    elif corruption=='middle':sources=(sources[0],replace(sources[1],metadata={}))
    elif corruption=='duration':policy['inserted_samples']+=1
    elif corruption=='timeline':policy['timeline'][0]['end']=float('nan')
    elif corruption=='source':policy['source_take_id']='f'*32
    elif corruption=='parent_pcm':blobs['parent_pcm'][0,0]=.123
    elif corruption=='context':policy['context']['left_pad_samples']+=1
    elif corruption=='transcript':policy['full_target_transcript']='different full script'
    elif corruption=='history':
        frames=original.frames.clone();frames[0,1]+=1;original=replace(original,frames=frames)
    elif corruption=='view_shape':policy['source_view']=None
    elif corruption=='parent_frames':
        frames=sources[0].frames.clone();frames[0,1]+=1;sources=(replace(sources[0],frames=frames),sources[1])
    else:sources=(replace(sources[0],metadata={**sources[0].metadata,'characters':'changed provenance'}),sources[1])
    changed=replace(original,metadata=meta,blobs=blobs,sources=sources)
    for action in (lambda:ri.validate_take(changed),lambda:ri.decode_rendered(changed),
            lambda:ri.history_take(changed),lambda:validate_session(Session(takes=(changed,))),
            lambda:save_session(Session(takes=(changed,)),tmp_path/'bad.safetensors')):
        with pytest.raises(ValueError):action()
    assert not (tmp_path/'bad.safetensors').exists()
