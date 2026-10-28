import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from drama3_t8 import editor
from drama3_t8.state import Session, Take, load_session, save_session


def fixture_audio():
    # Non-round duration and non-zero endpoints expose trimming/fade errors.
    wave=torch.sin(torch.arange(32017)*.017)*.25+.01
    wave[4000:16000]=0
    lines=[dict(start=0,end=2.,speaker=0,text='Hello cat'),
           dict(start=2.,end=wave.numel()/8000,speaker=1,text='Next meeting')]
    return wave,editor.analyze(wave,8000,lines,source='a'*64)


def edit(analysis,**patch):
    return editor.parse_edits(dict(source=analysis['source'],**patch),source=analysis['source'],duration=analysis['duration'])


def test_default_is_sample_exact_including_endpoints_and_line_boundaries():
    wave,a=fixture_audio();edits=edit(a)
    p=editor.plan(a,edits)
    result=editor.render(wave,a,p['pieces'],edits)
    assert torch.equal(wave,result)
    assert p['samples']==wave.numel() and p['duration']==wave.numel()/8000
    assert not any(pause['action']=='shortened' for pause in p['pauses'])


def test_cuts_overlap_and_insert_at_removed_location_preserve_exact_samples():
    wave,a=fixture_audio();edits=edit(a,cuts=[[.25,.75],[.5,1.]],inserts=[[.5,.125],[2.,.2]])
    p=editor.plan(a,edits);result=editor.render(wave,a,p['pieces'],edits)
    expected=torch.cat((wave[:2000],torch.zeros(1000),wave[8000:16000],torch.zeros(1600),wave[16000:]))
    assert torch.equal(result,expected)


def test_stereo_gain_mute_and_normalization():
    wave,a=fixture_audio();stereo=torch.stack([wave,wave*.5]);edits=edit(a,speakers={'0':{'gain_db':6},'1':{'mute':True}},normalize=True)
    p=editor.plan(a,edits);output=editor.render(stereo,a,p['pieces'],edits)
    assert output.shape==stereo.shape
    assert not output[:,16000:].count_nonzero()
    assert output.abs().max().item()==pytest.approx(10**(-1/20),abs=1e-6)
    assert torch.allclose(output[1],output[0]*.5)


@pytest.mark.parametrize('patch',[
    {'source':'b'*64},{'source':123},{'cuts':[[1,float('nan')]]},{'cuts':[[1,.5]]},
    {'cuts':[[0,5]]},{'inserts':[[-1,.2]]},{'inserts':[[1,61]]},{'trim_start':1},
    {'normalize':'false'},{'threshold_db':float('inf')},{'gap_min':5,'gap_max':1},
    {'pause_to':2,'pause_longer':1},{'speakers':{'16':{}}},{'speakers':{'0':{'gain_db':25}}},
    {'speakers':{'0':{'mute':1}}},{'speakers':{'0':{'name':'x'*81}}},{'invalid':True}])
def test_invalid_edits_rejected(patch):
    _,a=fixture_audio()
    with pytest.raises(ValueError):
        editor.plan(a,{'source':a['source'],**patch})


@pytest.mark.parametrize('change',[
    lambda a:a.update(source=123),lambda a:a.update(duration=float('nan')),
    lambda a:a['envelope'].append(0),lambda a:a['envelope'].__setitem__(0,float('inf')),
    lambda a:a['lines'][1].update(start=0),lambda a:a['lines'][0].update(speaker=17),
    lambda a:a['lines'][0].update(tokens=[dict(text='cat',start=1,end=.5)]),
    lambda a:a.update(sample_rate=True),lambda a:a.update(samples=12)])
def test_invalid_analysis_rejected(change):
    _,a=fixture_audio();change(a)
    with pytest.raises(ValueError):editor.plan(a,{})


def test_timed_captions_follow_cuts_and_silence_without_muted_voice():
    wave,a=fixture_audio()
    a['aligned']=True
    a['lines'][0]['tokens']=[dict(text='Hello',spoken=True,start=.1,end=.4),dict(text='cat',spoken=True,start=1.1,end=1.5)]
    a['lines'][1]['tokens']=[dict(text='meeting',spoken=True,start=2.1,end=2.8)]
    edits=edit(a,cuts=[[0,1]],inserts=[[1,.5]],speakers={'1':{'mute':True}})
    p=editor.plan(a,edits);captions=editor.captions(a,p['pieces'],edits)
    assert [w['text'] for c in captions for w in c['words']]==['cat']
    word=captions[0]['words'][0]
    assert word['start']==pytest.approx(.6) and word['end']==pytest.approx(1.)
    assert '00:00:00,600 --> 00:00:01,000' in editor.subtitle_srt(captions)


def test_line_caption_excludes_the_isolated_scene_gap():
    wave=torch.ones(40000)*.1
    a=editor.analyze(wave,8000,[dict(start=0,end=2,speaker=0,text='first'),
        dict(start=2.3,end=5,speaker=1,text='second')],source='a'*64)
    edits=edit(a);p=editor.plan(a,edits);subs=editor.captions(a,p['pieces'],edits)
    assert subs[0]['end']==2
    assert subs[1]['start']==2.3


def test_natural_pause_compression_keeps_untouched_speech():
    wave,a=fixture_audio();edits=edit(a,pause_longer=.75,pause_to=.25,gap_max=.25)
    p=editor.plan(a,edits);output=editor.render(wave,a,p['pieces'],edits)
    assert output.numel()==wave.numel()-10000
    assert torch.equal(output[:4000],wave[:4000])
    assert any(p['action']=='shortened' for p in p['pauses'])


def test_removing_all_audio_requires_explicit_silence_or_kept_samples():
    _,a=fixture_audio()
    with pytest.raises(ValueError,match='remove all audio'):
        editor.plan(a,edit(a,cuts=[[0,a['duration']]]))
    p=editor.plan(a,edit(a,cuts=[[0,a['duration']]],inserts=[[0,.1]]))
    assert p['samples']==800


def test_nonobject_empty_edits_are_rejected():
    for value in ([],False,0,'[]'):
        with pytest.raises(ValueError):editor.parse_edits(value)


def test_cjk_word_captions_do_not_insert_latin_word_spaces():
    _,a=fixture_audio();a['aligned']=True
    a['lines'][0]['tokens']=[dict(text='猫',spoken=True,start=.1,end=.2),dict(text='主',spoken=True,start=.2,end=.3)]
    edits=edit(a);p=editor.plan(a,edits)
    assert editor.captions(a,p['pieces'],edits)[0]['text']=='猫主'


def sample_take():
    frames=torch.zeros((8,11),dtype=torch.long);frames[:,0]=248100;frames[:,1:]=1
    return Take('Hello cat','Hello cat',frames)


def test_session_persistence_source_identity_and_original_history(tmp_path):
    take=sample_take();session=Session(takes=(take,))
    wave,a=fixture_audio();a['source']=editor.fingerprint(take)
    chosen=edit(a,cuts=[[.5,1]],speakers={'0':{'name':'猫主管','gain_db':-3}})
    branch=editor.edited_session(session,0,a,chosen)
    assert branch.session_id!=session.session_id
    assert branch.takes[0].take_id!=take.take_id and torch.equal(branch.takes[0].frames,take.frames)
    assert branch.takes[0].metadata['editor_parent_take_id']==take.take_id
    assert 'audio_editor' not in take.metadata
    path=tmp_path/'project.safetensors';save_session(branch,path);restored=load_session(path)
    rebound=editor.fingerprint(restored.takes[0])
    assert restored.takes[0].metadata['audio_editor']['edits']=={**chosen,'source':rebound}
    assert rebound!=a['source'] and restored.takes[0].metadata['audio_editor']['source']==rebound
    changed=replace(take,frames=take.frames+1)
    with pytest.raises(ValueError,match='no longer matches'):
        editor.edited_session(Session(takes=(changed,)),0,a,chosen)


def test_saved_edits_replay_on_output_but_raw_take_stays_original():
    take=sample_take();session=Session(takes=(take,))
    wave=torch.ones(1,16384)*.2
    audio={'waveform':wave[None],'sample_rate':44100}
    engine=SimpleNamespace(decode_take=lambda t,with_timeline=False:(audio,None) if with_timeline else audio)
    _,a,_=editor.prepare(engine,take)
    edits=edit(a,cuts=[[0,.1]])
    branch=editor.edited_session(session,0,a,edits)
    result=editor.decode_output(engine,branch.takes[0])
    assert result['waveform'].shape[-1]==16384-4410
    assert editor.decode_output(engine,take)['waveform'].shape[-1]==16384


def test_aligner_loaded_once_and_word_boundaries_survive_session_edits(monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import drama3_t8.alignment as alignment
    take=sample_take();audio={'waveform':torch.ones(1,1,16384)*.1,'sample_rate':44100}
    engine=SimpleNamespace(decode_take=lambda t,with_timeline=False:(audio,None) if with_timeline else audio)
    calls=[]
    def align(*args):
        calls.append(args)
        return [dict(text='Hello',start=.01,end=.1),dict(text='cat',start=.11,end=.3)]
    monkeypatch.setattr(alignment,'align_audio',align)
    _,analysis,edits=editor.prepare(engine,take,aligner='custom-model.safetensors')
    branch=editor.edited_session(Session(takes=(take,)),0,analysis,edits)
    _,reopened,_=editor.prepare(engine,branch.takes[0])
    assert len(calls)==1
    assert reopened['aligned'] is True and reopened['lines']==analysis['lines']


def test_empty_word_alignment_is_not_reported_as_aligned(monkeypatch):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import drama3_t8.alignment as alignment
    take=sample_take();audio={'waveform':torch.ones(1,1,16384)*.1,'sample_rate':44100}
    engine=SimpleNamespace(decode_take=lambda t,with_timeline=False:(audio,None) if with_timeline else audio)
    monkeypatch.setattr(alignment,'align_audio',lambda *args:[])
    with pytest.raises(ValueError,match='returned no words'):
        editor.prepare(engine,take,aligner='custom-model.safetensors')
