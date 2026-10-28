"""Speaker isolation, complete archival and script-order audio composition."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.engine import DramaEngine,Sampling
from drama3_t8.state import Reference,Session,Take,save_session,load_session
from drama3_t8.dialogue import isolated_scene,decode_isolated,audio_frames,own_history


class RecordingEngine:
    def __init__(self):self.calls=[]
    def make(self,lines,session,sampling):
        self.calls.append((lines,session,sampling))
        rows=[]
        for i,line in enumerate(lines):
            if i:
                separator=torch.zeros(1,11,dtype=torch.long);separator[0,0]=252196;rows.append(separator)
            # Source group + its line position have distinguishable audible codes.
            codes=torch.full((10,2+i),len(self.calls)*10+i,dtype=torch.long)
            rows.append(audio_frames(codes))
        take=Take('\n'.join(lines),'\n'.join(lines),torch.cat(rows),metadata={'elapsed_seconds':.1,'length_limit_reached':False,'seed':sampling.seed})
        return session.append(take)
    def speak(self,text,session,sampling,**kwargs):return self.make([text],session,sampling)
    def scene(self,lines,session=None,sampling=None,**kwargs):return self.make(lines,session,sampling)
    def decode(self,codes):
        wave=codes[:1].float().repeat_interleave(2048,dim=-1)
        return {'waveform':wave[None],'sample_rate':44100}


def session():
    refs=(Reference('<|speaker:0|> one',torch.ones(10,3,dtype=torch.long),'A'),
          Reference('<|speaker:1|> two',torch.full((10,3),2,dtype=torch.long),'B'))
    return Session(refs,direction='comedy')


@pytest.mark.parametrize('mode',['per_line','per_line_context','per_speaker'])
def test_isolated_conditioning_uses_local_voice_ids_and_scopes_original_cast(mode):
    engine=RecordingEngine()
    characters='A is speaker 0, the curious host. B is speaker 1, the deadpan manager.'
    isolated_scene(engine,['A: First.','B: Second.','A: Third.'],'an office',characters,
        None,Sampling(seed=99),None,session(),'keep the original voices',None,mode)
    names=['A','B'] if mode=='per_speaker' else ['A','B','A']
    for (lines,context,_),name in zip(engine.calls,names):
        # The native request and its reference have both been mapped to 0.
        assert all(line.startswith('<|speaker:0|>') for line in lines)
        assert all(ref.transcript.startswith('<|speaker:0|>') for ref in context.references)
        assert context.direction.rstrip().endswith('Voice only <|speaker:0|>.')
        assert f'<|speaker:0|> = {name}; original scene speaker {0 if name=="A" else 1}.' in context.direction
        assert 'Original scene character descriptions:\n'+characters in context.direction
        if mode=='per_line_context' and name=='B':
            assert '<|speaker:1|> = context only; original scene speaker 0.' in context.direction


@pytest.mark.parametrize('mode',['per_line','per_line_context','per_speaker'])
def test_isolated_modes_keep_references_sources_and_restore_script_order(mode,tmp_path):
    engine=RecordingEngine();parent=session();emitted=[]
    script=['A: First.','B: Second.','A: Third.']
    result=isolated_scene(engine,script,'space','A: low voice\nB: bright voice',None,Sampling(seed=99),
        None,parent,'slower',lambda frame:emitted.append(int(frame[0])),mode)
    take=result.takes[-1]
    assert parent.takes==() and result.references==parent.references
    assert take.metadata['scene_speaker_map']=={'A':0,'B':1}
    assert len(take.segments())==3
    for _,context,_ in engine.calls:
        assert len(context.references)==1
        assert context.references[0].transcript.startswith('<|speaker:0|>')
    if mode=='per_speaker':
        assert [len(lines) for lines,_,_ in engine.calls]==[2,1]
        assert [int(c[0,0]) for c in take.segments()]==[10,20,11]
    else:
        assert len(engine.calls)==3
        assert len(engine.calls[2][1].takes)==(2 if mode=='per_line_context' else 1)
    if mode=='per_line_context':
        assert engine.calls[1][1].takes[-1].prompt_text.startswith('<|speaker:1|>')
    file=tmp_path/'scene.safetensors';save_session(result,file);restored=load_session(file)
    assert torch.equal(restored.takes[-1].frames,take.frames)
    history=own_history(restored)
    assert len(history[0])==2 if mode!='per_speaker' else len(history[0])==1
    assert 1 in history
    audio,timeline=decode_isolated(engine,restored.takes[-1])
    assert [line['text'] for line in timeline]==['First.','Second.','Third.']
    assert [line['speaker'] for line in timeline]==[0,1,0]
    for line,codes in zip(timeline,take.segments()):
        first=round(line['start']*44100)
        assert audio['waveform'][0,0,first]==codes[0,0]
    assert audio['waveform'].shape[-1]==sum(c.shape[-1] for c in take.segments())*2048+2*round(.3*44100)
    assert emitted==[int(frame[0]) for codes in take.segments() for frame in codes.T]


def test_scene_revision_retains_strategy_and_archive(monkeypatch):
    engine=DramaEngine('model','codec');observed={}
    take=Take('A: hi','hi',audio_frames(torch.ones(10,3,dtype=torch.long)),metadata={
        'task':'scene','generation_mode':'per_line','script_lines':['A: hi'],'scene':'room','characters':'A: calm'})
    parent=Session(takes=(take,))
    def scene(*args,**kwargs):observed.update(kwargs);return parent
    monkeypatch.setattr(engine,'scene',scene)
    assert engine.revise(parent,'laughing') is parent
    assert observed['generation_mode']=='per_line' and observed['note']=='laughing'
    assert observed['session'].takes[0] is take


def test_isolated_scene_preserves_actual_cpu_work_and_fallback_diagnostics(tmp_path):
    class MeasuredEngine(RecordingEngine):
        def make(self,*args):
            result=super().make(*args)
            ordinal=len(self.calls)
            source=result.takes[-1]
            measured=replace(source,metadata={**source.metadata,
                'moe_backend':'cuda','moe_requested_backend':'auto',
                'moe_fallback':'one bank used Torch' if ordinal==2 else None,
                'expert_placement':'vram','expert_requested_placement':'vram',
                'expert_backend_calls':{'cuda':ordinal*7},'cpu_decode_calls':ordinal*2,
                'cpu_experts_run':ordinal*3,'gdn_calls_by_backend':{'torch':ordinal*4},
                'cuda_graph':{'captures':40,'replays':ordinal*20,'cpu_decode':True,
                    'fallback_reasons':['operation rejected capture'] if ordinal==2 else []}})
            return replace(result,takes=(*result.takes[:-1],measured))
    engine=MeasuredEngine()
    result=isolated_scene(engine,['A: First.','B: Second.'],'','',None,Sampling(),
        None,session(),'',None,'per_line')
    saved=tmp_path/'measured.safetensors';save_session(result,saved)
    metadata=load_session(saved).takes[-1].metadata
    assert metadata['cpu_experts_run']==9 and metadata['gdn_calls']==12
    assert metadata['expert_placement']==metadata['expert_requested_placement']=='vram'
    assert metadata['expert_backend_calls']=={'cuda':21} and metadata['cpu_decode_calls']==6
    assert metadata['moe_requested_backend']=='auto' and metadata['moe_fallback']=='one bank used Torch'
    assert metadata['cuda_graph']['captures']==80 and metadata['cuda_graph']['replays']==60
    assert metadata['cuda_graph']['cpu_decode'] is True
    assert metadata['cuda_graph']['fallback_reasons']==['operation rejected capture']


def test_incomplete_isolated_group_fails_before_publishing_audio():
    engine=RecordingEngine();original=engine.make
    def truncated(*args):
        result=original(*args);result.takes[-1].metadata['length_limit_reached']=True
        return result
    engine.make=truncated
    with pytest.raises(RuntimeError,match='did not complete'):
        isolated_scene(engine,['A: One.'],'','',None,Sampling(),None,session(),'',
            lambda frame:pytest.fail('Failed scene audio was published'),'per_line')


def test_isolated_live_preview_decodes_original_sources_then_reorders_with_actual_gaps(tmp_path,monkeypatch):
    import soundfile as sf
    import drama3_t8.live_audio as live_module
    calls=[]
    def decode(codes,dtype):
        calls.append(codes.shape[-1])
        # The first code changes the whole source, so a concatenated stream
        # would audibly differ from independently decoded A and B sources.
        return (codes[:,0].float()/100+codes[:,0,:1].float()/1000).repeat_interleave(2048,dim=-1)
    monkeypatch.setattr(live_module,'load_codec',lambda _:SimpleNamespace(model=SimpleNamespace(decode=decode)))
    files=[]
    live=live_module.LiveAudio('codec',tmp_path,chunk_frames=2,on_chunk=files.append)
    engine=RecordingEngine()
    result=isolated_scene(engine,['A: First.','B: Second.','A: Third.'],'','',None,
        Sampling(),None,session(),'',live.append,'per_speaker')
    live.finish()
    wave=torch.cat([torch.from_numpy(sf.read(file,dtype='float32')[0]) for file in files])
    assert calls==[5,2]
    decoder=SimpleNamespace(decode=lambda codes:{'waveform':decode(codes[None],torch.float32)[:,None],'sample_rate':44100})
    expected,_=decode_isolated(decoder,result.takes[-1])
    assert torch.allclose(wave,expected['waveform'][0,0],atol=1/32768)
    assert len(wave)/44100==pytest.approx(result.takes[-1].metadata['duration_seconds'])
    with pytest.raises(RuntimeError,match='finished'):live.append_scene(result.takes[-1])


def test_isolated_script_holds_one_expert_session_and_releases_it_on_group_failure():
    import contextlib
    engine=RecordingEngine();events=[]
    @contextlib.contextmanager
    def hold():
        events.append('acquire')
        try:yield
        finally:events.append('release')
    engine.generation_session=hold
    original=engine.make
    def checked(*args):
        assert events==['acquire']
        if engine.calls:raise RuntimeError('second group failed')
        return original(*args)
    engine.make=checked
    with pytest.raises(RuntimeError,match='second group'):
        isolated_scene(engine,['A: First.','B: Second.'],'','',None,Sampling(),None,session(),' ',None,'per_line')
    assert events==['acquire','release']
