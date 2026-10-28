"""Scene-node sampling/schema compatibility; no checkpoint/GPU inference."""
import hashlib
import json
from pathlib import Path

import pytest

from drama3_t8 import nodes
from drama3_t8.engine import Sampling
from drama3_t8.state import Session
from tools.make_workflows import widget_names

FIXTURE=Path(__file__).parent/'fixtures'
CONTROLS=['temperature','top_p','top_k','opening_retries','sampling_preset']
MODES=['together','per_line','per_line_context','per_speaker']

class EntryRecorder:
    def __init__(self):self.calls=[];self.result=Session()
    def scene(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        return self.result

def test_real_scene_schema_appends_sampling_without_changing_old_widget_prefix():
    source=json.loads((FIXTURE/'scene-before-sampling-controls-SOURCE.json').read_text(encoding='utf-8'))
    raw=(FIXTURE/'scene-before-sampling-controls.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['fixtures']['scene-before-sampling-controls.py']
    namespace={**vars(nodes),'__name__':'actual_prior_scene_schema'}
    exec(compile(raw,'scene_before_sampling_fixture','exec'),namespace)
    prior=namespace['T8DramaScene'].GET_NODE_INFO_V1()
    actual=nodes.T8DramaScene.GET_NODE_INFO_V1()
    assert widget_names(actual)==[*widget_names(prior),*CONTROLS]
    assert len(widget_names(prior))==10 and len(widget_names(actual))==15
    assert actual['input_order']['optional']==[*prior['input_order']['optional'],*CONTROLS,'session']
    assert actual['output']==prior['output']
    defaults={name:actual['input']['optional'][name][1]['default'] for name in CONTROLS}
    assert defaults==dict(temperature=.7,top_p=.7,top_k=30,opening_retries=0,sampling_preset='custom')

def test_old_positional_scene_call_preserves_default_sampling_and_existing_options():
    engine=EntryRecorder()
    result=nodes.T8DramaScene.execute(engine,'A: first\nB: second','scene','characters',None,
        44,1200,'','per_speaker',True,100)
    args,kwargs=engine.calls[0]
    assert result[0] is engine.result
    assert args[:4]==(['A: first','B: second'],'scene','characters',())
    assert args[4]==Sampling(seed=44,max_frames=1200)
    assert callable(kwargs.get('progress'))
    kwargs={key:value for key,value in kwargs.items() if key!='progress'}
    assert kwargs==dict(generation_mode='per_speaker',split_long_text=True,chunk_bytes=100)

@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('split',[False,True])
def test_scene_execute_forwards_actual_sampling_to_every_mode(mode,split):
    engine=EntryRecorder()
    nodes.T8DramaScene.execute(engine,'A: hello',generation_mode=mode,split_long_text=split,
        temperature=.85,top_p=.95,top_k=0,opening_retries=1,max_frames=111,
        seed=0,seed_text='18446744073709551615')
    args,kwargs=engine.calls[0]
    assert args[4]==Sampling(seed=2**64-1,temperature=.85,top_k=0,top_p=.95,max_frames=111,opening_retries=1)
    assert kwargs['generation_mode']==mode and kwargs['split_long_text'] is split

@pytest.mark.parametrize('mode',MODES)
def test_expressive_preset_reaches_real_engine_prompt_and_chunk_paths(mode):
    # Keep the actual Engine/Scene/prompt/history methods. Only model inference
    # is a deterministic tiny frame fixture shared with the source-map tests.
    from test_scene_chunks import Harness,references
    engine=Harness()
    result=nodes.T8DramaScene.execute(engine,'A: first sentence\nB: second sentence',references=references(),
        generation_mode=mode,split_long_text=True,chunk_bytes=48,seed=0,
        seed_text='18446744073709551615',sampling_preset='expressive',opening_retries=1,max_frames=50)[0]
    assert result.takes[-1].metadata['generation_mode']==mode
    assert engine.requests
    for index,request in enumerate(engine.requests):
        sampling=request['sampling']
        assert (sampling.temperature,sampling.top_p,sampling.top_k,sampling.opening_retries,sampling.max_frames)==(1.,.9,0,1,50)
        assert sampling.seed==(2**64-1+index)%2**64

@pytest.mark.parametrize('preset,expected',[('stable',(.7,.7,30)),('expressive',(1.,.9,0))])
def test_scene_presets_match_generate_and_preserve_seed_budget_and_retry(preset,expected):
    engine=EntryRecorder()
    nodes.T8DramaScene.execute(engine,'A: text',temperature=1.3,top_p=.8,top_k=7,
        sampling_preset=preset,opening_retries=1,seed=52,max_frames=123)
    sampling=engine.calls[0][0][4]
    assert (sampling.temperature,sampling.top_p,sampling.top_k)==expected
    assert (sampling.seed,sampling.max_frames,sampling.opening_retries)==(52,123,1)

@pytest.mark.parametrize('options',[
    {'temperature':0},{'temperature':float('nan')},{'temperature':float('inf')},
    {'top_p':0},{'top_p':1.1},{'top_p':float('nan')},
    {'top_k':-1},{'top_k':True},{'top_k':1.5},
    {'opening_retries':-1},{'opening_retries':2},{'opening_retries':True},
    {'max_frames':0},{'max_frames':True},{'sampling_preset':'unrecognized'},
    {'seed_text':'18446744073709551616'},{'seed_text':'-1'},
])
def test_invalid_scene_sampling_fails_before_calling_engine(options):
    engine=EntryRecorder()
    with pytest.raises((ValueError,TypeError)):
        nodes.T8DramaScene.execute(engine,'A: text',**options)
    assert engine.calls==[]
