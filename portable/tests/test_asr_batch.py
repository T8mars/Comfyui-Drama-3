"""ASR input/forwarding contracts on CPU; no model quality or UI claim."""
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()


@pytest.mark.parametrize('shape,expected', [((32,), (1,32)), ((2,32),(2,32)), ((1,2,32),(2,32))])
def test_single_clip_shapes_remain_single_asr_call(shape, expected):
    from drama3_t8.audio import asr_waveforms
    wave = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    before = wave.clone()
    clips = asr_waveforms(wave, 16000)
    assert len(clips) == 1 and clips[0].shape == expected
    assert clips[0].dtype == torch.float32 and torch.equal(clips[0], wave.reshape(expected).float())
    assert torch.equal(wave, before) and wave.dtype == torch.float64


def test_standard_batch_keeps_channels_samples_order_and_original_waveform():
    from drama3_t8.audio import asr_waveforms, audio_waveform
    wave = torch.arange(3*2*32).reshape(3,2,32)
    before = wave.clone()
    clips = asr_waveforms(wave, 22050)
    assert len(clips) == 3
    for index, clip in enumerate(clips):
        assert clip.shape == (2,32) and torch.equal(clip, wave[index].float())
    assert torch.equal(wave, before)
    # Reference/Codec/alignment still call this strict single-clip helper.
    with pytest.raises(ValueError, match='batch'):
        audio_waveform(wave, 22050)


@pytest.mark.parametrize('wave', [torch.zeros(0,1,8), torch.zeros(2,0,8), torch.zeros(2,1,0),
    torch.zeros(2,1,1,8), torch.ones(2,1,8,dtype=torch.complex64),
    torch.tensor([[[0.]], [[float('nan')]]]), torch.tensor([[[0.]], [[float('inf')]]])])
def test_invalid_batch_is_rejected_before_model_loading(monkeypatch, wave):
    from drama3_t8 import nodes
    from drama3_t8.qwen3_asr import asr
    monkeypatch.setattr(asr,'load_qwen3_asr',lambda *a:pytest.fail('Invalid batch reached model loading'))
    with pytest.raises(ValueError):
        nodes.T8DramaTranscribe.execute('unused',{'waveform':wave,'sample_rate':16000},model_path='unused.safetensors')


@pytest.mark.parametrize('rate', [True,False,0,-1,16000.5,float('nan'),float('inf'),'16000'])
def test_batch_sample_rate_keeps_existing_strict_contract(rate):
    from drama3_t8.audio import asr_waveforms
    with pytest.raises(ValueError,match='sample rate'):
        asr_waveforms(torch.zeros(2,1,8),rate)


def test_asr_node_loads_once_and_forwards_every_actual_clip_and_budget(monkeypatch, tmp_path):
    from drama3_t8 import nodes
    from drama3_t8.qwen3_asr import asr
    model = object(); loaded=[]; called=[]
    path=tmp_path/'qwen.safetensors'
    monkeypatch.setattr(asr,'load_qwen3_asr',lambda value:loaded.append(value) or model)
    outputs=iter([('first','English,Chinese'),('','Chinese,English'),('third','Japanese')])
    def forward(actual_model,waveform,sample_rate,language=None,context='',max_new_tokens=4096):
        assert actual_model is model
        called.append((waveform.clone(),sample_rate,language,context,max_new_tokens))
        return next(outputs)
    monkeypatch.setattr(asr,'transcribe',forward)
    wave=torch.arange(3*2*16,dtype=torch.float32).reshape(3,2,16)
    before=wave.clone()
    result=nodes.T8DramaTranscribe.execute('unused',{'waveform':wave,'sample_rate':22050},
        'auto','names only',str(path),max_new_tokens=32768)
    assert result.result == ('first\n\nthird','English,Chinese,Japanese')
    assert loaded == [str(path)] and len(called)==3
    for index,(clip,rate,language,context,budget) in enumerate(called):
        assert torch.equal(clip,wave[index])
        assert (rate,language,context,budget)==(22050,None,'names only',32768)
    assert torch.equal(wave,before)


@pytest.mark.parametrize('shape', [(64,),(2,64),(1,2,64)])
def test_single_node_result_and_default_budget_are_unchanged(monkeypatch, shape):
    from drama3_t8 import nodes
    from drama3_t8.qwen3_asr import asr
    monkeypatch.setattr(asr,'load_qwen3_asr',lambda _:object())
    calls=[]
    def forward(model,waveform,sample_rate,language=None,context='',max_new_tokens=4096):
        calls.append((tuple(waveform.shape),language,context,max_new_tokens))
        return ' original\ntext ','English,Chinese,English'
    monkeypatch.setattr(asr,'transcribe',forward)
    result=nodes.T8DramaTranscribe.execute('unused',{'waveform':torch.zeros(shape),'sample_rate':16000},
        'English','context',model_path='custom.safetensors')
    assert result.result == (' original\ntext ','English,Chinese,English')
    assert calls == [((1,64) if len(shape)==1 else (2,64),'English','context',4096)]


INVALID_BUDGETS=[True,False,None,0,-1,32769,1.5,4096.0,'4096',torch.tensor(4096)]


@pytest.mark.parametrize('budget', INVALID_BUDGETS)
def test_invalid_node_budget_fails_before_loading(monkeypatch, budget):
    from drama3_t8 import nodes
    from drama3_t8.qwen3_asr import asr
    monkeypatch.setattr(asr,'load_qwen3_asr',lambda *a:pytest.fail('Invalid budget reached model loading'))
    with pytest.raises(ValueError,match='max_new_tokens'):
        nodes.T8DramaTranscribe.execute('unused',{'waveform':torch.zeros(1,1,8),'sample_rate':16000},
            model_path='unused.safetensors',max_new_tokens=budget)


@pytest.mark.parametrize('budget', INVALID_BUDGETS)
def test_invalid_native_budget_fails_before_accessing_model(budget):
    from drama3_t8.qwen3_asr import asr
    with pytest.raises(ValueError,match='max_new_tokens'):
        asr.transcribe(None,torch.zeros(1,8),16000,max_new_tokens=budget)


@pytest.mark.parametrize('budget',[1,4096,32768])
def test_native_decoder_receives_explicit_integer_budget_on_cpu(monkeypatch,budget):
    from drama3_t8.qwen3_asr import asr
    calls=[]
    decoder=SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids,**kwargs:torch.zeros(len(ids),4)),
        generate=lambda **kwargs:calls.append(kwargs) or [asr.STOP_TOKENS[0]])
    model=SimpleNamespace(patcher=SimpleNamespace(load_device=torch.device('cpu')),
        compute_dtype=torch.float32,model=SimpleNamespace(decoder=decoder,audio_tower=lambda _:torch.zeros(1,4)),decode=lambda _: '')
    monkeypatch.setattr(asr,'log_mel',lambda _:torch.zeros(128,50))
    monkeypatch.setattr(asr,'prompt_ids',lambda *args:[asr.AUDIO_PAD])
    monkeypatch.setattr(asr.comfy.model_management,'load_models_gpu',lambda *args:None)
    monkeypatch.setattr(asr.comfy.model_management,'cuda_device_context',lambda *args:contextlib.nullcontext())
    asr.transcribe(model,torch.zeros(1,8000),16000,max_new_tokens=budget)
    assert len(calls)==1 and calls[0]['max_length']==budget and type(calls[0]['max_length']) is int


def test_real_schema_preserves_published_widget_prefix_and_appends_budget():
    from drama3_t8 import nodes
    from tools.make_workflows import widget_names
    fixed=Path(__file__).parent/'fixtures/asr-node-v0.1.11'
    source=json.loads((fixed/'SOURCE.json').read_text(encoding='utf-8'))
    assert source['commit']=='f838deaee67f2bce467ce6a76e9b3dc3de615a86'
    raw=(fixed/'transcribe.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['fixtures']['transcribe.py']
    # Define the exact published class against this real host API. Only its
    # metadata method runs; neither execute nor any model loader is invoked.
    namespace={**vars(nodes),'__name__':'published_asr_schema_fixture'}
    exec(compile(raw,str(fixed/'transcribe.py'),'exec'),namespace)
    published=namespace['T8DramaTranscribe'].GET_NODE_INFO_V1()
    actual=nodes.T8DramaTranscribe.GET_NODE_INFO_V1()
    assert widget_names(actual) == [*widget_names(published),'max_new_tokens']
    assert actual['input_order']['optional'] == ['model_path','max_new_tokens']
    options=actual['input']['optional']['max_new_tokens'][1]
    assert (options['default'],options['min'],options['max'])==(4096,1,32768)
    assert actual['output']==published['output']
