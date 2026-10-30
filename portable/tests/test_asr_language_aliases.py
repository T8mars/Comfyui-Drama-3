"""Real proposal entrypoints on CPU fixtures; never loads native/model weights."""
import contextlib
import threading
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest
import torch

from drama3_t8.asr_language import normalize_asr_language
from drama3_t8.r2t2 import native
import drama3_t8.r2t2.transcribe as r2
from drama3_t8.qwen3_asr import asr

ALIASES={"zh":"Chinese","en":"English","yue":"Cantonese","ja":"Japanese","ko":"Korean","de":"German","fr":"French","ru":"Russian","pt":"Portuguese","es":"Spanish","it":"Italian","中文":"Chinese","英语":"English","粤语":"Cantonese"}

@pytest.mark.parametrize("source,expected",ALIASES.items())
@pytest.mark.parametrize("backend,supported",[("Confucius",native.SUPPORTED_LANGUAGES),("Qwen3-ASR",asr.LANGUAGES)])
def test_aliases_are_real_backend_canonical_names(source,expected,backend,supported):
    assert normalize_asr_language("  "+source.upper()+"  ",supported,backend=backend)==expected

@pytest.mark.parametrize("value",[None,"","   ","auto","Auto","AUTO","  aUtO \n"])
def test_auto_none_empty_are_explicit_auto_without_token_text(value):
    assert normalize_asr_language(value,native.SUPPORTED_LANGUAGES) is None
    assert "language " not in native.build_prompt(language=value)

@pytest.mark.parametrize("canonical",asr.LANGUAGES)
def test_qwen_canonical_casefold_trim_preserves_all_existing_30_languages(canonical):
    assert normalize_asr_language(" "+canonical.swapcase()+" ",asr.LANGUAGES)==canonical

@pytest.mark.parametrize("bad",["Swedish","Arabic","sv","None","not-a-language","zh-UNKNOWN",1,True,[]])
def test_r2_never_silently_falls_back_or_expands_11_languages(bad):
    with pytest.raises(ValueError):normalize_asr_language(bad,native.SUPPORTED_LANGUAGES,backend="Confucius")
    assert len(native.SUPPORTED_LANGUAGES)==11 and len(asr.LANGUAGES)==30

def test_actual_backend_whitelist_is_used():
    assert normalize_asr_language(" arabic ",asr.LANGUAGES)=="Arabic"
    with pytest.raises(ValueError,match="Confucius.*Arabic"):normalize_asr_language("Arabic",native.SUPPORTED_LANGUAGES,backend="Confucius")

@pytest.fixture
def fake_r2(tmp_path,monkeypatch):
    calls=[]
    class Engine:
        def __init__(self,*a,**k):calls.append("construct")
        def __enter__(self):return self
        def __exit__(self,*a):calls.append("close")
        def transcribe(self,pcm,**kw):calls.append(("transcribe",kw));return {"text":"fixture","language":kw["language"] or "English","truncated":False,"finish_reason":"stop"}
    mm=SimpleNamespace(unload_all_models=lambda:calls.append("unload"),soft_empty_cache=lambda:calls.append("empty"));comfy=ModuleType("comfy");comfy.__path__=[];comfy.model_management=mm
    monkeypatch.setitem(sys.modules,"comfy",comfy);monkeypatch.setitem(sys.modules,"comfy.model_management",mm);monkeypatch.setattr(r2,"NativeQ8Engine",Engine)
    model=tmp_path/"m.gguf";projector=tmp_path/"p.gguf";model.touch();projector.touch()
    return SimpleNamespace(calls=calls,model=model,projector=projector,vad=tmp_path/"vad")

@pytest.mark.parametrize("alias,canonical",[("zh","Chinese"),("EN","English"),("粤语","Cantonese"),(" AuTo ",None)])
def test_single_batch_alias_semantics_and_result_metadata_identical(fake_r2,alias,canonical):
    f=fake_r2;audio={"waveform":np.ones((1,1,160),np.float32)*.1,"sample_rate":16000}
    single=r2.transcribe_audio(audio,f.model,f.projector,f.vad,language=alias)
    batch=r2.transcribe_audio_batch(iter([audio]),f.model,f.projector,f.vad,language=alias)[0]
    for key in ("text","language","segments","finish_reason","truncated","mode_executed","backend","duration_seconds","input_gain","model_released"):assert single[key]==batch[key]
    assert single["language"]==(canonical or "English")
    assert all(call[1]["language"]==canonical for call in f.calls if isinstance(call,tuple))

def test_invalid_batch_before_iterator_paths_and_engine(fake_r2,monkeypatch):
    touched=[]
    class Audios:
        def __iter__(self):touched.append("iterator");raise AssertionError("Invalid language touched audio iterator")
    monkeypatch.setattr(r2,"_validate_paths",lambda *a:touched.append("paths"))
    with pytest.raises(ValueError,match="Confucius"):r2.transcribe_audio_batch(Audios(),"missing","missing","missing",language="Swedish")
    assert touched==[] and fake_r2.calls==[]

def test_invalid_single_before_audio_conversion_or_model(fake_r2,monkeypatch):
    monkeypatch.setattr(r2,"comfy_audio_to_16k",lambda *a:pytest.fail("Invalid language converted audio"))
    with pytest.raises(ValueError,match="Confucius"):r2.transcribe_audio(object(),"missing","missing","missing",language="Arabic")
    assert fake_r2.calls==[]

def test_native_engine_alias_prompt_and_metadata_canonical_without_Dll_or_weights():
    prompts=[];engine=object.__new__(native.NativeQ8Engine);engine._lock=threading.RLock();engine._handle=1;engine._cancel_error=None;engine._callback=None;engine.check_cancel=lambda:None;engine.model_name="fixture";engine.projector_name="fixture"
    engine._lib=SimpleNamespace(r2t2_generate=lambda handle,ptr,n,prompt,*args:prompts.append(prompt.decode()) or "你好".encode(),r2t2_last_truncated=lambda:False)
    result=engine.transcribe(np.ones(8,np.float32),language=" zh ")
    assert result["language"]=="Chinese" and "language Chinese<asr_text>" in prompts[0]
    assert native.build_prompt(language=" 中文 ")==native.build_prompt(language="cHiNeSe")

def test_native_invalid_before_audio_array_or_generate():
    class Audio:
        def __array__(self,*a):pytest.fail("Invalid language touched PCM")
    with pytest.raises(ValueError,match="Confucius"):native.NativeQ8Engine.transcribe(object(),Audio(),language="Swedish")

@pytest.mark.parametrize("language,canonical",[("zh","Chinese"),(" en ","English"),("粤语","Cantonese"),("AUTO",None)])
def test_qwen_prompt_ids_canonicalize_before_tokenizer(language,canonical):
    calls=[];model=SimpleNamespace(encode=lambda text:calls.append(text) or [1]);asr.prompt_ids(model,"context",language,1)
    assert (f"assistant\nlanguage {canonical}<asr_text>" if canonical else "assistant\n") in calls

def test_qwen_prompt_invalid_before_tokenizer_and_transcribe_before_model_audio():
    model=SimpleNamespace(encode=lambda text:pytest.fail("Invalid language tokenized"))
    with pytest.raises(ValueError,match="Qwen3-ASR"):asr.prompt_ids(model,"", "invalid",1)
    with pytest.raises(ValueError,match="Qwen3-ASR"):asr.transcribe(object(),object(),16000,language="invalid")

def test_qwen_transcribe_alias_metadata_cpu_mock(monkeypatch):
    decoder=SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids,**kw:torch.zeros(len(ids),4)),generate=lambda **kw:[asr.STOP_TOKENS[0]])
    model=SimpleNamespace(patcher=SimpleNamespace(load_device=torch.device("cpu")),compute_dtype=torch.float32,model=SimpleNamespace(decoder=decoder,audio_tower=lambda _:torch.zeros(1,4)),decode=lambda _:"fixture")
    monkeypatch.setattr(asr,"log_mel",lambda _:torch.zeros(128,50));monkeypatch.setattr(asr,"prepare_audio",lambda *a:torch.zeros(1,8000));monkeypatch.setattr(asr,"split_long",lambda value:[value]);monkeypatch.setattr(asr,"prompt_ids",lambda model,context,language,tokens:([asr.AUDIO_PAD] if language=="Chinese" else pytest.fail("Not canonical")))
    monkeypatch.setattr(asr.comfy.model_management,"load_models_gpu",lambda *a:None);monkeypatch.setattr(asr.comfy.model_management,"cuda_device_context",lambda *a:contextlib.nullcontext())
    assert asr.transcribe(model,torch.zeros(1,8000),16000,language=" zh ")==("fixture","Chinese")

def test_qwen_node_invalid_before_audio_or_model_loading(monkeypatch):
    from drama3_t8 import nodes
    monkeypatch.setattr(asr,"load_qwen3_asr",lambda *a:pytest.fail("Invalid language allocated Qwen"))
    with pytest.raises(ValueError,match="Qwen3-ASR"):nodes.T8DramaTranscribe.execute("missing",object(),language="unknown",model_path="missing")

@pytest.mark.parametrize("backend,bad",[("r2t2","Swedish"),("qwen","invalid")])
def test_engine_integration_hunk_rejects_before_waveform_or_loading(backend,bad,monkeypatch):
    import drama3_t8.engine as engine
    monkeypatch.setattr(engine,"audio_waveform",lambda *a:pytest.fail("Invalid language touched engine audio"))
    with pytest.raises(ValueError):engine.DramaEngine.transcribe_detailed(SimpleNamespace(asr_backend=backend),object(),16000,language=bad)

def test_voice_catalog_language_label_is_not_rewritten(tmp_path):
    from drama3_t8.voice_library import VoiceLibrary
    from drama3_t8.state import Reference,Session
    library=VoiceLibrary(tmp_path/"voices");voice=library.create_voice("测试");session=Session((Reference("你好",torch.zeros((10,2),dtype=torch.long),"fixture"),))
    version=library.add_version(voice["id"],session,language="粤语")
    assert normalize_asr_language(version["language"],native.SUPPORTED_LANGUAGES)=="Cantonese"
    assert library.get_version(version["id"])["language"]=="粤语"
