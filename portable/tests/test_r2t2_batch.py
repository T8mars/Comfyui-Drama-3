"""CPU batch lifecycle/segmentation contracts; no native weights are allocated."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import drama3_t8.r2t2.transcribe as transcription


def audio(value=.1, seconds=1):
    return {"waveform":np.full((1,1,int(seconds*16000)),value,dtype=np.float32),"sample_rate":16000}


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    calls=[];instances=[]
    class Engine:
        error=None
        close_error=None
        def __init__(self,*args,**kwargs):
            calls.append("construct");self.inputs=[];self.closed=False;instances.append(self)
        def __enter__(self):calls.append("enter");return self
        def transcribe(self,pcm,**kwargs):
            self.inputs.append((pcm.copy(),kwargs));calls.append("transcribe")
            if self.error is not None:raise self.error
            return dict(text="one" if pcm[0]<.15 else "two",language="English",finish_reason="stop",truncated=False)
        def __exit__(self,kind,exc,tb):
            calls.append(("close",kind));self.closed=True
            if self.close_error is not None:raise self.close_error
    mm=SimpleNamespace(unload_all_models=lambda:calls.append("unload"),soft_empty_cache=lambda:calls.append("empty"))
    comfy=ModuleType("comfy");comfy.__path__=[];comfy.model_management=mm
    monkeypatch.setitem(sys.modules,"comfy",comfy);monkeypatch.setitem(sys.modules,"comfy.model_management",mm)
    monkeypatch.setattr(transcription,"NativeQ8Engine",Engine)
    model,projector=tmp_path/"模型.gguf",tmp_path/"投影.gguf";model.touch();projector.touch()
    return SimpleNamespace(calls=calls,instances=instances,Engine=Engine,model=model,projector=projector,vad=tmp_path/"missing-vad")


def batch(r,values,**kw):
    return transcription.transcribe_audio_batch(values,r.model,r.projector,r.vad,**kw)


def test_batch_shares_native_engine_and_mm_release_preserving_order(runtime):
    r=runtime
    result=batch(r,(a for a in [audio(.1),audio(0),audio(.2)]))
    assert [x["text"] for x in result]==["one","","two"]
    assert r.calls.count("construct")==r.calls.count("unload")==1 and r.calls.count("empty")==2
    assert len(r.instances)==1 and r.instances[0].closed
    assert [x["batch_index"] for x in result]==[0,1,2]
    assert all(x["model_released"] and x["batch_engine_loads"]==1 for x in result)
    assert all(x["timing_method"]=="segment_estimate" and x["execution"]=="in_process_native" for x in result)
    assert all(x["batch_elapsed_seconds"]>=x["elapsed_seconds"]>=0 for x in result)


def test_single_and_batch_keep_same_transcription_and_gain_semantics(runtime):
    r=runtime;values=[audio(.0001),audio(.2)]
    singles=[transcription.transcribe_audio(a,r.model,r.projector,r.vad,language="auto",context="scene",hotwords="one") for a in values]
    grouped=batch(r,values,language="auto",context="scene",hotwords="one")
    keys={"text","language","segments","truncated","finish_reason","mode_executed","backend","duration_seconds","input_gain","timing_method","model_path","projector_path","vad_path","execution","model_released"}
    assert [{k:a[k] for k in keys} for a in singles]==[{k:a[k] for k in keys} for a in grouped]
    assert all(opts["language"] is None and opts["context"]=="scene\nHotwords: one" for _,opts in r.instances[-1].inputs)


def test_all_silence_and_empty_batch_allocate_nothing(runtime):
    r=runtime
    result=batch(r,[audio(0),audio(0)])
    assert r.calls==[] and all(x["mode_executed"]=="digital_silence" and x["batch_engine_loads"]==0 for x in result)
    assert batch(r,[])==[] and r.calls==[]


@pytest.mark.parametrize("silent",[True,False])
def test_missing_configuration_is_not_success_even_for_silence(runtime,silent):
    r=runtime
    with pytest.raises(FileNotFoundError,match="model"):
        transcription.transcribe_audio_batch([audio(0 if silent else .1)],r.model.parent/"absent",r.projector,r.vad)
    assert r.calls==[]


def test_zero_split_does_not_require_vad_or_carry_recording_state(runtime):
    r=runtime
    wave=np.concatenate([np.full(16000,.1,np.float32),np.zeros(4000,np.float32),np.full(16000,.2,np.float32)])
    a={"waveform":wave[None,None],"sample_rate":16000}
    result=batch(r,[a,a])
    assert all(x["mode_executed"]=="offline_zero_split" and x["text"]=="one two" for x in result)
    assert len(result[0]["segments"])==len(result[1]["segments"])==2
    assert result[0]["segments"][0]["start_sample"]==result[1]["segments"][0]["start_sample"]==0
    assert len(r.instances[0].inputs)==4 and r.calls.count("construct")==1


def test_long_files_get_fresh_vad_state_but_one_native_engine(runtime,monkeypatch):
    from drama3_t8.r2t2.segmented import PresetBoundaryVAD
    import drama3_t8.r2t2.vad as vad
    created=[]
    class VAD(PresetBoundaryVAD):
        def __init__(self,path):super().__init__([]);created.append(self)
    monkeypatch.setattr(vad,"FireRedOnnxVAD",VAD)
    r=runtime;result=batch(r,[audio(.1,31),audio(.2,31)])
    assert len(created)==2 and all(v.seen==31*16000 for v in created)
    assert all(x["mode_executed"]=="segmented_offline" for x in result)
    assert r.calls.count("construct")==1 and len(r.instances[0].inputs)==4


def test_missing_required_vad_closes_shared_engine(runtime):
    r=runtime
    with pytest.raises(FileNotFoundError,match="FireRedVAD"):
        batch(r,[audio(.1),audio(.2,31)])
    assert r.instances[0].closed and r.calls.count("empty")==2


def test_base_exception_identity_and_exit_context_are_preserved(runtime):
    class Stop(BaseException):pass
    r=runtime;error=Stop();r.Engine.error=error
    with pytest.raises(Stop) as caught:batch(r,[audio(.1),audio(.2)])
    assert caught.value is error and r.instances[0].closed
    assert ("close",Stop) in r.calls and r.calls.count("transcribe")==1
    assert r.calls.count("empty")==2


def test_iterator_failure_after_first_recording_closes_once(runtime):
    r=runtime;error=RuntimeError("producer failed")
    def source():
        yield audio(.1)
        raise error
    with pytest.raises(RuntimeError) as caught:batch(r,source())
    assert caught.value is error and len(r.instances)==1 and r.instances[0].closed
    assert r.calls.count("empty")==2


def test_cancellation_after_inference_closes_without_returning_partial_success(runtime):
    class Cancel(BaseException):pass
    r=runtime;error=Cancel()
    def check():
        if r.instances and r.instances[0].inputs:raise error
    with pytest.raises(Cancel) as caught:batch(r,[audio(.1),audio(.2)],check_cancel=check)
    assert caught.value is error and r.calls.count("transcribe")==1 and r.instances[0].closed


def test_cancel_before_work_never_unloads_other_models(runtime):
    r=runtime
    def check():raise RuntimeError("cancel")
    with pytest.raises(RuntimeError,match="cancel"):batch(r,[audio(.1)],check_cancel=check)
    assert r.calls==[]


def test_close_failure_is_not_reported_as_model_released(runtime):
    r=runtime;r.Engine.close_error=RuntimeError("close failed")
    with pytest.raises(RuntimeError,match="close failed"):batch(r,[audio(.1)])
    assert r.calls.count("empty")==2


def test_invalid_later_audio_and_batch_limit_close_shared_engine(runtime):
    r=runtime
    with pytest.raises(ValueError):batch(r,[audio(.1),{"waveform":np.zeros((2,1,10)),"sample_rate":16000}])
    assert r.instances[0].closed
    with pytest.raises(ValueError,match="max_items"):batch(r,[audio(.1),audio(.2)],max_items=1)
    assert r.instances[-1].closed and len(r.instances[-1].inputs)==1
    with pytest.raises(TypeError):batch(r,audio(.1))
