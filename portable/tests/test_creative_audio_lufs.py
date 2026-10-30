import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from drama3_t8 import creative_audio as ca


def source_file(tmp_path, seconds=1, amplitude=.1):
    file=tmp_path/"source.wav"
    wave=np.asarray(amplitude*np.sin(2*np.pi*440*np.arange(round(16000*seconds))/16000),dtype=np.float32)
    sf.write(file,wave,16000,subtype="FLOAT")
    return file,wave


def measurement(**changes):
    return {"input_i":"-30.00","input_tp":"-12.00","input_lra":"1.00","input_thresh":"-40.00",
        "output_i":"-16.00","output_tp":"-0.50","output_lra":"1.00","output_thresh":"-26.00",
        "target_offset":"0.00","normalization_type":"linear",**changes}


def fake_process(monkeypatch, source, *, bad_stage=None, malformed=False, overrides=None):
    calls=[]
    def create(command,**kw):
        index=len(calls)+1
        calls.append(command)
        data=measurement(input_i="-30.00" if index==1 else "-16.00",input_tp="-12.00" if index==1 else "-8.00",**(overrides or {}))
        kw["stderr"].write(b"not json" if malformed else json.dumps(data).encode())
        if "-n" in command:
            audio,rate=sf.read(source,dtype="float32",always_2d=True)
            sf.write(command[-1],audio,rate,subtype="FLOAT")
        return SimpleNamespace(wait=lambda timeout:1 if index==bad_stage else 0)
    monkeypatch.setattr(ca,"ffmpeg_executable",lambda:"fake-ffmpeg")
    monkeypatch.setattr(ca.subprocess,"Popen",create)
    return calls


@pytest.mark.parametrize("stage",[1,2,3])
def test_child_failure_at_analysis_render_or_verification_cleans_only_private_temp(tmp_path,monkeypatch,stage):
    source,wave=source_file(tmp_path)
    before=source.read_bytes();witness=tmp_path/"existing-user.wav";witness.write_bytes(b"do not touch")
    calls=fake_process(monkeypatch,source,bad_stage=stage)
    with pytest.raises(RuntimeError,match="FFmpeg"):
        ca._run_loudnorm_file(source,tmp_path/"result.wav",16000,-16,.98,1,lambda:None)
    assert len(calls)==stage and source.read_bytes()==before and witness.read_bytes()==b"do not touch"
    assert not (tmp_path/"result.wav").exists() and not list(tmp_path.glob(".drama-loudness-*"))


@pytest.mark.parametrize("value",[None,True,"bogus","nan"])
def test_malformed_numeric_report_rejected_not_promoted_to_success(tmp_path,monkeypatch,value):
    source,_=source_file(tmp_path)
    fake_process(monkeypatch,source,overrides={"target_offset":value})
    with pytest.raises(RuntimeError,match="报告"):
        ca._run_loudnorm_file(source,tmp_path/"result.wav",16000,-16,.98,1,lambda:None)
    assert not (tmp_path/"result.wav").exists()


def test_missing_json_report_fails_explicitly(tmp_path,monkeypatch):
    source,_=source_file(tmp_path)
    fake_process(monkeypatch,source,malformed=True)
    with pytest.raises(RuntimeError,match="测量报告"):
        ca._run_loudnorm_file(source,tmp_path/"result.wav",16000,-16,.98,1,lambda:None)


def test_existing_destination_is_preserved_without_launching_child(tmp_path,monkeypatch):
    source,_=source_file(tmp_path);target=tmp_path/"result.wav";target.write_bytes(b"old result")
    monkeypatch.setattr(ca,"ffmpeg_executable",lambda:"fake-ffmpeg")
    monkeypatch.setattr(ca.subprocess,"Popen",lambda *a,**k:pytest.fail("must reject before launch"))
    with pytest.raises(FileExistsError):ca._run_loudnorm_file(source,target,16000,-16,.98,1,lambda:None)
    assert target.read_bytes()==b"old result"


@pytest.mark.parametrize("kill",[False,True])
@pytest.mark.parametrize("reason",["cancel","timeout"])
def test_child_cancel_timeout_and_kill_are_owned_and_bounded(tmp_path,monkeypatch,kill,reason):
    class Cancelled(RuntimeError):pass
    error=Cancelled("stop exactly this child")
    process=SimpleNamespace(terminated=False,killed=False,waits=[])
    def wait(timeout):
        process.waits.append(timeout)
        if process.killed or process.terminated and not kill:return 0
        raise ca.subprocess.TimeoutExpired("owned",timeout)
    process.wait=wait
    process.terminate=lambda:setattr(process,"terminated",True)
    process.kill=lambda:setattr(process,"killed",True)
    monkeypatch.setattr(ca.subprocess,"Popen",lambda *a,**kw:process)
    count=[0]
    def check():
        count[0]+=1
        if reason=="cancel" and count[0]>=3:raise error
    if reason=="cancel":
        with pytest.raises(Cancelled) as got:ca._loudness_process(["owned"],ca.time.monotonic()+60,check)
        assert got.value is error
    else:
        with pytest.raises(TimeoutError):ca._loudness_process(["owned"],0,check)
    assert process.terminated and process.killed is kill
    assert all(timeout in (.1,2) for timeout in process.waits)


def test_actual_changed_length_is_rejected_before_publishing(tmp_path,monkeypatch):
    source,_=source_file(tmp_path)
    calls=[0]
    def run(command,deadline,check):
        calls[0]+=1
        if "-n" in command:sf.write(command[-1],np.zeros(15999,dtype=np.float32),16000,subtype="FLOAT")
        return measurement()
    monkeypatch.setattr(ca,"ffmpeg_executable",lambda:"fake-ffmpeg")
    monkeypatch.setattr(ca,"_loudness_process",run)
    with pytest.raises(RuntimeError,match="时长"):
        ca._run_loudnorm_file(source,tmp_path/"result.wav",16000,-16,.98,1,lambda:None)
    assert not (tmp_path/"result.wav").exists()


def test_streamed_gain_never_uses_unsafe_read_write_handle_or_full_clip(tmp_path,monkeypatch):
    source,_=source_file(tmp_path,seconds=5)
    original=source.read_bytes();calls=[]
    old_read=ca.sf.SoundFile.read
    def bounded(self,frames=-1,*a,**kw):
        calls.append(frames);assert 0<frames<=ca.BLOCK_SAMPLES
        return old_read(self,frames,*a,**kw)
    monkeypatch.setattr(ca.sf.SoundFile,"read",bounded)
    owned=tmp_path/"owned.wav";owned.write_bytes(original)
    ca._gain_loudness_file(owned,-6,lambda:None)
    assert source.read_bytes()==original
    assert sf.info(owned).frames==80000
    assert calls


def test_real_silence_reports_unmeasurable_without_fake_lufs(tmp_path):
    source=tmp_path/"silence.wav";sf.write(source,np.zeros(16000,dtype=np.float32),16000,subtype="FLOAT")
    target=tmp_path/"result.wav"
    report=ca._run_loudnorm_file(source,target,16000,-16,.98,1,lambda:None)
    assert report["status"]=="unmeasurable" and report["actual_i"] is None and not report["target_achieved"]
    assert report["warning"] and report["render_attempts"]==0 and sf.info(target).frames==16000
    assert np.count_nonzero(sf.read(target)[0])==0


def test_real_sub_window_clip_keeps_original_pcm_and_reports_undefined_integrated_lufs(tmp_path):
    source,wave=source_file(tmp_path,seconds=.05)
    target=tmp_path/"result.wav"
    report=ca._run_loudnorm_file(source,target,16000,-16,.98,.05,lambda:None)
    assert report["status"]=="unmeasurable" and report["actual_i"] is None
    assert np.array_equal(sf.read(target,dtype="float32")[0],wave)


def test_real_peak_constrained_target_not_falsely_claimed(tmp_path):
    source,_=source_file(tmp_path,seconds=1,amplitude=.03)
    target=tmp_path/"result.wav"
    report=ca._run_loudnorm_file(source,target,16000,-1,.1,1,lambda:None)
    assert not report["target_achieved"] and report["status"]=="target_not_achieved" and report["warning"]
    assert report["actual_i"] < -10
    assert report["actual_sample_peak"] <= .1
    assert report["render_attempts"]<=3 and sf.info(target).frames==16000


def test_real_two_pass_report_uses_final_rate_measurement_not_internal_output_stat(tmp_path):
    source,_=source_file(tmp_path,seconds=1)
    target=tmp_path/"result.wav"
    report=ca._run_loudnorm_file(source,target,16000,-20,.98,1,lambda:None)
    assert report["target_achieved"] and report["actual_i"]==float(report["measured_after"]["input_i"])
    assert float(report["output_i"])==report["actual_i"]
    assert report["render_attempts"]<=3 and report["attempts"][0]["first_pass"]
    assert not list(tmp_path.glob(".drama-loudness-*"))


def test_above_unity_float_outside_measured_parameter_range_uses_explicit_dynamic_fallback(tmp_path):
    source,_=source_file(tmp_path,seconds=1,amplitude=10)
    before=source.read_bytes();target=tmp_path/"result.wav"
    report=ca._run_loudnorm_file(source,target,16000,-20,.98,1,lambda:None)
    assert float(report["analysis"]["input_i"])>0
    assert report["attempts"][0]["two_pass_parameters_supplied"] is False
    assert report["target_achieved"] and report["actual_sample_peak"]<=.98
    assert source.read_bytes()==before
