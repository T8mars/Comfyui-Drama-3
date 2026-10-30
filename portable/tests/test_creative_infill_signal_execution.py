"""CPU Studio/native contract fixtures only, no inference quality assertion."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import soundfile as sf
import torch

from test_creative_infill_execution import setup
from drama3_t8.state import Take
from drama3_t8.creative_workspace import CreativeWorkspace


def quiet_decoder(engine,amplitude=0):
    def decode(codes):
        return {"waveform":torch.full((1,1,codes.shape[-1]*2048),amplitude),"sample_rate":44100}
    engine.decode=decode


def test_new_speech_silent_insert_retains_WAV_and_Session_and_never_encodes_conditioning(tmp_path,monkeypatch):
    manager,engine,_,job,out=setup(tmp_path)
    quiet_decoder(engine,2e-5)
    monkeypatch.setattr(engine,"encode_reference",lambda *a,**k:pytest.fail("rejected edit must not create conditioning"))
    with pytest.raises(RuntimeError,match="接近静音"):
        manager._generate(job,out,lambda *_:None)
    assert list(out.glob("*.wav")) and (out/"session.safetensors").is_file()
    assert not (out/"conditioning.safetensors").exists()
    evidence=json.loads((out/"infill-energy.json").read_text(encoding="utf8"))
    assert evidence["rejected"] and evidence["requires_new_spoken_alnum"]


def test_delete_only_silent_insert_is_not_globally_prohibited(tmp_path):
    manager,engine,_,job,out=setup(tmp_path)
    quiet_decoder(engine,0.)
    job["lines"][0]["readout_text"]="Original utterance"
    result=manager._generate(job,out,lambda *_:None)
    evidence=json.loads((out/"infill-energy.json").read_text(encoding="utf8"))
    assert not evidence["rejected"] and not evidence["requires_new_spoken_alnum"]
    assert result["complete"] and Path(result["conditioning_session_path"]).is_file()


@pytest.mark.parametrize("amplitude",[2e-4,.02])
def test_new_speech_above_extreme_quiet_threshold_is_not_rejected(tmp_path,amplitude):
    manager,engine,_,job,out=setup(tmp_path)
    quiet_decoder(engine,amplitude)
    result=manager._generate(job,out,lambda *_:None)
    assert result["complete"] and not json.loads((out/"infill-energy.json").read_text(encoding="utf8"))["rejected"]


def test_legitimate_empty_FIM_is_not_globally_prohibited(tmp_path):
    manager,engine,_,job,out=setup(tmp_path)
    job["lines"][0]["readout_text"]="Original utterance"
    engine._take=lambda prompt,text,transcript,sampling,**kw:Take(text,transcript,torch.empty((0,11),dtype=torch.long),metadata={"seed":sampling.seed,"length_limit_reached":False})
    result=manager._generate(job,out,lambda *_:None)
    evidence=json.loads((out/"infill-energy.json").read_text(encoding="utf8"))
    assert result["complete"] and evidence["inserted_samples"]==0 and not evidence["rejected"]


def test_new_required_speech_with_empty_middle_is_failed_before_conditioning(tmp_path,monkeypatch):
    manager,engine,_,job,out=setup(tmp_path)
    engine._take=lambda prompt,text,transcript,sampling,**kw:Take(text,transcript,torch.empty((0,11),dtype=torch.long),metadata={"seed":sampling.seed,"length_limit_reached":False})
    monkeypatch.setattr(engine,"encode_reference",lambda *a,**k:pytest.fail("empty required speech cannot become conditioning"))
    with pytest.raises(RuntimeError,match="接近静音"):
        manager._generate(job,out,lambda *_:None)
    assert (out/"session.safetensors").is_file()
    evidence=json.loads((out/"infill-energy.json").read_text(encoding="utf8"))
    assert evidence["rejected"] and evidence["reason"]=="new_spoken_material_but_empty_insert"
    assert not (out/"conditioning.safetensors").exists()


def test_workspace_marks_failed_alternative_and_preserves_original_selection(tmp_path,monkeypatch):
    fixture=tmp_path/"fixture";fixture.mkdir()
    manager,engine,parent,source_job,_=setup(fixture)
    workspace=CreativeWorkspace(tmp_path/"workspace")
    model=source_job["model_identity"]
    project=workspace.create_project("signal guard",[{"line_id":"L01","role":"B","text":parent.transcript}],cast={"B":{"speaker":2}})
    def original(job,directory):
        wave=directory/"original.wav";session=directory/"session.safetensors"
        shutil.copyfile(source_job["source_candidate"]["audio_path"],wave)
        shutil.copyfile(source_job["source_candidate"]["session_path"],session)
        return {"complete":True,"seed":job["seed"],"audio_path":str(wave),"session_path":str(session),"sample_rate":44100,
            "duration_seconds":engine.original.shape[-1]/44100,"metadata":{"finish_reason":"eos"}}
    done=workspace.run_generation(project["project_id"],original,takes_per_line=1,mode="per_line_context",model_identity=model)
    source_id=done["generated"][0]
    manager.workspace=workspace
    quiet_decoder(engine,0.)
    monkeypatch.setattr(engine,"encode_reference",lambda *a,**k:pytest.fail("failed alternative cannot become context"))
    with pytest.raises(RuntimeError,match="接近静音"):
        workspace.run_variation(project["project_id"],"L01",source_id,
            lambda job,directory:manager._generate(job,directory,lambda *_:None),
            {"kind":"infill","text":"New required speech","start_seconds":9000/44100,"end_seconds":13000/44100},model_identity=model)
    actual=workspace.get_project(project["project_id"])
    assert actual["lines"][0]["selected_candidate_id"]==source_id
    failed=[c for c in actual["candidates"].values() if c["status"]=="failed"]
    assert len(failed)==1 and actual["candidates"][source_id]["status"]=="completed"
    directory=workspace.projects/project["project_id"]/"artifacts"/failed[0]["candidate_id"]
    assert (directory/"session.safetensors").is_file() and (directory/"infill-energy.json").is_file()
