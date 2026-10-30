"""CPU injection verifies benchmark orchestration, not synthesized voice quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf

from drama3_t8 import creative_benchmark as cb


SETTINGS = dict(gdn_backend="torch", moe_backend="torch", cuda_graph=False,
                expert_placement="ram", expert_cache_gb=0., asr_backend="r2t2", scene_head_mode="compact")


class FakeEngine:
    def __init__(self, owner):self.owner=owner;self.drama=None
    def unload(self):self.owner.events.append(("unload",));self.drama=None


class FakeStudio:
    def __init__(self, owner):
        self.owner=owner;self.settings=deepcopy(SETTINGS);self.model_paths={}
        self.engine=FakeEngine(owner)
    def apply_settings(self, settings, persist=True):
        assert persist is False
        self.owner.events.append(("settings",deepcopy(settings)))
        self.engine.unload();self.settings=deepcopy(settings);self.engine=FakeEngine(self.owner)
    def begin_commit(self, job_id, progress):progress();self.owner.events.append(("commit",job_id))


class Manager:
    def __init__(self, root):
        self.events=[];self.config={};self.reports=root/"userdata/performance"
        self.project={"project_id":"test-project","revision":1,
            "lines":[dict(line_id="original1",role="甲",readout_text="original"),dict(line_id="original2",role="乙",readout_text="other")],
            "cast":{"甲":dict(voice_id="v1",version_id="s1",speaker=2),"乙":dict(voice_id="v2",version_id="s2",speaker=5)},
            "candidates":{"must-stay":"original"}}
        self.workspace=SimpleNamespace(projects=root/"projects",get_project=lambda value:deepcopy(self.project),_directory=lambda value:root/"projects"/value)
        self.studio=FakeStudio(self);self.words={};self.source="a"*64
        self.versions={}
        for n in (1,2):
            path=root/f"s{n}.safetensors";path.write_bytes(f"real-frozen-test-reference-{n}".encode())
            self.versions[f"s{n}"]=path
        self.voices=SimpleNamespace(get_version=lambda version:dict(id=version,voice_id="v"+version[-1]),
            session_file=lambda version:self.versions[version],
            load_version=lambda version:SimpleNamespace(references=[SimpleNamespace(transcript="reference "+version)]))
        self.fail=None;self.missing_part=False;self.bad_words=False;self.fallback=False;self.no_actual=False
        self.long_evidence=dict(scene_chunking=dict(enabled=True,chunk_bytes=300),chunk_count=2)
    def model_identity(self, strong=True):
        return dict(model=dict(sha256="b"*64),codec=dict(sha256="c"*64),text_head=None,line_head=dict(sha256="d"*64))
    def source_identity(self):return self.source
    def performance_binding(self, models):
        return dict(machine="test-machine",gpu="test-gpu",torch="2.8.0+cu128",cuda="12.8",transformers="4.57.1",
            models=deepcopy(models),source=self.source,backends={"test":"injected-cpu"})
    def _generate(self, job, directory, progress, parent_recorder=None):
        progress();self.events.append(("generate",job["unit_id"],self.studio.engine.drama is not None))
        if self.fail:self.fail(job)
        self.studio.engine.drama=object()
        parent_recorder.mark_first_codes()
        rate=16000;wave=.1*np.sin(2*np.pi*220*np.arange(4800)/rate).astype(np.float32)
        target=directory/"whole.wav";sf.write(target,np.tile(wave,len(job["lines"])),rate,subtype="FLOAT")
        parts=[]
        for i,line in enumerate(job["lines"]):
            clip=directory/f"line-{i}.wav";sf.write(clip,wave,rate,subtype="FLOAT")
            self.words[str(clip.resolve())]=line["readout_text"]
            parts.append(dict(line_id=line["line_id"],audio_path=str(clip)))
        if self.missing_part and len(parts)>1:parts.pop()
        parent_recorder.mark_first_pcm(len(wave),rate)
        backends=[]
        for key,component in (("moe_backend","moe"),("gdn_backend","gdn"),("cuda_graph","graph"),("expert_placement","placement")):
            value=self.studio.settings[key]
            backends.append(dict(component=component,requested=value,actual=value,fallback_reason="test fallback" if self.fallback else None))
        if self.no_actual:backends=[]
        metadata=dict(finish_reason="eos",reference_versions=list(job["cast"].values()),performance=dict(backends=backends))
        if job["unit_id"]=="benchmark-long" and self.long_evidence is not None:metadata.update(deepcopy(self.long_evidence))
        return dict(complete=True,seed=job["seed"],audio_path=str(target),duration_seconds=.3*len(job["lines"]),line_audio=parts,metadata=metadata)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager=Manager(tmp_path)
    monkeypatch.setattr(cb,"_binding",lambda manager,models,original,progress:manager.performance_binding(models))
    def transcribe(manager, paths, original, progress):
        manager.events.append(("asr_batch",len(paths)));progress()
        return [dict(text="wrong words" if manager.bad_words else manager.words[str(Path(p).resolve())],finish_reason="stop",truncated=False) for p in paths]
    monkeypatch.setattr(cb,"_transcribe",transcribe)
    return manager


def payload(**values):
    return dict(project_id="test-project",quick=False,warm_repeats=2,include_cold=True,
        configurations=[dict(id="torch-eager",settings={k:SETTINGS[k] for k in ("moe_backend","gdn_backend","cuda_graph","expert_placement","expert_cache_gb")})],**values)


def run(manager, request=None):return cb.run_studio_benchmark(manager,"job-test",request or payload(),lambda *args:None)


def test_full_real_file_orchestration_all_timing_before_one_shared_quality(manager):
    before=deepcopy(manager.project);original=deepcopy(manager.studio.settings)
    result=run(manager);report=result["report"]
    assert report["status"]=="completed" and report["profile_saved"] and report["settings_restored"]
    assert set(report["coverage"])=={"short","long","reference","multi_speaker"}
    long_case=next(c for c in report["cases"] if c["id"]=="long")
    assert len(long_case["payload"]["readout_text"].encode("utf-8"))>long_case["payload"]["settings"]["chunk_bytes"]
    assert long_case["payload"]["mode"]=="together"
    assert len(report["results"])==12 and len([e for e in manager.events if e[0]=="asr_batch"])==1
    generations=[i for i,e in enumerate(manager.events) if e[0]=="generate"]
    asr=next(i for i,e in enumerate(manager.events) if e[0]=="asr_batch")
    assert max(generations)<asr and manager.events[asr][1]==18
    assert [e[2] for e in manager.events if e[0]=="generate"]==[False,True,True]*4
    assert manager.project==before and manager.studio.settings==original
    assert len(list((manager.reports/"profiles").glob("*.json")))==1
    assert all(r["diagnostics"]["first_codes_seconds"] is not None and r["diagnostics"]["first_pcm_seconds"] is not None for r in report["results"])
    assert report["quality_performance"]["phases"][0]["name"]=="asr.batch"
    for row in report["results"]:
        path=Path(row["artifact"]["directory"])
        assert "benchmarks" in path.parts and path.is_dir() and row["quality"]["passed"]
    saved=next((manager.reports/"profiles").glob("*.json"))
    assert "test-gpu" in saved.read_text("utf-8") and result["recommended"]["config_id"]=="torch-eager"
    proof=json.loads(saved.read_text("utf-8"))["qualification"]["report_sha256"]
    encoded=json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    assert proof==hashlib.sha256(encoded).hexdigest()
    assert all(row["quality"]["checks"]["long_chunked"] for row in report["results"] if row["case_id"]=="long")


def test_quick_has_only_short_and_never_saves_or_recommends(manager):
    request=payload();request.update(quick=True,warm_repeats=1)
    result=run(manager,request)
    assert result["report"]["coverage"]==["short"] and len(result["report"]["results"])==2
    assert result["recommended"] is None and not result["report"]["profile_saved"]
    assert not (manager.reports/"profiles").exists()


@pytest.mark.parametrize("problem",["missing-role","same-speaker","empty-session","ownership"])
def test_full_rejects_unqualified_cast_before_generation(manager,problem):
    if problem=="missing-role":manager.project["cast"].pop("乙")
    elif problem=="same-speaker":manager.project["cast"]["乙"]["speaker"]=2
    elif problem=="empty-session":manager.voices.load_version=lambda version:SimpleNamespace(references=[])
    elif problem=="ownership":manager.project["cast"]["乙"]["voice_id"]="other"
    with pytest.raises(ValueError):run(manager)
    assert not any(e[0]=="generate" for e in manager.events)


@pytest.mark.parametrize("failure",["bad_words","fallback","no_actual"])
def test_quality_or_backend_failure_cannot_qualify_profile(manager,failure):
    setattr(manager,failure,True)
    result=run(manager)
    assert result["report"]["status"]=="completed" and not result["report"]["profile_saved"]
    assert result["recommended"] is None and manager.studio.settings==SETTINGS


def test_missing_multispeaker_clips_fail_truthfully_and_restore(manager):
    manager.missing_part=True
    result=run(manager)
    assert result["report"]["status"]=="error" and not result["report"]["profile_saved"]
    assert not any(e[0]=="asr_batch" for e in manager.events) and manager.studio.settings==SETTINGS


def test_generation_error_stops_without_qa_and_restores(manager):
    def fail(job):raise RuntimeError("controlled CPU callback failure")
    manager.fail=fail
    result=run(manager)
    assert result["report"]["status"]=="error" and result["recommended"] is None
    assert manager.studio.settings==SETTINGS and not any(e[0]=="asr_batch" for e in manager.events)


def test_cancel_keeps_original_exception_and_restores_without_commit(manager):
    class Cancelled(RuntimeError):pass
    error=Cancelled("user cancelled")
    def fail(job):raise error
    manager.fail=fail
    # Cancellation originates in the UI progress callback, as the real engine does.
    calls=[0]
    def progress(*args):
        calls[0]+=1
        if any(e[0]=="generate" for e in manager.events):raise error
    manager.fail=None
    with pytest.raises(Cancelled) as caught:cb.run_studio_benchmark(manager,"job",payload(),progress)
    assert caught.value is error and manager.studio.settings==SETTINGS
    assert not any(e[0]=="commit" for e in manager.events) and not (manager.reports/"profiles").exists()


def test_no_cold_preserves_observed_first_load_and_config_immutability(manager):
    request=payload();request["include_cold"]=False
    before=deepcopy(request);result=run(manager,request)
    assert request==before and len(result["report"]["results"])==8
    first=result["report"]["results"][0]
    assert first["requested_load_state"]=="warm" and first["diagnostics"]["phases"][0]["cache_state"]=="cold"
    assert len([e for e in manager.events if e[0]=="settings"])==2


def test_source_change_blocks_save_even_after_measured_outputs(manager,monkeypatch):
    original=cb._quality
    def quality(*args):
        answer=original(*args);manager.source="e"*64;return answer
    monkeypatch.setattr(cb,"_quality",quality)
    with pytest.raises(ValueError,match="源码变化"):run(manager)
    assert manager.studio.settings==SETTINGS and not (manager.reports/"profiles").exists()


def test_r2_batch_is_blind_and_receives_all_ordered_audio(tmp_path,monkeypatch):
    path=tmp_path/"test.wav";sf.write(path,np.ones(1600,dtype=np.float32)*.1,16000)
    captured={}
    def batch(audios,*paths,**kwargs):
        captured.update(audios=list(audios),paths=paths,kwargs=kwargs);return [{"text":"recognized"}]*2
    monkeypatch.setitem(sys.modules,"drama3_t8.r2t2.transcribe",SimpleNamespace(transcribe_audio_batch=batch))
    manager=SimpleNamespace(studio=SimpleNamespace(model_paths=dict(r2t2_model="model",r2t2_projector="projector",r2t2_vad="vad")))
    result=cb._transcribe(manager,[path,path],SETTINGS,lambda:None)
    assert len(result)==2 and len(captured["audios"])==2
    assert captured["kwargs"]["language"] is None and captured["kwargs"]["context"]==captured["kwargs"]["hotwords"]==""
    assert captured["audios"][0]["waveform"].shape==(1,1,1600)


def test_qwen_bridge_uses_real_tensor_not_numpy(tmp_path,monkeypatch):
    path=tmp_path/"test.wav";sf.write(path,np.ones(1600,dtype=np.float32)*.1,16000)
    observed=[];sentinel=object()
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(from_numpy=lambda value:(observed.append(value.shape) or sentinel)))
    def transcribe(wave,rate,**kwargs):
        assert wave is sentinel and rate==16000 and kwargs["language"] is None
        assert kwargs["context"]==kwargs["hotwords"]==""
        return dict(text="recognized")
    manager=SimpleNamespace(studio=SimpleNamespace(engine=SimpleNamespace(transcribe_detailed=transcribe)))
    assert cb._transcribe(manager,[path],dict(asr_backend="qwen"),lambda:None)==[{"text":"recognized"}]
    assert observed==[(1,1600)]


def test_sampling_or_precision_changes_are_rejected_before_any_generation(manager):
    request=payload();request["configurations"][0]["settings"]["temperature"]=.1
    with pytest.raises(ValueError,match="backend/cache"):run(manager,request)
    assert manager.events==[] and not (manager.reports/"profiles").exists()


def test_full_rejects_single_warm_repeat_instead_of_qualifying_short_measurement(manager):
    request=payload();request["warm_repeats"]=1
    with pytest.raises(ValueError,match="两次"):run(manager,request)
    assert manager.events==[]


def test_reference_snapshot_mutation_is_not_silently_accepted(manager):
    called=[False]
    def fail(job):
        if not called[0]:manager.versions["s1"].write_bytes(b"reference changed");called[0]=True
    manager.fail=fail
    with pytest.raises(ValueError,match="参考音色版本"):run(manager)
    assert manager.studio.settings==SETTINGS and not (manager.reports/"profiles").exists()


def test_model_identity_mutation_is_not_silently_accepted(manager):
    original=manager.model_identity
    def identity(strong=True):
        answer=original(strong)
        if any(e[0]=="generate" for e in manager.events):answer["line_head"]={"sha256":"e"*64}
        return answer
    manager.model_identity=identity
    with pytest.raises(ValueError,match="控制头发生变化"):run(manager)
    assert manager.studio.settings==SETTINGS and not (manager.reports/"profiles").exists()


def test_restore_failure_prevents_profile_commit(manager):
    original=manager.studio.apply_settings;calls=[0]
    def apply(settings,persist=True):
        calls[0]+=1
        if calls[0]>1:raise RuntimeError("restore failed")
        return original(settings,persist=persist)
    manager.studio.apply_settings=apply
    with pytest.raises(RuntimeError,match="restore failed"):run(manager)
    assert not any(e[0]=="commit" for e in manager.events) and not (manager.reports/"profiles").exists()


def test_strong_quality_identity_cache_invalidates_on_file_change(tmp_path):
    manager=SimpleNamespace(reports=tmp_path/"reports")
    path=tmp_path/"artifact.bin";path.write_bytes(b"first")
    first=cb._strong_artifact(manager,path)
    assert cb._strong_artifact(manager,path)==first
    path.write_bytes(b"second revision")
    second=cb._strong_artifact(manager,path)
    assert second["sha256"]!=first["sha256"] and second["hash_kind"]=="full_sha256"


def test_default_pair_changes_only_measured_backends_and_restores(manager):
    request=dict(project_id="test-project",quick=True,warm_repeats=1,include_cold=True)
    result=run(manager,request)
    configs={x["id"]:x["settings"] for x in result["report"]["configs"]}
    assert set(configs)=={"torch-eager","cuda-graph"}
    assert configs["torch-eager"]["moe_backend"]=="torch" and configs["torch-eager"]["cuda_graph"] is False
    assert configs["cuda-graph"]["moe_backend"]=="cuda" and configs["cuda-graph"]["cuda_graph"] is True
    assert all(x["gdn_backend"]=="torch" and x["expert_placement"]=="ram" for x in configs.values())
    assert manager.studio.settings==SETTINGS and len(result["report"]["results"])==4
    assert result["recommended"] is None and "same process" in result["report"]["cold_scope"]


def test_profile_save_lookup_and_explicit_setting_apply_use_identical_strong_binding(tmp_path,monkeypatch):
    from drama3_t8.performance import BenchmarkProfileStore,artifact_identity
    manager=Manager(tmp_path);manager.studio.settings["asr_backend"]="qwen"
    paths={}
    for role in ("model","codec","line_head","asr_model"):
        path=tmp_path/(role+".bin");path.write_bytes((role+" actual CPU identity").encode());paths[role]=path
    manager.studio.model_paths=paths
    def identities(strong=True):
        return {role:None if role=="text_head" else artifact_identity(paths[role],hash_contents=True)
            for role in ("model","codec","text_head","line_head")}
    manager.model_identity=identities
    monkeypatch.setattr(cb,"_transcribe",lambda manager,paths,original,progress:
        [dict(text=manager.words[str(Path(path).resolve())],truncated=False) for path in paths])
    result=run(manager);store=BenchmarkProfileStore(manager.reports/"profiles")
    assert result["report"]["profile_saved"]
    profile=store.load(cb.current_profile_binding(manager));assert profile is not None
    manager.studio.apply_settings({**manager.studio.settings,"moe_backend":"cuda","cuda_graph":True,
        "gdn_backend":"fla","expert_placement":"vram","expert_cache_gb":12},persist=False)
    assert store.load(cb.current_profile_binding(manager))==profile
    manager.studio.apply_settings({**manager.studio.settings,**profile["recommendation"]["settings"]},persist=False)
    assert store.load(cb.current_profile_binding(manager))==profile
    paths["codec"].write_bytes(b"different actual Codec bytes")
    assert store.load(cb.current_profile_binding(manager)) is None


def test_custom_profile_freezes_effective_cache_and_placement_and_actual_apply_restores_them(manager,monkeypatch):
    from drama3_t8.creative_studio import CreativeStudio
    from drama3_t8.performance import BenchmarkProfileStore,SETTINGS as PERFORMANCE_SETTINGS
    manager.studio.settings.update(expert_cache_gb=1.,expert_placement="ram",private_ui_note="must not be exported")
    original=deepcopy(manager.studio.settings)
    request=payload()
    request["configurations"][0]["settings"]={"gdn_backend":"torch","moe_backend":"torch","cuda_graph":False}
    result=run(manager,request)
    effective={key:original[key] for key in PERFORMANCE_SETTINGS}
    assert result["report"]["configs"][0]["settings"]==effective
    assert result["recommended"]["settings"]==effective
    assert manager.studio.settings==original
    store=BenchmarkProfileStore(manager.reports/"profiles")
    profile=store.load(cb.current_profile_binding(manager))
    assert profile["recommendation"]["settings"]==effective
    manager.studio.settings.update(expert_cache_gb=2.5,expert_placement="vram")
    applied=[];native_apply=manager.studio.apply_settings
    def apply_patch(settings,persist=True):
        applied.append((deepcopy(settings),persist))
        native_apply({**manager.studio.settings,**settings},persist=False)
    monkeypatch.setattr(manager.studio,"apply_settings",apply_patch)
    output=CreativeStudio.execute_apply_profile(manager,"apply-tested-profile",{},lambda *args:None)
    assert output["settings"]==effective and applied==[(effective,True)]
    assert manager.studio.settings["expert_cache_gb"]==1.
    assert manager.studio.settings["expert_placement"]=="ram"
    assert manager.studio.settings["private_ui_note"]=="must not be exported"


def test_partial_custom_override_records_all_effective_fields_and_recomputes_rank(manager):
    from drama3_t8.performance import SETTINGS as PERFORMANCE_SETTINGS
    manager.studio.settings.update(gdn_backend="fla",moe_backend="cuda",cuda_graph=True,
                                  expert_placement="vram",expert_cache_gb=1.)
    original=deepcopy(manager.studio.settings)
    request=payload()
    request["configurations"][0]["settings"]={"moe_backend":"torch","cuda_graph":False,"expert_cache_gb":.75}
    result=run(manager,request)
    config=result["report"]["configs"][0]
    assert set(config["settings"])==PERFORMANCE_SETTINGS
    assert config["settings"]==dict(gdn_backend="fla",moe_backend="torch",cuda_graph=False,
                                     expert_placement="vram",expert_cache_gb=.75)
    assert config["stability_rank"]==4 and config["complexity"]==2
    assert result["recommended"]["settings"]==config["settings"] and manager.studio.settings==original


def test_current_binding_resolves_relative_speaker_and_invalidates_quality_model_change(tmp_path):
    from drama3_t8.performance import binding_key
    manager=Manager(tmp_path);manager.studio.settings["asr_backend"]="qwen"
    asr=tmp_path/"asr.bin";asr.write_bytes(b"quality ASR");speaker=tmp_path/"camp.onnx";speaker.write_bytes(b"speaker model")
    manager.studio.model_paths={"asr_model":asr};manager.config["speaker_model_path"]="camp.onnx"
    manager.resolve_path=lambda value:tmp_path/value
    first=cb.current_profile_binding(manager)
    assert first["models"]["quality_speaker"]["path"]==str(speaker.resolve())
    asr.write_bytes(b"changed quality ASR")
    assert binding_key(cb.current_profile_binding(manager))!=binding_key(first)


def test_imported_project_benchmark_and_reference_identity_use_private_library(tmp_path,monkeypatch):
    import torch
    from drama3_t8.state import Reference,Session
    from drama3_t8.voice_library import VoiceLibrary
    from drama3_t8 import creative_transcript
    manager=Manager(tmp_path)
    private=VoiceLibrary(manager.workspace._directory("test-project")/"voice-library")
    for index,role in enumerate(("甲","乙")):
        preview=tmp_path/f"private-{index}.wav";sf.write(preview,np.full(4800,.1+index*.1,dtype=np.float32),16000,subtype="FLOAT")
        voice=private.create_voice(role);version=private.add_version(voice["id"],
            Session((Reference("private reference",torch.ones((10,4),dtype=torch.long)),)),preview_path=preview)
        manager.project["cast"][role]=dict(voice_id=voice["id"],version_id=version["id"],speaker=index)
    manager.config["speaker_model_path"]="relative-camp.onnx";manager.resolve_path=lambda value:tmp_path/value
    monkeypatch.setattr(cb,"_binding",lambda manager,models,original,progress:manager.performance_binding(models))
    monkeypatch.setattr(cb,"_transcribe",lambda manager,paths,original,progress:
        [dict(text=manager.words[str(Path(path).resolve())],truncated=False) for path in paths])
    class Embedder:
        def __init__(self,path):assert path==tmp_path/"relative-camp.onnx"
        def embedding(self,wave,rate):return np.array([1.,0.]) if float(wave.mean())<.15 else np.array([0.,1.])
    monkeypatch.setattr(creative_transcript,"SpeakerEmbedder",Embedder)
    original=manager._generate
    def generate(job,directory,progress,parent_recorder=None):
        result=original(job,directory,progress,parent_recorder=parent_recorder)
        for part,line in zip(result["line_audio"],job["lines"]):
            sf.write(part["audio_path"],np.full(4800,.2 if line["role"]=="乙" else .1,dtype=np.float32),16000,subtype="FLOAT")
        return result
    manager._generate=generate
    result=run(manager)
    assert result["report"]["status"]=="completed" and result["report"]["profile_saved"]
    assert all("voice-library" in x["session_path"] for x in result["report"]["reference_snapshots"])
    identities=[line["identity"] for row in result["report"]["results"] for line in row["quality"]["details"]["lines"] if line["identity"]["checked"]]
    assert identities and all(x["matched"] for x in identities)


@pytest.mark.parametrize("evidence",[None,dict(scene_chunking=dict(enabled=False),chunk_count=3),
    dict(scene_chunking=dict(enabled=True),chunk_count=1),dict(scene_chunking=dict(enabled=True),chunk_count=True)])
def test_long_request_flag_without_real_multichunk_metadata_cannot_qualify(manager,evidence):
    manager.long_evidence=evidence;result=run(manager)
    assert result["report"]["status"]=="error" and result["recommended"] is None and not result["report"]["profile_saved"]
    assert manager.studio.settings==SETTINGS and not any(e[0]=="asr_batch" for e in manager.events)
    proof=list((manager.workspace.projects/"test-project/benchmarks").glob("*/0004-*/coverage-error.json"))
    assert len(proof)==1


def test_long_and_dialogue_cases_use_supported_real_conditioning_modes_and_planner(manager):
    from drama3_t8.creative_context import condition_creative_context
    from drama3_t8.scene_chunks import plan_scene
    from drama3_t8.state import Session
    cast,snapshots=cb._cast_snapshot(manager,manager.project)
    cases={case.id:case for case in cb._cases(manager.project,manager.model_identity(),False,cast)}
    assert cases["long"].payload["mode"]==cases["dialogue"].payload["mode"]=="together"
    # Exercise the real context helper rather than merely asserting a label.
    for case in (cases["long"],cases["dialogue"]):condition_creative_context(case.payload,Session())
    line=cases["long"].payload["lines"][0];speaker=cast[line["role"]]["speaker"]
    plan=plan_scene([f"<|speaker:{speaker}|> {line['readout_text']}"],mode="together",chunk_bytes=300)
    assert len(plan["units"])>1
