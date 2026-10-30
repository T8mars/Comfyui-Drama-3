"""Real workspace commits with injected CPU media/ASR; no model-quality claim."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import uuid
import wave

import pytest

from drama3_t8.creative_quality import NOTE,run_quality_retries
from drama3_t8.creative_workspace import CreativeWorkspace,InvalidCheckpoint,ProjectConflict


def media(job,directory):
    audio=directory/"audio.wav"
    with wave.open(str(audio),"wb") as handle:
        handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(8000)
        handle.writeframes(struct.pack("<80h",*([100]*80)))
    session=directory/"session.safetensors"
    metadata=dict(version=1,session_id=uuid.uuid4().hex,direction="",references=[dict(key="reference_0",transcript="固定音色的参考文本",
        name="CPU reference",codec_revision="fish-s2-codec-v1",processing={})],takes=[dict(key="take_0",
        take_id=uuid.uuid4().hex,transcript=job["readout_text"],prompt_text=job["readout_text"],semantic_start=248099,
        semantic_end=252194,separator_id=252196,metadata=dict(finish_reason="eos",length_limit_reached=False,
            seed=job["seed"],requested_seed=job["seed"],task="scene" if job["mode"] in {"together","per_speaker"} else "tts"))])
    header={"__metadata__":{"drama3_t8_session":json.dumps(metadata)},"take_0":dict(dtype="I64",shape=[1,11],data_offsets=[0,88]),
        "reference_0":dict(dtype="I64",shape=[10,1],data_offsets=[88,168])}
    encoded=json.dumps(header,separators=(",",":")).encode();encoded+=b" "*(-len(encoded)%8)
    session.write_bytes(struct.pack("<Q",len(encoded))+encoded+struct.pack("<11q",248099,*([0]*10))+struct.pack("<10q",*([0]*10)))
    return dict(complete=True,seed=job["seed"],audio_path=str(audio),session_path=str(session),duration_seconds=.01,
        sample_rate=8000,metadata=dict(finish_reason="eos",actual_segments=len(job["lines"])))


def review(passed):
    return dict(normalized_match=passed,independent=True,review_recommended=not passed,
        asr=dict(text="injected CPU ASR result",truncated=False,finish_reason="stop"))


class Engine:
    def __init__(self,owner):self.owner=owner;self.depth=0
    @contextmanager
    def generation_session(self):
        self.owner.events.append(("session.enter",));self.depth+=1
        try:yield
        finally:self.depth-=1;self.owner.events.append(("session.exit",))


class Manager:
    def __init__(self,workspace):
        self.workspace=workspace;self.events=[];self.jobs=[];self.fail=None;self.decisions=[]
        self.studio=SimpleNamespace(engine=Engine(self));self.after_quality=None;self.drop_revision=False
    def _generate(self,job,directory,progress):
        progress();self.events.append(("generate",job["line_ids"]));self.jobs.append(deepcopy(job))
        assert self.studio.engine.depth==1
        if self.fail:self.fail(job)
        return media(job,directory)
    def execute_quality(self,job_id,payload,progress):
        progress();assert self.studio.engine.depth==0
        ids=payload["candidate_ids"];self.events.append(("asr.batch",list(ids)))
        decision=self.decisions.pop(0) if self.decisions else True
        answers={key:review(decision(key) if callable(decision) else decision) for key in ids}
        result=self.workspace.run_quality(payload["project_id"],lambda candidate:answers[candidate["candidate_id"]],candidate_ids=ids)
        response=dict(checked=len(ids),reviews=answers,project_revision=result["project"]["revision"],errors=result["errors"])
        if self.drop_revision:response.pop("project_revision")
        if self.after_quality:self.after_quality(payload,result)
        return response


@pytest.fixture
def setup(tmp_path):
    workspace=CreativeWorkspace(tmp_path/"workspace");manager=Manager(workspace)
    def build(*,lines=None,takes=1,mode="per_line",sampling=None,base_seed=42,quality=None):
        project=workspace.create_project("CPU quality test",lines or [dict(line_id="L1",role="猫",text="原稿保留"),dict(line_id="L2",role="人",text="大家下班")])
        done=workspace.run_generation(project["project_id"],media,takes_per_line=takes,mode=mode,
            sampling=sampling,base_seed=base_seed,model_identity={"test":"frozen model identity"})
        answers={key:review(quality(key,n) if quality else False) for n,key in enumerate(done["generated"])}
        checked=workspace.run_quality(project["project_id"],lambda candidate:answers[candidate["candidate_id"]],candidate_ids=done["generated"])
        return checked["project"],dict(reviews=answers,checked=len(answers),project_revision=checked["project"]["revision"])
    return manager,build


def run(manager,project,initial,**payload):
    return run_quality_retries(manager,"CPU-job",{**dict(project_id=project["project_id"],max_quality_retries=2),**payload},initial,lambda *args:None)


def test_all_units_generated_before_one_shared_asr_and_never_auto_selected(setup):
    manager,build=setup;project,initial=build();original=deepcopy(project)
    result=run(manager,project,initial)
    assert len(result["rounds"])==1 and len(result["suggested_candidates"])==2
    assert [e[0] for e in manager.events]==["session.enter","generate","generate","session.exit","asr.batch"]
    assert len(manager.events[-1][1])==2 and result["selection_changed"] is False
    assert result["remaining_suspect"]==2 and result["unresolved_units"]==[]
    current=manager.workspace.get_project(project["project_id"])
    assert [x["selected_candidate_id"] for x in current["lines"]]==[x["selected_candidate_id"] for x in original["lines"]]
    assert len(current["candidates"])==4
    for key,candidate in original["candidates"].items():assert current["candidates"][key]==candidate
    for job in manager.jobs:
        source=original["candidates"][job["source_candidate"]["candidate_id"]]
        assert job["lines"]==source["input_snapshot"]["lines"] and job["model_identity"]==source["input_snapshot"]["model_identity"]
        assert job["operation"]["note"]==NOTE and job["operation"]["original_text"]==source["input_snapshot"]["lines"][0]["original_text"]


def test_default_zero_performs_no_generation_or_asr_and_keeps_initial_reviews(setup):
    manager,build=setup;project,initial=build()
    result=run_quality_retries(manager,"job",dict(project_id=project["project_id"]),initial,lambda:None)
    assert result["max_quality_retries"]==0 and result["rounds"]==[] and manager.events==[]
    assert result["reviews"]==initial["reviews"] and manager.workspace.get_project(project["project_id"])==project


@pytest.mark.parametrize("limit",[True,False,-1,3,1.,"1",None])
def test_retry_limit_is_strict_integer_zero_to_two(setup,limit):
    manager,build=setup;project,initial=build()
    with pytest.raises(ValueError):run_quality_retries(manager,"job",dict(project_id=project["project_id"],max_quality_retries=limit),initial,lambda:None)
    assert manager.events==[]


def test_already_passing_unselected_candidate_is_suggested_without_more_generation(setup):
    manager,build=setup;project,initial=build(lines=["原稿"],takes=2,quality=lambda key,n:n==1)
    result=run(manager,project,initial)
    assert manager.events==[] and result["rounds"]==[]
    assert len(result["suggested_candidates"])==1 and result["suggested_candidates"][0]["origin"]=="existing"
    assert manager.workspace.get_project(project["project_id"])["lines"][0]["selected_candidate_id"]==project["lines"][0]["selected_candidate_id"]


def test_locked_good_sentence_or_atomic_scene_is_untouched(setup):
    manager,build=setup;project,initial=build(mode="together")
    key=project["lines"][0]["selected_candidate_id"]
    project=manager.workspace.set_candidate_review(project["project_id"],"L1",key,locked=True)
    initial["project_revision"]=project["revision"]
    result=run(manager,project,initial)
    assert manager.events==[] and result["skipped"][0]["reason"]=="locked"
    assert manager.workspace.get_project(project["project_id"])==project


def test_multiline_atomic_unit_gets_only_one_variation_per_round(setup):
    manager,build=setup;project,initial=build(mode="together")
    result=run(manager,project,initial)
    assert len(manager.jobs)==1 and manager.jobs[0]["line_ids"]==["L1","L2"]
    assert len(result["suggested_candidates"])==1 and len(result["rounds"][0]["generated"])==1


def test_two_round_ceiling_when_asr_stays_suspect_and_old_sources_remain_selected(setup):
    manager,build=setup;project,initial=build(lines=["原稿"]);manager.decisions=[False,False]
    result=run(manager,project,initial)
    assert len(result["rounds"])==2 and len(manager.jobs)==2 and result["suggested_candidates"]==[]
    assert len([e for e in manager.events if e[0]=="asr.batch"])==2
    assert [job["seed"] for job in manager.jobs]==[43,44]
    assert len({job["source_candidate"]["candidate_id"] for job in manager.jobs})==1
    assert result["remaining_suspect"]==1 and len(result["unresolved_units"])==1


def test_seed_spans_include_existing_requested_actual_failed_and_uint64_wrap(setup):
    manager,build=setup;project,initial=build(lines=["原稿"],takes=2,sampling={"opening_retries":1},base_seed=2**64-2)
    manager.decisions=[False,True]
    result=run(manager,project,initial)
    assert [job["seed"] for job in manager.jobs]==[2,4]
    assert all(x["reserved_seed_count"]==2 for r in result["rounds"] for x in r["attempts"])
    assert len(result["suggested_candidates"])==1


def test_readout_and_subtitle_overrides_survive_original_text_reperformance(setup):
    manager,build=setup;project,initial=build(lines=[dict(line_id="L1",original_text="Drama3发布",readout_override="抓马三发布",subtitle_override="原字幕")])
    run(manager,project,initial)
    job=manager.jobs[0]
    assert job["readout_text"]=="抓马三发布" and job["subtitle_text"]=="原字幕"
    assert job["operation"]["original_text"]=="Drama3发布" and job["operation"]["note"]==NOTE


def test_structure_errors_are_bounded_and_recorded_as_failures_not_success(setup,monkeypatch):
    manager,build=setup;project,initial=build(lines=["原稿"])
    class SpeechGenerationError(RuntimeError):
        generation_attempts=[dict(seed=43,finish_reason="stalled")]
    monkeypatch.setitem(sys.modules,"drama3_t8.engine",SimpleNamespace(SpeechGenerationError=SpeechGenerationError))
    manager.fail=lambda job:(_ for _ in ()).throw(SpeechGenerationError("silent repetition"))
    result=run(manager,project,initial)
    assert result["status"]=="partial_structure_failure" and result["structure_failures"]==2
    assert len(manager.jobs)==2 and [job["seed"] for job in manager.jobs]==[43,44]
    assert result["suggested_candidates"]==[] and not any(e[0]=="asr.batch" for e in manager.events)
    current=manager.workspace.get_project(project["project_id"])
    assert sum(x["status"]=="failed" for x in current["candidates"].values())==2
    assert current["lines"][0]["selected_candidate_id"]==project["lines"][0]["selected_candidate_id"]


@pytest.mark.parametrize("error",[OSError("disk failure"),MemoryError("no RAM"),RuntimeError("CUDA error"),ValueError("configuration")])
def test_io_memory_cuda_and_configuration_failures_propagate_without_restart(setup,error):
    manager,build=setup;project,initial=build(lines=["原稿"])
    manager.fail=lambda job:(_ for _ in ()).throw(error)
    with pytest.raises(type(error)) as caught:run(manager,project,initial)
    assert caught.value is error and len(manager.jobs)==1 and not any(e[0]=="asr.batch" for e in manager.events)
    assert manager.studio.engine.depth==0


def test_equal_class_name_cannot_disguise_unrelated_runtime_failure(setup,monkeypatch):
    manager,build=setup;project,initial=build(lines=["原稿"])
    real=type("SpeechGenerationError",(RuntimeError,),{});unrelated=type("SpeechGenerationError",(RuntimeError,),{})
    monkeypatch.setitem(sys.modules,"drama3_t8.engine",SimpleNamespace(SpeechGenerationError=real))
    error=unrelated("different identity")
    manager.fail=lambda job:(_ for _ in ()).throw(error)
    with pytest.raises(unrelated) as caught:run(manager,project,initial)
    assert caught.value is error and len(manager.jobs)==1


def test_cancellation_propagates_and_persists_cancelled_variation(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    class Cancelled(RuntimeError):pass
    error=Cancelled("user cancellation")
    manager.fail=lambda job:(_ for _ in ()).throw(error)
    with pytest.raises(Cancelled) as caught:run(manager,project,initial)
    assert caught.value is error and len(manager.jobs)==1 and manager.studio.engine.depth==0
    current=manager.workspace.get_project(project["project_id"])
    assert sum(x["status"]=="cancelled" for x in current["candidates"].values())==1


def test_stale_initial_quality_revision_rejects_draft_change_before_generation(setup):
    manager,build=setup;project,initial=build()
    changed=manager.workspace.update_project(project["project_id"],{"name":"user new draft"})
    with pytest.raises(ProjectConflict,match="首次质检"):run(manager,project,initial)
    assert manager.workspace.get_project(project["project_id"])==changed and manager.events==[]


def test_source_text_no_longer_matches_draft_even_without_revision_receipt(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    current=deepcopy(project["lines"]);current[0]["original_text"]="用户新稿";current[0]["readout_override"]=None
    manager.workspace.update_project(project["project_id"],{"lines":current})
    initial.pop("project_revision")
    with pytest.raises(ProjectConflict,match="草稿不同"):run(manager,project,initial)
    assert manager.events==[] and manager.workspace.get_project(project["project_id"])["lines"][0]["original_text"]=="用户新稿"


def test_real_context_selection_drift_is_rejected(setup):
    manager,build=setup;project,initial=build(takes=2,mode="per_line_context")
    first=project["lines"][0]
    changed=manager.workspace.set_candidate_review(project["project_id"],first["line_id"],first["candidate_ids"][1],selected=True)
    initial["project_revision"]=changed["revision"]
    with pytest.raises(ProjectConflict,match="上下文已变化"):run(manager,project,initial,line_ids=["L2"])
    assert manager.events==[]


def test_valid_real_context_retries_keep_actual_source_session_chain(setup):
    manager,build=setup;project,initial=build(mode="per_line_context")
    result=run(manager,project,initial,line_ids=["L2"])
    assert len(manager.jobs)==1 and len(result["suggested_candidates"])==1
    assert manager.jobs[0]["context_candidate_ids"]==[project["lines"][0]["selected_candidate_id"],project["lines"][1]["selected_candidate_id"]]


def test_after_asr_concurrent_revision_is_rejected_without_overwriting_user_draft(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    manager.after_quality=lambda payload,result:manager.workspace.update_project(payload["project_id"],{"name":"saved concurrent draft"})
    with pytest.raises(ProjectConflict,match="版本"):run(manager,project,initial)
    assert manager.workspace.get_project(project["project_id"])["name"]=="saved concurrent draft"


def test_missing_quality_commit_receipt_cannot_be_reported_as_passed(setup):
    manager,build=setup;project,initial=build(lines=["原稿"]);manager.drop_revision=True
    with pytest.raises(InvalidCheckpoint,match="版本回执"):run(manager,project,initial)
    assert len(manager.jobs)==1 and manager.workspace.get_project(project["project_id"])["lines"][0]["selected_candidate_id"]==project["lines"][0]["selected_candidate_id"]


def test_initial_asr_error_is_not_misclassified_as_generation_issue(setup):
    manager,build=setup;project,initial=build(lines=["原稿"]);initial["errors"]=["some candidate"]
    with pytest.raises(InvalidCheckpoint,match="首次质检存在错误"):run(manager,project,initial)
    assert manager.events==[]


def test_new_quality_errors_propagate_without_second_round(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    original=manager.execute_quality
    def quality(*args):
        result=original(*args);result["errors"]=[result["reviews"].keys().__iter__().__next__()];return result
    manager.execute_quality=quality
    with pytest.raises(InvalidCheckpoint,match="补录质检存在错误"):run(manager,project,initial)
    assert len(manager.jobs)==1 and manager.studio.engine.depth==0


def test_nonindependent_asr_is_not_evidence_to_reperform_again(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    original=manager.execute_quality
    def quality(*args):
        result=original(*args)
        for value in result["reviews"].values():value["independent"]=False
        return result
    manager.execute_quality=quality
    with pytest.raises(InvalidCheckpoint,match="独立转写证据"):run(manager,project,initial)
    assert len(manager.jobs)==1


def test_one_round_limit_is_respected_when_new_asr_stays_suspect(setup):
    manager,build=setup;project,initial=build(lines=["原稿"]);manager.decisions=[False]
    result=run(manager,project,initial,max_quality_retries=1)
    assert len(manager.jobs)==1 and len(result["rounds"])==1 and result["suggested_candidates"]==[]


def test_callback_cannot_change_frozen_target_or_models(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    manager.fail=lambda job:job["lines"][0].update(original_text="changed by engine")
    with pytest.raises(ProjectConflict,match="修改了冻结"):run(manager,project,initial)
    assert len(manager.jobs)==1 and manager.workspace.get_project(project["project_id"])["lines"][0]["original_text"]=="原稿"


def test_actual_seed_outside_reserved_span_cannot_complete(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    original=manager._generate
    def generate(*args):
        result=original(*args);result["seed"]+=1000;return result
    manager._generate=generate
    with pytest.raises(InvalidCheckpoint,match="预留范围"):run(manager,project,initial)
    assert len(manager.jobs)==1


def test_quality_receipt_cannot_claim_a_different_result_than_durable_review(setup):
    manager,build=setup;project,initial=build(lines=["原稿"]);manager.decisions=[False]
    original=manager.execute_quality
    def quality(*args):
        result=original(*args)
        for key in result["reviews"]:result["reviews"][key]=review(True)
        return result
    manager.execute_quality=quality
    with pytest.raises(InvalidCheckpoint,match="已保存质检不一致"):run(manager,project,initial)
    assert len(manager.jobs)==1


def test_new_line_in_atomic_group_invalidates_old_source_snapshot(setup):
    manager,build=setup;project,initial=build(mode="together")
    changed=manager.workspace.update_project(project["project_id"],{"lines":[*project["lines"],dict(line_id="L3",text="new user draft")]})
    initial["project_revision"]=changed["revision"]
    with pytest.raises(ProjectConflict,match="原子单元分组"):run(manager,project,initial)
    assert manager.events==[] and manager.workspace.get_project(project["project_id"])["lines"][-1]["original_text"]=="new user draft"
