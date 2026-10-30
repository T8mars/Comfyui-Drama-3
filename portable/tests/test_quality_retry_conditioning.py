"""Durable independent-review and immutable-reference conditioning safeguards."""
from copy import deepcopy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from drama3_t8.creative_quality import (REFERENCE_ONLY_POLICY,confirmed_independent_mismatch,
    reference_retry_evidence,validate_reference_retry,run_quality_retries)
from drama3_t8.creative_studio import CreativeStudio
from drama3_t8.creative_workspace import CreativeWorkspace,InvalidCheckpoint
from test_creative_quality import setup,media,review,Manager


def reference_session(source):
    evidence=reference_retry_evidence(source)
    return SimpleNamespace(references=tuple(SimpleNamespace(transcript=item["transcript"]) for item in evidence["references"]))


def test_automatic_policy_is_persisted_and_validated_against_real_workspace(setup):
    manager,build=setup;project,initial=build(lines=["保留目标原稿"])
    source=project["candidates"][project["lines"][0]["selected_candidate_id"]]
    original_bytes=Path(source["session_path"]).read_bytes();generate=manager._generate;seen=[]
    def checked(job,directory,progress):
        assert validate_reference_retry(manager.workspace,job,reference_session(source)) is True
        seen.append(deepcopy(job["operation"]))
        return generate(job,directory,progress)
    manager._generate=checked
    result=run_quality_retries(manager,"cpu",dict(project_id=project["project_id"],max_quality_retries=1),initial,lambda *args:None)
    assert len(seen)==1 and seen[0]["conditioning_policy"]==REFERENCE_ONLY_POLICY
    assert seen[0]["quality_source_evidence"]["candidate_id"]==source["candidate_id"]
    current=manager.workspace.get_project(project["project_id"])
    created=current["candidates"][result["rounds"][0]["generated"][0]]
    assert created["variation"]["operation"]["conditioning_policy"]==REFERENCE_ONLY_POLICY
    assert current["lines"][0]["selected_candidate_id"]==source["candidate_id"]
    assert Path(source["session_path"]).read_bytes()==original_bytes


@pytest.mark.parametrize("change",[
    dict(independent=False),dict(independent=1),dict(normalized_match=True),dict(review_recommended=False),
    dict(asr={"text":"保留目标原稿","truncated":False}),dict(asr={"text":"错误录音","truncated":True}),
    dict(asr={"text":"错误录音"},expected="伪造目标"),dict(asr={"text":"错误录音"},recognized="伪造识别文本"),
    dict(asr=["不是转写回执"]),
])
def test_unconfirmed_or_false_labels_never_start_an_automatic_retake(setup,change):
    manager,build=setup;project,initial=build(lines=["保留目标原稿"])
    source_id=project["lines"][0]["selected_candidate_id"]
    answer={**review(False),**change}
    saved=manager.workspace.run_quality(project["project_id"],lambda candidate:answer,candidate_ids=[source_id])
    actual=saved["project"]["candidates"][source_id]
    assert not confirmed_independent_mismatch(actual)
    initial=dict(reviews={source_id:answer},checked=1,project_revision=saved["project"]["revision"])
    result=run_quality_retries(manager,"cpu",dict(project_id=project["project_id"],max_quality_retries=2),initial,lambda *args:None)
    assert manager.jobs==[] and result["rounds"]==[] and result["suggested_candidates"]==[]
    if answer.get("review_recommended") is True:
        assert result["skipped"][0]["reason"]=="independent_asr_mismatch_unconfirmed"


def test_initial_review_cannot_be_forged_separately_from_durable_quality(setup):
    manager,build=setup;project,initial=build(lines=["原稿"])
    key=project["lines"][0]["selected_candidate_id"]
    initial["reviews"][key]["asr"]["text"]="伪造一次新转写"
    with pytest.raises(InvalidCheckpoint,match="已保存质检"):
        run_quality_retries(manager,"cpu",dict(project_id=project["project_id"],max_quality_retries=1),initial,lambda *args:None)
    assert manager.jobs==[]


def test_missing_immutable_reference_skips_without_random_voice_or_fake_pass(tmp_path):
    workspace=CreativeWorkspace(tmp_path/"workspace");manager=Manager(workspace)
    project=workspace.create_project("No voice anchor",["原稿保留"])
    def no_references(job,directory):
        result=media(job,directory);path=Path(result["session_path"])
        raw=path.read_bytes();length=struct.unpack("<Q",raw[:8])[0];header=json.loads(raw[8:8+length])
        data=json.loads(header["__metadata__"]["drama3_t8_session"]);data["references"]=[]
        header["__metadata__"]["drama3_t8_session"]=json.dumps(data)
        header.pop("reference_0")
        encoded=json.dumps(header).encode();encoded+=b" "*(-len(encoded)%8)
        path.write_bytes(struct.pack("<Q",len(encoded))+encoded+raw[8+length:8+length+88])
        return result
    generated=workspace.run_generation(project["project_id"],no_references,model_identity={"test":"fixed"})
    source_id=generated["generated"][0];answer=review(False)
    quality=workspace.run_quality(project["project_id"],lambda candidate:answer,candidate_ids=[source_id])
    before=workspace.get_project(project["project_id"])
    result=run_quality_retries(manager,"cpu",dict(project_id=project["project_id"],max_quality_retries=2),
        dict(reviews={source_id:answer},checked=1,project_revision=before["revision"]),lambda *args:None)
    assert result["skipped"][0]["reason"]=="no_immutable_reference_for_clean_retry"
    assert "先选择音色" in result["message"]
    assert result["remaining_suspect"]==1 and result["suggested_candidates"]==[] and manager.jobs==[]
    assert workspace.get_project(project["project_id"])==before


@pytest.mark.parametrize("tamper",["policy","receipt","independent","reference"])
def test_conditioning_policy_validation_rejects_bypasses(setup,tamper):
    manager,build=setup;project,initial=build(lines=["原稿"])
    source=project["candidates"][project["lines"][0]["selected_candidate_id"]]
    evidence=reference_retry_evidence(source);session=reference_session(source)
    job=dict(project_id=project["project_id"],source_candidate=deepcopy(source),operation=dict(kind="revise",
        conditioning_policy=REFERENCE_ONLY_POLICY,quality_source_evidence=deepcopy(evidence)))
    if tamper=="policy":job["operation"]["conditioning_policy"]="unverified"
    if tamper=="receipt":job["operation"]["quality_source_evidence"]["quality_sha256"]="0"*64
    if tamper=="independent":job["source_candidate"]["quality"]["independent"]=False
    if tamper=="reference":session.references=(SimpleNamespace(transcript="替换的音色参考"),)
    with pytest.raises(InvalidCheckpoint):validate_reference_retry(manager.workspace,job,session)


def test_manual_variation_rejects_reserved_auto_policy_input_before_execution():
    with pytest.raises(ValueError,match="不接受手动"):
        CreativeStudio.execute_variation(SimpleNamespace(),"job",dict(operation={"kind":"revise","conditioning_policy":REFERENCE_ONLY_POLICY}),lambda *args:None)


def test_plain_manual_revision_has_no_forced_reference_only_policy():
    assert validate_reference_retry(None,dict(operation={"kind":"revise"}),None) is False
