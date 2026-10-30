"""Opt-in bounded, source-preserving re-performances after blind text review.

This is a soft ASR text gate. It never selects/rates a performance, claims MOS,
changes the script, or treats a structure failure as a successful generation.
"""
from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import json
from pathlib import Path
import struct
import sys

from .creative_workspace import (InvalidCheckpoint, ProjectConflict, _plain_line,
    _selection_fingerprint, fingerprint, text_quality, generation_settings)


NOTE = "保留原稿与角色声音，清晰完整地重新演出，不添加台词"
_UINT64 = 2**64
_SCOPE = "Independent ASR text soft gate; suggestions require audition, not MOS or acting ranking"
REFERENCE_ONLY_POLICY = "reference_only_v1"
_POLICY_FIELDS = {"conditioning_policy", "quality_source_evidence"}


def confirmed_independent_mismatch(source):
    """A label alone is insufficient: compare the saved ASR text to the Take target."""
    quality=source.get("quality") or {}
    if not isinstance(quality,dict):return False
    text=quality.get("text_check") or quality
    asr=quality.get("asr") or {}
    if not isinstance(asr,dict):return False
    if ("independent" in quality and quality["independent"] is not True
            or "normalized_match" in quality and quality["normalized_match"] is not False
            or "review_recommended" in quality and quality["review_recommended"] is not True):
        return False
    expected,heard=source.get("transcript"),asr.get("text")
    if (not isinstance(text,dict) or not isinstance(asr,dict) or not isinstance(expected,str)
            or not expected.strip() or not isinstance(heard,str) or text.get("independent") is not True
            or text.get("normalized_match") is not False or text.get("review_recommended") is not True
            or quality.get("review_error") or text.get("review_error") or asr.get("truncated")):
        return False
    if text_quality(expected,heard,independent=True)["normalized_match"]:
        return False
    if text.get("expected") is not None and (not isinstance(text["expected"],str) or not text_quality(expected,text["expected"])["normalized_match"]):
        return False
    return text.get("recognized") is None or isinstance(text["recognized"],str) and text_quality(heard,text["recognized"])["normalized_match"]


def _reference_summary(source):
    """Inspect immutable native references without loading a model or Torch."""
    path=Path(source["session_path"])
    records=[item for item in source.get("artifacts",[]) if item.get("kind")=="session"]
    if len(records)!=1 or not path.is_file() or path.stat().st_size!=records[0].get("bytes"):
        raise InvalidCheckpoint("补录源缺少唯一、未变的原生Session")
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
        checksum=digest.hexdigest()
        if checksum!=records[0].get("sha256"):raise InvalidCheckpoint("补录源Session校验不符")
        handle.seek(0)
        prefix=handle.read(8)
        if len(prefix)!=8:raise InvalidCheckpoint("补录源Session头损坏")
        length=struct.unpack("<Q",prefix)[0]
        if not 1<=length<=16*1024**2:raise InvalidCheckpoint("补录源Session头超出范围")
        header=json.loads(handle.read(length))
    data=json.loads(header.get("__metadata__",{}).get("drama3_t8_session","null"))
    if not isinstance(data,dict) or not isinstance(data.get("references"),list):
        raise InvalidCheckpoint("补录源Session缺少参考记录")
    references=[]
    for reference in data["references"]:
        if not isinstance(reference,dict):raise InvalidCheckpoint("补录音色参考元数据无效")
        tensor=header.get(reference.get("key"),{})
        shape=tensor.get("shape",[])
        if (not isinstance(reference.get("transcript"),str) or not reference["transcript"].strip()
                or len(shape)!=2 or shape[0]!=10 or type(shape[1]) is not int or shape[1]<1
                or tensor.get("dtype") not in {"I64","I32","I16","I8","U8"}):
            raise InvalidCheckpoint("补录音色参考不是有效的不可变编码")
        references.append(dict(transcript=reference["transcript"],key=reference["key"],shape=shape))
    return dict(session_sha256=checksum,references=references)


def reference_retry_evidence(source):
    if not confirmed_independent_mismatch(source):
        raise InvalidCheckpoint("补录必须有已保存的独立ASR读稿不符证据")
    summary=_reference_summary(source)
    if not summary["references"]:
        return None
    return dict(candidate_id=source["candidate_id"],quality_sha256=fingerprint(source["quality"]),**summary)


def validate_reference_retry(workspace,job,session):
    """Validate an internal auto-retake policy against the durable source record.

    HTTP/manual variation inputs cannot supply this policy. The second check
    here also rejects forged labels/evidence before model loading.
    """
    operation=job.get("operation") or {}
    if not _POLICY_FIELDS.intersection(operation):return False
    if operation.get("kind")!="revise" or operation.get("conditioning_policy")!=REFERENCE_ONLY_POLICY:
        raise InvalidCheckpoint("无效的自动质检补录条件策略")
    source=job["source_candidate"]
    current=workspace.get_project(job["project_id"])["candidates"].get(source["candidate_id"])
    if (current is None or current.get("status")!="completed"
            or Path(source["session_path"]).resolve()!=Path(current["session_path"]).resolve()
            or fingerprint(current["input_snapshot"])!=fingerprint(source["input_snapshot"])):
        raise ProjectConflict("自动补录源与当前工程不一致")
    expected=reference_retry_evidence(current)
    if expected is None or operation.get("quality_source_evidence")!=expected:
        raise InvalidCheckpoint("自动补录条件证据与已保存质检或参考不符")
    if (not confirmed_independent_mismatch(source) or fingerprint(source.get("quality"))!=expected["quality_sha256"]
            or [item.transcript for item in session.references]!=[item["transcript"] for item in expected["references"]]):
        raise InvalidCheckpoint("自动补录条件标签、参考或独立质检未验证")
    return True


def _review(candidate, reviews):
    return {**candidate.get("quality",{}),**reviews.get(candidate["candidate_id"],{})}


def _passed(candidate, reviews):
    quality=_review(candidate,reviews)
    text=quality.get("text_check") or quality
    asr=quality.get("asr") or {}
    return (candidate.get("status")=="completed" and text.get("normalized_match") is True
        and text.get("independent",quality.get("independent")) is True
        and text.get("review_recommended") is False and not quality.get("review_error")
        and asr.get("truncated") is not True and asr.get("finish_reason")!="length")


def _editor_key(project):
    return fingerprint(dict(name=project.get("name"),lines=[dict(line=_plain_line(line),
        selected_candidate_id=line.get("selected_candidate_id"),locked=line.get("locked",False)) for line in project["lines"]],
        cast=project.get("cast",{}),settings=project.get("settings",{}),dictionary=project.get("dictionary"),
        archived_lines=project.get("archived_lines",[])))


def _frozen_key(candidate):
    return fingerprint({k:v for k,v in candidate.items() if k not in ("rating","favorite","notes")})


def _is_speech_error(exc):
    # Do not import/initialize the host merely to classify an error, and never
    # swallow a different class just because it happens to share this name.
    module=sys.modules.get(__package__+".engine")
    expected=getattr(module,"SpeechGenerationError",None)
    return isinstance(expected,type) and isinstance(exc,expected)


def _uint(value):
    if type(value) is not int or not 0<=value<_UINT64:raise ValueError("候选种子记录无效")
    return value


def _retry_span(candidate):
    value=candidate["input_snapshot"]["sampling"].get("opening_retries",0)
    if type(value) is not int or value not in (0,1):raise ValueError("开头重试预留必须为0或1")
    return value+1


def _next_seed(project, source):
    """Reserve the entire requested opening-retry span, including failed takes."""
    lines=set(source["line_ids"]);used=set()
    for candidate in project["candidates"].values():
        if not lines.intersection(candidate.get("line_ids",[])):continue
        span=_retry_span(candidate)
        sampling=candidate["input_snapshot"]["sampling"]
        requested=candidate.get("requested_seed",sampling.get("seed",candidate.get("seed")))
        requested=_uint(requested)
        used.update((requested+i)%_UINT64 for i in range(span))
        for value in (candidate.get("seed"),sampling.get("seed")):
            if value is not None:used.add(_uint(value))
        for attempt in candidate.get("metadata",{}).get("generation_attempts",[]):
            used.add(_uint(attempt["seed"]))
    span=_retry_span(source)
    seed=(_uint(source.get("requested_seed",source["seed"]))+span)%_UINT64
    for _ in range(span*len(used)+1):
        if not any((seed+i)%_UINT64 in used for i in range(span)):return seed,span
        seed=(seed+1)%_UINT64
    raise ValueError("未找到可用且有界的补录种子范围")


def _verify_media(manager, project, candidate, cache, progress):
    if candidate.get("status")!="completed":raise InvalidCheckpoint("补录需要完整的源候选")
    root=(Path(manager.workspace.projects)/project["project_id"]).resolve()
    kinds=set()
    for artifact in candidate.get("artifacts",[]):
        progress();path=(root/artifact["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise InvalidCheckpoint("源候选媒体不存在或越界")
        stat=path.stat();signature=(str(path),stat.st_size,stat.st_mtime_ns,stat.st_ino,artifact["sha256"])
        if stat.st_size!=artifact["bytes"]:raise InvalidCheckpoint("源候选媒体长度发生变化")
        if signature not in cache:
            digest=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):
                    progress();digest.update(block)
            after=path.stat()
            if ((after.st_size,after.st_mtime_ns,after.st_ino)!=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
                    or digest.hexdigest()!=artifact["sha256"]):raise InvalidCheckpoint("源候选媒体校验失败")
            cache.add(signature)
        kinds.add(artifact["kind"])
    if not {"audio","session"}<=kinds:raise InvalidCheckpoint("源候选缺少实际WAV或Session")


def _assert_runtime(manager, inputs):
    frozen=inputs.get("model_identity",{})
    if hasattr(manager,"model_identity"):
        current=manager.model_identity(strong=False)
        for role in ("model","codec","text_head","line_head"):
            if role not in frozen:continue
            old,new=frozen[role],current.get(role)
            if old is None:
                if new is not None:raise ProjectConflict("源候选的控制头选择与当前模型不同")
            elif not isinstance(old,dict) or not isinstance(new,dict) or any(old.get(k)!=new.get(k) for k in ("sha256","bytes")):
                raise ProjectConflict("源候选模型或Codec与当前权重不同")
    runtime=frozen.get("runtime",{})
    if runtime.get("source") is not None and hasattr(manager,"source_identity"):
        if runtime["source"]!=manager.source_identity():raise ProjectConflict("源候选运行源码已变化，请重新生成后再补录")
    if runtime.get("settings") is not None and hasattr(manager,"studio"):
        if any(manager.studio.settings.get(k)!=v for k,v in runtime["settings"].items()):
            raise ProjectConflict("源候选运行设置已变化，请重新生成后再补录")
    if runtime.get("transformers") is not None:
        import importlib.metadata
        if importlib.metadata.version("transformers")!=runtime["transformers"]:
            raise ProjectConflict("源候选Transformers版本已变化")


def _assert_source(manager, project, source, cache, progress):
    inputs=source["input_snapshot"];lines=project["lines"]
    by_id={x["line_id"]:x for x in lines};ids=source["line_ids"]
    if not ids or any(key not in by_id for key in ids):raise ProjectConflict("源候选原子单元已不属于当前工程")
    if any(by_id[key].get("selected_candidate_id")!=source["candidate_id"] or by_id[key].get("locked") for key in ids):
        raise ProjectConflict("源候选已被改选或锁定，拒绝自动补录")
    if [_plain_line(by_id[key]) for key in ids]!=inputs["lines"]:
        raise ProjectConflict("源候选文本、朗读稿或字幕已与当前草稿不同")
    cast={x["role"]:project.get("cast",{}).get(x["role"]) for x in inputs["lines"]}
    if (cast!=inputs["cast"] or generation_settings(project["settings"])!=generation_settings(inputs["settings"])
            or project["dictionary"]["version"]!=inputs["dictionary_version"]):
        raise ProjectConflict("源候选选角、场景或读法字典已变化")
    indexes=[next(i for i,line in enumerate(lines) if line["line_id"]==key) for key in ids]
    mode=inputs["mode"];dependencies=[]
    expected_ids=([x["line_id"] for x in lines] if mode=="together" else
        [x["line_id"] for x in lines if x["role"]==by_id[ids[0]]["role"]] if mode=="per_speaker" else [ids[0]])
    if ids!=expected_ids:raise ProjectConflict("源候选原子单元分组已变化")
    policy=("own_first_latest_and_previous_v1" if mode in ("per_line_context","text_stream") else
        "first_anchor_v1" if mode=="long_form" else "independent_v1")
    if inputs.get("context_policy")!=policy:raise ProjectConflict("源候选上下文策略不匹配")
    if mode in ("per_line_context","text_stream"):
        if len(ids)!=1:raise ProjectConflict("逐句上下文源不是独立原子单元")
        index=indexes[0];own=[i for i in range(index) if lines[i]["role"]==by_id[ids[0]]["role"]]
        dependencies=sorted(set(([own[0],own[-1]] if own else [])+([index-1] if index else [])))
    elif mode=="long_form" and indexes[0]:dependencies=[0]
    elif mode not in ("per_line","together","per_speaker","long_form"):
        raise ProjectConflict("源候选生成模式未知")
    # Match the workspace's documented own-first/latest/previous or first-anchor
    # policy against actual selected context, not a newly invented substitute.
    observed=inputs.get("context",[])
    if len(observed)!=len(dependencies):raise ProjectConflict("源候选上下文单元已变化")
    for record,index in zip(observed,dependencies):
        key=lines[index].get("selected_candidate_id");candidate=project["candidates"].get(key)
        if candidate is None or record.get("candidate_id")!=key or record.get("fingerprint")!=candidate["fingerprint"]:
            raise ProjectConflict("源候选的已选上下文已变化")
        kind="conditioning" if candidate.get("conditioning_session_path") else "session"
        actual=next((a["sha256"] for a in candidate["artifacts"] if a["kind"]==kind),None)
        if record.get("session_sha256")!=actual:raise ProjectConflict("源上下文Session身份不匹配")
        if kind=="conditioning" and record.get("source_session_sha256")!=next((a["sha256"] for a in candidate["artifacts"] if a["kind"]=="session"),None):
            raise ProjectConflict("源上下文原生Session身份不匹配")
    look=inputs.get("lookahead_lines",[])
    if look!=[_plain_line(x) for x in lines[max(indexes)+1:max(indexes)+1+len(look)]]:
        raise ProjectConflict("源候选前瞻草稿已变化")
    _assert_runtime(manager,inputs)
    queue=[source["candidate_id"]];seen=set()
    while queue:
        key=queue.pop()
        if key in seen:continue
        seen.add(key);candidate=project["candidates"].get(key)
        if candidate is None:raise InvalidCheckpoint("源候选上下文链缺失")
        _verify_media(manager,project,candidate,cache,progress)
        queue.extend(candidate.get("context_candidate_ids",[]))


def _suggestion(source, candidate, round_index):
    return dict(source_candidate_id=source["candidate_id"],candidate_id=candidate["candidate_id"],
        line_ids=list(source["line_ids"]),round=round_index,origin="existing" if round_index==0 else "retry",
        asr_text_passed=True,selected=False,scope=_SCOPE)


def run_quality_retries(manager, job_id, payload, initial_quality, progress, *, progress_scope=None):
    """At most two real generation→shared-review rounds; selection stays human."""
    limit=payload.get("max_quality_retries",0)
    if type(limit) is not int or not 0<=limit<=2:raise ValueError("疑句补录次数必须为0至2的整数")
    if not isinstance(initial_quality,dict):raise ValueError("首次质检结果必须为对象")
    if initial_quality.get("errors"):raise InvalidCheckpoint("首次质检存在错误，请先修复转写或配置后再补录")
    reviews=deepcopy(initial_quality.get("reviews") or {})
    if not isinstance(reviews,dict) or any(not isinstance(k,str) or not isinstance(v,dict) for k,v in reviews.items()):
        raise ValueError("首次质检reviews无效")
    project_id=payload["project_id"];cancelled=[False]
    def check(*args):
        try:progress(*args)
        except BaseException as exc:
            if type(exc).__name__ in {"Cancelled","InterruptProcessingException","KeyboardInterrupt"}:cancelled[0]=True
            raise
    check();project=manager.workspace.get_project(project_id);revision=project["revision"]
    if initial_quality.get("project_revision",revision)!=revision:raise ProjectConflict("首次质检后工程已变化，请刷新后重试")
    for key,answer in reviews.items():
        current=project["candidates"].get(key)
        if current is None or any(current.get("quality",{}).get(field)!=value for field,value in answer.items()):
            raise InvalidCheckpoint("首次质检回执与已保存质检不一致，拒绝使用伪造标签补录")
    editor=_editor_key(project);cache=set();frozen={};skipped=[];suggested=[];pending=[];rounds=[]
    scope_ids=payload.get("line_ids")
    if scope_ids is not None and (not isinstance(scope_ids,list) or any(not isinstance(x,str) for x in scope_ids)):
        raise ValueError("补录台词范围无效")
    selected=list(dict.fromkeys(x.get("selected_candidate_id") for x in project["lines"] if x.get("selected_candidate_id")))
    by_id={x["line_id"]:x for x in project["lines"]}
    if scope_ids is not None and any(x not in by_id for x in scope_ids):raise ValueError("补录台词范围不属于当前工程")
    for key in selected:
        source=project["candidates"].get(key)
        if source is None or source.get("status")!="completed" or _review(source,reviews).get("review_recommended") is not True:continue
        if _review(source,reviews).get("review_error"):raise InvalidCheckpoint("所选候选质检失败，不能将转写错误当作模型疑句")
        if not confirmed_independent_mismatch(source):
            skipped.append(dict(source_candidate_id=key,line_ids=source["line_ids"],reason="independent_asr_mismatch_unconfirmed",
                message="没有已验证的独立ASR读稿不符证据，已跳过自动补录，请先重新质检或人工复核"))
            continue
        ids=source["line_ids"]
        if scope_ids is not None and not set(ids).intersection(scope_ids):continue
        reason=("locked" if any(by_id.get(i,{}).get("locked") for i in ids) else
            "partial_atomic_selection" if any(by_id.get(i,{}).get("selected_candidate_id")!=key for i in ids) else None)
        if reason:skipped.append(dict(source_candidate_id=key,line_ids=ids,reason=reason));continue
        _assert_source(manager,project,source,cache,check)
        compatible=[]
        for other in project["candidates"].values():
            if (other["candidate_id"]!=key and other.get("line_ids")==ids and _passed(other,reviews)
                    and _selection_fingerprint(other["input_snapshot"])==_selection_fingerprint(source["input_snapshot"])):
                _verify_media(manager,project,other,cache,check);compatible.append(other)
        if compatible:
            suggested.extend(_suggestion(source,other,0) for other in compatible)
            skipped.append(dict(source_candidate_id=key,line_ids=ids,reason="passing_candidates_available"));continue
        if reference_retry_evidence(source) is None:
            skipped.append(dict(source_candidate_id=key,line_ids=ids,reason="no_immutable_reference_for_clean_retry",
                message="缺少可用的不可变音色参考；请先选择音色后补录，原候选与选择保留"))
            continue
        pending.append(key)
        queue=[key]
        while queue:
            candidate_key=queue.pop()
            if candidate_key in frozen:continue
            candidate=project["candidates"][candidate_key];frozen[candidate_key]=_frozen_key(candidate)
            queue.extend(candidate.get("context_candidate_ids",[]))
    def refresh(expected):
        current=manager.workspace.get_project(project_id)
        if current["revision"]!=expected or _editor_key(current)!=editor:
            raise ProjectConflict("补录期间工程版本、选角、选择或草稿已变化，已生成候选保留")
        for key,value in frozen.items():
            if key not in current["candidates"] or _frozen_key(current["candidates"][key])!=value:
                raise ProjectConflict("补录期间源候选或上下文发生变化")
            _verify_media(manager,current,current["candidates"][key],cache,check)
        return current
    for number in range(1,limit+1):
        if not pending:break
        check();project=refresh(revision);record=dict(round=number,attempts=[],generated=[],reviews={},scope=_SCOPE)
        rounds.append(record)
        round_scope = progress_scope.round(number, tuple(pending)) if progress_scope is not None else None
        engine=getattr(getattr(manager,"studio",None),"engine",None)
        session=getattr(engine,"generation_session",None)
        with session() if callable(session) else contextlib.nullcontext():
            for source_id in list(pending):
                check();project=refresh(revision);source=project["candidates"][source_id]
                _assert_source(manager,project,source,cache,check)
                seed,span=_next_seed(project,source);line=source["input_snapshot"]["lines"][0]
                operation=dict(kind="revise",text=line["original_text"],original_text=line["original_text"],note=NOTE,
                    readout_override=line.get("readout_override"),subtitle_text=line.get("subtitle_override"),
                    conditioning_policy=REFERENCE_ONLY_POLICY,quality_source_evidence=reference_retry_evidence(source))
                if operation["quality_source_evidence"] is None:raise InvalidCheckpoint("补录前音色参考已不可用，请先选择音色")
                before_ids=set(project["candidates"]);before_revision=revision
                attempt=dict(source_candidate_id=source_id,line_ids=list(source["line_ids"]),seed=seed,reserved_seed_count=span)
                record["attempts"].append(attempt)
                def generate(job,directory):
                    check()
                    if (job["lines"]!=source["input_snapshot"]["lines"] or job["model_identity"]!=source["input_snapshot"]["model_identity"]
                            or job["sampling"]!={**source["input_snapshot"]["sampling"],"seed":seed}):
                        raise ProjectConflict("补录回调偏离已冻结的原稿或模型输入")
                    identity=fingerprint(job)
                    unit_progress = round_scope.start(source_id) if round_scope is not None else check
                    try:result=manager._generate(job,directory,unit_progress)
                    except BaseException as exc:
                        if type(exc).__name__ in {"Cancelled","InterruptProcessingException","KeyboardInterrupt"}:cancelled[0]=True
                        raise
                    if fingerprint(job)!=identity:raise ProjectConflict("生成回调修改了冻结补录输入")
                    if not isinstance(result,dict) or result.get("seed") not in {(seed+i)%_UINT64 for i in range(span)}:
                        raise InvalidCheckpoint("补录实际种子越出已预留范围")
                    check();return result
                try:
                    result=manager.workspace.run_variation(project_id,line["line_id"],source_id,generate,operation,
                        sampling={"seed":seed},model_identity=deepcopy(source["input_snapshot"]["model_identity"]),
                        expected_revision=revision,cancelled=lambda:cancelled[0])
                except Exception as exc:
                    if not _is_speech_error(exc):raise
                    # Current workspace contract saves the running variation and its
                    # failed result exactly once each; any additional revision is a
                    # concurrent edit and must never be accepted as our own write.
                    revision=before_revision+2;project=refresh(revision)
                    failed=set(project["candidates"])-before_ids
                    if (len(failed)!=1 or any(project["candidates"][key].get("status")!="failed" or
                        project["candidates"][key].get("variation",{}).get("source_candidate_id")!=source_id for key in failed)):
                        raise ProjectConflict("结构失败没有对应唯一失败候选记录") from exc
                    attempt.update(status="structure_failed",failed_candidate_ids=sorted(failed),error_type=type(exc).__name__,
                        message=str(exc)[:1500],generation_attempts=deepcopy(getattr(exc,"generation_attempts",[])))
                    if round_scope is not None:round_scope.complete(source_id)
                    continue
                if result.get("status")!="completed" or len(result.get("generated",[]))!=1 or result.get("reused"):
                    raise InvalidCheckpoint("本轮补录没有产生一个新的完整候选")
                revision=result["project"]["revision"];project=refresh(revision)
                key=result["generated"][0];attempt.update(status="generated",candidate_id=key);record["generated"].append(key)
                if round_scope is not None:round_scope.complete(source_id)
        if round_scope is not None:round_scope.generation_done()
        if record["generated"]:
            check();project=refresh(revision)
            quality_progress = round_scope.quality.start() if round_scope is not None else check
            quality=manager.execute_quality(job_id,dict(project_id=project_id,candidate_ids=list(record["generated"])),quality_progress)
            if not isinstance(quality,dict) or type(quality.get("project_revision")) is not int:
                raise InvalidCheckpoint("补录质检缺少实际提交版本回执")
            if quality.get("errors"):raise InvalidCheckpoint("补录质检存在错误，停止自动补录并保留候选")
            revision=quality["project_revision"];project=refresh(revision)
            answers=quality.get("reviews") or {}
            if not isinstance(answers,dict) or set(answers)!=set(record["generated"]) or type(quality.get("checked")) is not int or quality["checked"]!=len(record["generated"]):
                raise InvalidCheckpoint("补录质检未独立检查全部新候选")
            if any((v.get("text_check") or v).get("independent",v.get("independent")) is not True for v in answers.values()):
                raise InvalidCheckpoint("补录质检缺少独立转写证据")
            if any(any(project["candidates"][key].get("quality",{}).get(field)!=value for field,value in answer.items())
                    for key,answer in answers.items()):
                raise InvalidCheckpoint("补录质检回执与已保存质检不一致")
            if round_scope is not None:round_scope.quality.finish()
            reviews.update(deepcopy(answers));record["reviews"]=deepcopy(answers)
            for key in record["generated"]:
                candidate=project["candidates"][key];source_id=candidate["variation"]["source_candidate_id"]
                if _passed(candidate,reviews):
                    suggested.append(_suggestion(project["candidates"][source_id],candidate,number))
                    if source_id in pending:pending.remove(source_id)
        if round_scope is not None and not record["generated"]:round_scope.quality.finish(skipped=True)
        record["status"]="completed_with_structure_failures" if any(x["status"]=="structure_failed" for x in record["attempts"]) else "completed"
    project=refresh(revision);remaining=[]
    for key in selected:
        candidate=project["candidates"].get(key)
        if candidate and _review(candidate,reviews).get("review_recommended") is True:
            remaining.append(dict(source_candidate_id=key,line_ids=list(candidate["line_ids"])))
    failures=sum(x["status"]=="structure_failed" for r in rounds for x in r["attempts"])
    if progress_scope is not None:progress_scope.finish()
    missing_references=sum(item["reason"]=="no_immutable_reference_for_clean_retry" for item in skipped)
    message="补录仅提供转写通过的候选建议；原选择与原稿保留，请试听后明确选入成片"
    if missing_references:message+=f"；{missing_references}条缺少不可变音色参考已跳过，请先选择音色后补录"
    return dict(status="partial_structure_failure" if failures else "completed",structure_failures=failures,
        max_quality_retries=limit,rounds=rounds,reviews=reviews,suggested_candidates=suggested,
        remaining_suspect=len(remaining),remaining_suspect_units=remaining,unresolved_units=list(pending),skipped=skipped,
        project_revision=revision,selection_changed=False,scope=_SCOPE,
        message=message)


__all__=["run_quality_retries"]
