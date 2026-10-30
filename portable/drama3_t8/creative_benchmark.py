"""User-triggered Studio benchmark using real generation and blind ASR callbacks.

Synthetic jobs are isolated from project candidates. Timed TTS completes before
shared ASR quality work; settings are restored before any profile is committed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time
from urllib.parse import quote
import uuid

from .performance import (BENCHMARK_SCHEMA, BenchmarkCancelled, BenchmarkCase,
    BenchmarkConfig, BenchmarkOutcome, BenchmarkProfileStore, BenchmarkRun,
    PerformanceRecorder, QualityVerdict, SETTINGS as PERFORMANCE_SETTINGS,
    artifact_identity, run_benchmark)


SHORT = "别紧张，我只是把会议改成了午睡。"
LONG = ("今天开会讨论效率，猫主管宣布，所有会议统一改成午睡。"
        "主持人表示支持，因为大家终于可以带薪做梦。"
        "最后一条规定，梦里不准回复工作消息。"
        "技术同事提出疑问，午睡期间的梦境算不算项目进度。"
        "猫主管认真想了三秒，决定先建一个群，再开一次会讨论。"
        "刚才还在鼓掌的同事全部安静，只有打印机坚持加班。"
        "于是猫主管当场撤回通知，给每个人发了一个罐头，宣布今天的最佳方案就是准时下班。")
REFERENCE = "猫主管宣布，今天不开会，罐头照发。"
DIALOGUE = ("猫主管，今天还开会吗？", "不开了，大家带薪做梦。", "收到，梦里也不回工作消息。")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("."+uuid.uuid4().hex+".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("基准输出文件不属于本次测试目录")
    return path


def _version(package):
    try:return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:return "unavailable"


def _speaker_path(manager):
    value=manager.config.get("speaker_model_path")
    if not value:return None
    return manager.resolve_path(value) if hasattr(manager,"resolve_path") else Path(value)


def _project_voices(manager, project_id):
    from .creative_bundle import project_voice_library
    return project_voice_library(manager.workspace,project_id,manager.voices)


def _strong_artifact(manager, path):
    """Cache only verified full hashes, invalidating on file identity changes."""
    path = Path(path).resolve(strict=True)
    cache_path = Path(manager.reports)/"benchmark-identities.json"
    try:
        cache = json.loads(cache_path.read_text("utf-8")) if cache_path.stat().st_size <= 2*1024*1024 else {}
    except (OSError, ValueError):cache = {}
    stat = path.stat()
    signature = [stat.st_size,stat.st_mtime_ns,stat.st_ino]
    old = cache.get(str(path),{})
    if old.get("signature") == signature and old.get("sha256"):
        return artifact_identity(path,sha256=old["sha256"])
    identity = artifact_identity(path,hash_contents=True)
    cache[str(path)] = dict(signature=signature,sha256=identity["sha256"])
    _write(cache_path,cache)
    return identity


def _binding(manager, models, original, progress):
    progress()
    binding = manager.performance_binding(models)
    binding["models"] = deepcopy(models)
    binding["backends"] = {"comfy-kitchen":_version("comfy-kitchen"),
        "flash-linear-attention":_version("flash-linear-attention"),
        "onnxruntime":_version("onnxruntime"), "asr":original["asr_backend"],
        "scene_head_mode":original.get("scene_head_mode","compact")}
    paths = manager.studio.model_paths
    roles = ("r2t2_model","r2t2_projector") if original["asr_backend"]=="r2t2" else ("asr_model",)
    for role in roles:
        progress()
        binding["models"]["quality_"+role] = _strong_artifact(manager,paths[role])
    if original["asr_backend"]=="r2t2":
        from .r2t2.native import runtime_directory
        binding["models"]["quality_native"] = _strong_artifact(manager,runtime_directory()/"lib/drama_r2t2.dll")
        vad = Path(paths["r2t2_vad"])
        for filename in ("fireredvad_stream_vad_with_cache.onnx","cmvn.ark"):
            if (vad/filename).is_file():
                binding["models"]["quality_vad_"+filename] = _strong_artifact(manager,vad/filename)
    speaker = _speaker_path(manager)
    if speaker:binding["models"]["quality_speaker"] = _strong_artifact(manager,speaker)
    return binding


def current_profile_binding(manager, progress=None):
    """Exact benchmark/profile-lookup binding; candidate settings are excluded.

    Both saving and explicit application must call this helper. Changes to the
    measured MoE/GDN/Graph/cache/placement requests do not expire a profile;
    models, heads, ASR/native/CAM++, source and backend versions still do.
    """
    check=progress or (lambda:None)
    check()
    return _binding(manager,manager.model_identity(strong=True),deepcopy(manager.studio.settings),check)


def _cast_snapshot(manager, project):
    library=_project_voices(manager,project["project_id"])
    roles = list(dict.fromkeys(line["role"] for line in project["lines"]))
    cast, snapshots = {}, []
    for role in roles:
        value = project.get("cast",{}).get(role) or {}
        if not value.get("voice_id") or not value.get("version_id"):continue
        version = library.get_version(value["version_id"])
        if version["voice_id"] != value["voice_id"]:raise ValueError("Cast音色与版本不匹配")
        path = library.session_file(value["version_id"])
        session = library.load_version(value["version_id"])
        if not session.references:raise ValueError("Cast版本没有真实参考音频")
        speaker = value.get("speaker",len(cast))
        if type(speaker) is not int or not 0 <= speaker < 16:raise ValueError("Cast角色编号必须为0至15的整数")
        frozen = dict(voice_id=value["voice_id"],version_id=value["version_id"],speaker=speaker)
        cast[role] = frozen
        snapshots.append(dict(role=role,**frozen,session_path=str(path),session_sha256=_digest(path),
            transcripts=[r.transcript for r in session.references]))
        if len(cast)==2:break
    if len(cast)<2:raise ValueError("完整基准需要工程Cast中至少两个角色绑定真实参考音色版本")
    if len({v["speaker"] for v in cast.values()})<2:raise ValueError("完整基准需要两个不同的角色编号")
    return cast,snapshots


def _make_case(project, models, name, texts, roles, cast, tags, seed, max_frames, mode="per_line"):
    lines = [dict(line_id=f"bench-{name}-{i}",role=role,original_text=text,
        readout_text=text,subtitle_text=text,direction="",reference_group=None,sampling={})
        for i,(role,text) in enumerate(zip(roles,texts))]
    settings = dict(direction="Natural clear Mandarin. Speak only the exact script words, without music or sound effects.",
        scene="A playful office exchange.",characters="，".join(roles),split_long_text=True,chunk_bytes=300)
    sampling = dict(seed=seed,temperature=.7,top_p=.7,top_k=30,max_frames=max_frames,opening_retries=0)
    job = dict(project_id=project["project_id"],unit_index=0,unit_id="benchmark-"+name,
        line_id=lines[0]["line_id"],line_ids=[x["line_id"] for x in lines],line=lines[0],lines=lines,
        role=roles[0],roles=list(dict.fromkeys(roles)),readout_text="\n".join(texts),subtitle_text="\n".join(texts),
        take_index=1,seed=seed,sampling=sampling,cast=deepcopy(cast),settings=settings,
        model_identity=deepcopy(models),mode=mode,context_candidates=[],context_candidate_ids=[],lookahead_lines=[],status="pending")
    return BenchmarkCase(name,job,tuple(tags))


def _cases(project, models, quick, cast):
    short = _make_case(project,models,"short",[SHORT],["旁白"],{},["short"],42,300)
    if quick:return [short]
    roles = list(cast)
    return [short,
        _make_case(project,models,"long",[LONG],[roles[0]],{roles[0]:cast[roles[0]]},["long","reference"],516,900,"together"),
        _make_case(project,models,"reference",[REFERENCE],[roles[1]],{roles[1]:cast[roles[1]]},["reference"],42,300),
        _make_case(project,models,"dialogue",DIALOGUE,[roles[0],roles[1],roles[0]],cast,["multi_speaker","reference"],515,450,"together")]


def _long_chunk_evidence(metadata):
    policy=metadata.get("scene_chunking")
    count=metadata.get("chunk_count")
    return isinstance(policy,dict) and policy.get("enabled") is True and type(count) is int and count>1


def _transcribe(manager, paths, original, progress):
    """No target text, language hint, context or hotword reaches either ASR."""
    from .creative_audio import read_audio_file
    def audios():
        for path in paths:
            progress()
            wave,rate = read_audio_file(path)
            yield {"waveform":wave[None],"sample_rate":rate}
    if original["asr_backend"]=="r2t2":
        from .r2t2.transcribe import transcribe_audio_batch
        p = manager.studio.model_paths
        return transcribe_audio_batch(audios(),p["r2t2_model"],p["r2t2_projector"],p["r2t2_vad"],
            language=None,context="",hotwords="",check_cancel=progress)
    result=[]
    import torch
    for audio in audios():
        progress()
        result.append(manager.studio.engine.transcribe_detailed(torch.from_numpy(audio["waveform"][0]),audio["sample_rate"],
            language=None,context="",hotwords="",check_cancel=progress))
        progress()
    return result


def _quality(manager, root, pairs, original, snapshots, progress, binding):
    from .creative_audio import analyze_audio,read_audio_file
    from .creative_workspace import text_quality
    manager.studio.engine.unload()
    entries,per_job = [],[]
    for case,outcome in pairs:
        job,result = case.payload,outcome.artifact["result"]
        lines = job["lines"]
        parts = result.get("line_audio") or []
        if len(lines)==1 and not parts:parts=[dict(line_id=lines[0]["line_id"],audio_path=result["audio_path"])]
        index = {x["line_id"]:x for x in parts}
        if len(index)!=len(parts) or set(index)!={x["line_id"] for x in lines}:
            raise ValueError("基准缺少完整、无重复的实际逐句音频，不能伪造多人覆盖")
        selected=[]
        for line in lines:
            path=_inside(root,index[line["line_id"]]["audio_path"])
            selected.append(len(entries));entries.append(dict(path=path,expected=line["readout_text"],role=line["role"],line_id=line["line_id"]))
        per_job.append(selected)
    progress()
    observer=PerformanceRecorder({"kind":"benchmark.quality_batch"},binding)
    with observer:
        with observer.phase("asr.batch",cache_state="cold"):
            transcripts=_transcribe(manager,[e["path"] for e in entries],original,progress)
        if len(transcripts)!=len(entries):raise ValueError("ASR批处理条目数与基准音频不一致")
        speaker_path=_speaker_path(manager)
        library=_project_voices(manager,pairs[0][0].payload["project_id"] if pairs else Path(root).parent.parent.name)
        embedder,reference_vectors=None,{}
        if speaker_path and snapshots:
            from .creative_transcript import SpeakerEmbedder
            import numpy as np
            with observer.phase("speaker.load",cache_state="cold"):
                embedder=SpeakerEmbedder(speaker_path)
            for reference in snapshots:
                progress()
                wave,rate=read_audio_file(library.preview_file(reference["version_id"]))
                reference_vectors[reference["role"]]=embedder.embedding(wave,rate)
            if len(reference_vectors)>1:
                vectors=list(reference_vectors.values())
                if float(np.dot(vectors[0],vectors[1]))>=.98:
                    raise ValueError("两份参考声音过于接近，不能把最近参考结果当作角色区分验证")
        analyses=[]
        for entry,answer in zip(entries,transcripts):
            progress()
            wave,rate=read_audio_file(entry["path"]);stats=analyze_audio(wave,rate)
            comparison=text_quality(entry["expected"],answer.get("text",""),independent=True)
            identity=dict(checked=False,scope="No configured reference identity comparison; ASR is not MOS")
            if embedder is not None and entry["role"] in reference_vectors:
                import numpy as np
                vector=embedder.embedding(wave,rate)
                scores={r:float(np.dot(vector,v)) for r,v in reference_vectors.items()}
                nearest=max(scores,key=scores.get)
                identity=dict(checked=True,expected_role=entry["role"],nearest_role=nearest,
                    matched=nearest==entry["role"],cosine=scores,scope="Nearest among these references; not percent similarity or MOS")
            analyses.append(dict(line_id=entry["line_id"],audio_sha256=_digest(entry["path"]),ASR=answer,
                words=comparison,technical=stats,identity=identity))
    verdicts=[]
    for (case,outcome),ids in zip(pairs,per_job):
        result=outcome.artifact["result"];parts=[analyses[i] for i in ids]
        expected={(x["voice_id"],x["version_id"],x["speaker"]) for x in case.payload["cast"].values()}
        observed={(x.get("voice_id"),x.get("version_id"),x.get("speaker",0)) for x in result.get("metadata",{}).get("reference_versions",[])}
        checks=dict(words=all(x["words"]["normalized_match"] for x in parts),
            finite_nonsilent_audio=all(x["technical"]["duration_seconds"]>0 and x["technical"]["peak"]>1e-6 and x["technical"]["clipped_fraction"]<.001 for x in parts),
            ASR_not_truncated=all(not x["ASR"].get("truncated",False) and x["ASR"].get("finish_reason")!="length" for x in parts),
            reference_snapshot=expected==observed,
            seed_unchanged=result.get("seed")==case.payload["seed"],
            generation_ended=result.get("metadata",{}).get("finish_reason")=="eos")
        if "long" in case.tags:
            checks["long_chunked"]=_long_chunk_evidence(result.get("metadata",{}))
        if any(x["identity"]["checked"] for x in parts):
            checks["reference_identity"]=all(x["identity"].get("matched",True) for x in parts)
        verdicts.append(QualityVerdict(all(checks.values()),checks,dict(lines=parts,
            scope="Fixed-case text/technical/reference checks; not universal voice quality or MOS")))
    return verdicts,observer.report()


def _quick(binding,cases,configs,run_case,prepare,quality,warm_repeats,include_cold,progress):
    report=dict(schema=BENCHMARK_SCHEMA,status="running",quick=True,binding=binding,coverage=["short"],
        cases=[dict(id=c.id,payload=c.payload,tags=c.tags) for c in cases],
        configs=[dict(id=c.id,settings=c.settings) for c in configs],results=[],recommendation=None,
        qualification_scope="Short diagnostic only; no automatic profile or general quality claim")
    pending=[]
    try:
        for config in configs:
            case=cases[0]
            for spec in ([BenchmarkRun("cold",0)] if include_cold else [])+[BenchmarkRun("warm",i) for i in range(warm_repeats)]:
                progress();prepare(config,case,spec)
                recorder=PerformanceRecorder(case.payload,binding)
                row=dict(config_id=config.id,case_id=case.id,requested_load_state=spec.requested_load_state,repetition=spec.repetition)
                try:
                    with recorder:outcome=run_case(config,case,recorder,spec)
                except Exception as exc:
                    if isinstance(exc,BenchmarkCancelled):raise
                    row.update(status="error",error_type=type(exc).__name__,diagnostics=recorder.report())
                    report["results"].append(row);report["status"]="error";return report
                elapsed=recorder.report()["elapsed_seconds"]
                row.update(status="ok",elapsed_seconds=elapsed,audio_seconds=outcome.audio_seconds,
                    rtf=elapsed/outcome.audio_seconds,artifact=outcome.artifact,diagnostics=recorder.report())
                report["results"].append(row);pending.append((case,outcome))
        started=time.perf_counter();verdicts=quality(pending)
        if len(verdicts)!=len(pending):raise ValueError("快速基准质量条目数不符")
        for row,v in zip(report["results"],verdicts):row["quality"]=dict(passed=v.passed,checks=dict(v.checks),details=dict(v.details))
        report.update(status="completed",quality_batch_seconds=time.perf_counter()-started)
    except BenchmarkCancelled:report["status"]="cancelled"
    return report


def run_studio_benchmark(manager, job_id, payload, progress):
    """Run isolated real Studio jobs; restore settings before publishing a profile."""
    quick=payload.get("quick",True);repeats=payload.get("warm_repeats",1 if quick else 2);cold=payload.get("include_cold",True)
    if type(quick) is not bool or type(cold) is not bool or type(repeats) is not int or repeats not in (1,2):
        raise ValueError("基准选项格式无效")
    if not quick and repeats<2:raise ValueError("完整基准至少需要两次连续热运行")
    project=manager.workspace.get_project(payload["project_id"])
    project_id=project["project_id"]
    original=deepcopy(manager.studio.settings)
    raw=payload.get("configurations")
    if raw is None:
        base=dict(gdn_backend="torch",expert_placement="ram",expert_cache_gb=float(original.get("expert_cache_gb",8)))
        configs=[BenchmarkConfig("torch-eager",dict(base,moe_backend="torch",cuda_graph=False)),
                 BenchmarkConfig("cuda-graph",dict(base,moe_backend="cuda",cuda_graph=True))]
    else:
        if not isinstance(raw,list) or not 1<=len(raw)<=8:raise ValueError("基准需要1至8个配置")
        requested=[BenchmarkConfig(x["id"],x["settings"]) for x in raw]
        # Freeze every actually inherited performance field in report/profile;
        # applying a qualified profile must not inherit another cache/placement.
        inherited={key:original[key] for key in PERFORMANCE_SETTINGS}
        configs=[BenchmarkConfig(x.id,{**inherited,**x.settings}) for x in requested]
    if len({c.id for c in configs})!=len(configs):raise ValueError("基准配置ID重复")
    cancelled=[]
    def check(*args):
        try:progress(*args)
        except BaseException as exc:
            if type(exc).__name__ in {"Cancelled","InterruptProcessingException","BenchmarkCancelled"}:
                cancelled.append(exc);raise BenchmarkCancelled("基准已取消") from exc
            raise
    check()
    cast,snapshots=({},[]) if quick else _cast_snapshot(manager,project)
    models=manager.model_identity(strong=True)
    binding=current_profile_binding(manager,check)
    if any(binding["models"].get(key)!=models.get(key) for key in ("model","codec","text_head","line_head")):
        raise ValueError("基准初始化期间模型或控制头发生变化")
    from .performance import binding_key
    binding_key(binding,require_strong=True)
    cases=_cases(project,models,quick,cast)
    project_root=(Path(manager.workspace.projects)/project_id).resolve()
    if not project_root.is_relative_to(Path(manager.workspace.projects).resolve()):raise ValueError("工程路径越界")
    directory=project_root/"benchmarks"/uuid.uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    _write(directory/"inputs.json",dict(project_id=project_id,project_revision=project.get("revision"),
        quick=quick,models=models,binding=binding,reference_snapshots=snapshots,
        cases=[dict(id=c.id,payload=c.payload,tags=c.tags) for c in cases]))
    serial=0;active=None;quality_report=None
    def assert_source():
        if manager.source_identity()!=binding["source"]:raise ValueError("运行期间源码变化，请重启Studio后重新测试")
        current=manager.model_identity(strong=False)
        # Compare the selected artifacts, ignoring unrelated optional identity metadata.
        for role in ("model","codec","text_head","line_head"):
            if current.get(role)!=models.get(role):raise ValueError("运行期间模型或控制头发生变化，请重新测试")
        for reference in snapshots:
            if _digest(_project_voices(manager,project_id).session_file(reference["version_id"]))!=reference["session_sha256"]:
                raise ValueError("运行期间参考音色版本发生变化")
    def prepare(config,case,spec):
        nonlocal active
        check();assert_source()
        if active!=config.id:
            manager.studio.apply_settings({**original,**config.settings},persist=False)
            active=config.id
        if spec.requested_load_state=="cold":manager.studio.engine.unload()
    def run_case(config,case,recorder,spec):
        nonlocal serial
        check();serial+=1
        target=directory/f"{serial:04}-{config.id}-{case.id}-{spec.requested_load_state}-{spec.repetition}"
        target.mkdir()
        kwargs={"parent_recorder":recorder} if "parent_recorder" in inspect.signature(manager._generate).parameters else {}
        with recorder.phase("studio.generate",cache_state="warm" if manager.studio.engine.drama is not None else "cold"):
            result=manager._generate(deepcopy(case.payload),target,check,**kwargs)
        check()
        if result.get("complete") is not True:raise ValueError("基准生成没有完成")
        output=_inside(directory,result["audio_path"])
        duration=result.get("duration_seconds")
        if type(duration) not in (float,int) or not 0<float(duration)<86400:raise ValueError("生成音频时长无效")
        import soundfile as sf
        info=sf.info(str(output));measured_duration=info.frames/info.samplerate
        if not math.isclose(float(duration),measured_duration,rel_tol=0,abs_tol=1/info.samplerate):
            raise ValueError("生成元数据时长与实际音频不一致，不能计算有效RTF")
        if "long" in case.tags and not _long_chunk_evidence(result.get("metadata",{})):
            metadata=result.get("metadata",{})
            _write(target/"coverage-error.json",dict(case=case.id,required="scene_chunking.enabled=true and integer chunk_count>1",
                actual_scene_chunking=metadata.get("scene_chunking"),actual_chunk_count=metadata.get("chunk_count")))
            raise ValueError("长文基准未提供实际多块生成证据，拒绝将请求开关当作long覆盖")
        for observed in result.get("metadata",{}).get("performance",{}).get("backends",[]):
            recorder.record_backend(observed["component"],observed.get("requested"),observed.get("actual"),observed.get("fallback_reason"))
        return BenchmarkOutcome(float(measured_duration),dict(result=result,directory=str(target)))
    def quality(pairs):
        nonlocal quality_report
        check();assert_source()
        verdicts,quality_report=_quality(manager,directory,pairs,original,snapshots,check,binding)
        return verdicts
    report=None
    try:
        if quick:report=_quick(binding,cases,configs,run_case,prepare,quality,repeats,cold,check)
        else:report=run_benchmark(binding,cases,configs,run_case,quality_batch_gate=quality,prepare=prepare,
            warm_repeats=repeats,include_cold=cold,cancelled=lambda:(check() or False))
        report.update(quick=quick,project_id=project_id,reference_snapshots=snapshots,quality_performance=quality_report,
            cold_scope="Own engine unloaded in the same process; OS/disk/compiler caches are not reset",
            quality_scope="Independent text/technical/reference checks; not perceptual MOS or a universal quality score")
        assert_source()
    except BaseException as exc:
        _write(directory/"failure.json",dict(error_type=type(exc).__name__,message=str(exc),artifacts_preserved=True))
        raise
    finally:
        manager.studio.apply_settings(original,persist=False)
    metadata=f"/api/creative/projects/{project_id}/files/"+quote((directory/"report.json").relative_to(project_root).as_posix(),safe="/")
    report["profile_saved"]=False
    report["settings_restored"]=True
    _write(directory/"report.json",report)
    if cancelled:
        raise cancelled[0]
    try:check()
    except BenchmarkCancelled:
        raise cancelled[0]
    manager.studio.begin_commit(job_id,progress)
    profile=None
    if not quick and report.get("recommendation") is not None:
        # The profile must qualify the final report bytes, including these
        # bookkeeping flags; mutating it after save would break its evidence SHA.
        report["profile_saved"]=True
        try:profile=BenchmarkProfileStore(Path(manager.reports)/"profiles").save(report)
        except BaseException:
            report["profile_saved"]=False
            _write(directory/"report.json",report)
            raise
    _write(directory/"report.json",report)
    message=("快速检查完成，仅覆盖短句，不保存推荐档案" if quick else
        "完整基准完成，合格推荐已保存，未自动应用" if profile else "基准完成但未产生合格推荐，请查看质量与回退记录")
    if report["status"]!="completed":message="基准执行失败，诊断与已生成音频已保留，未保存推荐"
    return dict(report=report,metadata=metadata,download=metadata,
        recommended=None if quick else report.get("recommendation"),message=message)


__all__=["run_studio_benchmark","current_profile_binding"]
