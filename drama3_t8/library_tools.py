"""CPU-only voice compatibility descriptions and explicit immutable re-encoding."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid

from .workspace_lock import WorkspaceLock

_LOCK=threading.RLock()
_HEX=re.compile(r"[0-9a-f]{64}\Z")


def _hash(path, progress=None):
    result=hashlib.sha256();progress=progress or (lambda:None)
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            progress();result.update(block)
    return result.hexdigest()


def _identity(value):
    return (isinstance(value,dict) and _HEX.fullmatch(str(value.get("sha256",""))) is not None
        and type(value.get("bytes")) is int and value["bytes"]>0)


def _encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def _atomic(path, value):
    temporary=path.with_name("."+uuid.uuid4().hex+".pending")
    temporary.write_bytes(_encoded(value));temporary.replace(path)


def _codec(path, root):
    try:path=Path(path).expanduser().resolve()
    except (OSError,ValueError,TypeError):return None,"当前Codec路径不可用"
    if not path.is_file() or path.stat().st_size==0:return None,"当前Codec文件缺失或为空"
    from .reference_cache import codec_identity
    try:return codec_identity(path,root),None
    except (OSError,ValueError,KeyError,TypeError) as exc:
        # A metadata-cache failure must not turn a readable Codec into a false
        # missing-model report. This fallback hashes ONLY the Codec file.
        before=path.stat();digest=_hash(path);after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            return None,"Codec文件在核对期间变化"
        return dict(sha256=digest,bytes=after.st_size),"Codec身份缓存不可用，已直接核对文件："+type(exc).__name__


def _valid_metadata(value):
    if not isinstance(value,dict) or not isinstance(value.get("references"),list) or not value["references"]:return False
    for ref in value["references"]:
        if (not isinstance(ref,dict) or type(ref.get("code_frames")) is not int or ref["code_frames"]<=0
            or not isinstance(ref.get("transcript"),str) or not isinstance(ref.get("processing"),dict)):return False
    try:_encoded(value)
    except (ValueError,TypeError):return False
    return True


def _metadata(library, version, entries):
    # Always perform the library's real artifact SHA check. The metadata cache
    # never replaces validation on a later actual voice load.
    library.session_file(version["id"])
    key=version["content_sha256"];cached=entries.get(key)
    if (isinstance(cached,dict) and _valid_metadata(cached.get("data"))
        and cached.get("integrity")==hashlib.sha256(_encoded(cached["data"])).hexdigest()):
        return deepcopy(cached["data"]),"checksum+cached_cpu_validation"
    session=library.load_version(version["id"])
    if not session.references or session.takes:raise ValueError("音色版本不是纯参考Session")
    refs=[]
    for reference in session.references:
        if reference.codes.device.type!="cpu":raise ValueError("状态读取只允许CPU Session")
        refs.append(dict(transcript=reference.transcript,name=reference.name,codec_revision=reference.codec_revision,
            code_frames=int(reference.codes.shape[-1]),processing=deepcopy(reference.processing)))
    data=dict(references=refs)
    entries[key]=dict(data=data,integrity=hashlib.sha256(_encoded(data)).hexdigest())
    return deepcopy(data),"checksum+cpu_session_validation"


def _cache_entry(root, saved, codec):
    value=dict(last_result="unknown",key=None,entry_valid=False,encoded=True,
        scope="Saved result of the last encoding; entry validity is not a new encode/cache-hit measurement")
    if not isinstance(saved,dict):return value
    if saved.get("status") in {"hit","miss"}:value["last_result"]=saved["status"]
    key=saved.get("key")
    if not isinstance(key,str) or not _HEX.fullmatch(key):return value
    value["key"]=key
    receipt,archive=Path(root)/(key+".json"),Path(root)/(key+".safetensors")
    try:
        if not receipt.is_file() or not archive.is_file() or receipt.stat().st_size>1024*1024:return value
        data=json.loads(receipt.read_text(encoding="utf-8"));binding=data["binding"]
        if (hashlib.sha256(_encoded(binding)).hexdigest()!=key or binding.get("codec")!=codec
                or data.get("sha256")!=_hash(archive)):return value
        value["entry_valid"]=True
    except (OSError,ValueError,KeyError,TypeError):pass
    return value


def describe_voice_versions(library, voices, codec_path, cache_root):
    """Return copied catalog rows; no engine or main model is read/loaded.

    ready means that native CPU Session codes have a known matching Codec file
    identity. It is neither a speech-generation test nor a perceptual score.
    """
    if not isinstance(voices,(list,tuple)):raise ValueError("音色目录必须为列表")
    result=deepcopy(list(voices));root=Path(cache_root)
    codec,error=_codec(codec_path,root)
    try:root.mkdir(parents=True,exist_ok=True)
    except OSError:pass
    index=root/"voice-status-metadata.json"
    try:
        saved=json.loads(index.read_text(encoding="utf-8")) if index.is_file() and index.stat().st_size<=4*1024*1024 else {}
        entries=saved.get("entries",{}) if saved.get("schema")==1 and isinstance(saved.get("entries"),dict) else {}
    except (OSError,ValueError,TypeError):entries={}
    with _LOCK:
        for voice in result:
            for version in voice.get("versions",[]):
                try:
                    actual=library.get_version(version["id"])
                    if actual["voice_id"]!=voice["id"]:raise ValueError("版本与音色不匹配")
                    metadata,verification=_metadata(library,actual,entries)
                    refs=metadata["references"];known=[r["processing"].get("codec_identity") for r in refs]
                    incompatible=codec is not None and any(_identity(x) and (x["sha256"],x["bytes"])!=(codec["sha256"],codec["bytes"]) for x in known)
                    unknown=any(not _identity(x) for x in known)
                    status=("codec_missing" if codec is None else "incompatible" if incompatible else
                        "legacy_unknown" if unknown else "ready")
                    reason={"ready":"已核对编码与当前Codec身份一致","legacy_unknown":"旧版缺少完整Codec身份，需要显式重新编码后验证",
                        "incompatible":"编码来自不同Codec，请显式重编码创建新版本","codec_missing":"当前Codec文件不可用，无法核对兼容性"}[status]
                    caches=[_cache_entry(root,r["processing"].get("encoding_cache"),r["processing"].get("codec_identity")) for r in refs]
                    last={x["last_result"] for x in caches};last_result=next(iter(last)) if len(last)==1 else "mixed"
                    cache=dict(status=last_result,last_result=last_result,encoded=True,
                        entry_valid=all(x["entry_valid"] for x in caches),references=caches,
                        scope="Persisted last encode hit/miss and separately verified stored entries; no new encode performed")
                    description=dict(status=status,verified=status=="ready",encoded=True,reason=reason,
                        current_codec=codec,codec_note=error,reference_count=len(refs),session_verification=verification,
                        scope="Codec identity compatibility, not actual synthesis or voice quality",encoding_cache=cache)
                except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                    status="corrupt";cache=dict(status="unknown",last_result="unknown",encoded=False,entry_valid=False)
                    description=dict(status=status,verified=False,encoded=False,reason="参考Session不可验证："+type(exc).__name__,
                        current_codec=codec,encoding_cache=cache,scope="Corrupt or unavailable native reference artifact")
                version.update(status=status,encoding_status=status,encoding=description,encoding_cache=cache)
            latest=next((v for v in voice.get("versions",[]) if v["id"]==voice.get("latest_version_id")),None)
            if latest is not None:voice["encoding_status"]=latest["status"]
        # Best-effort metadata acceleration only. Actual use still goes through
        # VoiceLibrary's full checksum/native Session validation independently.
        try:
            data=dict(schema=1,entries=entries)
            if len(_encoded(data))<=4*1024*1024:
                with closing(WorkspaceLock(root/"voice-status-metadata.lock")):_atomic(index,data)
        except (OSError,ValueError,RuntimeError):pass
    return result


def _audio(path):
    import torch
    from .creative_audio import read_audio_file
    try:
        wave,rate=read_audio_file(path)
        return torch.from_numpy(wave.copy()),rate
    except Exception as exc:
        import soundfile as sf
        if not isinstance(exc,sf.LibsndfileError):raise
        from .studio import read_audio
        return read_audio(path)


def _optional_file(library, version, role):
    try:return getattr(library,role+"_file")(version["id"]),None
    except (OSError,ValueError) as exc:return None,type(exc).__name__+": "+str(exc)[:500]


def execute_reencode(manager, job_id, payload, progress):
    """Explicitly encode exactly one reference into a new immutable version."""
    from .state import Session
    from .reference_cache import codec_identity,encode_reference_cached
    from .audio_processing import prepare_reference
    import soundfile as sf
    import torch
    progress();library=manager.voices;version=library.get_version(payload["version_id"])
    if payload.get("voice_id",version["voice_id"])!=version["voice_id"]:raise ValueError("重编码不能变更音色归属")
    voice=library.get_voice(version["voice_id"]);session=library.load_version(version["id"])
    if len(session.references)!=1:
        raise ValueError("多参考版本无法可靠对应预览与原录音，请逐段保存为单参考后重编码")
    old=session.references[0]
    text=payload["transcript"] if "transcript" in payload else old.transcript
    if not isinstance(text,str) or not text.strip() or len(text)>100000 or "\x00" in text:
        raise ValueError("请提供与参考录音准确对应的非空台词")
    text=text.strip()
    name=payload.get("name") or voice["name"]
    if not isinstance(name,str) or len(name)>120 or "\x00" in name:raise ValueError("参考名称无效")
    language=payload.get("language",version["language"]);delivery=payload.get("delivery",version["delivery"])
    if (not isinstance(language,str) or len(language)>80 or "\x00" in language or not isinstance(delivery,str)
            or len(delivery)>160 or "\x00" in delivery):raise ValueError("参考语言或演出说明无效")
    original,source_error=_optional_file(library,version,"source")
    preview,preview_error=_optional_file(library,version,"preview")
    distinct_preview=preview is not None and version.get("preview_path")!=version.get("source_path")
    if distinct_preview:input_path=preview;input_kind="processed_preview"
    elif original is not None:input_path=original;input_kind="original_source"
    elif preview is not None:input_path=preview;input_kind="legacy_source_preview"
    else:raise FileNotFoundError("原录音与有效preview均不可用，不能重编码")
    progress();wave,rate=_audio(input_path)
    if (type(rate) is not int or rate<=0 or wave.ndim!=2 or not wave.numel() or not torch.isfinite(wave).all()):
        raise ValueError("重编码参考PCM无效")
    interval=deepcopy(old.processing.get("source_interval"))
    if interval is not None:
        if not isinstance(interval,dict):raise ValueError("旧版原录音区间无效")
        first,last=interval.get("start_seconds"),interval.get("end_seconds")
        if (type(first) not in (int,float) or type(last) not in (int,float) or not math.isfinite(first+last)
                or not 0<=first<last):raise ValueError("旧版原录音区间无效，不能猜测分段")
        if interval.get("source_sha256") is not None and version.get("source_sha256") is not None and interval["source_sha256"]!=version["source_sha256"]:
            raise ValueError("旧版原录音区间与保存来源SHA不一致")
    if input_kind!="processed_preview" and interval is not None:
        if last>wave.shape[-1]/rate+1/rate:raise ValueError("旧版原录音区间越界，不能猜测分段")
        wave=wave[:,round(first*rate):round(last*rate)]
        input_kind="original_source_interval"
    defaults=(False,False) if distinct_preview else (old.processing.get("trim_silence",False),old.processing.get("normalize",False))
    trim=payload.get("trim_silence",defaults[0]);normalize=payload.get("normalize",defaults[1])
    if type(trim) is not bool or type(normalize) is not bool:raise ValueError("裁静音和规范化选项必须为布尔值")
    prepared=wave.float().mean(0,keepdim=True)
    if rate!=44100:
        import comfy.audio
        prepared=comfy.audio.resample(prepared,rate,44100)
    prepared,processing=prepare_reference(prepared,44100,trim_silence=trim,normalize=normalize)
    if not prepared.numel() or not torch.isfinite(prepared).all():raise ValueError("预处理后参考PCM无效")
    progress();inputs=Path(manager.inputs);inputs.mkdir(parents=True,exist_ok=True)
    clip=inputs/(uuid.uuid4().hex+"-reencoded-preview.wav")
    # FLOAT preserves exactly the mono/prepared samples that the Codec receives.
    sf.write(clip,prepared.T.cpu().numpy(),44100,subtype="FLOAT")
    engine=manager.studio.engine;cache_root=library.root/"encoding-cache"
    if not Path(engine.codec_path).is_file() or Path(engine.codec_path).stat().st_size==0:raise FileNotFoundError("当前Codec文件缺失或为空")
    current=codec_identity(engine.codec_path,cache_root)
    if hasattr(engine,"unload"):engine.unload()
    progress()
    reference=encode_reference_cached(engine,prepared,44100,text,name,cache_root,trim_silence=False,normalize=False)
    if reference.processing.get("codec_identity")!=current or codec_identity(engine.codec_path,cache_root)!=current:
        raise ValueError("Codec在重编码期间变化，拒绝提交新版本")
    provenance=dict(voice_id=voice["id"],version_id=version["id"],session_sha256=version["content_sha256"],
        input_kind=input_kind,input_file_sha256=_hash(input_path,progress),original_source_available=original is not None,
        original_source_path=version.get("source_path"),original_source_sha256=version.get("source_sha256"),
        original_source_error=source_error,previous_preview_path=version.get("preview_path"),
        previous_preview_sha256=version.get("preview_sha256"),preview_error=preview_error,
        previous_codec_identity=deepcopy(old.processing.get("codec_identity")),transcript_changed=text!=old.transcript,
        source_interval_status="preserved" if interval is not None else "legacy_unknown")
    metadata={**reference.processing,"reencoded_from":provenance,"reencode_preprocessing":processing,
        "preview_pcm":dict(sample_rate=44100,channels=1,samples=prepared.shape[-1],subtype="FLOAT",sha256=_hash(clip,progress))}
    if interval is not None:metadata["source_interval"]=interval
    reference=replace(reference,processing=metadata)
    progress()
    # Recheck the original immutable archive and audio before crossing commit.
    library.session_file(version["id"])
    if original is not None:library.source_file(version["id"])
    if preview is not None:library.preview_file(version["id"])
    manager.studio.begin_commit(job_id,progress)
    new=library.add_version(voice["id"],Session((reference,)),source_path=original,preview_path=clip,transcript=text,
        language=language,delivery=delivery,defaults=deepcopy(version["defaults"]))
    return dict(voice_id=voice["id"],version_id=new["id"],name=voice["name"],reencoded_from=version["id"],
        encoding_cache=deepcopy(reference.processing.get("encoding_cache",{})),codec_identity=current,
        original_source_preserved=original is not None,source_available=original is not None,preview_input_kind=input_kind,
        message="已创建新的重编码版本，旧版本保留"+("；原录音不可用，已保留来源记录并使用有效preview" if original is None else ""))


__all__=["describe_voice_versions","execute_reencode"]
