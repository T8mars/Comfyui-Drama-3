"""Shared CPU-metadata/native-code cache for repeated reference encoding."""
from dataclasses import replace
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from .workspace_lock import WorkspaceLock

_LOCK = threading.RLock()


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def codec_identity(path, root):
    path, root = Path(path).resolve(), Path(root)
    stat = path.stat()
    root.mkdir(parents=True, exist_ok=True)
    index = root / "codec-identities.json"
    with _LOCK, closing(WorkspaceLock(root / "codec-identities.lock")):
        try:
            records = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
            if not isinstance(records, dict):
                raise ValueError("invalid codec cache index")
        except (ValueError, json.JSONDecodeError):
            index.rename(index.with_name(index.name + ".retired-" + uuid.uuid4().hex))
            records = {}
        key = str(path)
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
        record = records.get(key)
        valid = (isinstance(record, dict) and isinstance(record.get("signature"), list)
            and len(record["signature"]) == 3 and all(type(value) is int for value in record["signature"])
            and isinstance(record.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
            and type(record.get("bytes")) is int and record["bytes"] == record["signature"][0])
        if record is not None and not valid:
            index.rename(index.with_name(index.name + ".retired-" + uuid.uuid4().hex))
            records = {}
        if not valid or record["signature"] != signature:
            digest = _digest(path)
            after = path.stat()
            if signature != [after.st_size, after.st_mtime_ns, after.st_ino]:
                raise ValueError("Codec文件在核对期间发生变化")
            record = {"signature": signature, "sha256": digest, "bytes": stat.st_size}
            records[key] = record
            temporary = root / f".{uuid.uuid4().hex}.pending"
            temporary.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            temporary.replace(index)
        return {"sha256": record["sha256"], "bytes": record["bytes"]}


def encode_reference_cached(engine, waveform, sample_rate, transcript, name, root, *, trim_silence=False, normalize=False):
    from .state import Session, load_session, save_session
    wave = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or not wave.shape[-1] or not np.isfinite(wave).all():
        raise ValueError("参考必须是有限、非空的声道×采样音频")
    if type(sample_rate) is not int or sample_rate <= 0 or type(trim_silence) is not bool or type(normalize) is not bool:
        raise ValueError("参考编码参数无效")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    codec = codec_identity(engine.codec_path, root)
    pcm = hashlib.sha256()
    for channel in wave:
        for start in range(0, len(channel), 262144):
            pcm.update(np.ascontiguousarray(channel[start:start + 262144], dtype="<f4").tobytes())
    implementation = hashlib.sha256()
    for filename in ("engine.py", "audio_processing.py", "native/codec.py", "reference_cache.py"):
        implementation.update((Path(__file__).parent / filename).read_bytes())
    try:
        import importlib.metadata
        libraries = {key: importlib.metadata.version(key) for key in ("torch", "torchaudio")}
    except importlib.metadata.PackageNotFoundError:
        libraries = {}
    binding = {"schema": 1, "pcm_sha256": pcm.hexdigest(), "shape": list(wave.shape), "sample_rate": sample_rate,
               "codec": codec, "trim_silence": trim_silence, "normalize": normalize,
               "implementation": implementation.hexdigest(), "libraries": libraries}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    archive, receipt = root / f"{key}.safetensors", root / f"{key}.json"
    with _LOCK, closing(WorkspaceLock(root / f"{key}.lock")):
        if archive.is_file() and receipt.is_file():
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError):
                saved = {}
            if isinstance(saved, dict) and saved.get("sha256") == _digest(archive) and saved.get("binding") == binding:
                session = load_session(archive)
                if len(session.references) == 1 and not session.takes:
                    reference = session.references[0]
                    return replace(reference, transcript=transcript, name=name,
                        processing={**reference.processing, "codec_identity": codec,
                                    "encoding_cache": {"status": "hit", "key": key}})
        reference = engine.encode_reference(waveform, sample_rate, transcript, name,
            trim_silence=trim_silence, normalize=normalize)
        reference = replace(reference, processing={**reference.processing, "codec_identity": codec,
                            "encoding_cache": {"status": "miss", "key": key}})
        # Preserve corrupt/obsolete entries as evidence instead of replacing them.
        for file in (archive, receipt):
            if file.exists():
                file.rename(file.with_name(file.name + ".retired-" + uuid.uuid4().hex))
        temporary = root / f".{uuid.uuid4().hex}.safetensors"
        save_session(Session((reference,)), temporary)
        temporary.replace(archive)
        temporary = root / f".{uuid.uuid4().hex}.pending"
        temporary.write_text(json.dumps({"binding": binding, "sha256": _digest(archive)}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(receipt)
        return reference
