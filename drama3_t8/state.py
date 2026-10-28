"""Portable references, takes and immutable session branches."""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import uuid
import re
import os
import tempfile
import threading
import time

import torch
from safetensors.torch import save_file


INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
ARCHIVE_FEATURES = {'scene_chunking_v1', 'typed_takes_v1', 'render_infill_v1'}
# Only the atomic commit is serialized; tensor encoding and temporary-file IO
# can stay concurrent. Windows rejects simultaneous replacements of one target.
_COMMIT_LOCKS = tuple(threading.Lock() for _ in range(64))


def validate_codes(codes):
    if not isinstance(codes, torch.Tensor) or codes.dtype not in INTEGER_DTYPES:
        raise ValueError("Codec codes must be an integer tensor")
    if codes.ndim != 2 or codes.shape[0] != 10 or (codes.numel() and
            (codes.min() < 0 or codes.max() > 4095)):
        raise ValueError("Codec codes must have shape [10,frames] with values from 0 to 4095")


def validate_frames(frames):
    if not isinstance(frames, torch.Tensor) or frames.dtype not in INTEGER_DTYPES:
        raise ValueError("Take frames must be an integer tensor")
    if frames.ndim != 2 or frames.shape[1] != 11:
        raise ValueError("Invalid take frame shape")
    if frames.numel() and (frames[:, 0].min() < 0 or frames[:, 0].max() >= 252288
            or frames[:, 1:].min() < 0 or frames[:, 1:].max() > 4095):
        raise ValueError("Invalid take token values")


@dataclass(frozen=True)
class Reference:
    transcript: str
    codes: torch.Tensor
    name: str = ""
    codec_revision: str = "fish-s2-codec-v1"
    processing: dict = field(default_factory=dict)


def validate_reference(reference):
    if not isinstance(reference.processing,dict):
        raise ValueError("Reference processing must be an object")
    try:
        json.dumps(reference.processing,allow_nan=False)
    except (ValueError,TypeError) as exc:
        raise ValueError("Reference processing must be finite JSON data") from exc
    if not all(isinstance(value, str) for value in
               (reference.transcript, reference.name, reference.codec_revision)):
        raise ValueError("Reference transcript, name and codec revision must be text")
    validate_codes(reference.codes)
    if reference.codes.shape[-1] == 0:
        raise ValueError("Reference audio must contain at least one codec frame")


@dataclass(frozen=True)
class Take:
    prompt_text: str
    transcript: str
    frames: torch.Tensor
    semantic_start: int = 248099
    semantic_end: int = 252194
    separator_id: int = 252196
    metadata: dict = field(default_factory=dict)
    take_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    blobs: dict[str, torch.Tensor] = field(default_factory=dict)
    sources: tuple['Take', ...] = ()

    @property
    def codes(self):
        mask = (self.frames[:, 0] >= self.semantic_start) & (self.frames[:, 0] <= self.semantic_end)
        return self.frames[mask, 1:].T.contiguous()

    def segments(self):
        indices = (self.frames[:, 0] == self.separator_id).nonzero().flatten().tolist()
        bounds = [-1, *indices, len(self.frames)]
        chunks = [self.frames[a + 1:b] for a, b in zip(bounds, bounds[1:])]
        return [c[(c[:, 0] >= self.semantic_start) & (c[:, 0] <= self.semantic_end), 1:].T.contiguous()
                for c in chunks]


@dataclass(frozen=True)
class Session:
    references: tuple[Reference, ...] = ()
    takes: tuple[Take, ...] = ()
    direction: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def append(self, take):
        return replace(self, takes=(*self.takes, take), session_id=uuid.uuid4().hex)

    def branch(self, take_count):
        if type(take_count) is not int:
            raise ValueError("The branch point must be an integer take count")
        if not 0 <= take_count <= len(self.takes):
            raise ValueError("The branch point is outside this session")
        return replace(self, takes=self.takes[:take_count], session_id=uuid.uuid4().hex)


def all_takes(roots):
    """A bounded acyclic typed source graph, deduplicated by immutable ID."""
    found = {}; active = set(); checked_objects=set()
    def equal(a,b):
        try:
            return (all(getattr(a,key)==getattr(b,key) for key in ('prompt_text','transcript','semantic_start','semantic_end','separator_id'))
                and json.dumps(a.metadata,sort_keys=True,allow_nan=False)==json.dumps(b.metadata,sort_keys=True,allow_nan=False)
                and isinstance(a.frames,torch.Tensor) and isinstance(b.frames,torch.Tensor) and torch.equal(a.frames,b.frames)
                and set(a.blobs)==set(b.blobs) and all(a.blobs[k].dtype==b.blobs[k].dtype and
                    torch.equal(a.blobs[k].contiguous().view(torch.uint8),b.blobs[k].contiguous().view(torch.uint8)) for k in a.blobs)
                and [item.take_id for item in a.sources]==[item.take_id for item in b.sources])
        except (TypeError,ValueError,AttributeError,RuntimeError):
            return False
    def visit(take,depth):
        if not isinstance(take,Take) or not isinstance(take.take_id,str) or not re.fullmatch('[0-9a-f]{32}',take.take_id):
            raise ValueError('Invalid typed source Take identifier')
        if depth>64 or take.take_id in active:
            raise ValueError('Typed source graph is cyclic or too deep')
        if take.take_id in found:
            if found[take.take_id] is not take and not equal(found[take.take_id],take):
                raise ValueError('Conflicting Takes share a source identifier')
            if id(take) not in checked_objects:
                active.add(take.take_id)
                for source in take.sources:visit(source,depth+1)
                active.remove(take.take_id);checked_objects.add(id(take))
            return
        if len(found)+len(active)>=10000:
            raise ValueError('Typed source graph exceeds 10000 Takes')
        if not isinstance(take.sources,tuple) or not isinstance(take.blobs,dict) or len(take.blobs)>64:
            raise ValueError('Invalid typed Take sources/blobs')
        active.add(take.take_id)
        for source in take.sources:visit(source,depth+1)
        active.remove(take.take_id);found[take.take_id]=take;checked_objects.add(id(take))
    for root in roots:visit(root,0)
    return tuple(found.values())


def archive_features(takes):
    features=set()
    for take in takes:
        if 'scene_chunking' in take.metadata:features.add('scene_chunking_v1')
        if take.sources or take.blobs:features.add('typed_takes_v1')
        if 'render_infill' in take.metadata:features.add('render_infill_v1')
    return features


def save_session(session, path):
    validate_session(session)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors, refs, takes = {}, [], []
    graph=all_takes(session.takes)
    for i, ref in enumerate(session.references):
        key = f"reference_{i}"
        tensors[key] = ref.codes.to(device="cpu", dtype=torch.int16, copy=True).contiguous()
        refs.append({"key": key, "transcript": ref.transcript, "name": ref.name,
                     "codec_revision": ref.codec_revision,"processing":ref.processing})
    records={}
    for i, take in enumerate(graph):
        key = f"take_{i}"
        tensors[key] = take.frames.to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        record={"key": key, "prompt_text": take.prompt_text, "transcript": take.transcript,
                      "take_id": take.take_id, "semantic_start": take.semantic_start,
                      "semantic_end": take.semantic_end, "separator_id": take.separator_id,
                      "metadata": take.metadata}
        if take.blobs or take.sources:
            record['blobs']={}
            for name,blob in take.blobs.items():
                blob_key=f'take_{i}_blob_{name}'
                tensors[blob_key]=blob.detach().to(device='cpu',copy=True).contiguous()
                record['blobs'][name]=blob_key
            record['sources']=[source.take_id for source in take.sources]
        records[take.take_id]=record
    takes=[records[take.take_id] for take in session.takes]
    features = archive_features(graph)
    metadata = {"version": 2 if features else 1, "session_id": session.session_id, "direction": session.direction,
                "references": refs, "takes": takes}
    if features:
        metadata['features'] = sorted(features)
    root_ids={take.take_id for take in session.takes}
    source_records=[record for identifier,record in records.items() if identifier not in root_ids]
    if source_records:
        metadata['source_takes']=source_records
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(fd)
    part = Path(temporary)
    try:
        save_file(tensors, str(part), metadata={"drama3_t8_session": json.dumps(metadata, ensure_ascii=False)})
        commit_lock = _COMMIT_LOCKS[hash(os.path.normcase(str(path.resolve()))) % len(_COMMIT_LOCKS)]
        with commit_lock:
            for attempt in range(6):
                try:
                    part.replace(path)
                    break
                except PermissionError:
                    # Windows target mappings/handles can briefly block replacement.
                    # Keep six attempts with at most 750ms of scheduled backoff;
                    # never unlink the old archive, including on a permanent lock.
                    if os.name != "nt" or attempt == 5:
                        raise
                    time.sleep(.05 * (attempt + 1))
    finally:
        part.unlink(missing_ok=True)


def validate_session(session):
    if not isinstance(session.session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session.session_id):
        raise ValueError("Invalid session identifier")
    if not isinstance(session.direction, str):
        raise ValueError("Session direction must be text")
    for ref in session.references:
        validate_reference(ref)
    for take in all_takes(session.takes):
        if not all(isinstance(value, str) for value in (take.prompt_text, take.transcript, take.take_id)):
            raise ValueError("Take transcript, prompt and identifier must be text")
        if not re.fullmatch(r"[0-9a-f]{32}", take.take_id):
            raise ValueError("Invalid take identifier")
        if not isinstance(take.metadata, dict):
            raise ValueError("Take metadata must be an object")
        try:
            json.dumps(take.metadata, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Take metadata must contain finite JSON values") from exc
        if "scene_speaker_map" in take.metadata:
            from .scene_cast import validate_scene_speaker_map
            validate_scene_speaker_map(take.metadata["scene_speaker_map"])
        limits = (take.semantic_start, take.semantic_end, take.separator_id)
        if not all(type(value) is int and 0 <= value < 252288 for value in limits):
            raise ValueError("Invalid take token boundaries")
        if take.semantic_end - take.semantic_start != 4095 or take.semantic_start <= take.separator_id <= take.semantic_end:
            raise ValueError("Invalid take semantic range or separator")
        if take.metadata.get("task") == "text_stream" and not isinstance(take.metadata.get("lookahead"), str):
            raise ValueError("Streaming take metadata requires text lookahead")
        if take.metadata.get("task") == "scene":
            lines = take.metadata.get("script_lines")
            if lines is not None and (not isinstance(lines, (list, tuple))
                    or not all(isinstance(line, str) for line in lines)):
                raise ValueError("Scene script_lines must be a sequence of text lines")
            for field_name in ("scene", "characters", "director_note"):
                if field_name in take.metadata and not isinstance(take.metadata[field_name], str):
                    raise ValueError(f"Scene {field_name} must be text")
        validate_frames(take.frames)
        for name,blob in take.blobs.items():
            if (not isinstance(name,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}',name)
                    or not isinstance(blob,torch.Tensor) or blob.layout!=torch.strided or blob.dtype not in INTEGER_DTYPES|{torch.float32}
                    or not 1<=blob.ndim<=3 or blob.numel()>2000000000
                    or blob.is_floating_point() and not torch.isfinite(blob).all()):
                raise ValueError('Invalid typed Take tensor blob')
        if 'scene_chunking' in take.metadata:
            from .scene_chunks import validate_scene
            validate_scene(take)
        if 'render_infill' in take.metadata:
            from .render_infill import validate_take
            validate_take(take)


def load_session(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        if not meta or "drama3_t8_session" not in meta:
            raise ValueError("This file is not a Drama 3 T8 session")
        data = json.loads(meta["drama3_t8_session"])
        if not isinstance(data, dict):
            raise ValueError("Session metadata must be an object")
        if type(data.get('version')) is not int or data["version"] not in (1,2):
            raise ValueError(f"Unsupported session version {data.get('version')}")
        if data['version']==2:
            features=data.get('features')
            if (not isinstance(features,list) or not features or not all(isinstance(item,str) for item in features)
                    or len(set(features))!=len(features) or not set(features)<=ARCHIVE_FEATURES):
                raise ValueError('Unsupported session archive features')
        elif 'features' in data:
            raise ValueError('Version 1 archives cannot require additional features')
        if not isinstance(data["session_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", data["session_id"]):
            raise ValueError("Invalid session identifier")
        # Read through this same open file: reopening the path after reading
        # its header can combine an older archive's metadata with a replacement.
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    refs = tuple(Reference(r["transcript"], tensors[r["key"]], r["name"], r["codec_revision"],r.get("processing",{}))
                 for r in data["references"])
    root_records=data['takes'];extra_records=data.get('source_takes',[])
    if not isinstance(root_records,list) or not isinstance(extra_records,list) or len(root_records)+len(extra_records)>10000:
        raise ValueError('Invalid typed archive Take table')
    records={}
    for record in [*root_records,*extra_records]:
        if not isinstance(record,dict) or not isinstance(record.get('take_id'),str):
            raise ValueError('Invalid typed archive Take record')
        identifier=record['take_id']
        if identifier in records and records[identifier]!=record:
            raise ValueError('Conflicting archive source identifiers')
        records[identifier]=record
    restored={};active=set()
    def restore(identifier,depth=0):
        if depth>64 or identifier in active:
            raise ValueError('Archive source graph is cyclic or too deep')
        if identifier in restored:return restored[identifier]
        if identifier not in records:raise ValueError('Archive references an unknown source Take')
        item=records[identifier];blobs=item.get('blobs',{});source_ids=item.get('sources',[])
        if not isinstance(blobs,dict) or not isinstance(source_ids,list) or not all(isinstance(value,str) for value in source_ids):
            raise ValueError('Invalid archive typed source/blob references')
        if any(not isinstance(key,str) or key not in tensors for key in blobs.values()):
            raise ValueError('Archive references a missing tensor blob')
        active.add(identifier)
        sources=tuple(restore(source_id,depth+1) for source_id in source_ids)
        active.remove(identifier)
        take=Take(item['prompt_text'],item['transcript'],tensors[item['key']],item['semantic_start'],
            item['semantic_end'],item['separator_id'],item['metadata'],identifier,
            blobs={name:tensors[key] for name,key in blobs.items()},sources=sources)
        restored[identifier]=take
        return take
    takes=tuple(restore(record['take_id']) for record in root_records)
    if set(restored)!=set(records):raise ValueError('Archive contains unreferenced source Takes')
    session = Session(refs, takes, data["direction"], data["session_id"])
    validate_session(session)
    required=archive_features(all_takes(session.takes))
    if required != set(data.get('features',())) or required and data['version']!=2:
        raise ValueError('Session archive feature/version declaration differs from its takes')
    return replace(session, references=tuple(replace(ref, codes=ref.codes.long()) for ref in refs),
                   takes=tuple(replace(take, frames=take.frames.long()) for take in takes))
