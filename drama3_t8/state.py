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
        if not 0 <= take_count <= len(self.takes):
            raise ValueError("The branch point is outside this session")
        return replace(self, takes=self.takes[:take_count], session_id=uuid.uuid4().hex)


def save_session(session, path):
    validate_session(session)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors, refs, takes = {}, [], []
    for i, ref in enumerate(session.references):
        key = f"reference_{i}"
        tensors[key] = ref.codes.to(device="cpu", dtype=torch.int16, copy=True).contiguous()
        refs.append({"key": key, "transcript": ref.transcript, "name": ref.name,
                     "codec_revision": ref.codec_revision})
    for i, take in enumerate(session.takes):
        key = f"take_{i}"
        tensors[key] = take.frames.to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        takes.append({"key": key, "prompt_text": take.prompt_text, "transcript": take.transcript,
                      "take_id": take.take_id, "semantic_start": take.semantic_start,
                      "semantic_end": take.semantic_end, "separator_id": take.separator_id,
                      "metadata": take.metadata})
    metadata = {"version": 1, "session_id": session.session_id, "direction": session.direction,
                "references": refs, "takes": takes}
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
                    # Windows file scanners and a just-completed rename can
                    # briefly retain the target. Never unlink the old archive.
                    if os.name != "nt" or attempt == 5:
                        raise
                    time.sleep(.01 * (attempt + 1))
    finally:
        part.unlink(missing_ok=True)


def validate_session(session):
    if not isinstance(session.session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session.session_id):
        raise ValueError("Invalid session identifier")
    if not isinstance(session.direction, str):
        raise ValueError("Session direction must be text")
    for ref in session.references:
        if not all(isinstance(value, str) for value in (ref.transcript, ref.name, ref.codec_revision)):
            raise ValueError("Reference transcript, name and codec revision must be text")
        validate_codes(ref.codes)
    for take in session.takes:
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


def load_session(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        if not meta or "drama3_t8_session" not in meta:
            raise ValueError("This file is not a Drama 3 T8 session")
        data = json.loads(meta["drama3_t8_session"])
        if not isinstance(data, dict):
            raise ValueError("Session metadata must be an object")
        if data["version"] != 1:
            raise ValueError(f"Unsupported session version {data['version']}")
        if not isinstance(data["session_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", data["session_id"]):
            raise ValueError("Invalid session identifier")
        # Read through this same open file: reopening the path after reading
        # its header can combine an older archive's metadata with a replacement.
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    refs = tuple(Reference(r["transcript"], tensors[r["key"]], r["name"], r["codec_revision"])
                 for r in data["references"])
    takes = tuple(Take(t["prompt_text"], t["transcript"], tensors[t["key"]], t["semantic_start"],
                       t["semantic_end"], t["separator_id"], t["metadata"], t["take_id"])
                  for t in data["takes"])
    session = Session(refs, takes, data["direction"], data["session_id"])
    validate_session(session)
    return replace(session, references=tuple(replace(ref, codes=ref.codes.long()) for ref in refs),
                   takes=tuple(replace(take, frames=take.frames.long()) for take in takes))
