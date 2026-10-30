"""Durable local creative projects, candidate takes and content-bound checkpoints.

This module deliberately has no inference, Torch or web dependency. A host supplies
the generation callback and writes a real WAV plus native Session into the supplied
candidate directory. Completed checkpoints bind those bytes to their actual input.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import threading
import time
import unicodedata
import uuid
import wave
import zipfile

from .workspace_lock import WorkspaceLock, WorkspaceBusy


SCHEMA_VERSION = 1
MODES = ("per_line", "per_line_context", "together", "per_speaker", "long_form", "text_stream")
_ID = re.compile(r"[0-9a-f]{32}\Z")
_LINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PROTECTED = re.compile(r"(<\|[^>]*\|>|<[^>]+>)")
_LOCKS = {}
_LOCK_GUARD = threading.Lock()


class ProjectConflict(ValueError):
    """The editor tried to save over a newer durable project revision."""


class InvalidCheckpoint(ValueError):
    """A returned or stored candidate does not identify complete, intact media."""


def _cancellation_error(error):
    # Keep the durable core independent from Torch, Comfy and web host imports.
    # Their explicit cancellation signals must still end a batch immediately.
    return any(cls.__name__ in {"Cancelled", "CancelledError", "InterruptProcessingException"}
               for cls in type(error).__mro__)


def _resource_error(error):
    return isinstance(error, (MemoryError, OSError)) or any(
        cls.__name__ == "OutOfMemoryError" for cls in type(error).__mro__)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _error_text(error):
    try:
        return str(error)
    except BaseException:
        return "异常说明不可读取"


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _copy_json(value):
    return json.loads(_json_bytes(value))


def _int(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}必须是{low}至{high}的整数")
    return value


def _number(value, label, low=0.0, high=86400.0, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}必须是有限数字")
    if not low <= value <= high:
        raise ValueError(f"{label}超出允许范围")
    return float(value)


def _text(value, label, limit=200000, empty=True):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{label}必须是有效文本")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是对象")
    return _copy_json(value)


def _file_record(path):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise InvalidCheckpoint("校验期间文件内容发生变化")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def model_file_identity(paths, *, content_hash=False):
    """Capture actual model/Codec/head files; hosts may supply pinned SHA identities.

    The cheap default catches ordinary file replacement using resolved path, size,
    mtime and file identity. For a strict weight identity use the pinned model
    manifest SHA or content_hash=True; a path alone is not a content hash.
    """
    result = {}
    for name, value in paths.items():
        path = Path(value).expanduser().resolve()
        info = path.stat()
        if not path.is_file():
            raise ValueError(f"模型身份必须指向文件：{name}")
        result[name] = {"path": str(path), "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns, "file_id": info.st_ino}
        if content_hash:
            result[name].update(_file_record(path))
    return result


_DIGITS = "零一二三四五六七八九"


def _cardinal(raw):
    sign = "负" if raw.startswith("-") else ""
    raw = raw.lstrip("+-")
    if "." in raw:
        whole, fraction = raw.split(".", 1)
        return sign + _cardinal(whole) + "点" + "".join(_DIGITS[int(c)] for c in fraction)
    if len(raw) > 12 or (len(raw) > 1 and raw.startswith("0")):
        return sign + "".join(_DIGITS[int(c)] for c in raw)
    number = int(raw)
    if number == 0:
        return "零"

    def four(n):
        out, pending = [], False
        for divisor, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            digit, n = divmod(n, divisor)
            if digit:
                if pending:
                    out.append("零")
                out.extend((_DIGITS[digit], unit))
                pending = False
            elif out and n:
                pending = True
        return "".join(out)

    groups = []
    while number:
        number, part = divmod(number, 10000)
        groups.append(part)
    out, zero = [], False
    for i in reversed(range(len(groups))):
        part = groups[i]
        if part:
            if out and (zero or part < 1000):
                out.append("零")
            out.extend((four(part), ("", "万", "亿")[i]))
            zero = False
        elif out:
            zero = True
    value = "".join(out)
    if value.startswith("一十"):
        value = value[1:]
    return sign + value


def normalize_dictionary(value=None):
    """Normalize plain-text readout rules, never model-specific phoneme tokens."""
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError("发音词典必须是对象")
    raw_rules = value.get("rules", value.get("aliases", {}))
    if isinstance(raw_rules, dict):
        raw_rules = [{"source": key, "replacement": val} for key, val in raw_rules.items()]
    if not isinstance(raw_rules, list) or len(raw_rules) > 1024:
        raise ValueError("发音规则必须是最多1024项的列表或对象")
    rules = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise ValueError("发音规则必须是对象")
        source = _text(raw.get("source", raw.get("from")), "规则原文", 256, False)
        replacement = _text(raw.get("replacement", raw.get("to")), "规则读法", 2048, False)
        sensitive = raw.get("case_sensitive", True)
        whole = raw.get("whole_word", False)
        if type(sensitive) is not bool or type(whole) is not bool:
            raise ValueError("规则匹配选项必须是布尔值")
        rules.append({"source": source, "replacement": replacement,
                      "case_sensitive": sensitive, "whole_word": whole})
    # Stable order is part of the dictionary version; longer overlapping aliases
    # win when scanning. Replacement strings are never scanned again for aliases.
    mode = value.get("number_mode", "none")
    if mode not in ("none", "digits", "cardinal"):
        raise ValueError("数字展开模式必须是none、digits或cardinal")
    dates = value.get("expand_dates", False)
    if type(dates) is not bool:
        raise ValueError("日期展开必须是布尔值")
    normalized = {"rules": rules, "number_mode": mode, "expand_dates": dates}
    return {**normalized, "version": fingerprint(normalized)}


def preview_text(text, dictionary=None):
    """Return explicit original/readout/subtitle layers and applied-rule evidence."""
    _text(text, "原稿")
    dictionary = normalize_dictionary(dictionary)
    rules = sorted(enumerate(dictionary["rules"]), key=lambda pair: -len(pair[1]["source"]))
    compiled = []
    for index, rule in rules:
        pattern = re.escape(rule["source"])
        if rule["whole_word"]:
            pattern = r"(?<!\w)" + pattern + r"(?!\w)"
        compiled.append((index, rule, re.compile(pattern, 0 if rule["case_sensitive"] else re.I)))
    applied = []

    def rewrite(piece):
        parts, position = [], 0
        while position < len(piece):
            for index, rule, pattern in compiled:
                match = pattern.match(piece, position)
                if match:
                    parts.append(rule["replacement"])
                    applied.append({"rule_index": index, "source": match.group(),
                                    "replacement": rule["replacement"]})
                    position = match.end()
                    break
            else:
                parts.append(piece[position])
                position += 1
        rewritten = "".join(parts)

        def numbers(value):
            if dictionary["number_mode"] == "digits":
                return re.sub(r"\d", lambda m: _DIGITS[int(m.group())], value)
            if dictionary["number_mode"] == "cardinal":
                return re.sub(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?![\d.])", lambda m: _cardinal(m.group()), value)
            return value

        if not dictionary["expand_dates"]:
            return numbers(rewritten)
        # Match complete date/time candidates before ordinary numeric expansion.
        # Invalid calendar dates and clock times stay intact rather than getting
        # guessed, or partly rewritten by the digits/cardinal pass afterwards.
        temporal = re.compile(r"(?<!\d)(?:([0-9]{4})([-/])([0-9]{1,2})\2([0-9]{1,2})(?!\d)|"
                              r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日)|"
                              r"(?<![\d:])([0-9]{1,2}):([0-9]{2})(?![\d:])")
        result, position = [], 0
        for match in temporal.finditer(rewritten):
            result.append(numbers(rewritten[position:match.start()]))
            if match[8] is not None:
                hour, minute = int(match[8]), int(match[9])
                replacement = (_cardinal(str(hour)) + "点" + _cardinal(str(minute)) + "分"
                               if hour <= 23 and minute <= 59 else match.group())
            else:
                year, month, day = (match[1], match[3], match[4]) if match[1] else (match[5], match[6], match[7])
                try:
                    datetime(int(year), int(month), int(day))
                except ValueError:
                    replacement = match.group()
                else:
                    replacement = ("".join(_DIGITS[int(c)] for c in year) + "年" + _cardinal(str(int(month)))
                                   + "月" + _cardinal(str(int(day))) + "日")
            result.append(replacement)
            position = match.end()
        result.append(numbers(rewritten[position:]))
        return "".join(result)

    pieces = _PROTECTED.split(text)
    readout = "".join(piece if _PROTECTED.fullmatch(piece) else rewrite(piece) for piece in pieces)
    return {"original_text": text, "readout_text": readout, "subtitle_text": text,
            "dictionary_version": dictionary["version"], "applied_rules": applied,
            "number_mode": dictionary["number_mode"]}


def _normal_line(raw, old, line_id, dictionary):
    if isinstance(raw, str):
        raw = {"original_text": raw}
    if not isinstance(raw, dict):
        raise ValueError("每条台词必须是文本或对象")
    original = _text(raw.get("original_text", raw.get("text", old.get("original_text", ""))), "台词原稿", empty=False)
    readout_override = raw.get("readout_override", raw.get("readout_text", old.get("readout_override")))
    subtitle_override = raw.get("subtitle_override", raw.get("subtitle_text", old.get("subtitle_override")))
    if readout_override is not None:
        _text(readout_override, "朗读稿", empty=False)
    if subtitle_override is not None:
        _text(subtitle_override, "字幕稿")
    preview = preview_text(original, dictionary)
    role = _text(raw.get("role", old.get("role", "旁白")), "角色", 256, False)
    refs = raw.get("reference_ids", old.get("reference_ids", []))
    if not isinstance(refs, list) or not all(isinstance(i, str) for i in refs):
        raise ValueError("参考选择必须是字符串列表")
    line = {"line_id": line_id, "role": role, "original_text": original,
            "readout_override": readout_override, "readout_text": readout_override if readout_override is not None else preview["readout_text"],
            "subtitle_override": subtitle_override, "subtitle_text": subtitle_override if subtitle_override is not None else original,
            "dictionary_version": preview["dictionary_version"], "applied_rules": preview["applied_rules"],
            "direction": _text(raw.get("direction", old.get("direction", "")), "演出说明", 10000),
            "reference_group": _text(raw.get("reference_group", old.get("reference_group", "")), "参考组", 256),
            "reference_ids": list(refs), "sampling": _object(raw.get("sampling", old.get("sampling", {})), "句级参数"),
            "target_seconds": _number(raw.get("target_seconds", old.get("target_seconds")), "目标时长", low=.001, optional=True),
            "pause_after": _number(raw.get("pause_after", old.get("pause_after", 0)), "停顿", high=3600),
            "start_seconds": _number(raw.get("start_seconds", old.get("start_seconds")), "开始时间", optional=True),
            "end_seconds": _number(raw.get("end_seconds", old.get("end_seconds")), "结束时间", optional=True),
            "candidate_ids": list(old.get("candidate_ids", [])),
            "selected_candidate_id": old.get("selected_candidate_id"),
            "locked": bool(old.get("locked", False))}
    if line["start_seconds"] is not None and line["end_seconds"] is not None and line["end_seconds"] < line["start_seconds"]:
        raise ValueError("台词结束时间不能早于开始时间")
    return line


def _lines(raw_lines, previous, dictionary):
    if not isinstance(raw_lines, list) or not raw_lines or len(raw_lines) > 10000:
        raise ValueError("工程需要1至10000条台词")
    previous = {line["line_id"]: line for line in previous}
    used = set()
    explicit = [raw.get("line_id", raw.get("id")) for raw in raw_lines if isinstance(raw, dict)]
    if any(line_id is not None and (not isinstance(line_id, str) or not _LINE_ID.fullmatch(line_id)) for line_id in explicit):
        raise ValueError("台词ID无效")
    explicit = {line_id for line_id in explicit if line_id is not None}
    result = []
    for index, raw in enumerate(raw_lines):
        line_id = raw.get("line_id", raw.get("id")) if isinstance(raw, dict) else None
        if line_id is None:
            suffix = index + 1
            while f"L{suffix:02d}" in explicit or f"L{suffix:02d}" in used:
                suffix += 1
            line_id = f"L{suffix:02d}"
        if not isinstance(line_id, str) or not _LINE_ID.fullmatch(line_id) or line_id in used:
            raise ValueError("台词ID必须唯一，且只包含字母、数字、点、横线或下划线")
        used.add(line_id)
        result.append(_normal_line(raw, previous.get(line_id, {}), line_id, dictionary))
    return result


def _plain_line(line):
    return {key: deepcopy(value) for key, value in line.items()
            if key not in ("candidate_ids", "candidates", "selected_candidate", "selected_candidate_id", "locked")}


def generation_settings(settings):
    """Inference settings without duplicate editor and batch preferences.

    Mode, effective sampling/seed and actual text-stream lookahead have their
    own explicit checkpoint fields. Preserve unknown future settings rather
    than accidentally declaring an inference change compatible.
    """
    duplicate = {"script_source", "script_kind", "generation_mode", "takes_per_line",
                 "base_seed", "lookahead", "sampling", "quality_after", "max_quality_retries"}
    return {key: value for key, value in _object(settings, "工程生成参数").items() if key not in duplicate}


def _selection_fingerprint(inputs):
    """Content compatibility for a deliberate selection, independent of N/seed.

    The candidate retains its real generation fingerprint and actual seed. This
    identity only decides whether it can remain the user's chosen performance
    and supply real archived context when the requested candidate window changes.
    All other sampling, model, voice, direction and context inputs stay bound.
    """
    payload = deepcopy(inputs)
    payload["settings"] = generation_settings(payload.get("settings", {}))
    if payload.get("mode") != "text_stream":
        payload["lookahead_lines"] = []
        roles = {line["role"] for line in payload.get("lines", [])}
        payload["speaker_map"] = {role: speaker for role, speaker in payload.get("speaker_map", {}).items()
                                  if role in roles}
    payload.get("sampling", {}).pop("seed", None)
    for line in payload.get("lines", []):
        line.get("sampling", {}).pop("seed", None)
    return fingerprint(payload)


class CreativeWorkspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.projects = self.root / "projects"
        self.projects.mkdir(parents=True, exist_ok=True)

    def _directory(self, project_id):
        if not isinstance(project_id, str) or not _ID.fullmatch(project_id):
            raise ValueError("无效的创作工程ID")
        directory = self.projects / project_id
        resolved = directory.resolve()
        if resolved.parent != self.projects.resolve():
            raise ValueError("工程路径越出工作区")
        return directory

    def _read(self, project_id):
        path = self._directory(project_id) / "project.json"
        if not path.is_file():
            raise FileNotFoundError("创作工程不存在")
        project = json.loads(path.read_text(encoding="utf-8"))
        if project.get("schema_version") != SCHEMA_VERSION or project.get("project_id") != project_id:
            raise ValueError("不支持或损坏的创作工程")
        _json_bytes(project)
        return project

    @contextmanager
    def _operation(self, project_id):
        directory = self._directory(project_id)
        if not (directory / "project.json").is_file():
            raise FileNotFoundError("创作工程不存在")
        key = os.path.normcase(str(directory.resolve()))
        with _LOCK_GUARD:
            lock = _LOCKS.setdefault(key, threading.RLock())
        if not lock.acquire(blocking=False):
            raise WorkspaceBusy("此创作工程正在生成或保存，请稍后再修改")
        try:
            owner = WorkspaceLock(directory / "project.lock")
            try:
                yield self._read(project_id)
            finally:
                owner.close()
        finally:
            lock.release()

    def _save(self, project, *, initial=False, retain_history=True):
        directory = self._directory(project["project_id"])
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "project.json"
        project["revision"] = 1 if initial else project["revision"] + 1
        project["updated_at"] = _now()
        data = _json_bytes(project)
        if target.exists() and retain_history:
            history = directory / "history"
            history.mkdir(exist_ok=True)
            # Preserve user edits and review choices, not a full project copy
            # for every execution-status tick. Candidates/media remain append-
            # only, avoiding quadratic checkpoint-history disk use on novels.
            shutil.copyfile(target, history / f"{project['revision'] - 1:08d}-{uuid.uuid4().hex}.json")
        temporary = directory / f".{uuid.uuid4().hex}.pending"
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # UI readers can briefly hold the old document without delete-sharing
        # on Windows. Retry only this transient replacement, never inference.
        for attempt in range(6):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(.05 * (attempt + 1))

    def create_project(self, name, lines, cast=None, settings=None, dictionary=None):
        dictionary = normalize_dictionary(dictionary)
        project = {"schema_version": SCHEMA_VERSION, "project_id": uuid.uuid4().hex,
                   "name": _text(name, "工程名称", 256, False), "revision": 0,
                   "created_at": _now(), "updated_at": _now(),
                   "cast": _object({} if cast is None else cast, "选角快照"),
                   "settings": _object({} if settings is None else settings, "工程参数"), "dictionary": dictionary,
                   "lines": _lines(lines, [], dictionary), "archived_lines": [],
                   "candidates": {}, "runs": []}
        self._save(project, initial=True)
        return self._view(project)

    def list_projects(self):
        result = []
        for directory in sorted(self.projects.iterdir()):
            if _ID.fullmatch(directory.name) and (directory / "project.json").is_file():
                project = self._read(directory.name)
                result.append({key: project[key] for key in ("project_id", "name", "revision", "created_at", "updated_at")}
                              | {"line_count": len(project["lines"]), "candidate_count": len(project["candidates"])})
        return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def _candidate_view(self, project_id, candidate):
        result = deepcopy(candidate)
        if type(result.get("seed")) is int:
            result["seed_text"] = str(result["seed"])
        if type(result.get("requested_seed")) is int:
            result["requested_seed_text"] = str(result["requested_seed"])
        directory = self._directory(project_id).resolve()
        for key in ("audio_path", "session_path", "conditioning_session_path"):
            value = result.get(key)
            if value:
                if not (directory / value).resolve().is_relative_to(directory):
                    raise InvalidCheckpoint("候选路径越出工程目录")
                result["relative_" + key] = value
                result[key] = str((directory / value).resolve())
        for item in result.get("line_audio", []):
            path = (directory / item["audio_path"]).resolve()
            if not path.is_relative_to(directory):
                raise InvalidCheckpoint("分句音频路径越出工程目录")
            item["relative_audio_path"] = item["audio_path"]
            item["audio_path"] = str(path)
        return result

    def _view(self, project):
        result = deepcopy(project)
        result["candidates"] = {key: self._candidate_view(project["project_id"], value)
                                for key, value in project["candidates"].items()}
        for line in result["lines"]:
            # Shared atomic Scene candidates can cover hundreds of lines.
            # Expand once per candidate, not once per attached line; callers
            # still receive an independent project view, never durable objects.
            line["candidates"] = [result["candidates"][key] for key in line["candidate_ids"] if key in result["candidates"]]
            line["selected_candidate"] = result["candidates"].get(line["selected_candidate_id"])
        return result

    def get_project(self, project_id):
        return self._view(self._read(project_id))

    def update_project(self, project_id, patch, expected_revision=None, *, before_commit=None):
        patch = _object(patch, "工程修改")
        if before_commit is not None and not callable(before_commit):
            raise ValueError("工程预提交回调无效")
        if set(patch) - {"name", "lines", "cast", "settings", "dictionary"}:
            raise ValueError("工程修改包含不支持的字段")
        with self._operation(project_id) as project:
            if expected_revision is not None:
                _int(expected_revision, "工程版本", 1, 2**53 - 1)
            if expected_revision is not None and project["revision"] != expected_revision:
                raise ProjectConflict("工程已更新，请刷新后再保存")
            if "name" in patch:
                project["name"] = _text(patch["name"], "工程名称", 256, False)
            for key in ("cast", "settings"):
                if key in patch:
                    project[key] = _object(patch[key], key)
            if "dictionary" in patch:
                project["dictionary"] = normalize_dictionary(patch["dictionary"])
            if "lines" in patch or "dictionary" in patch:
                old_lines = project["lines"]
                project["lines"] = _lines(patch.get("lines", old_lines), old_lines, project["dictionary"])
                current = {line["line_id"] for line in project["lines"]}
                project["archived_lines"].extend({**line, "archived_at": _now()}
                                                   for line in old_lines if line["line_id"] not in current)
            if before_commit:
                before_commit(project, self._directory(project_id))
            self._save(project)
            return self._view(project)

    def preview_project(self, project_id, dictionary=None):
        project = self._read(project_id)
        dictionary = normalize_dictionary(dictionary if dictionary is not None else project["dictionary"])
        return [{"line_id": line["line_id"], **preview_text(line["original_text"], dictionary),
                 "readout_override": line["readout_override"], "subtitle_override": line["subtitle_override"]}
                for line in project["lines"]]

    def _usable(self, project_id, candidate):
        if candidate.get("status") != "completed" or not candidate.get("artifacts"):
            return False
        directory = self._directory(project_id).resolve()
        try:
            kinds = set()
            for artifact in candidate["artifacts"]:
                path = (directory / artifact["path"]).resolve()
                if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size == 0:
                    return False
                if _file_record(path) != {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}:
                    return False
                kinds.add(artifact["kind"])
                pointer = "conditioning_session_path" if artifact["kind"] == "conditioning" else artifact["kind"] + "_path"
                if artifact["kind"] in ("audio", "session", "conditioning") and candidate.get(pointer) != artifact["path"]:
                    return False
                if artifact["kind"].startswith("line:") and not any(
                        item.get("line_id") == artifact["kind"][5:] and item.get("audio_path") == artifact["path"]
                        for item in candidate.get("line_audio", [])):
                    return False
            return {"audio", "session"} <= kinds and len(kinds) == len(candidate["artifacts"])
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def _options(self, project, *, takes_per_line=3, base_seed=42, line_ids=None,
                 mode="per_line", sampling=None, model_identity=None, lookahead=0,
                 retry_failed=True, force=False):
        _int(takes_per_line, "候选数量", 1, 32)
        _int(base_seed, "种子", 0, 2**64 - 1)
        _int(lookahead, "前瞻句数", 0, 100)
        if mode not in MODES:
            raise ValueError("不支持的创作生成模式")
        if type(force) is not bool or type(retry_failed) is not bool:
            raise ValueError("重做选项必须是布尔值")
        known = {line["line_id"] for line in project["lines"]}
        if line_ids is not None and (not isinstance(line_ids, list) or not all(isinstance(i, str) and i in known for i in line_ids)):
            raise ValueError("请选择工程中的台词ID")
        return dict(takes_per_line=takes_per_line, base_seed=base_seed, line_ids=line_ids,
                    mode=mode, sampling=_object({} if sampling is None else sampling, "采样参数"),
                    model_identity=_object({} if model_identity is None else model_identity, "模型身份"), lookahead=lookahead,
                    retry_failed=retry_failed, force=force)

    def _jobs(self, project, options):
        lines, mode = project["lines"], options["mode"]
        speaker_map = {role: binding["speaker"] for role, binding in project["cast"].items()
                       if isinstance(binding, dict) and type(binding.get("speaker")) is int}
        if mode == "together":
            units = [list(range(len(lines)))]
        elif mode == "per_speaker":
            units = [[i for i, line in enumerate(lines) if line["role"] == role]
                     for role in dict.fromkeys(line["role"] for line in lines)]
        else:
            units = [[i] for i in range(len(lines))]
        jobs, context = [], {}
        usable = {key: self._usable(project["project_id"], candidate) for key, candidate in project["candidates"].items()}
        for unit_index, indexes in enumerate(units):
            unit_lines = [_plain_line(lines[i]) for i in indexes]
            roles = list(dict.fromkeys(line["role"] for line in unit_lines))
            if mode in ("per_line_context", "text_stream"):
                own = [i for i in range(unit_index) if lines[units[i][0]]["role"] == unit_lines[0]["role"]]
                dependencies = sorted(set(([own[0], own[-1]] if own else []) + ([unit_index - 1] if unit_index else [])))
            else:
                dependencies = [0] if mode == "long_form" and unit_index else []
            deps = [context.get(i) for i in dependencies]
            look = ([_plain_line(line) for line in lines[max(indexes) + 1:max(indexes) + 1 + options["lookahead"]]]
                    if mode == "text_stream" else [])
            used_speakers = set(roles) | {line["role"] for line in look}
            dependency_identity = [{"candidate_id": item["candidate_id"], "fingerprint": item["fingerprint"],
                                    "session_sha256": next(a["sha256"] for a in item["artifacts"] if a["kind"] ==
                                                           ("conditioning" if item.get("conditioning_session_path") else "session")),
                                    **({"source_session_sha256": next(a["sha256"] for a in item["artifacts"] if a["kind"] == "session")}
                                       if item.get("conditioning_session_path") else {})}
                                   if item else {"pending_unit": dependencies[n]} for n, item in enumerate(deps)]
            params = {**options["sampling"], **unit_lines[0]["sampling"]}
            if len(unit_lines) > 1 and any(line["sampling"] != unit_lines[0]["sampling"] for line in unit_lines):
                raise ValueError("场景原子单元内的句级采样参数必须一致，请改用逐句模式")
            unit_jobs = []
            for take_index in range(options["takes_per_line"]):
                seed_base = params.get("seed", (options["base_seed"] + unit_index * 1000) % 2**64)
                _int(seed_base, "实际种子", 0, 2**64 - 1)
                retry_span = _int(params.get("opening_retries", 0), "开头重试次数", 0, 1) + 1
                seed = (seed_base + take_index * retry_span) % 2**64
                actual_sampling = {**params, "seed": seed}
                inputs = {"schema_version": SCHEMA_VERSION, "lines": unit_lines, "cast": {role: project["cast"].get(role) for role in roles},
                          "speaker_map": {role: speaker_map[role] for role in used_speakers if role in speaker_map},
                          "settings": generation_settings(project["settings"]), "dictionary_version": project["dictionary"]["version"],
                          "mode": mode, "sampling": actual_sampling, "model_identity": options["model_identity"],
                          "context_policy": "own_first_latest_and_previous_v1" if mode in ("per_line_context", "text_stream") else "first_anchor_v1" if mode == "long_form" else "independent_v1",
                          "context": dependency_identity, "lookahead_lines": look}
                digest = fingerprint(inputs)
                selected_here = {lines[i]["selected_candidate_id"] for i in indexes}
                matches = [candidate for candidate in project["candidates"].values()
                           if (candidate.get("fingerprint") == digest or (candidate["candidate_id"] in selected_here and
                               candidate.get("compatible_fingerprint") == digest))
                           and candidate.get("line_ids") == [line["line_id"] for line in unit_lines]]
                completed = next((candidate for candidate in reversed(matches) if usable[candidate["candidate_id"]]), None)
                previous_failure = next((candidate for candidate in reversed(matches) if candidate.get("status") == "failed"), None)
                requested = options["line_ids"] is None or any(line["line_id"] in options["line_ids"] for line in unit_lines)
                status = ("not_requested" if not requested else "blocked" if any(item is None for item in deps)
                          else "completed" if completed and not options["force"] else "failed" if previous_failure and not options["retry_failed"] else "pending")
                job = {"project_id": project["project_id"], "unit_index": unit_index,
                       "unit_id": fingerprint([line["line_id"] for line in unit_lines])[:16],
                       "line_id": unit_lines[0]["line_id"], "line_ids": [line["line_id"] for line in unit_lines],
                       "line": unit_lines[0], "lines": unit_lines, "role": unit_lines[0]["role"],
                       "roles": roles, "readout_text": "\n".join(line["readout_text"] for line in unit_lines),
                       "subtitle_text": "\n".join(line["subtitle_text"] for line in unit_lines),
                       "take_index": take_index + 1, "seed": seed, "sampling": actual_sampling,
                       "seed_text": str(seed),
                       "cast": inputs["cast"], "settings": deepcopy(inputs["settings"]), "model_identity": options["model_identity"],
                       "speaker_map": dict(speaker_map),
                       "mode": mode, "fingerprint": digest, "input_snapshot": inputs,
                       "context_candidates": [self._candidate_view(project["project_id"], candidate) for candidate in deps if candidate],
                       "context_candidate_ids": [candidate["candidate_id"] for candidate in deps if candidate],
                       "lookahead_lines": look, "status": status,
                       "candidate": self._candidate_view(project["project_id"], completed) if completed else None}
                unit_jobs.append(job)
            selected_ids = {lines[i]["selected_candidate_id"] for i in indexes if lines[i]["selected_candidate_id"]}
            semantic = _selection_fingerprint(unit_jobs[0]["input_snapshot"])
            selected = next((project["candidates"][key] for key in selected_ids if key in project["candidates"]
                             and usable[key] and project["candidates"][key].get("line_ids") == unit_jobs[0]["line_ids"]
                             and _selection_fingerprint(project["candidates"][key]["input_snapshot"]) == semantic), None)
            chosen = selected or next((project["candidates"][job["candidate"]["candidate_id"]] for job in unit_jobs if job["candidate"]), None)
            locked = any(lines[i]["locked"] for i in indexes)
            if locked:
                for job in unit_jobs:
                    if job["status"] != "not_requested":
                        job["status"] = "locked" if selected and len(selected_ids) == 1 else "locked_stale"
                # A stale frozen edit remains available for audition/export, but
                # is never silently supplied as fresh generation context.
                if not selected or len(selected_ids) != 1:
                    chosen = None
            for job in unit_jobs:
                job["selected_candidate"] = self._candidate_view(project["project_id"], selected) if selected else None
                job["selection_compatible"] = bool(selected)
                job["selection_warning"] = ("所选候选与当前文本、参考、模型或非种子采样条件不一致" if selected_ids and not selected else "")
            context[unit_index] = chosen
            jobs.extend(unit_jobs)
        return {"project_id": project["project_id"], "revision": project["revision"],
                "mode": mode, "unit_count": len(units), "jobs": jobs,
                "pending": sum(job["status"] == "pending" for job in jobs),
                "completed": sum(job["status"] == "completed" for job in jobs),
                "blocked": sum(job["status"] in ("blocked", "locked_stale") for job in jobs)}

    def plan_generation(self, project_id, **options):
        project = self._read(project_id)
        return self._jobs(project, self._options(project, **options))

    def _result(self, project, candidate, job, result, artifact_dir):
        if not isinstance(result, dict) or result.get("complete") is not True:
            raise InvalidCheckpoint("生成回调必须确认complete=True；未完成的音频不能成为检查点")
        actual_seed = result.get("seed")
        if type(actual_seed) is not int or not 0 <= actual_seed < 2**64:
            raise InvalidCheckpoint("回调实际seed缺失或无效")
        metadata = _object(result.get("metadata", {}), "生成元数据")
        if metadata.get("length_limit_reached") or metadata.get("finish_reason") in ("stalled", "cancelled", "error"):
            raise InvalidCheckpoint("生成被截断、中断或停滞，不能标记完成")
        if "actual_segments" in metadata and metadata["actual_segments"] != len(job["lines"]):
            raise InvalidCheckpoint("实际场景段数与台词数不同")
        artifacts = []
        paths = {}
        directory = self._directory(project["project_id"]).resolve()
        for key, kind in (("audio_path", "audio"), ("session_path", "session")):
            value = result.get(key)
            if not isinstance(value, (str, os.PathLike)):
                raise InvalidCheckpoint(f"生成回调缺少{key}")
            path = Path(value)
            path = (path if path.is_absolute() else artifact_dir / path).resolve()
            if not path.is_relative_to(artifact_dir.resolve()) or not path.is_file() or path.stat().st_size == 0:
                raise InvalidCheckpoint("生成文件必须位于当前候选目录，且内容非空")
            record = {"kind": kind, "path": path.relative_to(directory).as_posix(), **_file_record(path)}
            artifacts.append(record)
            paths[key] = record["path"]
        audio_info = _pcm_wav_info(directory / paths["audio_path"])
        rate, channels, duration = audio_info["sample_rate"], audio_info["channels"], audio_info["duration_seconds"]
        if "duration_seconds" in result and abs(_number(result["duration_seconds"], "实际时长", high=604800) - duration) > max(2 / rate, 0.005):
            raise InvalidCheckpoint("回调时长与WAV实际帧数不符")
        if "sample_rate" in result and result["sample_rate"] != rate:
            raise InvalidCheckpoint("回调采样率与WAV不符")
        line_audio = result.get("line_audio", [])
        if not isinstance(line_audio, list):
            raise InvalidCheckpoint("分句音频必须是列表")
        checked_lines, seen = [], set()
        for raw in line_audio:
            item = _object(raw, "分句音频")
            line_id = item.get("line_id")
            if line_id not in job["line_ids"] or line_id in seen:
                raise InvalidCheckpoint("分句音频台词ID不匹配或重复")
            seen.add(line_id)
            path = Path(item["audio_path"])
            path = (path if path.is_absolute() else artifact_dir / path).resolve()
            if not path.is_relative_to(artifact_dir.resolve()) or not path.is_file():
                raise InvalidCheckpoint("分句音频必须位于当前候选目录")
            start = _number(item.get("start_seconds", item.get("start")), "分句开始时间", high=duration + .005)
            end = _number(item.get("end_seconds", item.get("end")), "分句结束时间", high=duration + .005)
            info = _pcm_wav_info(path)
            if (end <= start or info["sample_rate"] != rate or info["channels"] != channels
                    or info["sample_width"] != audio_info["sample_width"]
                    or abs(info["duration_seconds"] - (end - start)) > max(2 / rate, .005)):
                raise InvalidCheckpoint("分句PCM与实际时间范围不一致")
            if round(start * rate) + info["frames"] > audio_info["frames"]:
                raise InvalidCheckpoint("分句音频范围越出源PCM")
            with wave.open(str(directory / paths["audio_path"]), "rb") as full, wave.open(str(path), "rb") as part:
                full.setpos(round(start * rate))
                while block := part.readframes(65536):
                    if full.readframes(len(block) // (channels * info["sample_width"])) != block:
                        raise InvalidCheckpoint("分句PCM不是实际源区间的原样切片")
            relative = path.relative_to(directory).as_posix()
            artifacts.append({"kind": "line:" + line_id, "path": relative, **_file_record(path)})
            checked_lines.append({**item, "audio_path": relative, "start_seconds": start, "end_seconds": end,
                                  "sample_rate": rate, "duration_seconds": info["duration_seconds"]})
        if line_audio and seen != set(job["line_ids"]):
            raise InvalidCheckpoint("分句PCM未覆盖此原子单元的全部台词")
        session_info = _native_session_info(directory / paths["session_path"])
        conditioning = result.get("conditioning_session_path")
        if conditioning is not None:
            if len(job["lines"]) != 1 or not isinstance(conditioning, (str, os.PathLike)):
                raise InvalidCheckpoint("实际PCM条件归档只支持当前单角色台词")
            condition_path = Path(conditioning)
            condition_path = (condition_path if condition_path.is_absolute() else artifact_dir / condition_path).resolve()
            if not condition_path.is_relative_to(artifact_dir.resolve()) or not condition_path.is_file():
                raise InvalidCheckpoint("实际PCM条件归档必须位于当前候选目录")
            condition_info = _native_session_info(condition_path)
            binding = condition_info.get("from_source_view")
            expected_view = {"audio_sha256": next(item["sha256"] for item in artifacts if item["kind"] == "audio"),
                             "source_take_id": session_info["take_id"], "line_id": job["line_id"],
                             "codec_identity": job["model_identity"].get("codec")}
            if (condition_info["take_count"] != 1 or condition_info["transcript"] != session_info["transcript"]
                    or not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected_view.items())):
                raise InvalidCheckpoint("实际PCM条件归档未绑定当前WAV、目标台词与Codec身份")
            relative = condition_path.relative_to(directory).as_posix()
            paths["conditioning_session_path"] = relative
            artifacts.append({"kind": "conditioning", "path": relative, **_file_record(condition_path)})
        native_seed = session_info.get("seed")
        native_requested = session_info.get("requested_seed")
        if native_seed is not None and native_seed != actual_seed:
            raise InvalidCheckpoint("回调seed与原生Take记录不符")
        if actual_seed != job["seed"] and (native_seed != actual_seed or native_requested != job["seed"]):
            raise InvalidCheckpoint("实际seed不同但缺少原生Take的请求/重试证据")
        if native_requested is not None and native_requested != job["seed"]:
            raise InvalidCheckpoint("原生Take的请求seed与当前任务不同")
        candidate.update(status="completed", completed_at=_now(), **paths,
                         artifacts=artifacts, duration_seconds=duration, sample_rate=rate,
                         channels=channels, seed=actual_seed, requested_seed=job["seed"], metadata=metadata,
                         session_id=session_info["session_id"], native_take_id=session_info["take_id"],
                         transcript=session_info["transcript"], line_audio=checked_lines,
                         quality=_object(result.get("quality", {}), "质检记录"))
        if result.get("transcript") is not None:
            returned_text = _text(result["transcript"], "原生目标台词")
            if returned_text != session_info["transcript"]:
                raise InvalidCheckpoint("回调目标台词与原生Take归档不符")

    def run_generation(self, project_id, generate, *, cancelled=None, progress=None,
                       continue_on_error=False, expected_revision=None, **plan_options):
        """Run a serial, restartable native generation batch.

        ``generate(job, artifact_dir)`` must write a PCM WAV and real native
        Session below artifact_dir and return ``complete=True``, actual ``seed``,
        ``audio_path``, ``session_path``, plus optional metadata/quality. The host
        may hold one engine generation_session around this method. It must not
        run ASR inside every callback; use run_quality as a separate phase.

        takes_per_line is a target count: increasing 3 to 6 adds candidates 4–6
        while reusing intact 1–3. together and per_speaker are atomic scene/role
        units, not falsely advertised as independently resumable source lines.
        """
        if not callable(generate):
            raise TypeError("generate必须是本地推理回调")
        if type(continue_on_error) is not bool:
            raise ValueError("继续失败项选项必须是布尔值")
        with self._operation(project_id) as project:
            if expected_revision is not None:
                _int(expected_revision, "工程版本", 1, 2**53 - 1)
                if expected_revision != project["revision"]:
                    raise ProjectConflict("工程已更新，请刷新后再生成")
            options = self._options(project, **plan_options)
            for candidate in project["candidates"].values():
                if candidate["status"] == "running":
                    candidate.update(status="interrupted", error="上次进程在候选提交前退出", interrupted_at=_now())
            for previous in project["runs"]:
                if previous["status"] == "running":
                    previous.update(status="interrupted", completed_at=_now())
            run = {"run_id": uuid.uuid4().hex, "status": "running", "started_at": _now(),
                   "options": deepcopy(options), "generated": [], "failed": []}
            project["runs"].append(run)
            self._save(project)
            attempted = set()
            try:
                while True:
                    if cancelled and cancelled():
                        run["status"] = "cancelled"
                        break
                    plan = self._jobs(project, options)
                    job = next((item for item in plan["jobs"] if item["status"] == "pending"
                                and item["fingerprint"] not in attempted), None)
                    if job is None:
                        run["status"] = ("blocked" if plan["blocked"] else "partial" if run["failed"] or
                                         any(item["status"] == "failed" for item in plan["jobs"]) else "completed")
                        break
                    attempted.add(job["fingerprint"])
                    candidate_id = uuid.uuid4().hex
                    artifact_dir = self._directory(project_id) / "artifacts" / candidate_id
                    artifact_dir.mkdir(parents=True, exist_ok=False)
                    candidate = {"candidate_id": candidate_id, "line_ids": job["line_ids"],
                                 "take_index": job["take_index"], "seed": job["seed"], "requested_seed": job["seed"],
                                 "fingerprint": job["fingerprint"], "input_snapshot": job["input_snapshot"],
                                 "context_candidate_ids": job["context_candidate_ids"],
                                 "status": "running", "created_at": _now(), "run_id": run["run_id"],
                                 "artifacts": [], "rating": None, "favorite": False, "notes": ""}
                    project["candidates"][candidate_id] = candidate
                    for line in project["lines"]:
                        if line["line_id"] in job["line_ids"]:
                            line["candidate_ids"].append(candidate_id)
                    self._save(project, retain_history=False)
                    try:
                        returned = generate(deepcopy(job), artifact_dir)
                        self._result(project, candidate, job, returned, artifact_dir)
                        run["generated"].append(candidate_id)
                        # Auto-select only when no currently intact selection
                        # matches the current plan; extra takes don't override
                        # an existing deliberate selection.
                        semantic = _selection_fingerprint(job["input_snapshot"])
                        selected_ids = {line["selected_candidate_id"] for line in project["lines"] if line["line_id"] in job["line_ids"]}
                        valid_selection = any(key in project["candidates"] and self._usable(project_id, project["candidates"][key])
                                              and _selection_fingerprint(project["candidates"][key]["input_snapshot"]) == semantic
                                              for key in selected_ids if key)
                        if not valid_selection:
                            for line in project["lines"]:
                                if line["line_id"] in job["line_ids"]:
                                    line["selected_candidate_id"] = candidate_id
                        self._save(project, retain_history=False)
                    except BaseException as exc:
                        was_cancelled = _cancellation_error(exc) or bool(cancelled and cancelled())
                        candidate.update(status="cancelled" if was_cancelled else "failed" if isinstance(exc, Exception) else "interrupted",
                                         error=f"{type(exc).__name__}: {_error_text(exc)}", completed_at=_now())
                        run["failed"].append(candidate_id)
                        self._save(project, retain_history=False)
                        if was_cancelled or _resource_error(exc) or not isinstance(exc, Exception) or not continue_on_error:
                            raise
                    if progress:
                        progress({"run_id": run["run_id"], "generated": len(run["generated"]),
                                  "failed": len(run["failed"]), "plan": self._jobs(project, options)})
            except BaseException as exc:
                run.update(status="cancelled" if _cancellation_error(exc) or (cancelled and cancelled()) else "interrupted", completed_at=_now())
                self._save(project, retain_history=False)
                raise
            run["completed_at"] = _now()
            self._save(project, retain_history=False)
            return {"run_id": run["run_id"], "status": run["status"], "generated": list(run["generated"]),
                    "failed": list(run["failed"]), "plan": self._jobs(project, options), "project": self._view(project)}

    def run_variation(self, project_id, line_id, source_candidate_id, generate, operation,
                      *, sampling=None, model_identity=None, expected_revision=None,
                      cancelled=None, force=False):
        """Persist a real source-aware revise/infill candidate for A/B selection.

        The callback gets source_candidate (including its native Session), the
        complete edited unit lines, and the explicit operation. It must use the
        full job.readout_text as the target for a multi-line native source. No
        old candidate or current script is overwritten. Selecting with
        apply_text=True explicitly adopts the candidate's script layers.
        """
        if not callable(generate) or type(force) is not bool:
            raise ValueError("变体生成回调或强制重做选项无效")
        operation = _object(operation, "局部操作")
        if operation.get("kind") not in ("infill", "revise"):
            raise ValueError("局部操作仅支持infill或revise")
        text = _text(operation.get("text"), "改后台词", empty=False)
        with self._operation(project_id) as project:
            if expected_revision is not None:
                _int(expected_revision, "工程版本", 1, 2**53 - 1)
                if expected_revision != project["revision"]:
                    raise ProjectConflict("工程已更新，请刷新后再改配")
            line = next((item for item in project["lines"] if item["line_id"] == line_id), None)
            source = project["candidates"].get(source_candidate_id)
            if line is None or source is None or source_candidate_id not in line["candidate_ids"] or not self._usable(project_id, source):
                raise InvalidCheckpoint("局部改配需要本句完整、未损坏的源候选")
            if operation["kind"] == "infill":
                start = _number(operation.get("start_seconds"), "替换开始", high=source["duration_seconds"])
                end = _number(operation.get("end_seconds"), "替换结束", high=source["duration_seconds"])
                if end <= start:
                    raise ValueError("局部替换结束必须晚于开始")
                operation.update(start_seconds=start, end_seconds=end)
            desired = []
            for original in source["input_snapshot"]["lines"]:
                if original["line_id"] == line_id:
                    edited = _normal_line({**original, "original_text": operation.get("original_text", text),
                                           "direction": line["direction"],
                                           "readout_override": operation.get("readout_override"),
                                           "subtitle_override": operation.get("subtitle_text")},
                                          original, line_id, project["dictionary"])
                    desired.append(_plain_line(edited))
                else:
                    desired.append(deepcopy(original))
            roles = list(dict.fromkeys(item["role"] for item in desired))
            cast = {role: project["cast"].get(role) for role in roles}
            if cast != source["input_snapshot"]["cast"]:
                raise ValueError("源候选与当前选角不同，请先用当前音色生成，再做局部改配")
            params = {**source["input_snapshot"]["sampling"], **_object({} if sampling is None else sampling, "变体采样")}
            seed = _int(params.get("seed", source.get("requested_seed", source["seed"])), "变体请求种子", 0, 2**64 - 1)
            params["seed"] = seed
            operation = {**operation, "line_id": line_id, "original_text": operation.get("original_text", text),
                         "text": next(item["readout_text"] for item in desired if item["line_id"] == line_id)}
            inputs = {**deepcopy(source["input_snapshot"]), "lines": desired, "cast": cast,
                      "settings": generation_settings(project["settings"]), "dictionary_version": project["dictionary"]["version"],
                      "sampling": params, "model_identity": deepcopy(source["input_snapshot"]["model_identity"]) if model_identity is None else _object(model_identity, "变体模型身份")}
            if inputs.get("mode") != "text_stream":
                inputs["lookahead_lines"] = []
                inputs["speaker_map"] = {role: speaker for role, speaker in inputs.get("speaker_map", {}).items()
                                         if role in roles}
            variation = {"source_candidate_id": source_candidate_id, "source_fingerprint": source["fingerprint"],
                         "source_artifacts": deepcopy(source["artifacts"]), "operation": operation}
            digest = fingerprint({"input": inputs, "variation": variation})
            if not force:
                reused = next((item for item in reversed(list(project["candidates"].values()))
                               if item.get("fingerprint") == digest and self._usable(project_id, item)), None)
                if reused:
                    return {"status": "completed", "generated": [], "failed": [], "reused": True,
                            "candidate": self._candidate_view(project_id, reused), "project": self._view(project)}
            if cancelled and cancelled():
                return {"status": "cancelled", "generated": [], "failed": [], "project": self._view(project)}
            context_ids = list(dict.fromkeys([*source.get("context_candidate_ids", []), source_candidate_id]))
            job = {"project_id": project_id, "line_id": line_id, "line_ids": source["line_ids"],
                   "line": next(item for item in desired if item["line_id"] == line_id), "lines": desired,
                   "role": line["role"], "roles": roles, "readout_text": "\n".join(item["readout_text"] for item in desired),
                   "subtitle_text": "\n".join(item["subtitle_text"] for item in desired), "cast": cast,
                   "settings": inputs["settings"], "model_identity": inputs["model_identity"],
                   "speaker_map": deepcopy(inputs.get("speaker_map", {})),
                   "take_index": source["take_index"], "seed": seed, "sampling": params, "mode": inputs["mode"],
                   "operation": operation, "source_candidate": self._candidate_view(project_id, source),
                   "context_candidate_ids": context_ids,
                   "context_candidates": [self._candidate_view(project_id, project["candidates"][key]) for key in context_ids],
                   "lookahead_lines": inputs.get("lookahead_lines", []), "input_snapshot": inputs, "fingerprint": digest}
            candidate_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex
            candidate = {"candidate_id": candidate_id, "line_ids": list(source["line_ids"]),
                         "take_index": source["take_index"], "seed": seed, "requested_seed": seed,
                         "fingerprint": digest, "compatible_fingerprint": fingerprint(inputs), "input_snapshot": inputs,
                         "context_candidate_ids": context_ids, "variation": variation,
                         "status": "running", "created_at": _now(), "run_id": run_id,
                         "artifacts": [], "rating": None, "favorite": False, "notes": ""}
            project["candidates"][candidate_id] = candidate
            for item in project["lines"]:
                if item["line_id"] in source["line_ids"]:
                    item["candidate_ids"].append(candidate_id)
            run = {"run_id": run_id, "kind": "variation", "started_at": _now(), "status": "running",
                   "source_candidate_id": source_candidate_id, "operation": operation}
            project["runs"].append(run)
            directory = self._directory(project_id) / "artifacts" / candidate_id
            directory.mkdir(parents=True, exist_ok=False)
            self._save(project)
            try:
                self._result(project, candidate, job, generate(deepcopy(job), directory), directory)
                run.update(status="completed", completed_at=_now())
                self._save(project, retain_history=False)
            except BaseException as exc:
                state = "cancelled" if _cancellation_error(exc) or (cancelled and cancelled()) else "failed" if isinstance(exc, Exception) else "interrupted"
                candidate.update(status=state, error=f"{type(exc).__name__}: {_error_text(exc)}", completed_at=_now())
                run.update(status=state, completed_at=_now())
                self._save(project, retain_history=False)
                raise
            return {"run_id": run_id, "status": "completed", "generated": [candidate_id], "failed": [],
                    "reused": False, "candidate": self._candidate_view(project_id, candidate), "project": self._view(project)}

    def set_candidate_review(self, project_id, line_id, candidate_id, *, rating=None,
                             favorite=None, selected=None, locked=None, notes=None, apply_text=False):
        if rating is not None:
            _int(rating, "评分", 1, 5)
        for value in (favorite, selected, locked):
            if value is not None and type(value) is not bool:
                raise ValueError("候选选项必须是布尔值")
        if type(apply_text) is not bool:
            raise ValueError("应用台词选项必须是布尔值")
        if notes is not None:
            _text(notes, "候选备注", 10000)
        with self._operation(project_id) as project:
            line = next((item for item in project["lines"] if item["line_id"] == line_id), None)
            candidate = project["candidates"].get(candidate_id)
            if line is None or candidate_id not in line["candidate_ids"] or candidate is None:
                raise ValueError("候选不属于该台词")
            if selected is True and not self._usable(project_id, candidate):
                raise InvalidCheckpoint("只能选择完整、未损坏的候选")
            if selected is True and locked is not False and any(item["locked"] and item["selected_candidate_id"] != candidate_id
                     for item in project["lines"] if item["line_id"] in candidate["line_ids"]):
                raise ValueError("请先解锁，再替换已锁定的候选")
            if selected is True and apply_text:
                desired = {item["line_id"]: item for item in candidate["input_snapshot"]["lines"]}
                project["lines"] = [_normal_line(desired[item["line_id"]], item, item["line_id"], project["dictionary"])
                                    if item["line_id"] in desired else item for item in project["lines"]]
            if rating is not None:
                candidate["rating"] = rating
            if favorite is not None:
                candidate["favorite"] = favorite
            if notes is not None:
                candidate["notes"] = notes
            if selected is not None:
                for item in project["lines"]:
                    if item["line_id"] in candidate["line_ids"]:
                        if selected:
                            item["selected_candidate_id"] = candidate_id
                        elif item["selected_candidate_id"] == candidate_id:
                            if item["locked"] and locked is not False:
                                raise ValueError("请先解锁选中的候选")
                            item["selected_candidate_id"] = None
            if locked is not None:
                for item in project["lines"]:
                    if item["line_id"] in candidate["line_ids"]:
                        if locked and item["selected_candidate_id"] != candidate_id:
                            raise ValueError("只能锁定当前选中的候选")
                        item["locked"] = locked
            self._save(project)
            return self._view(project)

    def run_quality(self, project_id, review, *, candidate_ids=None, cancelled=None):
        """A separate ASR/technical-review phase; findings remain soft flags."""
        if not callable(review):
            raise TypeError("review必须是本地质检回调")
        with self._operation(project_id) as project:
            ids = list(project["candidates"]) if candidate_ids is None else candidate_ids
            if not isinstance(ids, list) or any(i not in project["candidates"] for i in ids):
                raise ValueError("质检候选ID无效")
            checked, errors = [], []
            # One durable pre-review snapshot suffices; per-candidate history is
            # stored on the candidate rather than copying the entire project.
            self._save(project)
            for candidate_id in ids:
                if cancelled and cancelled():
                    break
                candidate = project["candidates"][candidate_id]
                if not self._usable(project_id, candidate):
                    continue
                if candidate.get("quality"):
                    candidate.setdefault("quality_history", []).append(deepcopy(candidate["quality"]))
                try:
                    result = _object(review(self._candidate_view(project_id, candidate)), "质检返回值")
                    expected = candidate["transcript"]
                    if "transcript" in result:
                        result["text_check"] = text_quality(expected, _text(result["transcript"], "转写文本"),
                                                            independent=result.get("independent", False))
                    candidate["quality"] = {**candidate.get("quality", {}), **result, "reviewed_at": _now()}
                    checked.append(candidate_id)
                except Exception as exc:
                    candidate["quality"] = {**candidate.get("quality", {}), "review_error": f"{type(exc).__name__}: {_error_text(exc)}",
                                            "reviewed_at": _now()}
                    errors.append(candidate_id)
                self._save(project, retain_history=False)
            return {"checked": checked, "errors": errors, "project": self._view(project)}

    def export_project(self, project_id, target, *, include_all=False):
        """Write a new portable project archive with digest-checked candidate media.

        Voice identity/Cast snapshots are retained. External voice-library assets
        are not silently copied from arbitrary paths; a host can deliver its
        matching voice pack alongside this archive and resolve UUID/version IDs.
        """
        if type(include_all) is not bool:
            raise ValueError("候选打包选项必须是布尔值")
        target = Path(target).expanduser().resolve()
        with self._operation(project_id) as project:
            packed = deepcopy(project)
            included = set(project["candidates"]) if include_all else {
                line["selected_candidate_id"] for line in project["lines"] if line["selected_candidate_id"]}
            # Context Sessions must travel with selected takes, even if their
            # corresponding source-line candidate was not selected for export.
            pending = list(included)
            while pending:
                key = pending.pop()
                if key not in project["candidates"]:
                    raise InvalidCheckpoint("选中候选不存在")
                for dependency in project["candidates"][key].get("context_candidate_ids", []):
                    if dependency not in included:
                        included.add(dependency)
                        pending.append(dependency)
            members = {}
            for key, candidate in packed["candidates"].items():
                if key not in included or candidate["status"] != "completed":
                    candidate.update(archive_status=candidate["status"], status="not_included",
                                     artifacts=[], audio_path=None, session_path=None, conditioning_session_path=None, line_audio=[])
                    continue
                if not self._usable(project_id, candidate):
                    raise InvalidCheckpoint("打包候选已损坏或缺失")
                for artifact in candidate["artifacts"]:
                    name = _safe_member(artifact["path"])
                    members[name] = self._directory(project_id) / name
            packed["archive_note"] = "External voices require matching UUID/version voice packs; candidate input snapshots preserve frozen-media text."
            project_data = _json_bytes(packed)
            manifest = {"format": "drama3-creative-project", "schema_version": SCHEMA_VERSION,
                        "project_id": project_id, "created_at": _now(), "voice_assets_included": False,
                        "files": {"project.json": {"bytes": len(project_data), "sha256": hashlib.sha256(project_data).hexdigest()}}}
            manifest["files"].update({name: _file_record(path) for name, path in members.items()})
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects an existing user export. A failed
            # partial archive is retained rather than deleting any user file.
            with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", _json_bytes(manifest))
                archive.writestr("project.json", project_data)
                for name, path in sorted(members.items()):
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != manifest["files"][name]["sha256"]:
                        raise InvalidCheckpoint("打包期间候选内容发生变化")
                    archive.writestr(name, data)
            return {"path": str(target), **_file_record(target), "files": len(manifest["files"]),
                    "included_candidates": sorted(key for key in included if packed["candidates"][key]["status"] == "completed"),
                    "voice_assets_included": False}

    def import_project(self, archive_path, *, name=None, before_commit=None):
        """Validate a complete archive before committing a new project identity."""
        try:
            if before_commit is not None and not callable(before_commit):
                raise ValueError("工程预提交回调无效")
            return self._import_project(archive_path, name=name, before_commit=before_commit)
        except (KeyError, TypeError, AttributeError, IndexError, zipfile.BadZipFile) as exc:
            raise ValueError("损坏或不支持的创作工程包") from exc

    def _import_project(self, archive_path, *, name=None, before_commit=None):
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not 2 <= len(infos) <= 10000 or sum(info.file_size for info in infos) > 2 * 1024**3:
                raise ValueError("工程包文件数量或体积超限")
            names, folded = {}, set()
            for info in infos:
                member = _safe_member(info.filename)
                normalized = unicodedata.normalize("NFC", member).casefold()
                if normalized in folded or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                    raise ValueError("工程包存在重复、加密、链接或无效成员")
                folded.add(normalized)
                names[member] = info
            if "manifest.json" not in names or "project.json" not in names:
                raise ValueError("工程包缺少清单或工程文件")
            if names["manifest.json"].file_size > 4 * 1024 * 1024 or names["project.json"].file_size > 64 * 1024 * 1024:
                raise ValueError("工程包元数据超限")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "drama3-creative-project" or manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("不是支持的DRAMA创作工程包")
            records = manifest.get("files")
            if not isinstance(records, dict) or set(names) != set(records) | {"manifest.json"}:
                raise ValueError("工程包成员与清单不一致")
            for member, record in records.items():
                _safe_member(member)
                if member != "project.json" and not member.startswith("artifacts/"):
                    raise ValueError("工程包包含未支持的资产位置")
                if not isinstance(record, dict) or record.get("bytes") != names[member].file_size:
                    raise ValueError("工程包成员长度不符")
                digest = hashlib.sha256()
                with archive.open(member) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != record.get("sha256"):
                    raise ValueError("工程包成员校验失败")
            project = json.loads(archive.read("project.json"))
            if not isinstance(project, dict) or project.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("工程文档格式无效")
            dictionary = normalize_dictionary(project.get("dictionary"))
            _lines(project.get("lines"), [], dictionary)
            _object(project.get("cast"), "导入选角")
            _object(project.get("settings"), "导入参数")
            if not isinstance(project.get("archived_lines", []), list):
                raise ValueError("工程归档台词表无效")
            candidates = project.get("candidates")
            if not isinstance(candidates, dict):
                raise ValueError("工程候选表无效")
            for key, candidate in candidates.items():
                if not _ID.fullmatch(key) or candidate.get("candidate_id") != key:
                    raise ValueError("工程候选身份无效")
                if not isinstance(candidate.get("fingerprint"), str) or not re.fullmatch("[0-9a-f]{64}", candidate["fingerprint"]):
                    raise ValueError("工程候选指纹无效")
                _object(candidate.get("input_snapshot"), "候选输入快照")
                _int(candidate.get("take_index"), "候选序号", 1, 32)
                _int(candidate.get("seed"), "候选种子", 0, 2**64 - 1)
                if not isinstance(candidate.get("line_ids"), list) or not candidate["line_ids"] or not all(
                        isinstance(line_id, str) and _LINE_ID.fullmatch(line_id) for line_id in candidate["line_ids"]):
                    raise ValueError("工程候选台词ID无效")
                artifacts = candidate.get("artifacts")
                if not isinstance(artifacts, list):
                    raise ValueError("工程候选清单无效")
                for artifact in artifacts:
                    member = _safe_member(artifact["path"])
                    if (not member.startswith(f"artifacts/{key}/") or member not in records
                            or records[member] != {"bytes": artifact["bytes"], "sha256": artifact["sha256"]}):
                        raise ValueError("候选媒体与工程包清单不符")
                if candidate.get("status") == "completed":
                    kinds = {artifact["kind"] for artifact in artifacts}
                    if not {"audio", "session"} <= kinds or len(kinds) != len(artifacts) or any(
                            kind not in ("audio", "session", "conditioning") and not kind.startswith("line:") for kind in kinds):
                        raise ValueError("完成候选缺少音频或原生Session")
                    for kind in ("audio", "session"):
                        if candidate.get(kind + "_path") != next(a["path"] for a in artifacts if a["kind"] == kind):
                            raise ValueError("候选媒体路径不一致")
                    if "conditioning" in kinds and candidate.get("conditioning_session_path") != next(a["path"] for a in artifacts if a["kind"] == "conditioning"):
                        raise ValueError("实际PCM条件归档路径不一致")
                    if candidate.get("conditioning_session_path") and "conditioning" not in kinds:
                        raise ValueError("实际PCM条件归档缺少校验记录")
                elif candidate.get("audio_path") or candidate.get("session_path") or candidate.get("conditioning_session_path"):
                    raise ValueError("未打包候选不能引用外部媒体路径")
                for item in candidate.get("line_audio", []):
                    if (item.get("line_id") not in candidate["line_ids"] or
                            not any(artifact["kind"] == "line:" + item["line_id"] and artifact["path"] == item.get("audio_path") for artifact in artifacts)):
                        raise ValueError("分句音频没有对应的校验记录")
                if len({item["line_id"] for item in candidate.get("line_audio", [])}) != len(candidate.get("line_audio", [])):
                    raise ValueError("分句音频台词ID重复")
            for line in project["lines"]:
                if any(key not in candidates for key in line.get("candidate_ids", [])):
                    raise ValueError("台词引用了不存在的候选")
                selected = line.get("selected_candidate_id")
                if selected and (selected not in candidates or candidates[selected].get("status") != "completed"):
                    raise ValueError("选中候选没有包含在工程包")
            _json_bytes(project)
            old_id = project.get("project_id")
            project.update(project_id=uuid.uuid4().hex, origin_project_id=old_id,
                           name=_text(name if name is not None else project.get("name", "导入工程"), "工程名称", 256, False),
                           revision=0, imported_at=_now(), runs=[],
                           import_warnings=["音色资产按UUID/版本解析；请安装匹配的音色包或重新选角。"])
            project.setdefault("archived_lines", [])
            project.setdefault("created_at", _now())
            directory = self._directory(project["project_id"])
            directory.mkdir(parents=True, exist_ok=False)
            for member in records:
                if member == "project.json":
                    continue
                if not member.startswith("artifacts/"):
                    raise ValueError("工程包包含未支持的资产位置")
                path = directory / member
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            for candidate in project["candidates"].values():
                if candidate.get("status") == "completed":
                    if not self._usable(project["project_id"], candidate):
                        raise InvalidCheckpoint("导入媒体校验失败")
                    _native_session_info(directory / candidate["session_path"])
                    if candidate.get("conditioning_session_path"):
                        condition = _native_session_info(directory / candidate["conditioning_session_path"])
                        source = _native_session_info(directory / candidate["session_path"])
                        binding = condition.get("from_source_view")
                        expected = {"audio_sha256": next(a["sha256"] for a in candidate["artifacts"] if a["kind"] == "audio"),
                                    "source_take_id": source["take_id"], "line_id": candidate["line_ids"][0],
                                    "codec_identity": candidate["input_snapshot"]["model_identity"].get("codec")}
                        if (len(candidate["line_ids"]) != 1 or condition["take_count"] != 1 or condition["transcript"] != source["transcript"]
                                or not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected.items())):
                            raise InvalidCheckpoint("导入实际PCM条件归档与候选不一致")
            if before_commit:
                before_commit(project, directory)
            self._save(project, initial=True)
            return self._view(project)


def _pcm_wav_info(path):
    try:
        with wave.open(str(path), "rb") as audio:
            rate, frames, channels, width = audio.getframerate(), audio.getnframes(), audio.getnchannels(), audio.getsampwidth()
            if audio.getcomptype() != "NONE" or frames <= 0 or channels not in (1, 2) or rate <= 0:
                raise InvalidCheckpoint("候选必须是非空单/双声道PCM WAV")
            expected_bytes = frames * channels * audio.getsampwidth()
            actual_bytes = 0
            while block := audio.readframes(65536):
                actual_bytes += len(block)
            if actual_bytes != expected_bytes:
                raise InvalidCheckpoint("WAV音频数据不完整")
        return {"sample_rate": rate, "channels": channels, "duration_seconds": frames / rate,
                "frames": frames, "sample_width": width}
    except (wave.Error, EOFError) as exc:
        raise InvalidCheckpoint("候选音频必须是完整PCM WAV") from exc


def _native_session_info(path):
    """Check the native Session envelope without importing tensor runtimes."""
    try:
        with Path(path).open("rb") as handle:
            length_raw = handle.read(8)
            if len(length_raw) != 8:
                raise InvalidCheckpoint("Session归档头不完整")
            length = int.from_bytes(length_raw, "little")
            if not 2 <= length <= 32 * 1024 * 1024 or length > Path(path).stat().st_size - 8:
                raise InvalidCheckpoint("Session归档头长度无效")
            header = json.loads(handle.read(length))
        metadata = json.loads(header["__metadata__"]["drama3_t8_session"])
        _json_bytes(metadata)
        if type(metadata.get("version")) is not int or metadata.get("version") not in (1, 2) or not _ID.fullmatch(metadata.get("session_id", "")):
            raise InvalidCheckpoint("不是支持的原生DRAMA Session")
        takes = metadata.get("takes")
        if not isinstance(takes, list) or not takes:
            raise InvalidCheckpoint("生成检查点必须含实际Take，不能只有参考")
        last = takes[-1]
        frame = header[last["key"]]
        if (frame.get("dtype") != "I64" or len(frame.get("shape", [])) != 2 or type(frame["shape"][0]) is not int
                or type(frame["shape"][1]) is not int or frame["shape"][0] <= 0 or frame["shape"][1] != 11):
            raise InvalidCheckpoint("Session的Take帧布局无效")
        offsets = frame["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= Path(path).stat().st_size - 8 - length
                or offsets[1] - offsets[0] != frame["shape"][0] * 11 * 8):
            raise InvalidCheckpoint("Session的Take数据不完整")
        first = _int(last.get("semantic_start"), "原生语义起点", 0, 252287)
        final = _int(last.get("semantic_end"), "原生语义终点", 0, 252287)
        separator = _int(last.get("separator_id"), "原生分隔符", 0, 252287)
        if final - first != 4095 or first <= separator <= final:
            raise InvalidCheckpoint("Session语义范围无效")
        if not isinstance(last.get("take_id"), str) or not _ID.fullmatch(last["take_id"]):
            raise InvalidCheckpoint("Session的Take身份无效")
        semantic_frames = 0
        with Path(path).open("rb") as handle:
            handle.seek(8 + length + offsets[0])
            remaining = frame["shape"][0]
            while remaining:
                count = min(remaining, 65536)
                block = handle.read(count * 88)
                if len(block) != count * 88:
                    raise InvalidCheckpoint("Session帧数据被截断")
                for row in struct.iter_unpack("<11q", block):
                    if not 0 <= row[0] < 252288 or any(not 0 <= code <= 4095 for code in row[1:]):
                        raise InvalidCheckpoint("Session包含无效的语义/音频代码")
                    semantic_frames += int(first <= row[0] <= final)
                remaining -= count
        if not semantic_frames:
            raise InvalidCheckpoint("完成Take必须含真实音频代码，不能只有控制符")
        detail = last.get("metadata", {})
        if detail.get("length_limit_reached") or detail.get("finish_reason") in ("stalled", "error", "cancelled"):
            raise InvalidCheckpoint("原生Take未完整结束")
        for key in ("seed", "requested_seed"):
            if key in detail and (type(detail[key]) is not int or not 0 <= detail[key] < 2**64):
                raise InvalidCheckpoint("原生Take种子记录无效")
        return {"session_id": metadata["session_id"], "take_id": last.get("take_id"),
                "transcript": _text(last.get("transcript"), "原生Take目标台词"),
                "take_count": len(takes), "from_source_view": detail.get("from_source_view"),
                "seed": detail.get("seed"), "requested_seed": detail.get("requested_seed")}
    except (KeyError, TypeError, ValueError, AttributeError, json.JSONDecodeError) as exc:
        if isinstance(exc, InvalidCheckpoint):
            raise
        raise InvalidCheckpoint("不是完整的原生DRAMA Session归档") from exc


def _safe_member(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value:
        raise ValueError("工程包成员路径无效")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts) or path.as_posix() != value:
        raise ValueError("工程包成员路径越界或不规范")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.endswith((" ", ".")) or part.split(".")[0].upper() in reserved or len(part) > 128 for part in path.parts):
        raise ValueError("工程包成员名称不兼容本地文件系统")
    return value


def text_quality(expected, actual, *, independent=False):
    """Flag transcript discrepancies without deleting or selecting performances."""
    import difflib
    if type(independent) is not bool:
        raise ValueError("独立转写标记必须是布尔值")
    def normalize(value):
        return "".join(c for c in unicodedata.normalize("NFKC", value).casefold()
                       if not unicodedata.category(c).startswith(("P", "Z")) and not c.isspace())
    wanted, heard = normalize(expected), normalize(actual)
    ratio = difflib.SequenceMatcher(None, wanted, heard, autojunk=False).ratio()
    return {"expected": expected, "recognized": actual, "normalized_match": wanted == heard,
            "similarity": ratio, "review_recommended": wanted != heard,
            "independent": independent, "scope": "ASR text discrepancy only; not voice similarity or acting quality"}
