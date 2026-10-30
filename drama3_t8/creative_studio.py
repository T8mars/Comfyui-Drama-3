"""Native Studio integration for the creative desk; one existing GPU queue."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import threading
import time
from urllib.parse import quote
import uuid
import zipfile

from .creative_audio import analyze_audio, export_selected, file_digest, read_audio_file
from .creative_workspace import CreativeWorkspace, model_file_identity, preview_text, text_quality
from .voice_library import VoiceLibrary
from .local_paths import canonical_path

_INIT = threading.RLock()
_ID = re.compile(r"[0-9a-f]{32}\Z")
STYLES = {"自然口播": "清晰、自然，像对朋友说话", "温柔安慰": "[gently] 温柔、耐心，留出自然停顿",
          "轻声秘密": "[whispering] 轻声说一个秘密", "生气吐槽": "[angry] 克制地生气，吐槽但保持吐字完整",
          "开心分享": "[happily] 开心地分享一个好消息", "紧张悬疑": "紧张而克制，逐步揭开悬念"}


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
    temporary.replace(path)


def _uint(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
        value = int(value)
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("随机种子应为0至18446744073709551615的整数")
    return value


def _delivery_direction(job):
    """Only the current explicitly requested overall/line delivery."""
    return job["settings"].get("direction", "")


def _transcribe_bounded(audios, count, paths, progress, **options):
    """Keep one ASR model per bounded group without truncating a long batch."""
    from itertools import islice
    from .r2t2.transcribe import transcribe_audio_batch
    source, results = iter(audios), []
    for first in range(0, count, 10000):
        size = min(10000, count - first)
        progress()
        batch = transcribe_audio_batch(islice(source, size), paths["r2t2_model"], paths["r2t2_projector"],
            paths["r2t2_vad"], check_cancel=progress, max_items=size, **options)
        if len(batch) != size:
            raise ValueError("转写未返回全部音频结果，已停止提交质检")
        results.extend(batch)
    return results


def creative_manager(studio):
    with _INIT:
        result = getattr(studio, "_creative_manager", None)
        if result is None:
            result = CreativeStudio(studio)
            studio._creative_manager = result
        return result


class CreativeStudio:
    def __init__(self, studio):
        self.studio = studio
        self.root, self.data = studio.root, studio.data
        self.config_path = self.data / "creative-settings.json"
        self.config = {"voice_library_path": "userdata/voice-library",
                       "workspace_path": "userdata/creative", "speaker_model_path": (
                           "models/campplus-zh-en.onnx" if (self.root / "models/campplus-zh-en.onnx").is_file() else ""),
                       "idle_unload_seconds": 180, "unload_after_batch": False}
        if self.config_path.is_file():
            self.config.update(json.loads(self.config_path.read_text(encoding="utf-8")))
        self.voices = VoiceLibrary(self.resolve_path(self.config["voice_library_path"]))
        self.workspace = CreativeWorkspace(self.resolve_path(self.config["workspace_path"]))
        self.downloads = self.data / "creative-downloads"
        self.downloads.mkdir(exist_ok=True)
        self.reports = self.data / "performance"
        self.reports.mkdir(exist_ok=True)
        self._identity_path = self.data / "creative-model-identities.json"
        self._identities = json.loads(self._identity_path.read_text(encoding="utf-8")) if self._identity_path.is_file() else {}
        self._identity_lock = threading.RLock()
        self.last_activity = time.monotonic()
        self.last_unload = None
        self.legacy_errors = []
        self._input_lock = threading.RLock()

    def resolve_path(self, value):
        path = Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()
        return canonical_path(path if path.is_absolute() else self.root / path)

    @property
    def inputs(self):
        path = self.workspace.root / "inputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_input(self, path):
        path = canonical_path(path)
        if path.parent != canonical_path(self.inputs):
            raise ValueError("上传文件不属于当前工作区")
        index = self.data / "creative-input-index.json"
        with self._input_lock:
            records = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
            records[path.name] = str(path)
            _atomic_json(index, records)
        return path

    def input_file(self, name):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("无效的录音文件编号")
        for base in (self.inputs, self.data / "inputs"):
            file = canonical_path(base / name)
            if file.parent == canonical_path(base) and file.is_file():
                return file
        index = self.data / "creative-input-index.json"
        records = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
        if name in records:
            path = canonical_path(records[name])
            if path.name == name and path.is_file():
                return path
        raise FileNotFoundError("原录音不存在，请重新上传")

    def submit(self, payload):
        if payload.get("mode") not in {"creative_reference", "creative_batch", "creative_quality", "creative_recording",
                                       "creative_delivery", "creative_benchmark", "creative_variation", "creative_apply_profile", "creative_reencode"}:
            raise ValueError("不支持的创作任务")
        return {"job_id": self.studio.submit(payload)}

    def configuration(self):
        from .creative_audio import ffmpeg_executable
        try:
            ffmpeg, ffmpeg_error = ffmpeg_executable(), None
        except ValueError as exc:
            ffmpeg, ffmpeg_error = None, str(exc)
        return {**self.config, "ffmpeg_available": bool(ffmpeg), "ffmpeg_path": ffmpeg, "ffmpeg_error": ffmpeg_error,
                "resolved_paths": {key: str(self.resolve_path(self.config[key])) for key in ("voice_library_path", "workspace_path")},
                "styles": STYLES, "legacy_errors": self.legacy_errors}

    def configure(self, patch):
        if set(patch) - set(self.config):
            raise ValueError("不支持的创作设置")
        with self.studio._activity_lock:
            if self.studio._active_job or not self.studio.pending.empty():
                raise ValueError("请等待任务结束后再切换资料目录")
            chosen = {**self.config, **patch}
            if type(chosen["idle_unload_seconds"]) is not int or not 0 <= chosen["idle_unload_seconds"] <= 86400:
                raise ValueError("空闲释放时间应为0至86400秒；0表示关闭")
            if type(chosen["unload_after_batch"]) is not bool:
                raise ValueError("批次结束释放必须为布尔值")
            for key in ("voice_library_path", "workspace_path"):
                if not isinstance(chosen[key], str) or not chosen[key].strip():
                    raise ValueError("资料目录不能为空")
                chosen[key] = chosen[key].strip().strip('"')
            if not isinstance(chosen["speaker_model_path"], str):
                raise ValueError("说话人模型路径应为字符串")
            voices = VoiceLibrary(self.resolve_path(chosen["voice_library_path"]))
            workspace = CreativeWorkspace(self.resolve_path(chosen["workspace_path"]))
            _atomic_json(self.config_path, chosen)
            self.config, self.voices, self.workspace = chosen, voices, workspace
        return self.configuration()

    def sync_legacy(self):
        with self.studio.connect() as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM voice_references")]
            for row in rows:
                record = db.execute("SELECT payload FROM jobs WHERE id=?", (row["id"],)).fetchone()
                if record:
                    payload = json.loads(record["payload"])
                    input_id = payload.get("input_id", "")
                    if input_id and Path(input_id).name == input_id:
                        source = self.data / "inputs" / input_id
                        if source.is_file():
                            row["source_path"] = str(source)
        result = self.voices.migrate_legacy(rows, self.data)
        self.legacy_errors = result["errors"]
        return result

    def library_view(self, query="", tags="", favorites=False, archived=False):
        self.sync_legacy()
        from .library_tools import describe_voice_versions
        rows = self.voices.list_voices(query, tags=[x.strip() for x in tags.split(",") if x.strip()],
                                      favorites_only=favorites, include_archived=archived)
        return {"voices": describe_voice_versions(self.voices, rows, self.studio.engine.codec_path,
                self.voices.root / "encoding-cache"), "casts": self.voices.list_casts(),
                "configuration": self.configuration()}

    def _file(self, root, relative):
        root = canonical_path(root)
        path = canonical_path(root / relative)
        if not path.is_relative_to(root) or path == root or not path.is_file():
            raise FileNotFoundError("文件不存在或路径不属于本工程")
        return path

    def project_file(self, project_id, relative):
        if not _ID.fullmatch(project_id):
            raise ValueError("无效工程ID")
        return self._file(self.workspace.projects / project_id, relative)

    def download_file(self, name):
        if Path(name).name != name:
            raise ValueError("无效下载文件名")
        return self._file(self.downloads, name)

    def project_view(self, project_id):
        result = self.workspace.get_project(project_id)
        from .creative_bundle import project_voice_library
        library = project_voice_library(self.workspace, project_id, self.voices)
        if library.root != self.voices.root:
            result["embedded_voices"] = library.list_voices(include_archived=True)
            for voice in result["embedded_voices"]:
                for version in voice["versions"]:
                    version["audio_url"] = f"/api/creative/projects/{project_id}/versions/{version['id']}/audio"
        for candidate in result["candidates"].values():
            for key in ("audio_path", "session_path"):
                relative = candidate.get("relative_" + key)
                if relative:
                    candidate[key.replace("_path", "_url")] = f"/api/creative/projects/{project_id}/files/{quote(relative, safe='/')}"
            for clip in candidate.get("line_audio", []):
                relative = clip.get("relative_audio_path")
                if relative:
                    clip["audio_url"] = f"/api/creative/projects/{project_id}/files/{quote(relative, safe='/')}"
        for line in result["lines"]:
            line["candidates"] = list(line["candidate_ids"])
            selected = result["candidates"].get(line.get("selected_candidate_id"))
            line["selected_candidate"] = ({"candidate_id": selected["candidate_id"], "status": selected["status"]} if selected else None)
        return result

    def create_project(self, payload):
        from .creative_context import assign_speakers
        cast = payload.get("cast", {})
        if payload.get("cast_id"):
            cast = self.voices.get_cast(payload["cast_id"])["bindings"]
        cast = assign_speakers(cast, payload["lines"])
        result = self.workspace.create_project(payload.get("name", "未命名作品"), payload["lines"], cast=cast,
            settings=payload.get("settings", {}), dictionary=payload.get("dictionary"))
        return self.project_view(result["project_id"])

    def update_project(self, project_id, payload):
        self.assert_project_idle(project_id)
        patch = {key: value for key, value in payload.items() if key != "expected_revision"}
        if "cast_id" in patch:
            patch["cast"] = self.voices.get_cast(patch.pop("cast_id"))["bindings"]
        if "lines" in patch or "cast" in patch:
            from .creative_context import assign_speakers
            previous = self.workspace.get_project(project_id)
            patch["cast"] = assign_speakers(patch.get("cast", previous["cast"]), patch.get("lines", previous["lines"]))
        from .creative_bundle import ensure_project_cast_assets
        self.workspace.update_project(project_id, patch, expected_revision=payload.get("expected_revision"),
            before_commit=lambda document, directory: ensure_project_cast_assets(self.workspace, project_id,
                self.voices, document["cast"], extra_version_ids=[version_id for line in document["lines"]
                    for version_id in line.get("reference_ids", [])]))
        return self.project_view(project_id)

    def model_identity(self, *, strong=True):
        from .performance import artifact_identity
        result = {}
        with self._identity_lock:
            for role in ("model", "codec", "text_head", "line_head"):
                path = self.studio.model_paths[role]
                if not path.is_file():
                    result[role] = None
                    continue
                stat = path.stat()
                key = str(path.resolve())
                signature = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
                cached = self._identities.get(key, {})
                if cached.get("signature") == signature and cached.get("sha256"):
                    result[role] = artifact_identity(path, sha256=cached["sha256"])
                elif strong:
                    result[role] = artifact_identity(path, hash_contents=True)
                    self._identities[key] = {"signature": signature, "sha256": result[role]["sha256"]}
                else:
                    result[role] = artifact_identity(path)
            if strong:
                _atomic_json(self._identity_path, self._identities)
        return result

    def _options(self, payload, *, strong=True):
        keys = {"takes_per_line", "line_ids", "mode", "sampling", "lookahead", "retry_failed", "force"}
        options = {key: payload[key] for key in keys if key in payload and key != "mode"}
        options["mode"] = payload.get("generation_mode", "per_line")
        options["base_seed"] = _uint(payload.get("base_seed", 42))
        models = self.model_identity(strong=strong)
        options["model_identity"] = {**{key: ({"sha256": value["sha256"], "bytes": value["bytes"]} if value and value.get("sha256") else value)
            for key, value in models.items()}, "runtime": {
            "settings": {key: self.studio.settings.get(key) for key in
                         ("gdn_backend", "moe_backend", "cuda_graph", "expert_placement", "expert_cache_gb", "scene_head_mode")},
            "source": self.source_identity(), "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"), "torchaudio": importlib.metadata.version("torchaudio")}}
        if "sampling" in options and "seed" in options["sampling"]:
            options["sampling"] = {**options["sampling"], "seed": _uint(options["sampling"]["seed"])}
        return options

    def source_identity(self):
        from .source_identity import semantic_source_identity
        return semantic_source_identity(Path(__file__).parent)

    def performance_binding(self, models=None):
        import torch
        import comfy.model_management as mm
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            gpu = {"name": str(device.name), "index": 0, "capability": [device.major, device.minor],
                   "total_memory_bytes": int(device.total_memory), "uuid": str(getattr(device, "uuid", "unavailable"))}
        else:
            gpu = {"name": "cpu"}
        try:
            from comfyui_version import __version__ as comfy_version
        except ImportError:
            comfy_version = "unknown"
        try:
            kitchen_version = importlib.metadata.version("comfy-kitchen")
        except importlib.metadata.PackageNotFoundError:
            kitchen_version = "unavailable"
        backend_versions = {"comfy_kitchen": kitchen_version}
        for package in ("flash-linear-attention", "flash-attn", "triton-windows", "triton"):
            try:
                backend_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                backend_versions[package] = "unavailable"
        return {"machine": platform.node() or platform.system(),
                "gpu": gpu,
                "torch": str(torch.__version__), "cuda": str(torch.version.cuda or "none"),
                "transformers": importlib.metadata.version("transformers"),
                "models": {key: value for key, value in (models or self.model_identity()).items() if key != "runtime"},
                "source": self.source_identity(),
                "host": {"comfy_version": str(comfy_version), "model_management_sha256": file_digest(mm.__file__),
                         "kitchen_version": kitchen_version, "os": platform.platform()},
                "backends": backend_versions}

    def plan(self, project_id, payload):
        options = self._filter_options(project_id, payload, self._options(payload))
        result = self.workspace.plan_generation(project_id, **options)
        fields = {"unit_index", "unit_id", "line_id", "line_ids", "role", "roles", "readout_text", "subtitle_text",
                  "take_index", "seed", "seed_text", "mode", "fingerprint", "status"}
        result["jobs"] = [{key: value for key, value in job.items() if key in fields} |
                          {"candidate_id": job["candidate"]["candidate_id"] if job.get("candidate") else None}
                          for job in result["jobs"]]
        return result

    def _filter_options(self, project_id, payload, options):
        if not payload.get("only_failed") and not payload.get("suspect_only"):
            return options
        if payload.get("only_failed") and payload.get("suspect_only"):
            raise ValueError("仅失败与仅疑似需分别执行")
        project = self.workspace.get_project(project_id)
        requested = set(options["line_ids"]) if options.get("line_ids") is not None else {line["line_id"] for line in project["lines"]}
        if payload.get("only_failed"):
            failed = {item["fingerprint"] for item in project["candidates"].values() if item["status"] in {"failed", "interrupted", "running"}}
            plan = self.workspace.plan_generation(project_id, **options)
            chosen = {line_id for job in plan["jobs"] if job["fingerprint"] in failed for line_id in job["line_ids"]}
            return {**options, "line_ids": sorted(chosen & requested), "force": False, "retry_failed": True}
        chosen = set()
        for line in project["lines"]:
            selected = line["selected_candidate"]
            if selected and selected.get("quality", {}).get("review_recommended"):
                chosen.add(line["line_id"])
        return {**options, "line_ids": sorted(chosen & requested), "force": True}

    def review(self, project_id, payload):
        self.assert_project_idle(project_id)
        self.workspace.set_candidate_review(project_id, **payload)
        return self.project_view(project_id)

    def assert_project_idle(self, project_id):
        from .workspace_lock import WorkspaceBusy
        with self.studio._activity_lock:
            active = self.studio._active_job
            if active and self.studio.get_job(active)["payload"].get("project_id") == project_id:
                raise WorkspaceBusy("该工程正在生成或质检，请等待任务结束后再保存或选片")

    def export_voices(self, payload):
        target = self.downloads / f"voices-{uuid.uuid4().hex}.zip"
        result = self.voices.export_bundle(target, **{key: payload[key] for key in ("voice_ids", "cast_ids", "include_sources") if key in payload})
        return {"download": f"/api/creative/downloads/{target.name}", "manifest": result}

    def export_project(self, project_id, payload):
        from .creative_bundle import export_creative_bundle
        target = self.downloads / f"project-{uuid.uuid4().hex}.zip"
        result = export_creative_bundle(self.workspace, self.voices, project_id, target,
            include_all=payload.get("include_all", False), include_sources=payload.get("include_sources", True))
        return {"download": f"/api/creative/downloads/{target.name}", "manifest": result}

    def import_project(self, path):
        from .creative_bundle import import_creative_bundle
        result = import_creative_bundle(self.workspace, path)
        return self.project_view(result["project_id"])

    def script_preview(self, payload):
        if "lines" in payload:
            rows = []
            if not isinstance(payload["lines"], list) or not payload["lines"]:
                raise ValueError("请提供非空台词表")
            for line in payload["lines"]:
                original = line.get("original_text", "")
                preview = preview_text(original, payload.get("dictionary"))
                rows.append({**line, **preview,
                    "readout_text": line.get("readout_override") if line.get("readout_override") is not None else preview["readout_text"],
                    "subtitle_text": line.get("subtitle_override") if line.get("subtitle_override") is not None else original})
            return {"lines": rows, "styles": STYLES, "kind": "table"}
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入剧本")
        kind = payload.get("kind", "lines")
        rows = []
        if kind == "srt":
            from .subtitles import parse_srt
            for index, cue in enumerate(parse_srt(text)):
                rows.append({"line_id": f"L{index + 1:02}", "role": "旁白", "original_text": cue.text,
                             "start_seconds": cue.start, "target_seconds": cue.end - cue.start})
        else:
            if kind == "long":
                from .native.chunking import split_text
                chunks = split_text(text, int(payload.get("chunk_bytes", 300)))
            else:
                chunks = [line.strip() for line in text.splitlines() if line.strip()]
            for index, line in enumerate(chunks):
                match = re.match(r"^([^：:\n]{1,40})[：:]\s*(.+)$", line, re.S) if kind != "long" else None
                role, original = (match.group(1).strip(), match.group(2)) if match else ("旁白", line)
                rows.append({"line_id": f"L{index + 1:02}", "role": role, "original_text": original})
        for row in rows:
            row.update(preview_text(row["original_text"], payload.get("dictionary")))
            row.setdefault("readout_override", None)
            row.setdefault("subtitle_override", None)
        return {"lines": rows, "styles": STYLES, "kind": kind}

    def execute(self, job_id, payload, progress):
        operation = payload["mode"].removeprefix("creative_")
        self.last_activity = time.monotonic()
        try:
            return getattr(self, "execute_" + operation)(job_id, payload, progress)
        finally:
            self.last_activity = time.monotonic()
            if self.config["unload_after_batch"] and operation in {"batch", "quality", "recording", "benchmark", "variation", "reference", "reencode"}:
                self.studio.engine.unload()

    def execute_reference(self, job_id, payload, progress):
        from .studio import read_audio
        from .state import Session
        path = self.input_file(payload["input_id"])
        waveform, rate = read_audio(path)
        duration = waveform.shape[-1] / rate
        start = float(payload.get("start_seconds", 0))
        end = float(payload["end_seconds"]) if payload.get("end_seconds") is not None else duration
        if not math.isfinite(start + end) or not 0 <= start < end <= duration + 1 / rate:
            raise ValueError("参考选区必须位于原录音内")
        selected = waveform[:, round(start * rate):round(end * rate)]
        text = payload.get("transcript", "").strip()
        if not text:
            self.studio.engine.unload()
            result = self.studio.engine.transcribe_detailed(selected, rate, language=payload.get("language") or "auto", check_cancel=progress)
            if result.get("truncated") or not result["text"].strip():
                raise ValueError("参考转写为空或截断，请校稿后重试")
            text = result["text"]
        progress()
        from .reference_cache import encode_reference_cached
        reference = encode_reference_cached(self.studio.engine, selected, rate, text, payload.get("name", ""),
            self.voices.root / "encoding-cache", trim_silence=payload.get("trim_silence", False), normalize=payload.get("normalize", False))
        reference = replace(reference, processing={**reference.processing,
            "source_interval": {"start_seconds": start, "end_seconds": end, "source_sha256": file_digest(path)}})
        import soundfile as sf
        import comfy.audio
        from .audio_processing import prepare_reference
        preview = selected.mean(0, keepdim=True)
        if rate != 44100:
            preview = comfy.audio.resample(preview, rate, 44100)
        preview, _ = prepare_reference(preview, 44100, trim_silence=payload.get("trim_silence", False),
                                       normalize=payload.get("normalize", False))
        clip = self.inputs / f"{job_id}-reference.wav"
        sf.write(clip, preview.T.numpy(), 44100, subtype="FLOAT")
        progress()
        self.studio.begin_commit(job_id, progress)
        voice = self.voices.get_voice(payload["voice_id"]) if payload.get("voice_id") else self.voices.create_voice(
            payload.get("name") or "未命名音色", tags=[tag.strip() for tag in payload.get("tags", "").split(",") if tag.strip()],
            notes=payload.get("notes", ""))
        version = self.voices.add_version(voice["id"], Session((reference,)), source_path=path, preview_path=clip,
            language=payload.get("language", ""), delivery=payload.get("delivery", ""))
        cached = reference.processing.get("encoding_cache") or {}
        return {"voice_id": voice["id"], "version_id": version["id"], "name": voice["name"],
                "encoding_cache": cached, "message": "音色版本已保存" + (" · 已复用参考编码" if cached.get("status") == "hit" else "")}

    def execute_reencode(self, job_id, payload, progress):
        from .library_tools import execute_reencode
        return execute_reencode(self, job_id, payload, progress)

    def _session(self, job):
        from .state import Reference, Session, load_session
        from .creative_bundle import project_voice_library
        voices = project_voice_library(self.workspace, job["project_id"], self.voices)
        choices, seen = [], set()
        for line in job["lines"]:
            binding = job["cast"].get(line["role"]) or {}
            group = line.get("reference_group")
            explicit = line.get("reference_ids")
            versions = explicit or (binding.get("reference_groups", {}).get(group) if group else None)
            if not explicit and group and not versions:
                raise ValueError(f"角色 {line['role']} 没有参考组 {group}")
            versions = versions or ([binding["version_id"]] if binding.get("version_id") else [])
            owner = binding.get("voice_id")
            for version_id in versions:
                record = voices.get_version(version_id)
                if owner is None:
                    owner = record["voice_id"]
                if record["voice_id"] != owner:
                    raise ValueError(f"角色 {line['role']} 的参考版本必须属于同一绑定音色")
                key = (version_id, binding.get("speaker", 0))
                if key not in seen:
                    seen.add(key)
                    choices.append({"voice_id": owner, "version_id": version_id, "speaker": binding.get("speaker", 0)})
        direction = _delivery_direction(job)
        session = voices.load_selection(choices, direction=direction) if choices else Session(direction=direction)
        expected_codec = job.get("model_identity", {}).get("codec") or {}
        for reference in session.references:
            encoded_codec = reference.processing.get("codec_identity") or {}
            if encoded_codec.get("sha256") and expected_codec.get("sha256") and encoded_codec["sha256"] != expected_codec["sha256"]:
                raise ValueError("参考所用Codec与当前权重不同，请从原录音重新编码该音色版本")
        from .creative_context import condition_creative_context
        session, metadata = condition_creative_context(job, session)
        return session, choices, metadata

    @staticmethod
    def _infill_candidate_input(job, source_take):
        """Bind the selected WAV and native role IDs, independently of project IDs."""
        import soundfile as sf
        import torch
        from .creative_workspace import _file_record
        from .editor import fingerprint
        from .render_infill import normalize_subtype, playback_identity

        candidate, operation = job["source_candidate"], job["operation"]
        path = Path(candidate["audio_path"])
        records = [record for record in candidate.get("artifacts", []) if record.get("kind") == "audio"]
        before = _file_record(path)
        if len(records) != 1 or any(records[0].get(key) != before[key] for key in ("bytes", "sha256")):
            raise ValueError("源候选的实际音频已改变，不能继续局部修改")
        info = sf.info(str(path))
        subtype = normalize_subtype(info.subtype)
        pcm, rate = read_audio_file(path)
        if rate != 44100 or info.samplerate != rate or info.channels != 1 or pcm.shape != (1, info.frames):
            raise ValueError("局部修改需要完整的44.1kHz单声道候选音频")
        if _file_record(path) != before:
            raise ValueError("读取期间源候选音频改变，请重新选择")
        expected = playback_identity(torch.from_numpy(pcm), subtype, source=fingerprint(source_take))

        def actor(value):
            if type(value) is not int or not 0 <= value < 16:
                raise ValueError("原生目标speaker必须是0至15的整数")
            return value

        def tag(text):
            numbers = {int(value) for value in re.findall(r"<\|speaker:([0-9]+)\|>", text)}
            return actor(next(iter(numbers))) if len(numbers) == 1 else None

        metadata = source_take.metadata
        policy = metadata.get("render_infill") or {}
        structured = metadata.get("task") == "scene" or "scene_chunking" in metadata or policy.get("scene_text") is True
        ids = metadata.get("creative_line_ids") or candidate.get("line_ids") or job["line_ids"]
        if set(ids) != {line["line_id"] for line in job["lines"]} or len(set(ids)) != len(ids):
            raise ValueError("局部修改必须保留完整源场景的台词表")
        clips = candidate.get("line_audio") or []
        if not clips:
            recorded = candidate.get("metadata", {}).get("line_timings") or []
            clips = [{"line_id": ids[item["line"]], "start_seconds": item["start"], "end_seconds": item["end"]}
                     for item in recorded if type(item.get("line")) is int and 0 <= item["line"] < len(ids)]
        if not clips and len(ids) == 1:
            clips = [{"line_id": ids[0], "start_seconds": 0., "end_seconds": info.frames/rate}]
        by_id = {item["line_id"]: item for item in clips}
        if len(by_id) != len(clips) or set(by_id) != set(ids):
            raise ValueError("源候选缺少完整的逐句真实时间轴，请先重新生成")
        intervals, previous = [], 0
        for index, line in enumerate(job["lines"]):
            clip = by_id[line["line_id"]]
            first, last = clip["start_seconds"], clip["end_seconds"]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (first, last)):
                raise ValueError("源候选逐句时间轴无效")
            first, last = round(first*rate), round(last*rate)
            if not previous <= first < last <= info.frames:
                raise ValueError("源候选逐句时间轴越界或重叠")
            intervals.append({"line": index, "line_id": line["line_id"], "start_sample": first, "end_sample": last})
            previous = last
        selected_id = operation.get("line_id", job["line_id"])
        selected_clip = next((item for item in intervals if item["line_id"] == selected_id), None)
        first, last = float(operation["start_seconds"]), float(operation["end_seconds"])
        if not math.isfinite(first) or not math.isfinite(last):
            raise ValueError("局部修改选区必须是有限秒数")
        first, last = round(first*rate), round(last*rate)
        if selected_clip is None or not selected_clip["start_sample"] <= first <= last <= selected_clip["end_sample"]:
            raise ValueError("局部修改选区必须位于目标台词的真实音频范围内；跨行请使用导演重演")
        speakers = metadata.get("line_speakers")
        scripts = policy.get("history_text", "").splitlines() if policy.get("scene_text") else metadata.get("script_lines")
        mapping = policy.get("scene_speaker_map") or metadata.get("scene_speaker_map") or {}
        native = []
        for line in job["lines"]:
            index = ids.index(line["line_id"]) if line["line_id"] in ids else None
            selected = None
            if isinstance(speakers, list) and len(speakers) == len(ids) and index is not None:
                selected = actor(speakers[index])
            if selected is None and isinstance(scripts, (list, tuple)) and len(scripts) == len(ids) and index is not None:
                from .scene_cast import _render_line
                selected = tag(_render_line(scripts[index], mapping) if mapping else scripts[index])
            if selected is None and line["role"] in mapping:
                selected = actor(mapping[line["role"]])
            if selected is None and structured and index is not None:
                recorded = candidate.get("metadata", {}).get("line_timings") or []
                known = {actor(item["speaker"]) for item in recorded if item.get("line") == index and "speaker" in item}
                if len(known) == 1:
                    selected = known.pop()
            if selected is None and not structured:
                if metadata.get("creative_local_speaker") is not None:
                    selected = actor(metadata["creative_local_speaker"])
                else:
                    selected = tag(source_take.prompt_text)
                    if selected is None:
                        selected = actor(policy["target_speaker"]) if "target_speaker" in policy else 0
            native.append(selected)
        index = next((index for index, line in enumerate(job["lines"]) if line["line_id"] == selected_id), None)
        selected = native[index] if index is not None else None
        explicit = operation.get("target_speaker")
        if explicit is not None:
            selected = actor(explicit)
        if selected is None:
            # Rendered edits may divide one original line into several regions.
            # Their persisted timeline still uses actual native speaker IDs.
            timeline = policy.get("timeline") or candidate.get("metadata", {}).get("line_timings") or []
            start_seconds, end_seconds = float(operation["start_seconds"]), float(operation["end_seconds"])
            known = {actor(item["speaker"]) for item in timeline if "speaker" in item
                     and (item["start"] < end_seconds and item["end"] > start_seconds if start_seconds != end_seconds else item["start"] <= start_seconds < item["end"])}
            if len(known) == 1:
                selected = known.pop()
                if index is not None:
                    native[index] = selected
        if selected is None:
            raise ValueError("无法确认该选区的原生角色，请显式指定target_speaker")
        return {"expected_view": expected, "target_speaker": selected, "native_speakers": native,
                "source_intervals": intervals, "selected_line_id": selected_id,
                "edit_start_sample": first, "edit_end_sample": last}

    @staticmethod
    def _infill_delivery_timeline(selected, take, samples, sample_rate):
        """Project line bounds from preserved actual PCM, not FIM region indices."""
        policy = take.metadata["render_infill"]
        if (sample_rate != 44100 or policy["start_sample"] != selected["edit_start_sample"]
                or policy["end_sample"] != selected["edit_end_sample"] or policy["output_samples"] != samples):
            raise ValueError("局部修改的实际PCM与所选区间不一致")
        delta = policy["inserted_samples"] - (policy["end_sample"]-policy["start_sample"])
        timeline = []
        for item in selected["source_intervals"]:
            first, last = item["start_sample"], item["end_sample"]
            if item["line_id"] == selected["selected_line_id"]:
                last += delta
            elif first >= policy["end_sample"]:
                first, last = first+delta, last+delta
            if not 0 <= first < last <= samples:
                raise ValueError("局部修改后目标台词为空或时间轴越界，不能提交工程候选")
            timeline.append({"line": item["line"], "start": first/sample_rate, "end": last/sample_rate})
        return timeline

    def _generate(self, job, directory, progress, parent_recorder=None):
        from .engine import Sampling
        from .performance import PerformanceRecorder
        from .state import save_session
        import soundfile as sf
        engine = self.studio.engine
        recorder = PerformanceRecorder({"project_id": job["project_id"], "line_ids": job["line_ids"], "seed": job["seed"]},
                                       self.performance_binding(job["model_identity"]))
        recorder.start()
        try:
            operation = job.get("operation")
            if operation:
                from .state import load_session
                session = load_session(job["source_candidate"]["session_path"])
                old_codec = job["source_candidate"]["input_snapshot"].get("model_identity", {}).get("codec") or {}
                current_codec = job.get("model_identity", {}).get("codec") or {}
                if old_codec.get("sha256") and current_codec.get("sha256") and old_codec["sha256"] != current_codec["sha256"]:
                    raise ValueError("源候选的Codec与当前权重不同，请先重新生成后改稿")
                references = job["source_candidate"].get("metadata", {}).get("reference_versions", [])
                source = session.takes[-1]
                source_task = source.metadata.get("task")
                if operation["kind"] == "revise":
                    session = replace(session, direction=_delivery_direction(job))
                context_metadata = {"creative_context": {"policy": "source_aware_variation", "mode": job["mode"],
                    "operation": operation["kind"], "candidate_ids": [job["source_candidate"]["candidate_id"]]},
                    "creative_global_speaker": source.metadata.get("creative_global_speaker", 0),
                    "creative_local_speaker": source.metadata.get("creative_local_speaker", 0)}
                from .creative_quality import validate_reference_retry, REFERENCE_ONLY_POLICY
                if validate_reference_retry(self.workspace, job, session):
                    # The independently rejected speech cannot be paired with
                    # its intended target text as acoustic history again.
                    session = replace(session, takes=())
                    context_metadata["creative_context"].update(policy=REFERENCE_ONLY_POLICY,
                        candidate_ids=[], source_candidate_ids=[job["source_candidate"]["candidate_id"]])
            else:
                session, references, context_metadata = self._session(job)
            infill_input = self._infill_candidate_input(job, source) if operation and operation["kind"] == "infill" else None
            sampling = Sampling(**job["sampling"])
            sampling.validate()
            with recorder.phase("model.load", cache_state="warm" if engine.drama is not None else "cold"):
                engine.load_model()
            def on_codes(frame):
                recorder.mark_first_codes()
                if parent_recorder is not None:
                    parent_recorder.mark_first_codes()
            lines = []
            from .dialogue import recast
            for line in job["lines"]:
                binding = job["cast"].get(line["role"]) or {}
                speaker = binding.get("speaker", job.get("speaker_map", {}).get(line["role"],
                    list(dict.fromkeys(job["roles"])).index(line["role"])))
                if len(job["lines"]) == 1:
                    speaker = context_metadata.get("creative_local_speaker", speaker)
                from .prompts import directed_text
                text = recast(line["readout_text"], speaker)
                # FIM is governed by the full transcript and actual PCM region;
                # ordinary delivery controls are not a new FIM protocol.
                lines.append(text if infill_input is not None else directed_text(text, line.get("direction", "")))
            with recorder.phase("generate", cache_state="warm" if engine.drama is not None else "cold"):
                if operation and operation["kind"] == "infill":
                    selected = infill_input
                    infill_lines = [recast(text, actor) if actor is not None else text
                                    for text, actor in zip(lines, selected["native_speakers"])]
                    take = engine.infill(session.takes[-1], "\n".join(infill_lines), float(operation["start_seconds"]),
                        float(operation["end_seconds"]), sampling, operation.get("duration_frames"), progress,
                        target_speaker=selected["target_speaker"], expected_view=selected["expected_view"])
                    result = session.append(take)
                elif operation and operation["kind"] == "revise":
                    note = operation.get("note", "")
                    if len(lines) > 1 or source_task == "scene":
                        result = engine.scene(lines, job["settings"].get("scene", ""), job["settings"].get("characters", ""),
                            references=session.references, sampling=sampling, progress=progress, session=session,
                            note=note, on_codes=on_codes, generation_mode="together",
                            split_long_text=job["settings"].get("split_long_text", False), chunk_bytes=job["settings"].get("chunk_bytes", 300))
                    else:
                        result = engine.speak(lines[0], session, sampling, on_codes, progress, note=note)
                elif job["mode"] in {"together", "per_speaker"}:
                    result = engine.scene(lines, job["settings"].get("scene", ""), job["settings"].get("characters", ""),
                        references=session.references, sampling=sampling, progress=progress, session=session, on_codes=on_codes,
                        generation_mode="together",
                        split_long_text=job["settings"].get("split_long_text", False), chunk_bytes=job["settings"].get("chunk_bytes", 300))
                elif job["mode"] == "text_stream":
                    lookahead = context_metadata.get("creative_context", {}).get("lookahead_text", "")
                    result = engine.stream_text(lines[0], lookahead, session, sampling, on_codes, progress)
                else:
                    from .scene_chunks import plan_scene, chunked_scene
                    settings = job["settings"]
                    mode = "together" if job["mode"] == "per_line_context" else "per_line"
                    split = settings.get("split_long_text", False)
                    budget = settings.get("chunk_bytes", 300)
                    multiple = split and len(plan_scene(lines, session.references, session.takes,
                        mode=mode, chunk_bytes=budget, request_direction=session.direction)["units"]) > 1
                    if multiple:
                        result = chunked_scene(engine, lines, "", "",
                            session.references, sampling, progress, session, "", on_codes, mode, budget,
                            history_limit=3, fixed_reference_anchor=job["mode"] == "long_form", on_observed_codes=on_codes)
                    else:
                        result = engine.speak(lines[0], session, sampling, on_codes, progress)
            take = replace(result.takes[-1], metadata={**result.takes[-1].metadata, **context_metadata,
                                                     "creative_line_ids": job["line_ids"]})
            from .prompts import spoken_text
            expected_text = re.sub(r"\s+", "", spoken_text("\n".join(lines), scene=True))
            recorded_text = re.sub(r"\s+", "", spoken_text(take.transcript))
            if recorded_text != expected_text:
                raise ValueError("原生演出台词记录与当前朗读稿不一致，拒绝提交错误候选")
            # The project archives dependencies separately. Do not copy every
            # previous root Take into every later candidate Session.
            result = replace(result, takes=(take,))
            with recorder.phase("decode", cache_state="warm" if engine.codec is not None else "cold"):
                audio, timeline = engine.decode_take(take, with_timeline=True)
            recorder.mark_first_pcm()
            if parent_recorder is not None:
                parent_recorder.mark_first_pcm()
            progress()
            wave = audio["waveform"][0].cpu().numpy()
            if infill_input is not None:
                timeline = self._infill_delivery_timeline(infill_input, take, wave.shape[-1], audio["sample_rate"])
            stats = analyze_audio(wave, audio["sample_rate"])
            experts = engine.expert_stats()
            recorder.record_backend("moe", engine.moe_backend, experts["actual_backend"], experts["fallback_reason"])
            recorder.record_backend("placement", engine.expert_placement, experts["placement"])
            observed_gdn = [key for key, count in take.metadata.get("gdn_calls_by_backend", {}).items() if count > 0]
            actual_gdn = observed_gdn[0] if len(observed_gdn) == 1 else "mixed" if observed_gdn else "not_run"
            requested_gdn = getattr(engine, "requested_backend", engine.gdn_backend)
            recorder.record_backend("gdn", requested_gdn, actual_gdn,
                take.metadata.get("acceleration_fallback") or (f"实际使用 {actual_gdn}" if actual_gdn != requested_gdn else None))
            graphs = take.metadata.get("cuda_graph") or getattr(engine.drama, "last_graph_stats", None) or {}
            graph_used = bool(graphs.get("captures") or graphs.get("replays") or graphs.get("tower_captures") or graphs.get("tower_replays"))
            fallback = [str(graphs["disabled_reason"])] if graphs.get("disabled_reason") else []
            if graphs.get("disabled_reasons"):
                reasons = graphs["disabled_reasons"]
                fallback.append(json.dumps(reasons, ensure_ascii=False) if not isinstance(reasons, str) else reasons)
            if graphs.get("fallback_reasons"):
                reasons = graphs["fallback_reasons"]
                fallback.append(json.dumps(reasons, ensure_ascii=False) if not isinstance(reasons, str) else reasons)
            recorder.record_backend("graph", engine.cuda_graph, graph_used, "; ".join(fallback) or None)
            metadata = {**take.metadata, "reference_versions": references,
                        "creative_line_ids": job["line_ids"], "audio_analysis": stats}
            target = directory / f"{job['line_id']}_take{job['take_index']}.wav"
            archive = directory / "session.safetensors"
            sf.write(target, wave.T, audio["sample_rate"], subtype="PCM_24")
            if not timeline and len(job["lines"]) > 1:
                from .native.codec import FRAME_LENGTH
                pieces = take.segments()
                if len(pieces) == len(job["lines"]) and sum(piece.shape[-1] for piece in pieces) * FRAME_LENGTH == wave.shape[-1]:
                    cursor = 0
                    for index, piece in enumerate(pieces):
                        end = cursor + piece.shape[-1] * FRAME_LENGTH
                        timeline.append({"line": index, "start": cursor / audio["sample_rate"], "end": end / audio["sample_rate"]})
                        cursor = end
            if not timeline and len(job["lines"]) == 1:
                timeline = [{"line": 0, "start": 0, "end": stats["duration_seconds"]}]
            line_audio = []
            line_analysis = {}
            for timing in timeline:
                index = int(timing.get("line", len(line_audio)))
                if not 0 <= index < len(job["lines"]):
                    raise ValueError("实际场景行号与工程不符")
                line_id = job["lines"][index]["line_id"]
                first, last = round(timing["start"] * audio["sample_rate"]), round(timing["end"] * audio["sample_rate"])
                if not 0 <= first < last <= wave.shape[-1]:
                    raise ValueError("实际场景时间轴越界")
                file = directory / f"line-{line_id}.wav"
                sf.write(file, wave[:, first:last].T, audio["sample_rate"], subtype="PCM_24")
                line_analysis[line_id] = analyze_audio(wave[:, first:last], audio["sample_rate"])
                line_audio.append({"line_id": line_id, "audio_path": str(file),
                                   "start_seconds": first / audio["sample_rate"], "end_seconds": last / audio["sample_rate"]})
            metadata["line_timings"] = timeline
            metadata["line_audio_analysis"] = line_analysis
            save_session(result, archive)
            if operation and operation["kind"] == "infill":
                from .creative_infill_quality import analyze_infill_insert
                policy = take.metadata["render_infill"]
                edit_signal = analyze_infill_insert(
                    spoken_text(source.transcript), spoken_text(take.transcript),
                    wave, policy["start_sample"], policy["inserted_samples"], audio["sample_rate"])
                _atomic_json(directory / "infill-energy.json", edit_signal)
                if edit_signal["rejected"]:
                    raise RuntimeError(
                        "局部改配需要新语音，但实际替换段接近静音；本次候选失败，原选片保持。"
                        "实际WAV、Session与infill-energy.json已保留，请调整词级选区或换种子重试。")
            conditioning = None
            if operation and operation["kind"] == "infill" and len(job["lines"]) == 1:
                from .state import Session, Take
                from .dialogue import audio_frames
                import torch
                actual_pcm, actual_rate = read_audio_file(target)
                with recorder.phase("conditioning.encode", cache_state="warm"):
                    encoded = engine.encode_reference(torch.from_numpy(actual_pcm), actual_rate, take.transcript, job["role"])
                conditioning_take = Take(take.prompt_text, take.transcript, audio_frames(encoded.codes, take),
                    take.semantic_start, take.semantic_end, take.separator_id,
                    {**context_metadata, "conditioning_method": "codec_encode_selected_pcm", "from_source_view": {
                        "audio_sha256": file_digest(target), "source_take_id": take.take_id, "line_id": job["line_id"],
                        "codec_identity": job["model_identity"].get("codec")}})
                conditioning = directory / "conditioning.safetensors"
                save_session(Session(result.references, (conditioning_take,), result.direction), conditioning)
            performance = recorder.finish("ok")
            metadata["performance"] = performance
            _atomic_json(directory / "performance.json", performance)
            _atomic_json(self.reports / f"{uuid.uuid4().hex}.json", performance)
            returned = {"complete": True, "seed": int(take.metadata.get("seed", sampling.seed)), "audio_path": str(target),
                    "session_path": str(archive), "sample_rate": audio["sample_rate"], "duration_seconds": stats["duration_seconds"],
                    "metadata": metadata, "transcript": take.transcript, "line_audio": line_audio,
                    "quality": {"technical": stats, "asr_checked": False}}
            if conditioning is not None:
                returned["conditioning_session_path"] = str(conditioning)
            return returned
        except BaseException as exc:
            recorder.finish("cancelled" if type(exc).__name__ in {"Cancelled", "InterruptProcessingException"} else "error", error_type=type(exc).__name__)
            raise

    def execute_batch(self, job_id, payload, progress):
        from .performance import PerformanceRecorder
        from .creative_progress import BatchProgress
        cancelled = [False]
        report_progress = progress
        def progress(*args):
            try:
                return report_progress(*args)
            except BaseException as exc:
                if type(exc).__name__ in {"Cancelled", "InterruptProcessingException", "KeyboardInterrupt"}:
                    cancelled[0] = True
                raise
        retry_count = payload.get("max_quality_retries", 0)
        if type(retry_count) is not int or not 0 <= retry_count <= 2:
            raise ValueError("疑似漏词补录次数应为0、1或2")
        options = self._filter_options(payload["project_id"], payload, self._options(payload))
        import contextlib
        plan = self.workspace.plan_generation(payload["project_id"], **options)
        meter = BatchProgress(progress, plan,
            quality=bool(payload.get("quality_after") or payload.get("quality_check")), retries=retry_count)
        def generate(job, directory):
            local_progress = meter.start(job)
            return self._generate(job, directory, local_progress, parent_recorder=recorder)
        recorder = PerformanceRecorder({"kind": "creative_batch", "project_id": payload["project_id"]},
                                       self.performance_binding(options["model_identity"]))
        recorder.start()
        try:
            if plan["pending"]:
                with recorder.phase("model.load", cache_state="warm" if self.studio.engine.drama is not None else "cold"):
                    self.studio.engine.load_model()
            scope = self.studio.engine.generation_session() if plan["pending"] else contextlib.nullcontext()
            with scope, recorder.phase("tts.batch", cache_state="warm"):
                result = self.workspace.run_generation(payload["project_id"],
                    generate,
                    progress=meter.updated, expected_revision=payload.get("expected_revision"),
                    cancelled=lambda: cancelled[0],
                    continue_on_error=payload.get("continue_on_error", True), **options)
            if result["status"] != "completed":
                raise ValueError(f"本批次状态 {result['status']}；已生成 {len(result['generated'])}、失败 {len(result['failed'])}，产物保留在工程中")
            meter.generation_done(result["plan"])
            quality = None
            if payload.get("quality_after") or payload.get("quality_check") or retry_count:
                with recorder.phase("asr.batch", cache_state="unknown"):
                    quality = self.execute_quality(job_id, {"project_id": payload["project_id"],
                        **({"candidate_ids": result["generated"]} if result["generated"] else {})}, meter.quality.start())
                meter.quality.finish()
                if retry_count:
                    from .creative_quality import run_quality_retries
                    with recorder.phase("quality.retake", cache_state="unknown"):
                        quality = run_quality_retries(self, job_id, payload, quality, meter.check, progress_scope=meter.retries)
            meter.commit()
            self.studio.begin_commit(job_id, meter.check)
            meter.committed()
            report = recorder.finish("ok")
            _atomic_json(self.reports / f"batch-{job_id}.json", report)
            return {"project_id": payload["project_id"], "generated": result["generated"], "quality": quality,
                    "performance": report, "message": f"本批次完成，新增 {len(result['generated'])} 个候选；已完成项按指纹复用"}
        except BaseException as exc:
            report = recorder.finish("cancelled" if type(exc).__name__ in {"Cancelled", "InterruptProcessingException"} else "error", error_type=type(exc).__name__)
            _atomic_json(self.reports / f"batch-{job_id}.json", report)
            raise

    def execute_variation(self, job_id, payload, progress):
        if {"conditioning_policy", "quality_source_evidence"}.intersection(payload.get("operation") or {}):
            raise ValueError("自动质检补录条件不接受手动操作输入，请使用批次独立质检补录")
        sampling = dict(payload.get("sampling") or {})
        if "seed" in sampling:
            sampling["seed"] = _uint(sampling["seed"])
        result = self.workspace.run_variation(payload["project_id"], payload["line_id"], payload["source_candidate_id"],
            lambda job, directory: self._generate(job, directory, progress), operation=payload["operation"],
            sampling=sampling, model_identity=self._options({})["model_identity"],
            expected_revision=payload.get("expected_revision"))
        summary = {key: result[key] for key in ("status", "generated", "failed", "reused") if key in result}
        summary["candidate_id"] = (result.get("candidate") or {}).get("candidate_id")
        self.studio.begin_commit(job_id, progress)
        return {"project_id": payload["project_id"], "variation": summary,
                "message": "替代演出已保存，原候选保留；请试听后选入成片"}

    def execute_quality(self, job_id, payload, progress):
        from .studio import read_audio
        project = self.workspace.get_project(payload["project_id"])
        ids = payload.get("candidate_ids")
        candidates = [value for key, value in project["candidates"].items() if value["status"] == "completed" and (ids is None or key in ids)]
        if not candidates:
            return {"checked": 0, "reviews": {}, "project_revision": project["revision"], "errors": []}
        def audios():
            for candidate in candidates:
                progress()
                wave, rate = read_audio(candidate["audio_path"])
                yield {"waveform": wave[None], "sample_rate": rate}
        engine = self.studio.engine
        # R2T2 has allocations outside Torch. Release the main expert RAM banks
        # as well as GPU patchers before loading the next phase's model.
        engine.unload()
        if engine.asr_backend == "r2t2":
            paths = engine.r2t2_paths
            transcripts = _transcribe_bounded(audios(), len(candidates), paths, progress)
        else:
            transcripts = [engine.transcribe_detailed(audio["waveform"][0], audio["sample_rate"], check_cancel=progress) for audio in audios()]
        if len(transcripts) != len(candidates):
            raise ValueError("转写未返回全部候选，拒绝提交不完整质检")
        reviews = {}
        for candidate, actual in zip(candidates, transcripts):
            from .prompts import spoken_text
            expected = spoken_text("\n".join(line["readout_text"] for line in candidate["input_snapshot"]["lines"]))
            reviews[candidate["candidate_id"]] = {**text_quality(expected, actual["text"], independent=True),
                "asr": actual, "asr_checked": True, "manual_review_required": True}
            if actual.get("truncated"):
                reviews[candidate["candidate_id"]].update(review_recommended=True, review_error="转写被截断，请重新转写或人工复核")
        saved = self.workspace.run_quality(payload["project_id"], lambda candidate: reviews[candidate["candidate_id"]], candidate_ids=list(reviews))
        return {"checked": len(reviews), "reviews": reviews, "project_revision": saved["project"]["revision"],
                "errors": saved.get("errors", []), "message": "转写仅标疑，不代替音色与表演试听"}

    def execute_recording(self, job_id, payload, progress):
        from .studio import read_audio
        from .creative_transcript import speech_regions, suggest_speakers, asr_rows
        path = self.input_file(payload["input_id"])
        if payload.get("transcribe", True):
            self.studio.engine.unload()
        wave, rate = read_audio(path)
        duration = wave.shape[-1] / rate
        vad = self.studio.model_paths["r2t2_vad"]
        if (vad / "fireredvad_stream_vad_with_cache.onnx").is_file() and (vad / "cmvn.ark").is_file():
            regions = speech_regions(wave.numpy(), rate, vad, check_cancel=progress)
        elif payload.get("speakers"):
            raise ValueError("自动分角色需要先配置 FireRedVAD 目录")
        else:
            regions = {"segments": [], "timing_method": "manual_selection", "manual_review_required": True,
                       "warnings": ["未配置VAD，可手动选区；自动分段需配置FireRedVAD"]}
        result = {**regions, "input_id": path.name, "source_audio": f"/api/creative/inputs/{path.name}", "name": payload["name"]}
        import numpy as np
        count = min(800, wave.shape[-1])
        result["waveform_peaks"] = [float(np.max(np.abs(wave[:,
            wave.shape[-1] * i // count:wave.shape[-1] * (i + 1) // count].numpy()))) for i in range(count)]
        result["duration_seconds"] = duration
        if payload.get("speakers"):
            result.update(suggest_speakers(wave.numpy(), rate, regions["segments"], self.resolve_path(self.config["speaker_model_path"]),
                num_speakers=payload.get("num_speakers", 0), check_cancel=progress))
        if payload.get("transcribe", True):
            if payload.get("speakers") and result["segments"]:
                clips = [{"waveform": wave[None, :, round(item["start"] * rate):round(item["end"] * rate)], "sample_rate": rate}
                         for item in result["segments"]]
                paths = self.studio.model_paths
                if self.studio.engine.asr_backend == "r2t2":
                    texts = _transcribe_bounded(clips, len(clips), paths, progress, language=payload.get("language"))
                else:
                    texts = [self.studio.engine.transcribe_detailed(item["waveform"][0], rate, language=payload.get("language"), check_cancel=progress) for item in clips]
                result["lines"] = [{"line_id": f"L{index + 1:02}", "role": region["role"], "original_text": text["text"],
                    "start_seconds": region["start"], "end_seconds": region["end"], "timing_method": "speaker_window_estimate"}
                    for index, (region, text) in enumerate(zip(result["segments"], texts)) if text["text"].strip()]
            else:
                text = self.studio.engine.transcribe_detailed(wave, rate, language=payload.get("language"), check_cancel=progress)
                result["asr"] = text
                result["lines"] = asr_rows(text, source_duration=duration)
        self.studio.begin_commit(job_id, progress)
        target = self.studio.output / f"{job_id}.recording.json"
        _atomic_json(target, result)
        return {**result, "metadata": f"/api/files/output/{target.name}", "message": "录音已分析，请校正台词、边界与角色后导入工程"}

    def execute_delivery(self, job_id, payload, progress):
        project = self.workspace.get_project(payload["project_id"])
        stored = self.workspace._read(project["project_id"])
        verified = set()
        selections = []
        for line in project["lines"]:
            candidate = line["selected_candidate"]
            if not candidate or candidate["status"] != "completed":
                raise ValueError(f"{line['line_id']} 尚未选定可用候选")
            if candidate["candidate_id"] not in verified:
                if not self.workspace._usable(project["project_id"], stored["candidates"][candidate["candidate_id"]]):
                    raise ValueError(f"{line['line_id']} 的候选文件校验不通过，请重新选择完整候选")
                verified.add(candidate["candidate_id"])
            if len(candidate["line_ids"]) > 1:
                part = next((item for item in candidate.get("line_audio", []) if item["line_id"] == line["line_id"]), None)
                if not part:
                    raise ValueError("该多人候选缺少已验证的逐句PCM，请重新生成后交付；不会按错误顺序拼接")
                audio_path = part["audio_path"]
            else:
                audio_path = candidate["audio_path"]
            candidate_line = next(item for item in candidate["input_snapshot"]["lines"] if item["line_id"] == line["line_id"])
            if candidate_line["readout_text"] != line["readout_text"]:
                raise ValueError(f"{line['line_id']} 的朗读稿已修改，请重新生成或恢复所选候选的朗读稿后交付")
            selections.append({"line_id": line["line_id"], "role": line["role"],
                "text": line["original_text"], "subtitle": line["subtitle_text"], "readout": candidate_line["readout_text"],
                "audio_path": audio_path, "candidate_id": candidate["candidate_id"], "seed": candidate["seed"],
                "sampling": candidate["input_snapshot"]["sampling"],
                "references": candidate.get("metadata", {}).get("reference_versions", candidate["input_snapshot"]["cast"]),
                "backend": candidate.get("metadata", {}).get("performance", {}).get("backends", {}),
                "fingerprint": candidate["fingerprint"], "target_seconds": line["target_seconds"],
                "gap_after": line["pause_after"] + float(payload.get("gap_seconds", .15)),
                **({"start_seconds": line["start_seconds"]} if line["start_seconds"] is not None else {})})
        target = self.workspace.projects / project["project_id"] / "deliveries" / uuid.uuid4().hex
        archive = self.downloads / f"delivery-{uuid.uuid4().hex}.zip"
        progress()
        result = export_selected(project, selections, target, sample_rate=int(payload.get("sample_rate", 48000)),
            duration_mode=payload.get("duration_mode", "report"), loudness_mode=payload.get("loudness_mode", "none"),
            target_db=float(payload.get("target_db", -20)), gap_seconds=float(payload.get("gap_seconds", .15)),
            check_cancel=progress, archive_path=archive)
        self.studio.begin_commit(job_id, progress)
        overflow = sum(bool(line["timing"].get("overflow")) for line in result["lines"])
        message = "交付包已保存，原始候选保留"
        if overflow:
            message += f"；{overflow}句超出目标时长，已保留完整台词，请调整目标或改稿"
        loudness_warnings = [{"line_id": line["line_id"], "target_lufs": level.get("target_i"),
            "actual_lufs": level.get("actual_i"), "message": level["warning"]}
            for line in result["lines"]
            if (level := line.get("loudness", {}).get("loudnorm", {})).get("warning")]
        if loudness_warnings:
            message += f"；{len(loudness_warnings)}句响度未达目标或不可测，实际值已记录在交付报告"
        return {"project_id": project["project_id"], "manifest": result,
                "loudness_warnings": loudness_warnings,
                "download": f"/api/creative/downloads/{archive.name}", "message": message}

    def performance_view(self):
        from .performance import ResourceProbe, compare_performance, SCHEMA
        reports = []
        for path in sorted(self.reports.glob("*.json"), key=lambda value: value.stat().st_mtime, reverse=True):
            try:
                if path.stat().st_size > 8 * 1024**2:
                    continue
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(report, dict) and report.get("schema") == SCHEMA:
                reports.append(report)
                if len(reports) == 20:
                    break
        comparison = None
        if len(reports) > 1:
            try:
                comparison = compare_performance(reports[1], reports[0])
            except ValueError as exc:
                comparison = {"same_binding": False, "reason": str(exc)}
        profile, profile_error = None, None
        if any((self.reports / "profiles").glob("*.json")):
            from .creative_benchmark import current_profile_binding
            from .performance import BenchmarkProfileStore
            try:
                profile = BenchmarkProfileStore(self.reports / "profiles").load(current_profile_binding(self))
            except (OSError, ValueError) as exc:
                profile_error = str(exc)
        return {"resources": ResourceProbe()(), "reports": reports, "comparison": comparison,
                "profile": profile, "profile_error": profile_error,
                "last_idle_unload": self.last_unload, "configuration": self.configuration()}

    def maybe_unload(self):
        with self.studio._activity_lock:
            seconds = self.config["idle_unload_seconds"]
            if (seconds and not self.studio.pending.qsize() and self.studio._active_job is None
                    and time.monotonic() - self.last_activity >= seconds
                    and any(getattr(self.studio.engine, name, None) is not None for name in ("drama", "codec", "asr"))):
                self.studio.engine.unload()
                self.last_unload = {"at": time.time(), "reason": "idle", "idle_seconds": seconds}

    def execute_benchmark(self, job_id, payload, progress):
        from .creative_benchmark import run_studio_benchmark
        return run_studio_benchmark(self, job_id, payload, progress)

    def execute_apply_profile(self, job_id, payload, progress):
        from .performance import BenchmarkProfileStore
        from .creative_benchmark import current_profile_binding
        profile = BenchmarkProfileStore(self.reports / "profiles").load(current_profile_binding(self, progress))
        if profile is None:
            raise ValueError("当前硬件、代码与模型没有匹配的合格档案，请先完成本机完整效果对照")
        settings = profile["recommendation"]["settings"]
        self.studio.begin_commit(job_id, progress)
        self.studio.apply_settings(settings)
        return {"message": "已应用当前环境的已验证配置", "settings": settings}
