"""Native ComfyUI entrances to the same local creative project/voice core.

No Studio instance, HTTP request, server or GPU worker queue is started. Native
generation uses the engine supplied by the existing DRAMA loader in this graph.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

from .creative_workspace import CreativeWorkspace, preview_text
from .voice_library import VoiceLibrary, default_library_path


_UUID = re.compile(r"[0-9a-f]{32}\Z")


def _json(value, kind=dict):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, kind):
        raise ValueError("参数JSON类型不正确")
    json.dumps(result, allow_nan=False)
    return result


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _project_json(project):
    """Expose each rich atomic candidate once in the inspection JSON."""
    value = dict(project)
    value["lines"] = []
    for line in project["lines"]:
        item = {key: value for key, value in line.items() if key not in ("candidates", "selected_candidate")}
        item["candidates"] = list(line["candidate_ids"])
        selected = line.get("selected_candidate")
        item["selected_candidate"] = ({key: selected.get(key) for key in ("candidate_id", "status", "seed_text")}
                                      if selected else None)
        value["lines"].append(item)
    return _encode(value)


def _base():
    import folder_paths
    return Path(folder_paths.get_user_directory()).resolve() / "drama3_t8"


def _path(value, default):
    if not isinstance(value, str):
        raise ValueError("本地目录必须是文本")
    path = Path(os.path.expandvars(value.strip())).expanduser() if value.strip() else Path(default)
    if not path.is_absolute():
        path = _base() / path
    path = path.resolve()
    if path.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("请将创作资料放在插件源码目录之外")
    return path


def _library_path(value=""):
    return _path(value, default_library_path(_base()) if os.environ.get("DRAMA3_VOICE_LIBRARY", "").strip() else _base() / "voice-library")


def _input_path(value):
    import folder_paths
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请选择本地输入文件")
    path = Path(os.path.expandvars(value.strip())).expanduser()
    return (path if path.is_absolute() else Path(folder_paths.get_input_directory()) / path).resolve()


def _workspace_path(value=""):
    return _path(value, _base() / "creative")


def _file_hash(path):
    path = Path(path)
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_state(path, versions=()):
    path = Path(path)
    state = {name: _file_hash(path / name) for name in ("library.sqlite3", "library.sqlite3-wal")}
    if versions:
        library = VoiceLibrary(path)
        for version_id in sorted(set(versions)):
            version = library.get_version(version_id)
            relative = version["session_path"]
            artifact = (path / relative).resolve()
            if not artifact.is_relative_to(path.resolve()):
                raise ValueError("音色参考路径越出资料库")
            state[version_id] = _file_hash(artifact)
    return state


def _voice(library, value):
    if _UUID.fullmatch(value):
        return library.get_voice(value)
    matches = [voice for voice in library.list_voices(include_archived=False)
               if voice["name"].casefold() == value.strip().casefold()]
    if len(matches) != 1:
        raise ValueError("音色名称不存在或同名不唯一，请使用音色UUID")
    return library.get_voice(matches[0]["id"])


def _cast(library, value):
    if _UUID.fullmatch(value):
        return library.get_cast(value)
    matches = [cast for cast in library.list_casts() if cast["name"].casefold() == value.strip().casefold()]
    if len(matches) != 1:
        raise ValueError("选角名称不存在或同名不唯一，请使用选角UUID")
    return matches[0]


def _handle(project, workspace, library_path=""):
    return {"project_id": project["project_id"], "workspace_path": str(workspace.root),
            "library_path": str(_library_path(library_path))}


def _workspace(handle):
    if not isinstance(handle, dict) or not _UUID.fullmatch(handle.get("project_id", "")):
        raise ValueError("需要有效的原生创作工程输入")
    workspace = CreativeWorkspace(_workspace_path(handle.get("workspace_path", "")))
    return workspace, workspace.get_project(handle["project_id"])


def _project_state(handle, library_path=None):
    workspace, project = _workspace(handle)
    state = {"project": _file_hash(workspace._directory(project["project_id"]) / "project.json"), "artifacts": {}}
    directory = workspace._directory(project["project_id"]).resolve()
    for candidate in project["candidates"].values():
        for record in candidate.get("artifacts", []):
            artifact = (directory / record["path"]).resolve()
            if not artifact.is_relative_to(directory):
                raise ValueError("候选媒体路径越出工程")
            state["artifacts"][record["path"]] = _file_hash(artifact)
    from .creative_bundle import _voice_versions, project_voice_library
    fallback_path = _library_path(library_path or handle.get("library_path", ""))
    fallback = VoiceLibrary(fallback_path) if (fallback_path / "library.sqlite3").is_file() else None
    library = project_voice_library(workspace, project["project_id"], fallback)
    versions = _voice_versions(project, library)
    state["voices"] = _library_state(library.root, versions) if library else {}
    return state


class _Progress:
    def __init__(self):
        from comfy.utils import ProgressBar
        self.bar, self.total = ProgressBar(1), 1

    def __call__(self, value=None, total=None):
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        if value is not None and total is not None:
            self.total = max(1, int(total))
            self.bar.update_absolute(min(int(value), self.total - 1), self.total)

    def complete(self):
        self.bar.update_absolute(self.total, self.total)


class _NativeHost:
    """Only the synchronous fields used by the shared creative integration."""
    def __init__(self, engine):
        import folder_paths
        self.root = _base() / "native-host"
        self.data = self.root / "userdata"
        self.data.mkdir(parents=True, exist_ok=True)
        self.output = Path(folder_paths.get_output_directory()).resolve() / "drama3_t8_creative"
        self.output.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        models = Path(folder_paths.models_dir)
        self.model_paths = {role: Path(getattr(engine, attribute, None) or models / filename) for role, attribute, filename in (
            ("model", "model_path", "drama3-t8-fp8.safetensors"), ("codec", "codec_path", "drama3-codec.safetensors"),
            ("text_head", "text_head_path", "drama3-text-head.safetensors"), ("line_head", "line_head_path", "drama3-line-head.safetensors"),
            ("asr_model", "asr_path", "qwen3-asr-06b.safetensors"))}
        for role in ("model", "projector", "vad"):
            key = "r2t2_" + role
            self.model_paths[key] = Path((getattr(engine, "r2t2_paths", {}) or {}).get(key, models / "asr" / role))
        self.settings = {key: getattr(engine, key, default) for key, default in (
            ("gdn_backend", "torch"), ("moe_backend", "auto"), ("cuda_graph", False),
            ("expert_placement", "ram"), ("expert_cache_gb", 12), ("scene_head_mode", "compact"))}
        self._activity_lock = threading.RLock()

    def begin_commit(self, job_id, progress):
        progress()


def native_manager(engine, workspace_path="", library_path=""):
    """Reuse native generation/finishing functions, without constructing Studio."""
    from .creative_studio import CreativeStudio
    manager = CreativeStudio(_NativeHost(engine))
    manager.workspace = CreativeWorkspace(_workspace_path(workspace_path))
    manager.voices = VoiceLibrary(_library_path(library_path))
    manager.config.update(workspace_path=str(manager.workspace.root), voice_library_path=str(manager.voices.root))
    return manager


def get_creative_nodes(EngineType, ReferencesType, SessionType):
    from comfy_api.latest import io
    ProjectType = io.Custom("DRAMA3_T8_CREATIVE_PROJECT")
    category = "T8/Drama3/Creative"

    class T8DramaVoiceLibraryList(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaVoiceLibraryList", display_name="T8 Drama 3 · 音色资料库", category=category,
                inputs=[io.String.Input("library_path", default=""), io.String.Input("query", default=""),
                        io.Boolean.Input("favorites_only", default=False)], outputs=[io.String.Output()])

        @classmethod
        def fingerprint_inputs(cls, library_path="", **kwargs):
            return _library_state(_library_path(library_path))

        @classmethod
        def execute(cls, library_path="", query="", favorites_only=False):
            library = VoiceLibrary(_library_path(library_path))
            return io.NodeOutput(_encode({"voices": library.list_voices(query, favorites_only=favorites_only), "casts": library.list_casts()}))

    class T8DramaVoiceLibraryLoad(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaVoiceLibraryLoad", display_name="T8 Drama 3 · 载入音色版本", category=category,
                inputs=[io.String.Input("voice", default=""), io.String.Input("library_path", default=""),
                        io.String.Input("version_id", default="", optional=True), io.Int.Input("speaker", default=0, min=0, max=15),
                        io.String.Input("direction", default="", multiline=True)],
                outputs=[ReferencesType.Output(), SessionType.Output(), io.String.Output()])

        @classmethod
        def fingerprint_inputs(cls, voice, library_path="", version_id="", **kwargs):
            library = VoiceLibrary(_library_path(library_path))
            record = _voice(library, voice)
            return _library_state(library.root, [version_id or record["latest_version_id"]])

        @classmethod
        def execute(cls, voice, library_path="", version_id="", speaker=0, direction=""):
            library = VoiceLibrary(_library_path(library_path))
            record = _voice(library, voice)
            chosen = library.get_version(version_id or record["latest_version_id"])
            session = library.load_selection([{"voice_id": record["id"], "version_id": chosen["id"], "speaker": speaker}], direction=direction)
            return io.NodeOutput(session.references, session, _encode(chosen))

    class T8DramaVoiceLibrarySave(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaVoiceLibrarySave", display_name="T8 Drama 3 · 保存音色版本", category=category,
                inputs=[SessionType.Input("session"), io.Combo.Input("mode", options=["new_voice", "new_version"]),
                        io.String.Input("name", default="新音色"), io.String.Input("voice", default="", optional=True),
                        io.String.Input("library_path", default=""), io.String.Input("source_audio_path", default="", optional=True),
                        io.String.Input("language", default="zh"), io.String.Input("delivery", default="自然"),
                        io.String.Input("tags", default=""), io.String.Input("notes", default="")],
                outputs=[io.String.Output(display_name="voice_id"), io.String.Output(display_name="version_id")], is_output_node=True)

        @classmethod
        def execute(cls, session, mode="new_voice", name="新音色", voice="", library_path="", source_audio_path="",
                    language="zh", delivery="自然", tags="", notes=""):
            from .state import Session
            if mode not in ("new_voice", "new_version") or not session.references:
                raise ValueError("请选择新建/新增版本，并输入含参考声音的原生Session")
            library = VoiceLibrary(_library_path(library_path))
            record = library.create_voice(name, tags=tags, notes=notes) if mode == "new_voice" else _voice(library, voice)
            source = None
            if source_audio_path.strip():
                source = _input_path(source_audio_path)
            version = library.add_version(record["id"], Session(tuple(session.references)), source_path=source,
                                          language=language, delivery=delivery)
            return io.NodeOutput(record["id"], version["id"])

    class T8DramaVoiceCastLoad(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaVoiceCastLoad", display_name="T8 Drama 3 · 载入选角预设", category=category,
                inputs=[io.String.Input("cast", default=""), io.String.Input("library_path", default=""),
                        io.String.Input("reference_groups", default="{}", multiline=True)],
                outputs=[ReferencesType.Output(), SessionType.Output(), io.String.Output(display_name="cast_json")])

        @classmethod
        def fingerprint_inputs(cls, cast, library_path="", **kwargs):
            library = VoiceLibrary(_library_path(library_path))
            record = _cast(library, cast)
            ids = [binding["version_id"] for binding in record["bindings"].values()]
            ids += [version for binding in record["bindings"].values() for group in binding.get("reference_groups", {}).values() for version in group]
            return _library_state(library.root, ids)

        @classmethod
        def execute(cls, cast, library_path="", reference_groups="{}"):
            library = VoiceLibrary(_library_path(library_path))
            record = _cast(library, cast)
            session = library.load_cast(record["id"], reference_groups=_json(reference_groups))
            return io.NodeOutput(session.references, session, _encode(record["bindings"]))

    class T8DramaCreativeTextPreview(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeTextPreview", display_name="T8 Drama 3 · 朗读稿预览", category=category,
                inputs=[io.String.Input("text", default="", multiline=True), io.String.Input("dictionary_json", default="{}", multiline=True)],
                outputs=[io.String.Output(display_name="original"), io.String.Output(display_name="readout"),
                         io.String.Output(display_name="subtitle"), io.String.Output(display_name="rules_json")])

        @classmethod
        def execute(cls, text, dictionary_json="{}"):
            result = preview_text(text, _json(dictionary_json))
            return io.NodeOutput(result["original_text"], result["readout_text"], result["subtitle_text"], _encode(result))

    class T8DramaVoiceCastBuild(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaVoiceCastBuild", display_name="T8 Drama 3 · 角色选角快照", category=category,
                inputs=[io.String.Input("voice", default=""), io.String.Input("version_id", default=""),
                        io.String.Input("role", default="主角"), io.Int.Input("speaker", default=0, min=0, max=15),
                        io.String.Input("library_path", default=""), io.String.Input("previous_cast_json", default="{}", multiline=True),
                        io.String.Input("reference_groups_json", default="{}", multiline=True)],
                outputs=[io.String.Output(display_name="cast_json")])

        @classmethod
        def fingerprint_inputs(cls, voice, version_id="", library_path="", reference_groups_json="{}", **kwargs):
            library = VoiceLibrary(_library_path(library_path))
            record = _voice(library, voice)
            ids = [version_id or record["latest_version_id"]]
            for values in _json(reference_groups_json).values():
                if not isinstance(values, list):
                    raise ValueError("参考组必须是版本ID列表")
                ids.extend(values)
            return _library_state(library.root, ids)

        @classmethod
        def execute(cls, voice, version_id="", role="主角", speaker=0, library_path="", previous_cast_json="{}", reference_groups_json="{}"):
            library = VoiceLibrary(_library_path(library_path))
            record = _voice(library, voice)
            version = library.get_version(version_id or record["latest_version_id"])
            if not isinstance(role, str) or not role.strip() or len(role) > 120 or type(speaker) is not int or not 0 <= speaker <= 15:
                raise ValueError("角色名称或speaker编号无效")
            groups = _json(reference_groups_json)
            if len(groups) > 32:
                raise ValueError("角色参考组过多")
            ids = [version["id"]]
            for name, values in groups.items():
                if not isinstance(name, str) or not name.strip() or len(name) > 80 or not isinstance(values, list) or not values or len(values) > 64:
                    raise ValueError("参考组名称或版本ID列表无效")
                if len(set(values)) != len(values):
                    raise ValueError("同一参考组不能重复版本")
                ids.extend(values)
            for selected in ids:
                chosen = library.get_version(selected)
                if chosen["voice_id"] != record["id"]:
                    raise ValueError("参考版本必须属于当前绑定音色")
                library.session_file(selected)
            cast = _json(previous_cast_json)
            cast[role.strip()] = {"voice_id": record["id"], "version_id": version["id"], "speaker": speaker,
                "voice_name": record["name"], "language": version["language"], "delivery": version["delivery"],
                "defaults": version["defaults"], "voice_defaults": record.get("defaults", {}), "reference_groups": groups}
            return io.NodeOutput(_encode(cast))

    class T8DramaCreativeProjectCreate(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeProjectCreate", display_name="T8 Drama 3 · 新建创作工程", category=category,
                inputs=[io.String.Input("name", default="新工程"), io.String.Input("lines_json", default='["你好，世界。"]', multiline=True),
                        io.String.Input("cast_json", default="{}", multiline=True), io.String.Input("settings_json", default="{}", multiline=True),
                        io.String.Input("dictionary_json", default="{}", multiline=True), io.String.Input("workspace_path", default=""),
                        io.String.Input("library_path", default="")], outputs=[ProjectType.Output(), io.String.Output()])

        @classmethod
        def execute(cls, name, lines_json='["你好，世界。"]', cast_json="{}", settings_json="{}", dictionary_json="{}", workspace_path="", library_path=""):
            from .creative_context import assign_speakers
            workspace = CreativeWorkspace(_workspace_path(workspace_path))
            lines = _json(lines_json, list)
            roles = [{"role": line.get("role", "旁白") if isinstance(line, dict) else "旁白"} for line in lines]
            cast = assign_speakers(_json(cast_json), roles)
            project = workspace.create_project(name, lines, cast=cast, settings=_json(settings_json), dictionary=_json(dictionary_json))
            return io.NodeOutput(_handle(project, workspace, library_path), _project_json(project))

    class T8DramaCreativeProjectLoad(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeProjectLoad", display_name="T8 Drama 3 · 打开创作工程", category=category,
                inputs=[io.String.Input("project_id", default=""), io.String.Input("workspace_path", default=""), io.String.Input("library_path", default="")],
                outputs=[ProjectType.Output(), io.String.Output()])

        @classmethod
        def fingerprint_inputs(cls, project_id, workspace_path="", library_path=""):
            return _project_state({"project_id": project_id, "workspace_path": str(_workspace_path(workspace_path)),
                                   "library_path": str(_library_path(library_path))})

        @classmethod
        def execute(cls, project_id, workspace_path="", library_path=""):
            workspace = CreativeWorkspace(_workspace_path(workspace_path))
            project = workspace.get_project(project_id)
            return io.NodeOutput(_handle(project, workspace, library_path), _project_json(project))

    class T8DramaCreativeBatch(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeBatch", display_name="T8 Drama 3 · 原生工程批量配音", category=category,
                inputs=[EngineType.Input("engine"), ProjectType.Input("project"), io.Int.Input("takes_per_line", default=3, min=1, max=32),
                        io.String.Input("seed_text", default="42"), io.Combo.Input("generation_mode", options=["per_line", "per_line_context", "together", "per_speaker", "long_form", "text_stream"]),
                        io.String.Input("sampling_json", default="{}", multiline=True), io.Int.Input("lookahead", default=0, min=0, max=100),
                        io.String.Input("line_ids_json", default="[]", multiline=True), io.Boolean.Input("continue_on_error", default=False),
                        io.String.Input("library_path", default="", optional=True)],
                outputs=[ProjectType.Output(), SessionType.Output(display_name="last_selected_session"), io.String.Output()], is_output_node=True)

        @classmethod
        def fingerprint_inputs(cls, project, engine=None, **kwargs):
            state = _project_state(project, kwargs.get("library_path"))
            if engine is not None:
                state["models"] = {key: (str(value), Path(value).stat().st_size, Path(value).stat().st_mtime_ns)
                                   for key in ("model_path", "codec_path", "text_head_path", "line_head_path")
                                   if (value := getattr(engine, key, None)) is not None and Path(value).is_file()}
            return state

        @classmethod
        def execute(cls, engine, project, takes_per_line=3, seed_text="42", generation_mode="per_line", sampling_json="{}", lookahead=0,
                    line_ids_json="[]", continue_on_error=False, library_path=""):
            from .creative_studio import _uint
            from .state import load_session
            workspace, document = _workspace(project)
            manager = native_manager(engine, str(workspace.root), library_path or project.get("library_path", ""))
            payload = {"project_id": document["project_id"], "takes_per_line": takes_per_line, "base_seed": _uint(seed_text),
                       "generation_mode": generation_mode, "sampling": _json(sampling_json), "lookahead": lookahead,
                       "continue_on_error": continue_on_error, "expected_revision": document["revision"]}
            ids = _json(line_ids_json, list)
            if ids:
                payload["line_ids"] = ids
            plan = manager.plan(document["project_id"], payload)
            identity = manager._options(payload)["model_identity"]
            if plan["pending"] and engine.drama is not None and getattr(engine, "_creative_loaded_identity", None) != identity:
                engine.unload()
            progress = _Progress()
            result = manager.execute_batch(uuid.uuid4().hex, payload, progress)
            if plan["pending"]:
                engine._creative_loaded_identity = identity
            updated = workspace.get_project(document["project_id"])
            selected_lines = [line for line in updated["lines"] if not ids or line["line_id"] in ids]
            candidate = next((line["selected_candidate"] for line in reversed(selected_lines) if line["selected_candidate"]), None)
            if candidate is None:
                raise ValueError("本次选择的台词没有可用候选")
            session = load_session(candidate["session_path"])
            output = io.NodeOutput(_handle(updated, workspace, library_path or project.get("library_path", "")), session, _encode(result))
            progress.complete()
            return output

    class T8DramaCreativeDelivery(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeDelivery", display_name="T8 Drama 3 · 工程交付包", category=category,
                inputs=[ProjectType.Input("project"), io.Combo.Input("sample_rate", options=["48000", "44100"]),
                        io.Combo.Input("duration_mode", options=["report", "pad", "stretch"]), io.Combo.Input("loudness_mode", options=["none", "rms", "peak", "lufs"]),
                        io.Float.Input("target_db", default=-20, min=-60, max=-1), io.Float.Input("gap_seconds", default=.15, min=0, max=10)],
                outputs=[io.String.Output(display_name="archive_path"), io.String.Output(display_name="manifest_json")], is_output_node=True)

        @classmethod
        def fingerprint_inputs(cls, project, **kwargs):
            return _project_state(project)

        @classmethod
        def execute(cls, project, sample_rate="48000", duration_mode="report", loudness_mode="none", target_db=-20, gap_seconds=.15):
            workspace, document = _workspace(project)
            manager = native_manager(None, str(workspace.root), project.get("library_path", ""))
            progress = _Progress()
            result = manager.execute_delivery(uuid.uuid4().hex, {"project_id": document["project_id"], "sample_rate": int(sample_rate),
                "duration_mode": duration_mode, "loudness_mode": loudness_mode, "target_db": target_db, "gap_seconds": gap_seconds}, progress)
            path = manager.downloads / result["download"].rsplit("/", 1)[-1]
            output = io.NodeOutput(str(path), _encode(result["manifest"]))
            progress.complete()
            return output

    class T8DramaCreativeProjectExport(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeProjectExport", display_name="T8 Drama 3 · 导出完整工程", category=category,
                inputs=[ProjectType.Input("project"), io.String.Input("output_path", default=""),
                        io.Boolean.Input("include_all", default=False), io.Boolean.Input("include_sources", default=True)], outputs=[io.String.Output()], is_output_node=True)

        @classmethod
        def fingerprint_inputs(cls, project, **kwargs):
            return _project_state(project)

        @classmethod
        def execute(cls, project, output_path="", include_all=False, include_sources=True):
            import folder_paths
            from .creative_bundle import export_creative_bundle
            workspace, document = _workspace(project)
            library = VoiceLibrary(_library_path(project.get("library_path", "")))
            target = _path(output_path, Path(folder_paths.get_output_directory()) / "drama3_t8_creative" / f"project-{uuid.uuid4().hex}.zip")
            result = export_creative_bundle(workspace, library, document["project_id"], target, include_all=include_all, include_sources=include_sources)
            return io.NodeOutput(_encode(result))

    class T8DramaCreativeProjectImport(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="T8DramaCreativeProjectImport", display_name="T8 Drama 3 · 导入完整工程", category=category,
                inputs=[io.String.Input("bundle_path", default=""), io.String.Input("workspace_path", default=""), io.String.Input("library_path", default="")],
                outputs=[ProjectType.Output(), io.String.Output()], is_output_node=True)

        @classmethod
        def fingerprint_inputs(cls, bundle_path, **kwargs):
            return _file_hash(_input_path(bundle_path))

        @classmethod
        def execute(cls, bundle_path, workspace_path="", library_path=""):
            from .creative_bundle import import_creative_bundle
            workspace = CreativeWorkspace(_workspace_path(workspace_path))
            project = import_creative_bundle(workspace, _input_path(bundle_path))
            return io.NodeOutput(_handle(project, workspace, library_path), _project_json(project))

    return [T8DramaVoiceLibraryList, T8DramaVoiceLibraryLoad, T8DramaVoiceLibrarySave, T8DramaVoiceCastLoad, T8DramaVoiceCastBuild,
            T8DramaCreativeTextPreview, T8DramaCreativeProjectCreate, T8DramaCreativeProjectLoad, T8DramaCreativeBatch,
            T8DramaCreativeDelivery, T8DramaCreativeProjectExport, T8DramaCreativeProjectImport]
