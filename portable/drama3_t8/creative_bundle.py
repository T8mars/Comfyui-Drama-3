"""Complete local creative projects with isolated, version-exact voice assets.

An imported voice catalog lives inside the new project. Global catalogs are never
merged or renamed, so native UUIDs, Cast snapshots and checkpoint inputs survive.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import unicodedata
import uuid
import zipfile

from .creative_workspace import _file_record, _json_bytes, _now, _safe_member
from .voice_library import VoiceLibrary


FORMAT = "drama3-complete-creative-project"
SCHEMA = 1
MAX_OUTER_BYTES = 4 * 1024**3
MAX_INNER_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 64 * 1024**2


def embedded_library_path(workspace, project_id):
    return workspace._directory(project_id) / "voice-library"


def project_voice_library(workspace, project_id, fallback):
    path = embedded_library_path(workspace, project_id)
    return VoiceLibrary(path) if (path / "library.sqlite3").is_file() else fallback


def ensure_project_cast_assets(workspace, project_id, global_library, cast, *, extra_version_ids=()):
    """Append missing immutable versions to an imported project's private library.

    Intended for update_project's before_commit callback: the project lock and
    expected-revision check then cover the asset sync and final JSON commit.
    Existing private versions and all global data remain untouched. Optional
    missing originals are not mistaken for different encoded voice content.
    """
    root = embedded_library_path(workspace, project_id)
    if not (root / "library.sqlite3").is_file():
        return {"private_library": False, "added_version_ids": [], "already_present": []}
    if not isinstance(cast, dict) or not isinstance(extra_version_ids, (list, tuple, set)):
        raise ValueError("选角或额外参考版本无效")
    private = VoiceLibrary(root)
    owners = {}
    for binding in cast.values():
        if binding is None:
            continue
        if not isinstance(binding, dict):
            raise ValueError("角色音色绑定必须是对象")
        ids = [binding["version_id"]] if binding.get("version_id") else []
        groups = binding.get("reference_groups", {})
        if not isinstance(groups, dict):
            raise ValueError("参考组必须是对象")
        for group in groups.values():
            if not isinstance(group, list):
                raise ValueError("参考组必须是版本ID列表")
            ids.extend(group)
        for version_id in ids:
            owner = binding.get("voice_id")
            if (not isinstance(owner, str) or not re.fullmatch("[0-9a-f]{32}", owner)
                    or not isinstance(version_id, str) or not re.fullmatch("[0-9a-f]{32}", version_id)):
                raise ValueError("角色音色或版本UUID无效")
            if version_id in owners and owners[version_id] != owner:
                raise ValueError("同一版本不能属于不同音色")
            owners[version_id] = owner
    for version_id in extra_version_ids:
        if not isinstance(version_id, str) or not re.fullmatch("[0-9a-f]{32}", version_id):
            raise ValueError("额外参考版本UUID无效")
        owners.setdefault(version_id, None)

    def find(library, version_id):
        if library is None:
            return None
        # Do not treat malformed stored metadata as merely an unknown UUID.
        with library._db() as db:
            present = db.execute("SELECT 1 FROM versions WHERE id=?", (version_id,)).fetchone()
        return library.get_version(version_id) if present else None

    missing, present = [], []
    immutable = ("voice_id", "content_sha256", "transcript", "language", "delivery", "defaults")
    for version_id, owner in owners.items():
        old, incoming = find(private, version_id), find(global_library, version_id)
        record = old or incoming
        if record is None or (owner is not None and record["voice_id"] != owner):
            raise ValueError("选角引用了不存在或归属不符的音色版本")
        if old:
            private.session_file(version_id)
            if incoming:
                global_library.session_file(version_id)
                if any(old[key] != incoming[key] for key in immutable):
                    raise ValueError("私有与全局存在相同UUID但不同内容的音色版本，拒绝覆盖")
                for key in ("source_sha256", "preview_sha256"):
                    if old.get(key) and incoming.get(key) and old[key] != incoming[key]:
                        raise ValueError("相同版本UUID的原音/试听内容冲突，拒绝覆盖")
            present.append(version_id)
        else:
            missing.append(version_id)
    if missing:
        if global_library is None:
            raise ValueError("新选角版本不在项目库中，也没有全局来源")
        stage = root / ".asset-sync" / uuid.uuid4().hex
        stage.mkdir(parents=True, exist_ok=False)
        try:
            bundle = stage / "voices.zip"
            global_library.export_bundle(bundle, version_ids=sorted(missing), cast_ids=[], include_sources=True)
            private.import_bundle(bundle, conflict="rename")
            _receipt(stage, "synced", added_version_ids=sorted(missing))
        except BaseException as exc:
            _receipt(stage, "failed", error=type(exc).__name__)
            raise
    return {"private_library": True, "added_version_ids": sorted(missing), "already_present": sorted(present)}


def _inventory(archive, *, total_limit, member_limit):
    infos = archive.infolist()
    if not infos or len(infos) > member_limit:
        raise ValueError("工程包成员数量超限")
    total, result, folded = 0, {}, set()
    for info in infos:
        name = _safe_member(info.filename)
        key = unicodedata.normalize("NFC", name).casefold()
        if info.is_dir() or key in folded or info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("工程包包含重复、加密、链接或目录成员")
        total += info.file_size
        if info.file_size < 0 or total > total_limit:
            raise ValueError("工程包解压大小超限")
        folded.add(key)
        result[name] = info
    return result, total


def _verify(archive, records):
    for name, record in records.items():
        _safe_member(name)
        if not isinstance(record, dict) or type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError("工程包文件记录无效")
        if archive.getinfo(name).file_size != record["bytes"]:
            raise ValueError("工程包文件大小不符")
        digest = hashlib.sha256()
        with archive.open(name) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record.get("sha256"):
            raise ValueError("工程包文件校验失败")


def _voice_versions(project, library):
    needed = set()
    bindings = [project.get("cast", {})]
    bindings.extend(candidate.get("input_snapshot", {}).get("cast", {})
                    for candidate in project.get("candidates", {}).values() if candidate.get("status") == "completed")
    for cast in bindings:
        if not isinstance(cast, dict):
            raise ValueError("工程选角快照无效")
        for binding in cast.values():
            if binding is None:
                continue
            if not isinstance(binding, dict):
                raise ValueError("工程角色音色绑定无效")
            versions = [binding["version_id"]] if binding.get("version_id") else []
            groups = binding.get("reference_groups", {})
            if not isinstance(groups, dict):
                raise ValueError("工程参考组无效")
            for ids in groups.values():
                if not isinstance(ids, list):
                    raise ValueError("工程参考组必须是版本ID列表")
                versions.extend(ids)
            for version_id in versions:
                if library is None:
                    raise ValueError("工程引用音色，但没有可用音色库")
                version = library.get_version(version_id)
                if binding.get("voice_id") != version["voice_id"]:
                    raise ValueError("工程参考版本不属于绑定音色")
                needed.add(version_id)
    line_sets = [(project.get("lines", []), project.get("cast", {}))]
    line_sets.extend((candidate.get("input_snapshot", {}).get("lines", []), candidate.get("input_snapshot", {}).get("cast", {}))
                     for candidate in project.get("candidates", {}).values() if candidate.get("status") == "completed")
    for lines, cast in line_sets:
        for line in lines:
            owner = (cast.get(line.get("role")) or {}).get("voice_id")
            for version_id in line.get("reference_ids", []):
                if library is None:
                    raise ValueError("工程有参考版本，但未附带音色库")
                version = library.get_version(version_id)
                owner = owner or version["voice_id"]
                if version["voice_id"] != owner:
                    raise ValueError("台词参考版本不属于同一绑定音色")
                needed.add(version_id)
    return needed


def _stage(workspace, kind):
    path = workspace.root / ".bundle-staging" / (kind + "-" + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _receipt(stage, state, **fields):
    (stage / "receipt.json").write_bytes(_json_bytes({"state": state, "observed_at": _now(), **fields}))


def export_creative_bundle(workspace, voices, project_id, path, *, include_all=False, include_sources=True):
    """Export the frozen project and only its referenced voice versions/groups."""
    if type(include_sources) is not bool or type(include_all) is not bool:
        raise ValueError("工程导出选项必须是布尔值")
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("拒绝覆盖已有创作工程包")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = _stage(workspace, "export")
    try:
        project_zip = stage / "project.zip"
        project_export = workspace.export_project(project_id, project_zip, include_all=include_all)
        with zipfile.ZipFile(project_zip) as archive:
            packed_project = json.loads(archive.read("project.json"))
        library = project_voice_library(workspace, project_id, voices)
        versions = _voice_versions(packed_project, library)
        files = {"project.zip": project_zip}
        if versions:
            voice_zip = stage / "voices.zip"
            library.export_bundle(voice_zip, version_ids=sorted(versions), cast_ids=[], include_sources=include_sources)
            files["voices.zip"] = voice_zip
        manifest = {"format": FORMAT, "schema_version": SCHEMA, "created_at": _now(),
                    "project_id": project_id, "voice_version_ids": sorted(versions),
                    "include_all": include_all, "include_sources": include_sources,
                    "files": {name: _file_record(source) for name, source in files.items()}}
        size = sum(record["bytes"] for record in manifest["files"].values())
        if size > MAX_OUTER_BYTES or shutil.disk_usage(destination.parent).free < size + MIN_FREE_BYTES:
            raise ValueError("创作工程包过大或目标磁盘空间不足")
        # The fully written archive is linked into its requested name only after
        # success; an existing destination is never replaced. Staging/receipts
        # remain available for diagnosis without deleting any prior user file.
        temporary = destination.parent / f".creative-{uuid.uuid4().hex}.pending"
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest))
            for name, source in files.items():
                archive.write(source, name)
        with zipfile.ZipFile(temporary) as archive:
            _verify(archive, manifest["files"])
        os.link(temporary, destination)
        result = {"path": str(destination), **_file_record(destination), "voice_versions": len(versions),
                  "included_candidates": project_export["included_candidates"], "include_sources": include_sources,
                  "staging_path": str(stage)}
        _receipt(stage, "exported", **result)
        return result
    except BaseException as exc:
        _receipt(stage, "failed", error=type(exc).__name__)
        raise


def import_creative_bundle(workspace, path, *, name=None):
    """Import into a new project and its own catalog, publishing metadata last."""
    stage = _stage(workspace, "import")
    project_directory = None
    try:
        with zipfile.ZipFile(path) as archive:
            infos, _ = _inventory(archive, total_limit=MAX_OUTER_BYTES, member_limit=3)
            if "manifest.json" not in infos or infos["manifest.json"].file_size > 4 * 1024**2:
                raise ValueError("完整工程包缺少有效清单")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != FORMAT or manifest.get("schema_version") != SCHEMA:
                raise ValueError("不支持的完整创作工程包")
            records = manifest.get("files")
            if (not isinstance(records, dict) or "project.zip" not in records or set(records) - {"project.zip", "voices.zip"}
                    or set(infos) != set(records) | {"manifest.json"}):
                raise ValueError("完整工程包成员与清单不一致")
            _verify(archive, records)
            if shutil.disk_usage(workspace.root).free < sum(info.file_size for info in infos.values()) + MIN_FREE_BYTES:
                raise ValueError("工程目录磁盘空间不足")
            for member in records:
                with archive.open(member) as reader, (stage / member).open("xb") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
        expanded = 0
        with zipfile.ZipFile(stage / "project.zip") as project_archive:
            project_members, total = _inventory(project_archive, total_limit=MAX_INNER_BYTES, member_limit=10000)
            expanded += total
            if "project.json" not in project_members or project_members["project.json"].file_size > 64 * 1024**2:
                raise ValueError("工程文档缺失或超限")
            packed_project = json.loads(project_archive.read("project.json"))
        voice_versions = set()
        if "voices.zip" in records:
            with zipfile.ZipFile(stage / "voices.zip") as voice_archive:
                _, total = _inventory(voice_archive, total_limit=MAX_INNER_BYTES, member_limit=2048)
                expanded += total
                voice_manifest, _ = VoiceLibrary._bundle_manifest(voice_archive)
                voice_versions = {version["id"] for version in voice_manifest["versions"]}
        declared_versions = manifest.get("voice_version_ids")
        if (not isinstance(declared_versions, list) or len(set(declared_versions)) != len(declared_versions)
                or set(declared_versions) != voice_versions):
            raise ValueError("工程音色版本清单不符")
        if expanded + MIN_FREE_BYTES > shutil.disk_usage(workspace.root).free:
            raise ValueError("解压创作工程所需空间不足")

        def prepare(project, directory):
            nonlocal project_directory
            project_directory = directory
            library = None
            warnings = []
            if voice_versions:
                library = VoiceLibrary(directory / "voice-library")
                # Names may legitimately collide in the original catalog. The
                # isolated library preserves UUIDs/versions, disambiguating only
                # display names; immutable Cast/input snapshots stay untouched.
                library.import_bundle(stage / "voices.zip", conflict="rename")
                for voice in voice_manifest["voices"]:
                    actual_name = library.get_voice(voice["id"])["name"]
                    if actual_name != voice["name"]:
                        warnings.append(f"同名音色在项目库显示为{actual_name}；原UUID和选角快照未变。")
                project["embedded_voice_library"] = True
            needed = _voice_versions(project, library)
            if needed != voice_versions:
                raise ValueError("工程引用与附带音色版本不一致")
            project["import_warnings"] = warnings
            project["bundle_import"] = {"format": FORMAT, "voice_versions": len(voice_versions), "imported_at": _now()}

        # Workspace validates its media before this callback. Voice parsing and
        # import complete in the private directory before project.json exists.
        project = workspace.import_project(stage / "project.zip", name=name, before_commit=prepare)
        _receipt(stage, "imported", project_id=project["project_id"], voice_versions=len(voice_versions))
        return project
    except BaseException as exc:
        _receipt(stage, "failed", error=type(exc).__name__, project_directory=str(project_directory) if project_directory else None)
        if isinstance(exc, (KeyError, TypeError, AttributeError, zipfile.BadZipFile)):
            raise ValueError("损坏或不支持的完整创作工程包") from exc
        raise
