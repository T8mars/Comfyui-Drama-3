"""Complete project/voice portability, isolation and pre-commit failure tests."""
import hashlib
import json
from pathlib import Path
import shutil
import wave
import zipfile

import pytest
import torch

from drama3_t8.creative_bundle import export_creative_bundle, import_creative_bundle, project_voice_library, ensure_project_cast_assets
from drama3_t8.creative_workspace import CreativeWorkspace, ProjectConflict
from drama3_t8.state import Reference, Session, Take, save_session
from drama3_t8.voice_library import VoiceLibrary


IDENTITY = {"model": {"sha256": "a" * 64, "bytes": 123}, "codec": {"sha256": "b" * 64, "bytes": 45}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(code):
    return Session((Reference("<|speaker:0|> 你好，这是角色参考。", torch.full((10, 4), code, dtype=torch.int16)),))


def generate(job, directory):
    audio = directory / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x64\0" * 80)
    frames = torch.zeros((2, 11), dtype=torch.int64)
    frames[:, 0] = 248099
    take = Take(job["readout_text"], job["readout_text"], frames,
                metadata={"seed": job["seed"], "requested_seed": job["seed"], "finish_reason": "eos"})
    session = directory / "session.safetensors"
    save_session(Session().append(take), session)
    return {"complete": True, "seed": job["seed"], "audio_path": str(audio), "session_path": str(session)}


@pytest.fixture
def setup(tmp_path):
    voices = VoiceLibrary(tmp_path / "voices")
    voice = voices.create_voice("主角")
    original = tmp_path / "original.wav"
    with wave.open(str(original), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x12\0" * 2400)
    versions = [voices.add_version(voice["id"], reference(code), source_path=original, language="zh", delivery=delivery)
                for code, delivery in ((3, "平静"), (4, "喊话"), (5, "无关旧参考"))]
    workspace = CreativeWorkspace(tmp_path / "source-workspace")
    binding = {"voice_id": voice["id"], "version_id": versions[0]["id"], "speaker": 0,
               "reference_groups": {"喊话": [versions[1]["id"]]}}
    project = workspace.create_project("完整工程", [{"role": "主角", "text": "今天终于下班了"}], cast={"主角": binding})
    done = workspace.run_generation(project["project_id"], generate, takes_per_line=2, model_identity=IDENTITY)
    return voices, voice, versions, workspace, done["project"], original


def rewrite_outer(source, target, mutate):
    with zipfile.ZipFile(source) as archive:
        members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    manifest = json.loads(members.pop("manifest.json"))
    mutate(manifest, members)
    manifest["files"] = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()}
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in members.items():
            archive.writestr(name, data)


def test_complete_bundle_includes_only_referenced_versions_and_real_originals(setup, tmp_path):
    voices, voice, versions, workspace, project, original = setup
    path = tmp_path / "complete.drama.zip"
    result = export_creative_bundle(workspace, voices, project["project_id"], path)
    assert result["voice_versions"] == 2
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "project.zip", "voices.zip"}
        nested = tmp_path / "voices.zip"
        nested.write_bytes(archive.read("voices.zip"))
    with zipfile.ZipFile(nested) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert {version["id"] for version in manifest["versions"]} == {version["id"] for version in versions[:2]}
        assert versions[2]["id"] not in archive.read("manifest.json").decode()
        for version in manifest["versions"]:
            assert archive.read(version["source_path"]) == original.read_bytes()
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    embedded = project_voice_library(destination, imported["project_id"], None)
    assert embedded.get_voice(voice["id"])["name"] == "主角"
    assert len(embedded.get_voice(voice["id"])["versions"]) == 2
    assert imported["cast"] == project["cast"]
    assert imported["settings"] == project["settings"]
    assert destination.plan_generation(imported["project_id"], takes_per_line=1, model_identity=IDENTITY)["pending"] == 0
    selected = imported["lines"][0]["selected_candidate"]
    assert sha(selected["audio_path"]) == project["lines"][0]["selected_candidate"]["artifacts"][0]["sha256"]


def test_import_does_not_merge_same_uuid_conflicting_global_voice(setup, tmp_path):
    voices, voice, versions, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    global_library = VoiceLibrary(tmp_path / "global")
    other = global_library.create_voice("全局同UUID但不同声音")
    different = global_library.add_version(other["id"], reference(9))
    with global_library._db() as db:
        db.execute("INSERT INTO voices SELECT ?,name,tags,favorite,notes,archived,created,updated,NULL FROM voices WHERE id=?",
                   (voice["id"], other["id"]))
        db.execute("UPDATE versions SET id=?,voice_id=? WHERE id=?", (versions[0]["id"], voice["id"], different["id"]))
    old_hash = sha(global_library.database)
    old_content = global_library.get_version(versions[0]["id"])["content_sha256"]
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    embedded = project_voice_library(destination, imported["project_id"], global_library)
    assert embedded.root != global_library.root
    assert embedded.get_version(versions[0]["id"])["content_sha256"] != old_content
    assert global_library.get_voice(voice["id"])["name"] == "全局同UUID但不同声音"
    assert sha(global_library.database) == old_hash


def test_move_workspace_keeps_voice_uuid_candidate_bytes_and_resume(setup, tmp_path):
    voices, _, versions, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    moved_root = tmp_path / "another-drive-simulation"
    shutil.copytree(destination.root, moved_root)
    moved = CreativeWorkspace(moved_root)
    reloaded = moved.get_project(imported["project_id"])
    assert reloaded["cast"] == imported["cast"]
    candidate = reloaded["lines"][0]["selected_candidate"]
    assert Path(candidate["audio_path"]).is_relative_to(moved_root)
    assert sha(candidate["audio_path"]) == sha(imported["lines"][0]["selected_candidate"]["audio_path"])
    embedded = project_voice_library(moved, imported["project_id"], voices)
    assert embedded.root.is_relative_to(moved_root)
    assert embedded.session_file(versions[0]["id"]).is_file()
    assert moved.plan_generation(imported["project_id"], takes_per_line=1, model_identity=IDENTITY)["pending"] == 0
    # Re-export must use the embedded catalog even if an unrelated fallback is supplied.
    exported = export_creative_bundle(moved, VoiceLibrary(tmp_path / "empty-global"), imported["project_id"], tmp_path / "again.zip")
    assert exported["voice_versions"] == 2


def test_sources_can_be_excluded_without_copying_unrelated_recordings(setup, tmp_path):
    voices, _, _, workspace, project, _ = setup
    path = tmp_path / "no-originals.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path, include_sources=False)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    embedded = project_voice_library(destination, imported["project_id"], None)
    assert all(version["source_path"] is None for voice in embedded.list_voices() for version in voice["versions"])


def test_voice_failure_before_commit_never_exposes_half_project(setup, tmp_path, monkeypatch):
    voices, _, _, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    def failure(*args, **kwargs):
        raise RuntimeError("模拟音色归档读取失败")
    monkeypatch.setattr(VoiceLibrary, "import_bundle", failure)
    with pytest.raises(RuntimeError):
        import_creative_bundle(destination, path)
    assert destination.list_projects() == []
    partial = list(destination.projects.iterdir())
    assert partial and all(not (directory / "project.json").exists() for directory in partial)
    assert list((destination.root / ".bundle-staging").glob("*/receipt.json"))


def test_missing_voice_zip_with_cast_reference_is_rejected_before_commit(setup, tmp_path):
    voices, _, _, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    broken = tmp_path / "missing.zip"
    def remove(manifest, members):
        members.pop("voices.zip")
        manifest["voice_version_ids"] = []
    rewrite_outer(path, broken, remove)
    destination = CreativeWorkspace(tmp_path / "destination")
    with pytest.raises(ValueError):
        import_creative_bundle(destination, broken)
    assert destination.list_projects() == []


def test_outer_checksum_tampering_does_not_publish_project(setup, tmp_path):
    voices, _, _, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(broken, "w") as target:
        for entry in source.infolist():
            target.writestr(entry.filename, source.read(entry.filename) + (b"tampered" if entry.filename == "project.zip" else b""))
    destination = CreativeWorkspace(tmp_path / "destination")
    with pytest.raises(ValueError):
        import_creative_bundle(destination, broken)
    assert destination.list_projects() == []


@pytest.mark.parametrize("member", ["../bad", "C:/bad", "a\\b"])
def test_outer_path_escape_is_rejected(tmp_path, member):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr(member, "bad")
    destination = CreativeWorkspace(tmp_path / "destination")
    with pytest.raises(ValueError):
        import_creative_bundle(destination, path)
    assert destination.list_projects() == []


def test_self_generated_project_needs_no_voice_catalog(tmp_path):
    workspace = CreativeWorkspace(tmp_path / "source")
    project = workspace.create_project("无参考", ["你好"])
    workspace.run_generation(project["project_id"], generate, takes_per_line=1, model_identity=IDENTITY)
    path = tmp_path / "self.zip"
    export_creative_bundle(workspace, None, project["project_id"], path)
    with zipfile.ZipFile(path) as archive:
        assert "voices.zip" not in archive.namelist()
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    assert not imported.get("embedded_voice_library")
    assert destination.plan_generation(imported["project_id"], takes_per_line=1, model_identity=IDENTITY)["pending"] == 0


def test_explicit_voice_version_export_does_not_leak_other_versions(setup, tmp_path):
    voices, voice, versions, _, _, _ = setup
    unrelated = voices.create_voice("无关声音")
    voices.add_version(unrelated["id"], reference(8))
    path = tmp_path / "one-version.zip"
    voices.export_bundle(path, voice_ids=[voice["id"], unrelated["id"]], version_ids=[versions[0]["id"]], include_sources=False)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert [item["id"] for item in manifest["voices"]] == [voice["id"]]
        assert [item["id"] for item in manifest["versions"]] == [versions[0]["id"]]
        assert all(item["source_path"] is None for item in manifest["versions"])


def test_explicit_version_filter_cannot_leave_cast_group_dangling(setup, tmp_path):
    voices, voice, versions, _, _, _ = setup
    cast = voices.create_cast("角色组合", {"主角": {"voice_id": voice["id"], "version_id": versions[0]["id"],
                                                     "reference_groups": {"喊话": [versions[1]["id"]]}}})
    path = tmp_path / "bad-cast.zip"
    with pytest.raises(ValueError, match="cast reference"):
        voices.export_bundle(path, version_ids=[versions[0]["id"]], cast_ids=[cast["id"]])
    assert not path.exists()
    voices.export_bundle(path, version_ids=[versions[0]["id"], versions[1]["id"]], cast_ids=[cast["id"]])
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["versions"]) == 2
        assert manifest["casts"][0]["bindings"]["主角"]["reference_groups"]["喊话"] == [versions[1]["id"]]


def test_same_name_different_uuid_voices_keep_immutable_cast_and_ids(setup, tmp_path):
    voices, voice, versions, workspace, project, _ = setup
    other = voices.create_voice(voice["name"])
    other_version = voices.add_version(other["id"], reference(7))
    cast = {**project["cast"], "另一角色": {"voice_id": other["id"], "version_id": other_version["id"], "speaker": 1}}
    updated = workspace.update_project(project["project_id"], {"cast": cast})
    path = tmp_path / "same-names.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    embedded = project_voice_library(destination, imported["project_id"], None)
    assert imported["cast"] == updated["cast"]
    assert {item["id"] for item in embedded.list_voices()} == {voice["id"], other["id"]}
    assert embedded.get_version(other_version["id"])["voice_id"] == other["id"]
    assert imported["import_warnings"]


def test_imported_project_can_adopt_new_global_version_without_changing_old_assets(setup, tmp_path):
    voices, voice, versions, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    private = project_voice_library(destination, imported["project_id"], voices)
    old_file = private.session_file(versions[0]["id"])
    old_sha, global_sha = sha(old_file), sha(voices.database)
    new_cast = {"主角": {**imported["cast"]["主角"], "version_id": versions[2]["id"]}}
    sync = []
    updated = destination.update_project(imported["project_id"], {"cast": new_cast}, imported["revision"],
        before_commit=lambda document, directory: sync.append(ensure_project_cast_assets(destination, document["project_id"], voices, document["cast"])))
    assert sync[0]["added_version_ids"] == [versions[2]["id"]]
    assert sha(old_file) == old_sha and sha(voices.database) == global_sha
    selected = private.load_selection([{"voice_id": voice["id"], "version_id": versions[2]["id"], "speaker": 0}])
    assert torch.equal(selected.references[0].codes, reference(5).references[0].codes)
    assert updated["cast"] == new_cast
    done = destination.run_generation(imported["project_id"], generate, takes_per_line=1, model_identity=IDENTITY)
    assert done["generated"]
    again = tmp_path / "with-history.zip"
    exported = export_creative_bundle(destination, voices, imported["project_id"], again, include_all=True)
    assert exported["voice_versions"] == 3
    assert private.get_version(versions[0]["id"])["content_sha256"] == voices.get_version(versions[0]["id"])["content_sha256"]


def test_cast_sync_same_uuid_different_content_rejects_without_changing_project(setup, tmp_path):
    voices, _, versions, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    private = project_voice_library(destination, imported["project_id"], voices)
    original = private.session_file(versions[0]["id"])
    save_session(reference(9), original)
    with private._db() as db:
        db.execute("UPDATE versions SET content_sha256=? WHERE id=?", (sha(original), versions[0]["id"]))
    project_path = destination._directory(imported["project_id"]) / "project.json"
    old_project, old_asset = project_path.read_bytes(), original.read_bytes()
    desired = {"主角": {**imported["cast"]["主角"], "reference_groups": {"new": [versions[2]["id"]]}}}
    with pytest.raises(ValueError, match="UUID"):
        destination.update_project(imported["project_id"], {"cast": desired}, imported["revision"],
            before_commit=lambda document, directory: ensure_project_cast_assets(destination, document["project_id"], voices, document["cast"]))
    assert project_path.read_bytes() == old_project and original.read_bytes() == old_asset
    with pytest.raises(ValueError):
        private.get_version(versions[2]["id"])


def test_stale_revision_never_runs_asset_sync(setup, tmp_path):
    voices, _, versions, workspace, project, _ = setup
    path = tmp_path / "complete.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    destination.update_project(imported["project_id"], {"name": "新名称"}, imported["revision"])
    with pytest.raises(ProjectConflict):
        destination.update_project(imported["project_id"], {"cast": imported["cast"]}, imported["revision"],
            before_commit=lambda *_: pytest.fail("stale update mutated catalog"))


def test_sync_missing_sources_are_optional_and_extra_direct_refs_can_be_added(setup, tmp_path):
    voices, _, versions, workspace, project, _ = setup
    path = tmp_path / "without-originals.zip"
    export_creative_bundle(workspace, voices, project["project_id"], path, include_sources=False)
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    private = project_voice_library(destination, imported["project_id"], voices)
    assert private.get_version(versions[0]["id"])["source_path"] is None
    result = ensure_project_cast_assets(destination, imported["project_id"], voices, imported["cast"], extra_version_ids=[versions[2]["id"]])
    assert result["added_version_ids"] == [versions[2]["id"]]
    assert private.get_version(versions[0]["id"])["source_path"] is None
    assert private.get_version(versions[2]["id"])["source_path"] is not None


def test_complete_bundle_closure_includes_explicit_line_reference_versions(setup, tmp_path):
    voices, _, versions, workspace, project, _ = setup
    workspace.update_project(project["project_id"], {"lines": [{"line_id": "L01", "role": "主角", "text": "仅本句新参考", "reference_ids": [versions[2]["id"]]}]})
    path = tmp_path / "explicit-reference.zip"
    assert export_creative_bundle(workspace, voices, project["project_id"], path)["voice_versions"] == 3
    destination = CreativeWorkspace(tmp_path / "destination")
    imported = import_creative_bundle(destination, path)
    private = project_voice_library(destination, imported["project_id"], None)
    assert imported["lines"][0]["reference_ids"] == [versions[2]["id"]]
    assert private.session_file(versions[2]["id"]).is_file()
    other = voices.create_voice("外来音色")
    foreign = voices.add_version(other["id"], reference(9))
    workspace.update_project(project["project_id"], {"lines": [{"line_id": "L01", "role": "主角", "text": "错误混色", "reference_ids": [foreign["id"]]}]})
    with pytest.raises(ValueError, match="绑定音色"):
        export_creative_bundle(workspace, voices, project["project_id"], tmp_path / "foreign-reference.zip")
