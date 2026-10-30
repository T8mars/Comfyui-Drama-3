"""CPU tests for shared voice assets, immutable casts and hostile archives."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import wave
import zipfile

import pytest
import torch

from drama3_t8.state import Reference, Session, save_session
from drama3_t8.voice_library import VoiceLibrary, apply_explicit_defaults, default_library_path


@pytest.fixture
def session():
    return Session((Reference("<|speaker:0|> 你好，这是配音参考。", torch.full((10, 4), 3, dtype=torch.int16)),))


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "original.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(b"\x01\x00" * 2400)
    return path


def add(library, session, audio=None, name="主角"):
    voice = library.create_voice(name, tags=["普通话", "喜剧"], favorite=True, notes="片头主角")
    version = library.add_version(voice["id"], session, source_path=audio, language="zh", delivery="平静",
                                  defaults={"temperature": 0.7})
    return library.get_voice(voice["id"]), version


def rewrite_bundle(source, destination, mutate):
    with zipfile.ZipFile(source) as archive:
        members = {entry.filename: archive.read(entry.filename) for entry in archive.infolist()}
    manifest = json.loads(members["manifest.json"])
    mutate(manifest, members)
    members["manifest.json"] = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return destination


def test_catalog_import_and_preview_do_not_import_torch(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "lazy")
    _, version = add(library, session, audio)
    script = "import sys; from drama3_t8.voice_library import VoiceLibrary; lib=VoiceLibrary(sys.argv[1]); assert lib.list_voices(); assert lib.source_file(sys.argv[2]).is_file(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script, str(library.root), version["id"]], check=True, timeout=30)


def test_custom_library_path_and_explicit_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA3_VOICE_LIBRARY", str(tmp_path / "custom"))
    assert default_library_path(tmp_path) == (tmp_path / "custom").resolve()
    assert VoiceLibrary().root == (tmp_path / "custom").resolve()
    current = {"temperature": 0.8, "seed": 6}
    assert apply_explicit_defaults(current, {"temperature": 0.2}) == current
    assert apply_explicit_defaults(current, {"temperature": 0.2}, apply=True) == {"temperature": 0.2, "seed": 6}
    with pytest.raises(ValueError, match="finite"):
        apply_explicit_defaults({}, {"temperature": float("nan")}, apply=True)


def test_metadata_search_tag_and_archive(tmp_path, session):
    library = VoiceLibrary(tmp_path / "library")
    voice, version = add(library, session)
    assert library.list_voices("片头")[0]["id"] == voice["id"]
    assert library.list_voices("平静", tags=["喜剧"], favorites_only=True)[0]["latest_version_id"] == version["id"]
    assert library.list_voices(tags=["喜剧", "不存在"]) == []
    renamed = library.update_voice(voice["id"], name="新名字", tags=["中文", "中文"], favorite=False, notes="新备注")
    assert renamed["tags"] == ["中文"]
    assert renamed["versions"][0]["id"] == version["id"]
    library.archive_voice(voice["id"])
    assert library.list_voices() == []
    assert library.list_voices(include_archived=True)[0]["archived"]
    assert library.load_version(version["id"]).references[0].transcript == session.references[0].transcript


def test_source_copy_and_version_are_immutable(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "library")
    original = audio.read_bytes()
    voice, first = add(library, session, audio)
    source = library.source_file(first["id"])
    assert source.read_bytes() == original and source != audio
    audio.write_bytes(b"user-original-changed")
    assert source.read_bytes() == original
    second = library.add_version(voice["id"], session, language="en", delivery="angry", defaults={"temperature": 1.0})
    assert first["id"] != second["id"]
    assert library.get_voice(voice["id"])["latest_version_id"] == second["id"]
    assert library.get_version(first["id"])["language"] == "zh"
    assert torch.equal(library.load_version(first["id"]).references[0].codes, session.references[0].codes)
    with pytest.raises(ValueError, match="no associated"):
        library.source_file(second["id"])
    with pytest.raises(ValueError, match="re-encode"):
        library.add_version(voice["id"], session, transcript="改过但没有重新编码")


def test_corrupt_artifact_is_rejected(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "library")
    _, version = add(library, session, audio)
    library.source_file(version["id"]).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        library.source_file(version["id"])
    library.session_file(version["id"]).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        library.load_version(version["id"])


def test_migration_is_idempotent_and_preserves_exact_source(tmp_path, session, audio):
    base = tmp_path / "legacy"
    base.mkdir()
    save_session(session, base / "old.safetensors")
    original = (base / "old.safetensors").read_bytes()
    rows = [{"id": "legacy-reference", "name": "旧音色", "transcript": "旧展示字段", "path": "old.safetensors", "source_path": str(audio)}]
    library = VoiceLibrary(tmp_path / "library")
    first = library.migrate_legacy(rows, base)
    second = library.migrate_legacy(rows, base)
    assert not first["errors"] and not second["errors"]
    assert first["mapping"] == second["mapping"]
    assert len(library.list_voices()) == 1 and len(library.list_voices()[0]["versions"]) == 1
    version_id = first["mapping"]["legacy-reference"]["version_id"]
    assert library.source_file(version_id).read_bytes() == audio.read_bytes()
    assert (base / "old.safetensors").read_bytes() == original
    assert audio.is_file()
    assert "你好" in library.get_version(version_id)["transcript"]


def test_migration_missing_source_reports_error_without_deleting_legacy(tmp_path, session):
    save_session(session, tmp_path / "old.safetensors")
    library = VoiceLibrary(tmp_path / "library")
    result = library.migrate_legacy([{"id": "old", "name": "旧", "path": "old.safetensors",
                                      "source_path": "missing.wav"}], tmp_path)
    assert result["errors"] and (tmp_path / "old.safetensors").is_file()
    repaired = library.migrate_legacy([{"id": "old", "name": "旧", "path": "old.safetensors"}], tmp_path)
    assert not repaired["errors"]
    assert library.get_version(repaired["mapping"]["old"]["version_id"])["source_path"] is None


def test_cast_snapshot_survives_library_changes(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "library")
    voice, first = add(library, session, audio)
    cast = library.create_cast("双人短剧", {"小白": {"voice_id": voice["id"], "version_id": first["id"], "speaker": 3,
                                                    "defaults": {"top_p": 0.8}}}, defaults={"direction": "喜剧"})
    library.update_voice(voice["id"], name="已经改名", notes="新备注", archived=True)
    library.add_version(voice["id"], session, delivery="哭腔", defaults={"temperature": 1.1})
    assert library.get_cast(cast["id"])["bindings"]["小白"]["version_id"] == first["id"]
    assert library.get_cast(cast["id"])["bindings"]["小白"]["voice_name"] == "主角"
    generated = library.load_cast(cast["id"])
    assert generated.references[0].transcript.startswith("<|speaker:3|>")
    assert generated.references[0].name == "小白"
    assert generated.references[0].processing["voice_library"]["version_id"] == first["id"]
    assert generated.direction == ""  # saved defaults are explicitly applied by the host


def test_cast_rejects_mismatched_version_and_multi_speaker(tmp_path, session):
    library = VoiceLibrary(tmp_path / "library")
    a, first = add(library, session)
    b, _ = add(library, session, name="第二个")
    with pytest.raises(ValueError, match="does not belong"):
        library.create_cast("bad", {"a": {"voice_id": b["id"], "version_id": first["id"]}})
    multi = Session((Reference("<|speaker:0|>甲 <|speaker:1|>乙", session.references[0].codes),))
    second = library.add_version(a["id"], multi)
    with pytest.raises(ValueError, match="multi-speaker"):
        library.create_cast("bad", {"a": {"voice_id": a["id"], "version_id": second["id"], "speaker": 2}})
    with pytest.raises(ValueError, match="speaker"):
        library.create_cast("bad", {"a": {"voice_id": a["id"], "version_id": first["id"], "speaker": True}})


def test_bundle_roundtrip_relocation_cast_and_no_overwrite(tmp_path, session, audio):
    source = VoiceLibrary(tmp_path / "source")
    voice, version = add(source, session, audio)
    cast = source.create_cast("剧组", {"主角": {"voice_id": voice["id"], "version_id": version["id"], "speaker": 4}})
    bundle = tmp_path / "voices.t8drama-voices.zip"
    report = source.export_bundle(bundle)
    assert report["versions"] == 1 and report["casts"] == 1
    original = bundle.read_bytes()
    with pytest.raises(FileExistsError):
        source.export_bundle(bundle)
    assert bundle.read_bytes() == original
    destination = VoiceLibrary(tmp_path / "relocated")
    imported = destination.import_bundle(bundle)
    assert imported["voices"] == [voice["id"]]
    assert destination.source_file(version["id"]).read_bytes() == audio.read_bytes()
    assert destination.load_cast(cast["id"]).references[0].transcript.startswith("<|speaker:4|>")
    assert destination.import_bundle(bundle)["versions"] == [version["id"]]
    assert len(destination.list_voices()[0]["versions"]) == 1


def test_bundle_can_exclude_original_sources(tmp_path, session, audio):
    source = VoiceLibrary(tmp_path / "source")
    voice, version = add(source, session, audio)
    bundle = tmp_path / "without-original.zip"
    source.export_bundle(bundle, voice_ids=[voice["id"]], include_sources=False)
    destination = VoiceLibrary(tmp_path / "destination")
    destination.import_bundle(bundle)
    assert destination.get_version(version["id"])["source_path"] is None
    assert source.source_file(version["id"]).is_file()


def test_name_collision_rename_skip_and_error(tmp_path, session):
    source = VoiceLibrary(tmp_path / "source")
    voice, _ = add(source, session)
    bundle = tmp_path / "voice.zip"
    source.export_bundle(bundle)
    for mode in ("rename", "skip", "error"):
        destination = VoiceLibrary(tmp_path / mode)
        old, _ = add(destination, session)
        if mode == "error":
            with pytest.raises(ValueError, match="name"):
                destination.import_bundle(bundle, conflict=mode)
            assert len(destination.list_voices()) == 1
        else:
            result = destination.import_bundle(bundle, conflict=mode)
            assert destination.get_voice(old["id"])["name"] == "主角"
            if mode == "rename":
                assert destination.get_voice(voice["id"])["name"] == "主角 (2)"
            else:
                assert result["skipped_voices"] == [voice["id"]]


@pytest.mark.parametrize("attack", ["traversal", "windows_case", "unlisted", "checksum", "native"])
def test_hostile_bundle_is_rejected_atomically(tmp_path, session, attack):
    source = VoiceLibrary(tmp_path / "source")
    _, version = add(source, session)
    original = tmp_path / "original.zip"
    source.export_bundle(original)
    def mutate(manifest, members):
        if attack == "traversal":
            members["../escape.wav"] = b"evil"
        elif attack == "windows_case":
            members["MANIFEST.JSON"] = b"evil"
        elif attack == "unlisted":
            members["unlisted.wav"] = b"evil"
        elif attack == "checksum":
            members[version["session_path"]] += b"tampered"
        else:
            members[version["session_path"]] = b"not safetensors"
            manifest["files"][0]["bytes"] = len(members[version["session_path"]])
            digest = hashlib.sha256(members[version["session_path"]]).hexdigest()
            manifest["files"][0]["sha256"] = digest
            manifest["versions"][0]["content_sha256"] = digest
    hostile = rewrite_bundle(original, tmp_path / "hostile.zip", mutate)
    destination = VoiceLibrary(tmp_path / "destination")
    with pytest.raises(ValueError):
        destination.import_bundle(hostile)
    assert destination.list_voices(include_archived=True) == []
    assert not (tmp_path / "escape.wav").exists()


def test_uuid_version_conflict_rolls_back_new_voices_and_artifacts(tmp_path, session):
    source = VoiceLibrary(tmp_path / "source")
    first, original = add(source, session)
    second, extra = add(source, session, name="另一位")
    bundle = tmp_path / "source.zip"
    source.export_bundle(bundle)
    target = VoiceLibrary(tmp_path / "target")
    target.import_bundle(bundle)
    before = {item["id"] for item in target.list_voices()}
    before_files = sorted(path.relative_to(target.root).as_posix() for path in (target.root / "versions").rglob("*") if path.is_file())
    third, new_version = add(source, session, name="新演员")
    expanded = tmp_path / "expanded.zip"
    source.export_bundle(expanded)
    def mutate(manifest, members):
        manifest["versions"].sort(key=lambda item: item["id"] != new_version["id"])
        for item in manifest["versions"]:
            if item["id"] == original["id"]:
                item["delivery"] = "恶意改了不可变版本"
    hostile = rewrite_bundle(expanded, tmp_path / "conflict.zip", mutate)
    with pytest.raises(ValueError, match="UUID conflict"):
        target.import_bundle(hostile)
    assert {item["id"] for item in target.list_voices()} == before
    after_files = sorted(path.relative_to(target.root).as_posix() for path in (target.root / "versions").rglob("*") if path.is_file())
    assert after_files == before_files
    assert target.get_version(original["id"])["delivery"] == "平静"


def test_source_path_guard_rejects_external_traversal(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "library")
    _, version = add(library, session, audio)
    source = library.source_file(version["id"])
    # Use the database to model corruption: the public API does not accept paths.
    with library._db() as db:
        db.execute("UPDATE versions SET source_path=? WHERE id=?", ("../original.wav", version["id"]))
    with pytest.raises(ValueError, match="Unsafe"):
        library.source_file(version["id"])
    assert audio.is_file() and source.is_file()


def test_bad_archive_and_oversized_members_are_rejected(tmp_path, session, monkeypatch):
    from drama3_t8 import voice_library
    destination = VoiceLibrary(tmp_path / "target")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not zip")
    with pytest.raises(ValueError, match="archive"):
        destination.import_bundle(bad)
    source = VoiceLibrary(tmp_path / "source")
    add(source, session)
    bundle = tmp_path / "source.zip"
    source.export_bundle(bundle)
    monkeypatch.setattr(voice_library, "MAX_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="large"):
        destination.import_bundle(bundle)
    assert destination.list_voices(include_archived=True) == []


@pytest.mark.parametrize("value", [[], False, {"value": float("inf")}])
def test_invalid_defaults_do_not_publish_a_version(tmp_path, session, value):
    library = VoiceLibrary(tmp_path / "library")
    voice = library.create_voice("a")
    with pytest.raises(ValueError):
        library.add_version(voice["id"], session, defaults=value)
    assert library.list_versions(voice["id"]) == []


def test_old_bundle_without_groups_or_preview_fields_is_compatible(tmp_path, session, audio):
    source = VoiceLibrary(tmp_path / "source")
    voice, version = add(source, session, audio)
    cast = source.create_cast("old", {"a": {"voice_id": voice["id"], "version_id": version["id"]}})
    bundle = tmp_path / "new.zip"
    source.export_bundle(bundle)
    def old_schema(manifest, members):
        for item in manifest["versions"]:
            item.pop("preview_path", None)
            item.pop("preview_sha256", None)
        for item in manifest["casts"]:
            for binding in item["bindings"].values():
                binding.pop("reference_groups", None)
    old = rewrite_bundle(bundle, tmp_path / "old.zip", old_schema)
    target = VoiceLibrary(tmp_path / "target")
    target.import_bundle(old)
    assert target.preview_file(version["id"]).read_bytes() == audio.read_bytes()
    assert len(target.load_cast(cast["id"]).references) == 1
    assert target.import_bundle(old)["versions"] == [version["id"]]



def test_cast_reference_groups_are_explicit_immutable_snapshots(tmp_path, session):
    library = VoiceLibrary(tmp_path / "library")
    voice, first = add(library, session)
    second = library.add_version(voice["id"], session, language="en", delivery="生气")
    cast = library.create_cast("组剧", {"小白": {"voice_id": voice["id"], "version_id": first["id"], "speaker": 2,
                                                "reference_groups": {"生气": [first["id"], second["id"]], "英语": [second["id"]]}}})
    library.add_version(voice["id"], session, language="ja", delivery="新参考")
    assert cast["bindings"]["小白"]["reference_groups"]["英语"] == [second["id"]]
    default = library.load_cast(cast["id"])
    chosen = library.load_cast(cast["id"], reference_groups={"小白": "生气"})
    assert len(default.references) == 1 and len(chosen.references) == 2
    assert default.references[0].processing["voice_library"]["version_id"] == first["id"]
    assert {r.processing["voice_library"]["version_id"] for r in chosen.references} == {first["id"], second["id"]}
    assert all(r.transcript.startswith("<|speaker:2|>") for r in chosen.references)
    with pytest.raises(ValueError, match="Unknown reference group"):
        library.load_cast(cast["id"], reference_groups={"小白": "不存在"})
    with pytest.raises(ValueError, match="known cast roles"):
        library.load_cast(cast["id"], reference_groups={"不存在": "生气"})


@pytest.mark.parametrize("case", ["foreign", "empty", "blank", "duplicate"])
def test_cast_rejects_invalid_reference_groups(tmp_path, session, case):
    library = VoiceLibrary(tmp_path / "library")
    voice, first = add(library, session)
    _, foreign = add(library, session, name="别人")
    groups = {"foreign": {"a": [foreign["id"]]}, "empty": {"a": []},
              "blank": {" ": [first["id"]]}, "duplicate": {"a": [first["id"], first["id"]]}}[case]
    with pytest.raises(ValueError):
        library.create_cast("bad", {"a": {"voice_id": voice["id"], "version_id": first["id"], "reference_groups": groups}})
    assert library.list_casts() == []


def test_reference_groups_roundtrip_and_old_cast_interface(tmp_path, session):
    source = VoiceLibrary(tmp_path / "source")
    voice, first = add(source, session)
    second = source.add_version(voice["id"], session, language="en")
    cast = source.create_cast("分组", {"a": {"voice_id": voice["id"], "version_id": first["id"],
                                              "reference_groups": {"英语": [second["id"]]}}})
    old = source.create_cast("旧接口", {"a": {"voice_id": voice["id"], "version_id": first["id"]}})
    bundle = tmp_path / "voices.zip"
    source.export_bundle(bundle)
    target = VoiceLibrary(tmp_path / "target")
    target.import_bundle(bundle)
    assert len(target.load_cast(old["id"]).references) == 1
    assert target.load_cast(cast["id"], reference_groups={"a": "英语"}).references[0].processing["voice_library"]["version_id"] == second["id"]
    def mutate(manifest, members):
        next(item for item in manifest["casts"] if item["id"] == cast["id"])["bindings"]["a"]["reference_groups"]["英语"] = []
    hostile = rewrite_bundle(bundle, tmp_path / "bad-group.zip", mutate)
    empty = VoiceLibrary(tmp_path / "empty")
    with pytest.raises(ValueError, match="reference group"):
        empty.import_bundle(hostile)
    assert empty.list_voices(include_archived=True) == []


@pytest.mark.parametrize("include_sources", [True, False])
def test_original_and_processed_preview_are_separate_and_portable(tmp_path, session, audio, include_sources):
    preview = tmp_path / "clip.wav"
    with wave.open(str(preview), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(b"\x02\x00" * 1200)
    source = VoiceLibrary(tmp_path / "source")
    voice = source.create_voice("选段")
    version = source.add_version(voice["id"], session, source_path=audio, preview_path=preview)
    assert source.source_file(version["id"]).read_bytes() == audio.read_bytes()
    assert source.preview_file(version["id"]).read_bytes() == preview.read_bytes()
    assert version["source_path"] != version["preview_path"]
    bundle = tmp_path / "clip-only.zip"
    source.export_bundle(bundle, include_sources=include_sources)
    target = VoiceLibrary(tmp_path / "target")
    target.import_bundle(bundle)
    imported = target.get_version(version["id"])
    assert imported["preview_path"]
    assert target.preview_file(version["id"]).read_bytes() == preview.read_bytes()
    if include_sources:
        assert target.source_file(version["id"]).read_bytes() == audio.read_bytes()
    else:
        assert imported["source_path"] is None
        with pytest.raises(ValueError, match="no associated original"):
            target.source_file(version["id"])
    assert source.source_file(version["id"]).is_file() and audio.is_file() and preview.is_file()


def test_preview_without_original_and_alias_fallback(tmp_path, session, audio):
    library = VoiceLibrary(tmp_path / "library")
    voice, alias = add(library, session, audio)
    assert library.preview_file(alias["id"]) == library.source_file(alias["id"])
    clip = library.add_version(voice["id"], session, preview_path=audio)
    assert clip["source_path"] is None and clip["preview_path"]
    assert library.preview_file(clip["id"]).read_bytes() == audio.read_bytes()
    library.preview_file(clip["id"]).write_bytes(b"corrupted preview")
    with pytest.raises(ValueError, match="checksum"):
        library.preview_file(clip["id"])


def test_old_database_preview_columns_migrate_without_touching_audio(tmp_path, session, audio):
    import sqlite3
    import uuid
    root = tmp_path / "legacy-schema"
    root.mkdir()
    voice_id, version_id = uuid.uuid4().hex, uuid.uuid4().hex
    artifact = root / "versions" / version_id
    artifact.mkdir(parents=True)
    save_session(session, artifact / "reference.safetensors")
    (artifact / "source.wav").write_bytes(audio.read_bytes())
    native_hash = hashlib.sha256((artifact / "reference.safetensors").read_bytes()).hexdigest()
    source_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    with sqlite3.connect(root / "library.sqlite3") as db:
        db.executescript("""
            CREATE TABLE voices(id TEXT PRIMARY KEY,name TEXT NOT NULL,tags TEXT NOT NULL,
              favorite INTEGER NOT NULL,notes TEXT NOT NULL,archived INTEGER NOT NULL,
              created REAL NOT NULL,updated REAL NOT NULL,legacy_id TEXT UNIQUE);
            CREATE TABLE versions(id TEXT PRIMARY KEY,voice_id TEXT NOT NULL REFERENCES voices(id),
              session_path TEXT NOT NULL,content_sha256 TEXT NOT NULL,source_path TEXT,source_sha256 TEXT,
              transcript TEXT NOT NULL,language TEXT NOT NULL,delivery TEXT NOT NULL,defaults TEXT NOT NULL,created REAL NOT NULL);
        """)
        db.execute("INSERT INTO voices VALUES(?,?,?,?,?,?,?,?,?)", (voice_id, "old", "[]", 0, "", 0, 1.0, 1.0, None))
        db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (version_id, voice_id, f"versions/{version_id}/reference.safetensors", native_hash,
                    f"versions/{version_id}/source.wav", source_hash, session.references[0].transcript, "zh", "", "{}", 1.0))
    library = VoiceLibrary(root)
    assert library.get_version(version_id)["preview_path"] == f"versions/{version_id}/source.wav"
    assert library.preview_file(version_id).read_bytes() == audio.read_bytes()
    assert hashlib.sha256(library.session_file(version_id).read_bytes()).hexdigest() == native_hash
    assert VoiceLibrary(root).get_voice(voice_id)["latest_version_id"] == version_id



def test_low_disk_space_refuses_writes_before_changing_catalog(tmp_path, session, monkeypatch):
    from types import SimpleNamespace
    import drama3_t8.voice_library as module
    source = VoiceLibrary(tmp_path / "source")
    voice, _ = add(source, session)
    bundle = tmp_path / "valid.zip"
    source.export_bundle(bundle)
    target = VoiceLibrary(tmp_path / "target")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="disk space"):
        target.import_bundle(bundle)
    with pytest.raises(ValueError, match="disk space"):
        source.add_version(voice["id"], session)
    with pytest.raises(ValueError, match="disk space"):
        source.export_bundle(tmp_path / "not-created.zip")
    assert target.list_voices(include_archived=True) == []
    assert len(source.list_versions(voice["id"])) == 1
    assert not (tmp_path / "not-created.zip").exists()



def test_blank_tags_and_first_open_concurrency(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    root = tmp_path / "fresh-shared"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: VoiceLibrary(root), range(4)))
    library = VoiceLibrary(root)
    voice = library.create_voice("a", tags="")
    assert voice["tags"] == [] and library.list_voices(tags="")[0]["id"] == voice["id"]
    with library._db() as db:
        assert db.execute("SELECT COUNT(*) FROM library_meta").fetchone()[0] == 1


def test_failed_version_write_rolls_back_only_its_new_artifacts(tmp_path, session, audio, monkeypatch):
    import drama3_t8.state as state
    library = VoiceLibrary(tmp_path / "library")
    voice, existing = add(library, session, audio)
    saved = library.session_file(existing["id"]).read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(state, "save_session", fail)
    with pytest.raises(OSError, match="storage"):
        library.add_version(voice["id"], session, source_path=audio)
    assert len(library.list_versions(voice["id"])) == 1
    assert library.session_file(existing["id"]).read_bytes() == saved
    assert len(list((library.root / "versions").iterdir())) == 1
    assert audio.is_file()



def test_bundle_long_name_collisions_get_distinct_names(tmp_path, session):
    source = VoiceLibrary(tmp_path / "source")
    name = "a" * 120
    add(source, session, name=name)
    add(source, session, name=name)
    bundle = tmp_path / "voices.zip"
    source.export_bundle(bundle)
    target = VoiceLibrary(tmp_path / "target")
    add(target, session, name=name)
    target.import_bundle(bundle)
    names = [item["name"] for item in target.list_voices()]
    assert len(set(names)) == 3 and all(len(item) <= 120 for item in names)


def test_cast_bundle_invalid_binding_is_rejected_before_publication(tmp_path, session):
    source = VoiceLibrary(tmp_path / "source")
    voice, version = add(source, session)
    source.create_cast("cast", {"a": {"voice_id": voice["id"], "version_id": version["id"]}})
    bundle = tmp_path / "valid.zip"
    source.export_bundle(bundle)
    def mutate(manifest, members):
        manifest["casts"][0]["bindings"]["a"]["version_id"] = []
    hostile = rewrite_bundle(bundle, tmp_path / "bad.zip", mutate)
    target = VoiceLibrary(tmp_path / "target")
    with pytest.raises(ValueError, match="UUID"):
        target.import_bundle(hostile)
    assert target.list_voices(include_archived=True) == [] and target.list_casts() == []


def test_multiple_hosts_can_create_versions_in_shared_custom_library(tmp_path, session):
    from concurrent.futures import ThreadPoolExecutor
    root = tmp_path / "shared"
    first = VoiceLibrary(root)
    voice = first.create_voice("shared")
    def write(index):
        host = VoiceLibrary(root)
        return host.add_version(voice["id"], session, delivery=f"variant-{index}")["id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(write, range(8)))
    assert len(set(ids)) == 8
    assert len(first.list_versions(voice["id"])) == 8
    assert {item["delivery"] for item in first.list_versions(voice["id"])} == {f"variant-{index}" for index in range(8)}


def test_shared_selection_preserves_tags_unless_explicitly_mapped(tmp_path, session):
    library = VoiceLibrary(tmp_path / "library")
    voice, version = add(library, session)
    original = library.load_selection([{"voice_id": voice["id"], "version_id": version["id"]}])
    assert original.references[0].transcript == session.references[0].transcript
    mapped = library.load_selection([{"voice_id": voice["id"], "version_id": version["id"], "speaker": 6,
                                      "role": "旁白"}], direction="喜剧旁白")
    assert mapped.references[0].transcript.startswith("<|speaker:6|>")
    assert mapped.direction == "喜剧旁白" and mapped.references[0].name == "旁白"
    assert library.load_version(version["id"]).references[0].transcript.startswith("<|speaker:0|>")
    with pytest.raises(ValueError, match="does not belong"):
        library.load_selection([{"voice_id": "f" * 32, "version_id": version["id"]}])
    with pytest.raises(ValueError, match="Speaker"):
        library.load_selection([{"voice_id": voice["id"], "version_id": version["id"], "speaker": "3"}])


def test_native_session_must_not_include_generated_takes(tmp_path, session):
    from drama3_t8.state import Take
    library = VoiceLibrary(tmp_path / "library")
    voice = library.create_voice("a")
    with pytest.raises(ValueError, match="reference-only"):
        library.add_version(voice["id"], Session(session.references, (Take("a", "a", torch.zeros((0, 11), dtype=torch.int16)),)))
    assert library.list_versions(voice["id"]) == []



