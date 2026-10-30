"""Portable, versioned voice assets shared by Studio and ComfyUI.

No model is loaded on import. References use the existing native Session format.
Versions and cast snapshots are immutable; editing a voice only edits its catalog.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile

SCHEMA = 1
MAX_MEMBERS = 2048
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MIN_FREE_BYTES = 64 * 1024 * 1024
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma", ".webm", ".mp4"}


def _resolved(value):
    # Python 3.10 on Windows can return an extended prefix only for longer
    # descendants, even when the root has none. Compare one canonical spelling.
    resolved = str(Path(value).expanduser().resolve())
    if os.name == "nt" and resolved.startswith("\\\\?\\"):
        resolved = ("\\\\" + resolved[8:]) if resolved.startswith("\\\\?\\UNC\\") else resolved[4:]
    return Path(resolved)


def default_library_path(workspace=None):
    configured = os.environ.get("DRAMA3_VOICE_LIBRARY", "").strip()
    return _resolved(configured) if configured else _resolved(Path(workspace or Path.cwd()) / "userdata" / "voice-library")


def _id(value):
    try:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError()
        uuid.UUID(hex=value)
        return value
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid voice library UUID") from exc


def _text(value, field, limit=2000, empty=True):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(f"Invalid {field}")
    value = value.strip()
    if not empty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _json(value, field="defaults"):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must contain finite JSON data") from exc
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError(f"{field} is too large")
    return decoded


def _tags(values):
    if isinstance(values, str):
        values = [item.strip() for item in re.split(r"[,，;；\n]", values) if item.strip()]
    if not isinstance(values, (tuple, list)) or len(values) > 64:
        raise ValueError("Tags must be a list of at most 64 items")
    result, seen = [], set()
    for value in values:
        value = _text(value, "tag", 48, empty=False)
        if value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("Unsafe bundle member path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in {"", ".", ".."} or part.rstrip(" .") != part for part in path.parts):
        raise ValueError("Unsafe bundle member path")
    if any(re.fullmatch(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", part, re.I) for part in path.parts):
        raise ValueError("Reserved bundle member path")
    return value


def _native_session(value):
    # Deliberately lazy: listing and source audio preview do not import Torch.
    from .state import Session, load_session, validate_reference
    if isinstance(value, (str, Path)):
        from safetensors import SafetensorError
        try:
            session = load_session(value)
        except (OSError, ValueError, KeyError, TypeError, SafetensorError) as exc:
            raise ValueError("Invalid native reference Session") from exc
    else:
        session = value
    if not isinstance(session, Session) or not session.references or session.takes:
        raise ValueError("Voice versions require a nonempty reference-only Session")
    for reference in session.references:
        validate_reference(reference)
    return session


def apply_explicit_defaults(current, defaults, *, apply=False):
    """Only an explicit user action applies defaults; current values otherwise win."""
    if type(apply) is not bool:
        raise ValueError("Apply defaults must be an explicit boolean")
    current, defaults = _json(current, "current settings"), _json(defaults)
    return {**current, **defaults} if apply else current


class VoiceLibrary:
    def __init__(self, root=None):
        self.root = _resolved(root if root is not None else default_library_path())
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "library.sqlite3"
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS library_meta(version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS voices(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, tags TEXT NOT NULL,
                    favorite INTEGER NOT NULL, notes TEXT NOT NULL, archived INTEGER NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, legacy_id TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS versions(
                    id TEXT PRIMARY KEY, voice_id TEXT NOT NULL REFERENCES voices(id),
                    session_path TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                    source_path TEXT, source_sha256 TEXT, transcript TEXT NOT NULL,
                    language TEXT NOT NULL, delivery TEXT NOT NULL, defaults TEXT NOT NULL,
                    created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS voice_versions ON versions(voice_id,created);
                CREATE TABLE IF NOT EXISTS casts(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, snapshot TEXT NOT NULL, created REAL NOT NULL);
            """)
            db.execute("INSERT INTO library_meta(version) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM library_meta)", (SCHEMA,))
            from .catalog_trash import initialize
            initialize(db)
            version = db.execute("SELECT version FROM library_meta").fetchone()
            if version[0] != SCHEMA:
                raise ValueError("Unsupported voice library schema")
            columns = {row[1] for row in db.execute("PRAGMA table_info(versions)")}
            for column in ("preview_path", "preview_sha256"):
                if column not in columns:
                    db.execute(f"ALTER TABLE versions ADD COLUMN {column} TEXT")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _file(self, relative, digest=None):
        _member(relative)
        path = _resolved(self.root / relative)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Voice library artifact is missing or outside its library")
        if digest and _sha(path) != digest:
            raise ValueError("Voice library artifact checksum mismatch")
        return path

    @staticmethod
    def _version(row):
        if row is None:
            raise ValueError("Unknown voice version")
        result = dict(row)
        result["defaults"] = json.loads(result["defaults"])
        if result.get("preview_path") is None:
            result["preview_path"], result["preview_sha256"] = result["source_path"], result["source_sha256"]
        result["created_at"] = result.pop("created")
        return result

    def create_voice(self, name, *, tags=(), favorite=False, notes="", legacy_id=None):
        name, tags, notes = _text(name, "name", 120, False), _tags(tags), _text(notes, "notes", 8000)
        if type(favorite) is not bool:
            raise ValueError("Favorite must be a boolean")
        if legacy_id is not None:
            legacy_id = _text(legacy_id, "legacy ID", 256, False)
        voice_id, now = uuid.uuid4().hex, time.time()
        with self._db() as db:
            db.execute("INSERT INTO voices VALUES(?,?,?,?,?,?,?,?,?)",
                       (voice_id, name, json.dumps(tags, ensure_ascii=False), int(favorite), notes, 0, now, now, legacy_id))
        return self.get_voice(voice_id)

    def get_voice(self, voice_id, *, include_deleted_versions=False):
        with self._db() as db:
            row = db.execute("SELECT * FROM voices WHERE id=?", (_id(voice_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown voice")
        result = dict(row)
        result["tags"] = json.loads(result["tags"])
        result["favorite"], result["archived"] = bool(result["favorite"]), bool(result["archived"])
        result["created_at"], result["updated_at"] = result.pop("created"), result.pop("updated")
        versions = self.list_versions(voice_id, include_deleted=include_deleted_versions)
        result["versions"] = versions
        result["latest_version_id"] = versions[-1]["id"] if versions else None
        result["transcript"] = versions[-1]["transcript"] if versions else ""
        return result

    def list_voices(self, query="", *, tags=(), favorites_only=False, include_archived=False):
        query, requested = _text(query, "query", 1000).casefold(), {item.casefold() for item in _tags(tags)}
        if type(favorites_only) is not bool or type(include_archived) is not bool:
            raise ValueError("Library filters must be booleans")
        with self._db() as db:
            from .catalog_trash import hidden
            deleted = hidden(db, "voice")
            ids = [row[0] for row in db.execute("SELECT id FROM voices ORDER BY created DESC,id DESC") if row[0] not in deleted]
        results = []
        for voice_id in ids:
            voice = self.get_voice(voice_id)
            if voice["archived"] and not include_archived or favorites_only and not voice["favorite"]:
                continue
            if not requested.issubset({tag.casefold() for tag in voice["tags"]}):
                continue
            haystack = " ".join([voice["name"], voice["notes"], *voice["tags"],
                                  *(v["language"] + " " + v["delivery"] + " " + v["transcript"] for v in voice["versions"])]).casefold()
            if not query or query in haystack:
                results.append(voice)
        return results

    def update_voice(self, voice_id, *, name=None, tags=None, favorite=None, notes=None, archived=None):
        voice = self.get_voice(voice_id)
        for key, value in (("name", name), ("tags", tags), ("favorite", favorite), ("notes", notes), ("archived", archived)):
            if value is not None:
                if key == "name":
                    value = _text(value, key, 120, False)
                elif key == "tags":
                    value = _tags(value)
                elif key == "notes":
                    value = _text(value, key, 8000)
                elif type(value) is not bool:
                    raise ValueError(f"{key} must be a boolean")
                voice[key] = value
        with self._db() as db:
            db.execute("UPDATE voices SET name=?,tags=?,favorite=?,notes=?,archived=?,updated=? WHERE id=?",
                       (voice["name"], json.dumps(voice["tags"], ensure_ascii=False), int(voice["favorite"]),
                        voice["notes"], int(voice["archived"]), time.time(), voice["id"]))
        return self.get_voice(voice_id)

    def archive_voice(self, voice_id, archived=True):
        return self.update_voice(voice_id, archived=archived)

    def add_version(self, voice_id, reference_session, *, source_path=None, preview_path=None, transcript=None, language="", delivery="", defaults=None):
        voice = self.get_voice(voice_id)
        session = _native_session(reference_session)
        actual_transcript = "\n".join(reference.transcript for reference in session.references)
        transcript = actual_transcript if transcript is None else _text(transcript, "transcript", 100000)
        if transcript != actual_transcript:
            raise ValueError("Transcript differs from the encoded Session; re-encode before saving a changed transcript")
        language, delivery = _text(language, "language", 80), _text(delivery, "delivery", 160)
        defaults = _json({} if defaults is None else defaults)
        source = None
        if source_path is not None:
            source = _resolved(source_path)
            if not source.is_file() or source.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError("Original source must be an existing audio file")
            if source.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Original source is too large")
        preview = None
        if preview_path is not None:
            preview = _resolved(preview_path)
            if not preview.is_file() or preview.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError("Preview must be an existing audio file")
            if preview.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Preview audio is too large")
        estimate = sum(ref.codes.numel() * ref.codes.element_size() for ref in session.references) + 1024 * 1024
        if source is not None:
            estimate += source.stat().st_size
        if preview is not None and preview != source:
            estimate += preview.stat().st_size
        if shutil.disk_usage(self.root).free < estimate + MIN_FREE_BYTES:
            raise ValueError("Insufficient free disk space for a voice version")
        version_id, now = uuid.uuid4().hex, time.time()
        folder = _resolved(self.root / "versions" / version_id)
        if not folder.is_relative_to(self.root):
            raise ValueError("Version artifact path is outside the library")
        folder.mkdir(parents=True, exist_ok=False)
        destination = folder / "reference.safetensors"
        try:
            from .state import save_session
            save_session(session, destination)
            source_relative, source_sha = None, None
            if source is not None:
                copied = folder / ("source" + source.suffix.lower())
                with source.open("rb") as reader, copied.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                source_relative, source_sha = copied.relative_to(self.root).as_posix(), _sha(copied)
            preview_relative, preview_sha = source_relative, source_sha
            if preview is not None and preview != source:
                copied_preview = folder / ("preview" + preview.suffix.lower())
                with preview.open("rb") as reader, copied_preview.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                preview_relative, preview_sha = copied_preview.relative_to(self.root).as_posix(), _sha(copied_preview)
            row = (version_id, voice["id"], destination.relative_to(self.root).as_posix(), _sha(destination),
                   source_relative, source_sha, transcript, language, delivery,
                   json.dumps(defaults, ensure_ascii=False, sort_keys=True, allow_nan=False), now, preview_relative, preview_sha)
            with self._db() as db:
                db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                db.execute("UPDATE voices SET updated=? WHERE id=?", (now, voice["id"]))
        except BaseException:
            # Only this call's new UUID folder is rolled back; originals remain untouched.
            shutil.rmtree(folder)
            raise
        return self.get_version(version_id)

    def list_versions(self, voice_id, *, include_deleted=False):
        with self._db() as db:
            rows = db.execute("SELECT * FROM versions WHERE voice_id=? ORDER BY created,id", (_id(voice_id),)).fetchall()
            from .catalog_trash import hidden
            deleted = hidden(db, "version")
        return [self._version(row) for row in rows if include_deleted or row["id"] not in deleted]

    def get_version(self, version_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM versions WHERE id=?", (_id(version_id),)).fetchone()
        return self._version(row)

    def session_file(self, version_id):
        version = self.get_version(version_id)
        return self._file(version["session_path"], version["content_sha256"])

    def source_file(self, version_id):
        version = self.get_version(version_id)
        if not version["source_path"]:
            raise ValueError("This voice version has no associated original source audio")
        return self._file(version["source_path"], version["source_sha256"])

    def preview_file(self, version_id):
        version = self.get_version(version_id)
        if not version["preview_path"]:
            raise ValueError("This voice version has no associated preview audio")
        return self._file(version["preview_path"], version["preview_sha256"])

    def load_version(self, version_id):
        return _native_session(self.session_file(version_id))

    def migrate_legacy(self, rows, base_dir):
        """Idempotently adopt exact legacy associations; never guess input files."""
        base_dir = _resolved(base_dir)
        result = {"mapping": {}, "migrated": [], "errors": []}
        for incoming in rows:
            row = dict(incoming)
            legacy_id = str(row.get("id", ""))
            try:
                _text(legacy_id, "legacy ID", 256, False)
                with self._db() as db:
                    old = db.execute("SELECT id FROM voices WHERE legacy_id=?", (legacy_id,)).fetchone()
                if old:
                    voice = self.get_voice(old[0])
                    if voice["latest_version_id"]:
                        result["mapping"][legacy_id] = {"voice_id": voice["id"], "version_id": voice["latest_version_id"]}
                        continue
                    if self.list_versions(old[0], include_deleted=True):
                        # Deleted versions stay deleted across legacy catalog refreshes.
                        continue
                reference = Path(row["path"])
                reference = reference if reference.is_absolute() else base_dir / reference
                source = row.get("source_path") or None
                if source:
                    source = Path(source)
                    source = source if source.is_absolute() else base_dir / source
                preview = row.get("preview_path") or None
                if preview:
                    preview = Path(preview)
                    preview = preview if preview.is_absolute() else base_dir / preview
                session = _native_session(reference)
                voice = self.get_voice(old[0]) if old else self.create_voice(row.get("name") or "未命名音色", legacy_id=legacy_id)
                version = self.add_version(voice["id"], session, source_path=source, preview_path=preview,
                                           language=row.get("language", ""), delivery=row.get("delivery", ""),
                                           defaults=row.get("defaults"))
                result["mapping"][legacy_id] = {"voice_id": voice["id"], "version_id": version["id"]}
                result["migrated"].append(legacy_id)
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
                result["errors"].append({"legacy_id": legacy_id, "error": str(exc)})
        return result

    def create_cast(self, name, bindings, *, defaults=None):
        name = _text(name, "cast name", 120, False)
        if not isinstance(bindings, dict) or not bindings or len(bindings) > 64:
            raise ValueError("Cast bindings must contain 1 to 64 named roles")
        snapshot = {"bindings": {}, "defaults": _json({} if defaults is None else defaults)}
        for role, binding in bindings.items():
            role = _text(role, "role", 120, False)
            if role in snapshot["bindings"] or not isinstance(binding, dict):
                raise ValueError("Invalid or duplicate cast role")
            voice = self.get_voice(binding["voice_id"])
            version = self.get_version(binding.get("version_id") or voice["latest_version_id"])
            if version["voice_id"] != voice["id"]:
                raise ValueError("Cast version does not belong to this voice")
            speaker = binding.get("speaker", 0)
            if type(speaker) is not int or not 0 <= speaker < 16:
                raise ValueError("Cast speaker must be an integer from 0 to 15")
            groups = self._reference_groups(binding.get("reference_groups", {}), voice["id"])
            snapshot["bindings"][role] = {"voice_id": voice["id"], "voice_name": voice["name"], "version_id": version["id"],
                                          "speaker": speaker, "defaults": _json({} if binding.get("defaults") is None else binding["defaults"]),
                                          "voice_defaults": version["defaults"], "language": version["language"], "delivery": version["delivery"],
                                          "reference_groups": groups}
            for ids in groups.values():
                self.load_selection([{"voice_id": voice["id"], "version_id": item, "speaker": speaker, "role": role} for item in ids])
        # Validate actual speaker remapping before publishing the snapshot.
        self._cast_session(snapshot)
        cast_id, now = uuid.uuid4().hex, time.time()
        with self._db() as db:
            db.execute("INSERT INTO casts VALUES(?,?,?,?)", (cast_id, name, json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True), now))
        return self.get_cast(cast_id)

    def get_cast(self, cast_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM casts WHERE id=?", (_id(cast_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown cast")
        result = dict(row)
        result.update(json.loads(result.pop("snapshot")))
        result["created_at"] = result.pop("created")
        return result

    def list_casts(self):
        with self._db() as db:
            from .catalog_trash import hidden
            deleted = hidden(db, "cast")
            ids = [row[0] for row in db.execute("SELECT id FROM casts ORDER BY created DESC,id DESC") if row[0] not in deleted]
        return [self.get_cast(cast_id) for cast_id in ids]

    def trash_item(self, kind, item_id, *, restore=False):
        from .catalog_trash import mark
        item_id = _id(item_id)
        if kind == "voice":
            label = self.get_voice(item_id)["name"]
        elif kind == "cast":
            label = self.get_cast(item_id)["name"]
        elif kind == "version":
            version = self.get_version(item_id)
            label = self.get_voice(version["voice_id"])["name"] + " · " + (version["delivery"] or item_id[:8])
        else:
            raise ValueError("不支持的音色资料类型")
        with self._db() as db:
            mark(db, kind, item_id, label, restore=restore)
        return {"kind": kind, "id": item_id, "restored": restore}

    def trash_items(self):
        from .catalog_trash import entries
        with self._db() as db:
            return entries(db)

    def _reference_groups(self, groups, voice_id):
        if not isinstance(groups, dict) or len(groups) > 32:
            raise ValueError("Reference groups must be an object with at most 32 named groups")
        result = {}
        for name, versions in groups.items():
            if _text(name, "reference group name", 80, False) != name:
                raise ValueError("Reference group names must be canonical")
            if not isinstance(versions, list) or not versions or len(versions) > 64:
                raise ValueError("Reference group must contain 1 to 64 version IDs")
            normalized = [_id(item) for item in versions]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Reference group contains duplicate versions")
            if any(self.get_version(item)["voice_id"] != voice_id for item in normalized):
                raise ValueError("Reference group version does not belong to this voice")
            result[name] = normalized
        return result

    def _cast_session(self, snapshot, reference_groups=None):
        chosen = {} if reference_groups is None else reference_groups
        if not isinstance(chosen, dict) or set(chosen) - set(snapshot["bindings"]):
            raise ValueError("Group selection must map known cast roles to group names")
        selections = []
        for role, binding in snapshot["bindings"].items():
            ids = [binding["version_id"]]
            if role in chosen:
                name = _text(chosen[role], "reference group name", 80, False)
                if name not in binding.get("reference_groups", {}):
                    raise ValueError(f"Unknown reference group for role {role}")
                ids = binding["reference_groups"][name]
            selections.extend({**binding, "version_id": item, "role": role} for item in ids)
        return self.load_selection(selections)

    def load_selection(self, selections, *, direction=""):
        """Resolve explicit immutable versions for either host, without sampling defaults."""
        from .state import Session
        if not isinstance(selections, (list, tuple)) or not selections or len(selections) > 64:
            raise ValueError("Select 1 to 64 voice versions")
        direction = _text(direction, "direction", 32000)
        references = []
        for binding in selections:
            if not isinstance(binding, dict):
                raise ValueError("Voice selection must be an object")
            version = self.get_version(binding.get("version_id"))
            if version["voice_id"] != _id(binding.get("voice_id")):
                raise ValueError("Selected version does not belong to this voice")
            role = _text(binding.get("role", ""), "role", 120)
            speaker = binding.get("speaker")
            if speaker is not None and (type(speaker) is not int or not 0 <= speaker < 16):
                raise ValueError("Speaker must be an integer from 0 to 15")
            for reference in self.load_version(version["id"]).references:
                if speaker is not None and len(set(re.findall(r"<\|speaker:(\d+)\|>", reference.transcript))) > 1:
                    raise ValueError("Split a multi-speaker reference before assigning it to one cast role")
                transcript = reference.transcript
                if speaker is not None:
                    transcript = re.sub(r"<\|speaker:\d+\|>\s*", "", transcript)
                    transcript = f"<|speaker:{speaker}|> {transcript}"
                processing = {**reference.processing, "voice_library": {"voice_id": binding["voice_id"],
                              "version_id": binding["version_id"], "role": role}}
                references.append(replace(reference, name=role or reference.name, transcript=transcript, processing=processing))
        # Defaults stay metadata: callers explicitly apply them in their own UI.
        return Session(tuple(references), direction=direction)

    def load_cast(self, cast_id, *, reference_groups=None):
        return self._cast_session(self.get_cast(cast_id), reference_groups)

    def export_bundle(self, path, *, voice_ids=None, version_ids=None, cast_ids=None, include_sources=True):
        """Export native references and optional byte-identical original audio."""
        if type(include_sources) is not bool:
            raise ValueError("include_sources must be a boolean")
        destination = _resolved(path)
        if destination.exists():
            raise FileExistsError("Refusing to overwrite an existing voice bundle")
        destination.parent.mkdir(parents=True, exist_ok=True)
        casts = self.list_casts() if cast_ids is None and voice_ids is None and version_ids is None else [self.get_cast(item) for item in (cast_ids or [])]
        selected_versions = None
        if version_ids is not None:
            if not isinstance(version_ids, (list, tuple, set)) or not version_ids:
                raise ValueError("Select at least one explicit voice version")
            selected_versions = {_id(item) for item in version_ids}
            selected = {self.get_version(item)["voice_id"] for item in selected_versions}
            if voice_ids is not None and not selected <= {_id(item) for item in voice_ids}:
                raise ValueError("Selected versions belong to unselected voices")
        else:
            selected = {voice["id"] for voice in self.list_voices(include_archived=True)} if voice_ids is None else {_id(item) for item in voice_ids}
        for cast in casts:
            if selected_versions is not None:
                required = {binding["version_id"] for binding in cast["bindings"].values()}
                required.update(version for binding in cast["bindings"].values() for group in binding.get("reference_groups", {}).values() for version in group)
                if not required <= selected_versions:
                    raise ValueError("Explicit version selection leaves a cast reference missing")
            selected.update(binding["voice_id"] for binding in cast["bindings"].values())
        voices = [self.get_voice(item, include_deleted_versions=selected_versions is not None) for item in sorted(selected)]
        if not voices:
            raise ValueError("Select at least one voice")
        manifest = {"format": "drama3-voice-library", "schema": SCHEMA, "voices": [], "versions": [],
                    "casts": casts, "files": []}
        artifacts = {}
        for voice in voices:
            manifest["voices"].append({key: voice[key] for key in ("id", "name", "tags", "favorite", "notes", "archived", "created_at", "updated_at")})
            for original in voice["versions"]:
                if selected_versions is not None and original["id"] not in selected_versions:
                    continue
                version = dict(original)
                artifacts[version["session_path"]] = self.session_file(version["id"])
                if version["preview_path"] and version["preview_path"] != version["source_path"]:
                    artifacts[version["preview_path"]] = self.preview_file(version["id"])
                elif not include_sources:
                    version["preview_path"], version["preview_sha256"] = None, None
                if include_sources and version["source_path"]:
                    artifacts[version["source_path"]] = self.source_file(version["id"])
                else:
                    version["source_path"], version["source_sha256"] = None, None
                manifest["versions"].append(version)
        if not manifest["versions"]:
            raise ValueError("Selected voices have no reference versions")
        total = 0
        for relative, artifact in sorted(artifacts.items()):
            size = artifact.stat().st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("Voice bundle exceeds the portable size limit")
            manifest["files"].append({"path": relative, "bytes": size, "sha256": _sha(artifact)})
        encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_MANIFEST_BYTES or len(artifacts) + 1 > MAX_MEMBERS:
            raise ValueError("Voice bundle manifest is too large")
        if shutil.disk_usage(destination.parent).free < total + len(encoded) + MIN_FREE_BYTES:
            raise ValueError("Insufficient free disk space for a voice bundle")
        # A private temporary artifact is atomically linked without replacing a user file.
        handle, temporary = tempfile.mkstemp(prefix=".voice-export-", suffix=".zip", dir=destination.parent)
        os.close(handle)
        temporary = Path(temporary)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
                archive.writestr("manifest.json", encoded)
                for relative, artifact in sorted(artifacts.items()):
                    archive.write(artifact, relative)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(destination), "voices": len(voices), "versions": len(manifest["versions"]),
                "casts": len(casts), "include_sources": include_sources, "sha256": _sha(destination)}

    @staticmethod
    def _bundle_manifest(archive):
        infos = archive.infolist()
        if not infos or len(infos) > MAX_MEMBERS:
            raise ValueError("Invalid bundle member count")
        members, windows_names, total = {}, set(), 0
        for info in infos:
            name = _member(info.filename)
            folded = name.casefold()
            if info.is_dir() or name in members or folded in windows_names or info.flag_bits & 1:
                raise ValueError("Duplicate, encrypted or invalid bundle member")
            if info.file_size < 0 or info.file_size > MAX_FILE_BYTES:
                raise ValueError("Bundle member is too large")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Bundle is too large")
            members[name], _ = info, windows_names.add(folded)
        if "manifest.json" not in members or members["manifest.json"].file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Missing or oversized bundle manifest")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Invalid voice bundle manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != "drama3-voice-library" or type(manifest.get("schema")) is not int or manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported voice bundle format")
        for key, maximum in (("voices", 512), ("versions", 1024), ("casts", 256), ("files", MAX_MEMBERS - 1)):
            if not isinstance(manifest.get(key), list) or len(manifest[key]) > maximum:
                raise ValueError(f"Invalid bundle {key}")
        if not manifest["voices"] or not manifest["versions"]:
            raise ValueError("Voice bundle is empty")
        files = {}
        for record in manifest["files"]:
            if not isinstance(record, dict):
                raise ValueError("Invalid artifact manifest")
            name = _member(record.get("path"))
            if name == "manifest.json" or name in files or type(record.get("bytes")) is not int or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
                raise ValueError("Invalid artifact identity")
            if name not in members or members[name].file_size != record["bytes"]:
                raise ValueError("Missing artifact or size mismatch")
            files[name] = record
        if set(members) != {"manifest.json", *files}:
            raise ValueError("Unlisted or missing bundle artifacts")
        voice_ids, version_ids, used = set(), set(), set()
        for voice in manifest["voices"]:
            if not isinstance(voice, dict) or _id(voice.get("id")) in voice_ids:
                raise ValueError("Duplicate or invalid voice identity")
            voice_ids.add(voice["id"])
            _text(voice.get("name"), "name", 120, False)
            _tags(voice.get("tags"))
            _text(voice.get("notes"), "notes", 8000)
            if type(voice.get("favorite")) is not bool or type(voice.get("archived")) is not bool:
                raise ValueError("Invalid voice metadata flags")
            for field in ("created_at", "updated_at"):
                _timestamp(voice.get(field))
        for version in manifest["versions"]:
            if not isinstance(version, dict) or _id(version.get("id")) in version_ids:
                raise ValueError("Duplicate or invalid voice version")
            version_ids.add(version["id"])
            if _id(version.get("voice_id")) not in voice_ids:
                raise ValueError("Version references a missing voice")
            reference = f"versions/{version['id']}/reference.safetensors"
            if version.get("session_path") != reference or reference not in files or version.get("content_sha256") != files[reference]["sha256"]:
                raise ValueError("Invalid native reference identity")
            used.add(reference)
            source = version.get("source_path")
            if source is not None:
                _member(source)
                if PurePosixPath(source).parent.as_posix() != f"versions/{version['id']}" or PurePosixPath(source).stem != "source" or PurePosixPath(source).suffix.lower() not in AUDIO_EXTENSIONS:
                    raise ValueError("Invalid original audio path")
                if source not in files or version.get("source_sha256") != files[source]["sha256"]:
                    raise ValueError("Invalid original audio identity")
                used.add(source)
            elif version.get("source_sha256") is not None:
                raise ValueError("Source checksum has no source")
            # Old schema-1 bundles have no distinct preview fields.
            if "preview_path" not in version:
                version["preview_path"], version["preview_sha256"] = source, version.get("source_sha256")
            preview = version.get("preview_path")
            if preview is not None:
                _member(preview)
                if preview != source and (PurePosixPath(preview).parent.as_posix() != f"versions/{version['id']}" or PurePosixPath(preview).stem != "preview"):
                    raise ValueError("Invalid preview audio path")
                if PurePosixPath(preview).suffix.lower() not in AUDIO_EXTENSIONS or preview not in files or version.get("preview_sha256") != files[preview]["sha256"]:
                    raise ValueError("Invalid preview audio identity")
                used.add(preview)
            elif version.get("preview_sha256") is not None:
                raise ValueError("Preview checksum has no preview")
            _text(version.get("transcript"), "transcript", 100000)
            _text(version.get("language"), "language", 80)
            _text(version.get("delivery"), "delivery", 160)
            _json(version.get("defaults"))
            _timestamp(version.get("created_at"))
        if used != set(files):
            raise ValueError("Unreferenced bundle artifact")
        versions = {item["id"]: item for item in manifest["versions"]}
        cast_ids = set()
        for cast in manifest["casts"]:
            if not isinstance(cast, dict) or _id(cast.get("id")) in cast_ids:
                raise ValueError("Duplicate or invalid cast")
            cast_ids.add(cast["id"])
            _text(cast.get("name"), "cast name", 120, False)
            _timestamp(cast.get("created_at"))
            _json(cast.get("defaults"))
            bindings = cast.get("bindings")
            if not isinstance(bindings, dict) or not bindings or len(bindings) > 64:
                raise ValueError("Invalid cast bindings")
            canonical_roles = set()
            for role, binding in bindings.items():
                if _text(role, "role", 120, False) != role or role in canonical_roles or not isinstance(binding, dict):
                    raise ValueError("Invalid cast role")
                canonical_roles.add(role)
                version = versions.get(_id(binding.get("version_id")))
                if not version or version["voice_id"] != _id(binding.get("voice_id")):
                    raise ValueError("Cast references a missing or mismatched voice version")
                if type(binding.get("speaker")) is not int or not 0 <= binding["speaker"] < 16:
                    raise ValueError("Invalid cast speaker")
                for field in ("defaults", "voice_defaults"):
                    _json(binding.get(field))
                for field in ("voice_name", "language", "delivery"):
                    _text(binding.get(field), field, 160)
                groups = binding.get("reference_groups", {})
                if not isinstance(groups, dict) or len(groups) > 32:
                    raise ValueError("Invalid reference groups")
                for name, ids in groups.items():
                    if _text(name, "reference group name", 80, False) != name or not isinstance(ids, list) or not ids or len(ids) > 64:
                        raise ValueError("Invalid or empty reference group")
                    normalized = [_id(item) for item in ids]
                    if len(set(normalized)) != len(normalized):
                        raise ValueError("Duplicate reference group version")
                    if any(item not in versions or versions[item]["voice_id"] != binding["voice_id"] for item in normalized):
                        raise ValueError("Reference group references a missing or foreign voice version")
        return manifest, files

    def import_bundle(self, path, *, conflict="rename"):
        """Validate the entire bundle before one catalog transaction.

        Existing immutable UUIDs must agree. Name collisions can be renamed,
        skipped or rejected; existing versions are never replaced.
        """
        if conflict not in {"rename", "skip", "error"}:
            raise ValueError("Conflict mode must be rename, skip or error")
        result = {"voices": [], "versions": [], "casts": [], "skipped_voices": [], "skipped_casts": []}
        created_folders = []
        with _read_archive(path) as archive:
            manifest, files = self._bundle_manifest(archive)
            if shutil.disk_usage(self.root).free < 2 * sum(record["bytes"] for record in files.values()) + MIN_FREE_BYTES:
                raise ValueError("Insufficient free disk space to import a voice bundle")
            with tempfile.TemporaryDirectory(prefix=".voice-import-", dir=self.root) as staging:
                staging = Path(staging)
                for relative, record in files.items():
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(relative) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
                    if target.stat().st_size != record["bytes"] or _sha(target) != record["sha256"]:
                        raise ValueError("Bundle artifact checksum mismatch")
                # Validate native structure and exact encoded transcript before catalog changes.
                native_multispeaker = {}
                for version in manifest["versions"]:
                    session = _native_session(staging / version["session_path"])
                    if "\n".join(reference.transcript for reference in session.references) != version["transcript"]:
                        raise ValueError("Bundle transcript does not match encoded references")
                    native_multispeaker[version["id"]] = any(len(set(re.findall(r"<\|speaker:(\d+)\|>", ref.transcript))) > 1
                                                             for ref in session.references)
                for cast in manifest["casts"]:
                    for binding in cast["bindings"].values():
                        ids = [binding["version_id"], *(item for group in binding.get("reference_groups", {}).values() for item in group)]
                        if any(native_multispeaker[item] for item in ids):
                            raise ValueError("Cast cannot remap a multi-speaker reference")
                try:
                    with self._db() as db:
                        db.execute("BEGIN IMMEDIATE")
                        names = {row[0].casefold() for row in db.execute("SELECT name FROM voices")}
                        skipped = set()
                        for voice in manifest["voices"]:
                            existing = db.execute("SELECT id FROM voices WHERE id=?", (voice["id"],)).fetchone()
                            if existing:
                                result["voices"].append(voice["id"])
                                continue
                            name = voice["name"]
                            if name.casefold() in names:
                                if conflict == "error":
                                    raise ValueError("A different voice already uses this name")
                                if conflict == "skip":
                                    skipped.add(voice["id"])
                                    result["skipped_voices"].append(voice["id"])
                                    continue
                                suffix, stem = 2, name[:108]
                                while f"{stem} ({suffix})".casefold() in names:
                                    suffix += 1
                                name = f"{stem} ({suffix})"
                            names.add(name.casefold())
                            db.execute("INSERT INTO voices VALUES(?,?,?,?,?,?,?,?,NULL)",
                                       (voice["id"], name, json.dumps(voice["tags"], ensure_ascii=False), int(voice["favorite"]),
                                        voice["notes"], int(voice["archived"]), voice["created_at"], voice["updated_at"]))
                            result["voices"].append(voice["id"])
                        for version in manifest["versions"]:
                            if version["voice_id"] in skipped:
                                continue
                            old = db.execute("SELECT * FROM versions WHERE id=?", (version["id"],)).fetchone()
                            if old:
                                old = self._version(old)
                                keys = ("voice_id", "content_sha256", "transcript", "language", "delivery", "defaults")
                                if any(old[key] != version[key] for key in keys) or (version["source_sha256"] is not None and old["source_sha256"] != version["source_sha256"]):
                                    raise ValueError("Immutable voice version UUID conflict")
                                if version.get("preview_sha256") is not None and old["preview_sha256"] != version["preview_sha256"]:
                                    raise ValueError("Immutable voice preview UUID conflict")
                                self._file(old["session_path"], old["content_sha256"])
                                if old["source_path"]:
                                    self._file(old["source_path"], old["source_sha256"])
                                if old["preview_path"]:
                                    self._file(old["preview_path"], old["preview_sha256"])
                            else:
                                folder = _resolved(self.root / "versions" / version["id"])
                                if not folder.is_relative_to(self.root) or folder.exists():
                                    raise ValueError("Version artifact path already exists or is unsafe")
                                folder.mkdir(parents=True, exist_ok=False)
                                created_folders.append(folder)
                                shutil.copyfile(staging / version["session_path"], folder / "reference.safetensors")
                                if version["source_path"]:
                                    shutil.copyfile(staging / version["source_path"], folder / PurePosixPath(version["source_path"]).name)
                                if version["preview_path"] and version["preview_path"] != version["source_path"]:
                                    shutil.copyfile(staging / version["preview_path"], folder / PurePosixPath(version["preview_path"]).name)
                                db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                           (version["id"], version["voice_id"], version["session_path"], version["content_sha256"],
                                            version["source_path"], version["source_sha256"], version["transcript"], version["language"],
                                            version["delivery"], json.dumps(version["defaults"], ensure_ascii=False, allow_nan=False, sort_keys=True), version["created_at"],
                                            version["preview_path"], version["preview_sha256"]))
                            result["versions"].append(version["id"])
                        for cast in manifest["casts"]:
                            if any(binding["voice_id"] in skipped for binding in cast["bindings"].values()):
                                result["skipped_casts"].append(cast["id"])
                                continue
                            snapshot = {"bindings": cast["bindings"], "defaults": cast["defaults"]}
                            encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True)
                            old = db.execute("SELECT * FROM casts WHERE id=?", (cast["id"],)).fetchone()
                            if old:
                                if old["name"] != cast["name"] or json.loads(old["snapshot"]) != snapshot:
                                    raise ValueError("Immutable cast UUID conflict")
                            else:
                                db.execute("INSERT INTO casts VALUES(?,?,?,?)", (cast["id"], cast["name"], encoded, cast["created_at"]))
                            result["casts"].append(cast["id"])
                except BaseException:
                    # Only paths created by this failed import are rolled back.
                    for folder in reversed(created_folders):
                        if folder.is_relative_to(self.root) and folder.parent == self.root / "versions":
                            shutil.rmtree(folder)
                    raise
        return result


def _timestamp(value):
    import math
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid voice library timestamp")
    return value


@contextmanager
def _read_archive(path):
    try:
        with zipfile.ZipFile(_resolved(path)) as archive:
            yield archive
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("Invalid or corrupted voice bundle archive") from exc


