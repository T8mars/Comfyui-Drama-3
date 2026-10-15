import json
import shutil
import threading

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine

standalone_engine()
from drama3_t8.studio import Studio
from drama3_t8.state import Session, Reference, save_session
from drama3_t8.workspace_lock import WorkspaceBusy


def test_second_workspace_cannot_interrupt_active_queue(tmp_path):
    entered, release = threading.Event(), threading.Event()
    with Studio(tmp_path) as first:
        original = first.execute
        def execute(job_id, payload, progress):
            if payload["mode"] == "hold":
                entered.set()
                assert release.wait(10)
                return {"ok": True}
            return original(job_id, payload, progress)
        first.execute = execute
        job = first.submit({"mode": "hold"})
        try:
            assert entered.wait(10)
            with pytest.raises(WorkspaceBusy):
                Studio(tmp_path)
            assert first.jobs()[0]["status"] == "running"
        finally:
            release.set()
        first.pending.join()
        assert next(j for j in first.jobs() if j["id"] == job)["status"] == "done"
    with Studio(tmp_path) as restarted:
        assert restarted.jobs()[0]["status"] == "done"


def test_initialization_error_releases_owner_lock(tmp_path):
    data = tmp_path / "userdata"
    data.mkdir()
    settings = data / "runtime-settings.json"
    settings.write_text("broken")
    with pytest.raises(json.JSONDecodeError):
        Studio(tmp_path)
    settings.write_text("{}")
    with Studio(tmp_path) as studio:
        assert studio.worker.is_alive()


def test_queued_and_late_cancellation_do_not_cancel_next_job(tmp_path, monkeypatch):
    import comfy.model_management as mm
    entered, release = threading.Event(), threading.Event()
    interrupts = []
    monkeypatch.setattr(mm, "interrupt_current_processing", interrupts.append)
    with Studio(tmp_path) as studio:
        original = studio.execute
        called = []
        def execute(job_id, payload, progress):
            called.append(payload["mode"])
            if payload["mode"] == "hold":
                entered.set()
                assert release.wait(10)
                return {"ok": True}
            return original(job_id, payload, progress)
        studio.execute = execute
        running = studio.submit({"mode": "hold"})
        assert entered.wait(10)
        queued = studio.submit({"mode": "never_execute"})
        studio.cancel(queued)
        assert True not in interrupts
        studio.cancel(running)
        recovered = studio.submit({"mode": "unload"})
        release.set()
        studio.pending.join()
        jobs = {j["id"]: j for j in studio.jobs()}
        assert jobs[running]["status"] == jobs[queued]["status"] == "cancelled"
        assert jobs[recovered]["status"] == "done"
        assert "never_execute" not in called
        with pytest.raises(ValueError):
            studio.cancel(running)


def test_shutdown_stops_worker_and_refuses_new_jobs(tmp_path):
    studio = Studio(tmp_path)
    studio.close()
    assert not studio.worker.is_alive()
    with pytest.raises(RuntimeError, match="关闭"):
        studio.submit({"mode": "unload"})
    studio.close()


def test_cancel_after_result_is_finalized_is_rejected_instead_of_false_success(tmp_path, monkeypatch):
    before_update, release = threading.Event(), threading.Event()
    with Studio(tmp_path) as studio:
        original_connect = studio.connect
        class PausedConnection:
            def __init__(self):
                self.connection = original_connect()
            def __enter__(self):
                self.connection.__enter__()
                return self
            def __exit__(self, *args):
                return self.connection.__exit__(*args)
            def execute(self, sql, params=()):
                if sql.startswith("UPDATE jobs SET status=?,result="):
                    before_update.set()
                    assert release.wait(10)
                return self.connection.execute(sql, params)
        monkeypatch.setattr(studio, "connect", PausedConnection)
        job = studio.submit({"mode": "unload"})
        try:
            assert before_update.wait(10)
            with pytest.raises(ValueError, match="no longer active"):
                studio.cancel(job)
        finally:
            release.set()
        studio.pending.join()
        assert studio.jobs()[0]["status"] == "done"


def test_restart_recovers_abandoned_jobs_only_after_owner_is_gone(tmp_path):
    with Studio(tmp_path) as studio:
        with studio.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",
                       ("a" * 32, "running", "{}", None, None, 1, 1, 0))
    with Studio(tmp_path) as restored:
        assert restored.jobs()[0]["status"] == "interrupted"


def test_moved_workspace_reads_its_own_sessions_and_references(tmp_path):
    old, new = tmp_path / "before", tmp_path / "after move"
    reference = Reference("Hello", torch.ones((10, 3), dtype=torch.long), "Voice")
    session = Session((reference,))
    with Studio(old) as studio:
        studio.register_session(session)
        reference_id = "b" * 32
        reference_path = studio.data / "references" / f"{reference_id}.safetensors"
        save_session(session, reference_path)
        # Legacy 0.1.1 database rows used absolute paths.
        with studio.connect() as db:
            db.execute("UPDATE sessions SET path=?", (str(studio.data / "sessions" / f"{session.session_id}.safetensors"),))
            db.execute("INSERT INTO voice_references VALUES(?,?,?,?)", (reference_id, "Voice", "Hello", str(reference_path)))
    shutil.copytree(old / "userdata", new / "userdata", ignore=shutil.ignore_patterns("studio.lock"))
    reference_path.write_bytes(b"old copy no longer usable")
    (old / "userdata/sessions" / f"{session.session_id}.safetensors").write_bytes(b"old copy no longer usable")
    with Studio(new) as restored:
        assert restored.get_session(session.session_id).references[0].transcript == "Hello"
        assert restored.session_for({"reference_ids": [reference_id]}).references[0].name == "Voice"
        with restored.connect() as db:
            assert db.execute("SELECT path FROM sessions").fetchone()[0].startswith("sessions/")
            assert db.execute("SELECT path FROM voice_references").fetchone()[0].startswith("references/")
