"""Queue persistence fault contracts; no model or inference is required.

Prepared separately from canonical source. Run only after adopting the matching
Studio/app candidates and the usual real CPU host initialization.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from studio_app_loader import application

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.studio import Studio


def wait_drained(studio):
    deadline = time.monotonic() + 10
    while studio.pending.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(.01)
    assert studio.pending.unfinished_tasks == 0
    assert studio.worker.is_alive(), "storage faults must not kill the worker"


def close_bounded(studio):
    errors = []
    def close():
        try:
            studio.close()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "close must consume its sentinel and join"
    assert not studio.worker.is_alive()
    assert studio.pending.unfinished_tasks == 0
    assert errors == []


class TerminalFaultConnection:
    def __init__(self, connection, fault, reached=None, release=None):
        self.connection, self.fault = connection, fault
        self.reached, self.release = reached, release
        self.terminal = False

    def __enter__(self):
        self.connection.__enter__()
        return self

    def execute(self, sql, *args):
        if sql.startswith("UPDATE jobs SET status=?,result="):
            self.terminal = True
            if self.reached is not None:
                self.reached.set()
                assert self.release.wait(10)
            if self.fault == "execute":
                raise sqlite3.OperationalError("terminal execute fault")
        return self.connection.execute(sql, *args)

    def __exit__(self, typ, value, tb):
        try:
            if self.terminal and typ is None and self.fault == "commit_before":
                error = OSError("terminal commit-before fault")
                self.connection.__exit__(type(error), error, error.__traceback__)
                raise error
            result = self.connection.__exit__(typ, value, tb)
            if self.terminal and typ is None and self.fault == "commit_after":
                raise OSError("terminal ambiguous commit-after fault")
            return result
        finally:
            self.connection.close()


def install_terminal_fault(studio, monkeypatch, fault, reached=None, release=None):
    original = studio.connect
    def connect():
        if fault == "connect":
            raise OSError("terminal connect fault; database unavailable")
        return TerminalFaultConnection(original(), fault, reached, release)
    monkeypatch.setattr(studio, "connect", connect)
    return original


@pytest.mark.parametrize("fault", ["connect", "execute", "commit_before", "commit_after"])
def test_terminal_storage_fault_retains_result_and_never_reruns(tmp_path, monkeypatch, fault):
    entered, release = threading.Event(), threading.Event()
    calls, unloads = [], []
    studio = Studio(tmp_path)
    artifact = tmp_path / "committed-result.bin"
    def execute(job_id, payload, progress):
        calls.append(job_id)
        studio.begin_commit(job_id, progress)
        artifact.write_bytes(b"committed-once")
        entered.set()
        assert release.wait(10)
        return {"audio": artifact.name, "nested": {"value": [1, 2]}}
    monkeypatch.setattr(studio, "execute", execute)
    monkeypatch.setattr(studio.engine, "unload", lambda: unloads.append("normal"))
    try:
        first = studio.submit({"mode": "tts", "text": "first"})
        assert entered.wait(10)
        second = studio.submit({"mode": "tts", "text": "never execute"})
        created = {row["id"]: row["created"] for row in studio.jobs()}
        original = install_terminal_fault(studio, monkeypatch, fault)
        release.set()
        wait_drained(studio)
        first_row, second_row = studio.get_job(first), studio.get_job(second)
        assert calls == [first]
        assert first_row["status"] == second_row["status"] == "failed"
        assert first_row["execution_status"] == "done"
        assert second_row["execution_status"] == "not_started"
        assert first_row["result"]["audio"] == artifact.name
        assert second_row["result"] is None
        assert "执行结果已保留" in first_row["error"]
        assert "任务未执行" in second_row["error"]
        assert first_row["persistence_error"] == second_row["persistence_error"]
        assert first_row["progress"] == second_row["progress"] == 0
        assert not first_row["cancelable"] and not second_row["cancelable"]
        assert artifact.read_bytes() == b"committed-once"
        assert {row["id"]: row["created"] for row in studio.jobs()} == created
        assert unloads == ["normal"]  # fault path releases only after execute returned
        with pytest.raises(RuntimeError, match="暂停接单"):
            studio.submit({"mode": "unload"})
        # A reader must never get the mutable cache object itself.
        first_row["result"]["nested"]["value"].append(99)
        first_row["payload"]["text"] = "changed outside Studio"
        assert studio.get_job(first)["result"]["nested"]["value"] == [1, 2]
        assert studio.get_job(first)["payload"]["text"] == "first"
        # Even a complete DB outage cannot hide this round's fallback terminals.
        monkeypatch.setattr(studio, "connect", lambda: (_ for _ in ()).throw(OSError("all reads unavailable")))
        assert {row["id"]: row["status"] for row in studio.jobs()} == {first: "failed", second: "failed"}
        assert studio.get_job(first)["result"]["audio"] == artifact.name
        if fault == "commit_after":
            with original() as db:
                assert db.execute("SELECT status FROM jobs WHERE id=?", (first,)).fetchone()[0] == "done"
            # Ambiguous/failed persistence is still not reported as durable done.
            assert studio.get_job(first)["status"] == "failed"
    finally:
        release.set()
        close_bounded(studio)
    assert unloads == ["normal", "normal"]  # unchanged normal close owns final cleanup


@pytest.mark.parametrize("close_first", [False, True])
def test_fault_close_sentinel_race_balances_every_fetched_item(tmp_path, monkeypatch, close_first):
    entered, release = threading.Event(), threading.Event()
    terminal, finish_terminal = threading.Event(), threading.Event()
    studio = Studio(tmp_path)
    calls = []
    def execute(job_id, payload, progress):
        calls.append(job_id)
        studio.begin_commit(job_id, progress)
        entered.set()
        assert release.wait(10)
        return {"ok": True}
    monkeypatch.setattr(studio, "execute", execute)
    try:
        first = studio.submit({"mode": "tts"})
        assert entered.wait(10)
        second = studio.submit({"mode": "tts"})
        install_terminal_fault(studio, monkeypatch, "execute", terminal, finish_terminal)
        release.set()
        assert terminal.wait(10)
        close_errors = []
        def close():
            try:
                studio.close()
            except BaseException as exc:
                close_errors.append(exc)
        closer = None
        if close_first:
            closer = threading.Thread(target=close, daemon=True)
            closer.start()
            deadline = time.monotonic() + 10
            while not studio._closed and time.monotonic() < deadline:
                time.sleep(.01)
            assert studio._closed
        finish_terminal.set()
        if close_first:
            closer.join(10)
            assert not closer.is_alive()
            assert close_errors == []
        else:
            wait_drained(studio)
            close_bounded(studio)
        assert calls == [first]
        assert not studio.worker.is_alive()
        assert studio.pending.unfinished_tasks == 0
        assert studio.get_job(first)["status"] == "failed"
        assert studio.get_job(second)["status"] in {"failed", "cancelled"}
        assert studio.get_job(second)["execution_status"] == "not_started"
        # close must release ownership even though queue_error was set.
        with Studio(tmp_path):
            pass
    finally:
        release.set()
        finish_terminal.set()
        if not studio._closed:
            close_bounded(studio)


@pytest.mark.parametrize("fault", ["execute", "commit_after"])
def test_submit_storage_fault_does_not_unload_active_generation(tmp_path, monkeypatch, fault):
    entered, release = threading.Event(), threading.Event()
    studio = Studio(tmp_path)
    calls, unloads = [], []
    def execute(job_id, payload, progress):
        calls.append(job_id)
        entered.set()
        assert release.wait(10)
        return {"ok": True}
    monkeypatch.setattr(studio, "execute", execute)
    monkeypatch.setattr(studio.engine, "unload", lambda: unloads.append(studio._active_job))
    original = studio.connect
    class InsertFault(TerminalFaultConnection):
        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO jobs VALUES"):
                self.terminal = True
                if self.fault == "execute":
                    raise sqlite3.OperationalError("submit insert fault")
            return self.connection.execute(sql, *args)
    try:
        first = studio.submit({"mode": "tts"})
        assert entered.wait(10)
        second = studio.submit({"mode": "never"})
        monkeypatch.setattr(studio, "connect", lambda: InsertFault(original(), fault))
        with pytest.raises(RuntimeError) as rejected:
            studio.submit({"mode": "rejected"})
        assert rejected.value.__cause__ is not None
        assert calls == [first]
        assert studio._active_job == first
        assert unloads == []
        release.set()
        wait_drained(studio)
        rows = {row["id"]: row for row in studio.jobs()}
        assert len(rows) == 3
        assert rows[first]["status"] == "failed"
        assert rows[first]["result"] == {"ok": True}
        assert rows[second]["execution_status"] == "not_started"
        assert all(row["status"] == "failed" for row in rows.values())
        assert calls == [first]
        assert unloads == [None]
    finally:
        release.set()
        close_bounded(studio)


def test_generation_error_and_failed_cleanup_do_not_mask_first_error(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    studio = Studio(tmp_path)
    def execute(job_id, payload, progress):
        entered.set()
        assert release.wait(10)
        error = RuntimeError("original generation failure")
        error.generation_attempts = [{"nonfinite": float("nan")}]
        raise error
    monkeypatch.setattr(studio, "execute", execute)
    unloads = []
    def unload():
        unloads.append(studio._active_job)
        if len(unloads) == 1:
            raise RuntimeError("normal idle unload rejected")
    monkeypatch.setattr(studio.engine, "unload", unload)
    try:
        first = studio.submit({"mode": "tts"})
        assert entered.wait(10)
        install_terminal_fault(studio, monkeypatch, "execute")
        release.set()
        wait_drained(studio)
        row = studio.get_job(first)
        assert row["status"] == "failed" and row["result"] is None
        assert "original generation failure" in row["error"]
        assert "错误日志未能保存" in row["error"]  # allow_nan=False raised during logging
        assert "terminal execute fault" in row["error"]
        assert "normal idle unload rejected" in row["error"]
        assert row["resource_cleanup_error"]
        assert unloads == [None]
        assert row["execution_status"] == "failed"
        assert all(not isinstance(value, BaseException) for value in studio._job_records[first].values())
        assert not any(isinstance(value, BaseException) for value in vars(studio).values())
    finally:
        release.set()
        close_bounded(studio)


@pytest.mark.parametrize("log_fault", [OSError("log disk unavailable"), RuntimeError("log secondary fault")])
def test_secondary_log_write_failure_does_not_kill_worker_or_poison_next_job(tmp_path, monkeypatch, log_fault):
    studio = Studio(tmp_path)
    original, original_write = studio.execute, Path.write_text
    def execute(job_id, payload, progress):
        if payload["mode"] == "broken":
            raise RuntimeError("primary generation error")
        return original(job_id, payload, progress)
    def write(path, *args, **kwargs):
        if path.parent == studio.data / "logs":
            raise log_fault
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(studio, "execute", execute)
    monkeypatch.setattr(Path, "write_text", write)
    try:
        bad = studio.submit({"mode": "broken"})
        next_job = studio.submit({"mode": "unload"})
        wait_drained(studio)
        assert studio.get_job(bad)["status"] == "failed"
        assert "primary generation error" in studio.get_job(bad)["error"]
        assert str(log_fault) in studio.get_job(bad)["error"]
        assert studio.get_job(next_job)["status"] == "done"
        assert studio._queue_error is None
    finally:
        close_bounded(studio)


def test_api_queue_unavailable_is_503_with_readable_detail_and_no_new_job(tmp_path, monkeypatch):
    (tmp_path / "web").mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        studio = client.app.state.studio
        monkeypatch.setattr(studio, "connect", lambda: (_ for _ in ()).throw(OSError("submit storage unavailable")))
        response = client.post("/api/jobs", json={"mode": "unload"})
        assert response.status_code == 503
        assert "暂停接单" in response.json()["detail"]
        rows = client.get("/api/jobs").json()
        assert len(rows) == 1 and rows[0]["status"] == "failed"
        assert rows[0]["execution_status"] == "not_started"
        assert rows[0]["result"] is None
        again = client.post("/api/jobs", json={"mode": "unload"})
        assert again.status_code == 503
        assert len(client.get("/api/jobs").json()) == 1
        for route in ("/api/references", "/api/transcribe"):
            upload = client.post(route, files={"file": ("sample.wav", b"not decoded", "audio/wav")})
            assert upload.status_code == 503
            assert "暂停接单" in upload.json()["detail"]
            assert list((studio.data / "inputs").iterdir()) == []
        assert len(client.get("/api/jobs").json()) == 1
        invalid = client.post("/api/jobs", json={"mode": "unsupported"})
        assert invalid.status_code == 400
