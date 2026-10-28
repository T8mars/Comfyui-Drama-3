"""Real Windows target mappings exercise atomic Session commit retry limits."""
from dataclasses import replace
import json
import mmap
import os
from pathlib import Path
import threading
import time

import pytest
import torch

from drama3_t8.state import Reference, Session, Take, load_session, save_session


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows mapping/rename contract")


def _locked_save(tmp_path, monkeypatch, request, *, transient):
    codes = torch.ones(10, 3, dtype=torch.long)
    frames = torch.cat((codes[:1] + 248099, codes)).T.contiguous()
    original_session = Session(
        references=(Reference("Reference.", codes[:, :2].clone()),),
        takes=(Take("Hello.", "Hello.", frames),), direction="original")
    updated_session = replace(original_session, direction="updated")
    target = tmp_path / "locked.safetensors"
    save_session(original_session, target)
    original_bytes = target.read_bytes()
    with target.open("rb") as file:
        holder = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    # The mapping itself owns the Windows lock; the outer file is already closed.
    trace = {"transient": transient, "outer_file_closed": file.closed,
             "release_delay_seconds": .250 if transient else None,
             "attempts": [], "thread_errors": []}
    started = threading.Event()
    released = threading.Event()
    abort = threading.Event()
    original_replace = Path.replace

    def close_holder():
        trace["close_started"] = time.perf_counter()
        try:
            holder.close()
            trace["closed"] = time.perf_counter()
        finally:
            released.set()

    def release_later():
        try:
            while not started.wait(.01):
                if abort.is_set():
                    return
            remaining = trace["first_attempt"] + .250 - time.perf_counter()
            if not abort.wait(max(0, remaining)):
                close_holder()
        except BaseException as error:
            trace["thread_errors"].append(repr(error))

    worker = threading.Thread(target=release_later, name="session-own-mmap-release") if transient else None

    def observe_replace(source, destination):
        if source.parent != target.parent or Path(destination) != target:
            return original_replace(source, destination)
        entry = {"started": time.perf_counter(), "source": str(source)}
        if not started.is_set():
            trace["first_attempt"] = entry["started"]
            started.set()
        try:
            result = original_replace(source, destination)
        except OSError as error:
            entry.update(ended=time.perf_counter(), winerror=getattr(error, "winerror", None),
                         errno=error.errno, exception_id=id(error), error_type=type(error).__name__)
            try:
                entry["old_target_unchanged"] = target.read_bytes() == original_bytes
            except OSError as observation_error:
                entry["observation_error"] = repr(observation_error)
            trace["attempts"].append(entry)
            raise  # Preserve the very same real OS exception and traceback.
        else:
            entry.update(ended=time.perf_counter(), succeeded=True)
            trace["attempts"].append(entry)
            return result

    failure = None
    monkeypatch.setattr(Path, "replace", observe_replace)
    if worker is not None:
        worker.start()
    try:
        save_session(updated_session, target)
    except BaseException as error:
        failure = error
        trace["save_error"] = {"type": type(error).__name__, "winerror": getattr(error, "winerror", None),
                               "errno": getattr(error, "errno", None), "exception_id": id(error),
                               "observed": time.perf_counter()}
    finally:
        # Let an actually started transient release reach its fixed 250ms deadline,
        # including when the original short retry window has already failed.
        if worker is not None and started.is_set():
            worker.join(timeout=2)
        abort.set()
        if worker is not None:
            worker.join(timeout=2)
        if not released.is_set():
            close_holder()
        trace["worker_alive_after_cleanup"] = worker.is_alive() if worker is not None else False
        trace["mapping_closed_after_cleanup"] = holder.closed
        trace["part_files_after_save"] = [p.name for p in tmp_path.glob(target.name + ".*.part")]
        trace["target_is_original_after_save"] = target.read_bytes() == original_bytes
        request.node.user_properties.append(("actual_commit_trace", json.dumps(trace, sort_keys=True)))

    assert trace["outer_file_closed"] and trace["mapping_closed_after_cleanup"]
    assert not trace["worker_alive_after_cleanup"] and not trace["thread_errors"]
    assert not trace["part_files_after_save"]
    failures = [item for item in trace["attempts"] if not item.get("succeeded")]
    assert failures and all(item["winerror"] == 5 and item["error_type"] == "PermissionError" for item in failures)
    assert all(item["old_target_unchanged"] for item in failures)
    assert len(trace["attempts"]) <= 6
    if failure is not None:
        assert id(failure) == failures[-1]["exception_id"]
    return failure, trace, target, original_session, updated_session


def test_transient_target_mapping_released_after_250ms_commits(tmp_path, monkeypatch, request):
    error, trace, path, original, updated = _locked_save(tmp_path, monkeypatch, request, transient=True)
    if error is not None:
        raise error
    assert trace["attempts"][-1]["succeeded"]
    assert trace["closed"] >= trace["first_attempt"] + .250
    loaded = load_session(path)
    assert loaded.direction == updated.direction
    assert loaded.takes[0].take_id == original.takes[0].take_id
    assert torch.equal(loaded.takes[0].frames, original.takes[0].frames)
    assert torch.equal(loaded.references[0].codes, original.references[0].codes)


def test_permanent_target_mapping_keeps_original_and_stops_after_six_attempts(tmp_path, monkeypatch, request):
    error, trace, path, original, _ = _locked_save(tmp_path, monkeypatch, request, transient=False)
    assert isinstance(error, PermissionError) and error.winerror == 5
    assert len(trace["attempts"]) == 6
    assert all(item["ended"] < trace["close_started"] for item in trace["attempts"])
    assert trace["target_is_original_after_save"]
    loaded = load_session(path)
    assert loaded.direction == original.direction
    assert torch.equal(loaded.takes[0].frames, original.takes[0].frames)
    assert torch.equal(loaded.references[0].codes, original.references[0].codes)
