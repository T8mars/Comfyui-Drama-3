"""Real workspace ownership must end even when explicit unlock fails."""
import os
from pathlib import Path

import pytest

from drama3_t8.workspace_lock import WorkspaceBusy, WorkspaceLock


def test_unlock_failure_closes_handle_and_releases_actual_workspace(tmp_path, monkeypatch):
    path = tmp_path / "studio.lock"
    owner = WorkspaceLock(path)
    with pytest.raises(WorkspaceBusy):
        WorkspaceLock(path)
    if os.name == "nt":
        import msvcrt as locks
        method, unlock = "locking", locks.LK_UNLCK
    else:
        import fcntl as locks
        method, unlock = "flock", locks.LOCK_UN
    original = getattr(locks, method)
    failure = OSError("explicit unlock failed")

    def fail_unlock(handle, operation, *rest):
        if operation == unlock:
            raise failure
        return original(handle, operation, *rest)

    try:
        monkeypatch.setattr(locks, method, fail_unlock)
        with pytest.raises(OSError) as raised:
            owner.close()
        assert raised.value is failure
        assert owner.file.closed, "unlock errors must not retain the lock handle"
        owner.close()
        monkeypatch.setattr(locks, method, original)
        replacement = WorkspaceLock(path)
        replacement.close()
    finally:
        monkeypatch.setattr(locks, method, original)
        owner.close()


def test_normal_close_is_idempotent_and_releases_actual_workspace(tmp_path):
    path = tmp_path / "studio.lock"
    owner = WorkspaceLock(path)
    owner.close()
    owner.close()
    assert owner.file.closed
    replacement = WorkspaceLock(path)
    replacement.close()


@pytest.mark.parametrize("operation", ["seek", "write", "flush"])
def test_initialization_io_failure_closes_file_and_preserves_error(tmp_path, monkeypatch, operation):
    path = tmp_path / "studio.lock"
    original = Path.open
    handles = []
    failure = OSError("initial " + operation + " failed")

    class FaultyFile:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            if name == operation:
                def fail(*args, **kwargs):
                    raise failure
                return fail
            return getattr(self.stream, name)

    def opened(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == path:
            handles.append(stream)
            return FaultyFile(stream)
        return stream

    monkeypatch.setattr(Path, "open", opened)
    try:
        with pytest.raises(OSError) as raised:
            WorkspaceLock(path)
        assert raised.value is failure
        assert len(handles) == 1 and handles[0].closed
        monkeypatch.setattr(Path, "open", original)
        replacement = WorkspaceLock(path)
        replacement.close()
    finally:
        for stream in handles:
            stream.close()
