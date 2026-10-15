"""Independent cross-review of the main agent's result publication changes."""
from pathlib import Path

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import drama3_t8.studio as module
from drama3_t8.state import Session, Take
from drama3_t8.studio import Studio, result_files


def take():
    values = torch.ones(3, 11, dtype=torch.long)
    values[:, 0] = 248100
    return Take("Hello", "Hello", values, metadata={"elapsed_seconds": 0, "length_limit_reached": False})


def test_existing_last_result_path_blocks_all_publication_without_touching_it(tmp_path):
    first, existing = tmp_path / "new.wav", tmp_path / "existing.safetensors"
    existing.write_bytes(b"prior user archive")
    with pytest.raises(FileExistsError):
        with result_files(first, existing):
            pytest.fail("pre-existing last path entered a publication block")
    assert not first.exists() and existing.read_bytes() == b"prior user archive"


def test_sqlite_commit_failure_rolls_back_new_files_and_keeps_prior_project(tmp_path, monkeypatch):
    with Studio(tmp_path) as studio:
        original = Session(takes=(take(),))
        studio.register_session(original)
        prior = studio.data / "sessions" / (original.session_id + ".safetensors")
        original_bytes = prior.read_bytes()
        monkeypatch.setattr(studio, "generate_session", lambda mode, payload, session, *a: session.append(take()))
        monkeypatch.setattr(studio.engine, "decode", lambda *a: {"waveform": torch.zeros(1, 1, 20), "sample_rate": 44100})
        real_connect = studio.connect
        class Connection:
            def __init__(self):
                self.db = real_connect()
                self.new_session = False
            def __enter__(self):
                self.db.__enter__()
                return self
            def execute(self, sql, parameters=()):
                self.new_session = self.new_session or sql.startswith("INSERT INTO sessions")
                return self.db.execute(sql, parameters)
            def __exit__(self, kind, value, tb):
                if kind is None and self.new_session:
                    fault = OSError("commit unavailable")
                    self.db.__exit__(OSError, fault, None)
                    raise fault
                return self.db.__exit__(kind, value, tb)
        monkeypatch.setattr(studio, "connect", Connection)
        failed = studio.submit({"mode": "tts", "text": "Test", "session_id": original.session_id})
        recovered = studio.submit({"mode": "unload"})
        studio.pending.join()
        assert studio.get_job(failed)["status"] == "failed"
        assert "commit unavailable" in studio.get_job(failed)["error"]
        assert studio.get_job(recovered)["status"] == "done"
        assert prior.read_bytes() == original_bytes
        assert list((studio.data / "sessions").iterdir()) == [prior]
        assert len(studio.sessions()) == 1
        assert not [p for p in studio.output.rglob("*") if p.is_file()]


def test_failed_rollback_names_remaining_path_and_preserves_original_cause(tmp_path, monkeypatch):
    target = tmp_path / "new-result.wav"
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if path == target:
            raise PermissionError("file scanner holds result")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    fault = OSError("disk write failed")
    with pytest.raises(RuntimeError, match="部分结果文件未能清理") as caught:
        with result_files(target):
            target.write_bytes(b"partial")
            raise fault
    assert str(target) in str(caught.value) and caught.value.__cause__ is fault
