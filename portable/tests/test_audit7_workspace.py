"""Failure boundaries use real SQLite transactions and small real archives."""
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from studio_app_loader import application
from drama3_t8.studio import Studio
from drama3_t8.state import Reference, Session, Take, save_session
from drama3_t8.engine import Sampling


def recording(text='Original'):
    frames = torch.ones((3, 11), dtype=torch.long)
    frames[:, 0] = 248100
    return Take(text, text, frames)


@pytest.mark.parametrize('mode', ['scene', 'dialogue'])
def test_scene_queue_forwards_the_real_session_context(tmp_path, monkeypatch, mode):
    with Studio(tmp_path) as studio:
        source = Session((Reference('Voice.', torch.ones(10, 3, dtype=torch.long)),),
                         (recording(),), direction='Speak softly.')
        seen = []
        def scene(*args, **kwargs):
            seen.append(kwargs)
            return source.append(recording('New'))
        monkeypatch.setattr(studio.engine, 'scene', scene)
        result = studio.generate_session(mode, {'text':'A: New'}, source, Sampling(), lambda: None)
        assert seen[0].get('session') is source
        assert len(result.takes) == 2


def reject_inserts(studio, table):
    with studio.connect() as db:
        db.execute(f"CREATE TRIGGER reject_insert BEFORE INSERT ON {table} "
                   "BEGIN SELECT RAISE(ABORT, 'registration unavailable'); END")


def test_new_project_database_failure_does_not_leave_unregistered_archive(tmp_path):
    with Studio(tmp_path) as studio:
        reject_inserts(studio, 'sessions')
        source = Session(takes=(recording(),))
        with pytest.raises(sqlite3.IntegrityError, match='registration unavailable'):
            studio.register_session(source)
        assert not list((studio.data/'sessions').iterdir())
        assert studio.sessions() == []


def test_reusing_a_registered_identity_preserves_its_existing_archive(tmp_path):
    from dataclasses import replace
    with Studio(tmp_path) as studio:
        original = Session(takes=(recording(),))
        studio.register_session(original)
        path = studio.data/'sessions'/f'{original.session_id}.safetensors'
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            studio.register_session(replace(original, takes=(recording('Replacement'),)))
        assert path.read_bytes() == before
        assert studio.get_session(original.session_id).takes[0].transcript == 'Original'


@pytest.mark.parametrize('endpoint', ['references', 'transcribe'])
def test_upload_that_never_becomes_a_job_is_removed(tmp_path, endpoint):
    (tmp_path/'web').mkdir()
    with TestClient(application.create_app(tmp_path), raise_server_exceptions=False) as client:
        studio = client.app.state.studio
        kept = studio.data/'inputs'/'kept.wav';kept.write_bytes(b'Existing recording')
        reject_inserts(studio, 'jobs')
        response = client.post('/api/'+endpoint, files={'file':('new.wav', b'New uploaded recording')})
        assert response.status_code == 500
        assert list((studio.data/'inputs').iterdir()) == [kept]
        assert kept.read_bytes() == b'Existing recording'
        assert studio.jobs() == []


def test_failed_import_cleans_both_upload_and_unregistered_archive(tmp_path):
    (tmp_path/'web').mkdir()
    source = tmp_path/'source.safetensors';save_session(Session(takes=(recording(),)), source)
    with TestClient(application.create_app(tmp_path), raise_server_exceptions=False) as client:
        studio = client.app.state.studio
        reject_inserts(studio, 'sessions')
        response = client.post('/api/projects/import', files={'file':('source.safetensors', source.read_bytes())})
        assert response.status_code == 500
        assert not list((studio.data/'inputs').iterdir())
        assert not list((studio.data/'sessions').iterdir())
        assert studio.sessions() == []
