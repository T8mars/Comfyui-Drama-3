import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from studio_app_loader import application

from drama3_t8.state import Reference, Session, Take, save_session
from drama3_t8.studio import Studio


def make_take(text):
    frames = torch.ones((4, 11), dtype=torch.long)
    frames[:, 0] = 248100
    return Take(text, text, frames)


@pytest.mark.parametrize('mode', ['preview', 'align', 'revise', 'infill'])
@pytest.mark.parametrize('index', [0.9, True])
def test_take_index_does_not_silently_select_another_version(tmp_path, monkeypatch, mode, index):
    with Studio(tmp_path) as studio:
        session = Session(takes=(make_take('First'), make_take('Second')))
        studio.register_session(session)
        monkeypatch.setattr(studio.engine, 'decode', lambda *a: {'waveform': torch.zeros(1, 1, 100), 'sample_rate': 44100})
        monkeypatch.setattr(studio, 'generate_session', lambda mode, payload, session, *a: session.append(make_take('New')))
        import drama3_t8.alignment as alignment
        monkeypatch.setattr(alignment, 'align_audio', lambda *a: [])
        with pytest.raises(ValueError, match='版本|take'):
            studio.execute('a' * 32, {'mode': mode, 'session_id': session.session_id, 'take_index': index, 'text': 'New'}, lambda: None)


@pytest.mark.parametrize('index', ['0', 0, -1])
def test_valid_take_indices_keep_latest_and_decimal_string_compatibility(tmp_path, monkeypatch, index):
    with Studio(tmp_path) as studio:
        session = Session(takes=(make_take('First'), make_take('Second')))
        studio.register_session(session)
        monkeypatch.setattr(studio.engine, 'decode', lambda *a: {'waveform': torch.zeros(1, 1, 100), 'sample_rate': 44100})
        result = studio.execute('b' * 32, {'mode': 'preview', 'session_id': session.session_id, 'take_index': index}, lambda: None)
        assert result['transcript'] == ('Second' if index == -1 else 'First')


@pytest.mark.parametrize('speaker', [0.9, True])
def test_reference_speaker_does_not_silently_select_another_role(tmp_path, speaker):
    with Studio(tmp_path) as studio:
        rid = 'c' * 32
        archive = studio.data / 'references' / (rid + '.safetensors')
        save_session(Session(references=(Reference('Hello', torch.ones(10, 4, dtype=torch.long)),)), archive)
        with studio.connect() as db:
            db.execute('INSERT INTO voice_references VALUES(?,?,?,?)', (rid, 'Voice', 'Hello', archive.relative_to(studio.data).as_posix()))
        with pytest.raises(ValueError, match='角色|Speaker'):
            studio.session_for({'reference_ids': [rid], 'reference_speakers': {rid: speaker}})


def project_bytes(tmp_path):
    path = tmp_path / 'source.safetensors'
    save_session(Session(takes=(make_take('Original'),)), path)
    return path.read_bytes()


def test_successful_import_removes_unreferenced_upload_copy(tmp_path):
    (tmp_path / 'web').mkdir()
    data = project_bytes(tmp_path)
    with TestClient(application.create_app(tmp_path)) as client:
        response = client.post('/api/projects/import', files={'file': ('project.safetensors', data)})
        assert response.status_code == 200
        sid = response.json()['session_id']
        assert client.get('/api/sessions/' + sid).json()['takes'][0]['text'] == 'Original'
        assert not list((tmp_path / 'userdata/inputs').iterdir())


def test_failed_project_registration_also_removes_upload_copy(tmp_path, monkeypatch):
    (tmp_path / 'web').mkdir()
    data = project_bytes(tmp_path)
    with TestClient(application.create_app(tmp_path), raise_server_exceptions=False) as client:
        def fail(*a):
            raise OSError('Registration storage unavailable')
        monkeypatch.setattr(client.app.state.studio, 'register_session', fail)
        response = client.post('/api/projects/import', files={'file': ('project.safetensors', data)})
        assert response.status_code == 500
        assert not list((tmp_path / 'userdata/inputs').iterdir())


@pytest.mark.parametrize('metadata', ['broken json', json.dumps({'version': 999}), '{}'])
def test_existing_but_invalid_project_is_400_instead_of_missing_404(tmp_path, metadata):
    (tmp_path / 'web').mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        studio = client.app.state.studio
        session = Session(takes=(make_take('Original'),))
        studio.register_session(session)
        archive = studio.data / 'sessions' / (session.session_id + '.safetensors')
        save_file({}, str(archive), metadata={'drama3_t8_session': metadata})
        response = client.get('/api/sessions/' + session.session_id)
        assert response.status_code == 400, response.text
        assert response.json()['detail']
        assert client.get('/api/sessions/' + 'f' * 32).status_code == 404


def test_failed_repeat_preview_keeps_previous_complete_audio(tmp_path, monkeypatch):
    import drama3_t8.studio as module
    with Studio(tmp_path) as studio:
        session = Session(takes=(make_take('Original'),))
        studio.register_session(session)
        folder = studio.output / 'previews'; folder.mkdir()
        previous = folder / (session.session_id + '_0.wav')
        previous.write_bytes(b'previous complete audio')
        monkeypatch.setattr(studio.engine, 'decode', lambda *a: {'waveform': torch.zeros(1, 1, 100), 'sample_rate': 44100})
        real_write = module.sf.write
        def fail(path, *a, **k):
            # Playback identity serializes PCM in memory; fail only the
            # durable preview write this regression is meant to exercise.
            if not isinstance(path, Path):
                return real_write(path, *a, **k)
            path.write_bytes(b'partial write')
            raise OSError('Audio write failed')
        monkeypatch.setattr(module.sf, 'write', fail)
        with pytest.raises(OSError, match='write failed'):
            studio.execute('d' * 32, {'mode': 'preview', 'session_id': session.session_id}, lambda: None)
        assert previous.read_bytes() == b'previous complete audio'
        assert list(folder.iterdir()) == [previous]
