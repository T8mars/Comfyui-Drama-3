import io
import json
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from studio_app_loader import application


@pytest.fixture
def client(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web/index.html").write_text("test")
    instance = application.create_app(tmp_path)
    assert not (tmp_path / "userdata").exists()
    with TestClient(instance) as client:
        yield client


@pytest.mark.parametrize("content", [b"broken safetensors", None])
def test_malformed_project_is_400_and_removes_uploaded_file(client, tmp_path, content):
    if content is None:
        path = tmp_path / "bad-schema.safetensors"
        save_file({}, str(path), metadata={"drama3_t8_session": "true"})
        content = path.read_bytes()
    response = client.post("/api/projects/import", files={"file": ("bad.safetensors", content)})
    assert response.status_code == 400
    assert not list((tmp_path / "userdata/inputs").iterdir())


@pytest.mark.parametrize("endpoint,name", [("references", "a.wav"), ("transcribe", "a.wav"), ("projects/import", "a.safetensors")])
def test_oversized_upload_is_rejected_without_job_or_partial_file(client, tmp_path, monkeypatch, endpoint, name):
    monkeypatch.setattr(application, "MAX_UPLOAD_BYTES", 16)
    response = client.post("/api/" + endpoint, files={"file": (name, b"a" * 32)})
    assert response.status_code == 413
    assert not list((tmp_path / "userdata/inputs").iterdir())
    assert client.get("/api/jobs").json() == []


def test_transcript_keeps_its_original_audio_for_history_and_reference(client, monkeypatch):
    studio = client.app.state.studio
    monkeypatch.setattr(studio.engine, "transcribe_detailed", lambda *args, **kwargs:
                        {"text": "Original audio", "language": "English", "backend": "r2t2",
                         "duration_seconds": 1, "elapsed_seconds": 1})
    waveform = io.BytesIO()
    sf.write(waveform, np.ones(1000, dtype=np.float32) * .1, 16000, format="WAV")
    audio = waveform.getvalue()
    response = client.post("/api/transcribe", files={"file": ("original.wav", audio)})
    assert response.status_code == 200
    studio.pending.join()
    job = client.get("/api/jobs").json()[0]
    assert job["status"] == "done"
    assert job["result"]["source_name"] == "original.wav"
    assert client.get(job["result"]["source_audio"]).content == audio


@pytest.mark.parametrize("url,status", [("/api/files/input/..%5Cstudio.sqlite3", 404),
    ("/api/files/output/..%5Cuserdata%5Cstudio.sqlite3", 404),
    ("/api/jobs/not-a-uuid/chunks", 400), ("/api/sessions/not-a-uuid/takes/0.wav", 404),
    ("/api/stream/..%5C..%5Cuserdata/studio.sqlite3.wav", 404)])
def test_file_routes_keep_workspace_boundaries(client, url, status):
    assert client.get(url).status_code == status


def test_bad_job_type_is_a_client_error(client):
    assert client.post("/api/jobs", json={"mode": []}).status_code == 400
