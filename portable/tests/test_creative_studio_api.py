import io
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from studio_app_loader import application


def app_at(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("studio", encoding="utf-8")
    return application.create_app(tmp_path)


def test_creative_project_dictionary_roundtrip_and_full_uint64_seed(tmp_path):
    with TestClient(app_at(tmp_path)) as client:
        response = client.post("/api/creative/script/preview", json={"text": "旁白：第3场开始", "dictionary": {"number_mode": "cardinal"}})
        assert response.status_code == 200, response.text
        rows = response.json()["lines"]
        assert rows[0]["readout_text"] == "第三场开始"
        assert rows[0]["readout_override"] is None
        created = client.post("/api/creative/projects", json={"name": "短剧", "lines": rows, "dictionary": {"number_mode": "cardinal"}})
        assert created.status_code == 200, created.text
        project = created.json()
        pid = project["project_id"]
        response = client.patch(f"/api/creative/projects/{pid}", json={"expected_revision": project["revision"], "dictionary": {"number_mode": "none"}})
        assert response.status_code == 200, response.text
        assert response.json()["lines"][0]["readout_text"] == "第3场开始"
        plan = client.post(f"/api/creative/projects/{pid}/plan", json={"base_seed": "18446744073709551615", "takes_per_line": 1})
        assert plan.status_code == 200, plan.text
        assert plan.json()["jobs"][0]["seed"] == "18446744073709551615"
        stale = client.patch(f"/api/creative/projects/{pid}", json={"expected_revision": project["revision"], "name": "冲突"})
        assert stale.status_code == 409


def test_custom_library_location_and_table_preview_do_not_load_model(tmp_path):
    app = app_at(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/creative/configuration", json={"voice_library_path": str(tmp_path / "my voices"), "idle_unload_seconds": 0})
        assert response.status_code == 200, response.text
        voice = client.post("/api/creative/voices", json={"name": "女主", "tags": ["搞笑"], "favorite": True}).json()
        result = client.get("/api/creative/library?query=女主&favorites=true")
        assert result.status_code == 200, result.text
        assert result.json()["voices"][0]["id"] == voice["id"]
        response = client.post("/api/creative/script/preview", json={"lines": [{"line_id": "L99", "role": "女主", "original_text": "3次", "readout_override": "三回", "direction": "轻声"}], "dictionary": {"number_mode": "digits"}})
        assert response.status_code == 200, response.text
        line = response.json()["lines"][0]
        assert line["line_id"] == "L99" and line["direction"] == "轻声" and line["readout_text"] == "三回"
        assert app.state.studio.engine.drama is None
        assert client.get("/api/creative/downloads/absent.zip").status_code == 404


def test_selected_reference_keeps_original_and_exposes_real_clip(tmp_path, monkeypatch):
    from drama3_t8.state import Reference
    import torch
    app = app_at(tmp_path)
    with TestClient(app) as client:
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "drama3-codec.safetensors").write_bytes(b"synthetic-codec-identity")
        monkeypatch.setattr(app.state.studio.engine, "encode_reference", lambda wave, rate, text, name, **options:
            Reference(text, torch.zeros((10, 3), dtype=torch.long), name))
        raw = io.BytesIO()
        sf.write(raw, np.sin(np.arange(8000) * .1).astype(np.float32) * .1, 8000, format="WAV", subtype="PCM_24")
        original = raw.getvalue()
        response = client.post("/api/creative/reference", data={"name": "猫", "transcript": "喵", "start_seconds": ".25", "end_seconds": ".75", "delivery": "喊话"}, files={"file": ("cat.wav", original, "audio/wav")})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(.02)
        assert job["status"] == "done", job
        version = job["result"]["version_id"]
        source = client.get(f"/api/creative/versions/{version}/audio?original=true")
        preview = client.get(f"/api/creative/versions/{version}/audio")
        assert source.content == original
        clip, rate = sf.read(io.BytesIO(preview.content))
        assert rate == 44100 and len(clip) == 22050
        assert app.state.studio.engine.drama is None


def test_explicit_version_uses_its_real_audio_codes_and_rejects_foreign_voice(tmp_path):
    from drama3_t8.creative_studio import creative_manager
    from drama3_t8.state import Reference, Session
    import pytest
    import torch
    app = app_at(tmp_path)
    with TestClient(app):
        manager = creative_manager(app.state.studio)
        voice = manager.voices.create_voice("女主")
        default, explicit = [manager.voices.add_version(voice["id"], Session((Reference(
            "<|speaker:0|> 参考", torch.full((10, 3), code, dtype=torch.long)),))) for code in (1, 2)]
        job = {"project_id": manager.create_project({"lines": ["测试"]})["project_id"],
               "lines": [{"role": "女主", "readout_text": "台词", "reference_ids": [explicit["id"]]}],
               "cast": {"女主": {"voice_id": voice["id"], "version_id": default["id"], "speaker": 0}},
               "settings": {}, "mode": "per_line", "model_identity": {}}
        session, choices, _ = manager._session(job)
        assert choices[0]["version_id"] == explicit["id"]
        assert len(session.references) == 1 and int(session.references[0].codes[0, 0]) == 2
        other = manager.voices.create_voice("另一角色")
        foreign = manager.voices.add_version(other["id"], Session((Reference("参考", torch.ones((10, 3), dtype=torch.long)),)))
        job["lines"][0]["reference_ids"] = [foreign["id"]]
        with pytest.raises(ValueError, match="同一绑定音色"):
            manager._session(job)


def test_imported_project_audio_and_new_cast_are_self_contained(tmp_path):
    from drama3_t8.creative_studio import creative_manager
    from drama3_t8.creative_bundle import export_creative_bundle, import_creative_bundle, project_voice_library
    from drama3_t8.state import Reference, Session
    import torch
    app = app_at(tmp_path)
    with TestClient(app) as client:
        manager = creative_manager(app.state.studio)
        original = tmp_path / "source.wav"
        sf.write(original, np.zeros(80, dtype=np.float32), 8000)
        first = manager.voices.create_voice("原角色")
        version = manager.voices.add_version(first["id"], Session((Reference("原声", torch.ones((10, 2), dtype=torch.long)),)), source_path=original)
        source = manager.create_project({"lines": [{"role": "女主", "original_text": "你好"}],
            "cast": {"女主": {"voice_id": first["id"], "version_id": version["id"], "speaker": 0}}})
        archive = tmp_path / "project.zip"
        export_creative_bundle(manager.workspace, manager.voices, source["project_id"], archive)
        imported = import_creative_bundle(manager.workspace, archive)
        pid = imported["project_id"]
        view = client.get(f"/api/creative/projects/{pid}").json()
        preview = view["embedded_voices"][0]["versions"][0]["audio_url"]
        assert client.get(preview).content == original.read_bytes()
        second = manager.voices.create_voice("新选角")
        fresh = manager.voices.add_version(second["id"], Session((Reference("新声", torch.full((10, 2), 2, dtype=torch.long)),)), source_path=original)
        changed = client.patch(f"/api/creative/projects/{pid}", json={"expected_revision": view["revision"],
            "cast": {"女主": {"voice_id": second["id"], "version_id": fresh["id"], "speaker": 0}}})
        assert changed.status_code == 200, changed.text
        library = project_voice_library(manager.workspace, pid, None)
        assert library.get_version(fresh["id"])["voice_id"] == second["id"]
        assert library.get_version(version["id"])["voice_id"] == first["id"]
        assert client.get(preview).content == original.read_bytes()


def test_performance_catalog_skips_identity_caches_and_invalid_json(tmp_path):
    import json
    from drama3_t8.creative_studio import creative_manager
    app = app_at(tmp_path)
    with TestClient(app) as client:
        manager = creative_manager(app.state.studio)
        (manager.reports / "benchmark-identities.json").write_text(json.dumps({"codec.bin": {"sha256": "f" * 64}}), encoding="utf-8")
        (manager.reports / "broken.json").write_text("{broken", encoding="utf-8")
        result = client.get("/api/creative/performance")
        assert result.status_code == 200, result.text
        assert result.json()["reports"] == []
