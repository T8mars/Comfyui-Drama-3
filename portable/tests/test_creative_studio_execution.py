"""Host orchestration contracts with a small, explicitly synthetic engine.

These tests do not establish actual model quality, GPU speed or UI execution.
"""
from contextlib import contextmanager
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from studio_app_loader import application
from drama3_t8.state import Take


class TinyEngine:
    def __init__(self):
        self.drama = None
        self.codec = None
        self.moe_backend = "torch"
        self.gdn_backend = "torch"
        self.expert_placement = "ram"
        self.cuda_graph = False
        self.loads = self.calls = 0
        self.fail_text = None
        self.spoken_texts = []
        self.delivery_directions = []

    def load_model(self):
        if self.drama is None:
            self.loads += 1
            self.drama = SimpleNamespace(last_graph_stats={"captures": 0, "replays": 0})
        return self.drama

    @contextmanager
    def generation_session(self):
        self.load_model()
        yield

    def unload(self):
        self.drama = self.codec = None

    def expert_stats(self):
        return {"actual_backend": "torch", "placement": "ram", "fallback_reason": None}

    def _make(self, lines, session, sampling, on_codes):
        if self.fail_text and any(self.fail_text in line for line in lines):
            raise ValueError("synthetic failure")
        self.calls += 1
        self.spoken_texts.extend(lines)
        self.delivery_directions.append(session.direction)
        chunks = []
        for index, line in enumerate(lines):
            frames = torch.zeros((3, 11), dtype=torch.long)
            frames[:, 0] = 248099
            chunks.append(frames)
            if index + 1 < len(lines):
                separator = torch.zeros((1, 11), dtype=torch.long)
                separator[0, 0] = 252196
                chunks.append(separator)
        if on_codes:
            on_codes(chunks[0][0])
        from drama3_t8.prompts import spoken_text
        return session.append(Take("\n<|audio_separator|>\n".join(lines), spoken_text("\n".join(lines),scene=True), torch.cat(chunks), metadata={
            "seed": sampling.seed, "requested_seed": sampling.seed, "finish_reason": "eos",
            "sampling": sampling.__dict__,
            "length_limit_reached": False, "gdn_calls_by_backend": {"torch": 3},
            "elapsed_seconds": .01, "duration_seconds": len(lines) * 3 * 2048 / 44100}))

    def speak(self, text, session, sampling, on_codes=None, progress=None, *, note=""):
        from drama3_t8.prompts import directed_text
        return self._make([directed_text(text,session.direction,note)], session, sampling, on_codes)

    def revise(self, session, note, sampling, on_codes=None, progress=None):
        # Like the native director-only API, this has no replacement-script input.
        return self.speak(session.takes[-1].transcript, session, sampling, on_codes, progress)

    def scene(self, lines, scene, characters, *, references, sampling, progress, session,
              on_codes, generation_mode, split_long_text, chunk_bytes, note=""):
        from drama3_t8.prompts import directed_scene_lines
        lines,_,_=directed_scene_lines(lines,session.references,session.takes,session.direction,note)
        return self._make(lines, session, sampling, on_codes)

    def decode_take(self, take, with_timeline=False):
        if "scene_chunking" in take.metadata:
            from drama3_t8.scene_chunks import decode_scene
            audio, timeline = decode_scene(self, take)
            return (audio, timeline) if with_timeline else audio
        result = self.decode(take.codes)
        return (result, []) if with_timeline else result

    def decode(self, codes):
        count = codes.shape[-1] * 2048
        wave = torch.from_numpy((np.sin(np.arange(count) * .08) * .1).astype("float32"))[None, None]
        result = {"waveform": wave, "sample_rate": 44100}
        self.codec = object()
        return result


def client_at(tmp_path):
    (tmp_path / "web").mkdir()
    app = application.create_app(tmp_path)
    return app, TestClient(app)


def wait_job(client, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(.01)
    raise AssertionError("queue did not settle")


def test_real_workspace_artifacts_resume_without_loading_and_role_order(tmp_path):
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"name": "交错", "lines": [
            {"line_id": "L01", "role": "猫", "original_text": "第一句"},
            {"line_id": "L02", "role": "人", "original_text": "第二句"},
            {"line_id": "L03", "role": "猫", "original_text": "第三句"}]}).json()
        pid = project["project_id"]
        options = {"generation_mode": "per_speaker", "takes_per_line": 1, "base_seed": "42"}
        submitted = client.post(f"/api/creative/projects/{pid}/run", json=options)
        assert submitted.status_code == 200, submitted.text
        job = wait_job(client, submitted.json()["job_id"])
        assert job["status"] == "done", job
        assert engine.calls == 2
        view = client.get(f"/api/creative/projects/{pid}").json()
        assert all(line["selected_candidate"] for line in view["lines"])
        first = view["candidates"][view["lines"][0]["selected_candidate_id"]]
        assert len(first["line_audio"]) == 2
        assert all(client.get(item["audio_url"]).status_code == 200 for item in first["line_audio"])
        engine.unload()
        loads = engine.loads
        resumed = client.post(f"/api/creative/projects/{pid}/run", json=options).json()
        assert wait_job(client, resumed["job_id"])["status"] == "done"
        assert engine.loads == loads and engine.calls == 2
        delivered = client.post(f"/api/creative/projects/{pid}/delivery", json={"sample_rate": 44100, "gap_seconds": 0}).json()
        delivery = wait_job(client, delivered["job_id"])
        assert delivery["status"] == "done", delivery
        manifest = delivery["result"]["manifest"]
        assert [row["line_id"] for row in manifest["lines"]] == ["L01", "L02", "L03"]
        assert [row["role"] for row in manifest["lines"]] == ["猫", "人", "猫"]


def test_partial_batch_is_not_success_and_only_failed_does_not_generate_new_missing_lines(tmp_path):
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        engine.fail_text = "失败"
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"name": "失败保留", "lines": ["成功", "失败"]}).json()
        pid = project["project_id"]
        options = {"takes_per_line": 1, "base_seed": "42"}
        first = client.post(f"/api/creative/projects/{pid}/run", json=options).json()
        assert wait_job(client, first["job_id"])["status"] == "failed"
        project = client.get(f"/api/creative/projects/{pid}").json()
        assert sorted(item["status"] for item in project["candidates"].values()) == ["completed", "failed"]
        rows = project["lines"] + [{"line_id": "L03", "original_text": "新缺失"}]
        updated = client.patch(f"/api/creative/projects/{pid}", json={"lines": rows, "expected_revision": project["revision"]})
        assert updated.status_code == 200, updated.text
        engine.fail_text = None
        retry = client.post(f"/api/creative/projects/{pid}/run", json={**options, "only_failed": True}).json()
        assert wait_job(client, retry["job_id"])["status"] == "done"
        after = client.get(f"/api/creative/projects/{pid}").json()
        assert engine.calls == 2
        assert after["lines"][2]["candidate_ids"] == []
        assert after["lines"][1]["selected_candidate"]["status"] == "completed"


def test_ui_quality_after_option_runs_post_batch_stage(tmp_path, monkeypatch):
    from drama3_t8.creative_studio import creative_manager
    app, client = client_at(tmp_path)
    with client:
        app.state.studio.engine = TinyEngine()
        manager = creative_manager(app.state.studio)
        calls = []
        def quality(job_id, payload, progress):
            calls.append(payload)
            return {"checked": len(payload["candidate_ids"]), "reviews": {}, "project_revision": 1, "errors": []}
        monkeypatch.setattr(manager, "execute_quality", quality)
        project = client.post("/api/creative/projects", json={"lines": ["完整台词"]}).json()
        submitted = client.post(f"/api/creative/projects/{project['project_id']}/run", json={
            "takes_per_line": 1, "base_seed": "42", "quality_after": True, "max_quality_retries": 0}).json()
        finished = wait_job(client, submitted["job_id"])
        assert finished["status"] == "done", finished
        assert len(calls) == 1 and finished["result"]["quality"]["checked"] == 1


def test_delivery_rejects_changed_readout_instead_of_mislabeling_old_audio(tmp_path):
    app, client = client_at(tmp_path)
    with client:
        app.state.studio.engine = TinyEngine()
        project = client.post("/api/creative/projects", json={"lines": ["原来的台词"]}).json()
        pid = project["project_id"]
        job = client.post(f"/api/creative/projects/{pid}/run", json={"takes_per_line": 1}).json()
        assert wait_job(client, job["job_id"])["status"] == "done"
        current = client.get(f"/api/creative/projects/{pid}").json()
        rows = current["lines"]
        rows[0]["original_text"] = "修改后的另一句台词"
        updated = client.patch(f"/api/creative/projects/{pid}", json={"lines": rows, "expected_revision": current["revision"]})
        assert updated.status_code == 200, updated.text
        delivery = client.post(f"/api/creative/projects/{pid}/delivery", json={}).json()
        result = wait_job(client, delivery["job_id"])
        assert result["status"] == "failed" and "朗读稿已修改" in result["error"]


def test_tagged_script_and_replacement_revision_use_exact_requested_words(tmp_path):
    from drama3_t8.state import load_session
    from drama3_t8.prompts import spoken_text
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"lines": [
            {"role": "主角", "original_text": "<|speaker:3|> 周一开工。"}]}).json()
        pid = project["project_id"]
        submitted = client.post(f"/api/creative/projects/{pid}/run", json={"takes_per_line": 1}).json()
        assert wait_job(client, submitted["job_id"])["status"] == "done"
        assert engine.spoken_texts == ["<|speaker:3|> 周一开工。"]
        current = client.get(f"/api/creative/projects/{pid}").json()
        source_id = current["lines"][0]["selected_candidate_id"]
        old_file = current["candidates"][source_id]["session_path"]
        from pathlib import Path
        original_bytes = Path(old_file).read_bytes()
        submitted = client.post(f"/api/creative/projects/{pid}/variation", json={"line_id": "L01",
            "source_candidate_id": source_id, "expected_revision": current["revision"],
            "operation": {"kind": "revise", "text": "周五放假。", "note": "开心一点"}}).json()
        finished = wait_job(client, submitted["job_id"])
        assert finished["status"] == "done", finished
        assert engine.spoken_texts[-1] == "<|speaker:3|> [开心一点] 周五放假。"
        current = client.get(f"/api/creative/projects/{pid}").json()
        candidate = current["candidates"][finished["result"]["variation"]["candidate_id"]]
        assert spoken_text(load_session(candidate["session_path"]).takes[-1].transcript) == "周五放假。"
        assert current["lines"][0]["selected_candidate_id"] == source_id
        assert Path(old_file).read_bytes() == original_bytes


def test_long_quality_batches_have_explicit_native_bounds(tmp_path, monkeypatch):
    from drama3_t8.creative_studio import _transcribe_bounded
    from drama3_t8.r2t2 import transcribe
    calls = []
    def bounded(audios, *paths, max_items=256, **options):
        rows = list(audios)
        if len(rows) > max_items:
            raise ValueError("Too many audio items")
        calls.append(max_items)
        return rows
    monkeypatch.setattr(transcribe, "transcribe_audio_batch", bounded)
    paths = {key: "fixture" for key in ("r2t2_model", "r2t2_projector", "r2t2_vad")}
    assert _transcribe_bounded(iter(range(257)), 257, paths, lambda: None) == list(range(257))
    assert calls == [257]
    calls.clear()
    assert _transcribe_bounded(iter(range(10001)), 10001, paths, lambda: None) == list(range(10001))
    assert calls == [10000, 1]


@pytest.mark.parametrize("truncated", [False, True])
def test_independent_transcription_updates_saved_asr_checked_flag(tmp_path, monkeypatch, truncated):
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        engine.asr_backend = "qwen"
        monkeypatch.setattr(engine, "transcribe_detailed", lambda *args, **kwargs: {
            "text": "完整台词", "truncated": truncated}, raising=False)
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"lines": ["完整台词"]}).json()
        pid = project["project_id"]
        submitted = client.post(f"/api/creative/projects/{pid}/run", json={"takes_per_line": 1}).json()
        assert wait_job(client, submitted["job_id"])["status"] == "done"
        initial = client.get(f"/api/creative/projects/{pid}").json()
        assert all(candidate["quality"]["asr_checked"] is False for candidate in initial["candidates"].values())
        queued = client.post(f"/api/creative/projects/{pid}/quality", json={}).json()
        assert wait_job(client, queued["job_id"])["status"] == "done"
        current = client.get(f"/api/creative/projects/{pid}").json()
        for candidate in current["candidates"].values():
            assert candidate["quality"]["asr_checked"] is True
            assert candidate["quality"]["independent"] is True
            assert bool(candidate["quality"].get("review_error")) is truncated


def test_revision_uses_current_delivery_and_preserves_archived_direction(tmp_path):
    from pathlib import Path
    from drama3_t8.state import load_session
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"settings": {"direction": "旧的整体演出"},
            "lines": [{"role": "主角", "original_text": "今天开工。", "direction": "旧的句级演出"}]}).json()
        pid = project["project_id"]
        submitted = client.post(f"/api/creative/projects/{pid}/run", json={"takes_per_line": 1}).json()
        assert wait_job(client, submitted["job_id"])["status"] == "done"
        current = client.get(f"/api/creative/projects/{pid}").json()
        source_id = current["lines"][0]["selected_candidate_id"]
        source_path = Path(current["candidates"][source_id]["session_path"])
        original = source_path.read_bytes()
        assert load_session(source_path).direction == "旧的整体演出"
        rows = current["lines"]
        rows[0]["direction"] = "新的句级演出"
        updated = client.patch(f"/api/creative/projects/{pid}", json={"lines": rows,
            "settings": {**current["settings"], "direction": "新的整体演出"},
            "expected_revision": current["revision"]})
        assert updated.status_code == 200, updated.text
        current = updated.json()
        submitted = client.post(f"/api/creative/projects/{pid}/variation", json={"line_id": "L01",
            "source_candidate_id": source_id, "expected_revision": current["revision"],
            "operation": {"kind": "revise", "text": "今天放假。", "note": "开心一点"}}).json()
        finished = wait_job(client, submitted["job_id"])
        assert finished["status"] == "done", finished
        assert engine.delivery_directions[-1] == "新的整体演出"
        assert "[新的整体演出]" in engine.spoken_texts[-1]
        assert "[新的句级演出]" in engine.spoken_texts[-1]
        assert "[开心一点]" in engine.spoken_texts[-1]
        assert "旧的句级演出" not in engine.spoken_texts[-1]
        assert engine.spoken_texts[-1].endswith("今天放假。")
        assert source_path.read_bytes() == original
        assert load_session(source_path).direction == "旧的整体演出"


def test_long_single_line_creates_complete_source_aware_audio(tmp_path):
    from drama3_t8.state import load_session
    app, client = client_at(tmp_path)
    with client:
        engine = TinyEngine()
        app.state.studio.engine = engine
        project = client.post("/api/creative/projects", json={"lines": [{"role": "主角",
            "original_text": "<|speaker:7|> " + "这是一段要完整读出的较长对白。" * 8}],
            "settings": {"split_long_text": True, "chunk_bytes": 80}}).json()
        pid = project["project_id"]
        job = client.post(f"/api/creative/projects/{pid}/run", json={"takes_per_line": 1, "generation_mode": "long_form"}).json()
        finished = wait_job(client, job["job_id"])
        assert finished["status"] == "done", finished
        project = client.get(f"/api/creative/projects/{pid}").json()
        candidate = project["candidates"][project["lines"][0]["selected_candidate_id"]]
        take = load_session(candidate["session_path"]).takes[-1]
        assert take.metadata["chunk_count"] > 1
        assert take.metadata["scene_chunking"]["fixed_reference_anchor"] is True
        assert len(take.segments()) == 1 and len(candidate["line_audio"]) == 1
        assert all(text.startswith("<|speaker:0|>") for text in engine.spoken_texts)
