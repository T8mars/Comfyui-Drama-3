import time

from drama3_t8.bootstrap import standalone_engine

standalone_engine()
from drama3_t8.studio import Studio


def test_durable_queue_runs_and_records_real_result(tmp_path):
    studio = Studio(tmp_path)
    assert studio.references() == []
    assert studio.sessions() == []
    job_id = studio.submit({"mode": "unload"})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        jobs = studio.jobs()
        if jobs and jobs[0]["status"] not in {"queued", "running"}:
            break
        time.sleep(.05)
    job = next(j for j in studio.jobs() if j["id"] == job_id)
    assert job["status"] == "done", job["error"]
    assert job["result"]["message"] == "模型已释放"


def test_invalid_mode_fails_instead_of_fake_success(tmp_path):
    studio = Studio(tmp_path)
    job_id = studio.submit({"mode": "unknown"})
    studio.pending.join()
    job = next(j for j in studio.jobs() if j["id"] == job_id)
    assert job["status"] == "failed"
    assert "Unknown generation mode" in job["error"]
    assert job["result"] is None


def test_comfy_base_exception_cancellation_keeps_worker_alive(tmp_path,monkeypatch):
    import comfy.model_management as mm
    studio=Studio(tmp_path)
    original=studio.execute
    def execute(job_id,payload,progress):
        if payload["mode"]=="interrupt":
            raise mm.InterruptProcessingException()
        return original(job_id,payload,progress)
    monkeypatch.setattr(studio,"execute",execute)
    interrupted=studio.submit({"mode":"interrupt"})
    recovered=studio.submit({"mode":"unload"})
    deadline=time.monotonic()+10
    while studio.pending.unfinished_tasks and time.monotonic()<deadline:
        time.sleep(.05)
    jobs={job["id"]:job for job in studio.jobs()}
    assert jobs[interrupted]["status"]=="cancelled"
    assert jobs[recovered]["status"]=="done"
    assert studio.worker.is_alive()
