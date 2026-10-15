import threading
from dataclasses import replace

import torch
import pytest
from fastapi.testclient import TestClient

import app as application
from drama3_t8.state import Session, Take, save_session
from drama3_t8.studio import Studio


def make_take(text, code):
    frames=torch.full((3,11),code,dtype=torch.int64)
    frames[:,0]=248099
    return Take(text,text,frames)


def test_import_creates_a_copy_without_overwriting_existing_project_or_preview(tmp_path):
    (tmp_path/'web').mkdir()
    old=Session(takes=(make_take('Original',1),))
    source=tmp_path/'project.safetensors'
    save_session(old,source)
    with TestClient(application.create_app(tmp_path)) as client:
        first=client.post('/api/projects/import',files={'file':('project.safetensors',source.read_bytes())}).json()['session_id']
        preview=tmp_path/'outputs/previews'/f'{first}_0.wav'
        preview.parent.mkdir();preview.write_bytes(b'cached original preview')
        changed=replace(old,takes=(make_take('Changed',2),))
        save_session(changed,source)
        second=client.post('/api/projects/import',files={'file':('project.safetensors',source.read_bytes())}).json()['session_id']
        assert second!=first,'Reimport overwrote the existing project identity'
        assert client.get('/api/sessions/'+first).json()['takes'][0]['text']=='Original'
        assert client.get('/api/sessions/'+second).json()['takes'][0]['text']=='Changed'
        assert client.get(f'/api/sessions/{second}/takes/0.wav').status_code==409
        assert client.get(f'/api/sessions/{first}/takes/0.wav').content==b'cached original preview'


def test_active_jobs_remain_visible_beyond_the_recent_history_window(tmp_path):
    entered,release=threading.Event(),threading.Event()
    with Studio(tmp_path) as studio:
        def hold(*args):
            entered.set();assert release.wait(10)
            return {'ok':True}
        studio.execute=hold
        running=studio.submit({'mode':'hold'})
        try:
            assert entered.wait(10)
            with studio.connect() as db:
                db.executemany('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',[(f'{i:032x}','done','{}','{}',None,10**12+i,10**12+i,1) for i in range(120)])
            assert running in {job['id'] for job in studio.jobs()},'Running job disappeared after 100 newer jobs'
        finally:release.set()
        studio.pending.join()


def test_completed_job_can_be_observed_after_it_leaves_recent_history(tmp_path):
    (tmp_path/'web').mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        studio=client.app.state.studio
        job=client.post('/api/jobs',json={'mode':'unload'}).json()['job_id']
        studio.pending.join()
        with studio.connect() as db:
            db.executemany('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',[(f'{i:032x}','done','{}','{}',None,10**12+i,10**12+i,1) for i in range(120)])
        assert job not in {j['id'] for j in client.get('/api/jobs').json()}
        response=client.get('/api/jobs/'+job)
        assert response.status_code==200,'No API to observe a result outside recent history'
        assert response.json()['id']==job and response.json()['status']=='done'
        assert client.get('/api/jobs/'+'f'*32).status_code==404
        assert client.get('/api/jobs/not-a-uuid').status_code==404


@pytest.mark.parametrize("seed", ["9007199254740993", "18446744073709551615"])
def test_decimal_seed_reaches_torch_without_javascript_precision_loss(tmp_path, monkeypatch, seed):
    with Studio(tmp_path) as studio:
        observed=[]
        def generate(mode,payload,session,sampling,progress,on_codes):
            torch.Generator(device="cpu").manual_seed(sampling.seed)
            observed.append(sampling.seed)
            return session.append(make_take("Exact seed",1))
        monkeypatch.setattr(studio,"generate_session",generate)
        monkeypatch.setattr(studio.engine,"decode",lambda codes:{"waveform":torch.zeros(1,1,100),"sample_rate":44100})
        job=studio.submit({"mode":"tts","text":"Exact seed","sampling":{"seed":seed}})
        studio.pending.join()
        assert studio.get_job(job)["status"]=="done"
        assert observed==[int(seed)]


def test_invalid_decimal_seed_fails_before_model_loading_and_queue_recovers(tmp_path, monkeypatch):
    with Studio(tmp_path) as studio:
        monkeypatch.setattr(studio,"generate_session",lambda *a:pytest.fail("Invalid seed reached model generation"))
        job=studio.submit({"mode":"tts","text":"Invalid seed","sampling":{"seed":"1e3"}})
        recovered=studio.submit({"mode":"unload"})
        studio.pending.join()
        assert studio.get_job(job)["status"]=="failed"
        assert studio.get_job(recovered)["status"]=="done"
