import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from studio_app_loader import application

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import drama3_t8.studio as module
from drama3_t8.studio import Studio
from drama3_t8.state import Reference,Session,Take
from drama3_t8.model_paths import DEFAULT_PATHS


def take():
    codes=torch.ones((10,4),dtype=torch.long)
    return Take('Test','Test',torch.cat((codes[:1]+248099,codes)).T,metadata={'elapsed_seconds':0,'duration_seconds':4*2048/44100,'length_limit_reached':False})


def settings_files(tmp_path):
    settings={}
    for key,default in DEFAULT_PATHS.items():
        path=tmp_path/'模型'/key/Path(default).name
        if key in {'aligner','r2t2_vad'}:
            path.mkdir(parents=True)
            for name in (['config.json'] if key=='aligner' else ['cmvn.ark','fireredvad_stream_vad_with_cache.onnx']):(path/name).write_text('{}')
        else:path.parent.mkdir(parents=True);path.touch()
        settings[key]=str(path)
    return settings


@pytest.mark.parametrize('mode',['reference','tts','preview','align','transcribe'])
def test_cancel_is_rejected_once_result_writing_starts(tmp_path,monkeypatch,mode):
    entered,release=threading.Event(),threading.Event()
    with Studio(tmp_path) as studio:
        if mode=='align':
            # Optional assets are now checked at feature use; this test covers
            # the subsequent commit boundary, not an absent model configuration.
            studio.model_paths['aligner'].mkdir(parents=True)
            (studio.model_paths['aligner']/'config.json').write_text('{}')
        session=Session(takes=(take(),));studio.register_session(session)
        monkeypatch.setattr(module,'read_audio',lambda path:(torch.zeros(1,1000),44100))
        monkeypatch.setattr(studio.engine,'encode_reference',lambda *a:Reference('Test',torch.ones(10,4,dtype=torch.long),'Voice'))
        monkeypatch.setattr(studio.engine,'decode',lambda *a:{'waveform':torch.zeros(1,1,1000),'sample_rate':44100})
        monkeypatch.setattr(studio.engine,'transcribe_detailed',lambda *a,**k:{'text':'Test','language':'English','elapsed_seconds':0,'duration_seconds':1})
        monkeypatch.setattr(studio,'generate_session',lambda mode,payload,session,*a:session.append(take()))
        import drama3_t8.alignment as alignment
        monkeypatch.setattr(alignment,'align_audio',lambda *a:[{'text':'Test','start':0,'end':1}])
        original_save,original_write,original_text=module.save_session,module.sf.write,Path.write_text
        def pause():entered.set();assert release.wait(10)
        def save(value,target):
            original_save(value,target)
            if mode=='reference' and target.parent.name=='references':pause()
        def write(target,*a,**k):
            original_write(target,*a,**k)
            if isinstance(target,Path) and mode=='tts' and target.parent==studio.output:pause()
            if isinstance(target,Path) and mode=='preview' and target.parent.name=='previews':pause()
        def text(target,*a,**k):
            result=original_text(target,*a,**k)
            if (mode=='align' and target.name.endswith('.alignment.json')) or (mode=='transcribe' and target.suffix=='.txt'):pause()
            return result
        monkeypatch.setattr(module,'save_session',save);monkeypatch.setattr(module.sf,'write',write);monkeypatch.setattr(Path,'write_text',text)
        payload={'mode':mode,'text':'Test','input_id':'input.wav','transcript':'Test','session_id':session.session_id}
        job=studio.submit(payload)
        try:
            assert entered.wait(10)
            try:studio.cancel(job);rejected=False
            except ValueError:rejected=True
        finally:release.set()
        studio.pending.join()
        assert rejected,'Cancel was accepted after durable output writing had begun'
        assert studio.get_job(job)['status']=='done'
        if mode=='reference':assert len(studio.references())==1


def test_failed_engine_unload_does_not_persist_new_settings(tmp_path,monkeypatch):
    with Studio(tmp_path/'app') as studio:
        studio.apply_settings(settings_files(tmp_path))
        old,engine=studio.settings_path.read_bytes(),studio.engine
        def fail():raise RuntimeError('unload failed')
        monkeypatch.setattr(engine,'unload',fail)
        with pytest.raises(RuntimeError,match='unload failed'):studio.apply_settings({'expert_cache_gb':14})
        try:
            assert studio.settings_path.read_bytes()==old
            assert studio.engine is engine and studio.settings['expert_cache_gb']==12
            assert not studio.settings_path.with_suffix('.tmp').exists()
        finally:monkeypatch.setattr(engine,'unload',lambda:None)


def test_log_write_failure_does_not_kill_queue(tmp_path,monkeypatch):
    with Studio(tmp_path) as studio:
        original=Path.write_text
        def write(path,*a,**k):
            if path.parent.name=='logs':raise PermissionError('log folder unavailable')
            return original(path,*a,**k)
        monkeypatch.setattr(Path,'write_text',write)
        bad=studio.submit({'mode':'unknown'});recovered=studio.submit({'mode':'unload'})
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and studio.worker.is_alive() and studio.get_job(recovered)['status']!='done':time.sleep(.01)
        assert studio.worker.is_alive(),'A log I/O error terminated the GPU queue worker'
        assert studio.get_job(bad)['status']=='failed'
        assert studio.get_job(recovered)['status']=='done'


def test_cancel_during_final_decode_leaves_no_saved_result(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    with Studio(tmp_path) as studio:
        def decode(codes):
            entered.set();assert release.wait(10)
            return {'waveform':torch.zeros(1,1,1000),'sample_rate':44100}
        monkeypatch.setattr(studio.engine,'decode',decode)
        monkeypatch.setattr(studio,'generate_session',lambda mode,payload,session,*a:session.append(take()))
        job=studio.submit({'mode':'tts','text':'Test'})
        try:
            assert entered.wait(10);studio.cancel(job)
        finally:release.set()
        studio.pending.join()
        assert studio.get_job(job)['status']=='cancelled'
        assert studio.sessions()==[]
        assert not list(studio.output.rglob('*.wav'))
        assert not list((studio.data/'sessions').glob('*.safetensors'))


def test_configure_commit_rejects_late_cancel_and_reports_applied_settings(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    with Studio(tmp_path/'app') as studio:
        studio.apply_settings(settings_files(tmp_path))
        original=Path.write_text
        def write(path,*a,**k):
            result=original(path,*a,**k)
            if path==studio.settings_path.with_suffix('.tmp'):entered.set();assert release.wait(10)
            return result
        monkeypatch.setattr(Path,'write_text',write)
        job=studio.submit({'mode':'configure','settings':{'expert_cache_gb':14}})
        try:
            assert entered.wait(10)
            try:studio.cancel(job);rejected=False
            except ValueError:rejected=True
        finally:release.set()
        studio.pending.join()
        assert rejected,'Configuration was persisted despite a successful cancellation response'
        assert studio.get_job(job)['status']=='done'
        assert studio.settings['expert_cache_gb']==14


def test_unexpected_worker_baseexception_is_a_failed_job_and_queue_recovers(tmp_path,monkeypatch):
    with Studio(tmp_path) as studio:
        original=studio.execute
        def execute(job,payload,progress):
            if payload['mode']=='interrupt':raise KeyboardInterrupt('decoder worker error')
            return original(job,payload,progress)
        monkeypatch.setattr(studio,'execute',execute)
        bad=studio.submit({'mode':'interrupt'});recovered=studio.submit({'mode':'unload'})
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and studio.worker.is_alive() and studio.get_job(recovered)['status']!='done':time.sleep(.01)
        assert studio.worker.is_alive()
        assert studio.get_job(bad)['status']=='failed'
        assert studio.get_job(recovered)['status']=='done'


def test_nonfinite_api_payload_is_rejected_without_poisoning_job_history(tmp_path):
    (tmp_path/'web').mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        response=client.post('/api/jobs',content='{"mode":"tts","text":"Test","sampling":{"temperature":NaN}}',headers={'Content-Type':'application/json'})
        assert response.status_code==400
        assert client.get('/api/jobs').json()==[]


def test_queue_uses_an_immutable_json_snapshot_of_submission(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    with Studio(tmp_path) as studio:
        seen=[]
        def execute(job,payload,progress):
            if payload['mode']=='hold':entered.set();assert release.wait(10)
            else:seen.append(payload['nested']['text'])
            return {'ok':True}
        monkeypatch.setattr(studio,'execute',execute)
        studio.submit({'mode':'hold'})
        assert entered.wait(10)
        payload={'mode':'probe','nested':{'text':'Original'}}
        job=studio.submit(payload);payload['nested']['text']='Changed'
        release.set();studio.pending.join()
        assert seen==['Original']
        assert studio.get_job(job)['payload']['nested']['text']=='Original'


def test_invalid_result_json_does_not_terminate_worker_or_poison_history(tmp_path,monkeypatch):
    with Studio(tmp_path) as studio:
        original=studio.execute
        monkeypatch.setattr(studio,'execute',lambda job,payload,progress:{'bad':float('nan')} if payload['mode']=='bad-result' else original(job,payload,progress))
        bad=studio.submit({'mode':'bad-result'});recovered=studio.submit({'mode':'unload'})
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and studio.worker.is_alive() and studio.get_job(recovered)['status']!='done':time.sleep(.01)
        assert studio.worker.is_alive()
        assert studio.get_job(bad)['status']=='failed'
        assert studio.get_job(recovered)['status']=='done'


@pytest.mark.parametrize('broken,expected',[(False,404),(True,400)])
def test_missing_or_corrupted_saved_project_is_a_readable_api_error(tmp_path,broken,expected):
    (tmp_path/'web').mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        studio=client.app.state.studio;session=Session(takes=(take(),));studio.register_session(session)
        path=studio.data/'sessions'/f'{session.session_id}.safetensors'
        if broken:path.write_bytes(b'corrupted')
        else:path.unlink()
        response=client.get('/api/sessions/'+session.session_id)
        assert response.status_code==expected
        assert response.json()['detail']


def test_api_marks_saving_job_uncancelable_and_keeps_final_status_accurate(tmp_path,monkeypatch):
    (tmp_path/'web').mkdir();entered,release=threading.Event(),threading.Event()
    with TestClient(application.create_app(tmp_path)) as client:
        studio=client.app.state.studio
        monkeypatch.setattr(studio,'generate_session',lambda mode,payload,session,*a:session.append(take()))
        monkeypatch.setattr(studio.engine,'decode',lambda *a:{'waveform':torch.zeros(1,1,1000),'sample_rate':44100})
        original=module.sf.write
        def write(path,*a,**k):
            original(path,*a,**k)
            if isinstance(path,Path) and path.parent==studio.output:entered.set();assert release.wait(10)
        monkeypatch.setattr(module.sf,'write',write)
        job=client.post('/api/jobs',json={'mode':'tts','text':'Test'}).json()['job_id']
        try:
            assert entered.wait(10)
            assert client.get('/api/jobs/'+job).json()['cancelable'] is False
            assert client.post('/api/jobs/'+job+'/cancel').status_code==409
        finally:release.set()
        studio.pending.join()
        assert client.get('/api/jobs/'+job).json()['status']=='done'
