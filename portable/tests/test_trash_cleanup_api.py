"""Deletion UX persistence and validation; no model inference is needed."""
import json
import uuid

from fastapi.testclient import TestClient

from studio_app_loader import application
from drama3_t8.creative_studio import creative_manager
from drama3_t8.performance import SCHEMA


def app_at(path):
    (path / 'web').mkdir(exist_ok=True)
    return application.create_app(path)


def test_clear_mixed_stores_persists_and_rejects_stale_restore(tmp_path):
    app = app_at(tmp_path)
    with TestClient(app) as c:
        desk = creative_manager(app.state.studio)
        voice = c.post('/api/creative/voices', json={'name': '测试音色'}).json()
        project = c.post('/api/creative/projects', json={'name': '测试工程', 'lines': [{'role': '旁白', 'text': '你好'}]}).json()
        jid, sid = uuid.uuid4().hex, uuid.uuid4().hex
        with app.state.studio.connect() as db:
            db.execute('INSERT INTO jobs(id,status,payload,result,created,updated,progress) VALUES(?,?,?,?,?,?,?)',
                       (jid, 'done', json.dumps({'mode': 'generate'}), json.dumps({'session_id': sid}), 1, 1, 1))
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)', (sid, 'pinned.safetensors', '测试会话', 1))
        for kind, item in [('voice', voice['id']), ('project', project['project_id']), ('job', jid)]:
            assert c.request('DELETE', f'/api/creative/catalog/{kind}/{item}', json={}).status_code == 200
        rows = c.get('/api/creative/trash').json()
        assert len(rows) == 3
        first = c.post('/api/creative/trash/clear', json={'items': [rows[0], rows[0]]}).json()
        assert first == {'cleared': 1, 'errors': []}
        assert c.post('/api/creative/trash/restore', json=rows[0]).status_code == 400
        assert c.post('/api/creative/trash/clear', json={'all': True}).json() == {'cleared': 2, 'errors': []}
        assert c.get('/api/creative/trash').json() == []
        assert not desk.voices.list_voices() and not desk.workspace.list_projects()
        assert not app.state.studio.jobs() and not app.state.studio.sessions()
    with TestClient(app_at(tmp_path)) as c:
        assert c.get('/api/creative/trash').json() == []
        assert c.get('/api/creative/projects').json() == []
        assert c.get('/api/creative/library').json()['voices'] == []
        assert c.get('/api/jobs').json() == []


def test_clear_validates_before_mutation_and_reports_stale_items(tmp_path):
    with TestClient(app_at(tmp_path)) as c:
        voice = c.post('/api/creative/voices', json={'name': '待清理'}).json()
        c.request('DELETE', '/api/creative/catalog/voice/' + voice['id'], json={})
        row = c.get('/api/creative/trash').json()[0]
        for payload in ({'all': 'true'}, {'all': True, 'items': []}, {'items': [row, {'kind': 'voice', 'id': '../file'}]}, {'items': [{'kind': 'candidate', 'id': 'a'*32}]}):
            assert c.post('/api/creative/trash/clear', json=payload).status_code == 400
            assert len(c.get('/api/creative/trash').json()) == 1
        c.post('/api/creative/trash/restore', json=row)
        result = c.post('/api/creative/trash/clear', json={'items': [row]}).json()
        assert result['cleared'] == 0 and len(result['errors']) == 1
        assert c.get('/api/creative/library').json()['voices'][0]['id'] == voice['id']


def test_measurement_delete_only_removes_named_report_and_blocks_active_batch(tmp_path):
    app = app_at(tmp_path)
    with TestClient(app) as c:
        desk = creative_manager(app.state.studio)
        name = uuid.uuid4().hex
        path = desk.reports / (name + '.json')
        path.write_text(json.dumps({'schema': SCHEMA, 'status': 'done'}), encoding='utf-8')
        protected = desk.reports / 'profiles'
        protected.mkdir()
        (protected / 'keep.json').write_text('{}')
        assert c.get('/api/creative/performance').json()['reports'][0]['report_id'] == name
        assert c.delete('/api/creative/performance/not-a-report').status_code == 400
        assert path.exists()
        jid = uuid.uuid4().hex
        with app.state.studio.connect() as db:
            db.execute('INSERT INTO jobs(id,status,payload,created,updated,progress) VALUES(?,?,?,?,?,?)',
                       (jid, 'running', '{}', 1, 1, 0))
        batch = desk.reports / ('batch-' + jid + '.json')
        batch.write_text('{}')
        assert c.delete('/api/creative/performance/batch-' + jid).status_code == 400
        assert batch.exists()
        assert c.delete('/api/creative/performance/' + name).status_code == 200
        assert not path.exists() and (protected / 'keep.json').exists()


def test_active_project_clear_reports_failure_then_succeeds_when_idle(tmp_path):
    app = app_at(tmp_path)
    with TestClient(app) as c:
        project = c.post('/api/creative/projects', json={'name': '保护活动工程', 'lines': [{'role': '旁白', 'text': '你好'}]}).json()
        pid, jid = project['project_id'], uuid.uuid4().hex
        assert c.request('DELETE', '/api/creative/catalog/project/' + pid, json={}).status_code == 200
        with app.state.studio.connect() as db:
            db.execute('INSERT INTO jobs(id,status,payload,created,updated,progress) VALUES(?,?,?,?,?,?)',
                       (jid, 'running', json.dumps({'project_id': pid}), 1, 1, 0))
        result = c.post('/api/creative/trash/clear', json={'all': True}).json()
        assert result['cleared'] == 0 and len(result['errors']) == 1
        assert len(c.get('/api/creative/trash').json()) == 1
        with app.state.studio.connect() as db:
            db.execute('UPDATE jobs SET status=? WHERE id=?', ('done', jid))
        assert c.post('/api/creative/trash/clear', json={'all': True}).json() == {'cleared': 1, 'errors': []}


def test_clearing_one_job_keeps_shared_session_until_last_owner_deleted(tmp_path):
    app = app_at(tmp_path)
    with TestClient(app) as c:
        jid1, jid2, sid = [uuid.uuid4().hex for _ in range(3)]
        with app.state.studio.connect() as db:
            for jid in [jid1, jid2]:
                db.execute('INSERT INTO jobs(id,status,payload,result,created,updated,progress) VALUES(?,?,?,?,?,?,?)',
                           (jid, 'done', '{}', json.dumps({'session_id': sid}), 1, 1, 1))
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)', (sid, 'shared.safetensors', '共享会话', 1))
        for jid, visible in [(jid1, True), (jid2, False)]:
            c.request('DELETE', '/api/creative/catalog/job/' + jid, json={})
            assert c.post('/api/creative/trash/clear', json={'all': True}).json()['cleared'] == 1
            assert bool(app.state.studio.sessions()) == visible
        assert app.state.studio.get_job(jid1)['status'] == 'done'
