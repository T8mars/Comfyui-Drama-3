"""Catalog deletion preserves media/context and respects frozen selections."""
import json
import sqlite3
import threading
import uuid

import pytest

from drama3_t8.creative_workspace import CreativeWorkspace, ProjectConflict
from drama3_t8.voice_library import VoiceLibrary
from test_creative_workspace import media


def made(tmp_path, mode="per_line_context"):
    workspace=CreativeWorkspace(tmp_path/'workspace')
    project=workspace.create_project('演示',[{'line_id':'L01','role':'演员','text':'第一句'}, {'line_id':'L02','role':'演员','text':'第二句'}])
    workspace.run_generation(project['project_id'],media,takes_per_line=2,mode=mode)
    return workspace,workspace.get_project(project['project_id'])


def test_new_voice_and_cast_first_and_restorable(tmp_path):
    lib=VoiceLibrary(tmp_path/'voices')
    old=lib.create_voice('A最早');new=lib.create_voice('Z最新')
    assert [r['id'] for r in lib.list_voices()]==[new['id'],old['id']]
    ids=[uuid.uuid4().hex for _ in range(2)]
    with lib._db() as db:
        db.execute('INSERT INTO casts VALUES(?,?,?,?)',(ids[0],'A最早',json.dumps({'bindings':{}}),1))
        db.execute('INSERT INTO casts VALUES(?,?,?,?)',(ids[1],'Z最新',json.dumps({'bindings':{}}),2))
    assert [r['id'] for r in lib.list_casts()]==ids[::-1]
    lib.trash_item('cast',ids[1]);assert [r['id'] for r in lib.list_casts()]==ids[:1]
    assert lib.get_cast(ids[1])['name']=='Z最新'
    lib.trash_item('cast',ids[1],restore=True);assert lib.list_casts()[0]['id']==ids[1]
    lib.trash_item('voice',new['id']);assert [r['id'] for r in lib.list_voices()]==[old['id']]
    lib.trash_item('voice',new['id'],restore=True);assert lib.list_voices()[0]['id']==new['id']


def test_deleted_candidate_not_reused_media_and_context_retained(tmp_path):
    w,p=made(tmp_path);pid=p['project_id'];line=p['lines'][0];cid=line['selected_candidate_id']
    raw=w._read(pid);paths={a['path']:(w._directory(pid)/a['path']).read_bytes() for a in raw['candidates'][cid]['artifacts']}
    dependent=[c for c in raw['candidates'].values() if cid in c.get('context_candidate_ids',[])]
    assert dependent
    w.trash_item(pid,cid,expected_revision=p['revision'])
    visible=w.get_project(pid);assert cid not in visible['candidates'];assert visible['lines'][0]['selected_candidate_id'] is None
    assert all((w._directory(pid)/name).read_bytes()==data for name,data in paths.items())
    assert all(w._usable(pid,w._read(pid)['candidates'][c['candidate_id']]) for c in dependent)
    plan=w.plan_generation(pid,takes_per_line=2,mode='per_line_context')
    assert not any(job.get('candidate',{}).get('candidate_id')==cid for job in plan['jobs'] if job.get('candidate'))
    assert plan['pending']>=1
    w.trash_item(pid,cid,restore=True)
    assert cid in w.get_project(pid)['candidates']
    assert w.get_project(pid)['lines'][0]['selected_candidate_id'] is None


def test_locked_or_stale_delete_leaves_project_unchanged(tmp_path):
    w,p=made(tmp_path);pid=p['project_id'];cid=p['lines'][0]['selected_candidate_id']
    w.set_candidate_review(pid,'L01',cid,selected=True,locked=True)
    before=(w._directory(pid)/'project.json').read_bytes()
    with pytest.raises(ValueError,match='解锁'):w.trash_item(pid,cid)
    with pytest.raises(ProjectConflict):w.trash_item(pid,cid,expected_revision=p['revision'])
    assert before==(w._directory(pid)/'project.json').read_bytes()


def test_scene_delete_clears_all_selected_lines_and_restores_once(tmp_path):
    w,p=made(tmp_path,'together');pid=p['project_id'];cid=p['lines'][0]['selected_candidate_id']
    assert all(line['selected_candidate_id']==cid for line in p['lines'])
    w.trash_item(pid,cid);assert all(line['selected_candidate_id'] is None for line in w.get_project(pid)['lines'])
    assert len(w.trash_items())==1
    w.trash_item(pid,cid,restore=True);assert not w.trash_items()


def test_project_delete_restore_preserves_files_and_rejects_generation(tmp_path):
    w,p=made(tmp_path);pid=p['project_id'];candidate_files=list(w._directory(pid).glob('artifacts/*/*.wav'))
    w.trash_item(pid);assert not w.list_projects()
    assert all(file.exists() for file in candidate_files)
    with pytest.raises(FileNotFoundError):w.get_project(pid)
    with pytest.raises(FileNotFoundError):w.plan_generation(pid)
    assert w.trash_items()[0]['kind']=='project'
    w.trash_item(pid,restore=True);assert w.list_projects()[0]['project_id']==pid


def test_deleted_version_still_exports_exact_pinned_reference(tmp_path):
    import torch
    from drama3_t8.state import Reference,Session
    lib=VoiceLibrary(tmp_path/'voices');v=lib.create_voice('演员')
    version=lib.add_version(v['id'],Session((Reference('你好',torch.ones((10,4),dtype=torch.int16)),)))
    original=lib.session_file(version['id']).read_bytes()
    lib.trash_item('version',version['id'])
    assert lib.get_voice(v['id'])['versions']==[]
    assert lib.session_file(version['id']).read_bytes()==original
    assert lib.export_bundle(tmp_path/'pinned.zip',version_ids=[version['id']],cast_ids=[])['versions']==1
    lib.trash_item('version',version['id'],restore=True)
    assert lib.get_voice(v['id'])['latest_version_id']==version['id']


def test_completed_job_delete_restore_and_cached_records(tmp_path):
    from drama3_t8.studio import Studio
    from drama3_t8.catalog_trash import initialize
    s=Studio.__new__(Studio);s.database=tmp_path/'studio.sqlite3';s._activity_lock=threading.RLock();s._queue_error=None;s.cancel_events={};s._job_records={}
    sid=uuid.uuid4().hex;jid=uuid.uuid4().hex
    with s.connect() as db:
        db.executescript('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,payload TEXT,result TEXT,error TEXT,created REAL,updated REAL,progress REAL);CREATE TABLE sessions(id TEXT PRIMARY KEY,path TEXT,summary TEXT,updated REAL);')
        initialize(db);db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',(jid,'done',json.dumps({'mode':'generate'}),json.dumps({'session_id':sid}),None,1,1,1))
        db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(sid,'unchanged.safetensors','生成',1))
    s._job_records[jid]={'id':jid,'status':'done','payload':json.dumps({'mode':'generate'}),'result':json.dumps({'session_id':sid}),'error':None,'created':1,'updated':1,'progress':1}
    s.trash_item('job',jid);assert not s.jobs();assert not s.sessions();assert s.get_job(jid)['status']=='done'
    s.trash_item('job',jid,restore=True);assert s.jobs()[0]['id']==jid;assert s.sessions()[0]['id']==sid
    with s.connect() as db:db.execute('UPDATE jobs SET status=? WHERE id=?',('running',jid))
    with pytest.raises(ValueError,match='尚未结束'):s.trash_item('job',jid)
