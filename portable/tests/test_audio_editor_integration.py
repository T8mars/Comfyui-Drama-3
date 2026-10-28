import json
import uuid
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from studio_app_loader import application
from drama3_t8 import editor
from drama3_t8.state import Session,Take,load_session


def make_engine():
    wave=torch.sin(torch.arange(40960)*.03).reshape(1,1,-1)*.25
    audio=dict(waveform=wave,sample_rate=44100)
    return SimpleNamespace(decode_take=lambda take,with_timeline=False:(audio,None) if with_timeline else audio,
        unload=lambda:None),audio


def make_session():
    frames=torch.ones(20,11,dtype=torch.int64);frames[:,0]=248100
    return Session(takes=(Take('Cat meeting','Cat meeting',frames),))


def test_studio_queue_prepare_cpu_plan_render_and_restore(tmp_path):
    (tmp_path/'web').mkdir()
    with TestClient(application.create_app(tmp_path)) as client:
        studio=client.app.state.studio
        studio.engine,audio=make_engine()
        original=make_session();studio.register_session(original)
        submitted=client.post('/api/jobs',json=dict(mode='editor_prepare',session_id=original.session_id,take_index=-1)).json()
        studio.pending.join()
        job=client.get('/api/jobs/'+submitted['job_id']).json()
        assert job['status']=='done',job.get('error')
        source=job['result']['editor'];assert source['take_index']==0
        edits={**source['edits'],'cuts':[[0,.2]],'inserts':[[.3,.1]],'speakers':{'0':{'gain_db':-6}}}
        planned=client.post('/api/editor/plan',json={'source_id':source['source_id'],'edits':edits})
        assert planned.status_code==200,planned.text
        payload=dict(source_id=source['source_id'],session_id=original.session_id,take_index=0,edits=edits)
        result=client.post('/api/editor/render',json=payload)
        assert result.status_code==200,result.text
        result=result.json()
        assert result['take_index']==0
        assert result['session_id']!=original.session_id
        rendered,rate=sf.read(studio.output/result['audio'].rsplit('/',1)[-1],dtype='float32')
        assert len(rendered)==planned.json()['samples']
        saved=load_session(studio.data/'sessions'/f"{result['session_id']}.safetensors")
        assert saved.takes[0].metadata['audio_editor']['edits']=={**edits,'source':editor.fingerprint(saved.takes[0])}
        assert saved.takes[0].metadata['editor_parent_take_id']==original.takes[0].take_id
        assert 'audio_editor' not in studio.get_session(original.session_id).takes[0].metadata
        assert client.get(result['subtitle']).status_code==200
        assert client.post('/api/editor/render',json={**payload,'session_id':'b'*32}).status_code==400
        assert client.post('/api/editor/plan',json={'source_id':'../bad','edits':edits}).status_code==400
        assert client.post('/api/editor/plan',json={'source_id':source['source_id'],'edits':{**edits,'source':'b'*64}}).status_code==400
        assert client.post('/api/editor/plan',json={'source_id':source['source_id'],'edits':{'cuts':[[0,.1]]}}).status_code==400


def test_native_editor_outputs_session_audio_and_serializable_preview(tmp_path,monkeypatch):
    from drama3_t8 import nodes
    monkeypatch.setattr(nodes.folder_paths,'get_temp_directory',lambda:str(tmp_path))
    engine,audio=make_engine();session=make_session()
    initial=nodes.T8DramaEditor.execute(engine,session)
    entry=initial.ui['drama3_t8_editor'][0]
    assert torch.equal(initial[1]['waveform'],audio['waveform'])
    assert (tmp_path/entry['subfolder']/entry['audio']).is_file()
    edits={**entry['edits'],'cuts':[[0,.2]]}
    changed=nodes.T8DramaEditor.execute(engine,session,edits=json.dumps(edits))
    assert changed[1]['waveform'].shape[-1]==audio['waveform'].shape[-1]-8820
    assert changed[0].takes[0].metadata['audio_editor']['source']==editor.fingerprint(changed[0].takes[0])
    assert changed[0].takes[0].take_id!=session.takes[0].take_id and 'audio_editor' not in session.takes[0].metadata
    # A different seed is a fresh source, never an old cut accidentally applied.
    new=make_session()
    reset=nodes.T8DramaEditor.execute(engine,new,edits=json.dumps(edits))
    assert reset.ui['drama3_t8_editor'][0]['edits_reset'] is True
    assert torch.equal(reset[1]['waveform'],audio['waveform'])
    old_long={**edits,'cuts':[[0,100.]]}
    reset=nodes.T8DramaEditor.execute(engine,new,edits=json.dumps(old_long))
    assert torch.equal(reset[1]['waveform'],audio['waveform'])
    decoded=nodes.T8DramaDecode.execute(engine,changed[0])
    assert torch.equal(decoded[0]['waveform'],changed[1]['waveform'])


@pytest.mark.parametrize('edits',['[]','{"normalize":NaN}','{"source":"a"}','{"junk":0}'])
def test_native_rejects_bad_editor_json_before_publishing(tmp_path,monkeypatch,edits):
    from drama3_t8 import nodes
    monkeypatch.setattr(nodes.folder_paths,'get_temp_directory',lambda:str(tmp_path))
    engine,_=make_engine()
    with pytest.raises(ValueError):nodes.T8DramaEditor.execute(engine,make_session(),edits=edits)
    assert not list(tmp_path.iterdir())


def test_comfy_plan_route_is_same_process_and_rejects_nonfinite(monkeypatch):
    import asyncio
    from aiohttp import web
    from drama3_t8.editor_routes import register_routes
    import server
    routes=web.RouteTableDef()
    monkeypatch.setattr(server.PromptServer,'instance',SimpleNamespace(routes=routes),raising=False)
    register_routes();register_routes()
    assert len(routes)==1
    handler=next(iter(routes)).handler
    class Request:
        content_length=30
        async def text(self):return '{"analysis": {}, "edits":{"normalize":NaN}}'
    response=asyncio.run(handler(Request()))
    assert response.status==400


def test_studio_alignment_uses_edited_word_times_without_realigning_full_original(tmp_path,monkeypatch):
    from drama3_t8.studio import Studio
    import drama3_t8.alignment as alignment
    monkeypatch.setattr(alignment,'align_audio',lambda *args:pytest.fail('Must reuse mapped word timings'))
    with Studio(tmp_path) as studio:
        studio.engine,audio=make_engine();session=make_session()
        _,analysis,edits=editor.prepare(studio.engine,session.takes[0])
        analysis['aligned']=True
        analysis['lines'][0]['tokens']=[dict(text='Cat',spoken=True,start=.1,end=.3)]
        edits={**edits,'cuts':[[0,.05]],'inserts':[[.05,.2]]}
        branch=editor.edited_session(session,0,analysis,edits);studio.register_session(branch)
        job=studio.submit(dict(mode='align',session_id=branch.session_id,take_index=0))
        studio.pending.join();result=studio.get_job(job)
        assert result['status']=='done',result.get('error')
        assert result['result']['alignment'][0]['start']==pytest.approx(.25)
        assert result['result']['alignment'][0]['end']==pytest.approx(.45)


def test_studio_unaligned_edited_audio_requires_editor_word_alignment(tmp_path):
    from drama3_t8.studio import Studio
    with Studio(tmp_path) as studio:
        studio.engine,_=make_engine();session=make_session()
        _,analysis,edits=editor.prepare(studio.engine,session.takes[0])
        branch=editor.edited_session(session,0,analysis,edits);studio.register_session(branch)
        job=studio.submit(dict(mode='align',session_id=branch.session_id,take_index=0))
        studio.pending.join();result=studio.get_job(job)
        assert result['status']=='failed' and '多轨编辑器' in result['error']
