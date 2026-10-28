"""Real save/load archives expose reusable references without shifting Session slot 0."""
import torch
import hashlib
import json
from pathlib import Path
from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import folder_paths
from drama3_t8 import nodes
from drama3_t8.state import Reference,Session


def test_reference_only_session_roundtrips_through_actual_native_nodes_and_scene(tmp_path,monkeypatch):
    library=tmp_path/'voices';library.mkdir()
    monkeypatch.setattr(nodes,'SESSIONS',library)
    monkeypatch.setitem(folder_paths.folder_names_and_paths,'drama3_t8_sessions',([str(library)],{'.safetensors'}))
    references=(Reference('<|speaker:0|> Hello.',torch.ones(10,4,dtype=torch.int64),'Display A'),
                Reference('<|speaker:1|> Hi.',torch.full((10,5),2,dtype=torch.int64),'Display B'))
    original=Session(references=references,direction='Keep the original voices')
    nodes.T8DramaSaveSession.execute(original,'voices')
    archive=next(library.glob('*.safetensors'))
    loaded=nodes.T8DramaLoadSession.execute(archive.name)
    session,restored=loaded[0],loaded[1]
    assert session.session_id==original.session_id and session.takes==()
    assert restored is session.references and len(restored)==2
    assert [r.transcript for r in restored]==[r.transcript for r in references]
    assert all(torch.equal(a.codes,b.codes) for a,b in zip(restored,references))
    info=nodes.T8DramaLoadSession.GET_NODE_INFO_V1()
    assert info['output']==['DRAMA3_T8_SESSION','DRAMA3_T8_REFERENCES']
    observed={}
    class Engine:
        def scene(self,lines,scene,characters,references,sampling,**kwargs):
            observed.update(lines=lines,references=references)
            return Session(references=references)
    result=nodes.T8DramaScene.execute(Engine(),'A: New line.\nB: Answer.',references=restored)
    assert result[0].references is restored and observed['references'] is restored
    # Existing workflows connect output 0 to an input expecting Session. This
    # exact socket remains unchanged; the new references socket is appended.
    legacy_link=[1,10,0,20,0,'DRAMA3_T8_SESSION']
    assert info['output'][legacy_link[2]]==legacy_link[5]
    assert nodes.T8DramaSession.execute(references=restored)[0].references is restored


def test_actual_prior_editor_canvas_session_link_keeps_output_zero_and_type():
    fixture=Path(__file__).parent/'fixtures/load-session-output0-editor.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='47b3930516325d097d6d070662ca78c4e7c3df8ef5af81c9fdfcf4fcf27d8e60'
    data=json.loads(fixture.read_text(encoding='utf-8'))
    source=next(node for node in data['nodes'] if node['type']=='T8DramaLoadSession')
    assert len(source['outputs'])==1  # Archive predates the references output.
    by_id={node['id']:node for node in data['nodes']}
    link=next(link for link in data['links'] if link[1]==source['id'])
    assert link[2]==0 and link[5]=='DRAMA3_T8_SESSION'
    target=by_id[link[3]]
    assert target['inputs'][link[4]]['name']=='session'
    assert nodes.T8DramaLoadSession.GET_NODE_INFO_V1()['output'][0]==link[5]
    assert nodes.T8DramaEditor.GET_NODE_INFO_V1()['input']['required']['session'][0]==link[5]
