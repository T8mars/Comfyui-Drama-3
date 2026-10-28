"""Public Engine/Node/Studio/editor flow with actual CPU prompts and typed archives."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from test_scene_chunks import Harness, SCRIPT, references
from drama3_t8.engine import DramaEngine,Sampling
from drama3_t8.state import Session,save_session,load_session,validate_session
from drama3_t8 import editor,nodes,render_infill
from drama3_t8.scene_cast import prepare_scene_cast
from drama3_t8.scene_chunks import history_view
from drama3_t8.prompts import history_segments,FrameRows
from drama3_t8.studio import Studio,playback_identity


class CanvasEngine(Harness):
    def decode(self,codes):
        audio=super().decode(codes)
        return {**audio,'waveform':audio['waveform']/10000}
    def load_codec(self,*,memory_required=0):
        def encode(wave):
            padded=F.pad(wave,(0,-wave.shape[-1]%2048))
            codes=(padded.unfold(-1,2048,2048).mean(-1)*100+1000).round().long().clamp(0,4095)
            return codes[:,None].expand(-1,10,-1).contiguous()
        return SimpleNamespace(patcher=SimpleNamespace(load_device=torch.device('cpu')),model=SimpleNamespace(encode=encode))
    def unload(self):pass


def initial(engine=None):
    engine=engine or CanvasEngine()
    session=engine.scene(SCRIPT,references=references(),generation_mode='per_speaker',split_long_text=True,chunk_bytes=100)
    _,timeline=engine.decode_take(session.takes[-1],with_timeline=True)
    return engine,session,timeline


def infill_node(engine,session,timeline):
    return nodes.T8DramaInfill.execute(engine,session,'\n'.join(SCRIPT),timeline[1]['start']+.001,
        timeline[1]['start']+.003,target_speaker='1',seed=71,max_frames=80)[0]


def test_public_node_infill_decode_editor_and_real_archive_roundtrip(tmp_path,monkeypatch):
    import folder_paths
    engine,parent,timeline=initial()
    session=infill_node(engine,parent,timeline);take=session.takes[-1]
    assert take.metadata['render_infill']['subtype']=='FLOAT'
    audio=nodes.T8DramaDecode.execute(engine,session)[0]
    assert torch.equal(audio['waveform'][0],take.blobs['render_pcm'])
    assert audio['waveform'].abs().max()>1  # No implicit clipping in native AUDIO.
    wave,analysis,edits=editor.prepare(engine,take)
    assert not analysis['aligned'] and all(line['needs_alignment'] and line['text']=='' for line in analysis['lines'])
    assert editor.subtitle_srt(editor.captions(analysis,editor.plan(analysis,edits)['pieces'],edits))==''
    calls=[]
    def align(*args):
        calls.append(args)
        return [dict(text=f'word{i}',start=row['start']+.00001,end=min(row['end'],row['start']+.0001))
            for i,row in enumerate(analysis['lines'])]
    import drama3_t8.alignment as alignment
    monkeypatch.setattr(alignment,'align_audio',align)
    _,aligned,_=editor.prepare(engine,take,aligner='configured-native.safetensors')
    assert calls[0][3]==take.transcript and aligned['aligned']
    assert all(not line['needs_alignment'] for line in aligned['lines'])
    changed=editor.edited_session(session,-1,analysis,{**edits,'speakers':{'1':{'gain_db':6}}})
    assert changed.takes[-1].take_id!=take.take_id and torch.equal(changed.takes[-1].frames,take.frames)
    library=tmp_path/'voices';library.mkdir()
    monkeypatch.setattr(nodes,'SESSIONS',library)
    monkeypatch.setitem(folder_paths.folder_names_and_paths,'drama3_t8_sessions',([str(library)],{'.safetensors'}))
    nodes.T8DramaSaveSession.execute(changed,'rendered')
    restored=nodes.T8DramaLoadSession.execute(next(library.glob('*.safetensors')).name)[0]
    assert torch.equal(nodes.T8DramaDecode.execute(engine,restored)[0]['waveform'],editor.decode_output(engine,changed.takes[-1])['waveform'])
    assert torch.equal(parent.takes[0].frames,restored.takes[0].frames)


def test_render_only_reload_cast_reorder_second_infill_and_scene_revision_strategy(tmp_path):
    engine,parent,timeline=initial();first=infill_node(engine,parent,timeline).takes[-1]
    path=tmp_path/'only.safetensors';save_session(Session(references=references(),takes=(first,)),path)
    restored=load_session(path)
    cast,_,mapping=prepare_scene_cast(['B: reorder','A: stable'],restored.references,restored.takes,force=True)
    assert mapping=={'A':0,'B':1} and cast[0].startswith('<|speaker:1|>') and cast[1].startswith('<|speaker:0|>')
    second=engine.infill(restored.takes[0],'B: reorder\nA: stable',.001,.002,Sampling(seed=8),target_speaker=0)
    save_session(Session(references=references(),takes=(second,)),path);again=load_session(path)
    cast,_,mapping=prepare_scene_cast(['A: returns','B: same ID'],again.references,again.takes,force=True)
    assert mapping=={'A':0,'B':1} and cast[1].startswith('<|speaker:1|>')
    revised=engine.revise(again,'more cheerful',Sampling(seed=9,max_frames=80))
    last=revised.takes[-1]
    assert last.metadata['script_lines']==['B: reorder','A: stable']
    assert last.metadata['generation_mode']=='per_speaker'
    assert last.metadata['scene_chunking']['chunk_bytes']==100 and last.metadata['scene_speaker_map']==mapping


def test_all_history_views_use_full_actual_pcm_encoding_and_never_raw_middle():
    engine,parent,timeline=initial();rendered=infill_node(engine,parent,timeline).takes[-1]
    view=history_view((rendered,))
    assert len(view)==1 and view[0].metadata['history_actual_pcm']
    assert torch.equal(view[0].frames,rendered.frames)
    assert not torch.equal(view[0].frames,rendered.sources[1].frames)
    segments=history_segments((rendered,))
    frames=[entry.values for entry in segments if isinstance(entry,FrameRows)]
    assert len(frames)==1 and torch.equal(frames[0],rendered.frames)
    assert rendered.metadata['render_infill']['history_text'] in segments
    assert history_view((rendered.sources[1],))==()  # Raw FIM full-target text is not middle text.


def test_invalid_imported_render_map_and_conflicting_scene_identity_rejected(tmp_path):
    engine,parent,timeline=initial();rendered=infill_node(engine,parent,timeline).takes[-1]
    bad=replace(rendered,metadata={**rendered.metadata,'scene_speaker_map':{'A':99}})
    with pytest.raises(ValueError):save_session(Session(takes=(bad,)),tmp_path/'bad.safetensors')
    conflict=replace(rendered,metadata={**rendered.metadata,'scene_speaker_map':{'A':1,'B':0}})
    with pytest.raises(ValueError):prepare_scene_cast(['A: line'],references(),(parent.takes[-1],conflict),force=True)


@pytest.mark.parametrize('subtype',['FLOAT','PCM_16'])
def test_studio_real_queue_infill_writes_selected_format_and_identity(tmp_path,subtype):
    engine,parent,timeline=initial()
    selected=engine.decode_take(parent.takes[-1])
    identity=playback_identity(parent.takes[-1],selected,subtype)
    with Studio(tmp_path/'studio') as studio:
        studio.engine=engine;studio.register_session(parent)
        job=studio.submit(dict(mode='infill',session_id=parent.session_id,take_index=-1,text='\n'.join(SCRIPT),
            start_seconds=timeline[1]['start']+.001,end_seconds=timeline[1]['start']+.003,target_speaker=1,
            expected_view=identity,sampling={'seed':88,'max_frames':80}))
        studio.pending.join();result=studio.get_job(job)
        assert result['status']=='done',result['error']
        data=result['result'];path=studio.output/data['audio'].rsplit('/',1)[-1]
        assert sf.info(path).subtype==subtype and data['playback_view']['subtype']==subtype
        saved=studio.get_session(data['session_id'])
        actual=torch.from_numpy(sf.read(path,dtype='float32',always_2d=True)[0].T.copy())
        assert torch.equal(actual,saved.takes[-1].blobs['render_pcm'])
        assert data['playback_view']==playback_identity(saved.takes[-1],dict(waveform=actual[None],sample_rate=44100),subtype)
        if subtype=='FLOAT':assert actual.abs().max()>1


def test_node_infill_old_schema_prefix_and_invalid_expected_view_preserves_parent():
    info=nodes.T8DramaInfill.GET_NODE_INFO_V1()
    assert list(info['input']['optional'])[-2:]==['target_speaker','expected_view']
    engine,parent,timeline=initial();before=parent.takes[-1].frames.clone()
    for kwargs in ({'target_speaker':'16'},{'expected_view':'{"x":NaN}'}):
        with pytest.raises(ValueError):nodes.T8DramaInfill.execute(engine,parent,'full script',0,.01,**kwargs)
    assert torch.equal(parent.takes[-1].frames,before)
