"""Typed CPU source graphs roundtrip without putting PCM in JSON metadata."""
from dataclasses import replace
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from drama3_t8.state import Take,Session,save_session,load_session,validate_session


def take():
    frames=torch.ones((3,11),dtype=torch.int64);frames[:,0]=248100
    return Take('actual complete text','actual complete text',frames)


def test_typed_source_graph_deduplicates_shared_sources_and_preserves_exact_pcm_and_frames(tmp_path):
    parent=take();middle=take()
    rendered=replace(take(),blobs={'render_pcm':torch.arange(100,dtype=torch.float32)[None]/100,
        'prefix_codes':torch.ones((10,0),dtype=torch.int64)},sources=(parent,middle))
    later=replace(take(),blobs={'render_pcm':torch.zeros((1,200),dtype=torch.float32)},sources=(rendered,parent))
    session=Session(takes=(parent,rendered,later));path=tmp_path/'typed.safetensors';save_session(session,path)
    with safe_open(str(path),framework='pt') as file:
        metadata=json.loads(file.metadata()['drama3_t8_session'])
        assert metadata['version']==2 and metadata['features']==['typed_takes_v1']
        assert len(metadata['source_takes'])==1 and metadata['source_takes'][0]['take_id']==middle.take_id
        assert 'render_pcm' not in json.dumps(metadata['takes'][1]['metadata'])
        assert file.get_tensor(metadata['takes'][1]['blobs']['render_pcm']).dtype==torch.float32
    restored=load_session(path)
    assert restored.takes[-1].sources[0].take_id==rendered.take_id
    assert torch.equal(restored.takes[1].blobs['render_pcm'],rendered.blobs['render_pcm'])
    assert torch.equal(restored.takes[1].sources[1].frames,middle.frames)
    save_session(restored,tmp_path/'again.safetensors')


def test_shared_id_conflicts_cycles_and_nonfinite_blobs_rejected_before_save(tmp_path):
    original=take();other=replace(original,metadata={'changed':True})
    cases=[Session(takes=(original,other)),Session(takes=(replace(take(),blobs={'pcm':torch.tensor([float('nan')])}),))]
    cycle=take();object.__setattr__(cycle,'sources',(cycle,));cases.append(Session(takes=(cycle,)))
    child=take();duplicate_child=replace(child,metadata={'changed':True})
    root=replace(take(),sources=(child,));duplicate_root=replace(root,sources=(duplicate_child,))
    cases.append(Session(takes=(root,duplicate_root)))
    positive=replace(take(),blobs={'pcm':torch.zeros((1,2),dtype=torch.float32)})
    negative=replace(positive,blobs={'pcm':torch.tensor([[-0.,0.]],dtype=torch.float32)})
    cases.append(Session(takes=(positive,negative)))  # Float data identity includes the sign bit of zero.
    for index,session in enumerate(cases):
        path=tmp_path/f'bad-{index}.safetensors'
        with pytest.raises(ValueError):save_session(session,path)
        assert not path.exists()


@pytest.mark.parametrize('change',['missing_blob','cycle','unknown_source','unused_source','unknown_feature'])
def test_actual_typed_archive_loader_rejects_corrupt_graph_references(tmp_path,change):
    parent=take();root=replace(take(),sources=(parent,),blobs={'pcm':torch.zeros((1,100),dtype=torch.float32)})
    path=tmp_path/'valid.safetensors';save_session(Session(takes=(root,)),path)
    with safe_open(str(path),framework='pt') as file:
        metadata=json.loads(file.metadata()['drama3_t8_session']);tensors={key:file.get_tensor(key) for key in file.keys()}
    if change=='missing_blob':metadata['takes'][0]['blobs']['pcm']='missing'
    elif change=='cycle':metadata['source_takes'][0]['sources']=[root.take_id]
    elif change=='unknown_source':metadata['takes'][0]['sources']=['f'*32]
    elif change=='unused_source':metadata['takes'][0]['sources']=[]
    else:metadata['features'].append('future_codec_feature')
    bad=tmp_path/'bad.safetensors';save_file(tensors,str(bad),metadata={'drama3_t8_session':json.dumps(metadata)})
    with pytest.raises(ValueError):load_session(bad)
