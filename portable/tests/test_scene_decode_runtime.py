"""Exercise real sampling/control plumbing on CPU; this is not model inference."""
import contextlib
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native.generate import generate


def scripted_drama(monkeypatch,sequence):
    import drama3_t8.native.generate as module
    monkeypatch.setattr(module.comfy.model_management,'cuda_device_context',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(module.comfy.model_management,'throw_exception_if_processing_interrupted',lambda:None)
    ids=(198,248046,248100,248101,252196)
    calls=dict(backbone=[],controls=[])
    def embed(tokens,out_dtype):
        calls['backbone'].extend(tokens.tolist())
        return torch.zeros(len(tokens),4)
    def head(hidden):
        result=torch.full((1,len(ids)),-100.)
        result[0,ids.index(sequence.pop(0))]=100.
        return result
    def tower_run(hidden,labels,*args):
        calls['controls'].extend(labels[:,0].tolist())
    tower=SimpleNamespace(init_caches=lambda *a:[],embed_backbone=lambda frames,text:text,
        run=tower_run,context=lambda *a:None,is_semantic=lambda x:(x>=248099)&(x<=252194),
        step=lambda main,*args:torch.tensor([[main.item(),*([1]*10)]]))
    backbone=SimpleNamespace(embed_tokens=embed,init_caches=lambda *a:[],run_layers=lambda x,*a:x)
    drama=SimpleNamespace(config=SimpleNamespace(output_token_ids=ids,eos_token_id=248046,
        semantic_start_token_id=248099,semantic_end_token_id=252194,text=SimpleNamespace(max_position_embeddings=64)),
        model=SimpleNamespace(model=backbone,dtype=torch.float32),experts=None,
        tokenizer=SimpleNamespace(token_to_id=lambda s:252196,encode=lambda s:SimpleNamespace(ids=[198])),
        parts=lambda:([None],[torch.device('cpu')],lambda x:x,head,tower))
    return drama,calls


def test_early_eos_recovery_enters_real_backbone_and_tower_before_natural_separator(monkeypatch):
    drama,calls=scripted_drama(monkeypatch,[248100,248046,252196,198,248101,248046])
    correction=[]
    finished=[]
    audio=[]
    rows=generate(drama,torch.zeros(2,11,dtype=torch.long),12,.7,1,1.,47,
        expected_segments=2,on_boundary=correction.append,on_finish=finished.append,on_audio=audio.append)
    assert rows[:,0].tolist()==[248100,198,252196,198,248101]
    assert torch.count_nonzero(rows[1:4,1:])==0
    assert calls['backbone']==[0,0,248100,198,252196,198,248101]
    assert calls['controls']==[0,198,252196,198]
    assert finished==['eos'] and len(audio)==2
    assert [(c['raw_token_id'],c['emitted_token_id']) for c in correction]==[(248046,198)]


def test_non_scene_generation_keeps_original_eos_behavior(monkeypatch):
    drama,calls=scripted_drama(monkeypatch,[248100,248046])
    finished=[]
    rows=generate(drama,torch.zeros(2,11,dtype=torch.long),12,.7,1,1.,47,on_finish=finished.append)
    assert rows[:,0].tolist()==[248100] and finished==['eos']
    assert calls['controls']==[0]


def test_boundary_recovery_cannot_extend_the_total_frame_budget(monkeypatch):
    drama,calls=scripted_drama(monkeypatch,[248100,248046])
    finished=[]
    rows=generate(drama,torch.zeros(2,11,dtype=torch.long),2,.7,1,1.,47,
        expected_segments=2,on_finish=finished.append)
    assert rows[:,0].tolist()==[248100,198] and finished==['length']
