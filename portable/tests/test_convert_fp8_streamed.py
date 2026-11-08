"""Convert a real tiny checkpoint and verify lossless bytes and matched heads."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from drama3_t8.conversion.fp8 import convert_fp8_streamed
from convert_model import source_revision


def fixture(tmp_path,monkeypatch):
    from drama3_t8.conversion import fp8 as module
    tokenizer=SimpleNamespace(token_to_id=lambda token:{'<|audio_separator|>':4,'<|endofturn|>':5}[token],
        encode=lambda text:SimpleNamespace(ids=[3]))
    monkeypatch.setattr(module.Tokenizer,'from_file',lambda path:tokenizer)
    source=tmp_path/'source';source.mkdir()
    config={'eos_token_id':0,'semantic_start_token_id':1,'semantic_end_token_id':2,
        'text_config':{'num_hidden_layers':1,'num_experts':2}}
    (source/'config.json').write_text(json.dumps(config),encoding='utf-8')
    tensors={'lm_head.weight':torch.arange(32,dtype=torch.bfloat16).reshape(8,4),
        'model.language_model.embed_tokens.weight':torch.randn(8,4,dtype=torch.bfloat16),
        'model.language_model.layers.0.input_layernorm.weight':torch.randn(4,dtype=torch.bfloat16),
        'audio_decoder.codebook_embeddings.weight':torch.randn(6,4,dtype=torch.bfloat16)}
    for proj in ['gate_proj','up_proj','down_proj']:
        tensors['model.language_model.layers.0.mlp.shared_expert.'+proj+'.weight']=torch.randn(4,4,dtype=torch.bfloat16)
        for expert in range(2):
            base=f'model.language_model.layers.0.mlp.experts.{expert}.{proj}.'
            tensors[base+'weight']=torch.randn(4,4).to(torch.float8_e4m3fn)
            tensors[base+'weight_scale_inv']=torch.rand(1,1)
    for key in ['w1','w2','w3']:
        tensors[f'audio_decoder_mtp_shared.layers.0.transformer_layer.feed_forward.{key}.weight']=torch.randn(4,4,dtype=torch.bfloat16)
    save_file(tensors,source/'one.safetensors')
    (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'one.safetensors' for k in tensors}}),encoding='utf-8')
    return SimpleNamespace(src=source,out=tmp_path/'converted'/'drama.safetensors',source_revision='7'*40),tensors


def test_lossless_native_banks_dense_fusions_and_control_heads_match_the_source(tmp_path,monkeypatch):
    args,tensors=fixture(tmp_path,monkeypatch)
    convert_fp8_streamed(args)
    with safe_open(args.out,framework='pt') as archive:
        assert archive.metadata()['source_revision']=='7'*40
        assert torch.equal(archive.get_tensor('audio_head.weight'),tensors['lm_head.weight'][[0,1,2,4,5]])
        assert torch.equal(archive.get_tensor('model.embed_tokens.weight'),tensors['model.language_model.embed_tokens.weight'])
        for field,suffix in [('weight','weight'),('scale','weight_scale_inv')]:
            expected=torch.stack([torch.cat([tensors[f'model.language_model.layers.0.mlp.experts.{e}.{p}.{suffix}']
                for p in ['gate_proj','up_proj']],dim=0) for e in range(2)])
            assert torch.equal(archive.get_tensor('model.layers.0.mlp.experts.gate_up_'+field).view(torch.uint8),expected.view(torch.uint8))
        expected=torch.cat([tensors[f'model.language_model.layers.0.mlp.shared_expert.{p}.weight'] for p in ['gate_proj','up_proj']])
        assert torch.equal(archive.get_tensor('model.layers.0.mlp.shared_expert.gate_up_proj.weight'),expected)
    for name,rows in [('drama3-text-head.safetensors',None),('drama3-line-head.safetensors',[3,4])]:
        with safe_open(args.out.parent/name,framework='pt') as archive:
            assert archive.metadata()['source_revision']=='7'*40
            expected=tensors['lm_head.weight'] if rows is None else tensors['lm_head.weight'][rows]
            assert torch.equal(archive.get_tensor('weight'),expected)
            if rows is not None:assert archive.get_tensor('ids').tolist()==rows
    assert not args.out.with_suffix('.safetensors.part').exists()


def test_unknown_sources_are_not_falsely_stamped_with_the_old_official_revision(tmp_path):
    assert source_revision(SimpleNamespace(src=tmp_path))=='unknown'
    (tmp_path/'download-revision.json').write_text(json.dumps({'revision':'main'}),encoding='utf-8')
    with pytest.raises(ValueError):source_revision(SimpleNamespace(src=tmp_path))
    (tmp_path/'download-revision.json').write_text(json.dumps({'revision':'a'*40}),encoding='utf-8')
    assert source_revision(SimpleNamespace(src=tmp_path))=='a'*40
    assert source_revision(SimpleNamespace(src=tmp_path,source_revision='b'*40))=='b'*40
    with pytest.raises(ValueError):source_revision(SimpleNamespace(src=tmp_path,source_revision='main'))
