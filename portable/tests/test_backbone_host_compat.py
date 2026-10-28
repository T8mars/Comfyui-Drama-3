"""Real tiny hybrid backbone: same sequence in one prefill or several calls."""
from types import SimpleNamespace
import torch
import pytest

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35Config
from drama3_t8.native.model import Drama3Backbone


def test_hybrid_backbone_prefill_and_revision_chunks_are_numerically_consistent():
    text=Qwen35Config(vocab_size=64,hidden_size=128,intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=32,
        partial_rotary_factor=.5,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_key_head_dim=8,linear_value_head_dim=8,conv_kernel_size=4,
        layer_types=['linear_attention']*3+['full_attention'])
    config=SimpleNamespace(text=text,fp8_experts=True,num_experts=4,num_experts_per_tok=2,
        moe_intermediate_size=128,shared_expert_intermediate_size=128)
    torch.manual_seed(5090)
    model=Drama3Backbone(config,device='cpu',dtype=torch.float32,ops=comfy.ops.manual_cast)
    with torch.inference_mode():
        for param in model.parameters():param.copy_(torch.randn(param.shape)*.05)
        for layer in model.layers:
            layer.mlp.experts.gate_up_scale.fill_(.1)
            layer.mlp.experts.down_scale.fill_(.1)
        source=torch.randn(1,19,128)
        layers=list(model.layers);devices=[torch.device('cpu')]*4
        whole=model.init_layer_caches(64,layers,devices,torch.float32)
        chunked=model.init_layer_caches(64,layers,devices,torch.float32)
        expected=model.run_layers(source.clone(),whole,layers,devices)
        parts=[];start=0
        for count in (7,9,1,2):
            parts.append(model.run_layers(source[:,start:start+count].clone(),chunked,layers,devices))
            start+=count
        actual=torch.cat(parts,dim=1)
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-4)
        assert all(c.index==19 for c in chunked)
