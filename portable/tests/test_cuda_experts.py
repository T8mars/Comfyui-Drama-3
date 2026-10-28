"""Actual CUDA expert kernels reading pinned CPU weights, including cache misses."""
import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native.offload import FP8Experts,dequantize_fp8
from drama3_t8.native.offload_accelerated import ExpertStore
from drama3_t8.native import kernels


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA expert numerical test needs a GPU')
def test_cuda_fp8_host_banks_match_float32_reference_on_prefill_and_cache_misses():
    device=torch.device('cuda:0')
    if not kernels.available(device):pytest.skip('NVRTC kernels are unavailable')
    torch.manual_seed(510)
    experts=FP8Experts(8,128,128,device='cpu',dtype=torch.float32)
    with torch.inference_mode():
        experts.gate_up_weight.copy_(torch.randn(experts.gate_up_weight.shape)*.4)
        experts.down_weight.copy_(torch.randn(experts.down_weight.shape)*.4)
        experts.gate_up_scale.fill_(.08);experts.down_scale.fill_(.08)
        gate=dequantize_fp8(experts.gate_up_weight.flatten(0,1),experts.gate_up_scale.flatten(0,1),torch.float32).view(8,256,128).to(device)
        down=dequantize_fp8(experts.down_weight.flatten(0,1),experts.down_scale.flatten(0,1),torch.float32).view(8,128,128).to(device)
        store=ExpertStore(2*1024**2,pin_memory=True)
        routed=store.add(experts)
        for batch in (5,1,1):
            x=torch.randn(batch,128,device=device)
            ids=torch.stack([torch.randperm(8,device=device)[:2] for _ in range(batch)])
            weights=torch.rand(batch,2,device=device).softmax(-1)
            expected=[]
            for row in range(batch):
                pieces=[]
                for e,w in zip(ids[row],weights[row]):
                    g,u=F.linear(x[row:row+1],gate[e]).chunk(2,dim=-1)
                    pieces.append(F.linear(F.silu(g)*u,down[e])*w)
                expected.append(sum(pieces))
            with store.session(device,[]):
                actual=routed(x,ids,weights)
                assert store.sync_free
                assert torch.allclose(actual,torch.cat(expected),atol=2e-6,rtol=2e-4)
        assert store.backend_calls['cuda']==3 and store.last_backend=='cuda'
        store.cpu.shutdown(wait=True)
