"""Real resident BF16 regression contracts against production F.linear.

Resident banks stay whole on CUDA: they have no host slot-cache hit/miss path.
The seven decode calls change routed IDs, rows and mixture weights across
warmup/capture/replay to expose stale Graph inputs without inventing cache stats.
No tolerance, random fixture, model payload, or changed original FP32 test.
"""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from drama3_t8.native import kernels
from drama3_t8.native.acceleration import physical_tensors
from drama3_t8.native.graphs import StaticReplay
from drama3_t8.native.loader import load_checkpoint_storage
from drama3_t8.native.model import Experts
from drama3_t8.native.offload import ExpertStore as TorchStore
from test_cuda_quantized_experts import _bank, _modules, _cpu_threads
from test_resident_experts import holder
from test_resident_cuda import managed, cleanup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Root scheduled actual resident CUDA required')


def _inputs(width,device):
    h=_build_hadamard(256,device=device,dtype=torch.bfloat16)
    negative=(-h[0]*(170/128)).repeat(width//256)
    positive=-negative
    mixed=((torch.arange(width,device=device)%29)-14).to(torch.bfloat16)/32
    return torch.stack((torch.zeros_like(negative),negative,positive,mixed))


def _fixed_case(fmt):
    eye=torch.eye(256)
    gate=[];down=[]
    for expert in range(16):
        gate.append(torch.cat((eye*((16+expert)/32),eye*((32+expert)/64)),0))
        down.append(eye*((24+expert)/64))
    return SimpleNamespace(fmt=fmt,gate=_bank(torch.stack(gate),fmt),down=_bank(torch.stack(down),fmt))


@pytest.mark.parametrize('fmt',('asym_w4a8_int8','int8_tensorwise','w6a8_int8'))
def test_resident_bf16_prefill_decode_and_Graph_match_actual_F_linear(fmt):
    device=torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('Actual NVRTC unavailable; not a passed resident regression')
    reference=TorchStore(1,pin_memory=False)
    patcher=controller=None
    try:
        with _cpu_threads(),torch.inference_mode():
            case=_fixed_case(fmt)
            ref_op=reference.add(_modules(case))
            ops=comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
            expert=Experts(16,256,256,device='meta',dtype=torch.bfloat16,ops=ops)
            load_checkpoint_storage(expert.gate_up_proj,dict(case.gate))
            load_checkpoint_storage(expert.down_proj,dict(case.down))
            model=holder([expert]);model.dtype=torch.bfloat16
            patcher,controller=managed(model,'cuda')
            x=_inputs(256,device)[1:]
            ids=torch.stack((torch.arange(8,device=device),torch.arange(8,16,device=device),
                             (torch.arange(8,device=device)+3)%16))
            weights=(torch.arange(1,9,device=device,dtype=torch.float32)/36).to(torch.bfloat16).expand(3,-1)
            with reference.session(device,[]):
                expected=ref_op(x,ids,weights).clone()
            assert expected.dtype==torch.bfloat16 and torch.count_nonzero(expected)>0
            def counted_replay():
                # Same backend bookkeeping as production BackboneGraphs.
                controller.backend_calls['cuda']+=1
            runtime=StaticReplay(device,on_replay=counted_replay)
            with controller.session(device,[patcher]):
                assert controller.sync_free and expert.sync_free(device)
                assert not controller.fallback_reason and controller.selected_backend=='cuda'
                fields=dict(physical_tensors(model))
                assert fields and all(t.device==device and not t.is_meta for t in fields.values())
                pointers={name:t.data_ptr() for name,t in fields.items()}
                assert not hasattr(controller,'cache') and not hasattr(controller,'banks')
                actual=expert(x,ids,weights)
                assert torch.equal(actual,expected),fmt+' resident BF16 prefill differs from actual F.linear'
                for step,start in enumerate((0,0,0,8,8,0,8)):
                    row=x[step%3:step%3+1]
                    selected=torch.arange(start,start+8,device=device)[None]
                    mix=weights[0:1].roll(step,dims=-1)
                    with reference.session(device,[]):
                        wanted=ref_op(row,selected,mix).clone()
                    found=runtime(expert,row,selected,mix).clone()
                    assert found.dtype==torch.bfloat16
                    assert torch.equal(found,wanted),fmt+' resident BF16 decode/Graph differs from actual F.linear'
                assert runtime.calls==7 and runtime.captures==1 and runtime.replays==4
                assert runtime.fallback_reason is None
                assert controller.stats()['backend_calls']=={'cuda':8,'torch':0}
                assert controller.stats()['actual_backend']=='cuda'
                assert controller.cpu_side is None and controller.cpu_experts_run==controller.cpu_decode_calls==0
                assert pointers=={name:t.data_ptr() for name,t in physical_tensors(model)}
            assert not controller.active and not controller.sync_free
    finally:
        try:
            if controller is not None:cleanup(patcher,controller)
        finally:reference.close()
