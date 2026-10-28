"""Real INT8 host expert handoff, including cached graph replay and late-host fallback."""
import json
from types import SimpleNamespace
import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from drama3_t8.native.loader import load_checkpoint_storage
from drama3_t8.native.offload_accelerated import ExpertStore
from drama3_t8.native.graphs import StaticReplay
from drama3_t8.native import kernels


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real CPU/GPU expert handoff needs CUDA')
def test_int8_cpu_handoff_preserves_results_on_graph_replay():
    device=torch.device('cuda:0')
    if not kernels.available(device):pytest.skip('NVRTC expert kernels are unavailable')
    ops=comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    experts=SimpleNamespace(gate_up_proj=ops.MoEExperts(16,256,512,bias=False,device='meta'),
        down_proj=ops.MoEExperts(16,256,256,bias=False,device='meta'))
    generator=torch.Generator().manual_seed(290)
    for module,rows in ((experts.gate_up_proj,512),(experts.down_proj,256)):
        conf=dict(format='int8_tensorwise',num_experts=16,convrot=True,convrot_groupsize=256)
        load_checkpoint_storage(module,dict(weight=torch.randint(-127,128,(16,rows,256),dtype=torch.int8,generator=generator),
            weight_scale=torch.full((16,rows,1),.0002),
            comfy_quant=torch.tensor(list(json.dumps(conf).encode()),dtype=torch.uint8)))
    store=ExpertStore(1,pin_memory=True)
    routed=store.add(experts)
    torch.manual_seed(291)
    x=torch.randn(1,256,device=device,dtype=torch.bfloat16)
    ids=torch.arange(8,device=device)[None]
    weights=torch.full((1,8),.125,device=device,dtype=torch.bfloat16)
    try:
        with torch.inference_mode():
            with store.session(device,[],cpu_decode=False):
                expected=routed(x,ids,weights).clone()
            runtime=StaticReplay(device)
            with store.session(device,[],cpu_decode=True):
                assert store.sync_free and store.cpu_side is not None
                for _ in range(7):
                    actual=runtime(routed,x,ids,weights)
                    store.after_layer(0)
                    # The CPU path and grouped GPU kernels quantize at the same
                    # rotation; bf16 CPU matmul may round intermediate values.
                    error=(actual.float()-expected.float())
                    relative_rms=error.square().mean().sqrt()/expected.float().square().mean().sqrt()
                    assert torch.isfinite(actual).all() and relative_rms<.02,(float(relative_rms),float(error.abs().max()))
                    store.between_steps()
                assert runtime.captures==1 and runtime.replays==4
                assert store.cpu_decode_calls==7 and store.cpu_experts_run>0
    finally:store.close()
