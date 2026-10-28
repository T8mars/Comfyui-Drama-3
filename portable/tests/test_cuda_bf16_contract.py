"""Portable GPU regression draft; not collected/run during preparation.

Uses the actual production eager BF16 contract and real QuantizedTensor
F.linear dispatch. No saved model/audio or random fixture/seed search.
The original FP32 oracle tests and their tolerances remain unchanged.
"""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
from drama3_t8.native import kernels
from drama3_t8.native.offload import ExpertStore as TorchStore
from drama3_t8.native.offload_accelerated import ExpertStore as CUDAStore
from drama3_t8.native.graphs import StaticReplay
from test_cuda_quantized_experts import _bank, _modules, _cpu_threads

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires root scheduled actual CUDA')


def _actual_ready():
    device=torch.device('cuda:0')
    if not kernels.available(device):pytest.skip('Actual NVRTC unavailable, not a passed arithmetic regression')
    return device


def _inputs(width,device):
    # H is symmetric/involutive. A single -170/128 coefficient transforms
    # into exactly representable +/-A/16 input values. Its BF16 rounded
    # divisor makes the quotient -127.5, whose nearest-even integer is -128.
    h=_build_hadamard(256,device=device,dtype=torch.bfloat16)
    negative=(-h[0]*(170/128)).repeat(width//256)
    positive=-negative
    mixed=((torch.arange(width,device=device)%29)-14).to(torch.bfloat16)/32
    return torch.stack((torch.zeros_like(negative),negative,positive,mixed))


def test_bf16_rotation_A8_matches_actual_eager_zero_rounding_and_negative128():
    device=_actual_ready()
    one=torch.ones(1,device=device,dtype=torch.int32)
    with torch.inference_mode():
        for width in (256,512,2048):
            x=_inputs(width,device)
            h=_build_hadamard(256,device=device,dtype=torch.bfloat16)
            rotated=_rotate_activation(x,h,256)
            expected_q,expected_scale=quantize_int8_rowwise(rotated)
            actual_q,actual_scale=kernels.rotate_quantize(x.float(),one,bf16=True)
            assert rotated.dtype==torch.bfloat16 and expected_scale.dtype==torch.float32
            assert torch.all(expected_q[0]==0)
            assert torch.all(expected_scale[0]==torch.tensor(1e-30,device=device,dtype=torch.float32))
            assert torch.all(expected_q[1,::256]==-128),'Analytic fixture must exercise negative128, not symmetric127 clipping'
            assert torch.all(expected_q[2,::256]==127)
            assert torch.equal(actual_q,expected_q)
            assert torch.equal(actual_scale.view(torch.int32),expected_scale.reshape(-1).view(torch.int32))


def _fixed_case(fmt):
    # Deterministic, full-rank diagonal maps with distinct expert scaling.
    # The official production quantizers supply the packed bytes/codebooks.
    eye=torch.eye(256)
    gate=[];down=[]
    for expert in range(16):
        gate.append(torch.cat((eye*((16+expert)/32),eye*((32+expert)/64)),0))
        down.append(eye*((24+expert)/64))
    return SimpleNamespace(fmt=fmt,gate=_bank(torch.stack(gate),fmt),down=_bank(torch.stack(down),fmt))


def _store_contract(fmt):
    device=_actual_ready()
    with _cpu_threads(),torch.inference_mode():
        case=_fixed_case(fmt)
        reference=TorchStore(1,pin_memory=False)
        actual=CUDAStore(1,pin_memory=True)
        try:
            ref_op=reference.add(_modules(case));op=actual.add(_modules(case))
            assert not actual.pin_fallback_reason
            x=_inputs(256,device)[1:]
            ids=torch.stack((torch.arange(8,device=device),torch.arange(8,16,device=device),
                             (torch.arange(8,device=device)+3)%16))
            weights=(torch.arange(1,9,device=device,dtype=torch.float32)/36).to(torch.bfloat16).expand(3,-1)
            with reference.session(device,[]):
                wanted=ref_op(x,ids,weights).clone()
            assert wanted.dtype==torch.bfloat16 and torch.count_nonzero(wanted)>0
            with actual.session(device,[],cpu_decode=False):
                assert actual.sync_free and actual.banks[0][0].kernel is not None
                found=op(x,ids,weights)
                assert torch.equal(found,wanted),fmt+' BF16 prefill differs from actual production F.linear'
                assert actual.backend_calls=={'cuda':1,'torch':0} and actual.cpu_experts_run==0
            # Actual one-row decode and cache hit/miss/Graph use the same BF16
            # linear -> SiLU -> multiply -> A8 -> down -> weighted sum contract.
            def counted_replay():
                # Same bookkeeping as production BackboneGraphs: captured
                # kernels run without entering ExpertStore.forward in Python.
                actual.backend_calls['cuda']+=1
            runtime=StaticReplay(device,on_replay=counted_replay)
            with actual.session(device,[],cpu_decode=False):
                for start in (0,0,0,8,8,0,8):
                    selected=torch.arange(start,start+8,device=device)[None]
                    row=x[0:1];mix=weights[0:1]
                    with reference.session(device,[]):
                        expected=ref_op(row,selected,mix).clone()
                    observed=runtime(op,row,selected,mix).clone()
                    assert observed.dtype==torch.bfloat16
                    assert torch.equal(observed,expected),fmt+' BF16 decode/Graph differs from actual production F.linear'
                assert runtime.captures==1 and runtime.replays==4 and not runtime.fallback_reason
                assert actual.backend_calls=={'cuda':8,'torch':0} and actual.cpu_experts_run==0
        finally:
            # Close only the two stores owned by this regression.
            try:actual.close()
            finally:reference.close()


def test_bf16_W4_sparse_prefill_decode_and_graph_matches_real_F_linear():
    _store_contract('asym_w4a8_int8')


def test_bf16_INT8_and_W6_sparse_prefill_decode_and_graph_matches_real_F_linear():
    for fmt in ('int8_tensorwise','w6a8_int8'):
        _store_contract(fmt)
