"""Stateful graph replay must preserve cache progression and fresh frame storage."""
from types import SimpleNamespace
import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35Config
from drama3_t8.native.model import Drama3Backbone
from drama3_t8.native.graphs import BackboneGraphs,StaticReplay
from drama3_t8.native.offload_accelerated import ExpertStore
from drama3_t8.native import kernels


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph test needs a GPU')
def test_graph_replay_preserves_real_hybrid_backbone_state():
    device=torch.device('cuda:0')
    if not kernels.available(device):pytest.skip('NVRTC kernels unavailable')
    text=Qwen35Config(vocab_size=64,hidden_size=128,intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=32,
        partial_rotary_factor=.5,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_key_head_dim=8,linear_value_head_dim=8,conv_kernel_size=4,
        layer_types=['linear_attention']*3+['full_attention'])
    config=SimpleNamespace(text=text,fp8_experts=True,num_experts=8,num_experts_per_tok=2,
        moe_intermediate_size=128,shared_expert_intermediate_size=128)
    torch.manual_seed(512)
    model=Drama3Backbone(config,device='cpu',dtype=torch.bfloat16,ops=comfy.ops.manual_cast)
    store=ExpertStore(4*1024**2,pin_memory=True)
    with torch.inference_mode():
        for param in model.parameters():param.copy_(torch.randn(param.shape)*.05)
        for layer in model.layers:
            layer.mlp.experts.gate_up_scale.fill_(.1);layer.mlp.experts.down_scale.fill_(.1)
            layer.mlp.experts=store.add(layer.mlp.experts)
        model.to(device)
        layers=list(model.layers);devices=[device]*4
        source=torch.randn(1,16,128,device=device,dtype=torch.bfloat16)
        outputs=[]
        for use_graph in (False,True):
            caches=model.init_layer_caches(64,layers,devices,torch.bfloat16)
            graphs=BackboneGraphs(store) if use_graph else None
            pieces=[]
            with store.session(device,[]):
                assert store.sync_free
                pieces.append(model.run_layers(source[:,:8].clone(),caches,layers,devices))
                for i in range(8,16):
                    pieces.append(model.run_layers(source[:,i:i+1].clone(),caches,layers,devices,graphs=graphs).clone())
            outputs.append(torch.cat(pieces,dim=1))
            assert all(c.index==16 for c in caches)
            if graphs:
                report=graphs.report()
                assert report['captures']==4 and report['replays']==20,report
                assert not report['fallback_reasons']
        assert torch.isfinite(outputs[1]).all()
        assert torch.allclose(outputs[0],outputs[1],atol=.002,rtol=.002)
    store.cpu.shutdown(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph RNG test needs a GPU')
def test_graph_random_generator_advances_and_can_be_reproduced():
    def trial():
        generator=torch.Generator(device='cuda').manual_seed(513)
        runtime=StaticReplay(torch.device('cuda:0'),generator=generator)
        results=[]
        def operation(x):
            return x+torch.empty_like(x).exponential_(generator=generator)
        for i in range(7):results.append(runtime(operation,torch.full((128,),float(i),device='cuda')).clone())
        assert runtime.captures==1 and runtime.replays==4 and not runtime.fallback_reason
        return results
    with torch.inference_mode():
        first,second=trial(),trial()
        assert all(torch.equal(a,b) for a,b in zip(first,second))
        assert all(not torch.equal(a,b) for a,b in zip(first,first[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA capture-stream RNG ordering needs a GPU')
@pytest.mark.parametrize('caller_stream_kind',['default','side'])
def test_capture_rng_initialization_is_ordered_before_first_replay(monkeypatch,caller_stream_kind):
    # PyTorch initializes graph RNG seed/offset tensors on the capture stream
    # before capture begins. A slow stream exposes a missing edge to the caller
    # stream: its offset=0 fill can overwrite replay's offset=8 before sampling.
    original_begin=torch.cuda.CUDAGraph.capture_begin
    delayed=False
    def capture_begin(graph,*args,**kwargs):
        if delayed:torch.cuda._sleep(10_000_000)
        return original_begin(graph,*args,**kwargs)
    monkeypatch.setattr(torch.cuda.CUDAGraph,'capture_begin',capture_begin)
    caller=(torch.cuda.default_stream() if caller_stream_kind=='default' else torch.cuda.Stream())
    keepalive=[]
    with torch.inference_mode(),torch.cuda.stream(caller):
        eager_generator=torch.Generator(device='cuda').manual_seed(513)
        expected=[(torch.full((128,),float(i),device='cuda')+
                   torch.empty(128,device='cuda').exponential_(generator=eager_generator)).clone()
                  for i in range(7)]
        def trial():
            generator=torch.Generator(device='cuda').manual_seed(513)
            runtime=StaticReplay(torch.device('cuda:0'),generator=generator)
            keepalive.append((generator,runtime))
            def operation(x):
                if torch.cuda.is_current_stream_capturing():torch.cuda._sleep(50_000_000)
                return x+torch.empty_like(x).exponential_(generator=generator)
            outputs=[runtime(operation,torch.full((128,),float(i),device='cuda')).clone() for i in range(7)]
            assert runtime.captures==1 and runtime.replays==4 and not runtime.fallback_reason
            return outputs
        control=trial()  # Prime graph instantiation, so cold-start latency cannot hide the race.
        delayed=True
        actual=trial()
        caller.synchronize()
        for step,(want,normal,slow) in enumerate(zip(expected,control,actual)):
            assert torch.equal(want,normal),f'Unmodified capture differs from eager at step {step}'
            assert torch.equal(want,slow),f'Delayed capture RNG differs from eager at step {step}'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA capture fallback needs a GPU')
def test_capture_failure_restores_rng_and_does_not_double_commit_state():
    def trial(use_graph):
        generator=torch.Generator(device='cuda').manual_seed(527)
        runtime=StaticReplay(torch.device('cuda:0'),generator=generator)
        state=torch.zeros((),device='cuda')
        def operation(x):
            state.add_(1)
            noise=torch.empty_like(x).exponential_(generator=generator)
            if torch.cuda.is_current_stream_capturing():
                # Reject an operation in Python while capture_end can still
                # complete; this is the recoverable capture-fallback boundary.
                raise RuntimeError('Unsupported CUDA graph capture operation')
            return x+noise
        outputs=[]
        for i in range(6):
            source=torch.full((256,),float(i),device='cuda')
            outputs.append((runtime(operation,source) if use_graph else operation(source)).clone())
        return outputs,state.item(),runtime
    with torch.inference_mode():
        eager,eager_count,_=trial(False)
        actual,actual_count,runtime=trial(True)
        assert eager_count==actual_count==6
        assert runtime.fallback_reason and not runtime.captures
        assert all(torch.equal(a,b) for a,b in zip(eager,actual))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA invalidation check needs a GPU')
def test_invalidated_capture_is_rejected_without_attempting_corrupt_eager_fallback():
    import subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    # A real CUDA capture invalidation damages process-local RNG/allocator
    # state in Torch; isolate this destructive fault from the rest of pytest.
    script='''
import torch
from drama3_t8.native.graphs import StaticReplay,CudaGraphRuntimeError,ensure_healthy
r=StaticReplay(torch.device('cuda:0'))
x=torch.ones(8,device='cuda')
def op(x):
    if torch.cuda.is_current_stream_capturing():torch.cuda.synchronize()
    return x+1
with torch.inference_mode():
    r(op,x);r(op,x)
    try:r(op,x)
    except CudaGraphRuntimeError as e:assert 'Restart' in str(e)
    else:raise AssertionError('Invalid capture was treated as successful')
    try:ensure_healthy()
    except CudaGraphRuntimeError:pass
    else:raise AssertionError('Damaged context was reused')
print('EXPECTED_CAPTURE_INVALIDATION_REJECTED')
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'EXPECTED_CAPTURE_INVALIDATION_REJECTED' in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA generation graph test needs a GPU')
def test_tower_and_backbone_graphs_preserve_real_generated_frames_and_rng():
    from drama3_t8.native.model import Drama3Model
    from drama3_t8.native.config import TowerConfig
    from drama3_t8.native.generate import generate
    device=torch.device('cuda:0')
    if not kernels.available(device):pytest.skip('NVRTC kernels unavailable')
    text=Qwen35Config(vocab_size=252288,hidden_size=128,intermediate_size=128,
        max_position_embeddings=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=32,
        partial_rotary_factor=.5,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_key_head_dim=8,linear_value_head_dim=8,conv_kernel_size=4,
        layer_types=['linear_attention']*3+['full_attention'])
    config=SimpleNamespace(text=text,fp8_experts=True,num_experts=8,num_experts_per_tok=2,
        moe_intermediate_size=128,shared_expert_intermediate_size=128,
        tower=TowerConfig(128,4,2,32,256,1e-5,10,16,2,4,10000.,16),
        eos_token_id=248046,semantic_start_token_id=248099,semantic_end_token_id=248114,
        output_token_ids=(248046,*range(248099,248115)))
    torch.manual_seed(514)
    model=Drama3Model(config,device='cpu',dtype=torch.bfloat16,operations=comfy.ops.manual_cast)
    store=ExpertStore(4*1024**2,pin_memory=True)
    with torch.inference_mode():
        for p in model.parameters():p.copy_(torch.randn(p.shape)*.03)
        for layer in model.model.layers:
            layer.mlp.experts.gate_up_scale.fill_(.1);layer.mlp.experts.down_scale.fill_(.1)
            layer.mlp.experts=store.add(layer.mlp.experts)
        model.to(device)
        layers=list(model.model.layers);devices=[device]*4
        wrapper=SimpleNamespace(model=model,config=config,experts=store,
            tokenizer=SimpleNamespace(token_to_id=lambda token:252196),patchers=lambda:[],
            parts=lambda:(layers,devices,model.model.norm,model.audio_head,model.tower))
        prompt=torch.zeros(2,11,dtype=torch.long);prompt[:,0]=torch.tensor([198,100])
        results=[]
        for enabled in (False,True):
            wrapper.cuda_graph=enabled
            def fixed(step,hidden,logits):
                logits.fill_(-100)
                logits[...,step+1 if step<9 else 0]=100
            results.append(generate(wrapper,prompt,12,1.,0,1.,515,inspect_logits=fixed))
            if enabled:
                assert wrapper.last_graph_stats['captures']==4,wrapper.last_graph_stats
                assert wrapper.last_graph_stats['tower_captures']==1,wrapper.last_graph_stats
                assert wrapper.last_graph_stats['tower_replays']==6,wrapper.last_graph_stats
        assert results[0].shape==(9,11)
        assert results[1][:,0].tolist()==list(range(248099,248108))
        assert torch.equal(results[0],results[1])
    store.cpu.shutdown(wait=True)
