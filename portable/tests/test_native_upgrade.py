"""Numerical cache and bounded decode regressions for old/new host support."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.native.cache import LinearState, StaticAttentionState
from drama3_t8.native.codec import Drama3Codec, DECODE_WINDOW, DECODE_CONTEXT, FRAME_LENGTH
from drama3_t8.native.control_head import control_rows
from drama3_t8.native.generate import sample_top_k_top_p
from drama3_t8.engine import Sampling
from drama3_t8.engine import DramaEngine
from drama3_t8.audio_processing import prepare_reference,shorten_pauses
from safetensors.torch import save_file


def test_portable_attention_cache_matches_full_causal_gqa_through_continuations():
    torch.manual_seed(23)
    q=torch.randn(1,4,19,8)
    k=torch.randn(1,2,19,8)
    v=torch.randn(1,2,19,8)
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    cache=StaticAttentionState.zeros(1,2,32,8,'cpu',torch.float32,
        StaticAttentionState.shared(32,'cpu',torch.float32))
    parts=[]
    start=0
    for count in (9,1,3,6):
        end=start+count
        cache.prepare(count)
        if start==0:
            keys,values=cache.append(k[:,:,:end],v[:,:,:end])
            out=F.scaled_dot_product_attention(q[:,:,:end],keys,values,is_causal=True,enable_gqa=True)
            parts.append(out.transpose(1,2).reshape(1,count,32))
        else:
            parts.append(cache.decode(q[:,:,start:end],k[:,:,start:end],v[:,:,start:end],2))
        cache.advance(count)
        start=end
    actual=torch.cat(parts,dim=1).reshape(1,19,4,8).transpose(1,2)
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
    assert cache.index==19
    with pytest.raises(ValueError,match='capacity'):
        cache.prepare(14)


def test_windowed_codec_bounds_work_and_preserves_causal_seams_and_sample_count():
    visited=[]
    def quantize(codes,dtype):
        visited.append(codes.shape[-1])
        return codes[:,:1].to(dtype).repeat_interleave(FRAME_LENGTH,dim=-1)
    def decode(audio):
        kernel=torch.tensor([[[.2,.3,.5]]])
        return F.conv1d(F.pad(audio,(2,0)),kernel)
    codec=SimpleNamespace(quantizer=SimpleNamespace(decode=quantize),decoder=decode)
    codes=torch.arange(1301)[None,None].expand(1,10,-1)
    full=decode(quantize(codes,torch.float32))[:,0]
    visited.clear()
    actual=Drama3Codec.decode(codec,codes,torch.float32)
    assert actual.shape==(1,1301*FRAME_LENGTH)
    assert torch.equal(actual,full)
    assert len(visited)==6
    assert max(visited)<=DECODE_WINDOW+DECODE_CONTEXT


def test_immediate_linear_cache_is_independent_of_deferred_host_tracker():
    state=LinearState(torch.zeros(1,4,3),torch.zeros(1,2,4,4))
    state.prepare(17)
    state.conv_state.fill_(.5)
    state.advance(17)
    state.prepare(1)
    state.advance(1)
    assert state.index==18 and state.conv_state.eq(.5).all()


def test_compact_scene_head_rows_match_legacy_projection_exactly(tmp_path):
    torch.manual_seed(31)
    full=torch.randn(16,8)
    ids=(2,7)
    save_file({'weight':full},tmp_path/'full.safetensors')
    save_file({'weight':full[list(ids)],'ids':torch.tensor(ids)},tmp_path/'small.safetensors')
    left=control_rows(tmp_path/'full.safetensors',ids,8)
    right=control_rows(tmp_path/'small.safetensors',ids,8)
    hidden=torch.randn(5,8)
    assert torch.equal(left,right)
    assert torch.allclose(F.linear(hidden,left),F.linear(hidden,full)[:,list(ids)],atol=1e-6)
    with pytest.raises(ValueError,match='hidden size'):
        control_rows(tmp_path/'small.safetensors',ids,9)
    with pytest.raises(ValueError,match='missing required'):
        control_rows(tmp_path/'small.safetensors',(2,8),8)


def test_top_k_zero_retains_the_distribution_instead_of_collapsing_to_top_one():
    Sampling(top_k=0).validate()
    with pytest.raises(ValueError):Sampling(top_k=-1).validate()
    logits=torch.tensor([[.0,.1,.2,.3,.4]])
    def draws(limit):
        return {int(sample_top_k_top_p(logits,1.,limit,1.,torch.Generator().manual_seed(i))) for i in range(100)}
    assert draws(0)==set(range(5))
    assert draws(2)=={3,4}


def test_reference_processing_is_opt_in_and_records_source_interval_and_gain():
    wave=torch.zeros(1,4000)
    wave[:,1500:2500]=.02*torch.sin(torch.arange(1000)*.3)
    unchanged,original=prepare_reference(wave,1000)
    assert torch.equal(unchanged,wave) and original['gain']==1.
    prepared,edit=prepare_reference(wave,1000,trim_silence=True,normalize=True)
    assert edit['trim_start_seconds']>1. and edit['trim_end_seconds']<3.
    assert prepared.abs().max()<=.901 and edit['gain']==10.
    with pytest.raises(ValueError,match='人声'):
        prepare_reference(torch.zeros(1,1000),1000,trim_silence=True)


def test_pause_cleanup_preserves_speech_and_tracks_every_output_sample():
    speech=torch.full((1,100),.1)
    wave=torch.cat((torch.zeros(1,100),speech,torch.zeros(1,200),speech,torch.zeros(1,100)),dim=1)
    edited,mapping=shorten_pauses(wave,1000,maximum=.05,tail=.02)
    assert edited.shape==(1,270)
    assert torch.count_nonzero(edited)==200
    rebuilt=torch.cat([wave[:,round(i['source_start']*1000):round(i['source_end']*1000)] for i in mapping],dim=-1)
    assert torch.equal(edited,rebuilt)
    assert mapping[-1]['output_end']==.27


def test_opening_retry_archives_failed_frames_and_never_streams_discarded_audio(monkeypatch):
    import drama3_t8.engine as module
    engine=DramaEngine('model','codec')
    config=SimpleNamespace(semantic_start_token_id=248099,semantic_end_token_id=252194)
    drama=SimpleNamespace(config=config,model=SimpleNamespace(model=SimpleNamespace(layers=[])),
        tokenizer=SimpleNamespace(token_to_id=lambda s:252196))
    engine.load_model=lambda:drama
    calls=[];emitted=[]
    def generate(model,prompt,max_frames,temperature,top_k,top_p,seed,**callbacks):
        calls.append(seed)
        count=10 if len(calls)==1 else 60
        frames=torch.zeros(count,11,dtype=torch.long)
        frames[:,0]=248099
        frames[:,1]=13 if len(calls)==1 else torch.arange(count)
        for frame in frames:callbacks['on_audio'](frame[1:])
        callbacks['on_finish']('stalled' if len(calls)==1 else 'eos')
        return frames
    monkeypatch.setattr(module,'generate',generate)
    take=engine._take(torch.zeros(1,11,dtype=torch.long),'hello','hello',
        Sampling(seed=2**64-1,opening_retries=1),on_codes=lambda f:emitted.append(int(f[0])))
    assert calls==[2**64-1,0]
    assert emitted==list(range(60))
    assert take.metadata['generation_attempts'][0]['discarded_frames'][0][1]==13
    assert take.metadata['seed']==0 and take.metadata['requested_seed']==2**64-1
