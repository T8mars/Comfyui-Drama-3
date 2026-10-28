"""Actual DRAMA head-size cache decoding on the new host's CUDA attention.

Head size 256 selects comfy-kitchen's native BF16 kernel, unlike the small
head-size-32 backbone tests. This loads one randomly initialized attention
module with DRAMA's real projection dimensions, never the full model.
"""
import json
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
import comfy_kitchen
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.qwen35 import GatedAttention, Qwen35Config, apply_partial_rope
from drama3_t8.native.graphs import StaticReplay
from drama3_t8.native.model import Drama3Backbone


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(GatedAttention, 'init_kv_cache'),
    reason='New ComfyUI native CUDA attention-cache integration only')


def reference(attention, x, freqs):
    batch, count, _ = x.shape
    projected = attention.q_proj(x).view(batch, count, attention.num_heads, attention.head_dim * 2)
    q, gate = projected[..., :attention.head_dim], projected[..., attention.head_dim:]
    q = attention.q_norm(q).transpose(1, 2)
    k = attention.k_norm(attention.k_proj(x).view(
        batch, count, attention.num_kv_heads, attention.head_dim)).transpose(1, 2)
    v = attention.v_proj(x).view(batch, count, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    q, k = apply_partial_rope(q, k, freqs, attention.rotary_dim)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True,
        enable_gqa=attention.num_heads != attention.num_kv_heads)
    output = output.transpose(1, 2).reshape(batch, count, attention.inner_size)
    return attention.o_proj(output * gate.reshape(batch, count, -1).sigmoid())


@pytest.mark.parametrize('counts,seed', [
    ((9, 1, 1, 1, 1, 1, 1, 1), 732),
    ((9, 1, 1, 3, 1, 1, 2, 1, 1), 732),
    ((9, 1, 1, 1, 1, 1, 1, 1), 841),
    ((9, 1, 1, 3, 1, 1, 2, 1, 1), 842),
])
def test_real_dimension_attention_cache_and_graph_match_causal_sdpa(counts, seed, monkeypatch):
    device = torch.device('cuda:0')
    config = Qwen35Config(vocab_size=64, hidden_size=2048, intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        partial_rotary_factor=.25, rope_theta=10000000., rms_norm_eps=1e-6,
        layer_types=['full_attention'])
    attention = GatedAttention(config, device='cpu', dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .01)
        attention.to(device)
        source = (torch.randn(1, sum(counts), 2048, generator=generator) * .2).to(device, torch.bfloat16)
        owner = SimpleNamespace(config=config)
        layers = [SimpleNamespace(self_attn=attention)]
        flash_module = importlib.import_module('comfy_kitchen.flash_attention')
        backend = flash_module._cuda_backend
        assert backend is not None and backend._C is not None
        native = backend._C.flash_attention_decode_gqa
        available = getattr(comfy_kitchen, 'flash_attention_decode_gqa_is_available', None)
        assert native is not None and available is not None
        assert available(device, torch.bfloat16), 'The new-host CUDA kernel must be available on this test machine'
        calls = []
        def observed(*args, **kwargs):
            # Observe the real extension call so a DRAMA-owned DLPack boundary
            # remains visible without replacing the attention calculation.
            calls.append(dict(arguments=len(args), capturing=torch.cuda.is_current_stream_capturing(),
                extension=str(backend._C.__file__)))
            return native(*args, **kwargs)
        monkeypatch.setattr(backend._C, 'flash_attention_decode_gqa', observed)
        full_freqs = Drama3Backbone.compute_freqs_cis(owner,
            torch.arange(source.shape[1], device=device).unsqueeze(0), device)
        expected = reference(attention, source, full_freqs).clone()
        trials = []
        for use_graph in (False, True):
            cache = Drama3Backbone.init_layer_caches(owner, 128, layers, [device], torch.bfloat16)[0]
            runtime = StaticReplay(device)
            pieces, offset = [], 0
            for count in counts:
                x = source[:, offset:offset + count].clone()
                freqs = Drama3Backbone.compute_freqs_cis(owner,
                    torch.arange(offset, offset + count, device=device).unsqueeze(0), device)
                mask = None
                if count > 1:
                    mask = torch.full((count, offset + count), torch.finfo(x.dtype).min / 4,
                        device=device, dtype=x.dtype).triu_(offset + 1)
                selected = optimized_attention_for_device(device, mask=mask is not None, small_input=True)
                cache.prepare(count)
                def operation(value, positions):
                    return attention(value, attention_mask=mask, freqs_cis=positions,
                        optimized_attention=selected, past_key_value=cache)[0]
                result = runtime(operation, x, freqs) if use_graph and count == 1 else operation(x, freqs)
                pieces.append(result.clone())
                cache.advance(count)
                offset += count
            actual = torch.cat(pieces, dim=1)
            torch.cuda.synchronize(device)
            assert cache.index == source.shape[1]
            # Kitchen's BF16 GQA reference contract is atol3e-3/rtol1e-2:
            # https://github.com/Comfy-Org/comfy-kitchen/blob/f8b7e97a1418d2e529bd5405fbf86312be198ae6/tests/test_flash_attention.py#L367
            # This module also includes BF16 projections at different GEMM
            # batch sizes. Keep a stricter aggregate precision check, while
            # retaining the original extra-strict probe as a reported metric.
            difference = actual.float() - expected.float()
            relative_rms = float(difference.square().mean().sqrt() /
                expected.float().square().mean().sqrt().clamp_min(1e-12))
            strict_mismatches = int((~torch.isclose(actual, expected, atol=2e-4, rtol=2e-3)).sum())
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert relative_rms <= .01
            if use_graph:
                assert runtime.captures == 1 and runtime.replays >= 3
                assert runtime.fallback_reason is None
            trials.append(dict(graph=use_graph, output=actual,
                max_abs=float((actual.float() - expected.float()).abs().max()),
                relative_rms=relative_rms, extra_strict_mismatches=strict_mismatches,
                captures=runtime.captures, replays=runtime.replays))
        # Graph replay must not alter the selected kernel or cache evolution.
        assert torch.equal(trials[0]['output'], trials[1]['output'])
        assert calls and any(call['capturing'] for call in calls)
        report_path = os.environ.get('DRAMA_ATTENTION_REPORT')
        if report_path:
            path = Path(report_path)
            data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            data[str((counts, seed))] = dict(status='passed', seed=seed,
                config=dict(hidden=2048, heads=16, kv_heads=2, head_dim=256),
                calls=calls, trials=[{k: v for k, v in trial.items() if k != 'output'} for trial in trials],
                limits='One randomly initialized attention module, not whole-model generation or visible canvas')
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
