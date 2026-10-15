"""Fish Audio Drama 3: a Qwen3.5 MoE backbone that predicts each frame's main token, and a codebook tower that
predicts the frame's other codes.

A frame row is [main token id, code 0, ..., code N-1]. Text rows carry zeros for the codes; an audio row's main
token is the semantic token of its code 0.
"""
import math

import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope1, rope
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import FixedKVBias, RMSNorm, moe_experts_forward
from comfy.text_encoders.qwen35 import GatedAttention, GatedDeltaNet, LinearKV, Qwen35Transformer, Qwen35TransformerBlock
from .offload import FP8Experts, experts_decode, gather_experts
from .delta import DramaDeltaNet


class SharedExpert(nn.Module):
    def __init__(self, hidden, intermediate, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate_up_proj = ops.Linear(hidden, 2 * intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(comfy.ops._swiglu_eager(self.gate_up_proj(x)))


class Experts(nn.Module):
    def __init__(self, num_experts, hidden, intermediate, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = ops.MoEExperts(num_experts, hidden, 2 * intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts, intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x, topk_idx, topk_weight):
        if x.shape[0] > 1:
            return moe_experts_forward(x, topk_idx, topk_weight, self.num_experts, self.gate_up_proj, self.down_proj,
                                       comfy.ops._swiglu_eager)
        # one row: gather its experts on the GPU and run them as one batch
        with self.gate_up_proj.bank_resident(x) as gate_up, self.down_proj.bank_resident(x) as down:
            return experts_decode(x, gather_experts(gate_up._resident_bank[0], topk_idx[0], self.num_experts),
                                  gather_experts(down._resident_bank[0], topk_idx[0], self.num_experts), topk_weight[0])


class SparseMoeBlock(nn.Module):
    """Softmax router over all experts, renormalized top-k, plus a sigmoid-gated shared expert."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden = config.text.hidden_size
        self.top_k = config.num_experts_per_tok
        self.gate = ops.Linear(hidden, config.num_experts, bias=False, device=device, dtype=dtype)
        if config.fp8_experts:
            self.experts = FP8Experts(config.num_experts, hidden, config.moe_intermediate_size, device=device, dtype=dtype)
        else:
            self.experts = Experts(config.num_experts, hidden, config.moe_intermediate_size, device=device, dtype=dtype, ops=ops)
        self.shared_expert = SharedExpert(hidden, config.shared_expert_intermediate_size, device=device, dtype=dtype, ops=ops)
        self.shared_expert_gate = ops.Linear(hidden, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        probabilities = torch.softmax(self.gate(flat).float(), dim=-1)
        topk_weight, topk_idx = probabilities.topk(self.top_k, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        out = self.experts(flat, topk_idx, topk_weight.to(flat.dtype))
        out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return out.view_as(x)


class Drama3Block(Qwen35TransformerBlock):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        text = config.text
        self.layer_type = text.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = DramaDeltaNet(text, device=device, dtype=dtype, ops=ops)
        else:
            self.self_attn = GatedAttention(text, device=device, dtype=dtype, ops=ops)
        self.mlp = SparseMoeBlock(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)


class Drama3Backbone(Qwen35Transformer):
    def __init__(self, config, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        text = config.text
        self.config = text
        self.embed_tokens = ops.Embedding(text.vocab_size, text.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(Drama3Block(config, i, device=device, dtype=dtype, ops=ops) for i in range(text.num_hidden_layers))
        self.norm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)

    def init_caches(self, capacity, devices, dtype):
        """Per-layer decode state on each layer's device: DeltaNet conv and recurrent state, or a KV cache."""
        text = self.config
        conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim + text.linear_num_value_heads * text.linear_value_head_dim
        shared = {}
        caches = []
        for index, device in enumerate(devices):
            if text.layer_types[index] == "linear_attention":
                recurrent = torch.zeros((1, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim), device=device, dtype=torch.float32)
                conv = torch.zeros((1, conv_dim, text.conv_kernel_size - 1), device=device, dtype=dtype)
                caches.append(LinearKV(conv, recurrent, 0, None, None))
            else:
                if device not in shared:
                    shared[device] = FixedKVBias.shared(capacity, device, dtype)
                caches.append(FixedKVBias.zeros(1, text.num_key_value_heads, capacity, text.head_dim, device, dtype, shared[device]))
        return caches

    def run_layers(self, x, caches, layers, devices):
        """Hidden states [1, T, H] after `layers` (the whole stack, in order) for the rows `x` [1, T, H] continue,
        each layer on its device; the result is on the last one. Not final-normed.

        Prompt and continuation calls share the DeltaNet conv and recurrent state at arbitrary lengths.
        """
        past_len = caches[0].index
        start = 0
        while start < len(layers):
            device = devices[start]
            end = start
            while end < len(layers) and devices[end] == device:
                end += 1
            with comfy.model_management.cuda_device_context(device):
                x = self._segment(x.to(device), layers[start:end], caches[start:end], device, past_len)
            start = end
        return x

    def _segment(self, x, layers, caches, device, past_len):
        seq_len = x.shape[1]
        positions = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, past_len + seq_len), torch.finfo(x.dtype).min / 4, device=device, dtype=x.dtype).triu_(past_len + 1)
        freqs = self.compute_freqs_cis(positions, device)
        attention = optimized_attention_for_device(device, mask=mask is not None, small_input=True)
        queue = comfy.model_prefetch.make_prefetch_queue(list(layers), device, {"prefetch_dynamic_vbars": True})
        for layer, cache in zip(layers, caches):
            comfy.model_management.throw_exception_if_processing_interrupted()
            cache.prepare(seq_len)

            def core(layer=layer, cache=cache):
                nonlocal x
                x = layer(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention, past_key_value=cache)[0]

            comfy.model_prefetch.prefetch_queue_pop(queue, device, layer, x.dtype, core=core)
            cache.advance(seq_len)
        comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
        return x


class TowerAttention(nn.Module):
    """Gated GQA over every earlier row, q/k norm, interleaved partial RoPE. wqkv rows are [Q | gate | K | V]."""

    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.cfg = cfg
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        self.split = (q_width, q_width, kv_width, kv_width)
        self.wqkv = ops.Linear(cfg.dim, 2 * q_width + 2 * kv_width, bias=False, device=device, dtype=dtype)
        self.wo = ops.Linear(q_width, cfg.dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)

    def rotate(self, x, freqs):
        width = self.cfg.rotary_dim
        return torch.cat((apply_rope1(x[..., :width].contiguous(), freqs), x[..., width:]), dim=-1)

    def forward(self, x, context, cache):
        freqs, mask, attention = context
        cfg = self.cfg
        length = x.shape[1]
        q, gate, k, v = self.wqkv(x).split(self.split, dim=-1)
        q = self.rotate(self.q_norm(q.view(1, length, cfg.n_head, cfg.head_dim)).transpose(1, 2), freqs)
        k = self.rotate(self.k_norm(k.view(1, length, cfg.n_local_heads, cfg.head_dim)).transpose(1, 2), freqs)
        v = v.view(1, length, cfg.n_local_heads, cfg.head_dim).transpose(1, 2)
        if mask is None:
            out = cache.decode(q, k, v, cfg.n_local_heads)
        else:
            k, v = cache.append(k, v)
            out = attention(q, k, v, cfg.n_head, mask=mask, skip_reshape=True, enable_gqa=True)
        return self.wo(out * torch.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.w13 = ops.Linear(cfg.dim, 2 * cfg.intermediate_size, bias=False, device=device, dtype=dtype)
        self.w2 = ops.Linear(cfg.intermediate_size, cfg.dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(comfy.ops._swiglu_eager(self.w13(x)))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state (MTP style), then one pre-norm attention + MLP block."""

    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.embedding = ops.Embedding(cfg.vocab_size, cfg.dim, device=device, dtype=dtype)
        self.enorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.hnorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.eh_proj = ops.Linear(2 * cfg.dim, cfg.dim, bias=False, device=device, dtype=dtype)
        self.attention = TowerAttention(cfg, device=device, dtype=dtype, ops=ops)
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.feed_forward = TowerMLP(cfg, device=device, dtype=dtype, ops=ops)
        self.final_layernorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)

    def forward(self, hidden, codes, semantic, context, cache, shared_input=None):
        embedded = self.embedding(codes.clamp(0, self.embedding.num_embeddings - 1), out_dtype=hidden.dtype)
        embedded = torch.where(semantic[..., None], embedded, 0.0)
        if shared_input is not None:
            embedded = embedded + shared_input
        mixed = self.eh_proj(torch.cat((self.enorm(embedded), self.hnorm(hidden)), dim=-1))
        x = mixed + self.attention(self.attention_norm(mixed), context, cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return self.final_layernorm(x)


class Drama3Tower(nn.Module):
    """Predicts codes 1..N-1 of a frame from the backbone's hidden state and code 0, one layer per code, each
    attending over all earlier rows through its own KV cache. Row p of a prompt is labelled with frame p + 1."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        cfg = self.cfg = config.tower
        self.semantic_start = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.codebook_embeddings = ops.Embedding(cfg.vocab_size * cfg.num_codebooks, cfg.dim, device=device, dtype=dtype)
        self.shared_layers = nn.ModuleList(TowerLayer(cfg, device=device, dtype=dtype, ops=ops) for _ in range(cfg.shared_layers))
        self.residual_layers = nn.ModuleList(TowerLayer(cfg, device=device, dtype=dtype, ops=ops) for _ in range(cfg.num_codebooks - 1))
        self.heads = nn.ModuleList(ops.Linear(cfg.dim, cfg.vocab_size, bias=False, device=device, dtype=dtype) for _ in range(cfg.num_codebooks - 1))

    def is_semantic(self, tokens):
        return (tokens >= self.semantic_start) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames, first, end, dtype):
        """Summed embeddings of codes first..end-1 of each frame row."""
        offsets = torch.arange(first, end, device=frames.device) * self.cfg.vocab_size
        return self.codebook_embeddings(frames[:, 1 + first:1 + end] + offsets, out_dtype=dtype).sum(dim=-2)

    def embed_backbone(self, frames, text_embeddings):
        """Backbone input: an audio row adds the embeddings of codes 1..K to its main token's."""
        count = self.cfg.backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count, text_embeddings.dtype).to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        return torch.where(semantic[:, None], mixed, text_embeddings)

    def init_caches(self, capacity, device, dtype):
        cfg = self.cfg
        shared = FixedKVBias.shared(capacity, device, dtype)
        return [FixedKVBias.zeros(1, cfg.n_local_heads, capacity, cfg.head_dim, device, dtype, shared)
                for _ in range(len(self.shared_layers) + len(self.residual_layers))]

    def context(self, positions, mask_rows, device, dtype):
        """(freqs, mask, attention) for rows at `positions` [T]; decode rows (mask_rows == 0) attend without a mask."""
        freqs = rope(positions.unsqueeze(0), self.cfg.rotary_dim, self.cfg.rope_base).unsqueeze(1)
        mask = None
        if mask_rows:
            past = int(positions[0])
            mask = torch.full((mask_rows, past + mask_rows), torch.finfo(dtype).min / 4, device=device, dtype=dtype).triu_(past + 1)
        return freqs, mask, optimized_attention_for_device(device, mask=mask is not None, small_input=True)

    def run(self, hidden, labels, frames, context, caches):
        """Run every layer on rows whose next frames are `labels` [T, 1 + N]; returns the last hidden state.

        `labels` holds code 0 of the next frame and, for the residual layers, the codes before the one each
        layer predicts. During decode the residual layers' codes are sampled one at a time instead (`step`)."""
        semantic = self.is_semantic(labels[:, 0])
        unseen = self.embedding_sum(frames, 1 + self.cfg.backbone_codebooks, self.cfg.num_codebooks, hidden.dtype)
        unseen = torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = self._layer(layer, index, hidden, labels[:, 1], semantic, context, caches, unseen if index == 0 else None)
        for index, layer in enumerate(self.residual_layers):
            hidden = self._layer(layer, len(self.shared_layers) + index, hidden, labels[:, 1 + index], semantic, context, caches)
        return hidden

    def _layer(self, layer, index, hidden, codes, semantic, context, caches, shared_input=None):
        cache = caches[index]
        cache.prepare(hidden.shape[1])
        out = layer(hidden, codes[None], semantic[None], context, cache, shared_input[None] if shared_input is not None else None)
        cache.advance(hidden.shape[1])
        return out

    def step(self, main, hidden, frame, context, caches, sample):
        """Codes of the frame whose main token is `main` [1]: the shared layers, then one residual layer and head
        per code, each code sampled before the next layer runs. Returns the frame row [1, 1 + N]."""
        c0 = (main - self.semantic_start).clamp(0, self.cfg.vocab_size - 1)
        semantic = self.is_semantic(main)
        unseen = self.embedding_sum(frame, 1 + self.cfg.backbone_codebooks, self.cfg.num_codebooks, hidden.dtype)
        unseen = torch.where(self.is_semantic(frame[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = self._layer(layer, index, hidden, c0, semantic, context, caches, unseen if index == 0 else None)
        codes = [c0]
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.heads)):
            hidden = self._layer(layer, len(self.shared_layers) + index, hidden, codes[-1], semantic, context, caches)
            codes.append(sample(head(hidden[0])))
        return torch.stack((main, *codes), dim=-1)


class Drama3Model(nn.Module):
    """Backbone, the audio rows of the LM head, and the codebook tower. With a second GPU, the backbone's last layers,
    its final norm, the head and the tower move into a `SecondGPU` module that has a patcher of its own."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.model = Drama3Backbone(config, device=device, dtype=dtype, ops=operations)
        self.audio_head = operations.Linear(config.text.hidden_size, len(config.output_token_ids), bias=False, device=device, dtype=dtype)
        self.tower = Drama3Tower(config, device=device, dtype=dtype, ops=operations)
        # run every frame, so made resident first; `SecondGPU` takes them along
        self.every_frame = [self.audio_head, self.tower]

    def get_dynamic_vram__units(self):
        # an evicted embedding table is read row by row from its host copy
        return [*self.every_frame, *self.model.layers], [self.model.embed_tokens]


class SecondGPU(nn.Module):
    """The modules a second GPU runs: the backbone layers from `first` on, the final norm, the head and the tower."""

    def __init__(self, model, first):
        super().__init__()
        self.first = first
        self.layers = nn.ModuleList(model.model.layers[first:])
        self.norm = model.model.norm
        self.audio_head = model.audio_head
        self.tower = model.tower
        model.model.layers = nn.ModuleList(model.model.layers[:first])
        del model.model.norm, model.audio_head, model.tower
        model.every_frame = []

    def get_dynamic_vram__units(self):
        return [self.audio_head, self.tower, *self.layers], []
