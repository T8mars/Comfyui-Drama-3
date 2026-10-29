"""DRAMA-owned immediate DeltaNet state and a portable audio-tower KV cache."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LinearState:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    index: int = 0

    def prepare(self, num_tokens):
        # DramaDeltaNet commits both states immediately, including long revisions.
        pass

    def advance(self, num_tokens):
        self.index += num_tokens


@dataclass
class StaticAttentionState:
    key: torch.Tensor
    value: torch.Tensor
    index: int
    position: torch.Tensor
    bias: torch.Tensor
    tracker: dict

    @staticmethod
    def shared(capacity, device, dtype):
        return (torch.empty(6, device=device, dtype=torch.int64),
                torch.full((1, 1, 6, capacity), torch.finfo(dtype).min, device=device, dtype=dtype),
                {"step": None})

    @classmethod
    def zeros(cls, batch, kv_heads, capacity, head_dim, device, dtype, shared):
        key = torch.zeros(batch, kv_heads, capacity, head_dim, device=device, dtype=dtype)
        return cls(key, torch.zeros_like(key), 0, *shared)

    def prepare(self, num_tokens):
        if self.index + num_tokens > self.key.shape[2]:
            raise ValueError("DRAMA attention cache capacity exceeded")
        if self.tracker["step"] == (self.index, num_tokens):
            return
        self.tracker["step"] = (self.index, num_tokens)
        if num_tokens <= self.position.numel():
            torch.arange(self.index, self.index + self.position.numel(), out=self.position)
            query = torch.arange(self.index + num_tokens - self.position.numel(),
                                 self.index + num_tokens, device=self.key.device)
            visible = torch.arange(self.key.shape[2], device=self.key.device)[None, :] <= query[:, None]
            self.bias.copy_(torch.zeros_like(self.bias).masked_fill(~visible[None, None], torch.finfo(self.key.dtype).min))

    def advance(self, num_tokens):
        self.index += num_tokens

    def append(self, key, value):
        end = self.index + key.shape[2]
        self.key[:, :, self.index:end] = key
        self.value[:, :, self.index:end] = value
        return self.key[:, :, :end], self.value[:, :, :end]

    def decode(self, query, key, value, kv_heads):
        count = key.shape[2]
        self.key.index_copy_(2, self.position[:count], key)
        self.value.index_copy_(2, self.position[:count], value)
        out = F.scaled_dot_product_attention(query, self.key, self.value,
            attn_mask=self.bias[:, :, -count:], enable_gqa=query.shape[1] != kv_heads)
        return out.transpose(1, 2).reshape(query.shape[0], count, -1)


class AudioTowerState(StaticAttentionState):
    """Keep single-row GQA on the Fish Audio grouped SDPA path.

    A host may enable reduced-precision math SDPA. Passing unequal Q/KV
    head counts with ``enable_gqa`` can select that path for masked decode.
    Folding each KV head's query heads into rows (fish-speech 5415669,
    fish_omni/layers.py::grouped_attention) avoids the GQA fallback and its
    extra rounding without changing the host's global attention settings.
    Use this upstream path on older hosts as well: small rounding differences
    in their legacy audio-tower decode can change sampled residual codes.
    The full-capacity mask stays device-owned for CUDA graph replay.
    """

    def decode(self, query, key, value, kv_heads):
        count = key.shape[2]
        batch, heads, _, head_dim = query.shape
        if count != 1 or heads == kv_heads:
            return super().decode(query, key, value, kv_heads)
        self.key.index_copy_(2, self.position[:count], key)
        self.value.index_copy_(2, self.position[:count], value)
        grouped = query.reshape(batch, kv_heads, heads // kv_heads, head_dim)
        out = F.scaled_dot_product_attention(grouped, self.key, self.value,
            attn_mask=self.bias[:, :, -count:])
        return out.reshape(batch, heads, 1, head_dim).transpose(1, 2).reshape(batch, count, -1)


AudioTowerKV = AudioTowerState

try:
    from comfy.text_encoders.llama import FixedKVBias as TowerKV
except ImportError:
    TowerKV = StaticAttentionState
