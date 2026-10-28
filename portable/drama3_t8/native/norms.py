"""DRAMA normalization with Fish Audio's exact FP32 rounding order.

Based on fish-speech commit 541566932abbaebcb0a40e562a1f846738383f38,
fish_omni/layers.py; see licenses/LICENSE-FISH-AUDIO.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.model_management import cast_to


class ZeroCenteredRMSNorm(nn.Module):
    """Keep normalization and the zero-centered weight shift in FP32."""

    def __init__(self, dim, eps=1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        # Preserve ComfyUI's host-weight contract without rounding gamma + 1
        # in the checkpoint dtype before promotion.
        weight = cast_to(self.weight, dtype=torch.float32, device=x.device)
        return (out * (1.0 + weight)).type_as(x)


class GatedRMSNorm(nn.Module):
    """Round the weighted norm before the FP32 SiLU gate, as trained."""

    def __init__(self, dim, eps=1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x, gate):
        dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = cast_to(self.weight, device=x.device)
        out = weight * out.to(dtype)
        return (out * F.silu(gate.float())).to(dtype)
