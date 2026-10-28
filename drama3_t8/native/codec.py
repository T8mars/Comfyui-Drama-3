"""Fish S2-Pro codec (modded DAC) used by Drama 3: 44.1 kHz audio <-> 10 codebooks at 2048 samples per frame.

Encode: causal conv encoder (hop 512) -> 2x2 downsampling -> windowed transformer -> one semantic codebook (4096)
and nine residual codebooks (1024). Decode: codebooks -> windowed transformer -> upsampling -> causal conv decoder.
Weight norm is folded by the converter (`tools/convert.py codec`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.minimax_music.dav import Snake1d

SAMPLE_RATE = 44100
FRAME_LENGTH = 2048
DECODE_WINDOW = 256
DECODE_CONTEXT = 320
DECODE_FRAME_BYTES = int(5.3 * 1024 ** 2)


class CausalConv1d(nn.Module):
    """Left padding of the dilated kernel minus the stride; right padding only to complete the last window."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, dtype=dtype, device=device)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - stride

    def forward(self, x):
        frames = math.ceil((x.shape[-1] - self.kernel_size + self.padding) / self.stride + 1)
        extra = (frames - 1) * self.stride + self.kernel_size - self.padding - x.shape[-1]
        return self.conv(F.pad(x, (self.padding, extra)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dtype=dtype, device=device)
        self.trim = kernel_size - stride

    def forward(self, x):
        x = self.conv(x)
        return x[..., :x.shape[-1] - self.trim]


def residual_unit(dim, dilation, dtype, device, operations):
    return nn.Sequential(Snake1d(dim, dtype, device), CausalConv1d(dim, dim, 7, dilation=dilation, dtype=dtype, device=device, operations=operations),
                         Snake1d(dim, dtype, device), CausalConv1d(dim, dim, 1, dtype=dtype, device=device, operations=operations))


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation, dtype=None, device=None, operations=None):
        super().__init__()
        self.block = residual_unit(dim, dilation, dtype, device, operations)

    def forward(self, x):
        return x + self.block(x)


def rope_matrices(length, head_dim, base, device):
    """The codec's RoPE as 2x2 rotations; its cos/sin table was cached in bf16, so they are rounded the same way."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inv_freq)
    cos, sin = angles.cos().bfloat16().float(), angles.sin().bfloat16().float()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(1, 1, length, head_dim // 2, 2, 2)


class Attention(nn.Module):
    def __init__(self, dim, n_head, head_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.n_head, self.head_dim = n_head, head_dim
        self.wqkv = operations.Linear(dim, 3 * n_head * head_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(n_head * head_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope, mask):
        b, t, _ = x.shape
        q, k, v = (y.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for y in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rope1(q.float(), rope).to(x.dtype)
        k = apply_rope1(k.float(), rope).to(x.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dtype=None, device=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).type_as(x) * self.weight.to(x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.gamma.to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, hidden, dtype=None, device=None, operations=None):
        super().__init__()
        self.attention = Attention(dim, n_head, 64, dtype=dtype, device=device, operations=operations)
        self.feed_forward = FeedForward(dim, hidden, dtype=dtype, device=device, operations=operations)
        self.attention_norm = RMSNorm(dim, 1e-5, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(dim, 1e-5, dtype=dtype, device=device)
        self.attention_layer_scale = LayerScale(dim, dtype=dtype, device=device)
        self.ffn_layer_scale = LayerScale(dim, dtype=dtype, device=device)

    def forward(self, x, rope, mask):
        x = x + self.attention_layer_scale(self.attention(self.attention_norm(x), rope, mask))
        return x + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(x)))


class WindowTransformer(nn.Module):
    """Causal transformer over [B, C, T] where each step sees itself and the `window - 1` steps before it."""

    def __init__(self, dim, n_layer, window, dtype=None, device=None, operations=None):
        super().__init__()
        self.window = window
        self.layers = nn.ModuleList(TransformerBlock(dim, dim // 64, 3 * dim, dtype=dtype, device=device, operations=operations) for _ in range(n_layer))
        self.norm = RMSNorm(dim, 1e-5, dtype=dtype, device=device)

    def forward(self, x):
        x = x.transpose(1, 2)
        length = x.shape[1]
        rows = torch.arange(length, device=x.device)
        mask = (rows[None, :] <= rows[:, None]) & (rows[None, :] > rows[:, None] - self.window)
        rope = rope_matrices(length, 64, 10000, x.device)
        for layer in self.layers:
            x = layer(x, rope, mask)
        return self.norm(x).transpose(1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, 7, groups=dim, dtype=dtype, device=device, operations=operations)
        self.norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.pwconv1 = operations.Linear(dim, 4 * dim, dtype=dtype, device=device)
        self.pwconv2 = operations.Linear(4 * dim, dim, dtype=dtype, device=device)
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.pwconv2(F.gelu(self.pwconv1(y))) * self.gamma.to(x.dtype)
        return x + y.transpose(1, 2)


class VectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_proj = operations.Conv1d(dim, codebook_dim, 1, dtype=dtype, device=device)
        self.out_proj = operations.Conv1d(codebook_dim, dim, 1, dtype=dtype, device=device)
        self.codebook = operations.Embedding(codebook_size, codebook_dim, dtype=dtype, device=device)

    def encode(self, z):
        """(quantized z, codes): the nearest codebook entry by cosine distance."""
        latents = self.in_proj(z)
        flat = F.normalize(latents.transpose(1, 2).reshape(-1, latents.shape[1]).float())
        codebook = F.normalize(self.codebook.weight.float().to(flat.device))
        codes = (flat @ codebook.T).argmax(-1).view(latents.shape[0], -1)
        return self.decode(codes, z.dtype), codes

    def decode(self, codes, dtype):
        return self.out_proj(self.codebook(codes, out_dtype=dtype).transpose(1, 2))


class ResidualVectorQuantize(nn.Module):
    def __init__(self, n_codebooks, codebook_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.quantizers = nn.ModuleList(VectorQuantize(1024, codebook_size, 8, dtype=dtype, device=device, operations=operations) for _ in range(n_codebooks))


def resample_stage(down, dtype, device, operations):
    conv = (CausalConv1d(1024, 1024, 2, stride=2, dtype=dtype, device=device, operations=operations) if down
            else CausalConvTranspose1d(1024, 1024, 2, 2, dtype=dtype, device=device, operations=operations))
    return nn.Sequential(conv, ConvNeXtBlock(1024, dtype=dtype, device=device, operations=operations))


class Quantizer(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device, operations=operations)
        self.semantic_quantizer = ResidualVectorQuantize(1, 4096, **args)
        self.quantizer = ResidualVectorQuantize(9, 1024, **args)
        self.downsample = nn.Sequential(*(resample_stage(True, dtype, device, operations) for _ in range(2)))
        self.upsample = nn.Sequential(*(resample_stage(False, dtype, device, operations) for _ in range(2)))
        self.pre_module = WindowTransformer(1024, 8, 128, **args)
        self.post_module = WindowTransformer(1024, 8, 128, **args)

    def encode(self, z):
        z = self.pre_module(self.downsample(z))
        semantic, code = self.semantic_quantizer.quantizers[0].encode(z)
        residual = z - semantic
        codes = [code]
        for quantizer in self.quantizer.quantizers:
            quantized, code = quantizer.encode(residual)
            residual = residual - quantized
            codes.append(code)
        return torch.stack(codes, dim=1)

    def decode(self, codes, dtype):
        z = self.semantic_quantizer.quantizers[0].decode(codes[:, 0].clamp(max=4095), dtype)
        for index, quantizer in enumerate(self.quantizer.quantizers):
            z = z + quantizer.decode(codes[:, 1 + index].clamp(max=1023), dtype)
        return self.upsample(self.post_module(z))


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride, transformer_layers, dtype=None, device=None, operations=None):
        super().__init__()
        half = dim // 2
        stages = [ResidualUnit(half, d, dtype=dtype, device=device, operations=operations) for d in (1, 3, 9)]
        stages += [Snake1d(half, dtype, device), CausalConv1d(half, dim, 2 * stride, stride=stride, dtype=dtype, device=device, operations=operations)]
        if transformer_layers:
            stages.append(WindowTransformer(dim, transformer_layers, 512, dtype=dtype, device=device, operations=operations))
        self.block = nn.Sequential(*stages)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, dtype=None, device=None, operations=None):
        super().__init__()
        self.block = nn.Sequential(Snake1d(in_dim, dtype, device),
                                   CausalConvTranspose1d(in_dim, out_dim, 2 * stride, stride, dtype=dtype, device=device, operations=operations),
                                   *(ResidualUnit(out_dim, d, dtype=dtype, device=device, operations=operations) for d in (1, 3, 9)))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device, operations=operations)
        self.block = nn.Sequential(CausalConv1d(1, 64, 7, **args),
                                   *(EncoderBlock(dim, stride, layers, **args) for dim, stride, layers in ((128, 2, 0), (256, 4, 0), (512, 8, 0), (1024, 8, 4))),
                                   Snake1d(1024, dtype, device), CausalConv1d(1024, 1024, 3, **args))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device, operations=operations)
        self.model = nn.Sequential(CausalConv1d(1024, 1536, 7, **args),
                                   *(DecoderBlock(1536 // 2 ** i, 768 // 2 ** i, stride, **args) for i, stride in enumerate((8, 8, 4, 2))),
                                   Snake1d(96, dtype, device), CausalConv1d(96, 1, 7, **args), nn.Tanh())

    def forward(self, x):
        return self.model(x)


class Drama3Codec(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device, operations=operations)
        self.encoder = Encoder(**args)
        self.quantizer = Quantizer(**args)
        self.decoder = Decoder(**args)

    def encode(self, audio):
        """Mono audio [B, samples] at 44.1 kHz -> codes [B, 10, frames]; the audio is padded to whole frames."""
        audio = F.pad(audio, (0, -audio.shape[-1] % FRAME_LENGTH))
        return self.quantizer.encode(self.encoder(audio[:, None]))

    def decode(self, codes, dtype):
        """Codes [B, 10, frames] -> audio [B, frames * 2048]."""
        pieces = []
        for start in range(0, codes.shape[-1], DECODE_WINDOW):
            first = max(0, start - DECODE_CONTEXT)
            audio = self.decoder(self.quantizer.decode(codes[..., first:start + DECODE_WINDOW], dtype))[:, 0]
            pieces.append(audio[:, (start - first) * FRAME_LENGTH:])
        if not pieces:
            raise ValueError("Cannot decode empty DRAMA codes")
        return torch.cat(pieces, dim=-1)
