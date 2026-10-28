"""Preserve DeltaNet state for arbitrary continuation lengths; optional FLA kernels."""
import torch
import torch.nn.functional as F
import comfy.ops
import logging
from comfy.text_encoders.qwen35 import GatedDeltaNet as BaseDeltaNet

from .delta_reference import chunk_gated_delta_rule, recurrent_gated_delta_rule
from .norms import GatedRMSNorm


class DramaDeltaNet(BaseDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = GatedRMSNorm(self.value_head_dim, self.norm.eps,
                                 device=self.norm.weight.device, dtype=self.norm.weight.dtype)
        self.backend = "torch"
        self.backend_calls = {"torch": 0, "fla": 0}
        self.fallback_reason = None

    def forward(self, x, past_key_value=None, **kwargs):
        batch, length, _ = x.shape
        projected = self.in_proj_qkv(x).transpose(1, 2)
        z, beta, a = self.in_proj_z(x), self.in_proj_b(x).float().sigmoid(), self.in_proj_a(x)
        previous = past_key_value is not None and past_key_value.index > 0
        if previous:
            combined = torch.cat((past_key_value.conv_state, projected), dim=-1)
            with comfy.ops.CastBiasWeightContext(self.conv1d, combined, offloadable=True) as (weight, bias):
                mixed = F.silu(F.conv1d(combined, weight, bias, groups=self.conv1d.groups))
        else:
            combined = F.pad(projected, (self.conv_kernel_size - 1, 0))
            mixed = F.silu(self.conv1d(projected)[:, :, :length])
        if past_key_value is not None:
            past_key_value.conv_state.copy_(combined[:, :, -(self.conv_kernel_size - 1):])
        query, key, value = mixed.transpose(1, 2).split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        decay = -self.A_log.to(device=x.device, dtype=torch.float32).exp()
        g = decay * F.softplus(a.float() + self.dt_bias.to(device=x.device, dtype=torch.float32))
        query = query.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, length, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            repeat = self.num_value_heads // self.num_key_heads
            query, key = query.repeat_interleave(repeat, 2), key.repeat_interleave(repeat, 2)
        initial = past_key_value.recurrent_state if previous else None
        if self.backend == "fla":
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk, fused_recurrent_gated_delta_rule as recurrent
                operation = recurrent if length == 1 else chunk
                output, state = operation(query, key, value, g=g, beta=beta, initial_state=initial,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                compiler_failure = (isinstance(exc, (ImportError, OSError)) or type(exc).__module__.startswith('triton')
                                    or type(exc).__name__ in {'CalledProcessError','CompileError'})
                if not compiler_failure:
                    raise
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                self.backend = "torch"
                logging.warning("Drama FLA unavailable; using Torch: %s", self.fallback_reason)
        if self.backend == "torch":
            operation = recurrent_gated_delta_rule if length == 1 else chunk_gated_delta_rule
            output, state = operation(query, key, value, g, beta, initial_state=initial)
        self.backend_calls[self.backend] += 1
        if past_key_value is not None:
            past_key_value.recurrent_state.copy_(state)
        output = self.norm(output.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        return self.out_proj(output.reshape(batch, length, -1)), past_key_value
