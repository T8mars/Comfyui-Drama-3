"""Qwen3-ASR: an audio encoder whose output replaces the audio pad tokens of a Qwen3 decoder's prompt.

The encoder cuts the log-mel features into 1 s chunks (100 frames), downsamples each with three stride-2
convolutions (13 tokens), adds sinusoidal positions within the chunk, and attends within windows of 8 chunks, as
Qwen's vLLM backend does (and its streaming relies on). Qwen's transformers backend attends across the whole clip
instead; the two agree up to 8 s.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Qwen3_06B, Qwen3_06BConfig

CHUNK_FRAMES = 100
WINDOW_CHUNKS = 8
# chunks per convolution call; a 20 min clip has 1200
CONV_BATCH = 500


def conv_tokens(frames):
    """Tokens three stride-2 convolutions make from `frames` frames."""
    for _ in range(3):
        frames = (frames - 1) // 2 + 1
    return frames


def audio_tokens(frames):
    """Encoder tokens of a clip of `frames` mel frames: 13 per full chunk, then the tail chunk's."""
    full, tail = divmod(frames, CHUNK_FRAMES)
    return full * conv_tokens(CHUNK_FRAMES) + conv_tokens(tail)


def sinusoids(length, channels, device):
    scale = math.log(10000) / (channels // 2 - 1)
    angles = torch.arange(length, device=device, dtype=torch.float32)[:, None] * torch.exp(
        -scale * torch.arange(channels // 2, device=device, dtype=torch.float32))
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class EncoderAttention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x):
        attention = optimized_attention_for_device(x.device, small_input=True)
        return self.out_proj(attention(self.q_proj(x), self.k_proj(x), self.v_proj(x), self.heads))


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, ffn, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = EncoderAttention(dim, heads, device=device, dtype=dtype, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.fc1 = operations.Linear(dim, ffn, device=device, dtype=dtype)
        self.fc2 = operations.Linear(ffn, dim, device=device, dtype=dtype)
        self.final_layer_norm = operations.LayerNorm(dim, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class AudioEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        dim, hidden = config["d_model"], config["downsample_hidden_size"]
        self.conv2d1 = operations.Conv2d(1, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d2 = operations.Conv2d(hidden, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv2d3 = operations.Conv2d(hidden, hidden, 3, 2, padding=1, device=device, dtype=dtype)
        self.conv_out = operations.Linear(hidden * conv_tokens(config["num_mel_bins"]), dim, bias=False, device=device, dtype=dtype)
        self.layers = nn.ModuleList(EncoderLayer(dim, config["encoder_attention_heads"], config["encoder_ffn_dim"], device=device,
                                                 dtype=dtype, operations=operations) for _ in range(config["encoder_layers"]))
        self.ln_post = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.proj1 = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.proj2 = operations.Linear(dim, config["output_dim"], device=device, dtype=dtype)

    def forward(self, features):
        """Log-mel features [mels, frames] -> one embedding per audio token [tokens, output_dim]."""
        chunks = features.split(CHUNK_FRAMES, dim=-1)
        lengths = [chunk.shape[-1] for chunk in chunks]
        # every chunk is zero-padded to the longest, as Qwen's encoder batches them
        x = torch.stack([F.pad(chunk, (0, lengths[0] - length)) for chunk, length in zip(chunks, lengths)])[:, None]
        convolved = []
        for batch in x.split(CONV_BATCH):
            for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
                batch = F.gelu(conv(batch))
            convolved.append(self.conv_out(batch.permute(0, 3, 1, 2).flatten(2)))
        x = torch.cat(convolved)
        x = x + sinusoids(x.shape[1], x.shape[2], x.device).to(x.dtype)
        x = torch.cat([chunk[:conv_tokens(length)] for chunk, length in zip(x, lengths)])

        window = conv_tokens(lengths[0]) * WINDOW_CHUNKS
        full = x.shape[0] // window * window
        windows = []
        if full:
            windows.append(x[:full].unflatten(0, (-1, window)))
        if full < x.shape[0]:
            windows.append(x[full:][None])
        for layer in self.layers:
            windows = [layer(part) for part in windows]
        x = torch.cat([part.flatten(0, 1) for part in windows])
        return self.proj2(F.gelu(self.proj1(self.ln_post(x))))


class Qwen3ASRModel(nn.Module):
    """The audio encoder and the Qwen3 decoder (ComfyUI's, with its KV cache and generation loop)."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        text = config["text_config"]
        self.aligner = "classify_num" in config
        if text["head_dim"] != Qwen3_06BConfig.head_dim or not (text["tie_word_embeddings"] or self.aligner):
            raise ValueError("Qwen3-ASR decoders with another head size or an untied LM head are not supported")
        if self.aligner:
            self.timestamp_head = operations.Linear(text["hidden_size"], config["classify_num"], bias=False, device=device, dtype=dtype)
            self.timestamp_token_id = config["timestamp_token_id"]
            self.timestamp_segment_time = config["timestamp_segment_time"]
        self.audio_tower = AudioEncoder(config["audio_config"], device=device, dtype=dtype, operations=operations)
        self.decoder = Qwen3_06B({key: text[key] for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                                                              "num_attention_heads", "num_key_value_heads", "rms_norm_eps", "rope_theta")},
                                 dtype, device, operations)
