"""Fish Audio reference DeltaNet math, Community License; see licenses/."""
import torch
import torch.nn.functional as F
from torch import Tensor
GDN_CHUNK_SIZE = 64

def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = GDN_CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Gated delta rule over whole chunks, in fp32.

    query / key [B, T, H, dk] (not yet l2-normalized), value [B, T, H, dv], decay
    (log space, <= 0) and beta [B, T, H], initial_state [B, H, dk, dv]. Returns the
    output [B, T, H, dv] in the query dtype and the final fp32 state.
    """
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (
        t.transpose(1, 2).to(torch.float32).contiguous()
        for t in (query, key, value, beta, decay)
    )
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)

    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, decay = (F.pad(t, (0, pad)) for t in (beta, decay))
    chunk_count = (length + pad) // chunk_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = (
        t.reshape(batch, heads, chunk_count, chunk_size, t.shape[-1])
        for t in (query, key, key_beta, value_beta)
    )
    decay = decay.reshape(batch, heads, chunk_count, chunk_size)
    upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cumulative = decay.cumsum(dim=-1)
    pairwise = (
        (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2))
        .masked_fill(upper, float("-inf"))
        .exp()
    )

    system = (key_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_key_beta = key_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        system, value_beta, upper=False, unitriangular=True
    )
    key_cumulative_decay = torch.linalg.solve_triangular(
        system, decayed_key_beta, upper=False, unitriangular=True
    )

    if initial_state is None:
        state = query.new_zeros(batch, heads, key_dim, value_dim)
    else:
        state = initial_state.to(torch.float32)
    output = torch.empty_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    for index in range(chunk_count):
        corrected = new_values[:, :, index] - key_cumulative_decay[:, :, index] @ state
        output[:, :, index] = (
            query[:, :, index] @ state + intra[:, :, index] @ corrected
        )
        state = (
            state * chunk_decay[:, :, index]
            + key[:, :, index].transpose(-1, -2) @ corrected
        )

    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).contiguous().to(output_dtype), state

def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Token-by-token gated delta rule; same contract as chunk_gated_delta_rule."""
    output_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, decay = (
        t.transpose(1, 2).to(torch.float32).contiguous()
        for t in (query, key, value, beta, decay)
    )
    query = l2norm(query) * key_dim**-0.5
    key = l2norm(key)
    if initial_state is None:
        state = value.new_zeros(batch, heads, key_dim, value_dim)
    else:
        state = initial_state.to(torch.float32)
    output = torch.empty_like(value)
    for index in range(length):
        state = state * decay[:, :, index].exp()[..., None, None]
        k_t = key[:, :, index]
        remembered = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, index] - remembered) * beta[:, :, index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = (state * query[:, :, index].unsqueeze(-1)).sum(dim=-2)
    return output.transpose(1, 2).contiguous().to(output_dtype), state
