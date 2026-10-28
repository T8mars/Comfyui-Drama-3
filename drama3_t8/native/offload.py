"""Routed experts held in host memory instead of VRAM, for cards that cannot hold all 32 B of them.

A decode step uses 8 of a layer's 256 experts, so only those are copied to the GPU, and a least-recently-used
cache in the VRAM left free keeps the ones that come back. The cache lives for one generation. A prompt uses
nearly every expert, so its pass streams each layer's bank in chunks, keeping the experts the prompt used most.
"""
import collections
import contextlib
import dataclasses
import logging

import torch
import torch.nn as nn

import comfy.model_management
import comfy.ops
from comfy.quant_ops import QuantizedTensor

PREFILL_CHUNK = 64
CACHE_RESERVE = 2 * 1024 ** 3
FP8_BLOCK = 128


def pinned(tensor):
    if comfy.model_management.DISABLE_PINNED_MEMORY:
        return tensor.contiguous()
    out = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    return out.copy_(tensor)


def tensor_fields(weight):
    params = weight._params
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(params) if torch.is_tensor(getattr(params, f.name))}


def per_row_fields(weight):
    """The bank's tensors that hold one entry per weight row: these are sliced or gathered with the rows."""
    rows = weight._qdata.shape[0]
    return {"_qdata": weight._qdata, **{n: v for n, v in tensor_fields(weight).items() if v.ndim >= 1 and v.shape[0] == rows}}


def gather_experts(weight, ids, num_experts):
    """Experts `ids` [k] (a device tensor) of a flat [E * out, in] bank as one [k * out, in] weight, on the GPU."""
    if not isinstance(weight, QuantizedTensor):
        return weight.view(num_experts, -1, weight.shape[-1]).index_select(0, ids).flatten(0, 1)
    rows = weight._qdata.shape[0] // num_experts
    picked = {n: v.view(num_experts, rows, *v.shape[1:]).index_select(0, ids).flatten(0, 1) for n, v in per_row_fields(weight).items()}
    qdata = picked.pop("_qdata")
    return QuantizedTensor(qdata, weight._layout_cls,
                           dataclasses.replace(weight._params, orig_shape=(len(ids) * rows, weight._params.orig_shape[-1]), **picked))


def dequantize_fp8(weight, scale, dtype):
    """Float8 rows [R, C] with one float32 scale per 128 x 128 block [R / 128, C / 128] -> dtype, as Fish's code does."""
    rows, cols = weight.shape
    blocks = weight.float().view(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    return (blocks * scale[:, None, :, None]).view(rows, cols).to(dtype)


def experts_decode(x, gate_up, down, weights):
    """One row x [1, H] through k experts at once, as two matmuls. gate_up [k * 2I, H]: every expert sees the same
    row. down [k * H, I]: each expert's input row meets every expert's weights and only the diagonal blocks are
    kept; the weights are read once either way, which is what a one-row matmul costs."""
    k = weights.shape[0]
    hidden = comfy.ops._swiglu_eager(nn.functional.linear(x, gate_up).view(k, -1))
    if isinstance(down, QuantizedTensor):
        out = nn.functional.linear(hidden, down).view(k, k, -1).diagonal(dim1=0, dim2=1).T
    else:
        out = torch.bmm(down.view(k, -1, hidden.shape[-1]), hidden.unsqueeze(-1)).squeeze(-1)
    return (out * weights[:, None].to(out.dtype)).sum(0, keepdim=True)


class HostBank:
    """One CPU expert bank, optionally pinned; the first dimension holds each expert's rows in turn."""

    def __init__(self, num_experts, fields, *, pin_memory=True):
        self.num_experts = num_experts
        self.fields = {n: pinned(t) if pin_memory else t.contiguous() for n, t in fields.items()}

    def rows(self, name):
        return self.fields[name].shape[0] // self.num_experts

    def take(self, fields, first, last):
        """Experts [first, last) of `fields`, host or device, laid out like this bank's."""
        return {n: t[first * self.rows(n):last * self.rows(n)] for n, t in fields.items()}

    def to_device(self, first, last, device):
        return {n: t.to(device, non_blocking=True) for n, t in self.take(self.fields, first, last).items()}

    def selected_to_device(self, experts, device):
        ids = torch.tensor(experts, dtype=torch.long)
        return {n: t.view(self.num_experts, self.rows(n), *t.shape[1:]).index_select(0, ids)
                  .flatten(0, 1).to(device, non_blocking=True) for n, t in self.fields.items()}


class QuantHostBank(HostBank):
    """A flat `comfy.ops.MoEExperts` bank (int8 ConvRot, W4A8, W6A8); its experts run as QuantizedTensors."""

    def __init__(self, module, *, pin_memory=True):
        weight = module.weight
        if not isinstance(weight, QuantizedTensor) or weight._qdata.ndim != 2:
            raise ValueError("Drama 3 keeps only flat quantized or FP8 expert banks in host memory")
        fields = per_row_fields(weight)
        super().__init__(module.num_experts, fields, pin_memory=pin_memory)
        self.shared = {n: v for n, v in tensor_fields(weight).items() if n not in fields}
        self.shared_on = {}
        self.layout,self.in_features=weight._layout_cls,weight._params.orig_shape[-1]
        self.params=dataclasses.replace(weight._params,**{n:v for n,v in self.fields.items() if n!='_qdata'},**self.shared)

    def weight(self, fields, device):
        # a CPU codebook would send every W4A8 matmul down comfy_kitchen's eager path
        if device not in self.shared_on:
            self.shared_on[device] = {n: v.to(device) for n, v in self.shared.items()}
        fields = dict(fields)
        qdata = fields.pop("_qdata")
        return QuantizedTensor(qdata, self.layout, dataclasses.replace(self.params, orig_shape=(qdata.shape[0], self.in_features),
                                                                       **fields, **self.shared_on[device]))


class FP8HostBank(HostBank):
    """Experts as Fish ships them, float8 rows and block scales; their experts run in bf16."""

    def __init__(self, weight, scale, *, pin_memory=True):
        super().__init__(weight.shape[0], {"weight": weight.view(torch.uint8).flatten(0, 1), "scale": scale.flatten(0, 1)},
                         pin_memory=pin_memory)

    def weight(self, fields, device):
        return dequantize_fp8(fields["weight"].view(torch.float8_e4m3fn), fields["scale"], torch.bfloat16)


class ExpertStore:
    def __init__(self, cache_bytes=0, *, pin_memory=True):
        """`cache_bytes` 0 sizes the expert cache to the VRAM free when a generation starts."""
        self.banks = []
        self.cache = None
        self.cache_bytes = cache_bytes
        self.pin_memory = bool(pin_memory)
        self.backend_calls={"torch":0,"cuda":0}
        self.last_backend="torch"
        self.fallback_reason=None

    def close(self):
        if self.cache is not None:raise RuntimeError('Cannot unload an active expert generation')
        self.banks.clear()

    def add(self, experts):
        """Take a layer's experts into host memory; returns the module that replaces them in the model."""
        if isinstance(experts, FP8Experts):
            banks = (FP8HostBank(experts.gate_up_weight, experts.gate_up_scale, pin_memory=self.pin_memory),
                     FP8HostBank(experts.down_weight, experts.down_scale, pin_memory=self.pin_memory))
        else:
            banks = (QuantHostBank(experts.gate_up_proj, pin_memory=self.pin_memory),
                     QuantHostBank(experts.down_proj, pin_memory=self.pin_memory))
        self.banks.append(banks)
        return OffloadedExperts(self, len(self.banks) - 1)

    @contextlib.contextmanager
    def session(self, device, patchers):
        """One generation's expert cache, sized to the VRAM free at its start less what the patchers' weights on
        this device still need: a dynamic model's weights arrive on first use, and must not find the cache there."""
        if self.cache is not None:
            yield
            return
        pending = sum(p.model_size() - p.loaded_size() for p in patchers if p.load_device == device)
        budget = max(0, comfy.model_management.get_free_memory(device) - pending - CACHE_RESERVE)
        if self.cache_bytes:
            budget = min(budget, self.cache_bytes)
        try:
            self.cache = SlotPool(self.banks[0], budget, len(self.banks) * self.banks[0][0].num_experts, device)
            yield
        finally:
            import sys
            primary = sys.exc_info()[1]
            try:
                if self.cache is not None:
                    try:
                        self.cache.report()
                    except BaseException as exc:
                        if primary is None and not isinstance(exc, Exception):
                            raise
                        logging.warning('Drama expert cache diagnostics failed during cleanup: %s', exc)
            finally:
                self.cache = None
                try:
                    comfy.model_management.soft_empty_cache()
                except BaseException as exc:
                    if primary is None and not isinstance(exc, Exception):
                        raise
                    logging.warning('Drama expert device cache cleanup failed: %s', exc)

    def forward(self, layer, x, topk_idx, topk_weight):
        self.backend_calls["torch"]+=1
        banks = self.banks[layer]
        if x.shape[0] == 1 and self.cache is not None:
            slots = self.cache.slots(layer, topk_idx[0].tolist(), banks)
            gate_up, down = (bank.weight(self.cache.gather(index, slots), x.device) for index, bank in enumerate(banks))
            return experts_decode(x, gate_up, down, topk_weight[0])
        return self.prefill(layer, banks, x, topk_idx, topk_weight)

    def prefill(self, layer, banks, x, topk_idx, topk_weight):
        gate_up, down = banks
        num_tokens, top_k = topk_idx.shape
        order = torch.argsort(topk_idx.reshape(-1))
        counts = torch.bincount(topk_idx.reshape(-1), minlength=gate_up.num_experts).tolist()
        sorted_x = x[order // top_k]
        weight = topk_weight.reshape(-1)[order].unsqueeze(1)
        sorted_out = torch.empty_like(sorted_x)
        # the experts this prompt used most stay in the cache, up to the layer's share of it
        keep = set()
        if self.cache is not None:
            share = self.cache.count // len(self.banks)
            keep = set(sorted(range(len(counts)), key=counts.__getitem__, reverse=True)[:share])
        start = 0
        for first in range(0, len(counts), PREFILL_CHUNK):
            last = min(first + PREFILL_CHUNK, len(counts))
            active = [expert for expert in range(first, last) if counts[expert]]
            if not active:
                continue
            chunks = (gate_up.selected_to_device(active, x.device), down.selected_to_device(active, x.device))
            for index, expert in enumerate(active):
                n = counts[expert]
                rows = slice(start, start + n)
                hidden = comfy.ops._swiglu_eager(nn.functional.linear(sorted_x[rows], gate_up.weight(gate_up.take(chunks[0], index, index + 1), x.device)))
                sorted_out[rows] = nn.functional.linear(hidden, down.weight(down.take(chunks[1], index, index + 1), x.device)) * weight[rows]
                start += n
            for index, expert in enumerate(active):
                if expert in keep:
                    self.cache.insert((layer, expert), [bank.take(chunk, index, index + 1) for bank, chunk in zip(banks, chunks)])
        out = torch.empty_like(sorted_out)
        out[order] = sorted_out
        return out.view(num_tokens, top_k, -1).sum(dim=1)


class SlotPool:
    """A fixed number of expert slots on the GPU, filled from host memory on a miss, least recently used out first."""

    def __init__(self, banks, budget, total_experts, device):
        self.rows = [{n: bank.rows(n) for n in bank.fields} for bank in banks]
        per_expert = sum(t[:bank.rows(n)].nbytes for bank in banks for n, t in bank.fields.items())
        self.count = max(8, min(total_experts, budget // per_expert))
        self.fields = [{n: torch.empty((self.count * bank.rows(n), *t.shape[1:]), dtype=t.dtype, device=device) for n, t in bank.fields.items()}
                       for bank in banks]
        self.device = device
        self.owner = collections.OrderedDict()
        self.free = list(range(self.count))
        self.hits = self.misses = 0

    def _fill(self, slot, sources):
        for fields, rows, source in zip(self.fields, self.rows, sources):
            for name, target in fields.items():
                target[slot * rows[name]:(slot + 1) * rows[name]].copy_(source[name], non_blocking=True)

    def _claim(self, key):
        slot = self.free.pop() if self.free else self.owner.popitem(last=False)[1]
        self.owner[key] = slot
        return slot

    def slots(self, layer, experts, banks):
        for expert in experts:
            if (layer, expert) in self.owner:
                self.owner.move_to_end((layer, expert))
        slots = []
        for expert in experts:
            slot = self.owner.get((layer, expert))
            if slot is None:
                self.misses += 1
                slot = self._claim((layer, expert))
                self._fill(slot, [bank.take(bank.fields, expert, expert + 1) for bank in banks])
            else:
                self.hits += 1
            slots.append(slot)
        return torch.tensor(slots, device=self.device)

    def insert(self, key, sources):
        """Keep an expert already on the GPU (a prompt chunk), without counting a read."""
        if key not in self.owner:
            self._fill(self._claim(key), sources)

    def gather(self, index, slots):
        return {n: t.view(self.count, self.rows[index][n], *t.shape[1:]).index_select(0, slots).flatten(0, 1)
                for n, t in self.fields[index].items()}

    def report(self):
        total = self.hits + self.misses
        if total:
            logging.info("Drama 3 expert cache: %d experts (%.1f GiB), %.0f%% of %d expert reads hit", self.count,
                         sum(t.nbytes for f in self.fields for t in f.values()) / 1024 ** 3, 100 * self.hits / total, total)


class FP8Experts(nn.Module):
    """Routed experts as Fish ships them: float8 e4m3 weights with a float32 scale per 128 x 128 block. A use turns
    only the experts it needs into the compute dtype."""

    def __init__(self, num_experts, hidden, intermediate, device=None, dtype=None):
        super().__init__()
        self.num_experts = num_experts
        self.dtype = dtype
        for name, rows, cols in (("gate_up", 2 * intermediate, hidden), ("down", hidden, intermediate)):
            self.register_parameter(f"{name}_weight", nn.Parameter(torch.empty((num_experts, rows, cols), dtype=torch.float8_e4m3fn, device=device), requires_grad=False))
            self.register_parameter(f"{name}_scale", nn.Parameter(torch.empty((num_experts, rows // FP8_BLOCK, cols // FP8_BLOCK), dtype=torch.float32, device=device), requires_grad=False))

    def experts(self, name, ids):
        weight = getattr(self, f"{name}_weight").view(torch.uint8).index_select(0, ids).view(torch.float8_e4m3fn)
        return dequantize_fp8(weight.flatten(0, 1), getattr(self, f"{name}_scale").index_select(0, ids).flatten(0, 1), self.dtype)

    def forward(self, x, topk_idx, topk_weight):
        if x.shape[0] == 1:
            return experts_decode(x, self.experts("gate_up", topk_idx[0]), self.experts("down", topk_idx[0]), topk_weight[0])
        num_tokens, top_k = topk_idx.shape
        order = torch.argsort(topk_idx.reshape(-1))
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).tolist()
        sorted_x = x[order // top_k]
        weight = topk_weight.reshape(-1)[order].unsqueeze(1)
        sorted_out = torch.empty_like(sorted_x)
        start = 0
        for expert, n in enumerate(counts):
            if n == 0:
                continue
            ids = torch.tensor([expert], device=x.device)
            rows = slice(start, start + n)
            hidden = comfy.ops._swiglu_eager(nn.functional.linear(sorted_x[rows], self.experts("gate_up", ids)))
            sorted_out[rows] = nn.functional.linear(hidden, self.experts("down", ids)) * weight[rows]
            start += n
        out = torch.empty_like(sorted_out)
        out[order] = sorted_out
        return out.view(num_tokens, top_k, -1).sum(dim=1)


class OffloadedExperts(nn.Module):
    """Stands in for a layer's experts when they stay in host memory."""

    def __init__(self, store, layer):
        super().__init__()
        self.store = store
        self.layer = layer

    def forward(self, x, topk_idx, topk_weight):
        return self.store.forward(self.layer, x, topk_idx, topk_weight)
