"""Read only the required scene-control rows, including from legacy full heads."""
from pathlib import Path

import torch
from safetensors import safe_open


def control_rows(path, token_ids, hidden_size):
    path=Path(path)
    ids=tuple(token_ids)
    if not ids or len(set(ids))!=len(ids):
        raise ValueError("Scene control IDs must be nonempty and unique")
    with safe_open(str(path),framework="pt",device="cpu") as archive:
        if "weight" not in archive.keys():
            raise ValueError("Scene head is missing its weight tensor")
        source=archive.get_slice("weight")
        shape=source.get_shape()
        if len(shape)!=2 or shape[1]!=hidden_size:
            raise ValueError("Scene head hidden size does not match the DRAMA model")
        if "ids" in archive.keys():
            stored=archive.get_tensor("ids")
            if stored.ndim!=1 or stored.dtype!=torch.int64 or len(stored)!=shape[0] or len(stored)>32:
                raise ValueError("Invalid compact scene head IDs")
            lookup=stored.tolist()
            if len(set(lookup))!=len(lookup) or any(i not in lookup for i in ids):
                raise ValueError("Compact scene head is missing required control rows")
            rows=[source[lookup.index(i):lookup.index(i)+1] for i in ids]
        else:
            if min(ids)<0 or max(ids)>=shape[0]:
                raise ValueError("Full text head does not contain the required controls")
            rows=[source[i:i+1] for i in ids]
    weight=torch.cat(rows).contiguous()
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("Scene head must contain finite floating-point weights")
    return weight
