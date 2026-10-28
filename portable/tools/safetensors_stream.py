"""Write CPU tensors without materializing a second model-sized bytes dictionary."""
import json
import os
import struct
from dataclasses import dataclass

import torch

DTYPES = {
    torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8", torch.int16: "I16",
    torch.int32: "I32", torch.int64: "I64", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
}


@dataclass
class TensorSource:
    shape: tuple
    dtype: torch.dtype
    load: object


def write_tensor_sources(sources,path,metadata):
    """Plan the header from shapes, then materialize one tensor/bank at a time."""
    header={"__metadata__":metadata}
    offset=0
    for name,source in sorted(sources.items()):
        count=1
        for dimension in source.shape:count*=dimension
        size=count*torch.empty((),dtype=source.dtype).element_size()
        header[name]=dict(dtype=DTYPES[source.dtype],shape=list(source.shape),data_offsets=[offset,offset+size])
        offset+=size
    encoded=json.dumps(header,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    encoded+=b' '*(-len(encoded)%8)
    with open(path,'wb',buffering=0) as output:
        output.write(struct.pack('<Q',len(encoded)));output.write(encoded)
        for name,source in sorted(sources.items()):
            tensor=source.load()
            if tensor.device.type!='cpu' or not tensor.is_contiguous() or tuple(tensor.shape)!=tuple(source.shape) or tensor.dtype!=source.dtype:
                raise ValueError(f'Invalid lazy tensor source: {name}')
            view=memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
            for start in range(0,len(view),32*1024*1024):output.write(view[start:start+32*1024*1024])
            del view,tensor
        os.fsync(output.fileno())


def write_tensors(tensors, path, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"Streaming writer needs contiguous CPU storage: {name}")
        size = tensor.numel() * tensor.element_size()
        header[name] = {"dtype": DTYPES[tensor.dtype], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb", buffering=0) as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for name in sorted(tensors):
            view = memoryview(tensors[name].detach().reshape(-1).view(torch.uint8).numpy())
            for start in range(0, len(view), 32 * 1024 * 1024):
                file.write(view[start:start + 32 * 1024 * 1024])
        os.fsync(file.fileno())
