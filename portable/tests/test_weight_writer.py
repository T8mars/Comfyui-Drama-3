import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tools.safetensors_stream import write_tensors,TensorSource,write_tensor_sources


def test_streaming_checkpoint_preserves_shapes_dtypes_bytes_and_metadata(tmp_path):
    values = {"bf16": torch.randn(7, 9).bfloat16(), "fp8": torch.randn(4, 3).to(torch.float8_e4m3fn),
              "codes": torch.arange(33).reshape(3, 11), "scalar": torch.tensor(1.25)}
    path = tmp_path / "weights.safetensors"
    write_tensors(values, path, {"format": "测试 v2"})
    actual = load_file(str(path))
    for name, value in values.items():
        assert actual[name].dtype == value.dtype
        assert actual[name].shape == value.shape
        assert torch.equal(actual[name].reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
    with safe_open(str(path), framework="pt") as file:
        assert file.metadata()["format"] == "测试 v2"


def test_lazy_checkpoint_streams_sources_once_and_keeps_scalar_and_fp8_bytes(tmp_path):
    values={'a':torch.tensor(1.25),'b':torch.randn(5,8).bfloat16(),'c':torch.randn(4,8).to(torch.float8_e4m3fn)}
    calls=[]
    def load(name):calls.append(name);return values[name]
    sources={k:TensorSource(tuple(v.shape),v.dtype,lambda k=k:load(k)) for k,v in values.items()}
    path=tmp_path/'lazy.safetensors'
    write_tensor_sources(sources,path,{'source':'local FP8'})
    actual=load_file(str(path))
    assert calls==['a','b','c']
    for key in values:
        assert actual[key].shape==values[key].shape
        assert torch.equal(actual[key].reshape(-1).view(torch.uint8),values[key].reshape(-1).view(torch.uint8))
