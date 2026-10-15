import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tools.safetensors_stream import write_tensors


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
