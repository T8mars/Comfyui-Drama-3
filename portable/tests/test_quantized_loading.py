"""Meta construction must retain every physical quantized checkpoint tensor."""
import json
import contextlib
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from drama3_t8.native.loader import load_checkpoint_storage
from drama3_t8.native.offload import QuantHostBank, tensor_fields


def quantized_bank(fmt, experts=2, rows=16, width=256):
    generator = torch.Generator().manual_seed(27)
    conf = {"format": fmt, "num_experts": experts, "convrot": True, "convrot_groupsize": 256}
    if fmt == "int8_tensorwise":
        entries = {
            "weight": torch.randint(-127, 128, (experts, rows, width), dtype=torch.int8, generator=generator),
            "weight_scale": torch.rand(experts, rows, 1, generator=generator) * .02 + .01,
        }
    else:
        bits = 4 if fmt == "asym_w4a8_int8" else 6
        conf["group_size"] = 16
        entries = {
            "weight": torch.randint(-128, 128, (experts, rows, width * bits // 8), dtype=torch.int8, generator=generator),
            "weight_s_rel": torch.ones(experts, rows, width // 16).to(torch.float8_e4m3fn).view(torch.uint8),
            "weight_s_channel": torch.rand(experts, rows, generator=generator) * .02 + .01,
        }
        if bits == 4:
            entries["weight_codebook"] = torch.linspace(-7, 7, 16)
    entries["comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
    return entries


@pytest.mark.parametrize("fmt", ["int8_tensorwise", "asym_w4a8_int8", "w6a8_int8"])
def test_meta_quantized_bank_loads_packed_data_scales_and_host_selection_from_checkpoint(fmt):
    state = quantized_bank(fmt)
    source_pointer = state["weight"].data_ptr()
    expected = {name: value.clone() for name, value in state.items()}
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    bank = ops.MoEExperts(2, 256, 16, bias=False, device="meta")
    load_checkpoint_storage(bank, state)
    weight = bank.weight
    assert isinstance(weight, QuantizedTensor)
    assert weight._qdata.device.type == "cpu"
    assert weight._qdata.data_ptr() == source_pointer
    assert torch.equal(weight._qdata, expected["weight"].flatten(0, 1))
    fields = tensor_fields(weight)
    assert fields and all(value.device.type == "cpu" for value in fields.values())
    scale_name = "weight_scale" if fmt == "int8_tensorwise" else "weight_s_rel"
    scale = expected[scale_name].flatten(0, 1)
    if scale_name == "weight_s_rel":
        scale = scale.view(torch.float8_e4m3fn)
    assert torch.equal(fields["scale"].view(torch.uint8), scale.view(torch.uint8))
    if fmt != "int8_tensorwise":
        assert torch.equal(fields["s_channel"], expected["weight_s_channel"].flatten(0, 1))
    if fmt == "asym_w4a8_int8":
        assert torch.equal(fields["codebook"], expected["weight_codebook"])
    host = QuantHostBank(bank)
    selected = host.selected_to_device([1, 0], "cpu")
    reconstructed = host.weight(selected, torch.device("cpu"))
    assert torch.equal(reconstructed._qdata, expected["weight"][[1, 0]].flatten(0, 1))
    dequantized = reconstructed.dequantize()
    assert dequantized.shape == (32, 256) and torch.isfinite(dequantized).all()


def test_meta_dense_assign_uses_checkpoint_storage_and_materializes_buffers():
    model = torch.nn.Linear(4, 3, device="meta", dtype=torch.bfloat16)
    model.register_buffer("scale", torch.empty(3, device="meta"))
    state = {"weight": torch.arange(12).reshape(3, 4).to(torch.bfloat16),
             "bias": torch.ones(3, dtype=torch.bfloat16), "scale": torch.tensor([1., 2., 3.])}
    source = state["weight"]
    load_checkpoint_storage(model, state)
    assert model.weight.data_ptr() == source.data_ptr()
    assert model.scale.device.type == "cpu" and torch.equal(model.scale, torch.tensor([1., 2., 3.]))


def test_meta_mixed_precision_dense_modules_materialize_both_dense_and_quantized_weights():
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    model = torch.nn.ModuleDict({"dense": ops.Linear(256, 16, bias=True, device="meta"),
                                "quantized": ops.Linear(256, 16, bias=False, device="meta")})
    bank = quantized_bank("int8_tensorwise", experts=1)
    conf = json.loads(bytes(bank["comfy_quant"].tolist()))
    conf.pop("num_experts")
    source = torch.arange(16 * 256).reshape(16, 256).to(torch.bfloat16)
    state = {"dense.weight": source, "dense.bias": torch.ones(16, dtype=torch.bfloat16),
             "quantized.weight": bank["weight"][0], "quantized.weight_scale": bank["weight_scale"][0],
             "quantized.comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}
    load_checkpoint_storage(model, state)
    assert model["dense"].weight.data_ptr() == source.data_ptr()
    assert model["dense"].bias.device.type == "cpu"
    assert torch.equal(model["quantized"].weight._qdata, bank["weight"][0])
    assert all(value.device.type == "cpu" for value in tensor_fields(model["quantized"].weight).values())


def test_missing_quantized_weight_is_rejected_before_host_can_silently_drop_it():
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    model = ops.MoEExperts(2, 256, 16, bias=False, device="meta")
    state = quantized_bank("int8_tensorwise")
    state.pop("weight")
    with pytest.raises(ValueError, match="missing weight"):
        load_checkpoint_storage(model, state)


def test_unmaterialized_parameter_is_rejected_instead_of_reinitialized():
    model = torch.nn.Linear(4, 3, device="meta")
    with pytest.raises(ValueError, match="unmaterialized storage"):
        load_checkpoint_storage(model, {"weight": torch.empty(3, 4, device="meta"),
                                        "bias": torch.zeros(3)})


def test_guard_checks_quantized_scale_even_when_public_tensor_device_is_cpu():
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    params = TensorWiseINT8Layout.Params(scale=torch.empty(16, 1, device="meta"),
        orig_shape=(16, 256), orig_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256)
    weight = QuantizedTensor(torch.zeros(16, 256, dtype=torch.int8), "TensorWiseINT8Layout", params)
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    with pytest.raises(ValueError, match="unmaterialized storage: weight.scale"):
        load_checkpoint_storage(model, {"weight": weight})


def test_drama_session_supports_old_and_accelerated_stores_without_enabling_cpu_decode():
    from drama3_t8.native.loader import Drama3
    drama = object.__new__(Drama3)
    drama.patcher = SimpleNamespace(load_device=torch.device("cpu"))
    drama.second_patcher = None
    called = []
    @contextlib.contextmanager
    def old(device, patchers):
        called.append((device, patchers))
        yield
    drama.experts = SimpleNamespace(session=old)
    with drama.session(cpu_decode=True):
        pass
    assert called == [(drama.patcher.load_device, [drama.patcher])]
    scene_head = SimpleNamespace(load_device=torch.device("cpu"))
    with drama.session(extra_patchers=[scene_head]):
        pass
    assert called[-1] == (drama.patcher.load_device, [drama.patcher, scene_head])
    @contextlib.contextmanager
    def accelerated(device, patchers, cpu_decode=False):
        called.append(cpu_decode)
        yield
    drama.experts.session = accelerated
    with drama.session():
        pass
    with drama.session(cpu_decode=True):
        pass
    assert called[-2:] == [False, True]
    drama.experts = None
    with drama.session():
        pass


def test_reference_store_nested_session_reuses_cache_until_outer_exit(monkeypatch):
    from drama3_t8.native import offload
    calls = []
    class Pool:
        def __init__(self, *args):
            calls.append("allocate")
        def report(self):
            calls.append("report")
    monkeypatch.setattr(offload, "SlotPool", Pool)
    monkeypatch.setattr(offload.comfy.model_management, "get_free_memory", lambda device: 4 * 1024**3)
    monkeypatch.setattr(offload.comfy.model_management, "soft_empty_cache", lambda: calls.append("free"))
    store = offload.ExpertStore(1024)
    store.banks = [(SimpleNamespace(num_experts=4),)]
    with store.session(torch.device("cpu"), []):
        cache = store.cache
        with pytest.raises(RuntimeError, match="inner"):
            with store.session(torch.device("cpu"), []):
                assert store.cache is cache
                raise RuntimeError("inner")
        assert store.cache is cache
        assert calls == ["allocate"]
    assert store.cache is None and calls == ["allocate", "report", "free"]
