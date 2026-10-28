"""INT8/W4A8/W6A8 CUDA regressions against ComfyKitchen's CPU implementation.

The fixtures use the official CPU quantizers, not invented packed bytes.  The
primary oracle is the official eager INT8 linear (also used by eager W4/W6):
ConvRot, per-row dynamic A8, INT32 dot products, FP32 scales and SwiGLU.  A second
CPU check reconstructs A8 activations and the original-basis floating weight for
F.linear.  BF16 host matmul has different intermediate rounding; the four-layer
handoff test measures that separately from the strict FP32 kernel comparison.

Set DRAMA_CUDA_NUMERICS_REPORT to export measured errors as local JSON.  No model
download, server, or graph-invalidating fault injection is involved.
"""
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfy_kitchen.backends.eager import quantization as eager_int8
from comfy_kitchen.backends.eager import w4a8_int8 as eager_codes


FORMATS = ("int8_tensorwise", "asym_w4a8_int8", "w6a8_int8")
EXPERTS, WIDTH, INTERMEDIATE, TOP_K = 16, 256, 256, 8
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real expert numerics require CUDA")


@contextlib.contextmanager
def _cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _record(name, **values):
    path = os.environ.get("DRAMA_CUDA_NUMERICS_REPORT")
    if path:
        path = Path(path)
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        result[name] = values
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _errors(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = actual - expected
    rms = float(delta.square().mean().sqrt())
    reference_rms = float(expected.square().mean().sqrt())
    return dict(max_abs=float(delta.abs().max()), rms=rms,
                relative_rms=rms / max(reference_rms, 1e-30), reference_rms=reference_rms)


def _bank(weight, fmt):
    packed, scales, channels, table = [], [], [], None
    for source in weight:
        if fmt == "int8_tensorwise":
            q, scale = eager_int8.quantize_int8_convrot_weight(source, WIDTH)
        else:
            # This is the production converter's shared-codebook contract.  The
            # eager function is selected explicitly, so preparing fixtures never
            # schedules GPU work even on a machine with an active CUDA job.
            q, scale, channel, correction, table = eager_codes.quantize_w4a8_int8_weight(
                source.to(torch.bfloat16), group_size=16, convrot_groupsize=WIDTH,
                codebook_tensor=table, bits=4 if fmt == "asym_w4a8_int8" else 6)
            assert correction is None
            channels.append(channel)
        packed.append(q)
        scales.append(scale)
    conf = dict(format=fmt, num_experts=EXPERTS, convrot=True, convrot_groupsize=WIDTH)
    state = dict(weight=torch.stack(packed))
    if fmt == "int8_tensorwise":
        state["weight_scale"] = torch.stack(scales)
    else:
        conf["group_size"] = 16
        state["weight_s_rel"] = torch.stack(scales).view(torch.uint8)
        state["weight_s_channel"] = torch.stack(channels)
        if table is not None:
            state["weight_codebook"] = table
    state["comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
    return state


def _make_case(fmt, weight_scale=.025):
    generator = torch.Generator().manual_seed(314159)
    with _cpu_threads(), torch.inference_mode():
        # Modest, nonzero SwiGLU outputs avoid overflow and saturated sigmoid;
        # the comparison targets quantized execution, not a model-quality claim.
        gate = torch.randn(EXPERTS, 2 * INTERMEDIATE, WIDTH, generator=generator) * weight_scale
        down = torch.randn(EXPERTS, WIDTH, INTERMEDIATE, generator=generator) * weight_scale
        return SimpleNamespace(fmt=fmt, gate=_bank(gate, fmt), down=_bank(down, fmt))


@pytest.fixture(scope="module")
def case_factory():
    cases = {}

    def get(fmt):
        if fmt not in cases:
            cases[fmt] = _make_case(fmt)
        return cases[fmt]

    return get


@pytest.fixture(scope="module", params=FORMATS)
def bank_case(request, case_factory):
    return case_factory(request.param)


def _linear(case, state, expert, x):
    if case.fmt == "int8_tensorwise":
        return eager_int8.int8_linear(x.float(), state["weight"][expert],
            state["weight_scale"][expert], out_dtype=torch.float32,
            convrot=True, convrot_groupsize=WIDTH)
    return eager_codes.w4a8_int8_linear(x.float(), state["weight"][expert],
        state["weight_s_rel"][expert].view(torch.float8_e4m3fn),
        state["weight_s_channel"][expert], codebook=state.get("weight_codebook"),
        group_size=16, convrot_groupsize=WIDTH, out_dtype=torch.float32)


def _dense_weight(case, state, expert):
    if case.fmt == "int8_tensorwise":
        return eager_int8.dequantize_int8_convrot_weight(
            state["weight"][expert], state["weight_scale"][expert], WIDTH)
    return eager_codes.dequantize_w4a8_int8_weight(state["weight"][expert],
        state["weight_s_rel"][expert].view(torch.float8_e4m3fn),
        state["weight_s_channel"][expert], codebook=state.get("weight_codebook"),
        group_size=16, convrot_groupsize=WIDTH, output_dtype=torch.float32)


def _reference(case, x, ids, weights):
    x, ids, weights = x.detach().float().cpu(), ids.cpu(), weights.detach().float().cpu()
    result = []
    for row in range(len(x)):
        pieces = []
        for expert, weight in zip(ids[row].tolist(), weights[row]):
            gate, up = _linear(case, case.gate, expert, x[row:row + 1]).chunk(2, -1)
            pieces.append(_linear(case, case.down, expert, F.silu(gate) * up) * weight)
        result.append(torch.stack(pieces).sum(0))
    return torch.cat(result)


def _modules(case):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    import comfy.ops
    from drama3_t8.native.loader import load_checkpoint_storage

    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    modules = SimpleNamespace(
        gate_up_proj=ops.MoEExperts(EXPERTS, WIDTH, 2 * INTERMEDIATE, bias=False, device="meta"),
        down_proj=ops.MoEExperts(EXPERTS, INTERMEDIATE, WIDTH, bias=False, device="meta"))
    for module, state in ((modules.gate_up_proj, case.gate), (modules.down_proj, case.down)):
        # The real loader may pop/flatten dictionary entries; retain the oracle.
        load_checkpoint_storage(module, dict(state))
    return modules


def _cuda_store(case, layers=1):
    modules = [_modules(case) for _ in range(layers)]
    from drama3_t8.native.offload_accelerated import ExpertStore
    from drama3_t8.native import kernels

    device = torch.device("cuda:0")
    if not kernels.available(device):
        pytest.skip("NVRTC expert kernels are unavailable")
    store = ExpertStore(1, pin_memory=True)
    routed = [store.add(module) for module in modules]
    return device, store, routed


def test_official_packed_fixtures_and_a8_oracle_match_reconstructed_f_linear(bank_case):
    case = bank_case
    generator = torch.Generator().manual_seed(271828)
    x = torch.randn(3, WIDTH, generator=generator)
    metrics = {}
    with _cpu_threads(), torch.inference_mode():
        for name, state in (("gate", case.gate), ("down", case.down)):
            bits = 8 if case.fmt == "int8_tensorwise" else 4 if case.fmt == "asym_w4a8_int8" else 6
            assert state["weight"].shape[-1] == WIDTH * bits // 8
            dense = _dense_weight(case, state, 0)
            rotated = eager_int8.rotate_int8_convrot_weight(x, WIDTH)
            q, scale = eager_int8.quantize_int8_rowwise(rotated)
            a8_original_basis = eager_int8.rotate_int8_convrot_weight(q.float() * scale, WIDTH)
            oracle = _linear(case, state, 0, x)
            reconstructed = F.linear(a8_original_basis, dense)
            torch.testing.assert_close(oracle, reconstructed, atol=3e-6, rtol=3e-5)
            assert torch.isfinite(dense).all()
            metrics[name] = dict(oracle_vs_f_linear=_errors(oracle, reconstructed),
                a8_vs_unquantized_activation=_errors(oracle, F.linear(x, dense)))
    _record("cpu_layout_" + case.fmt, **metrics)


@CUDA
def test_quantized_cuda_prefill_and_decode_hit_miss_match_official_eager(bank_case, monkeypatch):
    # Import after standalone bootstrap; preparing/running the CPU oracle alone
    # does not import Comfy's GPU model manager.
    device, store, (routed,) = _cuda_store(bank_case)
    generator = torch.Generator().manual_seed(161803)
    samples = []
    try:
        with _cpu_threads(), torch.inference_mode():
            # Isolate grouped GPU kernels.  INT8 CPU cooperation is covered by
            # the separate real multi-layer handoff below, with its BF16 math.
            monkeypatch.setattr(store, "split", lambda experts, counts, nbytes: ([], 0.0, 1.0))
            with store.session(device, [], cpu_decode=False):
                assert store.sync_free and store.cache.count == 8
                for _ in range(2):
                    x = torch.randn(5, WIDTH, generator=generator)
                    ids = torch.tensor([[(row * 3 + j) % EXPERTS for j in range(TOP_K)] for row in range(5)])
                    weights = torch.rand(5, TOP_K, generator=generator).softmax(-1)
                    expected = _reference(bank_case, x, ids, weights)
                    actual = routed(x.to(device), ids.to(device), weights.to(device)).cpu()
                    metrics = _errors(actual, expected)
                    samples.append(dict(stage="prefill", **metrics))
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
                assert int((store.cache.table >= 0).sum()) == 8
            with store.session(device, [], cpu_decode=False):
                snapshots = []
                for start in (0, 0, 8, 0):
                    x = torch.randn(1, WIDTH, generator=generator)
                    ids = torch.arange(start, start + TOP_K)[None]
                    weights = torch.rand(1, TOP_K, generator=generator).softmax(-1)
                    expected = _reference(bank_case, x, ids, weights)
                    actual = routed(x.to(device), ids.to(device), weights.to(device)).cpu()
                    samples.append(dict(stage="decode", start=start, **_errors(actual, expected)))
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
                    snapshots.append(store.cache.reads.cpu().tolist())
                assert snapshots == [[0, 8, 0], [8, 8, 0], [8, 16, 0], [8, 24, 0]]
            assert store.backend_calls["cuda"] == 6 and store.last_backend == "cuda"
            _record("cuda_eager_" + bank_case.fmt, samples=samples, decode_reads=snapshots)
    finally:
        store.close()


@CUDA
def test_quantized_cuda_graph_replay_updates_input_and_replaces_cached_experts(bank_case):
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    from drama3_t8.native.graphs import StaticReplay

    device, store, (routed,) = _cuda_store(bank_case)
    runtime = StaticReplay(device)
    generator = torch.Generator().manual_seed(141421)
    samples = []
    try:
        with _cpu_threads(), torch.inference_mode(), store.session(device, [], cpu_decode=False):
            assert store.sync_free
            for start in (0, 0, 0, 8, 8, 0, 8):
                x = torch.randn(1, WIDTH, generator=generator)
                ids = torch.arange(start, start + TOP_K)[None]
                weights = torch.rand(1, TOP_K, generator=generator).softmax(-1)
                expected = _reference(bank_case, x, ids, weights)
                actual = runtime(routed, x.to(device), ids.to(device), weights.to(device)).clone().cpu()
                samples.append(dict(start=start, **_errors(actual, expected)))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            assert runtime.captures == 1 and runtime.replays == 4 and runtime.fallback_reason is None
            reads = store.cache.reads.cpu().tolist()
            assert reads[0] > 0 and reads[1] > 0 and reads[2] == 0
            _record("cuda_graph_" + bank_case.fmt, samples=samples, decode_reads=reads,
                    captures=runtime.captures, replays=runtime.replays)
    finally:
        store.close()


@CUDA
def test_four_layer_int8_cpu_handoff_graphs_preserve_residual_route_chain():
    # A larger branch than the operator fixture makes missed/corrupt handoffs
    # visible above BF16 input rounding, while retaining ordinary SwiGLU ranges.
    case = _make_case("int8_tensorwise", weight_scale=.04)
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    from drama3_t8.native.graphs import StaticReplay

    device, store, routed = _cuda_store(case, layers=4)
    generator = torch.Generator().manual_seed(173205)
    inputs = []
    for step in range(7):
        x = torch.randn(1, WIDTH, generator=generator).to(torch.bfloat16).to(device)
        ids = [torch.tensor([[(step * 7 + layer * 5 + j) % EXPERTS for j in range(TOP_K)]], device=device)
               for layer in range(4)]
        weights = torch.rand(1, TOP_K, generator=generator).softmax(-1).to(torch.bfloat16).to(device)
        inputs.append((x, ids, weights))

    def run(cpu_decode):
        runners = [StaticReplay(device) for _ in routed]
        results, served, requested = [], 0, 0
        with store.session(device, [], cpu_decode=cpu_decode):
            assert store.sync_free and store.cache.count == 8
            for x, ids, weights in inputs:
                # One narrow cache shared by all four layers deliberately causes
                # slot competition.  Per-layer post -> host service follows the
                # actual model's after_layer callback, between_steps once/frame.
                h = x.clone()
                for layer, (operation, runner) in enumerate(zip(routed, runners)):
                    def residual(source, selected, mix, operation=operation):
                        return source + operation(source, selected, mix)
                    h = runner(residual, h, ids[layer], weights).clone()
                    if cpu_decode:
                        store.after_layer(layer)
                        cpu_count = int((store.cpu_side.plan[layer, :, 1] == 2).sum())
                        requested += cpu_count
                        if cpu_count and not int(store.cache.took_over.item()):
                            served += cpu_count
                results.append(h.float().cpu())
                if cpu_decode:
                    store.between_steps()
            snapshots = store.cache.reads.cpu().tolist()
        assert all(runner.captures == 1 and runner.replays == 4 and runner.fallback_reason is None
                   for runner in runners)
        return torch.cat(results), served, requested, snapshots

    try:
        with _cpu_threads(), torch.inference_mode():
            reference, _, _, gpu_reads = run(False)
            actual, served, requested, hybrid_reads = run(True)
            metrics = _errors(actual, reference)
            initial = torch.cat([x.float().cpu() for x, _, _ in inputs])
            update = reference - initial
            update_rms = float(update.square().mean().sqrt())
            update_relative_error = metrics["rms"] / max(update_rms, 1e-30)
            _record("int8_four_layer_handoff", **metrics, cpu_experts_requested=requested,
                    cpu_experts_served=served, cpu_experts_run=store.cpu_experts_run,
                    cpu_decode_calls=store.cpu_decode_calls, gpu_reads=gpu_reads, hybrid_reads=hybrid_reads,
                    layer_graph_captures=4, layer_graph_replays=16,
                    net_update_rms=update_rms, error_relative_to_net_update=update_relative_error)
            assert store.cpu_decode_calls == 28 and store.cpu_experts_run > 0
            assert requested > 0 and served > 0, "Host work must actually reach a GPU consumer before timeout"
            # FP32 A8 GPU and BF16 INT8-packed host intermediates are deliberately
            # distinct.  Bound the final residual-chain error in BF16 output
            # units (two eps), and RMS in a single eps; this is not bit equality.
            eps = torch.finfo(torch.bfloat16).eps
            # A no-op branch must fail even if the original input dominates the
            # output norm.  Also constrain error relative to the actual update.
            assert update_rms > .05 * metrics["reference_rms"], metrics
            assert update_relative_error <= .05, metrics
            assert metrics["max_abs"] <= 2 * eps * max(float(reference.abs().max()), 1e-30), metrics
            assert metrics["relative_rms"] <= eps, metrics
    finally:
        store.close()
