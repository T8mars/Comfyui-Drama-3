"""Real CUDA norm offload, resident, and StaticReplay contracts.

The independent Fish Audio formulas and their provenance live in the CPU
contract test. CPU suites must exclude this CUDA-only file before collection.
Only tiny norms are allocated; no checkpoint or complete model is loaded.
"""
import pytest
import torch

from test_native_norm_precision import _official_gated, _official_zero_centered
from drama3_t8.native.graphs import StaticReplay
from drama3_t8.native.norms import GatedRMSNorm, ZeroCenteredRMSNorm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real norm CUDA contract needs a GPU')


def _norm(kind, device, dtype):
    cls = ZeroCenteredRMSNorm if kind == 'zero_centered' else GatedRMSNorm
    norm = cls(32, eps=1e-6, device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.fill_(1 / 256 if kind == 'zero_centered' else 1.5)
    return norm


def _inputs(kind, device, dtype, step=0):
    if kind == 'zero_centered':
        x = torch.tensor([1.5, .5, -1.5, -.5], device=device, dtype=dtype).repeat(8)
        return ((x + step / 16).reshape(1, 1, 32).repeat(2, 3, 1),)
    x = torch.ones(2, 32, device=device, dtype=dtype)
    gate = torch.full_like(x, .5 + step / 8)
    return x, gate


def _oracle(kind, norm, inputs):
    # Transfer a detached reference copy only. The production parameter stays
    # on its original device; this also exercises the actual host-weight path.
    weight = norm.weight.detach().to(device=inputs[0].device)
    if kind == 'zero_centered':
        return _official_zero_centered(inputs[0], weight, norm.eps)
    return _official_gated(inputs[0], inputs[1], weight, norm.eps)


def _snapshot(norm):
    assert set(dict(norm.named_parameters())) == {'weight'}
    assert set(norm.state_dict()) == {'weight'}
    return dict(parameter=norm.weight, pointer=norm.weight.data_ptr(),
                device=norm.weight.device, dtype=norm.weight.dtype,
                shape=norm.weight.shape, value=norm.weight.detach().cpu().clone())


def _unchanged(norm, before):
    assert norm.weight is before['parameter']
    assert norm.weight.data_ptr() == before['pointer']
    assert norm.weight.device == before['device']
    assert norm.weight.dtype == before['dtype']
    assert norm.weight.shape == before['shape']
    assert set(norm.state_dict()) == {'weight'}
    torch.testing.assert_close(norm.weight.detach().cpu(), before['value'], rtol=0, atol=0)


def _same(actual, expected, inputs):
    assert actual.device == inputs[0].device
    assert actual.dtype == inputs[0].dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['zero_centered', 'gated'])
def test_cpu_bf16_norm_weight_with_real_cuda_activation_preserves_parameter(kind):
    device = torch.device('cuda:0')
    norm = _norm(kind, torch.device('cpu'), torch.bfloat16)
    before = _snapshot(norm)
    with torch.inference_mode(), torch.cuda.device(device):
        inputs = _inputs(kind, device, torch.bfloat16)
        expected = _oracle(kind, norm, inputs)
        actual = norm(*inputs)
        torch.cuda.synchronize(device)
        _same(actual, expected, inputs)
    _unchanged(norm, before)


@pytest.mark.parametrize('kind', ['zero_centered', 'gated'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_resident_cuda_norm_matches_official_formula_and_dtype(kind, dtype):
    device = torch.device('cuda:0')
    with torch.inference_mode(), torch.cuda.device(device):
        norm = _norm(kind, device, dtype)
        before = _snapshot(norm)
        inputs = _inputs(kind, device, dtype)
        expected = _oracle(kind, norm, inputs)
        actual = norm(*inputs)
        torch.cuda.synchronize(device)
        _same(actual, expected, inputs)
        _unchanged(norm, before)


@pytest.mark.parametrize('kind', ['zero_centered', 'gated'])
def test_real_norm_cuda_graph_replays_match_eager_and_official_formula(kind):
    device = torch.device('cuda:0')
    with torch.inference_mode(), torch.cuda.device(device):
        # Graphs use resident weights, as the native generation path requires.
        # The separate test above covers CPU-owned weights without pretending
        # that pageable host-to-device transfers are a graph requirement.
        eager = _norm(kind, device, torch.bfloat16)
        captured = _norm(kind, device, torch.bfloat16)
        eager_before, captured_before = _snapshot(eager), _snapshot(captured)
        runner = StaticReplay(device)
        results = []
        for step in range(7):
            inputs = _inputs(kind, device, torch.bfloat16, step)
            expected = _oracle(kind, eager, inputs)
            eager_output = eager(*inputs)
            graph_output = runner(captured, *inputs).clone()
            torch.cuda.synchronize(device)
            _same(eager_output, expected, inputs)
            _same(graph_output, expected, inputs)
            torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
            results.append(graph_output)
        assert runner.graph is not None
        assert runner.captures == 1
        assert runner.replays > 0
        assert runner.fallback_reason is None
        assert any(not torch.equal(results[0], output) for output in results[1:])
        _unchanged(eager, eager_before)
        _unchanged(captured, captured_before)
        torch.testing.assert_close(eager.weight, captured.weight, rtol=0, atol=0)
