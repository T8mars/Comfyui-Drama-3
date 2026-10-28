"""Optional real-checkpoint layer/operator checks; never loads the full model.

Set DRAMA_REAL_CHECKPOINT_DIR to local converted files. Only layer 0, experts
0..15 are mapped into a small resident bank. This is not full-model synthesis,
speaker quality, canvas validation, or proof that the complete model fits VRAM.
"""
import hashlib
import json
import os
import struct
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.ops
from drama3_t8.native import kernels
from drama3_t8.native.acceleration import physical_bytes
from drama3_t8.native.loader import load_checkpoint_storage
from drama3_t8.native.model import Experts
from drama3_t8.native.graphs import StaticReplay
from test_resident_experts import holder
from test_resident_cuda import managed, cleanup
from test_cuda_quantized_experts import _reference, _cpu_threads, _record, _errors


CHECKPOINTS = Path(os.environ.get('DRAMA_REAL_CHECKPOINT_DIR', 'E:/codex-record/drama3-upgrade/models'))
FORMATS = {'int8_convrot': 'int8_tensorwise', 'w4a8': 'asym_w4a8_int8', 'w6a8': 'w6a8_int8'}
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Real-weight resident operator check requires CUDA')


def read_selected(path, fmt, experts=16):
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        raw_header = stream.read(header_size)
    header = json.loads(raw_header)
    states, selected = {}, {}
    dtypes = {'I8': torch.int8, 'U8': torch.uint8, 'F32': torch.float32}
    selected_bytes = 0
    # Bound the physical bytes explicitly. A >24GB Windows mmap view is not
    # needed to inspect a few experts and proved unsafe in the first probe.
    with path.open('rb') as stream:
        for bank in ('gate_up_proj', 'down_proj'):
            prefix = f'model.layers.0.mlp.experts.{bank}.'
            state = {}
            for key in header:
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix):]
                shape = header[key]['shape']
                dtype = dtypes[header[key]['dtype']]
                unit = torch.empty((), dtype=dtype).element_size()
                shape_selected = [experts, *shape[1:]] if name != 'comfy_quant' and len(shape) >= 2 and shape[0] == 256 else shape
                size = math.prod(shape_selected) * unit
                first, last = header[key]['data_offsets']
                assert last - first == math.prod(shape) * unit and size <= last - first
                selected_bytes += size
                assert selected_bytes < 2 * 1024**3
                raw = bytearray(size)
                stream.seek(8 + header_size + first)
                if stream.readinto(raw) != size:
                    raise ValueError('Selected checkpoint field is truncated')
                tensor = torch.frombuffer(raw, dtype=dtype).reshape(shape_selected)
                selected[key] = dict(source_shape=shape, selected_shape=list(tensor.shape),
                    source_file_byte_offset=8 + header_size + first, read_bytes=size,
                    sha256=hashlib.sha256(raw).hexdigest())
                state[name] = tensor
            conf = json.loads(bytes(state['comfy_quant'].tolist()))
            assert conf['format'] == FORMATS[fmt] and conf['num_experts'] == 256
            conf['num_experts'] = experts
            state['comfy_quant'] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
            states[bank] = state
    case = SimpleNamespace(fmt=FORMATS[fmt], gate=states['gate_up_proj'], down=states['down_proj'])
    return case, dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
        header_sha256=hashlib.sha256(raw_header).hexdigest(), layer=0, original_experts=256,
        selected_experts=list(range(experts)), selected_fields=selected, selected_read_bytes=selected_bytes,
        source_reader='Bounded seek/readinto bytearrays; no checkpoint-wide mmap or full bank read',
        full_checkpoint_weight_bytes=sum(value['data_offsets'][1] - value['data_offsets'][0]
            for key, value in header.items() if key != '__metadata__' and not key.endswith('.comfy_quant')))


@pytest.mark.parametrize('fmt', list(FORMATS))
def test_real_packed_2048_hidden_resident_operator_and_graph_against_official_a8(fmt):
    path = CHECKPOINTS / f'drama3-t8-{fmt}.safetensors'
    if not path.is_file():
        pytest.skip('Optional local converted checkpoint unavailable; no download is performed')
    device = torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('NVRTC expert kernels unavailable')
    case, source = read_selected(path, fmt)
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    expert = Experts(16, 2048, 512, device='meta', dtype=torch.bfloat16, ops=ops)
    load_checkpoint_storage(expert.gate_up_proj, dict(case.gate))
    load_checkpoint_storage(expert.down_proj, dict(case.down))
    model = holder([expert])
    budget = physical_bytes([model])
    assert budget < 2 * 1024**3
    patcher, controller = managed(model)
    generator = torch.Generator().manual_seed(605)
    samples = []
    def counted():
        controller.backend_calls['cuda'] += 1
    runtime = StaticReplay(device, on_replay=counted)
    try:
        with _cpu_threads(), torch.inference_mode(), controller.session(device, [patcher]):
            assert controller.sync_free
            x = torch.randn(2, 2048, generator=generator)
            ids = torch.tensor([list(range(8)), list(range(8, 16))])
            weights = torch.rand(2, 8, generator=generator).softmax(-1)
            expected = _reference(case, x, ids, weights)
            actual = expert(x.to(device), ids.to(device), weights.to(device)).cpu()
            samples.append(dict(stage='prefill', **_errors(actual, expected)))
            _record('resident_real_weight_' + fmt, status='running', source=source, resident_selected_bytes=budget, samples=samples)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            for start in (0, 0, 0, 8, 8, 0, 8):
                x = torch.randn(1, 2048, generator=generator)
                ids = torch.arange(start, start + 8)[None]
                weights = torch.rand(1, 8, generator=generator).softmax(-1)
                expected = _reference(case, x, ids, weights)
                actual = runtime(expert, x.to(device), ids.to(device), weights.to(device)).clone().cpu()
                samples.append(dict(stage='decode', start=start, **_errors(actual, expected)))
                _record('resident_real_weight_' + fmt, status='running', source=source, resident_selected_bytes=budget, samples=samples)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            assert runtime.captures == 1 and runtime.replays == 4 and not runtime.fallback_reason
            assert controller.backend_calls == {'torch': 0, 'cuda': 8}
            assert controller.cpu_experts_run == 0
        _record('resident_real_weight_' + fmt, status='passed', scope='real-weight layer/operator only; not full-model synthesis',
                source=source, resident_selected_bytes=budget, samples=samples, stats=controller.stats(),
                captures=runtime.captures, replays=runtime.replays,
                tolerance=dict(atol=2e-6, rtol=2e-4, oracle='official CPU eager A8 FP32; unchanged from synthetic checks'))
    finally:
        cleanup(patcher, controller)
