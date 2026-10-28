"""Exact sparse rotation and independent nearest-neighbor conversion contracts.

No model weights, transcripts or expected speech are used. FP64 accumulation
improves this cancellation case; it is not a full-exponent exact-sum promise.
"""
from array import array
from fractions import Fraction
import math
import struct

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
from drama3_t8.native import kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA rotation precision contract')


def _ready():
    device = torch.device('cuda:0')
    if not kernels.available(device):
        pytest.skip('NVRTC unavailable; arithmetic has not been verified')
    return device


def _nearest_bits(value):
    sign = 0x8000 if value < 0 else 0
    value = abs(value)
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, -126) - 7)
    ratio = value / quantum
    integer, remainder = divmod(ratio.numerator, ratio.denominator)
    if 2 * remainder > ratio.denominator or (2 * remainder == ratio.denominator and integer % 2):
        integer += 1
    if exponent < -126:
        return sign | integer
    if integer == 256:
        exponent += 1
        integer = 128
    return sign | ((exponent + 127) << 7) | (integer - 128)


def test_sparse_BF16_Hadamard_cancellation_matches_exact_rational_oracle():
    device = _ready()
    nonzero = {0: Fraction(9, 16), 1: -Fraction(1, 2**25),
               2: -Fraction(3, 8), 3: Fraction(9, 2048), 256: Fraction(213, 32)}
    # Explicit H4 tensor-product coefficients, independent of the CUDA butterfly.
    h4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
    rotated = []
    for column in range(512):
        total = Fraction(0)
        for index, value in nonzero.items():
            if index // 256 != column // 256:
                continue
            coefficient = math.prod(h4[(index >> (2*d)) & 3][(column >> (2*d)) & 3] for d in range(4))
            total += value * coefficient
        rotated.append(_nearest_bits(total / 16))
    exact = torch.tensor([[b if b < 32768 else b-65536 for b in rotated]], dtype=torch.int16).view(torch.bfloat16).to(device)
    assert rotated[0] == 0x3c3b
    x = torch.zeros(1, 512, device=device, dtype=torch.float32)
    for index, value in nonzero.items():
        x[0, index] = float(value)
    one = torch.ones(1, device=device, dtype=torch.int32)
    with torch.inference_mode():
        expected_q, expected_scale = quantize_int8_rowwise(exact)
        q, scale = kernels.rotate_quantize(x, one, bf16=True)
    assert expected_q[0, 0].item() == 3
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_scale.reshape(-1).view(torch.int32))


def test_direct_F64_BF16_conversion_all_nearest_neighbor_boundaries():
    device = _ready()
    values, wanted = array('d'), array('i')
    def scalar(bits):
        return struct.unpack('<f', struct.pack('<I', bits << 16))[0]
    def add(value, bits):
        values.extend((value, -value))
        wanted.extend((bits, bits | 0x8000))
    for lower in range(0x7f80):
        lo = scalar(lower)
        hi = scalar(lower+1) if lower < 0x7f7f else 2.0**128
        midpoint = (lo+hi)/2
        assert Fraction.from_float(midpoint)*2 == Fraction.from_float(lo)+Fraction.from_float(hi)
        add(lo, lower)
        add(math.nextafter(midpoint, -math.inf), lower)
        add(midpoint, lower if lower % 2 == 0 else lower+1)
        add(math.nextafter(midpoint, math.inf), lower+1)
    add(math.inf, 0x7f80)
    add(math.ulp(0.0), 0)
    add(float.fromhex('0x1.fffffffffffffp+1023'), 0x7f80)
    probe = r'''
extern "C" __global__ void drama_test_direct_bf16(const double* x, int* y, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) y[i]=(int)(__float_as_uint(qa_double_bf16(x[i]))>>16);
}
'''
    kernel = torch.cuda._compile_kernel(kernels.SOURCE+probe, 'drama_test_direct_bf16')
    with torch.inference_mode():
        x = torch.frombuffer(values, dtype=torch.float64).clone().to(device)
        expected = torch.frombuffer(wanted, dtype=torch.int32).clone().to(device)
        actual = torch.empty_like(expected)
        kernel(grid=((len(values)+255)//256, 1, 1), block=(256, 1, 1),
               args=[x, actual, len(values)], stream=torch.cuda.current_stream(device))
        assert torch.equal(actual, expected)
        payloads = (0x7ff0000000000001, 0x7ff123456789abcd, 0x7fffffffffffffff,
                    0xfff0000000000001, 0xfff123456789abcd, 0xffffffffffffffff)
        nx = torch.frombuffer(bytearray(struct.pack('<6Q', *payloads)), dtype=torch.float64).clone().to(device)
        ny = torch.empty(6, dtype=torch.int32, device=device)
        kernel(grid=(1, 1, 1), block=(256, 1, 1), args=[nx, ny, 6], stream=torch.cuda.current_stream(device))
        for source, found in zip(payloads, ny.cpu().tolist()):
            assert found & 0x7f80 == 0x7f80
            assert found & 0x40
            assert found & 0x8000 == (source >> 48) & 0x8000
