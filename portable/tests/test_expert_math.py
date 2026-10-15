from drama3_t8.bootstrap import standalone_engine
standalone_engine()

import torch
from drama3_t8.native.offload import experts_decode, HostBank


def test_decode_runs_only_each_experts_own_down_projection():
    torch.manual_seed(8)
    k, hidden, middle = 8, 32, 16
    x = torch.randn(1, hidden)
    gate_up = torch.randn(k * 2 * middle, hidden)
    down = torch.randn(k * hidden, middle)
    weights = torch.randn(k).softmax(0)
    expected = []
    for i in range(k):
        gate, up = (x @ gate_up[i*2*middle:(i+1)*2*middle].T).chunk(2,-1)
        value = (torch.nn.functional.silu(gate)*up) @ down[i*hidden:(i+1)*hidden].T
        expected.append(value * weights[i])
    assert torch.allclose(experts_decode(x,gate_up,down,weights),sum(expected),atol=2e-5,rtol=2e-5)


def test_sparse_prefill_gather_preserves_expert_order_for_each_field():
    bank = HostBank(4,{"weight":torch.arange(48).reshape(12,4),"scale":torch.arange(8).reshape(4,2)})
    fields=bank.selected_to_device([3,1],"cpu")
    assert torch.equal(fields["weight"],torch.cat((bank.fields["weight"][9:12],bank.fields["weight"][3:6])))
    assert torch.equal(fields["scale"],bank.fields["scale"][[3,1]])
