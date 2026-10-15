from drama3_t8.bootstrap import standalone_engine
standalone_engine()

import torch
import comfy.ops
from comfy.text_encoders.qwen35 import LinearKV, Qwen35Config
from drama3_t8.native.delta import DramaDeltaNet


def test_long_prefill_continuation_keeps_conv_and_recurrent_state():
    config = Qwen35Config(hidden_size=32, linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=8, linear_value_head_dim=8, conv_kernel_size=4)
    torch.manual_seed(17)
    layer = DramaDeltaNet(config, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
    with torch.inference_mode():
        for param in layer.parameters():
            param.normal_(0,.1)
        layer.norm.weight.fill_(1)
        layer.A_log.fill_(-2)
        layer.dt_bias.fill_(0)
        x = torch.randn(1,17,32)

        def cache():
            return LinearKV(torch.zeros(1,64,3), torch.zeros(1,4,8,8), 0, None, None)

        full_cache, split_cache = cache(), cache()
        full,_ = layer(x, full_cache)
        first,_ = layer(x[:,:7], split_cache)
        split_cache.advance(7)
        last,_ = layer(x[:,7:], split_cache)
        assert torch.allclose(full,torch.cat((first,last),1),atol=2e-5,rtol=2e-4)
        assert torch.allclose(full_cache.conv_state,split_cache.conv_state,atol=1e-6)
        assert torch.allclose(full_cache.recurrent_state,split_cache.recurrent_state,atol=1e-5)


def test_missing_optional_fla_uses_the_torch_operation(monkeypatch):
    import builtins
    config=Qwen35Config(hidden_size=32,linear_num_key_heads=2,linear_num_value_heads=4,
                       linear_key_head_dim=8,linear_value_head_dim=8,conv_kernel_size=4)
    layer=DramaDeltaNet(config,device="cpu",dtype=torch.float32,ops=comfy.ops.manual_cast)
    with torch.inference_mode():
        for param in layer.parameters():param.normal_(0,.1)
        x=torch.randn(1,7,32)
        expected,_=layer(x)
        original=builtins.__import__
        def importing(name,*args,**kwargs):
            if name.startswith('fla'):raise ImportError('optional FLA is unavailable')
            return original(name,*args,**kwargs)
        monkeypatch.setattr(builtins,'__import__',importing)
        layer.backend='fla'
        actual,_=layer(x)
    assert torch.equal(actual,expected)
    assert layer.backend=='torch'
    assert 'optional FLA' in layer.fallback_reason
