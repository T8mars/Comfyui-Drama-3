"""Real CPU Qwen forwards and tokenizer contracts across supported Transformers versions."""
import json
from pathlib import Path

import pytest
import torch
from transformers import AddedToken, Qwen2Tokenizer

from drama3_t8.vendor_qwen.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
from drama3_t8.vendor_qwen.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerTextModel


@pytest.fixture(params=('eager', 'sdpa'))
def text_model(request):
    torch.manual_seed(712)
    config = Qwen3ASRTextConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=32, pad_token_id=0, rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 0]},
        attn_implementation=request.param)
    with torch.device('cpu'):
        model = Qwen3ASRThinkerTextModel(config).eval()
    assert model.config._attn_implementation == request.param
    return model


def test_real_text_prefill_and_cached_token_match_full_sequence(text_model):
    with torch.inference_mode():
        prefix = text_model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
        cache = prefix.past_key_values
        assert cache.get_seq_length() == 3
        token = text_model(input_ids=torch.tensor([[4]]), past_key_values=cache, use_cache=True)
        full = text_model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
    assert prefix.last_hidden_state.shape == (1, 3, 8)
    assert token.last_hidden_state.shape == (1, 1, 8)
    assert token.past_key_values is cache and cache.get_seq_length() == 4
    assert torch.isfinite(token.last_hidden_state).all()
    torch.testing.assert_close(token.last_hidden_state, full.last_hidden_state[:, -1:], rtol=1e-5, atol=1e-6)


def test_real_text_causal_mask_prevents_future_tokens_changing_prefix(text_model):
    with torch.inference_mode():
        first = text_model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
        changed = text_model(input_ids=torch.tensor([[1, 2, 8, 9]]), use_cache=False)
    torch.testing.assert_close(first.last_hidden_state[:, :2], changed.last_hidden_state[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(first.last_hidden_state[:, 2:], changed.last_hidden_state[:, 2:])


def test_real_text_padding_mask_preserves_unpadded_outputs(text_model):
    with torch.inference_mode():
        clean = text_model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        padded = text_model(input_ids=torch.tensor([[0, 0, 1, 2, 3]]),
            attention_mask=torch.tensor([[0, 0, 1, 1, 1]]), position_ids=torch.tensor([[0, 0, 0, 1, 2]]), use_cache=False)
    torch.testing.assert_close(padded.last_hidden_state[:, -3:], clean.last_hidden_state, rtol=1e-5, atol=1e-6)


def test_native_qwen_asr_tokenizer_keeps_audio_protocol_ids_and_multilingual_text():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'vendor/ComfyUI/comfy/text_encoders/qwen25_tokenizer'
    tokenizer = Qwen2Tokenizer(str(folder / 'vocab.json'), str(folder / 'merges.txt'))
    added = json.loads((root / 'drama3_t8/qwen3_asr/added_tokens.json').read_text(encoding='utf-8'))
    tokenizer.add_tokens([AddedToken(text, special=special, normalized=False) for text, special in added['tokens']])
    assert tokenizer.convert_tokens_to_ids('<|audio_start|>') == 151669
    assert tokenizer.convert_tokens_to_ids('<|audio_end|>') == 151670
    assert tokenizer.convert_tokens_to_ids('<|audio_pad|>') == 151676
    assert tokenizer.convert_tokens_to_ids(added['tokens'][-1][0]) == added['first_id'] + len(added['tokens']) - 1
    for text in ('你好，这是本地转写。', 'The train leaves tomorrow morning.'):
        assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), clean_up_tokenization_spaces=False) == text
