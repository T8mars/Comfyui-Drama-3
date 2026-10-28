"""Twenty independent, bounded CPU contracts; no checkpoint inference."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drama3_t8.engine import Sampling
from drama3_t8.audio import audio_waveform, asr_waveforms
from drama3_t8.native.prompt import check_text
from drama3_t8.native import codec as codec_module
from drama3_t8.state import Reference, validate_reference, validate_codes
from drama3_t8.qwen3_asr import asr as qwen
from drama3_t8.r2t2.audio import limited_quiet_gain, to_mono_16k
from drama3_t8.r2t2.native import build_prompt, parse_raw
from drama3_t8.r2t2.segmented import OfflineSegment, exact_zero_boundaries
from drama3_t8.r2t2.hotwords import normalize_hotwords, read_hotword_file
from drama3_t8.model_paths import resolve_models, DEFAULT_PATHS


def test_round01_sampling_uint64_upper_edge_and_bool_rejection():
    Sampling(seed=2**64-1, top_k=0).validate()
    for seed in (True, -1, 2**64):
        with pytest.raises(ValueError):
            Sampling(seed=seed).validate()


def test_round02_nonfinite_sampling_does_not_enter_generation():
    for values in ({'temperature':float('nan')},{'temperature':float('inf')},{'top_p':float('nan')},{'top_p':float('inf')}):
        with pytest.raises(ValueError):
            Sampling(**values).validate()


def test_round03_opening_retry_is_bounded_to_one():
    Sampling(opening_retries=1).validate()
    for value in (True, 2, -1):
        with pytest.raises(ValueError):
            Sampling(opening_retries=value).validate()


def test_round04_audio_rate_and_multiclip_reference_rejected():
    for rate in (0, True, 44100.5, float('inf')):
        with pytest.raises(ValueError):
            audio_waveform(torch.ones(1,10),rate)
    with pytest.raises(ValueError,match='batch'):
        audio_waveform(torch.ones(2,1,10),44100)


def test_round05_asr_batch_checks_late_nonfinite_clip():
    wave=torch.ones(2,1,10)
    wave[1,0,-1]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        asr_waveforms(wave,16000)


def test_round06_prompt_reserved_tokens_rejected_before_tokenizer():
    for token in ('<|im_end|>','<|semantic:0|>','<|audio_separator|>'):
        with pytest.raises(ValueError):
            check_text('hello '+token,'text')


def test_round07_speaker_tags_are_canonical_decimal_and_bounded():
    check_text('<|speaker:15|> 最后的角色','text')
    for tag in ('<|speaker:16|>','<|speaker:01|>','<|speaker:-1|>'):
        with pytest.raises(ValueError):
            check_text(tag+' 你好','text')


def test_round08_reference_codes_require_nonempty_integer_vocabulary():
    validate_reference(Reference('voice',torch.full((10,1),4095,dtype=torch.long)))
    for codes in (torch.ones(10,1),torch.ones(10,1,dtype=torch.bool),torch.full((10,1),4096,dtype=torch.long)):
        with pytest.raises(ValueError):
            validate_codes(codes)
    with pytest.raises(ValueError):
        validate_reference(Reference('voice',torch.empty((10,0),dtype=torch.long)))


def test_round09_qwen_short_audio_padding_keeps_all_accepted_samples(monkeypatch):
    monkeypatch.setattr(qwen,'SAMPLE_RATE',100)
    monkeypatch.setattr(qwen,'MAX_SECONDS',10)
    raw=torch.tensor([.2,.3,.4])
    pieces=qwen.split_long(raw)
    assert len(pieces)==1 and len(pieces[0])==50
    assert torch.equal(pieces[0][:3],raw) and torch.count_nonzero(pieces[0][3:])==0


def test_round10_qwen_quiet_point_follows_target_within_upstream_tolerance(monkeypatch):
    monkeypatch.setattr(qwen,'SAMPLE_RATE',100)
    monkeypatch.setattr(qwen,'MAX_SECONDS',10)
    raw=torch.ones(1500)
    raw[1200:1210]=0
    pieces=qwen.split_long(raw)
    assert len(pieces[0]) == 1200
    assert max(map(len, pieces)) <= 1500
    assert torch.equal(torch.cat(pieces), raw)


def test_round11_qwen_long_partition_has_no_dropped_or_duplicate_samples(monkeypatch):
    monkeypatch.setattr(qwen,'SAMPLE_RATE',100)
    monkeypatch.setattr(qwen,'MAX_SECONDS',10)
    monkeypatch.setattr(qwen,'MIN_SECONDS',.01)
    raw=torch.arange(2501,dtype=torch.float32)+1
    pieces=qwen.split_long(raw)
    assert torch.equal(torch.cat(pieces),raw)
    assert all(len(p)>0 for p in pieces)


def test_round12_qwen_token_budget_rejects_bool_and_out_of_range():
    assert qwen.validate_max_new_tokens(32768)==32768
    for value in (True,0,32769,1.5):
        with pytest.raises(ValueError):
            qwen.validate_max_new_tokens(value)


def test_round13_quiet_gain_is_bounded_and_does_not_mutate_caller():
    raw=np.full(10,1e-7,dtype=np.float32);old=raw.copy()
    pcm,gain=limited_quiet_gain(raw)
    assert gain==10 and np.array_equal(raw,old)
    assert np.allclose(pcm,raw*10)


def test_round14_native_prompt_rejects_nul_context_and_unknown_language():
    for kwargs in ({'context':'bad\0context'},{'language':'not-a-language'}):
        with pytest.raises(ValueError):
            build_prompt(**kwargs)
    assert 'language Chinese<asr_text>' in build_prompt(language='Chinese')


def test_round15_native_raw_stop_marker_does_not_destroy_literal_pipe():
    assert parse_raw('language English<asr_text>a | b|')==('English','a | b')
    assert parse_raw('language Chinese<asr_text>你好<|im_end|>ignored')==('Chinese','你好')


def test_round16_offline_segment_owns_reused_input_buffer():
    accepted=[]
    engine=SimpleNamespace(transcribe=lambda audio,**kw:accepted.append(audio.copy()) or dict(text='hello',language='English'))
    segment=OfflineSegment(engine)
    chunk=np.arange(20,dtype=np.float32)
    expected=chunk.copy();segment.feed(chunk);chunk[:]=999
    assert segment.finish()['text']=='hello'
    assert np.array_equal(accepted[0],expected)
    assert segment.finish()['idempotent'] is True and len(accepted)==1


def test_round17_exact_silence_split_preserves_low_level_room_noise():
    wave=np.ones(20000,dtype=np.float32)
    wave[9000:11500]=1e-8
    assert exact_zero_boundaries(wave)==[]
    wave[9000:11500]=0
    assert exact_zero_boundaries(wave)==[11500]


def test_round18_hotword_file_utf8_size_and_escape_boundaries(tmp_path):
    path=tmp_path/'中文.txt';path.write_text('\ufeff语音, 语音；模型',encoding='utf-8')
    assert normalize_hotwords(read_hotword_file(tmp_path,'中文.txt').decode('utf-8-sig'))==('语音, 模型',2)
    with pytest.raises(ValueError):read_hotword_file(tmp_path,'../outside.txt')
    path.write_bytes(b'x'*(65536+1))
    with pytest.raises(ValueError,match='64 KiB'):read_hotword_file(tmp_path,'中文.txt')


def test_round19_custom_model_root_and_unused_aux_empty_path(tmp_path):
    custom=tmp_path/'模型 空间';custom.mkdir()
    paths=resolve_models({'asr_model':'','aligner':''},custom)
    assert paths['model']==(custom/DEFAULT_PATHS['model']).resolve()
    assert paths['asr_model']==(custom/DEFAULT_PATHS['asr_model']).resolve()
    external=tmp_path/'outside.safetensors';external.touch()
    assert resolve_models({'model':str(external)},custom)['model']==external.resolve()


def test_round20_windowed_codec_decode_crops_context_exactly_once(monkeypatch):
    monkeypatch.setattr(codec_module,'DECODE_WINDOW',4)
    monkeypatch.setattr(codec_module,'DECODE_CONTEXT',2)
    monkeypatch.setattr(codec_module,'FRAME_LENGTH',8)
    codec=codec_module.Drama3Codec.__new__(codec_module.Drama3Codec)
    torch.nn.Module.__init__(codec)
    codec.quantizer=SimpleNamespace(decode=lambda c,dtype:c[:,:1,:].repeat_interleave(8,dim=-1).to(dtype))
    codec.decoder=lambda x:x
    codes=torch.arange(11).reshape(1,1,11).expand(1,10,11)
    actual=codec.decode(codes,torch.float32)
    assert torch.equal(actual,torch.arange(11,dtype=torch.float32).repeat_interleave(8)[None])
