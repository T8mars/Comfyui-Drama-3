"""Editor integration with the real decode_take empty-timeline contract.

Codec output and auxiliary-model execution use deterministic CPU substitutes;
DramaEngine.decode_take, alignment.align_audio and editor preparation are real.
"""
from io import BytesIO
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch

from studio_app_loader import application  # Bootstrap the bundled Comfy engine.
from drama3_t8 import editor
from drama3_t8.engine import DramaEngine
from drama3_t8.state import Take


def _take(kind):
    def frames(count, code):
        rows = torch.full((count, 11), code, dtype=torch.int64)
        rows[:, 0] = 248099 + code
        return rows

    if kind == 'single':
        return Take('[cheerful] Cat meeting', 'Cat meeting', frames(20, 1)), ['Cat meeting'], [0]
    scripts = ['Captain: [exasperated] Why overtime?',
               'Cat: [deadpan] Galactic policy.',
               'Captain: [whispers] Please resign.']
    separator = torch.zeros(1, 11, dtype=torch.int64)
    separator[:, 0] = 252196
    rows = torch.cat((frames(6, 1), separator, frames(9, 2), separator, frames(7, 3)))
    metadata = dict(task='scene', generation_mode='together', script_lines=scripts)
    roles = [0, 1, 0]
    if kind == 'mapped_scene':
        metadata['scene_speaker_map'] = {'Captain': 4, 'Cat': 2}
        roles = [4, 2, 4]
    texts = ['Why overtime?', 'Galactic policy.', 'Please resign.']
    return Take('\n'.join(scripts), '\n'.join(texts), rows, metadata=metadata), texts, roles


def _pcm16(wave, rate):
    stream = BytesIO()
    sf.write(stream, wave.detach().cpu().T.numpy(), rate, format='WAV', subtype='PCM_16')
    return stream.getvalue()


@pytest.mark.parametrize('kind', ['single', 'scene', 'mapped_scene'])
@pytest.mark.parametrize('use_aligner', [False, True])
def test_actual_decode_take_empty_timeline_falls_back_to_spoken_lines(
        tmp_path, monkeypatch, kind, use_aligner):
    from drama3_t8.qwen3_asr import asr, align as native_align

    take, texts, roles = _take(kind)
    engine = DramaEngine('unused-model.safetensors', 'unused-codec.safetensors',
                         moe_backend='torch', cuda_graph=False)
    samples = take.codes.shape[-1] * 2048
    # Nonzero endpoints, non-round line durations and a real pause expose any
    # implicit trim, fade, normalization or segment-boundary alteration.
    wave = (torch.sin(torch.arange(samples) * .031) * .25 + .01).reshape(1, 1, -1)
    wave[..., 1000:1800] = 0
    original = dict(waveform=wave, sample_rate=44100)
    decode_calls = []

    def decode(codes):
        assert torch.equal(codes, take.codes)
        decode_calls.append(codes.shape[-1])
        return original

    # Do not replace decode_take: [] is its genuine output for both ordinary
    # takes and jointly generated scenes, including old archives without maps.
    monkeypatch.setattr(engine, 'decode', decode)
    assert engine.decode_take.__func__ is DramaEngine.decode_take
    assert engine.decode_take(take, with_timeline=True) == (original, [])
    align_calls = []
    loads = []
    aligner_path = tmp_path / 'forced-aligner.safetensors'
    aligner_path.touch()
    auxiliary = SimpleNamespace(model=SimpleNamespace(aligner=True))

    def load(path):
        assert use_aligner, 'The optional aligner must not load by default'
        assert path == str(aligner_path)
        loads.append(path)
        return auxiliary

    def align(model, audio, rate, transcript):
        assert model is auxiliary
        assert torch.equal(audio, original['waveform'][0])
        assert rate == 44100
        assert transcript == '\n'.join(texts) and transcript.strip()
        align_calls.append(transcript)
        # Replace only auxiliary-model execution, retaining the real public
        # align_audio transcript validation and result conversion.
        bounds = [0]
        for segment in take.segments():
            bounds.append(bounds[-1] + segment.shape[-1] * 2048 / rate)
        items = []
        for index, text in enumerate(texts):
            units = native_align.words(text)
            start, end = bounds[index], bounds[index + 1]
            span = (end - start) / (len(units) + 1)
            items.extend((word, start + (i + .1) * span, start + (i + .9) * span)
                         for i, word in enumerate(units))
        return items

    monkeypatch.setattr(asr, 'load_qwen3_asr', load)
    monkeypatch.setattr(native_align, 'align', align)
    audio, analysis, edits = editor.prepare(
        engine, take, aligner=str(aligner_path) if use_aligner else None,
        language='English')
    assert decode_calls == [take.codes.shape[-1]] * 2
    assert [line['text'] for line in analysis['lines']] == texts
    assert [line['speaker'] for line in analysis['lines']] == roles
    assert analysis['speakers'] == sorted(set(roles))
    assert analysis['aligned'] is use_aligner
    assert all(line['tokens'] for line in analysis['lines'])
    assert loads == ([str(aligner_path)] if use_aligner else [])
    assert align_calls == (['\n'.join(texts)] if use_aligner else [])
    if use_aligner:
        assert all('start' in token and 'end' in token
                   for line in analysis['lines'] for token in line['tokens'])
    planned = editor.plan(analysis, edits)
    rendered = editor.render(audio['waveform'][0], analysis, planned['pieces'], edits)
    assert torch.equal(rendered, original['waveform'][0])
    assert _pcm16(rendered, 44100) == _pcm16(original['waveform'][0], 44100)
    assert analysis['source'] == editor.fingerprint(take)
