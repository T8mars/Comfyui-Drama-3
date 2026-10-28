"""A failed preview callback must not retain the completed scene PCM."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.state import Take
from drama3_t8 import dialogue, live_audio


def test_failed_scene_preview_releases_its_owner_frame_pcm(tmp_path, monkeypatch):
    held = []
    original = dialogue.decode_isolated
    def record_decode(*args):
        audio, timeline = original(*args)
        held.append(weakref.ref(audio['waveform']))
        return audio, timeline
    monkeypatch.setattr(dialogue, 'decode_isolated', record_decode)
    monkeypatch.setattr(live_audio, 'load_codec', lambda *a: SimpleNamespace(model=SimpleNamespace(
        decode=lambda codes, dtype:torch.zeros(1, codes.shape[-1]*2048))))
    frames = torch.zeros(1, 11, dtype=torch.long)
    frames[:, 0] = 248099
    take = Take('hello', 'hello', frames, metadata=dict(decode_sources=[{'line_indexes':[0]}],
        line_speakers=[0], script_lines=['hello'], gap_seconds=0))
    def fail_preview(path):
        raise RuntimeError('preview callback failed')
    worker = live_audio.LiveAudio('unused', tmp_path, chunk_frames=1, on_chunk=fail_preview)
    worker.append_scene(take)
    with pytest.raises(RuntimeError, match='preview callback failed'):
        worker.finish()
    assert not worker.thread.is_alive() and len(held) == 1
    gc.collect()
    assert held[0]() is None, 'the retained worker error still owns scene PCM'
