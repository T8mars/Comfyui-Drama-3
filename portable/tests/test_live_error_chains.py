"""Retained error chains must not keep a completed worker's codec alive."""
import gc
from types import SimpleNamespace
import traceback
import weakref

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import drama3_t8.live_audio as live


@pytest.mark.parametrize('explicit', [False, True, 'cycle'])
def test_chained_worker_error_releases_codec_preserves_identity_and_stack(tmp_path, monkeypatch, explicit):
    models, errors = [], []
    class Model:
        def low_level(self):
            raise ValueError('inner decode failed')

        def decode(self, *args):
            try:
                self.low_level()
            except ValueError as inner:
                outer = RuntimeError('outer decode failed')
                errors.append((outer, inner))
                if explicit == 'cycle':
                    inner.__cause__ = outer
                if explicit:
                    raise outer from inner
                raise outer

    def load(*args):
        model = Model()
        models.append(weakref.ref(model))
        return SimpleNamespace(model=model)

    monkeypatch.setattr(live, 'load_codec', load)
    worker = live.LiveAudio('unused', tmp_path, chunk_frames=1)
    worker.append(torch.ones(10, dtype=torch.long))
    with pytest.raises(RuntimeError) as raised:
        worker.finish()
    assert not worker.thread.is_alive()
    assert raised.value is errors[0][0] is worker.error
    assert errors[0][1] is (raised.value.__cause__ if explicit else raised.value.__context__)
    assert {'decode'} <= {row.name for row in traceback.extract_tb(raised.value.__traceback__)}
    assert 'low_level' in {row.name for row in traceback.extract_tb(errors[0][1].__traceback__)}
    gc.collect()
    assert models[0]() is None, 'a retained linked exception still owns the codec'


def test_unrelated_prior_cause_keeps_its_debug_locals(tmp_path, monkeypatch):
    class Marker:
        pass
    marker = Marker()
    def previous_failure():
        kept_for_debugging = marker
        raise ValueError('prior unrelated failure')
    try:
        previous_failure()
    except ValueError as exc:
        previous = exc
    class Model:
        def decode(self, *args):
            raise RuntimeError('current decode') from previous
    monkeypatch.setattr(live, 'load_codec', lambda *a:SimpleNamespace(model=Model()))
    worker = live.LiveAudio('unused', tmp_path, chunk_frames=1)
    worker.append(torch.ones(10, dtype=torch.long))
    with pytest.raises(RuntimeError) as raised:
        worker.finish()
    assert raised.value.__cause__ is previous
    tb = previous.__traceback__
    frames = []
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    external = next(frame for frame in frames if frame.f_code.co_name == 'previous_failure')
    assert external.f_locals['kept_for_debugging'] is marker
