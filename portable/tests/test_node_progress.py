"""Real native-node to real Comfy ProgressBar contracts, without model inference."""
from types import SimpleNamespace

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine

standalone_engine()
import comfy.model_management as mm
import comfy.utils as comfy_utils
from drama3_t8 import nodes
from drama3_t8.engine import Sampling
from drama3_t8.state import Reference, Session, Take


ROUTES = ('speech', 'revision', 'long', 'stream', 'scene', 'infill')
METHODS = dict(speech='speak', revision='revise', long='long_form',
               stream='stream_text', scene='scene', infill='infill')
STEPS = {
    'speech': ((0, 91), (3, 91), (8, 91), (91, 91)),
    'revision': ((0, 154), (79, 154), (100, 154), (154, 154)),
    # Actual Engine.long_form currently forwards each chunk's raw progress.
    # Preserve its reset rather than inventing a node-side aggregate or total.
    'long': ((0, 77), (9, 77), (0, 77), (7, 77), (77, 77)),
    'stream': ((0, 101), (1, 101), (3, 101), (101, 101)),
    # Chunked/grouped Scene already emits aggregate progress in Engine.
    'scene': ((0, 462), (79, 462), (236, 462), (462, 462)),
    # Rendered infill uses stages, including fractional generation progress.
    'infill': ((0, 7), (1, 7), (3.25, 7), (6, 7), (7, 7)),
}


@pytest.fixture
def progress_spy(monkeypatch):
    """Observe the genuine ProgressBar method and genuine global-hook boundary."""
    updates, hook_events = [], []
    original = comfy_utils.ProgressBar.update_absolute

    def observed(bar, value, total=None, preview=None):
        updates.append((value, total, bar))
        return original(bar, value, total, preview)

    def hook(value, total, preview, **kwargs):
        hook_events.append((value, total, preview, kwargs))

    monkeypatch.setattr(comfy_utils.ProgressBar, 'update_absolute', observed)
    monkeypatch.setattr(comfy_utils, 'PROGRESS_BAR_HOOK', hook)
    # Recent hosts throttle hook delivery; the contract needs no real sleeps.
    # Older hosts lack these settings and already deliver every update.
    monkeypatch.setattr(comfy_utils, 'PROGRESS_THROTTLE_MIN_INTERVAL', 0.0, raising=False)
    monkeypatch.setattr(comfy_utils, 'PROGRESS_THROTTLE_MIN_PERCENT', 0.0, raising=False)
    return SimpleNamespace(updates=updates, hooks=hook_events)


@pytest.fixture
def source_session():
    reference = Reference('<|speaker:0|> fixed voice',
                          torch.ones((10, 2), dtype=torch.long, device='cpu'))
    frames = torch.ones((3, 11), dtype=torch.long, device='cpu')
    frames[:, 0] = 248100
    source = Take('fixed original', 'fixed original', frames)
    replacement = Take('new text', 'new text', frames.clone())
    parent = Session((reference,), (source,), 'Keep this original direction.')
    return parent, replacement


class CallbackEngine:
    """Controlled inference boundary; real nodes/Session/output/bar still run."""
    def __init__(self, route, parent, take, spy, failure=None):
        self.route = route
        self.parent = parent
        self.take = take
        self.result = parent.append(take)
        self.spy = spy
        self.failure = failure
        self.calls = []
        self.final_hooks_before_return = None

    def _call(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))
        callback = kwargs.get('progress')
        # Missing wiring must fail the assertions below, not fake a red via a
        # None-call TypeError before the original node's behaviour completes.
        if callback is not None:
            for value, total in STEPS[self.route]:
                callback(value, total)
        self.final_hooks_before_return = [event for event in self.spy.hooks
                                          if event[0] >= event[1]]
        if self.failure is not None:
            raise self.failure
        return self.take if method == 'infill' else self.result

    def speak(self, *args, **kwargs):
        return self._call('speak', args, kwargs)

    def revise(self, *args, **kwargs):
        return self._call('revise', args, kwargs)

    def long_form(self, *args, **kwargs):
        return self._call('long_form', args, kwargs)

    def stream_text(self, *args, **kwargs):
        return self._call('stream_text', args, kwargs)

    def scene(self, *args, **kwargs):
        return self._call('scene', args, kwargs)

    def infill(self, *args, **kwargs):
        return self._call('infill', args, kwargs)


def execute(route, engine, parent):
    if route == 'scene':
        return nodes.T8DramaScene.execute(engine, 'A: first\n \nB: second',
            'exact scene', 'exact characters', None, seed=7, max_frames=77,
            seed_text='18446744073709551615', generation_mode='per_speaker',
            split_long_text=True, chunk_bytes=48, temperature=.83, top_p=.91,
            top_k=11, opening_retries=1, sampling_preset='custom', session=parent)
    if route == 'infill':
        return nodes.T8DramaInfill.execute(engine, parent, 'new text',
            .125, .375, duration_frames=9, seed=7, max_frames=77,
            seed_text='18446744073709551615', target_speaker='1',
            expected_view='{"subtype":"FLOAT","sample_rate":44100}')
    return nodes.T8DramaGenerate.execute(engine, route, 'exact words', parent,
        lookahead='exact next', seed=7, temperature=.83, top_p=.91, top_k=11,
        max_frames=77, chunk_bytes=48, seed_text='18446744073709551615',
        opening_retries=1, sampling_preset='custom')


def assert_original_call(route, engine, parent):
    assert len(engine.calls) == 1  # No retry/re-submit introduced at the node.
    method, args, kwargs = engine.calls[0]
    assert method == METHODS[route]
    assert callable(kwargs.get('progress'))
    rest = {key: value for key, value in kwargs.items() if key != 'progress'}
    sampling = Sampling(2**64 - 1, .83, 11, .91, 77, 1)
    if route == 'scene':
        assert args == (['A: first', 'B: second'], 'exact scene', 'exact characters', None, sampling)
        assert rest == dict(generation_mode='per_speaker', split_long_text=True,
                            chunk_bytes=48, session=parent)
        assert rest['session'] is parent
    elif route == 'infill':
        assert args == (parent.takes[-1], 'new text', .125, .375,
                        Sampling(seed=2**64 - 1, max_frames=77), 9)
        assert args[0] is parent.takes[-1]
        assert rest == dict(target_speaker=1,
                            expected_view=dict(subtype='FLOAT', sample_rate=44100))
    elif route == 'revision':
        assert args == (parent, 'exact words', sampling) and args[0] is parent
        assert rest == {}
    elif route == 'stream':
        assert args == ('exact words', 'exact next', parent, sampling) and args[2] is parent
        assert rest == {}
    else:
        assert args == ('exact words', parent, sampling) and args[1] is parent
        assert rest == (dict(chunk_bytes=48) if route == 'long' else {})


@pytest.mark.parametrize('route', ROUTES)
def test_nodes_publish_real_nonzero_progress_preserve_engine_totals_and_complete_on_success(route, source_session, progress_spy):
    parent, take = source_session
    engine = CallbackEngine(route, parent, take, progress_spy)
    output = execute(route, engine, parent)
    assert_original_call(route, engine, parent)
    expected = [(value, total) for value, total in STEPS[route] if value < total]
    actual = [(value, total) for value, total, _ in progress_spy.updates]
    last_total = STEPS[route][-1][1]
    assert actual == expected + [(last_total, last_total)]
    assert any(0 < value < total for value, total, _, _ in progress_spy.hooks)
    assert progress_spy.hooks[-1][:2] == (last_total, last_total)
    assert engine.final_hooks_before_return == []
    assert len({id(bar) for _, _, bar in progress_spy.updates}) == 1
    bar = progress_spy.updates[-1][2]
    assert isinstance(bar, comfy_utils.ProgressBar)
    assert bar.current == bar.total == last_total
    if route == 'infill':
        assert output[0].takes[0] is parent.takes[0]
        assert output[0].takes[-1] is take
        assert output[0].references is parent.references
        assert output[0].direction == parent.direction
    else:
        assert output[0] is engine.result
    assert len(parent.takes) == 1


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('failure_type', (RuntimeError, mm.InterruptProcessingException))
def test_engine_error_or_cancellation_preserves_identity_and_never_commits_100_percent(route, failure_type, source_session, progress_spy):
    parent, take = source_session
    failure = failure_type('controlled engine boundary')
    engine = CallbackEngine(route, parent, take, progress_spy, failure=failure)
    with pytest.raises(failure_type) as raised:
        execute(route, engine, parent)
    assert raised.value is failure
    assert_original_call(route, engine, parent)
    assert any(0 < value < total for value, total, _, _ in progress_spy.hooks)
    assert not any(value >= total for value, total, _, _ in progress_spy.hooks)
    assert not any(value >= total for value, total, _ in progress_spy.updates)
    assert engine.final_hooks_before_return == []
    assert len(parent.takes) == 1


@pytest.mark.parametrize('route', ('speech', 'scene', 'infill'))
def test_output_construction_failure_does_not_commit_progress_endpoint(route, source_session, progress_spy, monkeypatch):
    parent, take = source_session
    engine = CallbackEngine(route, parent, take, progress_spy)
    failure = RuntimeError('controlled output construction')

    def reject_output(*args, **kwargs):
        raise failure

    monkeypatch.setattr(nodes.io, 'NodeOutput', reject_output)
    with pytest.raises(RuntimeError) as raised:
        execute(route, engine, parent)
    assert raised.value is failure
    assert_original_call(route, engine, parent)
    assert any(0 < value < total for value, total, _, _ in progress_spy.hooks)
    assert not any(value >= total for value, total, _, _ in progress_spy.hooks)
    assert not any(value >= total for value, total, _ in progress_spy.updates)
