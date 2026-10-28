"""Scene Session socket preserves real direction, references and prior history."""
import inspect

import pytest
import torch

from drama3_t8 import nodes
from drama3_t8.engine import Sampling
from drama3_t8.state import Reference, Session, Take
from tools.make_workflows import widget_names


MODES = ('together', 'per_line', 'per_line_context', 'per_speaker')
WIDGETS = ['script', 'scene', 'characters', 'seed', 'control_after_generate',
           'max_frames', 'seed_text', 'generation_mode', 'split_long_text',
           'chunk_bytes', 'temperature', 'top_p', 'top_k', 'opening_retries',
           'sampling_preset']


class Recorder:
    def __init__(self):
        self.calls = []
        self.result = Session()

    def scene(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_session_is_last_optional_socket_without_a_new_or_reordered_widget():
    info = nodes.T8DramaScene.GET_NODE_INFO_V1()
    assert info['input_order']['optional'][-1] == 'session'
    assert info['input']['optional']['session'][0] == 'DRAMA3_T8_SESSION'
    assert widget_names(info) == WIDGETS
    assert info['output'] == ['DRAMA3_T8_SESSION']


def test_session_is_last_execute_parameter_with_none_default():
    signature = inspect.signature(nodes.T8DramaScene.execute)
    assert list(signature.parameters)[-1] == 'session'
    assert signature.parameters['session'].default is None


def test_no_session_preserves_exact_inference_inputs_with_progress():
    engine = Recorder()
    result = nodes.T8DramaScene.execute(engine, 'A: first\nB: second', 'scene',
        'characters', None, 516, 300, '', 'per_speaker', True, 48,
        .7, .7, 30, 0, 'stable')
    args, kwargs = engine.calls[0]
    assert result[0] is engine.result
    assert args == (['A: first', 'B: second'], 'scene', 'characters', (),
                    Sampling(seed=516, max_frames=300))
    assert callable(kwargs.get('progress'))
    kwargs = {key: value for key, value in kwargs.items() if key != 'progress'}
    assert kwargs == dict(generation_mode='per_speaker', split_long_text=True,
                          chunk_bytes=48)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('split', [False, True])
def test_connected_session_is_identical_and_unconnected_refs_remain_none(mode, split):
    engine = Recorder()
    reference = Reference('<|speaker:0|> original', torch.ones((10, 3), dtype=torch.long))
    history = Take('prior', 'prior', torch.zeros((0, 11), dtype=torch.long))
    parent = Session((reference,), (history,), 'Keep this exact direction.')
    result = nodes.T8DramaScene.execute(engine, 'A: next', session=parent,
        generation_mode=mode, split_long_text=split, chunk_bytes=48,
        seed=516, max_frames=300, sampling_preset='stable')
    args, kwargs = engine.calls[0]
    assert result[0] is engine.result
    assert args[3] is None
    assert args[4] == Sampling(seed=516, max_frames=300)
    assert callable(kwargs.get('progress'))
    kwargs = {key: value for key, value in kwargs.items() if key != 'progress'}
    assert kwargs == dict(session=parent, generation_mode=mode,
                          split_long_text=split, chunk_bytes=48)
    assert kwargs['session'] is parent
    assert parent.references[0] is reference and parent.takes[0] is history
    assert parent.direction == 'Keep this exact direction.'


@pytest.mark.parametrize('explicit', [[], ['explicit-reference']])
def test_explicit_references_replace_inherited_refs_even_when_empty(explicit):
    engine = Recorder()
    parent = Session(direction='Original direction')
    nodes.T8DramaScene.execute(engine, 'A: next', references=explicit, session=parent)
    args, kwargs = engine.calls[0]
    assert args[3] == tuple(explicit)
    assert kwargs['session'] is parent


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('split', [False, True])
def test_actual_engine_prompt_history_and_direction_match_direct_scene(mode, split):
    # The real Node, Engine, cast/history and prompt builders all execute.
    # Only expensive model inference/Codec are the established tiny harness.
    from test_scene_chunks import Harness, references
    from drama3_t8.dialogue import audio_frames
    previous = Take('<|speaker:0|> Original words.', 'Original words.',
                    audio_frames(torch.ones((10, 3), dtype=torch.long)))
    parent = Session(references(), (previous,), 'Exact original Mandarin direction.')
    node_engine, direct_engine = Harness(), Harness()
    options = dict(generation_mode=mode, split_long_text=split, chunk_bytes=48)
    actual = nodes.T8DramaScene.execute(node_engine, 'A: first\nB: second',
        'same scene', 'same characters', session=parent, seed=516,
        max_frames=300, sampling_preset='stable', **options)[0]
    expected = direct_engine.scene(['A: first', 'B: second'], 'same scene',
        'same characters', None, Sampling(seed=516, max_frames=300),
        session=parent, **options)
    assert actual.direction == expected.direction == parent.direction
    assert actual.takes[0] is previous and expected.takes[0] is previous
    assert len(node_engine.requests) == len(direct_engine.requests) > 0
    for captured, direct in zip(node_engine.requests, direct_engine.requests):
        assert torch.equal(captured['prompt'], direct['prompt'])
        assert captured['text'] == direct['text']
        assert captured['sampling'] == direct['sampling']
    assert torch.equal(actual.takes[-1].frames, expected.takes[-1].frames)
