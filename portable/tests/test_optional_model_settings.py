"""Real Studio configuration can save TTS without unused auxiliary models."""
from pathlib import Path

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from drama3_t8.model_paths import DEFAULT_PATHS
from drama3_t8.studio import Studio


@pytest.fixture(autouse=True)
def cpu_backend():
    previous = mm.cpu_state
    mm.cpu_state = mm.CPUState.CPU
    try:
        yield
    finally:
        mm.cpu_state = previous


def core_settings(tmp_path):
    paths = {}
    for key in ('model', 'codec', 'line_head'):
        path = tmp_path / 'external' / Path(DEFAULT_PATHS[key]).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths[key] = str(path)
    return paths


@pytest.mark.parametrize('backend', ['r2t2', 'qwen'])
@pytest.mark.parametrize('blank', [False, True])
def test_actual_studio_saves_tts_core_with_no_optional_auxiliaries(tmp_path, backend, blank):
    settings = {**core_settings(tmp_path), 'asr_backend': backend}
    if blank:
        settings.update({key: '' for key in ('asr_model', 'aligner', 'r2t2_model', 'r2t2_projector', 'r2t2_vad')})
    with Studio(tmp_path / 'studio') as studio:
        studio.apply_settings(settings)
        assert studio.engine.model_path == Path(settings['model'])
        assert not studio.model_paths['aligner'].exists()
        assert studio.settings_path.is_file()
    with Studio(tmp_path / 'studio') as reopened:
        assert reopened.engine.codec_path == Path(settings['codec'])
        assert reopened.engine.asr_backend == backend


def test_missing_optional_r2t2_does_not_return_successful_empty_transcription(tmp_path):
    with Studio(tmp_path / 'studio') as studio:
        studio.apply_settings(core_settings(tmp_path), persist=False)
        with pytest.raises(FileNotFoundError, match='r2t2_model'):
            studio.engine.transcribe_detailed(torch.zeros(1, 16000), 16000)


def test_unused_auxiliary_formats_and_selected_head_still_validate_before_commit(tmp_path):
    with Studio(tmp_path/'studio') as studio:
        studio.apply_settings(core_settings(tmp_path))
        previous=studio.engine;data=studio.settings_path.read_bytes()
        with pytest.raises(ValueError,match='asr_model'):
            studio.apply_settings({'asr_model':'unused.bin'})
        with pytest.raises(FileNotFoundError,match='text_head'):
            studio.apply_settings({'scene_head_mode':'full'})
        assert studio.engine is previous and studio.settings_path.read_bytes()==data


def test_qwen_missing_path_is_an_operation_error_not_a_settings_error(tmp_path):
    with Studio(tmp_path/'studio') as studio:
        studio.apply_settings({**core_settings(tmp_path),'asr_backend':'qwen','asr_model':''})
        with pytest.raises(FileNotFoundError,match='asr_model'):
            studio.engine.transcribe_detailed(torch.ones(1,1600),16000)


def test_aligner_required_only_for_actual_alignment_or_editor_words(tmp_path):
    from drama3_t8.state import Session,Take
    with Studio(tmp_path/'studio') as studio:
        studio.apply_settings(core_settings(tmp_path))
        frames=torch.ones(4,11,dtype=torch.int64);frames[:,0]=248100
        session=Session(takes=(Take('test','test',frames),));studio.register_session(session)
        for payload in (dict(mode='align'),dict(mode='editor_prepare',align_words=True)):
            with pytest.raises(FileNotFoundError,match='aligner'):
                studio.execute('test',{**payload,'session_id':session.session_id},lambda *a:None)
