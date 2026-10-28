"""Session cards must report the duration of the audio they actually play."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from studio_app_loader import application
from test_render_consumers import CanvasEngine, initial, infill_node
from test_scene_chunks import SCRIPT, references
from drama3_t8 import editor


@pytest.mark.parametrize('kind', ['together', 'isolated', 'chunked', 'infill', 'edited'])
def test_session_card_matches_actual_decoded_or_edited_audio(tmp_path, monkeypatch, kind):
    engine = CanvasEngine()
    if kind in ('chunked', 'infill', 'edited'):
        engine, session, timeline = initial(engine)
    else:
        session = engine.scene(SCRIPT, references=references(),
            generation_mode='together' if kind == 'together' else 'per_line')
    if kind == 'infill':
        session = infill_node(engine, session, timeline)
    if kind == 'edited':
        _, analysis, edits = editor.prepare(engine, session.takes[-1])
        session = editor.edited_session(session, -1, analysis,
            {**edits, 'cuts': [[0, .02]], 'inserts': [[.03, .17]]})
    take = session.takes[-1]
    audio = editor.decode_output(engine, take)
    expected = audio['waveform'].shape[-1] / audio['sample_rate']
    (tmp_path / 'web').mkdir()
    (tmp_path / 'web/index.html').write_text('test', encoding='utf-8')
    app = application.create_app(tmp_path)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.studio, 'get_session', lambda identifier: session)
        monkeypatch.setattr(app.state.studio.engine, 'load_model',
            lambda *a, **k: pytest.fail('Listing a session loaded a model'))
        response = client.get('/api/sessions/' + session.session_id)
        assert response.status_code == 200, response.text
        actual = response.json()['takes'][-1]['duration']
        assert actual == pytest.approx(expected, abs=1 / audio['sample_rate'])
