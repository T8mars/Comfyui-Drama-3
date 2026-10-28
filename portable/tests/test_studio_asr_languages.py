"""Real Studio/HTTP metadata without loading either ASR model."""
import pytest
from fastapi.testclient import TestClient
from studio_app_loader import application
from drama3_t8.qwen3_asr.asr import LANGUAGES
from drama3_t8.r2t2.native import SUPPORTED_LANGUAGES
from drama3_t8.studio import Studio


@pytest.mark.parametrize('backend,expected', [('qwen',LANGUAGES),('r2t2',SUPPORTED_LANGUAGES)])
def test_studio_asr_languages_are_actual_backend_constants_without_models(tmp_path,monkeypatch,backend,expected):
    with Studio(tmp_path) as studio:
        studio.apply_settings({'asr_backend':backend},persist=False)
        monkeypatch.setattr(studio.engine,'load_model',lambda:pytest.fail('Metadata loaded main model'))
        monkeypatch.setattr(studio.engine,'load_codec',lambda:pytest.fail('Metadata loaded Codec'))
        assert studio.asr_languages()==['auto',*expected]
        assert studio.engine.drama is None and studio.engine.codec is None and studio.engine.asr is None


def test_studio_status_languages_follow_applied_backend_and_preserve_settings_snapshot(tmp_path):
    (tmp_path/'web').mkdir();(tmp_path/'web/index.html').write_text('test')
    with TestClient(application.create_app(tmp_path)) as client:
        studio=client.app.state.studio
        for backend,expected in [('r2t2',SUPPORTED_LANGUAGES),('qwen',LANGUAGES),('r2t2',SUPPORTED_LANGUAGES)]:
            studio.apply_settings({'asr_backend':backend},persist=False)
            response=client.get('/api/status')
            assert response.status_code==200
            status=response.json()
            assert status['settings']['asr_backend']==backend
            assert status['asr_languages']=={'backend':backend,'languages':['auto',*expected]}
            assert studio.engine.drama is None and studio.engine.codec is None and studio.engine.asr is None
