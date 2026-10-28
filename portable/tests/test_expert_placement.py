"""CPU checks for real public placement/settings/node wiring, no large model."""
from types import SimpleNamespace

import pytest
import torch
from drama3_t8.bootstrap import standalone_engine
standalone_engine()
import comfy.model_management as mm
from drama3_t8 import engine as module
from drama3_t8.engine import DramaEngine
from drama3_t8.studio import Studio
from test_optional_model_settings import core_settings


@pytest.fixture(autouse=True)
def cpu_backend():
    previous=mm.cpu_state
    mm.cpu_state=mm.CPUState.CPU
    try:yield
    finally:mm.cpu_state=previous


@pytest.mark.parametrize('placement',['ram','vram'])
def test_engine_delegates_complete_body_and_scene_heads_to_correct_loader(tmp_path,monkeypatch,placement):
    from safetensors.torch import save_file
    loaded=[];resident=[];load_options=[]
    body=object()
    drama=SimpleNamespace(config=SimpleNamespace(output_token_ids=(0,1),text=SimpleNamespace(hidden_size=4)),
        tokenizer=SimpleNamespace(encode=lambda text:SimpleNamespace(ids=[2]),token_to_id=lambda text:3),
        patchers=lambda:[body],parts=lambda:(None,None,None,lambda x:x[...,:2],None),
        load_resident=lambda extra_patchers=():resident.append(tuple(extra_patchers)))
    def load(*args,**kwargs):load_options.append(kwargs);return drama
    monkeypatch.setattr(module,'load_drama3',load)
    monkeypatch.setattr(mm,'load_models_gpu',lambda patchers:loaded.append(tuple(patchers)))
    engine=DramaEngine(tmp_path/'main.safetensors',tmp_path/'codec.safetensors',expert_placement=placement)
    monkeypatch.setattr(engine,'set_backend',lambda backend:None)
    save_file({'weight':torch.arange(16,dtype=torch.float32).reshape(4,4)},str(engine.text_head_path))
    assert engine.load_model() is drama
    assert load_options[0]['experts_in_ram'] is (placement=='ram')
    engine.load_scene_head()
    engine.load_text_head()
    if placement=='vram':
        assert loaded==[]
        assert resident==[(),(engine.line_head_patcher,),(engine.text_head_patcher,)]
    else:
        assert resident==[]
        assert loaded==[(body,),(body,engine.line_head_patcher),(body,engine.text_head_patcher)]


def test_explicit_vram_residency_error_propagates_without_streaming_fallback(tmp_path,monkeypatch):
    def fail(**kwargs):raise MemoryError('Full single-GPU residency needs more VRAM')
    drama=SimpleNamespace(load_resident=fail)
    monkeypatch.setattr(module,'load_drama3',lambda *a,**k:drama)
    monkeypatch.setattr(mm,'load_models_gpu',lambda *a,**k:pytest.fail('Must not fall back to managed streaming'))
    engine=DramaEngine('model','codec',expert_placement='vram')
    monkeypatch.setattr(engine,'set_backend',lambda *a:None)
    with pytest.raises(MemoryError,match='more VRAM'):engine.load_model()


@pytest.mark.parametrize('placement',['ram','vram'])
def test_studio_persists_expert_placement_and_native_node_appends_control(tmp_path,placement):
    from drama3_t8.nodes import T8DramaLoader
    with Studio(tmp_path/'studio') as studio:
        studio.apply_settings({**core_settings(tmp_path),'expert_placement':placement})
        assert studio.engine.expert_placement==placement
    with Studio(tmp_path/'studio') as studio:
        assert studio.settings['expert_placement']==placement and studio.engine.expert_placement==placement
    info=T8DramaLoader.GET_NODE_INFO_V1()
    assert list(info['input']['optional'])[-1]=='expert_placement'
    spec=info['input']['optional']['expert_placement']
    assert (spec[0] if isinstance(spec[0],list) else spec[1]['options'])==['ram','vram']
    paths=core_settings(tmp_path)
    engine=T8DramaLoader.execute('unused','unused',model_path=paths['model'],codec_path=paths['codec'],expert_placement=placement)[0]
    assert engine.expert_placement==placement


def test_placement_validation_rejects_unknown_value_before_settings_change(tmp_path):
    with pytest.raises(ValueError,match='placement'):DramaEngine('m','c',expert_placement='auto')
    with Studio(tmp_path/'studio') as studio:
        studio.apply_settings(core_settings(tmp_path))
        previous=studio.engine;data=studio.settings_path.read_bytes()
        with pytest.raises(ValueError):studio.apply_settings({'expert_placement':'both'})
        assert studio.engine is previous and studio.settings_path.read_bytes()==data


def test_studio_status_reports_selected_placement_without_claiming_backend_execution(tmp_path):
    from fastapi.testclient import TestClient
    from studio_app_loader import application
    (tmp_path/'studio'/'web').mkdir(parents=True)
    with TestClient(application.create_app(tmp_path/'studio')) as client:
        client.app.state.studio.apply_settings({**core_settings(tmp_path),'expert_placement':'vram'})
        result=client.get('/api/status')
        assert result.status_code==200
        acceleration=result.json()['acceleration']
        assert acceleration['expert_placement']=='vram'
        assert acceleration['moe_backend']=='not_run'
        assert acceleration['expert_stats']['cpu_decode_calls']==0
