from types import SimpleNamespace
import pytest
from studio_app_loader import application
from drama3_t8.state import Session

def test_loaded_session_applies_explicit_current_direction_without_mutating_archive():
    source=Session(direction='warm')
    host=SimpleNamespace(get_session=lambda _:source)
    assert application.Studio.session_for(host,{'session_id':'saved'}) is source
    updated=application.Studio.session_for(host,{'session_id':'saved','direction':'angry'})
    assert updated.direction=='angry' and source.direction=='warm'
    assert updated.references is source.references and updated.takes is source.takes
    assert application.Studio.session_for(host,{'session_id':'saved','direction':''}).direction==''

@pytest.mark.parametrize('mode',['tts','long','text_stream','scene','dialogue'])
def test_current_direction_override_reaches_each_generation_path(mode):
    seen=[]
    class Engine:
        def __getattr__(self,name):
            def call(*args,**kwargs):seen.append((name,args,kwargs));return Session()
            return call
    host=SimpleNamespace(engine=Engine())
    payload={'text':'<|speaker:0|> [warmly] 原文。[laughs]','replace_delivery_tags':True,'scene':'深夜','characters':'A: angry'}
    application.Studio.generate_session(host,mode,payload,Session(direction='angry'),None,None)
    text=seen[0][1][0]
    if isinstance(text,list):text='\n'.join(text)
    assert '[warmly]' not in text and '[laughs]' in text
    assert '[warmly]' in payload['text']
    if mode=='scene':assert seen[0][1][1:3]==('深夜','A: angry')


def test_scene_new_character_description_can_override_old_delivery_without_global_direction():
    seen=[]
    host=SimpleNamespace(engine=SimpleNamespace(scene=lambda *args,**kw:seen.append(args) or Session()))
    payload={'text':'导演：[warmly] 开始。','characters':'导演：生气','replace_delivery_tags':True}
    application.Studio.generate_session(host,'scene',payload,Session(),None,None)
    assert seen[0][0]==['导演：开始。']
