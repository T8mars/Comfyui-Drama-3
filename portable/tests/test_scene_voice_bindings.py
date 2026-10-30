import pytest
from drama3_t8.scene_voice_bindings import prepare_scene_voice_bindings, validate_role_voice_collisions

SCRIPT='导演：开始。\n女主：知道啦。\n旁白：稍后再说。'

def test_three_distinct_voices_defaulting_to_zero_are_rejected_before_generation():
    with pytest.raises(ValueError,match='同一个角色编号'):
        prepare_scene_voice_bindings({'mode':'scene','text':SCRIPT,'voice_versions':[
            {'voice_id':voice,'version_id':'fixed','speaker':0} for voice in ['male_narrator','female_lead','male_director']]})

def test_role_names_bind_voices_to_actual_script_ids_without_mutating_choices():
    original=[{'role':'女主','voice_id':'female','version_id':'fixed','speaker':0},
              {'role':'旁白','voice_id':'male_n','version_id':'fixed','speaker':0},
              {'role':'导演','voice_id':'male_d','version_id':'fixed','speaker':0}]
    result=prepare_scene_voice_bindings({'mode':'scene','text':SCRIPT,'voice_versions':original})
    assert [(x['voice_id'],x['speaker']) for x in result['voice_versions']]==[('female',1),('male_n',2),('male_d',0)]
    assert all(x['speaker']==0 for x in original)
    reordered=prepare_scene_voice_bindings({'mode':'scene','text':'女主：先来。\n导演：开始。\n旁白：之后。','voice_versions':original})
    assert reordered['voice_versions'][0]['speaker']==0 and reordered['voice_versions'][2]['speaker']==1

def test_explicit_tag_reservations_and_named_roles_use_one_canonical_map():
    out=prepare_scene_voice_bindings({'mode':'scene','text':'<|speaker:0|> Hi.\n女主：你好。',
        'voice_versions':[{'voice_id':'female','version_id':'fixed','role':'女主'}]})
    assert out['voice_versions'][0]['speaker']==1

def test_unknown_role_is_not_guessed_from_voice_name():
    with pytest.raises(ValueError,match='不在当前剧本'):
        prepare_scene_voice_bindings({'mode':'scene','text':SCRIPT,'voice_versions':[{'role':'小红','voice_id':'female','version_id':'fixed'}]})

def test_shared_voice_and_multiple_versions_remain_supported():
    validate_role_voice_collisions([{'role':'甲','voice_id':'one_voice','speaker':0}, {'role':'乙','voice_id':'one_voice','speaker':0}])
    prepare_scene_voice_bindings({'mode':'scene','text':SCRIPT,'voice_versions':[
        {'role':'女主','voice_id':'female','version_id':version} for version in ['soft','bright']]})
    with pytest.raises(ValueError,match='独立角色编号'):
        validate_role_voice_collisions([{'role':'甲','voice_id':'male','speaker':0},{'role':'乙','voice_id':'female','speaker':0}])
