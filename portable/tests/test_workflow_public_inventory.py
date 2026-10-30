"""Real host file discovery must not change distributable canvas selections."""
from copy import deepcopy
from pathlib import Path

import pytest

from tools import make_workflows as maker

PUBLIC={('T8DramaLoader','model'):'drama3-t8-fp8.safetensors',
        ('T8DramaLoader','codec'):'drama3-codec.safetensors',
        ('T8DramaTranscribe','asr_model'):'qwen3-asr-06b.safetensors',
        ('T8DramaAlign','aligner'):'qwen3_forced_aligner_0.6b_bf16.safetensors'}
EMPTY={('LoadAudio','audio'):'',('T8DramaLoadSession','session_file'):''}


@pytest.fixture(scope='module')
def real_inventory_examples(tmp_path_factory):
    import folder_paths
    from drama3_t8 import nodes

    root=tmp_path_factory.mktemp('real-workflow-inventories')
    results={}
    for variant in ('empty','private','public','mixed'):
        base=root/variant
        model_dirs={key:base/'models'/folder for key,folder in (
            ('diffusion_models','diffusion_models'),('vae','vae'),('asr','asr'),
            ('drama3_t8_sessions','sessions'))}
        for folder in model_dirs.values():folder.mkdir(parents=True)
        audio=base/'input';audio.mkdir()
        if variant in ('private','mixed'):
            for key in ('diffusion_models','vae','asr','drama3_t8_sessions'):
                (model_dirs[key]/'000-authors-private.safetensors').touch()
            align=model_dirs['asr']/'000-authors-private-aligner';align.mkdir();(align/'config.json').write_text('{}')
            (audio/'000-authors-private.wav').touch()
        if variant in ('public','mixed'):
            for key,name in (('diffusion_models','drama3-t8-fp8.safetensors'),
                             ('vae','drama3-codec.safetensors'),('asr','qwen3-asr-06b.safetensors')):
                (model_dirs[key]/name).touch()
            (model_dirs['asr']/'qwen3_forced_aligner_0.6b_bf16.safetensors').touch()
        with pytest.MonkeyPatch.context() as patch:
            patch.syspath_prepend(str(Path(folder_paths.__file__).resolve().parent))
            patch.setattr(folder_paths,'models_dir',str(base/'models'))
            patch.setattr(folder_paths,'filename_list_cache',{})
            patch.setattr(folder_paths.cache_helper,'cache',{})
            patch.setattr(folder_paths,'get_input_directory',lambda audio=audio:str(audio))
            for key,folder in model_dirs.items():
                patch.setitem(folder_paths.folder_names_and_paths,key,([str(folder)],{'.safetensors'}))
            maker.node_infos.cache_clear()
            try:
                info=deepcopy(maker.node_infos())
                assert nodes.folder_paths is folder_paths
                choices={}
                for key in PUBLIC.keys()|EMPTY.keys():
                    raw=dict(maker.input_specs(info[key[0]]))[key[1]]
                    choices[key]=raw[0] if isinstance(raw[0],list) else raw[1].get('options',[])
                graphs=maker.make_workflows()
                results[variant]=dict(infos=info,choices=choices,graphs=graphs)
            finally:maker.node_infos.cache_clear()
    return results


@pytest.mark.parametrize('variant',['empty','private','public','mixed'])
def test_real_empty_private_public_and_mixed_file_lists_keep_public_or_unselected_values(real_inventory_examples,variant):
    result=real_inventory_examples[variant]
    choices=result['choices']
    assert ('000-authors-private.safetensors' in choices['T8DramaLoader','model'])==(variant in ('private','mixed'))
    assert ('drama3-t8-fp8.safetensors' in choices['T8DramaLoader','model'])==(variant in ('public','mixed'))
    assert ('000-authors-private.wav' in choices['LoadAudio','audio'])==(variant in ('private','mixed'))
    assert ('qwen3_forced_aligner_0.6b_bf16.safetensors' in choices['T8DramaAlign','aligner'])==(variant in ('public','mixed'))
    for graph in result['graphs'].values():
        for node in graph['nodes']:
            values=dict(zip(maker.widget_names(result['infos'][node['type']]),node['widgets_values']))
            for (node_type,name),expected in (PUBLIC|EMPTY).items():
                if node_type==node['type']:
                    assert values[name]==expected
                    assert 'authors-private' not in str(values[name])


def test_all_19_canvas_graphs_are_identical_across_real_local_inventories(real_inventory_examples):
    first=real_inventory_examples['empty']['graphs']
    assert len(first)==19
    assert all(result['graphs']==first for result in real_inventory_examples.values())


@pytest.mark.parametrize('selection,expected_valid',[
    ('qwen-aligner',False),('qwen3_forced_aligner_0.6b_bf16.safetensors',True)])
def test_official_prompt_validation_uses_installed_single_file_combo_even_with_model_path(tmp_path,monkeypatch,selection,expected_valid):
    import asyncio
    import folder_paths
    # Packaging helpers may prepend Studio's app.py during collection.
    # Resolve execution/nodes against this test's real installed Comfy host.
    monkeypatch.syspath_prepend(str(Path(folder_paths.__file__).resolve().parent))
    import execution
    import nodes as host_nodes
    from drama3_t8.nodes import T8DramaAlign

    model=tmp_path/'models/asr/qwen3_forced_aligner_0.6b_bf16.safetensors'
    model.parent.mkdir(parents=True);model.touch()
    monkeypatch.setattr(folder_paths,'models_dir',str(tmp_path/'models'))
    actual=T8DramaAlign.GET_NODE_INFO_V1()['input']['required']['aligner']
    options=actual[0] if isinstance(actual[0],list) else actual[1]['options']
    assert options==[model.name]
    class AudioFixture:
        RETURN_TYPES=('AUDIO',)
        @classmethod
        def INPUT_TYPES(cls):return {'required':{}}
    monkeypatch.setitem(host_nodes.NODE_CLASS_MAPPINGS,'InventoryAudioFixture',AudioFixture)
    monkeypatch.setitem(host_nodes.NODE_CLASS_MAPPINGS,'T8DramaAlign',T8DramaAlign)
    prompt={'audio':{'class_type':'InventoryAudioFixture','inputs':{}},
            'align':{'class_type':'T8DramaAlign','inputs':dict(aligner=selection,audio=['audio',0],
                      transcript='你好',language='Chinese',model_path=str(model))}}
    valid,errors,_=asyncio.run(execution.validate_inputs('inventory-diagnostic',prompt,'align',{}))
    assert valid is expected_valid
    assert [error['type'] for error in errors]==([] if expected_valid else ['value_not_in_list'])


@pytest.mark.parametrize('node_type,name,value',[
    ('T8DramaLoader','gdn_backend','invented'),('T8DramaLoader','scene_head_mode','invented'),
    ('T8DramaLoader','expert_placement','invented'),('T8DramaGenerate','mode','invented'),
    ('T8DramaScene','generation_mode','invented'),('T8DramaScene','sampling_preset','invented'),
    ('T8DramaLoader','model','not-installed.safetensors'),('T8DramaLoader','codec','not-installed.safetensors'),
    ('T8DramaTranscribe','asr_model','not-installed.safetensors'),('T8DramaAlign','aligner','not-installed'),
    ('LoadAudio','audio','not-uploaded.wav'),('T8DramaLoadSession','session_file','not-saved.safetensors')])
def test_only_public_files_and_intentional_empty_inputs_bypass_inventory_membership(real_inventory_examples,monkeypatch,node_type,name,value):
    info=real_inventory_examples['private']['infos']
    monkeypatch.setattr(maker,'node_infos',lambda:info)
    with pytest.raises(ValueError,match='Invalid combo value'):
        maker.widget_values(node_type,{name:value})
