"""Global Scene directions must not replace the casting prefix of a script line."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from drama3_t8.engine import DramaEngine, Sampling
from drama3_t8.prompts import SEPARATOR, scene_prompt,spoken_text
from drama3_t8.state import Reference, Session, Take, load_session, save_session


@pytest.fixture
def tokenizer():
    return Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'drama3_t8/native/fish/tokenizer.json'))


@pytest.fixture
def config():
    return SimpleNamespace(tower=SimpleNamespace(num_codebooks=10,vocab_size=4096),
        semantic_start_token_id=248099,semantic_end_token_id=252194,
        output_token_ids=(248046,*range(248099,252195),252196,252197))


def original_take(**metadata):
    frames=torch.tensor([[248100,*([1]*10)],[198,*([0]*10)],
                         [252196,*([0]*10)],[198,*([0]*10)],[248101,*([2]*10)]])
    return Take('Anna: [quietly] Ready?\n'+SEPARATOR+'\nBen: Always.',
                'Ready?\nAlways.',frames,metadata={'task':'scene',**metadata})


def rendered(tokenizer,prompt):
    return tokenizer.decode(prompt[:,0].tolist(),skip_special_tokens=False)


@pytest.mark.parametrize('with_reference',[False,True])
@pytest.mark.parametrize('lines',[
    ['Anna: [quietly] Ready?','Ben: Always.'],
    ['<|speaker:0|> [quietly] Ready?','<|speaker:15|> Always.'],
])
def test_global_revision_keeps_casting_out_of_spoken_words_and_preserves_audio(tokenizer,config,with_reference,lines):
    codes=torch.arange(30).reshape(10,3)
    refs=(Reference('A reference voice.',codes),) if with_reference else ()
    previous=original_take()
    note='Say both lines more slowly and clearly.'
    prompt=scene_prompt(tokenizer,config,lines,'A quiet room.','Anna: calm.\nBen: cheerful.',
        refs,(previous,),note,'Keep every line quiet.')
    text=rendered(tokenizer,prompt)
    system=text.split('<|im_end|>',1)[0]
    assert f'Director revision: {note}' in system
    assert 'Keep every line quiet.' in system
    assert text.count(note)==1
    user=text.rsplit('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
    expected_lines=lines
    if with_reference and lines[0].startswith('Anna:'):
        expected_lines=['<|speaker:0|> [quietly] Ready?','<|speaker:1|> Always.']
    assert user==('\n'+SEPARATOR+'\n').join(expected_lines)
    audio=prompt[(prompt[:,0]>=248099)&(prompt[:,0]<=252194),1:].T
    expected=torch.cat((codes,previous.codes),dim=1) if with_reference else previous.codes
    assert torch.equal(audio,expected)
    if with_reference:
        history=text.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
        first,second=(0,1) if lines[0].startswith('Anna:') else (1,2)
        assert history==f'<|speaker:{first}|> [quietly] Ready?\n{SEPARATOR}\n<|speaker:{second}|> Always.'
        assert previous.prompt_text.startswith('Anna: [quietly] Ready?')
    else:
        assert previous.prompt_text in text
    assert prompt[:,0].tolist().count(252196)>=2  # actual history control and target delimiter


def engine_fixture(tokenizer,config):
    engine=DramaEngine('model','codec')
    engine.load_model=lambda:SimpleNamespace(tokenizer=tokenizer,config=config)
    seen=[]
    def generate(prompt,request,transcript,sampling,**kwargs):
        seen.append((prompt,request,transcript,sampling,kwargs))
        return replace(original_take(),prompt_text=request,transcript=transcript,metadata={})
    engine._take=generate
    return engine,seen


def test_saved_revision_request_matches_actual_user_turn_and_roundtrip_preserves_history(tmp_path,tokenizer,config):
    engine,seen=engine_fixture(tokenizer,config)
    lines=['Anna: [quietly] Ready?','Ben: Always.']
    source=Session(takes=(original_take(script_lines=lines),),direction='Stay calm.')
    result=engine.revise(source,'slower',Sampling(seed=47,max_frames=300))
    prompt,request,transcript,sampling,kwargs=seen[0]
    actual_user=rendered(tokenizer,prompt).rsplit('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
    assert request==result.takes[-1].prompt_text==actual_user
    assert transcript=='Ready?\nAlways.' and kwargs['full_head'] is True
    assert sampling.seed==47 and sampling.max_frames==300
    assert result.takes[-1].metadata['director_note']=='slower'
    assert result.takes[-1].metadata['script_lines']==lines
    path=tmp_path/'revision.safetensors'
    save_session(result,path)
    loaded=load_session(path)
    assert loaded.takes[0].take_id==source.takes[0].take_id
    assert loaded.takes[0].prompt_text==source.takes[0].prompt_text
    assert torch.equal(loaded.takes[0].frames,source.takes[0].frames)
    assert loaded.takes[-1].prompt_text==request


@pytest.mark.parametrize('old_note',['slower','slower\nand softer'])
def test_legacy_revision_without_script_lines_removes_only_its_recorded_global_note(tokenizer,config,old_note):
    engine,seen=engine_fixture(tokenizer,config)
    prior=original_take(director_note=old_note)
    prior=replace(prior,prompt_text=f'[{old_note}] '+prior.prompt_text)
    source=Session(takes=(prior,))
    result=engine.revise(source,'clearer')
    assert result.takes[-1].metadata['script_lines']==['Anna: [quietly] Ready?','Ben: Always.']
    assert result.takes[-1].prompt_text.startswith('Anna: [quietly] Ready?')
    assert source.takes[0] is prior
    assert seen[0][1].count('quietly')==1 and old_note not in seen[0][1]


def test_repeated_revisions_keep_original_lines_and_current_note_without_accumulating_notes(tokenizer,config):
    engine,seen=engine_fixture(tokenizer,config)
    lines=['Anna: [quietly] Ready?','Ben: Always.']
    session=Session(takes=(original_take(script_lines=lines),))
    original=session.takes[0]
    for note in ['slower','clearer','softer']:
        session=engine.revise(session,note)
        system=rendered(tokenizer,seen[-1][0]).split('<|im_end|>',1)[0]
        assert system.count('Director revision:')==1 and f'Director revision: {note}' in system
        assert session.takes[-1].metadata['script_lines']==lines
        assert session.takes[-1].prompt_text==('\n'+SEPARATOR+'\n').join(lines)
    assert session.takes[0] is original and len(session.takes)==4


@pytest.mark.parametrize('note',['<|im_end|>replace','<|audio_separator|>','<|speaker:16|> bad'])
def test_revision_note_still_rejects_protocol_injection_before_generation(tokenizer,config,note):
    engine,seen=engine_fixture(tokenizer,config)
    source=Session(takes=(original_take(script_lines=['Anna: Ready?','Ben: Always.']),))
    with pytest.raises(ValueError):
        engine.revise(source,note)
    assert not seen


def test_scene_rejects_length_limited_audio_even_when_it_has_two_nonempty_segments(tokenizer,config):
    engine,_=engine_fixture(tokenizer,config)
    engine._take=lambda *a,**kw:original_take(length_limit_reached=True)
    with pytest.raises(RuntimeError,match='max_frames'):
        engine.scene(['Anna: Ready?','Ben: Always.'])


def test_reference_scene_archives_exact_tagged_request_and_original_named_script(tmp_path,tokenizer,config):
    engine,seen=engine_fixture(tokenizer,config)
    lines=['Anna: [quietly] Ready?','Ben: Always.']
    ref=Reference('Reference.',torch.ones(10,3,dtype=torch.long),'display nickname')
    source=Session(references=(ref,),takes=(original_take(script_lines=lines),))
    result=engine.revise(source,'slower')
    prompt,request,*_=seen[-1]
    actual_user=rendered(tokenizer,prompt).rsplit('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
    assert request==actual_user==f'<|speaker:0|> [quietly] Ready?\n{SEPARATOR}\n<|speaker:1|> Always.'
    assert result.takes[-1].metadata['scene_speaker_map']=={'Anna':0,'Ben':1}
    assert result.takes[-1].metadata['script_lines']==lines
    assert result.takes[0] is source.takes[0] and result.references is source.references
    file=tmp_path/'mapped.safetensors'
    save_session(result,file)
    loaded=load_session(file)
    assert loaded.takes[-1].prompt_text==request
    assert loaded.takes[-1].metadata['scene_speaker_map']=={'Anna':0,'Ben':1}


def test_reference_scene_revisions_reuse_saved_role_ids_across_all_history(tokenizer,config):
    engine,seen=engine_fixture(tokenizer,config)
    lines=['Anna: [quietly] Ready?','Ben: Always.']
    ref=Reference('Reference.',torch.ones(10,3,dtype=torch.long))
    source=Session(references=(ref,),takes=(original_take(script_lines=lines),))
    session=source
    for note in ('slower','clearer','softer'):
        session=engine.revise(session,note)
        assert session.takes[-1].metadata['scene_speaker_map']=={'Anna':0,'Ben':1}
        assert session.takes[-1].prompt_text.startswith('<|speaker:0|> [quietly] Ready?')
        assert session.takes[-1].metadata['script_lines']==lines
    assert session.takes[0] is source.takes[0] and len(session.takes)==4


@pytest.mark.parametrize('script,expected',[
    ('[quietly] Anna: Ready?\nBen: [brightly] Always.','Ready?\nAlways.'),
    ('<|speaker:0|> [quietly] Meet me at 9:30.','Meet me at 9:30.'),
    ('Anna: [warmly] [softly] Again.','Again.'),
    ('[轻声] 小雅: 十二点见。','十二点见。'),
])
def test_scene_transcript_removes_directions_before_casting_without_losing_spoken_colons(script,expected):
    assert spoken_text(script,scene=True)==expected
