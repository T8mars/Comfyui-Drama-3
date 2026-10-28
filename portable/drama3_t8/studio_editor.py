"""Studio transport for the shared editor. GPU preparation uses Studio's queue;
plan and rendering use prepared CPU audio and immutable source identities.
"""
import json
import re
import time
import uuid

import soundfile as sf
import torch

from . import editor
from .state import save_session


def prepare_job(studio, job_id, payload, progress):
    from .studio import integer_option, result_files
    session=studio.get_session(payload.get('session_id'))
    index=integer_option(payload.get('take_index',-1),'演出版本 / take')
    if not session.takes or not -len(session.takes)<=index<len(session.takes):
        raise ValueError('The requested take does not exist')
    index %= len(session.takes)
    aligner=studio.model_paths['aligner'] if payload.get('align_words',False) else None
    if type(payload.get('align_words',False)) is not bool:
        raise ValueError('align_words must be a boolean')
    audio,analysis,edits=editor.prepare(studio.engine,session.takes[index],aligner=aligner,
        language=payload.get('language','Chinese'))
    progress()
    studio.begin_commit(job_id,progress)
    source=studio.output/f'{job_id}.editor-source.json'
    wav=studio.output/f'{job_id}.editor-source.wav'
    record=dict(analysis=analysis,session_id=session.session_id,take_index=index,edits=edits)
    with result_files(source,wav):
        sf.write(wav,audio['waveform'][0].T.numpy(),audio['sample_rate'],subtype='FLOAT')
        source.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return dict(editor=dict(source_id=job_id,session_id=session.session_id,take_index=index,
        analysis=analysis,edits=edits,audio=f'/api/files/output/{wav.name}'))


def prepared(studio,source_id):
    if not isinstance(source_id,str) or not re.fullmatch('[0-9a-f]{32}',source_id):
        raise ValueError('Invalid editor source ID')
    path=studio.output/f'{source_id}.editor-source.json'
    if not path.is_file():
        raise ValueError('Editor preview is unavailable; reopen the selected take')
    record=json.loads(path.read_text(encoding='utf-8'))
    editor.validate_analysis(record['analysis'])
    session=studio.get_session(record['session_id'])
    index=record['take_index']
    if type(index) is not int or not 0<=index<len(session.takes) or editor.fingerprint(session.takes[index])!=record['analysis']['source']:
        raise ValueError('Editor source no longer matches its session')
    return record,session


def preview_plan(studio,payload):
    record,_=prepared(studio,payload.get('source_id'))
    return editor.plan(record['analysis'],payload.get('edits',{}))


def render_result(studio,payload):
    from .studio import result_files,playback_identity
    record,session=prepared(studio,payload.get('source_id'))
    # Check the UI's selection again rather than quietly editing the prior project.
    if (payload.get('session_id')!=record['session_id'] or type(payload.get('take_index')) is not int
            or payload['take_index']!=record['take_index']):
        raise ValueError('Selected session/take changed; reopen the editor')
    analysis=record['analysis']
    edits=editor.parse_edits(payload.get('edits',{}),source=analysis['source'],duration=analysis['duration'])
    planned=editor.plan(analysis,edits)
    wave,rate=sf.read(studio.output/f"{payload['source_id']}.editor-source.wav",dtype='float32',always_2d=True)
    if rate!=analysis['sample_rate']:
        raise ValueError('Editor preview sample rate changed')
    output=editor.render(torch.from_numpy(wave.T.copy()),analysis,planned['pieces'],edits)
    branch=editor.edited_session(session,record['take_index'],analysis,edits)
    take=branch.takes[record['take_index']]
    view=playback_identity(take,dict(waveform=output[None],sample_rate=rate),'FLOAT')
    identity=uuid.uuid4().hex
    wav=studio.output/f'{identity}.editor.wav'
    metadata=studio.output/f'{identity}.editor.json'
    subtitle=studio.output/f'{identity}.editor.srt'
    project=studio.data/'sessions'/f'{branch.session_id}.safetensors'
    preview=studio.output/'previews'/f"{branch.session_id}_{record['take_index']}.wav"
    preview.parent.mkdir(exist_ok=True)
    with result_files(wav,metadata,subtitle,project,preview):
        sf.write(wav,output.T.numpy(),rate,subtype='FLOAT')
        import shutil
        shutil.copyfile(wav,preview)
        metadata.write_text(json.dumps(take.metadata['audio_editor'],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        subtitle.write_text(editor.subtitle_srt(take.metadata['audio_editor']['captions']),encoding='utf-8')
        save_session(branch,project)
        with studio.connect() as db:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(branch.session_id,project.relative_to(studio.data).as_posix(),
                '后期剪辑 · '+take.transcript[:90],time.time()))
    return dict(audio=f'/api/files/output/{wav.name}',metadata=f'/api/files/output/{metadata.name}',
        subtitle=f'/api/files/output/{subtitle.name}',project=f'/api/files/session/{project.name}',
        session_id=branch.session_id,take_id=take.take_id,transcript=take.transcript,
        take_index=record['take_index'],
        take_count=len(branch.takes),playback_view=view,
        metrics=dict(duration_seconds=planned['duration'],elapsed_seconds=0),editor_source=analysis['source'])
