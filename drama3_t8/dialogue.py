"""Isolated speaker generation with archived sources and script-order decoding."""
from dataclasses import replace
import contextlib
import math
import re

import torch

from .state import Session,Take
from .scene_cast import prepare_scene_cast
from .scene_conditioning import isolated_direction
from .prompts import SEPARATOR,spoken_text
from .native.codec import FRAME_LENGTH,SAMPLE_RATE

MODES=('together','per_line','per_line_context','per_speaker')
TAG=re.compile(r'<\|speaker:([0-9]+)\|>')


def speaker(text):
    tags={int(m.group(1)) for m in TAG.finditer(text)}
    if len(tags)>1:raise ValueError('Split each speaker into a separate script line for isolated generation')
    return next(iter(tags),0)


def recast(text,number=0):
    text=TAG.sub('',text).strip()
    return f'<|speaker:{number}|> {text}'


def audio_frames(codes,template=None):
    start=template.semantic_start if template is not None else 248099
    return torch.cat((codes[:1]+start,codes),dim=0).T.contiguous()


def restore_source(raw):
    return Take(raw['prompt_text'],raw['transcript'],torch.tensor(raw['frames'],dtype=torch.long),
        raw.get('semantic_start',248099),raw.get('semantic_end',252194),raw.get('separator_id',252196),
        raw['metadata'],raw['take_id'])


def own_history(session,_seen=None):
    result={}
    seen=set() if _seen is None else _seen
    for previous in session.takes:
        if previous.take_id in seen:continue
        seen.add(previous.take_id)
        if 'render_infill' in previous.metadata:
            from .render_infill import validate_take
            validate_take(previous)
            # Its middle transcript is unknown. Per-role conditioning may
            # reuse real parent turns, never relabel a mixed rendered stream.
            inherited=own_history(Session(takes=(previous.sources[0],)),seen)
            for actor,history in inherited.items():
                known={take.take_id for take in result.get(actor,())}
                result.setdefault(actor,[]).extend(take for take in history if take.take_id not in known)
            continue
        if 'scene_chunking' in previous.metadata:
            from .scene_chunks import source_history,validate_scene
            validate_scene(previous)
            for actor in set(previous.metadata['line_speakers']):
                result.setdefault(actor,[]).extend(source_history(previous,actor))
            continue
        sources=previous.metadata.get('isolated_sources')
        if sources:
            for source in sources:result.setdefault(source['speaker'],[]).append(restore_source(source))
        elif previous.metadata.get('task')=='scene':
            lines=previous.metadata.get('script_lines',[])
            cast,_,_=prepare_scene_cast(lines,session.references,session.takes,force=True)
            segments=previous.segments()
            if len(cast)!=len(segments):continue
            for line,codes in zip(cast,segments):
                actor=speaker(line)
                take=Take(recast(line),spoken_text(line,scene=True),audio_frames(codes,previous),
                    metadata={'source_take_id':previous.take_id,'scene_source_speaker':actor})
                result.setdefault(actor,[]).append(take)
    return result


def isolated_scene(engine,lines,scene,characters,references,sampling,progress,session,note,on_codes,mode):
    hold=engine.generation_session() if hasattr(engine,'generation_session') else contextlib.nullcontext()
    with hold:
        return _isolated_scene(engine,lines,scene,characters,references,sampling,progress,session,note,on_codes,mode)


def _isolated_scene(engine,lines,scene,characters,references,sampling,progress,session,note,on_codes,mode):
    session=session or Session(tuple(references or ()))
    if references is not None:session=replace(session,references=tuple(references))
    references=session.references
    cast,_,mapping=prepare_scene_cast(lines,references,session.takes,force=True)
    actors=[speaker(line) for line in cast]
    groups=([[i for i,actor in enumerate(actors) if actor==selected] for selected in dict.fromkeys(actors)]
            if mode=='per_speaker' else [[i] for i in range(len(lines))])
    histories=own_history(session)
    pieces=[None]*len(lines);sources=[];decode_sources=[];last_source=None;last_actor=None
    for group_index,indices in enumerate(groups):
        actor=actors[indices[0]]
        refs=tuple(replace(ref,transcript=recast(ref.transcript)) for ref in references if speaker(ref.transcript)==actor)
        history=histories.get(actor,[])
        history=history[:1]+history[1:][-1:]
        if mode=='per_line_context' and last_source is not None and last_actor!=actor:
            history=[*history,replace(last_source,prompt_text=recast(last_source.prompt_text,1))]
        name=next((key for key,value in mapping.items() if value==actor),f'speaker {actor}')
        direction=isolated_direction(session.direction,scene,characters,actor=actor,name=name,
            previous_actor=last_actor if mode=='per_line_context' and last_source is not None and last_actor!=actor else None,
            note=note)
        context=Session(refs,tuple(history),direction)
        local=replace(sampling,seed=(sampling.seed+group_index)%2**64)
        def advancing(step,total,index=group_index):
            if progress:progress(index*sampling.max_frames+step,len(groups)*sampling.max_frames)
        if progress:advancing(0,sampling.max_frames)
        if mode=='per_speaker' and len(indices)>1:
            generated=engine.scene([recast(cast[i]) for i in indices],scene='',characters='',
                references=refs,sampling=local,progress=advancing,session=context,note='',generation_mode='together')
        else:
            generated=engine.speak(recast(cast[indices[0]]),context,local,progress=advancing)
        take=generated.takes[-1]
        segments=take.segments()
        if take.metadata.get('length_limit_reached') or len(segments)!=len(indices) or any(not c.shape[-1] for c in segments):
            raise RuntimeError(f'Isolated scene group {group_index+1} did not complete every script line')
        for index,codes in zip(indices,segments):pieces[index]=codes
        sources.append(dict(speaker=actor,line_indexes=indices,take_id=take.take_id,prompt_text=take.prompt_text,
            transcript=take.transcript,frames=take.frames.tolist(),metadata=take.metadata,
            semantic_start=take.semantic_start,semantic_end=take.semantic_end,separator_id=take.separator_id))
        decode_sources.append(dict(line_indexes=indices,source_take_id=take.take_id))
        histories.setdefault(actor,[]).append(take)
        last_source,last_actor=take,actor
    rows=[]
    for i,codes in enumerate(pieces):
        if i:
            control=torch.zeros(1,11,dtype=torch.long);control[0,0]=252196;rows.append(control)
        rows.append(audio_frames(codes))
    metadata=dict(task='scene',generation_mode=mode,scene=scene,characters=characters,script_lines=list(lines),
        director_note=note,scene_speaker_map=mapping,scene_casting='isolated_speaker_tags_v1',
        expected_segments=len(lines),actual_segments=len(lines),isolated_sources=sources,decode_sources=decode_sources,
        line_speakers=actors,gap_seconds=.3,preview_mode='deferred_script_order',seed=sampling.seed,
        sampling=sampling.__dict__,length_limit_reached=False,finish_reason='eos',
        elapsed_seconds=sum(s['metadata'].get('elapsed_seconds',0) for s in sources),
        duration_seconds=(sum(c.shape[-1] for c in pieces)*FRAME_LENGTH+(len(pieces)-1)*round(.3*SAMPLE_RATE))/SAMPLE_RATE)
    details=[s['metadata'] for s in sources]
    metadata['moe_backend']='/'.join(sorted({item.get('moe_backend','torch') for item in details}))
    metadata['moe_requested_backend']='/'.join(sorted({item.get('moe_requested_backend','torch') for item in details}))
    metadata['moe_fallback']='; '.join(dict.fromkeys(item['moe_fallback'] for item in details if item.get('moe_fallback'))) or None
    metadata['cpu_experts_run']=sum(item.get('cpu_experts_run',0) for item in details)
    metadata['expert_placement']='/'.join(sorted({item.get('expert_placement','ram') for item in details}))
    metadata['expert_requested_placement']='/'.join(sorted({item.get('expert_requested_placement','ram') for item in details}))
    # Every source Take records deltas, not shared-cache lifetime totals.
    metadata['expert_backend_calls']={key:sum(item.get('expert_backend_calls',{}).get(key,0) for item in details)
        for key in {key for item in details for key in item.get('expert_backend_calls',{})}}
    metadata['cpu_decode_calls']=sum(item.get('cpu_decode_calls',0) for item in details)
    metadata['gdn_calls_by_backend']={key:sum(item.get('gdn_calls_by_backend',{}).get(key,0) for item in details)
        for key in ('torch','fla')}
    metadata['gdn_calls']=sum(metadata['gdn_calls_by_backend'].values())
    metadata['cuda_graph']={key:sum(item.get('cuda_graph',{}).get(key,0) for item in details)
        for key in ('captures','replays','tower_captures','tower_replays')}
    graphs=[item.get('cuda_graph',{}) for item in details]
    metadata['cuda_graph'].update(
        cpu_decode=any(graph.get('cpu_decode',False) for graph in graphs),
        fallback_reasons=list(dict.fromkeys(reason for graph in graphs for reason in graph.get('fallback_reasons',[]))),
        disabled_reasons=list(dict.fromkeys(graph['disabled_reason'] for graph in graphs if graph.get('disabled_reason'))))
    take=Take(('\n'+SEPARATOR+'\n').join(cast),spoken_text('\n'.join(lines),scene=True),torch.cat(rows),metadata=metadata)
    if on_codes is not None:
        # A codec preview must reset at each independently generated source and
        # preserve the source grouping when a speaker's lines are reordered.
        publish_scene=getattr(getattr(on_codes,'__self__',None),'append_scene',None)
        if publish_scene is not None:publish_scene(take)
        else:
            for codes in pieces:
                for frame in codes.T:on_codes(frame)
    return session.append(take)


def decode_isolated(engine,take):
    segments=take.segments()
    sources=take.metadata['decode_sources']
    if (not isinstance(sources,list) or not sources or
            len(take.metadata.get('line_speakers',[]))!=len(segments) or
            len(take.metadata.get('script_lines',[]))!=len(segments)):
        raise ValueError('Invalid isolated scene metadata')
    indexes=[i for source in sources for i in source['line_indexes']]
    if any(type(i) is not int for i in indexes) or sorted(indexes)!=list(range(len(segments))):
        raise ValueError('Invalid isolated scene source order')
    pieces=[None]*len(segments)
    for source in sources:
        indices=source['line_indexes']
        audio=engine.decode(torch.cat([segments[i] for i in indices],dim=-1))
        wave=audio['waveform'][0]
        cursor=0
        for index in indices:
            count=segments[index].shape[-1]*FRAME_LENGTH
            pieces[index]=wave[:,cursor:cursor+count]
            cursor+=count
        if cursor!=wave.shape[-1]:raise ValueError('Codec duration differs from scene source frames')
    gap=take.metadata.get('gap_seconds',.3)
    if not isinstance(gap,(int,float)) or not math.isfinite(gap) or not 0<=gap<=10:
        raise ValueError('Invalid isolated scene gap')
    output=[];timeline=[];cursor=0
    for index,wave in enumerate(pieces):
        if index:
            quiet=wave.new_zeros((wave.shape[0],round(gap*SAMPLE_RATE)))
            output.append(quiet);cursor+=quiet.shape[-1]
        start=cursor/SAMPLE_RATE
        output.append(wave);cursor+=wave.shape[-1]
        timeline.append(dict(speaker=take.metadata['line_speakers'][index],text=spoken_text(take.metadata['script_lines'][index],scene=True),
            start=start,end=cursor/SAMPLE_RATE,line=index))
    return {'waveform':torch.cat(output,dim=-1)[None],'sample_rate':SAMPLE_RATE},timeline
