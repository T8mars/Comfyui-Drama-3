"""Non-destructive, source-bound audio editing shared by Studio and native nodes.

The silence/piece model is adapted from PedroMarinhoDev/ComfyUI-Drama3,
commit 8cb3c64379d4988b9693cae9f3436b8fd54b22ad (GPL-3.0).
Validation, source binding, immutable sessions, captions and identity rendering
are T8 additions. Model history always keeps the original codec frames.
"""
import bisect
import copy
import hashlib
import json
import math
import re
import uuid
from dataclasses import replace

import torch

ENVELOPE_STEP = .01
MIN_PAUSE = .1
MAX_DURATION = 7200
MAX_ITEMS = 20000
DEFAULTS = dict(threshold_db=-50., trim_start=False, lead=.05, trim_end=False,
    tail=.15, pause_longer=7200., pause_to=7200., gap_min=0., gap_max=7200.,
    keep=[], cuts=[], inserts=[], speakers={}, normalize=False)


def finite(value, name, minimum=0., maximum=MAX_DURATION):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return float(value)


def json_object(value, name, maximum=4_000_000):
    if isinstance(value, str):
        if len(value) > maximum:
            raise ValueError(f"{name} is too large")
        try:
            value = json.loads(value)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    try:
        text = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{name} must contain finite JSON data") from exc
    if len(text) > maximum:
        raise ValueError(f"{name} is too large")
    return value


def fingerprint(take):
    if take.sources or take.blobs:
        from .state import Session,validate_session
        validate_session(Session(takes=(take,)))
    if 'scene_chunking' in take.metadata:
        from .scene_chunks import validate_scene
        validate_scene(take)
    return fingerprint_core(take)


def fingerprint_core(take):
    """Pure identity of this view; graph validators must not recursively reenter."""
    h = hashlib.sha256()
    h.update(take.take_id.encode('utf-8'))
    h.update(take.frames.detach().cpu().to(torch.int64).contiguous().numpy().tobytes())
    h.update(json.dumps(dict(text=take.transcript, prompt=take.prompt_text,
        semantic_start=take.semantic_start, semantic_end=take.semantic_end, separator_id=take.separator_id,
        script=take.metadata.get('script_lines'), sources=take.metadata.get('decode_sources'),
        speakers=take.metadata.get('line_speakers'), gap=take.metadata.get('gap_seconds'),
        scene_chunking=take.metadata.get('scene_chunking'), scene_units=take.metadata.get('scene_units'),
        scene_sources=take.metadata.get('scene_sources'), render_infill=take.metadata.get('render_infill'),
        typed_sources=[source.take_id for source in take.sources]),
        ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8'))
    for name,blob in sorted(take.blobs.items()):
        h.update(json.dumps([name,str(blob.dtype),list(blob.shape)]).encode('utf-8'))
        h.update(blob.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parse_edits(value=None, *, source=None, duration=MAX_DURATION):
    value = json_object({} if value is None else value, 'edits')
    unknown = set(value) - set(DEFAULTS) - {'source'}
    if unknown:
        raise ValueError(f"Unknown editor setting: {sorted(unknown)[0]}")
    edits = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(value)}
    if 'source' in edits and (not isinstance(edits['source'], str) or not re.fullmatch('[0-9a-f]{64}', edits['source'])):
        raise ValueError('Invalid editor source fingerprint')
    if source is not None:
        if 'source' in edits and edits['source'] != source:
            raise ValueError('Editor source changed; reopen the take before applying edits')
        if value and 'source' not in value:
            # Empty JSON is the only unbound request; persisted edits require identity.
            raise ValueError('Editor edits require a source fingerprint')
        edits['source'] = source
    for field in ('trim_start', 'trim_end', 'normalize'):
        if type(edits[field]) is not bool:
            raise ValueError(f'{field} must be a boolean')
    finite(edits['threshold_db'], 'threshold_db', -120, 0)
    for field in ('lead', 'tail', 'pause_longer', 'pause_to', 'gap_min', 'gap_max'):
        finite(edits[field], field)
    if edits['gap_min'] > edits['gap_max']:
        raise ValueError('gap_min must not exceed gap_max')
    if edits['pause_to'] > edits['pause_longer']:
        raise ValueError('pause_to must not exceed pause_longer')
    for field in ('cuts', 'inserts'):
        values = edits[field]
        if not isinstance(values, list) or len(values) > MAX_ITEMS:
            raise ValueError(f'{field} must be a bounded list')
        for item in values:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(f'Invalid {field} item')
            finite(item[0], field, 0, duration)
            finite(item[1], field, 0, duration if field == 'cuts' else 60)
            if field == 'cuts' and item[1] <= item[0]:
                raise ValueError('Cut end must be after start')
            if field == 'inserts' and item[1] <= 0:
                raise ValueError('Inserted silence must be positive')
    if sum(item[1] for item in edits['inserts']) > MAX_DURATION:
        raise ValueError('Inserted silence is too long')
    if (not isinstance(edits['keep'], list) or len(edits['keep']) > MAX_ITEMS or
            any(not isinstance(x, str) or len(x) > 32 for x in edits['keep'])):
        raise ValueError('Invalid kept pauses')
    speakers = edits['speakers']
    if not isinstance(speakers, dict) or len(speakers) > 16:
        raise ValueError('Invalid speaker settings')
    for speaker, setting in speakers.items():
        if not isinstance(speaker, str) or not re.fullmatch(r'(?:[0-9]|1[0-5])', speaker):
            raise ValueError('Speaker must be 0 to 15')
        if not isinstance(setting, dict) or set(setting) - {'name', 'gain_db', 'mute'}:
            raise ValueError('Invalid speaker setting')
        if 'gain_db' in setting:
            finite(setting['gain_db'], 'gain_db', -60, 24)
        if 'mute' in setting and type(setting['mute']) is not bool:
            raise ValueError('mute must be a boolean')
        if 'name' in setting and (not isinstance(setting['name'], str) or len(setting['name']) > 80):
            raise ValueError('Speaker name must be text, at most 80 characters')
    return edits


def source_lines(take, duration, timeline=None):
    from .prompts import spoken_text
    if 'render_infill' in take.metadata:
        from .render_infill import render_timeline
        timeline=render_timeline(take)
    if 'scene_chunking' in take.metadata:
        from .scene_chunks import validate_scene
        validate_scene(take)
        if not timeline:
            cursor=0;timeline=[]
            for index,(text,codes) in enumerate(zip(take.metadata['script_lines'],take.segments())):
                if index:cursor+=round(take.metadata['gap_seconds']*44100)
                end=cursor+codes.shape[-1]*2048
                timeline.append(dict(line=index,speaker=take.metadata['line_speakers'][index],
                    text=spoken_text(text,scene=True),start=cursor/44100,end=end/44100))
                cursor=end
    # Ordinary takes and jointly decoded scenes return [] when the codec has
    # no explicit source timeline; recover their transcript/segment metadata.
    if timeline:
        return [dict(line, chunk=i) for i, line in enumerate(timeline)]
    scripts = take.metadata.get('script_lines')
    segments = take.segments()
    if not scripts or len(scripts) != len(segments):
        return [dict(speaker=0, chunk=0, start=0., end=duration, text=take.transcript)]
    mapping = dict(take.metadata.get('scene_speaker_map', {}))
    occupied = set(mapping.values())
    out, cursor = [], 0.
    for index, (script, codes) in enumerate(zip(scripts, segments)):
        match = re.search(r'<\|speaker:(\d+)\|>', script)
        named = re.match(r'\s*(?:\[[^\]]*\]\s*)*([^<>:\[\]]{1,48}):', script)
        name = named.group(1).strip() if named else ''
        if name and name not in mapping:
            mapping[name] = next((i for i in range(16) if i not in occupied), 0)
            occupied.add(mapping[name])
        speaker = int(match.group(1)) if match else mapping.get(name, 0)
        end = min(duration, cursor + codes.shape[-1] * 2048 / 44100)
        out.append(dict(speaker=speaker, chunk=index, start=cursor, end=end,
                        text=spoken_text(script, scene=True)))
        cursor = end
    if out:
        out[-1]['end'] = duration
    return out


def _tokens(text):
    # Untimed display is honest; alignment replaces these with actual word units.
    return [dict(text=t, spoken=True) for t in re.findall(r'[\u3400-\u9fff]|[^\s\u3400-\u9fff]+', text)]


def analyze(waveform, sample_rate, lines, *, source, aligned_words=None):
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 192000:
        raise ValueError('Invalid editor sample rate')
    wave = waveform.detach().to(device='cpu', dtype=torch.float32)
    if wave.ndim == 2:
        wave = wave.mean(0)
    if wave.ndim != 1 or not wave.numel() or not torch.isfinite(wave).all():
        raise ValueError('Editor audio must be finite and nonempty')
    duration = wave.numel() / sample_rate
    finite(duration, 'Audio duration', 0, MAX_DURATION)
    step = max(1, round(ENVELOPE_STEP * sample_rate))
    padded = torch.nn.functional.pad(wave, (0, (-wave.numel()) % step))
    envelope = (10 * torch.log10(padded.reshape(-1, step).square().mean(-1) + 1e-12)).tolist()
    result = dict(version=1, source=source, sample_rate=sample_rate, samples=wave.numel(),
        duration=duration, envelope_step=step / sample_rate, envelope=envelope, lines=[],
        speakers=[], aligned=aligned_words is not None)
    for index, line in enumerate(lines):
        start, end = line['start'], line['end']
        tokens = _tokens(line['text'])
        if aligned_words is not None:
            tokens = [dict(text=w['text'], spoken=True, start=max(start,w['start']), end=min(end,w['end']))
                      for w in aligned_words if start <= (w['start'] + w['end']) / 2 < end]
        result['lines'].append(dict(speaker=line.get('speaker', 0), chunk=index,
            start=start, end=end, text=line['text'], tokens=tokens,
            needs_alignment=bool(line.get('needs_alignment',False) and aligned_words is None)))
    result['speakers'] = sorted({line['speaker'] for line in result['lines']})
    validate_analysis(result)
    return result


def validate_analysis(value):
    value = json_object(value, 'analysis', maximum=24_000_000)
    if not isinstance(value.get('source'),str) or not re.fullmatch('[0-9a-f]{64}', value['source']):
        raise ValueError('Invalid analysis source')
    duration = finite(value.get('duration'), 'duration', .000001, MAX_DURATION)
    rate = value.get('sample_rate')
    if type(rate) is not int or not 8000 <= rate <= 192000:
        raise ValueError('Invalid analysis sample rate')
    if type(value.get('samples')) is not int or abs(value['samples'] / rate - duration) > .5 / rate:
        raise ValueError('Analysis duration does not match sample count')
    step = finite(value.get('envelope_step'), 'envelope_step', .005, .1)
    env = value.get('envelope')
    if not isinstance(env, list) or len(env) != math.ceil(value['samples'] / round(step * rate)):
        raise ValueError('Invalid envelope length')
    for v in env:
        finite(v, 'envelope', -160, 80)
    lines = value.get('lines')
    if not isinstance(lines, list) or not lines or len(lines) > MAX_ITEMS:
        raise ValueError('Invalid editor lines')
    previous = 0.
    for line in lines:
        if not isinstance(line, dict) or type(line.get('speaker')) is not int or not 0 <= line['speaker'] < 16:
            raise ValueError('Invalid line speaker')
        start = finite(line.get('start'), 'line start', previous, duration)
        end = finite(line.get('end'), 'line end', start, duration)
        previous = end
        if not isinstance(line.get('text'), str) or len(line['text']) > 100000:
            raise ValueError('Invalid line text')
        if 'needs_alignment' in line and type(line['needs_alignment']) is not bool:
            raise ValueError('Invalid alignment provenance flag')
        tokens = line.get('tokens', [])
        if not isinstance(tokens, list) or len(tokens) > MAX_ITEMS:
            raise ValueError('Invalid word tokens')
        word_end = start
        for token in tokens:
            if not isinstance(token, dict) or not isinstance(token.get('text'), str) or len(token['text']) > 100000:
                raise ValueError('Invalid word token')
            if type(token.get('spoken', True)) is not bool:
                raise ValueError('Invalid spoken flag')
            if ('start' in token) != ('end' in token):
                raise ValueError('Word needs both start and end')
            if 'start' in token:
                ws = finite(token['start'], 'word start', max(start, word_end - .001), end)
                word_end = finite(token['end'], 'word end', ws, end)
    if value.get('speakers') != sorted({line['speaker'] for line in lines}) or type(value.get('aligned')) is not bool:
        raise ValueError('Invalid analysis speaker list or alignment flag')
    return value


def pauses(analysis, threshold_db):
    step, found, start = analysis['envelope_step'], [], None
    for index, value in enumerate([*analysis['envelope'], 80]):
        quiet = value < threshold_db
        if quiet and start is None:
            start = index
        elif not quiet and start is not None:
            end = min(index * step, analysis['duration'])
            if end - start * step >= MIN_PAUSE:
                found.append([start * step, end])
            start = None
    return found


def cut_pieces(duration, removed, inserted, line_starts):
    kept, position = [], 0.
    for a, b in sorted(removed):
        if a > position:
            kept.append([position, a])
        position = max(position, b)
    if position < duration:
        kept.append([position, duration])
    def line_at(t):
        return max(0, bisect.bisect_right(line_starts, t + 1e-8) - 1)
    pending, pieces = sorted(inserted), []
    for a, b in kept:
        points = sorted({t for t in line_starts if a < t < b} | {t for t, _ in pending if a < t < b})
        for x, y in zip([a, *points], [*points, b]):
            while pending and pending[0][0] <= x + 1e-8:
                pieces.append(dict(gap=pending.pop(0)[1], line=line_at(x)))
            pieces.append(dict(src=[x, y], line=line_at(x)))
    pieces += [dict(gap=seconds, line=line_at(t)) for t, seconds in pending]
    return pieces


def plan(analysis, edits):
    analysis = validate_analysis(analysis)
    duration, lines = analysis['duration'], analysis['lines']
    edits = parse_edits(edits, source=analysis['source'], duration=duration)
    boundaries = [line['end'] for line in lines[:-1]]
    removed, inserted, found, covered = [], [], [], set()
    for start, end in pauses(analysis, edits['threshold_db']):
        length, pid = end - start, f'{start:.2f}'
        boundary = next((b for b in boundaries if start - .05 <= b <= end + .05), None)
        kind = 'lead' if start == 0 else 'tail' if end >= duration else 'between' if boundary is not None else 'inside'
        action, to = 'kept', length
        if boundary is not None:
            covered.add(boundary)
        if pid in edits['keep']:
            action = 'kept by you'
        elif kind == 'lead' and edits['trim_start']:
            to = min(length, edits['lead'])
        elif kind == 'tail' and edits['trim_end']:
            to = min(length, edits['tail'])
        elif kind == 'between':
            to = min(max(length, edits['gap_min']), edits['gap_max'])
        elif kind == 'inside' and length > edits['pause_longer']:
            to = edits['pause_to']
        if to < length - .000001:
            action = 'shortened'
            removed.append([start, end-to] if kind == 'lead' else [start+to, end] if kind == 'tail' else [start+to/2, end-to/2])
        elif to > length + .000001:
            action = 'lengthened'
            inserted.append([(start+end)/2, to-length])
        found.append(dict(id=pid, start=start, end=end, kind=kind, action=action, to=to))
    for boundary in boundaries:
        if boundary not in covered and edits['gap_min']:
            inserted.append([boundary, edits['gap_min']])
    pieces = cut_pieces(duration, [*removed, *edits['cuts']], [*inserted, *edits['inserts']], [l['start'] for l in lines])
    # Quantize the *same* source endpoints to samples for Python and Web Audio.
    rate, total = analysis['sample_rate'], 0
    for piece in pieces:
        if 'src' in piece:
            a, b = [round(x*rate) for x in piece['src']]
            piece['src'] = [a/rate, b/rate]
            total += b-a
        else:
            count = round(piece['gap']*rate)
            piece['gap'] = count/rate
            total += count
    if total == 0:
        raise ValueError('The edits remove all audio; keep at least one source sample or inserted silence')
    if total > MAX_DURATION*rate:
        raise ValueError('Edited audio exceeds the duration limit')
    return dict(pieces=pieces, pauses=found, duration=total/rate, samples=total, source=analysis['source'])


def render(waveform, analysis, pieces, edits):
    analysis = validate_analysis(analysis)
    edits = parse_edits(edits, source=analysis['source'], duration=analysis['duration'])
    expected = plan(analysis, edits)['pieces']
    if pieces != expected:
        raise ValueError('Rendered pieces must match the validated editor plan')
    wave = waveform.detach().cpu()
    if not wave.is_floating_point() or wave.ndim not in (1, 2) or wave.shape[-1] != analysis['samples'] or not torch.isfinite(wave).all():
        raise ValueError('Audio does not match editor analysis')
    out, rate = [], analysis['sample_rate']
    for piece in pieces:
        if 'gap' in piece:
            out.append(wave.new_zeros((*wave.shape[:-1], round(piece['gap']*rate))))
        else:
            a, b = [round(v*rate) for v in piece['src']]
            audio = wave[..., a:b].clone()
            speaker = str(analysis['lines'][piece['line']]['speaker'])
            setting = edits['speakers'].get(speaker, {})
            if setting.get('mute', False):
                audio.zero_()
            elif setting.get('gain_db', 0):
                audio *= 10**(setting['gain_db']/20)
            out.append(audio)
    result = torch.cat(out, dim=-1) if out else wave[..., :0].clone()
    if edits['normalize'] and result.numel():
        peak = result.abs().max().item()
        if peak:
            result *= 10**(-1/20)/peak
    # No unconditional clamp/fades: the default is bit-exact identity.
    return result


def captions(analysis, pieces, edits):
    """Line and word captions mapped through actual cuts and inserted silence."""
    out, cursor = [], 0.
    for piece in pieces:
        if 'gap' in piece:
            cursor += piece['gap']
            continue
        a, b = piece['src']
        line = analysis['lines'][piece['line']]
        setting = edits['speakers'].get(str(line['speaker']), {})
        if not setting.get('mute'):
            words = []
            for token in line['tokens']:
                if 'start' in token and min(b,token['end']) > max(a,token['start']):
                    words.append(dict(text=token['text'],start=cursor+max(a,token['start'])-a,
                        end=cursor+min(b,token['end'])-a))
            visible_start=max(a,line['start'])
            visible_end=min(b,line['end'])
            if (words or not analysis['aligned']) and visible_end>visible_start:
                text=''
                for word in words:
                    unit=word['text']
                    if text and not re.search(r'[\u3400-\u9fff\u3040-\u30ff]',text[-1:]+unit[:1]):
                        text+=' '
                    text+=unit
                out.append(dict(speaker=line['speaker'],text=text if words else line['text'],
                    start=words[0]['start'] if words else cursor+visible_start-a,
                    end=words[-1]['end'] if words else cursor+visible_end-a,words=words))
        cursor += b-a
    return out


def subtitle_srt(items):
    def stamp(seconds):
        ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
        return f'{h:02}:{m:02}:{s:02},{ms:03}'
    visible=[c for c in items if c['end']>c['start'] and c['text'].strip()]
    return '\n\n'.join(f"{i}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{c['text']}"
        for i,c in enumerate(visible,1)) + ('\n' if visible else '')


def edited_session(session, index, analysis, edits):
    if type(index) is not int or not -len(session.takes) <= index < len(session.takes):
        raise ValueError('The requested take does not exist')
    take = session.takes[index]
    if fingerprint(take) != analysis['source']:
        raise ValueError('Editor source no longer matches the selected take')
    edits = parse_edits(edits, source=analysis['source'], duration=analysis['duration'])
    planned = plan(analysis, edits)
    # A new view is immutable and needs a distinct graph ID. Otherwise an old
    # rendered result can retain the same parent ID with different saved edits.
    viewed=replace(take,take_id=uuid.uuid4().hex,metadata={**take.metadata,
        'editor_parent_take_id':take.take_id,'editor_source_take_id':take.metadata.get('editor_source_take_id',take.take_id)})
    rebound=fingerprint(viewed)
    edits={**edits,'source':rebound}
    stored = dict(version=1, source=rebound, edits=edits,
        captions=captions(analysis, planned['pieces'], edits), duration=planned['duration'],
        source_lines=copy.deepcopy(analysis['lines']),aligned=analysis['aligned'])
    takes=list(session.takes)
    takes[index]=replace(viewed, metadata={**viewed.metadata,'audio_editor':stored})
    return replace(session, session_id=uuid.uuid4().hex, takes=tuple(takes))


def prepare(engine, take, *, aligner=None, language='Chinese'):
    if hasattr(engine,'decode_take'):
        audio,timeline=engine.decode_take(take,with_timeline=True)
    else:
        audio,timeline=engine.decode(take.codes),None
    wave=audio['waveform'][0].detach().cpu()
    rate=audio['sample_rate'];duration=wave.shape[-1]/rate
    lines=source_lines(take,duration,timeline)
    aligned=None
    if aligner:
        from .alignment import align_audio
        # One model load for the full take, preserving the low-memory path.
        transcript=(take.transcript if 'render_infill' in take.metadata else '\n'.join(l['text'] for l in lines))
        aligned=align_audio(aligner,wave,rate,transcript,language)
        if not aligned:
            raise ValueError('Word alignment returned no words; inspect the source audio and transcript')
    analysis=analyze(wave,rate,lines,source=fingerprint(take),aligned_words=aligned)
    stored=take.metadata.get('audio_editor',{})
    if not aligner and stored.get('source')==analysis['source'] and stored.get('source_lines'):
        analysis['lines']=copy.deepcopy(stored['source_lines'])
        analysis['aligned']=stored.get('aligned',False)
        validate_analysis(analysis)
    edits=stored.get('edits',{}) if stored.get('source')==analysis['source'] else {}
    return audio,analysis,parse_edits(edits,source=analysis['source'],duration=duration)


def output_duration(take):
    """Duration of the output view without loading a model or decoding audio."""
    from .native.codec import FRAME_LENGTH, SAMPLE_RATE
    stored = take.metadata.get('audio_editor', {})
    if not isinstance(stored, dict):
        raise ValueError('Invalid saved audio editor metadata')
    if stored and stored.get('source') == fingerprint(take):
        return finite(stored['duration'], 'Edited audio duration')
    if 'render_infill' in take.metadata:
        from .render_infill import validate_take
        validate_take(take)
        return take.blobs[take.metadata['render_infill']['pcm_blob']].shape[-1] / SAMPLE_RATE
    samples = take.codes.shape[-1] * FRAME_LENGTH
    if 'scene_chunking' in take.metadata or take.metadata.get('decode_sources'):
        gap = finite(take.metadata.get('gap_seconds', .3), 'Scene gap', 0, 10)
        samples += max(0, len(take.segments()) - 1) * round(gap * SAMPLE_RATE)
    return samples / SAMPLE_RATE


def decode_output(engine,take):
    """Apply saved post-production only at output; never alter model history."""
    if not take.metadata.get('audio_editor'):
        return engine.decode_take(take) if hasattr(engine,'decode_take') else engine.decode(take.codes)
    audio,analysis,edits=prepare(engine,take)
    planned=plan(analysis,edits)
    return {**audio,'waveform':render(audio['waveform'][0],analysis,planned['pieces'],edits)[None]}
