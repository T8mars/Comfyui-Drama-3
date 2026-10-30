"""Budgeted scene requests and their lossless source-to-playback map.

The composite has one segment per original line. Only the archived *actual*
source takes define Codec contexts; presentation codes are never decoded as
one stream. This module does not load models.
"""
from dataclasses import replace
from array import array
from bisect import bisect_right
import contextlib
import json
import math
import re
import unicodedata

import torch

from .state import Session, Reference, Take, validate_frames
from .scene_cast import prepare_scene_cast
from .native.chunking import PROTECTED_SPAN, SENTENCE_ENDS, CLAUSE_MARKS
from .native.prompt import WHITE_SPACE
from .native.codec import FRAME_LENGTH, SAMPLE_RATE

# Source-map layout remains v1; this version identifies the text planner.
# Keep the previous planner to validate/decode already saved source maps.
VERSION = 3
MAX_UNITS = 10000
SEP = '<|audio_separator|>'
JOIN = '\n' + SEP + '\n'
TAG = re.compile(r'<\|speaker:([0-9]+)\|>')
MARKUP = re.compile(r'<\|[^|>]*\|>|\[[^\[\]]*\]|</?[A-Za-z][^>]*>', re.S)
ATOMIC_SPAN = re.compile(PROTECTED_SPAN.pattern + r'|</?[A-Za-z][^>]*>', re.S)


def options(enabled, budget):
    if type(enabled) is not bool:
        raise ValueError('split_long_text must be a boolean')
    if type(budget) is not int or not 32 <= budget <= 3000:
        raise ValueError('Scene chunk_bytes must be an integer from 32 to 3000')


def _creative_context_options(history_limit, fixed_reference_anchor):
    if type(history_limit) is not int or history_limit not in (2, 3):
        raise ValueError('Scene initial history_limit must be 2 or 3')
    if type(fixed_reference_anchor) is not bool:
        raise ValueError('Scene fixed_reference_anchor must be a boolean')


def body_of(line):
    stripped=line.lstrip(WHITE_SPACE)
    match = TAG.match(stripped)
    if match is None:
        return 0, line
    offset = len(line) - len(stripped) + match.end()
    # Casting inserts one structural space. Remaining whitespace is content.
    if line[offset:offset + 1] == ' ':
        offset += 1
    return int(match.group(1)), line[offset:]


def tagged(body, actor):
    return f'<|speaker:{actor}|> {body}'


def byte_offsets(text):
    offsets=array('I',[0])
    for char in text:
        end=offsets[-1]+len(char.encode('utf-8'))
        if end>MAX_UNITS*3000:raise ValueError('Scene text exceeds the maximum chunked script size')
        offsets.append(end)
    return offsets


def feasible_parts(body,atoms,budget):
    """Linear fallback on protected boundaries; every unit contains content.

    Feasible suffix endpoints are indexed once. Greedy preferences are kept
    whenever they work; this runs only when they strand whitespace.
    """
    count=len(atoms);prefix=[0]
    for start,end in atoms:prefix.append(prefix[-1]+len(body[start:end].encode('utf-8')))
    next_text=[count]*(count+1)
    for index in range(count-1,-1,-1):
        start,end=atoms[index]
        next_text[index]=index if body[start:end].strip(WHITE_SPACE) else next_text[index+1]
    limit=[0]*count;right=0
    for left in range(count):
        while right<count and prefix[right+1]-prefix[left]<=budget:right+=1
        limit[left]=right
    reachable=bytearray(count+1);reachable[count]=1
    next_good=[count+1]*(count+2);next_good[count]=count
    for left in range(count-1,-1,-1):
        first=next_text[left]
        if first<count and next_good[first+1]<=limit[left]:reachable[left]=1
        next_good[left]=left if reachable[left] else next_good[left+1]
    if not reachable[0]:
        raise ValueError('Scene whitespace/protected spans cannot fit nonempty chunks at this byte budget')
    previous=[-1]*(count+1);last=-1
    for index,good in enumerate(reachable):
        if good:last=index
        previous[index]=last
    result=[];left=0
    while left<count:
        end=previous[limit[left]]
        if end<=next_text[left]:raise ValueError('No feasible nonempty scene chunk')
        result.append((atoms[left][0],atoms[end-1][1]));left=end
    return result


def split_body_v1(body, budget):
    """Exact character ranges; protected spans are indivisible, never dropped."""
    atoms = []
    cursor = 0
    for match in PROTECTED_SPAN.finditer(body):
        atoms.extend((i, i + 1) for i in range(cursor, match.start()))
        atoms.append((match.start(), match.end()))
        cursor = match.end()
    atoms.extend((i, i + 1) for i in range(cursor, len(body)))
    if not body.strip(WHITE_SPACE):
        raise ValueError('Scene line contains no text')
    for start, end in atoms:
        if len(body[start:end].encode('utf-8')) > budget:
            raise ValueError('A protected scene span cannot fit chunk_bytes including speaker/control overhead')
    result = []
    index = 0
    while index < len(atoms):
        end_index = index
        size = 0
        preferred = {}
        while end_index < len(atoms):
            a, b = atoms[end_index]
            added = len(body[a:b].encode('utf-8'))
            if size + added > budget:
                break
            size += added
            end_index += 1
            # Prefer full sentences, then clauses, then whitespace. Decimals
            # are not sentence boundaries. Every byte remains in some unit.
            char = body[b - 1]
            decimal = char == '.' and b > 1 and b < len(body) and body[b - 2].isdigit() and body[b].isdigit()
            rank = 0 if char in SENTENCE_ENDS and not decimal else 1 if char in CLAUSE_MARKS else 2 if char.isspace() else None
            if rank is not None and body[atoms[index][0]:b].strip(WHITE_SPACE):
                preferred[rank] = end_index
        if end_index < len(atoms) and preferred:
            end_index = preferred[min(preferred)]
        if end_index == index:
            raise ValueError('Scene chunk_bytes cannot fit the next UTF-8 character')
        result.append((atoms[index][0], atoms[end_index - 1][1]))
        index = end_index
    # A final whitespace-only range must join its predecessor; if it does not
    # fit, shift its leading text into that final unit instead of losing it.
    if result and not body[slice(*result[-1])].strip(WHITE_SPACE):
        a, b = result.pop()
        if not result:
            raise ValueError('Scene line contains no text')
        prior_a, _ = result[-1]
        if len(body[prior_a:b].encode('utf-8')) <= budget:
            result[-1] = (prior_a, b)
        else:
            movable=next((start for start,end in reversed(atoms) if prior_a<=start and end<=a
                and body[start:end].strip(WHITE_SPACE)),None)
            if (movable is None or not body[prior_a:movable].strip(WHITE_SPACE) or
                    len(body[movable:b].encode('utf-8'))>budget):
                return feasible_parts(body,atoms,budget)
            else:
                result[-1]=(prior_a,movable)
                result.append((movable,b))
    if any(not body[start:end].strip(WHITE_SPACE) for start,end in result):
        return feasible_parts(body,atoms,budget)
    return result


def _body_content(text):
    """Actual content excludes direction/control syntax and punctuation.

    XML scope contents still count. Symbols include emoji; whitespace, markup,
    punctuation and control characters cannot stand in for a spoken unit.
    """
    return any(char not in WHITE_SPACE and unicodedata.category(char)[0] not in 'PCZ'
        for char in (text if len(text)==1 else MARKUP.sub('',text)))


def _content_feasible(atoms,prefix,content,limits):
    """Linear suffix reachability over protected boundaries and content flags."""
    count=len(atoms)
    next_text=[count]*(count+1)
    for index in range(count-1,-1,-1):
        next_text[index]=index if content[index] else next_text[index+1]
    reachable=bytearray(count+1);reachable[count]=1
    next_good=[count+1]*(count+2);next_good[count]=count
    for left in range(count-1,-1,-1):
        first=next_text[left]
        if first<count and next_good[first+1]<=limits[left]:reachable[left]=1
        next_good[left]=left if reachable[left] else next_good[left+1]
    if not reachable[0]:
        raise ValueError('Scene directions/protected spans cannot fit chunks containing text; increase chunk_bytes')
    previous=[-1]*(count+1);last=-1
    for index,good in enumerate(reachable):
        if good:last=index
        previous[index]=last
    result=[];left=0
    while left<count:
        end=previous[limits[left]]
        if end<=next_text[left]:raise ValueError('No feasible scene chunk containing text')
        result.append((atoms[left][0],atoms[end-1][1]));left=end
    return result


def split_body_v2(body, budget):
    """V2 lossless partition: never manufacture instruction-only speech turns.

    An original pure effect line stays valid. Otherwise each produced unit must
    contain actual content. Prefix counters and indexed breakpoints avoid
    rescanning progressively longer substrings; fallback is linear in atoms.
    """
    if not body.strip(WHITE_SPACE):raise ValueError('Scene line contains no text')
    atoms=[];cursor=0
    for match in ATOMIC_SPAN.finditer(body):
        atoms.extend((index,index+1) for index in range(cursor,match.start()))
        atoms.append((match.start(),match.end()));cursor=match.end()
    atoms.extend((index,index+1) for index in range(cursor,len(body)))
    prefix=[0];content=[];ink=[];marks=[[],[],[]]
    for index,(start,end) in enumerate(atoms):
        piece=body[start:end];size=len(piece.encode('utf-8'))
        if size>budget:
            raise ValueError('A protected scene span cannot fit chunk_bytes including speaker/control overhead')
        prefix.append(prefix[-1]+size);content.append(_body_content(piece));ink.append(bool(piece.strip(WHITE_SPACE)))
        char=piece[-1]
        decimal=char=='.' and end>1 and end<len(body) and body[end-2].isdigit() and body[end].isdigit()
        rank=0 if char in SENTENCE_ENDS and not decimal else 1 if char in CLAUSE_MARKS else 2 if char.isspace() else None
        if rank is not None:marks[rank].append(index+1)
    if not any(content):content=ink  # User explicitly requested a pure effect original line.
    totals=[0]
    for flag in content:totals.append(totals[-1]+flag)
    count=len(atoms);limits=[0]*count;right=0
    for left in range(count):
        while right<count and prefix[right+1]-prefix[left]<=budget:right+=1
        limits[left]=right
    result=[];left=0
    while left<count:
        end=limits[left]
        if end<count:
            for positions in marks:
                candidate=bisect_right(positions,end)-1
                if candidate>=0:
                    cut=positions[candidate]
                    if cut>left and totals[cut]>totals[left]:end=cut;break
        if end<=left or totals[end]==totals[left]:
            return _content_feasible(atoms,prefix,content,limits)
        result.append((atoms[left][0],atoms[end-1][1]));left=end
    return result


def _tail_natural_boundary(body, cut):
    left, right = body[cut - 1:cut], body[cut:cut + 1]
    decimal = left == '.' and cut > 1 and cut < len(body) and body[cut - 2].isdigit() and right.isdigit()
    return bool((left and left in WHITE_SPACE) or (right and right in WHITE_SPACE)
        or (left in SENTENCE_ENDS | CLAUSE_MARKS and not decimal))


def _tail_safe_boundary(body, cut):
    """Do not introduce a cut inside combining/ZWJ/variation/flag sequences."""
    if not 0 < cut < len(body):return False
    left, right = body[cut - 1], body[cut]
    if left == '\u200d' or right == '\u200d':return False
    if unicodedata.category(right).startswith('M') or 0xFE00 <= ord(right) <= 0xFE0F:return False
    if 0x1F1E6 <= ord(left) <= 0x1F1FF and 0x1F1E6 <= ord(right) <= 0x1F1FF:return False
    return True


def _rebalance_short_tail(body, parts, budget):
    """Move only a final unnatural cut leaving <=2 actual body codepoints.

    Input is the already valid V2 partition. Re-evaluate just its final two
    units (at most twice the byte budget), preserving every byte, protected
    atom and unit count. No improving safe cut means the exact V2 result.
    """
    if len(parts) < 2:return parts
    start, old_cut = parts[-2]
    end = parts[-1][1]
    if _tail_natural_boundary(body, old_cut):return parts
    tail = body[start:end]
    mask = [char not in WHITE_SPACE and unicodedata.category(char)[0] not in 'PCZ' for char in tail]
    for match in MARKUP.finditer(tail):
        mask[match.start():match.end()] = [False] * (match.end() - match.start())
    content, byte_prefix = [0], [0]
    for char, flag in zip(tail, mask):
        content.append(content[-1] + int(flag))
        byte_prefix.append(byte_prefix[-1] + len(char.encode('utf-8')))
    local_cut = old_cut - start
    old_left, old_right = content[local_cut], content[-1] - content[local_cut]
    if not old_left or not 0 < old_right <= 2:return parts
    legal = bytearray(b'\1') * (len(tail) + 1)
    for match in ATOMIC_SPAN.finditer(tail):
        legal[match.start() + 1:match.end()] = b'\0' * (match.end() - match.start() - 1)
    best = None
    for offset in range(1, len(tail)):
        cut = start + offset
        if not legal[offset] or not _tail_safe_boundary(body, cut):continue
        left, right = content[offset], content[-1] - content[offset]
        if (not left or not right or byte_prefix[offset] > budget
                or byte_prefix[-1] - byte_prefix[offset] > budget):continue
        score = (abs(left - right), 0 if _tail_natural_boundary(body, cut) else 1,
                 -left, abs(cut - old_cut), cut)
        if best is None or score < best[0]:best = (score, cut)
    if best is None or best[0][:3] >= (abs(old_left - old_right), 1, -old_left):return parts
    cut = best[1]
    return [*parts[:-2], (start, cut), (cut, end)]


def split_body(body, budget):
    """V3 preserves V2 feasibility and balances eligible short final speech."""
    return _rebalance_short_tail(body, split_body_v2(body, budget), budget)


def plan_scene(lines, references=(), takes=(), *, mode='together', chunk_bytes=300, chunking_version=VERSION,
               request_direction=''):
    from .dialogue import MODES
    from .native.prompt import check_text
    from .prompts import direction_tags, directed_text
    request_direction = direction_tags(request_direction)
    direction_bytes = len(request_direction.encode('utf-8')) + 1 if request_direction else 0
    options(True, chunk_bytes)
    if type(chunking_version) is not int or chunking_version not in (1,2,VERSION):
        raise ValueError('Unsupported scene_chunking.version')
    if mode not in MODES:
        raise ValueError('Unknown dialogue generation mode')
    lines = tuple(lines)
    if not lines or len(lines) > MAX_UNITS:
        raise ValueError('Scene must contain 1..10000 original lines')
    cast, _, mapping = prepare_scene_cast(lines, references, takes, force=True)
    units = []
    normalized = []
    for line_index, line in enumerate(cast):
        check_text(line, 'scene line')
        actor, body = body_of(line)
        if not 0 <= actor < 16 or TAG.search(body):
            raise ValueError('Use exactly one leading speaker per original scene line')
        normalized.append(tagged(body, actor))
        offsets=byte_offsets(body)
        overhead = len(tagged('', actor).encode('utf-8'))
        split={1:split_body_v1,2:split_body_v2,VERSION:split_body}[chunking_version]
        available = chunk_bytes - overhead - direction_bytes
        if available < 1:
            raise ValueError('Direction controls cannot fit chunk_bytes including speaker overhead; increase chunk_bytes')
        for chunk_index, (start, end) in enumerate(split(body, available)):
            units.append(dict(unit_index=len(units), line_index=line_index, chunk_index=chunk_index,
                speaker_id=actor, text_start=start, text_end=end,
                byte_start=offsets[start], byte_end=offsets[end],
                text=body[start:end]))
            if not body[start:end].strip(WHITE_SPACE):
                raise ValueError('Scene chunk_bytes would produce a whitespace-only request; increase the budget or remove excess whitespace')
            if len(units) > MAX_UNITS:
                raise ValueError('Scene exceeds 10000 text chunks')
    actors = [body_of(line)[0] for line in normalized]
    if mode == 'per_speaker':
        ordered = [unit for actor in dict.fromkeys(actors) for unit in units if unit['speaker_id'] == actor]
    else:
        ordered = units
    requests = []
    batch = []
    def flush():
        if batch:
            requests.append(list(batch)); batch.clear()
    for unit in ordered:
        # Per-line modes intentionally use one unit/request; per-speaker may
        # pack several original lines, but never crosses a role group.
        local_actor = unit['speaker_id'] if mode == 'together' else 0
        candidate = [*batch, unit]
        text = JOIN.join(directed_text(tagged(u['text'], u['speaker_id'] if mode == 'together' else 0), request_direction) for u in candidate)
        if batch and (mode in ('per_line', 'per_line_context') or
                mode == 'per_speaker' and batch[0]['speaker_id'] != unit['speaker_id'] or
                len(text.encode('utf-8')) > chunk_bytes):
            flush()
        if len(directed_text(tagged(unit['text'], local_actor), request_direction).encode('utf-8')) > chunk_bytes:
            raise ValueError('Scene request exceeds chunk_bytes')
        batch.append(unit)
    flush()
    return dict(cast_lines=normalized, speaker_map=mapping, line_speakers=actors, units=units,
                requests=requests, mode=mode, chunk_bytes=chunk_bytes)


def source_take(raw):
    """Restore an already validated source; callers validate untrusted maps first."""
    return Take(raw['prompt_text'], raw['transcript'], torch.tensor(raw['frames'], dtype=torch.long),
        raw['semantic_start'], raw['semantic_end'], raw['separator_id'], raw['metadata'], raw['take_id'])


def validate_scene(take):
    """Strict archive/render boundary. Returns source Takes in generation order."""
    policy = take.metadata.get('scene_chunking')
    if policy is None:
        return None
    try:
        json.dumps(take.metadata,allow_nan=False)
    except (ValueError,TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('Scene source map must contain finite JSON data') from exc
    if not isinstance(policy, dict) or type(policy.get('version')) is not int or policy['version'] not in (1,2,VERSION):
        raise ValueError('Unsupported scene_chunking.version')
    options(policy.get('enabled'), policy.get('chunk_bytes'))
    _creative_context_options(policy.get('history_limit', 2), policy.get('fixed_reference_anchor', False))
    if policy['enabled'] is not True:
        raise ValueError('A versioned scene source map must be enabled')
    from .dialogue import MODES
    if take.metadata.get('task') != 'scene' or take.metadata.get('generation_mode') not in MODES:
        raise ValueError('Invalid chunked scene task/mode')
    original = take.metadata.get('script_lines')
    cast = take.metadata.get('scene_cast_lines')
    actors = take.metadata.get('line_speakers')
    units = take.metadata.get('scene_units')
    raws = take.metadata.get('scene_sources')
    if (not isinstance(original, list) or not original or len(original) > MAX_UNITS or
            not all(isinstance(line, str) for line in original) or
            not isinstance(cast, list) or len(cast) != len(original) or not all(isinstance(line, str) for line in cast) or
            not isinstance(actors, list) or len(actors) != len(original) or
            any(type(actor) is not int or not 0 <= actor < 16 for actor in actors) or
            not isinstance(units, list) or not 0 < len(units) <= MAX_UNITS or
            not isinstance(raws, list) or not 0 < len(raws) <= len(units)):
        raise ValueError('Invalid scene source-map shape')
    if policy.get('fixed_reference_anchor') and (take.metadata['generation_mode'] != 'per_line' or len(set(actors)) != 1):
        raise ValueError('Fixed reference anchoring requires one actor in per_line mode')
    # Recast using the persisted map, not a new first-seen allocation.
    from .scene_cast import _render_line
    mapping = take.metadata.get('scene_speaker_map', {})
    from .scene_cast import validate_scene_speaker_map
    validate_scene_speaker_map(mapping)
    from .native.prompt import check_text
    for line in original:
        check_text(line,'scene source-map script')
    rebuilt = [tagged(body_of(_render_line(line, mapping))[1], body_of(_render_line(line, mapping))[0]) for line in original]
    if cast != rebuilt or actors != [body_of(line)[0] for line in cast]:
        raise ValueError('Scene cast/script identity differs from its source map')
    request_direction = policy.get('request_direction', '')
    from .prompts import directed_text, direction_tags
    if not isinstance(request_direction, str) or direction_tags(request_direction) != request_direction:
        raise ValueError('Invalid canonical scene request direction')
    replanned=plan_scene(cast,mode=take.metadata['generation_mode'],chunk_bytes=policy['chunk_bytes'],
        chunking_version=policy['version'],request_direction=request_direction)
    expected_requests=[[unit['unit_index'] for unit in request] for request in replanned['requests']]
    if len(expected_requests)!=len(raws) or len(replanned['units'])!=len(units):
        raise ValueError('Scene source requests differ from their persisted chunk policy')
    if policy['version'] in (2,VERSION) and any(not isinstance(unit,dict) or
            any(unit.get(key)!=value for key,value in expected.items())
            for unit,expected in zip(units,replanned['units'])):
        raise ValueError('Scene units differ from their persisted text planner')
    validate_frames(take.frames)
    line_codes = take.segments()
    if len(line_codes) != len(original) or any(c.shape[-1] == 0 for c in line_codes):
        raise ValueError('Scene composite must retain one nonempty segment per original line')
    if take.metadata.get('expected_segments') != len(original) or take.metadata.get('actual_segments') != len(original):
        raise ValueError('Scene original-line segment counts differ')
    sources = []
    by_id = {}
    seen_units = set()
    for request_index,raw in enumerate(raws):
        if not isinstance(raw, dict):
            raise ValueError('Invalid scene source')
        identifier = raw.get('take_id')
        frames = raw.get('frames')
        indexes = raw.get('unit_indexes')
        if (not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier) or
                identifier in by_id or identifier == take.take_id or
                not isinstance(frames, list) or not 0 < len(frames) <= 1000000 or
                any(not isinstance(row, list) or len(row) != 11 or any(type(v) is not int for v in row) for row in frames) or
                not isinstance(indexes, list) or not indexes or any(type(i) is not int or not 0 <= i < len(units) for i in indexes) or
                not all(isinstance(raw.get(key), str) for key in ('prompt_text', 'transcript')) or
                not isinstance(raw.get('metadata'), dict) or 'scene_chunking' in raw['metadata'] or
                raw.get('request_index') != request_index or type(raw.get('request_index')) is not int or
                type(raw.get('request_seed')) is not int or not 0 <= raw['request_seed'] < 2**64 or
                any(type(raw.get(key)) is not int or raw[key] != getattr(take, key) for key in ('semantic_start', 'semantic_end', 'separator_id'))):
            raise ValueError('Invalid scene source frames/identity/units')
        if any(index in seen_units for index in indexes) or len(set(indexes)) != len(indexes):
            raise ValueError('Scene source unit coverage overlaps')
        if indexes!=expected_requests[request_index]:
            raise ValueError('Scene source request order differs from its mode/chunk policy')
        seen_units.update(indexes)
        actor = raw.get('speaker')
        if actor is not None and (type(actor) is not int or not 0 <= actor < 16):
            raise ValueError('Invalid scene source speaker')
        if (take.metadata['generation_mode']=='together') != (actor is None):
            raise ValueError('Scene source local/global role policy differs from its mode')
        restored = source_take(raw)
        validate_frames(restored.frames)
        segments = restored.segments()
        if len(segments) != len(indexes) or any(c.shape[-1] == 0 for c in segments):
            raise ValueError('Scene source segment coverage differs')
        try:
            request = JOIN.join(directed_text(tagged(units[i]['text'], units[i]['speaker_id'] if actor is None else 0), request_direction) for i in indexes)
        except (KeyError, TypeError) as exc:
            raise ValueError('Invalid scene source unit text') from exc
        if restored.prompt_text != request or len(request.encode('utf-8')) > policy['chunk_bytes']:
            raise ValueError('Scene source prompt differs from its budgeted request')
        from .engine import Sampling
        try:
            recorded=Sampling(**raw['metadata']['sampling']);recorded.validate()
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('Invalid scene source sampling record') from exc
        if (type(take.metadata.get('seed')) is not int or not 0 <= take.metadata['seed'] < 2**64 or
                raw['request_seed'] != (take.metadata['seed']+request_index)%2**64 or
                raw['metadata'].get('requested_seed',raw['request_seed']) != raw['request_seed'] or
                raw['metadata'].get('seed') != recorded.seed or raw['metadata'].get('length_limit_reached') is not False):
            raise ValueError('Scene source seed/sampling/completion differs from its request')
        if actor is not None and any(units[i].get('speaker_id') != actor for i in indexes):
            raise ValueError('Scene source role differs from its unit roles')
        by_id[identifier] = (raw, restored, segments)
        sources.append(restored)
    if seen_units != set(range(len(units))):
        raise ValueError('Scene source units do not cover the complete scene')
    text_cursors = [0] * len(original)
    code_cursors = [0] * len(original)
    chunk_cursors = [0] * len(original)
    previous_line = 0
    bodies=[body_of(line)[1] for line in cast]
    offsets=[byte_offsets(body) for body in bodies]
    for index, unit in enumerate(units):
        integer_fields = ('unit_index', 'line_index', 'chunk_index', 'speaker_id', 'text_start', 'text_end',
            'byte_start', 'byte_end', 'line_frame_start', 'line_frame_end', 'source_segment', 'source_frame_start', 'source_frame_end')
        if not isinstance(unit, dict) or any(type(unit.get(key)) is not int for key in integer_fields):
            raise ValueError('Invalid scene unit integer fields')
        line = unit['line_index']
        if (unit['unit_index'] != index or not 0 <= line < len(original) or line < previous_line or
                unit['speaker_id'] != actors[line] or unit['chunk_index'] != chunk_cursors[line]):
            raise ValueError('Invalid scene unit line/chunk/role order')
        previous_line = line
        body = bodies[line]
        start, end = unit['text_start'], unit['text_end']
        if (start != text_cursors[line] or not start < end <= len(body) or
                unit.get('text') != body[start:end] or
                unit['byte_start'] != offsets[line][start] or unit['byte_end'] != offsets[line][end]):
            raise ValueError('Scene unit text coverage differs from original cast text')
        identifier = unit.get('source_take_id')
        if not isinstance(identifier,str) or identifier not in by_id:
            raise ValueError('Scene unit references an unknown source')
        raw, restored, segments = by_id[identifier]
        segment_index = unit['source_segment']
        if not 0 <= segment_index < len(segments) or raw['unit_indexes'][segment_index] != index:
            raise ValueError('Scene source segment/unit order differs')
        codes = segments[segment_index]
        start_frame, end_frame = unit['line_frame_start'], unit['line_frame_end']
        if (unit['source_frame_start'] != 0 or unit['source_frame_end'] != codes.shape[-1] or
                start_frame != code_cursors[line] or end_frame != start_frame + codes.shape[-1] or
                not torch.equal(line_codes[line][:, start_frame:end_frame], codes)):
            raise ValueError('Scene source/composite frame coverage differs')
        text_cursors[line] = end
        code_cursors[line] = end_frame
        chunk_cursors[line] += 1
    if any(text_cursors[i] != len(body_of(cast[i])[1]) or code_cursors[i] != line_codes[i].shape[-1] for i in range(len(original))):
        raise ValueError('Scene source map has missing text or audio')
    gap = take.metadata.get('gap_seconds')
    if type(gap) not in (int, float) or not math.isfinite(gap) or not 0 <= gap <= 10:
        raise ValueError('Invalid scene gap')
    return sources


def playback_plan(take):
    """Validated real source takes plus frame slices, in presentation order.

    Each source is decoded once in full. ``source_start/end`` are Codec-frame
    offsets into that decoded source (control rows excluded); line_* offsets
    address the composite's original line. Gaps only occur between lines.
    """
    restored = validate_scene(take)
    if restored is None:
        raise ValueError('Playback plan requires a versioned chunked scene')
    sources = []
    for raw, source in zip(take.metadata['scene_sources'], restored):
        cursor = 0
        entries = []
        for index, codes in zip(raw['unit_indexes'], source.segments()):
            unit = take.metadata['scene_units'][index]
            entries.append(dict(unit, source_start=cursor, source_end=cursor + codes.shape[-1]))
            cursor += codes.shape[-1]
        sources.append(dict(take=source, units=entries))
    return dict(version=take.metadata['scene_chunking']['version'], sources=sources, units=take.metadata['scene_units'],
        script_lines=take.metadata['script_lines'], line_speakers=take.metadata['line_speakers'],
        gap_samples=round(take.metadata['gap_seconds'] * SAMPLE_RATE))


def decode_scene(engine, take):
    from .prompts import spoken_text
    plan = playback_plan(take)
    waves = {}
    for source in plan['sources']:
        codes = source['take'].codes
        audio = engine.decode(codes)
        wave = audio['waveform'][0]
        if audio['sample_rate'] != SAMPLE_RATE or wave.shape[-1] != codes.shape[-1] * FRAME_LENGTH:
            raise ValueError('Codec duration differs from the actual scene source')
        for unit in source['units']:
            waves[unit['unit_index']] = wave[:, unit['source_start'] * FRAME_LENGTH:unit['source_end'] * FRAME_LENGTH]
    output = []
    timeline = []
    cursor = 0
    for line, text in enumerate(plan['script_lines']):
        pieces = [waves[unit['unit_index']] for unit in plan['units'] if unit['line_index'] == line]
        wave = torch.cat(pieces, dim=-1)
        if line:
            output.append(wave.new_zeros((wave.shape[0], plan['gap_samples'])))
            cursor += plan['gap_samples']
        start = cursor / SAMPLE_RATE
        output.append(wave)
        cursor += wave.shape[-1]
        timeline.append(dict(line=line, speaker=plan['line_speakers'][line], text=spoken_text(text, scene=True),
            start=start, end=cursor / SAMPLE_RATE))
    return dict(waveform=torch.cat(output, dim=-1)[None], sample_rate=SAMPLE_RATE), timeline


def source_history(take, actor=None):
    """Actual acoustic turns; global tags for general history, local 0 per role."""
    from .dialogue import audio_frames
    validate_scene(take)
    result = []
    for raw in take.metadata['scene_sources']:
        restored = source_take(raw)
        local_actor = raw['speaker']
        if actor is None:
            text = TAG.sub(lambda match: f'<|speaker:{local_actor}|>' if local_actor is not None else match.group(0), restored.prompt_text)
            metadata = {**restored.metadata}
            if metadata.get('task') == 'scene':
                metadata['script_lines'] = text.split(JOIN)
            result.append(replace(restored, prompt_text=text, metadata=metadata))
        elif local_actor == actor:
            result.append(restored)
        elif local_actor is None:
            for unit_index, codes in zip(raw['unit_indexes'], restored.segments()):
                unit = take.metadata['scene_units'][unit_index]
                if unit['speaker_id'] == actor:
                    result.append(Take(tagged(unit['text'], 0), unit['text'], audio_frames(codes, restored),
                        restored.semantic_start, restored.semantic_end, restored.separator_id,
                        metadata={'scene_source_speaker': actor, 'source_take_id': restored.take_id}))
    return result


def history_view(takes):
    """Shared by all prompt builders, including ordinary TTS after a scene."""
    takes=tuple(takes)
    if not any('scene_chunking' in take.metadata or 'render_infill' in take.metadata or take.metadata.get('history_eligible') is False for take in takes):
        return takes
    from .render_infill import history_take
    expanded=[]
    for take in takes:
        if 'render_infill' in take.metadata:
            from .state import validate_session
            validate_session(Session(takes=(take,)))
            expanded.append(history_take(take))
        elif 'scene_chunking' in take.metadata:expanded.extend(source_history(take))
        elif take.metadata.get('history_eligible') is not False:expanded.append(take)
    return tuple(expanded if len(expanded) < 3 else (expanded[0], expanded[-1]))


def chunked_scene(engine, lines, scene, characters, references, sampling, progress, session, note, on_codes, mode, chunk_bytes,
                  *, history_limit=2, fixed_reference_anchor=False, on_observed_codes=None, system_context=''):
    """Generate the existing source-aware composite with optional creative views.

    The default is the unchanged Scene policy. Creative callers may retain up
    to three prepared acoustic turns in the first request only. Continuations
    use first/latest (at most two), or one fixed reference with empty history.
    Fixed anchoring accepts a single actor in per_line mode; mixed audio never
    becomes a relabelled single-actor reference.
    """
    from .dialogue import own_history, recast, audio_frames
    from .prompts import spoken_text, direction_tags, directed_text
    _creative_context_options(history_limit, fixed_reference_anchor)
    if on_observed_codes is not None and not callable(on_observed_codes):
        raise ValueError('Scene on_observed_codes must be a callable or None')
    session = session or Session(tuple(references or ()))
    if references is not None:
        session = replace(session, references=tuple(references))
    # Isolated voices already have real local reference/history tags. Only
    # actual user context contributes model language; no invented role mapping.
    request_direction = direction_tags(session.direction, note)
    plan = plan_scene(lines, session.references, session.takes, mode=mode, chunk_bytes=chunk_bytes,
                      request_direction=request_direction)
    if fixed_reference_anchor and (mode != 'per_line' or len(set(plan['line_speakers'])) != 1):
        raise ValueError('Fixed reference anchoring requires one actor in per_line mode')
    anchor_actor = plan['line_speakers'][0]
    fixed_references = tuple(ref for ref in session.references if body_of(ref.transcript)[0] == anchor_actor) if fixed_reference_anchor else ()
    anchor_source = None
    histories = own_history(session)
    global_history = list(history_view(session.takes))
    units = [dict(unit) for unit in plan['units']]
    sources = []
    line_codes = [[] for _ in lines]
    line_frames = [0 for _ in lines]
    generated_codes = {}
    active_line = None
    previous_actor = None
    previous_source = None
    line_other_context = None
    line_other_actor = None
    hold = engine.generation_session() if hasattr(engine, 'generation_session') else contextlib.nullcontext()
    with hold:
        for ordinal, request in enumerate(plan['requests']):
            actor = request[0]['speaker_id']
            together = mode == 'together'
            if mode == 'per_line_context' and request[0]['line_index'] != active_line:
                active_line = request[0]['line_index']
                line_other_context = (replace(previous_source,prompt_text=recast(previous_source.prompt_text,1))
                    if previous_source is not None and previous_actor != actor else None)
                line_other_actor=previous_actor if line_other_context is not None else None
            history = global_history if together else histories.get(actor, [])
            initial_limit = history_limit if ordinal == 0 else 2
            history = history if len(history) <= initial_limit else [history[0], history[-1]]
            if mode == 'per_line_context' and line_other_context is not None:
                history = [*history, line_other_context]
            refs = session.references if together else tuple(replace(ref, transcript=recast(ref.transcript))
                for ref in session.references if body_of(ref.transcript)[0] == actor)
            if fixed_reference_anchor and ordinal:
                if fixed_references:
                    refs = tuple(replace(ref, transcript=recast(ref.transcript)) for ref in fixed_references)
                else:
                    refs = (Reference(recast(anchor_source.prompt_text), anchor_source.codes, '首段音色锚'),)
                history = []
            if together:
                direction=session.direction
                context_description=system_context
            else:
                from .scene_conditioning import isolated_direction,isolated_system_context
                name=next((name for name,identifier in plan['speaker_map'].items() if identifier==actor),f'speaker {actor}')
                direction=isolated_direction(session.direction,scene,characters,actor=actor,name=name,
                    previous_actor=line_other_actor,note=note)
                context_description=isolated_system_context(scene,characters,actor=actor,name=name,
                    previous_actor=line_other_actor,system_context=system_context)
            context = Session(refs, tuple(history), direction)
            local = replace(sampling, seed=(sampling.seed + ordinal) % 2**64)
            def advancing(step, total, index=ordinal):
                if progress:
                    progress(index * sampling.max_frames + step, len(plan['requests']) * sampling.max_frames)
            advancing(0, sampling.max_frames)
            texts = [tagged(unit['text'], unit['speaker_id'] if together else 0) for unit in request]
            observed = {'on_codes': on_observed_codes} if on_observed_codes is not None else {}
            if context_description:
                observed['system_context']=context_description
            if len(texts) > 1 or together:
                generated = engine.scene(texts, scene=scene if together else '', characters=characters if together else '', references=refs,
                    sampling=local, progress=advancing, session=context, note=note if together else '',
                    generation_mode='together', split_long_text=False, **observed)
            else:
                generated = engine.speak(texts[0], context, local, progress=advancing, **observed)
            source = generated.takes[-1]
            import comfy.model_management as mm
            mm.throw_exception_if_processing_interrupted()
            segments = source.segments()
            if source.metadata.get('length_limit_reached') or len(segments) != len(request) or any(c.shape[-1] == 0 for c in segments):
                raise RuntimeError(f'Chunked scene request {ordinal + 1} did not complete every planned unit')
            expected_request = JOIN.join(directed_text(text, request_direction) for text in texts)
            if source.prompt_text != expected_request or len(expected_request.encode('utf-8')) > chunk_bytes:
                raise RuntimeError('Scene generation changed its budgeted request text')
            if fixed_reference_anchor:
                if ordinal == 0:
                    anchor_source = source
                else:
                    source = replace(source, metadata={**source.metadata, 'long_context': {
                        'version': 1, 'policy': 'explicit_references' if fixed_references else 'first_new_take_reference',
                        'source_take_id': None if fixed_references else anchor_source.take_id,
                        'explicit_reference_count': len(fixed_references), 'completed_history_turns': 0}})
            indexes = [unit['unit_index'] for unit in request]
            sources.append(dict(speaker=None if together else actor, unit_indexes=indexes,request_index=ordinal,request_seed=local.seed,
                prompt_text=source.prompt_text, transcript=source.transcript, frames=source.frames.tolist(),
                semantic_start=source.semantic_start, semantic_end=source.semantic_end, separator_id=source.separator_id,
                metadata=source.metadata, take_id=source.take_id))
            for segment_index, (unit, codes) in enumerate(zip(request, segments)):
                line = unit['line_index']
                mapped = units[unit['unit_index']]
                mapped.update(source_take_id=source.take_id, source_segment=segment_index,
                    source_frame_start=0, source_frame_end=codes.shape[-1])
                generated_codes[unit['unit_index']] = codes
            if together:
                global_history.append(source)
            else:
                histories.setdefault(actor, []).append(source)
                if mode == 'per_line_context':
                    # Freeze the immediately previous original line's context
                    # for every continuation; same-role next lines add none.
                    previous_source,previous_actor=source,actor
    rows = []
    for unit in units:
        line = unit['line_index']; codes = generated_codes[unit['unit_index']]
        unit.update(line_frame_start=line_frames[line], line_frame_end=line_frames[line] + codes.shape[-1])
        line_frames[line] += codes.shape[-1]; line_codes[line].append(codes)
    for index, pieces in enumerate(line_codes):
        if index:
            control = torch.zeros((1, 11), dtype=torch.long); control[0, 0] = 252196; rows.append(control)
        rows.append(audio_frames(torch.cat(pieces, dim=-1)))
    metadata = dict(task='scene', generation_mode=mode, scene=scene, characters=characters,
        script_lines=list(lines), scene_cast_lines=plan['cast_lines'], scene_speaker_map=plan['speaker_map'],
        scene_casting='chunked_source_map_v1', director_note=note, line_speakers=plan['line_speakers'],
        expected_segments=len(lines), actual_segments=len(lines), gap_seconds=.3,
        scene_chunking=dict(version=VERSION, enabled=True, chunk_bytes=chunk_bytes), scene_units=units, scene_sources=sources,
        seed=sampling.seed, sampling=sampling.__dict__, finish_reason='eos', length_limit_reached=False,
        preview_mode='deferred_script_order', chunk_count=len(units), request_count=len(sources),
        duration_seconds=(sum(line_frames) * FRAME_LENGTH + (len(lines) - 1) * round(.3 * SAMPLE_RATE)) / SAMPLE_RATE)
    if history_limit != 2:
        metadata['scene_chunking']['history_limit'] = history_limit
    if request_direction:
        metadata['scene_chunking']['request_direction'] = request_direction
    if system_context:
        metadata['system_context'] = system_context
    if fixed_reference_anchor:
        metadata['scene_chunking']['fixed_reference_anchor'] = True
        metadata['long_context'] = {'version': 1,
            'policy': 'explicit_references' if fixed_references else 'first_new_take_reference',
            'source_take_id': None if fixed_references else anchor_source.take_id,
            'explicit_reference_count': len(fixed_references), 'completed_history_turns': 0}
    details = [source['metadata'] for source in sources]
    for field in ('elapsed_seconds', 'cpu_experts_run', 'cpu_decode_calls'):
        metadata[field] = sum(item.get(field, 0) for item in details)
    for field in ('moe_backend', 'moe_requested_backend', 'expert_placement', 'expert_requested_placement'):
        metadata[field] = '/'.join(sorted({item.get(field, 'not_recorded') for item in details}))
    metadata['moe_fallback'] = '; '.join(dict.fromkeys(item['moe_fallback'] for item in details if item.get('moe_fallback'))) or None
    for field in ('expert_backend_calls', 'gdn_calls_by_backend'):
        metadata[field] = {key: sum(item.get(field, {}).get(key, 0) for item in details)
            for key in {key for item in details for key in item.get(field, {})}}
    metadata['gdn_calls'] = sum(metadata['gdn_calls_by_backend'].values())
    graphs = [item.get('cuda_graph', {}) for item in details]
    metadata['cuda_graph'] = {key: sum(item.get(key, 0) for item in graphs)
        for key in ('captures', 'replays', 'tower_captures', 'tower_replays')}
    metadata['cuda_graph'].update(cpu_decode=any(item.get('cpu_decode', False) for item in graphs),
        fallback_reasons=list(dict.fromkeys(reason for item in graphs for reason in item.get('fallback_reasons', []))),
        disabled_reasons=list(dict.fromkeys(item['disabled_reason'] for item in graphs if item.get('disabled_reason'))))
    take = Take(JOIN.join(plan['cast_lines']), spoken_text('\n'.join(lines), scene=True), torch.cat(rows), metadata=metadata)
    validate_scene(take)
    if progress:
        progress(len(sources) * sampling.max_frames, len(sources) * sampling.max_frames)
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    if on_codes is not None:
        publish = getattr(getattr(on_codes, '__self__', None), 'append_scene', None)
        if publish is not None:
            publish(take)
        # A bare frame callback has no source boundary protocol. Do not emit
        # misleading concatenated codes; the final AUDIO remains available.
    return session.append(take)
