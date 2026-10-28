"""Sample-accurate infill of the audio a user actually heard.

The complete new transcript conditions FIM. Its unknown middle transcript is
never paired with the generated middle as an ordinary TTS history turn. A real
encoding of the complete final PCM supplies that history instead.
"""
import copy
import hashlib
import io
import json
import math
import re
from dataclasses import replace

import torch
import torch.nn.functional as F

from .native.codec import FRAME_LENGTH, SAMPLE_RATE, DECODE_FRAME_BYTES
from .state import Take, validate_codes, validate_frames

VERSION = 1
CONTEXT_FRAMES = 128
MAX_SAMPLES = 7200 * SAMPLE_RATE


def pcm16(wave):
    """Canonical libsndfile PCM16 playback, returned as float32 and integers."""
    import soundfile as sf
    if (not isinstance(wave, torch.Tensor) or wave.ndim != 2 or wave.shape[0] != 1
            or not 0 < wave.shape[-1] <= MAX_SAMPLES or not torch.isfinite(wave).all()):
        raise ValueError('Rendered infill needs finite, nonempty mono PCM')
    buffer = io.BytesIO()
    sf.write(buffer, wave.detach().float().cpu().T.numpy(), SAMPLE_RATE, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    values, rate = sf.read(buffer, dtype='int16', always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError('Unexpected playback sample rate')
    integers = torch.from_numpy(values.T.copy())
    return integers.float().div(32768), integers


def normalize_subtype(value):
    if value == 'PCM16':
        return 'PCM_16'
    if not isinstance(value, str) or value not in ('FLOAT', 'PCM_16'):
        raise ValueError('Playback subtype must be FLOAT or PCM_16')
    return value


def canonical_pcm(wave, subtype):
    """Preserve a real FLOAT view, or perform the actual PCM16 writer mapping."""
    subtype = normalize_subtype(subtype)
    if subtype == 'PCM_16':
        return pcm16(wave)[0]
    if (not isinstance(wave, torch.Tensor) or wave.ndim != 2 or wave.shape[0] != 1
            or not 0 < wave.shape[-1] <= MAX_SAMPLES or not torch.isfinite(wave).all()):
        raise ValueError('Rendered infill needs finite, nonempty mono PCM')
    return wave.detach().to(device='cpu', dtype=torch.float32).contiguous().clone()


def pcm_hash(values, subtype='PCM_16'):
    subtype = normalize_subtype(subtype)
    header = f'{SAMPLE_RATE}:1:{values.shape[-1]}:{subtype}:'.encode('ascii')
    return hashlib.sha256(header + values.contiguous().numpy().tobytes()).hexdigest()


def playback_identity(wave, subtype, source=None):
    """Identity of the actual writer contract; never imply FLOAT is PCM16."""
    subtype = normalize_subtype(subtype)
    canonical = canonical_pcm(wave, subtype)
    stored = (canonical*32768).to(torch.int16) if subtype == 'PCM_16' else canonical
    digest = pcm_hash(stored, subtype)
    result = dict(pcm_sha256=digest, samples=canonical.shape[-1], sample_rate=SAMPLE_RATE, subtype=subtype)
    if subtype == 'PCM_16':
        result['pcm16_sha256'] = digest
    if source is not None:
        result['source'] = source
    return result


def _bits_equal(left, right):
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def _validate_playback_hash(wave, identity):
    subtype = normalize_subtype(identity.get('subtype'))
    stored = wave
    if subtype == 'PCM_16':
        scaled = wave*32768
        if not torch.equal(scaled, scaled.round()) or scaled.min() < -32768 or scaled.max() > 32767:
            raise ValueError('Rendered infill PCM16 quantization differs')
        stored = scaled.to(torch.int16)
    digest = pcm_hash(stored, subtype)
    if identity.get('pcm_sha256') != digest or (subtype == 'PCM_16' and identity.get('pcm16_sha256') != digest):
        raise ValueError('Rendered infill playback PCM hash differs')
    if subtype == 'FLOAT' and 'pcm16_sha256' in identity:
        raise ValueError('FLOAT playback cannot claim a PCM16 identity')


def _tensor_hash(tensor):
    # Session storage canonicalizes frame tensors to int64. Hash integer
    # values in that same representation so valid int32 sources roundtrip.
    if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        tensor = tensor.to(torch.int64)
    header = f'{tensor.dtype}:{tuple(tensor.shape)}:'.encode('ascii')
    return hashlib.sha256(header + tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _source_hash(take):
    """Bind direct parent provenance without recursively validating ancestors."""
    payload = dict(take_id=take.take_id, prompt=take.prompt_text, transcript=take.transcript,
        semantic_start=take.semantic_start, semantic_end=take.semantic_end, separator_id=take.separator_id,
        metadata=take.metadata, frames=_tensor_hash(take.frames),
        blobs={name:_tensor_hash(blob) for name,blob in sorted(take.blobs.items())},
        sources=[source.take_id for source in take.sources])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def _cancel(progress=None, done=0, total=1):
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    if progress is not None:
        progress(done, total)
    mm.throw_exception_if_processing_interrupted()


def context_windows(wave, start, end, frames=CONTEXT_FRAMES):
    """Exact boundary PCM; only the short left context is explicitly left-padded."""
    cap = frames * FRAME_LENGTH
    left_start = max(0, start - cap)
    left = wave[:, left_start:start]
    left_pad = 0
    if left.shape[-1]:
        if left.shape[-1] >= FRAME_LENGTH:
            count = left.shape[-1] // FRAME_LENGTH * FRAME_LENGTH
            left_start = start - count
            left = wave[:, left_start:start]
        else:
            left_pad = FRAME_LENGTH - left.shape[-1]
            left = F.pad(left, (left_pad, 0))
    right_end = min(wave.shape[-1], end + cap)
    right = wave[:, end:right_end]
    right_pad = -right.shape[-1] % FRAME_LENGTH if right.shape[-1] else 0
    return left, right, dict(left_start=left_start, left_end=start, left_pad_samples=left_pad,
        left_real_samples=start-left_start, right_start=end, right_end=right_end,
        right_pad_samples=right_pad, right_real_samples=right_end-end, max_frames=frames)


def encode_memory_budget(samples):
    """Activation request, not an empirically certified peak or OOM guarantee.

    Encoding also builds an encoder-transformer mask at hop512. Request both
    a per-code-frame activation allowance and that quadratic float-mask reserve.
    Long-input real-device validation remains separate from this accounting.
    """
    frames = math.ceil(samples / FRAME_LENGTH)
    # encode() pads to whole Codec frames before the hop512 encoder runs.
    encoder_steps = frames * FRAME_LENGTH // 512
    return frames * DECODE_FRAME_BYTES + 4 * encoder_steps * encoder_steps


def _encode(engine, wave):
    if not wave.shape[-1]:
        return torch.empty((10, 0), dtype=torch.long)
    import comfy.model_management as mm
    codec = engine.load_codec(memory_required=encode_memory_budget(wave.shape[-1]))
    device = codec.patcher.load_device
    with torch.inference_mode(), mm.cuda_device_context(device):
        codes = codec.model.encode(wave.to(device))[0].detach().to(device='cpu', dtype=torch.long)
    validate_codes(codes)
    if codes.shape[-1] != math.ceil(wave.shape[-1] / FRAME_LENGTH):
        raise ValueError('Codec encoding duration differs from the PCM context')
    return codes


def visible_view(engine, take, playback_subtype=None):
    """Compile saved Editor changes once; return playback PCM and role intervals."""
    from .editor import prepare, plan, render, fingerprint
    audio, analysis, edits = prepare(engine, take)
    if audio['sample_rate'] != SAMPLE_RATE:
        raise ValueError('Rendered infill requires native 44.1kHz audio')
    planned = plan(analysis, edits)
    wave = render(audio['waveform'][0], analysis, planned['pieces'], edits)
    selected = playback_subtype
    if selected is None:
        # A saved Editor overlay produces a new FLOAT playback view even if
        # its immutable base was an older PCM16 render. An explicit selected
        # asset format still takes priority over this native default.
        selected = ('FLOAT' if take.metadata.get('audio_editor') else
            take.metadata.get('render_infill', {}).get('subtype', 'FLOAT'))
    canonical = canonical_pcm(wave, selected)
    intervals = []
    cursor = 0
    for piece in planned['pieces']:
        if 'gap' in piece:
            cursor += round(piece['gap'] * SAMPLE_RATE)
            continue
        first, last = (round(value * SAMPLE_RATE) for value in piece['src'])
        for line in analysis['lines']:
            a = max(first, round(line['start'] * SAMPLE_RATE))
            b = min(last, round(line['end'] * SAMPLE_RATE))
            if b > a:
                setting = edits['speakers'].get(str(line['speaker']), {})
                intervals.append(dict(start=cursor+a-first, end=cursor+b-first,
                    speaker=line['speaker'], audible=not setting.get('mute', False)))
        cursor += last-first
    if cursor != canonical.shape[-1]:
        raise ValueError('Editor playback map differs from rendered PCM')
    identity = playback_identity(canonical, selected, source=fingerprint(take))
    return canonical, intervals, identity


def _selected_roles(intervals, start, end):
    return {item['speaker'] for item in intervals if item['audible'] and
        (item['start'] < end and item['end'] > start if start != end else item['start'] <= start < item['end'])}


def _role(intervals, start, end, explicit):
    if explicit is not None:
        if type(explicit) is not int or not 0 <= explicit < 16:
            raise ValueError('Target speaker must be an integer from 0 to 15')
        return explicit
    roles = _selected_roles(intervals, start, end)
    if len(roles) != 1:
        raise ValueError('Choose an explicit target speaker for a cross-role or silent selection')
    return roles.pop()


def _history_text(take, transcript, actor):
    from .native.prompt import check_text, SPEAKER_TAG
    from .scene_cast import prepare_scene_cast
    if not isinstance(transcript, str):
        raise ValueError('Complete new infill transcript must be text')
    check_text(transcript, 'complete new infill transcript')
    scene = take.metadata.get('task') == 'scene' or take.metadata.get('render_infill', {}).get('scene_text', False)
    if scene:
        lines = tuple(line for line in transcript.splitlines() if line.strip())
        casting = take
        if 'render_infill' in take.metadata:
            policy = take.metadata['render_infill']
            # Only a casting view: never pair this synthetic task label with
            # acoustic history. prepare_scene_cast scans scene records only.
            casting = replace(take, metadata=dict(task='scene',
                script_lines=policy['full_target_transcript'].splitlines(),
                scene_speaker_map=policy['scene_speaker_map']))
        cast, _, mapping = prepare_scene_cast(lines, (), (casting,), force=True)
        return '\n'.join(cast), mapping, True
    if SPEAKER_TAG.match(transcript.lstrip()) is None:
        transcript = f'<|speaker:{actor}|> {transcript}'
    return transcript, {}, False


def _timeline(intervals, start, end, middle_samples, actor):
    result = []
    shift = middle_samples - (end-start)
    for item in intervals:
        if item['start'] < start:
            a, b = item['start'], min(item['end'], start)
            if b > a:
                result.append(dict(start=a/SAMPLE_RATE, end=b/SAMPLE_RATE,
                    speaker=item['speaker'], text='', needs_alignment=True))
        if item['end'] > end:
            a, b = max(item['start'], end)+shift, item['end']+shift
            if b > a:
                result.append(dict(start=a/SAMPLE_RATE, end=b/SAMPLE_RATE,
                    speaker=item['speaker'], text='', needs_alignment=True))
    if middle_samples:
        result.append(dict(start=start/SAMPLE_RATE, end=(start+middle_samples)/SAMPLE_RATE,
            speaker=actor, text='', needs_alignment=True))
    result.sort(key=lambda item: (item['start'], item['end']))
    return [dict(item, line=index) for index, item in enumerate(result)]


def infill_take(engine, source_take, target_transcript, start_seconds, end_seconds,
                sampling=None, duration_frames=None, progress=None, target_speaker=None, expected_view=None,
                playback_subtype=None):
    """Use the complete new script, preserving all visible unselected PCM samples."""
    from .engine import Sampling
    from .prompts import infill_prompt, spoken_text
    from .dialogue import audio_frames
    sampling = sampling or Sampling()
    sampling.validate()
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start_seconds, end_seconds)):
        raise ValueError('Edit interval must be finite seconds')
    if not 0 <= start_seconds <= end_seconds:
        raise ValueError('Edit interval must be ordered and nonnegative')
    _cancel(progress, 0, 7)
    selected = playback_subtype
    if expected_view is not None:
        if not isinstance(expected_view, dict) or 'subtype' not in expected_view:
            raise ValueError('Invalid playback view/subtype')
        expected_subtype = normalize_subtype(expected_view['subtype'])
        if selected is not None and normalize_subtype(selected) != expected_subtype:
            raise ValueError('Playback subtype conflicts with the selected asset')
        selected = expected_subtype
    wave, intervals, view = visible_view(engine, source_take, selected)
    if expected_view is not None and expected_view != view:
        raise ValueError('Playback source changed; reopen the selected take before infill')
    start, end = round(start_seconds*SAMPLE_RATE), round(end_seconds*SAMPLE_RATE)
    if not 0 <= start <= end <= wave.shape[-1]:
        raise ValueError('Edit interval is outside the actual playback audio')
    actor = _role(intervals, start, end, target_speaker)
    text, mapping, scene_text = _history_text(source_take, target_transcript, actor)
    left, right, windows = context_windows(wave, start, end)
    _cancel(progress, 1, 7)
    prefix = _encode(engine, left)
    _cancel(progress, 2, 7)
    suffix = _encode(engine, right)
    _cancel(progress, 3, 7)
    drama = engine.load_model()
    # The actual surrounding speech already identifies a unique selected role.
    # Keep explicit conditioning for voice overrides or ambiguous/silent cuts.
    prompt_speaker = None if _selected_roles(intervals, start, end) == {actor} else actor
    prompt = infill_prompt(drama.tokenizer, drama.config, text, prefix, suffix, duration_frames,
        target_speaker=prompt_speaker)
    def advancing(step, total):
        _cancel(progress, 3+step/max(total, 1), 7)
    middle = engine._take(prompt, text, target_transcript, sampling, progress=advancing, allow_empty=True)
    if middle.metadata.get('length_limit_reached'):
        raise RuntimeError('Infill generation reached max_frames before completing')
    middle = replace(middle, metadata={**copy.deepcopy(middle.metadata), 'task':'infill_raw',
        'history_eligible':False, 'transcript_scope':'complete_FIM_target_not_middle',
        'parent_take_id':source_take.take_id, 'target_speaker':actor})
    _cancel(progress, 4, 7)
    middle_codes = middle.codes
    if middle_codes.shape[-1]:
        decoded = engine.decode(torch.cat((prefix, middle_codes), dim=-1))
        if decoded['sample_rate'] != SAMPLE_RATE:
            raise ValueError('Unexpected infill Codec sample rate')
        actual = decoded['waveform'][0]
        if actual.shape[-1] != (prefix.shape[-1]+middle_codes.shape[-1])*FRAME_LENGTH:
            raise ValueError('Codec middle duration differs from its actual codes')
        inserted = actual[:, prefix.shape[-1]*FRAME_LENGTH:]
    else:
        inserted = wave[:, :0]
    combined = canonical_pcm(torch.cat((wave[:, :start], inserted, wave[:, end:]), dim=-1), view['subtype'])
    output_view = playback_identity(combined, view['subtype'])
    _cancel(progress, 5, 7)
    history = _encode(engine, combined)
    _cancel(progress, 6, 7)
    timeline = _timeline(intervals, start, end, inserted.shape[-1], actor)
    history_frames = audio_frames(history, middle)
    policy = dict(version=VERSION, pcm_blob='render_pcm', parent_pcm_blob='parent_pcm', prefix_blob='prefix_codes', suffix_blob='suffix_codes',
        source_take_id=source_take.take_id, middle_take_id=middle.take_id, target_speaker=actor,
        start_sample=start, end_sample=end, inserted_samples=inserted.shape[-1], output_samples=combined.shape[-1],
        sample_rate=SAMPLE_RATE, subtype=view['subtype'], pcm_sha256=output_view['pcm_sha256'], source_view=view,
        conditioning='visible_pcm_reencode_v1', context=windows, prefix_frames=prefix.shape[-1], suffix_frames=suffix.shape[-1],
        middle_decode='prefix_then_middle_crop_prefix', history='complete_visible_pcm_encode', history_text=text,
        rendered_pcm_encoding=True, history_pad_samples=-combined.shape[-1] % FRAME_LENGTH,
        encoding_memory_budget_bytes=dict(prefix=encode_memory_budget(left.shape[-1]),
            suffix=encode_memory_budget(right.shape[-1]), history=encode_memory_budget(combined.shape[-1])),
        encoding_budget_status='activation_estimate_real_long_input_validation_pending',
        full_target_transcript=target_transcript, transcript_status='complete_target_intent_requires_alignment',
        scene_text=scene_text, scene_speaker_map=mapping, timeline=timeline, prompt_rows=len(prompt))
    policy['tensor_sha256'] = dict(prefix=_tensor_hash(prefix), suffix=_tensor_hash(suffix),
        history=_tensor_hash(history_frames), middle=_tensor_hash(middle.frames))
    policy['parent_source_sha256'] = _source_hash(source_take)
    if view['subtype'] == 'PCM_16':
        policy['pcm16_sha256'] = output_view['pcm16_sha256']
    result = Take(text, spoken_text(target_transcript, scene=scene_text), history_frames,
        middle.semantic_start, middle.semantic_end, middle.separator_id,
        metadata={**copy.deepcopy(middle.metadata), 'task':'render_infill', 'history_eligible':True,
            'render_infill':policy, 'duration_seconds':combined.shape[-1]/SAMPLE_RATE,
            'inserted_duration_seconds':inserted.shape[-1]/SAMPLE_RATE, 'source_take_id':source_take.take_id,
            'scene_speaker_map':mapping},
        blobs={'render_pcm':combined, 'parent_pcm':wave, 'prefix_codes':prefix, 'suffix_codes':suffix}, sources=(source_take, middle))
    validate_take(result)
    _cancel(progress, 7, 7)
    return result


def validate_take(take):
    """Validate only this node of the typed source graph; state validates closure."""
    policy = take.metadata.get('render_infill')
    if (not isinstance(policy, dict) or type(policy.get('version')) is not int or policy['version'] != VERSION
            or take.metadata.get('task') != 'render_infill'):
        raise ValueError('Unsupported rendered infill version/task')
    if not isinstance(take.blobs, dict) or not isinstance(take.sources, tuple) or len(take.sources) != 2:
        raise ValueError('Rendered infill requires typed PCM and original sources')
    parent, middle = take.sources
    if (parent.take_id != policy.get('source_take_id') or middle.take_id != policy.get('middle_take_id')
            or len({take.take_id, parent.take_id, middle.take_id}) != 3
            or middle.metadata.get('task') != 'infill_raw' or middle.metadata.get('history_eligible') is not False):
        raise ValueError('Rendered infill source identity differs')
    if policy.get('parent_source_sha256') != _source_hash(parent):
        raise ValueError('Rendered infill original source provenance differs')
    for key in ('start_sample','end_sample','inserted_samples','output_samples','prefix_frames','suffix_frames','target_speaker','prompt_rows'):
        if type(policy.get(key)) is not int or policy[key] < 0:
            raise ValueError('Invalid rendered infill integer metadata')
    view = policy.get('source_view')
    if (not isinstance(view, dict) or type(view.get('samples')) is not int or not 0 < view['samples'] <= MAX_SAMPLES
            or view.get('sample_rate') != SAMPLE_RATE or view.get('subtype') not in ('PCM_16', 'FLOAT')
            or not isinstance(view.get('source'), str) or not re.fullmatch('[0-9a-f]{64}', view['source'])):
        raise ValueError('Invalid original playback view')
    if not 0 <= policy['target_speaker'] < 16 or not 0 <= policy['start_sample'] <= policy['end_sample'] <= view['samples']:
        raise ValueError('Invalid rendered infill role/selection')
    from .editor import fingerprint_core
    if view['source'] != fingerprint_core(parent):
        raise ValueError('Original playback view differs from its parent source fingerprint')
    if (policy.get('sample_rate') != SAMPLE_RATE or policy.get('subtype') not in ('PCM_16', 'FLOAT')
            or policy['subtype'] != view['subtype']):
        raise ValueError('Invalid rendered infill playback contract')
    if policy.get('history') != 'complete_visible_pcm_encode' or not isinstance(policy.get('history_text'), str):
        raise ValueError('Rendered infill history must match the complete encoded PCM')
    from .native.prompt import check_text
    from .prompts import spoken_text
    if (not isinstance(policy.get('full_target_transcript'), str) or type(policy.get('scene_text')) is not bool
            or policy['history_text'] != take.prompt_text or policy.get('rendered_pcm_encoding') is not True
            or policy['prompt_rows'] > 8192 or policy['prompt_rows'] == 0
            or take.transcript != spoken_text(policy['full_target_transcript'], scene=policy['scene_text'])):
        raise ValueError('Rendered infill full transcript/history provenance differs')
    check_text(policy['history_text'], 'complete rendered infill history')
    canonical, mapping, scene_text = _history_text(parent, policy['full_target_transcript'], policy['target_speaker'])
    if (policy['history_text'] != canonical or policy['scene_text'] != scene_text
            or policy.get('scene_speaker_map') != mapping or take.metadata.get('scene_speaker_map') != mapping):
        raise ValueError('Rendered infill canonical history differs from the complete new target script')
    if (middle.prompt_text != policy['history_text'] or middle.transcript != policy['full_target_transcript']
            or middle.metadata.get('parent_take_id') != parent.take_id
            or type(middle.metadata.get('target_speaker')) is not int
            or middle.metadata['target_speaker'] != policy['target_speaker']
            or middle.metadata.get('transcript_scope') != 'complete_FIM_target_not_middle'
            or middle.metadata.get('length_limit_reached') is not False):
        raise ValueError('Raw infill source must retain its actual full FIM target, parent and role')
    for field in ('pcm_blob','parent_pcm_blob','prefix_blob','suffix_blob'):
        if not isinstance(policy.get(field), str) or policy[field] not in take.blobs:
            raise ValueError('Missing rendered infill tensor blob')
    wave = take.blobs[policy['pcm_blob']]
    if (wave.dtype != torch.float32 or wave.device.type != 'cpu' or wave.ndim != 2 or wave.shape[0] != 1
            or not 0 < wave.shape[-1] <= MAX_SAMPLES or wave.shape[-1] != policy['output_samples']
            or not torch.isfinite(wave).all()):
        raise ValueError('Invalid rendered infill PCM tensor')
    _validate_playback_hash(wave, policy)
    parent_pcm = take.blobs[policy['parent_pcm_blob']]
    if (parent_pcm.dtype != torch.float32 or parent_pcm.device.type != 'cpu' or parent_pcm.shape != (1, view['samples'])
            or not torch.isfinite(parent_pcm).all()):
        raise ValueError('Invalid original playback PCM tensor')
    _validate_playback_hash(parent_pcm, view)
    if (not _bits_equal(wave[:, :policy['start_sample']], parent_pcm[:, :policy['start_sample']])
            or not _bits_equal(wave[:, policy['start_sample']+policy['inserted_samples']:], parent_pcm[:, policy['end_sample']:])):
        raise ValueError('Rendered infill changed unselected playback PCM')
    left, right, windows = context_windows(parent_pcm, policy['start_sample'], policy['end_sample'])
    if (policy.get('context') != windows or policy.get('conditioning') != 'visible_pcm_reencode_v1'
            or policy['prefix_frames'] != math.ceil(left.shape[-1]/FRAME_LENGTH)
            or policy['suffix_frames'] != math.ceil(right.shape[-1]/FRAME_LENGTH)
            or policy.get('history_pad_samples') != -wave.shape[-1] % FRAME_LENGTH):
        raise ValueError('Rendered infill context window/padding policy differs')
    for name, count in (('prefix_blob','prefix_frames'), ('suffix_blob','suffix_frames')):
        codes = take.blobs[policy[name]]
        validate_codes(codes)
        if codes.device.type != 'cpu' or codes.shape[-1] != policy[count] or codes.shape[-1] > CONTEXT_FRAMES:
            raise ValueError('Invalid rendered infill context codes')
    validate_frames(take.frames)
    validate_frames(middle.frames)
    hashes = policy.get('tensor_sha256')
    if (not isinstance(hashes, dict) or hashes.get('prefix') != _tensor_hash(take.blobs[policy['prefix_blob']])
            or hashes.get('suffix') != _tensor_hash(take.blobs[policy['suffix_blob']])
            or hashes.get('history') != _tensor_hash(take.frames) or hashes.get('middle') != _tensor_hash(middle.frames)):
        raise ValueError('Rendered infill context/history/source tensor hash differs')
    if (take.codes.shape[-1] != math.ceil(wave.shape[-1]/FRAME_LENGTH)
            or middle.codes.shape[-1]*FRAME_LENGTH != policy['inserted_samples']
            or policy['output_samples'] != policy['source_view']['samples']-(policy['end_sample']-policy['start_sample'])+policy['inserted_samples']):
        raise ValueError('Rendered infill duration/history/source coverage differs')
    timeline = policy.get('timeline')
    if not isinstance(timeline, list) or not timeline:
        raise ValueError('Rendered infill requires a timeline')
    previous = 0
    for index, item in enumerate(timeline):
        if (not isinstance(item, dict) or type(item.get('line')) is not int or item['line'] != index or type(item.get('speaker')) is not int
                or not 0 <= item['speaker'] < 16 or item.get('text') != '' or item.get('needs_alignment') is not True):
            raise ValueError('Invalid rendered infill text/role annotation')
        a, b = item.get('start'), item.get('end')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not previous <= a < b <= wave.shape[-1]/SAMPLE_RATE:
            raise ValueError('Invalid rendered infill timeline')
        previous = b


def decode_rendered(take):
    validate_take(take)
    return dict(waveform=take.blobs[take.metadata['render_infill']['pcm_blob']][None].clone(), sample_rate=SAMPLE_RATE)


def render_timeline(take):
    validate_take(take)
    return copy.deepcopy(take.metadata['render_infill']['timeline'])


def history_take(take):
    """A real complete-PCM encoding; never truncate it to pair with full text."""
    validate_take(take)
    if take.frames.shape[0] > 8192:
        raise ValueError('Complete rendered infill history exceeds the model prompt context; branch without this history')
    return replace(take, prompt_text=take.metadata['render_infill']['history_text'],
        metadata={'task':'rendered_history', 'source_take_id':take.take_id, 'history_actual_pcm':True}, blobs={}, sources=())
