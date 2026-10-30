"""Bind scene references to explicit script roles, never to voice display names."""
from collections import defaultdict


def validate_role_voice_collisions(selections):
    groups = defaultdict(set)
    roles = defaultdict(set)
    for item in selections:
        if item.get('role') and item.get('speaker') is not None:
            if type(item['speaker']) is not int or not 0 <= item['speaker'] < 16 or not isinstance(item['role'],str):
                raise ValueError('角色绑定须使用有效角色名称和0到15的整数编号。')
            groups[item['speaker']].add(item.get('voice_id'))
            roles[item['speaker']].add(item['role'])
    for speaker, voices in groups.items():
        if len(voices) > 1 and len(roles[speaker]) > 1:
            raise ValueError(f'角色编号 {speaker} 同时绑定了不同音色：' + '、'.join(sorted(roles[speaker])) + '。请为不同音色使用独立角色编号。')


def prepare_scene_voice_bindings(payload):
    if payload.get('mode') not in {'scene', 'dialogue'} or not payload.get('voice_versions'):
        return payload
    from .scene_cast import prepare_scene_cast
    from .native.prompt import SPEAKER_TAG
    lines = [line for line in payload.get('text', '').splitlines() if line.strip()]
    _, _, mapping = prepare_scene_cast(lines, force=True)
    slots = set(mapping.values()) | {int(match.group(1)) for line in lines for match in SPEAKER_TAG.finditer(line)}
    selections = []
    grouped = defaultdict(set)
    for binding in payload['voice_versions']:
        if not isinstance(binding, dict):
            raise ValueError('音色绑定须包含固定音色版本和目标角色。')
        item = dict(binding)
        if item.get('role'):
            if item['role'] not in mapping:
                raise ValueError(f'音色绑定角色“{item["role"]}”不在当前剧本中，请为台词添加对应角色名。')
            item['speaker'] = mapping[item['role']]
        speaker = item.get('speaker')
        if type(speaker) is not int or not 0 <= speaker < 16:
            raise ValueError('请在角色选声中指定此音色对应的角色。')
        if slots and speaker not in slots:
            raise ValueError(f'音色绑定到角色 {speaker}，但当前剧本没有这个角色编号。')
        grouped[speaker].add(item.get('voice_id'))
        selections.append(item)
    if len(slots) > 1 and any(len(voices) > 1 for voices in grouped.values()):
        raise ValueError('多个不同音色被绑定到了同一个角色编号。请在“角色选声”中分别绑定导演、女主、旁白，或套用已保存的选角。')
    validate_role_voice_collisions(selections)
    return {**payload, 'voice_versions': selections}
