"""Keep original scene descriptions distinct from an isolated request's IDs."""


def isolated_direction(direction, scene, characters, *, actor, name, previous_actor=None, note=''):
    if type(actor) is not int or not 0 <= actor < 16:
        raise ValueError('Original scene speaker must be 0 to 15')
    if previous_actor is not None and (type(previous_actor) is not int or
            not 0 <= previous_actor < 16 or previous_actor == actor):
        raise ValueError('Context speaker must be a different original scene speaker')
    # Actual references, current text and completed history are already recast
    # to local 0/1. Keep that identity in the existing source-map metadata, not
    # in an invented instruction which the model might speak or misinterpret.
    from .prompts import direction_tags
    return direction_tags(direction, note)


def isolated_system_context(scene, characters, *, actor, name, previous_actor=None, system_context=''):
    """Keep real user scene/character descriptions in system, not user styles."""
    if type(actor) is not int or not 0 <= actor < 16:
        raise ValueError('Original scene speaker must be 0 to 15')
    if previous_actor is not None and (type(previous_actor) is not int or not 0 <= previous_actor < 16 or previous_actor==actor):
        raise ValueError('Context speaker must be a different original scene speaker')
    if not any((scene,characters,system_context)):
        return ''
    parts=[value for value in (system_context, 'Scene: '+scene if scene else '', 'Characters:\n'+characters if characters else '') if value]
    parts.append(f'Current character: {name}; original scene speaker {actor}; current local speaker 0.')
    if previous_actor is not None:
        parts.append(f'Previous context character: original scene speaker {previous_actor}; local context speaker 1.')
    return '\n\n'.join(parts)
