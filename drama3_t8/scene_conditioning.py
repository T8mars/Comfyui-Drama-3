"""Keep original scene descriptions distinct from an isolated request's IDs."""


def isolated_direction(direction, scene, characters, *, actor, name, previous_actor=None, note=''):
    if type(actor) is not int or not 0 <= actor < 16:
        raise ValueError('Original scene speaker must be 0 to 15')
    if previous_actor is not None and (type(previous_actor) is not int or
            not 0 <= previous_actor < 16 or previous_actor == actor):
        raise ValueError('Context speaker must be a different original scene speaker')
    parts=[]
    for label,value in (('Original scene direction',direction),('Original scene context',scene),
            ('Original scene character descriptions',characters),('Director revision for the original scene',note)):
        if value:
            parts.append(f'{label}:\n{value}')
    # References, the user turn and this role's prior turns are all recast to
    # local 0 by the isolated generators. The descriptions above retain the
    # user's original numbering; never tell local 0 to speak as global 1.
    mapping=f'Local voice mapping:\n<|speaker:0|> = {name}; original scene speaker {actor}.'
    if previous_actor is not None:
        mapping+=f'\n<|speaker:1|> = context only; original scene speaker {previous_actor}.'
    parts.append(mapping)
    parts.append('Original scene role numbers describe the scene; use the local voice mapping for this request.')
    parts.append('Voice only <|speaker:0|>.')
    return '\n\n'.join(parts)
