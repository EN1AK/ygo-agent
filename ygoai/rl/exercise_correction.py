"""Fail-closed replay checks for promoting a recorded mistake to a question."""
from ygoai.rl.exercise_actor import array_digest


def checked_prefix_action(record, step, snapshot, obs, responses):
    expected = record['decisions'][step]
    actual = dict(step=step, player=snapshot['player'], message=snapshot['message'],
                  menu=snapshot['menu'], observation_sha256=array_digest(obs),
                  core_response_count=len(responses))
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError(f'exact-history mismatch at step {step}: {key}')
    if responses != record['core_responses'][:len(responses)]:
        raise ValueError(f'exact-history response mismatch at step {step}')
    action = expected['action']
    if not 0 <= action < len(snapshot['menu']):
        raise ValueError('recorded action outside legal menu')
    return action


def observe_replayed_action(teacher, definition, snapshot, action):
    """Advance the supported reference's summon context from the actual action.

    This is reference-policy bookkeeping only, never an actor/RNN override.
    It covers OpeningReference and its contextual/material subclasses.
    """
    chosen = snapshot['menu'][action]
    if snapshot['player'] != definition['controlled'] or snapshot['message'] != 11:
        return
    if chosen['act'] not in ('Summon', 'SpSummon'):
        return
    code = chosen['code']
    materials = definition.get('context', {}).get('materials')
    if materials:
        from ygoai.rl.exercise_teaching import CYBER
        from ygoai.rl.exercise_starters import VEILER
        code = {materials['starter']: CYBER, materials['tuner']: VEILER}.get(code, code)
    teacher.summoning = code
