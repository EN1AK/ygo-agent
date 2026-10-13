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
