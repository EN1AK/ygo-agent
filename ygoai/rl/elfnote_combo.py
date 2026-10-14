"""Outcome targets for a bounded, teacher-only elfnote learnability study."""
from collections import Counter
import numpy as np
from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.observation_schema import tensor_contract


GOALS = {
    'red': dict(monsters=[50954680, 84815190], spells=[24092792, 64491754], hand=5),
    'blue': dict(monsters=[50954680, 84815190], spells=[24092792, 64491754], hand=5),
    'innocent': dict(monsters=[50954680, 33158448, 30983281], spells=[24092792, 64491754], hand=5),
    'gate': dict(monsters=[50954680, 59581480, 90953320], spells=[10266279, 24092792, 64491754], hand=9),
}


def grade(state, goal):
    cards = [c for c in state['cards'] if c['player'] == 0]
    monsters = Counter(c['code'] for c in cards if c['location'] == 4 and c['position'] & 5)
    spells = Counter(c['code'] for c in cards if c['location'] == 8 and c['position'] & 5)
    hand = Counter(c['code'] for c in cards if c['location'] == 2)
    checks = dict(monsters=not (Counter(goal['monsters']) - monsters),
                  spells=not (Counter(goal['spells']) - spells),
                  wind_in_hand=hand[56651978] >= 1, hand_resources=sum(hand.values()) >= goal['hand'])
    return dict(success=all(checks.values()), checks=checks, monsters=dict(monsters),
                spells=dict(spells), hand_count=sum(hand.values()))


def rollout(native, semantics, actor, definition, goals, *, intervention=None):
    env = native.ExerciseActor('', definition['initial'], str(semantics), 1, 'structured-lite-v1',
                              definition['seed_sequence'], definition['duel_flags'], 512)
    actor.reset()
    decisions, arrays, attempted = [], [], False
    record = dict(schema='ygo-elfnote-rollout-v1', definition=definition, training_ready=False)
    try:
        env.reset()
        for step in range(512):
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            outcomes = [grade(state, g) for g in goals]
            ended = sum(e['op'] == 40 for e in events) >= 2
            # An idle prompt or next turn implies that the previous chain has
            # resolved. Having target card names halfway through a chain is
            # insufficient. The opponent does not reveal its planned response.
            stable = ended or snap['message'] == 11
            if stable and any(g['success'] for g in outcomes):
                record.update(status='success', outcomes=outcomes)
                break
            if snap['invalid']:
                raise ValueError('native invalid duel')
            if ended or snap['done']:
                record.update(status='goal_not_reached', outcomes=outcomes)
                break
            obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if step == 0:
                hidden_identity_audit(obs)
            logits, value, rnn = actor.predict(obs, snap['player'])
            if snap['player'] == 0:
                action = int(np.argmax(logits[:len(snap['menu'])]))
            else:
                choices = [i for i, a in enumerate(snap['menu']) if intervention and not attempted
                           and snap['message'] == 16 and a['code'] == intervention and a['act'] == 'Activate']
                if choices:
                    action, attempted = choices[0], True
                else:
                    choices = [i for i, a in enumerate(snap['menu']) if a['act'] == 'Cancel']
                    action = choices[0] if choices else 0
            decisions.append(dict(step=step, player=snap['player'], message=snap['message'], menu=snap['menu'],
                                  action=action, observation_sha256=array_digest(obs), logits=logits[:len(snap['menu'])].tolist(),
                                  value=value, rnn_sha256=rnn, core_response_count=len(responses)))
            arrays.append({k: v[0].copy() for k, v in obs.items()})
            env.step(action)
        else:
            record.update(status='budget_exhausted', outcomes=outcomes)
    finally:
        snap = env.snapshot()
        events, state, responses = extract_trace(env, snap)
        record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                      intervention_attempted=attempted, core_seed=env.trace()['core_seed'])
        env.close()
    return record, {k: np.stack([a[k] for a in arrays]) for k in arrays[0]}
