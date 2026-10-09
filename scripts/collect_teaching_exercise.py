"""Record and verify normal-opening references, without optimization."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.exercise_core import sha
from ygoai.rl.exercise_teaching import OpeningReference, opening, root_matches, verdict, turn_boundary_crossed
from ygoai.rl.observation_schema import tensor_contract


def load_native(args, registration):
    os.chdir(args.scripts.parent)
    spec = importlib.util.spec_from_file_location('exercise_actor_native', args.native)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.init_module(str(args.database), str(args.code_list), {'_exercise': str(registration)})
    return native


def execute(native, definition, semantics, branch='reference', actor=None, depth='position', rng=None,
            observation_schema='structured-lite-v1'):
    schema = getattr(actor, 'observation_schema', observation_schema)
    env_args = ('', definition['initial'], str(semantics), 1)
    env = native.ExerciseActor(*env_args) if schema == 'structured-lite-v1' else native.ExerciseActor(*env_args, schema)
    contextual = 'context' in definition
    if contextual:
        from ygoai.rl.exercise_contexts import ContextReference, ready, context_verdict
    teacher = ContextReference(definition, branch) if contextual else OpeningReference(definition, branch)
    decisions, tensors, roots = [], [], {}
    root_event = None
    if actor:
        actor.reset()
    try:
        env.reset()
        for step in range(160):
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            if snap['invalid']:
                raise ValueError('native invalid duel')
            interaction = definition['kind'] == 'interaction'
            chain_ended = interaction and 'interaction' in roots and any(e['op'] == 74 for e in events[root_event:])
            if roots and (snap['done'] or chain_ended or turn_boundary_crossed(events, root_event)):
                break
            obs = {k: snap['observation'][k] for k in tensor_contract(schema)}
            depths = ('opening', 'interaction') if interaction else ('opening', 'combo', 'position') if definition['kind'] == 'combo' else ('opening', 'battle')
            for kind in depths:
                root_snap = dict(snap, player=1 if snap['player']==definition['controlled'] else 0)
                eligible = not contextual or ready(definition,snap,events,state)
                if eligible and kind not in roots and root_matches(kind, root_snap, events):
                    if root_event is None or kind == 'interaction':
                        root_event = len(events)
                    roots[kind] = step
            if step == 0 or step in roots.values():
                hidden_identity_audit(obs)
            logits, value, rnn = actor.predict(obs, snap['player']) if actor else (None, None, None)
            controlled = actor is not None and depth in roots and snap['player'] == definition['controlled']
            n = len(snap['menu'])
            if controlled:
                probs = np.exp(logits[:n] - logits[:n].max())
                probs /= probs.sum()
                action = int(rng.choice(n, p=probs)) if rng is not None else int(np.argmax(logits[:n]))
            else:
                teacher.state = state
                action = teacher.choose(snap, events)
            if action is None or not 0 <= action < n:
                raise ValueError(f'no legal teacher choice step={step} menu={snap["menu"]}')
            decisions.append(dict(step=step, player=snap['player'], message=snap['message'],
                menu=snap['menu'], action=action, actor=controlled, observation_sha256=array_digest(obs),
                rnn_sha256=rnn, logits=logits[:n].tolist() if actor else None, value=value,
                core_response_count=len(responses)))
            if contextual:
                decisions[-1]['teaching'] = 'opening' in roots and snap['player']==definition['controlled']
            tensors.append(obs)
            env.step(action)
        else:
            raise ValueError('exercise decision budget exhausted')
        snap = env.snapshot()
        events, state, responses = extract_trace(env, snap)
        record = dict(definition=definition, branch=branch, depth=depth, roots=roots,
            decisions=decisions, events=events, final_state=state, core_responses=responses,
            core_seed=env.trace()['core_seed'], result=context_verdict(state,events,definition) if contextual else verdict(state, events, definition['kind']),
            normal_opening=True, training_ready=False, boundary='first-own-turn-main2-or-end-events')
        arrays = {k: np.concatenate([o[k] for o in tensors]) for k in tensors[0]}
        return record, arrays
    except Exception:
        print(json.dumps(dict(partial=decisions, snapshot=snap['menu']), ensure_ascii=False), flush=True)
        raise
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'code-list', 'scripts', 'semantics', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--kind', choices=['combo', 'battle', 'battle-negated', 'interaction'], default='combo')
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path): setattr(a, k, v.resolve())
    a.output.mkdir(exist_ok=False, parents=True)
    definition = opening(a.database, a.code_list)
    reg = a.output / 'registration.ydk'
    reg.write_text('#main\n' + '\n'.join(map(str, definition['decks'][1])) + '\n#extra\n50588353\n85289965\n!side\n')
    native = load_native(a, reg)
    summary = []
    for variant in range(3):
        definition = opening(a.database, a.code_list, variant, a.kind)
        branches = {'combo': ('reference', 'revive_in_defense'), 'battle': ('direct', 'attack_monster'),
                    'battle-negated': ('end_battle', 'attack_monster'), 'interaction': ('veiler', 'ogre', 'pass')}
        for branch in branches[a.kind]:
            records = []
            for repeat in range(2):
                record, arrays = execute(native, definition, a.semantics, branch)
                name = f'v{variant}-{branch}-{repeat}'
                path = a.output / (name + '.json')
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                np.savez_compressed(a.output / (name + '.npz'), **arrays)
                records.append(record)
            assert records[0] == records[1], 'replay mismatch'
            assert records[0]['result']['success'] == (branch in ('reference', 'direct', 'end_battle', 'veiler')), 'unexpected reference outcome'
            summary.append(dict(variant=variant, branch=branch, result=record['result'], roots=record['roots']))
            print(json.dumps(summary[-1]), flush=True)
    (a.output / 'summary.json').write_text(json.dumps(dict(results=summary, native_sha256=sha(a.native)), indent=2))


if __name__ == '__main__':
    main()
