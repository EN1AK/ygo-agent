"""Convert audited first-turn YRP2 routes through the actual policy adapter."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

try:
    import _repo_bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _repo_bootstrap  # noqa: F401
import numpy as np
from scripts.collect_teaching_exercise import load_native
from ygoai.rl.exercise_actor import array_digest, extract_trace, hidden_identity_audit
from ygoai.rl.exercise_core import sha
from ygoai.rl.human_replay import response_action
from ygoai.rl.observation_schema import tensor_contract


def convert(args, native, source):
    replay = source['source']
    if source['status'] != 'responses_replayed':
        raise ValueError('raw replay has not passed admission')
    initial = [[code, p, loc, 0, 8] for p in (0, 1)
               for loc, deck in ((1, replay['main_decks'][p]), (64, replay['extra_decks'][p]))
               for code in deck]
    if (replay['initial_lp'], replay['initial_hand'], replay['draw_count']) != (8000, 5, 1):
        raise ValueError('actor initialization currently supports standard LP/hand/draw only')
    definition = dict(controlled=0, initial=initial, seed_sequence=replay['seed_sequence'],
                      duel_flags=replay['duel_flags'] | 16, max_steps=512, kind='human_combo')
    env = native.ExerciseActor('', initial, str(args.semantics), 1, 'structured-lite-v1',
                              replay['seed_sequence'], definition['duel_flags'], 512)
    decisions, arrays = [], []
    record = dict(schema='ygo-human-actor-replay-v1', definition=definition, source_sha256=source['source_sha256'],
                  core_audit_sha256=sha(args.input), native_sha256=sha(args.native), training_ready=False)
    try:
        env.reset()
        for step in range(512):
            snap = env.snapshot()
            events, state, responses = extract_trace(env, snap)
            for i, r in enumerate(responses[:len(replay['responses'])]):
                if r['data'] != replay['responses'][i]:
                    raise ValueError(f'raw response drift at {i}: {r["data"]} != {replay["responses"][i]}')
            count = len(responses)
            # Automatic empty chains after the recorded first-turn boundary do
            # not create supervised decisions for the opponent's next turn.
            if sum(e['op'] == 40 for e in events) >= 2 or count >= len(replay['responses']):
                record['status'] = 'converted'
                break
            if snap['invalid'] or snap['done']:
                raise ValueError('actor ended before the source endpoint')
            prompt = source['decisions'][count]['prompt']
            action = response_action(snap, prompt, replay['responses'][count])
            obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if step == 0:
                hidden_identity_audit(obs)
            decisions.append(dict(step=step, player=snap['player'], message=snap['message'],
                                  menu=snap['menu'], action=action, core_response_count=count,
                                  observation_sha256=array_digest(obs), teaching=snap['player'] == 0))
            arrays.append({k: v[0].copy() for k, v in obs.items()})
            env.step(action)
        else:
            raise TimeoutError('actor conversion decision budget exhausted')
    except (ValueError, IndexError, TimeoutError) as exc:
        record.update(status='rejected', error=str(exc))
    finally:
        snap = env.snapshot()
        events, state, responses = extract_trace(env, snap)
        record.update(decisions=decisions, events=events, final=state, core_responses=responses,
                      core_seed=env.trace()['core_seed'])
        env.close()
    tensors = {k: np.stack([a[k] for a in arrays]) for k in arrays[0]} if arrays else {}
    return record, tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'scripts', 'code-list', 'semantics', 'registration', 'input', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(args, k, v.resolve())
    source = json.loads(args.input.read_text(encoding='utf-8'))
    native = load_native(args, args.registration)
    record, arrays = convert(args, native, source)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    if arrays:
        np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    print(json.dumps(dict(status=record['status'], decisions=len(record['decisions']), error=record.get('error'))))
    if record['status'] != 'converted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
