"""Compare select/unselect input tensors to actual core prompt bytes.

This is an observation-contract diagnostic, not a policy score. It never
changes a legal menu, runtime module or training label.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from contracts import read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))


def main():
    from scripts.collect_teaching_exercise import load_native
    from ygoai.rl.exercise_actor import array_digest, extract_trace
    from ygoai.rl import exercise_core
    import ygoai.rl.exercise_actor as trace_module
    trace_module.messages = exercise_core.messages
    from ygoai.rl.observation_schema import tensor_contract
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = read(args.frozen / 'plan.json')
    trace = read(args.trace)
    runtime = Path(plan['runtime'])
    config = SimpleNamespace(native=runtime/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so',
        database=runtime/'assets/cards.cdb', scripts=runtime/'assets/script',
        code_list=runtime/'code_list.txt', semantics=runtime/'semantics')
    if sha(config.native) != plan['native_sha256']:
        raise ValueError('native runtime changed')
    native = load_native(config, runtime/'registration.ydk')
    d = trace['definition']
    env = native.ExerciseActor(d.get('lua', ''), d['initial'], str(config.semantics),
        trace.get('seed') or 1, 'structured-lite-v1', d['seed_sequence'], d['duel_flags'], 2048, True)
    probes = []
    try:
        env.reset()
        for decision in trace['decisions']:
            snap = env.snapshot()
            obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if (snap['menu'], array_digest(obs)) != (decision['menu'], decision['observation_sha256']):
                raise ValueError('failure input could not be reproduced')
            if snap['message'] == 26:
                events, _, _ = extract_trace(env, snap)
                prompt = next(e for e in reversed(events) if e['op'] == 26)
                raw = bytes.fromhex(prompt['raw'])
                nselect = raw[6]
                nunselect = raw[7 + 8 * nselect]
                expected = dict(minimum=raw[4], maximum=raw[5], selected_count=nunselect)
                selection = obs['selection_'][0].tolist()
                actual = dict(minimum=selection[5], maximum=selection[6], selected_count=selection[8])
                probes.append(dict(step=decision['step'], core_prompt_hex=prompt['raw'],
                    expected=expected, actual=actual, match=expected == actual,
                    selection_tensor=selection,
                    unselect_action_flags=obs['action_features_'][0, :len(snap['menu']), 14].tolist()))
            env.step(decision['action'])
    finally:
        env.close()
    result = dict(native_sha256=sha(config.native), trace_sha256=sha(args.trace),
        probes=probes, mismatches=sum(not p['match'] for p in probes),
        claim='raw prompt versus declared selection tensor; no strategic error labels')
    write(args.output, result)
    print(dict(probes=len(probes), mismatches=result['mismatches']), flush=True)


if __name__ == '__main__':
    main()
