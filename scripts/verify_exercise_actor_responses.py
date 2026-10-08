"""Re-execute native actor responses through the same core's independent C API."""
import argparse
import json
from pathlib import Path

import numpy as np

from ygoai.rl.exercise_core import Core, sha
from ygoai.rl.exercise_actor import actor_grade, array_digest
from ygoai.rl.exercise_starters import fixture


def verify(path, args):
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['status'] not in ('success', 'verified_failure') or record['core_seed'] is None:
        raise ValueError('cannot certify incomplete/invalid native trajectory')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        tensors = dict(archive)
    if array_digest(tensors) != record['observation_archive_digest']:
        raise ValueError('observation archive digest mismatch')
    if not tensors or any(len(a) != len(record['decisions']) for a in tensors.values()):
        raise ValueError('observation/decision count mismatch')
    for i, decision in enumerate(record['decisions']):
        obs = {key: value[i:i + 1] for key, value in tensors.items()}
        if array_digest(obs) != decision['observation_sha256']:
            raise ValueError('observation/decision association mismatch')
    core = Core(args.native, args.database, args.scripts, seed=record['core_seed'])
    try:
        lua, lp = fixture(record['case'])
        core.start(lua, lp)
        for row in record['core_responses']:
            prompt = core.next_prompt()
            if prompt is None:
                raise ValueError('response prefix exceeds duel lifetime')
            data = bytes.fromhex(row['data'])
            value = int.from_bytes(data, 'little', signed=True) if row['kind'] == 'int' else data
            core.respond(prompt, value)
        core.next_prompt()
        state = core.state()
        verdict = actor_grade(record['case'], state, core.events, record['stable_boundary'])
        # Card order, LP, phase and terminal result must agree, not just success.
        same_state = state == record['final_state']
        same_events = core.events == record['events']
        passed = same_state and same_events and verdict == record['status']
        return dict(path=path.name, sha256=sha(path), passed=passed,
                    same_state=same_state, same_events=same_events, status=verdict,
                    raw_responses=len(record['core_responses']))
    finally:
        core.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'database', 'scripts', 'input', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.input / 'summary.json').read_text(encoding='utf-8'))
    if not summary['passed']:
        raise ValueError('baseline integrity gate did not pass')
    for row in summary['results']:
        for evidence in row['evidence']:
            if sha(a.input / evidence['path']) != evidence['sha256']:
                raise ValueError('baseline evidence hash mismatch')
    native_path = str(a.native.resolve())
    if summary['artifacts_before'].get(native_path) != sha(a.native):
        raise ValueError('native runtime mismatch')
    rows = [verify(path, a) for path in sorted(a.input.glob('*-0.json'))]
    result = dict(schema='exercise-actor-response-audit-v1', results=rows,
                  passed=bool(rows) and all(r['passed'] for r in rows), native_sha256=sha(a.native))
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
