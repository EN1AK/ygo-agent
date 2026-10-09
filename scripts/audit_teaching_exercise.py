"""Independent core and tensor audit for deck-only normal opening records."""
import argparse
import json
from pathlib import Path
import numpy as np
from ygoai.rl.exercise_core import Core, sha
from ygoai.rl.exercise_actor import array_digest
from ygoai.rl.exercise_teaching import verdict


def audit(path, args):
    r = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        arrays = dict(data)
    assert len(r['decisions']) == len(next(iter(arrays.values())))
    for i, row in enumerate(r['decisions']):
        assert row['observation_sha256'] == array_digest({k: v[i:i+1] for k,v in arrays.items()})
    core = Core(args.native, args.database, args.scripts, seed=r['core_seed'])
    try:
        core.start_standard(r['definition']['initial'])
        for response in r['core_responses']:
            prompt = core.next_prompt()
            assert prompt is not None
            data = bytes.fromhex(response['data'])
            core.respond(prompt, int.from_bytes(data, 'little', signed=True) if response['kind'] == 'int' else data)
        core.next_prompt()
        assert core.events == r['events'], 'event mismatch'
        assert core.state() == r['final_state'], 'state mismatch'
        if 'context' in r['definition']:
            from ygoai.rl.exercise_contexts import context_verdict
            result=context_verdict(core.state(),core.events,r['definition'])
        else:
            result=verdict(core.state(),core.events,r['definition']['kind'])
        assert result == r['result']
        return dict(path=path.name, sha256=sha(path), observations_sha256=sha(path.with_suffix('.npz')),
                    responses=len(r['core_responses']), decisions=len(r['decisions']), passed=True)
    finally:
        core.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('native', 'database', 'scripts', 'input', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--pattern', default='v*-0.json')
    a = p.parse_args()
    rows = [audit(path, a) for path in sorted(a.input.glob(a.pattern))]
    assert rows
    out = dict(passed=all(r['passed'] for r in rows), results=rows, native_sha256=sha(a.native))
    with a.output.open('x') as f: json.dump(out, f, indent=2)
    print(json.dumps(out))
