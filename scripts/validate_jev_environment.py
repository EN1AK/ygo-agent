"""Validate actual roots, probe feedback and settled rewards before model training."""
import argparse
import json
import random
from pathlib import Path

from ygoai.rl.exercise_starters import CASES
from ygoai.rl.jev_experiment import BranchEnvironment, SCENARIOS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'database', 'scripts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    env = BranchEnvironment(args.core, args.database, args.scripts)
    reports = []
    for case, spec in SCENARIOS.items():
        for action in spec['actions']:
            env.reset(case)
            probe_reward = env.step(('probe', action))
            request = env.request(random.Random(1))
            assert len(request['state']['simulated_branches']) == 1
            assert not env.done
            reward = env.step(('commit', action))
            assert env.success == CASES[case]['branches'][action]
            reports.append(dict(case=case, action=action, reward=reward,
                                probe_reward=probe_reward, root_digest=env.root_digest,
                                evidence=env.evidence))
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(dict(passed=True, branches=len(reports), reports=reports,
                       scope='known synthetic scenarios only'), f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(passed=True, branches=len(reports))))


if __name__ == '__main__':
    main()
