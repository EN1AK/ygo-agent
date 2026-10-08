"""One isolated real-core branch for the direct Laya RL mechanism pilot."""
import argparse
import json
import sqlite3
from pathlib import Path

from ygoai.rl.exercise_core import Core
from ygoai.rl.exercise_starters import ReferencePolicy, fixture, finished, grade
from ygoai.rl.jev_experiment import SCENARIOS, digest, visible_state


def run(args):
    spec = SCENARIOS[args.case]
    if args.branch not in spec['actions']:
        raise ValueError('unsupported macro action')
    core = Core(args.core, args.database, args.scripts)
    policy = ReferencePolicy(args.case, args.branch)
    root, stable = None, False
    lua, lp = fixture(args.case)
    try:
        core.start(lua, lp)
        for _ in range(160):
            prompt = core.next_prompt()
            if finished(core, policy, prompt):
                stable = True
                break
            response = policy.choose(core, prompt)
            if policy.root_event is not None and root is None:
                root = dict(state=core.state(), prompt=prompt, prefix=core.decisions[:])
                if args.root_only:
                    break
            core.respond(prompt, response)
        else:
            raise TimeoutError('native branch decision budget exhausted')
        if root is None:
            raise ValueError('root not reached')
        root_view = visible_state(root['state'], spec['player'])
        db = sqlite3.connect(f'file:{Path(args.database).resolve().as_posix()}?mode=ro', uri=True)
        try:
            texts = []
            for code in sorted({c['code'] for c in root_view['cards']}):
                row = db.execute('SELECT texts.name, texts.desc, datas.atk, datas.def, datas.level '
                                 'FROM texts JOIN datas ON texts.id=datas.id WHERE texts.id=?', (code,)).fetchone()
                if row is None:
                    raise ValueError(f'missing card text {code}')
                texts.append(dict(code=code, name=row[0], effect=row[1], printed_atk=row[2],
                                  printed_def_or_link_markers=row[3], printed_level_rank_or_link=row[4] & 255))
        finally:
            db.close()
        state = core.state()
        return dict(schema='jev-real-core-branch-v1', case=args.case, branch=args.branch,
                    root_digest=digest(root), visible_root=root_view, card_texts=texts,
                    visible_final=None if args.root_only else visible_state(state, spec['player']),
                    status='root_only' if args.root_only else grade(args.case, state, core.events, stable=stable),
                    runtime=core.runtime, script_hashes=core.assets,
                    events=core.events, decisions=core.decisions, stable_boundary=stable,
                    information_mode='known_synthetic_scenario', full_duel_policy=False)
    finally:
        core.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'database', 'scripts'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--case', choices=list(SCENARIOS), required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--root-only', action='store_true')
    args = p.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
