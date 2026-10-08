"""Real-core starter verification; no actor inference, optimizer or checkpoint I/O."""
import argparse
import hashlib
import json
from pathlib import Path

from ygoai.rl.exercise_core import Core, sha
from ygoai.rl.exercise_starters import CASES, ReferencePolicy, fixture, finished, grade
from ygoai.rl.capability_exercises import validate_candidate


def replay(args, case, branch):
    core = Core(args.core, args.database, args.scripts)
    policy = ReferencePolicy(case, branch)
    lua, lp = fixture(case)
    status, error, stable = 'unknown', None, False
    state = None
    root = None
    try:
        core.start(lua, lp)
        for _ in range(args.max_decisions):
            prompt = core.next_prompt()
            if finished(core, policy, prompt):
                stable = True
                break
            response = policy.choose(core, prompt)
            if policy.root_event is not None and root is None:
                root = {'state': core.state(), 'prompt': prompt,
                        'prefix_decisions': len(core.decisions)}
                if case.startswith('battle'):
                    direct = next(c['extra'] for c in prompt['attack'] if c['code'] == 8491308)
                    if bool(direct) != (case == 'battle'):
                        raise ValueError('Hayate direct-attack precondition mismatch')
            core.respond(prompt, response)
        else:
            raise TimeoutError('decision budget exhausted')
        state = core.state()
        status = grade(case, state, core.events, stable=stable)
        if policy.root_event is None:
            raise ValueError('exercise root not reached')
    except TimeoutError as exc:
        status, error = 'budget_exhausted', str(exc)
    except (ValueError, KeyError, StopIteration) as exc:
        status, error = 'invalid_fixture', str(exc)
    finally:
        core.close()
    record = dict(schema='capability-exercise-run-v1', case=case, branch=branch,
                  family=CASES[case]['family'], split='development',
                  source_kind='synthetic_mechanism_fixture', status=status, error=error,
                  root_event_index=policy.root_event, stable_boundary=stable, final_state=state,
                  root=root,
                  events=core.events, decisions=core.decisions, runtime=core.runtime,
                  scripts_sha256=core.assets, fixture_sha256=hashlib.sha256(lua.encode()).hexdigest(),
                  process_calls=core.calls, actor_interface_verified=False, training_ready=False)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--scripts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', choices=list(CASES))
    p.add_argument('--branch')
    p.add_argument('--candidate', type=Path, help='validated starter candidate from build_capability_exercises draft')
    p.add_argument('--max-decisions', type=int, default=160)
    args = p.parse_args()
    if args.max_decisions <= 0:
        p.error('max-decisions must be positive')
    candidate = None
    if args.candidate:
        candidate = validate_candidate(json.loads(args.candidate.read_text(encoding='utf-8')))
        if candidate['template'] is None:
            p.error('unresolved scene needs a verified fixture/replay adapter; cannot guess a template')
        if args.case and args.case != candidate['template']:
            p.error('candidate/template mismatch')
        args.case = candidate['template']
    if args.branch and (not args.case or args.branch not in CASES[args.case]['branches']):
        p.error('--branch requires --case and a known branch')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case, definition in CASES.items():
        if args.case and case != args.case:
            continue
        for branch, expected in definition['branches'].items():
            if args.branch and branch != args.branch:
                continue
            runs = [replay(args, case, branch) for _ in range(2)]
            identical = runs[0] == runs[1]
            wanted = 'success' if expected else 'verified_failure'
            passed = identical and all(r['status'] == wanted for r in runs)
            paths = []
            for i, run in enumerate(runs):
                path = args.output / f'{case}-{branch}-{i}.json'
                path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')
                paths.append({'path': path.name, 'sha256': sha(path)})
            result = dict(case=case, branch=branch, expected=wanted, actual=runs[0]['status'],
                          error=runs[0]['error'], identical_replays=identical, passed=passed, evidence=paths)
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {'schema': 'capability-exercise-verification-v1', 'results': results,
               'candidate_id': candidate['id'] if candidate else None,
               'passed': bool(results) and all(r['passed'] for r in results),
               'families': len({CASES[r['case']]['family'] for r in results}),
               'training_ready': False, 'actor_evaluated': False,
               'source_sha256': {str(Path(f).name): sha(f) for f in (__file__,
                   Path(__file__).parents[1] / 'ygoai/rl/exercise_core.py',
                   Path(__file__).parents[1] / 'ygoai/rl/exercise_starters.py',
                   Path(__file__).parents[1] / 'ygoai/rl/capability_exercises.py',
                   Path(__file__).parent / 'exercises/core_bridge.cpp')}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
