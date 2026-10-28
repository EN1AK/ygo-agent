"""Search complete local-goal witnesses after actual autonomous deviations.

Exact-state offline teacher only. Reference suffixes propose continuations; an
independent full raw-core replay, not a state-key match, certifies each route.
"""
import argparse
import copy
from pathlib import Path
import sys
import time

from contracts import audit_opponent_policy, digest, ledger, passive_action, read, round_protocol, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts/studies/crystron_curriculum'))
from supervision_runtime import Runtime, array_digest, extract_trace, np, tensor_contract


def action_key(action):
    key = dict(action)
    # Own-deck prompt aliases may change after an equivalent search selection.
    # This proposes a candidate only; execution and full goal checks decide it.
    if str(key.get('spec', '')).isdecimal():
        key['spec'] = '#own-deck-alias'
        key['response'] = 0
    return key


def recover(rt, case, budget, output):
    item = case['item']
    if item['split'] != 'fit':
        raise ValueError('holdout cannot supply a recovery teacher')
    if sha(case['trace']) != case['trace_sha256'] or sha(case['checkpoint']) != case['checkpoint_sha256']:
        raise ValueError('failure evidence changed')
    trace = read(case['trace'])
    reference = rt.records[item['id']]
    if case.get('reference_record'):
        if sha(case['reference_record'])!=case['reference_record_sha256']:
            raise ValueError('retention reference changed')
        reference=read(case['reference_record'])
        if reference['source_sha256']!=item['source_sha256']:
            raise ValueError('retention reference belongs to another source')
    seat = reference['definition']['controlled']
    root = case['recovery_root']
    prefix = trace['decisions'][:root]
    attempts = []
    start = time.monotonic()

    def run(extra, suffix, capture=False, expected=None):
        env = rt.env(reference)
        decisions, arrays = [], []
        try:
            env.reset()
            actions = [d['action'] for d in prefix] + extra
            position = suffix
            for step in range(root + budget['max_suffix_steps']):
                if time.monotonic() - start > budget['seconds_per_root']:
                    return None, 'time_budget'
                snap = env.snapshot()
                events, state, responses = extract_trace(env, snap)
                if snap['invalid']:
                    return None, 'invalid'
                if step >= root and rt.achieved(item, state, snap, events):
                    record = dict(schema='ygo-reality-recovery-v1', definition=reference['definition'],
                        full_legal_menu=rt.full_legal_menu,native_sha256=rt.plan['native_sha256'],
                        source_sha256=reference['source_sha256'], decisions=decisions,
                        final=state, events=events, core_responses=responses,
                        core_seed=env.trace()['core_seed'], training_ready=False,
                        recovery_root=root, local_goal=item['goal'],
                        information_mode='offline_oracle_exact_state',
                        intervention=case.get('intervention', 'off_reference_recovery'),
                        failed_policy_sha256=case['checkpoint_sha256'],
                        failed_trace_sha256=case['trace_sha256'],
                        opponent_after_root='cancel optional, otherwise first legal')
                    if capture:
                        tensors = {k: np.stack([v[k] for v in arrays]) for k in arrays[0]}
                        return (record, tensors), 'success'
                    return digest(dict(final=state, core_responses=responses)), 'success'
                if snap['done'] or sum(e['op'] == 40 for e in events) > item['goal'].get('max_turn', 1):
                    return None, 'goal_not_reached'
                obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
                if step < root:
                    d = prefix[step]
                    if (snap['player'], snap['message'], snap['menu'], array_digest(obs)) != (
                            d['player'], d['message'], d['menu'], d['observation_sha256']):
                        raise ValueError('actual failed prefix did not reconstruct')
                    action = d['action']
                elif step < len(actions):
                    action = actions[step]
                elif snap['player'] != seat:
                    action = passive_action(snap['menu'])
                else:
                    while position < len(reference['decisions']) and reference['decisions'][position]['player'] != seat:
                        position += 1
                    if position == len(reference['decisions']):
                        return None, 'reference_exhausted'
                    d = reference['decisions'][position]
                    if d['message'] != snap['message']:
                        return None, 'suffix_prompt_mismatch'
                    options = [i for i, x in enumerate(snap['menu'])
                               if action_key(x) == action_key(d['menu'][d['action']])]
                    if not options:
                        return None, 'suffix_action_unavailable'
                    action = options[0]
                    position += 1
                if not 0 <= action < len(snap['menu']):
                    return None, 'extra_action_unavailable'
                if step >= root and snap['player'] != seat and action != passive_action(snap['menu']):
                    raise ValueError('search tried to change the fixed opponent policy')
                decisions.append(dict(step=step, player=snap['player'], message=snap['message'],
                    menu=snap['menu'], action=action, observation_sha256=array_digest(obs),
                    teaching=step >= root and snap['player'] == seat and len(snap['menu']) > 1))
                arrays.append({k: v[0].copy() for k, v in obs.items()})
                env.step(action)
            return None, 'decision_budget'
        finally:
            env.close()

    # Freeze proposals without reading any holdout trajectory. The entire
    # divergent prefix is preserved; an undo is a NEW explicit branch action.
    env = rt.env(reference)
    try:
        env.reset()
        for d in prefix:
            snap = env.snapshot()
            obs = {k: snap['observation'][k] for k in tensor_contract('structured-lite-v1')}
            if (snap['menu'], array_digest(obs)) != (d['menu'], d['observation_sha256']):
                raise ValueError('failed prefix mismatch')
            env.step(d['action'])
        snap = env.snapshot()
        extras = [[]]
        if snap['player'] == seat:
            extras += [[i] for i in range(len(snap['menu']))]
    finally:
        env.close()
    indexes = list(range(max(0, root - 2), min(len(reference['decisions']), root + 14)))
    indexes = [i for i in indexes if reference['decisions'][i]['player'] == seat]
    proposals = [(extra, i) for extra in extras for i in indexes]
    solution = None
    for extra, index in proposals[:budget['attempts']]:
        value, status = run(extra, index)
        attempts.append(dict(extra=extra, reference_index=index, status=status))
        if status == 'success':
            # Repeat the whole route before independent certification. The
            # expensive independent core is created only after actor closure.
            captured, second = run(extra, index, capture=True)
            if second != 'success':
                raise ValueError('successful teacher did not reproduce')
            record, arrays = captured
            opponent_policy = audit_opponent_policy(record, root)
            if digest(dict(final=record['final'], core_responses=record['core_responses'])) != value:
                raise ValueError('teacher repeat diverged')
            proof = rt.independent(record)
            if not proof['passed']:
                raise ValueError('independent recovery replay failed')
            if not any(d['teaching'] for d in record['decisions']):
                attempts[-1]['status'] = 'no_teaching_actions'
                continue
            record.update(training_ready=True, independent_endpoint_audit=proof,
                opponent_policy_audit=opponent_policy,
                admission=dict(split='fit', family=item['family'], same_actual_prefix=True,
                               repeated_adapter=True, correct_action_optimality_claim=False))
            target = output / (case['id'] + '.json')
            write(target, record)
            np.savez_compressed(target.with_suffix('.npz'), **arrays)
            solution = dict(id=case['id'], path=str(target), record_sha256=sha(target),
                observations_sha256=sha(target.with_suffix('.npz')), source_sha256=item['source_sha256'],
                family=item['family'], root_step=root, split='fit', group='repair', goal=item['goal'],
                own_steps=sum(d['player'] == seat for d in record['decisions']))
            break
        if status == 'time_budget':
            break
    report = dict(id=case['id'], training_ready=solution is not None,
        status='certified' if solution else 'unknown_budget_or_exhausted',
        seconds=time.monotonic() - start, attempts=attempts, solution=solution,
        negative_action_labels=False, source_case=case)
    write(output / (case['id'] + '-search.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observation', required=True)
    args = parser.parse_args()
    folder = args.run.resolve()
    protocol = round_protocol(folder, args.observation)
    cases = read(folder / args.observation / 'candidates.json')
    rt = Runtime(protocol['original'])
    output = folder / (args.observation + '-recovery')
    output.mkdir(exist_ok=False)
    reports = []
    for case in cases:
        if case['checkpoint_sha256'] != protocol['initial_checkpoint_sha256']:
            raise ValueError('failure came from a different policy generation')
        modes = protocol['search'].get('modes', ['off_reference_recovery'])
        budget = dict(protocol['search'])
        budget['attempts'] = max(1, budget['attempts'] // len(modes))
        budget['seconds_per_root'] /= len(modes)
        for mode in modes:
            candidate = dict(case, intervention=mode)
            if mode == 'counterfactual_root_witness':
                # Preserve the ACTUAL history up to the factual decision root.
                # A successful alternative is positive feasibility evidence;
                # the selected action is not labelled globally wrong.
                candidate.update(id=case['id']+'-counterfactual',
                    factual_action=read(case['trace'])['decisions'][case['recovery_root']-1]['action'],
                    recovery_root=case['recovery_root']-1,
                    prefix_actions=case['prefix_actions'][:-1])
            elif mode != 'off_reference_recovery':
                raise ValueError('unknown frozen intervention mode')
            report = recover(rt, candidate, budget, output)
            reports.append(report)
            print(candidate['id'], report['status'], report['seconds'], flush=True)
            if report['solution']:
                break
    solutions = [r['solution'] for r in reports if r['solution']]
    write(output / 'summary.json', dict(cases=len(cases), admitted=solutions,
        unknown=[r['id'] for r in reports if not r['solution']], information_mode='offline_oracle_exact_state'))
    ledger(folder, 'recovery', [output / 'summary.json'], admitted=len(solutions),
        claim='goal-reaching repeated adapter and independent raw-core witnesses')


if __name__ == '__main__':
    main()
