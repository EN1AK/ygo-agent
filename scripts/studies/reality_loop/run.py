"""Freeze and observe a reality-anchored research run; no production promotion."""
import argparse
import copy
import os
from pathlib import Path
import sys

from contracts import diagnose, first_deviation, ledger, read, round_protocol, sha, validate_splits, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts/studies/crystron_curriculum'))


def freeze(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = args.frozen.resolve()
    plan = read(original / 'plan.json')
    validate_splits(plan['rows'])
    for row in plan['rows']:
        for suffix, key in [('.json', 'record_sha256'), ('.npz', 'observations_sha256')]:
            path = (original / row['path']).with_suffix(suffix)
            if sha(path) != row[key]:
                raise ValueError('frozen input changed: ' + str(path))
    checkpoint = args.checkpoint.resolve()
    exam = None
    if args.independent_exam:
        exam_folder = args.independent_exam.resolve()
        exam_plan = read(exam_folder/'plan.json')
        fit_families = {r['family'] for r in plan['rows'] if r['split']=='fit'}
        if (any(r['split']!='holdout' for r in exam_plan['rows']) or
                fit_families & {r['family'] for r in exam_plan['rows']}):
            raise ValueError('independent exam overlaps fit families or contains fit rows')
        exam = dict(path=str(exam_folder), plan_sha256=sha(exam_folder/'plan.json'),
                    rows=len(exam_plan['rows']), checkpoint_selection=False)
    protocol = dict(schema='ygo-reality-loop-v1', original=str(original),
                    original_plan_sha256=sha(original / 'plan.json'),
                    initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=sha(checkpoint),
                    seed=28102026, rounds=2, roots_per_round=8,
                    observation_matmul_precision='highest',
                    independent_exam=exam,
                    search=dict(attempts=64, seconds_per_root=120, max_suffix_steps=384,
                                modes=['off_reference_recovery', 'counterfactual_root_witness']),
                    training=dict(updates_per_arm=512, learning_rate=2e-5,
                                  repair_mass=.5, optimizer='fresh Adam per arm',
                                  control='original curriculum only',
                                  candidate='original curriculum plus certified repairs'),
                    gate=dict(require_independent_replay=True, require_original_skills=True,
                              independent_transfer_pending=True, full_duels_pending=True,
                              post_ppo_retention_pending=True, promotion_eligible=False),
                    source_sha256=sha(__file__), contracts_sha256=sha(Path(__file__).with_name('contracts.py')),
                    prior_holdout_is_regression=True,
                    information_mode='offline_oracle_exact_state',
                    claims_excluded=['fair-play online search', 'global action optimality'])
    write(output / 'protocol.json', protocol)
    ledger(output, 'protocol', [output / 'protocol.json'], claim='frozen before new inference')
    print(output, flush=True)


def observe(args):
    from supervision_runtime import Runtime, jax
    folder = args.run.resolve()
    protocol = round_protocol(folder, args.name)
    # Old immutable protocols predate this field. Preserve their numerical
    # regime rather than silently changing a running experiment.
    precision = protocol.get('observation_matmul_precision', 'default')
    jax.config.update('jax_default_matmul_precision', precision)
    original = Path(protocol['original'])
    if sha(original / 'plan.json') != protocol['original_plan_sha256']:
        raise ValueError('original plan changed')
    output = folder / args.name
    output.mkdir(exist_ok=False)
    rt = Runtime(original)
    checkpoint = args.checkpoint.resolve()
    if sha(checkpoint) != protocol['initial_checkpoint_sha256']:
        raise ValueError('observation checkpoint is not the registered round parent')
    # Runtime.validate validates initial schema; do the same for each new endpoint.
    from ygoai.rl.checkpoint_compat import validate_checkpoint_compatibility
    validate_checkpoint_compatibility(checkpoint, observation_schema='structured-lite-v1',
        model_args=rt.modelargs, code_list_hash=rt.plan['code_list_sha256'],
        semantic_table_hash=rt.plan['semantics_sha256'], capacities=rt.capacities)
    results = rt.evaluate(rt.load(checkpoint), sha(checkpoint), output / 'evaluation')
    write(output / 'inference-contract.json', dict(matmul_precision=precision,
        devices=[str(d) for d in jax.devices()], checkpoint_sha256=sha(checkpoint),
        source_sha256=sha(__file__), protocol_sha256=sha(folder / 'protocol.json')))
    if not all(row['audit']['passed'] for row in results):
        raise ValueError('autonomous evidence failed independent replay')
    diagnoses = []
    candidates = []
    items = {r['id']: r for r in rt.plan['rows']}
    for row in results:
        path = output / 'evaluation' / row['path']
        trace = read(path)
        diagnosis = dict(id=row['id'], split=row['split'], seed=row['seed'], **diagnose(trace))
        diagnoses.append(diagnosis)
        item = items[row['id']]
        if row['split'] != 'fit' or row['status'] in ('success', 'invalid') or row['seed'] not in (None, 1):
            continue
        root = first_deviation(trace, rt.records[row['id']])
        if root is not None:
            candidates.append(dict(id=row['id'], item=item, trace=str(path),
                trace_sha256=sha(path), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                recovery_root=root, prefix_actions=[d['action'] for d in trace['decisions'][:root]],
                diagnosis=diagnosis['classification'], training_ready=False))
    # Deterministic balanced task order, never score/select on a holdout.
    candidates.sort(key=lambda x: (x['item']['group'], x['id']))
    by_group = {}
    for case in candidates:
        by_group.setdefault(case['item']['group'], []).append(case)
    selected = []
    while by_group and len(selected) < protocol['roots_per_round']:
        for group in list(by_group):
            selected.append(by_group[group].pop(0))
            if not by_group[group]:
                del by_group[group]
            if len(selected) == protocol['roots_per_round']:
                break
    write(output / 'diagnoses.json', diagnoses)
    write(output / 'candidates.json', selected)
    ledger(folder, 'observation', [output / 'evaluation/summary.json', output / 'diagnoses.json', output / 'candidates.json', output / 'inference-contract.json'],
           checkpoint_sha256=sha(checkpoint), recovery_candidates=len(selected),
           claim='audited autonomous development evaluation; no wrong-action labels')
    print('RECOVERY_CANDIDATES', len(selected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze')
    for name in ('frozen', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--independent-exam', type=Path)
    p = commands.add_parser('observe')
    for name in ('run', 'checkpoint'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--name', required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'observe': observe}[args.command](args)


if __name__ == '__main__':
    main()
