"""Matched-update control/repair arms with immutable fit-only admission."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from contracts import ledger, read, sha, validate_splits, write

PROJECT = Path(__file__).resolve().parents[3]


def prepare(run, observation, checkpoint):
    protocol = read(run / 'protocol.json')
    original = Path(protocol['original'])
    if sha(original / 'plan.json') != protocol['original_plan_sha256']:
        raise ValueError('original curriculum changed')
    observed = read(run / observation / 'evaluation/summary.json')
    if observed['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('teacher failures did not come from this training parent')
    recovery = run / (observation + '-recovery') / 'summary.json'
    repairs = read(recovery)['admitted']
    if not repairs:
        raise ValueError('no certified repairs; diagnose search instead of running empty treatment')
    plan = read(original / 'plan.json')
    holdout_families = {r['family'] for r in plan['rows'] if r['split'] == 'holdout'}
    holdout_sources = {r['source_sha256'] for r in plan['rows'] if r['split'] == 'holdout'}
    for row in plan['rows']:
        row['path'] = str((original / row['path']).resolve())
    original_mix = plan['stages'][-1]['mix']
    weight = sum(original_mix.values())
    original_mix = {g: w / weight for g, w in original_mix.items()}
    # An earlier round's admitted repairs are ordinary protected curriculum in
    # the next round. Never merge them into the new treatment group or silently
    # change the control's gradients when repeated feedback finds another root.
    new_group = 'repair' if 'repair' not in original_mix else 'repair-' + observation
    if new_group in original_mix:
        raise ValueError('new repair group already exists; use a fresh observation name')
    for repair in repairs:
        record = read(repair['path'])
        if (repair['split'] != 'fit' or repair['family'] in holdout_families
                or repair['source_sha256'] in holdout_sources):
            raise ValueError('repair derived from a holdout')
        if (not record['training_ready'] or not record['independent_endpoint_audit']['passed']
                or sha(repair['path']) != repair['record_sha256']
                or sha(Path(repair['path']).with_suffix('.npz')) != repair['observations_sha256']):
            raise ValueError('repair admission failed')
        if any(d['teaching'] for d in record['decisions'][:repair['root_step']]):
            raise ValueError('actual failed prefix cannot receive repair supervision')
        row = dict(repair, id=new_group + '-' + repair['id'], group=new_group)
        plan['rows'].append(row)
    validate_splits(plan['rows'])
    prepared = run / (observation + '-training')
    prepared.mkdir(exist_ok=False)
    for arm in ('control', 'repair'):
        folder = prepared / (arm + '-inputs')
        folder.mkdir()
        p = copy.deepcopy(plan)
        p.update(id='reality-loop-' + observation + '-' + arm,
                 parent=str(checkpoint), parent_sha256=sha(checkpoint))
        mix = original_mix
        if arm == 'repair':
            mass = protocol['training']['repair_mass']
            mix = {g: (1 - mass) * w for g, w in original_mix.items()}
            mix[new_group] = mass
        p['learning_rate'] = protocol['training']['learning_rate']
        p['stages'] = [dict(name='endpoint', updates=protocol['training']['updates_per_arm'], mix=mix)]
        p['evaluation']['local_fit_count'] = 8  # every admitted repair is evaluated in both arms
        p['promotion_eligible'] = False
        write(folder / 'plan.json', p)
    write(prepared / 'admission.json', dict(recovery_summary_sha256=sha(recovery),
        parent_sha256=sha(checkpoint), repairs=len(repairs),
        identical_evaluation_rows=True, control_repair_gradient=False, new_repair_group=new_group,
        curriculum_normalization='equal optimizer updates; total FLOPs and teacher cost reported separately'))
    ledger(run, 'training_protocol', [prepared / 'admission.json',
        prepared / 'control-inputs/plan.json', prepared / 'repair-inputs/plan.json'],
        claim='same parent and updates; repair gradients isolated to candidate arm')
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run, checkpoint = args.run.resolve(), args.checkpoint.resolve()
    prepared = prepare(run, args.observation, checkpoint)
    if args.prepare_only:
        return
    for arm in ('control', 'repair'):
        command = [sys.executable, str(PROJECT / 'scripts/studies/crystron_curriculum/train_supervision.py'),
            '--frozen', str(prepared / (arm + '-inputs')), '--output', str(prepared / arm)]
        with (prepared / (arm + '.log')).open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            write(prepared / (arm + '-process.json'), dict(pid=process.pid, command=command))
            code = process.wait()
        (prepared / (arm + '.exit')).write_text(str(code) + '\n', encoding='utf-8')
        if code:
            raise RuntimeError(arm + ' failed; preserve its directory and diagnose')
        completion = prepared / arm / 'completion.json'
        data = read(completion)
        if not data['completed']:
            raise ValueError('training did not complete')
        ledger(run, 'training_endpoint', [completion, prepared / arm / 'endpoint/summary.json'],
               arm=arm, claim='finite completed optimizer updates and audited autonomous evaluation')


if __name__ == '__main__':
    main()
