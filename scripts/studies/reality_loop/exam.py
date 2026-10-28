"""Freeze genuinely unexposed connected components before model evaluation.

Intake/adapter conversion uses human evidence only. Rejected conversions stay
visible in the frozen intake coverage and cannot be silently replaced by easier
recordings. Exam rows are never admitted to a teaching curriculum.
"""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

from contracts import digest, ledger, read, sha, write

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts/studies/crystron_curriculum'))


def freeze(args):
    from freeze_supervision import components
    run = args.run.resolve()
    output = run / 'independent-exam'
    output.mkdir(exist_ok=False)
    protocol = read(run / 'protocol.json')
    old = read(Path(protocol['original']) / 'plan.json')
    study = args.study.resolve()
    catalog_path = study / 'source/catalog.json'
    catalog = read(catalog_path)
    families = components(catalog)
    exposed = {r['family'] for r in old['rows']}
    exposed |= set(old['split']['known_components']) | set(old['split']['holdout_components'])
    fresh = [r for r in catalog if families[r['source_sha256']] not in exposed]
    # All previously audited complete games; no model score or preference is
    # involved. Partial games remain available for a future local-goal exam.
    selected = sorted([dict(row, family=families[row['source_sha256']])
                       for row in fresh if row['audit_status'] == 'complete'],
                      key=lambda r: digest(r['source_sha256']))
    if not selected:
        raise ValueError('no independent complete components; obtain new evidence')
    intake = dict(catalog_sha256=sha(catalog_path), exposed_components=sorted(exposed),
        unexposed_rows=len(fresh), unexposed_components=len({families[r['source_sha256']] for r in fresh}),
        selected=selected, selection='all audited complete recordings in unexposed components',
        no_model_inference=True, no_checkpoint_selection=True,
        opponents='fixed prefix to first own main; passive after root, not original human policy',
        invalids_not_replaced=True)
    write(output / 'intake.json', intake)
    ledger(run, 'independent_exam_protocol', [output / 'intake.json'],
        claim='unexposed player/deck/match components frozen without model scores')
    print('INDEPENDENT_INTAKE', len(selected), flush=True)


def convert(args):
    run = args.run.resolve()
    folder = run / 'independent-exam'
    intake = read(folder / 'intake.json')
    study = args.study.resolve()
    protocol = read(run / 'protocol.json')
    original = read(Path(protocol['original']) / 'plan.json')
    source = Path(original['source_root'])
    output = folder / 'converted'
    output.mkdir(exist_ok=False)
    rows, coverage = [], []
    for entry in intake['selected']:
        source_hash = entry['source_sha256']
        proofs = [p for p in (source / 'audit-repaired-v7' / (source_hash + '.json'),
                             study / 'audit-additional-v1' / (source_hash + '.json')) if p.exists()]
        if len(proofs) != 1 or sha(proofs[0]) != entry['audit_sha256']:
            raise ValueError('independent intake proof identity mismatch')
        target = output / source_hash[:12]
        command = [sys.executable, str(PROJECT / 'scripts/studies/crystron_curriculum/convert_replay.py'),
            '--source', str(source), '--catalog', str(study / 'source'),
            '--proof', str(proofs[0]), '--runtime', original['runtime'], '--output', str(target)]
        # Explicit process bound; a timed-out case remains part of the exam's
        # coverage denominator and is not a strategically negative example.
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            case = dict(source_sha256=source_hash, exit_code=completed.returncode,
                        stdout=completed.stdout[-1500:], stderr=completed.stderr[-1500:])
        except subprocess.TimeoutExpired:
            case = dict(source_sha256=source_hash, status='conversion_timeout_unknown')
            coverage.append(case)
            continue
        record_path = target / 'record.json'
        record = read(record_path) if record_path.exists() else {}
        if completed.returncode or record.get('status') != 'converted' or not record.get('input_verified'):
            case['status'] = 'conversion_rejected'
        else:
            seat = record['definition']['controlled']
            roots = [i for i, d in enumerate(record['decisions'])
                     if d['message'] == 11 and d['player'] == seat and d['turn_player'] == seat]
            if not roots:
                case['status'] = 'no_own_main_root'
            else:
                rows.append(dict(id=source_hash[:12], path=str(record_path),
                    record_sha256=sha(record_path), observations_sha256=sha(record_path.with_suffix('.npz')),
                    split='holdout', family=entry['family'], source_sha256=source_hash,
                    group='independent_practice', root_step=roots[0],
                    goal=dict(kind='win_after_first_own_main'),
                    own_steps=sum(d['player'] == seat for d in record['decisions'])))
                case['status'] = 'converted_needs_independent_endpoint_replay'
        coverage.append(case)
        print(source_hash[:12], case['status'], flush=True)
    write(folder / 'conversion-coverage.json', coverage)
    if not rows:
        raise ValueError('no independent exam rows converted; retain failure coverage')
    plan = copy.deepcopy(original)
    plan.update(id='reality-loop-independent-components-v1', rows=rows,
                parent=protocol['initial_checkpoint'], parent_sha256=protocol['initial_checkpoint_sha256'],
                stages=[], promotion_eligible=False)
    plan['evaluation'].update(local_holdout_count=len(rows), local_fit_count=0,
                              fit_practice_count=0, holdout_practice_count=0,
                              resource_decision_budget=1024)
    frozen = folder / 'frozen'
    frozen.mkdir()
    write(frozen / 'plan.json', plan)
    ledger(run, 'independent_exam_conversion', [folder / 'intake.json',
        folder / 'conversion-coverage.json', frozen / 'plan.json'],
        converted=len(rows), intake_cases=len(intake['selected']),
        claim='adapter conversion only; independent endpoint replay still required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'convert'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'convert': convert}[args.command](args)


if __name__ == '__main__':
    main()
