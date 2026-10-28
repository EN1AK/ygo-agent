"""Revoke opponent-incompatible witnesses without rewriting completed runs."""
import argparse
import copy
from pathlib import Path

from contracts import audit_opponent_policy, ledger, read, sha, write
from compare import audit_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'frozen', 'parent', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.run, args.frozen, args.parent, args.output = (
        getattr(args, k).resolve() for k in ('run', 'frozen', 'parent', 'output'))
    control = args.run/'round1-training/control'
    protocol = read(args.run/'protocol.json')
    completion, _, _ = audit_endpoint(control, protocol['training']['updates_per_arm'])
    control_plan = read(args.run/'round1-training/control-inputs/plan.json')
    if args.parent != (control/'endpoint/model.flax_model').resolve() or sha(args.parent) != completion['checkpoint_sha256']:
        raise ValueError('next parent must be the audited control-only endpoint')
    if any(g.startswith('repair') for stage in control_plan['stages'] for g in stage['mix']):
        raise ValueError('control parent used repair gradients')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = copy.deepcopy(read(args.frozen/'plan.json'))
    admitted, quarantined = [], []
    for item in plan['rows']:
        path = args.frozen/item['path']
        if sha(path) != item['record_sha256'] or sha(path.with_suffix('.npz')) != item['observations_sha256']:
            raise ValueError('frozen teacher inputs changed')
        if item['group'].startswith('repair'):
            try:
                audit_opponent_policy(read(path), item['root_step'])
            except ValueError as error:
                quarantined.append(dict(id=item['id'], reason=str(error), record_sha256=sha(path),
                    original_path=str(path), training_ready=False))
                continue
        admitted.append(dict(item, path=str(path.resolve())))
    present = {r['group'] for r in admitted if r['split']=='fit'}
    for stage in plan['stages']:
        mix = {g:w for g,w in stage['mix'].items() if g in present}
        total = sum(mix.values())
        if not total:
            raise ValueError('quarantine removed all curriculum groups')
        stage['mix'] = {g:w/total for g,w in mix.items()}
    plan.update(rows=admitted, parent=str(args.parent), parent_sha256=sha(args.parent),
        teacher_quarantine=dict(source_plan_sha256=sha(args.frozen/'plan.json'),
                                records=quarantined, no_invalid_teacher_gradients_in_parent=True))
    write(args.output/'plan.json', plan)
    write(args.output/'quarantine.json', dict(records=quarantined, admitted_rows=len(admitted),
        parent_sha256=sha(args.parent), previous_artifacts_preserved=True,
        control_plan_sha256=sha(args.run/'round1-training/control-inputs/plan.json'),
        control_completion_sha256=sha(control/'completion.json')))
    references = []
    for line in (args.run/'ledger.jsonl').read_text(encoding='utf-8').splitlines():
        import json
        entry = json.loads(line)
        if entry['kind'] in ('recovery','training_protocol','training_endpoint','comparison'):
            references.append(entry['entry_sha256'])
    ledger(args.run, 'correction', [args.output/'quarantine.json', args.output/'plan.json'],
        corrects_entries=references,
        claim='previously admitted teacher changed fixed opponent policy; revoke recovery-teaching eligibility',
        consequence='retain old experiment as invalid-condition evidence; next parent uses control-only gradients')
    print('ADMITTED', len(admitted), 'QUARANTINED', len(quarantined), flush=True)


if __name__=='__main__':
    main()
