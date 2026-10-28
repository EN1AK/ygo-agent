"""Audit paired autonomous endpoints without conflating completion and strength."""
import argparse
from collections import Counter
from pathlib import Path
from contracts import ledger, read, sha, write


def audit_endpoint(folder, expected_updates):
    if (folder.parent / (folder.name + '.exit')).read_text(encoding='utf-8').strip() != '0':
        raise ValueError('training process did not exit zero')
    completion = read(folder / 'completion.json')
    model = folder / 'endpoint/model.flax_model'
    if not completion['completed'] or completion['updates'] != expected_updates or sha(model) != completion['checkpoint_sha256']:
        raise ValueError('endpoint completion mismatch')
    metrics = [__import__('json').loads(line) for line in (folder / 'endpoint/metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(metrics) != expected_updates or not all(x['finite'] for x in metrics):
        raise ValueError('incomplete or nonfinite optimizer updates')
    before = read(folder / 'before/summary.json')['results']
    after = read(folder / 'endpoint/evaluation/summary.json')['results']
    if not all(x['audit']['passed'] and x['status'] != 'invalid' for x in before + after):
        raise ValueError('invalid or unaudited autonomous evaluation')
    return completion, before, after


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--observation', required=True)
    a = p.parse_args()
    run = a.run.resolve()
    training = run / (a.observation + '-training')
    protocol = read(run / 'protocol.json')
    data = {arm: audit_endpoint(training / arm, protocol['training']['updates_per_arm'])
            for arm in ('control', 'repair')}
    def index(rows):
        return {(r['id'], r['seed']): r for r in rows}
    base = index(data['control'][1])
    if base != index(data['repair'][1]):
        raise ValueError('same-parent pretraining observations differ')
    control, repair = index(data['control'][2]), index(data['repair'][2])
    if not base.keys() == control.keys() == repair.keys():
        raise ValueError('unmatched evaluation roots')
    rows = [dict(id=k[0], seed=k[1], group=base[k]['group'], split=base[k]['split'],
                 before=base[k]['status'], control=control[k]['status'], repair=repair[k]['status'])
            for k in base]
    regressions = [r for r in rows if r['before'] == 'success' and r['repair'] != 'success']
    improvements = [r for r in rows if r['repair'] == 'success' and r['control'] != 'success']
    losses = [r for r in rows if r['control'] == 'success' and r['repair'] != 'success']
    summary = dict(completed=True, matched_roots=len(rows),
        counts={arm: dict(Counter(r[arm] for r in rows)) for arm in ('before', 'control', 'repair')},
        candidate_improvements=improvements, candidate_losses=losses,
        original_success_regressions=regressions, local_retention_passed=not regressions,
        endpoint_sha256={arm: data[arm][0]['checkpoint_sha256'] for arm in data},
        prior_holdout_is_regression=True, independent_transfer_passed=False,
        full_duel_gate_passed=False, post_ppo_retention_passed=False,
        promotion_eligible=False, rows=rows,
        uncertainty='correlated development roots; no population confidence claim')
    path = training / 'comparison.json'
    if path.exists():
        raise FileExistsError(path)
    write(path, summary)
    ledger(run, 'comparison', [path], claim='matched development evaluation; full objective remains unproven')
    print({k: summary[k] for k in ('matched_roots', 'counts', 'local_retention_passed', 'promotion_eligible')}, flush=True)


if __name__ == '__main__':
    main()
