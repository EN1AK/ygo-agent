"""Audit a completed bounded elfnote study without selecting or changing a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ygoai.rl.exercise_core import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'native', 'database', 'scripts', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--old-record', type=Path)
    a = p.parse_args()
    if a.old_record:
        # A separate process per trajectory also isolates native callbacks.
        if a.old_record.stem == '4':
            from scripts.verify_exercise_actor_responses import verify
            result = verify(a.old_record, a)
        else:
            from scripts.audit_teaching_exercise import audit
            result = audit(a.old_record, a)
        assert result['passed'], result
        print(json.dumps(result))
        return
    assert Path(str(a.study) + '.exit').read_text().strip() == '0'
    manifest, summary = read(a.study/'manifest.json'), read(a.study/'summary.json')
    plan = manifest['plan']
    assert manifest['execute'] and summary['protected_unchanged']
    assert all(sha(path) == value for path, value in manifest['protected'].items())
    source = read(a.source/'source-manifest.json')
    for path, expected in source['all_files'].items():
        actual = hashlib.sha256((a.source/path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        assert actual == expected, path
    assert sha(a.source/'scripts/teach_elfnote_combo.py') == manifest['source_sha256']
    parity = read(a.study/'parity.json')
    assert parity['passed'] and parity['old_observations_identical'] and parity['max_logit_error'] < .002
    curves, models = [], []
    for arm in summary['arms']:
        name = arm['arm']['name']
        metrics = [json.loads(line) for line in (a.study/name/'metrics.jsonl').read_text().splitlines()]
        assert [m['update'] for m in metrics] == list(range(1, plan['updates_per_arm']+1))
        assert all(np.isfinite([m['loss'], m['grad_norm']] + m['ce'] + m['kl']).all() for m in metrics)
        checkpoint = a.study/name/'model.flax_model'
        assert sha(checkpoint) == arm['checkpoint_sha256']
        assert read(a.study/name/'endpoint/summary.json') == arm['result']
        assert arm['seconds'] < plan['max_seconds_per_arm']
        models.append(dict(arm=name, sha256=sha(checkpoint), seconds=arm['seconds'], final_metric=metrics[-1]))
        for step in plan['probe_updates']:
            probe = read(a.study/name/f'probe-{step}/summary.json')
            curves.append(dict(arm=name, update=step,
                               original=sum(c['status']=='success' for c in probe['cases']),
                               old=sum(c['success'] for c in probe['old'])))
    assert summary['seconds'] < plan['max_total_seconds']
    assert read(a.study/'before/summary.json') == summary['before']
    new_proofs = list(a.study.rglob('*.audit.json'))
    expected = len(plan['demonstrations']) + (1 + len(plan['arms'])) * (len(plan['demonstrations']) + len(plan['reserved']))
    assert len(new_proofs) == expected
    for path in new_proofs:
        proof = read(path)
        record = path.with_name(path.name.replace('.audit.json', '.json'))
        assert proof['passed'] and proof['status'] == 'actor_audited'
        assert proof['record_sha256'] == sha(record)
        assert proof['observations_sha256'] == sha(record.with_suffix('.npz'))
        assert proof['native_sha256'] == sha(a.native)
    old_paths = sorted(a.study.glob('**/old/[0-4].json'))
    assert len(old_paths) == 5 * (1 + len(plan['arms']) * (1 + len(plan['probe_updates'])))
    old_proofs = []
    for path in old_paths:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                 '--old-record', str(path)], check=True, capture_output=True, text=True)
        proof = json.loads(result.stdout.strip().splitlines()[-1])
        proof['path'] = str(path.relative_to(a.study))
        old_proofs.append(proof)
    result = dict(passed=True, source_commit=source['commit'], canonical_source_files=len(source['all_files']),
                  native_sha256=sha(a.native), new_actor_audits=len(new_proofs), old_audits=old_proofs,
                  models=models, learning_curve=curves, seconds=summary['seconds'],
                  protected_unchanged=True, summary_sha256=sha(a.study/'summary.json'),
                  scope='Five replay references and development probes; no global retention or duel-strength claim.')
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(passed=True, new_audits=len(new_proofs), old_audits=len(old_proofs), models=models)))


if __name__ == '__main__':
    main()
