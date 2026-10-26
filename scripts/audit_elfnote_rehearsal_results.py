"""Recount the complete frozen rehearsal study from raw evidence, without inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_digest(path):
    import numpy as np
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            array = np.asarray(arrays[name])
            digest.update(name.encode())
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def verify_old_proof(path, audit):
    path = Path(path)
    assert audit['passed'] and audit['sha256'] == sha(path)
    if 'observations_sha256' in audit:
        assert audit['observations_sha256'] == sha(path.with_suffix('.npz'))
    else:
        # This legacy fixture binds decompressed arrays through the hashed
        # record. Missing compressed-file hashes in other fixtures are errors.
        assert path.stem == '4' and audit['same_state'] and audit['same_events']
        record = read(path)
        assert record['case'] == 'combo' and record['status'] == audit['status']
        assert record['status'] in ['success', 'verified_failure']
        assert archive_digest(path.with_suffix('.npz')) == record['observation_archive_digest']


def recount(episodes):
    counts = Counter()
    for row in episodes:
        reward = row['reward']
        assert math.isfinite(reward)
        assert row['invalid_game'] in [0, 1] and row['termination_reason'] in [0, 1, 2, 3]
        outcome = ('invalid' if row['invalid_game'] or row['termination_reason'] != 1
                   else 'win' if reward > 0 else 'loss' if reward < 0 else 'draw')
        assert row['outcome'] == outcome
        counts[outcome] += 1
    n = len(episodes)
    valid = n - counts['invalid']
    return dict(attempts=n, valid_games=valid, invalid_games=counts['invalid'],
                wins=counts['win'], losses=counts['loss'], draws=counts['draw'],
                win_rate=counts['win'] / valid if valid else None)


def paired_delta(before, after):
    """Check actual environment seeds as well as the declared seed/seat keys."""
    def index(rows):
        result = {}
        for batch in rows:
            for episode in batch['episodes']:
                key = (batch['seed'], episode['candidate_seat'], episode['environment_index'])
                assert key not in result
                result[key] = (batch['environment_seed'], episode['outcome'])
        return result
    left, right = index(before), index(after)
    assert left and left.keys() == right.keys()
    assert all(left[k][0] == right[k][0] for k in left)
    return sum((right[k][1] == 'win') - (left[k][1] == 'win') for k in left) / len(left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'training', 'evaluation', 'retention', 'probe', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    protected = {}

    def evidence(path):
        path = Path(path).resolve()
        value = read(path)
        protected[str(path)] = sha(path)
        return value

    for path in [Path(str(args.training) + '.pipeline.exit'),
                 Path(str(args.evaluation) + '.pipeline.exit'),
                 Path(str(args.probe) + '.pipeline.exit')]:
        assert path.read_text().strip() == '0'
        protected[str(path.resolve())] = sha(path)
    training_audit = evidence(args.training / 'independent-training-audit.json')
    assert training_audit['passed']
    manifest = evidence(args.evaluation / 'manifest.json')
    plan = manifest['plan']
    assert plan['roles'] == ['control', 'rehearse']
    release = args.root / plan['runtime_release']
    runtime = evidence(release / 'frozen-manifest.json')
    assert runtime['native_sha256'] == plan['native_sha256']
    assert all(sha(release / item['path']) == item['sha256'] for item in runtime['artifacts'])
    assert sha(release / 'checkpoints/parent.flax_model') == plan['history_checkpoint_sha256']
    models = {'initial': plan['course_checkpoint_sha256']}
    for role in plan['roles']:
        proof = evidence(args.training / role / 'verification.json')
        extra = evidence(args.training / role / 'rehearsal-verification.json')
        assert proof['passed'] and proof['process_exit_zero'] and proof['zero_nonfinite_updates']
        assert proof['updates'] == extra['updates'] == 98 and extra['passed']
        assert proof['fitted_env_steps'] == plan['total_timesteps_per_arm']
        assert proof['checkpoint_sha256'] == extra['model'] == sha(proof['checkpoint'])
        assert extra['model'] == training_audit['roles'][role]['checkpoint_sha256']
        models[role] = proof['checkpoint_sha256']
    summary = evidence(args.evaluation / 'summary.json')
    assert summary['passed']
    rows = {model: {op: [] for op in ['history', 'greedy']} for model in models}
    expected = {(model, op, seed, seat) for model in models for op in ['history', 'greedy']
                for seed in plan['evaluation']['seeds'] for seat in plan['evaluation']['seats']}
    seen = set()
    for i, job in enumerate(manifest['jobs']):
        key = (job['model'], job['opponent'], job['seed'], job['seat'])
        assert key in expected and key not in seen
        seen.add(key)
        folder = args.evaluation / f'{i:02d}-{key[0]}-{key[1]}-{key[2]}-seat{key[3]}'
        assert evidence(folder / 'exit.json')['returncode'] == 0
        batch = evidence(folder / 'result.json')
        assert batch['checkpoint_sha256'] == models[key[0]]
        assert batch['opponent'] == key[1]
        assert batch['seed'] == key[2] and batch['candidate_seat'] == key[3]
        assert batch['native_sha256'] == plan['native_sha256']
        assert batch['runtime_manifest_sha256'] == sha(release / 'frozen-manifest.json')
        assert batch['deck_sha256'] == sha(release / 'decks/replay-elfnote.ydk')
        assert batch['opponent_checkpoint_sha256'] == (plan['history_checkpoint_sha256'] if key[1] == 'history' else None)
        assert len(batch['episodes']) == plan['evaluation']['episodes_per_seed_seat'] == 32
        assert {r['environment_index'] for r in batch['episodes']} == set(range(32))
        assert all(r['candidate_seat'] == key[3] for r in batch['episodes'])
        assert all(batch[k] == v for k, v in recount(batch['episodes']).items())
        rows[key[0]][key[1]].append(batch)
    assert seen == expected and len(seen) == 48
    games = {}
    for model, opponents in rows.items():
        games[model] = {}
        for op, batches in opponents.items():
            value = recount([r for batch in batches for r in batch['episodes']])
            assert all(summary['summary'][model][op][k] == v for k, v in value.items())
            games[model][op] = value
    for name, before, after in [('control_vs_initial', 'initial', 'control'),
                                ('rehearse_vs_initial', 'initial', 'rehearse'),
                                ('rehearse_vs_control', 'control', 'rehearse')]:
        for op in ['history', 'greedy']:
            delta = paired_delta(rows[before][op], rows[after][op])
            value = summary['paired_differences'][name][op]
            assert value['paired_attempts'] == 256 and value['seed_blocks'] == 4
            assert math.isclose(value['win_per_attempt_delta'], delta, abs_tol=1e-12)
    cross = []
    for seed in plan['evaluation']['seeds']:
        for seat in plan['evaluation']['seats']:
            folder = args.evaluation / f'cross-rehearse-control-{seed}-seat{seat}'
            assert evidence(folder / 'exit.json')['returncode'] == 0
            batch = evidence(folder / 'result.json')
            assert batch['checkpoint_sha256'] == models['rehearse'] and batch['opponent_checkpoint_sha256'] == models['control']
            assert batch['seed'] == seed and batch['candidate_seat'] == seat
            assert len(batch['episodes']) == 32 and {r['environment_index'] for r in batch['episodes']} == set(range(32))
            assert all(r['candidate_seat'] == seat for r in batch['episodes'])
            cross.extend(batch['episodes'])
    cross_counts = recount(cross)
    assert cross_counts['attempts'] == 256
    assert all(summary['cross_arm'][k] == v for k, v in cross_counts.items())
    retention = {}
    for model in models:
        folder = args.retention / model
        proof = evidence(folder / 'final-verification.json')
        m = evidence(folder / 'manifest.json')
        assert proof['passed'] and proof['checkpoint_sha256'] == m['checkpoint_sha256'] == models[model]
        assert all(sha(args.root / path) == digest for path, digest in m['runtime'].items())
        normal = evidence(folder / 'normal/summary.json')
        functional = evidence(folder / 'functional/summary.json')
        assert [(r['name'], r['split']) for r in normal['cases']] == [(r['name'], r['split']) for r in m['normal_cases']]
        counts = {split: sum(r['status'] == 'success' for r in normal['cases'] if r['split'] == split) for split in proof['normal'] if split != 'old'}
        assert len(normal['old']) == len(proof['old_audits']) == 5
        assert {Path(r['path']).stem for r in proof['old_audits']} == {str(i) for i in range(5)}
        counts['old'] = sum(r['success'] for r in normal['old'])
        assert counts == proof['normal']
        for i, row in enumerate(normal['cases']):
            record = evidence(folder / 'normal' / f'case-{i:02}.json')
            assert record['case'] == row['name'] and record['status'] == row['status']
        for old in proof['old_audits']:
            path = Path(old['path']); audit = old['proof']
            assert path.resolve().parent == (folder / 'normal/old').resolve()
            verify_old_proof(path, audit)
            protected[str(path.resolve())] = sha(path)
            protected[str(path.with_suffix('.npz').resolve())] = sha(path.with_suffix('.npz'))
        assert [(r['name'], r['mode']) for r in functional['cases']] == [(r['name'], mode) for r in m['functional_cases'] for mode in ['scripted', 'policy']]
        scores = {split: {mode: sum(r['status'] == 'success' for r in functional['cases'] if r['split'] == split and r['mode'] == mode) for mode in ['scripted', 'policy']} for split in functional['scores']}
        assert scores == functional['scores'] == proof['functional']
        for row in functional['cases']:
            path = Path(row['path'])
            assert path.resolve().parent == (folder / 'functional').resolve()
            record = evidence(path)
            assert row['record_sha256'] == sha(path) and row['observations_sha256'] == sha(path.with_suffix('.npz'))
            assert record['case'] == row['name'] and record['status'] == row['status'] and record['probe_mode'] == row['mode']
        audits = list((folder / 'normal').glob('*.audit.json')) + list((folder / 'functional').glob('*.audit.json'))
        assert len(audits) == proof['new_audits'] == len(m['normal_cases']) + 2 * len(m['functional_cases'])
        for path in audits:
            audit = evidence(path); record = path.with_name(path.name.replace('.audit.json', '.json'))
            assert audit['passed'] and audit['record_sha256'] == sha(record) and audit['observations_sha256'] == sha(record.with_suffix('.npz'))
        retention[model] = dict(normal=counts, functional=scores, retention_gate=proof['retention_gate'])
    probe = evidence(args.probe / 'final-verification.json')
    probe_manifest = evidence(args.probe / 'manifest.json')
    assert probe['passed'] and probe['independent_core_audits'] == 72
    assert probe_manifest['plan']['models'] == ['initial', 'control', 'rehearse']
    probe_plan = probe_manifest['plan']; probe_summary = evidence(args.probe / 'summary.json')
    terminal = {model: {challenge['name']: {mode: 0 for mode in ['scripted', 'pass', 'policy']} for challenge in probe_plan['challenges']} for model in models}
    for model in models:
        assert probe_plan[model + '_sha256'] == models[model]
        for case in probe_plan['cases']:
            for challenge in probe_plan['challenges']:
                folder = args.probe / f'{model}-{case}-{challenge["name"]}'
                for mode in ['scripted', 'pass', 'policy']:
                    path = folder / (mode + '.json'); record = evidence(path)
                    audit = evidence(folder / (mode + '.audit.json'))
                    assert audit['passed'] and audit['record_sha256'] == sha(path) and audit['observations_sha256'] == sha(path.with_suffix('.npz'))
                    assert record['case'] == case and record['mode'] == mode and not record['training_ready']
                    terminal[model][challenge['name']][mode] += record['status'] == 'success'
    assert probe_summary['audits'] == 72 and terminal == probe['scores'] == probe_summary['scores']
    for path, digest in probe_manifest['protected'].items():
        assert sha(path) == digest
    assert all(sha(path) == digest for path, digest in protected.items())
    result = dict(passed=True, models=models, primary_attempts=1536, cross_attempts=256,
                  games=games, cross=cross_counts, retention=retention, terminal_probe=terminal,
                  protected=protected, auditor_sha256=sha(Path(__file__)), production_default_replaced=False,
                  scope='Raw arithmetic, endpoint, actual-seed and artifact consistency audit; relies on separate independent core replays for physical legality. Success of audit does not imply ability/retention gate success or promotion. All exercise families are exposed; one training seed only.')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'protected'}), flush=True)


if __name__ == '__main__':
    main()
