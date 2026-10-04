"""Same-trajectory context and public outcome capture; no fitting or policy change."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from collect_interaction_windows import chains, label, partition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def raw_chains(rows):
    current = None
    for row in rows:
        p = bytes.fromhex(row['packet_hex'])
        assert p[0] == 1 and p[1] == row['message']
        if p[1] == 70:
            assert len(p) == 18
            if current is None:
                current = {'context_step': row['after_decision_step'], 'links': {},
                           'disabled_links': [], 'negated_links': [], 'packet_indices': []}
            assert p[-1] not in current['links']
            current['links'][p[-1]] = {'code': int.from_bytes(p[2:6], 'little'), 'actor': p[6]}
        assert current is not None, 'orphan chain packet'
        current['packet_indices'].append(row['index'])
        if p[1] in (75, 76):
            assert len(p) == 3 and p[2] in current['links']
            current['disabled_links' if p[1] == 76 else 'negated_links'].append(p[2])
        if p[1] == 74:
            yield current
            current = None
    assert current is None, 'incomplete chain'


def audit(folder, seed, seat):
    rows = [json.loads(s) for s in (folder / 'decisions.jsonl').read_text().splitlines()]
    decisions = [r for r in rows if r['record_type'] == 'decision']
    terminal = [r for r in rows if r['record_type'] == 'terminal']
    assert len(terminal) == 1 and len(decisions) == terminal[0]['steps']
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['returncode'] == 0 and manifest['device'] == 'cpu' and manifest['all_decisions']
    for key in ('checkpoint_sha256', 'native_sha256', 'frozen_native_sha256'):
        assert manifest[key] == manifest[key + '_after']
    fixtures = {}
    for i, row in enumerate(decisions):
        assert row['step'] == i and row['player'] == seat and not row['cycle_guard_intervened']
        p = folder / 'fixtures' / f'step-{i:06d}.npz'
        meta = json.loads(p.with_suffix('.json').read_text())
        assert meta['requested_seed'] == seed and meta['selected_action'] == row['selected_action']
        assert meta['checkpoint_sha256'] == row['checkpoint_sha256'] == manifest['checkpoint_sha256']
        assert sha(p) == meta['sha256']
        with np.load(p, allow_pickle=False) as data:
            assert all(not a.dtype.hasobject and np.isfinite(a).all() for a in data.values())
            np.testing.assert_array_equal(data['logits'][0, :meta['num_options']], row['policy_logits'])
        fixtures[i] = {'path': str(p.relative_to(folder)), 'sha256': sha(p)}
    packets = [json.loads(s) for s in (folder / 'chain-packets.jsonl').read_text().splitlines()]
    names = {70: 'chaining', 71: 'chained', 72: 'chain_solving', 73: 'chain_solved',
             74: 'chain_end', 75: 'chain_negated', 76: 'chain_disabled'}
    lines = (folder / 'engine.log').read_text().splitlines()
    expected = [n for line in lines for n, name in names.items() if line.startswith(f'Message {name},')]
    assert expected == [p['message'] for p in packets]
    raw, human = list(raw_chains(packets)), list(chains(lines))
    assert len(raw) == len(human)
    windows = []
    for index, (r, h) in enumerate(zip(raw, human)):
        if r['context_step'] is None:
            continue  # opponent acts before the first model observation
        assert r['context_step'] in fixtures
        h = label(h)
        assert len(r['links']) == h['activation_count']
        windows.append({'chain_index': index, 'seed': seed, 'seat': seat, 'group': f'seed:{seed}',
                        'context_step': r['context_step'], 'context': fixtures[r['context_step']],
                        'context_may_precede_opponent_actions': True,
                        'combination': h['combination'], 'split': partition(seed, h['combination']),
                        'raw_chain': r, 'human_evidence': h,
                        'labels': {'root_disabled_message': 1 in r['disabled_links'],
                                   'root_negated_message': 1 in r['negated_links'],
                                   'effect_success': None, 'cannot_escape': None, 'strategic_value': None},
                        'future_response_is_teacher_only': True, 'student_input_exported': False})
    return {'decision_count': len(decisions), 'fixtures_verified': len(fixtures),
            'terminal': terminal[0], 'windows': windows,
            'source_semantics': 'frozen legacy public-event attribution; no deployment',
            'same_run_labels': True, 'old_labels_imported': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    args = p.parse_args()
    old = json.loads(args.prior_plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for job in old['jobs']:
        command_path = Path(old['source']) / job['game'] / 'command.json'
        assert sha(command_path) == job['source_hashes']['command.json']
        command = json.loads(command_path.read_text())
        jobs.append({'game': job['game'], 'command_path': str(command_path), 'command_sha256': sha(command_path),
                     'seed': int(command[command.index('--seed') + 1]),
                     'seat': int(command[command.index('--player') + 1])})
    write(args.output / 'plan.json', {'jobs': jobs, 'max_windows': 500,
          'window_selection': 'first 500 complete chains in fixed game/chain order with a preceding model context',
          'dataset_version': 'live-cpu-v1', 'old_labels_imported': False,
          'prior_plan_sha256': sha(args.prior_plan), 'capture_sha256': sha(args.capture),
          'collector_sha256': sha(Path(__file__)), 'split_rule': 'existing seed/combination hash partition',
          'device': 'cpu', 'fit_allowed': False})
    all_windows, completed = [], []
    try:
        for job in jobs:
            folder = args.output / job['game'].replace('/', '-')
            assert sha(Path(job['command_path'])) == job['command_sha256']
            subprocess.run([sys.executable, str(args.capture), '--prior-command', job['command_path'],
                            '--evidence', str(args.evidence), '--output', str(folder),
                            '--release', str(args.release), '--all-decisions', '--steps', '0'], check=True)
            result = audit(folder, job['seed'], job['seat'])
            write(folder / 'live-verification.json', result)
            # Invalid attempts are preserved and not silently replaced.
            if result['terminal']['invalid_game']:
                raise RuntimeError(f"Invalid duel retained: {job['game']}")
            for row in result['windows']:
                if len(all_windows) < 500:
                    all_windows.append(dict(row, game=job['game'], id=f"{job['game']}/chain-{row['chain_index']:04d}"))
            completed.append(job['game'])
            print(json.dumps({'completed': job['game'], 'decisions': result['decision_count'],
                              'windows_so_far': len(all_windows)}), flush=True)
        for field in ('group', 'combination'):
            sets = [{r[field] for r in all_windows if r['split'] == s} for s in ('train', 'validation', 'test')]
            assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
        with (args.output / 'windows.jsonl').open('x') as f:
            for row in all_windows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        write(args.output / 'completed.json', {'games': completed, 'windows': len(all_windows),
              'windows_sha256': sha(args.output / 'windows.jsonl'),
              'splits': dict(Counter(r['split'] for r in all_windows)),
              'student_input_exported': False, 'fitting_performed': False})
    except Exception as error:
        write(args.output / 'failed.json', {'error': repr(error), 'completed_games': completed})
        raise


if __name__ == '__main__':
    main()
