"""Pre-registered, bounded CPU recapture of existing windows; never fits a model.

Snapshots precede chain start, but can precede an opponent action as well. They
are context snapshots, not an assertion that the model initiated the chain.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def decisions(path):
    return [r for r in read_rows(path) if r['record_type'] == 'decision']


def plan(inventory, source, output):
    rows = read_rows(inventory / 'windows.jsonl')
    assert len(rows) <= 500
    manifest = json.loads((inventory / 'manifest.json').read_text())
    assert sha(inventory / 'windows.jsonl') == manifest['windows_sha256']
    source_index = json.loads((source / 'delivery-verification.json').read_text())['sha256']
    selected_games = set()
    for split in ('train', 'validation', 'test'):
        games = {r['game'] for r in rows if r['split'] == split and r['game'].startswith('new186m/')}
        # IDs only: do not select by label, outcome, or probe performance.
        ordered = sorted(games, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        selected_games.update(ordered[:4])
    jobs = []
    for game in sorted(selected_games):
        folder = source / game
        hashes = {}
        for name in ('command.json', 'eval.log', 'decisions.jsonl', 'result.json'):
            hashes[name] = sha(folder / name)
            assert hashes[name] == source_index[f'{game}/{name}']
        lines = (folder / 'eval.log').read_text().splitlines()
        value_lines = [i for i, line in enumerate(lines, 1) if line.startswith('value:')]
        actions = decisions(folder / 'decisions.jsonl')
        assert len(value_lines) == len(actions)
        windows = []
        for row in rows:
            if row['game'] != game or row['split'] not in ('train', 'validation', 'test'):
                continue
            index = bisect_left(value_lines, row['start_line']) - 1
            if index < 0:
                continue  # no actor context yet; never substitute a future snapshot
            windows.append({'id': row['id'], 'split': row['split'], 'group': row['group'],
                            'combination': row['combination'], 'chain_index': row['chain_index'],
                            'step': actions[index]['step'], 'value_line': value_lines[index],
                            'chain_start_line': row['start_line']})
        steps = sorted({r['step'] for r in windows})
        assert 0 < len(steps) <= 32
        jobs.append({'game': game, 'source_hashes': hashes, 'steps': steps, 'windows': windows})
    for field in ('group', 'combination'):
        sets = [{r[field] for j in jobs for r in j['windows'] if r['split'] == s}
                for s in ('train', 'validation', 'test')]
        assert all(not sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3))
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'interaction-recapture-plan-v1', 'jobs': jobs,
              'inventory_sha256': sha(inventory / 'windows.jsonl'), 'source': str(source),
              'selection': 'first four game IDs by SHA256 per frozen split; new186m only',
              'max_duels': 12, 'max_seconds_per_duel': 360, 'device': 'cpu',
              'training_allowed': False, 'student_input_exported': False,
              'snapshot_semantics': 'last model decision strictly before chain; may be stale context',
              'event_semantics': 'frozen legacy native; known source attribution defect retained',
              'given_response': 'not exported; future packets are labels only',
              'split_counts': dict(Counter(r['split'] for j in jobs for r in j['windows']))}
    write(output / 'plan.json', record)
    print(json.dumps({'duels': len(jobs), 'splits': record['split_counts'], 'sha256': sha(output / 'plan.json')}))


def audit(original, replay, job):
    old, new = decisions(original / 'decisions.jsonl'), decisions(replay / 'decisions.jsonl')
    assert len(old) == len(new)
    for a, b in zip(old, new):
        for key in ('step', 'player', 'legal_actions', 'selected_action', 'checkpoint_sha256'):
            assert a[key] == b[key], (a['step'], key)
    manifest = json.loads((replay / 'manifest.json').read_text())
    assert manifest['returncode'] == 0 and manifest['device'] == 'cpu'
    for key in ('checkpoint_sha256', 'native_sha256', 'frozen_native_sha256'):
        assert manifest[key] == manifest[key + '_after']
    # Match entire public chain packet stream against ordered human log events,
    # including message type; the original chain index is safe only after this.
    names = {70: 'chaining', 71: 'chained', 72: 'chain_solving', 73: 'chain_solved',
             74: 'chain_end', 75: 'chain_negated', 76: 'chain_disabled'}
    lines = (original / 'eval.log').read_text().splitlines()
    expected = [n for line in lines for n, name in names.items() if line.startswith(f'Message {name},')]
    packets = read_rows(replay / 'chain-packets.jsonl')
    assert expected == [r['message'] for r in packets], 'chain stream drift'
    chains, links, disabled, negated = [], {}, [], []
    for row in packets:
        p = bytes.fromhex(row['packet_hex'])
        if p[1] == 70:
            assert len(p) == 18 and p[-1] not in links
            links[p[-1]] = {'code': int.from_bytes(p[2:6], 'little'), 'actor': p[6]}
        elif p[1] in (75, 76):
            assert len(p) == 3 and p[2] in links
            (disabled if p[1] == 76 else negated).append(p[2])
        elif p[1] == 74:
            chains.append({'links': links, 'disabled_links': disabled, 'negated_links': negated})
            links, disabled, negated = {}, [], []
    assert not links
    fixtures = {}
    for step in job['steps']:
        p = replay / 'fixtures' / f'step-{step:06d}.npz'
        meta = json.loads(p.with_suffix('.json').read_text())
        assert sha(p) == meta['sha256']
        with np.load(p, allow_pickle=False) as arrays:
            assert all(not a.dtype.hasobject and np.isfinite(a).all() for a in arrays.values())
        fixtures[str(step)] = {'sha256': sha(p), 'metadata_sha256': sha(p.with_suffix('.json'))}
    return {'same_trajectory_decisions': len(old), 'fixtures': fixtures,
            'windows': [dict(w, raw_chain=chains[w['chain_index']]) for w in job['windows']],
            'student_input_exported': False, 'training_performed': False,
            'hashes': {n: sha(replay / n) for n in ('manifest.json', 'engine.log', 'decisions.jsonl', 'chain-packets.jsonl')}}


def run(output, capture, evidence, release):
    frozen = sha(output / 'plan.json')
    record = json.loads((output / 'plan.json').read_text())
    write(output / 'started.json', {'plan_sha256': frozen})
    results = []
    try:
        for job in record['jobs']:
            original = Path(record['source']) / job['game']
            for name, expected in job['source_hashes'].items():
                assert sha(original / name) == expected
            replay = output / job['game'].replace('/', '-')
            subprocess.run([sys.executable, str(capture), '--prior-command', str(original / 'command.json'),
                            '--evidence', str(evidence), '--output', str(replay), '--release', str(release),
                            '--steps', *map(str, job['steps'])], check=True)
            result = audit(original, replay, job)
            write(replay / 'verification.json', result)
            results.append({'game': job['game'], 'verification_sha256': sha(replay / 'verification.json')})
            print(json.dumps({'completed': job['game'], 'decisions': result['same_trajectory_decisions'],
                              'windows': len(result['windows'])}), flush=True)
        assert frozen == sha(output / 'plan.json')
        write(output / 'completed.json', {'plan_sha256': frozen, 'results': results})
    except Exception as error:
        write(output / 'failed.json', {'error': repr(error), 'completed': results, 'plan_sha256': frozen})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('plan', 'run'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--capture', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--release', type=Path)
    a = p.parse_args()
    if a.mode == 'plan':
        plan(a.inventory, a.source, a.output)
    else:
        run(a.output, a.capture, a.evidence, a.release)
