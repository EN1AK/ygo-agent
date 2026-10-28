"""Pure provenance, split and diagnostic checks for the research loop."""
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ledger(folder, kind, evidence, **fields):
    """Append-only hash chain; claims cite concrete existing evidence files."""
    path = Path(folder) / 'ledger.jsonl'
    previous = None
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            signature = row.pop('entry_sha256')
            if row['previous'] != previous or digest(row) != signature:
                raise ValueError('corrupt research ledger')
            previous = signature
    entry = dict(kind=kind, previous=previous,
                 evidence=[dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in evidence], **fields)
    entry['entry_sha256'] = digest(entry)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def validate_splits(rows):
    for field in ('family', 'source_sha256'):
        fit = {r[field] for r in rows if r['split'] == 'fit'}
        holdout = {r[field] for r in rows if r['split'] == 'holdout'}
        if fit & holdout:
            raise ValueError('fit/holdout leakage: ' + field)


def passive_action(menu):
    if not menu:
        raise ValueError('empty passive-opponent menu')
    return next((i for i, action in enumerate(menu) if action['act'] == 'Cancel'), 0)


def audit_opponent_policy(record, root):
    """Independent of core legality: a legal response can still violate the task."""
    seat = record['definition']['controlled']
    checked = []
    for i, decision in enumerate(record['decisions'][root:], root):
        if decision['player'] != seat:
            expected = passive_action(decision['menu'])
            if decision['action'] != expected:
                raise ValueError('opponent-policy violation at step ' + str(i))
            checked.append(i)
    return dict(passed=True, opponent_steps=checked, rule='Cancel when legal; otherwise action 0')


def diagnose(trace):
    own = [d for d in trace['decisions'] if d['actor'] == 'model']
    repeated = Counter(d['observation_sha256'] for d in own)
    cycles = []
    for a, b in zip(own, own[1:]):
        x, y = a['menu'][a['action']], b['menu'][b['action']]
        if (a['message'] == b['message'] == 26 and x['code'] == y['code']
                and x['spec'] == y['spec'] and x['unselect'] != y['unselect']):
            cycles.append([a['step'], b['step']])
    return dict(status=trace['status'], repeat_observations=repeated.most_common(3),
                reversible_selection_pairs=cycles,
                classification='invalid_runtime' if trace['status'] == 'invalid' else
                'selection_cycle_candidate' if len(cycles) >= 2 else
                'unresolved_goal' if trace['status'] != 'success' else 'success',
                wrong_action_label=False)


def first_deviation(trace, reference):
    """Return root after an ACTUAL divergent choice, never its unexecuted alternative."""
    for i, decision in enumerate(trace['decisions']):
        if i >= len(reference['decisions']):
            break
        expected = reference['decisions'][i]
        if (decision['observation_sha256'], decision['menu'], decision['player']) != (
                expected['observation_sha256'], expected['menu'], expected['player']):
            break
        if decision['actor'] == 'model' and decision['action'] != expected['action']:
            return i + 1
    return None
