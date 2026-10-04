"""Conservative snapshot boundary audit; revealed identities need separate proof."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root):
    completed = json.loads((root / 'completed.json').read_text())
    windows_path = root / 'windows.jsonl'
    assert hashlib.sha256(windows_path.read_bytes()).hexdigest() == completed['windows_sha256']
    windows = [json.loads(s) for s in windows_path.read_text().splitlines()]
    issues, count, hidden_rows = [], 0, 0
    for folder in sorted(root.glob('new186m-attempt-*')):
        for p in sorted((folder / 'fixtures').glob('*.npz')):
            meta = json.loads(p.with_suffix('.json').read_text())
            assert hashlib.sha256(p.read_bytes()).hexdigest() == meta['sha256']
            with np.load(p, allow_pickle=False) as data:
                c = data['obs__cards_'][0]
                visible = data['obs__visible_card_ids_'][0]
                # Native location IDs: deck=1, hand=2, extra=7. Identity may be
                # legitimately revealed, but this auditor cannot certify that.
                hidden = (c[:, 4] == 1) & np.isin(c[:, 2], [1, 2, 7])
                hidden_rows += int(hidden.sum())
                identified = hidden & (np.any(c[:, :2] != 0, axis=1) | np.any(visible != 0, axis=1))
                if identified.any():
                    issues.append({'fixture': str(p.relative_to(root)), 'kind': 'opponent_private_zone_identity_needs_public_reveal_proof',
                                   'rows': np.flatnonzero(identified).tolist()})
                unknown = np.all(visible == 0, axis=1)
                for name in ('card_semantics_', 'effect_tags_', 'effect_tag_confidence_'):
                    if np.any(data['obs__' + name][0][unknown]):
                        issues.append({'fixture': str(p.relative_to(root)), 'kind': 'unknown_id_has_semantics', 'field': name})
                # Policy output/labels/engine packets must never enter raw inputs.
                allowed = set(meta['observation_keys'])
                assert allowed == {k[5:] for k in data.files if k.startswith('obs__')}
                count += 1
    summary = {'fixtures_checked': count, 'opponent_private_zone_rows_checked': hidden_rows,
               'issues': issues, 'issue_counts': dict(Counter(r['kind'] for r in issues)),
               'windows': len(windows), 'split_counts': completed['splits'],
               'root_disabled_by_split': {s: dict(Counter(str(r['labels']['root_disabled_message']) for r in windows if r['split'] == s))
                                          for s in ('train', 'validation', 'test')},
               'cannot_escape_known': sum(r['labels']['cannot_escape'] is not None for r in windows),
               'strategic_value_known': 0, 'student_input_exported': False,
               'note': 'No violation found is not a complete noninterference proof; history/RNN and public-reveal provenance need review.',
               'gate': 'review_required' if not issues else 'blocked_pending_visibility_proof'}
    with (root / 'input-boundary-audit.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k != 'issues'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    audit(p.parse_args().root)
