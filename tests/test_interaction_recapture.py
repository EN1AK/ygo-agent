import json
from pathlib import Path
import runpy

import pytest

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/capture_interaction_dataset.py'))


def pair(tmp_path, change=None):
    a, b = tmp_path / 'old', tmp_path / 'new'
    a.mkdir()
    b.mkdir()
    row = dict(record_type='decision', step=55, player=0, legal_actions=[0, 1],
               selected_action=0, checkpoint_sha256='frozen')
    (a / 'decisions.jsonl').write_text(json.dumps(row) + '\n')
    row.update(change or {})
    (b / 'decisions.jsonl').write_text(json.dumps(row) + '\n')
    return a, b


@pytest.mark.parametrize('change', [dict(selected_action=1), dict(legal_actions=[1, 0]),
                                  dict(checkpoint_sha256='other')])
def test_drift_rejects_old_label_transfer_before_opening_fixtures(tmp_path, change):
    a, b = pair(tmp_path, change)
    with pytest.raises(AssertionError):
        API['audit'](a, b, {'steps': [84], 'windows': []})


def test_chain_stream_drift_is_not_accepted_after_equal_actions(tmp_path):
    a, b = pair(tmp_path)
    manifest = {'returncode': 0, 'device': 'cpu'}
    for key in ('checkpoint_sha256', 'native_sha256', 'frozen_native_sha256'):
        manifest[key] = manifest[key + '_after'] = 'same'
    (b / 'manifest.json').write_text(json.dumps(manifest))
    (a / 'eval.log').write_text('Message chain_disabled, length 2\n')
    (b / 'chain-packets.jsonl').write_text(json.dumps({'message': 73, 'packet_hex': '014901'}) + '\n')
    with pytest.raises(AssertionError, match='chain stream drift'):
        API['audit'](a, b, {'steps': [], 'windows': []})
