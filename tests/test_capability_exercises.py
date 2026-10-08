import json
from pathlib import Path
import tempfile
import unittest

from ygoai.rl.capability_exercises import (assign_splits, draft, mine_decisions,
                                         validate_candidate, validate_verification, file_ref)
from ygoai.rl.exercise_core import messages
from ygoai.rl.exercise_starters import grade, SWORD


class CandidateTests(unittest.TestCase):
    def test_scene_cannot_self_certify_or_change_goal(self):
        row = draft('玻纤拉鳞茎', case='combo')
        self.assertEqual(validate_candidate(row)['labels'], {})
        row['training_ready'] = True
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            validate_candidate(row)
        row = draft('玻纤拉鳞茎', case='combo')
        row['goal'] = '8000_damage'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_candidate(row)

    def test_live_log_intake_preserves_provenance_without_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'decisions.jsonl'
            rows = [dict(record_type='decision', step=i, player=0,
                         legal_actions=[{'index': 0}, {'index': 1}], selected_action=1,
                         cycle_guard_intervened=(i == 1), checkpoint_sha256='a' * 64)
                    for i in range(3)]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            out = mine_decisions(path, steps=[2])
            self.assertEqual(len(out['candidates']), 2)
            for c in out['candidates']:
                self.assertEqual(validate_candidate(c)['status'], 'candidate')
                self.assertFalse(c['training_ready'])
                self.assertFalse(c['labels'])
                self.assertIn('both_players_legal_replay_history', c['missing'])
                self.assertEqual(c['source']['sha256'], out['source']['sha256'])
            with self.assertRaisesRegex(ValueError, 'absent'):
                mine_decisions(path, steps=[99])

    def test_no_suspicion_means_no_automatic_bad_label(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'decisions.jsonl'
            path.write_text(json.dumps(dict(record_type='decision', step=0,
                                             legal_actions=[0, 1], selected_action=1,
                                             state_value=-0.9)), encoding='utf-8')
            self.assertEqual(mine_decisions(path)['candidates'], [])

    def test_transitive_lineage_cannot_leak_into_test(self):
        rows = [dict(id='a', lineage=['duel:a', 'template:x'], split='development'),
                dict(id='b', lineage=['duel:b', 'template:x']),
                dict(id='c', lineage=['duel:b', 'suffix:y'])]
        splits = assign_splits(rows)
        self.assertEqual(len({r['group'] for r in splits}), 1)
        self.assertEqual({r['split'] for r in splits}, {'development'})

    def test_missing_lineage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lineage'):
            assign_splits([{'id': 'x'}])


class GradingTests(unittest.TestCase):
    def test_incomplete_resolution_is_unknown(self):
        self.assertEqual(grade('interaction', {}, [], stable=False), 'unknown')

    def test_attack_declarations_alone_do_not_satisfy_combo(self):
        state = {'cards': [dict(code=SWORD, player=0, location=4, sequence=5)]}
        attack = {'op': 110, 'raw': '6e0004050101040004'}
        self.assertEqual(grade('combo', state, [attack, attack], stable=True), 'verified_failure')
        events = [attack, {'op': 114, 'raw': '72'}, attack, {'op': 114, 'raw': '72'}]
        self.assertEqual(grade('combo', state, events, stable=True), 'success')
        state['cards'][0]['sequence'] = 6
        self.assertEqual(grade('combo', state, events, stable=True), 'verified_failure')

    def test_unknown_and_truncated_core_messages_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            list(messages(b'\xff'))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            list(messages(b'\x32\x00'))

    def test_pass_flag_without_replay_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'evidence'):
            validate_verification({'schema': 'capability-exercise-verification-v1',
                                   'passed': True, 'results': []}, '.')

    def test_tampered_evidence_and_wrong_pinned_runtime_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = dict(status='success', stable_boundary=True, root_event_index=0,
                       scripts_sha256={'test.lua': 'a' * 64}, runtime={'core_sha256': 'b' * 64},
                       root={'prompt': {}}, decisions=[{}], case='combo', branch='reference')
            refs = []
            for i in range(2):
                p = root / f'{i}.json'
                p.write_text(json.dumps(run), encoding='utf-8')
                refs.append({'path': p.name, 'sha256': file_ref(p)['sha256']})
            summary = dict(schema='capability-exercise-verification-v1', passed=True,
                           results=[dict(case='combo', branch='reference', actual='success',
                                         passed=True, identical_replays=True, evidence=refs)])
            with self.assertRaisesRegex(ValueError, 'runtime mismatch'):
                validate_verification(summary, root, expected_runtime={'core_sha256': 'c' * 64})
            (root / '0.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash/path mismatch'):
                validate_verification(summary, root)


if __name__ == '__main__':
    unittest.main()
