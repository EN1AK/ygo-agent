"""Safety checks for evidence admission rather than model-score snapshots."""
import json
from pathlib import Path
import tempfile
import unittest
from contracts import diagnose, first_deviation, ledger, validate_splits
from contracts import read, sha, write
from train_pair import prepare
from upgrade_inputs import assert_allowed_difference, expected_counts


class ContractsTest(unittest.TestCase):
    def test_raw_select_unselect_counts_include_selected_pool(self):
        # player=0, finish=1, cancel=0, min=1, max=3; two selectable
        # cards and one selected card. Each protocol card entry is 8 bytes.
        raw = bytes([26, 0, 1, 0, 1, 3, 2]) + bytes(16) + bytes([1]) + bytes(8)
        counts = expected_counts([dict(op=26, raw=raw.hex())])
        self.assertEqual(counts, dict(minimum=1, maximum=3, selected_count=1, must_continue=0))

    def test_input_upgrade_only_changes_declared_public_count_fields(self):
        import numpy as np
        original = dict(selection_=np.zeros((1, 16), np.uint8), cards_=np.zeros((1, 2), np.uint8))
        fixed = {k: v.copy() for k, v in original.items()}
        fixed['selection_'][0, [5, 6, 8, 11]] = [1, 3, 2, 1]
        assert_allowed_difference(original, fixed, 26)
        with self.assertRaisesRegex(ValueError, 'unexpected input change'):
            assert_allowed_difference(original, fixed, 15)
        fixed['cards_'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'unexpected input change'):
            assert_allowed_difference(original, fixed, 26)

    def test_connected_family_cannot_cross_split(self):
        with self.assertRaises(ValueError):
            validate_splits([dict(split='fit', family='p', source_sha256='a'),
                             dict(split='holdout', family='p', source_sha256='b')])

    def test_failure_prefix_keeps_executed_deviation(self):
        decisions = [dict(player=0, menu=['a', 'b'], observation_sha256='x', action=0)]
        reference = dict(decisions=decisions)
        trace = dict(decisions=[dict(decisions[0], action=1, actor='model')])
        self.assertEqual(first_deviation(trace, reference), 1)
        trace['decisions'][0]['observation_sha256'] = 'different root'
        self.assertIsNone(first_deviation(trace, reference))

    def test_cycle_diagnostic_does_not_create_negative_labels(self):
        decisions = []
        for i in range(6):
            decisions.append(dict(step=i, actor='model', message=26,
                observation_sha256=str(i % 2), action=0,
                menu=[dict(code=1, spec='m1', unselect=bool(i % 2))]))
        result = diagnose(dict(status='budget_exhausted', decisions=decisions))
        self.assertEqual(result['classification'], 'selection_cycle_candidate')
        self.assertFalse(result['wrong_action_label'])

    def test_ledger_rejects_tampered_prior_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / 'evidence.json'
            evidence.write_text('{}', encoding='utf-8')
            ledger(folder, 'observation', [evidence], count=1)
            path = Path(folder) / 'ledger.jsonl'
            row = json.loads(path.read_text(encoding='utf-8'))
            row['count'] = 99
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                ledger(folder, 'finding', [evidence])

    def pair_fixture(self, folder, bad_prefix=False):
        root = Path(folder)
        original = root / 'original'
        original.mkdir()
        plan = dict(rows=[dict(id='original', path='original.json', split='fit', family='f', source_sha256='s'),
                          dict(id='exam', path='exam.json', split='holdout', family='h', source_sha256='t')],
                    stages=[dict(mix={'foundation': 1})], evaluation={})
        write(original / 'plan.json', plan)
        checkpoint = root / 'parent'
        checkpoint.write_bytes(b'weights')
        write(root / 'protocol.json', dict(original=str(original), original_plan_sha256=sha(original / 'plan.json'),
              training=dict(repair_mass=.5, updates_per_arm=512, learning_rate=2e-5)))
        observation = root / 'round1/evaluation'
        observation.mkdir(parents=True)
        write(observation / 'summary.json', dict(checkpoint_sha256=sha(checkpoint)))
        recovery = root / 'round1-recovery'
        recovery.mkdir()
        path = recovery / 'repair.json'
        write(path, dict(training_ready=True, independent_endpoint_audit=dict(passed=True),
                         decisions=[dict(teaching=bad_prefix), dict(teaching=True)]))
        path.with_suffix('.npz').write_bytes(b'tensor evidence')
        repair = dict(id='x', path=str(path), split='fit', family='f', source_sha256='s',
                      root_step=1, record_sha256=sha(path), observations_sha256=sha(path.with_suffix('.npz')))
        write(recovery / 'summary.json', dict(admitted=[repair]))
        return root, checkpoint

    def test_control_has_identical_exam_but_no_repair_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder)
            prepared = prepare(root, 'round1', checkpoint)
            control = read(prepared / 'control-inputs/plan.json')
            candidate = read(prepared / 'repair-inputs/plan.json')
            self.assertEqual(control['rows'], candidate['rows'])
            self.assertNotIn('repair', control['stages'][0]['mix'])
            self.assertEqual(candidate['stages'][0]['mix']['repair'], .5)
            self.assertEqual(control['parent_sha256'], candidate['parent_sha256'])

    def test_failed_prefix_cannot_be_relabelled_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root, checkpoint = self.pair_fixture(folder, bad_prefix=True)
            with self.assertRaisesRegex(ValueError, 'failed prefix'):
                prepare(root, 'round1', checkpoint)


if __name__ == '__main__':
    unittest.main()
