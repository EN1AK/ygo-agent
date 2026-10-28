"""Reject historical-input reuse and incomplete autonomous replay coverage."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from contracts import write, sha
from corrected_lineage import teaching_bindings, actor_path_bindings


class CorrectedLineageTest(unittest.TestCase):
    def make_inputs(self, run, changed_native=None):
        for name, parent in [('round1', 'initial'), ('round2', 'updated')]:
            folder = run / (name + '-training')
            admission = dict(parent_sha256=parent, input_plan_sha256={})
            for arm in ('control', 'repair'):
                path = folder / (arm + '-inputs/plan.json')
                path.parent.mkdir(parents=True)
                write(path, dict(parent_sha256=parent, full_legal_menu=False,
                    native_sha256=changed_native if name == 'round1' and arm == 'repair' else 'correct'))
                admission['input_plan_sha256'][arm] = sha(path)
            write(folder / 'admission.json', admission)
            write(folder / 'comparison.json', dict(endpoint_sha256=dict(repair='updated')))

    def context(self, run, name):
        return dict(initial_checkpoint_sha256='initial' if name == 'round1' else 'updated')

    def test_exact_admission_hash_does_not_excuse_old_native_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.make_inputs(run, 'old')
            with patch('corrected_lineage.round_protocol', self.context):
                with self.assertRaisesRegex(ValueError, 'different runtime'):
                    teaching_bindings(run, 'correct')

    def test_corrected_two_round_lineage_binds_every_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.make_inputs(run, 'correct')
            with patch('corrected_lineage.round_protocol', self.context):
                bindings = teaching_bindings(run, 'correct')
            self.assertEqual(len(bindings), 6)
            self.assertTrue(all(sha(p) == value for p, value in bindings.items()))

    def test_round_two_must_use_updated_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.make_inputs(run, 'correct')
            with patch('corrected_lineage.round_protocol', return_value=dict(initial_checkpoint_sha256='initial')):
                with self.assertRaisesRegex(ValueError, 'terminal updated'):
                    teaching_bindings(run, 'correct')

    def test_failure_bridge_cannot_be_used_as_fresh_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            write(path, dict(passed=False))
            with self.assertRaisesRegex(ValueError, 'every actual actor path'):
                actor_path_bindings(Path(tmp), Path(tmp), path, 'correct')


if __name__ == '__main__':
    unittest.main()
