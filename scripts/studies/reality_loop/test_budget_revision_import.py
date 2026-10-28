"""Budget review may change unknown classification, never actual game decisions."""
import copy
import unittest
from freeze_engine_budget_revision import preserved_game_data
from audit_budget_study import verify_prefix
from engine_budget_audit import ACTOR_ERROR, trajectory_digest


class BudgetImportTest(unittest.TestCase):
    def record(self):
        record = dict(id='fixed', status='budget_exhausted', error=ACTOR_ERROR,
            native_sha256='native', final=dict(winner=None), events=[], core_responses=[],
            decisions=[dict(action=1)], core_process_cutoff=dict(limit=4096))
        record['independent_audit'] = dict(passed=True, kind='engine-process-budget-prefix-v1',
            core_process_calls=4096, responses=0, events=0, recorded_events=0,
            exact_recorded_prefix=True, natural_outcome_certified=False,
            independent_library_sha256='independent', trajectory_sha256=trajectory_digest(record))
        return record

    def test_review_keeps_every_actual_decision_event_and_field(self):
        revised = self.record()
        original = dict(copy.deepcopy(revised), status='invalid', independent_audit=dict(passed=False))
        self.assertTrue(preserved_game_data(original, revised))
        altered = copy.deepcopy(revised)
        altered['decisions'][0]['action'] = 2
        self.assertFalse(preserved_game_data(original, altered))

    def test_certificate_accepts_only_bound_unknown_prefix(self):
        record = self.record()
        verify_prefix(record, 'native', 'independent')
        for change in [dict(status='natural_terminal'), dict(candidate_result='win'),
                dict(native_sha256='other')]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    verify_prefix(dict(record, **change), 'native', 'independent')

    def test_tampering_with_a_completed_prefix_rejects(self):
        record = self.record()
        record['final']['winner'] = 0
        with self.assertRaises(ValueError):
            verify_prefix(record, 'native', 'independent')
        record = self.record()
        record['decisions'][0]['action'] = 2
        with self.assertRaises(ValueError):
            verify_prefix(record, 'native', 'independent')


if __name__ == '__main__':
    unittest.main()
