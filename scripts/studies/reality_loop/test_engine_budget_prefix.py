"""A process fuse certifies an unknown prefix only after exact full replay."""
import copy
import unittest
from engine_budget_audit import finish_prefix, eligible, trajectory_digest, ACTOR_ERROR, CORE_ERROR


class FakeCore:
    def __init__(self, record):
        self.calls = 4096
        self.decisions = [None] * len(record['core_responses'])
        self.events = copy.deepcopy(record['events'])
        self.final = copy.deepcopy(record['final'])
        self.error = CORE_ERROR

    def next_prompt(self, limit):
        assert limit == 4096
        if self.error is None:
            return None
        raise TimeoutError(self.error)

    def state(self):
        return self.final


class BudgetPrefixTest(unittest.TestCase):
    def record(self):
        return dict(status='invalid', error=ACTOR_ERROR, final=dict(winner=None, lp=[8000, 8000]),
            events=[dict(op=40, raw='2800')], core_responses=[dict(kind='int', data='00000000')])

    def test_exact_fuse_tail_is_unknown_not_a_natural_outcome(self):
        record = self.record()
        self.assertTrue(eligible(record))
        proof = finish_prefix(FakeCore(record), record)
        self.assertTrue(proof['passed'])
        self.assertFalse(proof['natural_outcome_certified'])

    def test_fuse_before_all_responses_or_wrong_limit_rejects(self):
        record = self.record()
        for field, value in [('decisions', []), ('calls', 4095), ('error', 'another timeout'), ('error', None)]:
            with self.subTest(field=field, value=value):
                core = FakeCore(record)
                setattr(core, field, value)
                with self.assertRaises(ValueError):
                    finish_prefix(core, record)

    def test_event_or_final_state_drift_rejects_even_at_correct_fuse(self):
        record = self.record()
        for field, value in [('events', [dict(op=40, raw='2801')]), ('final', dict(winner=None, lp=[7999, 8000]))]:
            with self.subTest(field=field):
                core = FakeCore(record)
                setattr(core, field, value)
                with self.assertRaises(ValueError):
                    finish_prefix(core, record)

    def test_other_errors_capture_failures_and_winners_are_ineligible(self):
        record = self.record()
        for change in [dict(error='invalid command'), dict(capture_error='failed'),
                dict(final=dict(winner=0)), dict(candidate_result='win')]:
            self.assertFalse(eligible(dict(record, **change)))

    def test_trajectory_certificate_changes_with_response_and_state(self):
        record = self.record()
        original = trajectory_digest(record)
        revised = copy.deepcopy(record)
        revised['core_responses'][0]['data'] = '01000000'
        self.assertNotEqual(original, trajectory_digest(revised))
        revised = copy.deepcopy(record)
        revised['final']['lp'][0] -= 1
        self.assertNotEqual(original, trajectory_digest(revised))


if __name__ == '__main__':
    unittest.main()
