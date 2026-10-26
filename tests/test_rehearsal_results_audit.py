import unittest
from scripts.audit_elfnote_rehearsal_results import recount, paired_delta


def episode(outcome, reward=1., invalid=0, reason=1):
    return dict(outcome=outcome, reward=reward, invalid_game=invalid,
                termination_reason=reason, candidate_seat=0, environment_index=0)


def batch(seed, outcome):
    return dict(seed=1, environment_seed=seed, episodes=[episode(outcome)])


class RehearsalAuditTests(unittest.TestCase):
    def test_positive_invalid_reward_cannot_be_counted_as_win(self):
        row = episode('invalid', invalid=1)
        self.assertEqual(recount([row])['wins'], 0)
        row['outcome'] = 'win'
        with self.assertRaises(AssertionError):
            recount([row])
        with self.assertRaises(AssertionError):
            recount([episode('win', reward=float('nan'))])

    def test_same_declared_seed_does_not_prove_same_actual_deal_seed(self):
        with self.assertRaises(AssertionError):
            paired_delta([batch(10, 'win')], [batch(11, 'loss')])
        with self.assertRaises(AssertionError):
            paired_delta([batch(10, 'win')] * 2, [batch(10, 'win')])

    def test_primary_pairing_keeps_invalid_attempt_as_nonwin(self):
        left, right = batch(10, 'invalid'), batch(10, 'win')
        self.assertEqual(paired_delta([left], [right]), 1.)


if __name__ == '__main__':
    unittest.main()
