import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from scripts.audit_elfnote_rehearsal_results import recount, paired_delta, verify_old_proof, sha


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

    def test_standard_old_observation_hash_must_still_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '0.json'; path.write_text('{}')
            arrays = path.with_suffix('.npz'); arrays.write_bytes(b'original')
            proof = dict(passed=True, sha256=sha(path), observations_sha256=sha(arrays))
            verify_old_proof(path, proof)
            arrays.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                verify_old_proof(path, proof)

    def test_synthetic_exception_requires_exact_fixture_and_array_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '4.json'
            path.write_text(json.dumps(dict(case='combo', status='success', observation_archive_digest='bound')))
            proof = dict(passed=True, sha256=sha(path), same_state=True, same_events=True, status='success')
            with patch('scripts.audit_elfnote_rehearsal_results.archive_digest', return_value='bound'):
                verify_old_proof(path, proof)
            with patch('scripts.audit_elfnote_rehearsal_results.archive_digest', return_value='changed'):
                with self.assertRaises(AssertionError):
                    verify_old_proof(path, proof)
            path.write_text(json.dumps(dict(case='combo', status='verified_failure', observation_archive_digest='bound')))
            proof.update(sha256=sha(path), status='verified_failure')
            with patch('scripts.audit_elfnote_rehearsal_results.archive_digest', return_value='bound'):
                verify_old_proof(path, proof)
            proof['same_events'] = False
            with self.assertRaises(AssertionError):
                verify_old_proof(path, proof)


if __name__ == '__main__':
    unittest.main()
