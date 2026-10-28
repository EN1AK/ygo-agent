"""Pairing all models on the same changed matrix must still fail the audit."""
import copy
import unittest
from audit_ppo_study import verify_original_duel_matrix


class OriginalMatrixTest(unittest.TestCase):
    def matrix(self):
        return dict(cases=[dict(id=str(i), seed=i // 2, candidate_seat=i % 2) for i in range(256)],
            budget=dict(decisions=2048, seconds_per_game=120), precision='highest', opponent_sha256='fixed')

    def test_same_original_matrix_accepts_corrected_runtime(self):
        original = self.matrix()
        current = dict(copy.deepcopy(original), native_sha256='corrected')
        verify_original_duel_matrix(current, original)

    def test_uniformly_changed_matrix_is_not_excused_by_pairing(self):
        original = self.matrix()
        current = copy.deepcopy(original)
        current['cases'][0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'original fixed obligation'):
            verify_original_duel_matrix(current, original)

    def test_unchanged_cases_cannot_hide_changed_opponent_or_budget(self):
        original = self.matrix()
        for key, value in [('opponent_sha256', 'another'), ('budget', dict(decisions=4096, seconds_per_game=120))]:
            with self.subTest(key=key):
                current = dict(copy.deepcopy(original), **{key: value})
                with self.assertRaisesRegex(ValueError, 'original fixed obligation'):
                    verify_original_duel_matrix(current, original)


if __name__ == '__main__':
    unittest.main()
