import unittest
from scripts.run_elfnote_formal_evaluation import paired_summary


def result(seed,outcomes):
    return dict(seed=seed,episodes=[dict(environment_index=i,candidate_seat=0,outcome=x) for i,x in enumerate(outcomes)])


class FormalPairsTests(unittest.TestCase):
    def test_invalid_attempts_remain_in_primary_denominator(self):
        before=[result(1,['win','invalid']),result(2,['loss','loss'])]
        after=[result(1,['invalid','win']),result(2,['win','loss'])]
        r=paired_summary(before,after)
        self.assertEqual(r['paired_attempts'],4)
        self.assertEqual(r['paired_valid_games'],2)
        self.assertEqual(r['win_per_attempt_delta'],.25)
        self.assertEqual(r['valid_pair_win_delta'],.5)
        self.assertEqual(r['seed_blocks'],2)

    def test_missing_or_duplicate_deals_reject_pairing(self):
        for rows in [[result(1,['win'])],[result(1,['win','loss']),result(1,['win','loss'])]]:
            with self.assertRaises(AssertionError):paired_summary([result(1,['loss','win'])],rows)


if __name__=='__main__':unittest.main()
