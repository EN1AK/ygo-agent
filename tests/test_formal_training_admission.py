import unittest
from scripts.run_elfnote_formal import finite_training_metrics


class FormalAdmissionTests(unittest.TestCase):
    valid='optimizer_finite=[ True], notfinite_count=[0], total_notfinite=[0], step=[40]\nlosses total=1.0, policy=-0.1, value=1.1, entropy=0.2, approx_kl=0.0\n'

    def test_finite_loss_does_not_hide_skipped_optimizer_updates(self):
        self.assertTrue(finite_training_metrics(self.valid,1)['zero_nonfinite_updates'])
        for text in [self.valid.replace('total_notfinite=[0]','total_notfinite=[1]'),self.valid.replace('[ True]','[False]'),self.valid.replace('total=1.0','total=nan')]:
            with self.assertRaises(AssertionError):finite_training_metrics(text,1)

    def test_every_expected_update_needs_both_metric_records(self):
        with self.assertRaises(AssertionError):finite_training_metrics(self.valid,2)
        with self.assertRaises(AssertionError):finite_training_metrics(self.valid.split('\n',1)[1],1)


if __name__=='__main__':unittest.main()
