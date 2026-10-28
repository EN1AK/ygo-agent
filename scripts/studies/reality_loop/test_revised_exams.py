"""Fixed-model retention and aliasing boundaries for corrected-rule exams."""
import unittest
from freeze_revised_exams import retain_historical


class RevisedEndpointTest(unittest.TestCase):
    def test_different_historical_weights_are_preserved(self):
        current=[dict(name='control-final',checkpoint='new',checkpoint_sha256='new-sha')]
        old=[dict(name='control-final',checkpoint='old',checkpoint_sha256='old-sha')]
        rows,aliases=retain_historical(current,old)
        self.assertEqual([r['name'] for r in rows],['control-final','historical-control-final'])
        self.assertEqual(rows[-1]['checkpoint'],'old')
        self.assertEqual(aliases,[])
        self.assertEqual(len(current),1)

    def test_alias_requires_exact_weight_identity(self):
        current=[dict(name='rehearse-final',checkpoint='new',checkpoint_sha256='same-sha')]
        old=[dict(name='control-final',checkpoint='old',checkpoint_sha256='same-sha')]
        rows,aliases=retain_historical(current,old)
        self.assertEqual(len(rows),1)
        self.assertEqual(aliases[0]['name'],'historical-control-final')
        self.assertEqual(aliases[0]['equivalent_endpoint'],'rehearse-final')
        self.assertEqual(aliases[0]['checkpoint'],'old')
        self.assertEqual(aliases[0]['checkpoint_sha256'],rows[0]['checkpoint_sha256'])

    def test_name_collision_cannot_hide_a_fixed_model(self):
        current=[dict(name='historical-control-final',checkpoint='new',checkpoint_sha256='new-sha')]
        old=[dict(name='control-final',checkpoint='old',checkpoint_sha256='old-sha')]
        with self.assertRaisesRegex(ValueError,'collision'):
            retain_historical(current,old)


if __name__=='__main__':unittest.main()
