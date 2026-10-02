import unittest

from ygoai.multideck_training import resume_sampler_counters


class SamplerRepartitionTest(unittest.TestCase):
    def test_preserves_flat_order_and_requires_opt_in(self):
        actors={'0':[2,3], '1':[4,5], '2':[6,7]}
        self.assertEqual(resume_sampler_counters(actors,2,3),['2,3','4,5','6,7'])
        with self.assertRaisesRegex(ValueError,'explicit'):
            resume_sampler_counters(actors,6,1)
        self.assertEqual(resume_sampler_counters(actors,6,1,repartition=True),['2,3,4,5,6,7'])
        with self.assertRaisesRegex(ValueError,'total'):
            resume_sampler_counters(actors,4,2,repartition=True)

    def test_rejects_corrupt_saved_state(self):
        for actors in ({'1':[0]}, {'0':[1], '1':[2,3]}, {'0':[-1]}, {'0':[True]}):
            with self.assertRaises(ValueError):
                resume_sampler_counters(actors,1,1)
        self.assertIsNone(resume_sampler_counters({},16,15))


if __name__=='__main__':
    unittest.main()
