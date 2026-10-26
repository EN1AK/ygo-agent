import unittest
from ygoai.rl.scene_numerics import objective_forward_report,qualify_objective


def report(delta=1e-4):
    return objective_forward_report([[1.,-1.],[.4,-.4]],[[1.+delta,-1.-delta],[.4+delta,-.4-delta]],
        [[.2,-.2],[.1,-.1]],[[.2,-.2],[.1,-.1]],[0,1],[[True,True]]*2,[1.,1.],[1.,0.],[0.,1.])


class SceneNumericTests(unittest.TestCase):
    def test_forward_difference_is_bounded_without_relaxing_same_input_arithmetic(self):
        r=report();proof=qualify_objective(r['reference_full'],r['reference_micro'],r)
        self.assertTrue(proof['passed'])
        self.assertGreater(proof['scalar_difference'],8*2**-23)
        with self.assertRaises(AssertionError):qualify_objective(r['reference_full']+.01,r['reference_micro']+.01,r)

    def test_large_forward_or_inconsistent_target_support_is_rejected(self):
        r=report(.01)
        with self.assertRaises(AssertionError):qualify_objective(r['reference_full'],r['reference_micro'],r)
        with self.assertRaises(AssertionError):
            objective_forward_report([[1.,-1.]],[[1.,-1.]],[[0.,0.]],[[0.,0.]],[1],[[True,False]],[1.],[1.],[0.])


if __name__=='__main__':unittest.main()
