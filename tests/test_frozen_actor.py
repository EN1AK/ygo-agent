import unittest
import numpy as np
from ygoai.rl.frozen_actor import FrozenActorGuard


class FrozenActorTest(unittest.TestCase):
    def test_parameters_optimizer_and_step_are_guarded(self):
        state = {'params': np.array([1., 2.]), 'optimizer': np.array([0.]), 'step': 0}
        guard = FrozenActorGuard(state)
        self.assertTrue(guard.verify(state)['exact'])
        for field in state:
            changed = dict(state)
            changed[field] = state[field] + 1
            with self.assertRaises(AssertionError):
                guard.verify(changed)


if __name__ == '__main__':
    unittest.main()
