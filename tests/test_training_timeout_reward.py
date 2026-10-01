import unittest

import numpy as np

from ygoai.rl.timeout_reward import training_timeout_outcome


class TrainingTimeoutRewardTest(unittest.TestCase):
    def test_policy_limit_is_loss_but_external_failure_is_masked(self):
        rewards, trainable = training_timeout_outcome(
            np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            [0, 1, 1, 0], [1, 2, 3, 1], 2.0)
        np.testing.assert_array_equal(rewards, [1.0, -2.0, 0.0, -1.0])
        np.testing.assert_array_equal(trainable, [True, True, False, True])
        self.assertEqual(rewards.dtype, np.float32)

    def test_rejects_bad_configuration_and_shapes(self):
        with self.assertRaises(ValueError):
            training_timeout_outcome([0.0], [1], [2], float("nan"))
        with self.assertRaises(ValueError):
            training_timeout_outcome([0.0, 0.0], [1], [2], 2.0)

    def test_zero_loss_restores_masking_of_timeouts(self):
        rewards, trainable = training_timeout_outcome([0.0], [1], [2], 0.0)
        np.testing.assert_array_equal(rewards, [0.0])
        np.testing.assert_array_equal(trainable, [False])
