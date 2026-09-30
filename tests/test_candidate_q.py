import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.jax.candidate_q import masked_policy_expectation, q_boost_targets


class CandidateQTargetsTest(unittest.TestCase):
    def test_two_seat_terminal_oracle_lambda_zero_and_one(self):
        # t=0 chooses slot 0; t=1 chooses slot 1 and ends the game.
        q = jnp.array([[[[2., 0.], [-2., 0.]]],
                       [[[4., 2.], [-4., -2.]]]])
        logits = jnp.zeros((2, 1, 2))
        mask = jnp.ones((2, 1, 2), dtype=bool)
        chosen = jnp.array([[0], [1]])
        rewards = jnp.array([[[0., 0.]], [[1., -1.]]])
        terminal = jnp.array([[False], [True]])
        valid = jnp.ones((2, 1), dtype=bool)
        boundary = jnp.array([[100., -100.]])  # Must not cross true terminal.
        expected_delta = np.array([[[1., -1.]], [[-1., 1.]]])
        for lam, expected_trace in (
            (0., expected_delta),
            (1., np.array([[[0., 0.]], [[-1., 1.]]])),
        ):
            result = q_boost_targets(
                logits, q, mask, chosen, rewards, terminal, valid,
                boundary, gamma=1., trace_lambda=lam)
            np.testing.assert_allclose(result.values, [[[1., -1.]], [[3., -3.]]])
            np.testing.assert_allclose(result.residuals, expected_delta)
            np.testing.assert_allclose(result.traces, expected_trace)
            np.testing.assert_allclose(
                result.q_targets,
                np.array([[[2., -2.]], [[2., -2.]]]) + expected_trace)

    def test_nonterminal_boundary_and_padding(self):
        q = jnp.array([[[[2., 2.], [3., 3.]]],
                       [[[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]]])
        result = q_boost_targets(
            jnp.zeros((2, 1, 2)), q,
            jnp.array([[[True, True]], [[False, False]]]),
            jnp.array([[0], [0]]), jnp.zeros((2, 1, 2)),
            jnp.zeros((2, 1), dtype=bool), jnp.array([[True], [False]]),
            jnp.array([[5., 7.]]), gamma=1., trace_lambda=1.)
        np.testing.assert_allclose(result.residuals[0], [[3., 4.]])
        np.testing.assert_allclose(result.traces[0], [[3., 4.]])
        np.testing.assert_allclose(result.q_targets[0], [[5., 7.]])
        np.testing.assert_allclose(result.advantages[1], [[0., 0.]])
        self.assertTrue(np.isfinite(np.asarray(result.q_targets)).all())

    def test_masked_and_single_action(self):
        probs, values = masked_policy_expectation(
            jnp.array([[[0., 100.]]]),
            jnp.array([[[[4., jnp.nan], [-4., jnp.nan]]]]),
            jnp.array([[[True, False]]]))
        np.testing.assert_allclose(probs, [[[1., 0.]]])
        np.testing.assert_allclose(values, [[[4., -4.]]])
        result = q_boost_targets(
            jnp.array([[[0., 100.]]]),
            jnp.array([[[[4., jnp.nan], [-4., jnp.nan]]]]),
            jnp.array([[[True, False]]]), jnp.array([[0]]),
            jnp.array([[[1., -1.]]]), jnp.array([[True]]),
            jnp.array([[True]]), jnp.zeros((1, 2)), 1., 1.)
        # The chosen Q is deliberately miscalibrated; terminal TD correction
        # moves its target to the observed reward.
        np.testing.assert_allclose(result.advantages, [[[-3., 3.]]])
        # A forced action can have a value target, but no policy-logit gradient.
        def actor_objective(raw_logits):
            terms = q_boost_targets(
                raw_logits, jax.lax.stop_gradient(jnp.array([[[[4., 0.], [-4., 0.]]]])),
                jnp.array([[[True, False]]]), jnp.array([[0]]),
                jnp.array([[[1., -1.]]]), jnp.array([[True]]),
                jnp.array([[True]]), jnp.zeros((1, 2)), 1., 1.)
            return -jnp.log(terms.policy_probs[0, 0, 0]) * jax.lax.stop_gradient(
                terms.advantages[0, 0, 0])
        np.testing.assert_allclose(jax.grad(actor_objective)(jnp.zeros((1, 1, 2))), 0.)


if __name__ == "__main__":
    unittest.main()
