import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.jax.agent import RNNAgent
from ygoai.rl.jax.candidate_q import masked_policy_expectation
from ygoai.rl.jax.candidate_q_model import ObservationCandidateQ, candidate_q_optimizer
from ygoai.rl.observation_schema import tensor_contract


class ObservationCandidateQTest(unittest.TestCase):
    def test_separate_critic_step_preserves_actor_logits(self):
        specs = tensor_contract(
            "structured-lite-v1", max_cards=4, max_options=4,
            history_actions=4, public_events=4, group_references=2)
        obs = {
            name: jnp.zeros((2,) + spec.shape, dtype=np.dtype(spec.dtype))
            for name, spec in specs.items()
        }
        obs["actions_"] = obs["actions_"].at[:, :2, 3].set(1)
        legal = jnp.array([[True, True, False, False]] * 2)
        actor = RNNAgent(
            num_channels=32, num_layers=1, rnn_channels=32,
            rnn_type="none", film=False, noam=False, embedding_shape=8,
            observation_schema="structured-lite-v1")
        rstate = actor.init_rnn_state(2)
        actor_variables = actor.init(jax.random.PRNGKey(1), obs, rstate)
        before = actor.apply(actor_variables, obs, rstate)[1]

        critic = ObservationCandidateQ(
            embedding_shape=8, channels=32, num_layers=1, noam=False)
        variables = critic.init(jax.random.PRNGKey(2), obs, legal)
        q = critic.apply(variables, obs, legal)
        self.assertEqual(q.shape, (2, 2, 4))
        np.testing.assert_allclose(q[..., 2:], 0.)
        probs, expectation = masked_policy_expectation(before, q, legal)
        self.assertEqual(expectation.shape, (2, 2))
        np.testing.assert_allclose(np.asarray(probs[..., 2:]), 0.)

        def loss(params):
            predicted = critic.apply({"params": params}, obs, legal)
            error = predicted - 1.
            return jnp.sum(jnp.where(legal[:, None, :], error ** 2, 0.))

        gradients = jax.grad(loss)(variables["params"])
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients)))
        optimizer = candidate_q_optimizer(1e-3, 1.)
        state = optimizer.init(variables["params"])
        updates, state = optimizer.update(gradients, state, variables["params"])
        new_params = jax.tree.map(lambda p, u: p + u, variables["params"], updates)
        self.assertTrue(np.isfinite(np.asarray(critic.apply({"params": new_params}, obs, legal))).all())
        after = actor.apply(actor_variables, obs, rstate)[1]
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
