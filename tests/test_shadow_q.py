import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from ygoai.rl.jax.candidate_q_model import ObservationCandidateQ, candidate_q_optimizer
from ygoai.rl.jax.shadow_q import shadow_update
from ygoai.rl.observation_schema import tensor_contract


class ShadowQTest(unittest.TestCase):
    def test_per_actor_seat_seed_is_independent_of_other_actor_seed(self):
        expected = np.random.RandomState(41).permutation([0, 0, 1, 1])
        np.random.seed(123)
        np.random.shuffle(np.arange(100))
        np.testing.assert_array_equal(expected, np.random.RandomState(41).permutation([0, 0, 1, 1]))

    def test_real_critic_updates_with_two_seat_terminal_rollout(self):
        specs = tensor_contract('structured-lite-v1', max_cards=4, max_options=4,
                                history_actions=4, public_events=4, group_references=2)
        obs = {k: jnp.zeros((2,) + v.shape, dtype=np.dtype(v.dtype)) for k, v in specs.items()}
        obs['actions_'] = obs['actions_'].at[:, :2, 3].set(1)
        mask = jnp.array([[True, True, False, False]] * 2)
        critic = ObservationCandidateQ(embedding_shape=8, channels=32, num_layers=1, noam=False)
        variables = critic.init(jax.random.PRNGKey(5), obs, mask)
        state = TrainState.create(apply_fn=None, params=variables['params'], tx=candidate_q_optimizer(1e-3, 1.))
        # Frozen rollout data: seat changes, each env terminates at t=1.
        storage = SimpleNamespace(
            obs=jax.tree.map(lambda x: jnp.stack([x, x]), obs),
            rewards=jnp.array([[0., 0.], [1., -1.]]),
            logits=jnp.zeros((2, 2, 4)),
            menu_valid_mask=jnp.stack([mask, mask]),
            menu_chosen_index=jnp.array([[0, 1], [1, 0]]),
            acting_seat=jnp.array([[0, 1], [1, 0]]),
            train_masks=jnp.ones((2, 2), dtype=bool),
            next_dones=jnp.array([[False, False], [True, True]]))
        new_state, metrics = shadow_update(
            critic, state, storage, (obs, jnp.zeros((2, 4)), jnp.array([0, 1])),
            minibatches=2, gamma=1., trace_lambda=1.)
        self.assertTrue(bool(metrics['finite']))
        self.assertEqual(int(metrics['valid_rows']), 4)
        self.assertEqual(int(new_state.step), 2)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(state.params), jax.tree.leaves(new_state.params))))
        # Exact deterministic repeat, including optimizer leaves.
        repeat, repeated_metrics = shadow_update(
            critic, state, storage, (obs, jnp.zeros((2, 4)), jnp.array([0, 1])),
            minibatches=2, gamma=1., trace_lambda=1.)
        jax.tree.map(np.testing.assert_array_equal, new_state, repeat)
        jax.tree.map(np.testing.assert_array_equal, metrics, repeated_metrics)


if __name__ == '__main__':
    unittest.main()
