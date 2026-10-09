import unittest
import numpy as np

from ygoai.rl.exercise_retention import retention_layout, validate_retention_plan


class RetentionLayoutTests(unittest.TestCase):
    def test_full_prefix_equal_scenes_padding_and_other_seat(self):
        def record(players):
            return (dict(decisions=[dict(player=p, action=0, menu=[{}, {}])
                                    for p in players]), {})
        legal, weights = retention_layout([record([0, 1, 0, 1]), record([1, 1, 1])], 4, 3)
        np.testing.assert_allclose(weights, [.25, 1/6, .25, 1/6, 0, 1/6, 0, 0])
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertTrue(legal.any(axis=1).all())
        self.assertFalse(legal[:, 2].any())
        self.assertEqual(legal[4].tolist(), [True, False, False])

    def test_invalid_menu_and_undeclared_scope_rejected(self):
        row = (dict(decisions=[dict(player=1, action=2, menu=[{}])] * 2), {})
        with self.assertRaises(ValueError):
            retention_layout([row], 4, 3)
        for coefficient in [-1, float('nan'), 11]:
            with self.assertRaises(ValueError):
                validate_retention_plan(dict(arms=[dict(parent_kl_coef=coefficient)]))
        with self.assertRaises(ValueError):
            validate_retention_plan(dict(arms=[dict(parent_kl_coef=.25)]))
        validate_retention_plan(dict(arms=[{}]))


try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


@unittest.skipIf(jax is None, 'JAX numerical checks require the training runtime')
class RetentionNumericsTests(unittest.TestCase):
    def test_direction_mask_and_frozen_anchor_gradient(self):
        from ygoai.rl.exercise_retention import parent_kl_rows
        p = jnp.array([[.8, .2, 1.], [.4, .6, 1.]])
        q = jnp.array([[.3, .7, 1.], [.4, .6, 1.]])
        legal = jnp.array([[True, True, False]] * 2)
        logits, anchor = jnp.log(p).at[:, 2].set(1e6), jnp.log(q).at[:, 2].set(-1e6)
        actual = parent_kl_rows(logits, anchor, legal)
        expected = .8*np.log(.8/.3)+.2*np.log(.2/.7)
        np.testing.assert_allclose(actual, [expected, 0.], atol=1e-6)
        grad_current, grad_parent = jax.grad(
            lambda x, y: parent_kl_rows(x, y, legal).sum(), argnums=(0, 1))(logits, anchor)
        self.assertTrue(np.isfinite(grad_current).all())
        self.assertGreater(float(jnp.linalg.norm(grad_current)), 0.)
        np.testing.assert_array_equal(grad_parent, 0.)
        np.testing.assert_array_equal(grad_current[:, 2], 0.)

    def test_zero_kl_and_padding_cannot_change_loss_or_gradient(self):
        from ygoai.rl.exercise_retention import parent_kl_rows
        logits = jnp.array([[1., -1.], [100., -100.]])
        anchor = jnp.array([[1., -1.], [-100., 100.]])
        legal = jnp.ones_like(logits, dtype=bool)
        weights = jnp.array([1., 0.])
        objective = lambda x: jnp.dot(weights, parent_kl_rows(x, anchor, legal))
        self.assertAlmostEqual(float(objective(logits)), 0., places=6)
        np.testing.assert_allclose(jax.grad(objective)(logits), 0., atol=1e-6)


if __name__ == '__main__':
    unittest.main()
