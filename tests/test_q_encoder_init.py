import unittest
import jax.numpy as jnp
import numpy as np
from ygoai.rl.jax.candidate_q_model import copy_actor_encoder


class EncoderInitTest(unittest.TestCase):
    def test_exact_copy_keeps_head_and_actor(self):
        actor = {'params': {'Encoder_0': {'w': jnp.ones((2, 3))}, 'policy': jnp.ones(2)}}
        critic = {'params': {'Encoder_0': {'w': jnp.zeros((2, 3))}, 'Dense_0': jnp.zeros(2)}}
        result = copy_actor_encoder(critic, actor)
        np.testing.assert_array_equal(result['params']['Encoder_0']['w'], 1.)
        np.testing.assert_array_equal(result['params']['Dense_0'], 0.)
        result['params']['Encoder_0']['w'] = result['params']['Encoder_0']['w'] + 2
        np.testing.assert_array_equal(actor['params']['Encoder_0']['w'], 1.)
        np.testing.assert_array_equal(critic['params']['Encoder_0']['w'], 0.)

    def test_mismatch_refuses(self):
        actor = {'params': {'Encoder_0': {'w': jnp.ones((2, 3))}}}
        for tree in ({'extra': jnp.ones(2)}, {'w': jnp.ones((3, 2))}, {'w': jnp.ones((2, 3), dtype=jnp.int32)}):
            with self.assertRaises(ValueError):
                copy_actor_encoder({'params': {'Encoder_0': tree}}, actor)
