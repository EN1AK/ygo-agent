"""Checks migration parity and learnable, menu-equivariant new residuals."""
from pathlib import Path
import sys

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from migrate_public_state_checkpoint import migrate_model, observation
from ygoai.rl.jax.agent import RNNAgent, ModelArgs
from dataclasses import asdict


@pytest.fixture(scope='module')
def models():
    capacities = dict(max_cards=8, max_options=4, history_actions=4, public_events=4, group_references=8)
    config = ModelArgs(observation_schema='structured-lite-v1', num_layers=1, rnn_channels=64)
    old = RNNAgent(embedding_shape=16, **asdict(config))
    obs = observation('structured-state-v2', capacities)
    obs['actions_'] = obs['actions_'].at[:, :2, 3].set(1)
    obs['action_features_'] = obs['action_features_'].at[:, :2, 0].set(1)
    state = old.init_rnn_state(1)
    source = old.init(jax.random.PRNGKey(4), obs, state)
    params, config2, report = migrate_model(source, config, 16, capacities)
    assert report['zero_residual_parity']
    obs['action_effect_semantics_'] = obs['action_effect_semantics_'].at[:, 0, :].set(1)
    obs['public_chain_'] = obs['public_chain_'].at[:, 0, :].set(jnp.array([1,1,1,1,13,1,1], dtype=jnp.uint16))
    obs['public_turn_effects_'] = obs['public_turn_effects_'].at[:, 0, :].set(jnp.array([1,1,1,1,3,2,1,0,1], dtype=jnp.uint16))
    return RNNAgent(embedding_shape=16, **asdict(config2)), params, obs, state


def test_new_features_receive_finite_nonzero_gradients(models):
    model, params, obs, state = models
    def objective(p):
        _, logits, value, _ = model.apply(p, obs, state)
        return logits[0,0] - logits[0,1] + value.sum()
    gradients = jax.grad(objective)(params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients))
    encoder = gradients['params']['Encoder_0']
    for key in ('effect_description_residual', 'public_state_action_residual', 'public_state_value_residual'):
        assert np.linalg.norm(np.asarray(encoder[key]['kernel'])) > 0, key


def test_action_permutation_after_residual_is_enabled(models):
    model, params, obs, state = models
    values = flax.core.unfreeze(params)
    for key in ('effect_description_residual', 'public_state_action_residual', 'public_state_value_residual'):
        kernel = values['params']['Encoder_0'][key]['kernel']
        values['params']['Encoder_0'][key]['kernel'] = jax.random.normal(jax.random.PRNGKey(len(key)), kernel.shape) * .01
    permutation = jnp.array([1,0,2,3])
    swapped = dict(obs)
    for key in ('actions_', 'action_features_', 'action_single_refs_', 'action_group_refs_',
                'action_group_mask_', 'action_effect_semantics_'):
        swapped[key] = obs[key][:, permutation]
    a = model.apply(values, obs, state)
    b = model.apply(values, swapped, state)
    np.testing.assert_allclose(a[1][:,permutation], b[1], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(a[2], b[2], atol=2e-5, rtol=2e-5)
    blank = dict(obs)
    blank['action_effect_semantics_'] = jnp.zeros_like(obs['action_effect_semantics_'])
    blank['public_chain_'] = jnp.zeros_like(obs['public_chain_'])
    blank['public_turn_effects_'] = jnp.zeros_like(obs['public_turn_effects_'])
    c = model.apply(values, blank, state)
    assert not np.allclose(a[1][:,:2], c[1][:,:2], atol=1e-6, rtol=1e-6)
