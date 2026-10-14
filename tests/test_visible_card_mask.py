"""Unused native masks must not erase the distinction between hand and field."""
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from scripts.migrate_public_state_checkpoint import observation
from ygoai.rl.jax.agent import ModelArgs, RNNAgent


def test_visible_zone_survives_unused_zero_mask():
    capacities = dict(max_cards=8, max_options=4, history_actions=4,
                      public_events=4, group_references=8)
    obs = observation('structured-lite-v1', capacities)
    # Two copies with identical identity/statics, one in hand and one on field.
    obs['cards_'] = obs['cards_'].at[:, :2, 1].set(1)
    obs['cards_'] = obs['cards_'].at[:, 0, 2].set(2).at[:, 1, 2].set(3)
    obs['visible_card_ids_'] = obs['visible_card_ids_'].at[:, :2, 1].set(1)
    obs['actions_'] = obs['actions_'].at[:, :2, 3].set(3)
    obs['actions_'] = obs['actions_'].at[:, 0, 0].set(1).at[:, 1, 0].set(2)
    obs['actions_'] = obs['actions_'].at[:, :2, 2].set(1)
    obs['mask_'] = jnp.zeros_like(obs['mask_'])
    model = RNNAgent(embedding_shape=16, **asdict(ModelArgs(
        observation_schema='structured-lite-v1', num_layers=1, rnn_channels=64)))
    state = model.init_rnn_state(1)
    params = model.init(jax.random.PRNGKey(17), obs, state)
    full = model.apply(params, obs, state)
    normal = model.apply(params, dict(obs, mask_=None), state)
    np.testing.assert_array_equal(full[1], normal[1])
    stale = model.apply(params, dict(obs, mask_=jnp.ones_like(obs['mask_'])), state)
    np.testing.assert_array_equal(full[1], stale[1])
    assert abs(float(full[1][0, 0] - full[1][0, 1])) > 1e-6
