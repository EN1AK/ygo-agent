"""Separate observation-only candidate-Q network and optimizer."""

import jax.numpy as jnp
import jax
from flax.core import freeze, unfreeze
import optax
from flax import linen as nn

from ygoai.rl.jax.agent import Encoder


def copy_actor_encoder(critic_variables, actor_variables):
    """Copy the complete compatible encoder; never import policy/value/RNN heads.

    No partial matching: a missing, extra, differently shaped or typed leaf is
    an incompatible initialization. Independent containers and immutable array
    copies keep subsequent Q optimization separate from the frozen actor.
    """
    target = unfreeze(critic_variables)
    source = unfreeze(actor_variables)
    q_encoder = target['params']['Encoder_0']
    actor_encoder = source['params']['Encoder_0']
    if jax.tree.structure(q_encoder) != jax.tree.structure(actor_encoder):
        raise ValueError('actor/Q encoder parameter trees differ')
    for q, a in zip(jax.tree.leaves(q_encoder), jax.tree.leaves(actor_encoder)):
        if q.shape != a.shape or q.dtype != a.dtype:
            raise ValueError('actor/Q encoder shape or dtype differs')
    target['params']['Encoder_0'] = jax.tree.map(lambda x: jnp.array(x, copy=True), actor_encoder)
    return freeze(target) if isinstance(critic_variables, type(freeze({}))) else target


class ObservationCandidateQ(nn.Module):
    """Independent observation-only critic; never shares actor parameters."""

    embedding_shape: int | tuple[int, int]
    observation_schema: str = "structured-lite-v1"
    structured_variant: str = "full"
    channels: int = 128
    num_layers: int = 2
    noam: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, observation, legal_mask):
        actions, state, _, encoder_invalid, _ = Encoder(
            channels=self.channels,
            num_layers=self.num_layers,
            embedding_shape=self.embedding_shape,
            dtype=self.dtype,
            noam=self.noam,
            observation_schema=self.observation_schema,
            structured_variant=self.structured_variant,
            oppo_info=False,
        )(observation)
        context = jnp.broadcast_to(state[:, None, :], actions.shape)
        features = jnp.concatenate((actions, context), axis=-1)
        hidden = nn.silu(nn.Dense(self.channels, dtype=jnp.float32)(features))
        values = nn.Dense(2, dtype=jnp.float32)(hidden).swapaxes(-1, -2)
        valid = jnp.asarray(legal_mask, dtype=jnp.bool_) & ~encoder_invalid
        return jnp.where(valid[:, None, :], values, 0.0)


def candidate_q_optimizer(learning_rate, max_grad_norm):
    """A distinct optimizer state for Q parameters, never the PPO TrainState."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )
