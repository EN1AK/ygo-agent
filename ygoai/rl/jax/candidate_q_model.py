"""Separate observation-only candidate-Q network and optimizer."""

import jax.numpy as jnp
import optax
from flax import linen as nn

from ygoai.rl.jax.agent import Encoder


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
