"""Candidate-Q targets over an already validated staged legal-action menu.

The last Q axis is the actor's menu order. The preceding axis contains the
two players' value perspectives, even when only one player is acting.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QBoostTargets(NamedTuple):
    policy_probs: jax.Array
    values: jax.Array
    chosen_q: jax.Array
    residuals: jax.Array
    traces: jax.Array
    advantages: jax.Array
    q_targets: jax.Array


def relative_q_to_absolute(q_relative, acting_seat):
    """Map [acting, opposing] Q channels to fixed [seat 0, seat 1]."""
    return jnp.where(
        jnp.asarray(acting_seat)[..., None, None] == 0,
        q_relative,
        q_relative[..., ::-1, :],
    )


def selfplay_reward_by_seat(reward_to_actor, acting_seat):
    """Convert the engine's previous-acting-player reward to absolute seats."""
    seat0 = jnp.where(jnp.asarray(acting_seat) == 0, reward_to_actor, -reward_to_actor)
    return jnp.stack((seat0, -seat0), axis=-1)


def masked_policy_expectation(logits, q_values, legal_mask):
    """Return menu probabilities and both-seat expected Q values.

    Shapes are ``logits/mask[..., M]`` and ``q_values[..., 2, M]``.
    An all-invalid row is padding and has zero probability and expectation.
    Menu identity and selected-index checks belong to the rollout boundary.
    """
    legal_mask = jnp.asarray(legal_mask, dtype=jnp.bool_)
    safe_logits = jnp.where(legal_mask, logits, -1e30)
    probs = jax.nn.softmax(safe_logits, axis=-1)
    probs = jnp.where(legal_mask, probs, 0.0)
    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-30)
    safe_q = jnp.where(legal_mask[..., None, :], q_values, 0.0)
    values = jnp.sum(probs[..., None, :] * safe_q, axis=-1)
    return probs, values


def q_boost_targets(
    reference_logits,
    q_values,
    legal_mask,
    chosen_index,
    rewards,
    terminals,
    valid_steps,
    boundary_values,
    gamma,
    trace_lambda,
):
    """Expected-SARSA residuals and backward traces for two-seat trajectories.

    Time is axis 0, environments axis 1. ``boundary_values[B, 2]`` is the
    frozen-reference expectation at the first uncollected state of each env.
    True terminals bootstrap zero; nonterminal collection boundaries bootstrap
    from that expectation. Padding contributes zero and breaks the trace.
    ``rewards[T, B, 2]`` must already follow the environment's seat convention.
    """
    probs, values = masked_policy_expectation(reference_logits, q_values, legal_mask)
    valid_steps = jnp.asarray(valid_steps, dtype=jnp.bool_)
    safe_index = jnp.where(valid_steps, chosen_index, 0)
    selected = jnp.take_along_axis(
        q_values, jnp.broadcast_to(safe_index[..., None, None], q_values.shape[:-1] + (1,)),
        axis=-1,
    )[..., 0]
    selected = jnp.where(valid_steps[..., None], selected, 0.0)
    values = jnp.where(valid_steps[..., None], values, 0.0)

    next_valid = jnp.concatenate(
        (valid_steps[1:], jnp.zeros_like(valid_steps[:1])), axis=0)
    next_values = jnp.concatenate((values[1:], boundary_values[None, ...]), axis=0)
    next_values = jnp.where(next_valid[..., None], next_values, boundary_values[None, ...])
    next_values = jnp.where(terminals[..., None], 0.0, next_values)
    residuals = jnp.where(
        valid_steps[..., None], rewards + gamma * next_values - selected, 0.0)

    def backward(carry, row):
        residual, terminal, valid, continuation = row
        trace = residual + gamma * trace_lambda * jnp.where(
            (terminal | ~continuation)[..., None], 0.0, carry)
        trace = jnp.where(valid[..., None], trace, 0.0)
        return trace, trace

    _, traces = jax.lax.scan(
        backward, jnp.zeros_like(boundary_values),
        (residuals, terminals, valid_steps, next_valid), reverse=True)
    advantages = jnp.where(valid_steps[..., None], selected - values + traces, 0.0)
    q_targets = jnp.where(valid_steps[..., None], selected + traces, 0.0)
    return QBoostTargets(probs, values, selected, residuals, traces, advantages, q_targets)
