"""Learner-only shadow Q: frozen-reference targets, no actor gradients."""

import jax
import jax.numpy as jnp

from ygoai.rl.jax.candidate_q import (
    masked_policy_expectation, q_boost_rollout_targets, relative_q_to_absolute,
)


def all_finite(tree):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc & jnp.all(jnp.isfinite(x)), tree,
        initializer=jnp.asarray(True))


def shadow_update(critic, state, storage, boundary, *, minibatches, gamma, trace_lambda):
    """One critic epoch; ``state`` never includes an actor parameter or optimizer.

    Targets are computed once with the pre-update Q and the *collected* policy
    logits. Both absolute-seat traces retain every opponent transition. The
    network channels are [acting, opposing], converted only at the trace edges.
    The PPO caller must continue using its original GAE update independently.
    """
    t, b = storage.rewards.shape
    n = t * b
    flatten = lambda x: x.reshape((n,) + x.shape[2:])
    chunk = lambda x: x.reshape((minibatches, n // minibatches) + x.shape[1:])
    obs = jax.tree.map(flatten, storage.obs)
    masks = flatten(storage.menu_valid_mask)
    obs_chunks = jax.tree.map(chunk, obs)
    mask_chunks = chunk(masks)

    def predict(_, batch):
        ob, mask = batch
        return None, critic.apply({'params': state.params}, ob, mask)

    _, predicted = jax.lax.scan(predict, None, (obs_chunks, mask_chunks))
    predicted = predicted.reshape((t, b, 2, masks.shape[-1]))
    boundary_obs, boundary_logits, boundary_seat = boundary
    boundary_mask = (boundary_obs['actions_'][..., 3] != 0).at[:, 0].set(True)
    boundary_q = critic.apply({'params': state.params}, boundary_obs, boundary_mask)
    _, boundary_v = masked_policy_expectation(
        boundary_logits, relative_q_to_absolute(boundary_q, boundary_seat), boundary_mask)
    targets = q_boost_rollout_targets(
        storage, predicted, boundary_v, gamma, trace_lambda, storage.train_masks)
    absolute = targets.both_seats.q_targets
    relative = jnp.where(storage.acting_seat[..., None] == 0, absolute, absolute[..., ::-1])
    relative = jax.lax.stop_gradient(relative)
    chosen = storage.menu_chosen_index.reshape(n)
    valid = storage.train_masks.reshape(n)
    flat_targets = relative.reshape(n, 2)

    def loss_fn(params, ob, mask, actions, labels, valid_rows):
        q = critic.apply({'params': params}, ob, mask)
        selected = jnp.take_along_axis(q, actions[:, None, None], axis=-1)[..., 0]
        error = jnp.where(valid_rows[:, None], selected - labels, 0.)
        return jnp.sum(error ** 2) / jnp.maximum(2 * valid_rows.sum(), 1)

    def step(carry, batch):
        state, finite = carry
        loss, grads = jax.value_and_grad(loss_fn)(state.params, *batch)
        healthy = jnp.isfinite(loss) & all_finite(grads)
        proposed = state.apply_gradients(grads=grads)
        healthy &= all_finite((proposed.params, proposed.opt_state))
        # Do not poison a checkpoint on a bad minibatch; caller aborts this run.
        state = jax.lax.cond(healthy, lambda: proposed, lambda: state)
        return (state, finite & healthy), loss

    (state, finite), losses = jax.lax.scan(
        step, (state, all_finite((predicted, relative, boundary_v))),
        (obs_chunks, mask_chunks, chunk(chosen), chunk(flat_targets), chunk(valid)))
    selected = targets.both_seats.chosen_q
    error = jnp.where(storage.train_masks[..., None], selected - absolute, 0.)
    count = jnp.maximum(2 * storage.train_masks.sum(), 1)
    metrics = {
        'loss': losses.mean(), 'preupdate_rmse': jnp.sqrt(jnp.sum(error ** 2) / count),
        'preupdate_bias': error.sum() / count,
        'target_abs_max': jnp.max(jnp.abs(relative)),
        'q_abs_max': jnp.max(jnp.abs(predicted)),
        'finite': finite, 'valid_rows': storage.train_masks.sum(),
    }
    return state, metrics
