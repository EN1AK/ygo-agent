"""Frozen-parent regularization on replayed, complete own-seat scenes.

This bounds local policy drift only; it is not a global retention guarantee.
JAX imports stay lazy so dataset validation works without a training runtime.
"""
import numpy as np


def retention_layout(rows, steps, action_capacity):
    """Legal support and equal-scene weights, including prefix, excluding padding."""
    legal = np.zeros((steps, len(rows), action_capacity), dtype=bool)
    weights = np.zeros((steps, len(rows)), dtype=np.float32)
    for b, (record, _) in enumerate(rows):
        own = [d for d in record['decisions'] if d['player'] == 1]
        if not 1 < len(own) <= steps:
            raise ValueError('retention needs a complete bounded own-seat scene')
        for t, decision in enumerate(own):
            n = len(decision['menu'])
            if not 0 < n <= action_capacity or not 0 <= decision['action'] < n:
                raise ValueError('invalid native legal menu or demonstrated action')
            legal[t, b, :n] = True
        weights[:len(own), b] = 1. / (len(rows) * len(own))
    # Dummy support prevents undefined softmax on padding; its weight is zero.
    legal[~legal.any(axis=-1), 0] = True
    return legal.reshape(-1, action_capacity), weights.reshape(-1)


def parent_kl_rows(logits, parent_logits, legal):
    """KL(current || frozen parent) over the same native-legal action support."""
    import jax
    import jax.numpy as jnp
    parent_logits = jax.lax.stop_gradient(parent_logits)
    current = jax.nn.log_softmax(jnp.where(legal, logits, -1e9), axis=-1)
    parent = jax.nn.log_softmax(jnp.where(legal, parent_logits, -1e9), axis=-1)
    return jnp.sum(jnp.where(legal, jnp.exp(current) * (current - parent), 0.), axis=-1)


def validate_retention_plan(plan):
    """Legacy studies remain unregularized; new studies declare scope explicitly."""
    for arm in plan['arms']:
        coef = arm.get('parent_kl_coef', 0.)
        if not isinstance(coef, (int, float)) or not np.isfinite(coef) or not 0 <= coef <= 10:
            raise ValueError('parent_kl_coef must be finite and in [0, 10]')
        if coef and plan.get('retention_scope') != 'full_own_scene':
            raise ValueError('parent KL requires declared full_own_scene scope')
