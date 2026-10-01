"""Keep policy-caused step-limit failures distinct from environment failures."""

import numpy as np


MAX_STEPS_TERMINATION_REASON = 2


def training_timeout_outcome(rewards, invalid_games, termination_reasons, loss):
    """Penalize step-limit loops; leave external failures out of PPO updates."""
    if not np.isfinite(loss) or loss < 0:
        raise ValueError("max_step_loss must be finite and nonnegative")
    invalid = np.asarray(invalid_games, dtype=np.bool_)
    reasons = np.asarray(termination_reasons)
    rewards = np.asarray(rewards)
    if rewards.shape != invalid.shape or rewards.shape != reasons.shape:
        raise ValueError("timeout outcome arrays must have matching shapes")
    policy_timeout = invalid & (reasons == MAX_STEPS_TERMINATION_REASON) & (loss > 0)
    learning_rewards = np.where(policy_timeout, -loss, rewards).astype(
        rewards.dtype, copy=False)
    trainable = ~invalid | policy_timeout
    return learning_rewards, trainable
