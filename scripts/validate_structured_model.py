"""Numerical, shape, and action-permutation diagnostics for Structured-lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _repo_bootstrap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from ygoai.rl.jax.agent import RNNAgent
from ygoai.rl.observation_schema import manifest


ACTION_KEYS = (
    "actions_", "action_features_", "action_single_refs_",
    "action_group_refs_", "action_group_mask_",
)


def fixture(batch: int, maximum: bool) -> dict[str, jax.Array]:
    specs = manifest("structured-lite-v1")["tensors"]
    obs = {
        name: jnp.zeros((batch,) + tuple(spec["shape"]), dtype=spec["dtype"])
        for name, spec in specs.items()
    }
    # Slot zero must be a usable action in both padding-only and maximum fixtures.
    obs["actions_"] = obs["actions_"].at[:, 0, 3].set(1)
    obs["action_features_"] = obs["action_features_"].at[:, 0, 0].set(1)
    if maximum:
        obs["cards_"] = obs["cards_"].at[..., 2].set(1)
        obs["actions_"] = obs["actions_"].at[..., 3].set(1)
        obs["action_features_"] = obs["action_features_"].at[..., 0].set(1)
        obs["selection_"] = obs["selection_"].at[..., 0].set(2)
        obs["public_events_"] = obs["public_events_"].at[..., 0].set(1)
        obs["action_single_refs_"] = obs["action_single_refs_"].at[..., 0].set(1)
        obs["action_single_refs_"] = obs["action_single_refs_"].at[..., 1].set(1)
        obs["action_single_refs_"] = obs["action_single_refs_"].at[..., 2].set(3)
        obs["action_group_refs_"] = obs["action_group_refs_"].at[..., 0].set(1)
        obs["action_group_refs_"] = obs["action_group_refs_"].at[..., 1].set(3)
        obs["action_group_mask_"] = jnp.ones_like(obs["action_group_mask_"])
    return obs


def run(variant: str) -> dict[str, object]:
    agent = RNNAgent(
        embedding_shape=2048,
        observation_schema="structured-lite-v1",
        structured_variant=variant,
    )
    obs = fixture(2, maximum=True)
    state = agent.init_rnn_state(2)
    params = agent.init(jax.random.PRNGKey(7), obs, state)
    _, logits, values, valid = agent.apply(params, obs, state)
    finite = bool(np.isfinite(np.asarray(logits)).all() and np.isfinite(np.asarray(values)).all())

    permutation = np.arange(obs["actions_"].shape[1])[::-1]
    permuted = dict(obs)
    for key in ACTION_KEYS:
        permuted[key] = obs[key][:, permutation]
    _, permuted_logits, _, _ = agent.apply(params, permuted, state)
    restored = np.asarray(permuted_logits)[:, permutation]
    max_error = float(np.max(np.abs(np.asarray(logits) - restored)))

    padding = fixture(2, maximum=False)
    _, padding_logits, padding_values, _ = agent.apply(params, padding, state)
    padding_finite = bool(
        np.isfinite(np.asarray(padding_logits)).all()
        and np.isfinite(np.asarray(padding_values)).all()
    )
    return {
        "variant": variant,
        "logits_shape": list(logits.shape),
        "value_shape": list(values.shape),
        "valid_shape": list(valid.shape),
        "maximum_fixture_finite": finite,
        "padding_fixture_finite": padding_finite,
        "action_permutation_max_abs_error": max_error,
        "action_permutation_tolerance": 1e-5,
        "passed": finite and padding_finite and max_error <= 1e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {variant: run(variant) for variant in ("relationship-only", "full")}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not all(item["passed"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
