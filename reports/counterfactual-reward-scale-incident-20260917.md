# Counterfactual reward-scale incident (2026-09-17)

## Summary

The first Veiler/Ogre counterfactual search inherited the native YGOPro
environment default `greedy_reward=True`. The Stage 3 checkpoint was trained
with `greedy_reward=False`. Terminal traversal therefore produced speed-shaped
returns such as `±2`, `±4`, and `±8`, while PUCT leaf values were trained on
terminal win/loss returns `±1`.

## Impact

- `formal-v2.json` and the original search report mix incompatible scales.
- Its terminal means, regret interval, and any direct leaf-value/terminal-value
  comparison are invalid.
- Its PUCT ranking happened to be unchanged after correction because the
  bounded search did not encounter a terminal node, but the old artifact must
  not be used as evidence or training data.

## Root cause

The counterfactual scripts omitted the `greedy_reward` environment argument.
The native environment defaults it to `true`, whereas `cleanba.py` explicitly
sets it to the training configuration, which was `false` for Stage 3.

## Permanent controls

1. All counterfactual entry points explicitly set `greedy_reward=False`.
2. Outputs record `reward_mode="terminal-win-loss"` and
   `greedy_reward=false`; the search report also records the terminal scale.
3. Every observed terminal reward is checked at runtime and the run aborts
   unless its absolute value is exactly 1 within numerical tolerance.
4. Counterfactual documentation states that checkpoint and rollout reward
   definitions must match and forbids relying on native defaults.
5. The original report is retained with a prominent superseded warning rather
   than silently rewritten.

## Corrected artifact

Use `formal-v3-winloss.json`, generated with the same snapshot, checkpoints,
seed, enumeration budget, rollout count, PUCT budget, and search depth as v2.
Only the explicit reward definition and its validation guard changed.
