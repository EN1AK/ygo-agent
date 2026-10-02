## Why

The current policy can only act from one forward pass, while Yu-Gi-Oh! combo ordering and interruption timing often require comparing several legal continuations. The repository already contains a replay-based offline PUCT diagnostic. A bounded candidate-action rollout is the first search experiment; native snapshots and belief particles are required before it can become a useful, fair-play evaluation path. PUCT remains an equal-budget comparator and possible later extension.

## What Changes

- Add a deterministic native duel snapshot/rollback backend with explicit ownership, lifetime, capacity, and identity checks.
- Build a bounded candidate-action rollout and regularized local policy update first, preserving legal-action ordering, recurrent state, terminal reward scale, chain-response boundaries, and root-player value perspective; retain the existing PUCT diagnostic for comparison.
- Add exact-state diagnostic search and belief-particle search modes, with exact-state results explicitly labelled as oracle diagnostics and prohibited from fair-play promotion.
- Add a separate autoregressive belief head and an offline self-play supervision pipeline, with causal label masking, known-deck assumptions, duel-separated validation, and no changes to PPO actor inputs or weights.
- Add opt-in search-assisted evaluation and self-play target generation with fixed budgets, raw-policy fallback, and complete per-rollout audit records.
- Emit mode-labelled searched-policy/value targets for later distillation without silently mixing them into PPO or relabelling critic values as per-action Q-values.
- Gate deployment on byte-identical snapshot round trips, replay/native differential checks, hidden-information leakage checks, throughput measurements, and paired strength evaluation.

## Capabilities

### New Capabilities

- `mcts-policy-search`: Native snapshot-backed, model-guided candidate rollout (with PUCT comparison) for diagnostic play, belief-aware evaluation, and auditable distillation targets.

### Modified Capabilities

None.

## Impact

- Native engine and bindings under `ygoenv/ygoenv/ygopro/` gain snapshot lifecycle support.
- Search logic currently embedded in `scripts/run_counterfactual_search.py` is reused behind compatible interfaces in `ygoai.rl`; PUCT extraction follows the candidate-rollout gate.
- Evaluation and self-play tooling gain optional search configuration and richer decision logs; default PPO training remains unchanged until search targets pass the promotion gates.
- Linux native modules must be rebuilt and synchronized across the local checkout, GitHub branch, home checkout, and H200 snapshot before GPU validation.
