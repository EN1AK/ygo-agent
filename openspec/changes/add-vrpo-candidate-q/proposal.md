## Why

The active self-play learner uses PPO with GAE and a state-value critic. In stochastic imperfect-information play, sampled future actions can make its advantage estimates noisy; the current `q_head` is not a trained action-value critic. We need a controlled way to test whether candidate-action Q and Q-boosting improve this agent before committing to expensive search or another long training run.

## What Changes

- Add an opt-in Candidate-Q research path over the exact policy-visible staged legal-action menu, with a separate action-value critic and Q-boosted advantage estimator. Keep the existing PPO/GAE path as the default and preserve its checkpoint and inference contracts.
- Stage the work: first instrument and validate an observation-only Candidate-Q ablation; then add a strictly learner-only privileged full-state critic input for a faithful centralized-Q experiment. Label these variants separately and never call the observation-only ablation full VRPO.
- Per the user's updated order, finish and evaluate the 100M PPO baseline first, then compare restarted GAE, Candidate-Q, and centralized Q-boosting from that same frozen 100M actor under matched environment-step and wall-clock budgets, fixed deck/seeds/both-seat evaluations, value calibration, combo and interruption metrics, and throughput/error gates. Retain 40M as a historical reference only.
- Keep search, belief modeling, self-play damping, and the running H200 100M continuation outside this change. No search targets or privileged fields may enter actor inference.

### User-directed evaluation budget revision (2026-10-03)

The user explicitly requested not waiting for 1,024 WindBot games before the
VRPO experiment. Retain the completed four-executor 40-game smoke as a small
diagnostic baseline, plus 10 paired-seat SkyStriker mirror attempts (100M actor
versus WindBot, both using the identical SkyStriker deck). Preserve any partial
expansion and user-interrupted attempt separately. Complete the already-running
historical matrix and a bounded replay/strategy review, then permit the short
shadow-Q gate; this is not high-confidence strength promotion. Critic quality,
numerical health, leakage, and matched-control requirements remain unchanged.

## Capabilities

### User-approved frozen specialist branch (2026-10-03)

For the 121,208,832-step SkyStriker specialist, first fit only the separate
observation critic with the actor frozen. Fresh self-play seeds supply fitting
data; selected WindBot loss diagnostics never enter fitting. This is not a
100M matched strength arm or a claim of full VRPO. Keep the 5M comparison gated.

### New Capabilities

- `vrpo-candidate-q-training`: Opt-in Candidate-Q critic, Q-boosted PPO training, leakage boundaries, provenance, and staged promotion against the unchanged GAE baseline.

### Modified Capabilities

None.

## Impact

- Likely touches `scripts/cleanba.py`, `ygoai/rl/jax/agent.py`, advantage/loss helpers in `ygoai/rl/jax/`, and (only for centralized Q) a separate privileged environment-to-learner observation path in `ygoenv/ygoenv/ygopro/ygopro.h`.
- Adds tests and experiment/report tooling for legal-menu alignment, alternating-player return perspective, critic calibration, leakage, finite targets, checkpoint compatibility, and matched-budget evaluations.
- Requires a new versioned checkpoint/experiment identity for VRPO variants; existing PPO checkpoints remain loadable by the unchanged default path and are not silently reinterpreted as Q-critic checkpoints.
