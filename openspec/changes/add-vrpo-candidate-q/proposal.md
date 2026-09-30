## Why

The active self-play learner uses PPO with GAE and a state-value critic. In stochastic imperfect-information play, sampled future actions can make its advantage estimates noisy; the current `q_head` is not a trained action-value critic. We need a controlled way to test whether candidate-action Q and Q-boosting improve this agent before committing to expensive search or another long training run.

## What Changes

- Add an opt-in Candidate-Q research path over the exact policy-visible staged legal-action menu, with a separate action-value critic and Q-boosted advantage estimator. Keep the existing PPO/GAE path as the default and preserve its checkpoint and inference contracts.
- Stage the work: first instrument and validate an observation-only Candidate-Q ablation; then add a strictly learner-only privileged full-state critic input for a faithful centralized-Q experiment. Label these variants separately and never call the observation-only ablation full VRPO.
- Compare GAE, Candidate-Q, and centralized Q-boosting under frozen 40M baselines, matched environment-step and wall-clock budgets, fixed deck/seeds/both-seat evaluations, value calibration, combo and interruption metrics, and throughput/error gates.
- Keep search, belief modeling, self-play damping, and the running H200 40M continuation outside this change. No search targets or privileged fields may enter actor inference.

## Capabilities

### New Capabilities

- `vrpo-candidate-q-training`: Opt-in Candidate-Q critic, Q-boosted PPO training, leakage boundaries, provenance, and staged promotion against the unchanged GAE baseline.

### Modified Capabilities

None.

## Impact

- Likely touches `scripts/cleanba.py`, `ygoai/rl/jax/agent.py`, advantage/loss helpers in `ygoai/rl/jax/`, and (only for centralized Q) a separate privileged environment-to-learner observation path in `ygoenv/ygoenv/ygopro/ygopro.h`.
- Adds tests and experiment/report tooling for legal-menu alignment, alternating-player return perspective, critic calibration, leakage, finite targets, checkpoint compatibility, and matched-budget evaluations.
- Requires a new versioned checkpoint/experiment identity for VRPO variants; existing PPO checkpoints remain loadable by the unchanged default path and are not silently reinterpreted as Q-critic checkpoints.
