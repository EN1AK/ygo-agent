## Context

See `proposal.md` for motivation and `specs/vrpo-candidate-q-training/spec.md` for behavior. The current `scripts/cleanba.py` rollout stores acting-player observations, selected actions, logits, scalar values, rewards, `mains`, termination flags, and training masks. Its `advantage_fn` selects GAE/V-trace; `truncated_gae_sep` carries separate player returns across alternating turns, and the learner applies a state-value MSE plus PPO policy loss. `RNNAgent.q_head=True` returns logits without a value; it is not Q(s,a). The normal `structured-lite-v1` observation deliberately hides opponent private cards and deck order. Current checkpoints deserialize the existing agent variables directly, so a new Q parameter tree cannot be silently loaded as an old checkpoint.

The paper's estimator is `A_i = Q_i(s,a) - V_i(s) + trace_i`, with `V_i(s) = sum_a pi(a|o) Q_i(s,a)` and `trace_i = sum_k (gamma*lambda)^k [r_i + gamma*V_i(next) - Q_i(s,a)]`. Its chosen-action regression target is `Q_i(s,a) + trace_i`. The critic's approximation error can erase the variance benefit, so critic calibration is a gate, not a presumed outcome. [Reference: arXiv:2605.19235, sections 3.2 and 4.2](https://arxiv.org/html/2605.19235).

## Goals / Non-Goals

**Goals:**

- Make the unchanged GAE path a byte-compatible default and give each Q experiment an explicit mode, checkpoint identity, and reproducible report.
- Test the estimator and menu alignment on small deterministic trajectories before a GPU pilot; separately measure whether a Q critic is accurate enough to help.
- Establish a learner-only full-state boundary so a centralized critic can be tested without changing legal actor inputs or inference behavior.

**Non-Goals:**

- Alter the active H200 100M run, its loaded training implementation, or its checkpoint format.
- Treat `q_head=True` as Q, put privileged fields in actor observations, add search to PPO collection, or combine Q-boosting with new damping/belief changes in the same ablation.
- Claim a game-strength or combo improvement from variance reduction alone.

## Decisions

### 1. Opt-in modes and controlled baseline

Add an explicit Q-training mode with `off` (current PPO), `shadow_observation` (train/log Q but keep GAE actor), `qboost_observation` (observation-only ablation), and `vrpo_centralized` (learner-only full-state critic). The first two gates let us test Q quality and estimator mathematics without conflating them with actor improvement. Keep existing `--value` and all default flags unchanged; reject unsupported combinations rather than silently falling back. In matched GAE/Q-boosting pilots, hold actor architecture, starting actor weights, deck distribution, optimizer budget, UPGO setting, and all other hyperparameters fixed. Disable UPGO in both pilot arms if no mathematically specified Q-boosting analogue exists; do not compare that result to the live 100M continuation as a one-variable ablation.

The user superseded the original 40M starting point with "continue to 100M, then experiment with VRPO". Freeze and evaluate the actual 100M endpoint before any experimental training, including shadow mode. All matched arms import that identical actor and restart their declared optimizer state consistently. Keep 40M and 45M reports as historical comparisons, not causal controls. Reopen the baseline task until endpoint provenance and the required combo/interruption metrics exist; a report with pending/null manual metrics does not pass it.

On 2026-10-03 the user reduced the slow WindBot budget: use the completed 40
four-executor smoke attempts and a new fixed 10-attempt SkyStriker mirror
(100M versus WindBot) instead of waiting for the 1,024-attempt expansion.
Keep all extra completed/interrupted expansion attempts in a separate ledger.
The reduced strategy slice is the first paired seed of each of the four smoke
blocks and the SkyStriker block (10 duels, five candidate-first opportunities),
plus the already reviewed greedy examples. Retain invalid or unobservable
cases and classify uncertain tactics explicitly. This small diagnostic slice
can unlock only the bounded shadow/short-pilot sequence, not a claim of stable
strength improvement. Apply this same reduced baseline review to matched
pilot arms. Do not weaken finite, menu alignment, calibration, leakage, or
matched-budget gates; higher-confidence promotion needs stronger evidence.

Alternative rejected: directly replace GAE in `advantage_fn` or toggle `q_head`, either of which changes current behavior without a real action-value estimate.

### 2. Separate Q critic and legal-menu identity

Introduce a separate candidate-Q critic/optimizer state rather than reshaping the actor's policy head or current state-value head. The critic receives the current staged action features and exact `num_options`/mask used by actor logits, and emits Q for every valid menu slot. Record selected menu index and a stable menu digest in the rollout; validate the digest before Q target construction. Padded slots never enter the policy expectation. For central Q, provide seat-specific return channels (or an equivalent explicitly tested zero-sum transformation) so opponent actions and delayed rewards can be evaluated from either player's perspective. At every state, the policy probabilities come from the acting player's legal observation and recurrent state, including states where the opposite seat acts.

Alternative rejected: infer Q for unobserved whole-card combinations or use legacy EDOPro combination expansion; it would not align with the actual staged policy action.

### 3. Shadow critic before actor use

Initialize actor parameters from the frozen, validated and evaluated 100M checkpoint in new run directories. First fit `shadow_observation` Q on fresh rollouts while GAE continues to drive the actor; hold out deck/seed slices for chosen-action return calibration and rank/order diagnostics by prompt class. No Q-boosted long run is allowed until Q values, policy expectations, traces, and gradients are finite, menu alignment has zero violations, and the cost is measured. Then run a bounded `qboost_observation` pilot against a separately restarted matched GAE control. Label its result an ablation, not full VRPO.

Alternative rejected: jump directly to a centralized long run; that would combine a new native data channel, new critic, and new estimator before identifying failures.

### 4. Q-boosting and critic targets follow full trajectories

User-approved specialist exception (2026-10-03): `shadow_observation` with
`q_freeze_actor` skips the entire PPO update. Byte-guard parameters, batch
statistics, optimizer and step on every collection batch; record the frozen
mode and parent hash in Q checkpoint context and require a matching baseline
manifest. Start from the frozen 121M SkyStriker actor with fresh seeds and Q
optimizer. Collected stochastic self-play is a different opponent/policy
distribution from argmax-vs-WindBot diagnostics; report that transfer gap, not
a calibrated WindBot Q claim. No fitting on the five selected diagnostic roots.
Use at most 262,144 fresh transitions for this initial critic-only pilot.

User extension (2026-10-04): after the warm-encoder pilot, continue its Q weights
and optimizer to cumulative 2M transitions (allow one batch of rounding up to
2,007,040). First benchmark isolated throughput configurations while retaining
minibatch128, horizon64 and unchanged objectives. Use a hash-bound continuation
manifest; the original initial-pilot cap stays unchanged. Explicit source-only
resume must name the old source and retain exact other context validation.
Benchmark weights are discarded, not used as continuation parents. Frozen actor
and no Q-boost promotion remain mandatory. Topology changes are reported.

Compute `V_i` as a masked policy expectation over each legal menu; form Expected-SARSA residuals and backward traces per seat through all observed transitions, not just the acting seat's decision rows. A true terminal has zero bootstrap; a collection boundary uses next-state policy expectation; explicit environment timeout follows the existing reward/termination convention and is separately counted. Recurrent padding is sanitized before softmax/Q arithmetic and excluded from reductions. The Q regression target is `Q_i(chosen) + trace_i`, not the scalar GAE target. Freeze reference policy probabilities for rollout/critic targets. In full VRPO actor minibatches, recompute policy-weighted terms with current actor probabilities while stop-gradient holds Q fixed; retain PPO ratio clipping and train actor only on newly collected on-policy data. Start without a replay buffer or new regularization so those do not confound the first comparison.

Alternative rejected: replace only `V(next)` with `sum pi Q` inside GAE; that omits `Q(chosen)-V(current)` and does not implement Q-boosting.

### 5. Learner-only privileged data channel

For `vrpo_centralized`, add a versioned, opt-in full-state record separate from `obs:*` and its existing exported actor schema. It may contain real zone identities/order and chain context required by Q; it must never feed actor encoder/logits/recurrent state, action selection, public replay, or evaluation requests. The actor-side rollout worker may transport this record to the learner, but actor inference must accept only the unchanged legal observation. The existing `oppo_info` switch is not this boundary and must not be repurposed. Add a perturbation fixture that changes hidden state with legal observation/menu held fixed: actor logits remain identical, Q may change. Fail closed if privileged records are missing, stale, or misaligned with a decision.

Alternative rejected: adding true hidden cards to `structured-lite-v1` or sharing a feature encoder trained with privileged inputs; either risks hidden-information leakage at inference.

### 6. Checkpoints, deployment, and promotion

Q modes use a versioned envelope with actor parameters, separate Q state/optimizer, mode, critic-input schema, reward convention, source/runtime hashes, and training context. Importing old PPO actor weights is explicit and does not pretend to restore a Q optimizer. Actor-only export preserves the existing evaluator interface. A checkpoint mode/schema mismatch fails before collection. Keep the 100M trainer, adapter, native module and runtime assets untouched by this experimental change while it runs; develop and validate in an isolated checkout/build. Preparatory helper-only source sync is allowed under the project's four-way convention after verifying that every existing training-path file is unchanged; it does not change the active run's recorded source identity or accept an experimental pilot. Deploy or start the separate experimental training variant only after the baseline gate, preserving machine-local assets and the running process.

Promotion order: deterministic estimator oracle and leakage tests; shadow-Q calibration/finite/throughput gate; matched short GAE vs observation-Q pilot; centralized data-channel/critic gate; matched centralized VRPO pilot; held-out both-seat games and combo/interruption review. Reports must show equal environment-step and wall-clock views, paired uncertainty, invalid/timeout rates, effective SPS, and which gate was actually passed.

## Risks / Trade-offs

- [Q critic error overwhelms variance reduction] → Shadow calibration, prompt-stratified return checks, and no long actor run before the gate.
- [Privileged data leaks through shared actor features or serialization] → Separate schema/parameter path, hidden-state perturbation tests, and actor-only inference export.
- [Alternating-turn reward sign or timeout bootstrap is wrong] → Hand-computed two-seat trajectory fixtures and parity checks against current GAE reward semantics.
- [Dynamic menu size makes Q expensive] → Use the existing capped staged menu, benchmark memory/SPS, and reject rather than truncate on overflow.
- [A/B result reflects changed UPGO, optimizer, or opponent mix] → Frozen baseline plus separately restarted, matched control; one major algorithm variable per comparison.
- [Existing 100M GPU job is disturbed] → No H200 production module replacement or changes to the active training path; use isolated CPU validation resources and verify preparatory helper-only sync leaves active file/asset hashes unchanged.

## Migration Plan

1. Implement and validate mode `off` parity and new checkpoint rejection before enabling Q modes.
2. Gate shadow/observation pilots and centralized critic work independently; never auto-migrate an existing PPO checkpoint into a Q checkpoint.
3. After the live run ends, deploy only an accepted pilot variant with a new run directory and provenance. Roll back by selecting `off` and the unchanged PPO checkpoint/source snapshot, not by overwriting training artifacts.
