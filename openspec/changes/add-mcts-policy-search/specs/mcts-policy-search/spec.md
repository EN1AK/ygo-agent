## Purpose

Provide deterministic, auditable, model-guided candidate rollout, with PUCT as a separately labelled comparator, that can improve Yu-Gi-Oh! decisions without leaking hidden information or corrupting the existing PPO training and evaluation contracts.

## ADDED Requirements

### Requirement: Duel states support deterministic snapshot and rollback
The system SHALL expose owned duel snapshots that restore the engine, random-number state, pending protocol prompt, response buffers, and environment-visible state to the exact captured point. Snapshot handles MUST be scoped to their originating duel, MUST be explicitly releasable, and MUST fail loudly on stale, foreign, released, or capacity-exhausted use.

#### Scenario: Snapshot round trip
- **WHEN** a duel is advanced after a snapshot, rolled back, and replayed with the same responses
- **THEN** every emitted engine message, legal-action menu, observation digest, acting player, reward, terminal flag, and subsequent random outcome matches the original branch

#### Scenario: Invalid snapshot ownership
- **WHEN** a snapshot is applied to another duel or after its lifetime has ended
- **THEN** rollback is rejected with a named error and neither duel is mutated

### Requirement: Search preserves the model and protocol contracts
The system SHALL evaluate candidate root actions over the current legal-action indices using one declared evaluator checkpoint, separate recurrent states for both player views, terminal rewards on the checkpoint's reward scale, and values converted exactly once to the root player's perspective. Multi-step selection and chain-response prompts MUST remain explicit protocol nodes, and chain-response prompts MUST NOT be treated as stable critic leaves. Any PUCT mode MUST obey the same contracts and be labelled separately.

#### Scenario: Search through a response chain
- **WHEN** a rollout or PUCT simulation reaches a `select_chain` response prompt before its depth or node budget is exhausted
- **THEN** the simulation expands through the response until a terminal state or resolved-chain boundary is reached before backing up a nonterminal critic value

#### Scenario: Stable legal-action identity
- **WHEN** candidate-action returns, updated-policy probabilities, or PUCT root visit probabilities are returned
- **THEN** they align with the exact legal-action ordering recorded at the root and can be mapped back to the same raw engine responses without permutation or truncation

### Requirement: Candidate rollout is paired and policy-regularized
The initial search mode SHALL evaluate eligible candidate actions with a declared particle, opponent-policy, continuation-policy, rollout-seed, horizon, and leaf-value protocol. It SHALL record per-action sample counts, mean returns, and uncertainty. The local action distribution SHALL be derived from those returns with an explicit regularization toward the raw policy, not an unlabelled hard maximum. PUCT SHALL be evaluated separately at matched inference cost before any claim that tree search is preferable.

#### Scenario: Paired candidate comparison
- **WHEN** two root actions are evaluated under a shared particle and rollout seed
- **THEN** both action-return records identify the shared conditions, the updated policy uses only completed valid comparisons, and the chosen action remains in the recorded root legal menu

### Requirement: Hidden-information modes are explicit and safe
The system SHALL distinguish exact-state diagnostic search from belief-particle search. Exact-state search MUST be labelled as oracle-only and MUST NOT be reported as fair-play strength or used to choose actions in promoted evaluation. Belief-particle search MUST replace hidden state before any branch-specific model query and MUST aggregate root statistics across auditable, reproducibly sampled particles.

#### Scenario: Exact-state diagnostic output
- **WHEN** search uses the original hidden opponent state
- **THEN** its output is marked `oracle_exact_state`, excluded from fair-play summaries, and contains no claim of deployable playing strength

#### Scenario: Belief particle search
- **WHEN** fair-play search is requested from an information set
- **THEN** every simulation identifies its particle seed and hidden-state hypothesis, and no model or search decision observes hidden identities unavailable to the acting player

### Requirement: Search is bounded and has a deterministic fallback
Every search request SHALL declare candidate count, particles, rollouts per action, depth, wall-time, and snapshot-memory budgets (and node budget for PUCT). If a budget or recoverable snapshot error prevents a complete comparison, the system MUST record the reason and use the raw policy action unless the caller explicitly selects a stricter failure mode.

#### Scenario: Wall-time budget expires
- **WHEN** search reaches its wall-time limit before all requested simulations complete
- **THEN** the result records completed simulations and the timeout reason, releases all snapshots, and returns the configured raw-policy fallback

### Requirement: Belief training keeps privileged labels outside actor inputs
The system SHALL train a separate autoregressive hidden-card predictor with legally available observation features and public slot descriptors. Hidden identities MAY be used as supervised labels but MUST NOT affect predictions for earlier slots or be appended to the PPO actor observation. Known-deck self-play priors MUST be declared and MUST NOT be silently applied to unknown-deck external opponents. Validation MUST split by whole duel and distinguish token calibration, joint-particle constraint validity, held-out deck generalization, and fair-play playing strength.

#### Scenario: Future label perturbation
- **WHEN** a supervised hidden identity at a later slot changes
- **THEN** earlier slot logits and masks remain unchanged and the PPO actor inputs remain unchanged

#### Scenario: Offline belief pilot completes
- **WHEN** separate belief updates and held-out metrics finish successfully
- **THEN** the report preserves data/runtime/checkpoint provenance and does not promote fair-play search without native particle-application and decision-quality gates

### Requirement: Search decisions are fully auditable
The system SHALL persist the raw policy logits and value, legal actions, snapshot identity, evaluator checkpoint hash, search configuration, per-rollout action prefix and leaf evaluation, candidate return distribution, regularization coefficient, updated policy, selected action, fallback status, reward mode, and terminal outcome. PUCT mode SHALL additionally persist root visits and backed-up edge values. Critic values, Monte Carlo returns, edge means, and per-action Q estimates MUST retain distinct labels.

#### Scenario: Replay a searched decision
- **WHEN** an auditor restores a recorded searched decision with the pinned runtime and checkpoint
- **THEN** root identity checks pass and the audit contains enough seeds, prefixes, particle identifiers, and model outputs to reproduce the selected action within the declared deterministic mode

### Requirement: Search integration is opt-in and does not silently alter PPO
Search-assisted evaluation and search-target generation SHALL be disabled by default. Search targets MUST be stored as a versioned auxiliary artifact and MUST NOT replace PPO actions, advantages, rewards, or losses unless a later explicitly configured training stage consumes them.

#### Scenario: Existing PPO command runs without search flags
- **WHEN** the existing training command is run without a search configuration
- **THEN** rollout actions, losses, checkpoint format, and throughput behavior remain unchanged

#### Scenario: Generate a distillation target
- **WHEN** an eligible decision completes a valid search
- **THEN** the artifact stores a mode-labelled updated-policy target (or PUCT visit-count target) and searched root value separately from the original policy/value output and links them to the source trace and checkpoint hashes

### Requirement: Promotion requires correctness and strength gates
The system MUST NOT promote search to standard evaluation or training until snapshot differential checks, hidden-information leakage checks, finite-value checks, bounded resource cleanup, and paired fixed-seed strength comparisons pass. Throughput and latency MUST be reported separately for replay and native snapshot backends.

#### Scenario: Native backend fails differential validation
- **WHEN** native rollback diverges from deterministic replay on any accepted fixture
- **THEN** native search is blocked from evaluation and target generation while replay diagnostics remain available
