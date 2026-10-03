## Purpose

Provide an opt-in, auditable action-value training path that can test whether Q-boosted advantages improve Yu-Gi-Oh! self-play without changing default PPO behavior or leaking hidden information into the acting policy.

## ADDED Requirements

### Requirement: Default PPO remains unchanged
The system SHALL keep the existing PPO/GAE collection, model inference, loss, checkpoint loading, and evaluation behavior when no Candidate-Q/VRPO option is selected. The active 100M continuation SHALL NOT be modified or resumed with a new critic format as part of this capability.

#### Scenario: Legacy training command
- **WHEN** a training command omits Candidate-Q/VRPO options
- **THEN** it uses the same observation and action contracts, GAE targets, actor and value losses, and checkpoint format as before this change

#### Scenario: Existing model evaluation
- **WHEN** an existing PPO checkpoint is evaluated without Candidate-Q/VRPO options
- **THEN** no privileged critic input is requested and the action distribution is unchanged

### Requirement: Q estimates correspond to the actual legal menu
An enabled Candidate-Q run SHALL estimate one value per action in the policy-visible, staged legal menu at each decision. It SHALL use the same order, validity mask, and chosen-action index as the actor. Missing, duplicate, out-of-range, or reordered action identities SHALL fail closed before forming a target or policy update.

#### Scenario: Staged selection prompt
- **WHEN** the core presents a staged multi-select prompt
- **THEN** Q values and policy probabilities cover the current staged menu only and the selected Q value uses the actor's selected menu index

#### Scenario: Invalid menu alignment
- **WHEN** the candidate-Q menu cannot be matched exactly to the actor's menu
- **THEN** the run reports the mismatch and refuses that update rather than silently truncating, padding, or changing action identities

### Requirement: Observation-only and centralized critics remain distinct
The system SHALL label and record an observation-only Candidate-Q ablation separately from a centralized-Q VRPO variant. In the centralized variant, privileged full-state information SHALL be available only to critic training; actor logits, recurrent state, exported inference inputs, search inputs, and opponent-visible observations SHALL depend only on information legally visible to the acting player.

#### Scenario: Observation-only ablation
- **WHEN** Q is trained from the actor-visible observation only
- **THEN** artifacts identify the run as `candidate_q_observation` and do not claim to implement centralized VRPO

#### Scenario: Privileged-state perturbation
- **WHEN** hidden cards or deck order are changed while the acting player's legal observation and menu are held fixed
- **THEN** actor logits and inference inputs remain identical, while the learner-only centralized critic may change

#### Scenario: Inference export
- **WHEN** a centralized-Q checkpoint is exported for play or evaluation
- **THEN** no privileged critic field is required to select an action and no hidden-state field is serialized into the actor-facing request

### Requirement: Q-boosted targets preserve game and trajectory semantics
The Q-boosting estimator SHALL compute the current chosen-action term minus the policy-weighted current Q baseline, plus a backward trace of Expected-SARSA residuals over legal actions. It SHALL use a declared frozen reference policy for rollout/critic targets and, in the full VRPO actor update, recompute policy expectations from the current actor while holding critic values fixed. It SHALL produce a distinct chosen-action Q target for critic learning. Terminal rewards, truncation/bootstrap rules, alternating player perspective, recurrent boundaries, and invalid or forced-action masks SHALL be handled explicitly and consistently with the existing training reward convention. PPO's clipped actor objective SHALL remain the actor update.

#### Scenario: Terminal decision
- **WHEN** a trajectory ends after a decision
- **THEN** the estimator uses no value beyond the terminal state and assigns rewards from the appropriate player's perspective

#### Scenario: Opponent acts between own decisions
- **WHEN** control changes between players in a self-play trajectory
- **THEN** each player's Q, baseline, reward, and trace remain in that player's value perspective without incorrectly swapping or dropping intervening transitions

#### Scenario: Bounded rollout ends without terminal
- **WHEN** a rollout stops at a collection boundary
- **THEN** the estimator bootstraps from the correct reference-policy-weighted Q at the boundary and does not treat the truncation as a game loss

#### Scenario: Padded or forced decision
- **WHEN** a recurrent row is padding or the legal menu has only one action
- **THEN** all targets remain finite, padding contributes no update, and any forced-action policy loss is zero while the transition still advances the duel and recurrent history

### Requirement: Critic quality and run provenance are measurable
Candidate-Q/VRPO runs SHALL record the estimator mode, critic input scope, source/runtime/checkpoint hashes, legal-menu capacities, training context, and separate actor/critic metrics. Before using Q-boosted advantages for a longer run, the system SHALL provide finite-target, calibration, menu-alignment, leakage, and throughput evidence. A failed gate SHALL leave the default PPO path available and unchanged.

#### Scenario: Uncalibrated critic
- **WHEN** the Q critic fails the declared calibration or finite-target gate on held-out decisions
- **THEN** the run is not promoted to a longer Q-boosted training experiment

#### Scenario: Resume mode mismatch
- **WHEN** a run attempts to resume a checkpoint with a different estimator, critic-input scope, observation contract, or Q-head layout
- **THEN** it refuses automatic resume and reports the incompatible metadata

#### Scenario: Frozen specialist critic-only pilot
- **WHEN** the user-approved 121M SkyStriker frozen-actor shadow option is selected
- **THEN** only the independent critic is optimized, actor parameters/optimizer/step remain byte-identical, checkpoint and baseline contexts declare frozen mode, and selected loss diagnostics are excluded from fitting
- **AND** fresh self-play calibration and transfer to WindBot diagnostics remain separate gates; this pilot does not replace the matched 100M comparison or authorize Q actor updates

### Requirement: Comparative evaluation separates algorithm effects
The system SHALL complete and evaluate the user-requested 100M PPO baseline before starting Candidate-Q experiments, and compare restarted GAE, observation-only Candidate-Q, and centralized-Q VRPO as separately labelled conditions from that same frozen 100M actor. The historical 40M result and live PPO continuation SHALL NOT substitute for a separately restarted matched control. Reports SHALL include matched environment-step and wall-clock budgets, fixed seeds/decks/both seats, uncertainty for paired outcomes, invalid/timeout rates, SPS, Q calibration, combo completion, and interruption quality. Search or belief assistance SHALL NOT be silently mixed into these conditions.

#### Scenario: Promotion report
- **WHEN** a pilot finishes
- **THEN** the report identifies the highest gate reached and attributes changes only to the variant actually enabled, without treating a short-term win-rate difference as proof of improved equilibrium play or combo learning

#### Scenario: User reduces the pre-pilot WindBot budget
- **WHEN** the 2026-10-03 user-requested small diagnostic baseline is used instead of the originally planned 1,024 WindBot attempts
- **THEN** the baseline records the completed 40-game four-executor smoke, 10 fixed both-seat SkyStriker mirror attempts with the 100M actor versus WindBot, partial expansion/interruption evidence separately, and the first paired seed per block as its explicitly limited strategy-review slice
- **AND** this evidence may unlock bounded shadow-Q and matched short pilots after the other baseline checks, but SHALL NOT be represented as high-confidence strength promotion or waive critic calibration, finite-target, menu-alignment, leakage, or matched-control gates
