## Purpose

Provide auditable interaction-mechanism diagnostics and executable capability exercises for combat, card interactions and multi-step combos. Keep verified facts, task success, strategic preferences and real playing strength distinct throughout evaluation and controlled teaching.

## ADDED Requirements

### Requirement: Evidence-linked event provenance

The diagnostic system SHALL associate chain events with their actual link and publicly established source, retaining message and observation evidence. Unknown identities MUST remain unknown rather than inherit the latest activation.

#### Scenario: Earlier link resolves after a response
- **WHEN** link two resolves before link one, including when either source leaves the field
- **THEN** each recorded solving, solved, negated or disabled event identifies the corresponding link and established source, and the report exposes any discrepancy with the frozen runtime

#### Scenario: A new chain or duel starts
- **WHEN** a previous chain or duel has ended
- **THEN** its source mapping does not contaminate the new chain or duel

### Requirement: Fact labels distinguish unknown and strategic judgment

Each diagnostic sample SHALL preserve evidence for cost, response and public outcome labels, with explicit unknown values. Chain resolution MUST NOT by itself imply successful effect application or a beneficial action.

#### Scenario: Target escapes before resolution
- **WHEN** a target leaves through a response and the original effect resolves
- **THEN** the record distinguishes the observed payment, movement and effect outcome from the hypothesis that passing was strategically preferable

### Requirement: Partial-observation-safe conditional prediction

Student inputs SHALL exclude engine-private opponent information. Retrospective predictions conditioned on an observed response MUST be identified as such and MUST NOT be reported as decision-time foresight or expected action value.

#### Scenario: Teacher sees a hidden hand
- **WHEN** engine-private state is used to verify a label
- **THEN** the private state is not present in student input or inference lookup keys

#### Scenario: Opponent response is not yet known
- **WHEN** an action is being selected
- **THEN** the system does not supply the actual future response as an input

### Requirement: Generalization and promotion gates

Evaluation SHALL freeze splits before fitting, group related samples by source duel/seed group and template/combo family, hold out interaction card combinations, report unknown coverage and compare with simple baselines. All descendants connected by a shared source SHALL remain in one split or be quarantined. A frozen-policy diagnostic MUST NOT change actor weights or automatically start policy training. Development exercises previously inspected or used for curriculum selection MUST NOT be relabelled as unseen tests.

#### Scenario: Prediction improves on development roots only
- **WHEN** performance improves only on previously examined roots or lacks sufficient held-out evidence
- **THEN** the report marks generalization unproven and blocks promotion

#### Scenario: Prediction passes its gate
- **WHEN** held-out prediction passes the predeclared gate
- **THEN** the output identifies eligibility for a separately configured matched policy experiment, not a claim of improved playing strength or an automatic training launch

#### Scenario: A combo is cut into suffix exercises
- **WHEN** several exercises share a full combo source or template ancestor
- **THEN** all suffixes and generated variants retain that group identity and do not cross training and held-out splits

### Requirement: Versioned executable exercise contract

Each accepted exercise SHALL declare its capability, lineage, runtime/assets, reproducible starting point, player perspective, legal observation/action contract, applicability conditions, opponent/randomness assumptions, goal and constraints, execution budget, grading version, answer evidence and split. Supervision SHALL be typed as conditional fact, supported action preference or goal attainment. Human-readable explanations and evaluator goals MUST NOT silently become actor inputs.

#### Scenario: Required evidence is missing
- **WHEN** an exercise lacks a reproducible root or evidence for a requested label
- **THEN** it is rejected or that label remains unknown, and it is excluded from the corresponding supervised target

#### Scenario: A goal is not part of the policy interface
- **WHEN** the evaluator assigns a target board or resource objective
- **THEN** the policy receives only its established legal inputs and the result is labelled as performance on that declared objective, not evidence of a general instruction-following policy

### Requirement: Engine-verified generation and scoped answers

Reference solutions SHALL be executed through the real engine with pinned assets. Exercises represented as real-game positions SHALL have a legal construction or replay prefix. Critical-condition contrasts and answer-preserving variations SHALL identify the changed condition and retain family lineage. A bounded failure to find a solution MUST NOT be labelled as proof that no solution exists; an unexamined action MUST NOT be labelled incorrect merely because it differs from the reference.

#### Scenario: A sacrifice gains a benefit
- **WHEN** a condition change causes destroying one's own monster to achieve the declared goal
- **THEN** the contrast is graded by the verified consequence rather than by a blanket prohibition on attacking stronger monsters

#### Scenario: A solution relies on one opponent response
- **WHEN** a route succeeds against a fixed legal response
- **THEN** its answer records that condition and does not claim success against all responses or use a hidden actual response as a decision-time preference label

#### Scenario: An artificial position is supplied
- **WHEN** a position cannot be shown reachable through the declared game rules
- **THEN** it is excluded from real-game exercise results and can only be retained as a separately labelled synthetic mechanism fixture

### Requirement: Goal-based combo grading

Combo exercises SHALL support finishing, continuation, complete-opening and interrupted-recovery difficulty classes with declared prerequisites and budgets. Success SHALL depend on legal execution and the goal/resource constraints at the declared stable resolution boundary, not exact reference-sequence matching. Rule-level chain links and combo decision length SHALL be reported separately. Execution outcomes SHALL distinguish success, verified failure, unknown, invalid fixture and budget exhaustion; rejected cases and coverage MUST be reported.

#### Scenario: An alternative route reaches the goal
- **WHEN** a different legal ordering or route meets all declared constraints
- **THEN** it receives success even when its sequence differs from the reference solution

#### Scenario: A pending effect changes the apparent end board
- **WHEN** the visible board meets the goal before mandatory resolution has completed
- **THEN** success is deferred until the declared stable boundary and the resulting constraints are checked again

#### Scenario: A rollout exceeds its budget
- **WHEN** execution stops at a resource or step limit without a verified result
- **THEN** the report records the exhausted budget and any task-defined noncompletion without asserting that the position is unsolvable

### Requirement: Faithful recurrent context restoration

Mid-sequence exercises SHALL restore both the environment context and each acting player's legal history. Recurrent states SHALL correspond to the evaluated checkpoint, either through verified same-checkpoint capture or recomputation from its legal history. A training consumer SHALL preserve sequence boundaries and the declared history warm-up policy. Restoration discrepancies MUST invalidate the exercise rather than become a policy error.

#### Scenario: A newer checkpoint attempts an old combo suffix
- **WHEN** the source trajectory was captured with a different checkpoint
- **THEN** recurrent state is recomputed for the new checkpoint and the old checkpoint's state is not reused or replaced with an unexplained zero state

#### Scenario: Replay restores a different menu
- **WHEN** the restored root or any reference step has a mismatched legal menu, observation or response mapping
- **THEN** execution stops with an invalid-fixture result and no previous trajectory label is attached to the divergent observation

### Requirement: Small verified pilot before teaching

The first milestone SHALL target 24 accepted development exercises, eight each for basic combat/lethal, interaction consequences/exceptions and combo progression, while inspecting at most 48 candidates. Each category SHALL include a verified critical-condition contrast; combo coverage SHALL include finishing, continuation and complete opening. Every accepted exercise SHALL receive manual evidence review and two consistent engine reference runs. The milestone SHALL deliver immutable manifests, evidence, rejection reasons and a frozen-checkpoint baseline through the existing policy interface without parameter updates. Unavailable metrics SHALL be explicitly marked rather than invented.

#### Scenario: The old collection contains 500 windows
- **WHEN** historical windows are considered for the pilot
- **THEN** only individually audited labels and correctly associated legal contexts count as accepted exercises, regardless of the raw collection size

#### Scenario: The candidate budget cannot supply reliable coverage
- **WHEN** 48 candidates have been inspected without the required accepted counts or category coverage
- **THEN** the pilot remains incomplete, reports the gaps and does not automatically expand its budget or start training

#### Scenario: The pilot achieves a perfect score
- **WHEN** the frozen model or a later learner solves all pilot development exercises
- **THEN** the report limits the claim to that development set and does not declare unseen-family generalization or improved playing strength

### Requirement: Explicit teaching and curriculum boundaries

Teaching SHALL use a separate opt-in entry point, a copied starting checkpoint and predeclared data, optimizer, sequence, compute-budget and stopping configurations. Verified demonstrations, supported preferences and conditional-fact supervision SHALL remain separately identified; unknown targets SHALL be masked out. The initial teaching experiment SHALL use a fixed capability/depth mixture. Adaptive sampling SHALL be introduced as a separate comparison, use development progress only, preserve coverage floors and ceilings, and never sample from the frozen test set. Exercise goals SHALL be compatible with the unchanged actor objective unless a separate goal-conditioned interface is specified.

#### Scenario: A diagnostic command completes
- **WHEN** labels, baseline scores or frozen probes have been produced
- **THEN** no actor optimizer is invoked and no production checkpoint is replaced

#### Scenario: A valid demonstration chooses one of several legal actions
- **WHEN** that demonstration is used for supervised teaching
- **THEN** its chosen route is identified as a demonstrated solution rather than an exhaustive set of strategically correct actions

#### Scenario: The curriculum finds a persistently failing category
- **WHEN** a sampling update considers that category
- **THEN** evidence validity and solvability status are checked, sampling respects the declared bounds, and unknown or invalid tasks are not repeatedly treated as learnable failures

### Requirement: Separate mechanism learning from strength promotion

Mechanism probes, demonstration teaching, auxiliary prediction losses, adaptive curricula and search distillation SHALL be evaluated as separately identified changes. Search-derived targets SHALL retain their accepted upstream mode, information assumptions and uncertainty. The initial teaching experiment MUST NOT add environment rewards for activation count, combo length or unverified intermediate advantage. Experiments SHALL report development retention, held-out exercise transfer and unassisted full-game performance separately, including invalid/timeouts and uncertainty. Teacher generation, search, offline updates and environment interaction SHALL be included in the declared total cost.

#### Scenario: Teaching is compared with ordinary continuation
- **WHEN** a policy-strength experiment is prepared
- **THEN** it freezes a common starting checkpoint/runtime, matched PPO controls, an equal-compute ordinary-update control, held-out decks/seeds/both seats/opponents, environment-step and total-time views, and meaningful-effect/stopping criteria before fitting

#### Scenario: Exercise accuracy improves but duel evidence is weak
- **WHEN** held-out games show insufficient or uncertain improvement
- **THEN** the result is reported as exercise transfer only, and neither playing-strength promotion nor automatic long training follows

### Requirement: Auditable scene and decision intake

Exercise intake SHALL accept user-described scenes and recorded suspicious decisions as versioned candidates with source hashes, lineage, suspicion reasons and missing prerequisites. Suspicion detectors MUST NOT assign correctness labels. Unsupported scenes and incomplete histories SHALL remain candidates rather than be silently mapped to an existing template. Mechanism verification alone SHALL NOT grant actor-training eligibility.

#### Scenario: A training log flags an apparently bad attack
- **WHEN** a decision is reported as suspicious without verified counterfactual outcomes
- **THEN** intake preserves the original decision and its provenance, assigns no negative action label and lists the required replay and goal evidence

#### Scenario: A known starter passes its engine branches
- **WHEN** two independent engine runs agree on each requested branch from a synthetic field
- **THEN** the report records the precise runtime and synthetic scope while actor-interface and current-checkpoint history gates remain explicit prerequisites for training

#### Scenario: A free-text scene has no recovery adapter
- **WHEN** the scene cannot be instantiated by a declared template or verified legal replay
- **THEN** intake emits a candidate with missing conditions and execution refuses to guess a compatible fixture

### Requirement: Isolated and reproducible execution

Diagnostics SHALL preserve runtime, checkpoint, dataset and evidence hashes and SHALL NOT replace an active training runtime or share its GPU without authorization.

#### Scenario: Native provenance requires a repair
- **WHEN** a discrepancy is confirmed during active training
- **THEN** the repair is tested in an isolated version, the active process remains unchanged and old evidence is retained
