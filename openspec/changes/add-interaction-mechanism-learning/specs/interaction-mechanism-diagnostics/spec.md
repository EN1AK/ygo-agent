## Purpose

Provide auditable interaction-mechanism diagnostics that separate engine facts, observable information, conditional predictions and strategic hypotheses before any policy-training promotion.

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

Evaluation SHALL freeze splits before fitting, group related samples by duel, hold out interaction card combinations, report unknown coverage and compare with simple baselines. A frozen-policy diagnostic MUST NOT change actor weights or automatically start policy training.

#### Scenario: Prediction improves on development roots only
- **WHEN** performance improves only on previously examined roots or lacks sufficient held-out evidence
- **THEN** the report marks generalization unproven and blocks promotion

#### Scenario: Prediction passes its gate
- **WHEN** held-out prediction passes the predeclared gate
- **THEN** the output is a recommendation for a separately authorized matched policy experiment, not a claim of improved playing strength

### Requirement: Isolated and reproducible execution

Diagnostics SHALL preserve runtime, checkpoint, dataset and evidence hashes and SHALL NOT replace an active training runtime or share its GPU without authorization.

#### Scenario: Native provenance requires a repair
- **WHEN** a discrepancy is confirmed during active training
- **THEN** the repair is tested in an isolated version, the active process remains unchanged and old evidence is retained
