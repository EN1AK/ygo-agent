## Purpose

Define a versioned, partial-observation-safe representation of cards, selection context, legal actions, their causal relationships, and recent public events so policy and value learning can distinguish strategically different legal responses.

## ADDED Requirements

### Requirement: Versioned observation contract
The system SHALL expose Structured-lite observations under an explicit schema version and SHALL provide card, global, selection-context, structured-action, role-reference, public-event, and padding-mask tensors with documented shapes, field meanings, ranges, and truncation behavior.

#### Scenario: Structured environment creation
- **WHEN** an environment is created with the Structured-lite schema version
- **THEN** every observation contains all required tensors with stable dtypes and configured capacities

#### Scenario: Capacity overflow
- **WHEN** a legal-action or role-reference collection exceeds its configured capacity
- **THEN** the environment fails the sample or records an explicit overflow indicator instead of silently presenting the truncated sample as complete

### Requirement: Partial-observation safety
The system MUST derive Structured-lite features only from information legally visible to the acting player at that decision, except in an explicitly named privileged-critic evaluation mode that cannot feed the policy path.

#### Scenario: Hidden opponent card
- **WHEN** an opponent card is not publicly identified
- **THEN** its identity, static semantics, effect tags, and historical references do not reveal the hidden card to the policy observation

#### Scenario: Publicly revealed card
- **WHEN** an opponent card becomes publicly identified by the engine
- **THEN** the observation may expose its identity and derived semantics for as long as the game rules and maintained public history permit

### Requirement: Deterministic static card semantics
The system SHALL generate versioned static card features from pinned card-database and card-script assets. The first Structured-lite version SHALL include deterministic card type and numeric attributes plus an audited, bounded effect-tag vocabulary, and SHALL record source hashes and coverage statistics.

#### Scenario: Repeated semantic build
- **WHEN** the semantic builder runs twice with identical source assets and configuration
- **THEN** it produces byte-identical semantic tables and metadata hashes

#### Scenario: Unsupported or ambiguous effect semantics
- **WHEN** a card effect cannot be classified reliably
- **THEN** the system emits an unknown or low-confidence representation rather than asserting an exact effect meaning

### Requirement: Structured selection context
The observation SHALL describe the active engine prompt, chain depth, response status, forced/finish/cancel constraints, and multi-selection progress needed to interpret the current legal-action set.

#### Scenario: Chain response menu
- **WHEN** the acting player is offered responses to an active chain link
- **THEN** the selection context identifies that the decision is a chain response and references the visible source of the active chain link when available

#### Scenario: Multi-card selection
- **WHEN** a prompt requires a bounded number of cards to be selected
- **THEN** the context distinguishes minimum, maximum, already-selected, finishable, cancelable, and must-continue states

### Requirement: Relationship-aware legal actions
Each legal action SHALL have a structured action record and zero or more role-aware references to visible cards. The supported roles SHALL distinguish at least action source, active effect source, candidate card, target, cost, material, tribute, and already-selected card, while allowing unknown roles and confidence levels.

#### Scenario: Response affects a different object
- **WHEN** a candidate response originates from one card, targets another card, and responds to an effect whose source is a third card
- **THEN** those three relationships are represented separately rather than collapsed into one card identity

#### Scenario: Relationship cannot be established
- **WHEN** the environment cannot reliably bind a role to a visible card
- **THEN** the reference is absent or marked with reduced confidence and never points to an unrelated card as an exact binding

### Requirement: Bounded public event history
The observation SHALL include a bounded, ordered history of public engine events sufficient to represent recent activations, targets, negations, destruction, movement, draw/search, summon, chain-solving, and chain-end outcomes, with actor and visible-card references where available.

#### Scenario: Continuous spell is destroyed during resolution
- **WHEN** a public chain response destroys a face-up continuous spell
- **THEN** subsequent observations record the destruction and chain resolution without reporting that the spell remained active

#### Scenario: Event history truncation
- **WHEN** public events exceed the configured history capacity
- **THEN** the retained ordering and truncation policy are deterministic and observable in diagnostics

### Requirement: State-conditioned action scoring
The Structured-lite model SHALL encode each legal action using its structured fields, role references, and the current visible scene before producing one masked logit per legal action. The value output SHALL remain a state value rather than being presented as per-action Q-values.

#### Scenario: Legal action scoring
- **WHEN** a valid Structured-lite observation contains N legal actions
- **THEN** the policy returns N usable logits in corresponding order and masks all padded action slots

#### Scenario: Value reporting
- **WHEN** inference returns a critic value without an explicit Q estimator
- **THEN** logs and reports label it as `V(s)` and do not label it as `Q(s,a)`

### Requirement: Explicit checkpoint compatibility
The system SHALL store observation schema, model architecture, semantic-table hashes, code-list hash, and capacity metadata with Structured-lite checkpoints. It MUST reject incompatible checkpoint loading unless an explicit, validated migration is requested.

#### Scenario: Legacy checkpoint evaluation
- **WHEN** a legacy checkpoint is selected with its matching legacy observation schema
- **THEN** it remains runnable for baseline evaluation without Structured-lite tensors

#### Scenario: Incompatible direct load
- **WHEN** a legacy checkpoint is loaded as if it were a Structured-lite checkpoint without migration
- **THEN** loading fails with a compatibility error before training or evaluation begins

#### Scenario: Approved warm-start migration
- **WHEN** a migration tool recognizes compatible legacy parameter subtrees
- **THEN** it copies only shape-compatible parameters, initializes new parameters explicitly, and writes a migration report containing source and output hashes

### Requirement: Reward and label separation
The system SHALL keep the existing terminal game-return objective for value learning. Public event outcomes and local tactical assertions SHALL be available for diagnostics, auxiliary representation learning, or regression evaluation, but SHALL NOT silently replace terminal return as the value target.

#### Scenario: Short combo trace
- **WHEN** a demonstration ends after a combo without a verified game terminal
- **THEN** it can supervise policy behavior but cannot create a terminal value label

#### Scenario: Tactical regression
- **WHEN** a snapshot proves that one response interrupts an effect and another does not
- **THEN** the report may assert the causal outcome while keeping any global action preference separately qualified by continuation-value evidence

### Requirement: Promotion evidence
Structured-lite SHALL be promoted only after producing reproducible coverage, leakage, snapshot-regression, combo, throughput, memory, value-calibration, and fixed-seed head-to-head reports against the legacy baseline.

#### Scenario: Veiler versus Ogre regression
- **WHEN** the validated decision `battle-62-seat1:14` is evaluated
- **THEN** diagnostics distinguish the continuous-spell effect source, Effect Veiler's monster target, Ghost Ogre's effect-source interaction, and their different chain outcomes

#### Scenario: Resource regression
- **WHEN** Structured-lite training throughput or memory is measured on the GPU server
- **THEN** the report records configuration, hardware, observation sizes, steps per second, peak memory, and comparable legacy-baseline measurements

#### Scenario: Promotion decision
- **WHEN** evaluation is complete
- **THEN** promotion is based on target-deck results and regression gates rather than aggregate training loss alone
