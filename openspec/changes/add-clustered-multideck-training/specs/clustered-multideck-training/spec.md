# Spec Delta

## Purpose

Defines training a new Structured-lite policy from random initialization across
many decks using auditable hierarchical cluster sampling, elevated elfnote
exposure, and reproducible resume behavior.

## ADDED Requirements

### Requirement: Hierarchical cluster-balanced sampling
The trainer SHALL sample each player deck through a hierarchy of anchor reserve,
cluster, near-duplicate family, and canonical deck rather than uniformly over raw
YDK files. The default per-seat elfnote reserve SHALL be 25%; the remaining mass
SHALL be distributed uniformly across eligible non-anchor clusters, then across
families and canonical decks within each cluster.

#### Scenario: Large cluster does not dominate
- **WHEN** one cluster contains substantially more deck files than another
- **THEN** both clusters receive the configured equal cluster-level probability before family/deck selection

#### Scenario: Elevated elfnote sampling
- **WHEN** the default curriculum is active
- **THEN** the realized per-seat elfnote frequency converges to 25% within the configured statistical tolerance

#### Scenario: Near-duplicate family has many variants
- **WHEN** one family contains many more canonical variants than other families in its cluster
- **THEN** that family does not gain probability merely from its variant count

### Requirement: Seat and matchup coverage
The sampler SHALL balance elfnote and non-elfnote decks across player 0 and player
1, maintain a nonzero floor for every ready cluster, and emit enough identifiers
to report games and learner decisions by seat, deck, family, cluster, and matchup.

#### Scenario: Realized distribution audit
- **WHEN** a deterministic sampler audit generates at least the configured minimum sample count
- **THEN** every cluster is observed and anchor, cluster, family, deck, and seat frequencies fall within declared tolerances

#### Scenario: Resume sampling
- **WHEN** training resumes from a saved checkpoint with the same manifest and sampler state
- **THEN** subsequent deck and seat selections match an uninterrupted run

### Requirement: Versioned initialization and asset compatibility
The initial multi-deck smoke and pilot SHALL use random initialization with a
recorded seed, feature schema, code-list and semantic hashes. The pinned 40M
`structured-lite-v1/full` checkpoint SHALL remain immutable as a reference and
optional migration source. If a later continuation uses it and the corpus adds
card codes, migration SHALL preserve existing code-list order, append new codes
deterministically, copy compatible parameters and old embedding rows exactly,
initialize only new rows with a recorded seed/method, regenerate matching
semantic assets, and write source/output hashes and a migration report.

#### Scenario: Fresh policy initialization
- **WHEN** the initial multi-deck run starts with all ready cards covered by the frozen assets
- **THEN** the trainer initializes the policy without loading 40M weights and records the initialization seed and asset hashes

#### Scenario: New card codes are required
- **WHEN** ready decks contain valid cards absent from the 40M code list
- **THEN** training is blocked until the code list and semantic assets are versioned; a 40M continuation additionally requires append-only checkpoint migration and compatibility checks

### Requirement: Universal multi-deck learning
Unlike the existing elfnote-specialist stage, the multi-deck trainer SHALL allow
the shared policy to act and learn on both seats for all ready decks. It SHALL NOT
expose the private deck identifier as a policy input; matchup inference SHALL use
only the normal observation schema and visible action history.

#### Scenario: Non-elfnote matchup
- **WHEN** both sampled decks are non-elfnote ready decks
- **THEN** valid transitions from both players can contribute to training under the same policy

#### Scenario: Hidden deck identity
- **WHEN** two games have identical visible observations but different unrevealed deck identities
- **THEN** the policy input tensors are identical

### Requirement: Curriculum and run metadata
Every run SHALL freeze and record its initialization mode and seed, optional
source checkpoint hash, code revision,
corpus/cluster revision, sampler configuration and state, schema and semantic
hashes, code-list hash, per-deck and per-cluster sample counts, invalid-game
counts, optimizer status, seeds, compute manifest, and all checkpoint hashes.

#### Scenario: Compute changes during later continuation
- **WHEN** a run is moved to different hardware or environment counts
- **THEN** the new run records the changed compute/batch configuration and links to its exact parent checkpoint and curriculum revision

#### Scenario: Corpus drift
- **WHEN** a resume request resolves to deck, cluster, code-list, or semantic hashes different from those in the checkpoint metadata
- **THEN** the trainer fails before rollout collection unless an explicit versioned migration is supplied

### Requirement: Staged safety gates
The system SHALL require, in order, corpus validation, deterministic sampler
audit, initialization/finite-forward validation (and checkpoint migration only
when continuing), a pinned engine-protocol
and model-input contract audit, short save/reload smoke training, and a
fixed-budget pilot before a long multi-deck continuation. Failed or invalid games
SHALL NOT be converted into wins or silently retained as normal training samples.

#### Scenario: Protocol branch is not covered
- **WHEN** the pinned ygopro-core can emit a notification or interactive message, parameter branch, payload form, or response form that lacks a passing adapter fixture and, where applicable, response-oracle check
- **THEN** smoke and pilot training are blocked before rollout collection

#### Scenario: Protocol value exceeds a model field domain
- **WHEN** a parsed engine value cannot be represented by the declared observation or action-feature schema
- **THEN** the environment fails with the raw message, payload, field, and bound instead of indexing an embedding or emitting a training transition

#### Scenario: Smoke training is non-finite
- **WHEN** any smoke run reports NaN/Inf parameters, losses, logits, values, or optimizer state
- **THEN** the long run is blocked and the failure artifacts are retained

#### Scenario: From-scratch pilot promotion
- **WHEN** a fresh pilot has protocol/domain violations, invalid games, non-finite learning or inference, failed save/reload, missing cluster exposure, or an incomplete held-out evaluation
- **THEN** the candidate is not promoted and the failure evidence is retained; the mature 40M elfnote baseline is reported for context rather than used as a regression gate

### Requirement: Pinned engine-protocol completeness
The environment SHALL derive a machine-readable protocol contract from the exact
ygopro-core revision used to build the native extension. The contract SHALL list
every core `MSG_*` value, classify notification versus interactive messages,
record payload forms for every core-emitted notification, record payload and
response forms for every interactive parameter branch, and map each supported
response to the environment action representation. Every core-emitted message
SHALL have a real production parser branch; a diagnostic name alone does not
count as support. An adapter change SHALL NOT be accepted solely because sampled
duels happened not to reach an unsupported branch.

#### Scenario: Core message inventory changes
- **WHEN** the pinned core source adds, removes, or renumbers a `MSG_*` definition relative to the frozen contract
- **THEN** protocol validation fails and identifies the exact inventory drift before native build or training

#### Scenario: Legal response enumeration
- **WHEN** an interactive core message fixture is decoded
- **THEN** every action exposed by the environment is accepted by the matching core response validator and every required normal response form has at least one representable action

#### Scenario: Boundary variants
- **WHEN** selection counts, optional minima, composite masks, 32-bit announced values, mandatory cards, or multi-card allocations take legal boundary values
- **THEN** the adapter decodes, represents, and encodes them without truncation, unsupported fallbacks, or out-of-range categorical indices

#### Scenario: Concatenated notification frames
- **WHEN** the core emits multiple notification messages in one process buffer
- **THEN** the adapter consumes exactly the current message payload, preserves every following message boundary, requires no policy response, and fails with message-local truncation diagnostics for an incomplete payload

### Requirement: Multi-level evaluation
Promotion evaluation SHALL report protected elfnote results, per-deck results,
macro averages across clusters, micro averages across games, worst-cluster
performance, held-out decks, both seating orders, invalid-game rates, and Wilson
confidence intervals with checkpoint and manifest hashes.

#### Scenario: Cluster contains many decks
- **WHEN** aggregate evaluation is produced
- **THEN** the cluster contributes once to the macro-cluster score regardless of member count while all games still contribute to the micro score

#### Scenario: Held-out generalization
- **WHEN** corpus construction reserves deterministic held-out decks or families
- **THEN** they remain absent from training sampling and are reported separately during evaluation
