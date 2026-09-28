# Spec Delta

## Purpose

Defines a reproducible large-deck corpus whose provenance, validation,
deduplication, representation, clustering, and immutable revision can be audited
and reused across training machines.

## ADDED Requirements

### Requirement: Pinned deck-source registry
The system SHALL ingest YDK files only through a source registry that records a
stable source identifier, fetch location, pinned revision or content hash,
license/provenance note, acquisition timestamp, and local file hash. Repository
decks and externally acquired decks SHALL use the same registry format.

#### Scenario: Rebuilding a corpus from its registry
- **WHEN** an operator rebuilds a corpus revision using the recorded registry
- **THEN** the system produces the same source-file hashes or fails with a source-drift error

#### Scenario: Unpinned external source
- **WHEN** an external deck source lacks a pinned revision or immutable content hash
- **THEN** the system excludes it from a releasable corpus revision and records the reason

### Requirement: Canonical deck normalization and deduplication
The system SHALL parse main, extra, and side deck zones separately, preserve card
multiplicity, produce a canonical normalized representation, merge exact
duplicates, and identify near-duplicate families without deleting their source
provenance.

#### Scenario: Same list from multiple sources
- **WHEN** two YDK files normalize to identical zone/card-count tuples
- **THEN** the corpus contains one canonical training deck with both sources recorded

#### Scenario: Near-duplicate variants
- **WHEN** lists differ only within the configured near-duplicate threshold
- **THEN** they receive a shared family identifier used to prevent variant-count sampling bias

### Requirement: Deck eligibility validation
The system SHALL validate deck sizes, card database coverage, script and token
availability, code-list coverage, semantic-row coverage, native environment load,
and deterministic smoke-duel completion. Each candidate SHALL be classified as
`ready`, `limited`, or `excluded` with machine-readable reasons, and only `ready`
decks SHALL enter training sampling.

#### Scenario: Missing runtime asset
- **WHEN** any card required by a deck lacks a database row, required script, token mapping, or code-list entry
- **THEN** the deck is not marked `ready` and the missing identifiers are reported

#### Scenario: Invalid smoke duel
- **WHEN** a deck produces an engine exception, illegal action, timeout, or incomplete reset in its validation games
- **THEN** the deck is quarantined from the training manifest pending review

### Requirement: Deterministic composition features
The system SHALL construct a deterministic sparse feature vector from normalized
main- and extra-deck card counts using corpus-level inverse-frequency weighting,
with zone separation and a recorded feature-schema version. Side-deck contents
SHALL be retained in provenance but SHALL NOT affect the initial clustering.

#### Scenario: Repeated feature build
- **WHEN** features are rebuilt from the same canonical corpus and feature configuration
- **THEN** the feature matrix shape, nonzero values, row ordering, and hash are identical

#### Scenario: Staple-heavy lists
- **WHEN** cards occur across most of the corpus
- **THEN** their clustering contribution is down-weighted relative to rarer archetype-defining cards

### Requirement: Reproducible strategic clustering
The system SHALL cluster canonical ready decks with a pinned algorithm,
hyperparameters, distance metric, seed, and deterministic row order. The emitted
manifest SHALL include cluster identifiers, medoid or representative decks,
member/family counts, quality metrics, and rejected alternatives. Exact and
near-duplicate variants SHALL NOT independently inflate cluster selection.

#### Scenario: Cluster rebuild
- **WHEN** clustering is repeated with the same corpus and configuration
- **THEN** every canonical deck receives the same cluster identifier and the manifest hash is unchanged

#### Scenario: Degenerate cluster solution
- **WHEN** a candidate solution violates configured minimum/maximum cluster sizes or produces a dominant catch-all cluster beyond its limit
- **THEN** the solution is rejected and cannot become the training manifest

### Requirement: Immutable corpus revision
The system SHALL assign every accepted corpus and clustering result a revision
identified by the hashes of source registry, canonical deck table, validation
report, feature schema/matrix, cluster configuration, and assignments. Existing
revisions SHALL NOT be modified in place.

#### Scenario: Adding newly acquired decks
- **WHEN** additional deck files are accepted after a revision is frozen
- **THEN** the system creates a new revision while preserving the prior manifest and hashes

