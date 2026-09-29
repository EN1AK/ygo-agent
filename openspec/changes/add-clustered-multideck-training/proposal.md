# Proposal

## Why

The 40M Structured-lite checkpoint was trained on a narrow deck distribution.
A new policy trained from random initialization over a large, deduplicated,
strategically diverse corpus can learn the new feature and protocol contract
without inheriting that distribution. Cluster sampling prevents common
archetypes or near-duplicate lists from dominating; an explicit elfnote reserve
keeps the anchor well represented.

## What Changes

- Build a versioned deck corpus from the repository and pinned external YDK
  sources, retaining source revision, license/provenance, normalized card lists,
  validation status, and hashes.
- Reject unusable decks before training: malformed or illegal-sized lists,
  missing card database rows/scripts/code-list entries, unsupported tokens, and
  decks that fail deterministic smoke duels are quarantined with reasons.
- Deduplicate exact lists and collapse near-duplicate variants before computing a
  deterministic card-composition representation and clustering the eligible
  corpus into strategically related groups.
- Add a hierarchical sampler that first reserves a configurable elevated share
  for elfnote, then samples clusters rather than raw decks, then samples a deck
  within the chosen cluster, with balanced seating and minimum coverage floors.
- Train the initial multi-deck policy from random initialization under the frozen
  Structured-lite/full feature, code-list, semantic, and protocol contracts.
  Keep the pinned 40M checkpoint and optional append-only migration as a
  separate compatibility path, not as the pilot's initialization.
- Record the corpus revision, cluster assignment, realized sampling counts,
  matchup results, invalid games, checkpoint hashes, and compute configuration so
  a later machine or larger GPU can resume the same curriculum.
- Derive a versioned engine-protocol contract from the pinned ygopro-core source,
  cover every emitted notification payload and every interactive message and
  response variant, and reject out-of-domain observation/action fields before
  they reach the policy.
- Add staged gates: corpus audit, sampler-distribution test, engine-protocol and
  model-input contract audit, short finite from-scratch smoke, fixed-budget
  from-scratch pilot, and only then a long run.
- Make deterministic policy loops observable and separately mitigable: encode
  cancellation in action history, fingerprint recurring public states/legal
  menus/actions, preserve raw-policy results, and expose opt-in cycle guards and
  replay-snapshot search artifacts without relabeling assisted play as raw play.

## Capabilities

### New Capabilities

- `deck-corpus-clustering`: Reproducible acquisition, normalization, validation,
  deduplication, feature construction, clustering, and versioning of a large YDK
  training corpus.
- `clustered-multideck-training`: Hierarchical cluster/deck/seat sampling,
  elfnote exposure, versioned initialization, telemetry, resume behavior, and
  promotion gates for a multi-deck policy.

### Modified Capabilities

None. The repository has no promoted main capability specs; this change replaces
neither the existing elfnote specialist plan nor the legacy runtime.

## Impact

- Training entrypoint and native environment deck selection need a versioned
  weighted-sampling manifest instead of the current uniform `deck=random` path.
- The native protocol adapter gains a generated message inventory, core-derived
  payload/response fixtures, response-oracle checks, and strict tensor-range
  validation; training is no longer used to discover protocol branches.
- New corpus and clustering tools will operate on `assets/deck`, the card
  database, scripts, code list, and semantic assets, producing immutable manifests
  rather than silently altering existing assets.
- New dependencies may include a pinned deterministic sparse clustering stack;
  if introduced, versions and generated assignments must be recorded.
- Training checkpoints gain deck-corpus and curriculum metadata. Existing 40M
  weights remain unchanged as a reference and optional migration source.
- Evaluation expands from elfnote mirrors to per-cluster holdouts and per-deck
  matchups. The 40M results are descriptive reference data, not a regression
  gate for a newly initialized 5M-step policy.
