# Tasks

## 1. Freeze source state and corpus contract

- [x] 1.1 Record the 40M checkpoint, repository revision, schema, semantic metadata, code list, elfnote deck, and current compute hashes in a parent manifest; verify every referenced file exists and recomputed hashes match.
- [x] 1.2 Define the source-registry, canonical-deck, validation, family, feature, cluster, split, and corpus-manifest JSON schemas; verify valid fixtures pass and missing provenance/hash fields fail schema validation.
- [x] 1.3 Inventory the repository's current 37 top-level YDK files and older multi-deck artifacts without modifying them; verify the inventory records zone counts, hashes, aliases, and source status.

## 2. Acquire and normalize a large deck corpus

- [x] 2.1 Implement registry adapters for local directories, pinned git trees, and immutable hashed archives; verify two acquisitions from each fixture produce identical source manifests and file hashes.
- [x] 2.2 Select and pin auditable external YDK sources with provenance/license notes, targeting at least 256 canonical ready decks; verify no unpinned source enters the releasable registry.
- [x] 2.3 Implement strict YDK parsing and canonical main/extra/side zone normalization with multiplicities; verify malformed inputs are rejected and canonical hashes are order-independent.
- [x] 2.4 Implement exact deduplication while retaining all aliases and sources; verify duplicate fixtures yield one canonical deck and complete provenance.
- [x] 2.5 Implement near-duplicate similarity and family assignment with a pinned threshold; verify exact duplicates, one-card variants, and unrelated decks receive the expected family relationships.

## 3. Validate runtime eligibility

- [x] 3.1 Validate deck sizes and card database, script, token, code-list, and semantic coverage; verify each failure class produces a machine-readable `limited` or `excluded` reason.
- [x] 3.2 Add deterministic per-deck native load/reset and both-seat smoke duels; verify engine exceptions, timeouts, illegal actions, and incomplete games quarantine the deck instead of becoming training outcomes.
- [x] 3.3 Build corpus revision 1 with frozen `ready`, `limited`, and `excluded` tables; verify only ready canonical decks appear in the sampling input and all excluded source records remain traceable.

## 4. Build deterministic deck features and clusters

- [x] 4.1 Implement zone-separated sparse TF-IDF card-composition features with recorded main/extra weights and side-deck exclusion; verify repeated builds have identical shapes, values, ordering, and hashes.
- [x] 4.2 Implement deterministic average-linkage cosine clustering and candidate-cut scoring; verify repeated runs produce identical assignments and reject configured degenerate size distributions.
- [x] 4.3 Emit cluster representatives, member/family counts, silhouette and size metrics, rejected alternatives, and a manual representative-audit sheet; verify every ready deck and family maps to exactly one cluster.
- [x] 4.4 Freeze deterministic train and family-level held-out splits; verify no exact deck or near-duplicate family crosses between training and held-out sets.

## 5. Add hierarchical sampling to the environment

- [x] 5.1 Define and validate the weighted deck-sampling manifest with the default 25% per-seat elfnote reserve, equal non-anchor cluster weights, family de-biasing, and cluster floors; verify probabilities normalize exactly.
- [x] 5.2 Extend the native environment reset path to sample anchor/cluster/family/deck for both seats and emit stable IDs in `info`; verify fixed sampler seeds reproduce the same sequence.
- [x] 5.3 Persist sampler seed/counter/state in run checkpoints and restore it on resume; verify an interrupted same-topology run produces the same subsequent deck/seat selections as an uninterrupted fixture.
- [x] 5.4 Run at least 100,000 sampler-only draws and verify elfnote, cluster, family, deck, and seat frequencies satisfy declared statistical tolerances with every ready cluster covered.
- [x] 5.5 Benchmark manifest sampling against current uniform `random` reset and verify the added reset overhead stays within the recorded acceptance budget.

## 6. Migrate the 40M checkpoint when assets expand

- [x] 6.1 Compare the ready corpus against the frozen 40M card/code/script/semantic assets and emit the exact set of required new card codes; verify no migration is performed when the set is empty.
- [x] 6.2 Implement append-only code-list and semantic-asset generation for new codes; verify all old row indices and semantic rows remain byte-identical and all appended rows are deterministic.
- [x] 6.3 Implement the expanded-embedding checkpoint migration from the pinned 40M source; verify non-embedding parameters and old rows copy exactly, only appended rows are initialized, and source/output hashes plus a full inventory are recorded.
- [x] 6.4 Run finite-forward, deterministic-action, save/reload, and cross-manifest rejection checks on the migrated checkpoint; verify no rollout starts with unexplained schema, corpus, cluster, code-list, or semantic hash drift.

## 7. Integrate universal multi-deck continuation

- [x] 7.1 Add trainer arguments and checkpoint metadata for corpus/cluster/curriculum revisions, sampler state, elfnote reserve, and per-level identifiers; verify a dry run prints and saves the fully resolved configuration.
- [x] 7.2 Allow the shared Structured-lite/full policy to learn valid transitions from both seats on all ready decks without deck-ID input; verify non-elfnote versus non-elfnote rollouts update the policy and hidden deck identities do not alter equal visible observations.
- [x] 7.3 Add per-deck, family, cluster, seat, matchup, invalid-game, and realized game/decision sampling telemetry; verify aggregate counts reconcile to total completed games and learner decisions.
- [x] 7.4 Add linked-run compute manifests for topology changes; verify resuming on changed hardware records the parent checkpoint and new batch/actor mapping instead of claiming bit-identical continuation.

## 8. Gate and evaluate multi-deck training

- [x] 8.1 Run a 100k-step multi-deck smoke continuation from the compatible 40M-derived checkpoint; verify finite optimizer/logits/values, valid games, checkpoint save/reload, and sampler-state resume.
- [x] 8.2 Pin the exact ygopro-core build revision and generate a machine-readable inventory of every `MSG_*` value, its notification/interactive class, source writer, payload fields, response form, and parameter branches; verify inventory drift or an unclassified message fails validation.
- [x] 8.3 Implement complete native parsing and response encoding for every normal interactive branch in the pinned contract, eliminating sampled-path `unsupported` and `not implemented` behavior; verify deterministic fixtures parse exactly and all emitted actions pass the matching core-derived response oracle.
  - [x] 8.3.1 Add an actual `MSG_ROCK_PAPER_SCISSORS` request/response handler for both core stages, with pinned-core fixtures and response validation.
  - [x] 8.3.2 Enumerate both select and unselect card pools and cancel/finish responses for `MSG_SELECT_UNSELECT_CARD`; verify response indices against the core validator.
  - [x] 8.3.3 Remove silent action-list truncation: represent every candidate within declared model domains or fail/quarantine the state before returning an incomplete legal-action set.
  - [x] 8.3.4 Match the pinned core's 26-defined-bit race validator: never expose undefined high bits as independent race choices, and verify how such bits affect accepted response masks.
  - [x] 8.3.5 Replace eager weighted tribute/sum combination materialization with bounded incremental choice and core-valid completion checks; do not enumerate permutations or all subsets.
  - [x] 8.3.6 Exercise every parser branch using framed messages from the pinned core and validate encoded responses through the matching core instance, not only duplicate helper predicates.
  - [x] 8.3.7 Represent `MSG_SORT_CARD` without materializing its `N!` permutations; preserve the no-reorder response and test any learned ordering as a staged sequence whose final response passes core validation.
  - [x] 8.3.8 Expose `MSG_SELECT_COUNTER` allocations as bounded sequential choices and verify allocation bytes against the core validator.
  - [x] 8.3.9 Parse every direct-core notification writer with exact payload boundaries, including `MSG_SWAP_GRAVE_DECK`, `MSG_CONFIRM_EXTRATOP`, `MSG_CANCEL_TARGET`, `MSG_HAND_RES`, `MSG_TAG_SWAP`, `MSG_RELOAD_FIELD`, `MSG_AI_NAME`, `MSG_SHOW_HINT`, and `MSG_MATCH_KILL`; remove whole-buffer notification skips and verify concatenated and truncated frames through the production parser.
- [x] 8.4 Add strict declared-domain validation for every observation, action, history, and structured feature consumed by model embeddings or heads; verify legal 32-bit masks/values are losslessly represented and any out-of-domain field fails before inference with raw protocol diagnostics.
- [x] 8.5 Build and run the protocol boundary suite for optional zero-minimum selections, cancel/finish forms, multi-place choices, weighted tribute/sum modes, arbitrary counter allocations, sort permutations, composite position/race/attribute masks, announce-card filters, and 32-bit announced numbers; add a large-candidate weighted-sum liveness fixture that rejects factorial response enumeration; verify the coverage report has no uncovered normal interactive branch and the stress fixture terminates with unique core-valid card sets.
- [x] 8.5.1 Make the protocol gate distinguish diagnostic message names from production parser handlers and fail if any direct core writer lacks a handler or notification-boundary fixture; rebuild the frozen contract and coverage evidence after the notification parser passes.
- [x] 8.6 Run a 100k-step multi-deck smoke from random initialization after the protocol gate; verify finite optimizer/logits/values/targets/advantages, zero protocol/domain violations, valid games, checkpoint save/reload, and sampler-state resume. Evidence: `multideck-scratch-smoke100k-v4-20260926-h200`, 100352 steps, 190 valid games, zero invalid games, exact save/reload and deterministic inference.
- [x] 8.6.1 Correct forced-termination and deterministic-opponent semantics: max-step/timeout episodes end invalid with zero reward and explicit reasons/counters, the learner and evaluator segregate them, `greedy` uses semantic action priority, and the legacy index-zero policy is available only as `first`; verify native cap behavior and a both-seat greedy evaluation on H200. Evidence: 54 native/Python tests passed; a four-game `max_steps=1` probe returned zero reward, `invalid_game=1`, and reason 2 for every game; `evaluation-greedy-semantic-256-20260929` records 32 natural and 224 invalid terminals without manufactured winners.
- [ ] 8.6.2 Add auditable deterministic-cycle handling: encode Cancel in action history, fingerprint public state/legal menu/raw choice independently of rolling history, keep raw argmax unchanged by default, add an opt-in next-ranked-action guard with explicit intervention logs, and export replayable cycle decision points for the existing snapshot-search path; verify synthetic A-B-A detection, non-cycle behavior, native Cancel-history encoding, and fixed-seed raw/guard H200 replays.
- [ ] 8.7 Run a fixed 5M-step pilot from random initialization with the frozen static 25% elfnote/equal-cluster curriculum and 128 action slots; preserve launch command, logs, checkpoints, hashes, compute data, realized sampling report, and the exact passing protocol-contract hash.
- [ ] 8.8 Evaluate the pilot on protected elfnote mirror/greedy/historical suites, every training cluster, family-level held-outs, both seats, and invalid-game rates; verify Wilson intervals and macro-cluster, micro-game, and worst-cluster metrics are reported.
- [ ] 8.9 Report the frozen 40M checkpoint as historical context, and block promotion on protocol/domain violations, invalid games, non-finite behavior, failed checkpoint reload/resume, missing cluster coverage, or incomplete evaluation; verify the signed decision report names accepted and rejected gates without applying a 40M win-rate regression gate to the fresh 5M policy.
- [ ] 8.10 Select and record the long-run step budget and actor/environment topology from measured pilot SPS, memory, utilization, and learning curves; verify the decision remains reproducible if later compute changes.
