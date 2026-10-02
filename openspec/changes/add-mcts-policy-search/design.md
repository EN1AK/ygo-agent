## Context

See `proposal.md` for motivation. The repository already has three relevant pieces: a replay snapshot contract in `ygoai.rl.counterfactual`, a script-local two-player PUCT diagnostic in `scripts/run_counterfactual_search.py`, and a generic C++ AlphaZero tree that assumes cheap cloneable environment states. The current YGOPro binding links the external `ygopro-core 0.0.4` package (upstream f969296 plus empty-deck and disable-check guards) and exposes no public clone operation, so replaying the seed and full action prefix is the only working restorer. Earlier pinned fixtures passed exact observation/action restore gates; that does not certify new runtime fixtures or make replay fast enough for online self-play.

The available MirrorForce archive documents an arena-copy design and ABI but does not contain its modified core sources. It is design evidence, not reusable implementation. Hidden information is a second independent constraint: exact rollback alone creates an oracle if branch outcomes are conditioned on the real opponent hand.

## Goals / Non-Goals

**Goals:**

- Establish a bounded candidate-action rollout on the replay backend, then a deterministic native snapshot primitive for useful throughput; retain PUCT as a matched-budget comparator.
- Preserve all current structured observation, legal-action, recurrent-state, reward-scale, and replay contracts.
- Support an oracle diagnostic mode immediately and a belief-aware mode before fair-play promotion.
- Make search results suitable for evaluation and later supervised distillation with complete provenance.
- Keep the active 100M PPO continuation free of search, belief labels, and auxiliary losses. The earlier 40M model stays a frozen reference.

**Non-Goals:**

- Replace PPO with AlphaZero training in the first implementation.
- Claim that the existing replay PUCT or exact-state search is deployable MCTS strength.
- Serialize Lua coroutines or arbitrary C++ object graphs field by field.
- Introduce MCTS into the currently running PPO job.

## Decisions

### 1. Patch and pin a snapshot-capable `ygopro-core` package

The native implementation will live in a pinned fork/patch set of `ygopro-core`, selected by the repository's xmake package recipe and recorded in the runtime release manifest. Each duel receives a fixed-address arena; engine and Lua allocations belonging to that duel are routed into it. A snapshot copies the committed arena extent plus small out-of-arena API metadata, and rollback copies it back to the same address.

This follows the documented arena design because internal engine objects and live Lua coroutine pointers remain valid when restored at identical addresses. Field-wise serialization was rejected because it is version-fragile and cannot safely reconstruct arbitrary Lua execution state. Process `fork` was rejected because it prevents safe CUDA/threaded leaf batching and has unacceptable process overhead. Seed/action replay remains the reference backend and fallback.

The public native ABI will be versioned and minimal: create snapshot, rollback, free snapshot, inspect extent/identity, and report named failures. Snapshot buffers are allocated outside the duel arena so rollback cannot overwrite them. Duel generation and snapshot generation counters prevent cross-duel and use-after-free handles.

### 2. Validate native snapshots before using them in search

Validation proceeds from smallest to widest scope:

1. C-level advance/snapshot/advance/rollback/replay equality, including RNG and pending Lua effects.
2. Environment equality for raw engine frames, legal responses, structured observations, public events, history, reward, and terminal state.
3. Native-versus-replay differential fixtures across prompt types, chains, summons, shuffles, and long combos.
4. Repeated snapshot stack creation/free and forced capacity failures under leak and sanitizer checks.

No search integration is enabled when this gate fails. This isolates allocator/rollback correctness from tree-policy correctness.

### 3. Build candidate rollout before extracting PUCT

The first reusable search path evaluates selected root legal actions with reproducible common rollout seeds, bounded depth, terminal returns or calibrated leaf values, and per-action mean/uncertainty. Its backend protocol provides root capture, restore/branch, observation/legal menu, terminal result, and cleanup. The replay backend implements the protocol first as a correctness oracle; the native backend then implements the same interface. Staged multi-select and chain-response menus remain prompt-local nodes, not imaginary joint actions.

The existing PUCT script is preserved and compared at equal inference cost after the candidate-rollout gate. Extract its `Edge`, `Node`, selection, expansion, backup, and audit logic only when that comparison or a later tree-search experiment warrants it. The generic C++ `mcts/alphazero` tree is not used initially because its interface assumes dense global action spaces and copyable batched states.

### 4. Batch model evaluation, not engine mutation

Engine traversal and rollback stay on the duel-owning CPU thread. Rollout policy and leaf-value queries are queued and evaluated in GPU batches by one checkpoint. Each player view has its own recurrent state trajectory; values are converted to the root perspective exactly once when forming action returns. Virtual loss applies only if parallel PUCT leaf selection is later introduced.

This avoids concurrent mutation of one duel arena and provides most of the useful GPU speedup. A deterministic single-leaf mode remains available for regression tests.

### 5. Treat information-set search as a separate layer over snapshots

Exact-state snapshots are useful for engine correctness, tactical probes, and debugging but are labelled oracle-only. Fair-play search samples hidden-state particles from information available to the acting player, restores the public root, applies the particle before any model query, searches under a fixed per-particle budget, and aggregates root edge statistics.

Particle creation must preserve public card multiset constraints, known reveals, prior public movements, deck legality, and the acting player's own hidden cards. An initial conservative sampler may draw only from exact remaining-card multisets in self-play fixtures; it cannot be promoted against external opponents until its information-set contract is validated. Reusing the true hidden state as the sole particle is never a fair-play fallback.

### 6. Integrate a regularized candidate policy through an explicit wrapper

Evaluation and target generation call a `SearchPolicy` wrapper around the existing model policy. The wrapper decides whether a position is eligible, runs bounded candidate rollouts, and makes a local policy update regularized toward the recorded raw policy. The update coefficient, action-return uncertainty, and selection rule are logged; it returns either a sampled/selected action from that updated policy or the original raw action with a named fallback reason. It does not call a critic output a Q-value. Default commands construct no wrapper.

Eligibility starts conservatively: at least two policy-visible legal actions, a configured uncertainty/strategic-prompt trigger, no unsupported external bot state, and sufficient remaining budget. Every assisted decision records both raw and searched actions. Search is not inserted into PPO collection in this change; instead it emits versioned target rows for a later, separately measured distillation stage. Search targets are updated-policy distributions for candidate rollout and visit distributions for PUCT; mode and semantics cannot be conflated.

### 6a. Train belief separately before connecting it to search

Use a separate autoregressive head, parameter file, and optimizer. Inputs are legal-view tokens and public hidden-slot descriptors. Teacher labels are shifted behind a causal mask; remaining-card constraints can use a declared known-deck self-play prior, never an unannounced opponent's actual deck list. Card 0 is padding. Supervision must be collected through a separate privileged-label channel, not appended to actor observations.

The offline pilot splits by whole duel, records teacher-forced token NLL, accuracy, calibration, and constraint-valid joint samples. These are not sufficient evidence for held-out deck generalization, joint-particle calibration, native application correctness, or fair-play strength. The first implementation contains a strict data consumer and synthetic regression fixture; a real self-play producer and native particle application remain explicit unfinished gates.

### 7. Use staged promotion gates

Promotion has four levels: replay diagnostic, native oracle diagnostic, belief-aware assisted evaluation, then search-target generation. Each level has explicit correctness, latency, cleanup, and paired-strength evidence. A failure disables only the higher level and preserves lower diagnostic modes.

## Risks / Trade-offs

- **[Allocator interception misses an allocation path]** → Pin one core revision, assert allocation ownership, test Lua and STL-heavy fixtures under ASan, and fail a duel rather than fall back to the system heap.
- **[Rollback restores dangling external pointers]** → Keep callbacks and immutable databases outside snapshots, enumerate all mutable globals, and require native/replay differential equality before promotion.
- **[Exact-state search leaks opponent information]** → Label it oracle-only in schema and code, prohibit it in fair-play summaries, and require a validated particle sampler for promoted play.
- **[Critic bias is amplified by PUCT]** → Preserve leaf audits, continue through chain prompts, calibrate against terminal outcomes, and compare searched choices against paired rollouts before distillation.
- **[Search latency collapses match throughput]** → Use native rollback, batched leaf inference, eligibility filters, hard budgets, and raw-policy fallback; report latency distributions rather than only average SPS.
- **[Native module work destabilizes the active runtime]** → Build in an isolated package/release, retain the production module hash and backup, and deploy beside the training runtime until gates pass.

## Migration Plan

1. Land bounded candidate rollout with replay-backend fixtures, action-return audit, and regularized policy update; leave default behavior unchanged.
2. Add the pinned core patch and native snapshot bindings; build a separate native artifact and run round-trip/differential gates.
3. Add opt-in oracle evaluation and compare replay versus native candidate decisions and latency.
4. Add belief particles and leakage tests, then run paired fixed-seed assisted evaluation against the raw policy.
5. Compare candidate rollout with the preserved PUCT diagnostic at matched compute; extract PUCT for reusable play only if evidence warrants it.
6. Enable versioned search-target export only after the evaluation gate passes.
7. Synchronize the exact commit and native artifact provenance across local, GitHub, home, and H200. Rollback consists of selecting the prior production native module and leaving search flags unset.

## Open Questions

- The first native benchmark will determine the arena capacity and leaf batch size; these are tuning values and do not change the contracts above.
- Search eligibility thresholds will be selected from recorded latency/quality curves rather than fixed in the initial API.
