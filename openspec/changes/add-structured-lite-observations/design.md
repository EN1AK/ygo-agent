## Context

See `proposal.md` for motivation and `specs/structured-lite-observations/spec.md` for the behavior contract. The current environment emits `cards_`, `global_`, `actions_`, `h_actions_`, and `mask_`. Its action tensor binds at most one scene-card index and card ID to a compact action description; recent history contains chosen actions but not a sufficiently explicit record of chain sources and outcomes. The JAX encoder embeds card identity and runtime fields, pools the card scene, action set, global state, and action history, then feeds a per-player 512-wide LSTM into the existing FiLM policy head and scalar critic.

The reference design describes a substantially different structured branch. It is useful as an architectural source, but it is not checkpoint-compatible with this repository and includes approximate Lua-effect binding and partially unfilled Target/Cost roles. GPU resources are constrained, so the first implementation must isolate the highest-value relationship features and measure their cost.

## Goals / Non-Goals

**Goals:**

- Make the active prompt, chain source, candidate action source, target, and selection constraints explicitly distinguishable.
- Preserve legal-information boundaries and separate player memories.
- Reuse the current FiLM actor, scalar critic, recurrent width, and core training objective for a controlled comparison.
- Establish deterministic semantic assets, schema metadata, migration reports, and diagnostic coverage.
- Make the Veiler/Ogre snapshot and future combo demonstrations first-class evaluation inputs.

**Non-Goals:**

- Full Lua execution semantics, natural-language embeddings, a world model, MCTS, or an explicit belief head.
- Treating local destruction, negation, search, or summon events as shaped value rewards.
- Claiming exact Target/Cost/effect-slot bindings where the engine protocol does not establish them.
- Preserving binary compatibility by pretending new tensors are legacy inputs.
- Replacing the current policy/value heads before Structured-lite features are independently evaluated.

## Decisions

### 1. Add a parallel schema instead of mutating the legacy schema

The environment will select a named observation version. Legacy evaluation keeps the existing tensors and shapes; Structured-lite adds new tensors and uses separate model metadata. This permits fixed-checkpoint baselines and rollback.

Alternative: append fields to existing tensors. Rejected because field offsets become ambiguous, silent checkpoint misuse becomes likely, and coverage cannot be audited independently.

### 2. Implement a minimal relationship graph as typed references

Structured-lite will use dense bounded tensors rather than a general graph library:

```text
visible card slots <-------------------------------+
       ^                                           |
       | typed references                          |
selection context --> active chain source          |
       |                                           |
legal action ------> action source/candidate/target+
       |
       +-----------> cost/material/tribute/selected sets
```

References are 1-based scene-card indices with zero meaning absent. Every binding carries an exact/fallback/unknown confidence class. Single-card roles remain separate; set-valued roles have independent masks and explicit overflow diagnostics.

Alternative: rely only on cross-attention to infer relationships. Rejected because attention cannot recover relations that were never represented and this is exactly the failure mode under investigation.

### 3. Separate prompt state from candidate-action state

A selection tensor describes properties shared by the current prompt: message type, chain depth, response status, selection role, min/max/must counts, selected count, and finish/cancel/continue flags. Each action record describes candidate-specific act, phase, effect descriptor, position/place/number/attribute, source semantic row, source confidence, and action ordinal for diagnostics.

The model must not depend on ordinal for semantic identity. An action-permutation diagnostic will measure whether reordering otherwise equivalent candidates changes scores beyond numerical tolerance after undoing the permutation.

### 4. Start semantics with exact CDB fields and audited effect tags

The offline builder will emit:

- card category and type bits;
- race and attribute bits;
- ATK/DEF/level/rank/link/scale/link-marker values plus applicability masks;
- a small versioned vocabulary derived from card scripts, initially including activation class and operations such as destroy, negate, send, banish, draw, search/add, summon, special summon, and target;
- per-tag confidence, source asset hashes, missing-script counts, and parser coverage.

The first version will not use high-dimensional hashed Lua features. They can be added later behind a new semantic-table version after effect-level alignment is measured.

Alternative: copy the reference branch's 16x64 hashed Lua table immediately. Deferred because hash collisions and effect-slot mismatch would make the first causal regression harder to interpret.

### 5. Record engine outcomes as public events, not rewards

A bounded ring records public activations, targeting, negation, destruction, zone movement, draw/search, summon, chain-solving, and chain-end messages. Events are observer-relative, link to currently resolvable visible card slots where safe, and retain a stable event type even when a historical card reference can no longer be bound.

This improves representation and debugging without asserting that a locally successful interruption is globally optimal. Terminal return remains the critic target.

### 6. Reuse heads and add scene-conditioned action encoding

Cards, prompt state, and public events form a visible scene. Each action combines its structured fields, static source semantics, and typed card-reference summaries, then performs a small action-to-scene attention block followed by one action-set comparison block. The existing per-player LSTM receives the scene state plus a masked mean of legal-action representations. The existing FiLM actor and scalar critic remain unchanged at their public interface.

This isolates feature quality from head redesign. An explicit Q head or afterstate critic becomes a later change only if better continuation policies still leave snapshot Q too noisy.

### 7. Treat migration as a warm start, not compatibility

The migration tool will copy only named parameter subtrees whose semantics and shapes are unchanged, such as compatible recurrent and head parameters. New structured encoders are initialized and listed in a machine-readable migration report. The first experiment will include both scratch and warm-start candidates if budget permits; neither may overwrite the source checkpoint.

### 8. Stage evaluation before long training

The implementation proceeds through gates:

1. Observation writer and semantic-table coverage on recorded self-play traces.
2. Visibility/leakage and deterministic replay checks.
3. Tensor-shape and numerical forward checks.
4. Short smoke training and throughput/memory comparison.
5. Fixed-budget A/B/C training: legacy, relationship-only, relationship plus events/CDB semantics.
6. Combo accuracy, Veiler/Ogre diagnostics, value calibration, and fixed-seed head-to-head evaluation.

The comparison uses the same deck pool, seeds, training steps, and evaluation harness. Reports retain configuration, code revision, semantic hashes, checkpoint hashes, and invalid-game counts.

## Risks / Trade-offs

- [Engine protocol does not identify every semantic role] -> Use confidence classes, unknown roles, coverage metrics, and never promote inferred bindings to exact without evidence.
- [Historical scene indices become stale after card movement] -> Preserve event type/card identity when publicly known, validate references at write time, and clear unsafe slot references.
- [Semantic extraction creates false confidence] -> Begin with a small audited vocabulary, publish per-tag precision samples, and version every builder change.
- [Observation growth reduces throughput] -> Keep bounded capacities, use compact integer tensors, benchmark each feature group, and retain a relationship-only ablation.
- [Value learning backpropagates through action features] -> Keep this behavior initially for comparability, then measure an optional stop-gradient ablation if critic instability appears.
- [Warm start biases results or hides broken initialization] -> Report scratch and migrated runs separately and compare early optimization curves.
- [Action ordinal leaks engine ordering] -> Retain ordinal only for diagnostics or remove it from learned inputs; enforce permutation testing.
- [Local regression improves while overall policy degrades] -> Require combo, calibration, target-deck head-to-head, and invalid-game gates before promotion.

## Migration Plan

1. Add the semantic builder and observation writer under a new schema flag while leaving legacy defaults unchanged.
2. Generate and validate versioned semantic assets; do not commit large generated tables.
3. Add the Structured-lite encoder and checkpoint metadata validation.
4. Produce a warm-start checkpoint in a new output directory with a parameter-copy report and hashes.
5. Run shape, leakage, replay, throughput, and smoke-training gates on the GPU server.
6. Run the fixed-budget A/B/C experiment and promotion evaluation.
7. If gates fail, continue serving and evaluating the legacy schema; remove no legacy assets or checkpoints.
8. Change training defaults only after the promotion report explicitly approves Structured-lite.

Rollback consists of selecting the legacy schema and its matching checkpoint. No checkpoint or dataset is rewritten in place.

## Open Questions

- Choose concrete initial capacities for public events and set-valued references after measuring overflow rates on existing traces; proposed starting values are 32 events and 8 references per role.
- Decide the maximum acceptable training-throughput regression after the first GPU microbenchmark; report both absolute and relative throughput before setting the promotion threshold.
