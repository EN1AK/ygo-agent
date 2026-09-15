## 1. Baseline and schema

- [ ] 1.1 Capture the legacy observation shapes, field definitions, checkpoint metadata, GPU throughput, peak memory, and fixed evaluation inputs in a baseline report; verify all referenced hashes and commands are recorded.
- [ ] 1.2 Define the Structured-lite schema version, tensor fields, role/event enums, confidence classes, capacities, dtypes, normalization ranges, and overflow indicators; verify a generated schema manifest is internally consistent and documents every field.
- [ ] 1.3 Add environment/model/checkpoint compatibility metadata and fail-fast validation; verify matching legacy and Structured-lite configurations load while cross-schema direct loads fail before execution.

## 2. Static semantic assets

- [ ] 2.1 Implement deterministic CDB feature extraction with applicability masks for type, race, attribute, ATK, DEF, level, rank, Link, scales, and link markers; verify two identical builds have the same hash.
- [ ] 2.2 Implement the bounded audited script-effect tag vocabulary with explicit unknown and confidence outputs; verify hand-reviewed positive and negative examples for every initial tag.
- [ ] 2.3 Emit semantic asset metadata containing code-list, database, script, configuration, table hashes, and coverage counts; verify unsupported cards map to the documented unknown row without changing table dimensions silently.

## 3. Structured environment observations

- [ ] 3.1 Add selection-context output for prompt type, chain depth/source, response state, multi-select progress, and finish/cancel/continue constraints; verify representative engine prompts produce the expected fields.
- [ ] 3.2 Add per-action structured records and exact/fallback/unknown bindings for action source, active effect source, candidate, target, cost, material, tribute, and selected roles; verify each supported message family with recorded verbose engine evidence.
- [ ] 3.3 Add bounded public events for activation, targeting, negation, destruction, movement, draw/search, summon, chain solving, and chain end; verify ordering and truncation on a trace longer than the configured history.
- [ ] 3.4 Add overflow diagnostics for actions and reference sets; verify overflow cannot be silently accepted as a complete training sample.
- [ ] 3.5 Audit acting-player visibility across hidden hand, deck, extra deck, set cards, revealed cards, and historical references; verify no policy tensor changes when hidden identities are permuted while all public information is held constant.

## 4. Structured-lite model

- [ ] 4.1 Implement encoders for exact semantics, effect tags, selection context, public events, and typed references; verify shape tracing and finite numerical forward output for padding-only and maximum-capacity fixtures.
- [ ] 4.2 Implement action-to-scene attention and bounded action-set comparison while retaining the existing FiLM policy head, scalar critic, and per-player LSTM interface; verify one masked logit per legal action and one `V(s)` output per state.
- [ ] 4.3 Add an action-permutation diagnostic and remove learned dependence on action ordinal if the unpermuted logits do not match within tolerance after reversing the permutation; verify the diagnostic report passes.
- [ ] 4.4 Add a named relationship-only ablation that excludes public events and static semantics; verify it shares all unrelated configuration with the full Structured-lite model.

## 5. Checkpoint migration

- [ ] 5.1 Implement a dry-run migration inventory that classifies every legacy parameter as copied, newly initialized, shape-incompatible, or intentionally excluded; verify counts sum to the full source and destination parameter trees.
- [ ] 5.2 Produce warm-start checkpoints only in new output directories and write source/output hashes plus parameter-copy reports; verify the source checkpoint hash is unchanged.
- [ ] 5.3 Compare scratch and warm-start finite forward passes and initial optimization metrics; verify neither run contains NaN/Inf values and retain both reports for experiment interpretation.

## 6. Server validation and experiment

- [ ] 6.1 Build and deploy the versioned environment and model to the GPU server without replacing the legacy runtime artifact; verify both schemas complete deterministic smoke duels.
- [ ] 6.2 Measure observation size, environment throughput, training steps per second, and peak GPU memory for legacy, relationship-only, and full Structured-lite configurations; verify results use identical hardware and batch settings.
- [ ] 6.3 Select public-event and group-reference capacities from measured overflow rates, recording the chosen values and rejected alternatives; verify the final schema manifest and model configuration agree.
- [ ] 6.4 Run short smoke training for scratch and warm-start Structured-lite candidates; verify checkpoint save/reload, finite losses, valid games, and deterministic evaluation complete.
- [ ] 6.5 Run the fixed-budget A/B/C experiment using identical deck pool, seeds, steps, and evaluation schedule for legacy, relationship-only, and full Structured-lite; verify all configurations, logs, checkpoints, hashes, and invalid-game counts are preserved.

## 7. Evaluation and promotion

- [ ] 7.1 Add the `battle-62-seat1:14` Veiler/Ogre snapshot diagnostic; verify feature dumps separately identify the continuous-spell effect source, Veiler monster target, Ogre interaction, and resulting summon/destruction events.
- [ ] 7.2 Add combo-route evaluation through the human-demonstration interface when compatible YRP recordings are available; verify combo-only trajectories score policy accuracy without terminal value labels.
- [ ] 7.3 Report critic calibration and paired snapshot rollout uncertainty without relabeling `V(s)` as Q; verify confidence intervals and sample counts accompany comparisons.
- [ ] 7.4 Run fixed-seed target-deck head-to-head evaluation in both seats against the legacy checkpoint; verify replay/result bundles and checkpoint hashes satisfy the repository evaluation conventions.
- [ ] 7.5 Write a promotion report comparing regression behavior, combo accuracy, target-deck strength, invalid games, throughput, memory, and value calibration; verify Structured-lite becomes a default only if the report explicitly approves it.
