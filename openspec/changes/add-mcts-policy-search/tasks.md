## 1. Freeze the replay reference and build candidate rollout

- [ ] 1.1 Capture deterministic replay-backend fixtures for the existing Veiler/Ogre case, policy-cycle case, chain-response case, and a long combo state; verify each records root observation digest, legal menu, raw policy/value, selected action, and the original PUCT audit without changing its meaning.
- [x] 1.2 Define backend-neutral result and audit schemas that distinguish candidate returns/updated policy from PUCT visits/edge Q; verify old diagnostic JSON can be upgraded without changing action identity or numeric meaning.
- [ ] 1.3 Record the exact deployed `ygopro-core 0.0.4` source revision (including the 2026-10-02 disable-check guards), package recipe, Lua linkage, exported symbols, and mutable global-state audit; verify a clean portable build reproduces the current native module behavior before snapshot patches are applied.
- [ ] 1.4 Implement replay-backed bounded candidate rollout with common particle and rollout seeds, separate player recurrent states, terminal/leaf returns, and prompt-local legal menus; verify paired fixed-seed fixtures cover alternating players, chains, staged multi-select, depth limits, and terminal outcomes.
- [x] 1.5 Produce per-action means/uncertainty and an explicitly regularized update from the raw policy; verify menu alignment, finite outputs, deterministic ties, and raw-policy fallback when comparison is incomplete.

## 2. Shared backend, budgets, and PUCT comparator

- [ ] 2.1 Define replay/native backend and evaluator protocols with explicit capture, branch, restore, legal-menu, terminal, recurrent-state, and cleanup semantics; verify a fake backend exercises every lifecycle transition and rejects use after close.
- [ ] 2.2 Add candidate-count, particle, rollout-per-action, depth, wall-time, and snapshot-memory budgets (plus PUCT node budget) with named raw-policy fallbacks; verify forced exhaustion releases resources and never presents partial comparisons as complete.
- [ ] 2.3 Preserve `run_counterfactual_search.py` CLI and output compatibility; use its existing PUCT diagnostic as the matched-compute comparator. Extract reusable PUCT tree bookkeeping only if the comparison gate warrants online tree search; verify frozen fixtures preserve visits, selected actions, value perspective, and chain behavior if extraction occurs.

## 3. Native duel snapshot backend

- [ ] 3.1 Add a repository-pinned `ygopro-core` snapshot patch/fork and xmake package override without changing the default production artifact; verify the unpatched and patched variants can be built side by side with recorded source and binary hashes.
- [ ] 3.2 Route per-duel core and Lua allocations through a fixed-address bounded arena while leaving immutable callbacks/databases outside it; verify allocation ownership assertions cover C++, STL, Lua, duel creation, and duel destruction paths under ASan.
- [ ] 3.3 Implement versioned snapshot, rollback, free, arena-extent, duel-generation, and snapshot-generation APIs with foreign/stale/released-handle rejection; verify named failures do not mutate the duel and snapshot buffers are never allocated inside the restored arena.
- [ ] 3.4 Expose snapshot lifecycle calls through the YGO environment binding and implement the native search backend; verify Python context-manager cleanup frees nested snapshots on success, exception, timeout, and terminal branches.
- [ ] 3.5 Add C-level round-trip fixtures covering RNG, shuffles, live Lua coroutine/effect resolution, pending prompts, response buffers, chains, staged multi-select, and terminal transitions; verify post-rollback engine message bytes and subsequent random outcomes are identical.
- [ ] 3.6 Run native-versus-replay differential fixtures across all recorded prompt classes and long combo prefixes; verify observations, public events, history, acting player, legal responses, reward, terminal state, and search choices match exactly.

## 4. Model evaluation and hidden information

- [ ] 4.1 Add a rollout/leaf evaluator that preserves separate recurrent states for both player views and converts checkpoint values exactly once to the root perspective; verify alternating-player and terminal-win/loss fixtures use the checkpoint's `greedy_reward=False` scale.
- [ ] 4.2 Add optional batched GPU leaf evaluation while engine rollback remains duel-thread-owned; verify deterministic single-leaf and batched modes agree within the declared numeric tolerance and report batch latency/throughput.
- [x] 4.3 Define a versioned information-set and belief-particle schema that tracks public history, known reveals, remaining-card multisets, particle seed, and hidden assignments; verify particles preserve all public constraints and never alter the acting player's known private cards.
- [ ] 4.3a Implement a separate autoregressive belief head and self-play training-data path, using legal observations as inputs and hidden identities only as supervised labels; verify causal label masking, remaining-count masks, finite updates, held-out negative log likelihood/calibration and constraint-valid sampling. Record known-deck self-play assumptions separately from unknown-deck external play.
- [ ] 4.4 Implement conservative self-play belief-particle sampling and a structural pre-query leakage guard; verify deliberate reads of the real hidden state before particle application fail and exact-state runs are labelled `oracle_exact_state`.
- [ ] 4.4a Connect trained belief-head samples to particle application and the candidate rollout wrapper; verify hidden-state perturbations cannot affect pre-particle actor/search queries, sample hashes are reproducible, and fair-play promotion requires held-out particle and decision-quality evidence.
- [x] 4.5 Aggregate candidate-action returns and uncertainty across reproducible particles with declared per-particle budgets; verify reordering particle execution does not change deterministic means, updated policy, or selected action.

## 5. Opt-in play and target generation

- [x] 5.1 Add an opt-in search-policy wrapper with eligibility filters, raw-action capture, searched-action selection, strict/fallback modes, and reason codes; verify existing evaluation and PPO commands are unchanged when search flags are absent.
- [ ] 5.2 Integrate the wrapper into model-versus-model evaluation and replay logging; verify every assisted decision links the playable replay, engine log, structured decision log, snapshot digest, checkpoint hashes, search audit, and terminal result.
- [ ] 5.3 Add a versioned search-target exporter containing mode-labelled updated-policy targets (or PUCT visits) and searched root values separately from original logits, critic values, MC returns, and per-action Q estimates; verify legal-action alignment and provenance survive serialization/reload.
- [ ] 5.4 Add an offline distillation data loader and loss plumbing behind an explicit training flag without enabling it in PPO collection; verify a bounded fixture consumes search targets while the default PPO loss and checkpoint format remain unchanged.

## 6. Promotion and deployment gates

- [ ] 6.1 Benchmark replay and native oracle candidate rollout on identical frozen decisions; verify action parity, byte-identical restore gates, snapshot memory bounds, cleanup, and latency distributions are reported separately.
- [ ] 6.2 Run paired fixed-seed raw-policy versus belief-candidate-rollout matches on held-out decks and both seats; verify the report includes confidence intervals, invalid-game rates, decision latency, fallback rates, and excludes oracle runs from fair-play strength claims.
- [ ] 6.2a Compare candidate rollout and the preserved PUCT diagnostic with the same model, root fixtures, and inference budget; report decision quality and wall-clock cost separately, without treating oracle PUCT results as fair-play strength.
- [ ] 6.3 Calibrate searched leaf values against eventual terminal outcomes by prompt type, player, phase, and chain depth; verify promotion is blocked when systematic viewpoint/prompt bias or non-finite values exceed declared thresholds.
- [ ] 6.4 Build the portable Linux native artifact, preserve/restore the production module during validation, and synchronize the accepted commit plus artifact provenance across local, GitHub, home, and H200; verify all four source revisions and deployed binary hashes match their manifests.
- [ ] 6.5 Update training/evaluation documentation with supported search modes, budgets, limitations, replay deliverables, and rollback procedure; verify the final change validation passes and the promotion report identifies the highest gate actually achieved.
