## 1. Freeze the baseline and protect defaults

- [ ] 1.1 After the active 40M run completes, freeze its validated checkpoint, source/native hashes, deck/seed/both-seat evaluation matrix, replay examples, SPS, timeout rate, combo completion, and interruption metrics in a baseline manifest; verify every referenced artifact exists and matches its recorded hash.
- [ ] 1.2 Add explicit `off`, `shadow_observation`, `qboost_observation`, and `vrpo_centralized` mode selection with incompatible-flag rejection; verify an unflagged command's parsed config, inference outputs, GAE targets, and legacy checkpoint bytes remain unchanged on fixed fixtures.
- [ ] 1.3 Add a versioned Q-mode checkpoint envelope and explicit old-PPO actor-weight import; verify matching Q checkpoints resume, mismatched mode/schema/reward context refuses before rollout, and old PPO evaluation still loads normally.

## 2. Establish legal-menu Q and a shadow critic

- [ ] 2.1 Record the staged legal-menu identity, validity mask, `num_options`, and chosen index alongside each Q-mode transition; verify multi-select, chain, forced-choice, and capacity-boundary fixtures preserve exact actor/Q slot alignment and fail closed on mismatch.
- [ ] 2.2 Implement a separate observation-only candidate-Q critic and optimizer that score only valid menu entries without using `RNNAgent.q_head` as Q; verify output shape, masked expectation, finite gradients, and unchanged actor logits on fixed observations.
- [ ] 2.3 Wire `shadow_observation` to train and log Q while GAE alone drives the actor; verify a short deterministic rollout produces critic loss/calibration metrics, no Q advantage enters PPO, and no hidden field appears in actor inputs.
- [ ] 2.4 Evaluate shadow Q on held-out deck/seed decisions by prompt type, chosen-action return calibration, and menu ranking; verify a machine-readable report contains finite/error counts, latency, memory, SPS, and declared pass/fail gates before enabling a Q-boosted actor pilot.

## 3. Implement and validate Q-boosting

- [x] 3.1 Implement masked policy expectation, Expected-SARSA residuals, backward Q-boosting trace, and chosen-action Q regression targets; verify hand-computed deterministic oracles for lambda 0/1, terminal zero bootstrap, padded rows, and a one-action policy.
- [ ] 3.2 Integrate Q traces with the existing two-seat recurrent rollout and collection boundaries; verify alternating-turn, delayed reward, true terminal, timeout, and nonterminal truncation fixtures against independently calculated seat-specific returns.
- [ ] 3.3 Keep the reference policy fixed for rollout/critic targets and recompute current actor expectations with fixed Q during full-VRPO PPO updates; verify sampled-action ratios, stop-gradient Q behavior, and critic target shape on a bounded minibatch.
- [ ] 3.4 Run `qboost_observation` against a separately restarted GAE control from the frozen actor checkpoint with identical decks, seeds, steps, wall-clock accounting, UPGO setting, and hyperparameters; verify a report labels observation-only Q as an ablation and records paired outcomes, uncertainty, SPS, calibration, and invalid/timeout counts.

## 4. Add a learner-only centralized critic channel

- [ ] 4.1 Add a versioned, opt-in full-state record separate from the public/actor observation schema, with decision identity and lifecycle checks; verify core-zone/chain identity fixtures and rejection of stale or misaligned records.
- [ ] 4.2 Prove the actor consumes only legal observations while the learner receives privileged Q inputs; verify a hidden-card/deck-order perturbation keeps actor logits and export bytes identical while Q can respond to the altered truth.
- [ ] 4.3 Train a separate centralized candidate-Q critic from the new record and current staged menu, retaining seat-specific Q semantics; verify held-out calibration, finite targets, exact menu alignment, and bounded memory/SPS before allowing `vrpo_centralized` actor updates.

## 5. Run matched pilots and promotion gates

- [ ] 5.1 Run preflight for all three experimental modes in isolated directories, including native protocol/boundary checks, checkpoint import, CUDA discovery, numeric finiteness, and a short real-engine rollout; verify no command touches the active 40M run or replaces its production native module.
- [ ] 5.2 Run a bounded `vrpo_centralized` pilot and matched restarted GAE control, then evaluate both seats on held-out decks without search/belief assistance; verify equal-step and equal-wall-clock views, paired uncertainty, combo/interruption metrics, calibration, SPS, invalid games, and timeout rates are reported.
- [ ] 5.3 Write a promotion report that states which gate actually passed and whether Q-boosting improved the declared metrics; verify weak/unstable critic calibration or non-finite targets block a longer run rather than being masked by a short win-rate change.

## 6. Synchronize only an accepted pilot

- [ ] 6.1 After the live 40M run ends, synchronize the accepted VRPO source and separate pilot artifact provenance across local, GitHub, home, and H200 while preserving untracked training assets; verify four source identities, deployed binary hashes, rollback backups, and a new run directory match the manifest.
