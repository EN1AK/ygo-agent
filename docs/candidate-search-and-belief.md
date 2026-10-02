# Candidate search and belief: implementation status

Status as of 2026-10-02: **isolated experimental primitives**, not promoted
fair-play search or a trained opponent-hand model. The active 100M PPO job does
not import these modules or consume search targets. Native snapshot/rollback,
real self-play belief-label collection, particle application, and held-out
strength tests remain unfinished OpenSpec tasks.

## What exists

- `ygoai.rl.candidate_search`: bounded paired rollouts, acting-player to
  root-player value conversion, per-action returns, particle-clustered
  uncertainty, and a KL-regularized policy update. The update follows
  [Ataraxos Appendix D.7](https://arxiv.org/html/2511.07312v1); coefficients must
  be calibrated for YGO, not copied from Stratego.
- `ygoai.rl.replay_search`: deterministic seed/prefix replay reference, separate
  recurrent states, explicit invalid-terminal rejection, and exception cleanup.
  Every result is **oracle_exact_state**. Replay cost includes replaying the
  entire prefix for every branch; this is not a fast native snapshot.
- `ygoai.rl.search_policy`: explicit eligibility wrapper; fair-play mode rejects
  the replay oracle before querying a branch. It is not wired into normal
  evaluation or PPO. Existing PUCT remains a separate diagnostic comparator.
- `ygoai.rl.search_audit`: lossless versioned envelope for earlier PUCT JSON.
  Visits, edge values, critic values, raw action identity and blocked actions
  keep their original meanings. A frozen historical cycle audit is tested.
- `ygoai.rl.belief`: known-deck fixture information sets, reproducible constrained
  multiset particles, and a pre-query application-order guard. The guard checks
  API order; it cannot certify a native implementation merely acknowledging a
  particle hash. The sampler does not access an engine or true hidden state.
- `ygoai.rl.jax.belief_head`: independent autoregressive Transformer with causal
  teacher-label masks, legal-context attention, remaining-count masks, and
  constrained seeded sampling. Hidden slots are an ordered sequence with padded
  suffixes; prediction token 0 is reserved for padding.
- `scripts/train_belief_head.py`: independent offline optimizer/checkpoint,
  finite-update checks, whole-duel holdout, teacher-forced NLL/top-1/ECE and
  particle constraint validity. No actor-weight or PPO-format changes.

## Diagnostic candidate CLI

The existing `scripts/run_counterfactual_search.py` accepts `--method candidate`,
`--candidate-count`, `--candidate-rollouts`, `--candidate-wall-seconds`,
`--candidate-policy-kl`, and `--candidate-magnet-kl`, plus its existing decision,
checkpoint, runtime-asset and depth arguments. The input must contain the
frozen root observation digest and legal menu. The candidate mode checks player,
digest, policy and value before search. Older `puct`, `enumeration`, and `both`
commands keep their existing behavior and are not relabelled as fair play.

Candidate subsets retain their original total probability mass; unsampled
actions retain their raw probabilities. Policy action defaults to argmax;
sampling is explicit. All candidates share the same particle/trial seeds.
If a comparison is incomplete, a chain is unresolved at the depth boundary,
or a resource gate fails, the result returns raw argmax with a reason. Strict
mode raises. Wall time is cooperative around backend calls, not a preemptive
timeout for a stalled native call. Snapshot-byte accounting covers the stored
snapshot representation, not total Python/JAX/engine process memory.

## Offline belief data contract

NPZ fields are exactly:

- `public_tokens [N,T,F]`: normalized legal-view encoding, including the acting
  player's legitimately known private information if the producer encodes it.
- `public_mask [N,T]`: boolean; padding features must be zero.
- `hidden_slot_features [N,S,G]`: public descriptors, never hidden identities.
- `hidden_tokens [N,S]`: supervised vocabulary indices, **label channel only**.
- `slot_mask [N,S]`: nonempty valid prefix then padded suffix.
- `remaining_counts [N,V]`: explicitly declared legal prior; PAD count is zero.
- `duel_ids [N]`: stable text IDs used to prevent same-duel train/holdout leakage.

The producer manifest declares `schema: ygo-belief-dataset-v1`,
`data_sha256`, `runtime_manifest_sha256`,
`information_assumption: known-deck-self-play`, and
`input_semantics: acting-player-legal-view`. The loader validates data integrity,
not the truthfulness of a producer's declaration. Real native producer leakage
tests are still required. Do not manufacture these declarations for arbitrary
full-state data.

Run `scripts/train_belief_head.py --data <data.npz> --producer-manifest <manifest.json>
--output <new-directory>`. Output includes a separate `belief.flax_model`,
`train.jsonl`, and `report.json`. A successful synthetic pipeline test is only a
code regression test; its metrics are not learned duel strength. Current
validation reports teacher-forced prefix calibration, not calibrated joint
particles, held-out deck generalization, or decision gains.

## Validation and rollout boundaries

Local NumPy tests cover paired seeds, viewpoint, menus, chain fallback,
budget cleanup, regularization, legacy JSON identity, information-set constraints,
guard order, data splits and calibration calculations. H200 CPU-only isolated
tests cover real JAX gradients, masks, constrained samples and a three-update
synthetic offline training/save/report cycle. No GPU search is inserted into
the long PPO run. No native artifact was changed for these Python additions.

Next: capture current-runtime tactical/chain/combo replay fixtures; validate the
replay candidate path end to end; route real self-play labels outside actor
observations; implement and differentially test fixed-address native snapshots
and particle application; only then measure paired fair-play strength and
consider target generation/distillation. Keep raw-policy rollback as default.
