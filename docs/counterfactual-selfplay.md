# Counterfactual self-play distillation

The implementation lives in `ygoai.rl.counterfactual`. It keeps engine restore
separate from analysis so the current deterministic replay fallback can later
be replaced by an arena-backed ocgcore snapshot without changing artifacts.

Pipeline:

1. Record each self-play decision with a stable duel/decision ID, player,
   engine action prefix, legal actions, selected action, policy logits and
   critic value. The portable snapshot is `{seed, actions, player}` and the
   trace stores a stable observation digest.
2. Rank ambiguous decisions with `counterfactual_pipeline.py select`.
3. For every selected point evaluate the full Cartesian product
   `legal action x belief particle x rollout seed` through
   `evaluate_decision` (or distributed workers that emit the same rollout
   JSONL schema).
4. Aggregate action returns into `Q(s,a)`, standard errors, selected-action
   regret, a temperature-controlled soft policy target, and pairwise
   preferences. `mine_errors` emits high-regret policy reversals such as the
   Veiler-vs-Ogre class of mistake.
5. Train the policy with cross entropy against `soft_policy_target`; optionally
   add a pairwise preference loss. `distillation_loss` and `preference_loss`
   define those objectives independently of the JAX/Torch training front end.
   Keep the original legal-action ordering.

Rollout JSONL rows contain `decision_id`, `action`, `particle`, `rollout_seed`,
and scalar `value`. Values and critic estimates are state/action-return
estimates; they must not be described as per-action Q-values until the
counterfactual aggregation has been performed.

## Snapshot backends

`ReplaySnapshotBackend` recreates a duel from the original seed and replays its
action prefix. It is intentionally slow but portable and deterministic.
`scripts/verify_replay_snapshot.py` independently restores sampled decision
points and requires exact digest, acting-player and legal-count agreement.

The supplied MirrorForce archive documents a faster arena implementation but
does not contain its modified core sources (`mfsnap.cpp` and allocator patch).
Its ABI is four calls: `duel_snapshot`, `duel_rollback`,
`duel_snapshot_free`, and `duel_arena_extent`. A future native backend should
implement the same Python `SnapshotBackend` protocol and pass a byte-identical
rollback test before use. Hidden information must be replaced by a sampled
belief particle before any model query; otherwise the search leaks the real
opponent hand.
