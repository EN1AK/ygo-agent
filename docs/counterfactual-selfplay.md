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

`scripts/run_counterfactual_mc.py` is the bounded mechanical baseline. It uses
the exact recorded hidden state (`K=1`) and seeded uniform-random continuations.
It must not be reported as belief-particle search; its purpose is to validate
the end-to-end action/Q/regret artifact path before adding hidden-state
permutation and model-policy rollouts.

The supplied MirrorForce archive documents a faster arena implementation but
does not contain its modified core sources (`mfsnap.cpp` and allocator patch).
Its ABI is four calls: `duel_snapshot`, `duel_rollback`,
`duel_snapshot_free`, and `duel_arena_extent`. A future native backend should
implement the same Python `SnapshotBackend` protocol and pass a byte-identical
rollback test before use. Hidden information must be replaced by a sampled
belief particle before any model query; otherwise the search leaks the real
opponent hand.

## Diagnostic search

`scripts/run_counterfactual_search.py` provides two deliberately offline
diagnostics on the same replay snapshot:

- `bounded-enumeration-terminal-rollout` enumerates every legal action for a
  bounded number of decisions, then samples the checkpoint policy to the end
  of the duel. Common rollout seeds are used across frontier leaves.
- `puct-leaf-value` restores action-prefix nodes and runs two-player PUCT,
  backing up the checkpoint critic value at the leaf instead of requiring a
  complete duel. Opponent nodes minimize the root player's value.

The YGOPro environment does not expose a cheap public clone operation. Each
node is therefore restored by replaying the seed and action prefix, including
both recurrent states. This is suitable for targeted CPU diagnostics, not for
online self-play throughput. The command refuses to search if the restored
root legal actions, policy probabilities, or critic value differ from the
recorded decision beyond the configured tolerance.

Leaf-value PUCT output is a search diagnostic, not automatically a training
target. Promote it only after critic calibration and hidden-information belief
particles have been validated; otherwise a confident but biased critic can
turn search visits into confidently wrong supervision.

### Leaf-value audit trail

Before leaf-value PUCT is used to create any preference or soft-policy target,
persist one audit row per simulation. Each row must contain the decision ID,
simulation index, complete action-index prefix, acting player at the leaf,
terminal flag, raw checkpoint `V(s)`, value after conversion to the root
player's perspective, and every backed-up edge value. When available, also
store the leaf observation digest and a human-readable action/state summary.
Root visit counts and aggregate Q values alone are insufficient to explain why
search preferred an action: they cannot distinguish learned tactical judgment
from value extrapolation, viewpoint errors, or a few extreme leaves.

PUCT must use one checkpoint as the evaluator for the entire tree, with
separate recurrent states for the two player views. It must not alternate
between independently trained critics and then assume that sign inversion
makes their scales compatible. `select_chain` response prompts are protocol
nodes: expand through them, but do not stop a simulation and back up their
critic value. Audit output must record the leaf prompt type and whether it is a
resolved-chain boundary.

### Detecting prompt-value bias

Prompt correlation must be measured against outcomes rather than inferred from
a few large value changes. Build a diagnostic table with one row per evaluated
state: duel/chain ID, observation digest, public-board fingerprint excluding
the legal-action prompt, prompt type, acting player, chain depth, phase, raw
value, common-player-perspective value, and eventual `±1` result.

Use three complementary checks:

1. Report value calibration and residual `V-result` by prompt type and acting
   player, including sample count, mean error, MSE/Brier score, and bootstrap
   confidence intervals.
2. Within one chain, match adjacent pass states whose public-board fingerprint
   is unchanged. Test the paired value change when only the response prompt or
   acting player changes. A systematic nonzero change is direct evidence of a
   prompt/viewpoint effect.
3. Fit a held-out mixed-effects or fixed-effects regression of value residual
   on prompt type, acting player, and chain depth while controlling for phase,
   board fingerprint, and duel/chain identity. Validate any detected effect on
   separate seeds and duels.

Feature masking or probing can help localize the source afterward, but masking
legal actions is out of distribution and is not sufficient evidence by itself.

### Reward-scale invariant

Counterfactual evaluation must use the same reward definition as the checkpoint
that supplies `V(s)`. The Stage 3 checkpoints were trained with
`greedy_reward=False`: nonterminal reward is zero and the terminal result is
`+1` for a win or `-1` for a loss. All counterfactual entry points therefore
set this flag explicitly and write `reward_mode`, `greedy_reward`, and/or the
terminal scale into their output. Never rely on the native environment default,
which is `greedy_reward=True` and changes terminal magnitude according to duel
length. Mixing that speed-shaped scale with a win/loss critic invalidates PUCT
backup and direct critic-versus-rollout calibration claims.
