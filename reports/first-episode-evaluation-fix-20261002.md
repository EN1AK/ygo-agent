# Paired evaluation collection correction

## Cause and change

The greedy evaluator's legacy collection took the first N completed episodes
across a vector batch. A fast environment could finish its second duel before
another environment finished its first. Equal seed and total count therefore
did not establish a fixed initial-duel sample or paired both-seat comparison.

Add opt-in `--first-episode-per-env` (episodes must equal environments), used
explicitly by `eval_vrpo_baseline.py`. Every accepted episode records its
environment index, and the baseline refuses missing/duplicate indices. Existing
unflagged evaluation collection and all actor/training behavior are unchanged.
`eval_mixed_match.py --candidate-seat 0/1` allows the same batch to be evaluated
in both seating orders, instead of mistaking half-seat batches for paired deals.

Replay collection now uses one fresh working directory per duel. The baseline
verifies every decision's identity, menu alignment, finite logits/value,
normalized probabilities, raw unassisted action and final result before
recording the artifact. Logs spanning multiple duels/environments are rejected
when requesting the currently single-duel decision-log format.

## Actual engine evidence

Home isolated CPU stage `/home/ygo/eval-first-episode-20261002-v1`, same deployed
native/fusion-procedure hashes as the active training runtime, real 53,002,240
PPO checkpoint SHA `efaa6448201e49778b1604f56efa294bd05c1518861aed30bdf55051078c6fda`.
This is a **validation fixture**, not the designated 100M baseline.

- 8 greedy first-duel attempts (four per seat): all natural, indices 0..3 once
  each. Original completion ordering for seat 0 was **3,0,3,2**; corrected
  collection is **3,0,2,1**. The old collector substituted environment 3's
  second 81-step duel for environment 1's 241-step initial duel. Both samples
  happen to give 2/4 wins; the fault is demonstrable sampling mismatch, not a
  claim that every reported win rate must numerically change.
- Two initial same-model self-play duels per seat: swapped-seat lengths are
  118/134 in both runs, with exact reversed rewards and matching termination
  reasons/environment indices. This validates seat orientation on real games.
- Two single-duel greedy replays have identical native duel seed 1104861466;
  their complete model logs contain 53/30 decisions respectively. Recorded
  first-environment results match the corresponding batch rows. All required
  numeric/menu/identity/terminal validation checks passed.
- Forced `max_steps=1`: two invalid terminals, reason 2, zero valid games,
  zero counted wins/losses, win rate and interval null (not zero or NaN).
- Eleven focused unit tests pass, including duplicate completions, missing indices,
  seat orientation, invalid rewards, and truncated/non-finite/assisted replays.
  A mocked orchestration test also covers the entire manifest flow with a
  single-deck matrix and isolated replay directories; this is not an additional
  real duel. Terminal logs preserve the original engine win-reason code rather
  than the boolean LP-victory statistic used in aggregate reporting.

Local primary evidence: `training-runs/eval-first-episode-evidence-20261002.tar.gz`,
SHA `34694dbc7a237ae61fb5124c001de367703140347e0ce150cae522545e72e5eb`.
Additional legacy/invalid-control logs are retained in the same home stage and
downloaded as `training-runs/eval-first-episode-controls-20261002.tar.gz`, SHA
`ce68130f06fea2fbf5a17e1b019f80351b737094d8aaec547c8ada3e263b815c`.
No strength promotion is based on these
small samples. The test recordings still need Chinese commentary if delivered
as user-facing replay examples; they are currently validation evidence only.

The revised prospective matrix and strategy-review denominators are in
`reports/100m-evaluation-protocol-20261002.md`. Re-run historical controls under
the corrected collection protocol; do not subtract an old unpaired headline
win rate from the future 100M score.
