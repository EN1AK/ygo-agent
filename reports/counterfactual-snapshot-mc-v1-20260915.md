# Counterfactual snapshot and MC v1 report (2026-09-15)

## Snapshot verification

The GPU server ran a 239-decision elfnote self-play duel with the Stage 3 mixed
checkpoint. Sixteen evenly spaced decision points (steps 0 through 238) were
restored independently from the recorded duel seed and action prefix.

Every restored point matched the source trace exactly on:

- SHA-256 digest of all observation arrays, including dtype and shape
- acting player
- legal-action count

Result: `REAL_ENGINE_SNAPSHOT_GATE_PASS`.

During the first attempt the verifier exposed a real API mismatch: the native
Gymnasium environment returns a five-element step tuple while the wrapped
evaluation path returns four. Commit `15fa086` normalizes both APIs. The failed
attempt's apparent pass marker was invalid because `tee` masked the pipeline
exit status; the accepted rerun used `set -euo pipefail`.

Server artifact directory:

`/root/ygo-agent-gpu-20260910/training-runs/counterfactual-snapshot-gate-20260915T0100Z`

## Bounded MC experiment

This is a mechanical end-to-end baseline, not belief-particle search. It uses
the original hidden state (`K=1`) and seeded uniform-random continuation policy.

- Source trace: the verified 239-decision duel above
- Decision selection: priority at least 0.35, at most four legal actions
- Selected decision points: 6
- Legal actions: all actions at every selected point
- Rollouts: 3 seeds per action
- Total branches: 57
- Target temperature: 0.25
- Preference margin: 0.05
- Error threshold: regret at least 0.15
- Wall time: approximately 67 seconds

Results:

- Policy top action overturned at 4/6 points
- Mean estimated regret: 0.523
- Maximum estimated regret: 1.131 (decision step 29)
- Mean per-action standard error: 0.280
- Four rows passed the provisional regret threshold

The return is the environment's existing shaped terminal reward, expressed from
the root player's perspective. It is a legitimate MC action-return estimate but
is not a calibrated win probability.

The sample count is deliberately too small for training. High standard errors
and uniform-random continuations mean the four reversals are candidates only;
they must not yet be labelled strategic errors such as Veiler-vs-Ogre. The next
statistical gate should rerun candidates with paired/common rollout randomness,
at least 32 seeds per action, and confidence-aware regret. Hidden-state belief
permutation and checkpoint-policy rollouts remain required before producing the
first retraining corpus.

Server artifact directory:

`/root/ygo-agent-gpu-20260910/training-runs/counterfactual-mc-v1-20260915T0130Z`

Relevant implementation revisions:

- `fae4beb`: counterfactual aggregation and replay snapshot backend
- `10aff82`: observation-digest verification
- `15fa086`: Gym/Gymnasium step normalization
- `d92c3c9`: bounded K=1 MC experiment runner
