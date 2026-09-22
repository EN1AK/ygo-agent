# Structured-lite Full 40M Evaluation (2026-09-22)

## Decision

Keep the 40M checkpoint as a valid candidate and a historical opponent, but do
not promote it as the new best model and do not continue the same training run
solely on the basis of the in-training evaluation score.

The checkpoint is measurably stronger than its 1M continuation start in direct
play, but it did not improve the independent greedy-bot anchor. The next compute
allocation should first expand the independent opponent suite and seed set.

## Artifacts

- Candidate: `/home/ygo/ygo-agent/training-runs/structured-lite-full-cont-39m-20260922-night/checkpoints/1790011620_step_000040000000.flax_model`
- Candidate SHA-256: `a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`
- Start/baseline: `/home/ygo/ygo-agent/training-runs/structured-lite-full-warmstart-1m-20260922/checkpoints/1790009101_step_000001000000.flax_model`
- Baseline SHA-256: `be04107298d167e0647c116632f01afea0380eb68541b7272a1d7cc0f5173584`
- Observation schema: `structured-lite-v1`, full variant
- Semantic metadata SHA-256: `3ebb70a61545767617704d7aa0ef1e9f376c08f5daffb3cb655fff743aac03c6`
- Code-list SHA-256: `f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`
- Raw results: `/home/ygo/ygo-agent/training-runs/structured-lite-full-cont-39m-20260922-night/evaluation-20260922/`

## Results

### Candidate versus 1M start

Four independent seed batches were run with 128 games each. In every batch the
candidate occupied player 0 in 64 games and player 1 in 64 games.

| Seed | Candidate wins | Games | Win rate | Mean episode length |
|---:|---:|---:|---:|---:|
| 42001 | 72 | 128 | 56.25% | 253.17 |
| 42002 | 75 | 128 | 58.59% | 241.82 |
| 42003 | 68 | 128 | 53.12% | 235.30 |
| 42004 | 82 | 128 | 64.06% | 261.88 |
| **Combined** | **297** | **512** | **58.01%** | — |

Wilson 95% confidence interval: **53.69% to 62.21%**. The point estimate is
approximately **+56 Elo** relative to the 1M start (descriptive only; the game
sampling is not a formal Elo pool).

Harness checks:

- 40M versus itself: 67/128, 52.34%.
- Reversed model ordering, 1M versus 40M: 47/128, 36.72%.
- 38.5M versus 1M spot check: 77/128, 60.16%; there is no evidence of a final
  1.5M-step collapse.

### Greedy-bot anchor

The same fixed evaluation settings and seed pair were used for the 40M candidate
and the 1M start.

| Checkpoint | Player 0 | Player 1 | Combined |
|---|---:|---:|---:|
| 40M | 110/256 (42.97%) | 123/256 (48.05%) | **233/512 (45.51%)** |
| 1M | 106/256 (41.41%) | 130/256 (50.78%) | **236/512 (46.09%)** |

The difference is **-0.59 percentage points** for 40M. An unpaired approximate
95% interval for the difference is **-6.69 to +5.52 points**, so this is
statistically indistinguishable from no change. The candidate also remains
below 50% against this anchor (40M Wilson 95% interval: 41.24% to 49.84%).

## Training integrity

- Training completed at global step 40,000,000 and wrote the final checkpoint.
- Final optimizer report was finite with `notfinite_count=0` and
  `total_notfinite=0`.
- Final metrics: total loss `-0.0073136`, policy loss `-0.0292785`, value loss
  `0.0285012`, entropy `0.6536338`, approximate KL `0.0056406`.
- The final training-window average return was `0.0120`; average episode length
  was `208`.
- The training-internal fixed-checkpoint evaluation rose from 62.50% to 96.25%,
  but that result did not reproduce on independent seeds. Treat it as a narrow,
  optimistic monitor rather than a promotion gate.

## Interpretation and next action

The 39M continuation learned a policy that beats its 1M ancestor reliably in
head-to-head play. It did not, however, produce a detectable improvement against
the fixed greedy bot. This pattern is consistent with specialization to the
self/history mixture or with an internal evaluation seed slice that is too
narrow.

Before spending a large new budget, run a promotion suite with fresh seeds and
at least three opponents: 1M start, 38.5M/40M historical checkpoints, and the
greedy bot (plus WindBot when available). Use at least 512 games per matchup,
balanced by seat. Promote or extend only if the candidate improves both the
historical pool and at least one external anchor without a significant regression
on the other.

