# K9VS Stage 3 mixed-opponent report (2026-09-11)

## Implementation

Git revision `dd0894f` adds mixed opponent training to `scripts/cleanba.py`.
Actor workers are assigned in a configurable cycle. The validated configuration
uses ten actors:

- 5 current-policy self-play actors (50%)
- 3 historical-checkpoint actors (30%)
- 2 built-in greedy-bot actors (20%)

Historical actors rotate through the configured checkpoint list every update.
Opponent actions in historical games are excluded from the training loss. Actor
JIT initialization is serialized to prevent concurrent XLA autotuning and CUDA
command-buffer capture failures when many actors share one GPU.

## Smoke test

A two-update smoke test exercised all three opponent modes simultaneously:

- All ten actors started with the intended 5:3:2 assignment
- `optimizer_finite=True`
- `notfinite_count=0`, `total_notfinite=0`
- Both updates completed and an exact-step checkpoint was saved

## Formal continuation run

The Stage 1 final checkpoint was used as the starting model. Six checkpoints
from the Stage 1 run formed the historical pool.

- Additional timesteps: 9,994,240
- Updates: 488
- Optimizer steps: 39,040
- Parallel environments: 160 total (16 per actor)
- Environment threads: 110 total (11 per actor)
- Rollout steps: 128
- Minibatches: 80
- Learning rate: `5e-5`, linearly annealed
- FP32, advantage normalization enabled
- Sustained throughput: approximately 6,743 SPS
- Approximate wall time including evaluation: 28m 04s
- Final optimizer state: finite, with zero rejected updates

The lower throughput compared with pure self-play is expected: ten actor models,
historical-policy inference, greedy-bot environments, and larger periodic
evaluation add overhead.

## Periodic greedy-bot evaluation

Each displayed result is the mean of ten actor evaluations, 256 games per actor
(2,560 games total per evaluation point).

| Update | Additional steps | Win rate | Mean return |
|---:|---:|---:|---:|
| 100 | 2,048,000 | 98.40% | 3.5262 |
| 200 | 4,096,000 | 99.30% | 3.9374 |
| 300 | 6,144,000 | 98.79% | 3.6933 |
| 400 | 8,192,000 | **99.92%** | **4.1654** |

An independent 256-game greedy-bot evaluation of the final checkpoint produced:

- Win rate: 97.66%
- Approximate 95% Wilson confidence interval: 95.04%-98.91%
- Mean reward: 3.9608
- Mean episode length: 92.09
- Evaluation throughput: 10,781 SPS

## Comparison with the Stage 1 baseline

The mixed-policy final model played 2,048 games against the Stage 1 final model:

- Mixed-policy win rate: **60.94%**
- Approximate 95% Wilson confidence interval: **58.80%-63.03%**
- Mean reward: 1.1711
- Mean episode length: 178.08
- Evaluation throughput: 28,591 SPS

The interval is well above 50%, providing strong evidence that mixed-opponent
training improved direct-play strength over continued pure self-play.

## Historical-pool evaluation

The mixed-policy final checkpoint played 256 games against each source
checkpoint:

| Historical step | Win rate | Mean reward | Mean length |
|---:|---:|---:|---:|
| 819,200 | 91.80% | 4.3913 | 145.49 |
| 3,276,800 | 82.81% | 3.3677 | 149.25 |
| 4,915,200 | 75.78% | 2.6578 | 158.12 |
| 6,553,600 | 67.97% | 1.7031 | 166.36 |
| 7,372,800 | 68.36% | 1.8001 | 172.58 |
| 8,192,000 | 65.23% | 1.4584 | 172.96 |

The final model beats every member of its historical pool by a clear margin.

## Artifacts

Remote run directory:

`/root/ygo-agent-gpu-20260910/training-runs/h200-k9vs-mixed-10m-20260911T020646Z`

Final checkpoint:

`checkpoints/1789092408_step_000009994240.flax_model`

Final SHA256:

`4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`

Starting Stage 1 checkpoint SHA256:

`3ca917eb2aeb8ee1e6c12b9f4d2b749bac6dc426a0b832b05b7146f3f665574b`

## Conclusion

Stage 3 succeeded. Mixed-opponent training is numerically stable, improves
direct-play strength substantially, and retains strong performance against both
the greedy bot and every historical opponent. The mixed-policy final checkpoint
is now the principal K9VS baseline for subsequent embedding experiments.
