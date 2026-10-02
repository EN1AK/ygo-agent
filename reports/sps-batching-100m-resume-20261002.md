# SPS tuning and resumed 100M baseline

The formal continuation is live, not finished. Status checked at
2026-10-02 20:17 Asia/Shanghai: cumulative 56,611,840 steps, approximately
5,758 SPS, finite optimizer/loss metrics, nonfinite counters zero.

## Measured configuration choice

All pilots started from the same verified 55,997,440-step checkpoint. Pilot
updates are **not** counted toward the formal continuation. SPS below is the
mean of the final four logged steady-state samples, excluding warm-up.

| Actors | Envs per actor | Total envs | PPO batch | Steady SPS |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 16 | 480 | 30,720 | 3,661.75 |
| 45 | 16 | 720 | 46,080 | 3,672.75 |
| 60 | 16 | 960 | 61,440 | 3,593.00 |
| 5 | 48 | 240 | 15,360 | **5,784.75** |
| 15 | 16 | 240 | 15,360 | 3,826.25 |

All five pilots exited 0 and passed finite-update checks. The fixed-total-env
comparison improves observed SPS by approximately 51%; it does not prove an
absolute hardware maximum. The 5x48 and 15x16 comparison preserves 80 minibatches
of size 192, 64 rollout steps, learning rate 0.0001, and the PPO update count.
Do not infer strength improvements from these short throughput pilots.

Increasing actor count alone did not help. Larger per-actor inference batches
reduced rollout/queue waiting; formal learner train time is about 2.6-2.7s per
update and data wait is near zero. GDB thread-event chatter was suppressed for
both fixed-total-env pilots; crash stack/register capture remains enabled.

## Formal continuation

- Run: `/root/ygo-agent-gpu-20260910/training-runs/multideck-fastest-56m-to100m-913b40d-20261002`.
- Training source: `913b40dfb1edbdb382fcf5130c709a42e3137f6c`.
- Starting cumulative steps: 55,997,440. Target: **100,003,840**.
- Parent checkpoint SHA-256:
  `53454af4d2e069274c80056ef156d6ab09519966323a9ef7d8ebdce215853c57`.
- Production native SHA-256 unchanged:
  `5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`.
- Configuration: 5 actors x 48 envs, 11 env threads per actor, total 240 envs;
  batch 15,360, 80 minibatches, Q off, max-step loss 2.
- Continuation restores model weights and sampler counters with a fresh optimizer,
  matching earlier resume semantics. Explicit sampler repartition preserves all
  240 counters in flattened logical order. Actor-dependent RNG streams change,
  so this is distributional continuation, **not bit-identical trajectory resume**.
- Initial detached launcher PID 322807; recheck current process identity rather
  than relying permanently on this PID.

Before launch, config-only gates passed and both observation schemas each
completed two deterministic 117-step real-engine duels without hidden-semantic,
event-reference, identity or overflow violations. The immutable runtime release
is `dist/runtime-releases/fastest-56m-913b40d-20261002` on H200 and in the local
`ygoai/dist/runtime-releases/` archive. All 17 artifact hashes were verified at
both locations; manifest SHA-256:
`bd4504248aa215d7cf0f1e3d9e927275d77537419ba795f3983a25cd64bd4aa9`.

Evidence directories: `training-runs/actor-scale-55997440-8eae2ac-20261002`
and `training-runs/actor-batching-55997440-913b40d-20261002` on H200. The latter's
`selection.json` records all pilots and the exact formal launch environment.
Its `completed.txt` means the tuning/handoff succeeded, **not** that 100M finished.
The prior 53M continuation was intentionally stopped for tuning; its old
termination marker is not a new training crash.

At 5.75k SPS, the remaining approximately 43.4M steps take about 2h06m, giving
an initial ETA near 22:25 Asia/Shanghai. This assumes no interruptions or sustained
throughput decline. Keep monitoring the formal run's checkpoint hashes, finite
metrics, process progress and completion markers. Search/belief experiments are
isolated, and no new search or belief weights enter this PPO run.
