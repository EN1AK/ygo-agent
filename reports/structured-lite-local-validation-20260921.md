# Structured-lite local GPU validation (2026-09-21)

## Scope

Validation ran in the home WSL environment on an NVIDIA GeForce RTX 5070 Ti.
The run used the frozen Stage 3 checkpoint with SHA-256
`4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`
and the pinned 13,492-entry code list with SHA-256
`f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`.

Artifacts are under
`training-runs/structured-lite-local-validation-20260921` in the Windows
workspace. Large generated checkpoints remain in the corresponding WSL run
directory under `/home/ygo/ygo-agent`.

## Schema, model, and environment checks

- Regenerated schema manifests contain 9,559 legacy bytes and 26,095
  Structured-lite bytes. Their parsed content matches the committed manifests;
  the WSL copies differ bytewise only because of line endings.
- The `full` and `relationship-only` model variants produced finite logits and
  scalar `V(s)` values for padding and maximum-capacity fixtures. Reversing the
  legal-action order and restoring it produced maximum logit error `0.0`.
- Fixed-seed real-engine duels passed for legacy and Structured-lite. The
  Structured-lite trace observed all event IDs 1 through 12, checked 3,006
  hidden card rows, found zero hidden-semantic violations and zero unsafe event
  references, and explicitly reported overflow when action capacity was two.

## Migration checks

An initial dry run exposed that the mutable WSL code list had grown to 14,981
entries and would discard the Stage 3 card embedding as shape-incompatible.
Using the pinned Windows code list resolved the mismatch.

The corrected migration accounted for all parameter leaves:

- source: 128 = 126 copied + 2 shape-incompatible;
- destination: 165 = 126 copied + 37 newly initialized + 2
  shape-incompatible;
- intentionally excluded: 0.

The two expected incompatible leaves are the gate and up projections of
`Encoder_0/GLUMlp_1`, whose input width changes from 512 to 768. The generated
full warm-start passed a finite forward check. Its SHA-256 is
`05583e71a4c7cfd7bdca74595fb819705bbe7fb028b45a5658c33d24018be06c`;
the source checkpoint hash remained unchanged.

## Short optimization and deterministic evaluation

Scratch and warm-start candidates each completed one 64-step PPO update and
saved a reloadable checkpoint.

| Candidate | Total loss | Policy loss | Value loss | Approx. KL | Finite optimizer |
| --- | ---: | ---: | ---: | ---: | --- |
| Scratch | -0.007827 | 0.003796 | 0.001437 | 0.00000558 | yes |
| Warm start | 0.106625 | 0.088069 | 0.029786 | 0.059789 | yes |

Each saved checkpoint was evaluated twice with seed 23 for four greedy-bot
games. Within each candidate the two runs had identical per-game length,
reward, win, and win-reason sequences.

- Scratch: 0/4 wins, mean length 63.00, mean reward -1.3049.
- Warm start: 4/4 wins, mean length 111.75, mean reward 1.6250.

These four-game results validate the execution path only and are not promotion
evidence. The warm-start initial KL is materially higher than the scratch KL
and must be monitored in a longer controlled smoke run.

## Matched resource benchmark

The three training variants used the same RTX 5070 Ti, seed 31, `elfnote`
deck, pinned code list, 8 local environments (16 effective player
environments), 64 rollout steps, 8 minibatches, four updates, and disabled
actor/learner concurrency. XLA GPU preallocation was disabled. Training SPS is
the mean of updates 3 and 4, after excluding compilation and parameter-transfer
warm-up. Peak memory is the maximum of 200 ms `nvidia-smi` samples spanning the
complete process.

| Variant | Observation bytes | Environment SPS | Steady training SPS | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 9,559 | 10,358 | 2,358.5 | 2,459 MiB |
| Relationship-only | 26,095 | 10,105 | 1,362.0 | 2,463 MiB |
| Full Structured-lite | 26,095 | 10,105 | 1,318.0 | 2,461 MiB |

The pure environment test used random-vs-random play for 128 completed games,
16 environments and threads, and seed 37. Legacy and Structured-lite produced
identical game trajectories and 23,904 environment steps. Structured
observation construction reduced environment throughput by 2.4%. Relative to
legacy, steady learner throughput was 42.3% lower for relationship-only and
44.1% lower for full Structured-lite. Peak memory remained within 4 MiB across
the three measurements; model/runtime compute, rather than persistent GPU
allocation, is the material cost in this short matched profile.

All runs completed with finite optimizer state. Logs, GPU samples, checkpoint
metadata, hashes, and the machine-readable summary are retained under
`training-runs/structured-lite-local-validation-20260921`. The benchmark
checkpoints themselves remain under the corresponding WSL directory.

## Capacity selection

A 512-game, 113,714-decision untruncated elfnote audit selected and froze 32
public events and 8 group references for `structured-lite-v1`. Capacity 32
retains a median 14 and p95 24 recent decisions; group-reference demand peaked
at 2, so 8 retains fourfold measured headroom. The full decision, rejected
alternatives, invalid attempts, consistency verification, and hardware-upgrade
policy are recorded in
`reports/structured-lite-capacity-selection-20260921.md` and the associated
machine-readable run artifacts.

## Remaining gates

The validation does not complete the prompt/message-family evidence,
hidden-identity permutation audit, the separate production action-overflow
gate, fixed-budget A/B/C experiment, or promotion evaluation.

