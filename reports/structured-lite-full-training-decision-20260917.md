# Structured-lite full-training decision — 2026-09-17

## Decision

Proceed toward one complete warm-start training run of the full
`structured-lite-v1` model. Keep it isolated from the main legacy route until
the fixed validation and promotion gates pass.

## Warm-start artifact

- Source checkpoint:
  `/root/ygo-agent-gpu-20260910/training-runs/h200-elfnote-stage3-continue-10m-20260912T180644Z/checkpoints/1789236406_step_000009994240.flax_model`
- Source SHA-256:
  `a4d2fe23330fa2cb464321d09beb52639e3f767601e6fb974858d2dcf0468e95`
- Output checkpoint:
  `/root/ygo-agent-gpu-20260910/training-runs/structured-lite-full-warmstart-20260917/migrated/warm-start.flax_model`
- Output SHA-256:
  `b8f8e1c5d40f1e7a9dc0768281fa0ada668f729ae78f25c45d528a1f6e2ae457`
- Schema/variant: `structured-lite-v1` / `full`
- Semantic table SHA-256:
  `8ffc1ffb825423cd725d050407c2975725befdcaae7c163faba665e5124776b9`
- Code-list SHA-256:
  `f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`

The source hash was unchanged after migration. The migrated model completed a
finite forward pass.

## Migration accounting

The dry run and real migration both accounted for every parameter leaf:

- source: 128 total = 126 copied + 2 shape-incompatible;
- destination: 165 total = 126 copied + 37 newly initialized + 2
  shape-incompatible;
- intentionally excluded: 0.

The two incompatible tensors are the gate and up projections of
`Encoder_0/GLUMlp_1`, whose input width grows from 512 to 768. They are newly
initialized in the migrated model together with the Structured-lite modules.
The complete machine-readable inventories remain beside the output under
`dry-run/migration-report.json` and `migrated/migration-report.json`.

## Launch status

The complete training run was requested on 2026-09-17, but was not started at
recording time because the configured GPU server rejected the SSH connection.
The internet-connected CPU/build server holds the verified warm-start artifact.
Long training must not be moved to that CPU server: the required GPU smoke,
throughput/memory comparison, and short scratch-versus-warm-start optimization
checks still precede it.

## First experiment

Use identical decks, seeds, rollout budget, and evaluation schedule for:

1. legacy Stage 3 baseline;
2. Structured-lite relationship-only ablation;
3. Structured-lite full warm start.

The full run may begin only after the short warm-start candidate saves and
reloads a checkpoint with finite losses and valid games. Preserve TensorBoard
events, console logs, resolved arguments, checkpoint sidecars and hashes,
invalid-game counts, overflow counters, training SPS, peak GPU memory, and all
fixed-seed evaluations.
