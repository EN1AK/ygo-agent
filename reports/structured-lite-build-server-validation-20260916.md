# Structured-lite build-server validation (2026-09-16)

## Scope

The GPU training server was unavailable. Validation used the internet-connected
build server in an isolated source copy and did not replace a deployed runtime.
GPU throughput, memory, training, migration, and promotion gates remain open.

## Native environment

- Build image: `ygo-build-cache:replay-fix`, portable release configuration.
- Native module SHA-256:
  `a0cd7d2d0a2fb41a47fb39121b095941086548e3c7ccd80b2dc5d08a3115efda`.
- Legacy manifest: 9,559 bytes; SHA-256
  `9d0594da37102f23a7218059c8f2d346cf56d8014d1e50a9af80f90fdb7a0227`.
- Structured-lite manifest: 26,095 bytes; SHA-256
  `dddf2966f01a00e8570c5bb338ac67e787cf85289fc32fd92bcc58362cddef9f`.
- Fixed-seed real-engine smoke duels repeated byte-identically for both schemas.
- Structured-lite checked 1,877 hidden card rows: zero semantic leakage and zero
  unsafe public-event references.
- Legal-action capacity 2 produced an explicit overflow diagnostic at reset.
- The observed public-event set covered activation, targeting, destruction,
  movement, draw, search, summon, special summon, chain solving, chain solved,
  and chain end. Negation requires a dedicated fixture before event coverage is
  considered complete.

Machine-readable details are in
`reports/structured-lite-environment-validation-20260916.json`.

## Model and checkpoint boundary

- CPU JAX diagnostics passed for `full` and `relationship-only` variants.
- Maximum-capacity and padding-only forward passes were finite.
- Output shape was `[2, 24]` policy logits and `[2, 1]` scalar `V(s)`.
- Reversed action order restored with maximum absolute logit error `0.0`.
- Metadata validation accepted matching legacy and Structured-lite loads and
  rejected both direct cross-schema directions before execution.

Machine-readable model details are in
`reports/structured-lite-model-validation-20260916.json`.

## Remaining server gates

The build server has no copy of the frozen Stage 3 checkpoint. Checkpoint
migration and warm-start validation therefore wait for either a checkpoint
transfer or GPU-server recovery. All GPU-specific tasks remain unverified.
