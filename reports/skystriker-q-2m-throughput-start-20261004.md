# SkyStriker frozen Q 2M continuation: launch and throughput

Status: formal training started, not yet complete. Verified at home 2026-10-04 02:06:52: cumulative Q step 532480, 33 finite updates, no menu mismatches, frozen actor update count zero. Full Q parameters and Adam state restored exactly from the 262144-step warm-Q checkpoint. This is observation-only Candidate-Q, not centralized VRPO or policy improvement.

## Bounded throughput sweep

Each pilot starts from the same Q/optimizer checkpoint and seed, runs 16 updates, excludes the first four updates, and measures end-to-end wall time between subsequent updates. Horizon 64, minibatch 128, Q LR 1e-4, gamma 1, lambda .95, four environment threads per actor; no concurrency or precision changes.

| Actors x environments per actor | Sustained SPS | Median interval SPS |
| --- | ---: | ---: |
| 2 x 16 | 702.03 | 716.12 |
| 2 x 32 | 970.66 | 962.83 |
| 4 x 16 | 812.09 | 841.44 |
| **4 x 32** | **1172.44** | **1176.44** |
| 6 x 16 | 864.00 | 862.24 |
| 6 x 32 | 1163.74 | 1165.42 |

4x32 is the highest measured among these six configurations, not a proven global maximum. Its lead over 6x32 is only 0.7%; fewer actors at similar throughput favors 4x32. Improvement over 2x16 is about 67%. Larger batches preserve minibatch size and updates per transition, but change rollout composition; this is not a pure training-length-only ablation.

The first three qualified measurements were retained from v2. v2 4x32 completed updates and saved weights but crashed at interpreter teardown, so its timing was disqualified. Code inspection found an extra shadow daemon rollout and missing environment close/thread join. Source 65443f7 fixes normal shutdown; retested 4x32 and both six-actor pilots exited with rc=0 and shadow_workers_closed=true. The precise native crash frame was not captured; this does not establish that every native failure is solved. Original failed logs remain preserved. An earlier run-name parsing failure occurred before training and is also preserved.

## Formal run

- Root: `/home/ygo/ygo-agent/training-runs/q2m-sweep-v3-20261004`.
- Formal directory: `formal-2m`; launcher PID 134758, initial trainer PID 142588.
- Active training source: `65443f76ba39a738aac79db924ba14defcc16cf2`, isolated at `/home/ygo/ygo-agent/training-runs/q2m-source-65443f7`.
- Start Q step 262144; additional 1744896 transitions; target cumulative **2007040**, rounded to full batches.
- 4 actors x 32 envs, 4 env threads/actor, rollout 64, batch 8192, 64 minibatches x 128, seed 81042002. Save every 30 updates and at terminal.
- Original warm-Q checkpoint: `/home/ygo/ygo-agent/training-runs/skystriker-warm-q-262k-e66a427-20261004/checkpoints/41032032_step_000000262144.flax_model.candidate_q`.
- Q SHA256: `92c45d476fa8c1b4593c3d9382dffea51b2e53fc7082cdd4cf957d334ed81895`.
- Frozen actor checkpoint SHA256: `cde8e3f6a06000319cbe3fe59f6112b2ffa37f5ece03b2f05b1ef06f16a03649`.
- Native SHA256: `a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`.
- Pilot-trained weights are not used. `q-resume-proof.json` confirms actor exact, critic parameters exact, optimizer restored. Explicit source-only transition from e66a427 is recorded; other resume metadata checks remain enforced.
- `shadow-q-metrics.jsonl` records cumulative Q steps; frozen-actor proof collection steps are local new steps and require adding 262144. Do not confuse actor-loop progress with consumed Q updates.

Latest console SPS is approximately 1280, but the matched pilot result is the basis of selection; final sustained throughput awaits completion. At the 532480-step check about twenty minutes remain, subject to saves and runtime variation.

## Completion requirements

Require normal exit, completed marker, cumulative endpoint, full-envelope 1090 finite arrays, sidecar/hash agreement and frozen actor identity. Compare selected 262k/0.5M/1M/2M Q checkpoints on the same live observations, separating reused development seeds from fresh held-out seeds. Five old diagnostic roots are not fitting data. Do not promote Q to policy, start 5M PPO, or claim improved critical-action ranking from training loss alone. OpenSpec task 2.7 remains open.

Heartbeat `monitor-home-q-2m-and-throughput` monitors progress quietly and performs endpoint verification and comparison. Local/GitHub/home code is synchronized; H200 is unavailable and pending synchronization. BO3 and user TRAINING_PLAN edits are preserved.
