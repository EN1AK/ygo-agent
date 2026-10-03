# Frozen SkyStriker Q 2M: training complete, validation running

The user-authorized warm-Q continuation completed normally at cumulative **2,007,040** transitions on home at approximately 2026-10-04 02:25:51. It resumed Q parameters and Adam from 262,144, adding 1,744,896 transitions. No pilot weights were promoted. This is observation-only Candidate-Q with a frozen actor, not a trained Q-boosted policy or centralized VRPO.

## Verified training evidence

- Formal run: `/home/ygo/ygo-agent/training-runs/q2m-sweep-v3-20261004/formal-2m`.
- Active source: `65443f76ba39a738aac79db924ba14defcc16cf2`.
- All 213 Q updates finite; zero menu mismatches. Frozen actor proof exact and actor update count zero. Resume proof confirms exact Q parameters and restored optimizer.
- Completed marker exists, failed marker absent, launcher and trainer exited, `shadow_workers_closed=true` recorded.
- Endpoint: `checkpoints/81042002_step_000002007040.flax_model.candidate_q`.
- SHA256: `14dce4c466a6bbe0a69b5a567c7edfbd15fe41174d12f097fb4c5ebd0914571a`; independent reread agrees with sidecar; **1090 arrays all finite**.
- Configuration: 4 actors x 32 envs, 4 env threads/actor, horizon64, batch8192, minibatch128, Q LR1e-4, gamma1, lambda.95.
- Formal steady end-to-end **1263.69 SPS**, median interval 1303.87 SPS; full run 1403.88 seconds (23.4 minutes). Matched short sweep selected 4x32 at1172.44 SPS, versus6x32 at1163.74; see the launch report for all six configurations and disqualified failures. These are different measurement windows, not contradictory figures or a proven global optimum.
- Endpoint/log archive: `training-runs/q2m-endpoint-20261004.tar.gz`, 139571469 bytes; SHA256 `36cb1121120f6e8673afa110f92df604d43c7f3bdfaf382e05bd96e6b2a7c1da`.

## Follow-up validation, not yet complete

Independent evaluation started at `/home/ygo/ygo-agent/training-runs/skystriker-q2m-validation-paired-20261004`, initial PID147231. Launcher log is that path plus `.launcher.log`. Script: `training-runs/evaluate-q2m-20261004.py`, based on the retained frozen-Q evaluation with assertion-checked transformations.

Four checkpoints (262144/507904/999424/2007040) score every identical live observation, while the unchanged stochastic actor alone selects actions. Each seed uses32 first episodes, both seats represented by self-play. Development seeds61042001/61042002 are reported separately from fresh seeds91042001/91042002. Invalid/capped games remain separate, without replacements. Paired uncertainty is clustered by game. Five retained loss fixtures are diagnostic-only, not fitting data or unbiased true expected-Q labels.

Startup diagnostics already ran: the final Q gives negative alternative-minus-original gaps on both previously win-versus-loss continuation fixtures (step189: -0.00438; step190: -0.01962). This does **not** show improved critical-action ordering; two roots from one duel are insufficient for a general conclusion. Full return calibration and fresh-seed comparisons are still pending. Task2.7 is complete; task2.4 and Q actor-promotion gates remain open. Do not start Q-boost/PPO/5M automatically.

The heartbeat remains active until evaluation and evidence delivery. Source/docs are synchronized local/GitHub/home; H200 remains offline, pending sync. No user BO3 or TRAINING_PLAN changes were included.
