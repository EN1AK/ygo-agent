# Frozen SkyStriker Q 2M: training and paired validation complete

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

## Follow-up validation completed

Independent evaluation started at `/home/ygo/ygo-agent/training-runs/skystriker-q2m-validation-paired-20261004`, initial PID147231. Launcher log is that path plus `.launcher.log`. Script: `training-runs/evaluate-q2m-20261004.py`, based on the retained frozen-Q evaluation with assertion-checked transformations.

Four checkpoints (262144/507904/999424/2007040) score every identical live observation, while the unchanged stochastic actor alone selects actions. Each seed uses32 first episodes, both seats represented by self-play. Development seeds61042001/61042002 are reported separately from fresh seeds91042001/91042002. Invalid/capped games remain separate, without replacements. Paired uncertainty is clustered by game. Five retained loss fixtures are diagnostic-only, not fitting data or unbiased true expected-Q labels.

Startup diagnostics already ran: the final Q gives negative alternative-minus-original gaps on both previously win-versus-loss continuation fixtures (step189: -0.00438; step190: -0.01962). This does **not** show improved critical-action ordering; two roots from one duel are insufficient for a general conclusion. Full return calibration and fresh-seed comparisons are still pending. Task2.7 is complete; task2.4 and Q actor-promotion gates remain open. Do not start Q-boost/PPO/5M automatically.

All128 first episodes completed naturally, zero invalids, 45,881 decisions. Unique(seed,env) count128 verified; every decision has all four Q menus. Neither Q selected actions. On the64 fresh-test games:

| Q steps | Chosen-action return RMSE (lower better) | Median menu spread |
| --- | ---: | ---: |
| 262144 | 0.909624 | 0.025997 |
| 507904 | 0.920699 | 0.017343 |
| 999424 | 0.924240 | 0.011893 |
| 2007040 | 0.944694 | 0.008115 |

Game-weighted paired MSE difference, 2M minus262k: +0.095259, game-bootstrap95%CI[+0.057921,+0.131084]. Development seeds independently show the same direction (RMSE0.910395→0.935413). These intervals condition on the two test seed blocks and one trained seed, not population-wide certainty. More fitting under this configuration did not fix Q quality; the later Q is worse on sampled-return calibration, and action contrasts are smaller. This is evidence against simply extending this setup, not a definitive diagnosis of target, representation, exploration, or optimizer causality. Rollout topology also changed in the continuation.

Combined final-Q RMSE0.940123 is better than zero predictor1.0, but worse than actor V0.902290. Menu-mean substitution is almost as accurate as chosen-action Q (fresh-test2M0.945626 vs0.944694), so the report does not support useful action-specific discrimination. Median single final-Q batch latency4.408ms; total four-model evaluation145.88sec/314.51decisionSPS. This latency measures the single endpoint call, not total four-Q inference. Only single-deck self-play calibration is tested; WindBot distribution transfer and unbiased action ranking remain unproven. Do not promote the Q into actor training. Task2.4 remains open.

Remote evidence directory above includes all raw decisions, games, protocol and reports. SHA256: four-way-comparison.json `146b6d3bec430f12b29ecd92bb7b507f967722a1f479ae7c5eb1ff77d47f5eb1`; decisions.jsonl `04512c3dc2a34e42b8c6423fd8c4a94f1ae7fa3e43f0cacc45659382913d61ef`; report.json `e8daab91cc5538771ecda9b117d1a4f2b4ff1a053e11f7746ecfb0a18e5a29cd`.

The heartbeat remains active only until local model/evidence delivery is verified (large endpoint download still in progress at report time). Source/docs are synchronized local/GitHub/home; H200 remains offline, pending sync. No user BO3 or TRAINING_PLAN changes were included.
