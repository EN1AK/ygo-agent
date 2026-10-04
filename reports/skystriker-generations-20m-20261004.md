# SkyStriker sequential 20M continuation

User authorization 2026-10-04: continue existing PPO until improvement against
the immediately preceding generation levels off; evaluate every 20M. This
supersedes the proposed centralized-Q experiment. No Q, belief, or search added.

## Frozen execution plan

- Host: home WSL; existing model-serving processes remain untouched.
- Initial parent: 186,277,888; SHA256
  `dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721`.
- Root: `/home/ygo/ygo-agent/training-runs/skystriker-generations-20m-v2-20261004`.
- Launcher: `scripts/run_skystriker_generations.py`; exclusive new run directory.
- Immutable training release: `skystriker-place-forward-505b855-20261003`;
  evaluation source: `q2m-source-65443f7`. Launcher synchronization does not
  overwrite either runtime tree. Native SHA256
  `a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`.
- Keep SkyStriker decks, legal actor observations, PPO/GAE, LR 3e-5, four actors
  x 32 environments, 8192 transition batches, two self/one frozen 100M history/
  one bot training actors. The rolling evaluation opponent does not change this
  training mixture. Q, UPGO, search and belief remain off.
- Each generation adds 20,004,864 transitions (2442 batches); saves every
  5,005,312 and at the endpoint. First endpoint 206,282,752.
- Existing 186M checkpoint contains actor weights, not Adam state. The initial
  generation initializes Adam once. In response to the user's concern about
  repeated resets, subsequent generations restore full TrainState (including
  Adam moments/counters) and learner RNG from an atomic sidecar. Each saved state
  is read back and checked for exact tree/array equality; resume verifies actor
  identity, hashes, finiteness and training configuration. Environment episodes,
  actor RNG and live recurrent rollouts restart at boundaries; this does not
  claim bitwise equivalence to uninterrupted data collection.
- The launcher derives a private trainer from frozen source SHA256
  `a26907a94d780884ac75e17a302c8bdf782f2bc57f99121db1396b0f995e65f0`,
  adding optimizer save/restore hooks and an unconditional config-only exit
  for the single-deck configuration. The original runtime and
  main training implementation are untouched. The generated helper and trainer
  hashes are recorded; changed source/configuration refuses optimizer resume.
- Config-only protocol/CUDA gate before training. Preserve finite metrics,
  checkpoint sidecars and hashes; train and evaluate serially. No extra test
  suite or discarded training smoke is added.

## Evaluation and stopping rule

Each endpoint faces its immediate parent: 206M vs 186M, 226M vs 206M, etc.
512 attempts = eight new seed blocks x 32 first environment episodes x both
seats. Raw argmax, 1000-step cap, no cycle intervention. Invalid games are
reported separately and stop the controller for investigation, never counted
as plateau evidence or silently replaced. Each process has a 30-minute limit.

Report actual win/loss/draw counts and win rate. For stopping, draws receive
half a point. Bootstrap 10,000 resamples of paired (seed, environment) deals,
keeping the two seats together. Clear gain requires score >=55% and lower95
>50%. After three consecutive screens without clear gain, run 1024 fresh
confirmation attempts against that generation's parent. Stop only if the
confirmation upper95 is below55%; otherwise reset patience and continue.
These are operational sequential stopping criteria, not multiple-testing-
adjusted proof of global convergence or improvement against every opponent.

Keep all generations; never deploy a checkpoint to the live serving process.
`schedule.json` contains current phase, exact child process, checkpoints and
results. An operator `STOP` file gracefully interrupts the managed child.
Failures preserve artifacts and halt rather than starting another generation.
No fixed elapsed-time cap was requested; plateau/failure/operator stop ends it.

## Status

Prepared; live launch and configuration acceptance are recorded in the remote
`schedule.json`. A preparation report alone does not establish training started.

The first launch (directory without `v2`) exposed a legacy config-only bug:
the flag returned only when a deck sampling manifest was supplied. Its config
process began collection and was stopped; all artifacts remain retained and no
resulting weights are promoted. V2 requires an explicit config-only completion
marker and absence of actor startup before launching the real first generation.

## Resident evaluation optimization

User requested keeping the two evaluated models resident across all seeds/seats.
`scripts/eval_generation_resident.py` loads both checkpoints once and keeps one
JIT prediction function for a full 512/1024-attempt matrix. Every seed/seat batch
still creates and closes its own environment and resets both recurrent states.
Seed derivation, batch size32, first-episode collector, argmax, two-player state
updates, invalid handling and scoring remain unchanged.

The already running scheduler is not restarted. An explicitly scoped dispatcher
is installed only in its isolated evaluation source, preserving the original
script byte-for-byte. Its first request writes the matrix's existing per-batch
JSON files; later scheduled requests verify model/config/result hashes and read
those results without loading JAX. Other output directories use the old entry.
The original recorded evaluator hash is superseded only for this scoped route;
`resident-eval-deployment.json` records the original, dispatcher and helper hashes.

The first matrix call retains the caller's 1800-second process limit. If it is
exceeded or any batch is invalid, evidence is retained and the existing scheduler
halts; incomplete cached matrices are never accepted or silently regenerated.
Per-batch timings and tracing counts appear in `resident-progress.json`, with
final matrix timing and file hashes in `resident-matrix.json`. No speedup or
bitwise trajectory equivalence is claimed before the first scheduled run. No
additional GPU benchmark or test suite was run alongside active training.
