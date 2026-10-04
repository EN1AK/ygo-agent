# SkyStriker sequential 20M continuation

User authorization 2026-10-04: continue existing PPO until improvement against
the immediately preceding generation levels off; evaluate every 20M. This
supersedes the proposed centralized-Q experiment. No Q, belief, or search added.

## Frozen execution plan

- Host: home WSL; existing model-serving processes remain untouched.
- Initial parent: 186,277,888; SHA256
  `dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721`.
- Root: `/home/ygo/ygo-agent/training-runs/skystriker-generations-20m-20261004`.
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
- Existing checkpoints contain actor weights, not Adam state. Every generation
  continues those weights with fresh Adam, consistent with the preceding
  eight-hour continuation. This is not uninterrupted optimizer continuation.
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
