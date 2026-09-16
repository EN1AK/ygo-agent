# Structured-lite implementation baseline (2026-09-15)

## Frozen legacy contract

- Observation schema: `legacy-v2` (historical checkpoints without a metadata
  sidecar are recognized as legacy-only).
- Default training capacities: 160 visible card slots (`max_cards=80` per
  player), 24 legal actions, 32 historical actions.
- Observation tensors: `cards_ [160,41]`, `global_ [23]`,
  `actions_ [24,12]`, `h_actions_ [32,14]`, and `mask_ [160,14]`, all `uint8`.
- Bytes per unbatched observation: 9,559. Field definitions are generated from
  `ygoai/rl/observation_schema.py` into `assets/observation-schema/legacy-v2.json`.
- Model: 2-layer 128-channel card encoder, per-player LSTM width 512, FiLM
  actor, scalar critic `V(s)`, FP32, trainable card-ID embedding.

## Frozen Stage 3 checkpoint

- Source run: `/root/ygo-agent-gpu-20260910/training-runs/h200-k9vs-mixed-10m-20260911T020646Z`.
- Source revision: `dd0894f`.
- Checkpoint: `checkpoints/1789092408_step_000009994240.flax_model`.
- Checkpoint SHA-256: `4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`.
- Training mix: 50% current self-play, 30% historical policies, 20% greedy;
  160 environments, 110 environment threads, rollout length 128, 80
  minibatches, one epoch, FP32.
- Sustained training throughput: approximately 6,743 steps/s.
- Fixed greedy evaluation: 256 games, 97.66% wins, mean reward 3.9608,
  mean length 92.09, 10,781 evaluation steps/s.

The older pure self-play H200 profile is retained as the directly measured
resource baseline because the Stage 3 run did not retain an equivalent GPU CSV:

- Hardware: NVIDIA H200, 143,771 MiB, driver 580.159.04; Python 3.10.12.
- Run: `/root/ygo-agent-gpu-20260910/training-runs/h200-k9vs-baseline-10m-20260910T085805Z`.
- Revision: `e188227`; sustained throughput 9,471 steps/s.
- Profile columns were GPU utilization, used memory MiB, and power W; 74
  samples recorded maxima of 43%, **108,833 MiB**, and 312.71 W.
- Selected checkpoint SHA-256:
  `dc6a7bc7674cff5754bbcb1a4eda048cf7012c6a0e59f8e82be8fa04b5798bd3`.

## Pinned inputs

- Chinese `cards.cdb` SHA-256:
  `5f13245de4e665450858f66ef2f736a3d2f48cc7f0036961373a96a650cb6797`.
- `scripts/code_list.txt` SHA-256:
  `f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`.
- `assets/deck/elfnote.ydk` SHA-256:
  `19b1c5aba1a5f00e0d29c9b9776c45de7ecafc6e7e25509a77c1e3ea5f133dcd`.
- Script asset commit from `assets/asset_manifest.json`:
  `5864b6f6e58d49738e0996b94e96655b51f420bf` (13,535 scripts).
- Fixed tactical decision: `battle-62-seat1:14`, effective seed `71900845`.
  Original policy probabilities are Veiler 0.554231, Ogre 0.390856, cancel
  0.054914. The full fixture is retained under
  `dist/veiler-ogre-snapshot-validation-20260915` and server directory
  `/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-forced-logs-20260915T1725Z`.
- Engine log SHA-256 values: Veiler
  `1d481410035300c84c3ced687620c6ea00e31d187c5a74263091f559cd4fdde4`;
  Ogre `01bd65bff8909356247409e0ea0d72bf6fa5f25d58d9f631f334fc1843f02e96`.

## Reproduction commands

The legacy training command is preserved verbatim in
`/root/ygo-agent-gpu-20260910/train-h200-k9vs-baseline-10m.sh`. Its effective
arguments are recorded in `reports/k9vs-baseline-10m-20260910.md`. Resource
maxima were verified with a small Python reduction over `gpu-profile.csv`; all
asset and checkpoint identities were verified with `sha256sum` on the GPU
server on 2026-09-15.

Structured-lite A/B/C evaluation will reuse the same H200, deck and code-list
assets, paired seeds, batch settings, and legacy Stage 3 checkpoint. Any
necessary batch reduction due to memory will be recorded rather than hidden.
