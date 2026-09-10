# K9VS 10M-step baseline report (2026-09-10)

## Configuration

- Git revision: `e188227`
- GPU: NVIDIA H200 (143771 MiB), driver 580.159.04
- Python: 3.10.12
- Language: Chinese
- Deck: K9VS
- Card vocabulary: 13,492 entries, trainable learned embedding
- External pretrained embedding: disabled
- Maximum options: 24
- Parallel environments / environment threads: 128 / 112
- Rollout steps: 128
- Minibatches: 64 (256 samples per minibatch)
- Update epochs: 1
- Learning rate: `1e-4`, linearly annealed
- Advantage normalization: enabled
- Precision: FP32
- Seed: 7
- Requested timesteps: 10,000,000
- Executed timesteps: 9,994,240 (610 complete updates)

## Training health and performance

- Sustained throughput near completion: 9,471 SPS
- Approximate wall time including periodic evaluation: 19m 14s
- Optimizer steps: 39,040
- `optimizer_finite=True` throughout sampled logs
- Final `notfinite_count=0`, `total_notfinite=0`
- Final policy loss: -0.00001035
- Final value loss: 0.0381571
- Final entropy: 0.396452
- Final approximate KL: 0.00000592

All 13 retained checkpoints have unique exact-step filenames and unique SHA256
values, confirming that parameters changed during training.

## Periodic 256-game evaluation

| Update | Steps | Win rate | Mean return |
|---:|---:|---:|---:|
| 100 | 1,638,400 | 96.48% | 3.0476 |
| 200 | 3,276,800 | 96.48% | 3.2217 |
| 300 | 4,915,200 | 98.44% | 3.2005 |
| 400 | 6,553,600 | 98.44% | 4.2700 |
| 500 | 8,192,000 | **98.83%** | **4.7216** |
| 600 | 9,830,400 | 98.44% | 4.6569 |

## Independent evaluation

Both selected checkpoints were independently evaluated for 256 games against
the fixed greedy bot.

| Checkpoint | Win rate | Approx. 95% Wilson CI | Mean reward | Mean length | SPS |
|---|---:|---:|---:|---:|---:|
| Step 8,192,000 (periodic best) | **98.44%** | 96.05%-99.39% | 4.2501 | 97.92 | 11,253 |
| Step 9,994,240 (final) | 97.27% | 94.46%-98.66% | 4.2633 | 102.21 | 12,253 |

The confidence intervals overlap. The 8,192,000-step checkpoint is retained as
the baseline's selected checkpoint because it also had the best periodic result.

## Selected artifacts

Remote run directory:

`/root/ygo-agent-gpu-20260910/training-runs/h200-k9vs-baseline-10m-20260910T085805Z`

Selected checkpoint:

`checkpoints/1789030687_step_000008192000.flax_model`

Selected checkpoint SHA256:

`dc6a7bc7674cff5754bbcb1a4eda048cf7012c6a0e59f8e82be8fa04b5798bd3`

Final checkpoint:

`checkpoints/1789030687_step_000009994240.flax_model`

Final checkpoint SHA256:

`3ca917eb2aeb8ee1e6c12b9f4d2b749bac6dc426a0b832b05b7146f3f665574b`

## Conclusion

Stage 1 succeeded: training remained numerically finite, parameters changed,
exact-step checkpoints were preserved, and the policy reached a strong result
against the fixed greedy bot. The next planned work is Stage 2: broaden the
evaluation suite to include the initial random model and historical checkpoints.
