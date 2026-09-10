# K9VS Stage 2 evaluation report (2026-09-10)

## Scope

This evaluation compares the selected 8,192,000-step checkpoint and the final
9,994,240-step checkpoint against the fixed greedy bot, a deterministic random
initial policy, and historical checkpoints from the same training run.

All primary matchups contain at least 256 games. Agent-vs-agent games split
starting positions evenly. The final-vs-selected comparison was expanded to
2,048 games and profiled for GPU performance.

## Fixed greedy bot

| Checkpoint | Games | Win rate | Approx. 95% Wilson CI | Mean reward | Mean length |
|---|---:|---:|---:|---:|---:|
| Step 8,192,000 | 256 | 98.44% | 96.05%-99.39% | 4.2501 | 97.92 |
| Step 9,994,240 | 256 | 97.27% | 94.46%-98.66% | 4.2633 | 102.21 |

The confidence intervals overlap. Greedy-bot performance is close to saturation
and is not sufficient by itself to rank the two late checkpoints.

## Random initial policy

| Checkpoint | Games | Win rate | Approx. 95% Wilson CI |
|---|---:|---:|---:|
| Step 8,192,000 | 256 | 100.00% | 98.52%-100.00% |
| Step 9,994,240 | 256 | 100.00% | 98.52%-100.00% |

Both trained policies won from both starting positions in every game. This
confirms that training acquired behavior far beyond the initial random policy.

## Selected checkpoint against training history

Agent 1 is the selected 8,192,000-step checkpoint. Each row contains 256 games.

| Opponent step | Agent 1 win rate | Mean reward | Mean length | Evaluation SPS |
|---:|---:|---:|---:|---:|
| 819,200 | 88.28% | 3.7319 | 154.35 | 16,050 |
| 1,638,400 | 82.42% | 3.1898 | 157.41 | 15,639 |
| 3,276,800 | 80.08% | 2.6358 | 162.92 | 15,757 |
| 4,915,200 | 70.31% | 1.8893 | 178.55 | 15,416 |
| 6,553,600 | 58.20% | 0.7630 | 191.29 | 15,931 |
| 7,372,800 | 54.69% | 0.1871 | 180.48 | 16,367 |
| 9,994,240 | 41.02% | -0.9512 | 173.65 | 15,866 |

The selected checkpoint consistently beats earlier policies, but the final
checkpoint beats it. Thus the checkpoint with the best greedy-bot score is not
the strongest policy in direct play.

## Final checkpoint against training history

Agent 1 is the final 9,994,240-step checkpoint. Each row contains 256 games.

| Opponent step | Agent 1 win rate | Mean reward | Mean length | Evaluation SPS |
|---:|---:|---:|---:|---:|
| 819,200 | 89.84% | 4.0643 | 150.97 | 15,861 |
| 1,638,400 | 87.11% | 3.6494 | 156.34 | 15,970 |
| 3,276,800 | 77.34% | 2.7326 | 160.53 | 16,091 |
| 4,915,200 | 67.97% | 1.7364 | 173.08 | 16,454 |
| 6,553,600 | 60.55% | 0.9311 | 173.19 | 16,436 |
| 7,372,800 | 56.64% | 0.6303 | 176.86 | 16,341 |

These results show broad improvement over the sampled history rather than a
regression that only exploits one particular checkpoint.

## High-confidence final-vs-selected comparison

The final checkpoint (Agent 1) played 2,048 games against the selected
8,192,000-step checkpoint:

- Final checkpoint win rate: **53.52%**
- Approximate 95% Wilson confidence interval: **51.35%-55.67%**
- Mean reward: 0.3375
- Mean episode length: 178.32
- Evaluation throughput: 26,590 SPS
- Wall time: 39.51s

The interval excludes 50%, so the final checkpoint is stronger in this direct
matchup despite its slightly lower greedy-bot win rate.

## GPU evaluation profile

The 2,048-game matchup used 2,048 environments and 112 environment threads.
Across 74 one-second samples:

- Average / peak GPU utilization: 16.3% / 43.0%
- Average / peak GPU memory: 102,914 MiB / 108,833 MiB
- Average / peak GPU power: 211.2 W / 312.7 W
- Model time: 12.19s
- Environment time: 26.90s

Evaluation remains primarily environment-bound. Increasing GPU compute alone is
unlikely to improve throughput without changing environment execution.

## Conclusion

Stage 2 succeeded:

- Both trained checkpoints decisively outperform the initial random policy.
- Both are near saturation against the fixed greedy bot.
- Later checkpoints broadly outperform earlier checkpoints.
- The final model is statistically stronger than the previous selected model in
  a 2,048-game direct matchup.

For subsequent work, use the final 9,994,240-step checkpoint as the principal
baseline while retaining the 8,192,000-step checkpoint as a distinct historical
opponent. The results support proceeding to Stage 3, a mixed historical opponent
pool, because greedy-bot evaluation alone no longer reliably ranks late models.

## Artifacts

Remote run directory:

`/root/ygo-agent-gpu-20260910/training-runs/h200-k9vs-baseline-10m-20260910T085805Z`

Final checkpoint SHA256:

`3ca917eb2aeb8ee1e6c12b9f4d2b749bac6dc426a0b832b05b7146f3f665574b`

Selected historical checkpoint SHA256:

`dc6a7bc7674cff5754bbcb1a4eda048cf7012c6a0e59f8e82be8fa04b5798bd3`
