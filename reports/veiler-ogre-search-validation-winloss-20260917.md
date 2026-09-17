# Veiler/Ogre search validation — corrected win/loss scale (2026-09-17)

## Integrity gates

- Decision: `battle-62-seat1:14`; exact hidden state (`K=1`).
- Reward mode: terminal win/loss, `greedy_reward=False`, win `+1`, loss `-1`.
- Runtime aborts on any terminal reward outside `±1`.
- Restored legal actions: 3; root policy maximum drift: `0.000637`.
- Root critic value: `0.998793`.
- Budget unchanged from the superseded v2 run.

## Bounded enumeration plus terminal rollout

The root action and next legal action were fully enumerated into seven leaves.
Each leaf used 16 common random rollout seeds, for 112 complete continuations.

| Root action | Leaves | Samples | Mean | Implied wins | SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Effect Veiler | 2 | 32 | -0.8125 | 3/32 | 0.1047 |
| Ghost Ogre | 2 | 32 | -0.8125 | 3/32 | 0.1047 |
| Cancel | 3 | 48 | -0.8333 | 4/48 | 0.0806 |

For the two equally sized Veiler/Ogre frontiers, paired Ghost Ogre minus Veiler
was `0.000`, paired SE `0.127`, approximate 95% interval
`[-0.249, 0.249]`. Both actions have winning continuations, but downstream
policy play makes terminal evaluation unable to distinguish them at this
budget and depth.

## Leaf-value PUCT

PUCT used 128 simulations, maximum depth 12, and `c_puct=1.5`.

| Root action | Prior | Visits | Visit target | Leaf Q |
| --- | ---: | ---: | ---: | ---: |
| Effect Veiler | 0.554 | 8 | 0.0625 | -0.186 |
| Ghost Ogre | 0.391 | 119 | 0.9297 | 0.841 |
| Cancel | 0.055 | 1 | 0.0078 | -0.194 |

PUCT strongly reverses the raw policy toward Ghost Ogre. Its numbers are now on
the same `±1` scale as terminal results. The result remains a candidate-ranking
signal, not sufficient training supervision: terminal continuation does not
confirm a Ghost Ogre advantage, and the search uses the exact hidden state
rather than belief particles.

## Conclusion

The corrected result is narrower than the original claim:

- The bounded terminal method finds wins after both actions but cannot rank
  Ghost Ogre above Veiler.
- Leaf-value PUCT strongly ranks Ghost Ogre above Veiler and agrees with the
  immediate rule-level causal analysis.
- This case is suitable for a human-reviewed tactical preference candidate,
  but not for automatically emitting a high-confidence soft target.

Machine-readable output:

- Local: `dist/veiler-ogre-search-formal-v3-winloss-20260917.json`
- CPU server:
  `/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-search-20260917/formal-v3-winloss.json`
