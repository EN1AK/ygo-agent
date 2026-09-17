# Veiler/Ogre counterfactual search validation (2026-09-17)

> Superseded: this first run accidentally inherited the native environment's
> `greedy_reward=True` default, while the Stage 3 critic was trained with
> `greedy_reward=False`. Its `±2/±4/±8` terminal returns and PUCT backups mix
> incompatible reward scales. Retain this report only as an incident record;
> use the corrected report and `formal-v3-winloss.json` for conclusions.

## Setup

- Decision: `battle-62-seat1:14`, root player 1.
- Legal actions: 0 Effect Veiler, 1 Ghost Ogre, 2 cancel.
- Exact hidden state (`K=1`); no belief-particle claim.
- CPU execution in the isolated Python 3.10 build container.
- Stage 3 Elfnote checkpoint versus the recorded K9VS mixed checkpoint.
- Snapshot gate: 3 legal actions, root value `0.998793`, and maximum root
  policy-probability drift `0.000637` (tolerance `0.001`).

## Bounded enumeration plus terminal rollout

Every root action and every following legal action was enumerated: seven
frontier leaves in total, with 16 common random rollout seeds per leaf (112
complete continuations).

| Root action | Leaves | Samples | Mean return | SE | Range | Positive paths |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Veiler | 2 | 32 | -4.428 | 0.447 | -8 to 0.833 | 3 |
| Ghost Ogre | 2 | 32 | -4.294 | 0.534 | -8 to 2 | 3 |
| Cancel | 3 | 48 | -5.049 | 0.440 | -8 to 2 | 4 |

Ghost Ogre minus Veiler was `+0.134` with paired SE `0.552` and approximate
95% interval `[-0.949, 1.216]`. The enumeration did find winning Ghost Ogre
continuations (best return `+2`, prefix `[1, 1]`), but terminal returns still
do not separate Ogre and Veiler confidently because later policy errors
dominate the outcome.

## Leaf-value PUCT

PUCT used 128 simulations, maximum depth 12, and `c_puct=1.5`. It expanded 49
nodes and backed up the critic in the root player's perspective.

| Root action | Prior | Visits | Visit target | Leaf Q |
| --- | ---: | ---: | ---: | ---: |
| Veiler | 0.554 | 8 | 0.0625 | -0.186 |
| Ghost Ogre | 0.391 | 119 | 0.9297 | 0.841 |
| Cancel | 0.055 | 1 | 0.0078 | -0.194 |

The search reverses the raw policy and strongly selects Ghost Ogre. This is a
useful tactical signal for the motivating error, but not yet proof that the
result is globally correct: the root critic predicts almost `+1` while sampled
complete continuations are mostly strongly negative. The critic is therefore
miscalibrated in this region even though its local ranking is helpful.

## Decision

Keep both methods as offline diagnostics. Terminal traversal supplies the
lower-confidence global check and concrete winning paths; leaf-value PUCT is
the better candidate generator for this case. Do not yet use PUCT visits as a
mainline soft-policy target. First validate across more human-reviewed tactical
positions and add belief particles for hidden information. A practical pilot
can require agreement between causal engine effects, PUCT ranking, and a
non-negative terminal-rollout trend before emitting preference data.

Machine-readable output is stored locally at
`dist/veiler-ogre-search-formal-v2-20260917.json` and on the CPU server at
`/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-search-20260917/formal-v2.json`.
