# WindBot full-duel evaluation

## Implemented path

`scripts/eval.py` can launch IceYGO WindBot and select the native environment's
`windbot` play mode. The environment hosts the YGOPro lobby, starts the duel,
streams engine messages to WindBot, accepts `CTOS_RESPONSE` decisions, and feeds
them directly back to `ygopro-core`. Forced prompts resolved locally are not
forwarded, preventing stale responses from shifting the protocol queue.

The bridge currently supports Linux, one environment, and one complete duel per
process. Independent invocations should be used for multi-game evaluation.

## GPU-server acceptance run

- Date: 2026-09-11
- WindBot revision: `b0a2355f00bd59491add14ff09efa9914a3b47c6`
- Learner deck: K9VS
- Opponent: WindBot `OldSchool` / `AI_OldSchool.ydk`
- Learner checkpoint: K9VS 10M-step final checkpoint
- Learner position: second
- Replay recording: enabled

Observed result:

| Metric | Value |
|---|---:|
| Complete duels | 1 |
| Learner wins | 1 |
| Environment decisions | 68 |
| Reward | 4.0 |
| Wall time | 1.3595 s |
| Model inference time | 1.0972 s |
| Environment + WindBot time | 0.2614 s |
| Effective learner SPS | 50 |

This run is an integration and performance acceptance test, not a statistically
meaningful strength estimate. Report win rate over many independent seeded runs
and over multiple compatible WindBot decks before comparing checkpoints.

## Operational constraints

1. The WindBot deck must exist in WindBot and the same card list must be loaded
   as the native environment opponent deck.
2. Every opponent card and Lua script must exist in the native engine assets.
3. Use the exact card-code mapping used during training when loading a checkpoint.
4. WindBot should be used for evaluation, not high-throughput PPO rollout.
