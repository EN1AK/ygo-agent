# Veiler/Ogre PUCT leaf audit (2026-09-17)

## Question

Why did 128-simulation PUCT assign 119 visits and aggregate leaf Q `+0.841`
to Ghost Ogre when terminal traversal could not distinguish it from Effect
Veiler?

## Audit result

The result is dominated by prompt-state and acting-player imbalance, not by a
large set of independently evaluated post-destruction positions.

| Root action | Visits | Unique prefixes | Positive leaves | Negative leaves | Leaf player 1 | Leaf player 0 | Terminal leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Effect Veiler | 8 | 8 | 1 | 7 | 1 | 7 | 0 |
| Ghost Ogre | 119 | 39 | 108 | 11 | 108 | 11 | 0 |
| Cancel | 1 | 1 | 0 | 1 | 0 | 1 | 0 |

For Ghost Ogre, 91 visits stopped at the depth-12 limit but represented only
11 unique prefixes. The same depth-limited leaves were therefore backed up
repeatedly. No simulation reached a terminal state.

## Where the value changes

The first evaluations along representative paths were:

| Prefix | Meaning/status | Leaf player | Raw V | Root-perspective V |
| --- | --- | ---: | ---: | ---: |
| `[0]` | Veiler selected | 0 | +0.21 | -0.21 |
| `[1]` | Ghost Ogre selected | 0 | +0.21 | -0.21 |
| `[1,1]` | Ogre chained; opponent passed Maxx C response | 1 | +0.99 | +0.99 |
| `[1,1,1]` | root player passed the next chain prompt | 0 | +0.25 | -0.25 |
| `[1,1,1,1]` | another pass returns prompt to player 1 | 1 | +1.00 | +1.00 |

The engine log shows that at `[1,1]` Ghost Ogre has been activated and sent to
the graveyard, but the chain has not yet resolved and the continuous spell has
not yet been destroyed. The `+0.99` value therefore cannot be evidence that
the critic recognized the resolved tactical consequence. It appears as soon as
the optional-chain prompt returns to player 1.

The Veiler path also eventually reaches a player-1 leaf valued `+0.89`, but it
requires a longer prompt sequence and received far fewer visits. This makes the
aggregate root Q sensitive to how many protocol-level selection prompts occur
before the search depth limit.

## Conclusion

The model does not currently demonstrate that it knows Ghost Ogre is better in
this position. The engine-level causal analysis still establishes why Ogre is
the sensible action, but this PUCT implementation's `92.97%` visit target is an
artifact-prone diagnostic and must not be used for training.

Required fixes before retesting:

1. Define search depth in strategic state transitions or resolved-chain
   boundaries, not raw engine response prompts.
2. Do not repeatedly back up the same deterministic depth-limited leaf as if it
   were new evidence.
3. Add a value-consistency check across consecutive pass/chain-prompt states
   and verify the exact player perspective used by the critic.
4. Re-evaluate root actions at comparable resolved-chain boundaries.
5. Keep per-simulation leaf audit mandatory.

Machine-readable audit:

- Local: `dist/veiler-ogre-puct-leaf-audit-v1-20260917.json`
- CPU server:
  `/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-search-20260917/puct-leaf-audit-v1.json`
