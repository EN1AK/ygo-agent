# PUCT evaluator and prompt-boundary incident (2026-09-17)

## Problems

The first PUCT diagnostic had two design errors:

1. It selected the checkpoint by acting player, mixing the Stage 3 and K9VS
   mixed critics in one search tree. Sign inversion does not calibrate two
   independently trained value functions.
2. It allowed `select_chain` response prompts to become scored leaves. In the
   Veiler/Ogre case, value changed between roughly `-0.2` and `+1.0` as the
   response prompt alternated, before Ghost Ogre's chain had resolved.

The legacy observation also exposes a zero-filled Structured-lite `selection_`
placeholder. Prompt detection initially read that zero instead of falling back
to the legacy `actions_[...,3]` message feature; this was caught because audit
rows reported `unknown-0` and was corrected before final validation.

## Fixes

- PUCT now uses the root player's checkpoint for policy and value throughout
  the tree, with independent recurrent state for each player view.
- Terminal rollouts may still use the recorded per-player policies; their
  evaluator choice is recorded separately from PUCT.
- `select_chain` nodes are expanded but cannot terminate a simulation for leaf
  backup. Search continues to a non-chain prompt, terminal state, or safety
  depth limit.
- Audit rows record prompt ID/name and `resolved_chain_boundary`.
- Legacy prompt detection falls back from zero-filled `selection_` to the
  nonzero legal-action message feature.

## Corrected case validation

With 128 simulations, depth 32, the single Stage 3 evaluator, and resolved-chain
leaf boundaries:

| Root action | Visits | Visit share | Leaf Q |
| --- | ---: | ---: | ---: |
| Effect Veiler | 6 | 4.69% | -0.581 |
| Ghost Ogre | 122 | 95.31% | +0.911 |
| Cancel | 0 | 0% | 0 |

All 128 leaves were non-chain prompts and all passed the resolved-boundary
gate. No leaf was terminal. Ghost Ogre leaves mostly reached player-1
`select_idlecmd`, `select_place`, or `select_card` states valued `+0.74` to
`+1.02`. Veiler leaves remained player-0 strategic prompts valued `+0.44` to
`+0.66` for player 0, or `-0.44` to `-0.66` from the root perspective.

This corrected result is consistent with the engine trace: Ghost Ogre destroys
the continuous spell and the opponent soon ends the turn, whereas the Veiler
route allows continued development. It is substantially more interpretable
than the original prompt-leaf result, but it remains diagnostic because no
simulation reached terminal and terminal rollout did not separate the actions.

## Prompt-bias follow-up

Use the matched-state, stratified calibration, and controlled-regression
protocol in `docs/counterfactual-selfplay.md`. The decisive question is whether
prompt type or acting player predicts held-out value residual after controlling
for the public board and eventual result. Do not infer bias solely from raw
values across unmatched states.

Corrected artifact:

- Local: `dist/veiler-ogre-puct-resolved-boundary-v2-20260917.json`
- CPU server:
  `/root/ygo-agent-gpu-20260910/training-runs/veiler-ogre-search-20260917/puct-resolved-boundary-v2.json`
