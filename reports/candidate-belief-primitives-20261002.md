# Candidate rollout / belief implementation milestone

Search source `c62fcfd42484ef0c16d861bc96ec10ceed0e2b23`, plus follow-up
documentation and validator changes. This is **replay-oracle diagnostic** level,
not native snapshot search, a trained belief model, or fair-play improvement.
OpenSpec `add-mcts-policy-search` remains active and incomplete.

## Validation evidence

- Local: candidate pairing, finite KL update, deterministic ties, subset mass,
  budget/chain fallback, resource cleanup, player viewpoint, terminal ±1 scale,
  invalid reward rejection, particle ordering, lossless old PUCT JSON migration,
  eligibility and oracle rejection.
- Local: known-deck information-set constraints, multiset sampling, known-private
  card preservation, API-order leakage guard, whole-duel data splits, strict
  data fields, padding/target validation and hand-calculated calibration metrics.
- H200 CPU-only isolated stage
  `training-runs/belief-pipeline-cpu-test-20261002`: four JAX tests passed in
  21.449 seconds. The head's causal masks, finite gradients, deterministic
  constrained sampling and padded rows passed. A **synthetic** three-update
  offline run saved a checksum-verified checkpoint and held-out report without
  promoting fair play. No GPU training data or production weights were modified.
- Home WSL CPU isolated stage `/home/ygo/search-c62fcfd-20261002`: real engine,
  53,002,240-step checkpoint
  `efaa6448201e49778b1604f56efa294bd05c1518861aed30bdf55051078c6fda`,
  native module
  `5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`,
  seed 71900845, structured-lite-v1/full, elfnote mirror, 128 actions, history
  and public events 32. Captured opening, chain, staged selection and 40-action
  prefix fixtures. Every restored digest/logit vector matched exactly; repeated
  candidate runs matched rollouts, updated distributions and selected actions.

## Actual search behavior

Two candidate actions, two paired rollouts per action, depth 8, wall cap 120s:

| Root | Outcome | First diagnostic runtime |
| --- | --- | --- |
| Opening | Raw fallback: unresolved chain at depth limit | 0.139s |
| Chain | Raw fallback: unresolved chain at depth limit | 0.150s |
| Staged selection | Complete comparison | 0.713s |
| 40-action prefix | Raw fallback: unresolved chain at depth limit | 0.959s |

These are CPU smoke timings, not a throughput benchmark. Three fallbacks are
evidence that the safety gate runs, **not successful tactical searches**.
The staged root digest is
`7b6e78346aba1a58ecd570e1b2f760037b26d2e9b2937f8cea1403f213745943`.
The normal candidate CLI independently reproduced that root and finished in
0.696s; raw and selected action were both 1. It recorded checkpoint/native/input
hashes and `fair_play_promoted=false`.

The unchanged PUCT CLI also ran on the same staged root: four simulations,
11 expanded nodes, root visits 1/2/1 for actions 0/1/2. This checks CLI/engine
compatibility; it is **not a matched-inference-cost comparison** and does not
show either method is stronger. The earlier historical policy-cycle PUCT audit
is frozen separately in `tests/fixtures/search/legacy-cycle-puct.json` and tested
for lossless envelope migration; it has not been silently reinterpreted under
this newer native runtime.

Local detailed artifacts:
`training-runs/search-c62fcfd-home-results/results/`,
`training-runs/search-c62fcfd-home-results/fixture-test.log`,
and `training-runs/search-c62fcfd-cli-results.tar.gz`.
These diagnostic fixtures are not full duel replay deliverables.

## Remaining gates

Real self-play label collection, a trained/held-out-calibrated belief model,
native arena ownership/rollback, applying hidden-state particles before model
queries, external unknown-deck priors, fair-play paired matches, and distillation
are still unfinished. Keep the current PPO continuation isolated. The new
`scripts/validate_candidate_replay.py` packages the repeatable smoke procedure;
inspect its missing-fixture and fallback report rather than treating a finished
process as complete search promotion.
