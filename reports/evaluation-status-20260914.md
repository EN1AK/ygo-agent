# Evaluation and replay status (2026-09-14)

## WindBot adapter

- A supervised batch evaluator now treats timeout, native abort, protocol error,
  and orphan cleanup failure as invalid attempts rather than wins.
- The deployed legacy protocol adapter translates `SELECT_CHAIN` and
  `CONFIRM_CARDS` for the pinned WindBot reader.
- The latest elfnote continuation was evaluated for 80 attempts across Blue-Eyes,
  Salamangreat, Swordsoul, and Labrynth with alternating seats: 77 valid games,
  76 wins, 1 loss, and 3 invalid attempts (98.70% of valid games won). All
  supervised process groups were cleaned successfully.

## Checkpoint comparison

The final checkpoint from
`h200-elfnote-stage3-continue-10m-20260912T180644Z` was compared with the Stage 3
mixed checkpoint on the same fixed-seed greedy matrix. Stage 3 scored 1272/1280
(99.38%); the continuation scored 1265/1280 (98.83%). A separate balanced-seat
direct battle produced 140/256 wins (54.69%) for the continuation against Stage
3. These results are not interchangeable: the first uses weak in-process greedy
opponents, while the second is checkpoint-versus-checkpoint self-play.

## Replay correction

Self-play replay generation initially terminated as `std::bad_function_call`.
The triggering setup omitted the required replay output directory, and the
native wrapper obscured the underlying setup failure. The corrected path creates
the directory before environment reset, permits recording without verbose debug
formatting, handles the unimplemented sort-card prompt as an automatic response,
and emits explicit diagnostics for missing action callbacks.

Two opposite-seat `.yrp` smoke replays completed after rebuilding the Python 3.10
module in an Ubuntu 22.04 container. Future replay bundles must include the `.yrp`,
engine log, JSONL decision trace (logits, probabilities, `V(s)`, selected action,
model/checkpoint identity), and a Chinese natural-language account.

## Limitations

- Policy logits and critic `V(s)` are available. The PPO model has no per-action
  Q head, so neither logits nor `V(s)` should be reported as Q-values.
- The 80-attempt WindBot run is a useful smoke matrix, not the 256-valid-games per
  matchup promotion gate.
- The continuation regressed slightly against greedy opponents despite winning
  the direct Stage 3 comparison; it should not replace Stage 3 solely on these
  results.
