# 100M baseline evaluation protocol (declared before endpoint evaluation)

This is a plan and collection-protocol correction, not a completed 100M result.
The active run is `multideck-fusion-canonical-58m-to100m-92f225c-20261002`,
target 100,003,840 cumulative steps. Freeze its actual endpoint and runtime
release before evaluating. Do not start GPU evaluation alongside training.

## Fixed conditions and collection

- Raw deterministic policy: no cycle intervention, search, belief or Q helper.
- Current schema `structured-lite-v1/full`, max options 128, history/public
  events 32, group references 8, max external decisions 1000. Preserve native
  SHA `5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`
  and fusion procedure SHA
  `c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322`.
- Natural finite outcomes only enter wins/losses/draws. Report **all attempts**,
  invalid reason counts and timeout rates alongside the valid-game win rate.
  Never silently retry an invalid seed until it wins or finishes.
- Every batch collects the first duel of each environment exactly once, not
  the fastest N completions. Pair by `(deck, seed, environment_index)` and run
  the identical batch with the candidate in the other seat. The old half-seat
  allocation inside one batch is balanced but is not paired initial deals.
- Keep a completed seed schedule immutable. If invalid games prevent a
  declared valid-game target, report this shortfall and use a separately named,
  predeclared extension block; retain all failed attempts in totals.

## Evaluation matrix

1. Greedy: four mirror decks `elfnote`, `deck-73240bea64f5d27b`,
   `deck-9e683ac4c1b75e0a`, `deck-1aaccbd536d18e5a`; seeds 30092026 and
   30102026; both seats; **64 initial duels per cell**, 1024 attempts total,
   256 per deck. Explicitly pass `--first-episode-per-env` with envs=episodes.
2. Historical model matches on elfnote: Stage 3 10M, the previous full-feature
   40M, and the older lite40M once its capacity migration is explicitly verified.
   Use seeds 42001 and 42002, 64 environments each, separately pinned candidate
   seat 0 and seat 1: 256 attempts per opponent. Start with a separate two-seat
   compatibility/invalidity smoke before the fixed matrix.
3. WindBot: Kashtira, Labrynth, Tearlaments and Voiceless, pinned executor build
   `b0a2355f00bd59491add14ff09efa9914a3b47c6` and matching executor deck files.
   Start with the existing 10-attempt both-seat pressure test at seed 2026100102
   as a separately labelled smoke. If its protocol/lifecycle gate passes, use
   a **new** seed block starting 2026100200, alternating seats with paired seeds,
   256 attempts per executor. Measure actual throughput and keep interruptions
   or incomplete blocks visible; 10 games are not a promotion-strength result.

Model hashes for historical comparisons:

- Stage 3 10M: `4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`.
- Previous full 40M: `6fbf9f01a49e290f07473d430407141ba27166bda4fd6aea3e6e9ddf9b925d0d`.
- Older lite40M: `a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`;
  its 24-action metadata currently fails the 128-action compatibility gate.
  Do not relabel or skip this failure; retain original bytes and validate any
  explicit derived capacity migration before comparison.

The old greedy 507/512 report used fastest-completion collection. It is still
historical evidence, but not a paired control for this corrected protocol.
Rerun the full 40M checkpoint on the same greedy schedule/runtime when comparing
training progress. Different checkpoint training histories are not a causal
feature ablation; do not claim the feature design is proven superior from them.

## Replay and strategy review

Every delivered example must include `.yrp`, full engine `.log`, complete
model decision `.jsonl` (legal menu, raw logits, normalized probabilities, V,
chosen action, context, model hashes and terminal), and Chinese commentary.
Record each duel in an isolated directory and fail if any required machine
log is missing or inconsistent. Chinese commentary and semantic annotations
remain explicit manual work; do not mark null fields complete.

Preselect the first four paired seeds of each main WindBot block for strategy
review: 32 duels, 16 candidate-first opportunities. Preserve all corresponding
invalid attempts and missing annotations separately. Report:

- First-turn end-board presence of Crystal Wing / Baronne, and whether at
  least one of their relevant interruptions is actually available (face-up
  presence alone is insufficient). Denominator: fully observed candidate-first
  turns; also show the full 16 scheduled opportunities and all exclusions.
- Tuner/resource handling: annotate each reviewed game's first observed tuner
  access/material decision, available alternatives, selected action, and the
  subsequent expansion or resource loss. Do not infer a missed forced combo
  solely from a card name; require the recorded legal state/sequence.
- Interruption quality: enumerate the reviewed model's chain-response choices,
  identify target/effect and own/opponent controller, and classify defensible,
  clearly wasteful or uncertain with an evidence pointer. Report uncertain
  cases explicitly rather than guessing tactical correctness.
- Repeated-action/cycle detections, interventions (must be zero), max-step
  terminals and wall-timeouts. No-policy-progress cycles are failures even if
  an evaluator can eventually escape them.

These small annotated slices are diagnostic rates with uncertainty, not proof
of broad combo mastery. Store seeds, denominator/exclusion rules and trace
references so the same review can be applied to later matched VRPO arms.

## Gate before VRPO

Require endpoint hash/finite checks, immutable source/native/asset provenance,
the completed evaluation report with failures disclosed, and non-null strategy
review before considering baseline task 1.1 complete. Then start shadow-Q and
matched restarted GAE/Q experiments from the identical 100M actor. The live PPO
continuation and old 40M run are not the matched experimental control. Keep
observation-only Candidate-Q distinct from a learner-only centralized VRPO.
