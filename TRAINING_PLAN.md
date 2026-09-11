# YGO Agent training plan

This is the default execution plan after the K9VS baseline. Follow the stages
in order and record configuration, asset revisions, checkpoints, metrics, and
conclusions for every experiment. Correct duel behavior and reproducible
evaluation take priority over increasing model size.

## Decisions and scope

- Keep the model's existing trainable card-ID embedding.
- Do not add external, card-text, or card-attribute pretrained embeddings.
- Use WindBot as a stronger, slower evaluation opponent, not as a rollout
  opponent for reinforcement-learning training.
- Use WindBot's supported `Normal` deck catalog to define the initial
  multi-deck matchup pool. The `.ydk` files define matchups; training opponents
  remain fast in-process policies or bots.
- Keep Chinese card assets as the default and pin every database, script,
  banlist, WindBot, and deck revision used for a result.

## Completed foundation

The following work remains the reference baseline rather than a future stage:

1. A stable 10,000,000-step K9VS baseline with finite optimizer metrics.
2. Greedy, random-initialized, and historical-checkpoint evaluation.
3. Mixed K9VS training with approximately 50% current self-play, 30%
   historical opponents, and 20% greedy opponents.

Preserve these checkpoints and reports as regression baselines. Do not compare
later multi-deck experiments only against their own training pool.

## Stage 4: Complete the WindBot evaluation adapter

Integrate [IceYGO/windbot](https://github.com/IceYGO/windbot) as an external,
local evaluation client. WindBot is a C# YGOPro client that connects to a
YGOSharp/SRVPro-compatible duel host and chooses actions through deck-specific
executors. It can also run in HTTP server mode, but one isolated process per
evaluation worker is the initial, easier-to-debug target.

Implementation order:

1. Pin and record a WindBot commit; build it with Visual Studio or Mono/.NET.
2. Install a matching `cards.cdb`, deck files, banlist, protocol version, and
   dialog assets beside the executable.
3. Finish the YGO Agent duel-host adapter that forwards legal game messages to
   WindBot and returns its responses to the environment.
4. Add lifecycle handling: process startup, readiness, port allocation,
   per-game reset, timeout, crash recovery, logs, and clean shutdown.
5. Support both seating orders and deterministic seeds where the protocol
   permits them.
6. Save `.yrp` replay, YGO Agent decision log, WindBot log, result, duration,
   and all asset revisions for failed and sampled successful games.

Acceptance gates:

- Complete at least 100 consecutive smoke games without protocol deadlock,
  illegal response, stale state, or orphaned process.
- Confirm card effects are loaded by checking representative
  `select_effectyn`, `select_chain`, `select_card`, and `select_option` paths.
- A timeout or WindBot crash must mark the game invalid, restart the worker,
  and never count as a win or loss.
- Measure games/hour and latency percentiles before selecting worker count.

WindBot is evaluation-only because external process/network round trips and
rule-based decision latency would substantially reduce training throughput.

## Stage 5: Build and validate the multi-deck pool

Take the initial deck list from WindBot's supported `Normal` executors. Do not
assume that every `.ydk` has a complete executor: WindBot notes that cards not
known by the selected deck AI may not be summoned or activated correctly.

For every candidate deck:

1. Pair the WindBot executor name with its exact `.ydk` file.
2. Import all card IDs, scripts, tokens, and extra-deck cards into YGO Agent's
   asset/code-list pipeline.
3. Validate 40-60 main-deck cards, legal extra deck, banlist compatibility,
   script availability, and code-list coverage.
4. Run legal-action smoke tests and at least 20 WindBot mirror games.
5. Classify the deck as `ready`, `limited`, or `excluded`, documenting why.

Begin with a small, strategically varied pool of roughly 4-8 validated Normal
decks rather than enabling the full catalog at once. Prefer decks whose WindBot
executors complete games reliably. Keep K9VS as the learner's first anchor deck
even though it is not currently a stock WindBot executor; evaluate it against
the selected WindBot decks.

## Stage 6: Multi-deck training curriculum

WindBot determines the opponent deck distribution, but WindBot itself does not
participate in training rollouts. For each selected WindBot deck, YGO Agent
loads the corresponding `.ydk` and uses fast in-process opponents.

Train in three controlled phases:

1. **Deck conditioning smoke test** — train/evaluate each deck separately and
   verify that observations identify the active deck or matchup unambiguously.
2. **Balanced multi-deck curriculum** — sample the validated deck pool
   uniformly, with both seating orders represented equally.
3. **Adaptive curriculum** — after the balanced baseline is stable, allocate
   more games to matchups with low win rate or high uncertainty while retaining
   a minimum sampling floor for every deck.

Within each matchup, retain the opponent-policy mixture as a starting point:

- 50% current policy/self-play
- 30% compatible historical checkpoint
- 20% in-process greedy bot

Checkpoint compatibility must include architecture metadata, code-list hash,
asset revision, deck-pool revision, and learned card-ID embedding dimensions.
Because expanding the code list changes embedding rows, introduce new cards
through an explicitly versioned model migration and initialize only new rows;
do not treat that migration as pretrained embedding work.

## Stage 7: Evaluation matrix and promotion gates

Evaluate candidate checkpoints outside the training loop. The primary matrix
is:

- Candidate versus WindBot for every validated deck
- Both seating orders
- At least 256 valid games per matchup when throughput permits
- Fixed seeds/deals for checkpoint-to-checkpoint comparisons
- K9VS regression matches against greedy, random-initialized, and historical
  policies

Report per matchup and macro/micro aggregates:

- Wins, losses, draws, invalid games, and Wilson confidence intervals
- Average turns, decisions, and wall-clock duration
- WindBot response latency p50/p95/p99 and timeout/crash rate
- Training SPS and evaluation games/hour
- GPU utilization, memory, and power during training
- Representative winning, losing, timeout, and illegal-action replays

Promote a checkpoint only if it improves the aggregate WindBot result without
materially regressing K9VS or any protected matchup. Never convert timeout,
protocol failure, or unsupported WindBot behavior into a model victory.

## Stage 8: Scaling after correctness

Only after the WindBot adapter and multi-deck evaluation matrix are stable,
consider longer runs, more environments, wider/deeper models, longer action
history, or bfloat16. Change one major variable at a time and retain the
no-external-pretrained-embedding decision.

## Current infrastructure

Use the GPU server and deployment paths documented in the workspace-level
`AGENTS.md`. Build or download WindBot and its dependencies on a machine with
public internet access, record checksums and revisions, then upload a complete
offline bundle to any isolated server.
