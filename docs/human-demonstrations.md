# Human demonstration pipeline

The v2 manifest preserves the eight `uint32` YRP2 seed words at 0x20–0x3f.
The word at 0x40 is the extended header version, **not a player seat**.
Old v1 manifests must be regenerated from the original replay before training.
Use the source rule flags and `create_duel_v2` RNG semantics: pseudo shuffle
does not disable hand shuffles. Seed zero can silently select different cards
after a search even when the initial hand is correct. These rules follow the
upstream [replay header](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay.h)
and [replay initialization](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay_mode.cpp).

`scripts/audit_human_replay.py` replays every response without fallback inputs.
`scripts/convert_human_replay.py` maps each audited response through the policy
adapter, including automatic replies and staged weighted selections; every
emitted response must match. A second conversion must reproduce all tensors,
and an independent core process must reproduce the actor's events and state.
No parser success alone grants training eligibility. The isolated bridge adds
an opt-in seed-sequence path; old default scenes require observation parity.

The elfnote pilot freezes five teaching routes and 24 reserved conditions in
`assets/exercises/teaching-v1/elfnote-single-card.json`. All remain one elfnote
development lineage. Compare autonomous first-turn outcome/resource targets,
not action-string accuracy. Multi-starter success at a single-starter target
is only a lower bound; disruption failures without a verified feasible recovery
remain diagnostic, not proof of overfitting. Short recordings have no value label.

Human recordings enter through a two-stage interface so that short combo
demonstrations and complete expert duels cannot accidentally share value
labels.

## Raw replay manifest

Import YGOPro2 recordings without committing the large binary files:

```bash
python scripts/import_human_replays.py /data/replays /data/combos.jsonl \
  --kind combo --deck-tag elfnote
```

The `ygo-human-demonstration-v2` record stores source hashes, exact deck lists,
raw response bytes and engine metadata. Combo records enable policy supervision
only. A future complete replay is imported with `--kind full_duel`; value
supervision is then permitted only after engine replay verifies a terminal
result.

## Engine-aligned decisions

The next conversion stage replays each response against a compatible OCG
engine and emits `ygo-human-decision-v1` records. Each record must contain:

- demonstration ID, decision index, acting player and deck hash;
- the exact model observation and recurrent-state boundary;
- legal-action count and the human-selected action index;
- trajectory kind, sample weight and source hashes;
- terminal result only for verified complete duels.

Training consumes this decision schema rather than depending on YRP2 directly.
This boundary allows later importers (live client capture, JSONL decision logs,
or other replay versions) without changing the behavior-cloning learner.

## Different decks

Different expert decks are useful for shared rules, interaction timing and
generic card play, but their policy targets are not interchangeable at every
state. Preserve deck hashes on every sample and use these safeguards:

1. Maintain a sampling floor for the target deck and report metrics per deck.
2. Weight same-archetype demonstrations highest, shared-card demonstrations
   next, and unrelated archetypes lowest.
3. Never train unknown card IDs as the generic unknown embedding. Expand and
   version the code list before using a replay containing unsupported cards.
4. Use a balanced deck sampler so a large off-deck corpus cannot overwhelm the
   target deck.
5. Validate promotion on the target deck's combo suite and duel evaluation,
   not aggregate imitation loss alone.

Short combos should not create value targets because they end by recording
choice rather than by a game terminal. Complete high-quality human duels can
provide policy targets immediately and terminal value targets after replay
verification; intermediate value targets may later be estimated separately.


## Turning combination failures into exercises

After a frozen evaluation, first verify that the same initial scene admits the
claimed outcome. A semantic transfer of an existing legal route can establish
a lower-bound witness, but it must pass two identical actor replays and an
independent raw-core audit. Preserve failed transfers too; never choose a
fallback action silently. The elfnote study archives six such post-hoc witnesses.

Store a failed policy trace and a successful same-scene witness as a scenario
candidate. This is not yet a negative label for the first differing action:
equivalent copies, placement choices, or valid alternative branches can differ.
For action-level correction, reconstruct the policy's actual prefix and prove
a useful alternative at the same decision root before granting eligibility.
Save source/model hashes, observations, decisions, rules and RNG, causal scope,
and `training_ready=false` until that gate is satisfied.

Disruption needs a feasible recovery target. Score resources and available
interactions after the chain resolves, including position and once-per-turn
availability when relevant. Failing the unopposed target does not prove failure
to adapt; in this study both learned policies independently used an alternate
Synchro route after Ash, although their positioning still needs functional review.

Once used for diagnosis or teaching, reserved cases become exposed development
cases. Freeze fresh combinations/interruption timings for the next round; keep
original-chain rehearsal and independent old-policy histories, and report each
metric separately. Do not promote based on the five rehearsal anchors alone.


## Search as a route teacher

The bounded elfnote study uses `assets/exercises/teaching-v1/elfnote-search.json`
and `scripts/run_elfnote_search.sh`. Its default stage is `preflight`; `collect`
explicitly enables frozen-policy discovery, and `train` explicitly consumes the
verified collection. These stages write new run directories and never select a
production checkpoint. Search emits complete goal-attaining action histories,
not a PUCT distribution, action-value estimate, or proof of an optimal route.

A case with a hidden hand trap first runs the parent policy until that response
has been observed and resolved at a stable own-player prompt. Discovery may
branch only after this root. The exact earlier observations, menus and actions
must replay, and their new teaching weights are zero. Both models still process
the complete history to reconstruct their recurrent state. A passive-opponent
case is separately declared and may search from the initial decision. This is
a deterministic conditional exercise, not a belief-search or fair-play guarantee
against arbitrary hidden hands and future responses.

Compare independent stochastic discovery with policy-guided prefix branching
under declared attempt and wall-time limits. Include failed attempts and all
prefix replay decisions in discovery costs. A finite search miss is not an
unsolvability label. Before a successful route can enter fitting, require two
identical actor/tensor replays plus an independent raw-core audit. Do not import
failed searches as negative action labels, or change the goal to count them as
successes. Save the original outcome/resource objective alongside the route.

Success-filtered self-training and search-route teaching are separate arms with
the same parent, optimizer budget and old-history KL coefficient. This tests
fixed-parent exploration and subsequent imitation; it does not substitute for
an on-policy PPO experiment or establish equal-FLOPs sample efficiency. Log
both restored-root continuation and fully autonomous opening outcomes, original
combo rehearsal, old skill anchors, and reserved within-family conditions.
Only fresh held-out conditions can test the next iteration after these cases
have been inspected or used for teaching.

Before interpreting a learning plateau, compare the exercise model inputs with
the ordinary environment preprocessing. The native `mask_` is populated only
when `oppo_info` is enabled. With ordinary visible observations its zero buffer
is unused; the model must ignore it, as `EnvPreprocess(skip_mask=True)` does.
Otherwise card identity, zone and other features collapse to the padding row,
and same-name choices in hand and on field can become indistinguishable. Require
a hand-versus-field regression with the old path as a failing negative control.
Keep the flawed run, fix the path in a separate source snapshot, and repeat
the baseline before claiming an optimizer or search failure. Engine replay alone
verifies legality and outcomes; it cannot detect erased policy input features.
