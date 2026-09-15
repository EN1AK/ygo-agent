# Human demonstration pipeline

Human recordings enter through a two-stage interface so that short combo
demonstrations and complete expert duels cannot accidentally share value
labels.

## Raw replay manifest

Import YGOPro2 recordings without committing the large binary files:

```bash
python scripts/import_human_replays.py /data/replays /data/combos.jsonl \
  --kind combo --deck-tag elfnote
```

The `ygo-human-demonstration-v1` record stores source hashes, exact deck lists,
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
