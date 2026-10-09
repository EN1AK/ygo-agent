# Effect declarations and public activation state

This is an opt-in `structured-state-v2` observation and model extension. The
default remains unchanged. It adds three tensors to `structured-lite-v1` and
preserves all existing tensor shapes and meanings. It does not change teaching,
rewards, legal menus, response encoding or the policy's information boundary.

## Inputs and meaning

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `action_effect_semantics_` | options × 32, uint8 | Binary declaration features for the action's description-bound effect. All zero means unknown. |
| `public_chain_` | 16 × 7, uint16 | Public source card ID, activating seat relative to viewer, description slot, link and status flags. |
| `public_turn_effects_` | 64 × 9, uint16 | Activation, resolution, activation-negation and effect-disable counts, grouped by activating seat/card/description for this turn. |

Description slots are 1..16 in public state, with 0 unknown. Chain status bits
are activated=1, solving=2, solved=4, activation-negated=8, effect-disabled=16.
They are independent flags: a resolution message does not prove an effect
succeeded. Repeated status notifications are idempotent. Counts aggregate
copies, not physical card instances, and must not be interpreted as remaining
once-per-turn uses. Unknown description slots can aggregate multiple effects.
No opaque engine effect pointer or opponent-private state enters the tensors.

The ledger consumes public CHAINING, SOLVING, SOLVED, NEGATED, DISABLED, END and
NEW_TURN messages. A chain end clears only the chain; a new turn or duel reset
clears both. The turn ledger survives the existing 32-event ring. Exceeding
16 links, 64 distinct turn keys or a uint16 count is an explicit error, never
silent truncation. Rows are viewer-relative and deterministic.

The parser supports unconditional initial_effect registrations, literal own-card
`aux.Stringid(code, slot)` or `aux.Stringid(id, slot)` with `local s,id=GetID()`,
and local clones. It extracts declared categories, speed/type, range, callback
presence, count-limit presence and card-target property. It does not execute Lua
or prove cost/target/operation behavior. Missing descriptions, multiple matching
registrations, unsupported mutation and dynamic initialization remain unknown.
Whole-card tags remain the fallback representation. A declaration's absence
means no positively identified syntax, not a proved negative rule.

This first version does not implement a revealed-hand identity ledger, infer
hidden deck recipes, track persistent restrictions like Maxx C, or replace LSTM
with turn memory. Those need separate contracts and raw-message audits.

## Assets and model compatibility

Build a NEW semantic directory from the SAME inputs as the source checkpoint:

```sh
python scripts/build_structured_semantics.py --cards-db <cards.cdb> \
  --code-list <code_list.txt> --scripts <script-directory> \
  --output-dir <new-semantic-directory> --effect-descriptions
```

The existing three tables are unchanged; `effect-descriptions.u8` and versioned
metadata are added. The table has one all-zero unknown card row, 16 descriptions
per card and 32 bytes per description. Metadata records table/input/parser
hashes and per-card binding coverage. The Python schema wrapper verifies asset
hashes; the native constructor checks table dimensions and requires the new
asset for v2. Both the native environment and RNNAgent must explicitly request
`structured-state-v2`; an old native build is rejected by missing tensor keys.

The full model adds named zero-initialized residual projections for effect
features and attention over public-state tokens. Global public-state pooling
also enters the recurrent state through a zero residual. Existing parameters
and optimizer code are unchanged. The relationship-only ablation excludes
these new inputs along with the existing semantic/event inputs.

Do not load old weights as v2 without migration:

```sh
python scripts/migrate_public_state_checkpoint.py --source <v1.flax_model> \
  --source-semantic-metadata <original/metadata.json> \
  --semantic-metadata <new/metadata.json> --code-list <code_list.txt> \
  --num-embeddings <original-count> --output-dir <new-checkpoint-directory>
```

Migration checks the source checkpoint and original semantic metadata, matching
code-list hashes, unchanged old table hashes, and a complete parameter inventory.
It verifies old/new outputs within 1e-6 on empty and populated new-input fixtures,
writes a new checkpoint and metadata, and preserves the source. It does not
migrate optimizer state or claim a trained strength gain. Equality assumes the
same legacy observations/runtime; rebuilding an engine with different core or
chain-provenance settings is not covered by weight parity.

## Verification gates

```sh
python -m pytest tests/test_effect_semantics.py tests/test_public_state_model.py -q
g++ -std=c++17 -I ygoenv tests/public_effect_state_test.cpp -o <test-executable>
<test-executable>
python scripts/validate_public_state_environment.py --deck <deck.ydk> \
  --code-list <code_list.txt> --semantic-assets <new-semantic-directory> \
  --games 4 --output <new-report.json>
```

The native check compares v1/v2 menus, every old tensor, rewards and termination
under identical actions/seeds and requires new inputs to be exercised. Its
public-ID check requires an observed activation in the retained event evidence;
if a long forced sequence loses that evidence, reject the check as incomplete
and audit the raw stream instead of treating missing evidence as proof of a leak.
Compile in an isolated runtime; preserve the production native and checkpoint.
