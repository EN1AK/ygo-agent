# Structured-lite effect-tag audit

The first vocabulary intentionally recognizes only explicit YGOPro Lua API or
category constants. It does not attempt natural-language or full Lua semantic
interpretation. `exact` means that a listed token is present in the card's
script, not that an effect slot has been proved to own that operation.

| Tag | Positive evidence | Negative boundary |
|---|---|---|
| activate | `EFFECT_TYPE_ACTIVATE` | ignition/trigger effects without that type |
| destroy | `CATEGORY_DESTROY`, `Duel.Destroy(` | text or helper names containing only “destroy” |
| negate | `CATEGORY_NEGATE`, `Duel.NegateActivation/Effect(` | disable-only effects without these APIs |
| send_to_grave | `CATEGORY_TOGRAVE`, `Duel.SendtoGrave(` | ordinary rule movement to graveyard |
| banish | `CATEGORY_REMOVE`, `Duel.Remove(` | temporary flags without removal operation |
| draw | `CATEGORY_DRAW`, `Duel.Draw(` | hand addition that is not a draw |
| search_or_add | `CATEGORY_SEARCH`, `Duel.Search(`, `Duel.SendtoHand(` | draw-only effects |
| summon | `CATEGORY_SUMMON`, `Duel.Summon(` | special summons |
| special_summon | `CATEGORY_SPECIAL_SUMMON`, `Duel.SpecialSummon(` | normal summons |
| target | `SetTarget(`, `Duel.SelectTarget(` | non-targeting selection |
| change_atk_def | explicit ATK/DEF effect constants | battle calculation without stat modification |
| move | `Duel.MoveToField/MoveSequence(` | send/destroy/remove operations |
| cost_release | `Duel.Release(` | release performed only as an effect operation |
| cost_discard | `REASON_DISCARD` | sending from hand without discard reason |
| continuous | exact CDB `TYPE_CONTINUOUS` bit | continuous monster effects |

Cards with a missing script use `unknown/unknown`. Scripts with no recognized
tag use `unknown/fallback`. All tag patterns and source hashes are recorded in
the generated metadata.
