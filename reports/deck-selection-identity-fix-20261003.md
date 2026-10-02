# Deck-menu actor identity repair and frozen-100M recheck

## Result and scope

The confirmed own-deck alias bug is repaired in source `3a42a4b6293ee2d21a2bf4e402f9e5c597c021eb`.
Its Git tree `12daa43a026e8e4b54077b62e219e94e5be4cc3b` exactly matches isolated
build source `ec5f8926dd2542aaa9154ca403751cd1575a4085`. Uncommitted BO3 native
changes and the user's `TRAINING_PLAN.md` edits were excluded and preserved.

- Both `MSG_SELECT_CARD` and `MSG_SELECT_UNSELECT_CARD` retain the selecting
  player's deck CID from the core message, independently of the alias string.
- The identity survives staged choices, both select/unselect pools and forced
  history. Prompt/reset lifecycle clears the mapping.
- Legacy/structured action CIDs agree. A deck alias binds to no physical scene
  row; action and selected-group references stay unbound. We do not guess a
  physical copy from equal codes or expose hidden opponent identities.
- Shapes, policy masks, response indices, legal choices and forward-selection
  behavior are unchanged. This is nevertheless a changed observation meaning,
  not an inference-only heuristic and not checkpoint-output parity with the bug.
  Both training and inference use the repaired adapter.
- The actor retains its separate action-ID embedding. Unbound references do
  not supply direct per-card scene features; schema-level prompt-card objects
  could be a later improvement, not part of this repair.

## Validation

Portable CPython 3.10 build on the documented build server, image
`ygo-build-cache:replay-fix`, native optimization disabled. Native SHA-256:

`0f686f233d00115a99bcf8a5bee5da0d0d86e7a914e6cd89bae0a8980cb5f2f4`

Six new test methods cover 76 real linked-core frame/response cases: both seats,
verbose/quiet, duplicate codes, shuffled/subset physical sequences, staged
multi-select, forced-choice resolver, both select/unselect pools, preserved-seq
mode, unchanged non-deck/opponent behavior and new-message cache clearing. The
same identity-binding helper and actual legacy/history/structured writers are
exercised, including selected-group references. All pass, as do the existing
55 native boundary tests and 7 Python protocol audit tests.

The first build failed downloading dependencies. Reused the prior local Git
core mirror at `/root/ygo-build/jobs/disable-check-guard-92154ab/pkgsearch`
and known SQLiteCpp archive (`70c67d5680c47460f82a7abf8e6b0329bf2fb10795a982a6d8abc06adb42d693`).
No global H200 compiler/toolchain changes were required. Two fixture-only
corrections (supported notification constant; mirroring `action.msg_` assignment)
were made before all tests passed. Failed logs were retained. Initial isolated
setup also stopped safely at an existing tracked `assets/deck`; only `locale`
and `scripts/script` need runtime links, not a replacement of tracked decks.

## Actual 100M model recheck

H200 isolated run:
`/root/ygo-agent-gpu-20260910/training-runs/deck-identity-ec5f892-validation`.
The production native was unchanged throughout this run; no trainer ran.

Frozen 100M weight SHA-256:
`aeca25126229eff22049a81087dd8db0c049f84ad01bd4595757b79fdfa9e64a`.
The original training source is still `92f225c`, not this repair's source.
Runtime procedure overlay SHA-256 remained
`c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322`.

Reused the first paired seed per original five WindBot blocks, not replacement
seeds: 2026100102 for the first four opponents, 2026100300 for SkyStriker.
The candidate uses elfnote for the first four blocks and AI_SkyStriker for the
mirror. All ten attempts naturally finished, no invalids or cycle intervention.

| Opponent | Candidate seat 0 | Candidate seat 1 |
| --- | --- | --- |
| Kashtira | loss | loss |
| Labrynth | win, opponent deck-out | win, opponent deck-out |
| Tearlaments | win | win |
| Voiceless | win | win |
| SkyStriker | win, opponent deck-out | loss |

Aligned 1,511 model decisions using the probs/value/chosen-action blocks.
Legal logits, probabilities and values are finite; probabilities normalize.
For fresh min=max=1 own-deck select-card menus, 246 candidate rows and 38
actual choices were checked: **zero identity mismatches, zero wrong scene
references, zero selected mismatches**. The old diagnostic slice had 247/271
candidate mismatches and 33/37 selected mismatches. Denominators differ because
corrected inputs change trajectories; this is not an identical-state A/B win
rate estimate. Real-duel menu checks use card names decoded through the pinned
database/code list; the native fixtures independently check exact numeric IDs.
Debug output omits zero CIDs for hidden cards; audit parsing preserves their
slots instead of collapsing them.

Raw evidence package `training-runs/deck-identity-ec5f892-raw-20261003.tar.gz`
SHA-256 `388ffc30a5068bef68b06249fd6144d89058b1279d492d7760954d8125b7956b`.
Local unpacked directory of the run name includes `identity-audit.json`, per-duel
review indexes and Chinese replay commentary in addition to all ten YRP files,
engine logs, logits/probability/V decision JSONL and executor metadata.

## What remains unproved

This is a protocol correctness repair, not evidence that combos or interruption
quality now pass. Kashtira seat 0 still attacks 3300-ATK Arise-Heart with 0-ATK
Effect Veiler, taking lethal damage (eval.log 8300–8353). Labrynth seat 1 targets
its own Regina with Chaos Angel despite opposing targets (decision 88), another
strategically suspect action to retain. Three wins are opponent deck-outs.

No Q/search/belief was enabled. Baseline task 1.1 remains open: complete the
corrected-runtime control/strategy accounting before shadow-Q. Both subsequent
GAE and Q arms must start from the same frozen actor and corrected runtime.
Changing the environment only for Q would invalidate the algorithm comparison.
The ten human-written notes are qualitative review, not complete metric labels.
