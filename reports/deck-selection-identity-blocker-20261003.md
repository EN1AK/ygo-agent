# 100M baseline audit: deck-selection identity mismatch (confirmed, not repaired)

The bounded strategy review for `add-vrpo-candidate-q` found a correctness
blocker before shadow/Q training. Keep baseline task 1.1 open. Do not start
Candidate-Q/VRPO with the current native module and call this gate passed.
No trainer, native binary, checkpoint or runtime procedure was changed during
this investigation. The frozen 100M checkpoint remains intact.

## Evidence from actual actor inputs

Review slice: attempts 0001 and 0002 of Kashtira, Labrynth, Tearlaments,
Voiceless (seed 2026100102), and the SkyStriker mirror (seed 2026100300).
All 10 selected duels have natural terminals. Exactly 1,351 model decisions
were aligned to the native log by the `probs` / `value` / chosen-action block;
automatic engine choices were not incorrectly counted as model decisions.

For fresh `MSG_SELECT_CARD`, min=max=1, own-deck numeric-menu entries:

| Measure | Count |
| --- | ---: |
| Candidate rows checked | 271 |
| Candidate rows whose serialized card identity differs from the core menu | 247 |
| Actual choices among those entries | 37 |
| Actual choices with a mismatched card identity | 33 |

These counts describe this diagnostic slice, not all prompts or training
steps. SkyStriker seat 1 had no matching own-deck entries and contributes zero
to this denominator. Both `legacy_features[1:3]` and
`structured_features[9:11]` contain the same wrong ID. The per-slot IDs also
match the native chosen-action debug record, ruling out merely reading the
wrong JSONL row. The incorrectly associated scene reference is nonzero too.

Concrete example: Labrynth attempt 0001, `decisions.jsonl` line 16 and
`eval.log` lines 299–326. The engine offers slot 1 (zero based) as
**耀圣之诗～回乡之平行体～**, and subsequently adds that spell to hand.
The actor receives CID **13484**, code **56651978**,
**耀圣之风诗 蕾吉娜**, with scene index **2**. Thus the engine response is
valid, but the actor is selecting using a different card identity.

The exact code list was copied from the evaluation command, SHA-256
`f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`.
The Chinese card database was verified identical on local and H200, SHA-256
`5f13245de4e665450858f66ef2f736a3d2f48cc7f0036961373a96a650cb6797`.

## Root cause

Pinned ygocore `f96929650ff8685b82fd48670126eae406366734`:

- `playerop.cpp:244–267` sorts the selection candidates and writes each
  candidate's actual code plus `get_select_info_location`.
- `card.cpp:1513–1528`: for the selecting player's deck, unless
  `select_deck_seq_preserved` is set, `get_select_sequence` returns a running
  **menu alias**, not the card's physical deck sequence. The default is false.
- `playerop.cpp:296–316` uses the same mechanism for select/unselect menus.

Current adapter `ygoenv/ygoenv/ygopro/ygopro.h`:

- `MSG_SELECT_CARD` builds only string specs from the prompt. The quiet branch
  discards each transmitted code; verbose mode uses the code to print the
  correct name but likewise does not carry it into the action identity.
- `init_multi_select` creates actions from those specs alone.
- `WriteState` resolves numeric specs against `_set_obs_cards`' physical deck
  positions, assigning that unrelated row's CID and scene index. Both feature
  schemas, action references and potentially subsequent history inherit it.
- `MSG_SELECT_UNSELECT_CARD` also discards the prompt identity, so it needs the
  same regression coverage; the numeric audit above does not quantify that
  branch. Forced-action history uses `spec_to_card_id` and also needs coverage.

This defect is in the actual actor-input construction, not a PPO numerical
failure or a WindBot executor failure. It can interfere with learning search,
tuner access and resource choices. It is **not** proof that this is the sole
cause of every weak move or the specialist/multideck strength difference.

## Required follow-up before restarting the experiment

1. Preserve prompt-scoped code/identity independently of physical zone specs
   for both affected messages, staged choices, selected groups and history.
2. Bind scene references only where a matching legally visible identity is
   established. Never reinterpret a deck alias as a physical position or read
   hidden opponent cards to repair the mapping. Retain response slot order,
   duplicate-card choices, finish and forward-selection behavior.
3. Test shuffled/subset deck menus against **real ygocore-generated frames**,
   both players, duplicate codes, selected/unselected sets, forced actions,
   verbose/quiet parity and prompt lifecycle. Existing legal-index/digest
   tests are insufficient because they can consistently hash wrong features.
4. Build an isolated native artifact, verify the same diagnostic seeds have
   zero identity mismatches, and run the existing native/protocol tests.
5. Keep old results labelled as the old native environment. Re-evaluate the
   frozen 100M actor on the corrected runtime with a bounded paired sample,
   then restart **both** matched GAE and Q arms under that same runtime.
   Changing only Q's environment would confound the algorithm comparison.

This changes baseline-environment semantics even if tensor dimensions stay
compatible. It is a prerequisite repair, not evidence that VRPO is implemented
or that the policy has learned stronger combos. Shadow/Q work is blocked at
the baseline gate while the fix and revised baseline are pending.

## Reproduction artifacts

Local `training-runs/ppo100m-strategy-review-20261003/` contains the per-duel
source-file hashes, complete aligned decision indexes, readable indexes, the
pinned code list, copies of the two relevant core source files, and
`prompt-card-identity-audit.json`. The operational scripts are
`training-runs/extract-strategy-review-20261003.py` and
`training-runs/audit-prompt-card-identity-20261003.py`.
The original raw replay/evaluation bundles remain unchanged.

Investigated source: `5650ef3d4027d7af4114595d27dfc861b83b371b`.
Production native SHA-256 rechecked:
`5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`.
Frozen 100M checkpoint SHA-256:
`aeca25126229eff22049a81087dd8db0c049f84ad01bd4595757b79fdfa9e64a`.
H200 process check found no active trainer or evaluator. No experiment was
started or restarted during this audit.
