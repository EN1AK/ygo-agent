# RPS protocol fixture work — 2026-09-25

## Scope

Continued OpenSpec change `add-clustered-multideck-training`, task 8.3.1.
No training was started.

## Changes made

- Extracted validation of the trailing `MSG_ROCK_PAPER_SCISSORS` player byte
  into a shared helper used by the production message handler.
- Added a native fixture that drives the linked ygopro-core
  `PROCESSOR_ROCK_PAPER_SCISSORS` processor and captures its emitted frames.
- Added coverage for both request stages, all nine hand pairs, non-repeat draw
  behavior, and a repeated draw followed by a decisive round.
- The fixture validates response range [1, 3] before feeding each integer to
  the core, then checks the actual core result and packed `MSG_HAND_RES` byte.

The pinned source contract used for the oracle is
`training-runs/protocol-source-ygopro-core-f969296`; its RPS processor emits
player requests 0 then 1, packs `hand0 + (hand1 << 2)`, and sets the winner
or `PLAYER_NONE` after a draw.

## Verification status

Verified on the GPU server in the isolated source snapshot
`/tmp/ygo-proto-build.pa9KjH/src2`. The first run found that the fixture
returned after a drawn round's `MSG_HAND_RES`; the fixture now waits for the
core processor to finish, and the test checks the complete second round.

- Built the native extension with isolated xmake 2.9.9 scripts: success.
- Ran `python -m unittest discover -s tests -p 'test_ygocore_protocol*.py' -v`:
  23 tests passed. The RPS tests cover all nine hand pairs, both player
  request frames, repeat after a draw, invalid player bytes, and out-of-range
  integer responses.
- Compiled extension SHA256:
  `ba82d0bbd36bd82e34de29e5b33dfbb80648234df9fe49f430d72959bbd36427`.
- The build cache's ygopro-core `operations.cpp` SHA256 matches the frozen
  local source: `67a3eb41b47bfd33f477514f386d99b12b4204c9a934ab0cc30fd41e095f4bce`.

The SSH key was present at `C:\Users\Administrator\.ssh\id_ed25519`.
This desktop process resolved its default SSH directory to `C:\Windows\.ssh`;
using the explicit key and existing known-hosts paths restored access.
The saved gateway fingerprint matched the server fingerprint observed during
connection: `SHA256:IVjkRgeMAWHa+gA/SVMZBy2yZBg+m04S1mBpzb2Ufrk`.

OpenSpec task 8.3.1 is complete. Other parser branches and the full protocol
gate remain in progress.
