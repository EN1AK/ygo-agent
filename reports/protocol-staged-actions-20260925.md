# Staged protocol actions — 2026-09-25

OpenSpec change: `add-clustered-multideck-training`; pinned ygopro-core:
`f96929650ff8685b82fd48670126eae406366734`.

## Implemented

- `MSG_SELECT_TRIBUTE` and both `MSG_SELECT_SUM` modes now expose sorted,
  incremental choices. Production no longer constructs all card subsets or
  permutations. Each offered card admits a valid completion under the matching
  core-derived predicate. Search work has an explicit one-million-node cap;
  exhaustion fails the protocol boundary rather than silently dropping options.
- `MSG_SORT_CARD` exposes no-reorder plus staged card ordering. The response is
  a complete permutation, and replay recording now writes every ordering byte.
- `MSG_SELECT_COUNTER` splits each feasible allocation range into binary
  decisions; every uint16 allocation remains reachable with two policy options
  per decision. Replay records the complete allocation.
- Added sort/counter/RPS message IDs by appending to the legacy ID table; the
  40M checkpoint's existing rows remain unchanged. Legacy action encoding now
  handles these three messages without an `Unsupported message` exception.

## Verification

- Built `ygopro_ygoenv` in the isolated GPU-server tree at
  `/tmp/ygo-proto-build.pa9KjH/src2` with the pinned core and xmake 2.9.9.
- `python -m unittest discover -s tests -p 'test_ygocore_protocol*.py'`:
  **40 passed**.
- Actual same-instance core-to-adapter-to-core fixtures cover staged sort,
  counter allocation, tribute, exact sum, sum-limit, select/unselect, and RPS.
  Other interactive branches are not yet covered by this stronger fixture type.
- The 30-card/15-selection sum liveness test returns 16 first-step choices
  without materializing any of its 155,117,520 subsets. A deliberately
  impossible case hits the explicit search budget and raises.

## Artifact hashes (SHA-256)

- Adapter header: `e0d3252a054da8c3a4b8230637a41bbfd8d303b31480438ec5f9dbd6080760ca`
- Adapter binding: `02adf8cedb4192dac3c9384ed1c0e6cc91b929b9c2744e27225448f6d2b49a8a`
- Boundary test: `3365e9e8f997a7cb2ae402891e0a41665ff630c3b4c0cf6202be12d7b230234e`
- Isolated native extension: `fa72782996a8a4922ad7044bd38dc8b35c4e6f333b3c647f1f8767337d1cadfd`
- Pinned protocol contract file: `d876ddb2b1f6da58aa8e42c17cf6e8585a293de1652df83a75475634aa0f49f6`

## Gate status

Tasks 8.3.5, 8.3.7, and 8.3.8 are implemented and tested. The complete
every-branch parser/core oracle (8.3.6) and refreshed protocol gate (8.5)
remain pending. The older `protocol-coverage.json` merely associates branch
names with test-function names; it does not establish same-instance core
acceptance for every branch. Therefore the protocol gate is not yet eligible
to authorize the 5M pilot. No training was started in this work segment.
