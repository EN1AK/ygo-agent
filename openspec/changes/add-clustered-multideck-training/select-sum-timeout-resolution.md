# MSG_SELECT_SUM timeout resolution

## Symptom

The first H200 pilot reached 102,400 steps but created persistent 100% CPU
environment workers. The timeout watchdog replaced the environments, while the
abandoned native threads remained alive and caused `double free or corruption`
when the process exited.

The 120,832-step diagnostic run captured two independent stalls:

- env 6, decks `deck-dfac0a778e1f2a7f` vs `deck-0c2a122d6661a1c5`,
  `MSG_SELECT_SUM`, step 34;
- env 4, decks `deck-7540128386b2b411` vs `deck-dfac0a778e1f2a7f`,
  `MSG_SELECT_SUM`, step 32.

Both stalls had a fully consumed 143-byte message buffer and occurred while the
adapter was generating legal responses, not while waiting for another engine
message.

## Root cause

The adapter's exact-sum response generator enumerated every permutation of the
optional cards up to `max_count`. Card selection order has no gameplay meaning
for this response, so large synchro-material prompts expanded toward factorial
complexity. The pinned ygocore validator searches selected card sets instead.

## Resolution

- Generate each optional-card subset once, in stable index order.
- Validate each candidate with the pinned core-equivalent exact-sum oracle.
- Cap `max_count` to the available candidate count and reject invalid count
  ranges without entering the search.
- Keep the full timeout diagnostic and correct the replacement-environment
  capacity guard.
- Add a 14-candidate, choose-7 regression that must yield exactly 3,432 unique,
  core-valid card sets rather than 7! permutations per set.

## Verification

- Native module SHA-256:
  `e34bd6db2c837c8a9f48ac0288fa072db838f9b3a675ff8bc41d1496ed3e3edd`.
- Protocol/domain suite: 23 tests passed.
- Same-corpus repair run: 120,832 steps, zero environment timeouts, finite
  optimizer, checkpoint saved, exit code 0, no double-free.
- Refreshed protocol contract internal SHA-256:
  `21d0d3493bb1437f6b1a344d0b4f0b449d180088c206dcb8665b36fbb7fde585`.
- Refreshed protocol contract file SHA-256:
  `d876ddb2b1f6da58aa8e42c17cf6e8585a293de1652df83a75475634aa0f49f6`.
- Refreshed coverage report: 20 interactive messages, 64 normal branches,
  zero uncovered branches; file SHA-256
  `7dd62d161a2690ecc333f172d98bde96dcc593c0c4c8b81556ebe035428c97da`.

The valid 5M pilot is the run whose directory and launch evidence declare the
refreshed contract above. Earlier diagnostic or aborted pilot directories are
retained only as failure evidence and are not promotion candidates.
