# Race mask protocol verification — 2026-09-25

OpenSpec change `add-clustered-multideck-training`, task 8.3.4.

Pinned ygopro-core defines `RACES_COUNT = 26`. Its
`field::announce_race` writer transmits a 32-bit available mask, but counts
and validates only the low 26 bits. A response containing undefined high bits
is accepted if its low bits satisfy the requested count and available mask;
the high bits are retained in the resulting hint. The adapter only offers
combinations of defined low bits, including the zero mask when the core
reduces the requested count to zero.

The native fixture invokes the core writer and validator on one duel instance.
Tests confirm high-bit behavior, unavailable low-bit rejection, and the
adapter's 26-bit option domain. A bounded binomial-count check now rejects
large race or attribute choice sets before combination materialization, so
26-choose-13 does not allocate millions of policy actions for a 24-action
model.

GPU isolated build: success. All 29 `test_ygocore_protocol*.py` tests
passed. Tested extension SHA256:
`3901bc7a9481945195fa88bc9d2e125b71824ca2df530dbd84a5b13e4aa92340`.

The broader protocol gate remains open. No training was started.
