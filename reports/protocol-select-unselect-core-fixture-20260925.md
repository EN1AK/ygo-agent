# Select/unselect protocol verification — 2026-09-25

OpenSpec change `add-clustered-multideck-training`, task 8.3.2.

The adapter reads both card pools from `MSG_SELECT_UNSELECT_CARD`. Select
responses use indices `0..select_count-1`; unselect responses use indices
`select_count..select_count+unselect_count-1`. The core accepts one byte
`[1, index]` for a card action, and integer `-1` only when `finishable` or
`cancelable` is set. Both flags share the same wire response, so the adapter
offers one finish action when both are true.

An unselect action now sets the structured action tensor's `reserved_0` byte
to 1. This distinguishes removing a selected card from adding one without
changing the frozen 40M checkpoint's tensor shape or field names.

The pinned-core fixture creates actual cards, invokes the core message writer,
then invokes its `check_response` path. Tests verify both pool counts and
payloads, every index in a two-plus-two fixture, invalid indices, all
finish/cancel flag forms, and the last representable index 255. If combined
pool size exceeds 256, the adapter fails closed: a byte response cannot
represent every core candidate.

Verification ran in the GPU server's isolated source snapshot
`/tmp/ygo-proto-build.pa9KjH/src2`. The native extension compiled with
isolated xmake 2.9.9 scripts, and all 27
`test_ygocore_protocol*.py` tests passed. The build cache's
`ygopro-core/playerop.cpp` SHA256 matched the frozen local source:
`7c0ee6a0bd15c0e459a4c7ddf4fb02f0adc0812a27b613d64de1c17a9d65da42`.
The tested extension SHA256 was
`37fe249413e157f5954a2589588fdac8c1a90dc5c7d63fa771f5caa3ab78a4ae`.

The action-capacity audit found that `info:num_options` had an inclusive
upper bound of `max_options - 1`, although a complete action list may have
exactly `max_options` members. The bound now includes `max_options`.
`WriteState` checks the complete candidate count before populating action
tensors; any larger set throws rather than returning a truncated list.

The broader protocol gate, including full parser-branch frame coverage, remains
open. No training was started.
