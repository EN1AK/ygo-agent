## Context

`scripts/card/code_list.py` currently generates `scripts/code_list.txt` from the active card database and Lua scripts. The current file contains two whitespace-separated columns: the card code and a `has_script` flag. The YGOPro-v1 native backend reads both columns and uses the second column to decide whether to preload card scripts.

Other readers are inconsistent. Python embedding utilities parse each line with `int(line.strip())`, and YGOPro-v0 / EDOPro native readers parse the whole line with `std::stoul(line)`. Those paths either fail on current files or rely on implementation quirks. The code-list format needs to be explicit and all readers need to agree on the first-column card id contract.

## Goals / Non-Goals

**Goals:**

- Make `code_list.txt` readable as either legacy single-column lines or current two-column lines.
- Preserve the current two-column generated format so script availability remains available to YGOPro-v1.
- Provide a shared Python helper for parsing code-list entries.
- Update native readers that do not need `has_script` to parse only the first column.
- Keep generated card-id ordering stable.
- Document and verify the update path.

**Non-Goals:**

- Do not change model architecture, embedding format, checkpoint format, or card feature tensors.
- Do not update external card databases or YGOPro Lua scripts as part of this change.
- Do not make YGOPro-v0 or EDOPro consume `has_script` metadata beyond ignoring it safely.

## Decisions

### Decision: Standardize on first-column compatibility

All code-list readers should treat the first whitespace-separated token as the card code. Additional columns are metadata. The first metadata column is currently `has_script`, encoded as `0` or `1`.

Rationale: this preserves the current generated file and keeps legacy single-column files usable.

Alternatives considered:

- Revert generation to single-column only: rejected because it loses useful script-availability metadata already consumed by YGOPro-v1.
- Require every reader to understand all metadata columns: rejected because most readers only need card ordering.

### Decision: Add a shared Python parser

Python code should use one helper in `ygoai.utils` for code-list parsing instead of open-coding `int(line.strip())`.

Rationale: embedding loading and embedding generation need identical ordering semantics, and future metadata columns should not break Python tools.

Alternatives considered:

- Patch each Python call site locally: acceptable short term, but it keeps duplicated parsing behavior.

### Decision: Keep generator output deterministic

`scripts/card/code_list.py` should continue sorting by card code and writing deterministic lines. If a compatibility output mode is added, it must be explicit rather than changing the default unexpectedly.

Rationale: checkpoints and embeddings depend on stable card id ordering.

## Risks / Trade-offs

- Existing embeddings may not cover newly generated card codes -> verification should fail with clear missing-code errors rather than silently reordering.
- Older native extensions may still be built from stale C++ sources -> native build verification is separate from parser unit checks.
- Code-list metadata could expand later -> readers should ignore unknown extra columns unless they explicitly require them.

## Migration Plan

1. Add shared Python code-list parsing that handles single-column and multi-column files.
2. Update Python embedding utilities to use the shared parser.
3. Update YGOPro-v0 and EDOPro native readers to parse only the first column.
4. Document the current `<card_code> <has_script>` generated format and update command.
5. Verify Python compile, parser behavior against current `scripts/code_list.txt`, and OpenSpec validation.
