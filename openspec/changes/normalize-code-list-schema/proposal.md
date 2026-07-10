## Why

`scripts/code_list.txt` is now generated as two columns (`card_code has_script`), but several Python and native readers still assume each line is a single integer. This causes embedding workflows and older environment backends to fail or parse inconsistently after the asset update workflow regenerates the code list.

## What Changes

- Define a stable code-list line format where the first column is always the card code and an optional second column records script availability.
- Add shared Python parsing behavior so embedding and runtime utilities can read both legacy single-column and current two-column code lists.
- Update native YGOPro-v0 and EDOPro readers to ignore optional metadata columns instead of parsing the full line as an integer.
- Keep YGOPro-v1 support for the `has_script` column while preserving card-id ordering.
- Document the supported code-list update command and file format.
- Add verification that the current generated `scripts/code_list.txt` can be parsed by affected workflows.

## Capabilities

### New Capabilities
- `code-list-schema`: Defines the generated code-list format and compatibility expectations for Python tooling and native environment readers.

### Modified Capabilities
- None. No baseline specs have been archived into `openspec/specs/`, so this change introduces a focused capability.

## Impact

- Affected code areas: `ygoai/utils.py`, `scripts/card/embedding.py`, `scripts/card/code_list.py`, `scripts/update_assets.py`, `ygoenv/ygoenv/ygopro0/ygopro.h`, `ygoenv/ygoenv/edopro/edopro.h`, README documentation, and tests or smoke commands.
- Affected workflows: asset validation with `--run-code-list`, training/evaluation startup, checkpoint embedding loading, and embedding generation.
- Compatibility impact: existing single-column code-list files remain readable; current two-column files become readable across Python tools and native backends.
