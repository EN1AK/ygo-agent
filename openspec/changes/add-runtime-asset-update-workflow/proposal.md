## Why

The project currently provisions card databases and YGOPro Lua scripts through static, pinned sources, so local runtime assets can drift behind upstream card/script changes. This blocks reliable use of newly supported cards and makes Chinese `cards.cdb` updates a manual, undocumented step.

## What Changes

- Add a documented runtime asset update workflow for YGOPro Lua scripts from `https://github.com/mycard/ygopro-scripts`.
- Add a documented Chinese card database update workflow from `https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb`.
- Preserve reproducibility by recording source URL, resolved revision or timestamp, checksum, destination path, and update time in a local manifest.
- Keep downloaded `cards.cdb`, `strings.conf`, and Lua script assets ignored by Git.
- Add verification requirements so updated assets are checked before evaluation or inference workflows use them.
- No model architecture, training algorithm, or native engine behavior changes are proposed.

## Capabilities

### New Capabilities

- `runtime-asset-update`: Covers updating, recording, validating, and consuming external card database and YGOPro Lua script assets.

### Modified Capabilities

- None. Existing baseline capabilities have not yet been archived into `openspec/specs/`, so this change introduces a focused capability for the update workflow.

## Impact

- Affected files likely include `Makefile`, `README.md`, optional helper scripts under `scripts/`, and OpenSpec documentation.
- Affected runtime assets include `assets/locale/zh/cards.cdb`, optional locale metadata such as `strings.conf`, and `scripts/script`.
- Network access becomes an explicit update-time dependency, not an implicit evaluation-time dependency.
- Evaluation and inference workflows continue to consume local files and should fail early when required assets are missing or invalid.
