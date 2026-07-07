## 1. Update Command

- [x] 1.1 Add an explicit asset update entrypoint, such as `scripts/update_assets.py` or `make update-assets`.
- [x] 1.2 Support updating `assets/locale/zh/cards.cdb` from `https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb`.
- [x] 1.3 Support updating `scripts/script` from `https://github.com/mycard/ygopro-scripts`.
- [x] 1.4 Support pinned script commits and optional expected database checksum arguments.

## 2. Safe Activation

- [x] 2.1 Stage downloads and Git checkouts in temporary paths before replacing active assets.
- [x] 2.2 Promote validated assets atomically or with a documented Windows-safe fallback.
- [x] 2.3 Preserve existing active assets when download, checkout, checksum, or validation fails.

## 3. Validation

- [x] 3.1 Validate downloaded `cards.cdb` as a readable SQLite database before activation.
- [x] 3.2 Validate required Lua files including `constant.lua`, `utility.lua`, and `procedure.lua`.
- [x] 3.3 Run or document code-list validation with `scripts/card/code_list.py` after asset updates.
- [x] 3.4 Report newly missing scripts or card-code mismatches without silently accepting inconsistent assets.

## 4. Provenance

- [x] 4.1 Write a local asset manifest with source URL, destination path, checksum, file size, timestamp, and script Git commit.
- [x] 4.2 Ensure downloaded assets, temporary update directories, and local manifests remain ignored unless a template manifest is intentionally added.
- [x] 4.3 Document how to reproduce a previous asset state from manifest metadata.

## 5. Documentation And Verification

- [x] 5.1 Update README with latest and pinned asset update commands.
- [x] 5.2 Document that evaluation, training, inference startup, and imports do not auto-update assets.
- [x] 5.3 Add verification commands for the update workflow and asset integrity checks.
- [x] 5.4 Validate the OpenSpec change with `openspec validate add-runtime-asset-update-workflow --strict`.
