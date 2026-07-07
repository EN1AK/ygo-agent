## Context

Runtime card assets are currently external to Git, which is correct for repository size and platform independence. The current provisioning path is still static:

- `Makefile` clones `https://github.com/mycard/ygopro-scripts.git` but checks out a fixed script revision.
- `Makefile` downloads locale databases from a pinned `mycard/ygopro-database` commit URL.
- `assets/locale/*/*.cdb`, `assets/locale/*/strings.conf`, and `scripts/script` are ignored and can be stale without any local trace of their source revision.

The requested update sources are:

- Lua scripts: `https://github.com/mycard/ygopro-scripts`
- Chinese card database: `https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb`

The environment loads `assets/locale/<lang>/cards.cdb` in Python and reads Lua files from `scripts/script` in the native YGOPro integration. Evaluation and inference should continue using local assets; network access should only be needed when an operator explicitly updates assets.

## Goals / Non-Goals

**Goals:**

- Add an explicit update workflow for YGOPro Lua scripts and Chinese `cards.cdb`.
- Preserve reproducibility by recording source metadata in a manifest.
- Keep runtime assets ignored by Git while making their provenance inspectable.
- Validate updated assets before they become the active local runtime inputs.
- Make the update workflow safe on Linux and Windows where practical.

**Non-Goals:**

- Do not commit downloaded `cards.cdb`, `strings.conf`, or Lua scripts to the repository.
- Do not automatically update assets during evaluation, training, inference startup, or import time.
- Do not change card feature encoding, model checkpoints, deck format, or native YGOPro rule behavior.
- Do not guarantee upstream compatibility for every new card before smoke tests pass.

## Decisions

### Decision: Use an explicit asset update command

Provide a script or Makefile target that operators run intentionally, for example `make update-assets` or `python scripts/update_assets.py`.

Rationale: evaluation and inference workflows need deterministic local inputs. Automatic network downloads during runtime would make experiments non-reproducible and brittle when upstream services are unavailable.

Alternatives considered:

- Update on every `make`: rejected because it changes assets implicitly.
- Update on Python import: rejected because imports should not perform network or filesystem mutations.

### Decision: Track source provenance in a local manifest

The update workflow should write a local manifest, such as `assets/asset_manifest.json`, containing source URL, resolved Git commit for scripts, HTTP metadata where available, SHA-256 checksums, file sizes, destination paths, and update timestamp.

Rationale: the manifest makes local state auditable without committing the large or generated runtime assets themselves.

Alternatives considered:

- Commit downloaded assets: rejected because `.gitignore` intentionally excludes them.
- Rely only on file modification time: rejected because it does not identify source revision or integrity.

### Decision: Update through temporary paths and promote atomically

Downloads and Git updates should be staged into temporary paths before replacing `assets/locale/zh/cards.cdb` or `scripts/script`.

Rationale: interrupted downloads or failed checkouts should not leave the active environment with a partial database or missing scripts.

Alternatives considered:

- Overwrite files in place: rejected because a failed download can corrupt the active runtime.
- Require manual copy after update: rejected because it is error-prone and hard to validate.

### Decision: Keep pinning optional but visible

The update workflow should support both latest-upstream updates and explicit pinned versions, such as a specific script commit or an expected database checksum.

Rationale: users need latest card data for play, while experiments and bug reports need reproducible assets.

Alternatives considered:

- Always update to latest: rejected because model evaluation results become difficult to reproduce.
- Always pin to a hard-coded revision: rejected because it recreates the current stale-asset problem.

### Decision: Verify assets before use

The update workflow should verify that the Chinese database is a readable SQLite database and that required script files such as `constant.lua`, `utility.lua`, and `procedure.lua` exist after the script update. When code-list validation is available, it should report cards missing scripts.

Rationale: upstream availability is not enough; the local runtime requires coherent database and script files.

Alternatives considered:

- Trust successful download/clone only: rejected because content can be incomplete or incompatible.

## Risks / Trade-offs

- Network endpoints can be unavailable -> The command should fail without modifying the active local assets and leave existing assets usable.
- Upstream latest scripts can break current engine assumptions -> The workflow should support pinned commits and run verification before promotion.
- CDN files may change without a versioned URL -> The manifest should record checksum, size, timestamp, and HTTP metadata when available.
- Windows symlink behavior differs from Linux -> The implementation should either avoid symlink-only behavior or fall back to copy/junction guidance.
- Updated card data may require code-list or embedding regeneration -> The workflow should warn when code-list validation detects new card codes.

## Migration Plan

1. Add the update command or script without changing default evaluation/training startup behavior.
2. Generate or update the local manifest only when the user explicitly runs the update workflow.
3. Update README/OpenSpec instructions to describe latest and pinned update modes.
4. Run validation against `assets/locale/zh/cards.cdb` and `scripts/script` after update.
5. If validation fails, keep existing active assets and report the failed staged path for inspection.

## Open Questions

- Should English `cards.cdb` and `strings.conf` also move to Moecube CDN or remain on the existing mycard database source?
- Should `strings.conf` be updated from the same CDN family if a stable URL is available?
- Should the manifest be ignored as local state or committed as an example/template?
- Should update verification run a full random evaluation smoke test or only fast asset integrity checks?
