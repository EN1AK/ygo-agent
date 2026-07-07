## ADDED Requirements

### Requirement: Runtime assets can be explicitly updated
The project SHALL provide an explicit operator-run workflow to update external runtime assets without requiring users to manually replace files.

#### Scenario: Updating Chinese card database
- **WHEN** an operator runs the documented asset update workflow for the Chinese card database
- **THEN** the workflow downloads `cards.cdb` from `https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb` and stages it for `assets/locale/zh/cards.cdb`

#### Scenario: Updating YGOPro Lua scripts
- **WHEN** an operator runs the documented asset update workflow for Lua scripts
- **THEN** the workflow fetches scripts from `https://github.com/mycard/ygopro-scripts` and stages them for `scripts/script`

### Requirement: Asset updates are not implicit runtime side effects
Evaluation, training, inference startup, and Python imports MUST NOT automatically download or mutate card database or Lua script assets.

#### Scenario: Starting evaluation
- **WHEN** a user starts an evaluation command
- **THEN** the command uses local assets and does not contact upstream asset sources unless the user explicitly requested an update command

### Requirement: Updated assets are validated before activation
The update workflow MUST validate staged assets before replacing the active local runtime assets.

#### Scenario: Database validation fails
- **WHEN** the downloaded Chinese `cards.cdb` is not a readable SQLite database
- **THEN** the workflow rejects the staged file and leaves the previous active `assets/locale/zh/cards.cdb` unchanged

#### Scenario: Script validation fails
- **WHEN** staged Lua scripts do not include required runtime files such as `constant.lua`, `utility.lua`, or `procedure.lua`
- **THEN** the workflow rejects the staged scripts and leaves the previous active `scripts/script` unchanged

### Requirement: Asset provenance is recorded
The update workflow SHALL record local provenance metadata for each updated asset source.

#### Scenario: Manifest is written
- **WHEN** an asset update succeeds
- **THEN** the workflow records source URL, destination path, checksum, file size, update timestamp, and resolved script Git commit when applicable

### Requirement: Pinned updates remain supported
The update workflow SHALL allow operators to update to a specified script revision or verify an expected card database checksum.

#### Scenario: Updating to a script commit
- **WHEN** an operator requests a specific `mycard/ygopro-scripts` commit
- **THEN** the workflow checks out that commit and records it in the manifest

#### Scenario: Checksum does not match
- **WHEN** an operator provides an expected checksum and the downloaded `cards.cdb` checksum differs
- **THEN** the workflow fails before activating the staged database

### Requirement: Generated assets remain outside Git
Downloaded card databases, locale string files, Lua scripts, temporary update directories, and local provenance manifests MUST remain excluded from source control unless a future change explicitly promotes a template or sample file.

#### Scenario: Reviewing after update
- **WHEN** an operator updates runtime assets locally
- **THEN** `git status` does not require staging downloaded databases, Lua scripts, or temporary update files

### Requirement: Update failures preserve existing runtime assets
The update workflow MUST fail safely without deleting or corrupting the active runtime assets.

#### Scenario: Network failure during update
- **WHEN** the upstream CDN or Git repository is unavailable during an update
- **THEN** the workflow exits with an error and the previous local `cards.cdb` and `scripts/script` remain usable
