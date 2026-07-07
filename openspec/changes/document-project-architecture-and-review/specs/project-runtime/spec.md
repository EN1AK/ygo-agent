## ADDED Requirements

### Requirement: Repository runtime layout is explicit
The project SHALL document the runtime role of the `ygoenv`, `ygoai`, `scripts`, and `ygoinf` packages so contributors can identify whether a change affects environment execution, model code, user workflows, or inference serving.

#### Scenario: Contributor reviews package ownership
- **WHEN** a contributor opens the OpenSpec runtime documentation before changing project code
- **THEN** the documentation identifies `ygoenv` as the environment package, `ygoai` as model/training utilities, `scripts` as workflow entrypoints, and `ygoinf` as the inference service

### Requirement: External runtime assets remain separately provisioned
The project SHALL treat locale card databases, locale strings, YGOPro Lua scripts, checkpoints, embeddings, replays, logs, and compiled native extensions as external or generated assets that are not committed to Git.

#### Scenario: Runtime assets are prepared
- **WHEN** a runtime workflow is executed from a fresh checkout
- **THEN** the workflow documentation identifies the required asset paths and the user can provision missing assets without force-adding ignored files

### Requirement: Python entrypoints declare import assumptions
User-facing Python scripts MUST either bootstrap the repository root for direct execution or document that an editable package install is required.

#### Scenario: Script is run directly
- **WHEN** a user runs a workflow script from the `scripts` directory
- **THEN** imports resolve through an explicit bootstrap path or the failure message/documentation points to editable installation as the required setup

### Requirement: Dependency declarations match runtime workflows
Python package metadata SHALL declare or intentionally separate dependencies required by supported evaluation, training, and inference workflows.

#### Scenario: Installing a supported workflow
- **WHEN** a user installs the package extras or workflow-specific package described by the project
- **THEN** the imports used by that supported workflow are satisfied without relying on undocumented manual dependency installation
