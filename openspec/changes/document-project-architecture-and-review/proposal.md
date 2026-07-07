## Why

YGO Agent now contains a mixed Python/C++ runtime, platform-specific build paths, ignored runtime assets, training/evaluation scripts, and an inference service, but no machine-readable project specification. This change initializes OpenSpec documentation so future changes can be reviewed against explicit runtime, build, evaluation, inference, and quality requirements instead of relying on README fragments and tribal knowledge.

## What Changes

- Add OpenSpec artifacts documenting the current project architecture, runtime assets, build requirements, and verification commands.
- Record the code review findings and convert them into actionable tasks for follow-up hardening.
- Define capability specs for the environment build, runtime assets, evaluation workflow, inference API, and code quality gates.
- No production code behavior changes are proposed by this change; this is a documentation and review baseline.

## Capabilities

### New Capabilities

- `project-runtime`: Repository-wide runtime, package layout, asset, and dependency expectations.
- `environment-build`: C++/pybind YGO environment build requirements for Linux and Windows.
- `evaluation-workflow`: Random, bot, checkpoint, and battle evaluation workflow requirements.
- `inference-service`: Model serving API requirements for `ygoinf`.
- `quality-review`: Review, verification, and known-risk tracking requirements.

### Modified Capabilities

- None. There are no existing OpenSpec capabilities in `openspec/specs/` yet.

## Impact

- Affected documentation: `openspec/config.yaml`, `openspec/changes/document-project-architecture-and-review/*`.
- Affected code areas reviewed: `xmake.lua`, `setup.py`, `ygoenv/`, `ygoai/`, `ygoinf/`, `scripts/`, `repo/packages/`.
- Affected workflows: environment build, asset preparation, random evaluation smoke test, checkpoint evaluation, battle evaluation, and FastAPI inference service deployment.
- External assets remain intentionally ignored by Git and must be provisioned separately: card databases, locale strings, YGOPro Lua scripts, checkpoints, embeddings, and compiled native extensions.
