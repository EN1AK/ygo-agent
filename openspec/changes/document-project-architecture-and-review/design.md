## Context

YGO Agent is a research-oriented Yu-Gi-Oh! reinforcement learning project with four primary layers:

- `ygoenv`: a high-throughput environment modeled after envpool, with Python registry/wrappers and C++/pybind extensions backed by `ygopro-core` and optional `edopro-core`.
- `ygoai`: JAX/Flax and Torch model definitions, PPO-style training utilities, rollout buffers, evaluation helpers, and checkpoint utilities.
- `scripts`: executable workflows for random/bot evaluation, checkpoint battle, single-node/distributed training variants, Torch training variants, and card embedding/code-list utilities.
- `ygoinf`: FastAPI inference server that maps external duel-state payloads into model inputs and serves JAX/TFLite predictions.

The repository intentionally ignores large/generated runtime files: compiled `.pyd/.so` extensions, xmake/build caches, locale `cards.cdb`/`strings.conf`, YGOPro Lua scripts, checkpoints, replay files, logs, and Python virtual environments. The README documents the Linux-first happy path, while the current codebase also contains Windows build support that requires explicit Python include/lib environment variables and a local xmake installation.

Code review scope covered:

- Packaging: `setup.py`, `ygoenv/setup.py`, `ygoinf/setup.py`.
- Native build: `xmake.lua`, `repo/packages/y/ygopro-core/xmake.lua`.
- Runtime initialization: `ygoai/utils.py`, `ygoenv/ygoenv/registration.py`, `scripts/_repo_bootstrap.py`.
- User workflows: `scripts/eval.py`, `scripts/battle.py`, training scripts.
- Inference service: `ygoinf/ygoinf/server.py`, `features.py`, `jax_inf.py`, `tflite_inf.py`.
- Native environment: `ygoenv/ygoenv/core/*`, `ygoenv/ygoenv/ygopro/*`, `edopro/*`.

## Goals / Non-Goals

**Goals:**

- Establish a durable OpenSpec baseline that describes what must be true for the project to build, run, evaluate, and serve predictions.
- Record current architecture and known quality risks from code review in a form that future implementation work can consume.
- Make runtime prerequisites explicit: Python packages, native extensions, locale databases, YGOPro scripts, code lists, decks, checkpoints, and embeddings.
- Define verification commands that can be run before future commits affecting build/runtime behavior.
- Keep this change documentation-only; implementation hardening tasks are listed but not applied here.

**Non-Goals:**

- Do not redesign the RL algorithms, model architecture, feature encoding, or Yu-Gi-Oh! rule handling.
- Do not vendor ignored runtime assets or generated native binaries into Git.
- Do not make `ygoinf` fully Windows-compatible in this documentation change.
- Do not replace the existing Makefile/xmake workflow with a new build system.
- Do not resolve every TODO in `ygopro.h`, `edopro.h`, or training scripts.

## Decisions

### Decision: Use repo-local OpenSpec artifacts

OpenSpec has been initialized in this repository with `schema: spec-driven`. Change artifacts live under `openspec/changes/document-project-architecture-and-review/`, and future archived specs should move into `openspec/specs/`.

Rationale: the repository is self-contained and does not currently use registered OpenSpec stores. Repo-local docs keep requirements versioned with the code they describe.

Alternatives considered:

- External planning store: rejected because code and generated requirements should be reviewed together.
- README-only documentation: rejected because it does not provide per-capability requirements, scenarios, or task gates.

### Decision: Model specs around capabilities, not source directories

The proposed specs are `project-runtime`, `environment-build`, `evaluation-workflow`, `inference-service`, and `quality-review`.

Rationale: capabilities map to user-visible guarantees and verification flows. Directory-based specs would duplicate implementation layout and become stale when files move.

Alternatives considered:

- One monolithic project spec: rejected because it would make build, runtime, inference, and quality requirements harder to validate independently.
- One spec per package: rejected because workflows cross package boundaries, especially evaluation and inference.

### Decision: Treat ignored assets as required external inputs

Locale databases, YGOPro scripts, checkpoints, embeddings, replay files, and compiled extensions remain out of version control. Specs define their required locations and verification behavior instead.

Rationale: these files are large, generated, platform-specific, or release artifacts. Committing them would increase repository size and couple the repo to one local machine.

Alternatives considered:

- Force-add assets and `.pyd` binaries: rejected because `.gitignore` intentionally excludes them and binaries are platform/Python-version-specific.
- Runtime auto-download on every script execution: rejected for now because it adds network dependence to deterministic evaluation workflows.

### Decision: Capture code review findings as follow-up tasks

The review found risks that are real but not appropriate to fix as part of documentation initialization:

- `ygoinf/setup.py` requires `tflite-runtime`, which is unavailable for some Windows/Python combinations.
- `setup.py` for `ygoai` does not declare JAX/Flax/Distrax/Chex dependencies needed by `scripts/eval.py` and training workflows.
- Several runtime validations use `assert`, which can be disabled with optimized Python execution.
- `ygoai.utils.load_embeddings()` uses pickle, which is unsafe for untrusted files.
- `ygoai/rl/ckpt.py` shells out with interpolated paths for `gsutil`.
- Several script entrypoints do not use `_repo_bootstrap.py`, so running them from `scripts/` may depend on editable-install behavior.
- `README.md` contains mojibake in several sections and is still Linux-first despite Windows support existing in `xmake.lua`.

Rationale: making these explicit gives future changes a prioritized backlog without mixing behavior changes into OpenSpec initialization.

### Decision: Keep Windows support documented as explicit build environment

Windows native build requires xmake, Visual Studio/MSVC, Python headers/libs, `pybind11`, and environment variables consumed by `xmake.lua`: `PYBIND11_INCLUDE`, `PYTHON_INCLUDE`, `PYTHON_LIBDIR`, and `PYTHON_VERSION_NODOT`.

Rationale: xmake's Python discovery can resolve a different `python3` than the runtime Python on Windows. Explicit environment variables make the build reproducible.

Alternatives considered:

- Rely on xmake package `pybind11`: rejected for Windows because its CMake Python discovery can select the wrong Python.
- Require WSL only: rejected because the current project now supports native Windows `ygopro_ygoenv` smoke evaluation.

## Risks / Trade-offs

- **Risk: Documentation can drift from code** -> Mitigation: require OpenSpec updates when build, runtime, inference, or evaluation behavior changes.
- **Risk: Generated specs may imply stronger guarantees than current code provides** -> Mitigation: specs include current known limitations and tasks distinguish documentation baseline from future hardening.
- **Risk: Windows instructions are machine-specific** -> Mitigation: specs describe required variables and prerequisites rather than hard-coding one user's absolute paths.
- **Risk: Asset provisioning is manual** -> Mitigation: specs define exact expected paths and smoke tests so missing assets fail early.
- **Risk: Existing scripts have inconsistent import behavior** -> Mitigation: tasks call out extending `_repo_bootstrap.py` or packaging fixes for all user-facing scripts.
- **Risk: `ygoinf` dependency constraints block installation on Windows** -> Mitigation: task backlog separates inference service dependency refactoring from evaluation runtime support.

## Migration Plan

1. Commit OpenSpec initialization and change artifacts.
2. Validate the change with `openspec validate document-project-architecture-and-review --strict`.
3. Keep current runtime behavior unchanged.
4. For future implementation work, choose tasks from `tasks.md`, update specs first when requirements change, then apply code changes and run verification commands.
5. When this documentation baseline is accepted, archive the change with `openspec archive document-project-architecture-and-review` so capability specs move to `openspec/specs/`.

## Open Questions

- Should Windows native build be a first-class supported platform in README/CI, or only a best-effort local workflow?
- Should `ygoinf` split optional extras such as `ygoinf[jax]` and `ygoinf[tflite]` to avoid platform-specific install failures?
- Should all scripts use `_repo_bootstrap.py`, or should packaging be fixed so editable installs are the only supported execution model?
- Should model checkpoints and embeddings be referenced by a manifest with checksums?
