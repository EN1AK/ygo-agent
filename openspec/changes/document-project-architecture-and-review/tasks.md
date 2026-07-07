## 1. OpenSpec Baseline

- [ ] 1.1 Review `proposal.md`, `design.md`, and all capability specs for accuracy against the current repository layout.
- [ ] 1.2 Validate the change with `openspec validate document-project-architecture-and-review --strict`.
- [ ] 1.3 Archive the accepted baseline with `openspec archive document-project-architecture-and-review` after review approval.

## 2. Runtime Documentation

- [x] 2.1 Document the supported Python version, package install modes, and workflow-specific dependency groups.
- [ ] 2.2 Document required external asset paths for `cards.cdb`, `strings.conf`, YGOPro scripts, decks, checkpoints, embeddings, and code lists.
- [ ] 2.3 Decide whether all user-facing scripts must use `_repo_bootstrap.py` or require editable installation, then update the scripts or documentation consistently.

## 3. Build Workflow

- [ ] 3.1 Document Linux native build prerequisites and the xmake command for `ygopro_ygoenv`.
- [x] 3.2 Document native Windows build prerequisites, including MSVC, xmake, pybind11, and the required Python path environment variables.
- [ ] 3.3 Verify ignored build outputs remain unstaged after a successful native build.

## 4. Evaluation Workflow

- [ ] 4.1 Document and verify the random evaluation smoke test that does not require a trained checkpoint.
- [ ] 4.2 Document checkpoint evaluation inputs, including checkpoint path, deck, language, seed, and environment options.
- [ ] 4.3 Document battle evaluation player configuration so each side's checkpoint, deck, model, and policy are attributable.

## 5. Inference Service

- [ ] 5.1 Document inference service startup requirements for model artifacts, code lists, embeddings, locale databases, and backend selection.
- [x] 5.2 Split or document JAX and TFLite dependencies so unsupported platforms are not blocked by unavailable backend packages.
- [ ] 5.3 Document health, duel initialization, prediction, CORS, and stale duel-state cleanup behavior.

## 6. Review Hardening Backlog

- [x] 6.1 Replace runtime `assert` validation used for user or asset input with explicit exceptions.
- [x] 6.2 Restrict or replace pickle-based embedding loading for untrusted files.
- [x] 6.3 Replace interpolated shell command construction in checkpoint utilities with argument-list subprocess calls.
- [x] 6.4 Align `setup.py`, `ygoenv/setup.py`, and `ygoinf/setup.py` with the dependencies required by supported workflows.
- [x] 6.5 Fix README mojibake and update the README to reflect the current Linux and Windows support boundaries.
