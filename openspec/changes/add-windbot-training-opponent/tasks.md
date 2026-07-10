## 1. WindBot Provisioning

- [x] 1.1 Document supported WindBot runtime targets, starting with local `WindBot.exe` and optional Mono/Linux if verified.
- [x] 1.2 Add configuration fields for WindBot executable path, working directory, deck, bot name, host, port, host info/password, dialog setting, timeout, and logs.
- [x] 1.3 Validate WindBot executable presence, runtime directory, `cards.cdb`, and configured deck before launch.
- [x] 1.4 Keep WindBot source checkouts, binaries, copied `cards.cdb`, and generated logs ignored by Git.

## 2. Adapter And Process Lifecycle

- [x] 2.1 Implement a WindBot process launcher using argument-list subprocess calls.
- [x] 2.2 Capture WindBot stdout/stderr or log files for setup and episode failures.
- [x] 2.3 Terminate WindBot child processes on timeout, workflow interruption, or training shutdown.
- [x] 2.4 Support explicit and dynamically allocated local ports for WindBot-backed sessions.

## 3. Environment / Protocol Integration

- [x] 3.1 Build a minimal local YGOPro-compatible host or adapter spike that WindBot can connect to.
- [ ] 3.2 Map WindBot-controlled decisions into the environment opponent side without changing existing observation/action tensors.
- [x] 3.3 Preserve current `self`, `random`, `bot`/greedy, and `human` play modes.
- [x] 3.4 Fail fast when WindBot mode is selected but the host/adapter is unavailable.

## 4. Training And Evaluation Entry Points

- [x] 4.1 Add WindBot as an opponent option to `scripts/eval.py`.
- [x] 4.2 Add WindBot opponent configuration to selected JAX training scripts after the adapter smoke test works.
- [x] 4.3 Decide whether Torch training scripts support WindBot in the first implementation or explicitly report unsupported mode.
- [x] 4.4 Emit run metadata including WindBot executable, source revision or version, deck, assets, host, port, and seed.

## 5. Smoke Tests And Verification

- [x] 5.1 Add a WindBot setup validation command that does not start long training.
- [x] 5.2 Add a one-duel or short-episode WindBot smoke test.
- [x] 5.3 Validate deck/card/script compatibility before long training jobs.
- [x] 5.4 Verify that missing executable, missing `cards.cdb`, port conflict, and connection timeout produce actionable errors.

## 6. Documentation

- [x] 6.1 Update README with WindBot setup, build, asset, deck, and smoke-test instructions.
- [x] 6.2 Document that WindBot is optional and not required for existing random/greedy/self-play workflows.
- [x] 6.3 Document expected throughput limitations and recommended use as evaluation or curriculum opponent before full rollout training.
- [x] 6.4 Validate the change with `openspec validate add-windbot-training-opponent --strict`.
