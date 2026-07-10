## Why

Current training and evaluation opponents are limited to built-in random and greedy bots, which provide weak and narrow opponent behavior. Adding IceYGO WindBot as an external training opponent gives agents access to scripted deck-specific play patterns and a stronger benchmark before self-play or model-vs-model evaluation.

## What Changes

- Add a WindBot opponent integration path using `https://github.com/IceYGO/windbot`.
- Support launching or connecting to WindBot as an external YGOPro-compatible bot process for training and evaluation.
- Add configuration for WindBot executable path, deck/deck file, host, port, host info/password, dialog, timeout, and process lifecycle.
- Add a training/evaluation opponent mode that can select WindBot instead of the current built-in `random` or `greedy` bot.
- Add verification commands and failure handling for missing WindBot binaries, unsupported platforms, connection failures, and deck/card asset mismatches.
- Keep WindBot source/binaries and downloaded third-party assets out of Git unless a future change explicitly vendors a small adapter.

## Capabilities

### New Capabilities

- `windbot-training-opponent`: Covers provisioning, configuring, launching, connecting, and using IceYGO WindBot as an opponent in training and evaluation workflows.

### Modified Capabilities

- None. The existing baseline specs have not been archived into `openspec/specs/`, so this proposal introduces a focused WindBot capability instead of modifying archived specs.

## Impact

- Affected code areas likely include training/evaluation scripts, environment opponent mode selection, process management utilities, README, and OpenSpec documentation.
- Affected workflows include random smoke evaluation, checkpoint evaluation, training-time local evaluation, and opponent selection in JAX/Torch training scripts.
- External dependency impact: WindBot is a C# project built with Visual Studio or Mono and expects `cards.cdb` beside `WindBot.exe`.
- Runtime impact: WindBot may run as a separate process or server-mode process and communicate with a YGOPro-compatible host over local networking.
- Security/operational impact: external process execution, port binding, timeouts, and cleanup must be controlled explicitly.
