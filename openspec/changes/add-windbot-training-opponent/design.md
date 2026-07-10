## Context

YGO Agent currently supports built-in opponent modes in the native environment: self-play, random bot, greedy bot, and human/client-facing play modes. The training scripts select built-in bot evaluation through `play_mode='bot'` or `play_mode='random'`, and `scripts/eval.py` exposes `--bot_type random|greedy`.

IceYGO WindBot is an external C# YGOPro bot project. Its upstream README describes building `WindBot.sln`, placing `cards.cdb` beside `WindBot.exe`, and launching the bot with command-line connection parameters such as deck, host, port, host info, dialog, and name. That model is materially different from the current in-process C++ `GreedyAI` and `RandomAI` classes.

The integration should therefore treat WindBot as an external process or service that connects to a local YGOPro-compatible host controlled by YGO Agent, rather than assuming WindBot can be linked into `ygopro_ygoenv`.

## Goals / Non-Goals

**Goals:**

- Add WindBot as a selectable training and evaluation opponent mode.
- Support configuring and launching an external WindBot binary from local paths.
- Provide a host/adapter layer that allows the RL environment to play against a WindBot-controlled opponent.
- Keep process lifecycle, ports, timeouts, logs, and cleanup explicit and testable.
- Make deck, cards database, script assets, and WindBot build prerequisites clear.
- Preserve existing random, greedy, self-play, and checkpoint evaluation behavior.

**Non-Goals:**

- Do not vendor WindBot source or binaries into this repository by default.
- Do not rewrite WindBot in Python or C++.
- Do not require WindBot for normal random/greedy/self-play training.
- Do not change model architecture, reward shaping, or existing replay/checkpoint formats.
- Do not guarantee every WindBot deck is compatible with the current card/script asset set before validation passes.

## Decisions

### Decision: Integrate WindBot as an external process

The implementation should launch or connect to WindBot as a separate process and communicate through a local YGOPro-compatible host/adapter.

Rationale: WindBot is a C# project with its own runtime assumptions. Keeping it out-of-process avoids coupling the native `ygopro_ygoenv` extension to .NET/Mono and reduces build risk.

Alternatives considered:

- Link WindBot into the C++ environment: rejected because WindBot is C# and not structured as a C++ library.
- Port WindBot logic into Python: rejected because it would be a large behavior rewrite and likely drift from upstream.
- Use WindBot only as a manual benchmark outside training: rejected because the user specifically wants it as a training opponent.

### Decision: Add a distinct opponent mode

WindBot should be configured as a distinct opponent mode such as `windbot`, not overloaded onto the existing `bot`/`greedy` behavior.

Rationale: WindBot has external process, deck, connection, and timeout requirements that do not apply to built-in bots.

Alternatives considered:

- Replace `bot` with WindBot: rejected because current training depends on cheap built-in greedy evaluation.
- Treat WindBot as `human`: rejected because training scripts need explicit lifecycle and automation.

### Decision: Keep WindBot artifacts externally provisioned

WindBot source checkout, build output, `WindBot.exe`, bot deck files, and any copied `cards.cdb` should remain local external artifacts by default.

Rationale: WindBot is third-party code with its own repository, build process, and binary outputs. Committing it would increase repo size and blur ownership.

Alternatives considered:

- Git submodule: possible later, but rejected for the first integration because users may want to pin local builds independently.
- Vendored binaries: rejected because binaries are platform/runtime-specific.

### Decision: Validate assets and connectivity before training

The implementation should provide a smoke test that launches the host/adapter and WindBot, confirms connection, runs at least one duel or short episode, and reports failure causes.

Rationale: WindBot failures can come from missing `cards.cdb`, incompatible deck files, port conflicts, missing .NET/Mono runtime, or protocol mismatch. Training should fail early with actionable errors.

Alternatives considered:

- Let training fail naturally: rejected because long training jobs should not discover opponent setup failures late.

## Risks / Trade-offs

- WindBot protocol may not map cleanly to the current in-process env -> Mitigation: build a minimal host/adapter spike before touching all training scripts.
- External process overhead can reduce throughput -> Mitigation: support WindBot evaluation first, then controlled training parallelism and process pooling.
- Port conflicts and orphan processes can destabilize training -> Mitigation: allocate ports explicitly or dynamically and ensure cleanup on exceptions.
- WindBot deck/card/script mismatch can produce invalid duels -> Mitigation: validate `cards.cdb`, Lua scripts, deck files, and code-list compatibility before launching training.
- Windows and Linux runtime behavior may differ -> Mitigation: document Windows-first support if using `WindBot.exe`, and treat Mono/Linux as an explicit compatibility target.
- Third-party repository changes can break integration -> Mitigation: support a pinned WindBot source revision/build path in configuration.

## Migration Plan

1. Add WindBot configuration and validation without changing default training behavior.
2. Implement a local adapter smoke test that can launch/connect WindBot and run one controlled duel.
3. Add `windbot` as an optional evaluation opponent mode.
4. Extend selected training scripts to use WindBot local evaluation or training opponent mode.
5. Document setup, build, deck, asset, and troubleshooting steps.
6. Keep `random`, `greedy`, and self-play modes as defaults until WindBot smoke tests are stable.

## Open Questions

- Should the first implementation target Windows `WindBot.exe` only, or also Mono on Linux?
- Should WindBot connect to an existing YGOPro host implementation, or should this project add a minimal local host/adapter?
- Which WindBot deck(s) should be the default training opponent?
- Should WindBot be used for full training rollouts, local evaluation only, or curriculum stages?
- How many concurrent WindBot processes are acceptable before throughput becomes impractical?
