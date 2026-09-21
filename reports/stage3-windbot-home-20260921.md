# Stage 3 + WindBot home validation (2026-09-21)

## Scope

This record covers recovery of the frozen Stage 3 checkpoint, deployment on the
`home` Windows/WSL2 machine, repair of the WindBot evaluation path, and the
resulting OldSchool and stronger-deck evaluations.

The runtime repository is `/home/ygo/ygo-agent` inside Ubuntu 22.04. The saved
Windows checkout is `D:\workspace\ygo-agent`.

## Frozen inputs

- Stage 3 checkpoint:
  `1789092408_step_000009994240.flax_model`
- Stage 3 checkpoint SHA-256:
  `4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`
- Frozen code list SHA-256:
  `f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`
- Frozen elfnote deck SHA-256:
  `19b1c5aba1a5f00e0d29c9b9776c45de7ecafc6e7e25509a77c1e3ea5f133dcd`
- WindBot revision:
  `b0a2355f00bd59491add14ff09efa9914a3b47c6`
- WindBot portable bundle SHA-256:
  `4a416f38f64417775087cf01a2321d54d83d79e9dbcd548c9be7118888823121`

The current project code list is not interchangeable with the frozen Stage 3
list. Evaluation used the frozen list so the checkpoint embedding shape and
card indices remained reproducible.

## Home runtime setup

The Stage 3 inputs are isolated under `/home/ygo/ygo-agent/stage3-frozen`.
WindBot is installed at `/home/ygo/windbot-b0a2355` and runs through Mono.
The home WSL image required these packages in addition to `mono-runtime`:

- `libmono-system-data4.0-cil`
- `libmono-system-transactions4.0-cil`
- `libmono-system-runtime-serialization4.0-cil`

A direct startup check loaded all 72 WindBot decks successfully.

## Problems found and fixed

### Environment deck name is not the WindBot executor name

The environment deck file and WindBot executor are related but use different
names. For example:

- environment deck: `AI_OldSchool.ydk`
- WindBot executor argument: `OldSchool`

Passing `AI_OldSchool` as the executor did not fail fast. WindBot fell back to
unrelated executors; observed examples were `PumpkingExecutor` and
`TimeThiefExecutor`. It also emitted continuous deck-tracking mismatches. All
subsequent jobs use `AI_<name>` for the environment and `<name>` for WindBot.

### Terminal TCP reset was classified as a failed duel

WindBot sometimes closes its socket with `ConnectionResetError` after receiving
the terminal WIN packet. The duel had already produced a valid episode and
summary, but `LegacyChainProxy` retained the reset as an error and `eval.py`
exited with code 1.

`ygoai/windbot_protocol.py` now treats this specific reset like EOF. A premature
disconnect still causes the duel worker itself to fail, so the supervisor
continues to classify real mid-duel failures as invalid.

### Replay bundles lacked structured model decisions

`scripts/eval.py` can now write an optional decision JSONL stream containing:

- acting player, step, turn, and phase;
- legal-action features and selected action;
- raw policy logits and normalized probabilities;
- critic state value `V(s)`;
- checkpoint path and SHA-256;
- terminal reward, result, and win reason.

`V(s)` is not a per-action Q-value. The PPO model has no Q head.

`scripts/eval_windbot.py --decision-logs` assigns a separate decision file to
each supervised attempt. `scripts/eval_battle.py` records the equivalent data
for checkpoint-versus-checkpoint duels.

## Validation results

### OldSchool regression

After both fixes, Stage 3 played 20 alternating-seat games against OldSchool:

- valid: 20/20
- invalid: 0
- Stage 3 result: 20 wins, 0 losses
- timeouts: 0
- remaining processes: 0
- non-empty WindBot stderr logs: 0/20
- average length: 96 steps (range 51-172)

This is a pipeline regression benchmark, not a strong-opponent estimate.

### Stronger WindBot decks

Each opponent was run for ten games, five in each Stage 3 seat:

| WindBot executor | Environment deck | Stage 3 result |
| --- | --- | ---: |
| Labrynth | `AI_Labrynth.ydk` | 10-0 |
| Kashtira | `AI_Kashtira.ydk` | 9-1 |
| Tearlaments | `AI_Tearlaments.ydk` | 10-0 |
| Voiceless | `AI_Voiceless.ydk` | 10-0 |

All 40 attempts were valid, with no non-empty WindBot stderr logs, protocol
errors, timeouts, or remaining processes. The frozen code list contains every
card used by these four WindBot decks.

Kashtira produced the only Stage 3 loss. It is reproducible with seed
`20261203`, Stage 3 in player seat 1. The duel lasted 80 model decisions and
ended because Stage 3 could not draw a card. The recorded run translated 66
legacy protocol messages and contains a playable replay, complete engine log,
and structured decision log.

## Artifact locations on home

- OldSchool bundle:
  `D:\workspace\ygo-agent\training-runs\stage3-windbot-eval-20260921.zip`
- Stronger-deck bundle:
  `D:\workspace\ygo-agent\training-runs\stage3-windbot-strong-decks-20260921.zip`
- Reproducible Kashtira loss:
  `D:\workspace\ygo-agent\training-runs\stage3-windbot-kashtira-loss-replay-20260921`

The stronger-deck bundle SHA-256 is
`e0b74f66fcc56258d2a87701e1a5ca343d40ab395aaf0446e85497678532e613`.

## Verification performed

- Python bytecode compilation succeeded for the changed evaluator modules.
- Structured JSON and JSONL artifacts were parsed after download.
- Every decision record had equal legal-action, logits, and probability counts.
- The corrected proxy completed a 20-game OldSchool batch with 20/20 valid
  results after the previous build produced terminal-reset false failures.
- Four additional two-seat smoke games and 40 stronger-deck games completed
  under the Linux process supervisor with complete cleanup.

The home virtual environment does not currently include `pytest`; validation
therefore used syntax checks plus end-to-end supervised duel batches.
