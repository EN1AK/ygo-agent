# Bounded shadow-Q start gate (2026-10-03)

User request: proceed until training starts, following the already reduced
WindBot budget. This is an observation-only shadow critic, **not** centralized
VRPO or a Q-boosted actor. PPO still uses GAE; Q/search/belief do not select actions.

## Frozen baseline

H200 manifest: `/root/ygo-agent-gpu-20260910/training-runs/shadow-start-gate-20261003/baseline-manifest.json`.
All 330 referenced artifacts were hashed, including the immutable 100M endpoint
and base release, corrected native/procedure, corrected replay bundle, strategy
records, historical matrices, and corrected greedy controls. Original training
source remains `92f225c`, endpoint SHA `aeca25126229eff22049a81087dd8db0c049f84ad01bd4595757b79fdfa9e64a`.
New runtime native is `0f686f233d00115a99bcf8a5bee5da0d0d86e7a914e6cd89bae0a8980cb5f2f4`;
procedure overlay remains `c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322`.

Corrected greedy controls use the same four decks, two seeds, both seats and
64 distinct first episodes per cell (1,024 attempts per checkpoint):

| Model | Wins | Losses | Invalid/max-step | Valid win rate |
| --- | ---: | ---: | ---: | ---: |
| 100M | 986 | 37 | 1 | 96.38% |
| full40M | 988 | 36 | 0 | 96.48% |

Evaluation source was `4d34d5d`. Logs and four replay sets per model are under
`training-runs/shadow-baseline-corrected-4d34d5d-20261003/` on H200.
Historical model/WindBot results are retained as old-observation evidence, not
causal controls for the repaired environment or the upcoming algorithm trial.
The 100M training's last SPS was 5,811; its 37 max-step messages have no episode
denominator and cannot be substituted for an evaluation timeout rate.

## Bounded strategic review, with uncertainty retained

`training-runs/shadow-strategy-review-20261003.json` on H200 records the previously
selected first paired seed of each of Kashtira, Labrynth, Tearlaments, Voiceless,
and SkyStriker: ten complete games, five candidate-first turns, no exclusions,
no max-step endings, no wall timeouts and no interventions. The corrected bundle
contains all ten YRP/engine/decision/Chinese commentary sets.

- First-turn Crystal Wing/Baronne presence and available target interruption:
  0/5 observed first turns (0/4 for Elfnote only; the fifth is SkyStriker).
  This is **not** a conditional feasible-combo completion rate. No missing
  forced combo is inferred solely from card names.
- Ten first tuner-access/material decisions retain the exact offered options,
  choice, subsequent eight decisions and a human review note. Elfnote first
  turns versus Labrynth and Voiceless visibly convert Prison Tuner into a
  Synchro/search/Regina sequence; versus Tearlaments the observed post-summon
  menu only permits end phase, so continuing cannot simply be assumed legal.
- All 495 model chain-choice prompts are enumerated with effect, controller,
  offered choices, subsequent target prompts and raw context line ranges.
  One own-turn Raye-to-Shizuku resource sequence is defensible; 494 choices
  remain tactically uncertain. Passing or activating is not automatically a
  successful interruption. There is **no established aggregate good-interruption rate**.
- Preserve separate non-chain policy errors: Veiler's lethal self-damage attack
  versus Kashtira, and Chaos Angel choosing own Regina versus Labrynth.
- The earlier greedy notes still expose self-negation/self-chaining; they are
  explicitly old-runtime evidence, not silently merged into corrected rates.

These disclosed diagnostic measurements satisfy only the user-approved bounded
pilot-start gate. They do not demonstrate strategic quality. More confident
labels and matched held-out calibration remain required for promotion.

## Shadow implementation and preflight

The independent observation-only encoder emits acting/opposing Q channels over
the exact staged menu. Frozen collected actor probabilities and pre-update Q
form two-seat Expected-SARSA trace targets. Only Q receives this regression
loss. Strip every Q-specific rollout field before the original PPO/GAE update.
Store the separate optimizer in the versioned Q envelope alongside the normal
actor-only checkpoint; incompatible resume or nonfinite values fail closed.

The bounded integration currently supports self-play, one learner device,
equal collect/update horizons, and no hidden actor inputs. Unsupported modes
continue to refuse before collection. Invalid nontrainable game transitions
abort the shadow gate rather than borrowing unvalidated targets. Policy-caused
max-step loss retains the existing declared reward convention.

Home isolated CPU validation: 18 Candidate-Q math/model/checkpoint/default-path
tests plus one deterministic real-critic update test passed. Real-engine CUDA
preflight and actual pilot start are recorded separately when verified; this
report alone does not claim that a GPU pilot is running.
