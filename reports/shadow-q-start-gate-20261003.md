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

### First real-engine preflight caught a Q identity-helper error

`shadow-preflight-a460f38-20261003` stopped before any learner update because
the earlier helper treated equal feature rows as duplicate action identities.
The diagnostic-only `shadow-menu-probe-20261003/probe.log` reproduced three
legal same-CID deck choices with unbound physical references. The corrected
native deliberately does not invent a physical deck index; the response slots
remain distinct. Menu identity v2 hashes the prompt-local slot together with
features and retains every legal response. Duplicated *captured identities*,
changed menus/counts/selected indices still fail. This does not change actor
features, core legality, the native module, or the frozen baseline. New tests
cover same-feature slots and corrupted captured identities separately.

### Real GPU integration evidence and stricter parity gate

`shadow-preflight-f081fad-20261003` completed 30,720 learner steps with two
finite Q/PPO updates and zero menu errors. Actor checkpoint has 165 finite
leaves; the complete Q/optimizer envelope has 1,090 finite leaves. The second
Q update took 2.85 s (first compilation/update 65.02 s). Targets remained finite.
These are on-rollout TD calibration diagnostics, not held-out Monte Carlo quality.

An independently collected same-seed GAE run also completed but was **not**
byte/allclose identical: maximum parameter delta 0.00156. The existing actors
seed and shuffle a process-global NumPy RNG from multiple threads; same seed
alone does not freeze seat assignment. Do not report these runs as exact
same-rollout parity or waive the failed comparison. Matched pilot commands now
opt into `--actor-seat-seed-mode per_actor` in both arms; the unflagged default
keeps shared legacy behavior. Additionally `--q-verify-actor-parity` performs
the identical PPO update from the same collected batch, initial state and key
before and after independent Q training; every output/optimizer leaf must be
exactly equal. This isolates actor contamination from fresh rollout variability.
The bounded manifest also caps shadow training at 1,013,760 new steps, and any
actor optimizer nonfinite counter now aborts shadow mode.

The first strict same-batch GPU run (`shadow-parity-preflight-1a070d8-20261003`)
also failed byte equality, so it was not promoted. The installed XLA explicitly
supports `--xla_gpu_deterministic_ops=true`; matched preflight/pilot launchers
now declare this flag for **both** arms. The trainer respects explicit XLA
flags before backend initialization, retaining its legacy fallback when unset.
This is a numerical reproducibility test, not permission to tolerate a Q-induced
actor change. Passing results must be recorded before starting the bounded pilot.

### Deterministic isolation passed; bounded pilot launched

The full-batch deterministic preflight was diagnostically interrupted after
over 15 minutes without a learner update. The saved GDB main-thread stack is
in `cuLaunchKernel` under nested XLA `WhileThunk` execution, **not compilation**.
This is an intentionally stopped diagnostic run, not a spontaneous crash.
Do not claim its failed/unfinished gate passed or that ordinary GPU runs are
bitwise deterministic.

`shadow-small-deterministic-5e2708f-20261003` then completed two real-engine
16-transition updates with deterministic GPU operations: two elfnote mirror
environments, 8 steps, one minibatch, original 100M weights and a fresh sampler.
This is an isolation fixture, not the corpus continuation or a matched strength
arm. All **679** same-batch PPO output/optimizer/key leaves before and after Q
training are exactly equal (maximum absolute difference **0**). Both Q updates
and actor optimizer states are finite; no menu mismatch. Final on-rollout
TD-target RMSE is 0.4640, not a held-out performance estimate.

The fixture actor checkpoint has 165 finite leaves and SHA-256
`5334df2f16bf1b71a3f656a3696432782eaeaf66ff9a18e2ba503b20cff0ba34`;
the complete Q envelope has 1,090 finite leaves and SHA-256
`380f049641c2609dd157d1d1d504ceb43ba74d4721b2dd1356b1ac4fb0ffc1d5`.
Both sidecars match; envelope actor equals standalone actor exactly.

The immutable pilot release is
`dist/runtime-releases/shadow-pilot-5e2708f-20261003`, manifest SHA-256
`cad17e4a6415d18b9e061807dfffd913a6e778c8bd83d7678f8865f756a810c3`.
It retains source, corrected native, dependency manifests, procedure overlay,
fresh dual-schema self-test, baseline hashes, launch/config and parity evidence.
It depends on the preserved fusion58m and ppo100m releases; restore their assets
and overlay in the declared order. The source is committed `5e2708f...`, not
the unrelated local BO3 working-tree changes.

The launched H200 run is
`training-runs/shadow-observation-100m-plus1m-5e2708f-20261003`.
It imports the **original** 100,003,840-step actor and its corpus sampler counters,
not either diagnostic checkpoint, and targets **101,017,600** after **1,013,760**
new steps. Five actors x 48 environments, 11 env threads/actor, batch 15,360,
80 minibatches, actor/Q LR 1e-4; both optimizers start fresh. Per-actor seat RNG,
no UPGO/concurrency, Q advantage/search/belief/privileged observation off.
Q is independently optimized; GAE alone updates PPO. Checkpoint interval is
8 learner batches (122,880 steps), plus final save.

The production pilot uses ordinary GPU operations for throughput; the strict
bitwise test is the separate bounded deterministic fixture, not a claim about
the entire pilot. Its launcher refuses to run without the completed same-source,
same-native, same-parent exact-parity proof. Task 2.4 held-out calibration and
all actor-Q-boost/centralized/promotion gates remain closed. Starting this run
is **not** starting full centralized VRPO or evidence of improved playing strength.
