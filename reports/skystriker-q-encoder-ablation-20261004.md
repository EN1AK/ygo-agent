# SkyStriker Q encoder initialization ablation

## Outcome

User-approved matched-budget experiment completed on home. Copying the frozen
121M actor encoder improved chosen-action return prediction on the development
validation set, but did **not** solve the selected critical-action ranking
failures. Do not enable Q-boosted actor updates or the 5M arm on this evidence.

| Same-observation comparison | Scratch Q | Actor-encoder initialized Q |
| --- | ---: | ---: |
| Chosen-action return RMSE | 0.953974 | 0.909876 |
| Replace chosen Q by uniform menu-mean Q: RMSE | 0.954370 | 0.910317 |
| Median menu max-minus-min Q | 0.026481 | 0.024915 |
| Selected win/loss branch-root order agreement | 1/2 | 1/2 |

Both networks scored exactly the same 22,519 decisions from 64 natural first
episodes (zero invalid), seeds 61042001 and 61042002, 32 environments each.
Equal-game mean warm-minus-scratch MSE was -0.0744635, with paired game bootstrap
95% interval [-0.1115973, -0.0398423], 10,000 draws, bootstrap seed 61042003.
This supports improved calibration on these games, not game-strength or reliable
action-ranking improvement. Almost unchanged error under the menu-mean ablation
still suggests that most predictive information is state-level.

## Controlled implementation and verification

- Training source: `e66a427bf1a73886215a769995dc71b2f585b29d`.
- New opt-in `--q-encoder-init actor` requires frozen shadow mode, compatible
  encoder configuration and an actor checkpoint. Default initialization remains
  random. Parameter tree, shape and dtype mismatches fail closed.
- All 135 encoder arrays (15,724,296 parameters) copied exactly. Q output layers
  remain random; critic optimizer is fresh. Q encoder remains trainable, its
  parameters independent from the actor. The LSTM and policy/value heads are
  not copied into Q. Initialization provenance participates in resume validation.
- Two new helper tests plus eight existing focused tests passed; real-model
  import/inference proof confirms exact encoder import, unchanged Q head and
  actor bytes. Independent proof initially needed NumPy-to-JAX conversion;
  this was a verification-script issue, not a training change.
- Real-engine preflight: two 16-transition batches, finite Q, actor frozen.
- Full run: 262,144 transitions / 128 updates, 2 actors x 16 environments,
  4 environment threads/actor, rollout64, batch2048, 16 minibatches, seed41032032,
  Q LR1e-4, gamma1, lambda0.95. Normalized commands match the scratch control
  except initialization and run/provenance paths. Duration385.10 seconds;
  do not attribute runtime differences to initialization from a single run.
- All128 Q updates finite, zero menu mismatches; terminal envelope has1,090
  finite arrays. Actor parameters, optimizer and step remain byte-identical,
  actor update count zero. Final TD RMSE0.12839 is not Monte Carlo calibration.

Frozen actor SHA256:
`cde8e3f6a06000319cbe3fe59f6112b2ffa37f5ece03b2f05b1ef06f16a03649`.
Native SHA256:
`a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`.
Warm terminal checkpoint SHA256:
`92c45d476fa8c1b4593c3d9382dffea51b2e53fc7082cdd4cf957d334ed81895`.
Scratch terminal checkpoint SHA256:
`a903bc2b7e0ba7004ccbea15de044b1bae0834256a75157e923de9c63a048567`.

## Critical roots and interpretation

Alternative-minus-original Q at the two roots from old attempt23:

| Root | Observed winning continuation alternative | Scratch gap | Warm gap |
| --- | --- | ---: | ---: |
| step189 | Enter Battle instead of activating Area Zero | +0.045188 | -0.066372 |
| step190 | Select Solemn Warning instead of Bomber | -0.002741 | +0.000971 |

Warm-start fixes the second ordering only by a tiny margin and reverses the
first ordering in the wrong direction. These are two correlated states from
one duel with one seeded continuation per branch, not known expected-Q labels.
The other three roots had loss/loss continuations and are not scored as ranking
successes or failures. All five roots remain excluded from fitting.

The training distribution is frozen stochastic self-play; these selected roots
are argmax-versus-WindBot transfer diagnostics. Reusing a validation set that
motivated the ablation makes it development evidence, not untouched final test
evidence. Only one training seed was used. Equal configuration/seed does not
prove identical fitting transitions: full training rollouts were not retained.

## Pairing correction, not hidden replacement

The first warm evaluation completed64 natural games but produced22,465 decisions,
whereas the earlier scratch evaluation had22,653. Exact trajectory pairing
therefore failed and no paired interval was claimed for those separate runs.
Same seed was insufficient to prove identical trajectories; the precise source
of divergence is not established here. Preserve this attempt under
`skystriker-warm-q-validation-20261004` and the evaluation launcher log.

The accepted comparison evaluates both critics on the **same live observations**
in one frozen-policy collection. Neither Q chooses actions or changes rewards.
Per-step decision records contain both complete Q menus and the shared chosen
action/return. No fitting or checkpoint selection used these new outcomes.

## Artifacts and next gate

Home root: `/home/ygo/ygo-agent/training-runs/`.

- Training: `skystriker-warm-q-262k-e66a427-20261004/`.
- Terminal: training directory plus
  `checkpoints/41032032_step_000000262144.flax_model.candidate_q`.
- Paired evaluation: `skystriker-warm-q-validation-paired-20261004/`.
- Evidence bundle: `q-encoder-ablation-evidence-20261004.tar.gz`, SHA256
  `53902844dabd467de6a5e1ea14e13e12fefb25f14462ca28eebb7618bc6add04`.
  Contains37 hash-indexed evidence files: logs, manifests, per-decision Q scores,
  paired statistics, scripts, original unpaired attempt, initialization proof,
  runtime verification and terminal sidecar. Checkpoint binary remains on home.
- Local bundle/extraction are under this repository's `training-runs/`.

Retain warm initialization as a promising starting point for further critic
experiments, not a promoted policy. Next isolate supervision: evaluate a bounded
terminal-return versus bootstrapped-target ablation on one cached data split;
collect independent multi-continuation action comparisons for ranking validation.
Keep current diagnostic roots out of fitting and do not stack memory/search
changes into that comparison. This next experiment has not been launched.

Implementation synchronized local/GitHub/home. H200 deployment remains pending
while unavailable; no claim of four-end consistency. User BO3 changes,
TRAINING_PLAN and native/runtime assets were left unchanged.
