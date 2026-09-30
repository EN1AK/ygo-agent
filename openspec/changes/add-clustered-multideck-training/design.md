# Design

## Context

See `proposal.md` for motivation. The current repository contains 37 top-level
YDK files and older multi-deck validation artifacts, but the production trainer
accepts fixed `deck1`/`deck2` names or native uniform `random` selection. Native
random selection samples the global deck list directly and cannot express
cluster-first probabilities, elfnote reserve, family de-biasing, or exact sampler
resume.

The historical reference checkpoint is the immutable 40M Structured-lite/full artifact with
SHA-256 `a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`.
It was trained primarily on elfnote and has fixed code-list and semantic hashes.
On 2026-09-26 the user selected random initialization for the new multi-deck
policy. The migrated 40M artifact remains a verified optional path, but the
100k smoke and 5M pilot do not load it.
The existing `TRAINING_PLAN.md` Stage 6 describes an elfnote specialist; this
change intentionally introduces a later universal-policy branch rather than
silently changing that specialist experiment.

## Goals / Non-Goals

**Goals:**

- Acquire hundreds of auditable YDK lists, reduce duplicate/archetype-size bias,
  and freeze a reproducible clustered corpus.
- Train a Structured-lite/full policy from random initialization across all
  ready decks while retaining elevated elfnote exposure.
- Make sampling exact enough to resume and compare after GPU or environment-count
  changes.
- Produce per-cluster observability and held-out generalization measurements.
- Make protocol support a source-derived, testable contract rather than relying
  on sampled training duels to discover unsupported engine branches.

**Non-Goals:**

- Claim current tournament legality or reproduce side-decking/best-of-three.
- Use deck IDs, archetype labels, or private deck contents as policy inputs.
- Train against latency-bound WindBot inside the rollout loop.
- Treat arbitrary downloaded lists as ready without runtime validation.
- Replace the original 40M checkpoint or existing specialist reports.

## Decisions

### 1. Corpus acquisition is registry-driven and append-only

Use a source registry with adapters for repository directories, pinned git trees,
and explicitly hashed archives. Every imported file is copied into a new corpus
revision, never directly merged into `assets/deck`. The initial implementation
targets at least 256 canonical ready decks; a pilot may proceed with at least 64
ready decks across at least eight accepted clusters.

This is preferred over manually copying a large folder because a raw directory
cannot explain provenance, licensing, later source drift, or why a deck was
excluded.

### 2. Canonical identity separates exact duplicates from families

Parse YDK zones into sorted `(card_code, count)` tuples. Hash all three zones for
exact identity. Compute near-duplicate similarity from main+extra vectors and
group highly similar lists into a family. Retain every source record, but expose
one canonical list per exact hash and make family selection uniform before list
selection.

This avoids a popular archetype receiving extra training weight merely because a
source contains many one-card variants. A simple filename/archetype-label grouping
was rejected because labels are missing, inconsistent, and easy to leak into the
policy design.

### 3. Cluster on zone-separated TF-IDF card composition

Construct an L2-normalized sparse vector with separate main- and extra-deck
coordinates, count-preserving term frequency, and corpus inverse document
frequency. Extra-deck coordinates receive a recorded moderate weight because
they are compact and strategically informative; side decks are excluded in v1.
Cluster canonical family representatives by cosine distance using deterministic
average-linkage agglomerative clustering. Select the cut from a bounded candidate
range using silhouette score plus hard minimum/maximum cluster-size and dominant-
cluster constraints. Store the dendrogram/candidate scores, final cut, medoids,
and assignments.

TF-IDF prevents ubiquitous staples from dominating. Agglomerative clustering is
chosen over filename labels and raw K-means because it supports cosine/card-overlap
geometry, is deterministic at the intended corpus size, and leaves an auditable
hierarchy. If the corpus grows beyond the measured memory budget, the same feature
contract can use a deterministic approximate first stage followed by exact
within-partition refinement, but that is not part of the initial implementation.

### 4. Reserve 25% elfnote probability per seat, then sample clusters

For each seat independently, the initial static curriculum is:

1. choose elfnote with probability `0.25`;
2. otherwise choose a non-anchor cluster uniformly;
3. choose a near-duplicate family uniformly inside the cluster;
4. choose a canonical deck uniformly inside the family.

Seats are counter-balanced in deterministic audit batches. Independent per-seat
sampling means about 43.75% of games involve elfnote and 6.25% are elfnote mirrors,
while about 25% of player decisions originate from elfnote before episode-length
effects. Reports show both requested probabilities and realized game/decision
fractions. The reserve is configurable but changing it creates a new curriculum
revision.

Uniform raw-deck sampling was rejected because large clusters and duplicated
families dominate. Forcing 20% elfnote mirrors, as in the specialist plan, was
rejected for the universal-policy branch because it would use too much game budget
on a single matchup; the per-seat reserve gives high retention plus cross-deck
elfnote games.

### 5. Native reset consumes a frozen weighted-sampling manifest

Extend the environment configuration with a deck-sampling manifest and sampler
seed/state. The native reset path selects cluster/family/deck for both seats and
emits stable deck, family, and cluster IDs in `info`. Python owns manifest
construction and validation; native code owns low-overhead per-reset draws.
Checkpoint/run state records the sampler counter/state needed for exact resume.

Spawning separate environment pools per cluster was rejected because it wastes
threads, complicates dynamic proportions, and makes exact global sampling and
resume difficult. Replicating YDK files to approximate weights was rejected
because it corrupts provenance and cannot de-bias families cleanly.

### 6. Versioned initialization and append-only compatibility migration

Validate the whole ready corpus against the current code list before training. If
new codes are required, preserve every existing row index and append new sorted
codes. Regenerate semantic assets for that exact list. For the initial pilot,
initialize all model parameters with a recorded seed and the frozen feature
schema. Keep the 40M migration tool as an optional later continuation path:
copy non-embedding parameters and old rows exactly, initialize only appended
rows with a pinned seed, and emit source/output hashes and compatibility tests.

Reordering the code list or rebuilding all embeddings was rejected because it
would destroy the learned 40M card identity mapping.

### 7. Universal self-play with controlled opponent mixture

Both seats use the shared current policy so all ready decks can generate learning
transitions. After the first finite pilot, add compatible historical snapshots and
the in-process greedy bot as bounded opponent components, recorded separately from
deck sampling. WindBot remains evaluation-only. No deck identifier enters model
observations.

The first pilot keeps the opponent policy mixture simple so failures can be
attributed to deck distribution rather than simultaneous curriculum changes.

### 8. Static pilot before adaptive reweighting

Begin with the frozen 25% anchor/equal-cluster distribution. Adaptive difficulty
weighting is deferred until static coverage and per-cluster evaluation are stable.
If later enabled, it must retain an explicit probability floor for every cluster,
cap per-revision weight changes, and create a new curriculum manifest.

This avoids feedback loops where noisy early losses cause one cluster to consume
the run or easy clusters disappear before the model learns basic play.

### 9. Derive the protocol contract from the pinned core

The native extension is built against ygopro-core revision
`f96929650ff8685b82fd48670126eae406366734`. Treat that source, especially its
`common.h` message definitions and `playerop.cpp` message writers/response
validators, as the protocol authority. Generate and freeze a machine-readable
inventory containing every `MSG_*` number, notification/interactive class,
payload layout, response representation, and meaningful parameter branches.

The core does not directly expose policy-ready legal-action tensors: it writes a
binary request and later validates bytes or an integer response. Therefore the
environment still owns deterministic legal-action enumeration, but every exposed
action is checked against a core-derived response oracle. Reference client code
may clarify presentation semantics, but cannot override the pinned core.

Protocol validation has three layers:

1. static inventory parity between the pinned core and adapter;
2. deterministic fixtures for every core-emitted notification payload and every
   interactive writer/validator branch, including concatenated notification
   frames, boundary counts, masks, optional selections, weighted sums, sorting,
   and arbitrary announced values;
3. strict observation/action tensor-domain checks before inference and before a
   transition enters the learner, with raw protocol diagnostics on failure.

Sampled deck smoke remains useful for integration coverage, but is not evidence
of protocol completeness. Any `unsupported`, `not implemented`, silent truncation,
or out-of-domain model field blocks the pilot.

### 10. Gates and default budgets

Use sequential gates rather than immediately launching a long run:

- sampler-only audit: at least 100,000 deterministic draws;
- environment audit: every ready deck loads and completes both-seat smoke games;
- protocol audit: pinned-core inventory parity, branch fixtures, response-oracle
  acceptance, and model tensor-domain checks;
- post-protocol training smoke: 100k steps from random initialization with finite
  optimizer and save/reload;
- pilot: 5M steps from random initialization with frozen sampling and an action
  capacity of 128, after observed valid 25-action menus exceeded the old 24-slot
  limit;
- evaluation: protected elfnote suite plus per-cluster and held-out matrix;
- long run: budget selected from pilot SPS, memory, and learning curves.

The pilot checkpoint is not automatically the new baseline. Promotion requires
zero protocol/domain violations and invalid games, finite learning and inference,
working checkpoint reload/resume, exposure of every training cluster, and a
reported held-out evaluation. A fresh 5M policy is not gated against the mature
40M elfnote specialist's win rate.

### 11. Forced termination and deterministic baselines are explicit

Keep the default decision cap at 1000 for now: it counts external policy
decisions, not turns or raw engine messages, and normal historical evaluations
finish far below it. Reaching the cap is therefore diagnostic evidence of a
stalled duel, not a legitimate life-point adjudication. The environment returns
done with zero reward, marks the game invalid, records a max-step termination
reason plus decision/turn counters, and the learner masks that transition.
Runtime timeouts use a distinct termination reason. Evaluation reports natural,
max-step, and timeout terminals separately and must not include forced terminals
in a win-rate denominator.

The old `GreedyAI` selected legal action zero regardless of meaning. Preserve
that behavior only as the explicitly named `first` diagnostic baseline. The
`greedy` baseline deterministically ranks semantic actions, prioritizing attacks
and phase progression over setup actions and cancellation. This baseline is
intentionally simple, but its name now describes a policy rather than an array
ordering accident. Raising the cap is only a comparison diagnostic after these
semantics are fixed; it is not the primary remedy for non-progress loops.

### 12. Detect policy cycles before applying explicit, auditable assistance

Deterministic argmax can repeat a reversible action pair even when every engine
message and response is valid.  Fingerprint the current public observation and
bounded legal-action menu while deliberately excluding rolling action/event
history, then pair that state fingerprint with the raw selected action.  A
repeat within a bounded period is a policy-cycle signal, not a protocol error.

Raw evaluation remains the default and never changes the selected action.  An
opt-in cycle guard may reject the repeated raw choice and select the next-ranked
legal action, but every intervention records the state fingerprint, cycle
period, raw action, replacement action, logits, and trigger configuration.
Reports keep raw and assisted results in separate named cohorts.  Cancellation
is recorded in the model-visible action history so continuation training can
learn that the immediately preceding optional selection was abandoned.

The existing deterministic replay-snapshot search is the first search backend.
Cycle events may be exported as replayable decision points for offline bounded
search and distillation.  Online search is not claimed until restore identity,
recurrent-state handling, compute budget, and chosen-action audit fields pass a
fixed-seed gate.  Raising the decision cap or silently changing argmax is
rejected because either hides rather than diagnoses the learned loop.

Protocol Cancel responses are not all policy actions. `MSG_SELECT_CARD`,
`MSG_SELECT_TRIBUTE`, and `MSG_SELECT_UNSELECT_CARD` use Cancel as UI-style
navigation back to the parent prompt. When one of these prompts has a selection
or Finish alternative, suppress that Back action only at the policy boundary;
keep its parser and response callback covered by the protocol suite. Do not
suppress chain pass, yes/no or effect rejection, or staged Finish, because those
are semantic decisions. This removes a dominated reversible edge without
changing tensor shapes or checkpoint compatibility and is preferred to asking
PPO to learn around a protocol-navigation artifact.

## Risks / Trade-offs

- **[Downloaded deck quality and legality vary]** → Pin sources, quarantine failed
  lists, preserve validation reasons, and make no tournament-legality claim.
- **[Card-overlap clusters may not equal playstyle clusters]** → Store medoids and
  quality metrics, manually audit representatives, and later compare behavioral
  embeddings without changing the v1 contract silently.
- **[A fresh policy may initially be weak on elfnote]** → Use the 25% per-seat
  reserve and fixed-seed evaluation; compare with historical 40M only as context.
- **[New cards invalidate checkpoint compatibility]** → Require append-only
  migration and refuse rollout on any unexplained hash mismatch.
- **[Some decks are unlearnable under the shared policy initially]** → Stage the
  pool, retain cluster floors, monitor legal-action/timeout metrics, and quarantine
  runtime failures rather than treating them as losses.
- **[Equal clusters can overweight tiny outlier clusters]** → Enforce cluster-size
  gates and family counts during cut selection; report both macro and micro scores.
- **[Exact resume can change when actor count changes]** → Record the hardware and
  actor mapping. A changed topology starts a linked run with a new compute manifest;
  it preserves distributional reproducibility even when bit-identical ordering is
  impossible.
- **[Sampled duels miss rare protocol branches]** → Generate coverage from the
  pinned core writers and response validators, require boundary fixtures, and
  treat duel coverage only as an integration supplement.
- **[A known notification name can hide a missing parser]** → Distinguish the
  diagnostic `msg_to_string` inventory from real `handle_message` branches,
  require a handler for every direct core writer, and forbid whole-buffer skips
  because the core may concatenate multiple notifications.

## Migration Plan

1. Freeze the current 40M checkpoint and all current hashes as reference data;
   do not modify it.
2. Build corpus revision 1, validate it, and freeze train/holdout assignments.
3. Freeze code-list and semantic assets for random initialization. Maintain the
   separately verified append-only checkpoint migration for optional future use.
4. Build and deploy the manifest-aware environment beside the existing runtime.
5. Generate and pass the pinned-core protocol inventory, response-oracle, branch,
   and model-input-domain gates.
6. Re-run deterministic sampler, smoke-duel, finite-forward, and save/reload gates.
7. Run the 5M pilot in a new run directory and evaluate protected/cluster suites.
8. Promote only the curriculum/checkpoint pair that passes the functional and
   evaluation gates. The immutable 40M source and old runtime remain available;
   no in-place state is changed.

## Open Questions

- The long-run step budget and actor/env topology will be chosen from the 5M pilot
  resource report because the user's available compute may change.
- External source registry entries and licenses will be finalized during corpus
  implementation; the behavior contract requires them to be pinned and auditable.
