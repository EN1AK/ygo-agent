# Frozen SkyStriker Candidate-Q start

The user approved freezing the 121,208,832-step actor and fitting only a
separate observation-only critic before any matched 5M continuation. This is
not full centralized VRPO, search, or an actor-strength improvement.

Implementation source: `356c6013bf249234a8f73f0cb2448eeac4da017d`.
Parent SHA-256: `cde8e3f6a06000319cbe3fe59f6112b2ffa37f5ece03b2f05b1ef06f16a03649`.
Native SHA-256: `a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`.
Runtime: immutable `skystriker-place-forward-505b855-20261003` assets, isolated
source at `/home/ygo/ygo-agent/training-runs/frozen-q-356c601/source`.

## Verified preflight

`/home/ygo/ygo-agent/training-runs/skystriker-frozen-q-preflight-356c601-20261003`
completed two 16-transition real-engine GPU updates. Q metrics were finite,
menu mismatch count zero, and the complete checkpoint contained 1,090 finite
arrays. Actor arrays exactly match the 121M parent. Serialized actor parameters,
batch statistics, optimizer and step stayed identical; optimizer step remained
zero. Q checkpoint SHA-256:
`c2a3579f651e534abae5d80eae813e29622cd208360d506377082db190142e26`.
Eight focused unit tests passed; wrong actor mode and >262,144 transition
requests were explicitly refused. Startup no-completed-episode average-return
NaN is an empty statistic, not a finite-Q failure.

## Pilot started, not yet accepted

Run: `/home/ygo/ygo-agent/training-runs/skystriker-frozen-q-262k-356c601-20261003`.
Budget 262,144 critic transitions, fresh seed 41032032, 2 actors x 16 envs,
4 environment threads/actor, horizon64, batch2048, 16 minibatches, Q LR1e-4,
128 channels, 2 layers. Both seats use the frozen stochastic self-play policy;
PPO update is skipped entirely. Q optimizer starts fresh, including after the
separate preflight. Checkpoint names count critic data, not extra actor training.
The legacy `actor_update` progress label means collection/update iteration here;
`frozen-actor-proof.json` is authoritative for actual actor update count zero.
Zero PPO loss fields are placeholders because PPO loss is not evaluated.

The selected five loss roots (four independent duels) are diagnostic-only and
never loaded by this fitting run. New self-play is not the argmax-versus-WindBot
distribution: diagnostics measure transfer, not unbiased WindBot-Q calibration.
Three loss/loss counterfactuals have no established long-term ranking; the two
resource roots come from one duel and cannot be split across fit/validation.
Evaluator shaped rewards are not used as fitting labels. Keep task2.4 open:
fresh held-out return calibration, prompt/menu ranking, throughput/memory and
numerical gates are still required before Q-boost or 5M training.

Native modules, production release, original actor and user BO3 edits were not
changed. Local/GitHub/home received implementation source; H200 remains pending
while unavailable. Run scripts and per-run manifests are separate artifacts.
