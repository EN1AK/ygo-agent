# 100M baseline progress and WindBot collection repair

The 100M PPO endpoint is complete, but baseline task 1.1 and VRPO experiments
are **not** complete. Training source remains
`92f225c3c2f7a7904ccc29958d1b8fb6db5c118e`; endpoint weight SHA-256 is
`aeca25126229eff22049a81087dd8db0c049f84ad01bd4595757b79fdfa9e64a`.

## Finished fixed-schedule evaluations

The greedy and historical runs used source `727a300`, native SHA
`5962816be0362007c751c43a515d18a50abcd7716c00be579aac0aba1e6705a8`,
and procedure SHA
`c2742f65f750092ef4298889391bc718c52ece5fad8d07c5a6e2ed69e0fe1322`.
No search, belief, or cycle intervention was enabled. Each batch collected
every environment's first duel; both seats used the same seed/environment map.

| Evaluation | Attempts | Wins | Losses | Invalid | Valid-game win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100M vs greedy, four decks | 1024 | 1005 | 18 | 1 | 98.24% |
| full40M vs greedy, identical schedule | 1024 | 1011 | 13 | 0 | 98.73% |
| 100M vs Stage3 10M, elfnote | 256 | 225 | 30 | 1 | 88.24% |
| 100M vs full40M, elfnote | 256 | 171 | 84 | 1 | 67.06% |

Historical matrix seeds were 42001 and 42002, 64 environments per seat/seed.
Each opponent first passed a separate four-attempt two-seat smoke at seed
52001. Invalid games stay outside wins/losses; their exact records remain in
the JSON evidence. This is evidence of direct-match gains on this schedule,
not a feature-only ablation, broad combo mastery, or a completed WindBot gate.

Evidence under `training-runs/` (local and H200):

- `ppo100m-historical-paired-727a300-20261002.tar.gz`, SHA-256
  `ed9a7ce200b783aec700d86c466ca0a54e4465c40bf8f1677ce8475ca40312dd`.
- `ppo100m-greedy-reviewed-20261002.tar.gz`, SHA-256
  `e3b0d0d808cb79e068a00f39f61f976255d913ce82389f179861019ede873b9d`.
  Four preselected examples each contain YRP, complete engine log, per-decision
  logits/probabilities/V JSONL, and Chinese commentary; also copied to home.

The elfnote first-seat example summons Baronne on total turn 9, then negates
its own graveyard tuner trigger (engine.log 2277–2316). The Star Seraph
second-seat example chains Zeus to its own board wipe (1120–1200), spending
additional materials. These legal but strategically weak actions remain
visible; do not mask them to inflate the baseline. The predetermined WindBot
strategy review is still pending and is not replaced by these examples.

## WindBot evaluator repair before its new runs

The first-episode collector added `environment_index=0` to structured terminal
lines. The supervisor's end-anchored legacy regex rejected that field, which
would misclassify a normally completed current worker as missing an episode.
Accept the optional field but require index zero for a single worker. For
structured records also require `termination_reason=1` and `invalid_game=0`;
legacy logs without those fields retain their previous behavior.

`eval_windbot.py --isolate-replays -- --record ...` now opts into per-attempt
working directories, so native relative `replay/*.yrp` paths cannot collide
between attempts or be confused with earlier runs. Pass absolute asset paths
in this mode. Default non-opted-in working-directory behavior is unchanged.

Validation: six parser tests pass on local Windows; all 13 tests pass in an
isolated H200 Linux source tree, including timeout/abort/interruption cleanup,
child exception rejection, and two workers producing the same replay filename
without overwriting one another. Real WindBot smoke remains the next gate.
No native module, actor input, learned weight, or training algorithm is changed.

Remaining: WindBot four-executor smoke then declared main seed blocks; explicit
old lite40M capacity migration validation; representative historical replays;
predeclared combo/interruption review; then matched 100M-start Q experiments.
