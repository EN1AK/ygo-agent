# YGO Agent training plan

## 下一轮实验执行顺序（2026-09-13）

学习方正式名称为 **elfnote**。此前 `k9vs` 为误用的文件/运行标签，
实际牌表是耀圣诗／狱神相关卡组；本地使用 `assets/deck/elfnote.ydk`。
历史报告、检查点和运行目录保留原名以便复核，不能据目录名判定卡组。
下列步骤是后续执行计划，本次仅修改名称与计划，不启动新训练。
本节优先于下文旧版 Stage 6 的实验建议。

1. **锁定基线和输入资产。**
   - [ ] 核对 elfnote 牌表、卡库、脚本、code list 和嵌入映射的哈希，记录实际卡组组成。
   - [ ] 从 Stage 3 混合模型出发；本轮多卡组模型只保留为退化样本，不作为新起点。
   - [ ] 固化水晶翼误无效己方的完整观测序列和 Stage 3 动作概率，加入回归集。
   - [ ] 查清 WindBot 弱怪撞强怪与终局连接重置，可靠性验收前不把其对局当成高质量示范。

2. **先做 elfnote 同卡组自博弈恢复实验。**
   - [ ] 使用现有 mixed 路径，初始沿用 Stage 3 的 50% 当前策略、30% 历史策略、20% greedy 配比；
         对手确实加载策略，同卡组对 greedy 不得标为自博弈。
   - [ ] 固定 Stage 3 历史对手，保留后续冻结版本；先运行短 GPU 验证，再做约 1M 步有界实验。
   - [ ] 保持输入、模型结构和牌表不变；训练内不并发启动评估，训练后串行独立评估。
   - [ ] 与 Stage 3 对照固定局面、greedy、历史策略和可信 WindBot 对局；记录无效局与置信区间。
         水晶翼局面应保持取消为首选，整体表现不明显退化，才考虑扩展训练预算。

3. **单独验证显式连锁输入。**
   - [ ] 设计当前连锁的发动方、卡片、效果索引、连锁序号和已公开目标编码，明确隐私边界。
   - [ ] 版本化观测与检查点兼容方案；若需原生模块构建，先准备隔离方案并另行取得授权。
   - [ ] 使用相同起点、对手、预算和种子比较有无连锁输入；本实验不同时改变对手体系。
         不通过禁止无效己方的硬编码来掩盖策略错误。

4. **给异卡组分别培养可学习的对手。**
   - [ ] 从青眼等少量卡组试点，每个卡组配置自己的策略、牌表和检查点，不能直接把 elfnote 策略当作成熟异卡组对手。
   - [ ] 可先做各卡组同卡组自博弈；WindBot 行为验收通过后，再试用其轨迹做模仿学习初始化。
   - [ ] 实现按对手卡组加载冻结策略的采样路径，验证模型、座位和学习方掩码。
         当前 specialist 模式仅支持 bot，必须完成此实现才能声称启用了策略对手。
   - [ ] 交替更新各卡组策略，定期冻结版本加入对手池，保留旧版本和最低采样份额，防止互相适应后退化。

5. **回到多卡组联赛与晋级评估。**
   - [ ] elfnote 学习方对抗各卡组策略池，保留约 20% 同卡组对抗份额；记录实际对手模型版本。
   - [ ] 从均衡采样开始，稳定后再考虑困难对手加权；逐项改变实验变量。
   - [ ] 使用双方座位、配对种子，独立评估各卡组与受保护镜像对局；目标每模型/对手至少 256 个有效局。
         失败尝试单列，受协议异常影响的样本不计作胜利。
   - [ ] 总体提升且关键对局不明显退化才晋级；否则保留 Stage 3 并定位失败边界。

每轮保存：起点/终点检查点、优化器是否恢复、资产哈希、对手版本及配比、种子、
预算、逐卡组结果、无效原因和固定局面回归。所有训练继续使用现有 GPU 环境，
不因计划修改而安装依赖、重编译或更改服务器配置。

This is the default execution plan after the elfnote baseline. Follow the stages
in order and record configuration, asset revisions, checkpoints, metrics, and
conclusions for every experiment. Correct duel behavior and reproducible
evaluation take priority over increasing model size.

## Decisions and scope

- Keep the model's existing trainable card-ID embedding.
- Do not add external, card-text, or card-attribute pretrained embeddings.
- Use WindBot as a stronger, slower evaluation opponent, not as a rollout
  opponent for reinforcement-learning training.
- Use WindBot's supported `Normal` deck catalog to define the initial
  multi-deck matchup pool. The `.ydk` files define matchups; training opponents
  remain fast in-process policies or bots.
- Keep Chinese card assets as the default and pin every database, script,
  banlist, WindBot, and deck revision used for a result.

## Completed foundation

The following work remains the reference baseline rather than a future stage:

1. A stable 10,000,000-step elfnote baseline with finite optimizer metrics.
2. Greedy, random-initialized, and historical-checkpoint evaluation.
3. Mixed elfnote training with approximately 50% current self-play, 30%
   historical opponents, and 20% greedy opponents.

Preserve these checkpoints and reports as regression baselines. Do not compare
later multi-deck experiments only against their own training pool.

## Stage 4: Complete the WindBot evaluation adapter

Progress (2026-09-12): the existing GPU deployment completed 100 consecutive
OldSchool evaluation games with 50 games per seat, 100 replays, and no remaining
evaluation/WindBot processes. The retained evaluation summary is
[`reports/evaluation-status-20260914.md`](reports/evaluation-status-20260914.md).
Follow-up fault injection found SIGABRT on timeout/disconnect and a surviving
WindBot after a stall; fresh processes recover only after external cleanup.
Preliminary latency statistics cover 21 matched responses from three complete
captures; the underlying raw logs remain on the evaluation server.
Supervised evaluation now handles those native aborts: GPU fault injection
verified automatic cleanup, invalid-game accounting, and successful next games.
See [`reports/evaluation-status-20260914.md`](reports/evaluation-status-20260914.md).
Use `scripts/eval_windbot.py` for batch evaluation. Broader latency sampling and
remaining protocol/private-state acceptance remain open.

Experimental exception (2026-09-12): the user explicitly authorized starting
the Stage 6 balanced elfnote specialist experiment while WindBot multi-deck
evaluation remains pending. Four opponent decks passed both-seat greedy-path
checks but failed WindBot chain-protocol checks; they remain limited rather
than fully accepted. A 5,120-step GPU smoke and bounded 10M-step continuation
completed; the consolidated status is recorded in
[`reports/evaluation-status-20260914.md`](reports/evaluation-status-20260914.md).
This exception does not complete Stage 4 acceptance or Stage 7 promotion gates.

Integrate [IceYGO/windbot](https://github.com/IceYGO/windbot) as an external,
local evaluation client. WindBot is a C# YGOPro client that connects to a
YGOSharp/SRVPro-compatible duel host and chooses actions through deck-specific
executors. It can also run in HTTP server mode, but one isolated process per
evaluation worker is the initial, easier-to-debug target.

Implementation order:

1. Pin and record a WindBot commit; build it with Visual Studio or Mono/.NET.
2. Install a matching `cards.cdb`, deck files, banlist, protocol version, and
   dialog assets beside the executable.
3. Finish the YGO Agent duel-host adapter that forwards legal game messages to
   WindBot and returns its responses to the environment.
4. Add lifecycle handling: process startup, readiness, port allocation,
   per-game reset, timeout, crash recovery, logs, and clean shutdown.
5. Support both seating orders and deterministic seeds where the protocol
   permits them.
6. Save `.yrp` replay, YGO Agent decision log, WindBot log, result, duration,
   and all asset revisions for failed and sampled successful games.

Acceptance gates:

- Complete at least 100 consecutive smoke games without protocol deadlock,
  illegal response, stale state, or orphaned process.
- Confirm card effects are loaded by checking representative
  `select_effectyn`, `select_chain`, `select_card`, and `select_option` paths.
- A timeout or WindBot crash must mark the game invalid, restart the worker,
  and never count as a win or loss.
- Measure games/hour and latency percentiles before selecting worker count.

WindBot is evaluation-only because external process/network round trips and
rule-based decision latency would substantially reduce training throughput.

## Stage 5: Build and validate the multi-deck pool

Protocol repair (2026-09-12): legacy SELECT_CHAIN and CONFIRM_CARDS messages
are now adapted in Python for the deployed WindBot. Twelve targeted GPU games
passed with no child exceptions and complete cleanup; see
[`reports/evaluation-status-20260914.md`](reports/evaluation-status-20260914.md).
This resolves the observed multi-deck protocol blocker, but does not replace
the larger per-deck stability and promotion gates below.

Take the initial deck list from WindBot's supported `Normal` executors. Do not
assume that every `.ydk` has a complete executor: WindBot notes that cards not
known by the selected deck AI may not be summoned or activated correctly.

For every candidate deck:

1. Pair the WindBot executor name with its exact `.ydk` file.
2. Import all card IDs, scripts, tokens, and extra-deck cards into YGO Agent's
   asset/code-list pipeline.
3. Validate 40-60 main-deck cards, legal extra deck, banlist compatibility,
   script availability, and code-list coverage.
4. Run legal-action smoke tests and at least 20 WindBot mirror games.
5. Classify the deck as `ready`, `limited`, or `excluded`, documenting why.

Begin with a small, strategically varied pool of roughly 4-8 validated Normal
decks rather than enabling the full catalog at once. Prefer decks whose WindBot
executors complete games reliably. Keep elfnote as the learner's first anchor deck
even though it is not currently a stock WindBot executor; evaluate it against
the selected WindBot decks.

## Stage 6: elfnote specialist multi-matchup curriculum

The first multi-deck model is a elfnote specialist, not a universal deck-playing
model. The learner always pilots elfnote. WindBot determines the opponent deck
distribution, but WindBot itself does not participate in training rollouts.
For each selected WindBot deck, YGO Agent loads the corresponding `.ydk` and
uses a fast in-process opponent policy or bot.

Only transitions selected by the elfnote learner contribute to the policy update.
Opponent transitions must be excluded from the learner batch; otherwise the
shared policy would also be trained to pilot every opponent deck. Do not expose
the opponent deck ID directly to the policy in the first version: infer the
matchup from revealed cards and action history, as in a real duel. Deck IDs are
still required in rollout metadata for sampling and reporting.

Train in three controlled phases:

1. **elfnote retention** — reserve approximately 20% of games for elfnote mirrors so
   basic combo execution and resource loops do not regress.
2. **Balanced opponent curriculum** — use the remaining games for elfnote against
   the validated WindBot deck pool, sampled uniformly at first, with both
   seating orders represented equally.
3. **Adaptive curriculum** — after the balanced baseline is stable, allocate
   more games to matchups with low win rate or high uncertainty while retaining
   a minimum sampling floor for every deck.

Within non-WindBot training matchups, retain the opponent-policy mixture as a
starting point when compatible opponent policies exist:

- 50% current policy/self-play
- 30% compatible historical checkpoint
- 20% in-process greedy bot

Because a elfnote checkpoint cannot initially pilot unrelated opponent decks
well, begin each new opponent deck with its fastest available rule policy or a
separately prepared baseline checkpoint. Periodically collect WindBot
evaluation trajectories. If in-process opponents remain too weak, use those
trajectories for behavior cloning of a fast WindBot proxy, then add the proxy
to the training opponent pool; do not put latency-bound WindBot calls in the
rollout loop.

Checkpoint compatibility must include architecture metadata, code-list hash,
asset revision, deck-pool revision, and learned card-ID embedding dimensions.
Because expanding the code list changes embedding rows, introduce new cards
through an explicitly versioned model migration and initialize only new rows;
do not treat that migration as pretrained embedding work.

## Stage 7: Evaluation matrix and promotion gates

Evaluate candidate checkpoints outside the training loop. The primary matrix
is:

- Candidate versus WindBot for every validated deck
- Both seating orders
- At least 256 valid games per matchup when throughput permits
- Fixed seeds/deals for checkpoint-to-checkpoint comparisons
- elfnote regression matches against greedy, random-initialized, and historical
  policies

Report per matchup and macro/micro aggregates:

- Wins, losses, draws, invalid games, and Wilson confidence intervals
- Average turns, decisions, and wall-clock duration
- WindBot response latency p50/p95/p99 and timeout/crash rate
- Training SPS and evaluation games/hour
- GPU utilization, memory, and power during training
- Representative winning, losing, timeout, and illegal-action replays

Promote a checkpoint only if the elfnote specialist improves the aggregate
WindBot result without materially regressing the elfnote mirror or any protected
matchup. Never convert timeout, protocol failure, or unsupported WindBot
behavior into a model victory.

## Stage 8: Scaling after correctness

Only after the WindBot adapter and multi-deck evaluation matrix are stable,
consider longer runs, more environments, wider/deeper models, longer action
history, or bfloat16. Change one major variable at a time and retain the
no-external-pretrained-embedding decision.

## Current infrastructure

Use the GPU server and deployment paths documented in the workspace-level
`AGENTS.md`. Build or download WindBot and its dependencies on a machine with
public internet access, record checksums and revisions, then upload a complete
offline bundle to any isolated server.
