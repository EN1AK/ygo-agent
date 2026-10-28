# YGO Agent training plan

## 最新用户约束：教学去除操作撤回（2026-10-28）

后续教学和模型继续使用现有原生动作过滤。人类撤选、返回上层等导航信息不作为
教学目标，也不能残留在重建的循环记忆输入中；拒绝发动、有效结束选择等战略动作保留。
已运行完整菜单实验保留为历史对照，不能当成过滤版路线验收。
先通过真实过滤环境重建动作索引、完整输入和历史，并逐条证明引擎响应、事件及
终点一致。含被过滤的人类撤回路径先隔离，待找到并独立验证等价简化轨迹后再准入。
新课程重新冻结、重新自主采集与对照；不直接沿用完整菜单实验的评测结论。

当前执行点为`training-runs/reality-loop-20261028/filtered-policy-v5`：
已提交源码ea386fc全量重建235条课程，235条通过、零隔离；53条轨迹共消去87个
导航环，235个原始游戏终点全部精确一致，独立核心复核共验证238条分支。
6条独立考试数据也全部通过清理，共消去8个导航环，6个原始终点精确一致。
两组均无核心撤选响应及已观测导航环残留。课程和考试兼容协议已分别准入，
新过滤路线首轮自主采集已执行，但事件复核失败使管线在教学前退出1；独立模型考试仍封存。
相同素材菜单现在保留待召唤卡身份；主菜单残留的失效选材计数只在环比较中忽略，
教学张量和循环历史均由实际重放产生。旧v4失败证据保留，不沿用其通过样本训练。

现已定位自主轨迹5002867ed20e的随机选牌差异：实际编译的训练核心采用重复抽取并
拒绝重复项，独立现代核心采用局部洗牌，导致固定种子结果及后续RNG状态不同。
首个兼容核心cc6f7317虽编译退出0，但实际加载发现Lua C接口链接错误，已失格保留。
f59e252版本修正构建参数并强制实际加载，新核心4099f875通过96组真实Lua/核心
采样核验（16组种子×6种抽取数量，每组连续4次调用），未修复核心96组均不同；
验证进程退出0，模型原生库不变。审核核心改版须绑定库和来源清单哈希，并全量
复核课程后另开研究协议。用户已重新开启GPU：5002867ed20e实际轨迹的716事件/
291响应复核通过，235+6条全量复核均通过，研究v4的59次自主评测全部通过核心
审核（30次达标），8个失败根获得4条正见证。控制臂在Runtime课程覆盖检查时
退出1，尚未进入优化：新见证未纳入原235条覆盖清单。现为派生课程保留精确
父记录证明，并对每条新增记录重新独立重放；覆盖检查不放宽，待新版本重跑。
用户说明GPU利用率低于40%持续4小时会自动关闭；运行要保留实际进程、退出码
和已完成证据，重连后先确认状态，不因断连盲目重启实验。

过滤版研究v5两轮四臂均各512次有限更新并真实退出0。第一轮63个开发配置为
before30、control47、repair46达标；第二轮66个配置为before46、control44、repair52。
两轮分别获得4条和3条正见证；两轮旧能力保持均未通过，不能晋级默认模型。
隔离完整运行时formal-runtime-v1在四副fit牌表上通过CPU真实重放/隐私/容量检查；
formal-protocol-v1的56个实际原生边界测试通过，64个交互分支及57个通知共121项
覆盖完整。尚无正式PPO优化及后保持证据，独立模型考试仍封存。
endpoint-exams-v1资格检查在第3个配置遇到uint32种子超过原生signed-int范围而
退出1；前一局达到2048步也被误标invalid。现仅调整保留32位内容的ABI表示及
预算分类，原种子、牌表、模型和决策预算不变。v2五个固定端点的小规模资格检查
全部通过，第二轮教学端点在完整原生环境完成实际GPU加载与32步决策，进程退出0。
这不是完整256局考试，也不是正式PPO更新；独立模型考试仍封存。

正式PPO前已确认当前58a7c5fb原生库缺少既有叠放素材事件位置修复：素材序号不能
作为战斗位置编码，否则第17张素材会崩溃。另建显式overlay-event修订的成对原生库，
不覆盖已完成SFT运行时；真实事件写入器须覆盖全部256个素材序号及普通战斗位置。
新运行时的教学/保持输入必须逐条实际重编码，只允许公开事件的位置列变化，动作、
引擎响应、事件和终点全部保持精确一致，新增见证也不能遗漏。随后全量独立复核，
重新完成四副fit牌表的CPU/协议/GPU资格检查；正式双方共用同一修订运行时，
必须另测该运行时上的PPO前基线，不将运行时变化归因于模型学习。

overlay-v3重编码与资格检查进程现已退出0：独立考试6条重编码通过，四副fit牌表
CPU检查及协议覆盖均通过，新正式原生库7371515e在固定第二轮端点上完成实际GPU
加载和决策。此前两次重编码工具失败保留：终局前场面捕获节奏遗漏已修复；另一次
是连续CARD_HINT通知的无序发送，现只规范连续提示块，保留其他所有原始通知和
发动窗口。模型独立成绩仍未读取。后续由ppo_study.py冻结32条fit完整历史（最长
138步、256步容量）和普通PPO/完整历史保持配对；正式每臂98更新、1,003,520步，
两次试训更新丢弃，共同固定起点28ec8cef。必须先完成修订运行时基线及双方真实
梯度/试训资格检查，再开始正式优化；这些工程步骤均不能替代全量考试和保持结论。

overlay-v3实际保留242/242条课程（含两轮7条新增见证），重新编码3858个决策输入；
独立考试6/6条重新编码450个输入。动作、响应和最终场面精确一致，事件只允许
既有契约中的连续CARD_HINT块排序，所有其他原始通知和发动窗口保留。
PPO协议v1在模型加载前因旧JAX不接受环境变量highest而退出1，保留为失败证据；
v2使用该版本支持的完整精度float32启动名称，模型、数据、预算与精度语义不变。
实际协议为research-v5/ppo-study-v2-protocol.json，before的66个开发配置已完成，
普通组smoke真实运行中；独立模型分数仍封存。完整对局评测增加逐局哈希完成标记，
显式续跑只跳过同协议同源码的已完成局（包括无效和超时），不能重抽失败局；
中断的未完成文件另存保留。最终统一评测全部5个教学前后端点与2个PPO固定终点，
每端点6条独立题及256个同种子、同对手、双座位自然对局，不能以4局资格前缀代替。

## 当前研究路线（2026-10-28，用户明确授权）

按 arXiv:2610.33845v1 的现实反馈路线推进实际研究闭环，完整目标及验收见
`docs/reality-anchored-learning.md`。先复用已冻结235条水晶机巧课程与831a78a局部
技能端点，核对实际自主失败，执行两轮“当前模型偏离→有界恢复搜索→重复引擎/
独立核心验证→同起点同更新预算教学对照→重新自主评测”。选卡/撤选循环只作
诊断候选，不屏蔽合法动作，也不把未知搜索结果标为负例。

继续完成新玩家/牌表/对局分组的独立迁移、双座位完整对局与兼容正式PPO后的
保持验证；已有留出题降为回归，不能用开发得分替代未见泛化或晋级。每轮在推理
前冻结机器可读协议，保留来源/模型/运行时哈希、成本、失败及追加式研究记录。
不因实现脚本或完成局部实验就宣称整体目标达成；现有原生模块和默认模型保护。

2026-10-28阶段进展：首轮两组各512更新、60个配对配置与独立审核已完成并退出0，
普通组27成功、教学组25成功；唯一新见证因搜索改变了固定对手动作而撤销准入。
旧产物保留并追加更正，不沿用教学端点。原选卡输入计数已在独立9e5fa085模块中修复，
236条记录/1961窗口及6条独立考试/141窗口迁移审核通过，游戏动作和结算不变。

当前执行点是`training-runs/reality-loop-20261028/selection-counts-v1/research-v2`：
使用仅接受原课程梯度的control端点3103f344，隔离失格见证后保留235条课程；固定highest
精度及新独立考试。8个真实失败根的后偏离恢复尚无见证；在各自实际决策根找到3条
对手策略一致、重复执行和独立核心验证通过的可行替代，均只作正见证，不标原动作错误。
新的同预算对照已启动，结果仍待完成。独立迁移模型分数、双座位完整对局及PPO保持
尚未验证；完整对局入口已有256尝试的冻结协议能力，代码准备不算实际评测完成。

## 当前持续目标（2026-10-23，用户明确授权）

继续当前能力教学路线，直到具备训练资格后执行正式训练并评测，不以题库实验结束替代该目标。先从已验收212111保护候选重验并教会抗灰/泡影的功能恢复路线；新资源、干扰时点与双启动条件在推理前固定，保持已暴露elfnote家族身份。每轮保留可学习性、原有能力、迁移和未知结论，允许新增经核验题目；不按未验证怀疑赋负标签。

随后冻结与题库Selected作用域等输入语义一致的隔离完整运行时，完成真实环境、观测/合法菜单、隐私及短GPU优化检查。沿用下文同卡组50%当前策略、30%固定兼容历史策略、20%greedy路径，先执行约1M环境步的正式训练及同预算对照；牌表、预算、种子、对手和停止条件在运行前另存配置。训练后串行执行双座位完整对局与题库回归，记录无效/超时、置信区间及计算成本。正式模型默认值仍由独立晋级证据决定，不能由题库得分自动替换。

这是对当前教学候选继续训练的明确授权，历史Stage3/100M等保留为版本化对照；不授权覆盖其他活动进程或未备份的原生库，不要求跳过运行时检查。详细执行任务见`openspec/changes/add-interaction-mechanism-learning/tasks.md`的7.20–7.21。

2026-10-24进度：首轮普通PPO约1M及1024完整对局/双端点题库复核已完成，历史胜率51.17%→61.33%，greedy均99.22%；严格旧24与原五目标回落、保持门槛失败，不替换默认模型。详见reports/elfnote-formal-ppo-20261023.md。同预算防御教学组仍未执行；先区分替代终场功能与目标过严，再以新冻结协议测试PPO期间持续旧题CE/场景重建KL，不能把这次普通PPO结果当作该约束已生效的证据。

2026-10-27完成：持续复习PPO对照见reports/elfnote-ppo-rehearsal-20261026.md；后续四响应SFT两组各256更新已完成，teach自主/恢复根4/4、正常49/53及旧五保持。87a7a7通过自己的20480步GPU准入后，按原1003520步/50-30-20预算完成after_defense，98批有限更新、退出0，端点e1144c。四端点2048自然终局/零无效及训练后回归、157新/5旧审核、8根补充诊断、独立原始复算和三包4109成员本地核验全部完成。普通PPO历史157/256、教学后PPO158/256，差+0.39百分点、区间[-8.98,+7.03]，未证额外教学稳健收益。PPO后正常0/53、功能自主1/28、拟合0/4、旧五5/5；固定原合法前缀后仍4/4响应，定位优先检查自主展开/上下文漂移，保持门槛失败不推广。60个真实目标退步候选已绑定证明但仍未准入。用户本阶段要求的正式训练与评测闭环完成，不代表泛化/保持问题解决；后续持续展开保护须另冻结协议。详见reports/elfnote-defense-teaching-20261026.md及reports/elfnote-after-defense-ppo-20261027.md。

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

## Structured-lite first full-training decision (2026-09-17)

The next complete training candidate is the `structured-lite-v1` full variant,
warm-started from the completed Stage 3 Elfnote checkpoint. This is an
experimental branch and does not replace the legacy model or runtime.

Before the long run starts, preserve the following order:

1. finish the recorded prompt/action/event/visibility checks for the native
   Structured-lite observations;
2. deploy beside, rather than over, the legacy runtime on the GPU server;
3. run deterministic legacy and Structured-lite smoke duels;
4. measure overflow, throughput, and memory with fixed settings;
5. compare scratch and warm-start short optimization runs and verify finite
   losses plus checkpoint reload;
6. only then start the fixed-budget full run, retaining its configuration,
   logs, checkpoints, hashes, invalid-game counts, and evaluation seeds.

The Veiler/Ogre case remains a required diagnostic, but it is not by itself a
promotion criterion. The corrected counterfactual-search result is a legacy
baseline; the new model must be compared on rule causality, value calibration,
paired rollout uncertainty, regression matches, and target-deck strength.

## PPO-to-search research route (recorded 2026-09-30)

This is a proposed sequence of controlled experiments, not a claim that these
components already work. Do not change the active 40M continuation or its
checkpoint format. Freeze its checkpoint, runtime hashes, deck list, seeds,
both-seat evaluation matrix, replay examples, and wall-clock/SPS baseline first.
Measure combo completion, interruption quality, invalid/timeout rates, advantage
distribution, and value calibration by decision type. Use equal environment-step
and wall-clock comparisons; promote only statistically credible paired gains.

1. **Candidate-action Q / Q-boosting pilot.** The current `q_head` setting is not
   a trained action-value critic. Keep PPO as the actor update, add a separately
   validated Q estimator over the *policy-visible staged legal menu*, and compare
   ordinary GAE against Q-boosted advantages with one major variable changed at
   a time. An observation-only Q ablation is cheap; a faithful centralized-Q
   variant requires a separate full-state critic input and a strict actor/critic
   information-leakage boundary. Check finite targets, Q calibration, menu-index
   alignment, and training throughput before longer runs.
2. **Stabilize self-play.** Test actor-update damping/anchor regularization,
   opponent-pool diversity, and selective policy-loss weighting as separate
   ablations. A forced single-action prompt may contribute zero policy gradient,
   but its transition must still advance the duel, recurrent state, and history.
   Do not stack multiple stabilizers and attribute the result to one of them.
3. **Model hidden information.** Train a belief sampler from self-play ground
   truth using only the acting player's legal information at inference. Validate
   public-card constraints, calibration, held-out decks, and leak guards. Exact
   opponent-hand rollouts remain oracle diagnostics, never fair-play scores.
4. **Add test-time candidate rollout.** On selected high-impact prompts, evaluate
   a bounded set of legal actions across reproducible belief particles with a
   common rollout budget, terminal returns or a calibrated leaf value, then form
   a regularized local policy update rather than an unqualified max-Q choice.
   Compare raw policy, candidate rollout, and PUCT at matched inference cost.
   Search is opt-in and must fall back to the raw policy on budget/error gates.
5. **Distil only after a fair-play gain.** Version searched policy/value targets
   separately from PPO data and test offline distillation against the frozen
   baseline. Test-time search alone does not teach the base model its combos.
   Gumbel/tree-search or ReBeL-style training remains a later option if the
   simpler candidate rollout fails at adequate compute and belief quality.

The search implementation and gates are tracked in
`openspec/changes/add-mcts-policy-search/`. The Q-critic and self-play ablations
are research items here, not hidden tasks in that search change. Relevant
research: [VRPO/Q-boosting](https://arxiv.org/pdf/2605.19235) and
[Ataraxos candidate rollout](https://arxiv.org/pdf/2511.07312); neither paper's
reported gain is assumed to transfer to this project.

## Current infrastructure

Use the GPU server and deployment paths documented in the workspace-level
`AGENTS.md`. Build or download WindBot and its dependencies on a machine with
public internet access, record checksums and revisions, then upload a complete
offline bundle to any isolated server.

Native training environments must be deployed from an immutable runtime
release following `docs/reproducible-runtime-releases.md`. Never reconstruct a
previous experiment from an unpinned online package repository or choose among
multiple `.so` files by filename. Verify the release manifest and real-engine
schema self-test before loading a checkpoint or starting optimization.
