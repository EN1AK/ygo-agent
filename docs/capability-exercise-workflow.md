# 能力题库：从场景或实战失误到可重复训练的题

本流程适用于“给出一个局面”“解释一次不合理选择”“补一项机制或 combo 能力”。首版入口是 `scripts/build_capability_exercises.py`，真实引擎验证入口是 `scripts/run_capability_exercises.py`。首批只包含三个开发家族，不视作未见测试集。

## 先把问题写成可判定目标

出题者（人或模型）从自然语言提取以下信息，缺失项记录为待补，不能猜成已证事实：

| 字段 | 必须回答的问题 |
| --- | --- |
| 能力与目标 | 要检验哪一个选择或机制？在什么结算边界算完成？ |
| 起点 | 回合、阶段、座位、LP、区域/表示、连锁、合法菜单、已经消耗的资源和次数是什么？ |
| 恢复方法 | 能从双方完整历史重放，还是明确标注的合成残局？ |
| 前提与对手 | 对手有什么可用响应？答案只对哪些假设成立？ |
| 正解与反例 | 至少一个可重放成功方案和一个改变关键条件的对照；未知动作保持未知。 |
| 来源与版本 | 对局/种子、决策编号、模板/切片家族、引擎/脚本/数据库/模型哈希是什么？ |
| 预算 | 最多多少候选、引擎步数、决策数和搜索时间？ |

目标写结果，不写必须背诵的动作顺序。例如“刺刀枪管龙完成两次攻击”，允许先发动效果或先攻击一次再发动；不能悄悄加上“必须打出 8000”。“阻止交闪回手”在整条连锁结束后检查区域，不能看到幽鬼兔破坏燎里就算成功。

## 可执行工作流

在仓库根目录使用配置好的 Python 环境运行以下命令；不需要模型权重或 GPU。输出采用独占创建，修订或重跑使用新目录。

### 1. 输入场景，创建候选

```sh
python -m scripts.build_capability_exercises draft --scene '玻纤拉鳞茎，用最少场上怪兽做枪管龙并攻击两次' --case combo --output training-runs/exercise-draft/combo.json
```

已支持的精确模板：`combo`、`battle`、`battle-negated`、`interaction`。`--case` 是出题者明确选择的模板，不是自然语言识别结果；场景正文是需求说明，**不会改变模板的固定参数**。不指定模板时，会生成包含缺失条件的通用候选，验证器拒绝把它强行套到某个题目上。

任意新场景由出题模型按上表结构化，再新增恢复适配器、目标判定和对照分支。自然语言到任意可运行局面的自动合成仍是后续能力，首版没有假装实现这一点。

### 2. 从训练/评估日志自动收集候选

```sh
python -m scripts.build_capability_exercises mine --decisions path/to/decisions.jsonl --output training-runs/exercise-draft/mined.json
python -m scripts.build_capability_exercises mine --decisions path/to/decisions.jsonl --step 42 --output training-runs/exercise-draft/flagged-42.json
```

兼容 `eval_structured` 和同轨迹交互采集的决策 JSONL。自动候选来源：循环保护干预；训练端写入的 `suspicions: ["原因"]`；同回合同座位中，公开观测摘要及菜单同时重复。人工指出的步数用 `--step`，可重复指定。每次最多 48 个候选。

这些规则只决定“值得检查”。低 V、低攻击力、高策略置信度、输掉对局都不是错误的充分证据：低攻怪兽可能为了遗言或触发条件主动撞上去。初次导入不产生正负标签。

输出保留日志哈希、行号、决策内容哈希、模型哈希、原始动作菜单、怀疑原因、可用观测文件引用。原始索引不是跨运行时通用动作；单方 WindBot 日志没有对手完整历史，不能只拿自身动作索引恢复根局面。缺少双方历史、环境版本、根观测/菜单一致性时继续隔离。

训练端以后每段评估完成即可调用 `mine`，根据来源 ID 去重后加入队列；只将已完成日志传入，避免读到半行。首版没有安装后台监控或修改训练循环。

### 3. 在独立真实引擎中验证

先用现有、固定版本的核心和 Lua 静态库构建小型桥接库，输出必须是新路径：

```sh
bash scripts/exercises/build_core_bridge.sh /path/to/core-install /path/to/liblua.a /path/to/new/libexercise_bridge.so
python -m scripts.run_capability_exercises --core /path/to/libexercise_bridge.so --database /path/to/cards.cdb --scripts /path/to/script --candidate training-runs/exercise-draft/combo.json --output training-runs/exercise-proof/combo-v1
```

省略 `--candidate` 可以执行全部首批模板；`--case` / `--branch` 允许定位某个分支。每条分支在新 duel 中重复两次，记录原始消息、菜单与响应、根局面、从合成初始场恢复根的响应前缀、稳定结算状态、实际加载脚本与数据库/核心哈希。未知消息、错误响应、脚本错误直接拒收；默认至多 160 次决策、4096 次 process 调用。脚本回调是核心全局状态，**一个进程不得并发多个 Core**；更广泛的无人值守验证应外加进程级时间限制。

首版布置的是合成残局。玻纤和燎里都经过真实 Link 召唤及触发过程；疾风的无效由对手实际发动遮蒙者完成。之后一切选择和结算使用真实卡片脚本，不手动改攻击力或宣告效果结论。合成初始场尚未证明能从某张标准牌表的起手到达。

判定状态分别为 `success`、`verified_failure`、`unknown`、`invalid_fixture`、`budget_exhausted`。`verified_failure` 只表示已执行的这一分支未完成目标，不表示整个局面无解。

```sh
python -m scripts.build_capability_exercises audit --summary training-runs/exercise-proof/combo-v1/summary.json --output training-runs/exercise-proof/combo-v1/audit.json
```

审计检查证据哈希、根、稳定边界、同运行时、两次一致和预期结果。Python API 的 `expected_runtime` 可强制匹配调用者预先固定的版本。通过后仅授予机制验证资格，`training_ready` 仍为 false。

### 4. 拆分并扩充，但保留同源关系

先完成完整解，再切出“玻纤拉谁”“Link 素材选择”“鳞茎复活表示”“枪管龙发动对象”“第二次攻击”等根。切片必须从同一前缀恢复，不能丢掉鳞茎一局一次、效果已用次数等隐藏在历史中的条件。首版记录了完整轨迹；独立切片运行入口尚未实现。

每个家族依次增加关键条件对照、无关变量变化、替代成功解、实战故障重现。不以换卡名、换一个 LP 数字充独立题数。`assign_splits` 对完整对局、种子组、模板、combo 切片的所有 lineage 键做联合分组；具有间接共享关系也不得跨集合。开发家族的整个连通分组都固定为 development。

```sh
python -m scripts.build_capability_exercises split --candidates training-runs/exercise-draft/mined.json --output training-runs/exercise-draft/split.json
```

划分文件写出后冻结；加入新数据时须重新做跨来源审计，不独立重算多个批次后直接拼接。首批三个家族及其变体全部是开发题。

### 5. 从已验证题进入模型评测与教学

固定四个合成局面已连接原 actor，冻结基线见 `reports/capability-exercise-actor-baseline-20261008.md`。另有正常起手的 combo/疾风轨迹，以及默认关闭的短序列教学入口，见下文第 7 节。通用实战恢复和更广泛教学仍按以下门槛推进：

1. 用现有模型观测/动作接口重放，核验根观测、每个菜单和实际引擎响应关联；核心教师状态不传给 actor。
2. 每次评测/更新都用当前参数从完整合法历史重建双方 RNN。旧模型保存的 RNN 只能用于原轨迹核对，不能复用作新模型的起点。
3. 冻结模型跑分能力基线，再按批准的预算进行短序列示范或练习更新。接受多个达成目标的方案，未知标签屏蔽；不用“combo 越长奖励越高”。
4. 与相同起点、环境步数及总计算量的普通 PPO 对照，检查未见家族、完整对局迁移和原有能力回退。教师搜索成本也要计入。

流水线状态因此为：**原始场景/可疑选择 → 候选 → 可恢复且真实结算验证 → actor 与历史审计 → 冻结评测 → 有预算的教学 → 回归与实战评估**。失败时保留原因和证据，不循环自我生成“标准答案”。

### 6. 对首批固定局面跑原 actor 基线

在干净 Git archive 中、固定 `ygo-build-cache:replay-fix` 构建镜像内执行；不要直接拿含未提交改动的生产 header 构建：

```sh
bash scripts/exercises/build_actor_bridge.sh /path/to/sqlitecpp-3.2.1.tar.gz /path/to/new-build
```

该诊断模块继承原 `YGOProEnvImpl`，沿用 `reset/step/next/WriteState`、合法菜单、强制动作和历史编码。仅替换 duel 工厂及透明核心消息/响应跟踪，生成 `exercise_ygopro.json` 记录五处修改和输入 header 哈希。核心使用仓库的 0.0.3/0.0.4 安全补丁；模块须保存为独立 `exercise_actor_native`，不能覆盖生产 native。同一进程只允许顺序运行一个实例，核心回调和诊断轨迹为进程全局状态。

```sh
timeout 180 env JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/eval_capability_exercises.py \
  --native /path/to/exercise_actor_native.cpython-310-x86_64-linux-gnu.so \
  --database /path/to/cards.cdb --scripts /path/to/script \
  --code-list /path/to/frozen-code-list.txt --semantics /path/to/pinned-semantics \
  --checkpoint /path/to/frozen.flax_model --output /path/to/new-baseline
timeout 90 env PYTHONPATH=. python scripts/verify_exercise_actor_responses.py \
  --native /path/to/exercise_actor_native.cpython-310-x86_64-linux-gnu.so \
  --database /path/to/cards.cdb --scripts /path/to/script \
  --input /path/to/new-baseline --output /path/to/new-response-audit.json
```

省略 `--checkpoint` 只验证原适配器参考路线。每条路线重复两次；带模型时，每题首条参考分支还运行模型自主接手分支。模型从根开始控制被考座位，另一方继续固定参考策略。两边记忆均从**声明的合成 episode 起点**初始化，并用当前冻结参数对每个原适配器暴露的观测推进；到题根不重新清零，也不加载旧 RNN。原适配器自动处理的强制动作仍按原方式记入历史，没有人为增加模型调用。初始化注册用的占位牌组不参与 fixture 布局。

输出包含每步实际 actor 张量 `.npz`、菜单/选择/响应、logits/归一化概率、V、输入观测和 RNN 哈希、完整事件、终局面，以及模型/语义表/脚本/数据库前后哈希。教师状态和答案不加入 actor 张量。审计独立重放同一核心的实际响应，要求每个事件和末局面精确相同，并核对张量逐行与决策关联。`summary.passed` 表示评测可信，不表示模型答对；答题结果应读各 `mode=policy` 行的 `status`。

这些旧合成局面仍未验证从标准牌表正常起手可达，因此保留 `full_standard_duel_history_verified=false`、`training_ready=false`。第 7 节新采集的正常起手轨迹有独立证据，不能反过来授予旧合成记录训练资格。无效后的战斗选择、燎里响应窗口的正常起手恢复仍待补齐；切片和母题必须属于同一开发家族。

### 7. 正常起手、分段教学与前缀回归

`collect_teaching_exercise.py` 的 `--kind combo` / `--kind battle` 从双方 40 张主卡组、至多三张同名卡、正常初始抽牌和首回合限制启动，没有 Debug 场面注入。规则是冻结 MR4 核心且未启用禁限卡表，不等于现行赛事合法性。固定牌序、固定对手和三个无关填充牌变体仍属于开发家族。

```sh
python -m scripts.collect_teaching_exercise \
  --native "$NATIVE" --database "$DATABASE" --scripts "$CARD_SCRIPTS" \
  --code-list "$CODES" --semantics "$SEMANTICS" --kind combo --output "$COMBO"
python -m scripts.audit_teaching_exercise \
  --native "$NATIVE" --database "$DATABASE" --scripts "$CARD_SCRIPTS" \
  --input "$COMBO" --output "$COMBO_AUDIT"
# 以 --kind battle 和新的输出目录重复采集/审计。
```

combo 中，对手先用古之规则布置两只守备青眼；己方通过电子龙、速攻同调士和通常召唤遮蒙者进入玻纤，拉鳞茎，完成枪管龙及二连击。完整对局前缀保存在每条记录中。`position` 从鳞茎复活表示窗口接手，`combo` 从玻纤效果窗口接手。疾风题从通常召唤零衣、Link 疾风进入战阶，考直接攻击并避免自身伤害。二者均非任意指令条件策略；清场/伤害是这个固定对手下的局部目标，不证明战略最优。

采集记录本身保持 `training_ready=false`。独立审计需逐行检查 `.npz` 与决策观测哈希，再通过另一 C API duel 重放所有响应、核对全部事件及终局面；教学入口核对审计与记录/张量/native 哈希后，只对指定实验授予使用资格。

```sh
python -m scripts.train_exercise_demonstration \
  --native "$NATIVE" --database "$DATABASE" --scripts "$CARD_SCRIPTS" \
  --code-list "$CODES" --semantics "$SEMANTICS" --checkpoint "$PARENT" \
  --combo "$COMBO" --combo-audit "$COMBO_AUDIT" \
  --battle "$BATTLE" --battle-audit "$BATTLE_AUDIT" \
  --runtime-manifest "$ASSET_BASELINE" --plan assets/exercises/teaching-v1/plan.json \
  --output "$NEW_OUTPUT"
```

该命令默认仅评估和核验顺序/批量 RNN 一致性。显式增加 `--execute` 才运行清单内固定 32 次更新；输出目录必须不存在，模型始终另存。`ASSET_BASELINE` 是冻结资产前后哈希清单，不是任意路径列表；重用第一轮清单也要求原 parent/native 的确切哈希一致。运行时须独立打包源码、native、牌库、脚本、语义表和 checkpoint 并先验证 schema。

每次更新用当前参数从本座位的完整前缀重算 RNN；前缀参与状态传播和梯度回传，但首轮只在鳞茎之后及疾风战阶的示范动作上计损失，padding 不计分。两种能力各占一半权重，未探索选择不标为错误。只拟合 variant 0；variant 1/2 是同家族检查，不能称为未见机制测试。

**每个接续题都必须同时复测更早的接手点。** 第一轮虽学会鳞茎表示，却在玻纤效果窗口改为取消发动：正确重建历史不等于保住前缀决策。后续实验应冻结新的清单，从原父模型重新开始，将已经会的前缀决策纳入复习或锚定，并把各深度回退作为拒绝推广条件；不因后缀成功自动追加预算。

`evaluate_teaching_checkpoints.py` 支持重复传入 `--checkpoint`，按相同根只读复测多个模型。对每个评测目录运行 `audit_teaching_exercise.py --pattern '*v*-0.json'` 独立核对正常起手轨迹。普通 PPO、教学后 PPO、增加普通更新的对照需分别保留命令、最终模型、sidecar、有限指标及退出码。第一轮额外 PPO 只匹配 learner 输入样本数，**未匹配总 FLOPs/墙钟**；不能据此宣称等算力优势。详见 `reports/capability-exercise-teaching-20261008.md`。

### 8. 教学退步时的研究流程

先核对同根观测、菜单和当前参数历史，排除环境漂移；再冻结可证伪假设。不要从一个训练终点推断“没学够”或“遗忘”。同时保存分项损失、关键动作概率和真实自主 rollout，在多个更新点观察有没有先学会再退步。各更新点使用同一前缀，避免把状态分布变化当作同条件概率变化。

首个研究清单 `regression-study.json` 分别比较延长训练、降低学习率、移除复习任务和加入前缀目标；测量梯度方向和固定父模型隐藏状态的解释性干预。后一种干预只用于分析，不能作为合法 RNN 恢复或正式评测。调查结论见 `reports/capability-teaching-regression-study-20261008.md`：原后缀方法第 8 次成功、第 16 次退步、256 次仍未恢复；前缀监督在第 32 次已学会 variant 0 的完整起手，256 端点通过三变体，短 PPO 后仍保留。

复现四组研究使用 `study_teaching_regression.py`，参数与第 7 节类似，但 `--release` 必须指向完整冻结运行时目录（含 `manifest.json`），替代旧入口的 `--runtime-manifest`；`--plan` 传入 `assets/exercises/teaching-v1/regression-study.json`。只复现已经验证的修复组则传入 `assets/exercises/teaching-v1/prefix-rehearsal.json`。两种配置都须显式增加 `--execute` 才更新参数，默认只预检。原 32 次入口保留为历史对照，不作为修复后的推荐课程。

后续出题/教学记录应同时声明：教学根、完整前缀、前缀监督或锚定、旧技能复习、各深度自主验收和预算。`opening` 现在从己方第一个原 actor 决策接手。判定停止必须检查根之后原始 NEW_PHASE/NEW_TURN 事件，不能只检查下一次可操作窗口的 phase：引擎可能自动跳过主要阶段二/结束阶段。预算耗尽与明确失败仍分开报告。

损失平台需定位到具体动作：若只是多个成功目标的参考交叉熵，不能因 loss 不为零就反复加训；新增等价动作集合标签必须先重放验证。只教同一条路线更多遍，也不能替代新起手、历史变化及未见机制的测试。开发修复、跨家族迁移和完整对局强度分别验收。

### 场景重建式原策略保护

在同一个 `study_teaching_regression.py` 入口传入 `assets/exercises/teaching-v1/scene-parent-kl.json`。默认仅预检；`--execute` 运行已冻结的四组各 256 次对照，无自动 PPO 续训或推广。原 `regression-study.json` 和 `prefix-rehearsal.json` 未指定 KL，仍保持原目标。

场景重建分两层：真实引擎从合法起手沿响应前缀恢复局面；新旧模型分别从完整本座位观测序列重建记忆。可以缓存冻结父模型的策略输出，不能把父模型隐藏状态借给学生。KL 采用 `current || parent`，只计合法动作，完整前后缀等场景权重，padding 权重为零。参考 MirrorForce 的 `sky_specialize.py`；0.25 是待验证起点而非通用最佳值。

同时查看教学损失、平均/最大 KL、父策略首选一致率和从起手/玻纤/鳞茎接手的自主完成率。原策略也会犯错，KL 可能阻碍纠错；保留有无 KL 的匹配对照和前缀示范，不以保护为由冻结原来的错误。题目分数、局部保留与全局牌技仍分别验收。

2026-10-09 四组对照已完成：前缀＋KL 通过三个正常起手变体及旧合成 combo，平均策略偏移相对前缀对照下降约 19.5%；后缀＋KL 仍未学会完整起手。10 项测试、76 条代表轨迹的独立核心审计通过；不构成未见家族或整体牌技结论。完整报告见 `reports/scene-parent-kl-study-20261009.md`。

### 同时补齐多项短板

联合教学入口为 `scripts/teach_three_capabilities.py`，清单为 `assets/exercises/teaching-v1/three-capabilities.json`。按 `collect` → `preflight` → `fit` 分开执行；前两步不启动优化器。每次使用全新输出目录，保留失败记录。运行环境和父模型由清单哈希固定，原始题解、观测张量、采集审计和数据清单均需要可追溯。

1. 为每个短板补齐正常起手和对手响应前缀。至少验证一个成功分支和一个失败分支；改变动作前，两分支必须具有完全相同的根观测。反例只能修改指定决策，不能顺手改动前面的召唤表示或其他行为。
2. 保留容易与新知识混淆的旧能力作为复习。例如教“疾风被无效后停手”时，同时训练“未被无效时直接攻击”。交互题的答案限定为当前目标、已知场面和固定响应；“遮蒙者优于幽鬼兔”不是普遍规则。
3. 按场景均衡训练完整本座位决策序列，父策略和学生分别重建记忆；对合法动作计算冻结父策略 KL。父策略也可能对错误动作十分自信，因此预先设置较强/较弱约束对照，并同时记录纠错进展与旧能力保持情况。
4. 学习曲线记录多个固定更新点，终点评估同时包含正常起手、后段接管和最初的合成题。后段学会、完整起手失败、合成题失败分别报告，不用平均分掩盖差异。训练步数只能结合这些曲线判断，不以训练损失下降代表实际通关。
5. 独立核心审计必须在独立进程运行。C API 的脚本读取回调是进程全局状态；在 actor 进程内加载另一套读取回调会破坏合成题预载，形成工具错误。题目加载错误应阻断流程，不能记成模型答错。

当前入口仍是人工审定模板的有界实验，尚未实现“任意实战失误自动出题并无人值守训练”。开发变体不计作未见家族，题库通关不替代完整对局强度评估。

2026-10-09 联合教学结果：0.25/0.05 KL 两组各 1024 更新，正常重建题均 27/27，旧合成题均 2/4；交互题改善，但旧 combo 在玻纤发动处退步，旧无效题仍撞青眼。完整验收未通过，不推广。下一轮优先扩大不同合法历史的复习覆盖，再做座位/生命值/历史的单因素对照；不能用重复同四条示范替代覆盖。详见 `reports/three-capability-teaching-20261009.md`。

## 扩充上下文时的审查

`context-coverage.json` 与 `teach_context_coverage.py` 把相同机制置于不同合法历史：区域选择、替代展开、先后手座位、真实效果造成的低生命值及组合变化。先确定配置和用途，再采集正反例；预留的开发组合不得在训练中参与采样或挑选模型。它们仍继承原开发家族的 lineage，不能更名为未见测试。

准备过程也是需要重建的模型历史，但不一定是教学答案。例如为低 LP 题支付卡牌成本、为了后手场面先结束首回合，仅用于建立考点。用逐决策 `teaching` 标记排除这些准备动作的 CE/KL 权重，仍保留其观测和 RNN 推进；判分依据准备完成后的 LP，不把准备成本误记为答题损失。

审查题目数时同时统计上下文配置、关键决策观测根、模板家族和重复重放。替代路线可能共享起手，因此除了完整展开，还要从每种历史的关键接管点复测；不能用同一个起手测试的重复成绩证明历史覆盖。无效后停手题要求实际进入该机制并保留疾风，不能靠不展开获得成功。参考分支须同根、两次一致，独立核心审计须与模型运行分进程。

实验对照同时固定父模型、每次批量大小、更新数和学习率，单独比较覆盖与保护约束。扩大题库意味着每条示范的重复次数减少，因此还要记录每条示范曝光量、不同接管深度及最后一个完整采样周期的指标；不能直接比较来自不同场景的相邻 loss。即使学习预算相同，数据构造和完整评估成本也应另列。

本轮结果见 `reports/context-coverage-teaching-20261009.md`：同样1536更新，窄＋KL为57/77，扩＋KL为74/77，扩无KL为76/77；扩题改善开发组合迁移，但三组均丢失旧合成combo，未推广。已将低血量题中“合法零衣效果展开后仍撞青眼”的真实失败导入候选队列。后续优先重建模型实际选择的历史，验证其成功续行；不将较早的合法替代解误标为错误，也不将单条参考的模仿loss当作多解题唯一指标。

## 首批三组验收点

| 家族 | 正确路线 | 对照 |
| --- | --- | --- |
| 玻纤—鳞茎—枪管龙 | 玻纤拉鳞茎；与另一个效果怪兽共三只做 Link 4；鳞茎攻击表示复活；转守备获二连击 | 守备表示复活无法作为转守备对象；对方两只守备怪兽也不能代替它。另验先攻击一次再发动的成功解 |
| 疾风战斗选择 | 1000 对 1500 LP，疾风越过攻击表示青眼直接攻击获胜 | 真正被遮蒙者无效后结束战阶保住本回合；两种局面撞青眼都输。无效分支仅考当前战阶存活 |
| 燎里回收交闪 | 连锁遮蒙者，整链结束后交闪仍在墓地 | 幽鬼兔破坏燎里后仍回收；不响应同样回收；只在本题固定条件下比较 |

10 个执行分支不是 10 个独立题家族，更不是 10 条已验证训练样本。
