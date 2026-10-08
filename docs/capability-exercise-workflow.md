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

这是下一实施门槛，当前首版**没有**连接 actor 或优化器：

1. 用现有模型观测/动作接口重放，核验根观测、每个菜单和实际引擎响应关联；核心教师状态不传给 actor。
2. 每次评测/更新都用当前参数从完整合法历史重建双方 RNN。旧模型保存的 RNN 只能用于原轨迹核对，不能复用作新模型的起点。
3. 冻结模型跑分能力基线，再按批准的预算进行短序列示范或练习更新。接受多个达成目标的方案，未知标签屏蔽；不用“combo 越长奖励越高”。
4. 与相同起点、环境步数及总计算量的普通 PPO 对照，检查未见家族、完整对局迁移和原有能力回退。教师搜索成本也要计入。

流水线状态因此为：**原始场景/可疑选择 → 候选 → 可恢复且真实结算验证 → actor 与历史审计 → 冻结评测 → 有预算的教学 → 回归与实战评估**。失败时保留原因和证据，不循环自我生成“标准答案”。

## 首批三组验收点

| 家族 | 正确路线 | 对照 |
| --- | --- | --- |
| 玻纤—鳞茎—枪管龙 | 玻纤拉鳞茎；与另一个效果怪兽共三只做 Link 4；鳞茎攻击表示复活；转守备获二连击 | 守备表示复活无法作为转守备对象；对方两只守备怪兽也不能代替它。另验先攻击一次再发动的成功解 |
| 疾风战斗选择 | 1000 对 1500 LP，疾风越过攻击表示青眼直接攻击获胜 | 真正被遮蒙者无效后结束战阶保住本回合；两种局面撞青眼都输。无效分支仅考当前战阶存活 |
| 燎里回收交闪 | 连锁遮蒙者，整链结束后交闪仍在墓地 | 幽鬼兔破坏燎里后仍回收；不响应同样回收；只在本题固定条件下比较 |

10 个执行分支不是 10 个独立题家族，更不是 10 条已验证训练样本。
