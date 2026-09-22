# Structured-lite 40M vs Stage3 batch evaluation

日期：2026-09-22

## 配置

- 牌组：双方均使用冻结的 `elfnote.ydk`
- 每组：4 个独立 seed，每个 seed 128 局，候选先手/后手各 64 局
- 40M：`structured-lite-v1`，`structured_variant=full`
- Stage3：`legacy-v2`
- 40M SHA-256：`a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`
- 1M SHA-256：`be04107298d167e0647c116632f01afea0380eb68541b7272a1d7cc0f5173584`
- Stage3 SHA-256：`4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`

## 结果

| 候选 | Seed 胜率 | 合计 | Wilson 95% CI | 平均长度 | 平均回报 |
|---|---|---:|---:|---:|---:|
| 40M full | 68.75%, 66.41%, 67.97%, 68.75% | **348/512 = 67.97%** | 63.81%–71.86% | 190.15 | 1.511 |
| 1M full | 13.28%, 13.28%, 12.50%, 9.38% | **62/512 = 12.11%** | 9.56%–15.22% | 168.05 | -4.017 |
| 40M，structured-only 输入全置零 | 8.59%, 5.47%, 3.13%, 9.38% | **34/512 = 6.64%** | 4.79%–9.14% | 175.51 | -4.436 |

交换模型装载顺序的 128 局校验中，Stage3 对 40M 为 31.25%，与同 seed 正向评测中 40M 的 68.75% 互补，未发现 A/B 装载顺序偏差。

## 判断

40M 已经稳定强于 Stage3。1M full 对 Stage3 很弱，而继续训练到 40M 后提高约 55.9 个百分点，说明后续训练学到了对 Stage3 有效的策略。把所有 structured-only 张量置零后，40M 从 67.97% 降到 6.64%，说明模型确实强依赖新增结构化输入，不是仅靠 legacy 字段运行。

不过，零输入消融属于分布外干预，不能把 61.3 个百分点的下降解释成新增特征的纯因果收益。要严格证明 Structured-lite 优于旧特征，还需要从同一初始化出发，以相同 deck、seed、actor mixture、训练预算和评测入口训练一个 40M `legacy-v2` 对照；最好同时保留 `relationship-only`，形成 legacy / relationship-only / full 三臂实验。

## `lite` 与 `full` 的含义

`structured-lite-v1` 是观测 schema 的名称。`lite` 表示它是完整结构化参考设计的有界、紧凑子集：保留 legacy 张量，并增加可见卡 ID、CDB 静态语义、审计过的 effect tags、选择上下文、动作结构字段、单卡/集合关系引用、公共事件环及溢出诊断。容量被固定为 32 个公共事件和每个集合角色 8 个引用，并使用 compact integer tensors。

它有意不实现完整 Lua 执行语义、自然语言嵌入、通用动态图、world model、显式 belief head、MCTS 或逐动作 Q/afterstate critic；Target/Cost 等关系无法由协议精确确认时还会使用 exact/fallback/unknown 置信等级。

`structured_variant=full` 是 **Structured-lite schema 内部**的模型变体，表示启用该 lite schema 已实现的关系、公共事件和静态语义等全部特征；另一个变体是 `relationship-only`，用于消融语义和事件特征。因此它是“Structured-lite v1 范围内的完整实现”，不是原始宏大参考设计的全量实现，两者并不矛盾。

