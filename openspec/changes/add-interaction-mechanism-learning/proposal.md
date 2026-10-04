## Why

闪刀姬诊断提示：模型需要学习可迁移的交互机制，而不是逐个背诵“这次应当保留遮蒙者”。当前静态语义粒度、连锁事件归属和训练监督各有待核验的缺口；这些还不能直接解释全部败局或 Q 实验失败。

## What Changes

- 建立带来源证据、可见性边界和未知标签的交互诊断数据集。
- 首先对真实连锁消息与模型可见事件逐项核对；若确认来源错配，在隔离版本修复并做回归，不覆盖活动训练环境。
- 用冻结策略的小规模条件结果预测实验，区分信息缺失、表示不足和监督不足。
- 按卡片交互组合划分留出集，检验机制迁移而非种子记忆。通过门槛后才另行申请匹配 PPO 辅助学习实验。

## Capabilities

### New Capabilities

- `interaction-mechanism-diagnostics`: 可审计的连锁来源核验、条件后果标签、冻结策略预测探针及分阶段实验门槛。

### Modified Capabilities

无已归档主规范变更。与进行中的 `add-structured-lite-observations` 和 `add-vrpo-candidate-q` 协调；不替代其未完成验收。

## Impact

候选涉及 `ygoenv/ygoenv/ygopro/ygopro.h` 的连锁事件记录、`scripts/build_structured_semantics.py`、结构化观测及独立诊断工具。当前阶段只新增规划文档；不改张量布局、奖励、合法动作、活动 PPO、模型参数或线上 native。后续若改变观测语义必须独立版本化，不能将运行时修复带来的差异算作训练收益。
