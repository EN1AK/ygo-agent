# 过滤教学与正式PPO研究验收

全部固定端点和预定分母完成；完成实验不代表可以晋级默认模型。

| 端点 | 独立题成功 / 6 | 自然胜 | 自然负 | 和局 | 未知 / 256 |
|---|---:|---:|---:|---:|---:|
| parent | 2 | 98 | 109 | 0 | 49 |
| round1-control | 2 | 95 | 105 | 0 | 56 |
| round1-repair | 2 | 83 | 95 | 0 | 78 |
| round2-control | 1 | 97 | 104 | 0 | 55 |
| round2-repair | 2 | 101 | 101 | 0 | 54 |
| control-final | 2 | 131 | 95 | 0 | 30 |
| rehearse-final | 2 | 141 | 80 | 0 | 35 |
| historical-ppo1-round1-control | 1 | 96 | 98 | 0 | 62 |
| historical-ppo1-round1-repair | 3 | 108 | 90 | 0 | 58 |
| historical-ppo1-round2-control | 1 | 105 | 105 | 0 | 46 |
| historical-ppo1-round2-repair | 2 | 74 | 123 | 0 | 59 |
| historical-ppo1-control-final | 3 | 135 | 89 | 0 | 32 |
| historical-ppo1-rehearse-final | 1 | 126 | 98 | 0 | 32 |
| historical-ppo2-control-final | 2 | 120 | 97 | 0 | 39 |
| historical-ppo2-rehearse-final | 2 | 124 | 88 | 0 | 44 |

独立题原始接收7条，其中1条转换失败保留为未知；表中列出可评测的6条，report.json另报原始接收分母上的成功比例边界。
独立题从固定人类前缀进入己方主阶段，之后对手被动；完整策略对手的双方决策另由自然对局检验。

无效和预算耗尽均保留为未知，未折算成胜负。

相对PPO起点的能力保持：普通组未通过，完整历史组未通过。
相对最初模型的旧能力保持（在同一修订环境重测）：普通组未通过，完整历史组未通过。

比例边界仅反映未知结果，不是统计置信区间；固定组件、四副训练牌表及配对座位存在相关性，不能直接外推人类天梯胜率。

两轮教学本身的旧能力回归、完整对局配对边界及所有输入/产物哈希见同目录 report.json。默认模型不自动晋级。

相同权重的历史端点显式登记为别名，共用同一运行时、同一完整题矩阵的结果：
- historical-ppo1-parent → parent（权重SHA完全相同）
- historical-ppo2-parent → parent（权重SHA完全相同）
- historical-ppo2-round1-control → historical-ppo1-round1-control（权重SHA完全相同）
- historical-ppo2-round1-repair → historical-ppo1-round1-repair（权重SHA完全相同）
- historical-ppo2-round2-control → historical-ppo1-round2-control（权重SHA完全相同）
- historical-ppo2-round2-repair → historical-ppo1-round2-repair（权重SHA完全相同）

原独立组件的最初模型旧运行时尝试已因工程审核失败而暴露；本次是规则修订后的固定实验复测，未用这些成绩调参或选模型，不能称作一套全新未观察的留出集。
