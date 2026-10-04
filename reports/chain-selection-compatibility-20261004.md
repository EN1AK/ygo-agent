# 连锁来源修复：旧模型选择观测兼容版本

## 处理结果

用户要求解决连锁修复迁移时的额外选择标志差异。本次新增可复现的独立构建脚本 `scripts/build_chain_compat_runtime.py`，从固定 b81516e 源码归档构建候选：启用 `chain-source-by-link-v2`，同时将 `selection[9]` 明确保留为 `legacy-prompt-field-v1`（`selection_finishable_`）。合法菜单、响应编码和 PPO 不改。该兼容模式不等于以菜单 finish 求 any 的新语义，未来切换新语义应另做实验。

候选目录为 home `/home/ygo/chain-compat-b81516e-20261004`。脚本拒绝已有目录，核对归档哈希，禁用编译缓存，显式指定 xmake 项目路径，完整重编译，并保存原始/补丁后头文件哈希、编译参数、退出码与模块哈希。原始源码和现有运行时不被覆盖。

## 实测证据

CPU 重放同一 186M 遮蒙者—零衣对局，原始 argmax，无动作覆盖，63 决策，第 9 回合自然结束，无无效局。目录：`/home/ygo/ygo-agent/training-runs/chain-compat-replay-20261004`。

- 直接与旧冻结 native `a2dbd260...` 的既有重放比较，63 项菜单及选中动作一致，整局公开链消息逐字节一致。
- 前 12 个决策的所有非事件观测字段一致，包含此前失败的 step 8 `selection[9]`。
- step 2 row 19/20 的开始/完成结算由错误的零衣 actor 0 改为正确的遮蒙者 actor 1、link 1。
- `verify_chain_replay.py` 生成 `paired-verification.json`，`passed=true`。本轮没有新增或重跑 77 项测试；77 项属于此前 b81516e 隔离来源修复记录。
- 原生模块与 186M 权重在重放前后哈希一致；冻结模块未被替换。

这是一个真实对局的兼容性证据，不是全局等价证明、胜率提升证据或新模型训练。logits/V 不要求相同，事件输入修复可能改变输出。

## 构建追溯结论

已找回旧任务中的成功容器构建输出，其二进制哈希确为 `a2dbd260...`，日志显示 `cache compiling.release`。本地 `training-runs/sky-place-forward-505b855-20261003/container-build.sh` 显示源码在独立目录解包；此前查看的 home release/build.log 只是失败的早期本机构建，不是最终构建日志。

505b855 源码把 `selection[9]` 从原始提示字段改为对合法菜单 finish 求 any。旧二进制在已捕获窗口的输出与原始提示字段相容；本次兼容构建直接与它通过逐字段比较。旧二进制实际为何未体现所声明源码语义仍不能据此定案：编译缓存是线索，不是已证明根因。

解决迁移的办法是显式保留旧选择语义并验证，不是抹掉差异、篡改旧报告或声称重新编译必然等价。新构建禁用缓存并提供双语义版本和完整哈希链，避免将未标明的选择变化混入连锁来源实验。

## 固定版本

- 源码：`b81516e34be25365f6d274ea0829980c388c0e10`
- 源码归档 SHA256：`963fd88aa452738f2e5a2fb8c6307daaf303dead17f3c7920a0119e2a1abe14b`
- 兼容模块 SHA256：`7bc098653b5336b159215168a4680b5f77b0015939bae6cf061e732016e79358`
- 修补后头文件 SHA256：`c886ec41de9c77f465ae211595aee2bfb017e9e04818eab065a2976989c0d599`
- 选择语义绑定：`selection_finish_version=legacy-prompt-field-v1`
- 连锁语义绑定：`chain_event_provenance_version=chain-source-by-link-v2`

## 使用边界

这份候选供后续独立迁移/续训使用。当前逐代 20M 任务继续使用原固定 runtime，不能在一个仍进行中的可比性实验里静默替换。线上对战服务也未切换。未来启动新 run 应同时记录上述两个语义版本与 native 哈希，并保持匹配的训练/评估环境；若网络结构不变，可继承权重与 Adam，但环境重置和观测版本变化必须记录。
