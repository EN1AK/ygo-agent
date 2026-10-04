# 连锁来源核验：真实运行时错配已确认，隔离修复尚未验收

## 结论

遮蒙者—零衣案例不再只是静态风险：冻结 native 确实把链 1 遮蒙者的开始/完成结算事件记成链 2 零衣，发动者也由我方变成对方。这证明公开事件输入存在错误，不证明它就是 Q 失败或 PPO 波动的原因。

## 实际执行与证据

- 单个 CPU 对局，nice 15、CPU 14/15，原始 argmax，无动作覆盖，退出 0。
- 冻结 native SHA256 `a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`。
- 186M checkpoint SHA256 `dc29c11254c0418a4b43cc29083db3d05e17b5c8e3dacd5c27b17d63c8670721`，捕获前后不变。
- 既有三分支证据 52 文件逐项哈希通过；捕获 manifest 记录冻结源码头文件哈希、命令和进程。
- 根目录 `training-runs/chain-provenance-20261004/`：公开链消息 `chain-packets.jsonl`、前 12 决策的完整 NPZ 观测、`decisions.jsonl`、`engine.log`、录像及 `event-comparison.json`。
- 本地原始包 `training-runs/chain-provenance-20261004.tar.gz` SHA256 `d6c574a2411555a5c36ba90129f905c0d2b718f70f166d0ad79522b9def9434c`，与 home 相同。解码对照表在原始包之后生成，作为独立文件交付，不冒称在原包内。

## 明确错配位置

报文 index 为零起始；观测行号也是零起始。卡片 ID 为游戏卡号而非嵌入行号。

| 证据 | 正确含义 | 冻结观测 |
|---|---|---|
| packet 10 CHAINING | link 1，97268402 遮蒙者，actor 1 | 发动事件正确 |
| packet 12 CHAINING | link 2，26077387 零衣，actor 0 | 发动事件正确 |
| packet 14/15 | link 2 开始/完成结算，零衣 actor 0 | step 2 row 15/18 正确 |
| packet 16 `014801` | link 1 开始结算，遮蒙者 actor 1 | step 2 row 19：零衣 actor 0，错误 |
| packet 17 `014901` | link 1 完成结算，遮蒙者 actor 1 | step 2 row 20：零衣 actor 0，错误 |

step 2 的公开事件还保留同链两次发动和链结束，足以明确关联。后续 step 3/4 重复保留的历史不是新的独立错配样本。链完成不代表遮蒙者成功无效了目标。

## 实施状态

已实现但尚未通过构建/测试的候选：编译开关 `YGO_CHAIN_EVENT_PROVENANCE_V2`，默认不启用；按 link 保存公开来源，用消息编号查询结算/无效事件；未知来源使用空身份，链结束和对局重置清理。绑定版本名 `chain-source-by-link-v2`。新增六项针对性 native 测试，**未执行，不计通过**。新旧运行时必须分开，不得直接用于当前训练。

版本链：0452c35 捕获工具与规划，5ef266b 解码工具，525adf5 初始候选的选择性暂存位置错误，106eb98 已纠正；525adf5 不可作为构建来源。用户 BO3 和 TRAINING_PLAN 改动仍留在工作区，没有纳入上述提交。

## 构建隔离异常与暂停点

第一次 WSL `git archive` 读取 Windows 仓库失败（bad pack-int-id），没有编译。改为本地精确归档后，xmake 自动选择父项目 `/home/ygo/ygo-agent`，而不是预定隔离目录，意外写入父项目构建配置及 `ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so`。该文件当前 SHA256 `e34bd6db2c837c8a9f48ac0288fa072db838f9b3a675ff8bc41d1496ed3e3edd`；未取得写入前哈希，不能声称已恢复原样，也不能把这一构建算作候选通过。

立即核验 `/proc/276238/maps`：活动训练加载的是不可变 release 下的独立 native，而不是父项目副本。该实际加载文件重新哈希仍为 `a2dbd260...`，PID 276238 仍存活。捕获用的 q2m 冻结模块也未变。没有停止、重启或切换活动训练。

按实施流程在异常处暂停：后续必须显式指定 xmake 项目路径并断言输出目录；父项目副本/配置应先保留并核验可信备份，不能猜测恢复。当前仅 1.1/1.2 完成，1.3/1.4 未完成，未开始数据拟合或 PPO 辅助训练。H200 离线未同步。

## 继续后的隔离构建核验

用户确认继续后，已将父副本和配置原样备份至 home `/home/ygo/chain-build-incident-106eb98/`；没有可信的写入前哈希，仍不宣称恢复。新构建在训练树外 `/home/ygo/chain-provenance-build-106eb98`，配置和构建命令均显式传入 `-P`。

本次配置需要 `ygopro-core 0.0.4`，home 仅缓存 0.0.2/0.0.3；GitHub 下载连接超时，配置退出 1，未进入编译，六项候选测试仍未执行。没有降级依赖。退出后 `cmp` 确认父 native 与配置均和本次备份一致，冻结生产模块重验 SHA256 仍为 `a2dbd2604fec0e01ad722d887c639c00ed4c5aa74c912bd8f37d69f57a815486`。

按 AGENTS 检查构建服务器，SSH 可达且存在 `ygo-build-cache:replay-fix` 等镜像；只读检查该镜像仅见 core 0.0.2 缓存，不能冒称依赖已齐。候选源码归档上传请求被安全审查拒绝，未执行传输；需要用户明确授权将私有候选源码上传到 `ws-d1fd808734bb0029` 构建服务器。暂停于此，不绕过拦截，不部署，不拟合，不将未运行测试计为通过。进度仍为 2/11。
