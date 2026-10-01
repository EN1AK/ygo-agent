# 选卡撤选循环：向前动作优先过滤（2026-10-01）

## 结论

模型在 `MSG_SELECT_UNSELECT_CARD` 中反复选择／撤选，导致与 WindBot 对战达到 1000 步上限。只在核心允许 `Finish` 时隐藏撤选不够：Voiceless 的同种子首局仍会在“继续选另一张卡／撤选”之间循环。最终规则是：非人类策略只要有任一向前选择或 `Finish`，就不暴露撤选；没有向前动作时保留撤选，以免堵死纠错路径。该规则位于训练、评测共用的环境合法动作菜单，不是仅对推理器的特殊处理。人类模式不变。

## 配对评测

冻结 45,015,040 步模型 `28b4bd084a92e760fd54905887dda92c4cdac9d2ba39cc963114de98d9144322`，四种 WindBot 牌组各 10 局，种子从 `2026100102` 开始、交替座位、原始 argmax、关闭推理 cycle guard、每局最多 1000 步。只更换环境原生模块；没有续训。

| WindBot 牌组 | 旧模块有效胜–负／无效 | 新模块有效胜–负／无效 |
| --- | ---: | ---: |
| Kashtira | 1–3／6 | 5–5／0 |
| Labrynth | 6–0／4 | 8–2／0 |
| Tearlaments | 6–1／3 | 9–1／0 |
| Voiceless | 5–1／4 | 9–1／0 |
| 合计 | 18–5／17 | 31–9／0 |

这证明在该固定压力测试中撤选超限消失，不证明同模型的一般对局强度提高了多少。过滤会牺牲部分策略性重新选材能力；需在后续训练和更广的牌组／greedy／镜像评测中继续观察。此变更也改变了模型看到的合法动作分布，后续训练应使用同一环境版本，旧检查点可以直接用于本次评测但不应把热启动收益全部归因于训练。

## 验证与产物

- 源码提交：`f7517828bae524a90f6e7fdbff7c0939abf5381c`，已同步本地、GitHub `experiment/counterfactual-search`、home `/mnt/d/workspace/ygo-agent` 与 H200 `/root/ygo-agent-gpu-20260910/ygo-agent`。
- home WSL 和 H200 均完成 55 项固定 ygocore 协议边界测试。H200 协议报告为 121 个已覆盖分支、0 个未覆盖分支。
- H200 原生模块 SHA-256：`748f969a5e112b90b536f3b2e55042fe079ca173d26d7cbbdc8342f5ce52d22b`；协议覆盖报告 SHA-256：`6fe9f7c92ee2ed68c9b4b9cd81c7574d12b279ffa139e1f664427b9e54b77933`。源码和安装标记分别是 H200 `training-runs/source-deploy-f751782.txt` 与 `training-runs/native-deploy-f751782.txt`；旧模块和旧报告保存在 `training-runs/unselect-forward-f751782-production-backup/`。
- 完整 40 局的 `summary.json`、逐局 `result.json`、`eval.log`、`decisions.jsonl` 和 WindBot 日志保存在 H200 `/root/ygo-agent-gpu-20260910/training-runs/unselect-forward-f751782-windbot40/`。本轮结果目录未生成 `.yrp` 文件，不能当作可播放回放交付。第一版仅隐藏 `Finish` 时撤选的 8 局诊断保存在 `training-runs/unselect-finish-16ca223-windbot8/`，其中 Voiceless 首局仍于 `msg=26` 超限；第二版对应的 8 局均有效，保存在 `training-runs/unselect-forward-f751782-windbot8/`。
- 指定构建服务器当时无法连接 GitHub 获取已有依赖，因此采用 home Ubuntu 22.04 WSL 的 Python 3.10、`native_optimization=false` 便携 release 构建。H200 也是 Ubuntu 22.04；先在 H200 以候选模块隔离导入、跑 55 项测试和 40 局对战，确认后才备份并安装到生产路径。
