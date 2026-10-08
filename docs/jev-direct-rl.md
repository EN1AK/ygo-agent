# 直接后训练 Jev 类模型：独立分支试验

分支：`experiment/jev-direct-rl`。用户于 2026-10-08 授权将此方向作为独立实验。
本分支使用 `convaiinnovations/laya-multilingual`，固定模型 revision
`1720e3e3357cfe1e281542e223f8273b0890ca34`，Laya 0.4.0。
这是公开可训练的 Jev 类模型，不是 TypeSafe 官方 Jev 权重。

## 首轮范围

直接对预训练 Laya 编码器及其原有决策头进行 on-policy REINFORCE 更新。
同一个模型选择“模拟某条路线”或“提交路线”；模拟器结果进入下一次输入。
没有额外 PPO actor、critic、监督答案或奖励模型。原有置信度头不使用、不训练。
组内轨迹来自相同固定根；使用其他轨迹回报均值作为独立 baseline，逐操作
return-to-go 计算策略损失。采样后只进行一次更新，重算 log probability 必须一致。
关闭 dropout 保持采样和训练一致，但开启梯度。

首轮只覆盖三个开发场景：燎里回收交闪、疾风直攻、疾风已被无效。
它们是已知设定的合成机制环境，使用现有固定对手和 ReferencePolicy 续行。
动作是已枚举的宏路线，不代表完整原始合法动作空间。奖励是达到场景子目标
的 +1 / 未达到的 -1；推演扣 0.02，每个决策最多两次。不是全局胜率奖励。
本分支不把原能力题的 `training_ready=false` 改为 true，不宣称完成原 actor 教学门槛。
新实验使用自己的文本决策接口，需独立完成后续实战资格验证。

每次 probe/commit 都在新进程中运行真实核心，重放到根后执行分支。
验证根、核心、卡库和已见脚本身份；未知或预算耗尽的结果拒绝进入奖励。
初始输入读取卡库原始卡文；分支输入只提供模拟动作及玩家可见的结算状态，
不给 grade、参考动作名称、核心私有状态或“正确答案”标记。
所有原始核心事件和教师验证资料独立保留在日志中。

## 已知限制

- 场景和对手响应已知，不能当作一般隐藏信息对局的公平搜索。
- 可见状态投影不能代替 belief sampling；接完整对局前仍需实现信息集推演。
- 三个场景是开发集，前后得分只能验证闭环，不能证明泛化、牌技提升或优于现有 PPO。
- 此版本最多比较两条完整宏路线，不支持自由展开搜索树或自主决定分支深度。
- 输入长于 token 预算、选项描述被截断或合并时直接报错。
- 正式比较还需相同模型无推演对照、多种子、未见机制家族和完整对局评测。
- `optimizer.pt` 保存实验现场，但 CLI 当前只支持权重初始化；不宣称恢复完整训练连续性。

## 运行

在独立 Python 3.10 环境安装 `requirements-jev-experiment.txt`，复用可用的 PyTorch。
离线主机需预先传入所有 wheel、固定模型权重、encoder config 和 tokenizer。
原 GPU 训练环境是 JAX 环境，不在其中安装这些依赖。

```sh
python -m unittest discover -s tests -p test_jev_experiment.py
python -m scripts.validate_jev_environment --core /path/libexercise_bridge.so \
  --database /path/cards.cdb --scripts /path/script --output /new-path/engine-proof.json
python -m scripts.train_jev_direct_rl --model-dir /path/laya-multilingual \
  --core /path/libexercise_bridge.so --database /path/cards.cdb --scripts /path/script \
  --output /new-path/pilot --updates 3 --group-size 4 --max-probes 2 --device cuda
```

checkpoint 为 Laya 完整 `model.safetensors`，评测原模型目录加
`--init-weights /path/pilot/model.safetensors --updates 0`。默认输出目录不得已存在。
保留 config、输入输出模型哈希、完整交互 JSONL、模型概率、引擎事件、梯度指标、
编码器及决策头参数变化、优化器及 RNG。缺少正常退出和成功 summary 不计为完成。

## 分支隔离与同步

本次用户明确要求独立分支，因此沿用四端留证原则，但不覆盖
`experiment/counterfactual-search` 或两个现有训练目录。
本地独立 worktree、GitHub 同名实验分支；服务器使用
`/root/ygo-agent-gpu-20260910/experiments/jev-direct-rl/`。
home 若可达，使用独立 checkout；不可达时在报告中保留待同步状态。

## 上游

- https://huggingface.co/convaiinnovations/laya-multilingual
- https://github.com/NandhaKishorM/laya
- 模型和 Laya 代码标注 Apache-2.0；本分支调用安装库，不复制上游实现。
