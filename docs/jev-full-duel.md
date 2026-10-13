# 决策模型直接控制完整对局

已在 H200 完成 Qwen3-4B 的正常对局、真实分支回传、9373-token 输入与直接 LoRA 更新。
实际运行数据、零梯度首轮和限制见 [验证报告](../reports/jev-full-duel-20261008.md)。

入口 `python -m scripts.run_jev_duel`。本接口使用正常牌表、8000 LP、正常起手和抽牌，
直到模拟器自然终局；超时、原生无效局、上下文溢出都报错，不记作完成或胜利。

## 输入和输出

原生 `YGOProEnvImpl` 负责合法菜单和响应编码。模型从当前菜单选择原生索引，
`env.step(index)` 执行选择；下一次菜单仍由模型回答。发动、对象、素材、表示形式、
连锁响应、阶段切换均使用这个接口。多选类决策按原适配器逐步选择，完成后才提交
核心字节响应。适配器原有的唯一/强制响应保留并记入原始轨迹；不调用 ReferencePolicy。

输入来自核心动态 query 和已经处理的事件帧，包括当前攻守、无效状态、卡片区域、
表示形式、素材、指示物、装备/对象关系、LP、阶段、回合和连锁事件。卡文及具体
效果选项读取卡库。近期事件明确标记窗口大小，未解释事件类型单列；原始帧保留审计。
状态按表格序列化，减少重复字段名，任何 token 截断均拒绝执行。
`--recent-events` 控制可见事件窗口：Laya 默认 20，Qwen 默认 100；省略的更早事件数量
始终明确写入输入。窗口可扩大，但仍受上下文上限约束，不声称保留整局全部历史。

双方分别获得自己的可见观察；不输入牌库顺序或对方手牌/盖卡身份。已公开的连锁、
移动、抽牌、结算信息通过白名单转换。当前隐藏信息处理仍应在更多机制上审计。

## 分支推演

默认 `--search-budget 0`。诊断时使用 `--search-mode oracle --search-budget 2 --horizon 4`：

1. 同一模型可选择执行动作或模拟动作。
2. 新子进程从相同牌组、种子和实际动作前缀重放，校验根状态、菜单、事件、响应摘要。
3. 执行所选动作，再由同一模型通过相同逐步接口选择后续动作，直到指定深度或自然终局。
4. 将该玩家能看到的状态变化、新公开事件和新卡文送回模型；截断分支只标记 cutoff。
5. 校验主对局没有变化，由模型决定继续模拟还是实际执行。

精确重放使用真实隐藏状态，属于 **oracle 诊断**。输出观察隐藏私有卡牌不等于搜索公平：
未来结果仍依赖真实对方手牌及牌库。尚未实现信息集采样，不能将搜索结果用作公平实战胜率。

## 模型后端

- `--backend laya`：既有 Laya multilingual，8192 token 上限，直接更新编码器和选择头。
- `--backend openjev-qwen`：Qwen3-4B，固定 revision
  `1cfa9a7208912126459214e8b04321603b3df60c`，使用原生 32768 token 范围。
  它采用 OpenJev 开源实现中的单 token 选项评分思路，不是 `openjev/openjev` 的 27B 权重。
  不生成解释或解析自由文本。超过 26 个选项时使用分组条件分布，保留全部动作的概率和梯度。
  这不是上游的完整服务或校准复现，概率只表示当前选项集下的策略概率。

Qwen 使用注意力 Q/V 投影内的 rank-8、alpha-16 LoRA，冻结基座；可学习参数仍位于
语言模型内部，没有第二个 PPO actor。每次请求保留全部编码、采样概率和模型输入快照。

## 后训练

默认 `--updates 0` 只验证完整对局。`--updates 1 --group-size 2` 采集同一个开局的
多条采样轨迹，使用 REINFORCE 和其他轨迹的回报均值作 baseline，只更新座位 0。
座位 1 在整组采样期间使用冻结的当前参数，下一组同步，是最小自博弈设置。
真实对局自然胜负为 +1/-1，平局 0；学习方自己调用推演扣除成本。
学习方发起的分支里，由学习方采样的后续动作也接收后续真实回报，假设分支胜利不当作真实奖励。
采样和更新的 log probability 必须一致；模型 dropout 关闭或为 0。
默认 `--baseline leave-one-out`；整组回报相同且没有中间成本时，优势为零，可能不会更新。
`--baseline zero` 可显式选择固定零 baseline 的普通 REINFORCE，用于小批量链路检查，
代价是更高的梯度方差；不会按当前组的成败自动切换 baseline。零梯度运行保留原记录，
摘要的 `parameter_update_verified` 只有在实际参数发生变化时才为真。

Qwen 保存 `adapter.safetensors` 和描述文件、优化器及 RNG；Laya 保存完整权重。
`--init-weights` 支持重新加载相应权重/适配器，不宣称恢复完整训练连续性。

## 运行示例

```sh
python -m scripts.run_jev_duel \
  --backend openjev-qwen --model-dir /path/qwen3-4b \
  --native /path/jev_duel_native.cpython-310-x86_64-linux-gnu.so \
  --database /path/cards.cdb --scripts /path/script \
  --code-list /path/code_list.txt --semantics /path/semantics \
  --deck1 assets/deck/unused/OldSchool.ydk --deck2 assets/deck/unused/OldSchool.ydk \
  --output /new/path --duels 1 --search-mode oracle --search-budget 2
```

`scripts.download_jev_qwen` 下载固定公开基座并校验分片哈希；国内镜像只是传输备选，
仍要求与固定 Hugging Face 权重完全一致。原生模块由
`scripts.exercises.prepare_jev_duel` 在独立构建目录生成，复用已验证的 0.0.4 核心对象；
生产原生模块及生产训练环境不替换。

每局保留输入/模型概率/动作 JSONL、精确原生响应、事件、状态、初始牌序和独立重放验证。
标准牌表只检查主卡数量、额外数量和单卡上限，不声称符合当前赛事禁限卡表。
当前对手是同模型而非 WindBot，完整运行不意味着有牌技或对现有策略的强度收益。

## 并行采集和计算

`scripts.run_jev_parallel` 从已有完整对局的 `config.json` 读取固定模型和模拟器资产。
它参考 JAX 训练入口的 actor 队列与参数同步方式：多个独立原生对局进程通过队列请求
同一个 GPU 策略，每组采样完成后才更新 LoRA，随后双方同时使用新参数。模型仍直接选择
模拟器操作，不增加 PPO 策略。每局有独立采样 RNG，不由线程到达顺序决定采样结果。

```sh
python -m scripts.run_jev_parallel \
  --config /path/to/existing-duel/config.json --output /new/result/path \
  --actors 4 --batch-size 1 --episodes 4 --updates 1 --baseline zero \
  --no-checkpoint
```

`--no-checkpoint` 用更多显存换取更少的训练重计算，仅适用于显存足够的设备和输入长度；
默认保留梯度检查点。H200 的数千 token 输入中，较大批量未必更快；`--batch-size` 同时
控制服务聚合和提示微批量，需要以实际测量选择。4 个 actor 表示 4 局同时推进，不等于
4 张 GPU。当前入口使用单张可见 CUDA 卡，尚无多卡梯度同步实现。

批量计算按长度分组，右侧补齐，并用因果注意力读取最后一个真实 token；完整输入和全部
合法选项保留。训练按原始采样批次重算，包括必要的零损失对手样本，严格检查采样与更新的
log probability 差异不超过 `1e-4`。每局独立重放检查、分支审计和自然终局要求仍然执行。

入口保存配置、运行时资产哈希、推理批次成员、逐局日志、分阶段耗时、峰值显存、梯度及
实际参数变化。`--vary-seeds` 可改变每局初始种子；默认同开局采样不同轨迹。性能实验使用
零 baseline，不因一组的回报恰好相同而在运行中切换算法。留一均值 baseline 仍可显式选择。
`--init-weights` 只加载适配器，不恢复优化器或声明完整断点续训。

完整性能对照见 [2026-10-12 测试报告](../reports/jev-parallel-performance-20261012.md)。

## 前缀 KV 缓存

在上述并行命令上添加 `--prefix-mode reuse --no-checkpoint --batch-size 1` 启用。
`--prefix-mode recompute` 使用完全相同的卡文前置输入、注意力实现和前缀切分，
但每次重新计算前缀，作为严格概率一致的对照；默认 `off` 保留原输入格式。
切换输入布局可能改变模型策略，不能直接用不同布局的两次对局时长推断缓存收益。

缓存按真实 token 前缀匹配，默认最多 8 项、合计 32768 个前缀 token；可以用
`--prefix-cache-entries`、`--prefix-cache-tokens` 调整。超过容量的单个前缀正常重算。
输入仍包含当前可见卡文、模拟器状态、近期事件、分支结果和合法操作，无截断，
不把假设分支的卡文累积进主对局。只有完全相同的前缀跨请求复用；变化后的场面后缀
仍然计算，并关注缓存中的全部历史 token。当前不是整局事件流的无限增量缓存。

推理 KV 在参数版本变化时失效。每个后缀用独立缓存容器，防止假设分支污染其他请求。
训练中，后缀对缓存 KV 的梯度逐次以 FP32 累加，淘汰或组末时重建一次前缀计算图，
将累积梯度反传到同一个 LoRA；没有丢掉前缀梯度，也没有增加 PPO 策略。
BF16 梯度求和次序不同可能产生数值差异，不能声称参数更新逐位相同。
模型概率的 `1e-4` 校验阈值保持不变。当前实现限单条提示、不补齐、零 dropout、
全注意力，并要求关闭梯度检查点；没有实现多卡共享缓存。

`scripts.benchmark_jev_prefix` 可在固定真实输入上比较单次完整前向、拆分重算与缓存，
并保存推理/训练耗时、缓存 token 命中、完整梯度和峰值显存。

上游依据：
- https://github.com/kw2828/OpenJev （Qwen3-4B 单 token 选项读取实现）
- https://huggingface.co/Qwen/Qwen3-4B （模型、32K 原生上下文、Apache-2.0）
- https://huggingface.co/openjev/openjev （另一个同名 27B 项目，本实验未使用）
