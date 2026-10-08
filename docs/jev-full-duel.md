# 决策模型直接控制完整对局

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

上游依据：
- https://github.com/kw2828/OpenJev （Qwen3-4B 单 token 选项读取实现）
- https://huggingface.co/Qwen/Qwen3-4B （模型、32K 原生上下文、Apache-2.0）
- https://huggingface.co/openjev/openjev （另一个同名 27B 项目，本实验未使用）
