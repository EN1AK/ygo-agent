# Structured-lite 40M 对 Stage3 / WindBot 回放

日期：2026-09-22

## 固定资产

- 40M checkpoint：`1790011620_step_000040000000.flax_model`
- 40M SHA-256：`a6ddf4b08d5dba2cbe786b3df8b465cc4b01d8d8b85104187a98cb166a14496f`
- Stage3 checkpoint：`stage3.flax_model`
- Stage3 SHA-256：`4df3832ebdff8cdbffb1d59429eeceff4e8dfa16496db30c3d1d50d1e9537817`
- code list SHA-256：`f54528c6927d7ddd92d411b1423560700fc07c008eba86235e7edb963c45e437`
- elfnote 牌组 SHA-256：`19b1c5aba1a5f00e0d29c9b9776c45de7ecafc6e7e25509a77c1e3ea5f133dcd`
- structured semantic metadata SHA-256：`3ebb70a61545767617704d7aa0ef1e9f376c08f5daffb3cb655fff743aac03c6`
- WindBot revision：`b0a2355f00bd59491add14ff09efa9914a3b47c6`

## 对局一：40M 对 Stage3

- seed：`2026092201`
- 40M：先手，structured-lite-v1/full
- Stage3：后手，legacy-v2
- 双方牌组：冻结的 `elfnote.ydk`
- 结果：Stage3 胜，164 步；40M reward `-4.0`

40M 开局展开蕾吉娜、福尔图娜、狄娜和朱诺拉，Stage3 后手则通过狱神精与耀圣轴持续回收资源，做出加速同调星尘龙和强袭黑羽。中盘强袭黑羽以 5500 攻击力战破福尔图娜，40M 从 8000 LP 降至 4700。Stage3 随后继续铺出朱诺拉等怪兽，并在终局同调召唤地狱俯冲轰炸机。

转折点是地狱俯冲轰炸机发动破坏效果：Stage3 清掉己方多张场面卡以及 40M 的朱诺拉，对 40M 造成 8600 效果伤害并直接结束对局。40M 当时可以用手中的幽鬼兔连锁，但策略选择取消；从结果看，这是本场最可疑、最值得进一步用反事实分支验证的决策。

文件：`stage3-duel/replay.yrp`、`stage3-duel/eval.log`、`stage3-duel/decisions.jsonl`、`stage3-duel/result.json`。决策日志同时记录两个模型，共 164 次决策，其中 40M 61 次、Stage3 103 次。

## 对局二：40M 对 Kashtira WindBot

- seed：`2026092202`
- WindBot：先手，`AI_Kashtira` / `Kashtira` executor
- 40M：后手，elfnote
- 结果：40M 胜，53 步；40M reward `2.0`

40M 后手通过福尔图娜和狱神精展开，做出 3500 攻击力的混沌之双翼。随后利用朱诺白化精、两只米底乌斯和混沌之双翼连续直击，把 WindBot 从 8000 LP 压到 600。WindBot 最后一回合展开独角兽、芬里尔狼、恐吓爪牙族型俱舍怒威族及珠泪哀歌族型俱舍怒威族，但没有形成有效解场。

终局 WindBot 用 2700 攻击力的珠泪哀歌族型俱舍怒威族主动攻击 3500 攻击力的混沌之双翼，承受 800 战斗伤害后 LP 变为 -200，直接败北。这是明显的策略弱点，不是协议或进程故障：对局完整结束、WindBot stderr 为空、49 条 legacy 协议消息成功转换，且没有残留进程。

文件：`windbot-kashtira-duel/attempt-0001/replay.yrp`、`eval.log`、`decisions.jsonl`、`metadata.json`、`command.json`、`result.json` 及 WindBot stdout/stderr。决策日志记录 40M 的 53 次决策；`state_value` 为 critic 的 `V(s)`，不是逐动作 Q 值。

## 说明

这两场是可复现的单局样本，用于回放和策略检查，不能作为胜率估计。Stage3 与 40M 的比较需要后续进行平衡先后手、多 seed 的批量评测；WindBot 的终局自杀攻击表明其 Kashtira executor 仍不适合作为高质量强度上限。

