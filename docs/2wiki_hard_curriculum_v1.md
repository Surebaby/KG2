# 2Wiki ProofKG hard-contrastive curriculum v1

## 一句话结论

在未消费、family-disjoint 的82题reserve上，reward-v2.1通过全部预注册的“随机rollout内部可排序”
门，可以进入PPO-O/PPO-K**配置准备**；这不是PPO效果，也没有批准或启动训练。

## 为什么需要新队列

此前普通train-only队列的greedy EM为`.90–.92`，oracle@4只多约5pp。reward-v2.1虽然在未见
题上pairwise达到`.80`，但top1仍比greedy低5pp，因此旧实验保持FAIL，不能改写。

本轮改变的是训练队列而不是reward：从421个已经生成greedy+K4且有完整自动ProofKG的train
qid中，只保留K=4同时含正确和错误轨迹的208题。它们提供同题正负对比：

- recovery：25题，greedy错且K4内有正误轨迹；
- stability：183题，greedy对但K4内也会采到错误轨迹；
- 采样权重：两层各50%，层内均匀；
- 所有题ProofKG complete=true；
- 与reserve的qid/family交叉均为0。

更宽的“greedy错且至少一个sample正确”共有30题，其中5题四个sample全对、没有正负对比，
所以不进入contrastive curriculum。第一次协议冻结脚本因Python布尔字面量拼写错误在协议写出前
失败，失败目录保留；正式协议在新的v2目录冻结，未覆盖旧目录。

## reserve82 结果

每题同一个strong-SFT checkpoint生成1条greedy与4条temperature=1.0 rollout，共410条；reward
评分不读取Gold，Gold只用于事后EM门判定。

| 指标 | 预注册门 | 实测 | 判定 |
|---|---:|---:|---|
| sampled valid rate | ≥.90 | .9878 | PASS |
| mixed-outcome qid | ≥25 | 45 | PASS |
| correct-vs-wrong pairwise | ≥.65 | .6961 | PASS |
| reward-top1 − random sampled EM | ≥+.10 | +.1944 | PASS |
| runtime errors | 0 | 0 | PASS |

必须报告但不作为本协议门的结果：greedy EM=`.6829`、oracle@4 EM=`.8902`，探索空间
`+20.73pp`；reward-top1 EM=`.7439`，比greedy高`+6.10pp`。18/82题属于greedy错但K4
可答对的recovery题。

这说明reward-v2.1在这个新的难题/对比队列上能够提供PPO所需的相对优势信号。它不证明PPO
更新后一定超过SFT，也不把旧top1-vs-greedy失败改成成功。

## 奖励信号边界

在valid mixed轨迹上，组合分数与EM的Spearman为`.352`。主要正信号仍是
`A_answer_consistency`（`.408`）；P/H/O/G单项相关分别约`−.017/.082/.114/.064`。因此后续
必须通过唯一变量的PPO-K−PPO-O判断KG过程项是否真的带来训练增益，不能仅凭本轮排序门声称
“引用覆盖奖励有效”。

## 下一步允许与禁止

允许：准备两份锁定配置，起点、208题schedule、seed、KL、replay、rollout和更新预算完全一致；
PPO-O只用outcome，PPO-K只增加冻结reward-v2.1。

仍禁止：未经研究者批准启动训练；把reserve82用于训练；修改reward公式/权重后复用本reserve；
把零更新rankability写成最终模型提升。

## 证据

- 正式协议：`outputs/audits/2wiki_hard_curriculum_v1_protocol_v2/protocol.json`；
- 训练qid：`outputs/audits/2wiki_hard_curriculum_v1_protocol_v2/train_contrastive_qids.jsonl`；
- 410条候选：`outputs/validation/2wiki_hard_curriculum_v1_reserve82_candidates/candidates.jsonl`；
- 最终机器结果：`outputs/audits/2wiki_hard_curriculum_v1_reserve82_result/result_record.json`。
