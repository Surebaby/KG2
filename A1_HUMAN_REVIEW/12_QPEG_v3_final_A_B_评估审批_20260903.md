# QPEG-v3 final300×3 A/B 评估审批

## 要批准的动作

只批准一次新的版本化 Evaluation protocol 和本地推理评估；不批准 SFT、PPO、Reward/Loss
修改，也不覆盖任何 baseline 或旧结果。

- A：strong SFT + frozen top-10 passages + no graph；
- B：同 checkpoint、同 qid、同 passages、同 prompt template、同 decoding/scorer + QPEG-v3；
- 唯一变化：B 多一个由相同 passages 确定性派生的 typed evidence graph block。

QPEG-v3 不使用 Wikidata，不新增检索文档，不读取 final answer/support/decomposition；每条边为：

```text
(passage_title, evidence_sentence, full_source_sentence)
```

它应被称为 passage evidence graph，不应声称为 Wikidata factual KG。

## 已完成的前置门

1. train-only family-disjoint holdout：AUC=.8836、runtime precision=.5657、route=.9517；
2. 分数据集 precision=.4682/.7270/.5406，均过冻结门；
3. final900 answer-free 图：nonempty=.830/.877/1.000，max=4，provenance=1，gold=false；
4. 历史 no-graph A 臂与冻结 final prompt SHA256=900/900，可逐题复用；
5. QPEG-v1/v1.1/v2 的失败结果和已消费 confirmation 均保留，不参与 v3 调参。

## 冻结后不可修改的设置

- checkpoint：`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`；
- base model：`models/llama3-8b`；
- 数据集：HotpotQA / 2WikiMultiHopQA / MuSiQue，各300题；
- passages：`outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl`；
- graph：`data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/question_graph_records.jsonl`；
- seed=42，greedy，max_new_tokens=512，top-k passages=10；
- canonical EM/F1 scorer；
- final 打开后不再改 selector、阈值、top-4、edge表示或qid。

## 决策门

主门按预先缩放到 n=300 的规则：

- macro ΔEM(B−A) > 0；
- macro ΔF1(B−A) > 0；
- 任一数据集净损不得超过6/300（2pp）；
- parse rate 不退化超过1pp；
- 同时报 paired bootstrap 95% CI、McNemar、gained/lost/tied；显著性与工程门分开表述。

通过：才允许准备三数据集同资源的 QPEG-grounded SFT 数据，并另行申请 continued-SFT。

失败：QPEG-v3 停止；不继续在 final 上调规则。论文保留“2Wiki额外Wikidata ProofKG强增益 +
跨数据集同资源passage graph负结果”，PPO仍不启动。

## 成本和产物

- A臂不重新生成；只生成B臂900条；
- 本地RTX 4090预计约15–30分钟，精确时长为`UNKNOWN`；
- 新产物必须包含 protocol/manifest、输入和模型hash、逐题prediction、report；
- 不写回 raw、baseline、legacy index 或任何旧实验目录。

## 审批项

请研究者明确回复是否批准：

> 批准冻结并运行 QPEG-v3 final300×3 matched A/B；仅复用已核验的A臂，新增B臂900次推理；
> 不授权训练或Reward/Loss修改。
