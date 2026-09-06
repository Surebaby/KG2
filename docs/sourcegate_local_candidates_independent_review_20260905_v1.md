# 本地 α 候选生成独立复核（2026-09-05）

Review ID：`SOURCEGATE-LOCAL-CANDIDATES-INDEPENDENT-REVIEW-20260905-V1`。

**结论：主审计的关键数字已独立复现；生成产物可继续用于既定评分流程，但这些数字还不能证明 α 或 PPO 有效。** 当前结果来自冻结 SFT 的 train-only、K=2 随机候选。ReaRAG 评分和新 α 校准尚未完成，没有 PPO 参数更新。

## 复核对象与方法

- 原始候选：[generations.jsonl](../outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1/generations.jsonl)。
- 冻结输入：[候选银行 manifest](../outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1/manifest.json)。
- 主审计：[report.json](../outputs/audits/sourcegate_local_candidates_quality_20260905_v1/report.json) 与 [candidate_diagnostics.jsonl](../outputs/audits/sourcegate_local_candidates_quality_20260905_v1/candidate_diagnostics.jsonl)。
- 实现：[audit_sourcegate_generated_candidates_v1.py](../scripts/pilot/audit_sourcegate_generated_candidates_v1.py)。

复核在 CPU 上读取原始生成、冻结输入及原始 train Gold，重新调用冻结答案解析/指标函数、轨迹校验器、ProofKG 评分器和特征提取器。使用独立的聚合代码，不调用主审计的 `summarize` 或 `pair_summary`。全部 1,660 条的 EM、F1、valid、m_graph、graph 三个分量及门控特征与主审计逐条一致，差异数为 0。输入和生成 release 的加载校验通过。

这是对数据连接、逐条评分及聚合逻辑的独立复算，仍共享冻结底层评分器，因此不能排除评分器本身的缺陷；下文记录了一个实际解析缺陷。本次仅新建本文档，未修改任何冻结资产、Gold、reward、baseline 或 evaluation protocol；诊断 Gold 不得作为 α 训练目标或输入。

## 关键统计

| 指标 | 复核值 | 正确解释 |
|---|---:|---|
| 问题 / 候选 | 830 / 1,660 | 每题恰好 2 条 |
| 单候选平均 EM | 55.1204819277% | 将两次随机采样作为候选求均值 |
| 单候选平均 F1 | 61.5383829041% | 与主审计仅存在浮点求和末位差异，绝对差小于 1e−15 |
| PPO 格式门后 EM / F1 | 52.6506024096% / 58.6637007571% | 非法轨迹计零后的答案诊断，不是实际 PPO reward 或 PPO 评测 |
| Oracle EM@2 | 67.3493975904% | 事后知道 Gold 才能从两条中选中命中者的上界 |
| 严格格式通过 | 1,472 / 1,660（88.6747%） | 现有解析器/校验器定义下通过，见后述反例 |
| 触及生成上限且无 EOS | 87 / 1,660（5.2410%） | 触及 384-token cap；不自动代表答案残缺 |

Oracle@2 不是可部署单次回答的成绩，也不能据此宣称 PPO 能达到 67.35% EM。两条均 EM 命中的问题为 356，均未命中的为 271，一条命中一条未命中的为 203；这些计数恰好对应上表的单候选 EM 与 oracle@2。

银行由 800 道 graph-eligible 问题和 30 道缺图控制问题组成，全部 830 题中 810 题来自 2Wiki，HotpotQA/MuSiQue 各 10 题。因此总体数字主要描述这批图候选，不能当作三数据集综合 baseline 表，也不能用少量 HotpotQA/MuSiQue 候选推断泛化水平。

## 图分数与门控特征

以下排序只使用 **同一问题的两条轨迹均 valid、均 m_graph=1，且恰好一条 canonical EM 命中、一条未命中** 的 138 组。胜表示命中者分数更高，负表示未命中者更高，差值绝对值不超过 `1e−10` 计平局。

这里的“命中/未命中”是冻结 canonical EM 的机械判定，**不是人工语义正确性标注**。未命中可能包含别名、表达方式或归一化边界，不能一概称为事实错误。

| 分数 | 胜 / 平 / 负 | 平局计半的排序准确率 |
|---|---:|---:|
| Graph 总分 | 112 / 17 / 9 | 87.3188% |
| 纯结构分量 | 31 / 68 / 39 | 47.1014% |
| 图推导答案一致性分量 | 103 / 34 / 1 | 86.9565% |

这支持“Graph 总分在这批可排序候选中与 EM 命中有关联”，但尚不支持“结构过程评分独立识别高质量推理”。纯结构分量在这批样本上没有显示可靠的有利排序；总分的表面效果主要伴随图推导答案一致性分量。后者是图侧推导答案的一致性信号，并非人工过程质量标注，也不等同于直接将 benchmark Gold 输入图评分器。

138 组中 **110 组（79.7101%）两条轨迹的全部门控特征完全相同**。对于仅接收这些特征的确定性 gate，两条会得到相同 α。因此当前特征不能在这些配对中提供逐轨迹的区分依据。主报告的 1,420 条 eligible-valid 候选只出现 54 个唯一特征向量，中心化特征矩阵秩为 3；`link_confidence` 全为 1，也提示输入信息有限。

这不直接证明 learned α 没有价值：相同 α 仍可作用于不同的 Graph/Text 原始分数，跨问题的权重分配也可能有用。是否超过固定 α，必须在 ReaRAG 评分和校准之后，通过既定 matched A−F 对照检验。不能仅凭当前 Graph 排序率替代这项检验。

## 空 Final Answer 解析反例与长度上限

真实候选 `2wikimultihopqa::4827b81c0baf11ebab90acde48001122::k1` 的输出以空的 `[Final Answer]\n` 结束，触及 cap。现有 `extract_final_answer` 返回字面量 `']'`，现有严格轨迹校验仍返回 `valid=True`；其本次 EM/F1 均为 0。最小复现为：

```python
from kgproweight.data.parsers import extract_final_answer
assert extract_final_answer("[Final Answer]") == "]"
```

现有第二个兼容正则会从 `Final Answer` 后将右方括号当作答案。本银行中返回恰好 `']'` 的候选有 1 条。因此 1,472 条 valid 的数字只能解释为当前实现判定通过，不能宣称每条都有非空有效答案。该问题涉及进入 PPO 有效/非法分支的边界，应在正式 PPO 前作为明确版本化修复项评估；本复核未修改解析器，也未重写旧分数。

另一方面，87 条 capped 候选中有 7 条按当前实现 valid，其中 1 条 EM 命中。反例 `2wikimultihopqa::81ff9976086a11ebbd5fac1f6bf848b6::k0` 以完整的 `[Final Answer]\nyes` 结束，`valid=True`、EM=1。故不能把全部 87 条统称为“答案被截断”，也不能仅据 cap 推断增大长度必然修复其余错误。

## 后续判断边界

可以继续既定的 passage-only ReaRAG 评分，以获得比较图/文来源所需的另一侧信号。进入正式 PPO 前，应记录解析边界缺陷的处理决策，并检查新 α 的输出分布、与固定 α 的差异及独立零更新检查结果。若要增强特征或改变过程 reward，需要以新的可追踪版本处理，不能用本轮诊断的 Gold 偷换冻结校准目标。

与主审计配套的 [CPU 聚合测试](../tests/test_audit_sourcegate_generated_candidates_v1.py) 已通过 6/6，覆盖 K=2、同题排序方向、平局、eligible 筛选、相同特征和 oracle@2 与单候选 EM 的区别；这些测试不构成 GPU 或 PPO 验证。

## 产物身份

| 文件 | SHA-256 |
|---|---|
| 主审计 `report.json` | `92edc084d31eb546bf2ce80be8a5a5ac2ccee684f8a2ebf1555e83dd950b6f78` |
| 主审计 `candidate_diagnostics.jsonl` | `a9d3bc1a5e87e14c329e51d4afacd6d55462d8de09d3faacd371f485b0535e03` |
| 原始 `generations.jsonl` | `65d10093b3b922f0e49533dd3d85f760ed4841d12ea2008943b0212513fd6c75` |

主审计记录的 Git HEAD 为 `76f174f8e1206d75bfd43a03dce5fb9d83ad4c43`；具体未提交代码身份以主审计 `source_bindings` 中逐文件 SHA 为准。
