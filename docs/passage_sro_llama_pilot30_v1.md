# Passage-SRO 本地模型小样本最终记录

## 结论

本轮没有进入 EM/F1 A/B。HotpotQA 与 MuSiQue 都未通过在模型抽取前冻结的结构门，正式状态为
`FAIL_STOP_STRUCTURE_NO_UTILITY_EVALUATION`。因此不能声称该方法提升或降低 EM/F1；正确结论仅是：
当前 Llama-3-8B 零样本抽取器无法稳定地产出足够多、可机械验证的 passage-SRO 图。

## 这次实际测试了什么

输入只包含问题与已经冻结的 Top-10 retrieved passages；不读取答案、supporting facts 或
decomposition，也不访问网络。模型最多提出四条：

```text
(head entity, canonical relation, tail entity/literal)
```

`evidence_quote` 与 `relation_trigger` 是独立 provenance，不进入三元组 relation。只有 passage、
quote、head、tail、trigger、冻结关系表全部一致的边才保留；其余 fail-closed。

这与历史实验的差异是：QPEG-v1/v1.1 用固定正则抽三元组，QPEG-v3 用完整证据句作为 passage
edge；本轮使用本地模型发现开放表述，再用原文逐字校验。它仍是 passage-derived evidence graph，
不是 Wikidata/DBpedia fact。

## 冻结门与结果

| 数据集 | JSON 可解析 | 非空图 | 保留/拒绝边 | 判定 |
|---|---:|---:|---:|---|
| HotpotQA | 20/30 = 66.7% | 13/30 = 43.3% | 26/46 | FAIL |
| MuSiQue | 18/30 = 60.0% | 13/30 = 43.3% | 27/37 | FAIL |

冻结要求为每数据集 parse≥90%、nonempty≥50%、抽取错误为0。两数据集均未通过。22个抽取错误
都是模型生成了非法 JSON（主要是 passage 原文内双引号未转义），不是系统崩溃；但在冻结协议
里它们仍按抽取失败处理，不能事后修改口径。

拒绝边的主要原因也一致：两数据集各有22条 relation 不在冻结词表；其次是 quote/head/tail
或 trigger 无法在指定 passage 中逐字核验。严格校验降低了覆盖，但阻止了模型自由生成的文本
被误当作事实边。

此外，逐字 provenance 只证明“文本来自原 passage”，不自动证明模型赋予的 canonical
relation 在语义上正确。事后查看发现个别关系标签并不由原句蕴含，因此即便把 JSON 修好、把
nonempty 从43.3%推过50%，也不能直接假设 utility 会变正。

## 为什么没有继续跑 A/B

A/B 的前置条件已经在任何模型输出产生前冻结。结构门失败后继续跑 utility 会违反项目的门控
规则，并产生“看见结果后放宽方法”的选择偏差。本轮没有打开 confirmation，也没有把边加入
SAEG训练数据。

## 与已有路线合并后的边界

- QPEG-v1.1 regex SRO：macro ΔEM −0.67pp，已停止；
- QPEG-v3 full-sentence graph：macro ΔEM −1.22pp，已停止；
- claim-constrained historical Wikidata：Hotpot/MuSiQue 都 gained=0、lost=1，已停止；
- 本轮 local-model passage SRO：结构门失败，未做 utility A/B。

所以“简单从 retrieved passages 构图”已经覆盖了正则、证据句、零样本模型 SRO 三种实现，
目前都没有形成可晋级路线。DBpedia、训练过的关系抽取器或 learned semantic verifier 尚未测试，
不能写成无效；但它们是新的资源/模型变量，不应在本轮失败后自动展开。

## 证据

- 冻结协议：`outputs/audits/passage_sro_llama_pilot30_v1_protocol/protocol.json`；
- 原始生成与逐边审计：`data/derived/passage_sro_llama_pilot30_v1_gpu/`；
- 机器结果：`outputs/audits/passage_sro_llama_pilot30_v1_result/result_record.json`；
- fail-closed 测试：`tests/test_passage_sro.py`。
