# HotpotQA / MuSiQue 标准 Wikidata 三元组最后一次 pilot

日期：2026-09-03  
Experiment ID：`CLAIM-CONSTRAINED-WIKIDATA-HOTPOT-MUSIQUE-PILOT30-V1`  
状态：`FAIL_STOP_WIKIDATA_ONLY_FOR_HOTPOT_AND_MUSIQUE`

## 1. 这次与此前有什么不同

此前执行器依赖 planner 先给出一个 PID，再对精确 `(QID, PID)` 查询。HotpotQA 的主要失败是
PID 与实体/问题关系不匹配；MuSiQue 还存在 subquery 到 canonical relation 的转换缺口。

本次改为 passage-constrained claim grounding：

1. 仍只读 question、冻结 Top-10 passages 和冻结 planner 输出；
2. 用 Wikipedia 标题解析 QID；
3. 对精确 subject QID 扫描其 2020-12-09 截止前的真实 Wikidata claims；
4. 只有 claim tail 被当前 passages 的标题或正文支持，并且 property 与计划关系相容，或它是
   唯一的非元数据 passage-title claim 时才保留；
5. 输出仍为标准 `(head entity, Wikidata property, tail entity/value)`，每跳最多2条、每题最多12条；
6. 不读取 Gold、supporting facts 或 MuSiQue decomposition answer。

这条路线使用 passage **约束真实 Wikidata claim**，但没有从 passage 自由生成关系，因此不同于
passage-derived 临时 KG。

## 2. 构建结果

| 数据集 | n | planner recognized | KG非空 | 完整执行 | 边数 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 30 | 30 | 7（23.3%） | 1（3.3%） | 10 |
| MuSiQue | 30 | 19 | 4（13.3%） | 1（3.3%） | 6 |

完整性门：identity/hash join=1.0、runtime error=0、历史claim验证率=1.0、passage-tail
支持率=1.0、最大每题2条，全部通过。但既有全量替换门 `nonempty>=.80 / complete>=.70`
在两个数据集均失败。

## 3. 同 passages 零训练 A/B

A 为 legacy KG；B 在完全相同 passages、checkpoint、greedy decoding 和 scorer 下，把本次可信边
优先合并进 legacy。无新边的题精确回退 A。

| 数据集 | A EM/F1 | B EM/F1 | EM变化 | gained/lost | 引用率变化 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | .4000/.5686 | .3667/.5283 | -3.33pp | 0/1 | +6.67pp |
| MuSiQue | .2333/.3533 | .2000/.3333 | -3.33pp | 0/1 | +6.67pp |

Hotpot 23 个 fallback 和 MuSiQue 26 个 fallback 的预测逐题一致率均为1.0，说明回退实现正确。
两个数据集都是“模型更多引用新 KG，但答案各少对1题”，因此不是模型没看 KG，而是这些真实
Wikidata边仍不是问题所需的关系。

Hotpot 首次 A/B 的60条预测已全部落盘，但汇总阶段因协议缺少只用于报告的 `scope` 字段报错。
失败manifest保留；修正后的v2输入与原输入逐字节一致，结果直接从完整预测恢复，没有重新生成，
也没有改变prompt、模型、Gold或评分规则。

## 4. 决策与边界

- 停止 HotpotQA/MuSiQue 的 Wikidata-only 开发，不扩到 confirmation、全量训练或 SAEG-v1 W分支。
- 不把本轮16条边注入正式数据；当前 SAEG-v1 对这两个数据集继续使用 P 分支和 no-graph replay。
- 该负结果不影响 2Wiki canonical ProofKG 的既有 `.640/.6944` 结果。
- 本轮不能推出 passage-derived SRO graph 或 DBpedia 无效；这两条尚未做完整实验。

若未来另开方法线，最合理的新变量是：从 retrieved passages 抽取带原句 provenance 的
`(head, relation, tail)` 临时图，并将“passage抽取边”和“Wikidata/DBpedia验证边”分来源记录。
DBpedia 只能作为新增外部资源表，不能静默混入同资源 baseline。

## 5. 可追溯产物

- 协议：`outputs/audits/claim_constrained_wikidata_pilot30_v1_protocol/protocol.json`
- KG记录：`data/derived/claim_constrained_wikidata_pilot30_v1/`
- A/B输入：`outputs/audits/claim_constrained_wikidata_pilot30_v1_ab_inputs_v2/`
- 最终结果：`outputs/audits/claim_constrained_wikidata_pilot30_v1_result/result_record.json`
- Hotpot完整预测/失败manifest：`outputs/validation/claim_constrained_wikidata_hotpot_pilot30_ab_v1/`
- MuSiQue正式A/B报告：`outputs/validation/claim_constrained_wikidata_musique_pilot30_ab_v1/`

