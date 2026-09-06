# 固定20题的 Gold-free 证据供给探测

2026-09-06。数据准备实验 `EVIDENCE-SUPPLY-CONSUMED-MUSIQUE20-20260906-V1` 已完成，状态为 `COMPLETE_DEVELOPMENT_ONLY`。这是已消费训练问题上的供给诊断，不是新数据封版、PPO结果或独立确认。Reader的40条候选及答案评价由另一个冻结实验处理。

## 已完成内容

沿用原60题长度/格式探测中的全部20题MuSiQue，保持题目、family、SFT起点、原system提示、空KG、最终10段和Reader输入6144/output384预算。新供给不读取Gold、原始QA答案、官方support、先前语义审阅记录或Reader输出。

冻结的一次算法为：从原前3段中抽取英文大写专名词面；剔除题面已有专名及包含题面已有专名的长串；按出现段数、最早段rank、字符位置、规范化串顺序取最多3个；查询为“原问题 + 专名”。每条查询执行本地Wiki18 E5@100、BM25@100、RRF(k=60)@50、BGE重排。BGE使用对应扩展查询，输入passage前1200字符。

最终保留原rank顺序的前4个不同docid且不同规范化全文的段，然后从3路新文档排名轮询补至10，短缺时按原段顺序回填。所有20题保留，没有按检索结果替换问题或重试查询。`train_12312` 原段1/2、4/5的规范化全文重复，因此该题保留原rank1/3/4/6，其余19题保留原top4。

## 实际运行与审计

| 项目 | 结果 |
|---|---:|
| 完整保留问题 | 20/20 |
| 固定扩展查询 | 60 |
| E5 / BM25 query calls | 60 / 60 |
| BGE query-passage pairs | 3000 |
| 每题原独立段 / 新段 | 4 / 6 |
| 原 / 新prompt总tokens | 35,441 / 35,340 |
| 新prompt最大tokens | 1904 |
| 总耗时 | 148.95秒 |
| GPU peak allocated / reserved | 2.71 / 4.33 GiB |

本地模型为E5-base-v2和bge-reranker-v2-m3；BGE实际float32、max_length8192、CUDA，sentence-transformers5.6.0。canonical passage pack只是既有的估算函数，不能被描述为严格token裁剪；最终实际tokenizer计数和6144上限另外验证。

新脚本15项单测通过。独立CPU重构检查对60条原始检索trace重算RRF值和次序、验证BGE排序，对20条新输入重建final10、messages、prompt及input hash，全部一致；ID和规范化全文去重、原system/KG/其他来源字段保持不变。该审计不评价语义支持充分性，也不计算答案指标。

## 排序合同疑点的核查

阅读旧代码时发现NumPy mmap搜索未显式排序argpartition结果；在没有实证前没有修改该函数。合成例以及独立 `DENSE-RANK-CONTRACT-CONSUMED20-20260906-V1` 实检显示：当前NumPy2.2.6上20/20实际dense top100已降序，0逆序，top1全为最高分。13题共28个相邻完全相等分数；引入docid tie规则只会在10题的等分段内重排，20题分数序列逐值不变。

因此没有采用所谓排序修复，也没有声称旧数据或baseline存在该错误。原始诊断report记为 `RETURNED_ORDER_DIFFERS_FROM_STABLE_SORT`，随后追加的 `tie_scope_addendum.json` 明确这是新tie政策的区别，当前20题降序缺陷未复现。合成审计最早报告的status先于结果错误标记reproduced，原文件保留；权威manifest指向修正后的 `report_v3.json=NOT_REPRODUCED`。

## 比较边界

- 本实验增加上游检索预算，只匹配Reader最终资源。不能将供给变化直接归因于PPO、α或过程奖励。
- 新prep顺序执行E5/BM25以确保每个分支都返回100项，不允许原router吞异常后降级；以固定dense+sparse次序调用既有RRF。旧router并发按完成顺序拼接，等RRF分数的插入顺序可能不同。因此不声称严格复现旧router的全部tie行为。
- 专名抽取是通用词面heuristic，不是多跳语义规划。可能漏桥接、加入无关实体，也可能移除原rank靠后的必要证据；所有移除原rank与新段来源已经记录。120个新文档本身不是质量改善证据。
- corpus/index/model/tokenizer共59,429,745,503字节在冻结前全SHA一次；运行首尾核对device/inode/bytes/mtime_ns，代码首尾全SHA验证。这不等于运行首尾对整库再独立重hash。
- 原3000题、replay、baseline、Gold、reward和α没有改动；未消费fresh132，也没有模型参数更新。

## 权威入口

- [供给协议](../outputs/audits/evidence_supply_v1_consumed20_20260906_v1/protocol.json)，SHA `2efad55b04c3949370dbeaedd5bbf6c0bf40df19b25f51e32823ed766eceed3f`。
- [供给manifest](../outputs/audits/evidence_supply_v1_consumed20_20260906_v1/manifest.json)、[独立输入重构](../outputs/audits/evidence_supply_v1_consumed20_20260906_v1/materialized_input_audit.json)。
- 新 `inputs.jsonl` SHA：`d6bb5105867efcb40566b3b4cc0f7b199702b9df91516d48b845e2bd8fe91251`。
- [实际20题dense排序审计](../outputs/audits/dense_rank_contract_consumed20_20260906_v1/manifest.json)、[tie边界补记](../outputs/audits/dense_rank_contract_consumed20_20260906_v1/tie_scope_addendum.json)。
- [合成排序审计权威manifest](../outputs/audits/memmap_rank_contract_readonly_20260906_v1/manifest.json)。
- [Reader实验预先冻结协议](../outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1/protocol.json)。

实际检索命令（已执行，产物禁止重跑覆盖）：

```bash
PYTHONPATH=/home/zjulab/kgpaper:/home/zjulab/kgpaper/flashrag_src \
KGPW_PROJECT_ROOT=/home/zjulab/kgpaper HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/zjulab/anaconda3/envs/kgpaper/bin/python -u \
outputs/audits/evidence_supply_v1_consumed20_20260906_v1/probe.executed.py run
```
