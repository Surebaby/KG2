# SAEG-v1 训练与评估数据集说明

> 版本：2026-09-03 / v1 release  
> 状态：\`PASS_DATASET_RELEASE_NOT_TRAINED_NOT_EVALUATED\`  
> 总审计：\`outputs/audits/saeg_v1_dataset_release_audit/report.json\`

## 1. 这套数据解决了什么

旧 QPEG 把 passage 证据句写成 \`(title, evidence sentence, sentence)\`，虽然内部标记为 passage
edge，但模型输出仍把它放进 \`Knowledge Used\`，容易被误解为标准 KG 三元组。SAEG-v1 正式拆开：

\`\`\`text
[Wikidata Knowledge Graph]
  (Ed Wood, occupation, film director)
[End Wikidata Knowledge Graph]

[Passage Evidence]
[P1] Ed Wood
Sentence: Ed Wood was an American filmmaker.
[End Passage Evidence]
\`\`\`

模型每步输出：

\`\`\`text
Knowledge Used: [(Ed Wood, occupation, film director)]
Passage Used: [P1]
\`\`\`

因此：

- \`Knowledge Used\` 只包含标准 \`(head entity, relation, tail entity)\`；
- passage 证据只用 \`[P<n>]\` 引用，不再伪装成 KG；
- W/P 引用使用独立 parser 验证；legacy parser 和历史实验保持不变。

中间资产 \`data/derived/saeg_v1_training_graph_assets_v1/\` 为了兼容旧构图代码，passage edge
仍有内部 \`triple\` 容器；它不是模型输入，也不是正式 KG。最终 release 已转换为
\`passage_evidence\` 对象，\`kg_subgraph\` 中此类伪三元组为 0。

## 2. 训练集

正式 release：

\`data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl\`

共 4,860 条：

| 数据集 | P-only | W-only | P+W fused | no-graph replay | 合计 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 600 | 0 | 0 | 200 | 800 |
| 2WikiMultiHopQA | 600 | 1230 | 1230 | 200 | 3260 |
| MuSiQue | 600 | 0 | 0 | 200 | 800 |
| 合计 | 1800 | 1230 | 1230 | 600 | 4860 |

说明：

- P supervision 使用 raw train split 的 supporting facts/decomposition，明确标为
  \`gold_train_only=true\`，不能用于评估；
- W 构图不读取 Gold，保留 QID、PID、hop、cutoff provenance；
- master 中 1,191 个 fused 题采用 P+W 联合 target，40 个跨源对齐不完整题 fail-closed
  为 W-only target；另有一组 W/fused 两个变体因跨数据集 held-out family 碰撞被 release-v2
  排除，因此正式 release 联合 target 为 1,190；
- 未调用 Teacher API，所有 target 是已有 train-only 证据的确定性格式转换；
- 每行带归一化 \`sampling_probability\`，正式训练前仍需冻结 epoch size/采样器配置。

原始 4,862 条 master 与排除记录均保留：

- \`data/silver_data/saeg_v1_train4862_seed42_v1/\`；
- \`data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/excluded.identity_only.jsonl\`。

## 3. 评估输入与 Gold 隔离

Answer-free 输入：\`data/derived/saeg_v1_evaluation_inputs_seed42_v1/\`

| 分区 | 规模 | 用途 | 能否调参/选 checkpoint |
|---|---:|---|---|
| development | 150（50/数据集） | 方法开发与 checkpoint 门 | 可以 |
| confirmation | 300（100/数据集） | development 全门通过后一次性确认 | 当前不可以，仍 sealed |
| canonical reporting | 900（300/数据集） | 与历史 baseline 同 qid/retrieval 主表 | 不可以，非确认集 |

Scorer-only Gold：\`data/derived/saeg_v1_scorer_gold_seed42_v2/\`

- 与 answer-free 输入按 \`(role, dataset::qid, question_sha256)\` 1.0 join；
- inference input 中答案/Gold 字段为 0；
- confirmation Gold 文件存在但标记 \`sealed=true\`，尚无 SAEG 模型预测或评分。

## 4. 资源口径

同资源主表：

- A：frozen Top-10 passages，无图；
- B：相同 passages + Passage-QPEG；
- 新 SFT/PPO 模型也必须复用相同 qid、passages、decoding 与 scorer。

这能与已有 passage-based baselines 公平比较，因为 P 图不读取新的外部语料。

额外资源表：

- 只在已通过结构门的 canonical 2Wiki 300 题报告 Wikidata-only / fused；
- 该分支消费额外 Wikidata，不能与同资源 baseline 直接作单变量归因；
- fresh 2Wiki 150 闭包结果 nonempty=.760、complete=.600，未过 \`.80/.70\` 门，故 fail-closed；
- HotpotQA/MuSiQue W 分支沿用此前冻结的结构失败结论，不注入低质量 W 边。

2026-09-03又完成一次独立的claim-constrained历史Wikidata pilot：即便扫描精确QID的真实
claims并用冻结passages约束tail，Hotpot/MuSiQue仍只有7/30与4/30非空；同passages A/B均
gained=0、lost=1。因此release保持不变，不将这些边回填到W分支。详见
`docs/claim_constrained_wikidata_pilot30_v1.md`。

同日新增的零样本本地模型 passage-SRO pilot 也未改变release：Hotpot/MuSiQue各30题都只有
13题非空，且JSON可解析率为`.667/.600`，未过预注册结构门，因而没有进入utility A/B或回填
P分支。该失败与QPEG-v4训练资产相互独立，详见`docs/passage_sro_llama_pilot30_v1.md`。

## 5. 已通过的数据门

- train/eval qid overlap = 0；
- train/development family overlap = 0；
- train/confirmation family overlap = 0；
- development/confirmation family overlap = 0；
- passage 伪三元组进入 KG 字段 = 0；
- 无效训练 citation step = 0；
- answer-free/Gold identity join = 1.0；
- token 审计：训练 max=3,287，评估 prompt max=2,743，全部低于 4,096；
- QPEG-v4 25/50/75 的失败结果和 fresh 2Wiki W 结构失败均保留。

## 6. 尚未完成

本次只完成数据与数据接口，不代表模型效果：

1. 冻结实际训练 epoch size、weighted sampler 与 continued-SFT 超参；
2. 先跑 CPU/GPU 小规模 SFT data-loading smoke；
3. 只在 development 运行新的 SAEG 四臂；
4. development 全门通过后才允许一次性打开 confirmation；
5. reward、α-v2、PPO-O/PPO-K 仍未修改或批准。
