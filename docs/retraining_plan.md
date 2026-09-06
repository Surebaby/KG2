# KG-ProWeight 当前重训计划

> 活动版：2026-09-03
>
> 历史完整版本：`archive/project_plans/20260902/retraining_plan.full.md`
> 本文件只保留当前已验证事实、下一轮实验和停止门。历史失败与旧决策没有删除，统一在归档版追溯。

## 1. 当前目标

一周内完成一个覆盖 HotpotQA、2WikiMultiHopQA、MuSiQue 的可审计闭环，分别回答：

1. 同资源条件下，结构化 evidence graph 是否比只给 passages 更有效；
2. continued-SFT 是否让模型更会使用该图，同时不破坏原有能力；
3. PPO 是否超过 SFT；
4. KG process reward 是否比普通 outcome-only PPO 提供额外收益。

完整执行方案见：
`A1_HUMAN_REVIEW/11_一周三数据集_SFT_PPO_KG_完整执行方案_20260902.md`。

## 2. 已验证且必须保留的结论

### 2.1 强 SFT 标准 pipeline

| 数据集 | EM |
|---|---:|
| HotpotQA | 0.383 |
| 2WikiMultiHopQA | 0.427 |
| MuSiQue | 0.247 |

### 2.2 2Wiki ProofKG 供给有效

在 canonical fresh passages、同 checkpoint、同 decoding、同 scorer 的 matched control 中：

| 供给 | EM | F1 |
|---|---:|---:|
| strong SFT + legacy KG | 0.4033 | 0.4582 |
| strong SFT + ProofKG | 0.6400 | 0.6944 |

ProofKG 相对 legacy 的 EM 增益为 `+23.7pp`，95% paired bootstrap CI
`[+17.0,+30.3]pp`，McNemar `p=5.3e-11`。该结果使用额外历史 Wikidata 资源，必须作为
resource-augmented 结果单列，不能与不消费该资源的 baseline 冒充严格同资源比较。

### 2.3 当前 PPO-smoke600 没有超过 SFT

canonical 2Wiki n=300 四臂结果：

| 模型 | legacy EM/F1 | ProofKG EM/F1 |
|---|---:|---:|
| strong SFT | .4033/.4582 | .6400/.6944 |
| ProofKG PPO-smoke600 | .3100/.3653 | .6100/.6458 |

`PPO@Proof = -3.0pp`，CI `[-8.33,+2.33]pp`；正式状态是
`NO_PPO_GAIN_OVER_SFT_ON_ALIGNED_PROOFKG`。ProofKG 供给有效不等于现有 PPO 有效。

### 2.4 现有跨数据集 Wikidata-only 路径未走通

- HotpotQA relation-graph v2：plan recognized=1.0，但 nonempty=`0.30`、complete=`0.10`；
- MuSiQue subquery conversion v2：recognized=`0.633`；
- 不降低原冻结结构门，不把这两个失败包装成三数据集通用成功。

### 2.5 旧 continued-SFT 路径已失败

- 8k/50% ProofKG continued-SFT：replay EM `.790→.755`，hidden EM `.576→.485`；
- light20 5k 各 checkpoint 未同时通过 replay 与 hidden 门；
- 新实验不继续扩大这两种数据配方，也不从失败 checkpoint 启动 PPO。

### 2.6 外部 checkpoint baseline 的比较口径

既有 `naive_rag/self_rag/r1_searcher/corag/trace/rearag` 结果继续进入**同资源端到端主表**。
兼容性审计已通过：三数据集各 n=300 的 qid/question 全匹配，均使用 Wiki18、seed42、
确定性解码，且900题×6方法的既有逐题 EM/F1 与当前 canonical scorer 重算结果完全一致。
`naive_rag/self_rag/r1_searcher/corag` 与冻结 final 的
top-10 文档 ID/顺序逐题一致；TRACE 的文档 ID/顺序一致但 passage 序列化字段不同；ReaRAG
保留其原生检索，因此只作端到端系统比较，不作 matched-passage 因果对照。

QPEG 是本方法的一部分，不能为了“公平”同时送给外部 baseline；主表比较完整系统最终
EM/F1。QPEG、SFT、PPO 分别贡献多少，使用同 checkpoint、同 qid、同 frozen passages、
同 decoding/scorer 的内部 A–F 消融回答。额外 Wikidata ProofKG `.640/.694` 继续单列为
resource-augmented 结果。审计证据：
`outputs/audits/qpeg_v1_external_baseline_compatibility_v2/report.json`。

## 3. 新主线：同资源 QPEG-v1

`question + frozen top-10 passages → Question-conditioned Passage Evidence Graph`

QPEG 只读取 baseline 已获得的问题和 passages，不读取 dev/test answer、support、decomposition，
不查询 Wikidata。每条边保存：

```text
(head_surface, relation_surface, tail_surface,
 passage_id, sentence_index, sentence_sha256, extractor_version)
```

固定规则：实体/日期/数值 mention 抽取；passage title 作为稳定 subject；同句 predicate/object
抽取；跨 passage 同名实体只建立 adjacency、不创造事实；按问题相关性与 bridge connectivity 排序；
规范化去重后最多12边；无法回指原句的边不注入；空图显式走 no-graph。

## 4. 数据计划

| 数据集 | 新 QPEG-grounded | accepted silver replay | 目标合计 |
|---|---:|---:|---:|
| HotpotQA | 800 | 800 | 1,600 |
| 2Wiki | 800 | 800 | 1,600 |
| MuSiQue | 800 | 800 | 1,600 |
| 合计 | 2,400 | 2,400 | 4,800 |

- 原 silver 只读，所有派生数据写新版本并记录 manifest/hash；
- 新轨迹由当前强 SFT 在 QPEG prompt 上生成，不重跑全量外部 Teacher；
- train Gold 只用于接受/拒绝与 outcome reward，不进入图构建；
- 仅接受 EM=1、格式合法、引用都属于可见 QPEG 的轨迹；
- 任一数据集不足600条合格新轨迹时，不复制凑数，数据门记 FAIL。

## 5. 最小实验矩阵

| Arm | checkpoint | 输入 | 主要差值 |
|---|---|---|---|
| A | strong SFT | passages, no-QPEG | 原始能力 |
| B | strong SFT | same passages + QPEG | `B-A` 供给效应 |
| C | QPEG-SFT | same passages + QPEG | `C-B` SFT-learning |
| D | QPEG-SFT | same passages, no-QPEG | `C-D` KG依赖 |
| E | PPO-O | same passages + QPEG | `E-C` outcome PPO |
| F | PPO-K | same passages + QPEG | `F-E` KG reward 净增益 |

最终统一使用 frozen qid/passages、seed42、greedy、canonical scorer；三数据集 n=300 报
EM/F1，n=100 用同一 judge/template 报 IHR，同时给 paired CI、McNemar、gained/lost/tied。

## 6. 训练配置与门

### 6.1 continued-SFT

- 起点：`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`；
- LoRA r=32、alpha=64、dropout=.05，bf16，max length=6144；
- lr=`2e-6`，effective batch=32，最多120 updates；保存step40/80/120；
- 选择最早通过门的 checkpoint：legacy replay parse≥.995、EM≥.780、hidden EM≥.545；
- 三数据集 QPEG dev macro EM/F1 不低于 strong SFT，任一数据集 EM 退化≤1pp；
- 无 checkpoint 通过即保留 strong SFT，SFT 分支记 FAIL，不自动增加步数。

### 6.2 PPO-O / PPO-K

- 起点、train qid、K4 schedule、seed、更新预算和稳定性参数完全一致；
- 每臂 `300 prompts × K4 = 1,200 trajectories`；
- lr=`5e-7`、batch=4、ppo_epochs=1、显式 SFT reference、KL coef=.25、target KL=8、
  10% replay、max_new_tokens=384；
- PPO-O：`4*EM + 0.4*F1`；
- PPO-K：只增加归一化 process-v2.1，拟议权重`.2`；
- 零更新启动门：valid≥.90、oracle@4−greedy≥8pp、pairwise≥.65、process-top1≥greedy；
- 任一门失败则 PPO-K 停止，不在确认集继续手调权重。

## 7. 七天顺序

1. 冻结 pilot50/confirmation100/final300，完成 QPEG schema/builder/tests；
2. pilot50×3 运行 strong-SFT A/B，只允许一次通用规则修正；
3. unseen confirmation100×3，通过后才物化约4.8k SFT 数据；
4. 96GB 远端跑 continued-SFT，本地并行评估三个 checkpoint；
5. 构建 train-only hard cohort，完成 K4 rankability 并冻结 PPO 配置；
6. 96GB 远端顺序运行 PPO-O、PPO-K；
7. 完成三数据集六臂 n=300、统一 IHR 和闭环报告。

## 8. 成功目标与停止条件

- QPEG：三数据集 nonempty≥.80、provenance=1、join=1、gold_access=false；
- SFT：legacy 不退化超过1pp，QPEG macro 不低于 strong SFT；
- PPO：PPO-K 相对 QPEG-SFT macro EM 目标≥+1pp，相对 PPO-O≥+.5pp；
- IHR：PPO-K 相对 QPEG-SFT 或 PPO-O 下降≥3个绝对百分点；
- 上述是工程晋级门，不等同统计显著；论文结论仍看 CI、多seed和协议一致性；
- 任一门失败即保留负结果并停止对应昂贵后续，不换qid、不改Gold、不事后降低门。

## 9. 启动权限

执行前仍需研究者明确批准：

1. 新建版本化 QPEG evaluation protocol；
2. 一次最多120 updates的continued-SFT和两次各1,200 trajectories的PPO；
3. PPO-K接入process-v2.1权重`.2`这一核心Reward修改。

未获批准前，只允许只读审计、普通代码、测试和小规模验证。

## 10. 正式资产入口

- 一周方案：`A1_HUMAN_REVIEW/11_一周三数据集_SFT_PPO_KG_完整执行方案_20260902.md`；
- 当前完整闭环：`A1_HUMAN_REVIEW/10_完整实验闭环与待补文件_20260902.md`；
- 2Wiki canonical 四臂：`outputs/audits/2wiki_n300_sft_ppo_x_legacy_proof_v1/`；
- 项目机器快照：`outputs/audits/project_closed_loop_snapshot_20260902_v1/result_record.json`；
- 历史完整计划：`archive/project_plans/20260902/retraining_plan.full.md`。

## 11. Day 1 完成记录（2026-09-02）

- 已在图构建前冻结三数据集 `pilot50/confirmation100/final300`；各角色 qid 与
  answer-free family 交叉均为0，Gold访问为false；
- pilot/confirmation 共450题已按 canonical Wiki18 检索物化，三数据集各150题、每题10篇；
- 已实现确定性 QPEG-v1、eval fail-closed join、版本化配置及40项相关回归测试；
- 已物化 `data/derived/qpeg_v1_n1350_seed42/`：共1,350题，三数据集各450题，
  nonempty=1.0、provenance=1.0、identity unique、每题≤12边；
- 上述只证明结构和可追溯性，不证明图语义有效。下一门仍是 pilot50×3 的强 SFT A/B，
  未跑出 utility 前不得宣称 QPEG 提升 EM/F1。

## 12. Day 2 pilot 结果与停止决定（2026-09-03）

初版 QPEG pilot50×3 在同 qid、同 passages、同 strong-SFT、同 decoding/scorer 下：

| 数据集 | no-QPEG EM | QPEG EM | ΔEM | gained/lost |
|---|---:|---:|---:|---:|
| HotpotQA | .360 | .320 | −.040 | 2/4 |
| 2Wiki | .400 | .360 | −.040 | 3/5 |
| MuSiQue | .180 | .200 | +.020 | 3/2 |

初版 macro ΔEM=`−2.0pp`，未过门。错误分解发现245条自环和87条 verbatim sentence
伪三元组；按协议使用唯一一次通用修正，QPEG-v1.1 仅删除这两类边，不改 qid、passages、
关系词典、排序或边预算。修正版结果：

| 数据集 | no-QPEG EM | QPEG-v1.1 EM | ΔEM | gained/lost |
|---|---:|---:|---:|---:|
| HotpotQA | .360 | .380 | +.020 | 4/3 |
| 2Wiki | .400 | .340 | −.060 | 3/6 |
| MuSiQue | .180 | .200 | +.020 | 3/2 |

修正版 macro ΔEM=`−0.67pp`；2Wiki 净损3题，超过冻结上限2题。最终状态为
`FAIL_STOP_FINAL_NO_CONFIRMATION`：不再手调规则，不打开 confirmation，不物化4.8k训练数据，
也不启动该路线的 continued-SFT/PPO。该负结果不影响既有2Wiki额外Wikidata ProofKG结论。

证据：`outputs/validation/qpeg_v1_1_precision_pilot50x3_strong_sft_ab_seed42/`。

## 13. QPEG-v2 train-only selector 与 confirmation 最终停止（2026-09-03）

QPEG-v1.1 停止后，新建了独立的 QPEG-v2 方法，而不是继续事后修改 v1.1。v2 仅用
train split 的 supporting facts / decomposition 标注监督边选择器，按 question family
划分 train/dev/holdout，特征不读取答案。train-only holdout 的 AUC=`.8761`、选择边精度
=`.7006`、qid route=`.9307`，说明选择器能识别“来自相关证据句的候选边”。

但在此前未打开的 family-disjoint confirmation100×3 上，同 qid、同 passages、同 strong-SFT、
同 decoding/scorer 的最终结果为：

| 数据集 | no-QPEG EM/F1 | QPEG-v2 EM/F1 | ΔEM | gained/lost |
|---|---:|---:|---:|---:|
| HotpotQA | .420/.558 | .410/.549 | −.010 | 2/3 |
| 2Wiki | .370/.473 | .360/.433 | −.010 | 7/8 |
| MuSiQue | .150/.259 | .120/.277 | −.030 | 1/4 |

macro ΔEM=`−1.67pp`，MuSiQue 净损3题，超过冻结上限2题，正式状态为
`FAIL_STOP_FINAL`。confirmation 已消费，不允许再调阈值、改边或换题；当前 QPEG-v2 不进入
4.8k 数据构建、continued-SFT 或 PPO。

失败分解给出了明确但不改变判定的表示层根因：评估题中答案出现在全部 passages 的比例为
Hotpot/2Wiki/MuSiQue=`64/59/34%`，出现在选择器选中 provenance 原句的比例为
`25/25/13%`，但出现在最终注入的 regex 三元组 tail 中仅为`11/6/3%`。即监督目标是“证据句
相关”，运行时表示却只保留被正则截断的 tail，大量答案承载文本在图序列化时丢失。

证据：

- `outputs/validation/qpeg_v2_selector_confirmation100x3_strong_sft_ab_seed42/`；
- `outputs/audits/qpeg_v2_confirmation_failure_audit_v1/decision_addendum.json`。

## 14. QPEG-v3 证据句图：final A/B 已完成并停止（2026-09-03）

下一候选不再调整 v2 阈值，而是修复上面的监督—表示错配：对 passage 中的完整句子做
train-only、family-disjoint 选择，推理期把选中句序列化为带 provenance 的 typed evidence edge：

```text
(passage_title, evidence_sentence, full_source_sentence)
```

它仍只使用 baseline 已检索到的 question + top-10 passages，不查询 Wikidata，不读取评估
Gold，也不把句子伪装成 Wikidata fact。模型只在三数据集各1,000条train样本上拟合，family
交叉为0。冻结阈值`.77`、每题最多4句。初始报告错误地用“阈值以上全部句子”计算精度，
得到`.509`并记FAIL；原报告保留。按协议真实运行时top-4原样重算（不重训、不改阈值）后：

- holdout AUC=`.8836`；
- runtime selected precision=`.5657`，qid route=`.9517`；
- Hotpot/2Wiki/MuSiQue precision=`.4682/.7270/.5406`，route=`.8586/1.0/1.0`；
- 所有原冻结门通过，状态=`PASS_TRAIN_HOLDOUT_ADVANCE_CORRECTED_RUNTIME_CAP`。

final900 answer-free 图已物化：Hotpot/2Wiki/MuSiQue nonempty=`.830/.877/1.000`，平均
边数=`2.42/2.61/4.00`，最大4，identity唯一、gold_access=false、provenance全部验证通过。
图块平均增加约`131/152/212` tokens，未改变 passages 或检索资源。

已核验旧 n=300 no-KG 预测与冻结 final contexts 的实际 prompt SHA256 为`900/900`一致，
因此未来 final A 臂可逐题复用，B 臂仍必须新生成。证据：

- `outputs/audits/qpeg_v3_sentence_selector_runtime_cap_correction_v1/decision_addendum.json`；
- `data/derived/qpeg_v3_sentence_selector_final900_seed42_v2/report.json`；
- `outputs/audits/qpeg_final_no_graph_prompt_reuse_v1/report.json`。

研究者批准后，已冻结并完成 final300×3 matched-passage A/B。A/B 使用同 qid、同 frozen
passages、同 strong-SFT、同 greedy decoding 和同 canonical scorer；A 臂复用900条逐字节一致
的历史生成，B 臂只增加 QPEG-v3 graph 并新生成900条。结果为：

| 数据集 | no-QPEG EM/F1 | QPEG-v3 EM/F1 | ΔEM | gained/lost |
|---|---:|---:|---:|---:|
| HotpotQA | .390/.515 | .370/.490 | −.020 | 9/15 |
| 2Wiki | .370/.440 | .387/.446 | +.017 | 31/26 |
| MuSiQue | .247/.365 | .213/.323 | −.033 | 9/19 |

macro ΔEM=`−1.22pp`、macro ΔF1=`−2.01pp`。MuSiQue 净损10/300；2Wiki 与 MuSiQue
parse rate 分别下降3.33pp和2.67pp。四项冻结门（macro EM正、macro F1正、单数据集净损
不超过6、parse下降不超过1pp）全部未同时满足，最终状态=`FAIL_STOP_FINAL_VERIFIED`。

独立完整性复核：1800条预测、1800个唯一dataset/qid/arm、每臂每数据集300题、19/19
冻结输入/模型hash、A/B非图字段900/900一致、空图样本逐生成一致，相关测试12项通过。
因此不得在此 final 集上继续调selector，也不构建原拟议4.8k QPEG数据，不启动该路线的
continued-SFT/PPO-O/PPO-K。额外Wikidata 2Wiki ProofKG `.640/.694`仍是独立的
resource-augmented 正结果，不受此同资源负结果影响。

事后诊断（不作为晋级门）进一步表明：三个数据集中实际输出已知图引用的样本，相对匹配
no-graph 的 ΔEM 均为负（Hotpot/2Wiki/MuSiQue=`−4.50/−4.39/−3.76pp`）；2Wiki 的小幅
总增益反而集中在未输出已知图引用的样本。因此当前结果不能解释为“模型学会使用QPEG”。
若继续研究，只能新建 train-only 图schema适配实验并使用全新未见评估 cohort，不能重开或
调参到本 final900。

证据：

- `outputs/validation/qpeg_v3_final300x3_strong_sft_ab_seed42_v2/`；
- `outputs/audits/qpeg_v3_final300x3_result_verification_v1/result_record.json`；
- `outputs/audits/qpeg_v3_final300x3_result_verification_v1/posthoc_failure_decomposition.json`。

## 15. QPEG-v4 train-only 图格式适配（2026-09-03，已获批待远程训练）

QPEG-v3 final 的直接注入结果表明，旧 strong-SFT 并未稳定学会新的全句图引用格式。v4 是
独立实验，不重开或修改已消费 final900。它检验一个更窄的假设：只用 raw-train 的
supporting facts / decomposition 监督完全一致的 evidence-sentence 引用格式，能否让模型在
answer-free QPEG 上受益，同时保留无图能力。

研究者批准后已在构图和训练前冻结：

- train：三数据集各600个train qid，family 与新评估集交叉为0；
- development：全新50题/数据集，仅用于checkpoint选择；
- confirmation：全新100题/数据集，development全门通过后才允许推理；
- 以前QPEG pilot/confirmation/final共450题/数据集按qid和family全部排除。

已物化2400条训练数据：1800条图适配（每数据集600）+600条no-graph replay（每数据集
200）。图样本的每个引用都逐字属于prompt图；no-graph样本引用全空；所有轨迹3–5步，
parser与引用一致性均为2400/2400；未调用Teacher API。完整序列平均1544 tokens、p95=2448、
最大3511，全部低于6144。

全新评估检索和answer-free图也已在训练前完成：

| 数据集 | development | confirmation | QPEG nonempty | 平均边数 |
|---|---:|---:|---:|---:|
| HotpotQA | 50 | 100 | .847 | 2.59 |
| 2Wiki | 50 | 100 | .833 | 2.44 |
| MuSiQue | 50 | 100 | 1.000 | 4.00 |

训练从strong-SFT adapter继续，lr=`2e-6`、effective batch=32、bf16、max length=6144、1 epoch、
严格75 updates，保存step25/50/75。评估四臂为A=strong/no-graph、B=strong/QPEG、
C=adapted/QPEG、D=adapted/no-graph；主效应是interaction=`(C-D)-(B-A)`，避免把普通
多数据集SFT收益误归因于图。仅最早通过全部development门的checkpoint可打开一次
confirmation，之后不得调参。

CPU preflight全部通过后，训练包已同步到远程96GB RTX PRO 6000。前两次启动只在pytest
收集阶段因同步清单遗漏新builder/QPEG模块而退出，GPU训练步为0；两份失败launcher日志
保留。补全导入链后第三次远端38项测试通过，数据/config/protocol hash均一致，模型和strong-SFT
adapter正确加载并完成75 updates。最终manifest=`COMPLETE`，step25/50/75/final四个adapter
均为109,086,416 B；loss按step20/40/60为`.3766/.3005/.2465`，grad norm保持有限；峰值
allocated/reserved显存为`39.25/43.42GB`，无OOM/CPU offload。四个adapter、manifest、loss与
全部成功/失败日志均已下载到本地并核验。

训练期间并行冻结全新development50×3配对输入（A/B除`kg_subgraph`外完全一致，136/150
QPEG非空，confirmation零行）。strong-SFT基线A/B已完成：HotpotQA EM=`.36/.36`、2Wiki
`=.30/.30`、MuSiQue=`.10/.08`；直接加QPEG没有development EM增益。最早checkpoint-25
的C/D现仅在本地4090评估；只有冻结interaction与retention全门通过才允许打开一次
confirmation。远程曾尝试自动衔接评估，但在加载模型前因远程缺少本地base-model路径
fail-fast，GPU未开始推理；该失败日志保留，后续评估统一在本地执行。证据：

checkpoint-25的development C/D已经完成且未过门：macro interaction EM/F1分别为
`−.0267/−.0321`，三个数据集没有一个正interaction；C相对B的macro EM为`−.0067`，
D相对A反而为`+.0200`。这表明无图能力没有遗忘，失败集中在图条件利用与parse稳定性。
按冻结规则不打开confirmation。checkpoint-50也已完成且未过门：macro interaction EM/F1分别为
`−.0200/−.01585`，仅1/3数据集interaction为正。checkpoint-75最终 macro interaction
EM/F1=`+.01333/+.00643`，但仍仅1/3数据集为正，no-graph净损与parse门未过。三候选选择结果
正式为`FAIL_STOP_DEVELOPMENT`，selected checkpoint=null，confirmation保持未打开。

- `outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/`；
- `data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/`；
- `outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/`；
- `data/derived/qpeg_v4_schema_adaptation_eval450_seed42/`；
- `outputs/audits/qpeg_v4_schema_adaptation_sft_preflight_v1/report.json`。
- `outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/`；
- `outputs/validation/qpeg_v4_development_strong_sft_ab_v1/`；
- `checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42/manifest.json`。

## 16. SAEG-ProWeight 双源证据图提案（2026-09-03，方案草案）

研究者同意推进多分支方向：Passage-QPEG负责跨数据集高覆盖，Wikidata-ProofKG负责可对齐
问题的高精度关系事实，no-graph负责fail-closed回退；统一source-aware router只注入通过来源内
有效性门的边。数据release和首轮SFT配置现已冻结，但模型仍为`NOT YET TRAINED`，不能写成
已经提升；development零训练utility门仍是启动GPU训练的前置条件。

完整方案见`docs/source_adaptive_evidence_graph_plan.md`，其中已定义统一schema、P/W/N分支、
融合预算、SFT四臂interaction、source-aware连续reward、rankability硬门、PPO-O/PPO-K配对、
同资源主表与额外Wikidata增强表、一周排期和失败停止条件。最近动作是在不重开QPEG-v4
confirmation的前提下先完成checkpoint-50/75，再做CPU-only P/W资产重叠审计；审计得到真实
重叠率前，训练规模与fused样本量保持`UNKNOWN`。

CPU-only Gate-0资产审计随后完成。初版审计错误地直接比较了QPEG的answer-free lexical
family hash与query-planner的PID/结构family hash；该字段已用append-only addendum标记失效，
其余结果未变。v2统一用answer-free lexical family重算，得到：现有QPEG-2Wiki 600题与
ProofKG 1299题仅6个qid直接重合，但1299/1299均可从raw-train support确定性重建P分支；
重建passages与ProofKG训练passages的标题集、正文集均1299/1299完全一致。排除当前QPEG-v4
development/confirmation的统一family后，仍有1231个安全P+W候选。该结果只证明数据可配对，
不证明融合或训练有效；P分支使用train Gold support，必须标为SFT schema supervision。

候选池已按identity-only形式准备并已物化：P-only 1800、N replay 600、W-only 1231、
P+W fused 1231，共4862个variant、3025个唯一dataset/qid。由于W/fused只存在于2Wiki，不能
直接把4862条等权训练；正式物化前必须先冻结source/dataset-balanced采样策略。

随后已完成train-only统一图资产物化（仍不是SFT targets）：4862/4862 record schema、identity、
question hash和graph hash全部通过；输出包含passage边7551条、Wikidata边6434条、no-graph
600条。1231个fused record均保留P和W两源，asset cap=12下无截断；W边保持
`gold_access=false`及QID/PID/hop/cutoff provenance，P边明确标注使用raw-train Gold support。
对1299个ProofKG上下文的补充复核确认标题与正文不但集合一致，顺序也1299/1299一致，确保
passage rank provenance未错位。

数据采样权重已冻结但训练长度未冻结：先按dataset各`1/3`；Hotpot/MuSiQue内部P/N=`.9/.1`；
2Wiki内部P/W/fused/N=`.2/.3/.4/.1`。汇总来源概率为P=`.6667`、W=`.10`、fused=`.1333`、
N=`.10`，所有4862个候选均有唯一权重并与图资产identity join=1.0。

answer-free跨源对齐审计显示，1231个fused record的3082个W hops中3041个（98.67%）能与不同
P edge一对一对齐；1191/1231（96.75%）题的全部hop均可对齐。研究者批准后已按1191题P+W
联合、40题W-only fail-closed策略生成双字段SFT target；严格family过滤后联合target为1190条。

QPEG-v4 checkpoint-50 development也已完成并失败：macro interaction EM/F1=`−.0200/−.01585`，
仅1/3数据集interaction为正；confirmation未打开。按冻结顺序继续checkpoint-75。

语义检索与门控边界也已明确：canonical E5+BM25/RRF+BGE两阶段检索保留并对所有同资源系统
固定；旧alpha checkpoint不直接继承，source-aware router负责输入证据可用性，未来alpha-v2仅
负责逐步结构奖励/文本奖励权重，且必须经过重新校准、rankability和固定alpha消融后才能进入PPO。

证据：

- `outputs/audits/saeg_v1_training_asset_overlap_v2/`；
- `outputs/audits/saeg_v1_training_candidate_pool_v1/`；
- `data/derived/saeg_v1_training_graph_assets_v1/`；
- `outputs/audits/saeg_v1_training_sampling_weights_v1/`；
- `outputs/audits/saeg_v1_cross_source_alignment_v1/`；
- `outputs/audits/qpeg_v4_schema_adaptation_development_scores_v1/step50.json`。

## 17. SAEG-v1 正式数据 release（2026-09-03，已完成，未训练）

模型可见citation contract已经纠正：Wikidata只用标准`(head, relation, tail)`进入
`Knowledge Used`；passage证据序列化为`[P1] title + sentence`并只进入`Passage Used`。
legacy prompt/parser保持不变，SAEG使用独立双字段prompt/parser。

训练master为4862条。release审计发现一个跨数据集train/confirmation模板family碰撞，对应
W/fused两个variant；不覆盖master，另建family-disjoint release-v2共4860条并重新归一化
采样概率。最终train citation contract错误=0、KG字段中的passage伪三元组=0、Teacher API
调用=0。

评估协议分为：

- development=150（50/数据集），只用于方法/checkpoint决策；
- sealed confirmation=300（100/数据集），尚未生成或评分任何SAEG模型预测；
- canonical reporting=900（300/数据集），只用于与历史baseline对齐，不得调参或选模型。

1350条inference input均answer-free，Gold另存scorer-only v2文件，identity/hash join=1.0。
train/dev/confirmation qid与family重叠均为0。token审计训练最大3287、评估prompt最大2743，
均低于4096。

fresh 2Wiki 150题answer-free ProofKG闭包虽planner合法149/150，但nonempty=.760、
complete=.600，未过预注册.80/.70门，状态`FAIL_STRUCTURAL_NOT_ELIGIBLE`，因此在fresh
development/confirmation中fail-closed。canonical 2Wiki 300的既有ProofKG仍只用于额外资源表。

关键产物：

- `data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/`；
- `data/derived/saeg_v1_evaluation_inputs_seed42_v1/`；
- `data/derived/saeg_v1_scorer_gold_seed42_v2/`；
- `outputs/audits/saeg_v1_evaluation_protocol_v1/`；
- `outputs/audits/saeg_v1_dataset_release_audit/`；
- `docs/saeg_v1_dataset_release.md`。

具体SFT训练设计已在§21冻结；数据release和CPU preflight通过仍不等于模型utility已通过，
也不等于批准修改reward/alpha或绕过development门直接启动PPO。

## 18. HotpotQA/MuSiQue 标准 Wikidata 三元组最后一次尝试（2026-09-03，FAIL_STOP）

研究者批准后，新增独立的`claim-constrained historical Wikidata` pilot：不再只信planner给出的
单一PID，而是在精确subject QID的2020-12-09历史revision中扫描真实claims，再用冻结Top-10
passages约束tail，并要求关系相容或唯一的passage-title claim。全程gold_access=false，输出只含
标准`(head, Wikidata property, tail)`，不把passage句子当relation。

结构结果仍不足：HotpotQA nonempty=`7/30`、complete=`1/30`；MuSiQue nonempty=`4/30`、
complete=`1/30`。16条保留边的历史claim验证率与passage-tail支持率均为1.0，runtime error=0，
但两个数据集均远低于既有`.80/.70`全量替换门。

同passages零训练A/B也失败：Hotpot legacy→增强 EM/F1 `.4000/.5686→.3667/.5283`，
MuSiQue `.2333/.3533→.2000/.3333`；二者均gained=0、lost=1，引用率反而各提升6.67pp。
这证明模型确实消费了新边，但真实Wikidata claim仍不是所需问题关系。按预注册规则停止
Hotpot/MuSiQue Wikidata-only，不开confirmation、不进SAEG-v1 W分支、不训练。

完整报告：`docs/claim_constrained_wikidata_pilot30_v1.md`；机器记录：
`outputs/audits/claim_constrained_wikidata_pilot30_v1_result/result_record.json`。该结论不覆盖
2Wiki额外Wikidata结果，也不能外推为passage-derived临时KG或DBpedia无效。

## 19. Retrieved-passage 标准 SRO 最小试验（2026-09-03，FAIL_STOP）

在确认QPEG-v1/v1.1已经做过regex SRO、QPEG-v3已经做过完整证据句图后，本轮没有重复旧方法，
而是冻结了一个新的本地模型抽取变量：Llama-3-8B从相同Top-10 passages最多提出4条标准
`(head, canonical relation, tail)`，`evidence_quote/relation_trigger`只作为独立provenance；每条边
必须通过原文、head、tail、trigger与冻结关系词表的fail-closed校验。全程不读Gold、不访问网络。

HotpotQA/MuSiQue各30题的结果均为nonempty=`13/30=.433`，低于冻结`.50`门；JSON可解析率仅
`.667/.600`，分别有10/12个模型输出因未转义原文引号等非法JSON失败。其余拒绝主要来自
relation不在冻结词表（两数据集各22条）或surface/provenance无法逐字核验。结构门失败后按协议
没有运行EM/F1 A/B、没有打开confirmation、没有接入训练。

该结果把“简单从retrieved passages构临时图”覆盖到regex SRO、完整证据句和零样本模型SRO
三种实现，但不能外推到DBpedia、训练过的关系抽取器或learned semantic verifier。完整记录：
`docs/passage_sro_llama_pilot30_v1.md`；机器结果：
`outputs/audits/passage_sro_llama_pilot30_v1_result/result_record.json`。

## 20. 2Wiki hard-contrastive curriculum 零更新门（2026-09-03，PASS_PREP_ONLY）

旧reward-v2.1/L0 verifier的top1-vs-greedy失败保持不变。本轮不改reward，只针对旧队列
greedy `.90–.92`、探索空间过窄的问题，新建train-only对比课程：421个完整ProofKG候选题中
208题的K4同时含正确/错误轨迹，其中recovery=25、stability=183，按`.5/.5`两层采样。

在此前冻结且从未生成过候选的family-disjoint reserve82上，生成greedy+K4共410条。全部新门
通过：valid=`.9878`、mixed qid=`45`、pairwise=`.6961`、reward-top1相对随机sample
`+19.44pp`、runtime error=0。必须报告的非门指标为greedy/oracle/reward-top1
`.6829/.8902/.7439`，即该队列探索空间`+20.73pp`且reward-top1比greedy高`+6.10pp`。

主要排序信号仍来自A_answer_consistency（valid mixed Spearman `.408`），其余P/H/O/G较弱。
因此该结果只批准“准备”PPO-O/PPO-K配对配置；能否训练提升必须由两臂唯一变量实验回答，当前
没有启动PPO。完整记录：`docs/2wiki_hard_curriculum_v1.md`。

## 21. SAEG-v1 continued-SFT 一轮训练日程冻结（2026-09-03，PASS_NOT_TRAINED）

结合本地references和2025–2026 SFT→RL研究重新核对后，确认`1,200 trajectories`只能作为
PPO smoke，不能当正式收敛训练；SFT也不能只按最高pass@1选最终checkpoint，而要保留多个
深度检查格式、旧能力保持和K=4探索空间。研究者同意从数据/配置冻结开始推进。

活动TODO中“SAEG SFT targets尚未生成”的旧描述已纠正。实际可训练的family-disjoint release为
`data/silver_data/saeg_v1_train4860_family_disjoint_seed42_v2/silver_train.jsonl`，包含4860条完整
监督轨迹；不是只有图资产。其train/eval qid与family overlap均为0，citation错误和passage伪KG
三元组均为0。

为了避免普通Hugging Face Trainer忽略每行`sampling_probability`而让2Wiki占到67%，新增了
确定性一轮日程：

- epoch size=`4860`，seed=`42`；每数据集精确`1620`次暴露；
- HotpotQA：P/N=`1458/162`；
- 2Wiki：P/W/fused/N=`324/486/648/162`；
- MuSiQue：P/N=`1458/162`；
- 组内先遍历候选再允许重复，最后统一seeded shuffle；不修改target、answer、passage或graph；
- 实际暴露3144个不同variant、2467个不同dataset/qid，1716行为重复暴露，单variant最多3次。

continued-SFT配置已冻结为：strong SFT adapter起点、LoRA r32/alpha64/dropout.05、lr=`2e-6`、
bf16、batch=`4`、grad accumulation=`8`、effective batch=`32`、max length=`4096`、1 epoch。
`ceil(4860/32)=152`次optimizer updates，保存step`38/76/114/152(final)`，用于25/50/75/100%
深度的RL-readiness选择。此前“4860条、effective batch32、最多120 updates”的算术不一致已消除。

CPU preflight共31项全过：4860/4860原release内容逐字段一致、4860/4860 SAEG parser合法、
adapter 256个LoRA张量可读、config继承值正确、输出目录不存在。真实tokenizer/data-loader smoke也
得到4860/4860行，长度`534–3287`、supervised tokens`125–871`、零监督行为0、超过4096为0。

关键产物：

- `scripts/prepare/freeze_saeg_v1_sft_epoch.py`；
- `data/silver_data/saeg_v1_sft_balanced_epoch4860_seed42_v1/`；
- `configs/training/phase3_sft_saeg_v1_balanced_epoch4860_seed42.yaml`；
- `outputs/audits/saeg_v1_sft_balanced_epoch4860_preflight_v1/`。

下一步不是立即远程训练，而是完成SAEG eval runner的N/P/W/F路由与sealed guard，并在development
上先跑strong-SFT零训练utility。只有该门通过，才按已冻结配置启动152-update continued-SFT。
PPO正式预算暂定为两臂各`1,200` smoke → `8k`中程门 → 至少`20k`、最多条件扩展到`40k`
trajectories；正式值要在SFT checkpoint和train-only rankability通过后再锁定。

## 22. SAEG-v1 development runner 与 ReaRAG 职责核对（2026-09-03，IN_PROGRESS）

已实现N/P/W/F路由、answer-free inference/scorer Gold identity join、confirmation sealed guard、
同qid配对统计、bootstrap/McNemar和fail-closed一致性检查。首次冻结的v1 runner发现一个实现口径错误：
它把14个P为空的题从B/D臂删除，而冻结设计要求这些题保留在同一150题配对总体中并精确回退A。
v1在模型加载后、第一题完成前人工中止，没有写prediction、没有消费结果；原`RUNNING` manifest不覆盖，
另写`ABORTED_ADDENDUM.json`说明原因。

修正后的v2协议已在生成前独立冻结：A/B/D各150题，C/W为0题`NOT_EVALUABLE`；其中P非空136题，
P为空14题的B/D必须复用A的逐字节相同prompt/generation。相关SAEG runner/router/parser测试13项全过。
关键产物：

- `outputs/audits/saeg_v1_development_zero_training_protocol_v2/`；
- `scripts/eval/evaluate_saeg_v1_development_utility.py`；
- `tests/test_saeg_development_utility.py`；
- `outputs/validation/saeg_v1_development_strong_sft_npdf_v1/ABORTED_ADDENDUM.json`。

v2首次本地运行模型加载后首题超过90秒仍未返回，已中止等待单题device/latency探针；这不是utility
负结果，也不能据此打开SFT训练门。下一步先确认实际模型device与单题生成时延，再启动完整150题评估。

ReaRAG没有从本阶段漏接：零训练推理utility和continued-SFT均不计算reward，只加载Llama-3-8B与
强SFT LoRA；PPO-O只有outcome reward，也不应加载ReaRAG。只有PPO-K在启用`R_text(t)`时才必须
显式加载冻结ReaRAG-9B并fail-hard。本地`models/rearag-9b`已核到41/41分片、索引与tokenizer完整，
约21.0GB；旧`alpha_gate.pt`不直接继承，source-aware alpha-v2须在新SAEG train-only数据上重新
校准和消融后才可启用。ReaRAG外部baseline结果保持只读，不在每次方法utility中重复运行。

并行完成了不依赖utility数值的SFT可观测性准备：SFT runtime新增默认关闭、显式配置才启用的
TensorBoard转发；SAEG配置写入`/root/tf-logs/SAEG-V1-SFT-BALANCED-EPOCH4860-SEED42`，只记录
SFT loss/lr/grad-norm（SFT没有KL）。新增远程启动器在运行前硬检查utility必须为
`PASS_ZERO_TRAINING_UTILITY`、preflight必须为`PASS_NOT_TRAINED`、输出目录不存在和起点adapter存在。
更新配置后的preflight-v2共33/33项通过，仍未启动训练。产物：

- `outputs/audits/saeg_v1_sft_balanced_epoch4860_preflight_v2/`；
- `scripts/deploy/launch_saeg_v1_sft_remote.sh`。

PPO数据侧同步完成了capacity-only审计，状态`PASS_CAPACITY_ONLY_NOT_MATERIALIZED`。拟议统一池为
三数据集各1800个唯一问题、共5400 prompts，K4后每臂21600条on-policy trajectories；可用同一
有序池的300/2000/5400 prompt前缀分别执行1200 smoke、8000中程和21600正式训练。现有SFT
release含3024个唯一source qid（Hotpot 600、2Wiki 1824、MuSiQue 600），补齐均衡池需新增
Hotpot/MuSiQue各1200，2Wiki无需新增。排除1350个SAEG评估qid/family与533个历史reward审计qid/
family后，三个raw-train的可新增容量仍为88300/55404/19281，容量不是瓶颈。

该审计没有冻结5400题身份、没有构图、没有写Gold/答案、没有批准PPO。只有development utility
通过后，才会单独冻结精确qid/source route、预留fresh family-disjoint reward dev/confirmation，
再物化缺失的P图与训练context。机器记录：`outputs/audits/saeg_v1_ppo_pool_capacity_v1/report.json`。

精确身份冻结器代码已准备但未执行：`scripts/prepare/freeze_saeg_v1_ppo_pool.py`。它在入口硬要求
utility=`PASS_ZERO_TRAINING_UTILITY`，否则拒绝创建输出；通过后才会冻结5400题main schedule，且
预先从SFT/eval/历史reward family之外各保留fresh reward-development/confirmation 100题。main
schedule按数据集轮转，前300题固定为100/100/100；2Wiki使用720 fused、510 W、390 P、180 N，
Hotpot与MuSiQue各使用1620 P、180 N。三个确定性/隔离helper测试已通过。这里的“代码准备完成”
不等于身份已冻结，更不等于图、passages或outcome labels已物化。

## 23. SAEG-v1 development 零训练效用门（2026-09-03，FAIL_STOP_BEFORE_SFT）

v2完整运行结束，A/B/D各150行、C/W=`NOT_EVALUABLE`、14个空P题精确fallback均通过完整性检查。
但主效用门明确失败：Hotpot、2Wiki、MuSiQue的D−A EM分别为`-.08/-.08/-.04`，macro EM/F1为
`-.0667/-.0663`；逐题总计gained=1、lost=11。P非空136题自身EM差`-.0735`，不是空图回退造成。
P臂已知passage citation使用率`.7533`，说明模型会消费P块，但高相关候选句经常缺终点或只覆盖
比较/桥接的一侧，因而把原本正确答案带错。

训练P数据来自raw-train Gold supporting facts/decomposition完整正例；evaluation P来自answer-free
sentence selector。现有release没有自动P partial/misleading输入的回退监督，因此存在明确train/eval
evidence-quality错配。按预注册门，当前continued-SFT保持BLOCKED，PPO 5400题冻结器也不得执行。

建议的下一单变量版本是`P hard-negative alignment v2`：用与eval相同的自动selector构建train输入，
只用train Gold离线分complete/partial/misleading；complete监督引用P，后两类保留可见P块但监督从full
passages作答且不引用错误P。该方案需研究者批准，且必须新建协议/数据版本，不能覆盖本次负结果。
详见`docs/saeg_v1_development_utility_v2.md`。

## 24. Passage-QPEG hard-negative alignment v2（2026-09-03，FAIL_STOP_DATA_GATES）

研究者已批准该单变量数据修复。现有负结果不能简单解释为“Passage句子全部是错误事实”：更准确的
根因是自动selector经常选择局部真实但不构成答案链的相关句、漏掉必需跳，或选择指向错误候选实体的
句子；强SFT又会实际消费该块（known passage citation rate=`.7533`），因此这些不完整/错位证据会
产生误导。训练分布进一步放大了问题：旧P监督来自raw-train Gold supporting facts/decomposition，
MuSiQue训练输入甚至只含Gold support paragraphs；评估P则来自canonical top-10 retrieval后的answer-free
selector。旧训练集中没有“错误P可见、目标选择性忽略”的样本。

已冻结新协议`SAEG-P-HARD-NEGATIVE-ALIGNMENT-V2-SEED42`，保持retriever、selector（threshold `.77`、
top-4）、强SFT起点、loss和evaluation输入不变。自动检索/选择阶段禁止Gold；之后仅在train split用Gold
把P分为`complete/partial/misleading/empty`。target只引用与必需支撑跳精确匹配的P边：partial保留整个
含噪P块但只引用正确子集，misleading保留P块但所有`Passage Used`为空。当前只准备候选数据，不改reward、
不打开confirmation、不启动SFT/PPO。

冻结后复核发现旧QPEG-v4 train1800与后来加入的SAEG全评估角色有0个dataset/qid交叉、但有19个family
交叉（2Wiki 18、MuSiQue 1）。原协议未覆盖这项后来扩展的评估集合，因此用append-only isolation
addendum纠正并排除，形成有效train cohort 1781（Hotpot 600 / 2Wiki 582 / MuSiQue 599），原协议不覆盖。

工程状态：分型/选择性citation candidate builder与相关回归测试已完成；第一次retrieval尝试
因运行中发现完整性谓词会把合法`forbidden_fields=0`当False而主动中止，0数据行、0模型更新，失败目录
和addendum保留。修正后1800题answer-free canonical retrieval全部完成：各数据集600题、全部10 passages、
Gold/forbidden字段0。

冻结exact分型门失败：Hotpot的complete/partial/misleading/empty=`0/48/455/97`，2Wiki=`0/9/497/76`，
MuSiQue=`0/13/586/0`，三集均未达到complete≥20，因此没有冻结SFT日程、没有训练。抽查确认exact匹配受
Wiki版本/标题重复影响，故另做不改变门的post-failure near-exact诊断（同标题、去重复标题前缀、token-F1
≥.90）：Hotpot=`26/208/269/97`，2Wiki=`26/251/229/76`，MuSiQue=`2/182/415/0`；逐边precision仅
`.181/.237/.078`。即exact口径夸大失败，但自动P仍以partial/misaligned为主，MuSiQue尤其严重。

当前v2保持`FAIL_STOP_DATA_GATES`。建议研究者另行批准`paired hard-negative curriculum v2.1`：保留
train-only Gold-complete P作为正例，同问题automatic partial/misleading P作为可见硬负例，并保留no-P
replay；不改loss/reward/eval。它可测试是否恢复当前`−6.7pp`伤害，但要让P产生稳定正增益，后续仍须
单独训练hop-aware evidence-set selector。完整记录见`docs/saeg_p_alignment_v2_result.md`。

## 25. 主线收缩：强 SFT + 2Wiki 配对 PPO（2026-09-03，ACTIVE_NOT_STARTED）

研究者要求以尽快获得可解释、可与baseline比较的结果并完成论文为优先级。SAEG/Passage-QPEG
三数据集扩展到此停止继续迭代：第23节零训练utility为负，第24节数据门失败；near-exact诊断只说明
exact口径偏严，并未改变自动P的低precision/低required-hop recall。失败资产与结论保留，后续不再
围绕同一development集合调selector、数据门或continued-SFT。

活动主线回到现有强SFT起点
`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`，不重新训练SFT。最有证据支持的
后训练数据是2Wiki完整ProofKG hard-contrastive cohort：208个train-only mixed-outcome qid，
recovery/stability按`.5/.5`采样；family-disjoint reserve82上的零更新审计已得到valid=`.9878`、
pairwise=`.6961`、reward-top1相对greedy=`+6.10pp`。这不是PPO结果，但说明该队列具备真实探索空间。

正式因果实验必须成对运行：PPO-O与PPO-K共享SFT、208题、冻结的1,200-trajectory schedule、K=4、
seed、KL、critic、10% replay、decoding和优化预算；唯一研究变量是
`proofkg_process_reward=false/true`。PPO-O奖励为`4*(EM+0.1*F1)`；PPO-K只额外加入
`.2*process-v2.1`。这里“只改reward”是指两条新PPO臂之间；相对历史hybrid PPO，训练cohort与KG供给
已经改变，不能写成历史实验的单变量续跑。

两份配置、v2 lock、共同schedule和CPU preflight已经完成且通过，但尚未启动训练：

- `configs/training/phase3_ppo_2wiki_hard_curriculum_v1_outcome1200_seed42.yaml`；
- `configs/training/phase3_ppo_2wiki_hard_curriculum_v1_process_v2_1_1200_seed42.yaml`；
- `outputs/audits/2wiki_hard_curriculum_v1_paired_ppo_configs_v2/`。

论文比较分两层，不混口径：第一层在同一2Wiki canonical ProofKG pipeline中比较strong-SFT、PPO-O、
PPO-K，`PPO-K−PPO-O`归因过程奖励，`PPO-O−SFT`归因outcome PPO；第二层把ProofKG系统与既有2Wiki
baseline并列，但明确标注额外Wikidata资源。三数据集legacy主表继续报告既有同资源结果，不把2Wiki
ProofKG结果冒充三数据集通用提升。

执行采用一次预注册的逐级预算，而不是反复临时改配置：先两臂各1,200 trajectories；仅当训练健康且
PPO-K在未见development上相对PPO-O方向为正、相对strong-SFT无实质退化时，才为两臂同时冻结相同的
中程扩展预算。checkpoint选择规则必须在评估前冻结；最终重要结论需要多seed或配对置信区间，不能只
挑单seed/单checkpoint。当前状态为`ACTIVE_NOT_STARTED`，启动远程训练仍需研究者明确确认。

## 26. 更正：三数据集混合 PPO 为正式主训练（2026-09-03，PROPOSED_NOT_MATERIALIZED）

第25节把2Wiki配对PPO写成“活动主线”不足以支持跨数据集通用模型结论。研究者明确要求正式PPO必须
混合HotpotQA、2WikiMultiHopQA和MuSiQue；因此第25节的2Wiki-only实验降级为机制消融，不能作为最终
通用模型。该更正不删除第25节，保留决策演变记录。

复核训练资产得到一个此前未充分披露的事实：当前strong-SFT实际只使用HotpotQA train的4,751条
accepted silver；历史hybrid PPO的600 trajectories也全部来自HotpotQA（559 unique qid）。它们在
2Wiki/MuSiQue上的结果是跨数据集泛化，不是三数据集混合训练效果。新的正式主训练需要补上这一实验
设计缺口。

拟议的干净混合奖励为所有数据统一：valid轨迹使用`4*(EM+0.1*F1)`，invalid轨迹为`-4`；PPO-K仅在
完整、identity-safe、`gold_access=false`的自动ProofKG eligible行额外加入`.2*process-v2.1`，PPO-O
不加入。HotpotQA/MuSiQue当前没有通过供给门的ProofKG，因此只获得outcome更新；失败的QPEG/
Passage-QPEG边不得进入prompt或process reward。PPO-O/PPO-K仍必须共享完全相同的混合数据、prompt、
schedule、seed、KL、critic、10% replay与预算，唯一研究变量为process项开关。

当前可直接复用且互相family-disjoint的训练身份为HotpotQA 600、MuSiQue 599和2Wiki hard ProofKG 208，
共1,407 unique questions；canonical answer-free Top-10 passages及2Wiki ProofKG均已落盘。为减少
ProofKG-train/legacy-eval错配，正式冻结前还应为2Wiki加入同源no-graph/legacy view，并使用按数据集
均衡而非按文件行数均匀的sampler。此处只是可用容量与设计结论，混合silver、question-KG records、
共同schedule和配置尚未物化，不能声称已经可启动。

现有代码不能直接干净运行该公式：v2.1启动检查目前错误地要求所有行都具有完整execution，且
ineligible行会回到旧PRM/alpha/ReaRAG composite而非统一outcome。需要新建默认关闭的混合reward路径，
只对eligible行检查execution并加process；不得改变任何历史配置语义。本轮无需重训PRM、旧alpha gate
或加载ReaRAG。v2.1由hop/citation/order等过程特征产生整轨迹分数，但当前落在最终token，论文不得写成
逐hop即时dense reward。

预算采用成对递进：第一阶段100/100/100 prompt groups、K=4，即每臂1,200 trajectories，只作为健康与
方向性smoke；通过后至少扩到每臂约4,800 trajectories，正式预算再由未见development曲线冻结。最终
标准legacy pipeline在三数据集n=300报告strong-SFT、mixed PPO-O、mixed PPO-K；2Wiki canonical
ProofKG另作额外资源/机制表。HotpotQA/MuSiQue上的K−O只能解释为共享策略迁移或保持，不能声称它们
直接接受了KG过程监督；直接process机制归因来自eligible 2Wiki行。

## 27. 冻结：三数据集 ReaRAG + ProofKG 配对 PPO（2026-09-03，CPU_PASS_GPU_NOT_STARTED）

本节是对第26节“尚未物化、且不加载ReaRAG”状态的append-only后续更正；不删除第23–26节的失败
实验和决策演变。研究者已批准正式配对方案。起点固定为
`checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`，**不重新训练SFT**，也不采用
失败的SAEG/QPEG continued-SFT checkpoint。

### 27.1 冻结数据与共同训练日程

- mixed population：`data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/`，共1799个唯一问题，
  HotpotQA/2Wiki/MuSiQue=`600/600/599`；每题均有冻结的10篇passages；
- 固定日程为1800个prompt groups，每组`K=4`，即每臂7200条on-policy trajectories；三个数据集各
  600组。日程覆盖1674个不同身份，并按预注册策略重复hard-recovery和MuSiQue题，不能写成1799题
  恰好各遍历一次；
- 208个2Wiki问题具有identity-safe、complete、`gold_access=false`的ProofKG execution；日程中为
  300组/1200 trajectories，可接受ProofKG-v2.1过程奖励；
- 其余1591个问题没有通过门的图，`kg_subgraph=[]`，仍完整接受outcome与文本过程反馈；不得把失败的
  QPEG/SAEG passage edges重新注入；
- 两臂都读取同一份question-KG records、prompt、passages、顺序和K=4结果空间。ProofKG内容在两臂输入
  中相同，只有是否把其过程分数加入reward不同。

### 27.2 两臂奖励与唯一变量

轨迹不满足格式门时，两臂均令`R=-4`，并且不调用ReaRAG、不更新其EMA、不计算ProofKG reward。合法
轨迹对primary answer及冻结aliases分别做canonical normalization，EM和F1各自在别名集合上独立取最大：

`R_out = 4 * (max_alias EM + 0.1 * max_alias F1)`。

当前1799行中163行带多个原始alias，规范化去重后149行仍有多个实质不同alias，均来自MuSiQue；aliases
只用于train-only outcome判分，不进入prompt。共享文本项使用冻结ReaRAG-9B：第`t`步原始分为`s_t`，
用更新前因果EMA基线`b_t`中心化并裁剪，`c_t=clip(s_t-b_t,-1,1)`；若有`n`个合法推理步，则每个
对应step-end放置`0.3*c_t/n`，总文本项等价于`R_text=0.3*mean_t(c_t)`。ReaRAG必须显式成功加载，
任何缺模型、推理异常或非有限值都fail-hard，禁止自动切换旧text head或dummy scorer。

- **PPO-T**：合法轨迹`R = R_out + R_text`；
- **PPO-TK**：合法轨迹`R = R_out + R_text + I_eligible * 0.2 * R_ProofKG-v2.1`。

`R_out`与全局ProofKG-v2.1项放在final generated token；ReaRAG按推理步骤放置。PPO-TK相对PPO-T允许
的有效配置差异只有`training.ppo.proofkg_process_reward: false -> true`以及独立的输出/Experiment ID；
它回答`PPO-TK−PPO-T`的KG过程奖励净效应。旧`alpha_gate.pt`、旧Phase2 PRM/composite reward均从该
mixed路径禁用，不能把它们的历史行为混入新归因。

### 27.3 共享稳定配置与回放边界

两臂共享：lr=`1e-6`、batch=`4`、mini-batch=`1`、PPO epochs=`2`、`kl_coef=.25`、
`target_kl=8`、KL horizon=`2000`、gamma/lambda=`.95/.95`、clip/value-clip=`.2/.2`、
max-grad-norm=`1`、vf-coef=`.5`、critic zero-init/dropout=`0`、max input/new tokens=`6144/384`、
max reasoning steps=`5`，每600 trajectories保存一次。`10%` SFT replay及`.10` anchor weight也完全
共享；该replay来自历史强SFT的HotpotQA train，仅作为防遗忘锚点，不得描述为三数据集均衡replay。
训练健康门从step 200起检查15窗：valid rate≥`.70`、length-capped≤`.20`、mean KL≤`10`；任何
non-finite立即停止。

### 27.4 论文比较、当前门和复现要求

同资源主表继续使用冻结的标准legacy pipeline，报告strong-SFT、PPO-T、PPO-TK在HotpotQA、2Wiki、
MuSiQue相同n/qid/decoding下的EM/F1；`PPO-T−SFT`是outcome+ReaRAG后训练总体效应，
`PPO-TK−PPO-T`才是KG reward因果量。2Wiki canonical ProofKG的EM/F1=`.640/.6944`及matched
legacy→ProofKG `+23.7pp EM`必须放在**额外Wikidata资源增强表**，不得与legacy同资源baseline冒充
严格同条件比较。HotpotQA/MuSiQue没有直接KG过程监督，其变化只能解释为共享策略迁移/保持。

版本化pair lock位于`outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v2/`。CPU preflight状态为
`PASS_NO_GPU_PREFLIGHT`：相关测试通过、1799/1799 alias转发一致、KG identity join=`1.0`、7200日程及
两臂唯一差异均核验通过。此前data report中的`RUNTIME_SCHEDULE_BLOCKED`作为历史状态保留，由本次
后续v2 runtime/preflight记录解除；不回写旧报告。

**尚未启动GPU训练。** 启动前仍必须在96GB远端完成fail-closed remote preflight及不超过8条trajectory
的真实连线探针，验证policy/reference/ReaRAG同时加载、step reward位置、EMA遥测、TensorBoard目录和
显存；探针失败不得进入7200。正式主终点预注册为每臂final-7200，不得根据中间checkpoint事后挑最好。
当前只有seed42：任何重要“超过SFT”或“KG过程奖励有效”结论至少还需相同冻结协议的额外seeds，或在
同qid上报告paired bootstrap CI、McNemar、gained/lost/tied并明确单seed限制；不得仅凭一次有利结果
宣称普适提升。配置为：

- `configs/training/phase3_ppo_mixed3_rearag_v1_text7200_seed42.yaml`（PPO-T）；
- `configs/training/phase3_ppo_mixed3_rearag_v1_text_kg_v2_1_7200_seed42.yaml`（PPO-TK）。

## 28. Proof400 扩容与统一 family 隔离更正（2026-09-03，DATA_CONFIG_PASS）

本节以append-only方式取代第27节的v1训练数据选择，不删除或覆盖v1数据、配置、失败记录。复审发现
v1把不同namespace生成的`family_sha256`直接比较，因此“family overlap=0”不能成立；统一用
`answer-free-lexical-family-v1`重算后，v1实际与受保护评估集重叠51个dataset-scoped families、83行。
该旧family结论已标记`SUPERSEDED_NAMESPACE_INCOMPARABLE`，其余v1结果不变。

新v2仍为1799个唯一训练问题：HotpotQA/2Wiki/MuSiQue=`600/600/599`。2Wiki改为200个普通空图
outcome问题加400个完整自动ProofKG问题；400中保留125个通过新隔离门的hard问题，再从冻结的
n1500完整ProofKG池补275个。四种问题类型严格各100个（inference/comparison/compositional/
bridge-comparison），400条均具有identity-safe execution、`complete=true`和`gold_access=false`。
其余1399条显式空图，只接收outcome+ReaRAG文本反馈。

隔离门按`(dataset, lexical-family-v1 hash)`判断，受保护A类固定为canonical main每数据集300题
（共900）加尚未打开的confirmation每数据集100题（共300）；v2 population与A类的qid和
dataset-scoped family overlap均为0。跨数据集相同问题模板不算泄漏，但单独记录了2个模板、6个
训练行的raw-family碰撞。train-side已消费rankability/verifier题允许进入训练；一旦使用，其旧结果
只能标为development/consumed，不能再充当独立confirmation。

总计算预算不变：三个数据集仍各600 prompt groups、每组K=4，共7200 trajectories；1799个唯一身份
全部至少出现一次，仅MuSiQue确定性重复1题。ProofKG覆盖从v1的208 unique提升到400 unique，固定日程
中的eligible暴露从1200提升到1600 trajectories。数据和两臂配置已通过CPU解析/契约检查，GPU连线
探针及正式训练均未启动。版本化资产为：

- protocol：`outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/`；
- materialized data：`data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/`；
- family/code lock：`outputs/audits/mixed_ppo_three_dataset_v2_proof400_family_scope_addendum_v2/`；
- PPO-T：`configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml`；
- PPO-TK：`configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml`。

当前状态为`DATA_CONFIG_PASS / GPU_PROBE_NOT_STARTED / TRAINING_NOT_STARTED`。后续必须先为v2重新绑定
正式pair lock并运行最多8 trajectories的GPU runtime wiring probe；探针通过后，才允许启动两臂7200。

## 29. Proof400 GPU 连线探针 v3（2026-09-03，CPU_PASS_GPU_NOT_RUN）

已为第28节Proof400数据/配置新建完全独立的v3探针，旧v1/v2探针资产均保留。探针只取两个问题：
PPO-T使用1个HotpotQA空图、process-ineligible问题，PPO-TK使用1个2Wiki完整ProofKG、
process-eligible问题；每题K=4，总预算严格为8 trajectories。它只验证运行时连线，不能估计训练效果。

CPU freeze/preflight已通过：question-KG identity join=`1.0`，T为0 triples，TK为2 triples且完整
execution；真实CLI相对各自正式臂只改变微型数据/输出/`total_steps=4`/`save_every=4`，90个runtime
代码文件和10个配置依赖均已hash绑定，内嵌回归测试通过。为支持fail-closed检查，只新增了遥测字段
（ReaRAG更新前EMA均值、未裁剪centered均值、process-applied计数），未改变reward数值或优化逻辑。

GPU postflight会硬性要求：显式SFT reference为真；首批KL有限且绝对值≤1；至少一个合法ReaRAG步骤，
raw/pre-update EMA/unclipped-centered/clipped及n_obs均有效；T的process严格为0；TK至少一个合法
`process_applied`且加权process非零；PPO/critic统计有限，并写出K4 history、adapter、manifest、日志和
TensorBoard event。任一门失败即停止，不进入正式7200。

资产位置：

- freeze：`outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_freeze/`；
- micro data：`data/silver_data/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42/`；
- CPU preflight：`outputs/audits/mixed3_rearag_runtime_wiring_probe_v3_proof400_seed42_local_preflight/`；
- launcher：`launch_ppo_mixed3_rearag_runtime_probe_v3_proof400_remote.sh`。

当前未分配CUDA、未生成checkpoint、未启动探针或正式训练。

## 30. 强SFT探索空间门：Proof400新增fill275的n100零更新审计（2026-09-03，FROZEN_NOT_RUN）

为避免在强SFT已接近上限、K=4候选没有可学习差异时直接花费PPO训练成本，正式训练前新增一个只测
候选空间的train-side development门。cohort只从Proof400新增的275条automatic ProofKG中选择，明确
排除125条safe-hard；按inference/comparison/compositional/bridge-comparison各25条确定性选取，共
100题、100个不同lexical families。它们从冻结时起标为`development/consumed`，不得再作为独立确认集。

模型固定为现有强SFT `sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final`，每题沿用mixed-v2中
原封不动的10篇passages和完整ProofKG。prompt输入与train-only outcome labels分文件冻结；Gold只在生成
完成后计算EM/F1，不进入prompt。每题生成1个greedy与4个sampled候选，参数与正式PPO rollout一致：
max-new-tokens=384、temperature=1、top-p=1、top-k=0、seed=42；总计500次推理，不建reward/critic，
optimizer updates=0。

预注册三门必须同时满足：sample valid rate≥0.90、sampled oracle@4 EM−greedy EM≥0.05、K4内部同时
出现正确/错误结果的qid比例≥0.20。若失败，只说明本cohort的可观测学习空间不足；允许另建版本化
cohort/配额方案，但不得改本轮门、在本轮内挑题，或宣称PPO/reward无效。当前仅完成CPU冻结和13项相关
测试，未运行GPU推理。首版v1因运行器未主动复核已记录的模型文件hash而在运行前被append-only v2取代，
v1文件未覆盖且无科学结果。有效资产为：

- protocol/input：`outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v2_preregistration/`；
- runner：`scripts/pilot/audit_proof400_fill275_sft_headroom.py`；
- local launcher：`launch_proof400_fill275_sft_headroom_n100_local.sh`。

## 31. 强SFT探索空间门v3修复冻结（2026-09-03，FROZEN_NOT_RUN）

第30节v2实际本地启动后在生成前因运行器错误读取`generation.temperature`而失败；冻结协议中的字段实际
位于`generation.sampled.temperature/top_p`。失败目录与日志完整保留，manifest状态为`FAILED_RUNTIME`：
模型权重已加载，但候选生成0条、optimizer updates=0，因此没有可解释的科学结果，不能据此判断强SFT
学习空间或PPO是否有效。

append-only v3保持相同n100、四类25×4、同一passages/ProofKG、强SFT和三道门，只修复采样参数读取，
并强化运行前锁：主动复核Llama `config/generation_config/index`及4个实际safetensors分片、base tokenizer，
强SFT父manifest/loss与final目录全部关键文件，共19个模型资产；代码闭包补入question-KG、ProofKG判定和
logging等直接依赖（共10项）；同时严格匹配Python、PyTorch/Transformers/PEFT/Accelerate/NumPy/
Safetensors/TRL及PyTorch CUDA build。CPU侧15项相关测试通过，包含全部模型大文件hash复核。

当前唯一有效待运行版本为
`outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration/`；launcher已改指v3
的独立protocol/output/log，并在启动前要求目标均不存在。当前未再次启动GPU推理或任何训练。

## 32. 子问题依赖检索可行性 pilot（2026-09-03，FAIL_STOP_DEVELOPMENT_FEASIBILITY）

本轮只回答一个开发问题：在固定强SFT、legacy KG、prompt、10篇passage预算及greedy/512解码时，
把原问题一次性检索替换为“question-only冻结QueryPlan -> 根跳检索 -> 从第一跳候选确定性抽取bridge ->
实例化依赖查询 -> 再检索 -> 6篇原passage加每逻辑hop最多2篇、总数仍为10”的B臂，是否优于冻结的
完整问题canonical top10 A臂。样本为已消费的HotpotQA30和MuSiQue30；未修改canonical pipeline、
checkpoint、reward、训练数据或PPO，因此本轮是零训练、单变量的development feasibility test。

Gold隔离严格执行：检索/物化进程不读取answer、support、decomposition或target；A/B passage先冻结并
hash，随后由独立CPU finalizer从raw dev附加答案，仅用于评分。当前规范化dev没有supporting-fact标签，
所以support recall为`NOT_AVAILABLE`；gold字符串是否出现在passage中只作为诊断，不能写成证据召回率。
本地Wiki18 corpus/dense/BM25均为21,015,324篇，ID边界为0至21,015,323；真实输入dry-run中两数据集
plan均30/30可执行，57项相关测试通过。

版本演进均append-only保留：v1因dependent-hop分母错误、v2因修正后代码hash变化，均在检索前
supersede且无结果；v3实际启动后发现逐query扫描21M memmap不可行，在首个query、0行物化时中止，
状态为`ABORTED_RUNTIME_INEFFICIENT_NO_RESULT`，不是科学负结果。v4只将同一query、顺序、bridge与
merge语义改成按依赖层批量检索/重排。有效预注册为
`outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v4_preregistration/protocol.json`，
runner SHA256为`6bbd50c6c24e6a6bcc923f9f12dc26cae0ac0b89a2deeb7c4f3e9c61eaae54a7`。

v4检索物化结构门全部通过：HotpotQA/MuSiQue均30/30执行、0 runtime error、0 fallback；声明依赖的
eligible题分别为24/30和30/30，依赖查询非空率均1.0；60/60题得到新的plan-query候选且最终passage
集合发生变化。强SFT A/B评分结果为：

- pooled：EM `.3167 -> .3500`（`+3.33pp`），F1 `.4610 -> .4716`（`+1.06pp`），gained/lost/tied
  `3/1/56`，净增2题，EM 95% paired-bootstrap CI `[-3.33,+10.0]pp`，McNemar `p=.625`；
- HotpotQA：EM `.4000 -> .4333`，F1 `.5686 -> .5688`，gained/lost=`2/1`；
- MuSiQue：EM `.2333 -> .2667`，F1 `.3533 -> .3744`，gained/lost=`1/0`；但B臂一题在512 tokens内
  重复推理并缺失Final Answer，parse从30/30降至29/30；
- gold字符串可见率仅作诊断，从`.4833`升至`.5333`；该值不是support recall。

预注册效用门未通过：pooled净增2题小于门槛3题，且parse count delta为-1；置信区间包含0。因此正式
状态为`FAIL_STOP_DEVELOPMENT_FEASIBILITY`。这不是“拆解无效”：工程机制门全部通过，两个数据集的
EM方向均为+3.3pp，并有明确救回样例；但证据不足以打开fresh confirmation或宣称稳定提升。逐题审计
显示主要剩余瓶颈是bridge语义/类型精度及固定10篇预算下误删有用旧passage：例如MuSiQue `dev_198`
正确形成“Permission to Fly co-writer -> Jordan Pruitt record label”并答出Hollywood Records；而
HotpotQA `dev_205`把人物Nicki Minaj当成应查询host的中间实体，替换掉有用节目passage后答错。

有效结果位于：

- Gold-free物化：`outputs/audits/plan_once_dependent_retrieval_pilot30x2_v4/`；
- Gold附加后冻结协议：`outputs/audits/plan_once_dependent_retrieval_pilot30x2_freeze_v4/`；
- 配对结果：`outputs/validation/plan_once_dependent_retrieval_pilot30x2_eval_v4/report.json`；
- 完整日志：`logs/eval/subquestion_dependent_retrieval_pilot30x2_v4_materialize.log`与
  `logs/eval/subquestion_dependent_retrieval_pilot30x2_v4_eval.log`。

科学边界：本轮测的是planner、bridge抽取、依赖检索和固定预算merge的整体，不单独证明“拆解模块”
因果有效，也未隔离额外检索调用次数。当前不据此修改SFT/PPO/reward或打开confirmation。若继续该线，
应先在已消费development题上另冻版本，加入bridge类型约束/重排并解决错误passage置换；只有重新通过
效用门，才在fresh family/QID-disjoint confirmation中加入budget-matched generic bridge臂，以区分
结构化子问题拆解收益与单纯增加检索预算的收益。

## 33. 子问题依赖检索 v5 精度优先复测（2026-09-04，FAIL_STOP_GOLD_FREE_MECHANISM_GATE）

v5 是在已消费的同一 HotpotQA30/MuSiQue30 development cohort 上进行的 append-only 组合改进：加入
类型/关系约束的 bridge admission，并将最终合并改为保护 Arm-A 前8篇、只允许完整问题 cross-encoder
严格胜过 A 尾部文档的新依赖文档替换。该轮不是单组件消融，不能分别归因 bridge selector 与 merge；
强SFT、legacy KG、prompt、10篇 passage 预算、greedy/512 解码及 v4 scorer 均保持冻结。

正式预注册为
`outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v5_preregistration/protocol.json`，
SHA256=`a6177803ded2e59a366386e2f6b3a4161a0e4ecc903c9313e46f6b772fb92f22`。本地完整 Wiki18
corpus/dense/BM25 均为 21,015,324 篇；正式 Gold-free 物化在
`outputs/audits/plan_once_dependent_retrieval_pilot30x2_v5/`，status 为
`COMPLETE_INPUTS_NOT_ANSWER_EVALUATED`。

工程安全门全部通过：两数据集 plan executable 均为 30/30，runtime error=0，输出均为 top10，A前8篇
保持不变，unauthorized displacement=0，root passage injection=0，所有 fallback 与A逐题逐字节一致。
但预注册机制门失败：

- HotpotQA：24题存在依赖step，只有11/24形成非空依赖查询（`.4583 < .80`），最终仅3/30保留新文档
  （`.10 < .50`）；状态为6题无依赖、13题bridge abstain、8题新候选未严格胜过A尾部、3题改变；
- MuSiQue：30题存在依赖step，22/30形成非空依赖查询（`.7333 < .80`），最终8/30保留新文档
  （`.2667 < .50`）；状态为8题bridge abstain、14题候选未严格胜过A尾部、8题改变。

finalizer 已按协议在加载 Gold 前 fail-fast，故本轮没有答案生成，也没有 EM/F1；不得把本轮描述为
“模型效果下降/提升”。Gold-free 失败分解冻结于
`outputs/audits/plan_once_dependent_retrieval_pilot30x2_v5_failure_decomposition/`：Hotpot/MuSiQue 分别有
13/8题卡在 bridge admission；另有8/14题虽完成依赖检索，但所有新候选均被完整问题 CE 拒绝。被拒候选
相对待替换A文档的平均分差分别为`-0.2230/-0.1412`，说明安全回退在按设计工作；与此同时 v5 对
bridge 的硬筛选和单一实体替换过于保守，覆盖不足。

后续不得降低 v5 冻结门或打开 confirmation。若继续，另行预注册 v6 development combination：采用
“原始完整问题锚定 + 冻结子问题/关系 + 最多两个候选bridge”的多查询扩展，不再让单个可能错误的bridge
替换完整问题语义；保留A前8篇、相同CE严格替换和10篇预算。v6仍须先过同样的Gold-free安全/覆盖门，
再运行既定效用门；通过后才允许一次fresh confirmation，并加入检索调用数匹配的generic expansion臂。

## 34. 子问题依赖检索 v6 问题锚定多查询复测（2026-09-04，FAIL_STOP_DEVELOPMENT_FEASIBILITY）

v6 仍使用已消费的同一 HotpotQA30/MuSiQue30 development cohort，是一次 append-only 组合实验，
同时改变三项：bridge 由“证据断言”降为 query hint、每个依赖跳最多生成两个候选查询、每个查询都用
`原始完整问题 + 换行 + 冻结子问题/关系`锚定。每个 query variant 独立保留 top2 候选，最终仍保护
Arm-A 前8篇、总输入10篇、最多替换2篇，并要求新文档在同一完整问题 cross-encoder 下严格胜过
可替换的 A 尾部文档；平分保留 A。该轮不能分别归因上述三个组件。

正式设计先冻结在
`outputs/audits/subquestion_dependent_retrieval_v6_design_freeze/`；正式 preregistration 为
`outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v6_preregistration/protocol.json`，
SHA256=`182a812d06c5014a7a0e7d7546f8dd9eaef845e5d4020830fe612dafb955a69b`。协议逐字节锁定
same60 输入、21,015,324篇 Wiki18 corpus/dense/BM25、E5、cross-encoder、强SFT、base model及代码。
相关 v5/v6 回归测试通过，真实输入 dry-run 60/60 通过；检索阶段不读取 Gold。

Gold-free 正式物化位于
`outputs/audits/plan_once_dependent_retrieval_pilot30x2_v6/`，工程与机制门全部通过：

- HotpotQA：plan executable=`30/30`，依赖题 query nonempty=`24/24`，换入新依赖文档=`15/30=.50`；
- MuSiQue：plan executable=`30/30`，依赖题 query nonempty=`30/30`，换入新依赖文档=`18/30=.60`；
- 两数据集均 runtime error=`0`、duplicate query/document=`0`、root-only injection=`0`、越权替换=`0`；
  A前8篇、top10、问题前缀、最终完整问题CE和27个fallback逐题一致性全部通过。

只有在上述门通过后，finalizer 才从冻结 raw dev 附加 scorer Gold；强SFT paired 结果位于
`outputs/validation/plan_once_dependent_retrieval_pilot30x2_eval_v6/report.json`：

- pooled：EM `.3167 -> .3000`（`-1.67pp`），F1 `.4610 -> .4215`（`-3.95pp`），
  gained/lost/tied=`1/2/57`，EM 95% paired-bootstrap CI=`[-8.33,+3.33]pp`；
- HotpotQA：EM `.4000 -> .3667`，F1 `.5686 -> .4697`，gained/lost=`0/1`；
- MuSiQue：EM `.2333 -> .2333`，F1 `.3533 -> .3733`，gained/lost=`1/1`；
- 两臂 parse rate 均为`1.0`，27个fallback的prompt与prediction逐字节一致。

因此效用门未通过，正式状态为`FAIL_STOP_DEVELOPMENT_FEASIBILITY`；不得打开 fresh confirmation，
不得据此修改SFT/PPO训练数据，也不得把机制覆盖提升写成答案效果提升。v6把问题进一步定位为：
“查询能生成、候选能召回、受控替换能执行”，但当前完整问题相关性分数不足以判断新文档是否真正支持
所需推理跳；HotpotQA是主要负向来源，MuSiQue只有F1方向性改善且样本不足。

本地多步检索文献核验见`docs/multistep_retrieval_reference_notes.md`。CoRAG支持后续查询同时依赖原问题、
历史子问题和子答案；ReaRAG/Search-R1支持observation-conditioned状态；R1-Searcher支持近似单三元组
的原子查询；Self-RAG支持在接纳证据前评估相关性/支持性。它们只构成机制启发：其中Gold过滤、自由
Search/Finish、RL、learned critic或sub-answer生成均未进入v6，不能声称v6复现这些完整方法。

下一步先完成已解锁结果的只读逐题诊断，区分错误bridge、文档支持性不足与固定预算误替换；不继续对
same60手调阈值。若仍推进多步检索，应另冻新方法变量（例如明确生成/抽取中间sub-answer，并使用
support-sensitive而非仅question-relevance的证据门）及新的开发证据，避免在同一60题上反复试版本。

只读事后诊断已落盘于
`outputs/audits/plan_once_dependent_retrieval_pilot30x2_v6_outcome_diagnosis/`，状态为
`COMPLETE_DEVELOPMENT_ONLY_EXPLORATORY_DIAGNOSIS`。实际替换的33题 EM `.242 -> .212`、F1
`.383 -> .311`；27题fallback逐题一致。47篇替入文档中44篇命中子查询有效词、33篇含query hint，
但只有8篇含Gold答案字面，最终top10的Gold字面可见题反从29降至28。这只能作为“词面相关不等于
决定性多跳支持”的机械证据，不能称support recall：当前落盘数据没有Hotpot supporting facts或
MuSiQue decomposition。两个lost问题的平均CE margin为`.813`，唯一gain为`.326`，也直接否定
“完整问题CE margin越高就必然越支持答案”的假设。代表性错误是Hotpot `dev_211`：新文档强调
中间演员Armie Hammer，模型从正确的Jackson Storm转而回答中间实体；唯一gain是MuSiQue `dev_261`，
新La Liga文档补出了最终数值38。该诊断不构成v7调参依据，v6正式失败状态不变。

## 35. 子问题“先回答、再检索”v7可行性测试（2026-09-04，FAIL_STOP_BEFORE_GOLD）

v7针对v6“子查询词面相关但不一定支持目标hop”的问题，测试显式回答中间子问题是否比只抽一个实体
更适合作为下一跳检索条件。该轮没有改SFT、PPO、reward、KG、alpha门控或canonical评估；使用全局已消费的
HotpotQA20/MuSiQue20 development题，仅检验检索机制：

- A：冻结的canonical一次检索Top-10，无KG；
- B：从当前producer passages中确定性抽取top-1实体，再执行依赖检索；
- C：同一强SFT读取原问题、当前plan step和同一producer passages，输出严格JSON子答案；只有通过
  passage-local机械验证的`entity/number/date`答案才能触发依赖检索；
- B/C按每题、每depth、每logical hop匹配检索预算。根query相同时producer passages逐字节相同；上游
  bridge不同后，下游arm-specific passages属于递归策略的因果中介，因此主比较解释为C相对B的完整递归
  策略效应，而不是“每一层只改bridge、其余输入永远相同”。

正式预注册位于
`outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/protocol.json`；递归估计量的
pre-execution append-only澄清位于
`outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration_addendum_recursive_trajectory_v1/`。
实现主要落在`kgproweight/retrieval/subanswer_v7.py`、
`scripts/pilot/generate_grounded_subanswers_v7.py`、
`scripts/pilot/materialize_paired_dependent_retrieval_v7.py`、
`scripts/prepare/finalize_paired_dependent_retrieval_v7.py`和
`scripts/eval/evaluate_paired_dependent_retrieval_v7.py`。另补模型全树hash、父协议链、stage descriptor链、
Gold递归禁字段、scorer闭包、B/C预算及fallback一致性等fail-closed检查。

retry1正式链中planner 40/40 schema-valid且可执行；root物化产生41个depth-1 C reader任务。强SFT严格JSON
parse为HotpotQA `12/19=.632`、MuSiQue `18/22=.818`，但机械验证仅分别为`3/19=.158`和
`3/22=.136`。41条中24条因答案被模型标为`answer_type=other`而被拒（其中可见多个实体型短语），10条因把`abstain`
输出成非布尔值被拒，1条citation数量不符，仅6条通过。这里的“通过”只表示答案字面可在唯一引用passage
中定位，不代表语义蕴含或最终答案正确。

depth-1后发现runner对无依赖plan的终止状态标记错误；依赖检索计算已发生，但新stage在写盘前fail-closed，
因此该未完成stage不进入科学计数。当前代码已修复该状态机问题和同depth sibling重复query的配对skip，
并补回归测试；由于代码hash改变，不能用修后代码续跑retry1旧锁链。

为避免无意义地重锁并继续GPU/检索，新增CPU-only、Gold-free单调上界审计：
`outputs/audits/subquestion_dependent_retrieval_v7_depth1_monotonic_upper_bound_retry1/report.json`。在“以后所有
可达检索、实体抽取和子答案验证都成功”的最乐观假设下，按冻结runner实际只对有下游consumer的producer
建立reader任务的口径，最终机械验证率上界仍只有：

- HotpotQA：`3/19=.1579 < .40`；
- MuSiQue：`4/23=.1739 < .40`。

所以一个必要Gold-free机制门已被数学上确定为不可达，正式状态为
`FAIL_STOP_BEFORE_GOLD_MONOTONIC_UPPER_BOUND`。本轮未打开Gold、未运行最终答案生成、无EM/F1/IHR，
也不能据此断言“回答子问题这一思想无效”；它只否定了“当前强SFT + 当前严格JSON/type契约 +
锁定单次生成、且已观测`retry_count=0`”的具体实现。

若继续，先另冻v8单变量schema适配：不再让模型自报`answer_type`决定接纳，模型只返回answer、citation和
严格布尔abstain，由机械程序从answer surface确定性推断entity/number/date；可先对已冻结的41条raw response
做Gold-free反事实parser审计。只有HotpotQA和MuSiQue各自的潜在验证率均达到`.40`，才允许在新的
append-only v8链上从头物化；否则停止该线。不得把v7 development题或其诊断当fresh confirmation，也不据
本轮修改训练或论文主表。

本地原文复核还表明，IRCoT并非预先冻结完整子问题树，而是每轮基于原问题、累计passages和已有CoT只生成
下一句reasoning，并用该句检索；Decomposed Prompting则用独立decomposer和single-hop handler，将上一答案
及其documents显式传给后续子问题。若v8接口反事实仍过不了门，不应继续给固定PID plan打补丁；应另立
reference-aligned动态控制协议：`root retrieve -> evidence-conditioned thought/subquestion -> retrieve ->
accumulate -> independent final reader`。公平比较固定corpus、模型和query/passage/token预算，A为canonical
one-shot，B为调用次数匹配的generic iterative expansion，C为动态拆解；`C-B`隔离动态拆解价值，`C-A`
只报告系统收益。已有trace/IRCoT baseline意味着多轮检索本身不能作为本文创新；若以后进入训练，动态
controller必须作为所有SFT/PPO消融的共同backbone，再比较outcome-only PPO与alpha-gated KG reward。

## 36. 动态问题拆解 v8（2026-09-04，DEVELOPMENT90_FAIL_STOP_BEFORE_GOLD）

详细方案已写入`docs/subquestion_decomposition_v8_test_plan.md`。研究者只批准了 Phase 0 与随后容量审计；
新 A/B/C 检索、Gold 评估、SFT/PPO 和 reward 修改仍未获授权。计划采用三级漏斗：

1. 已对v7冻结41条raw response完成append-only、CPU-only、Gold-free单变量审计：只忽略模型自报
   `answer_type`，其余严格JSON、真布尔abstain、唯一citation、subject/locality规则全部保持；正式结果为
   Hotpot `11/19=.579`、MuSiQue `17/22=.773`，且明确只作接口诊断，不当作语义或答案质量结果；
2. 再冻结A/B/C零训练机制：A=canonical one-shot；B=不读取observation/中间答案的plan-once子问题；
   C=将验证后的single-hop answer和observation写回状态后动态生成q2。首轮只测一个依赖转换，不加入q3；
   A/B/C共享root，B/C共享q1，B/C每题固定3次实际检索（root/q1/q2）、2次controller、1次q1 reader；
   q2失败使用预冻结静态/root fallback，调用数仍匹配；final context唯一采用root6+q1-2+q2-2共10篇；
3. Gold-free动作/预算/验证门通过后，才按L0格式、L1字面、L2语义角色、L3完整support chain、L4最终
   EM/F1的漏斗评估；development只做可行性，prospective-validation300×3与未打开的论文确认reserve须
   预先封存且family/QID-disjoint。

历史FlashRAG trace不作为matched control：实查其n=300最终累计文档长度均值为Hotpot/2Wiki/MuSiQue
`26.19/28.74/28.64`，并非固定top-10或IRCoT论文15篇cap；相关append-only更正已写入
`docs/baselines_final.md`和`docs/multistep_retrieval_reference_notes.md`。若零训练动态机制成立，后续才将
controller作为共同backbone，比较controller-SFT、PPO outcome-only与PPO+alpha-gated KG process reward；
多轮检索本身已有baseline，不能作为本文创新归因。

Phase 0 已按 append-only 协议正式完成。Experiment ID 为
`SUBQUESTION-DECOMPOSITION-V8-PHASE0-V7-CONTRACT-COUNTERFACTUAL-DEV41-SEED20260904-V1`，协议 SHA256 为
`7ccc7cf0aa3b4aa66fa04dd4d88d98e1b4ea32b248762cf6fda67f586677f45d`。72 项相关回归测试通过，运行中
0 GPU、0 model、0 retrieval、`gold_access=false`。P0 完整复现后，P1 只把 verifier admission 的类型改为
由答案表面按 date→number→entity-like 确定性推断：HotpotQA `3/19 -> 11/19`，MuSiQue
`3/22 -> 17/22`，两者均越过冻结 `.40` 机械门；P2 乐观空字符串 abstain 诊断为 `15/19` 与 `20/22`。
正式状态为`PASS_P1_TYPE_ONLY_INTERFACE_DIAGNOSIS`。它只证明 v7 存在明显接口误拒，不证明中间答案
语义正确或最终 EM/F1 提升，也不单独授权 v8。v8 reader 因而保留短答案/sentinel、由系统定位 provenance，
删除模型自报类型这一控制字段；P2 coercion 不进入正式实现。

独立复核确认41条数值与Gold-free边界可信，但发现两项未来运行加固要求：当前聚合P0计数门在P1/P2计算
之后执行；部分间接祖先及`kgproweight/kg/question_kg.py`未被Phase 0协议直接重验hash。现有结果不覆盖、
不重写；缺陷已记录于
`outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_counterfactual_v1_review_addendum/metadata_addendum.json`。
未来只能以新协议/新Experiment ID修复复跑，当前不得宣称完整闭包已认证。

随后完成正式capacity-only审计：
`outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1/`，状态为
`SCOPE_A_ONLY_PASS_DEV30_PROSPECTIVE300_FAIL_BALANCED_RESERVE1000`。审计锁定58个历史evaluation/protocol
registry、20个本地训练/保守排除输入和12个证据文件；全部本地训练输入均可按精确question hash回接对应
raw-train，outside qid/family=`0/0`。严格Scope A的可冻结family为Hotpot `6213`、2Wiki `331`、MuSiQue
`1644`，因此三数据集各`30+300`可行，但2Wiki只余1个family，统一`reserve1000×3`不可行。Scope B因
完整历史training ledger不可得而标为`INVALID_FOR_FREEZE`，不能用于凑样本。

该审计打开了可能含Gold的源文件，但只使用identity投影且不输出个体身份或Gold；它不是Gold-free selector，
也没有冻结任何题。当前停止等待研究者决定是否采用严格Scope A仅冻结`30+300`。若批准，还需单独实现
custodian export；若坚持reserve1000，则需新增独立2Wiki来源或事先改变隔离/样本协议。正式L3评分所需
Hotpot supporting facts与MuSiQue decomposition来源仍为`UNKNOWN`，在Gold阶段前必须解决。

研究者随后批准严格Scope A，并取消本轮`reserve1000`。identity-only custodian export已一次性冻结：

- development=90（Hotpot/2Wiki/MuSiQue各30），SHA256=`dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2`；
- prospective=900（各300），SHA256=`36b680cabef059dae7370bb131b1bafc0f120baf372f4e7666aa0e2d13b13c99`；
- 每行只含`dataset/qid/question`，与history/training/raw-train及两split间的qid/family overlap均为0；
- development-only loader绑定manifest/dev hash，prospective role/path在打开前即拒绝，无unlock开关。

正式冻结位于
`outputs/audits/subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_seed20260904_v1/`。这是
procedural seal，不是加密物理保管；后续runner必须只走锁定loader。

同时完成v8 Gold-free纯函数核心：一行query、短subanswer/`NO_RELEVANT_ANSWER`、系统自动绑定唯一
passage provenance、static/dynamic q2 action选择、稳定去重及`root6+q1 novel2+q2 novel2`固定10篇合并。
Unicode casefold offset、数值标点/复合数边界、伪造provenance和identity/content冲突均fail-closed；新功能及相关历史
回归由主代理重跑全绿。本阶段没有GPU、模型、检索、Gold scoring或训练。

历史Trace/IRCoT-style n300×3预算审计也已append-only完成：最终累计文档数均值为
`26.19/28.7433/28.6367`，不是fixed-10 matched control；每调用top-k、物理calls、runtime code、cache/retry和
wall time无法由历史产物恢复，全部记为`UNKNOWN`。

原方案的“B第2次call产生static、C第2次call产生dynamic、C所有失败都回退B static、每臂只2 calls”
存在预算矛盾。研究者已在2026-09-04批准两调用澄清：B slot2只看
`Q+q1+NO_VERIFIED_SUBANSWER`；C在a1无效时使用同prompt，有效时才看`a1+provenance`；dynamic response
格式无效时C确定性回退Q且不产生第3次call。byte-identity门仅限a1-ineligible子集，dynamic-invalid按ITT
单列。正式runner、逻辑/物理call ledger、append-only driver与首版implementation freeze已完成，相关
回归测试全绿。真实engineering smoke attempt001启动后确认模型、RRF100/BGE10和Wiki18索引均可加载，
但逐query dense全库扫描实测67--83秒；当前串行实现最多48次full scan，会使后续90题耗时不可接受。
该尝试在0行科学输出时停止，失败manifest已append-only保留。下一步只将相同查询按root/q1/q2三个依赖层
批量执行，保持84次logical request、候选/排序/预算与A/B/C定义不变；新代码必须新锁并以attempt002重跑。
batch smoke通过前不启动development90，不打开Gold或sealed prospective。

等价batch实现随后以implementation V2重新逐字节锁定并通过133项相关测试。真实smoke attempt002已完成：
12题三数据集各4题、84/84 logical retrieval、3次full-index passes（12/10/10 unique queries）、同一强SFT
物理模型对象、固定10篇无重复、cache/ledger/fallback/Gold-free边界共16门全部PASS，状态`COMPLETE`。
该工程结果仅解锁冻结development90的Gold-free物化；不包含Gold、EM/F1/IHR或效用结论。

冻结development90现已完整运行并保留7.1MB逐题输出，但正式Gold-free门失败：q1有效率
Hotpot/2Wiki/MuSiQue=`1.000/.933/.967`，a1 admissible=`0/.133/.067`，B q2有效率=`.678`，C dynamic
transition ITT=`0/.033/.033`。失败分解为a1跨文档surface歧义42题、surface未找到29题、q1回声6题、
sentinel 7题，仅6题通过；B q2另有29题重复原问题或q1。故不打开Gold、不计算EM/F1/IHR、不读取
sealed prospective，也不在同一90题上改prompt/门后重测。结论边界是当前强final-answer SFT无法零训练
稳定承担controller/subanswer职责；不外推为专门controller-SFT或动态拆解思想无效。

append-only诊断位于
`outputs/audits/subquestion_decomposition_v8_development90_gold_free_failure_diagnosis_v1/`。其中同时更正性能
遥测：正式字段`full_index_passes=3`实际代表顶层batch调用；底层query chunk=64使90/87/62三层实际
memmap遍历为5次。该命名错误不影响输出、机制失败或停止判定。

## 37. canonical 子问题回答 v9 Phase 0（2026-09-04，FAIL_STOP）

研究者批准了“仅在评估时做子问题拆解，并让现有checkpoint用其熟悉的QA格式回答每个子问题”的低成本
可行性测试。为避免一开始再次扫描Wiki18或消耗新题，本阶段只复用已消费的v8 development90中的固定
`q1 query + reranked top-10 passages`，不重新检索、不读Gold、不打开或hash sealed prospective900。唯一
变量是把v8的一行短答案prompt替换为正式SFT/eval共用的`build_inference_messages()`，输出预算固定512，
并由权威`extract_final_answer()`提取`[Final Answer]`；缺标记时fail-closed。原v8唯一文档surface binder、
同一强SFT checkpoint、greedy decoding和全部q1输入保持不变。

协议与结果分别位于：

- `outputs/audits/subquestion_decomposition_v9_canonical_subqa_phase0_protocol_v1/`，protocol SHA256=
  `a22500e7325ea77323063870cc658bc8f4acc809f0bddd604f7dda3d2ebd304c`；
- `outputs/audits/subquestion_decomposition_v9_canonical_subqa_phase0_consumed_dev90_seed42_v1/`，report SHA256=
  `8dc02e7f4ace2559f85fc4b541b8b05d2e51b692b106c46a8b47017df5463495`；
- 相关v8/v9测试共90项通过，正式运行90次模型生成、0 retrieval、0 Gold scoring。

结果表明接口格式问题被完全消除：HotpotQA/2Wiki/MuSiQue的Final Answer解析率与step-trace率均为
`1.000/1.000/1.000`。但未改的唯一文档binder只接纳`2/30、5/30、5/30`，低于预注册每数据集`.40`门，
所以正式状态为`FAIL_STOP_CANONICAL_SUBANSWER_INTERFACE`，不得消耗fresh pilot90或声称EM/F1收益。
失败分解为：跨文档答案surface歧义55/90、surface未找到11/90、boolean/null/echo共12/90、唯一文档
绑定成功12/90。相对v8旧接纳数6/90仅增至12/90。

该结果只否定“canonical answer + 唯一文档surface绑定”这个具体零训练实现，不说明子答案本身错误：
`unique success + ambiguous multi-doc surface = 67/90`个答案至少出现在一篇q1文档中，是一个Gold-free
诊断上界，不是语义正确率。若继续，必须由研究者另行批准v9.1并在运行前冻结：将binder单变量改为
“至少一篇文档命中后按最高rerank rank确定性绑定”，同时明确它只证明lexical locality、不冒充semantic
support；然后只在全新的train-side family-disjoint pilot30×3上检验完整链。不得在本次已消费90题上
改门重判；q2 controller仍是独立的OOD瓶颈，也不能因q1格式通过而宣称已解决。

## 38. rank-first 子答案绑定 v9.1（2026-09-04，FAIL_STOP）

研究者批准后，v9.1 将唯一实现变量限定为多文档答案词面的绑定规则：只要canonical子答案至少在一篇
q1 top-10文档中命中，就确定性绑定到检索排名最靠前的命中文档。该规则只认证lexical locality，不认证
semantic entailment。正式协议、runner/binder、fresh identity-only pilot30×3和append-only driver均在运行前
冻结；相关v8/v9/v9.1回归测试共118项通过。真实engineering smoke n=4×3通过全部工程门后，才启动新的
family-disjoint pilot90；整个过程未读取Gold，也未打开或hash sealed prospective900。

正式pilot90结果为`FAIL_STOP_GOLD_FREE_GATES`：canonical answer parse与step trace均为1.0；rank-first
a1 admissible在Hotpot/2Wiki/MuSiQue分别为`.767/.600/.833`，三者均通过`.40`门；C dynamic transition
ITT分别为`.600/.533/.633`，也均通过`.32`门。由此可确认binder单点瓶颈已机械解除。完整链仍失败于两个
预注册门：q1 schema valid为`.900/.933/1.000`，Hotpot与2Wiki低于每数据集`.95`；B observation-blind
static q2 pooled valid仅`.633`，低于`.90`。逐题Gold-free诊断确认33个无效B-q2全部是重复原问题或q1。

终态产物为
`outputs/audits/subquestion_decomposition_v9_1_rank_first_fresh_pilot30x3_seed42_attempt001/`；report SHA256=
`fb04defc2fe4c8363c7018daa56caee3aca6d2c29e1a31b6b5e751d57de45a23`，rows SHA256=
`0c716c1d1db9a2109e17f8c39d8ad3fadf1f6d121b99101349a714ee63fa8b7d`。独立机械诊断记录在同名
`_diagnosis/`目录。按冻结门，Gold、EM/F1/IHR与prospective均未打开，且不得在same pilot90上继续改prompt、
parser或阈值重判。

本轮结论不是“问题拆解无效”，而是“现有强final-answer SFT不能稳定零样本承担query controller”。若继续
动态检索线，下一项实现应是用train-side监督/蒸馏训练专门controller，再建立新的family-disjoint Gold-free
验证；停止继续对binder做局部补丁。该新训练属于新的核心实验变量，需另行预注册和批准。

并行完成论文首版重写：`docs/paper/polished_draft.md`现为完整中英双语稿，按问题—缺口—方法—RQ结果—
局限组织，并只使用本地原论文可核验的9条参考文献。稿件严格分离same-resource legacy结果与额外Wikidata
ProofKG结果，保留PPO、跨数据集ProofKG、旧α门控及动态检索负结果；尚未完成的PPO-T/PPO-TK、统一IHR、
完整canonical baseline主表等均标为TBD。首版当时的稿件SHA256=
`c67538acdd9d104a6065aef6b97eb53a4355d51de6bb1a2e1093a77502852ebb`；这是首版、已被本文件第39节所述
方法优先重写取代的历史草稿哈希，不是当前稿件哈希，也不是新的实验依据。

## 39. 方法优先的论文叙事重构（2026-09-04）

研究者指出首版稿件仍受开发过程牵引，要求按完整方法框架重写，并以“方法设计成立、实验负责验证”作为
论文组织原则。现已新增`docs/paper/PAPER_REWRITE_BLUEPRINT.md`并据此重构
`docs/paper/polished_draft.md`。本轮只改论文与计划文件，没有修改数据、Gold、baseline、evaluation protocol、
reward代码或训练配置，也没有启动新实验。

论文训练主线现固定为：citation-aware Strong SFT Reader → 冻结的一次检索mixed3 contexts → matched
PPO-T/PPO-TK。ProofKG只在eligible 2Wiki轨迹上提供图过程项，PPO-TK−PPO-T是训练主张的决定性差值。
Query Controller是另行训练、冻结并独立评估的推理期检索扩展；它没有参与当前`7,200`条PPO schedule的
生成，也不阻塞该配对训练。2Wiki额外Wikidata matched
`0.4033/0.4582 -> 0.6400/0.6944`只解释输入供给效用，不冒充PPO训练收益。

正文已删除“项目失败流水账”式摘要/结论，将开发版本集中到附录；方法章节用完成态说明设计，但尚未完成的
结果只保留HTML内部槽位或表中破折号。数据章现严格区分训练单位并写入可追溯数量：Strong SFT实际使用
HotpotQA train的`4,751`条accepted trajectories；PPO为`1,799`个唯一问题、`1,800`个prompt groups、
`K=4`和`7,200`条scheduled rollouts；若PPO-TK启动，其中`400`个2Wiki groups具备ProofKG过程奖励资格，
对应`1,600`条scheduled eligible rollouts，其余`1,399`个问题只配置outcome/text信号。当前数据状态仍是
`COMPLETE_DATA_NOT_TRAINED`，正式paired checkpoints和optimizer updates均为0。Controller release尚未物化，
其数量只留内部槽位，不把capacity或目标规模写成已完成数据。

论文中的预注册目标奖励写成`R_out + R_text + 0.2 I_eligible R_Proof`，其中
`R_Proof=(0.05P+0.15G+0.35HO+0.45m_AA)/(0.55+0.45m_A)`；非法轨迹两臂均为`-4`，
`gamma=.95`只用于return/GAE。历史learned alpha不再与当前PPO配对混写，只能在新建fixed-vs-learned
matched协议后作为独立消融。当前仍有一个训练前P0阻断：`proofkg_process_v2.py`的temporal derivation
错误检查tail字符串长度而非triple结构，四位年份会被跳过；修复属于核心reward变更，必须另获确认、升版本、
补测试并重冻PPO pair lock后才能正式训练。数据与Gold边界同时更正：Gold不进入prompt、ProofKG边或
execution；训练Gold用于SFT筛选/target和PPO outcome scoring；400个ProofKG qids中的125个hard/A-safe
qids还使用train-only Gold outcome行为完成上游难度分层，275个automatic qids的图构建与入组不使用Gold outcome。

论文结果顺序现固定为：Wiki18 canonical端到端主表 → PPO-TK−PPO-T → eligibility/IHR → 2Wiki额外Wikidata
supply → Controller扩展。外部baseline只能称为共同reporting qids、Wiki18和canonical scorer下的端到端参照；
TRACE/ReaRAG保留原生检索。Strong SFT的`0.383/0.490`、`0.427/0.485`、`0.247/0.342`已标为历史
canonical reference，正式matched主表必须与PPO-T/PPO-TK在最终Scheme-A contexts上同批fresh重跑。

本轮方法优先重写的最终文件哈希为：`docs/paper/polished_draft.md` =
`70db2ffe7211c96e916f2084424c7f0f9a291e28e171dd4978e1843ba1d557ef`；
`docs/paper/PAPER_REWRITE_BLUEPRINT.md` =
`e042875dd10d3b18b71d0c9c349d58dd0e01f1e457d19bcc277ca790b0f0bfee`。二者只冻结写作状态，不能作为实验结果凭证。

## 40. 完整成功态目标论文重写（2026-09-04）

研究者进一步明确：`polished_draft.md`不是当前最小可执行协议的实验报告，而是按照“最终完整方法中每个环节
均有效”来组织的目标论文。因此正文主线由第39节的最小paired-PPO叙事提升为：

`observation-conditioned query controller -> text/KG dual-source acquisition -> provenance-aware evidence state
-> validity-masked learned alpha -> text/graph process supervision -> modular Controller/Reader SFT -> Reader PPO`。

本轮只修改论文和写作蓝图，没有修改数据、Gold、baseline、evaluation protocol、reward实现或训练配置，也
没有启动实验。摘要与引言按PILOTRAG式“问题—缺口—统一方法—训练原则—贡献”组织，不出现Strong SFT、
PPO臂名称、样本数、路径、hash或工程门。Strong SFT被抽象为Reader的监督轨迹初始化；“相同初始化和预算、
只移除图过程信号”的PPO对照只在实验设计/消融中出现。

方法定义同步修正为可闭合的目标形式：检索轮次与推理步骤使用不同索引；Controller输出
`(query, anchor, relation intent, dependencies, output slot)`结构化subgoal并显式选择text/graph/hybrid/
answer动作；subgoal plan是required hops唯一来源，evaluation不使用Gold answer/supporting facts构造plan；
中间答案只有在绑定到检索provenance后才写回controller状态。标准图事实为
`(head entity, relation, tail entity or value)`；来源权重以
`precision*required-hop relevance`为连续可信目标并用`BCEWithLogitsLoss`训练，PPO rollout中采用
`alpha_graph=m_graph*sigmoid(logit)`与`alpha_text=1-alpha_graph`，使不合格图证据严格回退文本；
测试时不计算reward，门控作用由优化后的Reader参数体现；
P/H/O/G/A沿用可追溯图过程公式，图属性按完整轨迹判定，不伪装成逐token答案一致性。
Controller-SFT与Reader-SFT使用不同target/adapter但共享provenance contract；alpha在train-only轨迹上校准、
独立calibration split选择并在PPO中冻结；Reader PPO采用正式clipped objective并在训练期间冻结Controller，
以隔离证据获取与证据利用。

实验叙事按目标成功态写入四类正向结论槽位：三数据集总体提升、来源加权过程监督有效、自适应控制器有效、
不同证据结构下均可泛化。槽位没有编造数值，并在文件顶部明确为投稿前必须用冻结实验替换或删除的研究假设。
唯一填入的定量段仍是已完成的2Wiki enhanced-resource matched control：dev n=300、seed42、greedy、同一SFT
checkpoint和同一10篇canonical fresh passages，仅替换KG block，EM `0.4033 -> 0.6400`、F1
`0.4582 -> 0.6944`、paired EM CI `[.170,.303]`、McNemar `p=5.3e-11`。该结果继续只解释额外历史
Wikidata供给效用，不冒充PPO收益；由于使用development而非未接触confirmation split，正文已降格为
exploratory supporting analysis，最终投稿前必须在未参与reward/alpha选择的held-out 2Wiki数据上复验。

第39节及更早版本不删除，作为历史写作/最小实验蓝图保留。当前目标正文与已冻结实现之间仍存在明确差额：
专门Controller release、三数据集双源轨迹、learned-alpha matched消融和完整Reader PPO尚需正式物化与验证；
若任一正式结果不支持成功态叙事，投稿版必须收缩相应主张。

本轮目标稿写作状态哈希：`docs/paper/polished_draft.md` =
`c95e9a97b554d1e81e539581d603d52388de6df5b164a7a81c80dd7ef45ee587`；
`docs/paper/PAPER_REWRITE_BLUEPRINT.md` =
`6c7b257b0debdb0d39fe55d850d4b59045f0c7de54a7a186e99e5577191b3f`。二者只标识当前写作版本，
不构成未完成实验的证据。

## 41. 专门 Query Controller exact-20 可训练性与 Gold-free 机制评估（2026-09-04）

为检验 v9.1 的失败究竟来自“动态拆解不可行”，还是来自让 final-answer Reader 零样本兼任检索控制器，
本阶段训练了独立的 observation-conditioned Query Controller。Controller 每次只输出一个规范检索动作；
`q1` 只读取原问题，`q2_dynamic` 读取原问题、已执行的 q1 和一条绑定到文档/句子 provenance 的中间观察。
训练与评估只覆盖 2WikiMultiHopQA 和 MuSiQue；HotpotQA 的精确动作标签覆盖尚为 `UNKNOWN`，没有混入本轮。
本轮不执行检索、不读取最终答案 Gold，也不计算 EM/F1/IHR。

冻结 action release 包含 train `2,400` 条、dev `240` 条；每题严格成对包含 q1/q2_dynamic，train/dev 的
qid 与 family overlap 均为 0。exact-20 probe 从 base instruct 初始化，QLoRA r=16，学习率 `1e-4`，
gradient accumulation=8，只执行 20 个 optimizer steps。v4.2、v4.3 的失败记录均 append-only 保留：
前者暴露“并存第二 adapter”的 PEFT 校验假差异；后者暴露验证器把 FP32 磁盘权重强制按 BF16 重载后又
要求逐位相等。诊断确认 v4.3 的 256/256 张量均精确等于相应 BF16 cast，非 checkpoint 损坏。v4.4 仅将
clean reload 对齐真实 PEFT 推理默认 FP32，并继续使用 `torch.equal` 零容差；saved/live/reload 三份 dtype
inventory 均为 `torch.float32: 256`。

v4.4 probe 正式完成：20/20 steps，train loss=`3.5022`，逐步 loss 从 `7.2104` 降至末步 `1.5319`；
20 条 loss/gradient 均有限且观察到非零训练梯度；峰值 GPU reserved=`10.01 GiB`；adapter 保存、卸载和
clean single-adapter reload 256/256 张量逐位一致。训练 manifest 位于
`outputs/probes/query_controller_action_v1_probe20_seed42_v4_4/manifest.json`，状态为 `COMPLETE`。

首次 dev 启动在 CUDA 前因 runner 漏传已验证的 protocol report/manifest 两个 SHA 而停止，产生 0 条预测；
该事实保留在 append-only E0 addendum。没有重训，而是冻结 eval-only successor `v4_4_eval_e1`，唯一实现
变化为在 runner 与 formal scorer 的两个调用点转发这两个哈希；父训练 protocol、checkpoint、dev240、
greedy decoding 和机制门均不变。E1 完成 240/240 条生成，独立 CPU scorer 的正式结果为：

- 2Wiki q1 schema/dependency/state-use=`1.000`，q2_dynamic=`0.9667`，通过 `.97/.95` 分槽门；
- MuSiQue q1=`1.000`，q2_dynamic=`0.8500`，q2 低于 `.95`；
- overall identity join=`1.000`、Gold boundary=`1.000`，但 placeholder-free 与 query-nonrepeat 均为
  `0.9542<1.000`；11/240 条无效，集中在 q2 的未使用中间观察、重复查询或 JSON/PID 格式异常；
- 因此联合正式状态为 `FAIL_STOP_ACTION_MECHANICS`，不得写成“两数据集 Controller 已通过”。

结果证据位于：

- 训练：`outputs/probes/query_controller_action_v1_probe20_seed42_v4_4/`；
- eval-only 协议：`outputs/audits/query_controller_v4_4_dev_eval_e1_protocol/`；
- 240 条不可变预测：`outputs/validation/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1/`；
- 独立机制评分：`outputs/audits/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1_score/`。

科学结论应分层表述：专门 Controller 的动作学习在 2Wiki 上已通过小样本 Gold-free 机制门，说明子问题拆解
并非不可实现；MuSiQue 的 q2 尚未通过，且本阶段没有验证真实 Reader 预测中间答案下的在线检索收益。
下一步不再修改同一 dev 门或用 20-step checkpoint 直接主张 EM/F1，而是冻结完整 `2,400` action 的正式
Controller-SFT 配置，优先提高 MuSiQue q2 格式/状态使用；在新的 family-disjoint mechanism split 通过后，
再运行 matched-budget `one-shot vs dynamic Controller` 的真实 `q1 -> retrieve -> Reader observation -> q2 ->
retrieve -> final Reader` A/B，最终以 EM/F1、检索调用数和 latency 决定是否进入论文主结果。

## 42. HotpotQA Controller 银标覆盖 pilot30（2026-09-05，生成前阶段）

为避免在缺少 HotpotQA 精确动作监督时直接训练三数据集 Controller，本阶段先验证能否从 HotpotQA train 的
bridge 样本稳定构造两条可执行查询。这里验证的是**训练侧银标标签覆盖**，不是正式 dev/test 结果，也不是
Controller 或 Reader 的 EM/F1。最终在线链仍固定为：`Controller q1 -> Wiki18 -> 强 SFT Reader 预测并绑定
observation -> Controller q2 -> Wiki18 -> 同一强 SFT Reader 回答原问题`；运行时禁止注入标注 bridge。

已实现 `kgproweight/data/hotpot_controller_silver.py`：仅接受方向唯一的两文档 bridge chain，producer 只看
遮蔽 bridge/final 的语义证据；q1 不得看到中间值，q2-template 只含一个 `#1`，经审核后才用训练标注中间值
物化训练 target。Hotpot 使用真实 `metadata.supporting_facts.title[i]` provenance，不能伪装成 2Wiki/MuSiQue
annotation path；输出暂为 companion schema，正式训练前必须建立三数据集 successor validator/trainer，冻结
v4.4 不原地修改。

正式 identity freeze 位于
`outputs/audits/query_controller_hotpot_silver_label_coverage_pilot30_seed20260904_v1/`。在冻结口径下，raw
train 90,447 条中有 14,137 条通过严格链筛选；排除已枚举的 58 个历史、20 个训练和 7 个追加 registry
entry 后仍有 9,981 个 dataset-scoped family。固定 pilot 为 easy/medium/hard 各 10 条，qid/family consumed
overlap 均为 0，sealed prospective 内容未打开，失败不允许换题。旧 checkpoint 的完整输入账本仍为
`UNKNOWN`，因此只能称“与已枚举 consumed union 不重叠”，不得称全局从未见。筛选使用了 train final answer
和 Gold supporting facts 来绑定/排除链，这一点已通过 append-only `metadata_addendum_v1_1.json`补充披露；
identity 输出本身仍只有 dataset/qid/question。

独立预调用审查发现并修复了三类通用缺陷：冠词/重音别名可绕过泄漏检查、带中间名实体未被完整掩码、
`proposal_sha256`未与最终 q1/q2 action 重算绑定。修复后相关回归共 83 项通过。固定 30 题中 29 题通过
最新 source/mask 门；1 题确实在原问题中含 bridge 扩展别名，按固定分母记为预调用失败且不替换。此前
“5 题不可掩码”是加固过程中的中间诊断：其中 4 题只是支持句使用了扩展人名，统一有序别名掩码后已安全
恢复；该演变已写入 append-only addendum。

DeepSeek 单生成/双审执行协议已冻结在
`outputs/audits/query_controller_hotpot_silver_pilot30_execution_protocol_seed20260904_v1/`，protocol SHA256=
`f690063157f8818a79168f2f511fd7ec1becb2a431e81dcf26d372336a48360d`。Producer 固定
`deepseek-v4-flash`、temperature=0、每题仅一个候选；Reviewer-1 为不看 Gold 的 Flash 盲审，Reviewer-2
为看 train chain 的 `deepseek-v4-pro` adjudicator。二者使用新上下文、互不可见，只允许称
context-isolated dual review，不能声称统计独立；格式/语义失败不重试，仅 transport 失败可用完全相同请求
重试。模型可见 producer payload 不含 dataset/qid，并用每题固定长度 opaque nonce 替换命名 mask。

当前阶段仍为 `NOT_GENERATED_NOT_TRAINED`：API、Wiki18 retrieval、Reader inference 和训练调用均为 0。
下一步先完成固定30题的单次生成、机械校验和隔离双审；随后另冻 retrieval/Reader successor protocol，使用
canonical Wiki18（E5+BM25->RRF->BGE Top-10）与固定强 SFT checkpoint 验证真实预测 observation 下的 q1
与 q2 链。只有固定分母至少 24/30 同时通过生成、双审、检索支持和 Reader provenance 门，才扩到
HotpotQA 600/60/30 并申请三数据集 Controller-SFT；否则保留失败并停止，不降低门、不补题。
