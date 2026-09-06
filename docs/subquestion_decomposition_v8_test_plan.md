# 动态问题拆解与交错检索 v8 测试方案

> 状态：`DEVELOPMENT90_FAIL_STOP_BEFORE_GOLD`
>
> 日期：2026-09-04
>
> Phase 0、容量审计、严格Scope A队列冻结、历史Trace预算审计与v8 Gold-free纯函数核心已完成。
> 研究者已在2026-09-04批准两次controller调用协议；正式runner、调用账本、append-only driver和首版
> implementation freeze已经完成。工程smoke attempt001在真实运行中测得逐query扫描Wiki18需67--83秒，
> 因逐题串行将使smoke约需50--66分钟、development约需数小时，故在0行科学输出时人工停止并写入
> `FAILED_RETAINED_APPEND_ONLY`。当前只做结果等价的分阶段batch改造；当前仍不得打开Gold、读取sealed
> prospective、启动SFT/PPO或修改reward/loss。

> 等价batch实现已冻结为implementation V2。真实engineering smoke attempt002于2026-09-04完成：
> 12题三数据集各4题、84/84 logical retrieval、3次full-index passes（unique queries 12/10/10）、
> 16项runtime/mechanism gates全部PASS；同一强SFT物理模型对象、固定10篇、cache守恒及Gold/prospective=0
> 均已验证。该结果只解锁冻结development90的Gold-free运行，不是EM/F1或方法效用结果。

> 冻结development90随后已完整运行并append-only落盘，但未通过Gold-free机制门：q1有效率为
> Hotpot/2Wiki/MuSiQue=`1.000/.933/.967`；a1 admissible仅`0/.133/.067`；B q2有效率`.678`；
> C dynamic transition ITT为`0/.033/.033`。因此状态为`FAIL_STOP_GOLD_FREE_GATES`，未打开Gold、
> 未计算EM/F1/IHR、未读取sealed prospective。无Gold分解显示42/90个a1因跨文档surface歧义被拒、
> 29/90未找到surface，B q2另有29/90重复原问题或q1。结论仅否定“强final-answer SFT零训练兼任
> controller/subanswer reader”的具体实现，不否定专门controller-SFT后的方法。

## 1. 要回答的研究问题

本轮不再问笼统的“复杂问题能否拆解”，而分成三个可证伪问题：

1. **接口问题**：v7 的中间答案是否主要被 `answer_type/abstain/citation` 契约误杀？
2. **机制问题**：在相同检索调用与最终上下文预算下，显式使用上一跳答案及其绑定 observation，能否比
   “不使用上一跳答案/observation 的静态拆解”召回并保留更有效的下一跳证据？
3. **训练问题**：如果零训练动态控制器可行，后续 controller-SFT 和 on-policy PPO 是否有足够探索空间，
   以及 alpha-gated KG reward 是否在 outcome-only PPO 之上提供增量？

Phase 0 是对已消费 v7 轨迹的接口诊断，不是 v8 动态路线的晋级门；v8 的机制门通过后才能打开答案
效用，效用门通过后才能进入训练。这样可以避免再次直接跑昂贵 PPO，却不知道失败来自 planner、reader、
retrieval、evidence merge 还是 reward。

## 2. 已有证据与本方案的逻辑

### 2.1 v7 已经证明什么

- Planner 在 HotpotQA20/MuSiQue20 上均为 20/20 schema-valid、可执行，因此不是“完全拆不出结构”。
- schema-valid 不代表语义正确：例如 `hotpotqa::dev_1107` 需要找任教大学，plan 却使用
  `educated at/P69`；`dev_1670` 需要找市长对应的城市，却使用人物的 `position held/P39`。
- 41 个 depth-1 reader 任务中严格 JSON 通过 30 个，机械验证仅 6 个；24 个被
  `answer_type=other` 拒绝，10 个因 `abstain` 非布尔而拒绝，1 个 citation 数量不符。
- 在锁定单次生成、观测到 `retry_count=0` 的 v7 轨迹下，机械验证率最乐观上界为 HotpotQA
  `3/19=.158`、MuSiQue `4/23=.174`，低于冻结的 `.40` 门。因此 v7 已在 Gold 前正式停止，
  没有 EM/F1/IHR。

### 2.2 从本地论文借什么

- IRCoT：不是先冻完整 DAG，而是 `Q -> retrieve -> 生成下一句 reasoning -> retrieve -> ...`；下一步
  由当前 observation 决定，旧 passages 累积，最后由独立 reader 回答。原论文每轮只保留新生成
  reasoning 的第一句作下一 query，最多 8 个 reasoning steps、累计 passages 上限 15，并用独立 QA
  reader 给出正式答案。本文只借动态交错机制，不照搬其用 Gold recall 在 dev 上选择检索 K 的调参方法；
  v8 的 slot/K/merge 必须在运行前一次性冻结。
- Decomposed Prompting：decomposer 逐轮产生单跳问题；专门 handler 返回 answer 和 documents；上一答案
  写回 history 后才生成下一问题。
- CoRAG：下一 query 依赖原问题与历史 `(Q_i,A_i)`，并专门训练 next-query、subanswer、final-answer 三类
  能力，说明 final-answer SFT 不会自然等价于交互控制器 SFT。
- Search-R1/R1-Searcher：若进入 RL，检索必须处于 on-policy rollout 内，环境返回的 document tokens
  需要从 policy loss/KL 中 mask；先验证 action 格式、有效查询与探索空间，再谈 PPO。

因此 v8 的核心变量不是“再调一个 PID 规则”，而是：**把第二个子问题从 plan-once 改成由第一子问题的
经证据验证答案动态生成。**

## 3. 必须先解决的 baseline 口径冲突

现有 `trace (IRCOT)` 可作为端到端参考，但不能直接作为新实验的预算匹配因果对照。只读核查其 n=300
`intermediate_data.json` 后，`output.retrieval_result` 实际长度为：

| 数据集 | 最小 | 最大 | 平均 |
|---|---:|---:|---:|
| HotpotQA | 11 | 67 | 26.19 |
| 2WikiMultiHopQA | 11 | 69 | 28.74 |
| MuSiQue | 12 | 67 | 28.64 |

这与 `docs/paper/baseline_results.md` 中“逐样本 retrieval_result 10 docs”的文字不一致。正式 v8 比较前必须
单独生成 append-only baseline-budget 审计，查清 `10` 指每轮 rerank top-k 还是最终累计 passages；不得
覆盖历史结果。历史 trace EM/F1 可以继续作为旧系统参考，但不得称为与 v8 同预算。

该append-only审计已完成，产物位于
`outputs/audits/historical_trace_ircot_budget_n300_seed42_v1/`。三数据集各n=300的历史最终累计文档数
均值为Hotpot/2Wiki/MuSiQue=`26.19/28.7433/28.6367`，记录的generation rounds均为2–6。900题
的容器/schema/cardinality和最终文档ID唯一性检查通过。但产物不能恢复每调用top-k、真实物理
检索次数、精确runtime code hash、cache/retry或wall time，这些均正式记为`UNKNOWN`。因此历史
Trace只是IRCoT-style系统参考，不是fixed-10 matched control，也不是已确认的论文忠实复现。

## 4. Phase 0：冻结 41 条回答的 Gold-free 接口反事实审计

### 4.1 输入

- `outputs/validation/subquestion_dependent_retrieval_v7_subanswers_depth1_v1_retry1/subanswers.jsonl`
- 对应 41 条 task、producer passages、implementation/plan/stage/model locks。
- 禁止读取 `data/raw`、answer、supporting facts、MuSiQue decomposition 或任何 scorer Gold。

### 4.2 三种只读口径

1. `P0-current`：完整复现 v7 parser/verifier，预期 6/41，用作一致性检查。
2. `P1-surface-class-inferred`：先完整通过原 v7 JSON parser，因此模型自报 `answer_type` 仍须属于原枚举，
   `abstain` 仍须为真布尔，并仍要求唯一有效 citation；然后唯一变化是：admission 不再使用模型自报
   `answer_type`，而以原 v7 `_looks_like_date()`、`_looks_like_number()`、否则 entity-like 的固定顺序生成
   effective surface class，再原样调用 v7 verifier。原 answer bytes、长度上限、subject-echo、citation 和
   cited-passage lexical-locality 规则全部不变。这里的 entity-like 只是排除式匹配分支，不是 NER 或实体
   链接结论。
3. `P2-contract-upper-bound`：仅作乐观诊断；除 P1 外，允许在 answer 非空且 citation 唯一时把
   `abstain:""` 视作 `false`。P2 不能直接成为正式 parser，不能作为论文结果。

确定性类型推断在执行前写死并测试：完全复用 v7 normalization/helper；日期/年份优先，数值其次，其余
进入 entity-like 分支。不得在 P1 额外加入 128 字符、token 数或“是否长句”等新门，否则就不再是
type-only 反事实。

### 4.3 输出与门

每数据集报告 task 数、parse 数、locality-pass 数、subject-echo、citation miss、按原失败原因可恢复数。

在写本方案时做过一次**未注册、未落盘的只读探索性重放**：只忽略模型自报 type，仍先经过原版严格
JSON parser，保持 `abstain` 必须为真布尔、唯一 citation、subject/locality 和原 verifier 全部不变，得到
HotpotQA `11/19=.579`、MuSiQue `17/22=.773`。这不能替代正式 append-only 审计，但说明 Phase 0 很可能
以极低成本确认“主要是接口误杀”，而不是 passage 必然没有中间答案。

- P1 两数据集均 `verified_rate >= .40`：可确认“模型自报 type”是 v7 的主要接口误杀之一，并支持 v8
  直接删除该输出字段；v8 的单行 subanswer contract 本身不再运行 type-inference。
- P1 未过而 P2 两数据集均过 `.40`：需要新 reader schema 或 constrained decoding；禁止在正式评估中
  直接宽松 coercion。
- P2 任一数据集仍低于 `.40`：接口修复不足，停止静态 v7 分支，直接检验动态拆解。

Phase 0 是已经被探索性重放预知的 consumed-development 复现诊断，不能打开 Gold、不能宣称答案更好，
也不决定是否执行 v8 动态路线；v8 仍须使用 fresh development 和自身机制门。

### 4.4 Phase 0 正式结果（2026-09-04）

研究者批准后，Phase 0 已先冻结协议、再以冻结代码和输入正式运行。Experiment ID 为
`SUBQUESTION-DECOMPOSITION-V8-PHASE0-V7-CONTRACT-COUNTERFACTUAL-DEV41-SEED20260904-V1`；协议与结果分别位于：

- `outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_protocol_v1/protocol.json`；
- `outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_counterfactual_v1/`。

72 项相关回归测试通过；正式审计为 CPU-only、0 model call、0 retrieval call、`gold_access=false`。冻结
P0 完整逐题复现旧结果后才允许计算 P1/P2，结果为：

| 数据集 | P0 原契约 | P1 仅程序推断 surface class | P2 诊断上界 |
|---|---:|---:|---:|
| HotpotQA | 3/19（.158） | 11/19（.579） | 15/19（.789） |
| MuSiQue | 3/22（.136） | 17/22（.773） | 20/22（.909） |
| 合计 | 6/41（.146） | 28/41（.683） | 35/41（.854） |

P1 在两个数据集都越过预先写死的 `.40` 机械门，正式状态为
`PASS_P1_TYPE_ONLY_INTERFACE_DIAGNOSIS`。这支持一个有限但重要的工程结论：v7 让模型自报
`answer_type` 确实造成了主要接口误拒，v8 应采用“短答案或 sentinel + 系统绑定 provenance”的输出契约，
不再让自报类型控制接纳。该结果不测语义蕴含、support recall、EM/F1/IHR，也不是 v8 动态方法的晋级门；
P2 的空字符串 coercion 仍只保留为乐观诊断，不进入正式 reader。

独立复核另发现两项不改变本次数值、但必须在未来运行前修复的审计加固项：聚合 P0 cardinality/count 门
在当前实现中晚于 P1/P2 计算；此外，部分间接祖先和 `question_kg.py` 没有在 Phase 0 协议中直接重验哈希。
现有结果保持不动并由
`outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_counterfactual_v1_review_addendum/metadata_addendum.json`
追加披露；不得称当前版本已经认证“完整代码/祖先闭包”。若重跑，必须先修复并使用新协议、新 Experiment ID。

## 5. Phase 1：数据与协议一次性冻结

### 5.1 数据分层

先运行纯 CPU capacity audit，再一次性冻结：

- `engineering-smoke`：每数据集 4 题，复用已经消费的 HotpotQA/MuSiQue/2Wiki 题或合成 fixture，只检查
  代码和 Gold-free telemetry，正式 development 不能首次承担 2Wiki 接口调试；
- `development`：HotpotQA/2Wiki/MuSiQue 各 30 题；
- `prospective-validation`：三数据集各 300 题，在任何 development 输出前冻结并封存；这是周内方向验证，
  不是逐数据集高功效论文确认；
- `paper-confirmation-reserve`：若容量允许，另冻结三数据集各 1000 题的完全独立 reserve，本轮不检索、不
  推理、不打开；只有 prospective validation 通过且研究者再次批准后，才按同一协议一次性使用；
- 如任一数据集无法获得足够的 QID/family-disjoint 样本，应停止并报告容量不足，不得临时降 n。

development/prospective-validation/paper-confirmation-reserve 必须排除：所有训练/SFT/PPO/silver qid 与
family、v4–v7 development、所有
ProofKG/reward/verifier consumed cohorts、历史 canonical main900 qid/family，以及彼此之间的 qid/family。
只保存 `dataset/qid/question/question_sha256/family_sha256`；Gold 另存、不可被检索物化进程访问。

隔离键冻结为历史一致的 dataset-scoped `(dataset,qid)` 与 `(dataset,family_sha256)`；这里的 family 使用
`answer-free-lexical-family-v1`，只能称 lexical-family proxy，不能声称是真实语义 family。若现有
answer-free registry 容量不足，selector 不得直接打开含 Gold 的 dev 文件后仍宣称 Gold-free；必须另设
唯一有权读取源文件的 custodian-export，诚实记录 `source_may_contain_gold=true`，仅投影严格 allowlist 的
question identity，确保下游 selector/materializer 的 `gold_access=false`。

### 5.1.1 正式 capacity-only 审计结果（2026-09-04）

正式产物位于
`outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1/`，Experiment ID 为
`SUBQUESTION-DECOMPOSITION-V8-COHORT-CAPACITY-AUDIT-V1`，状态为
`SCOPE_A_ONLY_PASS_DEV30_PROSPECTIVE300_FAIL_BALANCED_RESERVE1000`。审计显式锁定58个历史
evaluation/protocol registry、20个本地可核实训练或保守排除输入及12个manifest/config证据；20个输入中
`training family/qid outside corresponding raw-train = 0/0`。58个registry不能冒充完整历史training ledger；
缺失旧checkpoint输入的精确身份仍为`UNKNOWN`，所以较宽Scope B已正式标为
`INVALID_FOR_FREEZE_INCOMPLETE_TRAINING_LEDGER_DIAGNOSTIC_ONLY`，不得用于选题。

唯一可行动的是严格Scope A：排除全部历史registry qid/family、本地训练ledger qid/family、raw-train qid，
并排除同数据集的全部raw-train lexical family；每个dataset-scoped family最多选一题。

| 数据集 | eligible qid | eligible family/可冻结容量 | 30+300 | 再加reserve1000 |
|---|---:|---:|---:|---:|
| HotpotQA | 6214 | 6213 | PASS | PASS |
| 2WikiMultiHopQA | 355 | 331 | PASS（仅余1 family） | FAIL（缺999） |
| MuSiQue | 1762 | 1644 | PASS | PASS |

本次容量审计不是Gold-free selector：它打开了可能含Gold的train/dev与训练输入文件，但程序只使用
`dataset/id/qid/question`身份投影，`gold_fields_used=false`、`gold_fields_emitted=false`，且报告不输出任何
逐题qid、question、question hash或family hash。现有已物化answer-free registry没有未消费的新题
（`existing_answer_free_unused_pool=0`），因此正式冻结必须另做custodian export。

当前按预案停止在冻结前，等待研究者二选一：

1. 推荐：接受本周只冻结严格Scope A的`development30 + prospective300`/数据集，不设不对称reserve1000；
2. 若坚持`reserve1000×3`，先增加独立2Wiki来源或重新批准改变family隔离/规模，不能静默放宽。

无论选哪条，本轮都未冻结cohort。另一个独立阻塞仍为：当前转换后的dev文件没有已确认可用的官方
supporting-facts/decomposition字段，L3 support-chain scorer的数据源为`UNKNOWN`，必须在打开Gold前解决。

### 5.1.2 严格Scope A正式冻结结果（2026-09-04）

研究者选择严格Scope A并取消本轮reserve1000后，已一次性冻结三数据集各
`development30 + prospective300`。Experiment ID为
`SUBQUESTION-DECOMPOSITION-V8-COHORT-FREEZE-DEV30-PROSPECTIVE300-SEED20260904-V1`，产物位于
`outputs/audits/subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_seed20260904_v1/`。

- `development.identity_only.jsonl`：90行，每数据集30，SHA256=`dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2`；
- `prospective.identity_only.jsonl`：900行，每数据集300，SHA256=`36b680cabef059dae7370bb131b1bafc0f120baf372f4e7666aa0e2d13b13c99`；
- 每行严格只有`dataset/qid/question`；与historical registry、已知training ledger及raw-train的
  dataset-scoped qid/family overlap全为0，两个split之间qid/family overlap也为0；
- source JSONL可能含Gold，custodian进程只访问`id/qid/question`，选择不使用也不输出Gold。

新development-only loader已绑定freeze manifest和development SHA；传入prospective role/path会在读取
题目前拒绝，且无unlock参数。这是可审计的procedural seal，不是加密或外部物理保管；
未来打开prospective必须新增append-only授权与新loader/protocol版本。当前prospective不得用于
检索、模型推理、Gold评分或方法调整。

### 5.2 冻结资源

- Wiki18 corpus/dense/BM25：21,015,324 文档；
- E5 + BM25 RRF、BGE reranker；
- 同一个 Llama3-8B base 和强 SFT adapter；
- greedy decoding、seed42；controller 每次最多 96 new tokens，final reader 512；
- 无 KG、无 alpha、无 reward，先隔离问题拆解本身；
- 全部输入、模型树、index、代码闭包、prompt、parser、merge、scorer 和依赖版本写 SHA256 manifest。

## 6. Phase 2：最小可归因 A/B/C 实验

### 6.1 三臂定义

#### Arm A：canonical one-shot

- 原问题一次检索，冻结 canonical Top-10；
- 同一强 SFT final reader；
- 这是标准系统基线，不与 B/C 声称逐调用预算相同。

#### Arm B：plan-once 子问题控制

- 用原问题先做一次共享 root retrieval；
- controller 从 `Q + root observation` 生成一次共享 `q1`；执行共享q1 retrieval和single-hop reader后，
  B的第二次controller调用只接收固定allowlist `Q + q1 + NO_VERIFIED_SUBANSWER`，看不到root/q1
  observation、检索分数、reader raw output、中间答案、binding telemetry或arm标签；这里的“static”指
  信息静态/observation-blind，不要求物理上早于q1检索；
- q1 retrieval 后仍运行与 C 相同的 single-hop reader，但其输出不进入 B 的 q2；
- 这是与 C 检索/模型调用预算匹配的“拆解但不使用中间答案”控制。

#### Arm C：answer-conditioned 动态拆解

- 与 A/B 共享逐字节相同的 root passages，并与 B 共享逐字节相同的 q1 及 q1 retrieval；
- 检索 q1 后，single-hop reader 从 q1 documents 产生简短 a1 或唯一 sentinel
  `NO_RELEVANT_ANSWER`；
- 系统根据 answer surface 在 q1 documents 中自动绑定 supporting doc/provenance，不要求模型自报
  type 或 citation；
- 只有通过冻结的短答案/locality/subject-echo 检查时，第二次 controller 调用才以
  `Q + q1 + verified a1 + bound evidence excerpt/provenance` 生成 `q2_dynamic`；这里把答案与其证据状态
  视为一个预注册 treatment bundle，不把 C-B 缩写成“只改变 answer 字符串”；
- a1不通过时，C的第二次调用使用与B逐字节相同的
  `Q + q1 + NO_VERIFIED_SUBANSWER` prompt和content key；控制器输出可以安全复用，C/B的selected q2、
  q2 retrieval、最终passages、final prompt和prediction必须逐字节一致；
- a1已通过但`q2_dynamic`输出无效、缺失、重复或含placeholder时，C确定性回退原问题Q并仍执行q2
  retrieval，不读取B static产物、不产生第3次controller调用；该子集保留在ITT并单独报告，不要求与B一致；
- 最后由同一 strong-SFT final reader 基于合并后的固定预算 passages 回答原问题。

因此主比较 `C-B` 测试：**证据约束的 answer/observation-conditioned q2 策略，相对静态 q2 策略是否改善
下一跳证据和最终答案。** 第一轮故意只测一个依赖转换，不加入 q3；若通过，再另立版本测试递归深度。
`C-A` 只报告完整系统相对标准一次检索的收益。

#### 6.1.1 q2预算冲突及研究者裁决

原文同时要求：B第二次controller call生成`q2_static`，C第二次call生成`q2_dynamic`，
C失败时又使用预生成的`q2_static`，且每臂只计2次controller call。若`q2_static`由模型生成，
这四项不能同时成立：独立运行C要么需要第3次call，要么免费读取B的产物。

研究者已在2026-09-04批准把“static”定义为**信息静态/observation-blind**，而不要求物理上早于q1检索：
B/C均在q1 reader后执行自己的第2次无状态controller call；B的prompt只含`Q+q1+
NO_VERIFIED_SUBANSWER`，C在a1有效时才含`a1+provenance`，否则与B prompt逐字节相同。
这样a1-ineligible子集上的prompt/output/retrieval/final context可严格一致，每臂仍只有2次逻辑call。
若a1有效但dynamic q2输出无效，为不隐藏第3次call，C确定性回退原问Q，该子集仍保留
在ITT，但不强制与B逐字节一致。byte-identity门只作用于a1-ineligible子集。该裁决必须由新
append-only protocol与implementation lock绑定；正式development只有在runner测试和engineering smoke
通过后才可启动。

### 6.2 预算和 evidence merge

- A/B/C 的 root query 与 root reranked top-10 必须逐字节相同。B/C 每题固定 3 个 retrieval calls：
  1 个共享 root + 1 个共享 q1 + 1 个 q2；固定 2 个 controller slots和1个q1 single-hop-reader slot；
- 无效 q1 使用原问题 Q 作为预注册 fail-closed retrieval query；无效 q2_static 使用 Q；a1有效但
  q2_dynamic无效时直接使用Q。fallback仍真实执行对应逻辑检索请求并计入成本，不计作有效动态query；
  因此 B/C 不只是最大 logical slots 相同，而是每题实际 retrieval/controller/reader 调用数也相同；
- 每个真实 query 均调用同一 canonical 检索栈：E5 top-100 + BM25 top-100，RRF(k=60) 合并为 top-50，
  再由同一 BGE reranker 产生 top-10；最终 merge 只从各轮 top-10 按下述固定槽位取文档。任何实现期发现的
  配置差异都必须在 protocol 冻结前解决，不能在 development 后调整；
- B/C 使用相同 controller 调用数、single-hop reader 调用数、retrieval slots、candidate-k、reranker和
  final-reader调用；
- 最终证据槽固定为 root top-6 + q1 top-2 novel + q2 top-2 novel，按 document identity 稳定去重。
  q1 reader 绑定的 provenance doc 若尚未在 root top-6，须在 q1 的两个槽中优先保留；该规则对 B/C
  相同。子问题无效、重复或不足时，从共享 root rank7–10 按原序补足；最终严格10篇；
- B/C的共享root、q1和a1-ineligible static slot2可按完整content key缓存复用，但每个独立系统仍登记完整
  逻辑调用。冻结守恒式为`logical requests = cache hits + cache misses`；joint physical execution按唯一
  验证content key/cache miss报告，不要求B/C物理调用数相同，也不能用复用掩盖standalone-equivalent成本；
- 不再使用 full-question CE 将新文档与 A 尾部作事后替换，因为 v6 已证明该分数不能代表决定性 hop
  support；B/C 只采用上述唯一的 `6+2+2` allocation；
- 所有 query、document、round、source rank、score、dedup 和 provenance 逐题记录；
- 另报实际 prompt tokens、generated tokens、wall time 和物理检索次数。A 与 B/C 的系统成本差异必须公开。

### 6.3 Query 与 subanswer 契约

query 只允许一行自然语言单跳问题，不要求 PID/关系标签/JSON。机械门仅检查：非空、无 unresolved
placeholder、不是上一 query 的规范化重复、长度 3–256 字符、无 Gold 字段。

subanswer 只允许一行短答案或精确 sentinel `NO_RELEVANT_ANSWER`。系统在 q1 reranked top-10 中搜索
唯一/可追溯
surface location并写 provenance；不要求模型输出 `answer_type/cited_doc_ids/abstain`。locality 仍只称
lexical locality，不称 semantic entailment/support recall。

## 7. Phase 3：Gold-free 机制门

先在 engineering-smoke 通过完整性测试，再对冻结 development90 运行。任何一项失败均在 Gold 前停止：

| 门 | 每数据集阈值 |
|---|---:|
| q1 schema-valid | >= .95 |
| q2_static schema-valid | >= .90 |
| C 的机械验证 a1 rate | >= .40 |
| C 的有效 dynamic q2 transition / 全部题 | >= .32 |
| 重复/空/padding query rate | <= .05 |
| B/C 每题logical retrieval/controller/reader ledger identity | 1.00 |
| logical request = cache hit + cache miss守恒 | 1.00 |
| B static observation-blind prompt allowlist | 1.00 |
| C dynamic state/binding完整性 | 1.00 |
| B/C 最终 passages 数为10、无重复 | 1.00 |
| C在a1-ineligible子集与B的slot2/q2/passage/final-prompt/prediction byte identity | 1.00 |
| C在a1-eligible但dynamic-invalid子集精确回退Q且无第3 call | 1.00 |
| A/B/C root query、top-10 byte identity | 1.00 |
| runtime error | 0 |
| Gold access / forbidden recursive fields | 0 |

其中 `.32=.40×.80` 是预注册的全体 ITT 分母门，不能通过只保留 C 成功题缩小分母。另报告但不冒充
support 指标：条件 dynamic-q2 valid rate、new-document rate、q2 query lexical novelty、a1 surface
locality、每轮 retrieval score、文档来源和 token/call cost。

## 8. Phase 4：development 效用门

只有 Phase 3 全过，才用完全不含 Gold 的冻结 prompts 调用同一 final reader，先落盘并哈希 A/B/C
predictions。之后由独立 scorer 只读 join Gold；final reader 进程永远不能访问 Gold 文件。

按以下顺序报告诊断漏斗，防止再把“格式通过”误称为“证据正确”：L0/L1 在 Gold-free materialization 后
即可计算；L2 可在不向审阅者暴露最终 prediction/EM 的情况下盲审；L3/L4 只能在 predictions 已冻结后由
独立 support/scoring job 计算。

1. L0：query/action 格式有效；
2. L1：subanswer 可在来源 passage 做字面定位；
3. L2：subanswer 是否回答了当前单跳问题、是否填入正确变量/关系方向；
4. L3：最终10篇是否包含完整 supporting-evidence chain；
5. L4：最终答案 EM/F1。

L2在不知道arm、最终prediction和EM的条件下，对 development90 的 B/C q2 全量双人盲审，随机化呈现
顺序；rubric固定为“单一未知量、与当前状态一致、关系/变量方向正确、回答后能推进原问题”四项合取。
机制门要求 C 的合取通过率至少 `.65`、相对 B 至少 `+15pp`、任一数据集 C-B 不低于 `-5pp`，且
Cohen's kappa `>=.60`；低于kappa门视为量表失败，先修审阅协议，不能判模型失败。AI只能作预审，若用于
正式结论必须明确模型和局限。L3由独立scorer
在所有query/passages冻结后，版本化读取Hotpot supporting facts、MuSiQue decomposition/support paragraphs
和2Wiki evidence并映射到Wiki18。映射失败题仍留在ITT总体，只报告corpus-support ceiling；若无法可靠映射，
L3写`NOT_AVAILABLE`，绝不能用Gold答案字面、query词重叠或CE分数替代。

证据机制门是 prospective validation 解锁的必要条件，而不是可选描述：在 official-support 映射覆盖率每数据集
`>=.80` 时，C 相对 B 的完整 support-chain recall@10 pooled 至少 `+5pp`，任一数据集不得低于 `-2pp`。
若某数据集映射覆盖率低于 `.80`，该数据集 L3 写 `NOT_AVAILABLE`，不能对它声称“召回了更完整证据”；
此时只允许把零训练结果称为系统效用开发结果，不自动解锁跨数据集 prospective validation，需另行批准
映射修复协议。若 query 语义正确且 union recall 提高、但 final-10 未提高，归因 fusion；若 final-10
提高但 EM 不提高，归因 final reader，而不是继续改 decomposer。

### 8.1 主比较

- `C-B` pooled（n=90）ITT EM 至少 `+5pp`，即净增正确题至少 `5`；
- `C-B` pooled F1 `> 0`；
- 每数据集 `gained-lost >= -1`；
- parse count 不下降；
- a1-ineligible fallback子集C/B prediction逐字节一致；a1-eligible但dynamic-invalid子集只验证回退Q并
  留在ITT，不要求与B一致。a1-admissible子集属于treatment后诊断，只报告，不参与晋级门。

### 8.2 次比较

- `C-A` pooled EM 至少 `+5pp` 且 F1 为正；该门只决定是否同时宣称端到端系统收益，不参与 C-B
  动态机制是否进入 prospective validation 的判定；
- 分数据集如实报告，不以 pooled 正值掩盖单数据集明显退化；
- paired bootstrap CI、McNemar、gained/lost/tied全部输出，但小样本 CI 不作为唯一晋级门。

失败时保留结果并停止，不改阈值、不换题、不在同一 development90 上继续手调。

## 9. Phase 5：fresh prospective validation 与主表口径

只有 Phase 3 全过、Phase 4 的 C-B ITT outcome 门与 evidence mechanism 门同时通过，才可经研究者再次
批准，一次性打开已封存的 prospective-validation300×3：

- 所有协议、模型、query/parser/merge和预算不变；
- 主门：dataset-stratified `C-B EM >= +5pp`、paired-bootstrap 95% CI 下界 > 0、F1方向为正、
  gained > lost，且三个数据集的点估计均不得为负；
- exact McNemar与分数据集CI同时报告。若点估计达门但CI含0，只能写“方向性结果”，不能写稳定优于；
- A/B/C 同一 qid、同一 scorer；报告每数据集、macro、pooled、query/token/latency成本；
- 现有 trace/IRCOT 数值只作历史系统参考；必须在同一 prospective-validation qid 上重跑其 frozen implementation，
  并明确其实际累计 passage/query预算，不能把旧 n300 数值当 matched causal control。

prospective validation 通过后，才允许在标准 canonical n300×3 上跑描述性系统表。由于历史 main qids
已被多次分析，该表不是独立确认集；本轮 validation 只能作为跨数据集前瞻验证，canonical n300 作为与旧
baseline 对齐的系统结果。

需要强调统计能力：n=300/数据集适合一周内完成跨数据集方向与 pooled 效应验证，但未必足以让每个
数据集的 5pp 差异都达到显著。因此本协议不在看完 development 后改变 n，也不把 n300 称为最终论文
确认。若容量审计已成功封存独立 `paper-confirmation-reserve1000×3`，则在本轮结果完成后另行批准是否按
完全相同协议一次性打开；n1000 与 n300 cohort 不合并、不得选择性扩样。若只做到 n300，论文必须明确
“prospective validation / per-dataset underpowered”，不能把不显著写成等效。

## 10. Phase 6：只有零训练机制成立后才进入训练

### 10.1 Controller-SFT

训练样本统一成三个任务：

1. `state -> next single-hop query`；
2. `query + retrieved docs -> concise subanswer / NO_RELEVANT_ANSWER`；
3. `Q + verified (q,a,docs) history -> final answer/reasoning`。

来源只使用 train split：Hotpot supporting-fact chains、MuSiQue decomposition、2Wiki ProofKG/Gold-train
evidence；保留 provenance 和 source type。先做 500–1000 条小样本 headroom 门，再决定是否扩到 5k–8k。
起点为当前强 SFT，保留一部分原 final-QA replay，防止 controller 格式学习破坏最终回答能力。

### 10.2 PPO

PPO 必须把 search/observation 放进 on-policy rollout，而不是先固定 passages 再一次性回答：

- 环境返回的 passage tokens 从 policy loss 和 KL 中 mask；
- `PPO-O`：最终答案 outcome reward + 固定 action-validity/成本项；
- `PPO-K`：与 PPO-O 完全相同，只在 eligible 证据图上增加 alpha-gated KG/process reward；
- `PPO-K - PPO-O` 才能归因 KG reward，`PPO-O - controller-SFT` 解释 RL outcome 学习；
- Hotpot/MuSiQue没有经过验证的KG process边时alpha必须为0，不能把跨数据集迁移称为直接KG监督；
- 新动态rollout先做 K=4 零更新 headroom/rankability：valid>=.90、oracle-greedy>=5pp、reward-selected
  gain>=2pp；失败则不启动正式 PPO。

已有 trace/IRCOT baseline 意味着“多轮检索”本身不是本文创新。它应成为共同 controller backbone；本文
贡献需由同资源、同 controller 下的 SFT/PPO-O/PPO-K 与 legacy/ProofKG供给消融体现。

## 11. 工程文件规划

拟新增，不覆盖 v7：

- `scripts/diagnose/audit_v7_subanswer_contract_counterfactual.py`
- `scripts/prepare/freeze_dynamic_decomposition_v8.py`
- `kgproweight/retrieval/dynamic_decomposition_v8.py`
- `scripts/pilot/materialize_dynamic_decomposition_v8.py`
- `scripts/prepare/finalize_dynamic_decomposition_v8.py`
- `scripts/eval/evaluate_dynamic_decomposition_v8.py`
- `tests/test_v7_subanswer_contract_counterfactual.py`
- `tests/test_dynamic_decomposition_v8.py`
- `outputs/audits/dynamic_decomposition_v8_*/protocol.json|manifest.json`

复用但不修改其历史语义：Wiki18加载与RRF、BGE reranker、强SFT loader、现有EM/F1 scorer、v7的Gold递归
禁字段和模型/资产tree-hash工具。不得覆盖任何v4–v7 artifact或历史baseline。

截至2026-09-04，已实现`freeze_dynamic_decomposition_v8.py`、
`dynamic_decomposition_v8.py`、`dynamic_decomposition_v8_cohort.py`及对应测试。纯核心已覆盖一行query、
短subanswer/sentinel、系统绑定provenance、static/dynamic action选择和`6+2+2`合并；
development loader只能读取已锁定的90题，默认拒绝prospective。这些是Gold-free工程实现，
不是方法效果。两调用q2协议已获批准，`materialize` runner与call ledger正在实现；
`finalize/evaluate`仍须在Gold-free机制门通过后才可进入。

实际运行结果覆盖上述“仍须”表述：implementation V2、工程smoke和development90均已完成；development
在Gold前失败并正式停止。正式报告把3次顶层backend batch误命名为3次`full_index_passes`；底层按64-query
分块，90/87/62三个阶段实际memmap遍历为`2+2+1=5`。该性能遥测更正已append-only记录于
`outputs/audits/subquestion_decomposition_v8_development90_gold_free_failure_diagnosis_v1/`，不影响检索输出、
其他机制指标或FAIL_STOP结论。

## 12. 执行顺序、成本与停止点

1. Phase 0：CPU-only，0 GPU/0 retrieval；先得到接口上界。
2. baseline-budget append-only audit：CPU-only；修正比较口径，不改历史分数。
3. capacity audit + development/prospective-validation/paper-confirmation-reserve 预冻结：CPU-only，0 Gold。
4. 实现和单测；用已消费/合成输入跑 n=4×3 smoke，先实测每题时间和显存，再给出可信总时长。
5. 本地4090运行 development30×3；先 Gold-free 物化，过门后才运行无Gold final reader，再由独立
   scorer读取Gold。
6. development失败即停止；成功后由研究者再次确认是否打开 prospective-validation300×3。
7. prospective validation通过后才准备 controller-SFT；SFT通过再准备 PPO，不提前占用远程大显存。

所有运行必须有唯一 Experiment ID、配置/代码/数据/model/index hash、seed、硬件与软件版本、日志、逐题
telemetry、report和manifest。任何失败均append-only保留。

## 13. 研究者授权状态

截至2026-09-04：

1. 已批准核心策略变量：`q2_static` 对比
   `q2_dynamic(Q,q1,verified a1,bound evidence/provenance)`；
2. 已批准B/C固定3次logical retrieval calls（共享root、共享q1、各自q2）以及唯一的
   `root top-6 + q1 top-2 + q2 top-2` 最终严格10篇预算；
3. 已批准新增Evaluation Protocol，并冻结development30×3与sealed prospective-validation300×3；
   本轮因2Wiki容量不足不设paper-confirmation-reserve1000×3；
4. 已批准Phase 3/4的门与两调用q2澄清；
5. 当前仅批准到本地零训练 development，不自动授权 prospective validation、paper confirmation、SFT、
   reward/loss或PPO修改。
