# SAEG Passage-QPEG hard-negative alignment v2：数据审计结果

日期：2026-09-03  
Experiment ID：`SAEG-P-HARD-NEGATIVE-ALIGNMENT-V2-SEED42`  
最终状态：`FAIL_STOP_DATA_GATES`（未采样最终SFT日程、未训练）

## 1. 要回答的问题

SAEG-v1 development零训练效用中，加入自动Passage-QPEG后，strong SFT的macro EM/F1分别下降
6.67/6.63pp，逐题gained/lost=1/11；同时known passage citation rate为0.7533。需要判断：

1. P块是否被模型读取；
2. P块是否经常只含局部相关但不构成完整答案链的证据；
3. 能否直接用evaluation同分布的自动P构造complete/partial/misleading hard-negative课程。

## 2. 冻结设计与执行

- 复用QPEG-v4冻结的train-only 600×3问题身份；
- 使用与evaluation相同的检索链：E5@100 + BM25@100 → RRF@50 → bge reranker@10 → pack3860；
- 使用相同selector：threshold=0.77，top-4；检索和选择阶段`gold_access=false`；
- 自动P生成后，才用raw-train Gold supporting facts/decomposition做后验诊断与SFT target；
- exact质量类：complete=覆盖全部必需支撑单元；partial=覆盖部分；misleading=P非空但0覆盖；empty=P为空；
- target只允许引用匹配的P边，任何未匹配边即使在输入可见也不得引用；loss/reward不变。

冻结后复核发现train1800与后来扩展的SAEG全评估集合存在19个family交叉、0个dataset/qid交叉。
append-only addendum排除这些family后，有效候选为1781：Hotpot 600、2Wiki 582、MuSiQue 599。

本地检索最终1800/1800成功，三数据集各600，全部10 passages，Gold/forbidden字段为0。第一次运行因
预完成代码审计发现完整性布尔谓词bug而中止，0行数据、0模型更新；失败目录和addendum均保留，修复后
第二次运行成功。

## 3. 冻结exact门结果

| 数据集 | complete | partial | misleading | empty | selected-edge exact precision | required-hop exact recall | full-context exact recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0 | 48 | 455 | 97 | 3.20% | 3.47% | 16.55% |
| 2Wiki | 0 | 9 | 497 | 76 | 0.64% | 0.56% | 5.96% |
| MuSiQue | 0 | 13 | 586 | 0 | 0.54% | 0.93% | 6.84% |

三个数据集均未达到预注册`complete≥20`门，因此状态为`FAIL_STOP_DATA_GATES`。输出仍完整保留，但不生成
最终SFT schedule，也不启动模型更新。

## 4. 为什么exact结果不能解释为“几乎所有句子事实都错”

人工抽查发现，Wiki18 retrieval文本相对raw dataset support存在标题重复、引号、标点和小幅版本差异。
例如raw support的`Lake Managua ... is a lake in Nicaragua`在retrieval中以
`Lake Managua Lake Managua ... is a lake in Nicaragua`出现；它在语义上是同一事实，却未通过exact匹配。

因此另做了明确标记为post-failure diagnostic的保守近精确分析：标题token序列必须相同，去掉句首重复标题，
之后multiset token-F1≥0.90。它不证明一般语义蕴含、不改exact label，也不反转失败门。

| 数据集 | complete | partial | misleading | empty | selected-edge near-exact precision | required-hop near-exact recall |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 26 | 208 | 269 | 97 | 18.09% | 19.24% |
| 2Wiki | 26 | 251 | 229 | 76 | 23.71% | 20.41% |
| MuSiQue | 2 | 182 | 415 | 0 | 7.80% | 13.32% |

近精确修正确认exact统计高估了失败程度，但核心结论不变：完整P只占Hotpot 4.3%、2Wiki 4.5%、
MuSiQue 0.3%；partial/misleading占绝大多数，尤其MuSiQue供给质量很差。

## 5. 对用户问题的准确回答

问题主要出在Passage-QPEG证据质量，但应区分三种情况：

1. **事实本身局部真实，但缺一跳**：最常见；模型沿局部证据回答错误候选。
2. **同名实体/相邻主题被选中**：例如地点、影视作品、人物的错误消歧，属于实际误导。
3. **文本版本差异**：事实可用，但exact审计误判；near-exact诊断已单独量化。

因此不能写成“P句子几乎全是假事实”；可以有证据地写成：当前answer-free selector输出的P大多不是完整
多跳证明，其中大量是partial或misaligned evidence；strong SFT会读取这些P，因而产生显著负效用。

## 6. 为什么当前候选不能直接训练

只用自动P候选时，三数据集几乎没有complete正例。直接训练会把目标退化为“看到P就忽略”，无法学习
内容条件化的选择性信任，也不能证明KG/graph reasoning有效。尤其MuSiQue只有2个near-exact complete，
不具备独立学习“好P要用”的正例容量。

## 7. 建议的新单变量版本（待批准）

`P paired hard-negative curriculum v2.1`：

- 正例：保留train-only Gold-complete P轨迹，教模型在完整证据下引用并推理；
- 硬负例：同一训练问题的canonical retrieval + automatic P，partial只引用正确子集，misleading完全不引用；
- replay：保留no-P样本，约束旧能力；
- 不改模型、loss、reward、evaluation输入或strong-SFT起点；
- 先冻结配对身份、精确配额和SFT深度，再跑一次小规模continued-SFT；
- development只复用已消费的150题，按已冻结interaction/no-P retention门选最早checkpoint；门通过后才请求
  打开sealed confirmation。

该版本预计首先把P造成的`−6.7pp`伤害拉回接近0；它不能单独保证P相对no-P产生正增益。要获得稳定正增益，
还需把当前句分类selector升级为hop-aware evidence-set selector，直接优化必需跳覆盖，而不是独立句相关性。

## 8. 可追溯产物

- 冻结协议：`outputs/audits/saeg_p_hard_negative_alignment_v2_protocol/`；
- 隔离addendum：`outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum/`；
- 成功retrieval：`outputs/audits/saeg_p_alignment_v2_train1800_retrieval/`；
- 工程失败保留：`outputs/audits/saeg_p_alignment_v2_train1800_retrieval_aborted_integrity_gate_bug_v1/`；
- exact候选与报告：`data/silver_data/saeg_p_alignment_v2_train1781_candidates_seed42/`；
- near-exact诊断：`outputs/audits/saeg_p_alignment_v2_near_exact_diagnostic_v1/`；
- 实现：`scripts/prepare/freeze_saeg_p_hard_negative_alignment_v2.py`、
  `scripts/prepare/materialize_saeg_p_alignment_v2_train_retrieval.py`、
  `scripts/prepare/build_saeg_p_alignment_v2_candidates.py`、
  `scripts/diagnose/audit_saeg_p_alignment_v2_near_exact.py`；
- 测试：`tests/test_saeg_p_alignment_v2.py`（5项）。
