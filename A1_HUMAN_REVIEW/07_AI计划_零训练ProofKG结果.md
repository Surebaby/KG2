# AI-plan 零训练 Proof-KG 工程 Pilot

日期：2026-08-29  
Experiment ID：`query_planner_v2_ai_plan_proofkg_n30_seed20260829`  
正式状态：`PASS_AI_ENGINEERING_ONLY`

## 结论

30题的AI预审计划已经实际执行，而不只是检查格式。运行时仅使用问题、AI给出的anchor
QID、PID依赖图和当前Wikidata属性；`answer`、2Wiki evidence和supporting facts均未进入
构建。全部KG与执行详情先写盘并锁定SHA256，之后才加载2Wiki evidence做覆盖审计。

预注册工程门全部通过：

| 指标 | 实测 | 门槛 |
|---|---:|---:|
| anchor QID非空 | 30/30 = 100% | ≥90% |
| proof KG非空 | 30/30 = 100% | ≥90% |
| plan完整执行 | 29/30 = 96.7% | ≥70% |
| 完整relation+value链 | 25/30 = 83.3% | ≥65% |
| 运行错误 | 0 | =0 |

AI plan的62个PID步骤与2Wiki后置reference PID共62/62一致。每题平均生成2.2条有目标关系的
三元组。相较旧通用邻域KG，这说明“按题QID→目标PID→tail QID继续下一跳”的路径确实能
生成高密度proof KG，不需要把整个Wikidata索引或模型同时装入显存。

## 剩余5题

- `A1-007`：当前值为`Matilda of Saxony`，reference tail为`Matilda Billung`，而reference
  下一跳head又是`Matilda of Saxony`。两步关系都取到，属于数据集快照内部可见的别名衔接。
- `A1-011`：当前值为`The Notorious B.I.G.`，reference tail为`Christopher Wallace`，
  下一跳head为`The Notorious B.I.G.`；出生日期正确取到。属于数据集内部可见的别名衔接。
- `A1-012`：当前值为`Princess Isabella, Duchess of Genoa`，reference使用
  `Princess Isabella of Bavaria`；下一跳父亲值正确。很可能是同实体title/alias差异，
  但未做人审，保持为`UNKNOWN_REQUIRES_SOURCE_REVIEW`。
- `A1-023`：当前producer值为`Ilaiyaraaja`，reference tail写`Raja`，reference下一跳head
  为`Ilaiyaraaja`；两位producer的P27均为India。属于数据集内部可见的别名衔接。
- `A1-028`：第一跳父亲QID与值存在，但当前Wikidata没有第二跳P27。这是唯一实际property
  缺失，不能通过增大top-k修复；需要版本化历史快照、Wikipedia文本证据或缺失时abstain。

以上诊断不修改预注册评估，也不把25/30事后上调。严格指标继续报告83.3%；alias-aware
数值只能在另行冻结Evaluation protocol并完成来源/人工核验后计算。

## 证据与边界

- 预注册协议：
  `outputs/audits/query_planner_v2_ai_plan_proofkg_preregistration_20260829/protocol.json`
- 完整运行结果：
  `outputs/audits/query_planner_v2_ai_plan_proofkg_n30_seed20260829/report.json`
- runtime KG SHA256：
  `0013ede3fb9a1460c0477f2858fb1ef1a2d71d5eab196b600c645995239a29e0`
- runtime details SHA256：
  `6384043f7b85daffd2a64520ccc32f2eab7806f72ff4f0d853d0763e21d4b181`
- post-build audit SHA256：
  `97cfebbe7db15e46b17780e368fc838667a43c846d1e693f5fd71211d79303e4`

这仍是30题、AI-authored plan、已查看过的工程pilot，不是人工Gold、独立确认或三数据集结论。
通过只允许设计一个另行冻结的零训练模型利用率对照；不授权替换正式KG、修改Reward/Loss、
训练planner或重训PPO。HotpotQA和MuSiQue的可执行规划仍未由本结果解决。

## 下一步

先做零训练、同qid、同passages的利用率对照：固定现有SFT和hybrid PPO checkpoint，只替换
KG输入为`legacy`与本proof KG，报告配对EM/F1、证据可见/隐藏分层和冲突题。该对照需要GPU
推理，但不需要训练；96GB显存足够，因为属性cache在CPU/磁盘预先构建，prompt中每题最多
只注入12条三元组。只有模型利用率无退化且有配对收益，才值得另冻更大的跨数据集结构确认
与新的PPO smoke；否则先解决planner/知识源，不重训。
