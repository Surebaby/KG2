# Source trajectory features v2：train-only 架构审查

Experiment ID：`SOURCE-TRAJECTORY-FEATURES-V2-TRAIN-CAPACITY-20260905-V1`。本记录只设计并检查 α 的来源引用表示，不拟合 gate，不启动 PPO，不修改旧候选、Gold、KG、scorer、baseline 或 evaluation protocol。

## 冻结的单一方案

模块为 `kgproweight/reward/source_trajectory_features_v2.py`，版本 `source-quality-trajectory-features-v2`。保留四个 v1 输入 `density`、`link_confidence`、`cite_any`、`cite_match`，新增两个 [0,1] 输入：

| 名称 | 定义 | 能观察到的差异 |
|---|---|---|
| `source_edge_coverage` | 全轨迹唯一匹配可见 KG triple 数 / 唯一可见 KG triple 数；无图为 0 | 漏引可见边，即使旧 `cite_match` 仍为 1 |
| `min_step_citation_precision` | 每步先计算 unique matched / (unique cited + unique unknown surfaces + malformed indicator)，无引用为 0，再取所有步骤的最小值；无步骤为 0 | 同一全局引用集合被集中在部分步骤，其他步骤没有引用；或某步混入未知来源 |

`compute_gate_features_v2(spec, steps, proof_result)` 沿用原 hard gate，调用者随后必须应用冻结的 `FrozenSourceCreditMask`。新增表示不能自行给予任何问题 Graph 信用。仅从 proof result 读取显式 `scorer_version`；空 proof 保留原 pre-score 契约。不得读取 score、components、derived answer、prediction、answer consistency、ReaRAG scalar 或 Gold。此模块也不读取 Reasoning、Conclusion、passages 或 live policy entropy。

两个新增量仅作为可学习的结构 proxy，不手设正向奖励，不假定“更多引用必然更可信”。独立的 feature version/schema 必须与六维 artifact 绑定；原四维及其 artifact 保留，用于 norm-only 对照。归一化与特征一起变化的实验应标记组合实验。

## Train-only 表示容量证据

读取冻结 family assignment 的 `candidate_id` 与 `split`，仅对 train 候选计算表示；不使用 ratio target、Graph/Text scalar、EM、F1 或 Gold，也不读取 calibration/confirmation 候选内容。设计选择只用下面这一次 train 容量审计，不根据已消费的 holdout 再挑变体。

Train 有 481 题 / 962 候选；source PASS 有 382 题 / 764 候选，其中 format-valid 为 376 题 / 683 候选。两条候选同时有效的 source PASS 题为 307 题。

| 表示 | 307 个有效 PASS 同题 K2 对完全相同 | 全部 481 个 train K2 对完全相同 |
|---|---:|---:|
| 原四维 | 251 / 307 = 81.76% | 375 / 481 = 77.96% |
| 四维 + edge coverage | 196 / 307 = 63.84% | 276 / 481 = 57.38% |
| **最终六维** | **160 / 307 = 52.12%** | **214 / 481 = 44.49%** |
| 七维探索方案，未采纳 | 52 / 307 = 16.94% | 72 / 481 = 14.97% |

未采纳的七维是在旧四维上增加引用步骤比例、unique/occurrence 引用 novelty、逐步引用 occurrence 分布的归一化熵。这些变量能区分更多候选，但重复合法综合步骤就会改变值，均匀与新颖也没有可靠的来源可信度方向。非约束 logistic 权重可能利用长度和重复，不能保证 padding 没有收益。因此选择结构含义更直接且对合法引用重复严格不变的六维，而非选择容量最大的表示。

可复现产物位于 `outputs/audits/source_trajectory_features_v2_train_capacity_20260905_v1/`：`audit.py`、`report.json`、`train_features.jsonl`、`manifest.json`。源文件、冻结输入和 parser SHA 均绑定，运行开始与结束复核一致。

## 科学解释与未解决风险

六维将这组 train 有效候选的表示碰撞率从 81.76% 降到 52.12%，仅证明可观察引用结构的分辨率增加。它不是 α 更准确、排序更好或 PPO EM/F1 改善的证据。仍有超过一半的有效同题候选无法区分；如果答案或推理相反而 citations 相同，输出必须相同。

Proof v2.3 的 Graph score 包含 citation precision、hop coverage × order 和源答案一致性；其中答案项权重为 0.45。直接把这些已算出的组件、Graph scalar 或 Text scalar作为 gate 输入，会接近复现其 heuristic ratio target。六维没有读取这些结果，也未读取执行 hop coverage；但可见 edge coverage 仍与 Graph 结构部分相关，故 target 拟合只能解释为 heuristic distillation，不能当独立的来源可靠度验证。

逐步最小 citation precision 会把合法的无引用综合步骤记为 0；这是“该步没有显式 KG 引用”，不是“该步事实错误”。因此它不能成为未经验证的惩罚规则。所有输入边都照抄进每个步骤仍可得到高结构值，而自由文本可能错误。词面 Reasoning/passage overlap 也无法解决这一点：复制、否定与同词不同关系均容易欺骗它，本版本未引入该类语义伪验证。

独立反事实测试证明：重复 citation occurrence、重复完整合法步骤、重复所有步骤不会改变六维，因此相同 gate 的 α 不会仅因这类重复增加。该结论限于 α 表示；原 ReaRAG 步骤均值、格式约束、生成长度和总 reward 并未因此自动具有重复不变性。新加空引用步骤可降低最小值，也不能单凭非约束权重保证总 reward 单调下降。端到端 anti-padding 仍应在过程 reward 审计中单独验证。若重复的是 malformed 步骤，旧四维中的 cite_match 会按旧合同增加 malformed 计数；新增两维保持不变，但不得声称全部六维不变，也不应静默修订 v1 计数。

## 验证与下一步判定边界

`tests/test_source_trajectory_features_v2.py` 的 20 个 CPU 检查覆盖旧值与 hard gate 不变、重复不变、漏边/空引用步骤可分辨、未知与 malformed 引用计数、空图、source identity 失败、严格禁止读取 score/free text、Final/否定/Reasoning/passages 反事实保持不变及输入无副作用。

推荐固定同一 mask、训练族 split、初始化和拟合预算，对比旧四维 + norm-v2 与六维 + norm-v2。已消费的 calibration/confirmation 只能作为 development 再分析；新 fit 的较好指标不能自动签 PPO 放行。需要预先冻结新的、未消费的 confirmation，且来源 mask、归一化镜像、artifact 身份和真实 PPO loader 人口绑定均通过。这个特征版本没有修复输入 KG，也不放宽 671 / 800 的来源 Graph 信用人口。
