# Source-credit v1 修复与真实校准记录

2026-09-05；研究者授权“可以按照你的想法进行修复”。本轮是 format-v2 与 source-credit-v1 的组合工程修复，PPO 参数更新为 0。目标仍为答案奖励、过程奖励与 α 全部启用的 PPO-A；F/T 为同输入、同信用门、同预算对照。

当前结论：格式边界与图信用误发已修复，真实 ReaRAG 评分和 α 校准已完成；已有过程排序诊断没有显示 learned α 胜过 fixed α，不能把校准通过写成方法收益，也不能据此直接启动 full PPO。

## 1. 修复范围与数据边界

- 原 3000 题、12000 条 schedule、2000 条 replay、830 题 × K2 的 1660 条生成均保留。Gold、Reader prompt/passages/KG、旧评价协议未修改。
- format-v2 在训练奖励入口检查唯一 Final Answer 标记之后确有字母或数字，拒绝空答案、纯标点；v1 校验和 canonical 答案解析保持历史行为。1660 条中仅原来误判为有效的 1 条空 Final 改为无效：有效数 1472 → **1471（88.61%）**。没有通过丢弃失败候选提高有效率。
- Gold-free 来源复核发现历史实体名投票和 root projection 存在身份混淆。800 个有图问题：**PASS 671、UNVERIFIED 100、FAIL 29**。FAIL 为可确认的不同 QID 显示名折叠；UNVERIFIED 表示本地证据不足，不能称为 100 个已确认事实错误。PASS 也仅表示通过当前身份/来源检查，不等于人工确认推理语义正确。
- 本版修复的是**奖励信用**：旧图资格门与冻结来源 PASS 取交集。129 个未通过图的题仍保留输入和答案/Text 奖励，但令 α_eff=0、Graph reward=0。未知题、变化的 record/question/可见图、缺失绑定均关闭图信用。
- 没有声称旧输入 KG 已修好。新 artifact 特意区分 `source_credit_clearance=true` 与 `source_integrity_clearance=false`；信用门不能认证原始 Reader 输入的事实完整性。

来源证据与保守检查边界见 [来源分歧复核](sourcegate_source_disagreement_review_20260905_v1.md)。原代码先保存于 `outputs/audits/sourcegate_format_v2_parent_code_snapshot_20260905_v1/`；新评分、mask、校准、诊断独立存放，旧失败记录保留。

## 2. 已执行的真实评分与校准

本地 RTX4090 使用原 BF16 ReaRAG，未量化或截断上下文；49 个模型/tokenizer 文件完成 SHA 核验。最长评分输入 2589 tokens，小于 4096 预算。1660 条评分完整结束，1471 条有效轨迹共有 **4517 个真实 Text step scores**；所有无效候选也保留。

先生成 format-only 诊断校准，再在冻结来源信用 mask 下拟合新 gate；format-only artifact 不用于当前 PPO。新门保留原四维特征、启发式 ratio target、seed42、family split 与既定校准门槛。仅 train split 拟合权重、归一化及 fixed α；Gold 仅用于拟合结束后单独存放的结果诊断。

| split，非 abstain 样本 | Brier | train constant 的 Brier | R² vs train constant |
|---|---:|---:|---:|
| train，683 | 0.021771 | 0.026478 | 0.1778 |
| calibration，242 | 0.025740 | 0.031158 | 0.1739 |
| internal confirmation，277 | 0.020446 | 0.030711 | 0.3343 |

**既定启发式校准门 9/9 通过。** 这只说明门能预测所定义的 ratio 标签；不是独立来源可靠性或 EM/F1 提升证据。原版与信用屏蔽后的拟合人群不同，不能将两个版本的指标直接当作同条件改进。

有效且获图信用的 1202 条候选中，learned α 均值 0.3988、标准差 0.0714、范围 0.0219–0.4833；fixed α 为 train 均值 **0.4021112662**。其余 269 条有效候选的实际 A/F Graph reward 全为 0。

## 3. 当前实际奖励与配置

合法轨迹的 token reward 累计（不含 PPO 的 KL penalty）：

`R = 4 × (max_alias EM + 0.1 × max_alias F1) + 0.30 × (1 − α_eff) × T_N + 0.20 × α_eff × G_N`

`G_N = clip((raw_G − 0.5716556857) / 0.2929044118, −1, 1)`

`T_N = mean_t clip((raw_T_t − 0.7167349010) / 0.1025195960, −1, 1)`

非法轨迹总任务奖励 −4。Text 奖励按步末发放并除以步数，Graph 与 outcome 放在最后 response token。T 使用 α_eff=0，F 使用合格 mask×0.4021112662，A 使用合格 mask×frozen learned α。A/F/T 共用原 Proof v2.3、passage-only ReaRAG、归一化与格式判定；来源信用被取消不改变该题原来的最小格式步数要求。

四个数值特征仍为 graph density、固定来源 confidence、cite-any、cite-match。本轮没有依据已看过的 confirmation 或 EM 标签反复扩特征/调权重。原来源门之外新增有证据绑定的硬信用门；并明确保留 learned α 区分能力不足这一结果。

当前配置为 `configs/training/phase3_ppo_mixed4_source_credit_v1_{a,f,t}_{probe,smoke,full}_seed42.yaml`，9 份均经过 actual CLI 解析核对。共同设置：lr=1e-6，batch=4，mini-batch=1，PPO epochs=2，γ=1，λ=.99，KL coefficient=.25、target=8，max_new_tokens=384，显式 Strong SFT reference，10% replay、anchor=.1。阶段预算为 12/600/12000 scheduled trajectories；配置存在不代表训练已执行。TensorBoard 沿用独立 run 目录及 AutoDL `/root/tf-logs` 入口。

新配置显式要求 `source_gate_format_version=v2`、`source_gate_credit_version=v1`，绑定新 schema 的 gate 与 mask。旧模式默认不变；新旧 artifact 混用、mask/代码/证据 hash 变化或未获对应 clearance 均显式报错，不静默降级成未校准门。

## 4. 独立过程效用诊断：正向信号和局限

同题 K2 中，一条 EM 命中、一条未命中且两条均合法、均获图信用，共 **114 对**。以下比较只使用过程分，排除答案 outcome；平局计半。

| 评分 | 正确排序比例 |
|---|---:|
| Graph 完整原始分，仅诊断 | 90.79% |
| Graph 结构部分，仅诊断 | 47.81% |
| ReaRAG 原始 step mean，仅诊断 | 55.26% |
| 实际 T 过程奖励 | 53.51% |
| 实际 A 过程奖励 | **83.33%** |
| 实际 F 过程奖励 | **83.33%** |

A−F 仅 **1 对改善、1 对变差、112 对不变**；94/114 对四个数值特征完全相同。全银行 143 对 EM 命中/未命中双有效候选，A=F 为 74.13%，T 为 50.35%。这支持图通道具有候选排序信号，但没有证明 learned α 优于固定门；Graph 的强信号主要包含图推导答案一致性，不能当作逐步推理语义已验证。

Text 的 **2256/4517=49.94%** steps 被裁剪到边界。当前统计按轨迹 raw mean 拟合，却逐 step 应用；这是原先明定且此次未改的合同，但饱和与 graph-heavy 采样会压缩文本分辨率。新增来源门没有解决此问题。不能在已看过的 confirmation 上寻找最优 scale 后仍宣称独立确认；后续必须建立新的 train-only 归一化/代表性普通样本研究版本，另留未消费的确认集。

整个银行为训练候选诊断，不是 canonical baseline；原生成答案 EM/F1 仍是 **55.12%/61.54%**，工程修复没有生成新的更高分模型。当前 valid 88.61% 也未满足 todo 中建议的 90% 候选健康线；尚缺独立过程确认、完整 reward-top1/oracle 及引用操纵反事实，未签发 PPO 效用放行。

## 5. 验证与可追溯产物

16 个相关测试文件合计 **341 tests 通过**，覆盖格式兼容、Gold-free 来源判定、mask 防篡改、A/F/T runtime、真实分数适配、校准、过程诊断、PPO 配置传递和 TensorBoard。

- cached-real-ReaRAG 检查：32 条候选 × A/F/T，**96/96通过**；独立逐token奖励最大误差 2.22×10⁻⁷。绑定输入与真实tokenizer验证通过；首次审计器自身的parent行比较错误及修订记录独立保留，未修改生产奖励或原评分。
- 正式 PPO loader 全人群检查：**9/9通过**；3000题确有671题保留图信用；830个bank成员的query、完整record、可见KG、prompt、passages与正式PPO装配全部一致，身份不匹配为0。其余2170普通题均无图信用。12000 schedule中2684条可获图信用、9316条无图信用；K4和每题10passages不变。
- 这两项均不加载policy/critic、没有优化器更新，也不是GPU PPO probe。本轮改动在本地完成，旧远端release不会自动包含本版代码和校准产物。

| 产物 | 当前权威路径 |
|---|---|
| 真实 ReaRAG 分数 | `outputs/audits/source_quality_candidates_format_v2_scored_local_seed42/` |
| Gold-free 来源复核 | `outputs/audits/sourcegate_source_disagreement_review_20260905_v1/integrity_clearance_v2/` |
| 冻结信用 mask | `outputs/audits/source_credit_mask_v1_local_seed42/manifest.json` |
| **当前 α gate** | `outputs/calibration/source_credit_gate_v1_local_seed42/gate.json` |
| 校准、分割、来源与源码绑定 | `outputs/calibration/source_credit_gate_v1_local_seed42/manifest.json` |
| 过程效用诊断 | `outputs/audits/source_credit_reward_utility_v1_local_seed42/report.json` |
| A/F/T 配置复核 | `outputs/audits/source_credit_v1_config_revision_20260905_v1/report.json` |
| runtime 回归 | `outputs/audits/sourcegate_source_disagreement_review_20260905_v1/runtime_wiring/report.json` |
| cached-real-ReaRAG token 核对 | `outputs/audits/source_credit_runtime_cached_v1_local_seed42/` |
| PPO loader 全人群信用核对 | `outputs/audits/source_credit_v1_ppo_population_check_20260905_v1/` |
| 整体修复交付快照与绑定 | `outputs/audits/source_credit_v1_repair_release_20260905_v1/manifest.json` |

下一阶段先处理文本尺度与 ordinary 分布、α 同题区分及格式有效率，再依据预先冻结的规则进行独立效用确认和小规模 GPU probe。不能把这次 9/9 ratio 拟合通过当作 full12000 放行。A−F 是当前信用门下的 learned α 对照；如果论文要归因来源信用屏蔽本身，需另设固定其余条件的旧信用门对照。
