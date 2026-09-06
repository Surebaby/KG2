**KG-ProWeight：正式 PPO 前的方案与论文证据审阅**

审阅日期：2026-09-05。Review ID：PPO-ALPHA-SCIENTIFIC-REVIEW-20260905-V1。

本报告完整阅读 AGENTS.md、RESEARCH_WORKFLOW.md、docs/todo2.md，并交叉核对生产代码、数据 preflight、历史结果与日志。研究者已在本轮明确确认 v4 数据封版，当前状态为 **DATA_READY_PPO_NOT_STARTED**。本轮未修改生产代码、核心 reward/loss、数据集、baseline 或正式 evaluation protocol，未更新任何科研 checkpoint。新增的两个 CPU 诊断及其范围见文末。

**1. 总体判断与项目进度**

建议保留当前收窄后的研究方向：固定 Strong SFT Reader、固定证据输入、固定预算，用 PPO 研究图结构评分与文本评分的信用分配。暂不恢复 Controller、QPEG、SAEG 或 continued-SFT。数据准备已经足以推进下一阶段；现在最需要解决的是奖励的测量有效性、α 的研究含义和 token 级训练接线。

**不建议直接按当前 α/reward 提案进入正式 PPO。** 数据门通过不能替代奖励门；目前有已经复现的评分盲点和训练接线缺陷。把这些问题带入 12,000 条轨迹训练，可能得到更高的代理奖励，却无法判断模型是否真正改善推理。

| 层次 | 已核验的进度 | 论文边界 |
|---|---|---|
| Reader | Strong SFT：4,751 条 accepted Hotpot 轨迹，1 epoch；checkpoint 冻结 | 所有 PPO 臂的统一起点 |
| 数据 | H/W/M 各 1,000；Proof800，四类各 200；每题 10 passages；replay 2,000 | 无需重做封版数据 |
| 调度 | 3,000 groups × K4 = 12,000 scheduled trajectories；Graph 暴露 3,200 条 | 是已冻结采样计划，尚未完成 PPO 采样/训练 |
| 数据验门 | materializer 23/23、独立 preflight 46/46；qid/hash/family/replay/ledger 的相应隔离门通过 | 证明冻结协议定义下的数据完整性与隔离 |
| 工程底座 | explicit SFT reference、critic zero-init、replay、截断控制已有 smoke 经验 | 不能推出新 reward/α 的训练效用 |
| Proof scorer | v2.2 日期和多值处理修复已有静态证据 | v4 真实候选上的排序与语义可靠性尚未通过确认 |
| α 与正式配置 | successor checkpoint、T/F/A YAML/lock、零更新确认尚未完成 | 不可直接把旧 Proof400/text7200 或旧 α 接入 v4 |
| 主张 | Graph 供给有开发集正证据；历史 PPO 有局部正信号，也有迁移退化 | 新 v4 的 F−T、A−F、A−SFT 和 IHR 均未知 |

数据权威证据：[v4 preflight](/home/zjulab/kgpaper/outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_preflight/report.json:4)。本轮核对的 preflight SHA256 为 94876f0e…fbb2，其绑定的 data/replay report 和 manifest 哈希吻合。

实际实验链是：离线 planner/实体解析/闭包产出逐题冻结证据 → Strong SFT 一次生成完整 Step 轨迹 → outcome + 文本评分 + 图评分组成 PPO reward → 用新 α 调节两个评分通道的信用。α 发生在生成之后，不负责检索，也不改变测试时的输入。Phase2 三分类 PRM head 未进入当前 PPO reward path，因此不需要为本方案重训一个 8B PRM adapter。

已有结果需要同时保留正负两面：

| 已有试验 | 结果 | 正确解释 |
|---|---|---|
| Strong SFT，标准 legacy n=300/数据集 | H/W/M EM：.3833/.4267/.2467 | 冻结基线 |
| 2Wiki，固定相同 passages，SFT，development n=300 | legacy .4033 → Proof .6400；ΔEM +23.67pp，CI [17.0,30.3]pp | Graph 供给收益；不能作为 PPO/α 收益 |
| 历史 hybrid PPO，标准 legacy | H/W/M EM：.3600/.4233/.2433 | 工程稳定后仍未超过 SFT |
| 历史 ProofKG-PPO，family-disjoint train n=100，Proof 输入 | 相对 SFT +8pp，CI [2,14]pp；McNemar p=.02148 | 存在局部训练正信号，尚非新 v4 或 learned α 结论 |
| 同一历史 n=100 的 KG 利用率 interaction | DID +6pp，CI [−1,14]pp | 不能认定 interaction 已成立 |
| 历史 ProofKG-PPO，标准 legacy n=300 | H/W/M EM：.3433/.2967/.2433 | 供给迁移与遗忘需要认真控制，尤其 2Wiki |

证据：[供给层配对结果](/home/zjulab/kgpaper/outputs/audits/2wiki_matched_control_score_v2/matched_control_score.json:3)、[历史 PPO 局部正结果及禁止外推的边界](/home/zjulab/kgpaper/outputs/validation/proofkg_ppo_standard_retrieval_paired_n100_seed42_decision/report.json:3)、[历史 PPO legacy 2Wiki 分数](/home/zjulab/kgpaper/outputs/proofkg_ppo_pipeline_eval/2wikimultihopqa/seed_42/2wikimultihopqa_2026_09_01_01_41_kg_proweight/metric_score.json:1)。

**2. P0：Proof reward 目前能检查部分结构合规，不能充当完整语义 verifier**

当前 v2.2 的结构部分包括引用 precision、hop coverage、dependency order；结论 grounding 主要通过输出实体的词面命中判断，答案一致性还包含双向子串判断。可确定答案时，45% 权重直接来自图派生的最终答案一致性。这是轨迹级混合评分，不能直接称逐步事实正确性的机器证明。

本轮用完全合成的三跳链 Film Z → Person A → Person B → Country C 调用真实 scorer，得到：

| 合成干预 | v2.2 score |
|---|---:|
| 正常三跳轨迹 | 1.0 |
| 每个 Conclusion 改为 “It is false that …” | 1.0 |
| Reasoning 全换成无关胡话 | 1.0 |
| Reasoning 与 Conclusion 同时否定图事实 | 1.0 |
| Final Answer 改为 “not Country C” | 1.0 |
| Final Answer 改为 “Country C or Country D” | 1.0 |
| Final Answer 只写 “Country” | 1.0 |

这已经证明这些特定输入上的评分盲点。它**不证明实际 rollout 已经利用这些漏洞，也不估计其发生频率**。其中错误最终答案仍会失去 canonical gold outcome 奖励；“错误过程、正确最终答案”则尤其需要独立过程指标发现。

证据：[可复现合成诊断](/home/zjulab/kgpaper/outputs/audits/ppo_alpha_reward_readonly_review_20260905_v1/synthetic_probe_report.json:1)、[grounding 实现](/home/zjulab/kgpaper/kgproweight/reward/proofkg_process_v2_2.py:147)、[答案等价实现](/home/zjulab/kgpaper/kgproweight/reward/proofkg_process_v2_2.py:395)、[组合公式](/home/zjulab/kgpaper/kgproweight/reward/proofkg_process_v2_2.py:493)。

建议：

- 新建 scorer successor，修正可确定的否定、多答案和子串误判；规范化答案等价应有明确的 entity/date/alias 合同，对不可判定语义 abstain。只加一个否定词黑名单仍不足以解决普遍蕴含问题。
- 把“引用与路径结构质量”和“最终答案与图一致性”分别记录。零更新增加移除 answer-consistency 分量的对照，检验结构过程分自身是否有信息；是否保留其训练权重由新 reward 协议决定。
- 建立真实轨迹的派生反事实集：正确答案下破坏推理、保留引用但反转结论、交换关系/主体、遗漏必要 hop、堆砌所有图边。另含语义等价改写与引用顺序的正控制，避免只奖励模板。
- 不为绕过 scorer 问题改 Gold、删难题、放宽 hard gate。语义评分暂时不可靠时，应缩小“过程验证”主张或暂停该奖励进入正式 PPO。

日期修复的 45/45 也不代表 Graph 是完美 oracle：同一 Proof400 静态审计有 9 个 terminal singleton 的派生答案与 gold surface 不一致；整体可确定 360/400，其中 351 个对齐，40 个 abstain。该结果需要在 v4 下重新审阅，但不得按 gold 对齐结果筛选训练题。[静态审计](/home/zjulab/kgpaper/outputs/audits/proofkg_process_v2_2_static_audit_v3/report.json:86)

**3. P0：零更新门必须隔离 gold outcome，否则可能得到自证式通过**

提案总 reward 含 \(4(EM+0.1F1)\)。对合法轨迹，EM 正确时该项为 4.4；EM 错误时不超过 0.4。在候选过程权重 λT=.30、λG=.20 的尺度下，gold outcome 足以主导正确/错误排序。因此用总 reward 的 top1 EM 或 pairwise 正确率检验过程奖励，很容易把答案标签本身的效果归给 α。

训练使用 train-only gold outcome 是合理的；问题在于如何验证新增过程奖励。建议冻结同一个 candidate bank，分开报告：

| 诊断 | 用途 |
|---|---|
| outcome-only / 完整训练 reward | 检查训练接线、正确答案奖励是否正确；不作为过程有效性证据 |
| ReaRAG-only / Proof-only / fixed process mix / learned process mix | 测试不读取 benchmark gold 的排序能力 |
| 去掉 graph answer-consistency 的 mix | 检验结构/中间结论评分的独立作用 |
| 同一最终答案或同一 outcome 桶内排序 | 用独立过程标签区分推理质量；这时不能再仅用 EM 作标签 |
| 直接 ratio α / shuffled α / 固定 α | 测试 learned α 的增量与对应关系 |

主诊断应包括按问题聚合的 pairwise、ties、reward-top1−greedy、oracle@K、有效混合正误候选的问题数及配对 CI。不能把每题 K4 的所有配对或多个 steps 当独立样本。0.65 的 point estimate 也不能替代样本量和不确定性说明。

项目已经出现过类似警示：旧 hybrid 审计中 full-reward pairwise≈.9434，而 process-only≈.4717；greedy EM=.69，process-top1=.50。旧 scorer、候选池和协议都不同，此证据只能说明风险真实存在，不能直接判定新 v4 会失败。[历史审计](/home/zjulab/kgpaper/outputs/audits/ppo_reward_rankability_sft_hybrid_train_n100_k4_seed20260828_v1/summary.json:1)

**4. P0：gamma 与 lambda 实际逐 token 递推，末端 Graph 信用衰减过快**

生产 trainer 的 GAE 遍历每个 token，而非每个 reasoning step。zero-init critic 下，孤立末端奖励对前方 d 个 token 的原始 GAE 贡献为：

\[
\Delta A_{T-d}=(\gamma\lambda)^dR_T.
\]

本轮直接调用真实 compute_advantages 做单位脉冲验证：

| gamma / lambda | 提前 100 tokens | 提前 200 tokens |
|---|---:|---:|
| .95 / .95 | 0.00003505 | 0.000000001229 |
| 1 / .95 | 0.005921 | 0.00003505 |
| 1 / .99 | 0.3660 | 0.1340 |
| 1 / 1 | 1.0 | 1.0 |

该推导描述原始、白化前的末端奖励贡献；不等同于训练后的全部梯度，因为 critic、其他奖励和 advantage whitening 也起作用。但它与“Proof 只放 final token，希望改善早期多跳推理”的目标明显不协调。只把 gamma 改为 1 而保留 lambda=.95，仍然存在很短的 trace horizon。

建议把 **gamma=1、lambda=.99** 作为待验证的共同底座候选，先在冻结真实长度/奖励序列上重算 raw GAE、returns、各步骤信用；1/1 作为 Monte Carlo 诊断对照。随后用小 probe 判断方差和 critic 稳定性。上述候选未经正式 GPU 验证，不能当成已知最优超参数；若同时改 gamma/lambda，应登记为组合配置，或者按 gamma→lambda 顺序逐变量验证。

证据：[生产 GAE](/home/zjulab/kgpaper/kgproweight/training/step_reward_ppo_trainer.py:172)、[CPU 脉冲报告](/home/zjulab/kgpaper/outputs/audits/ppo_pretraining_cpu_review_20260905_v1/report.json:31)。GAE 的偏差/方差权衡依据已核验的[原始论文](https://arxiv.org/abs/1506.02438)；这里的具体风险判断来自本项目的 token 粒度和数值验证。

**5. P0：正式 successor 还应修复 EOS 与 rollout 模式接线**

EOS：Strong SFT tokenizer 的 PAD 与 EOS 相同；现有生成后处理按“最后一个非 PAD token”截断，会连真正 EOS 一并删除。合成输入 [7,8,EOS,EOS] 被截成 [7,8]，已用真实函数复现。停止动作因此不进入当前响应 loss/KL 区间，末端奖励也落到最后一个答案 token。

建议显式保留首个实际终止 token，只去除之后的 filler；覆盖左 padding、不同长度、无 EOS、达到长度上限和多个终止 ID。新路径遇到 response/reward/mask 长度不一致应 fail-hard，当前 min(...) scatter 与外层 padding/truncation 不应掩盖错误。

rollout 模式：在已安装 TRL 0.11.4 上，微型随机 CPU 模型执行一次 PPO step 后，policy.training=True；随后的项目 _generate 调用仍处于 True。no_grad 并不关闭 dropout，而 SFT LoRA dropout=.05。数值层面的真实模型 logprob 偏差尚待 runtime probe，但模式未固定的接线问题已复现。建议 rollout 临时 eval 并恢复原模式，检查生成分布、重算 old logprob 和 reference 使用同一确定的 dropout/temperature/top-p 合同。

证据：[生成与 EOS 代码](/home/zjulab/kgpaper/kgproweight/training/phase3_ppo.py:1732)、[长度 scatter](/home/zjulab/kgpaper/kgproweight/training/step_reward_ppo_trainer.py:141)、[模式与 EOS 诊断](/home/zjulab/kgpaper/outputs/audits/ppo_pretraining_cpu_review_20260905_v1/report.json:81)。

这些属于进入新 reward 试验前需要统一修复的底座，T/F/A 均应使用同一修复版本；不能只给 A 修复。

**6. P0：ReaRAG 通道的真实含义与输入合同需要明确**

当前 ReaRAG scorer 计算候选 step 的 mean NLL，再映射为 tanh((2.5−NLL)/1.5)。这首先衡量模型对该段文本的似然，不等于已经校准的事实支持概率。原始 ReaRAG 是用推理/动作轨迹做 SFT 的生成模型，原论文没有为本项目这个 NLL 映射提供过程 correctness 校准。[本地实现](/home/zjulab/kgpaper/kgproweight/reward/text_reward_model.py:72)、[ReaRAG 原论文 §3.3](https://arxiv.org/html/2503.21729v1)

而且 mixed scorer 的 prompt 明确传入了 spec.kg_subgraph。因此当前两个通道准确地说是“图结构/一致性规则分”和“全证据条件下的语言模型似然分”，不能默认当作两个独立的 Graph / Passage correctness 专家。[评分输入](/home/zjulab/kgpaper/kgproweight/training/reward_function.py:448)

建议在新 reward 协议中明确选择：

- 若研究问题是 **Graph 与 Passage 证据的互补**，文本 scorer 使用冻结 passages 与轨迹前缀，移除输入 KG；Reader 本身的输入在全部臂中保持一致。随后验证 passage 去除/置乱是否改变 scorer 的事实判别。
- 若保留当前全证据评分，论文就应准确称为 **结构评分与文本似然评分的混合**，并实证验证其互补性。

不要仅凭 ReaRAG 名称宣称 PRM 已可靠。独立反事实应控制长度和格式，检验否定、错误实体、无支持结论、改写，以及证据置乱的影响。

还有一个需诊断的输入预算差异：PPO max_input_length=6144，而 ReaRAG wrapper 默认 max_length=4096，超长时从 prompt 左侧截断；可能丢掉问题/早期 passages，保留尾部 KG/轨迹。应记录 scorer 实际可见的 question/passages/KG、tokenizer 和截断率，不能只统计 Reader prompt 长度。评分 prompt 当前还是拼接 message.content，不是 ReaRAG 自身 chat template；是否改用其模板也要以新版本离线验证，不能假定改模板必然更好。

**7. α：保留 hard gate，重新定义 soft target 的科学含义**

\[
\alpha_{\mathrm{eff}}=m_{\mathrm{graph}}\sigma(z/\tau)
\]

这个拆分值得保留：不可学习的 mask 负责身份/来源/执行可用性，learned gate 不重复学习“有没有图”。m=0 严格 α=0，PPO 期间 gate 参数冻结，且只在 eligible 集合上报告校准，都是正确边界。

但拟议的 \(q_G/(q_G+q_T+\epsilon)\) **目前只是启发式标签，不是已经识别的相对可靠性**：

- qG 包含引用合规和图答案一致性；qT 是似然映射。两者没有针对同一种独立质量事件进行校准。
- (qG,qT)=(.1,.1) 和 (.9,.9) 都约为 .5，无法区分“两边都差”与“两边都好”。分母为零的 abstain 只处理退化点，未解决两边都弱的情况。
- “可靠 scorer 对错误轨迹给低分”是有价值的负反馈。如果因为分低就降低该 scorer 的权重，可能把正确惩罚也关掉。需要区分 **被评分轨迹的质量** 与 **评分器判断是否可信**。
- eligible 全是 strict complete 图，completeness 在拟合集合中接近常数；剩余变化主要来自 policy 的引用/输出行为。图完整性也只相对 QueryPlan 成立，计划可能遗漏真正必要的关系。
- 在这种合成 target 上取得更低 Brier、更高 R² 或正确分桶单调性，只证明 gate 学会复现标签。它们不能单独证明可靠性或 PPO 效用。

我更建议把路线分两步：

1. **低成本可证伪阶段**：将 ratio 明确当作 heuristic，先与 constant、直接 ratio、shuffled α 在相同 bank 上比较独立过程质量与 process-only 排序。没有超越简单规则的证据，就不把“学习 α”定为论文必须成功的核心。
2. **若保留 learned reliability 核心**：在 family-disjoint 的 train-only 候选上建立独立过程判定，分别测量两个 scorer 对同一质量目标的误差/排序可靠性；再训练/校准 gate，并在未参与拟合的 reward confirmation 上验证。可用新增人工标注或经人工检验的冻结语义 judge，均需保留来源和协议。当前 ratio 不应未经这一步就被称为最优信任分配。

BCEWithLogitsLoss 支持 soft target 本身没有问题；问题是 target 的含义。若需要新标签或 loss，应先形成具体版本方案再由研究者批准。

**8. α 的另一个风险：冻结网络不等于冻结可优化激励**

设 \(G=\lambda_GR_{\mathrm{Proof}}\)，\(T=\lambda_TR_{\mathrm{Text}}\)，则：

\[
R_{\mathrm{proc}}=T+\alpha(x)(G-T).
\]

只给 G 平移常数 c，不改变 G 内部的候选排序，却给混合奖励新增 \(c\alpha(x)\)。α 依赖生成轨迹 x，因此这个项并非常数。策略即使不能更新 gate 参数，也可能通过引用格式、堆砌图边或改变文本风格提高 α。

当前方案只中心化 Text，而合法 Graph 分非负，必须检查这个标尺效应。连续 α 还有另一个隐蔽点：Text 的 EMA 基线乘以 (1−α(x)) 后变成 action-dependent 项，不能简单当作普通 policy-gradient baseline 被抵消。

建议：

- 冻结 train-only 的两通道 normalization、中心、尺度与温度，报告分布与不同来源的有效 advantage，而不仅是 raw reward share。可先测试离线固定统计，避免 α 演化与在线 EMA 同时引入漂移。
- 若保留 EMA，明确更新时序、初始化、跨臂重置、candidate bank 重排顺序；恢复训练时保存其状态，或明确禁止将重启当作同一条连续轨迹。当前实现明确未 checkpoint baseline。
- 零更新做 offset/scale stress 和 citation manipulation，观察独立质量不变时 reward/α 的变化。
- 新 entropy 特征若继承当前 policy 的 −mean(logp)，同一轨迹在不同 checkpoint 下可能有不同 α；冻结 gate 不能使这个 reward 完全固定。优先评估冻结 SFT reference surprisal，或先移除此特征。−mean(logp) 应称 sampled-token surprisal，不能称 semantic entropy。
- 用 evidence-only gate 作为廉价消融，检验收益是否确实需要 trajectory-conditioned 特征。α 均值增大和跨度变宽本身均非成功指标。

证据：[EMA 状态与更新](/home/zjulab/kgpaper/kgproweight/reward/composite_reward.py:193)、[所谓 entropy 实现](/home/zjulab/kgpaper/kgproweight/reward/alpha_gate.py:244)。优化代理奖励可能降低真实质量已有原始研究证据；该研究不能量化本项目风险，但支持保留独立质量评价的必要性。[Gao、Schulman、Hilton，2023](https://proceedings.mlr.press/v202/gao23h.html)

**9. 我建议的 PPO 共同配置与训练预算口径**

下表是审阅建议，**不是已批准或已冻结的配置**。优先保留有稳定性经验的设置，先解决信用与接线，不同时搜索大量超参数。

| 配置项 | 建议 | 冻结前检查 |
|---|---|---|
| policy/reference | 同一个 Strong SFT；reference 显式冻结 | 初始权重、logits/KL、adapter 可追溯 |
| learning rate | 先保留 1e−6 | 学习曲线与参数变化；不能从低 clipfrac 直接推断应加 LR |
| PPO epochs | 保留 2 | 每批真实 optimizer steps 与样本复用次数 |
| K | 保留 4 | 与零更新候选的 decode 保持一致 |
| batch/mini-batch | 先保留 4/1 跑 probe | 当前一批可能只有同题四轨迹；检查 all-correct/all-wrong、梯度方差 |
| gamma/lambda | 候选 1/.99，先做脉冲与真实长度验证 | 与旧 .95/.95 区分登记，不能冒称单变量收益 |
| KL | 初始 .25、controller target 8 暂保留 | 明确序列 KL、每 token KL、adaptive coefficient；target 8 不是硬停止门 |
| critic | zero-init、dropout=0 | 看白化前 returns/advantages、value loss、EV |
| decoding | temperature=1、top-p=1、top-k=0；rollout eval 模式 | 实际采样分布与 old logprob 重算一致 |
| 长度 | 384 先作为候选 cap；最多 5 steps | 在 v4 每数据集/题型重测截断；若不足，用同一新 cap 重建所有臂的比较 bank |
| validity | 三臂逐项相同 | 明确 ordinary min steps 与 eligible dynamic validity，检查所有实际 steps 后再判超限 |
| replay | clean v2 pool；保留目标 .10 和独立 CE coefficient | 分清 replay样本/PPO轨迹比例、CE权重、实际token/梯度预算 |
| reward | outcome 4×(EM+.1F1)、invalid −4 可作为起点 | Graph/Text 参数、mask、位置须经新协议审阅；不能沿用失配 scorer |
| 保存/选择 | 每600轨迹保存；最终只按冻结 validation 规则选 | 测试集和 confirmation 不参与追选 |

10% replay 按当前代码口径是 replay 样本数 / PPO trajectories，不是总优化样本中的10%，更不是 token 或 gradient 的10%。12,000 rollouts 若完整消费，名义上是3,000个 batch_size=4 的 trainer.step 调用；PPO epochs=2 不会变成24,000条新轨迹，但会重复优化。mini-batch=1、grad accumulation=1 时名义 optimizer 更新更多，并另有 replay CE 更新，实际计数应写进 resolved manifest，不能只写“训练12,000 steps”。必要时扩大 batch 中的不同问题数，以单变量 pilot 测试，不要把同题 K4 自动当作四个独立问题。

旧 YAML 继承链仍带旧 replay 路径、v2.1 scorer、旧健康门以及无 α 配置。因此新 T/F/A 必须输出完整 resolved config，绑定 v4 data、clean replay-v2、scorer successor、mask、α checkpoint、tokenizer/chat template 和代码 hash；只替换目录名不足以完成接线。

历史 combined/hybrid smoke 的 valid 均值约95.83%/96.33%，截断约1.33%/.83%，实际 replay ratio=.10；支持保留部分工程设置，但不是 v4 的性能证据。[日志再核算](/home/zjulab/kgpaper/outputs/audits/ppo_pretraining_cpu_review_20260905_v1/historical_smoke_summary.json:1)

**10. solid 论文所需的对照与统计**

T/F/A 必须保留，且全部使用同一输入、完整预算与配对种子：

- F−T：加入 Graph/Text 混合、同时降低 Text 权重的净效应；不能写成“仅增加 Graph bonus”。
- A−F：按轨迹变化的 α 相对固定 α 的贡献。
- A−SFT：完整 PPO 方法的净收益。

为了让对照足够有说服力，建议补充：

| 对照 | 优先级与预算定位 | 回答的问题 |
|---|---|---|
| PPO-O：outcome-only | 强烈建议；若声称过程奖励优于纯结果奖励，应正式同预算 | Text/Graph过程信号是否超越 outcome RL |
| 协议规定的 conditional-mean fixed α | T/F/A 主实验必需 | 控制平均初始混合比例 |
| validation 选优的固定 α | 先少量预注册候选 pilot，再冻结常数 | learned α 是否仅击败一个不合适的固定值 |
| 直接 ratio gate | 先零更新；如“学习 gate”是核心，应有匹配训练对照 | 已有 qG/qT 都已算出，为何还需要拟合 ratio |
| shuffled α | eligible内保留边际分布的零更新诊断；正式机制主张需相应训练证据 | 收益是否来自 α 与轨迹/证据的正确对应 |
| evidence-only gate / 去掉 surprisal | 小规模机制消融 | trajectory 特征是否增益，是否引入操控或漂移 |

零更新或短 pilot 不能替代正式同预算的因果对照。预算不足时应减少论文主张，而非把少训练的 F 或简单规则当成已被 A 公平击败。F 的初始 mean-matching 也不保证 PPO 演化后仍匹配 A 的均值/尺度，需记录 drift，不能事后重调 F。

建议把数据用途写成四层：α-train、α-validation、未参与调参的 reward-confirmation、最终模型 evaluation。α-train/validation 按 family 整组隔离；独立 reward-confirmation 应与 α 拟合和 PPO 训练按 qid/hash/family 隔离并纳入保护账本。训练池内的保留集只称 calibration/diagnostic，不冒称独立 confirmation；不能将受保护确认集回填到已冻结 v4，任何角色变更需另行批准。Proof800 只有728个 current families，K4仍然不能变成3,200个独立校准问题。v4 的 exact prompt/decode 应用于新 bank；旧400×K2条款已经过时。

两套 evaluation 输入都保留：标准 legacy 表检验与冻结基线同输入下的迁移；source-adaptive 表检验同一供给下的训练收益。不能只向 A 提供 ProofKG，也不能因为某表结果不理想就事后改主终点。若论文希望把 source-adaptive 设为主要内部因果场景，应在训练前形成新预注册说明，保留既有 canonical 表完整报告。

主要比较和最小有意义效应需预先指定。建议正式 T/F/A 至少3个配对 seeds，分别报告 seed 波动和按 qid/family 配对的 CI，不把同题多个 seed、候选或步骤当独立题。seed42 用于方向筛选的事实必须保留；后续 seeds 不能抹去这个选择过程。

n=300 可能不足以检出1–3pp的 α 改善。作为规划用近似，二分类配对差的 discordance=.10、双侧.05、80% power 时，MDE≈2.8√(.10/300)=5.1pp；具体样本量应由预注册 pilot 的 discordance 和聚类结构估计。重复使用过的 n300 仍可保留作 canonical development/reference；论文最终确认应新增未触碰题池与冻结规则，需研究者批准，不覆盖旧评估。

**11. IHR、资源说明和论文叙事**

当前 IHR judge 只见 question、gold answer 与孤立单步，不见 passages/KG；API 重试失败被返回 hallucination=False。helper 对空结果聚合为0，但实际评估脚本对无可解析steps的回答直接跳过，因此还存在模型间可评分coverage不同的问题。这不足以支撑强的“证据支持度/中间幻觉下降”结论，且失败率/覆盖率变化可能影响指标。[judge prompt](/home/zjulab/kgpaper/kgproweight/reward/ihr_judge.py:25)、[错误与聚合路径](/home/zjulab/kgpaper/kgproweight/reward/ihr_judge.py:123)、[评估脚本跳过空轨迹](/home/zjulab/kgpaper/scripts/eval/run_ihr_judge.py:96)

建议经批准另建 IHR protocol：对模型身份盲评，给同一冻结证据与必要推理前缀，区分 supported/contradicted/insufficient-evidence；API error、空轨迹、不可解析均单列 coverage。提供人工审阅子集与一致性/错误分析，同时报告轨迹长度、claim 数、有效回答率，避免少说话或无法评分被当成低幻觉。旧 IHR 保留，不能覆盖重算后当成同协议结果。

资源表应透明区分“当前目标问题 gold 未进入 prompt/图/trace”和“整个供给完全无监督”。v6 store 明确使用其他2Wiki train问题的官方 evidence annotations，经当前 qid/family 排除后保留5,887 rows、14,007 evidence hops，并叠加历史 Wikidata fallback。该流程的目标题隔离与本轮确认一致，但不能简称纯外部Wikidata、无人工监督。[store manifest](/home/zjulab/kgpaper/indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42/store_manifest.json:15)

内部 T/F/A 共享全部训练资源，可以做因果比较；与外部 checkpoint 使用同一推理输入，并不表示训练资源相同。还应报告 graph eligible率、失败题型与资源成本，不只报告筛出的800个成功图。词法 family 隔离不等于已经证明所有语义重述、共享实体/事实或外部模型预训练污染均不存在。

目前 graph eligible 全来自2Wiki。三数据集结果能检验混合训练的整体收益与 passage-only fallback；它不能证明 learned graph-quality gate 已跨数据集泛化。可先用2Wiki内未见family/题型与受控图扰动做有限泛化诊断；若要宣称真实跨数据集Graph泛化，仍需其他数据集的合格图证据。

论文贡献建议围绕三个可追踪问题组织：供给是否有用、固定混合是否有效、学习混合是否优于可靠的简单对照。不要继续把暂停的 Controller 写成已实现主方法，也不要预先要求所有数据集都显著提高。现 polished draft 开头标明目标叙事，但摘要和方法仍含动态Controller与未验证的正结论，需要在后续新稿中对齐实际固定Reader方法；本轮保留旧稿未改。[当前草稿边界](/home/zjulab/kgpaper/docs/paper/polished_draft.md:5)

过程监督和结果监督应明确区分，这也是原始过程监督研究采用逐步正确性反馈的核心实验问题；不能仅因为奖励在 step-end 发放，就把语言似然或轨迹答案一致性称为经过验证的逐步事实监督。[Lightman等，2023](https://arxiv.org/abs/2305.20050)

**12. 建议的下一阶段交付与停止条件**

顺序建议如下，已完成的数据封版不重跑：

1. 形成 reward/α successor 的具体审阅包：scorer语义、Text输入合同、normalization、α标签与特征、token位置、validity、总预算和代码修复清单。用户批准核心变量后再实现，不覆盖历史版本。
2. 修复并验证共同 PPO 接线：EOS、rollout eval、长度硬检查、prompt/chat-template，以及 gamma/lambda 的真实长度信用诊断。
3. 在 v4 exact inputs 上冻结 family-disjoint bank。先比较 scorer与简单混合，再决定是否值得训练 learned α。
4. 通过 process-only / 同outcome过程判别 / 反事实 / headroom 门后，做三臂各最多8条的真实 GPU runtime probe；记录实际显存、时间、模型版本、评分路径与回退情况。
5. matched 600×3 smoke 只用于工程健康和继续/停止，保留所有臂及失败；reward上升但独立质量下降时停止，不能只延长A。
6. 冻结正式配置、确认集、checkpoint选择与统计规则后，再按授权启动完整T/F/A及必要对照；重要结论补多seed。

如果修复后的 process reward 仍不能区分正确与错误过程，应停在 reward 阶段。如果 direct ratio、固定权重与 learned α 表现相当，优先保留简单方法，降低“learned reliability”的主张。如果仅source-adaptive获益而legacy退化，应报告适用边界与迁移代价。论文的可信度来自这些可证伪条件，而非必须使某个预设模块获胜。

**13. 本轮验证、文档同步与可复现资产**

- 奖励诊断：[脚本](/home/zjulab/kgpaper/outputs/audits/ppo_alpha_reward_readonly_review_20260905_v1/synthetic_probe.py)、[报告](/home/zjulab/kgpaper/outputs/audits/ppo_alpha_reward_readonly_review_20260905_v1/synthetic_probe_report.json)。完全合成数据、CPU、0次优化，绑定 scorer/parser hashes。
- PPO诊断：[报告](/home/zjulab/kgpaper/outputs/audits/ppo_pretraining_cpu_review_20260905_v1/report.json)。真实GAE函数与生成切片；为验证TRL模式转换，随机微型CPU模型执行1次PPO call、2次optimizer step，生成仅以stub检查模式/切片。**没有加载或更新科研checkpoint，不能把这个微型子探针称为绝对零更新，也不能当真实GPU训练验证。**
- 61项现有相关测试通过，覆盖explicit reference、mixed reward、replay、schedule、diagnostics、config forwarding；这组测试未涵盖本轮全部新发现。未启动大规模训练、外部judge/API标注或正式evaluation。
- 当前Git HEAD为76f174f8e1206d75bfd43a03dce5fb9d83ad4c43，工作区已有大量未提交变更；两诊断绑定实际源码hash，不能只用HEAD假称代码已完整冻结。
- 旧Strong SFT manifest的训练代码版本字段仍不完整。可追加可核验的说明与当前artifact hashes，不得伪造当时commit；新PPO必须绑定完整source snapshot、配置、数据与模型版本。
- 研究者本轮已确认以v4最终数据/协议/preflight为准。RESEARCH_WORKFLOW 的Stage A/B/C与todo2的400×K2 calibration等旧段落仍需后续文档同步；本轮未修改这些执行条款或已冻结资产，未对α bank的新样本划分作未经批准的决定。

上述建议涉及的核心 reward/loss 与正式 evaluation 升版、smoke/full PPO，执行前仍按 [AGENTS.md](/home/zjulab/kgpaper/AGENTS.md:64) 的权限规则形成具体可审阅方案并取得相应确认。本报告本身只完成评估，不构成训练授权。
