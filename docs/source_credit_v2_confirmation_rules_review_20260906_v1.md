# Source-credit v2 下一阶段规则与离线来源准备审查

2026-09-06；审查 ID：`SOURCE-CREDIT-V2-CONFIRMATION-RULES-REVIEW-20260906-V1`。

本记录区分已有约束、历史实验门槛与尚待冻结的下一阶段建议。它不修改 reward、Gold、baseline、评估协议、旧 mask、已发布 gate 或训练放行标志，也不启动确认集生成或 GPU 训练。本轮当前可执行工作为长度诊断及新确认身份/输入准备。

## 1. 现有规则的原始依据

| 项目 | 原依据 | 适用性 |
|---|---|---|
| v2 正常训练加载 | [source_credit_gate_v2.py](../kgproweight/reward/source_credit_gate_v2.py) 要求 `training_clearance is True`，且必须同时 `independent_confirmation_clearance is True`；未确认产物只能显式诊断加载 | 当前代码硬约束。新两组 gate 均尚未放行；不能手改布尔值充当确认 |
| 当前零更新候选健康/效用 | [todo2.md §4](todo2.md) 建议 valid≥.90、eligible pairwise≥.65、oracle−greedy≥3pp、A top1≥greedy、A top1≥F；原文注明“最终阈值必须在看 confirmation 前冻结” | 是待冻结的研究建议，不能倒称当前 v2 已有相同硬门 |
| 最小 mixed-outcome 数量 | [旧 hard-curriculum protocol](../outputs/audits/2wiki_hard_curriculum_v1_protocol_v2/protocol.json) 要求至少25个 mixed-outcome qid、valid≥.90、pairwise≥.65、top1−random sampled EM≥.10、runtime errors=0 | 该旧82题 reserve/旧reward/旧O-K训练设计的门；可作为新协议的样本量参考，不能自动套用或拼接成当前 v2 硬门 |
| 更早的 rankability 门 | [historical v2 n100 protocol](../outputs/audits/proofkg_process_rankability_historical_v2_n100_k4_seed20260831_v1_preregistration/protocol.json) 使用 oracle−greedy≥5pp、process top1−greedy≥2pp、pairwise≥.60、valid≥.90 | 旧 scorer/人口/代际，不能用这些较低阈值覆盖后来规则，也不能合并比较不同 protocol 的结果 |
| smoke 健康目标 | [todo2.md §5.2](todo2.md) 写 valid≥.95、length-capped≤.05、replay=.10±.01，检查 KL/critic；还要求 A 对 SFT/F 的效用方向 | 600-trajectory 后续决策目标，不是12条工程样本的统计充分性标准 |
| 现代码的成本止损 | [基础配置](../configs/training/phase3_ppo_mixed4_emf1_v1_outcome_seed42.yaml)：200 trajectories 后看15个batch窗口，valid<.70、cap>.20、mean KL>10停止；非有限状态立即停止。[实现](../kgproweight/training/phase3_ppo.py) 明确它只停止明显不健康运行 | 不能将“.70未触发停止”解释为“.95健康门通过”；也不能将未运行满窗口解释为健康 |
| 当前工程 probe 预算 | [probe 配置](../configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml) 是12 trajectories、K4×3 groups；[当前 todo](todo2.md) 要覆盖一次 replay | 早期章节“每臂8条”是历史设计；当前明确的probe12更具体。它是接线/更新健康验证，不是α有效性试验 |

未找到当前 v2 已冻结的 `n_discordant≥30` 或“工程 probe 前 A 必须统计显著优于 F”的规定。也没有单独数值化、已经冻结的当前 v2 `F−T` 显著性门。新增这些条件必须作为新协议，在新确认结果揭示之前明确登记。

## 2. 过程效用必须真正排除答案奖励

本轮的排序分数应继续使用生产公式中的过程部分：

`T = .30 * TN`

`F = .30 * (1 − alpha_F) * TN + .20 * alpha_F * GN`

`A = .30 * (1 − alpha_A) * TN + .20 * alpha_A * GN`

不得把 `4*(EM+.1*F1)` 加入排序键后，再声称过程奖励独立偏好正确答案。Gold 应在模型输出、真实 ReaRAG/Graph 原始评分、α预测与排序冻结后，仅用于评估排序所选答案。

特别地，[旧 dynamic-validity confirmation scorer](../scripts/pilot/score_proofkg_dynamic_validity_confirmation.py) 的 `full_reward_selected_gain` 使用含 Gold outcome 的 full reward；其≥2pp条件不能直接转用为本轮的独立 process utility 证据。

必须分开报告：A−F 是动态α相对matched固定α的差值；F−T 是固定图文信用混合相对Text-only的差值。F同时缩小Text权重，不能把F−T描述为“固定其他过程项后额外加一个Graph奖励”的纯加法效应。raw Graph排序可以附报，但不替代真实F/T过程项比较。

top1 必须预先固定无Gold tie-break；全无效题保留并明确赋零，不丢弃以提高 top1。有效候选上的pairwise与全部候选上的ITT top1/valid须同时报告，避免只展示容易的合法子集。

## 3. 新题与新随机轨迹是两种不同确认

原800个有图题全部出现在已经分析过的830题银行中。原train、calibration和internal-confirmation都已用于设计或开发重分析；代表性120题也用于Text统计拟合，不能再次命名为独立确认。

在原671个有图信用题上换seed、增加K或延长输出，只能检验**既见问题条件下的新随机轨迹表现**。这对工程诊断有用，但不是新family泛化，也不能单凭这类复验把 `independent_confirmation_clearance` 改为True。

真正的新family确认要在揭示生成结果前固定身份，排除此前gate拟合、normalization拟合、开发重分析人口及保护账本的qid、question hash和current family；不按新source-PASS、valid或EM结果替换题目。所谓“fresh”必须明确相对于哪些已消费集合，不能只检查qid不同。

并行库存审查给出的可用strict fresh人口为431题：`bridge_comparison=161`、`comparison=153`、`compositional=117`、`inference=0`。在全部1287个strict题中，inference有256题；原Proof800已选200，剩56个qid的family均与旧830/代表性120冲突。因此目前身份提案为三种graph类型各32题，共96题，再加H/M/W ordinary各12题，共132个独立family。

权威身份与数量证据是 [capacity report](../outputs/audits/source_credit_v2_fresh_confirmation_capacity_20260906_v1/report.json) 和 [132题身份提案](../outputs/audits/source_credit_v2_fresh_confirmation_capacity_20260906_v1/candidate_cohort.question_only.jsonl)；该提案尚未冻结K或启动候选生成。

[输入scope-v2补充](../outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1/manifest.scope_v2.json) 进一步区分训练人口：Graph96对正式3000题的raw qid、question hash、family交集均为0；ordinary36的三层交集均为36。ordinary题此前未被α开发消费，但属于未来PPO训练集，因此只用于训练前Text/mask/格式诊断，不能在PPO之后作为独立测试。该说明没有改变132行输入或训练3000题。

这份提案必须标注 **inference 未覆盖**。三类型确认通过不能外推为四种Proof任务全面确认，更不能外推为HotpotQA/MuSiQue的learned graph α泛化；后两数据集在当前供应中主要检验Text与α=0回退。原strict资格也不等于新source-PASS，96题之后的PASS/FAIL/UNVERIFIED均须原样保留并分类型统计。

## 4. 低开销的下一阶段顺序与决策建议

1. **先做已消费输入上的长度诊断。** 只改变384→512输出预算，保留prompt、passages、KG、SFT、采样温度、EOS规则与format-v2。可采用固定identity hash的小型三数据集分层子集，先只生成并作CPU格式校验，不急于支付ReaRAG评分费用。问题数、配额和停止规则先写入新protocol。
2. 原生成器使用 `[candidate-seed-v1, seed, question_key, candidate_index]` 的确定性seed，见 [candidate_seed](../scripts/prepare/source_quality_candidate_bank_v1.py)。因此可复用对应384候选作为配对基线，并要求512版本在共同长度内逐token复现原前缀；若前缀不一致，先归因与核验执行条件，不能把结果差异全归因于长度。
3. 分别报告cap/EOS、缺Final/字段、步数不足/超步数、原valid变invalid和invalid变valid；保留所有失败。90%是候选健康建议线，不能因小型偏置诊断子集超过90%就宣称代表性格式问题已经解决。不要增加EOS遮罩、重采样过滤、补空步骤或放宽validator。
4. **确认输入准备与上述诊断并行。** 先冻结132身份及来源记录，离线核source信用、每题10 passages和渲染/Gold隔离；此阶段不需要打开Gold或进行确认generation。Graph类型缺口必须进入protocol。
5. **正式确认generation之前锁定全部决策。** 建议Graph使用K4，以匹配PPO组规模并提高出现不同答案结果的机会；同时准备同输入、同长度的冻结SFT greedy参考。所有A/F/T及N-only/N+F共享一份候选与真实原始评分，α/过程项在CPU重排，不分别生成六遍或重复调用六遍ReaRAG。ordinary K和宏观汇总权重也须先明确。
6. 可参考历史设计把“至少25个不同family包含有效EM命中与未命中候选”预注册为解释pairwise的最低信息量；这是下一协议建议，不是已经存在的v2硬门。若固定人口未达到，结论是信息不足，不能事后替题或继续采样直到通过。
7. 可将todo的valid≥.90、eligible pairwise≥.65、oracle−greedy≥3pp、A top1≥greedy及A≥F建议逐项审定并冻结；另明确F/T比较是报告项还是投资门。A≥F的点估计判断不等于统计学非劣性，真正的非劣性必须预先给margin和CI规则。所有规定同时保留，不按结果选择较容易的一组门或较优版本。

以上不承诺某个尚未冻结的小样本会通过，也不将旧83.33%或新90.35%开发结果当成已完成确认。本次执行长度诊断和确认输入准备；确认generation及PPO full的运行协议与结果分别记录，不能由准备阶段的PASS代替其放行证据。

## 5. K4统计与工程 probe 的边界

K4的同题正确/错误交叉组合共享同一问题和生成条件，不能作为互相独立的样本。例如一个2正确/2错误的问题可以产生4个crosspairs，但只贡献1个mixed-outcome family；它不能“凑成4个discordant独立样本”。

新协议需预先决定pairwise是按pair加权还是先题内平均再按family平均，不能在结果后改聚合口径。CI/bootstrap/配对检验应以family为抽样单位，同时报告mixed-outcome family数、crosspair数和tie数。相同132身份下A/F/T以及两种gate的比较均配对。若用已有micro-pairwise作主指标，也不能把micro pairs当独立bootstrap单位。

12条/3个K4 group的工程probe应核对真实policy/reference、采样与logprob一致性、EOS/mask、token奖励守恒、Graph信用屏蔽、Text保留、critic/梯度/optimizer有限、replay与TensorBoard，以及checkpoint/manifest可追溯。它无法合理证明α统计显著优于固定门，因此不应将“单独显著A>F”发明为3batch接线检查的必要条件。

但这不等于现在可绕过未确认的生产loader开始正式训练。现实现默认仍拒绝未确认gate；工程probe若需在独立效用确认之前执行，必须有明确、隔离、限额的另版工程授权与加载路径，不得伪造gate确认标志或沿用同一开关放行smoke/full。可先做不含optimizer更新的GPU工程验证。进入600 matched smoke和full仍遵循各自的健康、效用与审批流程；最终α贡献依赖matched A−F训练及多seed结果。

## 6. Fresh source mask 的最小离线物化路径

旧 [mask manifest](../outputs/audits/source_credit_mask_v1_local_seed42/manifest.json) 和validation目录内没有独立的通用mask构建CLI。可复用的完整来源证据构建逻辑在 [run_integrity_clearance.py](../outputs/audits/sourcegate_source_disagreement_review_20260905_v1/run_integrity_clearance.py)，初始版本另保存在同audit的 `integrity_clearance/run_integrity_clearance_initial.py`。这些旧脚本硬编码原input/output路径，不能直接运行并覆盖原产物；下一步应新增参数化封装和新目录。

该runner依据每份 `fullsource_record.execution` 收集anchor QID、hop input QID和PID，再用同一 `VersionedEvidenceStore` 与 `StoreFirstCombinedRetriever` 逐请求重放typed edges。Historical retriever显式 `offline=True`、cutoff=`2020-12-09T23:59:59Z`；[offline分支](../kgproweight/kg/historical_wikidata_retriever.py) 在cache miss时返回缺失，不请求网络、不写回缓存。

旧证据绑定的底层资源是：

- `data/external/2wikimultihopqa_official_ids/data_ids/id_aliases.json`；
- `data/derived/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3/closure_historical_property_cache.jsonl`；
- `indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42/{edges,aliases}.jsonl`；
- historical retriever、versioned store、store-first retriever三个源文件。

本次只读重新计算这7项SHA，均与旧 [qid_source_evidence.json](../outputs/audits/sourcegate_source_disagreement_review_20260905_v1/integrity_clearance_v2/qid_source_evidence.json) 的绑定一致。n1500 closure cache有2931行、2927个带非空entity的独立QID；旧已物化证据只有2269个实体，因为它只按旧830输入的实际请求取子集。因此**底层来源来自n1500闭包资源，旧evidence JSON本身不是对全n1500或新96题的覆盖证明**。

可以不联网、从这些原绑定缓存为新96题重新派生独立evidence与question_checks；是否96题全部PASS必须在实际请求重放后判断，不能根据缓存总体数量承诺。官方QID aliases/demonyms与历史label是名称证据，store别名投票不能升格为权威QID名称。缺名称/typed edge或cache miss仍为UNVERIFIED，明确身份冲突为FAIL。

新mask最小发布要求：新 `inputs.jsonl`（每题身份、question/input SHA、原m_graph、完整record及record SHA）→ 新 `qid_source_evidence.json` → 同一个冻结 `validate_source_integrity_v1` 生成 `question_checks.jsonl` → 新 `source-credit-mask-manifest-v1`，绑定inputs/checks/evidence/verifier_code并计算payload SHA。用原 [FrozenSourceCreditMask.load](../kgproweight/reward/source_credit_gate_v1.py) 全量重验每项来源SHA、record身份和逐题决定。

当前mask按精确question/record身份绑定，新family直接使用旧mask必然MISSING→α_eff=0，不能据此测Graph泛化。确认应使用独立确认mask/绑定gate视图；α权重、feature标准化、Text/Graph统计、reward系数均逐项保持冻结，原671个训练信用资格与主3000题输入不变。它仅新增确认对象的来源信用检查，不是来源输入修复，也不能提前签发PPO放行。

## 7. 本轮离线物化执行结果

上述准备现已完成，权威产物为 [fresh source manifest](../outputs/audits/source_credit_v2_fresh_confirmation_source_20260906_v1/manifest.json)。新增参数化脚本是 [freeze_source_credit_v2_confirmation_sources.py](../scripts/prepare/freeze_source_credit_v2_confirmation_sources.py)，其执行快照与全部来源SHA保存在产物内；[13项测试](../tests/test_freeze_source_credit_v2_confirmation_sources.py) 已通过。

只消费固定132题输入中的96道graph题，没有按来源检查结果替换身份：

| Graph类型 | 原提案 | PASS | UNVERIFIED | FAIL |
|---|---:|---:|---:|---:|
| bridge_comparison | 32 | 18 | 9 | 5 |
| comparison | 32 | 30 | 1 | 1 |
| compositional | 32 | 31 | 1 | 0 |
| 合计 | 96 | 79 | 11 | 6 |

这些是来源信用结果，仍不是候选格式、EM/F1或过程效用确认结果。后续Graph信用有效人口为79题，bridge_comparison仅18题；需按实际PASS人口和类型报告，不将96个原strict资格全称为有效图信用，17个非PASS题保留并α_eff=0。普通36题继续无图回退。

新manifest绑定96行原字节内容派生的 `graph_inputs.jsonl`、独立QID证据、逐题checks与原冻结verifier。实际 `FrozenSourceCreditMask.load` 复验通过；两组确认专用gate wrapper各96个身份没有MISSING，真实PASS/FAIL/UNVERIFIED与mask一致。wrapper只变化确认来源/ID/mask绑定与false放行标志，其他字段逐项比较完全不变；原训练mask与671个信用资格没有修改。

调用链以 `offline=True` 运行，并对historical网络请求与cache persist入口设置失败断言。运行开始/结束所有绑定来源和代码SHA一致。未读取Gold值、未联网、未写原cache、未生成确认候选、未启动GPU或optimizer；两组wrapper正常生产加载仍拒绝，诊断加载不等于正式PPO放行。
