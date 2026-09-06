# Source-credit v2：进一步修复记录

2026-09-06。研究者明确要求“需要进一步修复，先修复”。本轮修复 Text 归一化的拟合单位、训练样本代表性，以及 α 的来源引用分辨率；仍没有 PPO 参数更新。本轮所有新数据、校准、配置和检查使用独立版本，旧资产保留。

## 1. Text 尺度的根因与固定修复

旧 Text 统计使用每条轨迹 raw step mean 的方差，但实际奖励对每个 raw step 标准化并硬裁剪。在旧 train 的 848 条有效候选、474 题、2608 steps 中，约 **59.8% 的 step 方差来自轨迹内部**，旧拟合方式漏掉了这部分变化。

固定新版合同为：问题等权 → 题内有效候选等权 → 轨迹内 step 等权，拟合训练集 raw step 的中心 μ 和标准差 σ；尺度 `s=max(σ,0.1)`。应用顺序为：

`z_t = (raw_Text_t − μ) / s`

`S(z_t) = z_t / (1 + |z_t|)`

`T_N = mean_t S(z_t)`

这是固定 softsign 映射，没有搜索温度、分位点或最优 EM 尺度。它保留不同有限 step 分数的差别，避免硬 clip 将尾部分数压成相同 ±1。记录 hard clipping、`|S(z)|≥.95` 的软饱和，以及 `|z|>1` 的尾部比例；**hard clipping=0 是映射的数学性质，不是模型效果提升**。

只修拟合单位时，旧 train 银行得到 μ=0.7144668706、s=0.1648410928，旧尺度为0.1025195960。这是第一阶段受控诊断的统计，最终代表性银行使用另版统计，不覆盖它。

## 2. 补充 Text 统计的代表性

旧 train 的有效来源为 2Wiki 463 题、HotpotQA 7 题、MuSiQue 4 题。不能靠把这几个 H/M 样本强行加权来宣称已代表正式 1000/1000/1000 PPO 分布。

已冻结独立的小规模 Text 归一化银行：**H40 / W40 / M40，共120题，K2=240候选**；2Wiki 按原供给比例包含32道有图题、8道ordinary题。复用47道旧 train 题的94条原始候选及真实分数，新生成73题×K2=146条（H33、M36、W ordinary4）。

选题只用冻结身份、family、原图资格与固定 hash，未使用 Gold、EM/F1、轨迹有效率或评分。排除旧 calibration/confirmation 的304个family及保护账本；新增family只进入normalization train，不伪作独立确认。全部失败轨迹保留；不会为了凑齐有效40题反复替换问题。

生成与评分沿用冻结 Strong SFT、原 prompt/passages、384-token sampling合同和真实 BF16 ReaRAG，评分上下文保持passage-only。代表性版本只替换 Text 的拟合人群及统计，**α 权重、Graph 统计、来源mask和1660条主诊断候选均保持不变**。

权威补充运行目录：`outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2/`。r1 的新输入manifest有三项旧描述统计残留，已停止并保留32条部分生成；r2的输入与选题逐字节相同、元数据修正后从零重跑。这不是根据答案或有效率重试，r1未纳入最终银行。

**r2 已完成并独立复验通过**：240条全部保留，192条有效（80.00%），覆盖118题、654个真实ReaRAG步骤。实际有效支持为HotpotQA 38题/62候选/209步、2Wiki 40题/71候选/226步、MuSiQue 40题/59候选/219步。48条无效输出不进入评分统计，也未被替换；因此只能称选题均衡，不能称最终有效样本严格等量。这个80%与原graph-heavy银行88.61%的人群不同，不能直接推断模型退化；同时说明格式健康仍未解决。

最终 `μ=0.6359942726986936`、`s=0.20923710334704368`。重绑定脚本从正式问题池、旧train分配、保护账本和固定seed重新复算抽样，并核对全部240条generation/score来源SHA；保护账本及已消费holdout family重叠均为0。两组门的α逐条预测、Graph统计、source mask以及原候选/分配文件均精确不变。GPU生成与评分进程已退出。

## 3. α 特征的修复与抗重复边界

四维 v1 对许多同题候选给出完全相同的表示。只在原 train 内、不读取 Gold 或 ratio target 的表示容量审查中，最终选择两个含义明确的新增量：

| 新增特征 | 定义 |
|---|---|
| source_edge_coverage | 全轨迹唯一引用中与可见KG精确匹配的边数 / 可见KG唯一边数 |
| min_step_citation_precision | 每步唯一匹配边数 /（该步唯一已解析引用 + 唯一未知引用 + malformed标记），取全轨迹最小值；空引用为0 |

旧 density、固定 link confidence、cite_any、cite_match 保留。未加入原 Graph/Text scalar、派生答案、Final 的正确性、当前policy entropy或 Gold。重复正常引用、重复整步不改变新增两维；malformed步骤重复仍会影响旧v1的malformed计数，这一历史边界明确保留，没有声称六维对任何非法输入都完全不变。

train 中 **307 对均有效且获图信用的 K2 候选**，不按EM筛选：

| 表示 | 完全相同的候选对 |
|---|---:|
| 原四维 | 251/307 = 81.76% |
| 最终六维 | 160/307 = 52.12% |

曾在train-only容量审查中检查过重复敏感的novelty/引用分布熵；虽然它们能进一步减少特征相同率，但缺乏可信方向且容易受重复影响，未进入最终方法。开发选择记录全部保留。六维改善的是引用结构的分辨率；相同引用但 Final 或推理语义相反，仍可能具有相同特征。

## 4. 如何分开归因

本轮固定两个后继对照，不根据后验分数更换设计：

- **N-only**：旧四维 α 的全部权重、标准化和每条预测值逐项保持，只改Text归一化。
- **N+F**：在相同新Text归一化下用六维 α；保持原 train family split、启发式 ratio target、seed42、800轮logistic fit、优化器和预算。

N-only与旧版的差别归于Text归一化；N+F与N-only的差别归于门的表示/拟合。每一版内部的A/F/T共享相同来源mask、Text/Graph尺度、输入与预算；F使用对应train条件均值。代表性银行统计的替换再单独记录为人口修订，不重新拟合α。

合法轨迹总任务奖励（KL另算）：

`R = 4 × (max_alias EM + 0.1 × max_alias F1) + .30 × (1 − α_eff) × mean_t S(z_t) + .20 × α_eff × clip(G_z, −1, 1)`

非法轨迹仍为−4；来源未通过或缺失仍α_eff=0；Graph原始v2.3与其归一化不变。Text按步末分配并除以步数，outcome/Graph放最后token。既定10% replay、Strong SFT reference、PPO学习率/预算未在本轮改变。

## 5. 固定拟合后的开发诊断

旧 calibration 与 internal confirmation 已在上轮消费，所以本节明确称为**开发重分析**，不能当作新的独立确认。

| 原split | 四维Brier | 六维Brier | 四维R² | 六维R² |
|---|---:|---:|---:|---:|
| train | .021771 | .013791 | .1778 | .4791 |
| 已消费calibration | .025740 | .019176 | .1739 | .3846 |
| 已消费confirmation | .020446 | .014569 | .3343 | .5256 |

这些结果证明对所定义的启发式ratio的拟合改善，不证明独立来源可靠性。六维fixed α为0.4021405882，四维为0.4021112662。

第一阶段在同一114对均有效且获图信用的EM命中/未命中候选上，仅比较过程奖励（平局计半、无outcome）：

| 版本，原银行Text统计 | learned A | matched fixed F |
|---|---:|---:|
| 原v1 | 83.33% | 83.33% |
| N-only | 88.60% | 87.72% |
| N+F | 89.47% | 87.72% |

N+F的A−F差别只来自2对改善、0对退化，样本少且为开发诊断。主1660条候选的生成答案没有改变；不能把reward选择候选的top1当作新PPO模型成绩，也不能按此选择最优Text尺度。

**最终代表性Text统计版本**使用相同114对、相同规则复算：

| 版本 | learned A | matched fixed F | Text T |
|---|---:|---:|---:|
| 原v1 | 83.33% | 83.33% | 53.51% |
| 代表性 N-only | 90.35%（103/114） | 88.60%（101/114） | 56.14% |
| 代表性 N+F | 90.35%（103/114） | 88.60%（101/114） | 56.14% |

两个新版本A−F均为2对改善、0退化。**新增两维在这个排序指标上没有比只修Text多带来收益**，不能将整个v2对v1的变化都归因于α特征。主830题按process选K2 top1时，A的EM分别为v1 61.325%、N-only 62.048%、N+F 62.048%；F1分别67.698%、68.268%、68.142%。不同结果如实保留，不按此重选变体或参数；这仍是固定SFT候选选择诊断，PPO没有更新。

权威最终复算：`outputs/audits/source_credit_v2_representative_cached_and_utility_20260906_v1/{report,utility}.json`。原始K2 oracle@2为EM 67.349%、F1 74.628%；只允许格式有效候选时为65.783%/72.860%，不能用oracle当可实现模型成绩。

## 6. 版本边界与核验

- 新 `SourceCreditGateV2`、feature-v2、Text normalization-v2独立版本；source-credit-mask-v1继续冻结使用。
- 正常PPO CLI仅在显式 `source_gate_credit_version=v2` 时加载新门，且拒绝尚未获得独立确认的artifact。CPU零更新检查可显式加载diagnostic gate；不存在静默绕过或把开发分数当成确认的流程。
- 首版v2派生特征发现2条超过5步的invalid候选使用了全部steps，而生产遥测只消费前5步；修订calibrator改为共享validator的steps视图并另存v2。原189条invalid、1471条valid、所有有效fit数值与权重不变，旧版本保留。
- 首阶段192/192真实缓存ReaRAG生产奖励检查通过，逐token独立复算误差≤2.23×10⁻⁷；同32候选×A/F/T×两版。Graph屏蔽、Text保留、负例−4及代码/数据绑定均检查。
- 最终代表性版再次192/192通过，最大token误差2.22156×10⁻⁷；两组各1660条（1471valid+189invalid）的全部feature values和m_graph与生产视图一致。正常生产loader拒绝两组未独立确认gate，审计输入与代码开始/结束SHA一致。
- 最终32条真实SFT候选的A奖励分别写入两组TensorBoard诊断run并读回；N+F有257 scalar/98 histogram标签，六维与hardclip/softsat/尾部关键值逐项一致。事件明确`optimizer_updates=0`，是零更新诊断，不是PPO训练曲线。
- 剩余格式失败不是已发现的EOS裁剪bug：189条中81触及长度上限、108由模型主动EOS结束。统一min_length会同时阻止合法短输出；没有放宽validator或强行改EOS。后续如比较384→512必须建立同输入同seed的独立小probe。

当前协议：`outputs/audits/source_credit_v2_stepfeatures_protocol_20260906_v2/protocol.json`。
校准底座：`outputs/calibration/source_credit_gate_v2_stepfeatures_local_seed42_20260906_v2/`。
代表性统计继任：`outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1/`（已完成且来源重放通过）。

已生成18份新配置：`configs/training/phase3_ppo_mixed4_source_credit_v2_{norm,features}_{a,f,t}_{probe,smoke,full}_seed42.yaml`。两组分别绑定代表性版 `norm_only/gate.json` 与 `features_v2/gate.json`；每个阶段的A/F/T共享其他配置，CLI解析18/18通过。它们是可追溯的下一阶段配置，尚不具有独立confirmation放行。

集成回归465项通过（JUnit保存在 `outputs/audits/source_credit_v2_integrated_regression_20260906_v1/`）；随后代表性重绑定相关97项通过。TensorBoard实际writer的尾部分数误报也已修复：v2的 `text_clip_frac` 读取真实硬裁剪指标，另记 `text_raw_z_outside_unit_frac` 和 `text_softsign_saturation_frac`，六维特征均可写入事件。该追加修改的12项相关测试通过；这些测试集存在重叠，不累加为独立测试总数。

本轮的交付目标是修复可解释的统计与实现缺陷并如实报告剩余不确定性。独立确认、PPO probe/smoke以及正式baseline EM/F1仍是后续实验，不用本轮开发重分析代替。

诊断图：[PNG](../outputs/audits/source_credit_v2_repair_figures_20260906_v1/repair_summary.png) / [SVG](../outputs/audits/source_credit_v2_repair_figures_20260906_v1/repair_summary.svg)。图中展示统计映射、表示相同率、实际有效拟合人口和固定114对开发排序，未展示PPO学习曲线。

最终归档：[release manifest](../outputs/audits/source_credit_v2_repair_release_20260906_v1/manifest.json)，Experiment ID=`SOURCE-CREDIT-V2-REPAIR-RELEASE-20260906-V1`。工作区已有历史未提交修改，因此以逐文件SHA及源码/配置/文档快照绑定，未将既有修改混入新Git commit。
