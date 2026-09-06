# 完整 α / 双源过程奖励 PPO：EM/F1 实施协议 v1

> 最新继任（2026-09-06）：Text逐step/softsign、代表性统计与六维α已实现；18份v2配置、校准和生产奖励检查见
> [source-credit-v2修复记录](source_credit_v2_repair_20260906.md)。独立confirmation未完成，PPO未开始；下方v1命令保留历史含义。
>
> 后续执行记录（2026-09-05）：format-v2与source-credit-v1已完成真实评分和新gate校准；当前配置及证据见
> [source-credit-v1修复记录](source_credit_v1_repair_20260905.md)。本文件原sourcegate-v1路径及无卡/待生成状态保留历史含义；
> 当前不可用旧gate/命令覆盖新版本。新信用门仅修复奖励发放，原Reader输入KG未改；尚无PPO更新或独立效用放行。

2026-09-05。研究者已明确主目标：答案奖励、过程奖励、α 全部加入后训练一个 EM/F1 较好的版本。
PPO-A 是主模型，PPO-F/T/O 是解释收益来源的对照。此前 O-first 是执行误读；已准备的 O 产物保留，
没有启动任何训练。当前远端无卡，允许完成环境/代码/配置及 CPU 数据准备。

## 1. 版本和不变资产

- 主配置：`configs/training/phase3_ppo_mixed4_sourcegate_v1_a_{probe,smoke,full}_seed42.yaml`；
  对照将 `a` 替换为 `f/t`。不同臂同阶段仅 mode 与 output identity 不同。
- 各阶段 12/600/12000 trajectories，对应 3/150/3000 PPO batch updates，K4、epochs2。
  每阶段独立从 Strong SFT 开始，失败记录保留，不覆盖或原地恢复同 ID。
- 数据仍为 final v4：3000 questions、800 strict ProofKG、12000 schedule rows；replay 为干净 v2 的2000条。
  probe/smoke 沿用已验证的父 schedule 精确字节前缀，绝不依答案选择输入。
- shared runtime v2；γ=1、λ=.99、lr1e-6、KL .25/target8、384 new tokens、6144 input tokens；
  10% replay、anchor .1、同一显式 SFT reference。rollout10 passages、replay保留15。

## 2. 生产奖励合同

合法输出统一获得 `R_out=4*(max_alias EM+.1*max_alias F1)`；非法输出**总 reward 为 −4**。
Graph mask 使用完整原始 record 的 schema、dataset::qid、question hash、provenance/cutoff、Gold-free
声明、完整执行和边来源；不能只用子图非空推断资格。`m_graph=0` 时 `α_eff=0`，不把低质量图送入过程奖励。

轨迹级 Graph 分来自保守 v2.3 scorer，未验证语义不加分；每步 Text 分来自冻结 ReaRAG，评分上下文只含
question、冻结 passages 与此前轨迹，**不含输入 KG**。Reader 看到的图不变；改变的是 reward scorer 的视图。
评分预算必须覆盖完整问题/证据，超预算须显式失败/新协议处理，不能静默左裁剪。

用 α train split 上固定统计中心化/缩放两个通道，并 clip 到 [-1,1]。PPO 中不使用 causal EMA；
T/F/A 共用同一统计文件。定义标准化 `G_i` 和 `T_it`，合法 n 步：

`r_text(step_end_t) = (1-α_eff) × .30 × T_it / n`

`r_graph(final_token) = α_eff × .20 × G_i`

Text 统计拟合对象是 train 每轨迹 raw step mean；应用顺序固定为每步标准化→clip→mean，
与先均值再 clip 不等价。所有臂、校准记录和零更新重排必须使用同一顺序。

outcome 仍放 final response token，包含真实 EOS；所有 token reward 之和必须等于三通道总和。
T 为 α=0；F 为 mask×eligible-train conditional mean α；A 为 mask×frozen learned α。
gate 与 normalization 在 PPO 期间冻结；记录 raw/normalized channels、α、features、mask 和奖励占比。

## 3. α successor 与其证据边界

新模块 `source_quality_gate_v1.py`，独立 JSON artifact，严格版本和 payload hash；不复用 legacy `.pt`。
四个稳定特征：graph density、冻结 provenance/link confidence、cite-any、cite-match。
不使用当前 policy logprob；因此同一证据/轨迹不会因 PPO checkpoint 变动而改变该特征。

第一版 target 明确是**启发式质量分配标签**：
`qG=R_Proof_v2.3/.85`，`qT=mean((raw_ReaRAG+1)/2)`，`y=qG/(qG+qT)`。
共同低质量或零分的 abstain 规则写入校准 artifact；不把 qG/qT 的名称当作已校准正确率。
候选仅来自 train pool；同 current family 的全部题和轨迹固定在同一个 train/calibration/internal-confirmation
split。只有 train split 拟合标准化、模型权重和 fixed α，calibration/internal-confirmation 只报告检验。
分割为独立 families 的60/20/20；共同低质量阈值为 `max(qG,qT)≤.05`，分母阈值为1e-8。
α-only 轻量训练使用 soft-label BCE，输出 Brier/constant、R²、分布、tie 等。

即便上述拟合通过，也只证明它能预测启发式标签，不证明 gate 是独立的可靠性判断器。
正式 PPO 之前还需在真实候选上检查 process-only correct/wrong 排序、reward-top1、oracle headroom、
constant/direct-ratio/shuffled α、offset/scale 与引用操纵反事实。v2.3 仍可能给伪造推理高分，不能删去负例。
如果真实排序失败，应修过程代理并重开版本，不能只为凑齐“α 已加入”启动大规模训练。

## 4. EM/F1 选模与论文目标

开发集沿用已消费且允许选模的150题，每数据集50；与训练/replay/canonical/confirmation三层身份隔离。
A-smoke registry 在生成结果前冻结 SFT、step200、step400、final600；legacy视图为主，no_graph为保留能力诊断。
按 macro EM→macro F1→同分优先 SFT/较早 checkpoint；允许 SFT 胜出，不能强制选择 PPO。
O 消融已有独立银行，不能把其 registry 直接改名为 A。

完整正式预算的候选 registry 在运行前另冻。选定模型后，canonical 三数据集各 n300 只用于最终报告：
保持 Scheme A 标准 pipeline、检索/重排/KG/prompt/greedy512/scorer，并 fresh 跑 SFT 与 PPO。
源代码升级后的兼容性证据必须追加，不能修改原 baseline 或绕过旧 hash gate。
开发 fixed-context proxy 分数不能当主表。单 seed 的“较好 checkpoint”不是 solid 论文结论；
随后通过同预算 T/F/A、配对统计和 seeds13/2024，区分完整方法收益、过程收益与 learned α 贡献。

## 5. 当前交付边界

无卡模式可以完成代码/环境同步、配置解析、契约测试、Gold-free candidate 输入准备和模型文件 hash。
真实 SFT 候选生成、ReaRAG 评分、α artifact、零更新效用验证、GPU probe/smoke/full 尚需 GPU。
缺少这些产物时不得声称完整训练就绪，更没有新 EM/F1 成绩。
远端独立目录：`/root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905/`。

CPU 实施已完成：T/F/A 三阶段 actual CLI 配置一致性、3000 条真实 loader/tokenizer、800/2200
mask census 通过。候选输入银行为 `outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_gitless_fix1`：
800 个 eligible + 三数据集各10个 ordinary controls，共830题、K2=1660条**待生成**，758 families，
最长2399 tokens；有效 EOS 为[128001,128009]。真实候选/分数/α权重仍不存在。
`gitless_fix1` 是候选输入的代码绑定修订：原银行保留，输入 JSONL 与原版逐字节相同；只将校准脚本的
无 Git 部署兼容性修复纳入新 manifest，不能修改旧 bank 的源码 hash 来绕过检查。
A 开发银行为 `outputs/audits/source_gated_mixed4_emf1_v1_development_a_smoke`，已独立注册四个候选。

候选银行为了 α 校准而富集 Graph，不能当作 PPO 的自然分布：Graph=800/830，而正式 PPO=800/3000。
当前 Text 标准化统计由该银行的 valid train candidates 估计；零更新报告必须按 dataset/m_graph
检查中心偏移与裁剪饱和率。若该估计令 ordinary H/M 大面积饱和，需追加代表性普通候选/加权统计的
新版本后再训练，不能因 aggregate 校准通过就忽略这个问题。

## 6. GPU 恢复后的命令与顺序

2026-09-05 TensorBoard 遥测修订覆盖本节部署入口：当前使用独立目录
`/root/autodl-tmp/kgpaper_releases/source_gated_mixed4_emf1_v1_20260905_tensorboard_v1/`，
release manifest 为 `outputs/audits/source_gated_mixed4_emf1_v1_release_tensorboard_v1/manifest.json`。
下方候选命令已指向逐字节相同输入的 successor bank，旧银行和旧 release 保留作历史审计。
监控说明：`docs/ppo_tensorboard_autodl_20260905.md`。

以下命令在上述独立远端目录执行，`bash scripts/sourcegate_python.sh` 固定新代码和现有训练环境。
本次没有执行这些 GPU 命令。

最终部署绑定使用 `outputs/audits/source_gated_mixed4_emf1_v1_release_v2/manifest.json`。
复验时向 `verify_sourcegate_deployment_v1` 显式传入该路径作为 `--release-manifest`；v1部署记录保留。

```bash
bash scripts/sourcegate_python.sh -m scripts.prepare.source_quality_candidate_bank_v1 generate \
  --bank-dir outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1 \
  --output-dir outputs/audits/source_quality_candidate_bank_v1_generated_seed42 \
  --experiment-id SOURCE-QUALITY-CANDIDATE-BANK-V1-GENERATE-SEED42

bash scripts/sourcegate_python.sh -m scripts.prepare.source_quality_candidate_bank_v1 score \
  --bank-dir outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1 \
  --generation-dir outputs/audits/source_quality_candidate_bank_v1_generated_seed42 \
  --output-dir outputs/audits/source_quality_candidate_bank_v1_scored_seed42 \
  --experiment-id SOURCE-QUALITY-CANDIDATE-BANK-V1-SCORE-SEED42

bash scripts/sourcegate_python.sh -m scripts.train.calibrate_source_quality_gate_v1 \
  --bank-manifest outputs/audits/source_quality_candidate_bank_v1_scored_seed42/manifest.json \
  --isolation-proof outputs/audits/source_quality_candidate_bank_v1_scored_seed42/isolation_proof.json \
  --output-dir outputs/calibration/source_quality_gate_v1_seed42 \
  --experiment-id SOURCE-QUALITY-GATE-V1-CALIBRATION-SEED42
```

之后检查真实候选的独立零更新效用和各层饱和率；校准成功本身不自动通过该门。GPU probe 的生产入口为：

```bash
bash scripts/sourcegate_python.sh -m scripts.train.phase3_ppo \
  --config configs/training/phase3_ppo_mixed4_sourcegate_v1_a_probe_seed42.yaml
```

probe通过后使用 `a_smoke` 配置；检查 TensorBoard、history、checkpoint 与开发集后才进入 `a_full`。
这些阶段无需先训练 O。正式配置不允许缺失或未校准 gate 静默替代为固定 α。
