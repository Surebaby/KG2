# PPO EM/F1 performance pilot v1

> 最新研究者澄清：主目标是含 α 与过程奖励的完整模型。本文件仅保留已经准备的 PPO-O 消融说明；
> O-probe/O-smoke 不是 A/T/F 的必跑前置。主线见 `ppo_sourcegate_execution_20260905_v1.md`。

日期：2026-09-05。决策依据：研究者已授权实施审查修复，优先取得综合 baseline 下表现较好的
EM/F1 checkpoint；远端当前无卡，先同步代码、环境和配置。没有 GPU probe、PPO checkpoint 或新评估分数。

## 1. 实验身份与首轮目标

| 阶段 | Experiment ID / output basename | 轨迹 / PPO updates | 起点 |
|---|---|---|---|
| probe | `ppo_mixed4_emf1_v1_outcome_probe12_seed42` | 12 / 3 | frozen Strong SFT |
| smoke | `ppo_mixed4_emf1_v1_outcome_smoke600_seed42` | 600 / 150 | frozen Strong SFT |
| full | `ppo_mixed4_emf1_v1_outcome12000_seed42` | 12000 / 3000 | frozen Strong SFT |

三阶段独立运行；不把 probe/smoke 更新接到 full 中，也不累计成同一预算。K=4 是每题四条在线采样轨迹，
封版文件里的 12000 行是问题调度，并非提前生成的策略回答。smoke 与 probe 使用父 schedule 的完整字节前缀，
不依答案挑题。probe 三数据集各 1 个 group，smoke 各 50 个 group；full 各 1000 个 group。

这是组合工程修订：runtime、格式检查与 γ/λ 同时变化。目标是先得到可信的性能候选，不作单变量或 α 因果主张。
数据、SFT、legacy baseline、历史失败记录及 v2.1/v2.2 scorer 保留不动。

## 2. 奖励与配置

合法轨迹奖励为 `4 × (max_alias canonical_EM + 0.1 × max_alias canonical_F1)`；非法轨迹为 `−4`。
EM/F1 只在 reward/scoring 中读 gold。policy 输入仍为封版 question/passages/可用 KG。
第一轮关闭 text process、Proof process 和 α，不加载未被使用的 ReaRAG/PRM/α 模型。
过程格式只作为语法条件，不能声称格式合法意味着推理正确。

`runtime_contract_version=v2` 保留旧默认 `legacy`：

- rollout 生成暂时切换 eval，随后恢复各模块原 mode；dropout 不再使采样策略和计算 logprob 的策略失配。
- 保留首个实际 EOS（包含 PAD=EOS 情况），区分正常停止与长度截断。
- response、attention mask、policy/ref logprob、逐 token reward 长度必须严格一致；总奖励须守恒且有限。
- 对完整输出检查步数上限与唯一答案字段，再做评分裁剪；每步 Reasoning/Knowledge Used/Conclusion 各一次。

核心配置：lr=1e-6，batch4/mini1，K4，ppo_epochs2，γ=1，λ=.99，KL coefficient .25，target8，
clip=.2，max_new384，input6144，temperature1/top_p1/top_k0，zero-init critic/value dropout0。
policy/reference 都来自冻结 Strong SFT；10% replay，anchor weight .1。
`ppo_max_passages=15` 是上限：v4 每行只有 10 篇，replay 的原 15 篇得以完整保留。

自动 smoke 健康停止门：从 200 trajectories 后观察最后 15 updates，valid<.70、length-capped>.20、
mean KL>10 或任何非有限状态触发停止并保留失败 checkpoint。它是灾难性退化成本门，不能当作效用通过门。
继续 full 前还要看 KL/critic/clip/长度曲线、replay 实际比例和开发集 EM/F1。

新 `proofkg-structural-answer-v2-3-frozen-1` 只供后续诊断：取消未验证语义的 .15 加分，严格全答案匹配，
总上限 .85；答案无法确定时上限 .40。它依旧可能给伪造推理与正确推理相同分数，不能称为语义 verifier，
也未通过真实候选 rankability/训练收益验证。PPO-O 不消费这个分数。

## 3. 选模与综合评估

采用已被消耗、允许 method/checkpoint 决策的 development150：H/W/M 各 50。
与 canonical900、confirmation300、rollout3000、clean replay2000 按 dataset::qid、question hash、current
family 全部隔离。沿用历史开发问题/答案/10 passages；不沿用 QPEG/SAEG 的图、模型或输出格式。

两个开发视图都用标准 Step prompt：legacy 通过当前标准 pipeline 的离线 KG 方法物化，no_graph 为空图。
输入 JSONL 不含 gold，labels 独立。smoke 开始前冻结候选 SFT、step200、step400、final600。
greedy、max_new512；按 legacy 三数据集 macro EM → macro F1 → 同分优先 SFT → 更早步数选择，
每个候选必须保留两个视图的完整结果。选择允许返回 SFT，不强迫宣布 PPO 获胜。
full 候选 registry 在 full 开始前另冻，禁止根据报告集成绩筛选 checkpoint。

开发银行是 fixed-context proxy，不能冒充 canonical Scheme A 主表。最终报告仍须在冻结三数据集各 n300、
seed42、greedy512、原检索/重排/KG/答案 scorer 条件下，通过标准 pipeline fresh 跑 SFT 与所选 PPO。
源文件 hash 变化需独立追加兼容性记录，不能修改原 protocol 或忽略 integrity gate。
报告 EM/F1/macro、每数据集 gained/lost/tied 与配对区间；外部 baseline 资源/模型差异单列。
单 seed pilot 只支持工程选择；论文核心结论需要后续多 seed 与 matched 消融。

## 4. 操作入口

配置：`configs/training/phase3_ppo_mixed4_emf1_v1_outcome{,_probe12,_smoke600}_seed42.yaml`。
确切文件名以三个实际配置为准，full 文件名为 `..._outcome_seed42.yaml`。
准备脚本：`python -m scripts.prepare.prepare_ppo_emf1_pilot_v1`，只读验证父数据并生成独立协议，已存在目录拒绝覆盖。
运行入口：`python -m scripts.train.launch_ppo_emf1_v1 --stage probe --check-only`；GPU 恢复后用 `--execute`。
smoke/full 使用相应 stage，必须有完成的前序阶段；不通过时保留原目录，用新版本修订。
选模工具：`python -m scripts.eval.ppo_emf1_development_v1 --help`，提供 prepare/generate/score/select。

远端使用独立 `/root/autodl-tmp/kgpaper_releases/ppo_mixed4_emf1_v1_20260905/`，
复用 `/root/autodl-tmp/kgpw_env/bin/python` 并显式把新 release 放在 import 路径首位。
最终源文件/配置/数据/模型绑定在 `outputs/audits/ppo_mixed4_emf1_v1_release/manifest.json`；
机器环境与同步结果写独立报告。CPU 通过只表示准备完成，CUDA/GPU 更新、显存峰值和新 EM/F1 均尚未验证。
