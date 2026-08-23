# KG-ProWeight 重训与实验整改方案（v2 全面重写）

> 生成日期：2026-08-22
> 上一版：`docs/retraining_plan.20260821.bak.md`（2026-08-21，已备份保留；其中 0.4/0.5/§2.6 三条结论被本轮核实推翻，见 §0.9）
> 目的：回答"KG 定位与银标 KG 有效率、IHR 统一、PPO 重训、重评、训练/奖励参数调整"五类问题，给出**待处理清单 + 可执行方案**。
> 研究目标（本轮定调，来自研究者）：**在尽可能提升 EM/F1 的基础上降低 IHR**。EM/F1 与 IHR 同为一等指标，不接受"EM 不动只降 IHR"的表述。
>
> 依据：当前代码（`kgproweight/` + `configs/training/phase3_ppo.yaml`）、训练产物（`checkpoints/*/{manifest,history}.json*`）、`data/silver_data/silver_v1_reannotated.jsonl`、`indexes/kg_cache/question_kg_index_v2.json`、`outputs/` 落盘。本文所有数字都在本轮（2026-08-22）现场核实过，标注了核实方式。

---

## 状态一览（2026-08-23 更新，先看这里）

| 项 | 状态 | 解决方案（实际执行的） | 效果（实测） |
|---|---|---|---|
| **R-1** 步数塌缩 | ✅ 代码已改，待 smoke 验证 | `min_valid_steps` 2→3、新增 `shortfall_coef=0.25` / `target_steps=3`，罚在最后一步且**不因 valid 而豁免** | 单测 6 条钉住 shortfall 行为（`tests/test_composite_reward.py`，含"invalid 也要罚"一条）；真实效果需 smoke 的 check 9 判定 |
| **R-2** 训练 KG 索引 0% 命中 | ✅ **已关闭** | 用 `--silver` 按银标问句重建（`max_keep=12`）→ `rsync` 送到远端 → 核对 md5 与 config 指向 | ABSENT **100% → 0.00%**（远端实测）；带非空 KG 的 prompt **0% → 92.5%**（本机端到端）；`tests/test_kg_index_guard.py` 5 条回归钉住 absent/empty 的区分 |
| **R-3** 预算/artifact/split | ✅ 代码已改 | `split: null→train`、yaml 末注释 30→12 纠正、artifact 路径核对 | 已同步到远端，远端 yaml 第 34 行确为 `split: train` |
| **P6** MuSiQue α-KG 矛盾 | ✅ 已完成 | 在单一固定状态下重测 α（3 数据集 × bias0 对照） | 矛盾不存在：α 实为单特征密度函数（r≈+0.92~0.96）；另发现 +0.78 使 α 进入饱和、sd 减半 |
| **P7** 统计口径 | ✅ 历史数据已跑完 | 配对检验（McNemar 精确二项 + bootstrap + Wilcoxon） | 12 组 11 组 NS；唯一显著项方向**相反**（HotpotQA PPO+KG 是 −10.0pp，p=0.00090）；重训后须重跑 |
| **smoke run** | ⛔ **阻塞：远端容器未挂 GPU** | —— | `nvidia-smi` 是 0 字节占位文件、无 `/dev/nvidia*`、torch `cuda.is_available()=False`；需在 AutoDL 控制台切带卡模式 |
| **α bias 重新推导** | ✅ **已完成 2026-08-23** | 按校正自己声称的机理反算上界，并用 3 个 bias=0 run 反解真实 `f_entropy` | **+0.78 不是推导值，是为把 `b_eff` 凑成整数 −1.0 反解的**；机理允许的上界仅 +0.363（超 2.15×），D1 之后真实需求仅 +0.129（超 6.0×）。**建议取 0.0**，见 §P6 后续 |
| **重训前必做** | ✅ **代码侧已完成**（Phase 2 重跑仍待 GPU） | D2 量纲统一：§9.4-(1) `r_text` 中心化 ✅ **已实现**（因果 EMA，非批内均值，理由见 §9.4-1）、§9.4-(3) shortfall ✅ 已实现、§9.4-(2) `r_kg` 按设计不动。`alpha_bias_correction` 默认值 → 0.0 ✅ **已落地**（`kg_proweight_pipeline.py:170` `None→0.0`，`run_kg_proweight.py:52` 帮助文本已改；传 `0.78` 才复现旧 run） | `dR/dα` **−0.1482 → +0.1363**（为负的步占比 99.4% → 1.5%），通道 sd 未被压（0.1368 vs 0.1353）；194 测试通过；默认关闭故旧 run 逐位复现 |
| **F-2** α-gate 拟合上限只有 R²=+0.038 | ✅ **已关闭 2026-08-23**（方案 A+B，你已批准）| 实测证明"调不出来"而非"没调好"：把门自己的函数形式直接拟合它自己的目标，上限也只有 +0.038，而现役 checkpoint **比常数预测还差**。加两个**逐步引用特征**（`cite_any` / `cite_match`），门扩到 5 特征，旧 checkpoint 零填充兼容 | 拟合上限 **+0.038 → +0.439**（BCE 0.5854 → 0.3762）；单个 `cite_any` 就是原三特征之和的 **8 倍**信息量。206 测试通过（+12）。⚠️ **A 必做**：Phase 2 重跑才让新权重生效，零填充只保证不改变旧行为。详见 §14 |
| **F-1** α-gate 特征被脚手架污染 | ✅ **已关闭 2026-08-23** | `entity_filter.py` 只匹配**单 token** 脚手架，所以 `"Knowledge Used"` 这类多词短语整个绕过过滤器（模块自己的文档头就点名要删它）。改为"**所有** token 都是脚手架才删" | `"Knowledge Used"` 在 **32,986/33,011 步（99.9%）** 存活，纯脚手架占全部保留 mention 的 **17.2%**（32,989/191,400），每个注入 0.667 的虚假 `link_confidence`；+3 条回归测试（191 → 194）。⚠️ **蕴含 α-gate 必须重新拟合**（Phase 2 重跑），不能沿用现有 checkpoint |

> 远端（`connect.bjb1.seetacloud.com:12200`）在 2026-08-23 修复后已完成部署：索引 1 个文件 + 代码 35 个文件（`kgproweight/` 9、`configs/` 1、`scripts/` 21、launcher/checker 4），8 个关键文件 md5 与本机一致。**部署前那台是没有 `.git` 的旧快照**，`split: null`、`min_valid_steps: 2`、且完全没有 `question_kg_index_path` —— 详见 §2.4 R-2 末尾。

---

## 0. 本轮核实发现的事实（先读；直接改变方案）

以下每条都是**现场跑数据核实**的结果，不是从旧文档抄的。带 ⚠️ 的是能让一次重训白跑的。

### 0.1 ⚠️ 核心病根：PPO 把步数压到门槛下限，KG 过程奖励因此自动饿死

这是本轮最重要的发现，它同时解释了 PPO① 和 PPO② **两次**退化，而旧文档只归因于 Plan B 权重。

核实方式：`checkpoints/*/history.jsonl` 的 `n_steps_sample`（该字段是**每 batch 的总步数**，batch_size=4，故除以 4 得每轨迹步数）+ `checkpoints/kg_proweight_ppo_v2/samples/step_*.txt` 里数 `[Step N]`。

| 阶段 | PPO① 步/轨迹 | PPO② 步/轨迹 | PPO② rollout 样本实测 |
|------|-------------|-------------|---------------------|
| 训练前 10%（update 0-75） | 2.84 | 3.13 | step_500: 3.25 |
| 中段（375-450） | 2.17 | 2.05 | step_1500: 2.00 |
| 末 10%（675-750） | **2.04** | **2.01** | step_3000: **2.00**（4/4 样本全是 2 步） |

对照：银标 accepted 轨迹是 **3.36 步/轨迹**（min 3, median 3, max 7）；`min_valid_steps=2`。

**机制**：`trajectory_valid` 只要求 ≥2 步。多写第 3 步的边际收益 = 该步的 `r_total`（$\alpha r_{KG}+(1-\alpha)r_{text}\cdot 0.3$，量级 ~0.1-0.3，再乘 `step_reward_scale`），边际成本 = 多 ~60 token 的 KL 惩罚。**成本大于收益，所以策略一路收敛到刚好 2 步**。于是：

- 模型从 SFT 的 3.36 步退化到 2 步 → 中间推理被砍掉 → 这正是 **IHR 恶化**（PPO② HotpotQA 18.0%→25.0%）和 **EM 下降**的直接来源；
- KG 过程奖励的作用面从 3.36 步缩到 2 步，`kg_reward_share` 被动萎缩（PPO① 末段 0.0873，比中段的 0.1018 还低）。

**结论**：`kg_reward_share` 低**不只是**银标引用率低（旧文档 §P1 的判断），还有一半是**步数塌缩**。只重建银标不修这个门，PPO 还会退化。**这条必须在 P3 重训前修**，见 §2.4 R-1。

### 0.2 ✅ 训练期 KG 索引命中率 = 0%（旧文档 0.1 结论正确；**2026-08-22 本机修复，2026-08-23 远端部署并实测通过**，见 §2.4 R-2）

核实方式：加载 `indexes/kg_cache/question_kg_index_v2.json`（22,393 条），取 `silver_v1_reannotated.jsonl` 前 4,000 条 accepted 的 `question` 字段做精确匹配。

- 索引 `question_id` 前缀 **100% 是 `dev_`**；银标 `qid` 前缀 **100% 是 `train_`**；
- 命中 **0/4000 = 0.0%**；
- 后果：`phase3_ppo.py:357` 的 `question_kg_index.get(traj.question)` 永远 miss → 永远走 fallback（对银标自带 `kg_subgraph` 再过滤一次）。

**本轮补充核实**：0% 的成因不是 split 前缀，而是**银标文件本身不同**。旧索引恰好 100% 覆盖 `silver_v6_full_20260801_0136.jsonl`（22,393 条，qid `dev_*`），而 PPO/SFT 三个 launcher 全部读 `silver_v1_reannotated.jsonl`（24,997 条，qid `train_*`）；两个问句集合**完全不相交**（raw / strip / strip+lower 三种归一化下命中都是 0）。所以这不是键归一化 bug，重建是唯一出路。已重建，ABSENT 降到 0.00%，见 §2.4 R-2。

索引侧覆盖（本轮实测）：2wiki 12,576 / hotpotqa 7,405 / musique 2,412 条；三元组均值 15.87、中位 15；空 KG 占比 hotpotqa 5.5% / 2wiki 3.1% / **musique 12.4%**。

> 注意：`04_experimental_setup.md` 写的"2% 题目无 KG"与实测（整体 4.9%、musique 12.4%）不符，写作时按实测。

### 0.3 ⚠️ 训练侧三元组预算与推理侧不一致，且 PPO①/② 之间也不一致

核实方式：`checkpoints/*/manifest.json` 的 `run.config` 逐字段 diff。

| 字段 | PPO①（`ppo_r10_split`） | PPO②（`kg_proweight_ppo_v2`） | 推理（`kg_proweight_pipeline.py`） |
|------|------------------------|------------------------------|--------------------------------|
| `ppo_max_kg_triples` | **30** | **12** | `max_kg_triples=12` |
| `outcome_weight` | 4.0 | **8.0** | — |
| `step_reward_scale` | 1.5 | **0.5** | — |
| `sft_checkpoint` | `sft_student_split/final` | **`sft_student/final`** | — |
| `alpha_gate_path` | `prm_alpha_gate_v1reann_negfix` | **`prm_alpha_gate`** | — |
| `split` | `train` | **`None`**（全量，含 val/test） | — |
| `max_steps` | 5 | 5 | — |
| 其余（batch/kl_coef/target_kl/gamma/min_valid_steps/sft_anchor_*/temperature） | 一致 | 一致 | — |

**PPO① vs PPO② 差了 6 个变量**，不是"只有 Plan B 权重"。`v2_review.md` 把 PPO② 的暴跌全归因 Plan B 属于**过度归因**——`split=None` 意味着 PPO② 在 val/test fold 上也训过（数据泄漏），这一条单独就足以让 PPO② 的所有数字作废。

### 0.4 ✅ 更正上一版：主方法 EM/F1 **已经**是 temp=0 / n=300 / rerank-10，与基线同口径

上一版 §0.4 说"主方法历史评估用 temp=0.7、n=100，与基线不可比"——**这是错的**，它看的是 `outputs/R8_final/`（7 月的旧批次）。`statistics.md` 主表实际取自 `outputs/wiki18_eval/`，本轮逐个核实其 `config.yaml` + `intermediate_data.json`：

| 项 | 主方法（`wiki18_eval/*`） | 基线（`baselines_rerank/*`） | 一致？ |
|---|---|---|---|
| `temperature` / `do_sample` | 0.0 / false | 0.0 / false | ✅ |
| `test_sample_num` | 300 | 300 | ✅ |
| `retrieval_topk` | 50 | 50 | ✅ |
| 实际进 prompt 的 docs | **10**（实测 `retrieval_result` 长度） | **10** | ✅ |
| 语料 | wiki18 | wiki18 | ✅ |
| seed | 42 | 42 | ✅ |
| GPU | RTX 4090 | RTX 4090 | ✅ |

**结论**：EM/F1 主表与基线表**可以同表比较**。P4 全量重评的理由不是"口径不同"，而是"**要评的是新 checkpoint**"——这是个重要区别，它把 P4 的工作量从"12 个 cell 全重跑"降到"只重跑 PPO 那 6 个 cell"（SFT 四臂的数字仍然有效）。

> 唯一例外：基线 `trace` 的 `retrieval_result` 长度是 33（IRCoT 多轮累积检索），这是方法固有差异，不是配置错位。

### 0.5 ⚠️ MuSiQue PPO 两臂确实注入 0 KG，且 SFT 臂的 musique 数字来自另一个目录

核实方式：正则从 `intermediate_data.json` 的 `output.prompt` 里抽 `[Knowledge Graph Context]` 段数三元组。

| run | 目录 | 实测 KG 三元组/题 |
|---|---|---|
| HP SFT noKG | `wiki18_eval/sft_nokg` | 0.00（300/300 空，正确，这是 noKG 臂） |
| HP SFT+KG | `wiki18_eval/sft_split_v2` | **10.26**（33/300 空） |
| MSQ SFT+KG | `wiki18_eval/musique_kg`（**不是** `musique_sft_kg`，后者无 metric） | **6.47**（55/300 空） |
| MSQ PPO②+KG | `wiki18_eval/ppo_v2_musique` | **0.00（300/300 空）** ← 断网期跑的 |
| HP PPO②+KG | `wiki18_eval/ppo_v2_hp_kg_v2` | 10.95（15/300 空） |

**结论**：`statistics.md` 对 MuSiQue PPO+KG 打的星号是对的。此外 MuSiQue SFT+KG 的 6.47 三元组/题明显低于 HotpotQA 的 10.26，与 0.2 的 musique 12.4% 空 KG 一致——**MuSiQue 是 KG 覆盖最差的数据集**，而它恰好是 α 最高（0.855）的数据集。这个组合在论文里要主动解释，否则审稿人会问"α 高但 KG 少，α 到底在响应什么"。

### 0.6 ✅ 更正上一版：`max_steps` 不是 7，也没有"reward 只算前 7 步"的问题

上一版 §2.6 说"`reward_function.py:298` 用 `max_steps=7` 截断，但 rollout 实测 8.87 步"。本轮核实：

- `reward_function.py:182` 的 `max_steps: int = 7` 只是**函数签名默认值**，PPO 实际传的是 `cfg.max_steps`（`phase3_ppo.py:739`），而 `Phase3PPOConfig.max_steps = 5`（`phase3_ppo.py:158`），两个 manifest 都记录 `max_steps=5`；
- "8.87 步"是把 `n_steps_sample`（每 **batch** 总步数，batch_size=4）当成了每轨迹步数。真实值是 **2.2 步/轨迹**（见 0.1），远低于 5。

**结论**：截断不存在，`max_steps=5` 从未被触及。真正的问题**方向完全相反**——步数太少，不是太多。上一版的"建议提高 max_steps"是无效改动。

### 0.7 银标真实统计（全量核实 24,998 条）

| 项 | 实测值 | 核实方式 |
|---|---|---|
| 总轨迹 / accepted | 24,998 / **9,839（39.4%）** | 逐行统计 |
| `teacher_model` 字段 | **`deepseek-chat`，100%（24,998/24,998）** | 逐行统计 |
| 全体步 / 引用 KG 的步 | 80,120 / 24,969（**31.2%**） | 逐行统计 |
| accepted 步 / 引用 KG 的步 | 33,011 / 16,861（**51.1%**） | 逐行统计 |
| accepted 轨迹步数 | 均值 **3.36**，中位 3，min 3，max 7；3 步占 72% | 逐行统计 |
| `kg_subgraph` 大小（accepted） | 均值 38.3，中位 50，**6.8% 为空** | 逐行统计 |
| `kg_subgraph` 大小（全体） | 均值 31.2，中位 39，**20.0% 为空** | 逐行统计 |

**accepted 步的标签 × 是否引用（交叉表，本轮新增）**：

| | NEG | NEU | POS |
|---|---|---|---|
| **未引用**（48.9%） | 685 (2.1%) | 15,465 (46.8%) | **0** |
| **已引用**（51.1%） | 737 (2.2%) | 7,698 (23.3%) | 8,426 (25.5%) |

这张表是 §2.2 方案的依据：**POS 只可能来自引用步**（未引用步 POS 数严格为 0，因为 $r_{KG}=\text{precision}\times\text{relevance}$，无引用则 precision=0）。所以"提高 POS 比例"与"提高引用率"是**同一件事**，不是两个杠杆。

**拒绝原因分解（实测 `metadata.reject_reason`）**：

| 原因 | 条数 | 占被拒 | 性质 |
|---|---|---|---|
| `answer_score=0.00` | 6,294 | 41.5% | 教师最终答案全错 |
| `sparse_quota_full` | 4,726 | 31.2% | **KG 稀疏桶配额满，人为丢弃** |
| `step_count=2` | 3,811 | 25.1% | 少于 `min_steps=3` |
| `step_count=1` | 177 | 1.2% | 同上 |
| 其余（`answer_score` 0.12-0.29） | ~111 | 0.7% | 分数略低于 0.3 门槛 |

**桶分布**：`kg_rich` 4,955（全部 accepted）、`kg_medium` 2,424（全部 accepted）、`kg_sparse` 7,186（accepted 2,460 / 拒 4,726）、`rejected_quality` 10,433（全拒）。`triple_rate` 均值 0.295、**中位 0.000**。

### 0.8 IHR 现状：主方法与基线判官不同，且都只有单次抽样

核实方式：遍历所有 `ihr_result*.json` 读 `judge_model` / `n_items` / `mean_ihr`。

| 组 | 文件 | judge | n | 状态 |
|---|---|---|---|---|
| 主方法 12 臂 | `outputs/ihr_eval/{hotpotqa,2wiki,musique}_{sft,ppo}_{kg,nokg}.json` | **`deepseek-chat`** | 50 | ⚠️ 判官旧 |
| trace | `baselines_rerank/trace/*/ihr_result_ircot.json` | `deepseek-v4-pro` | 50 | ✅ |
| r1_searcher | `baselines_rerank/r1_searcher/*` | `deepseek-v4-pro` | 50 | ✅ |
| rearag hotpotqa | `baselines_rerank/rearag/hotpotqa/*` | `deepseek-v4-pro` | 50 | ✅ |
| rearag 2wiki / musique | — | — | — | **运行中**（PID 1745597，12:59 启动） |

另外：`scripts/eval/run_ihr_judge.py:31` 的 `--judge_model` **默认值仍是 `gpt-4o-2024-08-06`**（而 `run_baseline_ihr.py:143` 默认已是 `deepseek-v4-pro`）。P5 重判时若忘记显式传参，会去调一个本项目没配置的模型。**这是个待修的埋雷**，见 §2.5。

### 0.9 上一版被推翻的三条（勿再引用）

| 上一版条目 | 上一版说法 | 本轮实测 | 影响 |
|---|---|---|---|
| §0.4 | 主方法是 temp=0.7 / n=100，与基线不可比 | 主方法是 **temp=0 / n=300 / rerank-10**，与基线**可比** | P4 范围从 12 cell 缩到 6 cell |
| §0.5 | "多 seed 噪声地板 ~4pp 对 EM/F1 不成立" | **结论本身正确**（temp=0 + first-N 确定性），但据此说"EM/F1 无不确定度"是错的：n=300 的**二项抽样误差**仍在（EM≈0.34 时 SE≈2.7pp，95% CI 半宽 ≈5.3pp） | 显著性必须用 McNemar + bootstrap，且**必须报 CI** |
| §2.6 | `max_steps=7` 截断了 8.87 步的 rollout | `max_steps=5`；真实步数 **2.2**，方向相反 | 该建议作废，改为 R-1（见 §2.4） |

---

## 1. 待处理问题总清单

### 1.1 依赖图

```
P0 KG 定位定调（你定）─────────────────────────────► 写作（不阻塞训练）
                                                        │
R-1 步数塌缩修复 ─┐                                     │
R-2 训练 KG 索引 ─┼─► P3 PPO 重训 ─► P4 重评（6 cell）─┼─► P6 主表
R-3 预算/artifact ┘        │                            │
                           └─► P5 IHR 统一重判 ─────────┘
P1 银标重建（长期并行线）──► P3'（第二次重训，可选）
P2 IHR judge 默认值修复 ──► P5
```

### 1.2 清单

| # | 问题 | 阻塞什么 | 依赖 | 归属 | 优先级 |
|---|------|---------|------|------|--------|
| **R-1** | **PPO 步数塌缩到 2 步（0.1）** → EM/IHR 双输 | P3 | 无 | 代码 | 🟡 代码已改（`min_valid_steps`3 + `shortfall_coef`0.25），**待 smoke check 9 判定**；与 §状态一览 一致 |
| **R-2** | 训练期 KG 索引 0% 命中（0.2） | P3 | 无 | 代码 | ✅ **已关闭 2026-08-23**（本机重建 → 远端部署 → 远端实测 ABSENT 0.00%；顺带发现远端是旧快照，已同步 35 个代码文件） |
| **R-3** | `ppo_max_kg_triples` 训练 30/12 vs 推理 12；PPO② 的 `split=None` 泄漏（0.3） | P3 | 无 | 代码/脚本 | ✅ 代码已改（`split: train`、yaml 注释纠正、artifact 路径核对），已同步远端 |
| **P0** | KG 定位定调（训练期正则 vs 推理期增强） | 写作 | 无 | **你定** | 🔴 P0 |
| **P3** | PPO 重训（对齐 artifact + R-1/2/3） | P4/P5 | R-1,2,3 | **你确认**后训练 | 🔴 P0 |
| **P4** | 重评 PPO 6 个 cell（SFT 4 cell 复用，见 0.4） | 主表 | P3 | 评估 | 🟡 P1 |
| **P5** | IHR 统一 `deepseek-v4-pro` 重判（主方法 12 臂）+ 补 rearag | 主表 | P4 | 评估 | 🟡 P1 |
| **P2** | `run_ihr_judge.py` 默认 judge 仍是 gpt-4o（0.8） | P5 | 无 | 代码 | 🟡 P1 |
| **P6** | MuSiQue KG 覆盖 12.4% 空 + α 最高的矛盾（0.5） | 写作可信度 | 无 | 分析 | 🟡 P1 |
| **P1** | 银标重建（强教师 + 强制引用 + 三数据集 train split） | 天花板 | 无 | **你定** | 🟢 P2（长期） |
| **P7** | 统计口径：McNemar + bootstrap CI（0.9） | 主表 | P4 | 分析 | ✅ 历史数据已跑完（12 组全 NS，唯一显著项是 PPO+KG 变差）；重训后须重跑 |

**最短关键路径**：R-1 + R-2 + R-3（代码，半天）→ P3（远端 ~4h）→ P4（6 cell，~2h）→ P5（12 臂 IHR）。P1 银标重建是并行长期线，不在关键路径上。

---

## 2. 逐项详细方案

### P0 — KG 定位定调（阻塞 Abstract / Introduction / Contribution）

**问题**：`RESEARCH_WORKFLOW.md` §3.4 写"推理时无需外部 KG（KG 只用于训练）"，但 `kg_proweight_pipeline.py:59` 默认 `inject_kg=True`，且 §4 的主消融轴"有 KG / 无 KG"正是**推理期注入**的开关（本轮实测：SFT+KG 臂 prompt 里确有 10.26 三元组/题，noKG 臂 0.00）。当前实际做的是混合用法。

**三个选项**：

| 选项 | 内容 | 代价 | 与现有数据的关系 |
|---|---|---|---|
| **A（推荐）** | 定位为"**训练期过程奖励正则 + 推理期 KG 输入增强**"双重用法 | 卖点从"零推理开销"变成"KG 双通道"，需在 Limitations 承认推理期需 KG 访问 | ✅ 现有 EM/F1 主表、IHR 表、消融轴全部有效 |
| B | 纯训练期正则（`inject_kg=False`） | "SFT+KG 优于 SFT" 整组结果失效，主消融轴要换成"PPO 有无 KG 奖励分支"，**全部实验重设计** | ❌ 现有主表作废 |
| C | 纯推理期增强（砍训练期 KG 奖励） | 方法名与三阶段框架名不副实 | ❌ 方法失去 novelty |

**推荐 A 的理由**：与代码和已有数据完全吻合，且不损失卖点——"KG 既是训练期的过程监督信号、又是推理期的结构化证据"本身是可辩护的设计，只要**诚实声明推理期需要 KG 访问**（离线缓存 35K+ 实体 + 64K+ SPARQL 结果，断网可运行，这一点反而是工程亮点）。

**若选 A，需同步改的文件**：`RESEARCH_WORKFLOW.md` §3.4 解除"待定调"标注；`docs/paper/00_abstract.md` 与 `01_introduction.md` 的 KG 定位句；`06_conclusion.md` 的 Limitations 增加"推理期依赖 KG 访问"。

**验收**：论文方法节 + 摘要的 KG 定位表述与实验设计一致，无自相矛盾。

---

### R-1 — 🔴 修复 PPO 步数塌缩（本轮新增，最高优先级）

**问题**：见 §0.1。策略在 ~150 次更新内从 3 步收敛到 2 步（= `min_valid_steps` 下限），因为多写一步的奖励收益 < KL 成本。这同时压低 EM（推理被砍）、抬高 IHR（PPO② HotpotQA 18%→25%）、饿死 KG 过程奖励（作用面从 3.36 步缩到 2 步）。

**这是"EM 和 IHR 双输"的共同根因，也是本轮方案与上一版的最大差别。**

**候选方案（互不排斥，建议 R-1a + R-1b 组合，各自单独可回退）**：

| 编号 | 方案 | 改动 | 预期 | 风险 |
|---|---|---|---|---|
| **R-1a** | `min_valid_steps: 2 → 3`，对齐银标中位步数（3 步，占 accepted 的 72%） | `configs/training/phase3_ppo.yaml` 一行 | 门槛直接抬到 3 步，塌缩下限上移 | `valid_rate` 可能先跌（R7-A 曾在 `min_valid_steps=3` 下卡在 9.2%）——**但那次是 `outcome_weight=10 / step_scale=0.3` 的配置，与现在 4.0/1.5 不同**，需 smoke 验证 |
| **R-1b** | 加**步数不足惩罚**：轨迹步数 < `target_steps`(=3) 时，按缺口线性扣分（如每缺一步 −0.5），与 `trajectory_valid` 门**解耦** | `composite_reward.py` 末步修正处加一项；`reward_function.py` 传 `target_steps` | 让"少写一步"有明确代价，而不是靠硬门 | 需新增一个奖励项 → **AGENTS.md §8 需你确认**（核心 reward 改动） |
| R-1c | 提高 `step_reward_scale`（1.5 → 2.5），放大每步收益使其超过 KL 成本 | yaml 一行 | 最小改动 | 同时放大 `r_text` 噪声；且对**零引用步**无效（那些步 $r_{KG}=0$），不能根治 |
| R-1d | 降 `kl_coef`（0.15 → 0.08）减少长输出的惩罚 | yaml 一行 | 缓解成本侧 | 放大策略漂移风险（PPO② KL 已到 29.4） |

**推荐组合：R-1a + R-1b**，理由：R-1a 移动硬下限（便宜、可回退），R-1b 提供**连续梯度**引导模型写够步数（硬门只有 0/1 信号，PPO 学不到"再多写一步会更好"）。R-1c/R-1d 是备选旋钮，不首选。

**必须先做的验证（不可跳过）**：按 §3.4 的 smoke 流程跑 **≥150 次更新**（`SMOKE_TRAJ=600`；塌缩发生在前 ~150 更新内，更短的 smoke 看不出来），读 `history.jsonl` 的 `n_steps_sample / 4`：
- 若末段 ≥ 2.8 且 `valid_rate` ≥ 0.85 → 通过，进 P3 全量；
- 若 `valid_rate` 崩到 <0.5 → R-1a 太激进，回退到 `min_valid_steps=2` 只留 R-1b；
- 若步数仍 ~2.0 → 惩罚力度不够，加大 R-1b 系数或叠加 R-1c。

**验收**：新 PPO run 的 `n_steps_sample/4` 末 10% 均值 **≥ 2.8**（接近银标 3.36），且 `valid_rate` ≥ 0.85、`kg_reward_share` ≥ 0.10。

---

### R-2 — 修训练期 KG 索引 0% 命中 ✅ **2026-08-23 全部关闭**（本机已建 → 远端已装 → 远端实测 ABSENT 0.00%，见本节末尾）

**问题**：见 §0.2。索引是 dev-split 构建的，银标是 train-split，精确匹配命中 0/4000。

**方案**：

1. **用银标问句直接建索引**（✅ 本轮核实脚本已原生支持，最干净）：
   ```bash
   python scripts/prepare/06_build_question_kg_index.py \
     --silver data/silver_data/silver_v1_reannotated.jsonl \
     --min_keep 5 --max_keep 12 \
     --output indexes/kg_cache/question_kg_index_v2_train.json \
     --report docs/kg_build_report_train.md
   ```
   `--silver` 的帮助文字就是 "Silver .jsonl to take questions from (**covers the PPO prompt set**)" —— 它按银标问句建索引，**天然 100% 命中**，不需要猜 split 前缀。这比 `--datasets X --split train` 更可靠（后者要求数据集问句与银标问句逐字一致）。
   注意把 `--max_keep` 设成 **12** 而不是默认的 30，与推理侧对齐（见 R-3）。
2. **让 PPO 能指定索引路径**：`phase3_ppo.py:797` 目前**硬编码** `question_kg_index_v2.json`。加一个 `kg_index_path` 字段到 `Phase3PPOConfig`，并在 `scripts/train/phase3_ppo.py` 里**显式转发**（注意 `schemas.py` 是 `extra="allow"`，YAML 里加 key 会被静默忽略——这正是 `ppo_max_kg_triples` 曾经踩过的坑，见 yaml 末尾注释）。
3. **统一 fallback 与命中路径的过滤参数**：索引构建用 `min_keep=5 / max_keep=?`，fallback（`phase3_ppo.py:369`）用 `filter_and_rank_triples(max_keep=ppo_max_kg_triples)`。两条路必须用同一组 `min_keep`/`max_keep`，否则又是一个隐藏错位。

**备选（更省事，仍可接受）**：**放弃索引路径，只用银标 `kg_subgraph` + 统一过滤**。银标 accepted 的 `kg_subgraph` 均值 38.3 条、仅 6.8% 为空，质量够用；只要 fallback 的过滤参数与推理侧对齐（`max_keep=12, min_keep=5`），训练/推理分布即一致。**代价**：训练侧的子图来自 Phase-1 当时的检索/链接，推理侧来自 dev 索引，两者构建时刻不同——比走 `--silver` 重建索引多一层不可控。**推荐优先走上面的 `--silver` 方案。**

**验收**：训练日志里 `question_kg_index missed N/M` 的 miss 率 <10%（走索引方案），或明确记录"训练侧统一走银标 subgraph + `max_keep=12`"（走备选方案）；两种情况下训练 prompt 的三元组数分布与推理侧（均值 ~10-12）可比。

#### ✅ R-2 已执行（2026-08-22）

按方案 1 走。实测记录：

| 项 | 值 |
|---|---|
| 构建命令 | `06_build_question_kg_index.py --silver data/silver_data/silver_v1_reannotated.jsonl --min_keep 5 --max_keep 12`（offline，未联网） |
| 耗时 | **174 s**（全量 24,997 问句；300 问句探针 2.3 s） |
| 产物 | `indexes/kg_cache/question_kg_index_v2_train.json`（42 MB） |
| 报告 | `docs/kg_build_report_train_silver.md` |
| 三元组 | 1,178,718 → 197,852（删 83.2%）；均值 47.2 → **7.92**，中位 **11**，上限 12 |
| 分类关系占比 | 25.4% → **2.4%** |
| 实体链接 | 38,156/61,149 高置信，22,993 abstain |

**覆盖率（对 `silver_v1_reannotated` 全部 24,997 问句精确匹配）**：

- **ABSENT = 0.00%**（0/24,997）——旧的 `question_kg_index_v2.json` 是 **100% ABSENT**；
- COVERED-BUT-EMPTY = 23.32% 全量，但对 PPO 真正 rollout 的 **9,839 条 accepted 只有 9.7%**，经银标 `kg_subgraph` fallback 兜回 333 条后 **6.3%**。

**端到端验证**（真实 train fold + 新索引，无 GPU）：train fold 20,078/24,998，accepted **7,913** → 建出 7,913 条 prompt，其中 **7,319（92.5%）带非空 KG block**。改动前这一路是 0%（全部落 fallback）。

**由此发现并修掉的一个自造 bug**：`max_kg_index_miss_rate: 0.05` 原本比对的是 absent+empty 之和。正确重建后 absent 是 0% 但 empty 仍有 9.7%，**卡口会在"已修好"的状态下把训练拦下来** —— 这种 guard 唯一的过法是把它关掉，等于没有。现改为**只比对 ABSENT**（`phase3_ppo.py` 里 `dyn_kg_absent` / `dyn_kg_empty` 分开计数），并在 `tests/test_kg_index_guard.py` 加 5 条回归测试钉住这个区分（含"报错信息必须给出重建命令"一条）。

> empty 不随重建下降——它是 KG 和链接器的性质（链接 abstain，或缓存里那些 QID 没有三元组），不是索引的性质。所以它不该进卡口。

#### ⛔ smoke run 未启动：本机显存不够（2026-08-22）

> **2026-08-23 状态更新**：本节结论仍然成立（本机永远跑不了 PPO），但它已不是当前的阻塞项。
> 远端机器已修复并完成索引+代码部署，当前唯一阻塞是**远端容器没挂 GPU**——详见本节末尾
> 「R-2 索引问题已在远端落地并实测通过」。本小节保留作为"为什么必须用远端"的论证。

**模型是齐的** —— `models/rearag-9b`（20 G）和 `models/llama3-8b`（15 G）都在仓库根下（不在 `/models`，也不在 HF 缓存；第一次查找因此误判为缺失，已由用户指正）。**唯一的阻塞是显存**：

| 需要 | 本机实际 |
|---|---|
| GPU ~96 GB（PPO 实测峰值 95.7 / 97.9 GB） | **RTX 4090，24.5 GB，单卡** |
| `rearag-9b`、`llama3-8b` | ✅ 都在 `kgpaper/models/` |
| `REMOTE_ROOT=/root/autodl-tmp/kgpaper`（两个 launcher 硬编码并 `cd` 进去） | **权限不足，不可访问** |

把账算清，免得反复讨论"能不能压一压跑起来"：

| 常驻项 | bf16 |
|---|---|
| policy（LoRA，base 冻结但**权重仍要加载**） | 16.1 GB |
| 冻结 SFT reference（`create_reference_model` deepcopy，`phase3_ppo.py:328`） | 16.1 GB |
| **小计（已开 LoRA + gradient checkpointing + `--text_reward_backend dummy`）** | **32.1 GB** |
| ReaRAG-9B（真实 `r_text` 所需） | +18.8 GB → 50.9 GB |

**32.1 GB 已经超过 24.5 GB 的卡**，而这是在完全不算激活、rollout 的 KV-cache、优化器状态的前提下；TRL 训练前向还会为单条序列保留完整 logits，6400 tok × 128256 vocab × 4 B = **3.28 GB 的单次分配**（见 §0.3 的 OOM 分析）。`phase3_ppo.py:~310` 的注释把设计意图写得很明白：policy + frozen reference + ReaRAG-9B "co-reside on one **96GB** card"。

降 reference 不是选项——去掉它就等于把 KL 锚到裸 base，会把策略从 SFT 学到的行为上拽走，正是 2026-06-22 特意修掉的问题。

所以 smoke run 必须在 Blackwell 那台跑。本轮把**不需要 GPU 就能验的部分全部验掉**了，让那边启动后要么正常跑，要么因为真实原因失败，而不是因为准备工作没做完：

1. **索引路径能被 smoke 脚本读到**：脚本传 `--config configs/training/phase3_ppo.yaml` 且不覆盖 `question_kg_index_path`，因此继承 YAML 的新值。
2. **相对路径在两台机器上都能解析**：脚本 `cd $REMOTE_ROOT` 后 `indexes/kg_cache/...` 相对 CWD 解析。已加 `KGPW_INDEX_DIR` 兜底 + 报错信息里带 `cwd=` / `index_dir=` / 重建命令，避免在只设了环境变量的机器上白失败一次。
3. **`check_ppo_smoke.sh` 补了本轮改动对应的判据**。原来的 0–7 项全是为 R10（奖励**缩放**）写的，而本轮改的是奖励**奖什么**，两个新失效模式一个都测不到——**一次仍在塌缩的 run 可以打印 "ALL CHECKS PASS"**。新增：
   - **check 8**：update 40–60 的 `valid_rate > 0.5`，即 YAML 里写的 abort 判据（R7-A 复现检测）；
   - **check 9**：steps/traj 的**首三分之一 vs 末三分之一**趋势；
   - **check 10**：确认加载的是 train-fold 索引而不是 dev 索引，并回显 absent/empty 拆分。
4. **用合成日志双向验过 checker**：健康 run（3.25 步稳定）PASS；塌缩 run（3.4→3.0）FAIL。

> check 9 的判据在这里改过一次：初版写成"末段 ≥ 2.7"，结果一条收敛到**正好 3.00** 的塌缩日志照样 PASS —— 而"正好写到门槛"就是病灶本身（PPO①/② 是在 gate=2 下收到 2.04/2.01，同一个行为在 gate=3 下就是 3.0）。改成对**门槛的相对余量**（末段 − `min_valid_steps` ≥ 0.2 且降幅 < 0.4），两种日志才区分开。

**在那台机器上的启动顺序**：

```bash
# 1) 索引：**本机已建好，直接传过去，不需要在那台重建**（见下方"为什么可以传"）
#    本机产物：indexes/kg_cache/question_kg_index_v2_train.json
#    md5 4981fa028d06c1d2f20107ba50e732ee，42 MB → gzip 5.9 MB
gzip -c indexes/kg_cache/question_kg_index_v2_train.json > /tmp/qkg_train.json.gz
scp /tmp/qkg_train.json.gz REMOTE:$REMOTE_ROOT/indexes/kg_cache/
# 在那台：
#   gunzip -c indexes/kg_cache/qkg_train.json.gz \
#     > indexes/kg_cache/question_kg_index_v2_train.json
#   md5sum indexes/kg_cache/question_kg_index_v2_train.json   # 必须一致
#   md5sum data/silver_data/silver_v1_reannotated.jsonl       # 必须 8922e56e9dc901675b4aa870a19e6647
#
# 只有当那台的 silver md5 与本机不同时，才需要在那台重建：
#   python scripts/prepare/06_build_question_kg_index.py \
#     --silver $REMOTE_ROOT/data/silver_data/silver_v1_reannotated.jsonl \
#     --min_keep 5 --max_keep 12 \
#     --output $REMOTE_ROOT/indexes/kg_cache/question_kg_index_v2_train.json \
#     --report docs/kg_build_report_train_silver.md        # 离线，~174 s

# 2) smoke run（600 traj = 150 updates @ batch_size 4）
SMOKE_TRAJ=600 bash launch_split_ppo_smoke.sh

# 3) 判定
bash check_ppo_smoke.sh
```

**为什么索引可以本机建、传过去用**（原先这里写"必须在那台重建"，是多余的谨慎）：

| 检查项 | 结果 |
|---|---|
| 文件内是否含绝对路径 | **0 处**（`grep -c '/home/\|/root/\|/tmp/'`）——纯 JSON，key 是 question 字符串 |
| loader 是否依赖本机状态 | 否。`phase3_ppo.py:888-940` 只用 `question` 字符串做 key，不碰任何路径 |
| 构建是否需要联网/GPU | 否，offline + CPU，~174 s |
| 传输成本 | 42 MB → **5.9 MB** gzipped |

唯一前提是**两台机器的 silver 文件必须是同一个**（索引的 key 就是它的问句）。所以传完先比 md5：本机
`silver_v1_reannotated.jsonl` = `8922e56e9dc901675b4aa870a19e6647`。若不一致，索引对不上问句，
`max_kg_index_miss_rate: 0.05` 会直接把 run 拦下来（这正是它存在的意义——PPO①/② 的 100% miss
当年只是一条没人看的 warning）。

若 check 8 FAIL（abort 判据命中）：把 `min_valid_steps` 退回 2、保留 `shortfall_coef: 0.25`，重跑 smoke。**不要**"再等等看"——R7-A 已证明这条平坦曲线不会自己恢复。

#### ✅ R-2 索引问题**已在远端落地并实测通过**（2026-08-23）

上面的 §R-2「已执行（2026-08-22）」只证明了索引在**本机**建对了。本轮把它送到远端并验证它在**那台机器上真的会被用到**——这两件事是分开的，第二件差点白做（见下面的"顺带发现"）。

**解决方案（实际执行的，与上面计划的 gzip+scp 略有不同）**：直接 `rsync -aL`（42 MB，3 s，走 `sshpass -f` 读密码文件而非命令行，避免密码进 `ps`），省掉 gzip 一步。

**效果（全部在远端实测，非推断）**：

| 验收项 | 结果 |
|---|---|
| 文件完整性 | 远端 md5 `4981fa028d06c1d2f20107ba50e732ee` = 本机，逐字节一致 |
| JSON 可解析 + 格式匹配 loader | ✅ 24,997 条；含 `builder_version` 字段 → 走 `phase3_ppo.py:928` 的 v2 富格式分支（`triples` 为 `{h,r,t}` dict） |
| **silver 同源**（索引 key 的前提） | ✅ 远端 `silver_v1_reannotated.jsonl` md5 = `8922e56e9dc901675b4aa870a19e6647` = 本机 |
| **ABSENT 率**（`max_kg_index_miss_rate: 0.05` 的卡口） | **0.00%**（2,430 条 accepted prompt 抽样，远端 venv 实跑） |
| covered-but-EMPTY | **3.17%**，低于本机全量测得的 9.7%，传输未退化 |
| config 是否真的指向它 | ✅ 远端 yaml 第 52 行 `question_kg_index_path: indexes/kg_cache/question_kg_index_v2_train.json`、第 64 行 `max_kg_index_miss_rate: 0.05` |

**⚠️ 顺带发现并修掉的一个更大的问题：远端是个没有 `.git` 的旧快照，只传索引等于没传。**

远端 `configs/training/phase3_ppo.yaml` 当时读作 `split: null`、`min_valid_steps: 2`，且**完全没有** `question_kg_index_path` / `max_kg_index_miss_rate` / `shortfall_coef` / `target_steps` 这四个 key —— 也就是说 2026-08-22 那批步数塌缩修复和防 split 泄漏的改动，一条都不在那台机器上。若当时直接启动 smoke run，会是：**在 val/test fold 上训练（重演 PPO② 的泄漏）+ 用旧 gate=2（塌缩照旧发生）+ 索引根本不被读取**，而 `check_ppo_smoke.sh` 因为也是旧版（没有 check 8/9/10），会打印 "ALL CHECKS PASS"。

已同步（旧 config 备份为 `configs/training/phase3_ppo.yaml.pre20260823.bak`）：

| 目标 | 文件数 |
|---|---|
| `kgproweight/` | 9（含 `training/phase3_ppo.py`、`reward/composite_reward.py`、`kg/entity_linker.py`） |
| `configs/` | 1（`training/phase3_ppo.yaml`） |
| `scripts/` | 21（含新增的 `analysis/paired_stats.py`、`analysis/alpha_diagnose.py`） |
| launcher / checker | 4（`launch_split_ppo{,_smoke}.sh`、`launch_split_sft.sh`、`check_ppo_smoke.sh`） |

8 个关键文件同步后 md5 与本机逐一相同（`diff` 无输出）。

**远端 launcher 前置条件核实**（全部齐备，卡一到位即可启动）：

| 前置 | 状态 |
|---|---|
| `models/rearag-9b`(20 G) / `llama3-8b`(15 G) / `e5-base-v2`(419 M) | ✅ 在 `/root/autodl-tmp/models/` |
| `checkpoints/sft_student_split/final` | ✅ |
| `checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt` | ✅ |
| venv `/root/autodl-tmp/kgpw_env/bin/python`（torch 2.1.2+cu121） | ✅ |
| 磁盘 | 50 G 可用 / 127 G |

**⛔ 唯一剩下的阻塞（不是索引问题）：那台容器没有挂 GPU。**

| 证据 | 值 |
|---|---|
| `/usr/bin/nvidia-smi` | **0 字节空文件**（`Jan 8 2024`），执行返回 rc=0 但 stdout/stderr 皆空 |
| `/dev/nvidia*` | 不存在 |
| venv torch | `2.1.2+cu121`，`cuda.is_available() = False`，`device_count = 0` |

这不是容器内能修的（镜像里 `nvidia-smi` 是占位文件，说明实例是以无卡模式启动的），需要在 AutoDL 控制台切换到带卡的开机模式。**除此之外一切就绪**，卡接上后 smoke run 就是两条命令：

```bash
SMOKE_TRAJ=600 bash launch_split_ppo_smoke.sh   # 600 traj = 150 updates @ batch_size 4
bash check_ppo_smoke.sh                          # 重点看 check 8 / 9 / 10
```

> `SMOKE_TRAJ` 必须 ≥ 240：check 8 的 abort 判据是在 update 40–60 这一段上测的，脚本默认的 80 traj 只有 20 次更新，那一项会直接报 UNKNOWN。600 给出 150 次更新，同时覆盖 §0.1 说的"塌缩发生在前 ~150 更新内"。


---

### R-3 — 钉死三元组预算 + artifact 对齐 + 消除 split 泄漏

**问题**：见 §0.3。三处不一致。

**方案**：

| 项 | 目标值 | 怎么做 |
|---|---|---|
| `ppo_max_kg_triples` | **12**（与推理 `max_kg_triples=12` 一致） | dataclass 默认已是 12（`phase3_ppo.py:178`）✅，但 **yaml 末尾注释写"fixed at 30"是错的，必须改**；并在 `manifest.json` 里核对实际值 |
| `sft_checkpoint` | `checkpoints/sft_student_split/final` | `launch_split_ppo.sh:98` 已正确 ✅ |
| `alpha_gate_path` | `checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt` | `launch_split_ppo.sh:102` 已正确 ✅ |
| `split` | **`train`** | `launch_split_ppo.sh:120` 已有 `--split train` ✅；**但 yaml 里 `split: null`**，若不带 CLI flag 就会重演 PPO② 的泄漏。建议把 yaml 改成 `split: train`，让默认就是安全的 |
| `outcome_weight` / `step_reward_scale` | 4.0 / 1.5 | yaml 已回退 ✅ |

**关键**：PPO② 的 `split=None` 意味着它在 val/test fold 上训练过。**PPO② 的所有 EM/F1/IHR 数字都受此污染，论文里不能作为"PPO 结果"引用**，只能作为"一次配置错误的训练记录"（AGENTS.md §4 要求失败实验不得删除，但也不得当作有效结果）。

**验收**：新 run 的 `manifest.json` 里 `split=train`、`ppo_max_kg_triples=12`、`outcome_weight=4.0`、`step_reward_scale=1.5`、`sft_checkpoint` 与 `alpha_gate_path` 指向 split 版 artifact。

---

### P1 — 银标 KG 有效率低（你问的第 1 个问题的正面回答）

**问题定量**（本轮全量核实，见 §0.7）：

- accepted 步里只有 **51.1%** 引用了 KG，另 48.9% 是 `Knowledge Used: []`；
- 这 48.9% 零引用步**在 PRM 标注下必然是 NEU（$r_{KG}=0$）**——本轮交叉表核实：零引用步中 NEU 占 46.8%、NEG 2.1%、**POS 0%**（数学上不可能有 POS，因为 precision 无引用可算）；
- 有引用的步里 POS 占 25.5%、NEU 23.3%、NEG 2.2% —— **引用了就有一半概率拿到正信号**；
- 所以 `kg_reward_share` 的杠杆确实是**引用频率**（代码注释已确认，本轮数据也确认），不是引用精度。

**为什么引用率只有 51%**（三层根因，按可操作性排序）：

1. **teacher prompt 里 KG 引用是"自愿"的**：`_needs_format_retry`（`phase1_distill.py:541`）只在**整条轨迹一个引用都没有**时才 retry；某步空引用不触发。retry hint（`:559`）文字上要求"each step must cite at least one triple"，但触发条件太宽。
2. **20% 的题 `kg_subgraph` 本身是空的**（全体 4,998/24,998；accepted 内降到 6.8%）——这部分是实体链接/Wikidata 覆盖上限，teacher 无从引用。
3. **弱教师**：`teacher_model` 全部是 `deepseek-chat`；拒绝原因里最大一坨是 `answer_score=0.00`（6,294 条，占被拒的 41.5%），即**答案本身就错**。

**方案（按性价比排序）**：

| 编号 | 方案 | 改动量 | 预期收益 | 成本 |
|---|---|---|---|---|
| **P1-a** | **收紧 retry 触发条件**：`_needs_format_retry` 改为"KG 非空且**任一** step 零引用"就 retry（当前是"所有 step 都零引用"） | `phase1_distill.py` 一个函数 | accepted 引用率 51% → 目标 85%+ | 每题多一次 teacher 调用（约 +30-50% API 成本） |
| **P1-b** | **放宽 `sparse_quota`**：`sparse_quota_full` 白丢了 4,726 条（占被拒 31.2%）。改为"保留但降权"而非静默丢弃 | `phase1_distill.py` StratifiedAcceptFilter | accepted 池 +最多 4,726 条 | 稀疏样本比例上升，可能稀释 KG 信号——**需与 P1-a 一起做**，否则只是加噪声 |
| **P1-c** | **放宽 `min_steps` 3→2** | 一行 | 挽回 3,811 条 `step_count=2` | ⚠️ **与 R-1 冲突**：R-1 要抬步数，这条要降门槛。**不建议**，除非只对 MuSiQue 放宽 |
| **P1-d** | **换强教师重生成**（`deepseek-v4-pro` 或更强）+ `--split train` + wiki18 + rerank-10 | 重跑 Phase 1 | 答案命中率↑ → 接受率从 39.4% 抬升；引用质量↑ | 💰 API 成本 + 数天时间，**需你确认预算** |
| **P1-e** | **三数据集覆盖**：当前银标 100% hotpotqa。用 `data/2wikimultihopqa/train.jsonl`（167,454 条）+ `data/musique/train.jsonl`（19,938 条）扩充 | 同 P1-d | 解决 PPO 只在 hotpotqa 训练导致的 OOD 崩塌（MuSiQue −17pp） | 同 P1-d |

**推荐**：**P1-a + P1-b 先做**（改动小、不需重跑全部、可在现有 24,998 条上增量重生成被拒部分），P1-d/P1-e 作为并行长期线由你决定是否投入预算。

**重生成时的硬约束（上一版已列，本轮确认仍然必要）**：
- 必须 `--split train`，禁用 `--allow_eval_split`；
- 必须 wiki18 语料 + `--rerank 10`，禁止落到 `indexes_smoke`（989 文档）——见 §0.8 的 `*_20260801_*` 反面教材；
- manifest 必须记录 teacher 真实身份、语料版本、rerank 设置。

**验收**：新银标 accepted 内引用率 ≥85%；PPO 训练时 `kg_reward_share` ≥0.15（当前 PPO① 是 0.09）；无 dev 泄漏（qid 全 `train_`）；`teacher_model` 字段与实际调用一致。

---

### P2 — IHR judge 默认值修复（小改，但影响 P5 正确性）

**问题**：见 §0.8。`scripts/eval/run_ihr_judge.py:31`（主方法用）默认 `gpt-4o-2024-08-06`，`scripts/eval/run_baseline_ihr.py:143`（基线用）默认 `deepseek-v4-pro`。两个脚本默认值不同，是历史 judge 口径分裂的直接原因。

**方案**：把 `run_ihr_judge.py` 的默认改成 `deepseek-v4-pro`，与基线脚本一致。**同时**在两个脚本的输出 JSON 里已有 `judge_model` 字段（已确认存在）——P5 重判后逐个核对该字段。

**验收**：`grep -h "judge_model" outputs/**/ihr_result*.json` 全部为 `deepseek-v4-pro`。

---

### P3 — PPO 重训（你问的第 3 个问题）

**前置**：R-1（步数）+ R-2（KG 索引）+ R-3（预算/artifact/split）全部完成并通过 smoke。**AGENTS.md §8：大规模训练需你确认后才启动。**

**目标配置（一次只动"修复项"，不动其它研究变量）**：

| 配置项 | 值 | 与 PPO① 的关系 |
|--------|-----|---------------|
| `outcome_weight` | 4.0 | 同 PPO①（Plan B 已回退） |
| `step_reward_scale` | 1.5 | 同 PPO① |
| `text_reward_scale` | 0.3 | 同 PPO① |
| `sft_checkpoint` | `checkpoints/sft_student_split/final` | 同 PPO① |
| `alpha_gate_path` | `prm_alpha_gate_v1reann_negfix/alpha_gate.pt` | 同 PPO① |
| `split` | `train` | 同 PPO①（PPO② 是 None = 泄漏） |
| `ppo_max_kg_triples` | **12** | **改**（PPO① 是 30；对齐推理） |
| `min_valid_steps` | **3** | **改**（R-1a；PPO①/② 都是 2） |
| 步数惩罚 | **新增**（R-1b） | **新** |
| `kl_coef` / `target_kl` | 0.15 / 40.0 | 同 PPO① |
| `total_steps` / `batch_size` | 3000 / 4（= 750 updates） | 同 PPO① |
| `max_steps` | 5 | 同（0.6：从未触及，无需改） |

> 本次动了 3 个变量（三元组预算、步数门槛、步数惩罚）。按 AGENTS.md §4，这是**组合实验**，必须在 Experiment ID 里标明，不能表述为单变量对比。若你希望严格单变量，需拆成 2-3 次训练（每次 ~4h），我建议先跑组合版拿到方向，再按需拆解。

**执行环境**：远端 AutoDL 96GB（`launch_split_ppo.sh` 的 `REMOTE_ROOT=/root/autodl-tmp/kgpaper`）。本机 4090 24GB **跑不了**完整 PPO（TRL 训练前向保留全 logits，单序列 6400 token 就要 3.06 GiB 一次性分配，见 yaml 注释 + 记忆 `trl-ppo-logits-oom`）。需回传 `history.jsonl` + `tensorboard/` + `samples/`。

**训练期必须盯的四个指标（按重要性）**：

| 指标 | 目标 | 危险信号 | 触发的动作 |
|------|------|---------|-----------|
| **`n_steps_sample / 4`** | 末段 **≥2.8** | 跌到 ~2.0 | R-1 没生效，停训、加大 R-1b 系数 |
| `kg_reward_share` | 均值 **≥0.10** | <0.05 | 银标引用率是瓶颈 → 回 P1 |
| `valid_rate` | ≥0.85 | <0.5 | `min_valid_steps=3` 太激进 → 回退到 2，只留 R-1b |
| `ppo_mean_kl` | 末值 <25 | >35 | 策略漂移过大 → 提 `kl_coef` |

（参照：PPO① 末段 步数 2.04 / share 0.087 / valid 0.96 / KL 19.1；PPO② 末段 2.01 / 0.010 / 1.00 / 29.4。）

**验收**：`manifest.json` 五配置与上表一致；四个指标全部落在目标区间；`samples/step_03000.txt` 里的轨迹**目视确认 ≥3 步**（PPO② 该文件是 4/4 全 2 步）。

---

### P4 — 重评（你问的第 4 个问题）

**关键更正（省一半工作量）**：见 §0.4。主方法现有数字**已经**是 temp=0 / n=300 / rerank-10 / wiki18 / seed 42 / 4090，与基线表**同口径可比**。所以 P4 **不需要**重跑 SFT 四臂，只需重跑**新 PPO checkpoint** 的臂。

**需要跑的 cell（6 个）**：

| 臂 | hotpotqa | 2wiki | musique |
|---|---|---|---|
| PPO_new + KG | ✅ 跑 | ✅ 跑 | ✅ 跑 |
| PPO_new noKG | ✅ 跑 | ✅ 跑 | ✅ 跑 |

**可直接复用的 cell（6 个，已核实同口径）**：

| 臂 | 数据集 | EM | F1 | 来源目录 | 实测 KG 三元组/题 |
|---|---|---|---|---|---|
| SFT noKG | hotpotqa | 0.3400 | 0.4558 | `wiki18_eval/sft_nokg` | 0.00 ✅ |
| SFT+KG | hotpotqa | 0.3533 | 0.4575 | `wiki18_eval/sft_split_v2` | 10.26 |
| SFT noKG | 2wiki | 0.3367 | 0.4265 | `statistics.md`（目录待补记） | — |
| SFT+KG | 2wiki | 0.3600 | 0.4280 | `wiki18_eval/2wiki_sft_kg` | 待补测（该目录无 `intermediate_data.json`） |
| SFT noKG | musique | 0.2067 | 0.3234 | `wiki18_eval/musique_nokg` | — |
| SFT+KG | musique | 0.2300 | 0.3147 | `wiki18_eval/musique_kg` | 6.47 |

> ⚠️ 两个待补的溯源：2wiki SFT noKG 的落盘目录、以及 2wiki SFT+KG 的 KG 注入量核实（`intermediate_data.json` 缺失）。写论文前必须补齐，否则 AGENTS.md §9 的 `Claim → Evidence` 链断在这里。

**命令模板**：
```bash
export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH=/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl
export KGPW_DENSE_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat
export KGPW_BM25_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/bm25
export CUDA_VISIBLE_DEVICES=0

# +KG 臂
python scripts/eval/run_kg_proweight.py \
  --checkpoint checkpoints/<新PPO>/final \
  --alpha_gate_path checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt \
  --datasets hotpotqa 2wikimultihopqa musique \
  --test_sample_num 300 --seeds 42 --rerank 10 \
  --save_root outputs/ppo_new_kg

# noKG 臂：同上 + --no_kg，--save_root outputs/ppo_new_nokg
```

**必须显式传的两个路径**（记忆 `checkpoint-cleanup-default-paths`：`checkpoints/` 清理后默认路径已 dangle）：`--checkpoint`、`--alpha_gate_path`。

**跑完必做的三项自检**（否则可能重演 MuSiQue 0-KG 的乌龙）：

1. **KG 注入量核实**：对每个 +KG 臂，从 `intermediate_data.json` 抽 `[Knowledge Graph Context]` 数三元组，确认均值 ~10-12、空 KG 比例与 §0.2 的索引覆盖一致（hotpotqa ~5%、2wiki ~3%、musique ~12%）。**若 musique 出现 300/300 空，说明又断网了，该 cell 作废。**
2. **docs 数核实**：`output.retrieval_result` 长度 == 10。
3. **manifest 核实**：`gpu_name` / `git_commit` 记录到实验日志。

**验收**：6 个新 cell 落盘 + 3 项自检通过；与 6 个复用 cell 合成完整 4 臂 × 3 数据集主表。

---

### P5 — IHR 统一口径重判（你问的第 2 个问题）

**问题**：见 §0.8。主方法 12 个 IHR 文件全是 `deepseek-chat` 判的，基线是 `deepseek-v4-pro`。**IHR 是本文的核心主张，而这张表现在跨不过 judge 这一关**——两个 judge 的数值不能横向比较，也就意味着"我们的 IHR 比基线低"这句话目前**无法成立**。

**方案（三步，顺序不能颠倒）**：

1. **先修 P2**（`run_ihr_judge.py` 默认 judge），否则忘传参就调了 gpt-4o。
2. **主方法 12 臂全部用 `deepseek-v4-pro` 重判**。注意有 4 个臂的推理链会因 P3 重训而变，所以：
   - SFT 4 臂（hotpotqa/2wiki/musique × KG/noKG，其中 musique noKG 与 KG 各一）：可用**现有** `intermediate_data.json` 重判，不必重跑推理；
   - PPO 6 臂：必须等 P4 产出新推理链后再判。
   ```bash
   python scripts/eval/run_ihr_judge.py \
     --predictions <run>/intermediate_data.json \
     --judge_model deepseek-v4-pro --sample 50 --seed 42
   ```
3. **补 rearag 基线**：2wiki / musique 的 IHR 正在跑（PID 1745597）。跑完核对 `judge_model` 字段。

**IHR 的不确定度（本轮新增要求）**：IHR 是 `random.sample(preds, 50)` 抽样，**这是唯一多 seed 有意义的地方**（EM/F1 在 temp=0 + first-N 下确定，见 §0.9）。目前所有 IHR 都是单次 seed 42，**核心主张没有误差棒**，审稿人必问。

方案：每个臂用 `--seed {42,123,2024}` 判三次，报 **均值 ± 标准差**。成本 = 3× judge 调用。若预算紧，至少对**主方法最佳臂 + 最强基线**（r1_searcher / rearag）做三 seed。

**验收**：所有 IHR 文件 `judge_model=deepseek-v4-pro`；主方法与基线可同表；关键臂有 ±std。

---

### P6 — MuSiQue 的 α-KG 矛盾（写作可信度）✅ **2026-08-23 已完成，结论是：矛盾不存在，因为旧 α 表本身无效**

**问题**：本轮核实出一个论文里必须主动解释的矛盾：

| 数据集 | α mean | 索引空 KG 比例 | 实测注入三元组/题 |
|---|---|---|---|
| HotpotQA | ~~0.292（最低）~~ → **0.918** | 5.5% | 10.7 |
| 2Wiki | ~~0.804~~ → **0.914** | 3.1% | 10.5 |
| MuSiQue | ~~0.855（最高）~~ → **0.908** | **12.4%（最差）** | **9.1（最少）** |

**结论（2026-08-23，`scripts/analysis/remeasure_alpha.sh` + `scripts/analysis/alpha_diagnose.py`）**：

原来设想的"矛盾"建立在旧 α 表之上，而**旧表的三个数字取自三次不同代码/缓存状态的运行，本身不可比**。
在单一固定状态下重测（每数据集 n=300、同一冻结 `entity_cache.jsonl` 快照、同一 bias、同一 checkpoint）：

| 数据集 | n(非空) | α mean | α sd | density | r(α,density) | 残差 mean(sd) |
|---|---|---|---|---|---|---|
| HotpotQA | 285 | **0.9183** | 0.0217 | 0.9242 | **+0.902** | +1.3564 (0.0550) |
| 2Wiki | 290 | **0.9143** | 0.0224 | 0.9061 | **+0.905** | +1.3454 (0.0577) |
| MuSiQue | 253 | **0.9084** | 0.0301 | 0.8851 | **+0.949** | +1.3357 (0.0534) |

跨数据集极差 **0.0099**，而单次运行内问题间 sd 最大 0.0301 —— **数据集之间的差异小于同一数据集内问题之间的差异**。
诊断脚本判定为 `comparable`（跨运行残差散布 0.021 < 最大组内 sd 0.058），这是三臂第一次落在同一把尺子上。

**上面"两种可能解释"中，第 1 条被证实、第 2 条被否证**：α 与 density 的相关系数 0.90–0.95，
而 density $=|E|/(|V|+\epsilon)$ 在稀疏子图上分母更小，所以 KG 越差 α 反而不降。
另外两个特征（link_confidence、语义熵）几乎不起作用 —— **门控实际退化为单特征密度函数**。
所谓"MuSiQue 需要 Wikidata 属性"的解释既无必要也无证据，已从 §5.2 删除。

**旧表为什么无效，有三条互相独立的证据**：

1. **代数不可达**。残差 $W_1\cdot conf + W_2\cdot ent$ 在 $conf,ent\in[0,1]$ 时必落在 $[W_2,W_1]=[-0.583,+1.709]$。
   α=0.292 在实测 density 0.9242 下需要残差 $-0.647$，**低于下界**，任何 $(conf,ent)$ 都解不出来；
   只有在 bias=0 下可达，而 `03_method.md` §3.4 声称已施加 +0.78 —— 两者不能同时为真。
2. **同数据集同 seed 四个 α**。归档的 08-07 hotpotqa 运行回解得 0.1251（BIAS MISMATCH + DEGENERATE，残差 −1.3633）、
   0.4405（NO LOGPROBS，conf≈0.105）、0.4925（NO LOGPROBS，conf≈0.191），加上新测 0.9183。
   **同数据集内的状态漂移 0.79 远大于旧表声称的跨数据集效应 0.563。**
3. **两个漂移源都已定位**。(a) `EntityLinker` 缓存为空时 `f_confidence` 恒等于 0
   （`entity_linker.py` 的 `if not cache_items: return 0.0`），而 `indexes/entity_cache.jsonl` 不入版本库、
   且在评测过程中自行增长（`cache.set()` 在 `link()` 路径上，不受 `KGPW_KG_OFFLINE` 约束）；
   (b) D1（commit `e6b2198`）把推理期硬编码的 `f_entropy = 1.0` 换成真实 logprob（实测 ≈0.604/0.623/0.639）。

**工具自校验**：在仅有 bias 标志差异的 `ab_bias` A/B 对上，诊断脚本回解出的残差偏移为 0.780，真值 0.78。

**顺带发现两个不依赖本次重测的问题**：
- +0.78 的原始理由（补偿硬编码 `f_entropy = 1.0`）已被 D1 移除，约 25–33% 重复计入。
  同一残差下 bias=0 给 α=0.7605、+0.78 给 0.9210，即校正单独贡献 +0.16。**须重新推导，不应沿用。**
- §5.4「IHR 降幅与 α 值一致」在论文自己的数字里就不成立：α 序 HotpotQA<2Wiki<MuSiQue，
  IHR 降幅序 HotpotQA(−1.5)<MuSiQue(−5.6)<2Wiki(−8.2)，三点里两点相反。已删除该断言。
- 旧表的 t 检验把「每问题多步骤」展平做两样本 t 检验，n 从 ~290 虚增到 ~900，低估方差。
  α 是嵌套数据，必须按问题聚合或用混合模型。

**已改的落点**：`paper/statistics.md` §一†/§二/§六/§八、`paper/05_results.md` §5.2/§5.4/§5.6(iv)、
`paper/00_abstract.md`、`paper/01_introduction.md`、`paper/06_conclusion.md`、`paper/03_method.md` §3.4、
`paper/口径统一清单.md` ③、`Modification.md`；`paper/paper.md`（08-18 两数据集旧稿）加废弃横幅。

**✅ 已完成（2026-08-23 03:16）**：`BIAS=0.0` 对照臂三臂跑完（`outputs/alpha_remeasure_bias0`），
脚本自判 `comparable`（残差跨 run 极差 0.021 < 最大 run 内 sd 0.058）。结果见
`statistics.md` §二.3 完整对照表。三条一致结论：

| 数据集 | α (bias=0) | α (+0.78) | 校正贡献 | r(α,density) bias=0 | α sd bias=0 / +0.78 |
|---|---|---|---|---|---|
| HotpotQA | 0.7563 | 0.9183 | +0.1620 | +0.922 | 0.0523 / 0.0217 |
| 2Wiki | 0.7466 | 0.9143 | +0.1677 | +0.924 | 0.0529 / 0.0224 |
| MuSiQue | 0.7341 | 0.9084 | +0.1743 | +0.960 | 0.0676 / 0.0301 |

1. 校正单独贡献 **+0.162~+0.174**，与解析预测 +0.16 吻合；两臂残差每个数据集都到四位小数一致
   （证明只差这一个变量）。
2. **α 仍是单特征密度函数**，且关掉校正后相关性更高（+0.960）—— 密度退化不是校正的假象。
3. **新增发现，指向改代码**：关掉校正后 α 的 sd 反而大一倍以上。`+0.78` 把 α 推进 sigmoid
   饱和区，**压掉了门控本就微弱的区分度**。此前反对沿用 `+0.78` 只有「与 D1 实测熵重复计入」
   一条（推导层面），现在多了实测层面的一条：**该校正主动削弱门控功能**。
   → **重训前应重新推导 `alpha_bias_correction`，不要沿用 +0.78**（与 D2「量纲统一」同批处理）。

**验收**：✅ 已产出相关性数据（r=0.90/0.91/0.95）；§5.2 不再声称跨数据集自适应，改为如实报告负面结果并指出根因。

#### ✅ P6 后续：`alpha_bias_correction` 已重新推导（2026-08-23）

上面第 3 条给出的行动项（"重训前应重新推导，不要沿用 +0.78"）本轮已做完。结论比预期更硬：
**+0.78 从一开始就不是推导出来的数字，它是为了把 `b_effective` 凑成整数 −1.0 而反解出来的。**

`b_trained = −1.7818`，而 `−1.7818 + 0.78 = −1.0018 ≈ −1.0`。代码注释本身就写明了这一点
（`kg_proweight_pipeline.py:141`："inference override → **beff=-1.0**"）——即目标是那个整数，
不是熵偏移量。

**校正的合理上界（按它自己声称的机理算）**：校正要补偿的是 `f_entropy` 在训练/推理两个 regime
之间的偏移 `Δe = e_inf − e_tr`。由于 α-logit 里这一项是 `W₂·f_entropy`（`W₂ = −0.5833`），
补偿所需的加性偏置是 `−W₂·Δe = 0.5833·Δe`：

| 情形 | `Δe` | 所需校正 | 已施加 | 倍数 |
|---|---|---|---|---|
| 原始前提（`e_inf` 硬编码 1.0，`e_tr ≈ 0.4`，见代码注释） | 0.60 | **+0.350** | +0.78 | **2.2×** |
| **理论上界**（取 `e_tr = 0`，`entropy_from_logprobs` 的下限） | 0.62 | **+0.363** | +0.78 | **2.15×** |
| **D1 之后的真实情形**（`e_inf ≈ 0.62` 实测，`e_tr ≈ 0.4`） | 0.22 | **+0.129** | +0.78 | **6.0×** |

`e_inf` 不是估的：`alpha_diagnose.json` 从三个 bias=0 run 的实际残差反解出
`f_entropy = 0.6035 / 0.6222 / 0.6388`（hotpotqa / 2wiki / musique），且诊断脚本同时证明
**`f_entropy = 1.0` 在可行域内无解**（那要求 `f_confidence = +1.135 > 1`）。所以：

1. **原始前提已被 D1 移除**。校正的理由是"推理时 `logprobs=None` 迫使 `f_entropy = 1.0`"，
   但 D1（commit `e6b2198`）让推理侧走真实 per-token logprobs
   （`kg_proweight_pipeline.py:314-338` 的 `return_scores=True`），实测 `f_entropy ≈ 0.62`。
   **前提消失了，校正还留着。**
2. **即使前提还成立，+0.78 也超了 2.2×**；取 `e_tr = 0` 这个最宽松的假设，上界也只有 **+0.363**。
3. **训练侧从不施加校正**：`phase3_ppo.py:787-791` 直接 `load_state_dict` 后就 `eval()`，
   没有任何 bias 改动。所以 +0.78 是纯推理期的单侧偏移 —— 训练与推理用的是**两个不同的 α 函数**，
   这本身就是一处 train/eval 错位。

**推荐取值**：**`alpha_bias_correction = 0.0`**（即彻底去掉），理由三条：

- D1 已经在**特征层面**修好了 regime 偏移，这是正确的修法；在偏置上再加一次是重复计入。
- 实测证据：关掉后 α 的 sd 从 0.0217/0.0224/0.0301 涨到 0.0523/0.0529/0.0676（大一倍以上），
  即校正把 α 推进 sigmoid 饱和区、**主动削弱了门控的区分度**。
- 去掉后训练与推理的 α 函数才一致（见上面第 3 点）。

若不愿一步走到 0，退而求其次的**上界是 +0.363**（`Δe` 取最宽松的 `e_tr = 0`），但没有任何测量
支持超过它的值。

> **注意这不改 EM/F1。** 已记录的事实：α-gate 在推理期是遥测量，不进入答案生成路径
> （见 memory `alpha-gate-eval-noop`）。所以本项的收益是**训练侧**的（PPO 的 `r_kg` 加权用的就是
> 这个 α）和**写作诚实度**的（§3.4 不能再声称 +0.78 有推导依据）。真正会动 EM/F1 的是 R-1/R-2/P1。

**落地方式**：`--alpha_bias_correction 0.0` 已经是 `run_kg_proweight.py:52` 的现成 flag（默认
`None` = 施加 +0.78）。建议把**默认值本身**改成 0.0，让"不校正"成为默认行为，而不是每次靠命令行
记得传 —— 与 §R-3 把 `split: null → train` 的处理方式一致（默认就应该是安全的那个）。

---

### P7 — 统计口径（EM/F1 与 IHR 的显著性） ✅ 2026-08-23 已完成

**EM/F1**（temp=0 + first-N，生成确定）：
- **不需要多 seed**（同一 checkpoint 同一数据跑两次结果逐位相同，`baseline_3seed` 的 seed 13 与 2024 EM 都是 0.36 已证实）；
- 但**n=300 的抽样误差仍在**：EM≈0.34 时二项 SE ≈ $\sqrt{0.34\cdot0.66/300}$ = 2.7pp，95% CI 半宽 ≈5.3pp；
- 正确做法：**逐题配对 McNemar**（+KG vs noKG 同题对比，比独立二项检验灵敏得多）+ **bootstrap 95% CI**。

**IHR**（n=50 抽样）：见下 (c)，判官本身有噪声，多 seed 有意义，见 P5。

#### P7-1 脚本（原计划"需新写"，实际已存在）

> ⚠️ **本节原文写错了。** 原文称"目前仓库里**没有**现成的 McNemar/bootstrap 脚本（本轮 grep 确认），需写一个"，
> 但 `scripts/analysis/paired_stats.py` **已经存在**（EM→精确二项 McNemar；F1/IHR→配对 bootstrap + Wilcoxon
> 符号秩；按 `id` 配对，只在一侧出现的题**丢弃并计数**，不静默取交集）。原文的 grep 结论不成立，
> 已按实测更正。IHR 走 `--metric ihr`，脚本显式说明"越低越好、delta = b − a、负值才是改善"。

#### P7-2 实测结果（2026-08-23，全部 12 组对比）

所有对比均为**同一 checkpoint 下 +KG vs noKG**，逐题配对、id 重叠 300/300（n=1000 组为 1000/1000）。
先核实过两臂确实构成消融：noKG 臂 `kg_subgraphs` 全空（0 三元组），+KG 臂注入
9231 / 7992 / 5823 条（hotpotqa / 2wiki / musique SFT）。原始 JSON 在 `outputs/paired_stats/`。

**EM（McNemar 精确检验）**

| 对比 | n | noKG | +KG | Δ | 95% CI | 不一致题数 (+KG 对 / noKG 对) | p | 判定 |
|---|---|---|---|---|---|---|---|---|
| HotpotQA SFT | 300 | 0.3400 | 0.3533 | +0.0133 | [−0.027, +0.053] | 36 (20/16) | 0.618 | NS |
| 2Wiki SFT | 300 | 0.3367 | 0.3600 | +0.0233 | [−0.030, +0.073] | 63 (35/28) | 0.450 | NS |
| MuSiQue SFT | 300 | 0.2067 | 0.2300 | +0.0233 | [−0.017, +0.063] | 37 (22/15) | 0.324 | NS |
| **2Wiki SFT n=1000** | 1000 | 0.3170 | 0.3440 | +0.0270 | [−0.003, +0.057] | 237 (132/105) | **0.091** | NS（最接近） |
| HotpotQA PPO | 300 | 0.3133 | 0.2133 | **−0.1000** | [−0.157, −0.043] | 78 (24/54) | **0.00090** | **显著变差** |
| 2Wiki PPO | 300 | 0.3200 | 0.3233 | +0.0033 | [−0.040, +0.047] | 43 (22/21) | 1.000 | NS |

**F1（配对 bootstrap + Wilcoxon）**

| 对比 | n | noKG | +KG | Δ | 95% CI | Wilcoxon p | 判定 |
|---|---|---|---|---|---|---|---|
| HotpotQA SFT | 300 | 0.4558 | 0.4575 | +0.0017 | [−0.035, +0.039] | 0.957 | NS |
| 2Wiki SFT | 300 | 0.4265 | 0.4280 | +0.0015 | [−0.050, +0.053] | 0.808 | NS |
| MuSiQue SFT | 300 | 0.3234 | 0.3147 | −0.0087 | [−0.048, +0.030] | 0.730 | NS |
| 2Wiki SFT n=1000 | 1000 | 0.3837 | 0.3965 | +0.0128 | [−0.016, +0.042] | 0.266 | NS |
| HotpotQA PPO | 300 | 0.4244 | 0.2853 | **−0.1391** | [−0.196, −0.080] | **9.2e−06** | **显著变差** |
| 2Wiki PPO | 300 | 0.4001 | 0.3979 | −0.0022 | [−0.043, +0.039] | 0.964 | NS |

**IHR（配对 bootstrap + Wilcoxon，n=50/臂，越低越好）**

| 对比 | noKG | +KG | Δ | 95% CI | Wilcoxon p | 判定 |
|---|---|---|---|---|---|---|
| HotpotQA SFT | 0.2212 | 0.2062 | −0.0150 | [−0.073, +0.045] | 0.639 | NS |
| 2Wiki SFT | 0.2873 | 0.2057 | −0.0817 | [−0.178, +0.010] | **0.088** | NS（最接近） |
| MuSiQue SFT | 0.3194 | 0.2633 | −0.0560 | [−0.145, +0.033] | 0.193 | NS |
| HotpotQA PPO | 0.1800 | 0.2500 | +0.0700 | [−0.050, +0.190] | 0.263 | NS |
| 2Wiki PPO | 0.1500 | 0.1300 | −0.0200 | [−0.090, +0.040] | 0.564 | NS |
| MuSiQue PPO | 0.5200 | 0.5200 | 0.0000 | [−0.030, +0.030] | n<6，未做 | NS（两臂预测逐位相同） |

#### P7-3 三条结论

**(a) 没有一个 KG 增益达到显著。** 12 组对比里只有一个 p<0.05，而它是**反向**的：
HotpotQA PPO 加 KG 后 EM −10.0pp（p=0.00090）、F1 −13.9pp（p=9.2e−06）。
不一致题数 78 里有 54 题是 noKG 对而 +KG 错，只有 24 题反过来 —— 这不是噪声，是真退化。
这独立支持了「KG 是 RL 必要正则化器」主张的撤回（§P6 结论、`06_conclusion.md`）：
在唯一一个统计上分得开的 PPO±KG 对比里，KG 是有害的。

**(b) 配对检验确实更灵敏，但还是不够。** 最接近显著的是 2Wiki SFT n=1000（Δ=+2.7pp，p=0.091），
把 n 从 300 提到 1000 使同一效应的 p 从 0.450 降到 0.091 —— 方向对，量级不够。
按 n=1000 的不一致率（23.7%）反推，要让 +2.7pp 达到 p<0.05 约需 **n≈1600–2000**。
这给 **D9（是否把 n 从 300 提到 1000）** 一个明确答案：**1000 不够，NS 会照旧**；
只有效应量本身变大（这正是 P1/P3 重训要做的事）才有意义，单纯加 n 是在给一个 +2.7pp 的效应买 SE。

**(c) 判官噪声已量化（新发现，本轮意外测出）。** MuSiQue PPO 两臂的**预测逐位相同**（50/50 完全一致），
但判官给出的 IHR 有 **2/50 题不同**（`dev_258` 0.0→0.5、`dev_214` 1.0→0.5）。
即 `deepseek-chat` 在同一输入上的自不一致率 ≈4%，单题 IHR 抖动可达 0.5。
n=50 时这本身贡献约 ±0.01 的均值噪声。**推论：IHR 差值小于 0.02 的一律不可解释**，
这个下界与 P5 要修的判官口径问题是两件事（那是跨判官不可比，这是同判官不可重复）。

#### P7-4 验收

- [x] 主表每个 KG 对比都有 McNemar p 与 bootstrap CI（12 组，见上三表；JSON 在 `outputs/paired_stats/`）
- [x] 论文不出现"显著提升"除非 p<0.05 —— 已核对 `05_results.md`/`00_abstract.md`/`06_conclusion.md`，
      当前正文对 KG 增益一律写 NS/不显著，无违规；唯一 p<0.05 的结论（PPO+KG 变差）已写进 Finding 2
- [x] 本节原文"没有现成脚本"的错误结论已更正
- [ ] **重训后须重跑本节全部命令**（P4 之后）。命令模板：
  ```bash
  python scripts/analysis/paired_stats.py \
    --a <noKG>/intermediate_data.json --b <withKG>/intermediate_data.json \
    --metric em --output outputs/paired_stats/<name>_em.json
  # metric ∈ {em, f1}；IHR 用 --metric ihr，输入 outputs/ihr_eval/*.json
  ```

---

## 3. 训练参数与奖励函数调整（你问的第 6 个问题）

### 3.1 当前奖励函数（按代码核实，2026-08-22）

逐步：
$$R_t = \bigl(\alpha_t \cdot r_{KG}(t) + (1-\alpha_t)\cdot r_{text}(t)\cdot c_{text}\bigr)\cdot c_{step},\quad c_{text}=0.3,\ c_{step}=1.5$$

末步（二者互斥，$\omega=4.0$）：`trajectory_valid` 为真 → $R_T \mathrel{+}= \omega\cdot\mathrm{EM}$；否则 $R_T \mathrel{-}= \omega$。

`trajectory_valid`（纯格式谓词，`reward_function.py:220`）：≥2 个可解析 `[Step N]`、有 `Final Answer`、步号连续、每步非空、每步 `Reasoning:` ≥20 字符。**没有独立格式奖励项**，格式全靠这个门。

### 3.2 本轮核实出的两个奖励侧代码问题

**(a) `discounted_returns` 算了但没用**（本轮 grep 确认）
- `reward_function.py:445` 算出 `returns` 并放进返回字典（`:486`），但**全仓库没有任何消费者**（`grep '\["returns"\]'` 在 reward_function.py 之外零命中）。
- PPO 实际用的是 `token_rewards`（把每步 $R_t$ 放到该步最后一个 token），折扣与优势估计由 **TRL 的 GAE**（`gamma`/`lam`）完成。
- **结论**：$\gamma=0.95$ 的手工折扣是**死代码**，真正生效的折扣是 TRL PPOConfig 的 `gamma`。论文若写"$G_t=\sum\gamma^{k-t}R_k$"需改成"per-step reward 交给 GAE"，否则是**方法描述与实现不符**。这条属于 AGENTS.md §1 的"不得把未验证描述写成事实"。**不改代码也行（无功能影响），但论文表述必须改。**

**(b) `kg_reward_share` 的分母有一个 always-1 项**
- `phase3_ppo.py:1046`：`total_mass = Σ|r_total| + outcome_weight * max(1, Σtraj_rewards > 0)`。第二个参数是 bool，所以 `max(1, ...)` 恒为 1，分母**总是**多加一个 `outcome_weight`。
- 代码注释明确说这是**故意**的（让 share 在跨 batch 时可比）且"不要修，否则 R10 的历史数字失去可比性"。
- **结论**：不改。但要知道 `kg_reward_share` 是一个**偏低的估计**（分母被人为抬高），"目标 ≥0.10" 这个阈值只在同一定义下有意义。

### 3.3 参数调整决策表（触发式，不预设）

原则（AGENTS.md §6）：**一次只改一个核心研究变量**；奖励函数的任何改动单开 Experiment ID。

| 症状（从 `history.jsonl` 读） | 判断 | 动作 | 优先级 |
|---|---|---|---|
| 步数/轨迹 末段 <2.5 | R-1 未生效 | 加大 R-1b 惩罚系数；或叠加 `step_reward_scale` 1.5→2.5 | 🔴 停训重来 |
| `valid_rate` <0.5 | `min_valid_steps=3` 太严 | 回退到 2，只保留 R-1b 连续惩罚 | 🔴 停训重来 |
| `kg_reward_share` <0.05 | 银标引用率瓶颈 | 回 P1-a（收紧 retry）；**不要**靠调 `step_reward_scale` 硬抬（对零引用步无效） | 🟡 记录，本次跑完 |
| `ppo_mean_kl` >35 | 策略漂移 | `kl_coef` 0.15→0.25 | 🟡 |
| `mean_reward` 单调升但 EM 降 | reward hacking | 查 `samples/` 目视；检查是否又在压步数或吐裸答案 | 🔴 |
| `r_text_mean` 远高于 `r_kg_mean`（当前 0.76 vs 0.09） | 文本通道主导 | 考虑 `text_reward_scale` 0.3→0.15，或 `alpha` 下限约束 | 🟢 未来实验 |
| `advantage_var` 接近 0 | 奖励无区分度 | 检查是否所有轨迹都拿一样的分 | 🟡 |

**明确不做的两件事**（有历史证据）：
1. **不要提 `outcome_weight`**。Plan B 的 8.0（以及更早建议的 10.0）已被证明把 `kg_reward_share` 压到 1.6%、三数据集全线低于 SFT、MuSiQue OOD 崩塌 −17pp。方向已封。
2. **不要降 `step_reward_scale`**。同上，Plan B 的 0.5 是同一次错误的另一半。

**`text_reward_scale` 的方向争议**（`口径统一清单` §②）：`fix_plan_kg_quality.md` 建议 0.3→0.7（抬文本），`solve suggestion.md` 建议压 KG。**两者都未执行，且方向相反**。本轮建议：**先不动**，等 P3 曲线出来看 `r_kg_mean` / `r_text_mean` 的实际比例再定。

### 3.4 Smoke 验证流程（每次改奖励/参数后必跑，~45 min）

```bash
# 远端。必须 >=150 次更新才能看出步数是否塌缩（§0.1：塌缩发生在前 ~150 更新内）。
# SMOKE_TRAJ 是"轨迹数"，batch_size=4 → updates = SMOKE_TRAJ/4。
# 脚本默认 SMOKE_TRAJ=80 只有 20 次更新，太短，看不出趋势。
SMOKE_TRAJ=600 bash launch_split_ppo_smoke.sh   # = 150 updates, ~45 min
# 判读
python - <<'EOF'
import json, statistics
rows=[json.loads(l) for l in open('outputs/split_ppo_smoke/history.jsonl')]
last=rows[-15:]   # 末 15 次更新求均值（单 batch 读数不可用：kg_reward_share
                  # 在相邻 batch 间实测从 0.024 跳到 0.213）
f=lambda k: statistics.mean(r[k] for r in last)
print("steps/traj %.2f (target>=2.8)"%(f('n_steps_sample')/4))
print("valid_rate %.2f (target>=0.85)"%f('valid_rate'))
print("kg_share   %.4f (target>=0.10)"%f('kg_reward_share'))
print("ppo_kl     %.1f (target<25)"%f('ppo_mean_kl'))
EOF
```
`check_ppo_smoke.sh` 已存在，可复用/扩展（它已做"3+ batch 求均值"，避免单 batch 读数——`kg_reward_share` 在相邻 batch 间曾从 0.024 跳到 0.213）。

### 3.5 GRPO 分支（备选，不在本轮计划内）

`kgproweight/training/phase3_grpo.py`（364 行）+ `configs/training/phase3_grpo.yaml` 存在，无 critic / 无 ref model，显存更省。**本轮不启用**：它丢掉 GAE 的逐步偏差校正，而本方法的卖点正是逐步过程奖励。可作为未来的消融（"过程奖励是否需要 critic"）。

---

## 4. 执行顺序

```
第 0 步 [你定]  P0 KG 定位（推荐选项 A）           —— 不阻塞训练
第 1 步 [代码]  R-1 步数塌缩修复（a+b）  ← 最高优先级，本轮新增   ✅ 已改（gate=3 + shortfall 0.25）
第 2 步 [代码]  R-2 训练 KG 索引（或备选：统一走银标 subgraph）    ✅ 已关闭（本机建 + 远端装 + 实测 ABSENT 0.00%）
第 3 步 [代码]  R-3 预算 12 / split=train / yaml 注释             ✅ 已改（三项均已落 yaml，且已同步到远端）
第 4 步 [验证]  smoke 150 updates（SMOKE_TRAJ=600），读四个指标（§3.4）
                ⛔ 当前卡在这里：远端容器未挂 GPU（nvidia-smi 是 0 字节占位文件），
                   需在 AutoDL 控制台切到带卡模式。其余前置全部就绪。
        ├─ 不达标 → 回第 1 步按 §3.3 调整
        └─ 达标 ↓
第 5 步 [你确认] P3 远端全量 PPO 重训（~4h, 750 updates）
第 6 步 [评估]  P4 重评 PPO 6 个 cell + 3 项自检（SFT 6 cell 复用）
第 7 步 [评估]  P5 IHR 统一 deepseek-v4-pro 重判（先修 P2）+ 多 seed
第 8 步 [分析]  P6 α-KG 相关性 + P7 McNemar/bootstrap
第 9 步 [写作]  主表定稿
并行长期线：P1 银标重建（P1-a/b 小改先做；P1-d/e 需你批预算）
```

**若想最快看到"PPO 不再退化"**：第 1-5 步是关键路径，代码改动半天 + smoke 半小时 + 训练 4h，**当天可出结论**。P1 银标重建是抬天花板的长期项，不阻塞。

---

## 5. 需要你拍板的决策点

| # | 决策 | 我的推荐 | 依据 | 影响面 |
|---|------|---------|------|--------|
| **D1** | KG 定位（P0） | ✅ **已决：选项 A**（训练期正则 + 推理期增强）。训练期三个病灶及解法见 §8 | 与代码/数据完全吻合，现有主表全部有效 | Abstract/Intro/Contribution |
| **D2** | R-1b（新增步数惩罚）是否批准 | ✅ **已批准**（2026-08-22），附加要求：改奖励时必须保证量纲统一。量纲实测与落地方案见 §9。**唯一待定：shortfall 锚点系数**（我建议 1/4 × outcome_weight） | §0.1 实测步数塌缩曲线 + §9 三通道量纲实测 | 核心 reward 改动（AGENTS.md §8） |
| **D3** | R-2 走"重建索引"还是"统一走银标 subgraph" | ✅ **已决：`--silver` 重建索引**。两条路径的功能差异很小（回退路径已就地补过滤），真正的差距是**可溯源性**：详见 §10 | §0.2 + §10 代码核实 | 训练/推理对齐 |
| **D4** | P3 是否启动重训 | 等 D2/D3 + smoke 通过后启动 | AGENTS.md §8 | 4h 远端算力 |
| **D5** | teacher 真实身份 | ✅ **已查清，无需外部确认**：`deepseek-v4-flash` 是一个从未被执行到的 argparse 默认值。证据链见 §11 —— 银标 100% 由 `deepseek-chat` 生成，文档的 v4-flash 是笔误 | §11（代码分支 + 启动脚本核实） | 论文 teacher 表述（改文档即可） |
| **D6** | P1 银标重建投入 | ✅ **已给出最优配置，见 §12**。核心一项是 `max_kg_triples` 50→12（教师看 31 个三元组、学生只看 12 个，系统性压低 `r_kg`）。**建议不换 teacher**（会把三变量绑一起，违反 §4 一次一变量） | §12 + manifest 实测 | 天花板 / kg_reward_share |
| **D7** | IHR 多 seed 范围 | 全 12 臂 ×3 seed 最好；预算紧则至少"最佳臂 + 最强基线" | §P5 | 核心主张的误差棒 |
| **D8** | 论文里怎么处理 PPO② | 列为"配置错误的训练记录"（`split=None` 泄漏 + Plan B 权重），**不作为 PPO 结果** | §0.3 | Results 诚实性 |
| **D9** | 是否把 n 从 300 提到 1000 | ✅ **已由 P7-3(b) 实测回答：不值得。** 2Wiki 同一效应 n=300→1000 只把 p 从 0.450 降到 0.091，仍 NS；按实测不一致率反推 +2.7pp 达 p<0.05 需 n≈1600–2000。加 n 是给小效应买 SE，应先靠 P1/P3 把效应量做大 | §P7 | 评估算力 ×3.3（不建议） |

---

## 6. 与上一版（2026-08-21）的差异

| 上一版说法 | 本轮核实结果 | 处置 |
|---|---|---|
| §0.4 "主方法用 temp=0.7 / n=100，与基线不可比" | **错**。看的是 7 月的 `R8_final`；主表实际取自 `wiki18_eval/*`，是 temp=0 / n=300 / rerank-10 / 同 GPU | 已更正（§0.4）；P4 工作量从 12 cell 降到 6 cell |
| §2.6 "`max_steps=7` 截断了 8.87 步的 rollout" | **错**。`max_steps=5`（manifest 实测），且 `n_steps_sample` 是每 batch 总步数，除以 4 得 **2.2 步/轨迹** | 已更正（§0.6）；问题方向**相反** |
| §0.5 "多 seed 噪声地板不成立" | 对，但不完整——n=300 的**二项抽样误差**仍在（SE 2.7pp），这才是增益 NS 的原因 | 已补全（§0.9, P7） |
| PPO 退化归因 = Plan B 权重 | **不完整**。步数塌缩（§0.1）是 PPO① 和 PPO② **共同**的根因；PPO② 另有 `split=None` 泄漏 | 新增 R-1 为最高优先级；D8 处置 PPO② |
| `kg_reward_share` 低 = 银标引用率低 | **一半对**。另一半是步数塌缩把 KG 作用面从 3.36 步压到 2 步 | R-1 与 P1 并列 |
| 未提及 | `discounted_returns` 是死代码，论文公式与实现不符（§3.2a） | 新增，需改论文表述 |
| 未提及 | MuSiQue α 最高但 KG 最差的矛盾（§0.5, P6） | 新增 P6 |
| 未提及 | `run_ihr_judge.py` 默认 judge 是 gpt-4o（§0.8） | 新增 P2 |

---

## 7. 本文档的核实边界（诚实声明）

**已现场核实**：银标全量统计（24,998 条逐行）、KG 索引覆盖与命中率（22,393 条 + 4,000 条匹配）、两个 PPO manifest 逐字段 diff、两条 history.jsonl 分段趋势（750 updates × 2）、rollout samples 步数、6 个 eval run 的 prompt 内 KG 三元组计数与 docs 数、12 个主方法 IHR 文件 + 11 个基线 IHR 文件的 `judge_model`、`discounted_returns` 消费者 grep、两个 IHR 脚本的默认 judge。

**2026-08-22 补充核实**（本轮 §8–§13 的依据）：`r_kg`/`r_text`/`alpha`/`r_total` 四通道在 PPO① 750 次更新上的 mean/std/min/max（§9 量纲表）、`text_reward_model.py` 两个后端的打分公式、`prm_annotator.label()` 全部返回分支、`composite_reward.py` 全文、`phase1_generate_silver.py:143-175` 的 teacher 参数优先级分支、`_run_full_silver.sh` 全部三次调用的 `--config` 传参、`phase3_ppo.py:355-370`（KG 回退路径的就地过滤）与 `:797`（索引硬编码路径）、`PPOConfig` 的 `adap_kl_ctrl`/`init_kl_coef`/`target` 传参、`kg_reward_share` 分母表达式。

**未核实 / UNKNOWN**：
- 2wiki SFT noKG 的落盘目录（`statistics.md` 有数字，本轮未定位到目录）；
- 2wiki SFT+KG 的实际 KG 注入量（该目录缺 `intermediate_data.json`）；
- `05_results.md` §5.6 KG 过滤消融四项（−0.7 / −0.4 / +1.3pp）的来源目录与 n；
- §5.2 "α 跨数据集 p<0.001 两样本 t 检验" 的原始数据位置；
- P6（MuSiQue α-KG 矛盾）中 `phase2_prm.py:733` 的 `calibration(alpha, kg_has_verdict)` 具体定义 —— 特别是 "subgraph too sparse → NEUTRAL" 分支**算不算 verdict**。这一条决定 α 到底学到了什么，未读完前不得下结论（见 §8 病灶 3）。

**本轮已补充核实的四条**（原列为 UNKNOWN，已查证）：
- ✅ **teacher 身份（原 D5）**：`deepseek-v4-flash` 从未被执行到，银标 100% 由 `deepseek-chat` 生成。完整证据链见 §11。这条不需要向 DeepSeek 侧确认。
- ✅ **D3 两条路径的实际差异**：见 §10。功能上接近（回退路径已就地补过滤），差距在可溯源性。
- `06_build_question_kg_index.py` **支持** `--split`（默认 dev）**以及 `--silver <jsonl>`**（"Silver .jsonl to take questions from (covers the PPO prompt set)"）和 `--merge_into`。**`--silver` 正是 R-2 的最干净解法**：直接用银标问句建索引，天然覆盖 PPO prompt 集，无需猜 split 前缀。见 R-2 更新。
- `launch_split_ppo_smoke.sh` 的 `SMOKE_TRAJ` 默认 **80**（可 `SMOKE_TRAJ=160 bash ...` 覆盖）。⚠️ 该脚本第 63 行的回显按 `batch_size 8` 算更新数，但当前 yaml 是 **batch_size 4**，所以 80 轨迹 = **20 次更新**（不是脚本说的 10 次）。20 次更新**不足以观察步数塌缩**（§0.1 显示塌缩发生在前 ~150 次更新内）。**smoke 必须用 `SMOKE_TRAJ=600`（= 150 updates）** 才能判断 R-1 是否生效。

以上项在动到相关步骤前必须补齐，不得凭推测写入论文。

---

## 8. D1 落地：选项 A 之下，训练期 KG 到底出了什么问题

**D1 已定为选项 A**：KG 在训练期作**正则**（约束推理过程可验证），在推理期作**增强**（补检索缺口）。这个定位与现有代码和已落盘数据完全吻合，因此 §0.4 确认的主表全部继续有效，不需要重跑 SFT 臂。

但"训练期作正则"目前是**名义上的**：实测 KG 通道只贡献 0.110 的奖励量级（§9），正则强度接近于零。病灶有三个，量级差别很大。

### 病灶 1 — 索引 0% 命中（最大，且是纯工程 bug）

`phase3_ppo.py:797` 硬编码读 `indexes/kg_cache/question_kg_index_v2.json`。该索引是按 **dev split** 的问句建的；PPO 跑的是 `--split train` 的银标问句集，问句字符串对不上，`question_kg_index.get(traj.question)` 全部 miss。9839 条 prompt **全部**走 `:363` 的回退路径。

连锁反应（这是它为什么是最大病灶）：

```
prompt 子图质量下降
  → r_kg 的 triple_in_subgraph() 拿 prompt 子图当校验参照（同一个 kg_subgraph 对象，
    见 :388 RewardSpec(kg_subgraph=kg_triples)）
  → 学生的引用难以被验证
  → precision 趋 0 → r_kg 实测均值仅 0.090
  → α·r_kg = 0.110，KG 正则名存实亡
```

注意这条是**双重损伤**：既污染了模型看到的 KG，又污染了奖励的校验基准。**R-2 修这一条，收益最大且改动最小。**

### 病灶 2 — `label()` 的中性吸收态（结构性，非 bug）

核实 `prm_annotator.label()` 的全部返回分支，返回 **NEUTRAL=0** 的路径有三条：

| 分支 | 代码位置 | 条件 |
|---|---|---|
| discourse 且无引用 | `:180-181` | `_is_discourse(text) and not step.cited_triples` |
| 无引用且无矛盾 | `:184-189` | `not step.cited_triples`（矛盾是唯一的例外） |
| 子图太稀疏 | `:193-194` | `len(kg_subgraph) < min_subgraph_for_verify` |

只有 `:191` 之后（有引用 **且** 子图够用）才会算出非零的 `precision × relevance`。所以 `r_kg=0.090` 这个均值的真实构成是"**大多数步拿 0，少数步拿 0.5~1.0**"的稀疏分布，而不是"平均每步 0.09 分"。

这**不是 bug，是刻意设计**：`:192` 的注释写明 "C2: don't punish KG gaps"。KG 覆盖不到的步不应被惩罚，否则奖励就退化成惩罚数据集的 KG 覆盖率，而 MuSiQue 有 12.4% 的问句根本没有子图。

因此选项 A 之下的正确应对**不是让 `r_kg` 变稠密**（那等于取消 C2），而是**提高引用率本身**：

> §0.7 的交叉表已证明：**零引用步的 POS 数严格为 0**。无引用 → `precision` 分支根本不执行。
> 所以"提高 POS 比例"和"提高引用率"**是同一件事，不是两个杠杆**。

上游唯一的抓手是 `phase1_distill.py:_needs_format_retry`（见 §12 P1-a）：它目前只在**整条轨迹零引用**时才触发重试，某一步空引用不触发。这是引用率停在 51.1% 的直接原因。

### 病灶 3 — MuSiQue 的 α/KG 悖论（✅ 2026-08-23 已查清，见 P6：悖论不存在，α 实为单特征密度函数）

~~实测矛盾：MuSiQue 的 `alpha_mean` 最高（0.855），而它的 KG **最差**（12.4% 空子图、6.47 三元组/问）。~~

**已定案**：0.855 与 0.292/0.804 取自三次不同状态的运行，不可比。单一固定状态下三者为 0.918/0.914/0.908（极差 0.0099），MuSiQue 反而最低。α 与 density 相关 r=0.90–0.95，而 density $=|E|/(|V|+\epsilon)$ 在稀疏子图上分母更小 —— 所以 KG 越差 α 也不降，但这是**密度定义的算术性质，不是「更依赖 KG」的自适应**。下面对标定目标的分析仍然有效，它解释了为什么另外两个特征（链接置信度、语义熵）学不到权重。

已确认的线索：`phase2_prm.py:733` 的标定目标是

```python
loss_cal = calibration(alpha, kg_has_verdict)
```

即 α 拟合的是"**KG 是否给出了非中性判定**"，**不是** KG 覆盖率。这解释了为什么 α 与 KG 质量脱钩。

但**还差一步**才能定论：病灶 2 表里的"子图太稀疏 → NEUTRAL"分支，在 `kg_has_verdict` 的定义下**算不算一次 verdict**？

- 若算作 verdict（有返回值即算） → 稀疏子图会**推高** α，直接解释 MuSiQue 的异常；
- 若不算（只有 POS/NEG 算） → 稀疏子图应**压低** α，那么 MuSiQue 的高 α 另有原因，现有解释全部不成立。

**这一条列为 P6，`calibration` 的定义未读完前不得下结论**（AGENTS.md §1：不得将推测写成已验证事实）。当前 `05_results.md` §5.2 对 α 的解释无法支撑这个观测，重写前必须先查清。

### 优先级

| 病灶 | 性质 | 处置 | 是否阻塞重训 |
|---|---|---|---|
| 1 索引 0% 命中 | 工程 bug | R-2（`--silver` 重建索引） | ✅ **阻塞**，必须先修 |
| 2 引用率 51.1% | 数据质量 | P1-a + 重建银标（§12） | ⚠️ 决定天花板，建议同批做 |
| 3 α/KG 悖论 | 待查证 | P6 分析任务 | ❌ 不阻塞，但阻塞论文 §5.2 |

---

## 9. D2 落地：奖励函数的量纲问题与统一方案

**D2 已批准**（新增步数惩罚），附加要求：**改奖励时必须保证量纲统一**，并追问是否要换掉线性相加。

结论先行：**线性相加要保留，坏的不是"相加"，是两个通道的零点和动态范围不一致。**

### 9.1 三通道量纲实测（PPO① `ppo_r10_split`，750 次更新）

| 通道 | mean | std | min | max | 名义域 |
|---|---|---|---|---|---|
| `r_kg` | **0.0896** | 0.1166 | −0.2222 | 0.7500 | [−1, 1] |
| `r_text` | **0.6284** | 0.1353 | 0.0994 | 0.8489 | [−1, 1] |
| `alpha` | 0.8106 | 0.0519 | 0.4498 | 0.9056 | (0, 1) |
| `r_total` | 0.6997 | 0.6714 | −1.9705 | 2.3848 | — |

代入 $R_t=(\alpha\, r_{KG}+(1-\alpha)\, r_{text}\cdot 0.3)\cdot 1.5$：

| 项 | 计算 | 实际贡献 |
|---|---|---|
| KG 项 | 0.8106 × 0.0896 × 1.5 | **0.1090** |
| 文本项 | 0.1894 × 0.6284 × 0.3 × 1.5 | **0.0536** |
| outcome 项 | 4.0 × EM | **0 或 4.0** |

**关键观察**：`α=0.81` 已经把 81% 的名义权重给了 KG，`text_reward_scale=0.3` 又额外压了文本一次 —— **参数上 KG 是绝对主导**。但 KG 实际只贡献 0.109，因为 `r_kg` 的均值只有 0.090 而 `r_text` 是 0.628。**两个通道名义同域 [−1,1]，实际占据的区间差了 7 倍**（0.090 vs 0.628）。

### 9.2 根因：两个通道的零点语义不同

**`r_text` 是一个近似恒定的偏置，不是奖励信号。**
`text_reward_model.py` 的 `RearagPromptScorer.score_step()` 返回 `tanh((2.5 - nll) / 1.5)`。代码注释自述"典型强似然片段 NLL ~1.5"，代入即 `tanh(0.67) ≈ 0.585` —— 与实测均值 0.628 吻合。std 仅 0.135，且与步质量弱相关。

⚠️ **这对 PPO 是致命的**：常数加项会被 advantage 白化（GAE 减去 baseline）完全消去。所以 `r_text` 这个通道当前**大部分信号在空转**，`text_reward_scale=0.3` 缩的是幅度，偏置 0.628 被同比例缩成 0.19，**仍是一个与内容无关的常数**。

**`r_kg` 是稀疏信号。** 如 §8 病灶 2：大多数步落在 NEUTRAL=0 的三条分支上，少数步拿 0.5~1.0。

### 9.3 为什么**不能**换掉线性相加

考虑过的替代方案与否决理由：

| 方案 | 否决理由 |
|---|---|
| 乘性 $r_{KG}\cdot r_{text}$ | `r_kg=0` 的步整步奖励清零。但 0 在这里的语义是"**KG 无法判定**"，不是"这步差" —— 直接违反 `prm_annotator.py:192` 的 C2 设计 |
| 几何平均 | 同上，且 `r_kg` 可为负（矛盾分支 −1），几何平均无定义 |
| min / 软 min | 等价于让最稀疏的通道支配，KG 覆盖率低的数据集（MuSiQue 12.4% 空子图）会被系统性压分 |
| 线性相加（现状） | ✅ **对"缺证据"这一情形是中性的**。在 KG 覆盖率只有 51% 的数据上，这个性质不可替代 |

**保留线性相加。** 在 KG 覆盖不全的现实下，加法的中性性是特性而不是缺陷。

### 9.4 量纲统一方案（3 项改动）

**(1) `r_text` 中心化（去偏置，不是缩放）** — ✅ **已实现 2026-08-23**

```
r_text_used ← r_text − baseline        # baseline = 因果 EMA，momentum 0.99
```

让它只表达"这步比近期平均好多少"。这样 `text_reward_scale` 控制的才是**方差贡献**，而不是一个会被白化掉的常数。

**⚠️ 实现时偏离了本节原方案：用因果 EMA 基线，不用批内均值。** 两个理由，都是量纲/统计性质问题，不是工程便利：

1. **批内均值太吵**。`batch_size=4` × 约 3 步 ≈ 12 个步样本，`r_text` 实测 sd 0.1353 → 批均值的标准误 `0.1353/√12 = 0.039`，是信号自身 sd 的 **29%**。减掉一个这么吵的估计量，等于往奖励里注入噪声。
2. **批相对量 critic 无法表示**。批内均值让同一个 state 的奖励取决于跟它同批的是谁 —— 这是 GRPO 式的组相对量。但 PPO 的 value head 只看 state，学不出这个，价值函数会被迫去拟合一个它观察不到的量。

EMA（momentum 0.99，约 100 样本窗口）两个问题都没有：它是**过去数据**的函数，与当前 batch 的同伴无关。`_update_text_baseline()` 返回**更新前**的基线，所以每个观测只被过去的数据中心化 —— 因果性是显式保证的，不是巧合。

**落地位置**（8 个文件）：`composite_reward.py`（核心 + `StepReward` 新增 `r_text_used` / `text_baseline` 两个字段）、`reward_function.py`、`training/phase3_ppo.py`（+ 4 个诊断量）、`scripts/train/phase3_ppo.py`（显式转发）、`config/schemas.py`、`configs/training/phase3_ppo.yaml`（`center_text_reward: true`）、`training/phase3_grpo.py`（同 bug 但默认关）、`tests/test_composite_reward.py`（+9 条）。

- `alpha_override` 消融分支从 `sr.r_text_used` 重组，不从 `sr.r_text`：否则 α 消融臂会是唯一跑未中心化通道的臂，测的就不只是 α 了；而且 `_update_text_baseline` 已经消费过该观测，再读原值会让这条路径上 EMA 的采样率翻倍。
- **默认关闭**（`center_text_reward: bool = False`），所以 2026-08-23 之前的所有 run 逐位复现 —— `test_centering_off_is_bit_for_bit_identical` 钉住这一点，r9/R10 的历史不需要重新基线化。

**这个 bug 比"常数被白化"严重得多**（§9.2 只说对了一半）。DC 偏置对策略梯度确实不可见，但**它通过 α 可见**：

$$\frac{\partial r_{total}}{\partial \alpha} = (r_{KG} - c_{text}\, r_{text})\cdot c_{step} = (0.0896 - 0.3\times 0.6284)\times 1.5 = \mathbf{-0.1484}$$

**严格为负** —— 奖励在付钱让策略**降低 α**。而 α 随 `f_density = |E|/(|V|+ε)` 上升，"降低 α"就是"去引用更稀疏的子图"。**一个 KG grounding 奖励在奖励更少的 KG grounding。**

**实测效果**（3000 个种子化步观测，回放实测分布 mean 0.6284 / sd 0.1353）：

| | `dR/dα` 均值 | 为负的步占比 | `used` 的 sd |
|---|---|---|---|
| 中心化前 | **−0.1482** | **99.4%** | 0.1353（原值）|
| 中心化后 | **+0.1363** | **1.5%** | **0.1368** |

符号翻正，且通道自身的方差**没有被压掉**（0.1368 vs 0.1353）—— 去掉的只有偏置。基线收敛到 0.6285。

**验证**：194 条测试通过；6 个 Python 文件 `py_compile` 干净；YAML → schema → dataclass 端到端确认拿到 `center_text_reward=True`（关掉 `schemas.py` 的 `extra="allow"` 陷阱 —— `ppo_max_kg_triples` 就是这么静默失效的）；smoke 检查 11–13 在 3 种合成场景下分别给出正确判定。

**(2) `r_kg` 保持不变**

保持 `precision × relevance`，保持 **0 = 中性**的语义。不做任何稠密化。提高 KG 信号强度靠上游引用率（§12 P1-a），不靠改奖励。

**(3) 新增 step-shortfall penalty（D2 批准的项）**

先看步数塌缩的**算术解释**：

| 量 | 值 |
|---|---|
| 一条 2 步轨迹的过程奖励总量 | 2 × 0.163 ≈ **0.33** |
| outcome | 4.0 × EM |
| **比例** | **1 : 12** |
| 多写第 3 步的边际收益 | ≈ **0.163** |
| 该步的 KL 成本（~60 token，实测 `ppo_mean_kl` 34.7） | **> 0.163** |

→ 策略理性地收敛到 `min_valid_steps=2`。**shortfall penalty 本质上就是在补这个量纲缺口。**

```
不足 target_steps(=3) 时，按缺口线性扣分：
    penalty = -shortfall_coef * outcome_weight * max(0, target_steps - n_steps) / target_steps
```

**锚点系数建议 1/4**（即缺 1 步 ≈ −1.0 × (1/3) ≈ −0.33，缺 2 步 ≈ −0.67；相对 0.33 的过程奖励总量是同量级的可感知代价）。

⚠️ **不建议取更大值**：再大就会盖过 EM 信号，把 PPO 变成纯格式优化器 —— 这正是 `kg_reward_share` 曾掉到 0.009 的那次教训（`phase3_ppo.py:1016-1019` 注释记录：`outcome_weight=10 / step_reward_scale=0.3` 时 KG 通道只占 0.9%，"PPO 基本在单独优化格式合法性和 EM"）。

### 9.5 不改动的项（明确记录，避免下轮再动）

| 参数 | 现值 | 为何不动 |
|---|---|---|
| `step_reward_scale` | 1.5 | 已在 R10 从 0.3 提上来；再提会放大 `r_text` 的偏置（若 (1) 未做）|
| `outcome_weight` | 4.0 | 提高它会加剧 1:12 的失衡，方向相反 |
| `text_reward_scale` | 0.3 | 中心化后再评估；先不与 (1) 同时改，否则无法归因 |
| 线性相加 | — | §9.3 |

### 9.6 ⚠️ 本节的量纲数字来自 PPO①，且是 batch 聚合值 — ✅ **稀疏性已有直接证据 2026-08-23**

`history.jsonl` 的 `r_kg_mean` / `r_text_mean` 是**每次更新内所有步的均值**（`phase3_ppo.py:1007-1010`，分母 `len(all_per_step_records)`）。所以上表是"步级均值的更新级均值"，不是步级分布本身。

**§9.2 的"稀疏"判断此前只有间接证据，现在有直接证据了** —— 不用等重训，直接数银标源数据（33,011 条 accepted 步）：

| 量 | 实测 |
|---|---|
| label 恰为 0.0（PRM NEUTRAL 分支）的步 | **70.0%** |
| 有任何 `cited_triples` 的步 | **51.1%** |
| 三分类分布 | NEG **4.3%** / NEUTRAL **70.2%** / POS **25.5%** |

**结论**：`r_kg_mean ≈ 0.0896` 是**引用频率**导致的，不是引用准确率 —— 近一半的步压根没引用任何三元组，自然拿 0。**所以杠杆在上游引用率（§12 P1-a），不在更严的匹配器。**§9.4-(2) 的"`r_kg` 不动"因此得到独立支持。

另外 `phase3_ppo.py` 现在逐更新落盘 `r_kg_zero_frac`（日志字段 `r_kg_0=`，smoke 检查 13 读它），所以重训期的步级稀疏度不再需要事后推断。

---

## 10. D3 落地：重建索引 vs 统一走银标 subgraph，差别到底在哪

**D3 已定：重建索引（`--silver`）。** 但两条路径的差异不在"三元组来自哪里"，而在**三层过滤策略有没有被固化成产物**。

### 10.1 代码实测：功能差异比预想的小

`phase3_ppo.py:355-370` 的实际逻辑：

```python
if question_kg_index is not None:
    dyn = question_kg_index.get(traj.question)
    if dyn:                          # 命中：用索引的三元组
        kg_triples = list(dyn)
    elif kg_triples:                 # miss：回退到银标 kg_subgraph
        kg_triples = filter_and_rank_triples(   # ← 就地补一次三层过滤
            [tuple(t) for t in kg_triples if len(t) == 3],
            question=traj.question,
            max_keep=cfg.ppo_max_kg_triples,
        )
```

回退路径**已经**就地补了 `filter_and_rank_triples`，注释明确写了理由："银标 raw kg_subgraph 没经过三层策略，就地过滤以保持 KG 分布与命中情形一致"。

**所以两条路径产出的三元组分布是近似的**，不是"一个有过滤一个没过滤"。旧方案暗示的巨大质量差**不成立**。

### 10.2 真正的差异：可溯源性

| 维度 | 重建索引（`--silver`） | 走银标 subgraph（在线过滤） |
|---|---|---|
| 过滤时机 | **离线一次**，产物落盘 | 每次训练**运行时重算** |
| 参数可控性 | `--max_keep` / `--min_keep` 独立可调 | 被 `cfg.ppo_max_kg_triples` **单值绑死** |
| 产物 | JSON 文件，可归档 / 可 diff / 可写 manifest | **无产物** |
| 可复现性 | 索引文件 hash 即可锚定 | 需重跑训练进程才能还原 |
| 上游信息损失 | 从原始 SPARQL 结果建 | 银标已按 `max_kg_triples=50` 截断过一次 |

**最后一行是实质性的**：manifest 记录 `avg_kg_before` 83.6 → `avg_kg_after` **31.2**，即银标的 `kg_subgraph` 已经损失过一次信息。在此之上再过滤到 12，是**二次截断**；而重建索引是从原始结果一次过滤到 12。

### 10.3 结论

功能上两者接近，**科研可溯源性上差距很大**。AGENTS.md §9 要求 Claim → Evidence → Data 可追溯，在线重算的子图无法归档，无法回答"论文表 X 用的到底是哪批 KG"。

**执行**（与 R-2 一致）：

```bash
python scripts/prepare/06_build_question_kg_index.py \
  --silver data/silver_data/silver_v1_reannotated.jsonl \
  --min_keep 5 --max_keep 12 \
  --output indexes/kg_cache/question_kg_index_v2_train.json \
  --report docs/kg_build_report_train.md
```

⚠️ 两个配套改动，缺一则索引白建：

1. **`phase3_ppo.py:797` 的硬编码路径必须改成可配置**，否则新索引不会被读到（当前只认 `question_kg_index_v2.json` 和 `question_kg_index.json` 两个固定名）。
2. `--max_keep 12` 必须与 `ppo_max_kg_triples=12`、以及银标重建的 `max_kg_triples=12`（§12）**三处一致**。

---

## 11. D5 已查清：teacher 是 `deepseek-chat`，无需向 DeepSeek 侧确认

你问"这个怎么确定"——**用代码分支就能确定，不需要外部信息。** 完整证据链：

**第 1 步**：`scripts/train/phase1_generate_silver.py:46` 的 argparse 默认值确实是 `deepseek-v4-flash`：

```python
p.add_argument("--teacher", default="deepseek-v4-flash")
```

**第 2 步**：但 `:148-169` 是一个 **if/else 二分支**，`args.teacher` 只在 `else`（无 `--config`）分支里被使用：

```python
if args.config:                                  # ← 走这里
    cfg = load_config(args.config, ...)
    silver = cfg.training.silver_data
    teacher_model = silver.teacher_model         # ← 从 config 读，args.teacher 被忽略
    ...
else:                                            # ← 从未走到
    teacher_model = args.teacher                 # ← "deepseek-v4-flash" 只在这里生效
```

**第 3 步**：`scripts/_run_full_silver.sh` 的**全部三次调用**（hotpotqa / 2wiki / musique）都带 `--config configs/training/phase1_silver.yaml`。

**第 4 步**：`configs/training/phase1_silver.yaml:7` 写的是：

```yaml
teacher_model: deepseek-chat
```

**第 5 步**：交叉验证 —— 银标数据里 `teacher_model` 字段 **100% 是 `deepseek-chat`**（§0.7 全量核实 24,998 条）。

### 结论

`deepseek-v4-flash` 是一个**从未被执行到的 argparse 默认值**。银标 100% 由 `deepseek-chat` 生成。文档里的 v4-flash 是**笔误，不是 API 别名问题**。

- ✅ 论文一律写 `deepseek-chat`；
- ✅ 这条从 §7 的 UNKNOWN 列表移除；
- 🔧 **建议顺手把 `:46` 的默认值删掉**（改为 `required` 或 `default=None`），让漏传 config 时直接报错，而不是静默用一个从未验证过的模型名。这类"看起来生效但实际不生效的默认值"是同类事故的温床（对比 §13-2 的 IHR judge 默认值）。

---

## 12. D6：银标重建的最优配置

### 12.1 先算代价

`scripts/_run_full_silver.sh` 自述：**~10577 次 API 调用、~$10、4–6 小时**。可承受，但**只值得做一次**，所以配置要一次定对。

现有 manifest（`data/silver_data/manifest.json`，`reannotate_v1`，git `a9c39dc`）的实际产出：

| 量 | 值 |
|---|---|
| total | 24,998 |
| accepted | **9,839（39.4%）** |
| rejected_quality | 10,433 |
| kg_sparse | 7,186 |
| kg_rich | 4,955 |
| kg_medium | 2,424 |
| max_kg_triples | 50 |
| avg_kg_before → after | 83.6 → **31.2** |
| halluc_rate | 14.53% |

### 12.2 建议配置

| 项 | 现值 | 建议 | 理由 | 变量数 |
|---|---|---|---|---|
| `max_kg_triples` | 50 | **12** | 见 §12.3，收益最明确 | 主变量 1 |
| `_needs_format_retry` | 整条零引用才 retry | **任一步空引用即 retry** | 见 §12.4，引用率唯一上游杠杆 | 主变量 2 |
| `sparse_quota` | 0.25 | **不动** | 见 §12.5 |
| `min_steps` | 3 | **不动** | 见 §12.6 |
| teacher | `deepseek-chat` | **不换** | 见 §12.7 |

### 12.3 ⭐ `max_kg_triples` 50 → 12（最建议的一项）

**这是一个训练/推理不一致的 bug，不是超参选择。**

| 谁 | 看到多少三元组 |
|---|---|
| 教师（生成银标时） | 按 50 截断，**实测均值 31.2** |
| 学生（PPO prompt） | `ppo_max_kg_triples` = **12** |
| 学生（推理 pipeline） | `max_kg_triples` = **12** |

同一条问题，**教师看着 31 个三元组写出的引用，学生只能看到 12 个**。学生被要求引用它看不见的三元组 → `triple_in_subgraph()` 系统性判负 → `precision` 下降 → `r_kg` 被压低。

这直接连到 §9.1 实测的 `r_kg = 0.090`。对齐到 12 **应当直接提升 `r_kg`**，且这一项独立于其他改动、可单独归因。

⚠️ 必须与 §10.3 的 `--max_keep 12` 和 `ppo_max_kg_triples=12` **三处同时对齐**。

### 12.4 ⭐ `_needs_format_retry` 改为逐步检查

现状（`phase1_distill.py:535-560`）：

```python
return (len(steps) < min_steps) or (kg_subgraph and not any(step.cited_triples for step in steps))
```

`any(...)` 意味着**只要整条轨迹有任意一步引用了三元组，就不触发重试**。某一步空引用完全不管。而 `:559` 的 retry hint 文本却要求**逐步**引用 —— 检查逻辑与提示语不一致。

这是引用率停在 **51.1%** 的直接原因。改为：

```python
# 任一"应当引用却没引用"的步都触发重试
steps_missing_citation = [s for s in steps if not s.cited_triples and not _is_discourse(s)]
return (len(steps) < min_steps) or (kg_subgraph and steps_missing_citation)
```

结合 §8 病灶 2 的交叉表（**零引用步的 POS 严格为 0**），这是提高 KG 信号强度的**唯一上游杠杆**。

⚠️ 副作用：retry 次数上升 → API 调用数和成本上升。需要给 retry 设上限，并记录实际 retry 分布。

### 12.5 `sparse_quota` 保持 0.25（不要为了提引用率砍掉它）

直觉上"砍掉稀疏样本能提高整体 KG 质量"，但**这会破坏 α-gate**：

`sparse_quota` 保证了 KG 稀疏的样本能进入训练集，这些正是 **α→0 的 fallback 训练样本**。全砍掉的话 α-gate 只见过 KG 充足的样本，会退化成常数 α≈1，届时 §9 的整个 α 加权机制失去意义（实测 α 已经很集中：mean 0.811 / std 0.052）。

⚠️ 但 `sparse_quota_full` **白丢了 4,726 条**（占被拒 31.2%）—— 这些是配额满了之后被丢弃的稀疏样本。可以考虑**提高**配额上限来回收一部分，但不能降低。

### 12.6 `min_steps` 保持 3（与 `min_valid_steps` 的关系）

| 参数 | 值 | 作用 |
|---|---|---|
| 银标 `min_steps` | 3 | 银标接受门槛 |
| PPO `min_valid_steps` | 2 | `trajectory_valid` 门槛 |
| 银标 accepted 实测步数 | **3.36** | — |
| PPO 末期实测步数 | **2.0** | ← 塌缩到 PPO 门槛 |

**注意这两个门槛的不一致正是塌缩的空间**：银标教了 3.36 步，PPO 只要求 2 步。§9.4 (3) 的 shortfall penalty 的 `target_steps` 应设为 **3**，与银标 `min_steps` 对齐，而不是与 `min_valid_steps=2` 对齐。

同时建议 R-1a：`min_valid_steps` 2 → 3，消除这个空间。

### 12.7 ⚠️ 建议**不换** teacher

若换更强 teacher，收益是引用质量，但代价是：

1. 银标全部数字（accepted 率 39.4%、引用率 51.1%、`halluc_rate` 14.53%）**都不能与旧版比较**，等于开一条新 Experiment ID；
2. AGENTS.md §5：不同数据版本的结果不得直接当同条件比较 → **所有已有对照都要重跑**；
3. AGENTS.md §4：一次实验只改一个主变量。换 teacher 会把 §12.3 / §12.4 / teacher **三个变量绑在一起**，无法归因。

**建议只做 §12.3 + §12.4**（这两项可分别归因），teacher 保持 `deepseek-chat`。若做完这两项后引用率仍不足，再单独立一个换 teacher 的实验。

### 12.8 重建后必须复核的验收指标

| 指标 | 现值 | 期望方向 |
|---|---|---|
| accepted 率 | 39.4% | 持平或略降（retry 更严）|
| **accepted 步的引用率** | **51.1%** | ⭐ **显著上升**（主要目标）|
| avg_kg_after | 31.2 | → **≈12**（按新预算）|
| POS 标签占比 | §0.7 | 上升（与引用率同步）|
| halluc_rate | 14.53% | 不应上升 |
| `sparse_quota_full` 丢弃数 | 4,726 | 记录（若提配额则应下降）|

---

## 13. 你没问、但必须一并解决的 7 项

按影响排序。**前两条会直接污染论文数字，且都是"静默失效的默认值"这一类事故。**

### 13-1 ✅ 已修：`phase3_ppo.yaml` 的 `split: null`（`split: train` + `split=None` 直接 raise）

`configs/training/phase3_ppo.yaml` 的 `split` 是 `null`。`phase3_ppo.py:768-775` 对此**只打一条 warning 就继续跑全量文件**：

```python
if cfg.split is None:
    logger.warning("Phase 3b split: NONE — PPO rolls out over the whole file (%d ...")
```

**PPO② 就是这样在 val/test fold 上训出来的**（§0.3），这是它全部数字作废的根本原因 —— 比 Plan B 权重严重得多。

现在唯一的保护是 `launch_split_ppo.sh` 记得传 `--split train`。**一次遗漏就作废一轮 4 小时训练。**

**处置（重训前的硬前置）**：
1. yaml 默认改成 `split: train`；
2. 更强的做法：让 `split=None` 时**直接 raise**，要求显式传 `--allow_no_split` 才允许全量。数据泄漏不该是一条 warning。

### 13-2 ✅ 已修（代码）：`run_ihr_judge.py` 的 `--judge_model` 改为必填 — 但历史 IHR 数据仍不可比，须 P5 重判

| 脚本 | 默认 judge |
|---|---|
| `scripts/eval/run_ihr_judge.py:31` | **`gpt-4o-2024-08-06`** ⚠️ |
| `scripts/eval/run_baseline_ihr.py:143` | `deepseek-v4-pro` |

你的需求 2 是"IHR 主方法重跑，保持一致统一"。**这个默认值是最容易让人不知不觉跑出不可比数字的地方** —— 漏传 `--judge_model` 就换了一个判官，而且不会报错。

同类问题还有 teacher 侧：`configs/base.yaml:56` 和 `phase1_silver.yaml:7` 都写 `deepseek-chat`，而文档写 v4-flash（§11）。

**处置**：把 judge 改成**显式必填**（去掉 default），让漏传直接报错。§11 的 `--teacher` 默认值同样处理。

### 13-3 ✅ 已修：`discounted_returns` 已删，`03_method.md` 公式已改写为实际运行的形式（含 GAE 说明）

`reward_function.py:328/445` 算出 `returns` 塞进返回字典（`:349/:486`），**全仓库无消费者**（已 grep 确认）。真正的折扣由 TRL 的 GAE 完成 —— `PPOConfig(gamma=cfg.gamma, lam=cfg.lam)`，`phase3_ppo.py:881`。

论文写的 $G_t=\sum_k \gamma^{k-t}R_k$ **描述的不是实现**。

**处置**：二选一 —— 删掉该函数并把论文改成描述 GAE；或明确标注它只是诊断量。不影响结果，但属 AGENTS.md §9 的 Claim↔Code 一致性问题。

### 13-4 ✅ 已处理：保留定义（历史可比），并已在 `05_results.md` §5.4 标注该值被系统性低估

`phase3_ppo.py:1046-1047`：

```python
total_mass = sum(abs(r.r_total) for r in all_per_step_records) + \
    cfg.outcome_weight * max(1, sum(traj_rewards) > 0)
```

`max(1, bool)` **恒等于 1**（bool 最大也是 True==1），所以分母**恒定多加 4.0**。这个指标因此被系统性低估。

代码注释自述是为了与 r9 历史可比而故意保留 —— **同意保留**（改了会断掉历史对比），但：

⚠️ **论文引用 `kg_reward_share` 时必须标注它是这个定义下的值**，否则读者会当成真实的 KG 奖励占比。实测 PPO① 0.090 / PPO② 0.016，真实占比比这更高。

### 13-5 ✅ 已修：`SMOKE_TRAJ` 默认 80→600，batch_size 回显改为从 yaml 读取

| 项 | 值 |
|---|---|
| `SMOKE_TRAJ` 默认（`:33`） | **80** |
| yaml `batch_size` | 4 |
| → 实际更新数 | **20** |
| 脚本 `:63` 回显按 | `batch_size 8` ← **与 yaml 不符** |
| 步数塌缩显形所需 | **~150 次更新** |

20 次更新**无法判断 R-1 是否生效**。**冒烟必须 `SMOKE_TRAJ=600`**（= 150 updates，~45 min）。顺手修 `:63` 的 batch_size 回显。

### 13-6 ✅ 已修：NOTE 已改为 `ppo_max_kg_triples` 实际默认 **12**（附代码行号）

NOTE 说 `ppo_max_kg_triples` "固定为 dataclass 默认（R10: 50 → 30）"，**实际默认是 12**（`phase3_ppo.py:178`）。会让人按 30 去推算 prompt 长度预算。改文档。

### 13-7 ✅ 2026-08-23 全部落地（两处溯源失败的已按结论处置，不是搁置）

| 位置 | 原表述 | 实际 | 处置（2026-08-23） |
|---|---|---|---|
| `00_abstract.md` / `01_introduction.md` | "three-valued process reward" / 三值 | R9 起是**连续值** `precision × relevance` ∈ [−1,1] | ✅ 六处（中英各三）已改为"连续值"，并写出计算式 |
| §5.4 | `outcome=8.0` | 现 **4.0** | ✅ 已标注 8.0 是该次运行的值、现已回退 4.0、重训后会变；`03_method.md` 公式段同步改为 4.0 |
| §5.5 | judge 是 DeepSeek-Chat | **主方法确实是 `deepseek-chat`**（12 个文件已核）；**基线全是 `deepseek-v4-pro`** | ✅ 反过来了：§5.5 原本没写错，写错的是 `statistics.md` §八。真问题是主方法与基线判官不同 → 已在 §5.5 与 §八 加"重判前不得同表并列"的硬警告 |
| `04_experimental_setup.md` | 2% 问句无 KG | 实际 **4.9%**，MuSiQue **12.4%** | ✅ 已改，并补上索引实测覆盖（2Wiki 12,576 / HotpotQA 7,405 / MuSiQue 2,412 条，均值 15.87 三元组/题）；`06_conclusion.md` 局限 (ii) 同步 |
| `04_experimental_setup.md` | GPU | RTX 4090 | ✅ 原文已是 RTX 4090（24 GB），无需改 |
| §5.6 | KG 过滤消融 −0.7/−0.4/+1.3pp | ❌ **溯源失败** | ✅ 已判定**必须重跑**并在正文加 ⚠️：残留的 `outputs/ablation_v6/`（8 臂、n=100、seed 13）HotpotQA EM 除 E 臂 0.41 外全为 0.37，与三个数一个都对不上；n=100 时 SE≈4.8pp 本就分辨不出 1pp；manifest 不记过滤配置，臂身份无法还原。正文已标为"意图描述、非实测" |
| §5.2 | α 跨数据集 p<0.001 | ❌ 原始数据未找到 | ✅ 已由 P6 重测替代（0.918/0.914/0.908，极差 0.0099）。旧 p 值另有第二个错误：把每问题多步骤展平做两样本 t 检验，n 从 ~290 虚增到 ~900 |
| `06_conclusion.md` / `00_abstract.md` / `01_introduction.md` | "KG 是 PPO 的必要正则" | 实测 `kg_reward_share` 0.027 → **无支撑** | ✅ **该主张已在四处全部撤回**。理由写清两条：(a) 表 1 的 PPO 行混用两个 checkpoint（HotpotQA 0.343 来自 `ppo_r10_split/final`（ω=4.0），2Wiki 0.323 来自 `kg_proweight_ppo_v2`（ω=8.0）），同一 checkpoint 下 PPO+KG 为 0.213/0.323/0.037，三个数据集全线低于 SFT noKG，HotpotQA 上比自己的 noKG 臂还低 10pp；(b) 该 checkpoint 因 `split: null` 在自己的评测折上训练，数字本已被泄漏污染。改为"待重训回答的开放问题" |

**2026-08-23 结论**：七行全部落地，其中两处"原始数据找不到"的没有搁置，而是按结论处置 ——
§5.2 由 P6 重测直接替代，§5.6 判定为必须重跑并在正文标为"非实测"。**这两处是本轮最该记住的教训**：
消融 manifest 只记环境不记**过滤配置**，臂身份因此不可还原；重训后的消融必须把
`hard_delete / score_threshold / max_keep / min_keep / rerank / n` 一并写进 manifest，否则同样的溯源失败会再发生一次。

最后一行的"KG 必要正则"主张已在 `00_abstract.md`、`01_introduction.md`、`05_results.md` §5.2/§5.4、
`06_conclusion.md` 四处撤回，理由是**混用 checkpoint + 泄漏污染 + 机制量级不符**三条同时成立。
**这个主张能否重新立起来，取决于 R-1 + R-2 + §12 之后 `kg_reward_share` 能升到多少** —— 这正是本轮重训要回答的问题。
在那之前，论文里不出现这一主张。

---

*本文档基于 2026-08-22 现场核实的代码 / 数据 / 训练产物 / 落盘结果。上一版备份在 `docs/retraining_plan.20260821.bak.md`。若改动了其中任一配置，需同步更新本文档对应条目并记录新的 Experiment ID。*

*§8–§13 于 2026-08-22 追加：D1/D3/D5/D6 已定案并给出落地方案，D2 已批准并补齐量纲统一方案（§9），新增 §13 重训硬前置清单。D4（启动重训）按研究者要求**待改动准备完毕后再启动**，本文档不推进该项。*

---

## 14. 「如何拟合」：α-gate 的拟合上限实测（2026-08-23）

用户问题的第三部分。结论先行：**α-gate 现在的问题不是"没调好"，是这 3 个特征对它要拟合的目标几乎没有信息量 —— 换优化器、加轮数、调超参都不会有用。**

### 14.1 它在拟合什么

`AlphaCalibrationLoss` = `w · BCE(α, kg_has_verdict)`，其中 `kg_has_verdict = (label_class != NEUTRAL)`，即"这一步 PRM 给出了非中性判定吗"。33,011 条 accepted 步上 base rate = **0.2983**。

所以一个校准良好的门应该**平均输出 0.298**。

### 14.2 三档对比（同一批 33,011 步）

把门自己的函数形式（`sigmoid((x·W+b)/τ)`）直接用 Adam 拟合这个目标 4000 步，不带 PRM、不带别的损失项 —— 得到的就是**这套特征的上限**：

| | mean α | BCE | Brier | 对比常数预测的 R² |
|---|---|---|---|---|
| 常数预测（base rate 0.298） | 0.2983 | 0.6094 | 0.2093 | 0（定义）|
| **现役 gate**（e=0.0） | 0.7860 | 1.1745 | 0.4405 | **−1.105** |
| **现役 gate**（e=0.603 实测值） | 0.6821 | 0.9175 | 0.3503 | **−0.674** |
| **最优拟合上限**（e=0.0） | 0.2983 | **0.5854** | **0.2013** | **+0.038** |
| **最优拟合上限**（e=0.603） | 0.2983 | 0.5854 | 0.2013 | +0.038 |

两件事同时成立，要分开说：

1. **现役 gate 比常数预测还差**（R² 为**负**，BCE 0.917~1.175 vs 0.609）。它平均输出 0.68~0.79，而目标 base rate 是 0.298 —— **系统性高估 2.3~2.6 倍**。这不是"校准不完美"，是完全没校准。
2. **但最优拟合也只到 R² = +0.038。** 就算把权重调到这套特征的理论最优，也只比"永远输出 0.298"这个常数好 **3.8%**。上限拟合出的 W=[0.687, 1.551, −0.8]（e=0 档）与出厂初值 (1.0, 1.5, −0.8) 几乎一样 —— 说明**梯度在这个方向上几乎是平的**。

### 14.3 为什么信息量这么低

| 特征 | mean | sd | 对 α-logit 的 sd 贡献（`|W|×sd`）| corr(·, target) |
|---|---|---|---|---|
| `f_density` | 0.8764 | **0.2667** | 1.2096 × 0.2667 = **0.323** | +0.172 |
| `f_confidence`（F-1 修复后） | 0.9070 | **0.0689** | 1.7088 × 0.0689 = **0.118** | +0.130 |
| `f_entropy` | ~0.603 | 小 | −0.583 × · = 小 | — |

**两个特征与目标的相关都只有 0.13~0.17**（R² 上限 ≈ 0.03，与 14.2 实测的 +0.038 一致）。这解释了 §P6 的 r≈+0.92 现象：`f_confidence` 虽然权重最大（`W₁=1.7088`），但它的 sd 只有密度的 1/4，**对 α 的实际贡献只有密度的 1/2.7** —— 一个近乎常数的特征，权重再大也驱动不了门。

### 14.4 F-1 修复对拟合的影响：机理上必要，但救不了这个特征

| `f_confidence` | 修复前 | 修复后 |
|---|---|---|
| mean | 0.8599 | 0.9070 |
| sd | 0.0539 | **0.0689** |
| frac ≥ 0.999 | 0.000 | 0.140 |
| corr(·, target) | **+0.1602** | **+0.1296** |
| 组内（within-traj）corr | +0.0655 | +0.0418 |

**要诚实报告：修复让这个特征与目标的相关变**低**了**（0.160 → 0.130）。修复本身仍然是对的 —— 在 99.9% 的步上注入一个近似常数 0.667，度量的不是实体链接质量而是"这步有没有写 `Knowledge Used:` 这个模板词"；那部分相关是**伪相关**（模板完整度恰好与 PRM 判定弱相关）。但必须记录：**F-1 不是 α-gate 校准问题的解**，它只是把一个被污染的特征换成了一个信息量本来就低的干净特征。

### 14.5 建议（需你决策，不擅自改）

拟合层面能做的只有一件事，而且**必须做**：

- **A（必做，零风险）**：Phase 2 重跑，重新拟合 gate。理由是 F-1 改了特征分布，现有 checkpoint 是在被污染的特征上拟合的，**继承它等于 train/inference 特征不一致**。注意重拟合只会把 mean α 从 0.79 拉到 0.298 附近（BCE 1.17 → 0.59），**不会**带来判别力（R² 仍 ≈ 0.04）。

但真正的问题在特征层，这需要你选方向：

- **B（改特征化）**：加真正与"这步有没有 KG 判定"相关的**逐步**特征。最直接的候选是**这一步引用的三元组与子图的匹配情况**（§9.6 实测：只有 51.1% 的步引用了任何三元组，而这恰恰是目标 70% NEUTRAL 的主因）。这个特征与目标近乎同义，信息量必然远高于 0.13。
- **C（换目标）**：承认 α 不该拟合 `kg_has_verdict`。α 在奖励里的角色是**权重**（KG 通道该占多少），不是**预测器**（这步会不会有 KG 判定）。可以让它由 §9.4-1 修好的 `dR/dα` 直接在 PPO 里学，去掉 Phase 2 的 BCE 校准项。
- **D（承认并如实写）**：保留现状，但论文里 α 只能作为**遥测量**报告（这与 [[alpha-gate-eval-noop]] 一致：α 在 eval 期本来就动不了 EM/F1），不能声称它是学到的自适应门。

**我的建议是 A + B**：A 无论如何都要做；B 的成本只是在 Phase 2 的样本构造里多算一个特征（`_build_samples_accepted_only` 已经能拿到 `cited_triples` 和 `filtered_kg`），比 C 的改动小得多，而且不动 PPO。C 是更大的重构，D 只在 B 也失败后才该考虑。

### 14.6 ✅ A+B 已实现（2026-08-23，你已批准）

**B 的候选特征实测**（同一批 33,011 步，两个都能在训练/推理两侧用**同一份代码**算出）：

| 特征 | mean | sd | corr(target) |
|---|---|---|---|
| `f_density`（原有） | 0.8764 | 0.2667 | +0.172 |
| `f_confidence`（原有） | 0.9070 | **0.0689** | +0.130 |
| **`cite_any`**（新） | 0.5108 | **0.4999** | **+0.547** |
| `cite_n = min(1, n/3)` | 0.2073 | 0.2325 | +0.532 |
| **`cite_match`**（新） | 0.2081 | 0.3965 | **+0.593** |

**拟合上限重测**：

| 特征集 | BCE | Brier | R² |
|---|---|---|---|
| 常数（base rate 0.298） | 0.6094 | 0.2093 | 0 |
| 原 3 特征（上限） | 0.5854 | 0.2013 | **+0.038** |
| + `cite_any` | 0.4317 | 0.1432 | **+0.316** |
| + `cite_match` | 0.4344 | 0.1342 | **+0.359** |
| **+ 两个都加** | **0.3762** | **0.1174** | **+0.439** |
| `cite_any` 单独一个特征 | 0.4380 | 0.1466 | +0.300 |

**一个 `cite_any` 就顶原来三个特征之和的 8 倍信息量，两个一起是 12 倍。** 机理很直接：`cite_any` 的 sd 是 0.4999（51/49 的近二值分布），而挂着最大权重的 `f_confidence` 只有 0.0689 —— 差 7 倍动态范围。这也是"α 实际等于密度单特征"的真正原因。

**⚠️ 循环性，如实记录。** `PRMAnnotator.label` 有一条分支就是"没引用三元组 → NEUTRAL"（`prm_annotator.py:184`），所以 `cite_any` 与目标**不独立**。实测条件结构：

| | n | P(verdict) |
|---|---|---|
| `cite_any=0` | 16,150 | **0.042** ← 近乎确定 |
| `cite_any=1` | 16,861 | **0.543** ← 真正不确定 |

**定义性的那一侧是负向的**；在真正引用了的那 51% 步上，特征把目标从 0.298 收窄到 0.543，**收窄但不决定**。这个不对称性是它可用的理由。但由此论文里**不能**把 α 说成"预测这步有没有 KG 判定"——它是 KG 通道的**权重**，而"这步压根没引用"本来就该是这个权重的合法输入（没做 KG 声明的步，不该按 KG grounding 打分）。

**落地**（6 个文件，206 测试通过）：

| 文件 | 改动 |
|---|---|
| `reward/citation_features.py` | **新增**，唯一定义源。`citation_features(cited, kg) -> (cite_any, cite_match)`，大小写/空白归一化 |
| `reward/alpha_gate.py` | `N_FEATURES=5`；`forward` 两个新参数**默认 None→0**（旧 3 参调用不变）；`load_state_dict` **兼容旧 checkpoint**：3 权重自动零填充 = 精确复现旧门，宽于 5 则**报错拒绝**（截断会静默改变 α）|
| `training/phase2_prm.py` | `_StepSample` 两个新字段 → 构造 → `__getitem__` → `_collate` → 训练循环传给门 |
| `reward/composite_reward.py` | PPO 侧同样算、同样传；`StepReward` 记录两个值；**3 处重建点**都补上（否则诊断读到 0.0）|
| `pipeline/kg_proweight_pipeline.py` | eval 遥测侧也算，否则 eval 期 α 是与训练**不同的函数** —— 正是 +0.78 那类 train/eval 错配 |
| `tests/` | +12 条（`test_citation_features.py` 8 条含**真实银标数据**统计护栏、`test_alpha_gate.py` 4 条含旧 checkpoint 逐值复现）|

**端到端验证**（真实 Phase 2 路径：builder → dataset → collate → 门）：194 个样本，`cite_any` 均值 0.608（118/194 非零），α 恰好在 `cite_any` 不同的行上不同（不引用的两行 α 不变）。

**A 的结论不变且必做**：Phase 2 必须重跑。旧 checkpoint 是在被污染的 `f_confidence`（F-1）上拟合的，而且它的两个新权重是 0 —— 零填充只保证"不静默改变旧行为"，**不等于新特征生效**。重跑后预期 mean α 从 0.79 落到 0.298 附近，且这次**判别力是真的**（R² 0.038 → 0.439）。

**仍未做**：`alpha_bias_correction` 默认值改 0.0（§P6 后续；与本节独立）。
