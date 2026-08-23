本仓库里 "v1 / v2" 有两个不同含义，容易混淆。为彻底消除歧义，本文给两个 PPO checkpoint 定**序列号 PPO① / PPO②**，全篇每个 PPO 数字都标上，一眼能看出是哪个 checkpoint 跑的：

| 序列号 | checkpoint 目录 | 训练日期 | 奖励权重（关键） | 一句话身份 |
|--------|----------------|---------|-----------------|-----------|
| **PPO①** | `ppo_r10_split/final` | 8/7 | `outcome 4.0 / step 1.5`（正确） | 旧 KG 时代训的，但**权重是对的**；换新 KG 复测 = +1.7pp |
| **PPO②** | `kg_proweight_ppo_v2` | 8/10 | `outcome 8.0 / step 0.5`（Plan B） | 换新 KG 时**把奖励权重也改坏了** → 全线低于 SFT |

> 其余差异：PPO① 锚点 `sft_student_split` + α-gate `prm_alpha_gate_v1reann_negfix` + `split=train`；PPO② 锚点 `sft_student` + α-gate `prm_alpha_gate` + `split=None`（见 §2.1）。

另外两个独立概念，别和 ①/② 混：

1. **KG 管线版本**：`旧KG` = R9 v6 之前（旧 linker、BM25 重排、`question_kg_index.json`）；`新KG` = R9 v6 之后（linker 升级、`bge-reranker-v2-m3`、`question_kg_index_v2.json`）。**同一个 checkpoint 可以分别用旧KG/新KG 各测一次**。
2. **SFT**：只有一个 SFT checkpoint（`sft_student_split`），无 ①/② 之分。

**关键结论（提前说）**：

- 干净的 KG 改进 = **PPO① 旧KG → 新KG**：HotpotQA EM +1.7pp（§1.1）；
- PPO② 的暴跌 = **Plan B 权重**造成，不是 KG 的锅（§2）；
- 2Wiki 的 "+3.3pp" = PPO①(旧KG) → PPO②(新KG) 两个变量叠加，**不可归因**（§1.1 注）。

---

## 一、v1 vs v2 评估对比（EM / F1 / IHR）

### 1.1 唯一干净的对比：同一 checkpoint 换 KG（**PPO①** `ppo_r10_split/final`）

| 数据集 | 指标 | v1（旧KG） | v2（新KG） | Δ |
|--------|------|-----------|-----------|-----|
| HotpotQA PPO+KG (n=300) | EM | 0.3267 | 0.3433 | **+1.7pp** |
| HotpotQA PPO+KG (n=300) | F1 | 0.4241 | 0.4427 | **+1.9pp** |
<!-- | 2Wiki PPO+KG (n=300) | EM | 0.2900 | 0.3233* | +3.3pp* | -->

> \* 2Wiki 的 0.3233 来自 **PPO②**（换 checkpoint 了），不是同一 checkpoint 换 KG 的干净对比，`+3.3pp` **不可单独归因于 KG**。真正的 PPO① 新KG 的 2Wiki 数字本周期没跑。

**结论（v2 KG 管线）**：KG 质量修复在 HotpotQA 上带来 **+1.7pp EM / +1.9pp F1**（同 checkpoint）。方向一致、幅度在 n=300 单 seed 的 ~4–7pp 噪声地板内，**不可表述为"显著提升"**，只能说"方向为正"。

### 1.2 完整 EM/F1 主表（PPO 用 ①/② 分开标注）

| 数据集 | 模型 | EM | F1 |
|--------|------|-----|-----|
| HotpotQA | SFT noKG | 0.3400 | 0.4558 |
| HotpotQA | SFT+KG | 0.3533 | 0.4575 |
| HotpotQA | PPO② noKG | 0.3133 | 0.4244 |
| HotpotQA | PPO① +KG（新KG，干净对比用） | 0.3433 | 0.4427 |
| HotpotQA | PPO② +KG | **0.2133** | 0.2853 |
| 2Wiki | SFT noKG | 0.3367 | 0.4265 |
| 2Wiki | SFT+KG | 0.3600 | 0.4280 |
| 2Wiki | PPO② noKG | 0.3200 | 0.4001 |
| 2Wiki | PPO② +KG | 0.3233 | 0.3979 |
| MuSiQue | SFT noKG | 0.2067 | 0.3234 |
| MuSiQue | SFT+KG | 0.2300 | 0.3147 |
| MuSiQue | PPO② noKG | 0.0367 | 0.1117 |
| MuSiQue | PPO② +KG* | 0.0367 | 0.1117 |

> \* MuSiQue +KG 实际注入 0 KG（wikidata 断网），与 noKG 相同，不作 +KG 结论。

### 1.3 IHR（中间幻觉率，DeepSeek-Chat 判，n=50/集；PPO 两列均为 **PPO②**）

| 数据集 | SFT noKG | SFT+KG | SFT Δ | PPO② noKG | PPO② +KG | PPO② Δ |
|--------|---------|--------|-------|----------|--------|-------|
| HotpotQA | 22.1% | 20.6% | **−1.5pp** | 18.0% | 25.0% | +7.0pp |
| 2Wiki | 28.7% | 20.6% | **−8.2pp** | 15.0% | 13.0% | −2.0pp |
| MuSiQue | 31.9% | 26.3% | **−5.6pp** | 52.0% | 52.0% | 0.0pp* |

> 没有 PPO① 的 IHR 数据——IHR 是本周才补的，只有 SFT（当前管线）+ PPO② 两组。所以 **IHR 无法做 ① vs ② 对比**，只能看"当前管线里 KG 是否有效"（见 §四）。

### 1.4 总结：这一版做了什么 / 什么效果 / 什么缺陷

| # | 改进 | 内容 |
|---|------|------|
| 1 | 实体链接升级 | 两段式检索 + 升级 linker，`entity_cache` 35K+ |
| 2 | KG 评分对齐 spec 2.3 | `evidence_score` / `triple` 评分重写，修了 0.10×0.10=0.01 的 bug |
| 3 | 交叉编码器重排 | `bge-reranker-v2-m3` 替换 BM25（检索质量） |
| 4 | KG 关系过滤 | Python 侧 relation filter + QA 关系优先级 |
| 5 | 关键词→PID 扩展 | ~200 关键词映射 |
| 6 | SPARQL ORDER BY | 按重要性排序 |
| 7 | α-gate 偏置修正 | +0.78 bias correction |
| 8 | D1 真实 logprobs | Phase 1 用真实 logprobs |
| 9 | 数据泄漏防护 | train/eval split、`split_seed` 分离 |

**效果**（可量化）：

- HotpotQA PPO+KG：EM +1.7pp / F1 +1.9pp（干净对比）；
- KG 噪声：三元组 22.8/q → 10.3/q（−55%），但"噪声/错误"仍占 48.4%；
- SFT+KG 在三数据集：+1.3 ~ +2.7pp（**全部 McNemar 不显著**）。

**仍存在的缺陷**（详见 [statistics.md §十](statistics.md)）：

1. **"Plan B" 奖励权重训坏了 PPO**（`outcome 4→8`、`step 1.5→0.5`）→ PPO② 全线低于 SFT，已回退待重训；
2. **wikidata 断网** → MuSiQue +KG 无有效结论；
3. **银标数据质量是上游瓶颈**（§三）；
4. 单 seed、噪声地板 ~4–7pp → KG 效果无法显著化。

---

## 二、v2 最后一次 PPO 训练日志

### 2.1 训练配置（v1 vs v2 差异不止 Plan B）

**PPO②**（`kg_proweight_ppo_v2`，8 月 10 日，remote `checkpoints/kg_proweight_ppo/`）与 **PPO①**（`ppo_r10_split`，8 月 7 日）在**五个**配置项上不同，不止 "Plan B" 两个（来自两份 `manifest.json` 实测）：

| 配置项 | PPO①（ppo_r10_split） | PPO②（kg_proweight_ppo_v2） | 影响 |
|--------|--------------------|--------------------------|------|
| `outcome_weight` | 4.0 | **8.0** | EM 奖励翻倍，EM 独大（Plan B） |
| `step_reward_scale` | 1.5 | **0.5** | KG 过程奖励砍到 1/3（Plan B） |
| `sft_checkpoint` | `sft_student_split/final`（8/6 split SFT） | `sft_student/final`（7/6 旧 SFT） | SFT 锚点换成更旧的 |
| `alpha_gate_path` | `prm_alpha_gate_v1reann_negfix`（8/5） | `prm_alpha_gate`（8/4 旧版） | α-gate 换成未做 reann+negfix 的旧版 |
| `split` | `train` | `None` | 训练 fold 不同 |

其余相同：`lr=1e-6, batch_size=4, mini_batch_size=1, ppo_epochs=1, kl_coef=0.15, target_kl=40, total_ppo_steps=3000(轨迹)=750 更新, max_new_tokens=256, temperature=1.0`。基座 `llama3-8b`。

> ⚠️ **所以 v2 不是一个干净的 "v1 + Plan B" 对比**——它同时把 SFT 锚点、α-gate 都换成了更旧的 artifact、fold 也从 train 换成 None。Plan B 是 KG 正则化失效的直接原因（见 §2.2 的 `kg_reward_share`），但 SFT 锚点 / α-gate / fold 是额外的混杂变量。重训时必须一并统一回 v1 的 artifact。

> Plan B 两个值**已在 2026-08-17 回退**到 4.0/1.5（`configs/training/phase3_ppo.yaml` 有注释）；SFT 锚点与 α-gate 需在重训脚本里对齐回 `sft_student_split/final` + `prm_alpha_gate_v1reann_negfix`，`--split train`。

### 2.2 训练指标汇总（v2，TensorBoard 读数）

| 指标 | PPO② 值 | PPO①（ppo_r10_split 终值，对比） | 说明 |
|------|-------|-------------------------------|------|
| KL divergence（终） | 29.42 | ~19.1 | 策略偏离 SFT 较大 |
| valid_rate | 100% | 75~100% 波动 | 全部轨迹过格式检查 |
| α mean（训练中） | 0.848 | ~0.86 | 门控高 KG 权重 |
| mean_reward | 6.26 | ~1.06 | 正向 |
| r_kg_mean | 0.250 | ~0.0 | KG 引用质量 |
| r_text_mean | 0.764 | ~0.67 | 文本流畅度 |
| **kg_reward_share** | **2.7%** | **~17.8%** | **KG 分量被 Plan B 稀释到近噪声** |
| n_steps | 8.0 | ~7–10 | 步数打满 |
| sft_anchor_loss | 10.46 | ~1.2 | 偏离 SFT 输出较多 |
| epoch timing | 18s/update | ~17.9s/update | rollout ~9s + reward ~2.5s + ppo ~6.5s |

**关键信号**：`kg_reward_share` 从 PPO① 的 ~17.8% 掉到 PPO② 的 **2.7%** —— 这直接证实了 Plan B 把 KG 过程奖励从奖励函数里稀释掉了，PPO 实际上在优化"EM + 格式合法性"，KG 正则化失效。这与 §一 里 PPO② 全线低于 SFT、MuSiQue IHR 飙到 52% 完全自洽。

**逐 step 轨迹（750 更新，三段均值）**：

| 段 | mean_reward | ppo_mean_kl | valid_rate | kg_reward_share | sft_anchor_loss |
|----|------------|-------------|-----------|-----------------|-----------------|
| 前 10%（step 1–75） | 1.80 | 62.8 | 0.80 | 2.10% | 0.46 |
| 中 10%（step 376–450） | 2.59 | 37.2 | 0.90 | 1.78% | 0.62 |
| 后 10%（step 676–750） | 3.46 | 37.2 | 0.96 | 1.02% | 0.61 |
| **最终（step 3000）** | **6.26** | **29.4** | **1.0** | **2.70%** | **10.46** |

`kg_reward_share` 全 750 步分布：**min 0.0 / p25 0.0 / 中位数 1.07% / p75 2.56% / max 17.4%**——即**一半的更新 KG 分量为 0**，全程被压在 ~1%。`mean_reward` 一路涨（1.8→3.5→6.26）是 EM 奖励（outcome 8.0）驱动的，不是 KG；`ppo_mean_kl` 从 62.8 降到 29.4 说明策略已大幅飘离 SFT（高 KL）。

### 2.3 详细逐 step 日志已回传（2026-08-18）

v2 的完整逐 step 训练数据已从 AutoDL 机（remote `checkpoints/kg_proweight_ppo/`）回传到本机 `checkpoints/kg_proweight_ppo_v2/` 下：

| 文件 | 大小 | 内容 |
|------|------|------|
| `history.jsonl` | 426 KB | 750 更新，每步 20+ 字段（`mean_reward / ppo_mean_kl / valid_rate / r_kg_mean / kg_reward_share / sft_anchor_loss / alpha_mean / policy_entropy / clipfrac`…） |
| `manifest.json` | 6 KB | 完整训练配置 + 依赖版本 + GPU（见 §2.1 的五个差异） |
| `tensorboard/events.out.tfevents…` | 689 KB | 完整训练曲线（可 `tensorboard --logdir checkpoints/kg_proweight_ppo_v2/tensorboard`） |
| `samples/step_00{500..3000}.txt` | 24 KB | 每个 checkpoint 的 rollout 样本 |

> 注意：remote 目录叫 `kg_proweight_ppo`，本机叫 `kg_proweight_ppo_v2`（回传时改名）；训练没有单独的 `train.log`（stdout 未重定向），逐 step 数据以 `history.jsonl` + tensorboard 为准。

### 2.4 v2 完整 history.jsonl 样例（末 2 步）

```json
{"step": 2996, "mean_reward": 1.931, "ppo_mean_kl": 23.10, "valid_rate": 1.0,
 "alpha_mean": 0.843, "r_kg_mean": -0.125, "r_text_mean": 0.763,
 "kg_reward_share": 0.0253, "n_steps_sample": 8, "sft_anchor_loss": 0.0}
{"step": 3000, "mean_reward": 6.258, "ppo_mean_kl": 29.42, "valid_rate": 1.0,
 "alpha_mean": 0.848, "r_kg_mean": 0.250, "r_text_mean": 0.764,
 "kg_reward_share": 0.0270, "n_steps_sample": 8, "sft_anchor_loss": 10.46}
```

---

## 三、EM 要有"巨大突破"需要完成的重构

先把结论放前面：**当前框架的 EM 天花板大约就是 ~0.34**（`llama3-8b` + 检索 + 少量 KG）。KG 作为杠杆只贡献 +1~3pp（且不显著）。要"巨大突破"（0.40+ 乃至 0.50+），瓶颈不在检索/KG 细节，而在**基座 + 银标数据质量**。按优先级：

### 3.1 银标数据（最上游、最值钱的重构）

现状（实测 `data/silver_data/silver_v1_reannotated.jsonl`，24,998 条 / 80,120 步）：

| 指标 | 现状 | 问题 | 重构 |
|------|------|------|------|
| teacher | `deepseek-v4-flash`（100%） | 弱教师，轨迹质量天花板低 | 换强教师（DeepSeek-R1 / GPT-4o / Claude），或蒸馏更强推理模型 |
| KG 引用率 | **31.2%** 步有 `cited_triples` | 2/3 推理步不看 KG → 过程奖励"无 KG 可奖" | 强制 KG 锚定：只保留/生成有 `cited_triples` 的步 |
| accepted 率 | **39.4%** | 6 成轨迹被拒，有效样本少 | 修正 accepted 判定（当前可能过严） |
| PRM 标签分布 | **80.4% = 0.0**（正 15.4% / 负 4.2%） | 中性类占绝对多数，PRM 学不到判别信号 | 标签重采样 / 换连续分数 / 5-way，或对正负样本过采样 |

这四条的修复直接决定 phase1→phase2→phase3 的信号质量，比任何下游改动收益都大。

### 3.2 PRM / 过程奖励链路

- 当前 phase3 **只消费 `alpha_gate`（规则 `prm_annotator.label()`）**，训练出来的 3-way PRM head 没进 reward（见 `launch_split_ppo.sh` 注释）。→ 打通"训练出的 PRM → 过程奖励"这条链路，或干脆用更强 PRM（如 R1 式逐 token 评分）。
- 80% 中性标签让 PRM 输出近乎常数，α-gate 在 eval 阶段又是 telemetry（不门控），所以"α 突出 KG"目前是训练期自说自话。见 [[alpha-gate-eval-noop]] 记忆。

### 3.3 基座模型（天花板本身）

- `llama3-8b` 在多跳 QA 上 ~0.34 就是天花板。要突破，换 **`llama3-70b` / Qwen2.5-72B / DeepSeek-R1** 级基座，或走长 CoT 推理模型（R1-searcher 已在 `models/r1-searcher/`，但未接入主链）。
- 这一步是"巨大突破"的**必要条件**，8B 上做再多微调也撞不穿 ~0.35。

### 3.4 检索 / KG（收益有限，但要修对）

- KG 噪声仍占 48.4%（实体链接错误、无关属性）→ 换更强的实体链接 + 关系过滤，能把 +1~3pp 的 KG 效果提到 +3~5pp，但**不会质变**；
- MuSiQue/2Wiki 的 `question_kg_index` 缺失（只覆盖 hotpotqa-dev）→ 补齐离线 KG 缓存，消除对 live Wikidata 的依赖（当前断网直接归零）；
- KG 键 strip bug、α-gate 路径错位等"静默不一致"已修（见记忆），但要持续对齐 train/eval 口径。

### 3.5 训练 / 奖励设计

- **数据**：银标只有 hotpotqa → PPO 在 MuSiQue OOD 崩溃 -17pp。补 2wiki/musique 银标，或至少多数据集 SFT 锚点；
- **奖励**：EM-only（Plan B）→ 奖励黑客 + OOD 忘记。回退到 outcome 4.0 / step 1.5 后，进一步用"过程奖励 + 多样性 + 更强 KL"替代 EM 独大；
- **KL**：v2 的 KL 29.42 已说明策略飘离 SFT，`sft_anchor_weight 0.10 / interval 10` 拉不住 → 加 SFT replay 强度。

### 3.6 优先级排序（一条可执行的路线）

1. **换基座**（70B 或推理模型）—— 直接抬天花板；
2. **重建银标**（强 teacher + 强制 KG 引用 + 修 accepted + 修标签分布）—— 抬信号质量；
3. **回退 Plan B 后重训 PPO**（已回退，待跑）—— 恢复 KG 正则化，先回到 SFT 附近再谈超越；
4. 修检索/KG 细节（entity linker、离线缓存）—— 边际 +1~3pp，锦上添花。

---

## 四、v2 的修改改进：对"突出 KG 作用"是否有效？（IHR 等）

### 4.1 SFT 阶段：KG 有效，且方向一致

IHR（中间幻觉率，越低越好）是 KG 作用最干净的证据——它在 SFT 阶段三个数据集上**全部为负**：

| 数据集 | SFT noKG → SFT+KG | ΔIHR |
|--------|-------------------|------|
| HotpotQA | 22.1% → 20.6% | −1.5pp |
| 2Wiki | 28.7% → 20.6% | **−8.2pp** |
| MuSiQue | 31.9% → 26.3% | −5.6pp |

**结论**：加 KG 一致地降低中间幻觉率，且幅度与推理复杂度正相关（MuSiQue/2Wiki 多跳 > HotpotQA 少跳）。这说明 v2 的 KG 管线在 SFT 阶段**是有效突出 KG 作用的**——KG 的贡献主要体现为"减少中间步骤胡说"，而不是"提 EM"（EM 只 +1~2.7pp 且不显著）。

### 4.2 PPO② 阶段：KG 失效/反作用（Plan B 的后果，不是 KG 的问题）

| 数据集 | PPO② noKG → PPO② +KG | ΔIHR |
|--------|-------------------|------|
| HotpotQA | 18.0% → 25.0% | **+7.0pp（KG 使幻觉升高）** |
| 2Wiki | 15.0% → 13.0% | −2.0pp（微弱） |
| MuSiQue | 52.0% → 52.0% | 0.0pp* |

> \* MuSiQue 两臂注入 0 KG（wikidata 断网），Δ=0 是环境故障。

**解读**：在 PPO②（Plan B）上，KG 不再降低幻觉、HotpotQA 反而升高。这不是"KG 有害"，而是 Plan B 把 `kg_reward_share` 压到 2.7%（§2.2），模型在 PPO 阶段根本没被 KG 过程奖励约束，过拟合 EM 后，额外的 KG 上下文反而成了分布外噪声。**PPO 阶段 KG 失效是 Plan B 的后果，回退重训后应恢复 SFT 阶段的正确方向**。

### 4.3 结论

- **v2 的 KG 突出是有效的**：IHR 在 SFT 阶段三个数据集一致下降（−1.5 / −8.2 / −5.6pp），这是比 EM（+1~2.7pp NS）更有说服力的证据——KG 的作用是"抑制中间幻觉"，而这个维度 EM 反映不出来。
- **但 EM 上的效果微弱且不显著**（噪声地板内），所以"KG 提升 EM"目前不能作为结论写进 paper，只能写"KG 降低中间幻觉率（IHR 一致为负）"。
- **PPO 阶段的 KG 失效是 Plan B 造成的**，需要回退重训后才能对 PPO 阶段的 KG 作用下结论；MuSiQue +KG 则被 wikidata 断网挡住，暂无数据。

---

*配套：[statistics.md](statistics.md)（主表 + §十 已知限制）；记忆条目：[[musique-ppo-collapse]]、[[alpha-gate-eval-noop]]、[[eval-noise-floor-n100]]、[[eval-artefact-alignment]]。*



IHR judge 已彻底解锁并验证通过。当前状态总结：

事项	状态
rearag × hotpotqa	运行中（batch 2/3，~04:15 完成 batch2，~05:15 全部完成）
后台等待器	已挂（biodwna1w），完成后自动通知我接 trace
IHR judge	✅ deepseek-v4-pro + https://api.deepseek.com + 新 key 验证通过，凭据已写入 gitignored 的 .env
trace 修复	✅ 已就位（period 恢复 + max_iter=6），等 rearag 完成后 smoke(n=5) → 重跑(n=300)
Stage 2 (2wiki/musique)	待 stage 1 干净后补
一个小提醒（记录在案，不影响你现在的决定）：现有所有 IHR 结果文件（outputs/ihr_eval/、R8_final/）是用 deepseek-chat 判的，基线 IHR 改用 deepseek-v4-pro 后，这两批数字在 judge 上不一致、不能直接横向对比——paper 里需要注意统一口径。

接下来我保持静默等 rearag，完成后自动接上 trace smoke + 重跑 + IHR。