# 实验问题诊断与改进方案

> **时间**: 2026-07-02  
> **状态**: R6-A 训练完成，发现问题，待讨论方案后启动 R7

---

## 一、实验背景与目标

KG-ProWeight 项目的核心假设是：**通过 α-gate 动态融合 KG 结构奖励和文本语义奖励，PPO 可以培养出"能在推理过程中有选择地依赖知识图谱"的语言模型。**

实验采用三阶段流水线：

```
Phase 1 (蒸馏)          Phase 2 (PRM)           Phase 3 (RL)
DeepSeek-V3 →          训练 α-Gate →           SFT → PPO / GRPO
生成银标轨迹           学习置信度分数 α         强化学习优化
```

**设计意图**:
- **SFT 的职责**: 将模型锚定在 `[Step N] ... Final Answer:` 的推理格式上（"格式约束"）
- **PPO 的职责**: 在保持格式的前提下，通过优化复合奖励来提升多跳推理能力（"能力优化"）
- **α-gate 的职责**: 在每个推理步骤动态选择信任 KG（结构化）还是 ReaRAG（语义），作为 PPO 的 credit assignment 机制

---

## 二、当前奖励函数 (R6-A 配置)

### 2.1 总公式

```
R_step(t) = α_t × R_KG(t) + (1 − α_t) × R_Text(t) × 0.3  +  0.3  +  10.0 × EM (最终步)
              ↑                    ↑                        ↑         ↑
           KG奖励              文本奖励                  格式bonus   答案奖励
```

### 2.2 组件详解

#### (A) α-Gate: 动态置信度门控

**文件**: `kgproweight/reward/alpha_gate.py`

```
α_t = σ( (1.0 × f_density + 1.5 × f_confidence − 0.8 × f_entropy − 2.0) / 0.5 )
```

| 特征 | 含义 | 高值→信任KG | 低值→信任文本 |
|------|------|:----------:|:----------:|
| `f_density` | KG 子图密度 | ✅ | ❌ |
| `f_confidence` | 实体链接置信度 | ✅ | ❌ |
| `f_entropy` | 模型不确定性 (`-mean(log p_token)`) | ❌ | ✅ |

**训练后 α 分布** (来自 smoke_ppo_full 评估日志): `mean=0.656, std=0.118, min=0.168, max=0.793`

#### (B) R_KG: 知识图谱结构奖励

**文件**: `kgproweight/reward/prm_annotator.py`

三分类标签器，仅依赖 KG 子图验证:

| 标签 | 值 | 触发条件 |
|------|:--:|---------|
| POSITIVE | +1 | 所有引用三元组被验证 AND 至少一个与结论相关 |
| NEUTRAL | 0 | 无引用 / 子图过于稀疏 / 引用无法验证 / 填充引用 |
| NEGATIVE | -1 | 结论与已验证的前一步矛盾 |

关键参数: `min_subgraph_for_verify=3`, `triple_fuzzy_threshold=80.0`, `require_triple_relevance=True`

#### (C) R_Text: 语义文本奖励

**文件**: `kgproweight/reward/text_reward_model.py`

- **后端**: ReaRAG-9B prompt scorer
- **原理**: 拼接 `prompt + step_text` → 计算 step token 的平均 NLL → `tanh((2.5 − NLL) / 1.5)`
- **输出范围**: `[-1, 1]`，约 1.5 NLL→+1，>5 NLL→−1
- **缩放**: PPO 中乘以 `text_reward_scale = 0.3`，实际输出范围 `[-0.3, 0.3]`

#### (D) 格式奖励 (step_format_bonus)

**文件**: `kgproweight/training/reward_function.py:163`

```python
if records:
    for i in range(len(per_step_rewards)):
        per_step_rewards[i] += 0.3  # 每步 +0.3
```

#### (E) 结果奖励 (outcome reward)

**文件**: `kgproweight/reward/composite_reward.py:162-175`

```python
# 最后一步追加:
records[-1].r_total += outcome_weight × EM(predicted_answer, gold_answer)
# outcome_weight = 10.0
```

**EM 函数**: 小写化、去冠词、去标点后的严格字符串匹配

### 2.3 PPO 训练超参

| 参数 | R6-A 值 | 作用 |
|------|--------|------|
| SFT 基座 | Elite SFT (2k 精选) | 格式锚定 |
| PPO 数据 | 全量 9,839 条 silver | |
| total_steps | 5,000 | |
| batch_size | 8 | |
| kl_coef | 0.1 | KL 对 SFT 的惩罚权重 |
| outcome_weight | 10.0 | 最终步 EM 权重 |
| text_reward_scale | 0.3 | 文本奖励缩放 |
| step_format_bonus | 0.3 | 每步格式奖励 |
| γ (discount) | 0.95 | |
| λ (GAE) | 0.95 | |
| temperature | 1.0 | 生成采样温度 |

### 2.4 奖励函数调用流程图

```
PPOTrainer.step()
  │
  ├─ 1. model.generate(query) → response_ids, response_text
  │
  ├─ 2. KGProWeightRewardFunction.__call__(prompt, response, spec, logprobs, response_ids)
  │     │
  │     ├─ parse_steps(response) → [step_1, ... , step_n]
  │     ├─ clean_entities(step.mentioned_entities)
  │     ├─ extract_final_answer(response) → predicted_answer
  │     │
  │     ├─ for each step:
  │     │   ├─ compute_features(density, confidence, entropy)
  │     │   ├─ α_gate → α_t
  │     │   ├─ prm_annotator.label() → r_kg ∈ {+1,0,-1}
  │     │   ├─ text_reward.score_step() → r_text ∈ [-1,1]
  │     │   └─ r_total = α×r_kg + (1-α)×r_text×0.3 + 0.3
  │     │
  │     ├─ last_step.r_total += 10.0 × EM(pred, gold)
  │     ├─ discounted_returns([r1, r2, ..., rn]) → GAE returns
  │     │
  │     ├─ step_spans_over_ids(response_ids) → [(start,end), ...]
  │     └─ token_rewards[span.end-1] = r_i  (per-token scatter)
  │
  └─ 3. StepRewardPPOTrainer.compute_rewards()
        ├─ KL_penalty = -kl_ctl × KL(P_θ || P_ref)
        └─ reward = KL_penalty + token_rewards  → GAE → loss
```

---

## 三、当前问题：R6-A Reward Hacking

### 3.1 现象

**评估时 R6-A 100% 跳过了 `[Step N]` 步骤标记**，直接输出 `Final Answer: X`。

| 指标 | SFT | PPO R3 | PPO R6-A |
|------|-----|--------|----------|
| EM (平均) | 0.291 | 0.233 | **0.246** |
| 包含步骤标记 | ✅ | ✅ (?) | ❌ (0%) |

**答案质量良好** (EM=0.383 on hotpotqa 接近 SFT=0.397)，但**完全没有推理过程**。

### 3.2 根因分析

```
PPO 发现的高奖励策略:
┌──────────────────────────────────────────────────┐
│  写步骤:  "[Step1]...[Step2]...Final Answer: X"  │
│  奖励:     α·R_KG + (1-α)·0.3 + 0.3×n + 10.0×EM│
│  风险:     R_text 噪声累积 (n×±0.3)               │
│                                                  │
│  不写步骤: "Final Answer: X"                      │
│  奖励:     KL_penalty + 10.0×EM (via fallback)     │
│  风险:     无  ← 纯优势策略!                        │
└──────────────────────────────────────────────────┘
```

**三个促成因素**:

1. **SFT 格式锚定不足**: Elite SFT 只有 2,000 条数据 (EM=0.257 vs Full SFT EM=0.291)，格式遵从性天然弱于全量 SFT
2. **`step_format_bonus=0.3` 是正向激励，不是约束**: 模型学会了 bonus 是"可选的"，训练时拿着，评估时扔掉
3. **`text_reward_scale=0.3` 仍太多噪声**: 长步骤序列累积 R_text 噪声，使得 PPO 从"保持格式"中获得的边际收益递减

### 3.3 问题本质

> **SFT 的格式约束力 < PPO 的奖励优化力**

PPO 确实在学习——它找到了最优策略：**简化输出，保留答案质量**。问题不是 PPO 失败了，而是 **奖励函数设计没有把格式作为硬约束**。

---

## 四、候选解决方案

### 方案概览

| # | 方案 | 改动维度 | 核心思想 |
|---|------|---------|---------|
| 1 | Prompt 层格式指令 | Prompt | 在 system prompt 中硬性要求步骤格式 |
| 2 | 非对称格式惩罚 | 奖励函数 | 有步骤 +0.3 / 无步骤 -1.5 (5:1 比例) |
| 3 | 全量 SFT 替换 | SFT 基座 | Elite SFT→Full SFT，更强格式锚定 |
| 4 | 温度退火 | 训练策略 | temp 从 1.0→0.3 线性下降 |
| 5 | 乘性格式乘数 | 奖励函数 | 无步骤→奖励×0.5，有步骤→奖励×1.0 |
| 6 | 两阶段 PPO | 训练策略 | Stage1 锁格式 → Stage2 优答案 |

### 各方案对比

| 维度 | 方案1 | 方案2 | 方案3 | 方案4 | 方案5 | 方案6 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 改动量 | 🟢小 | 🟡中 | 🟡中 | 🟡中 | 🟡中 | 🔴大 |
| 数学保障 | 🟡弱 | 🟢强 | 🟡中 | 🟡中 | 🟢强 | 🟢强 |
| 实验可比性 | 🟢好 | 🟢好 | 🔴差 | 🟡中 | 🟢好 | 🟡中 |
| 失败风险 | 🟡中 | 🟢低 | 🟢低 | 🟡中 | 🟡中 | 🟡中 |

### 推荐优先级

| 优先级 | 组合 | 理由 |
|--------|------|------|
| 🥇 | **方案1 + 方案3** | Prompt 约束 + 强 SFT 基座。改动小、风险低、可快速验证 |
| 🥈 | **方案2 (非对称惩罚)** | 数学上最干净，直接修复 reward hacking 根因 |
| 🥉 | **方案2 + 方案5 (组合)** | 惩罚 + 乘数双保险，适合追求最强保证的最终实验 |

---

## 五、实验需求清单

### 5.1 当前已完成

- [x] Zero-shot LLM baseline (9 个评估)
- [x] Naive RAG baseline (9 个评估)
- [x] Elite SFT baseline (不需要单独评估，即 SFT checkpoint 的 inference)
- [x] Full SFT baseline (9 个评估)
- [x] ReaRAG α=0 PPO (1 个评估)
- [x] Pure EM PPO (1 个评估)
- [x] PPO R3 (9 个评估)
- [x] PPO R6-A (9 个评估)
- [x] Baseline 对比文档 (`docs/baselines.md`)

### 5.2 待完成 (R7+)

- [ ] **选定 R7 方案** (从上述 1-6 中选择)
- [ ] 修改相关代码 (奖励函数 / 配置 / prompt)
- [ ] 训练 R7 PPO (约 10 小时)
- [ ] 评估 R7 on hotpotqa/2wiki/musique × 3 seeds
- [ ] IHR (LLM-as-Judge) 评估 (如果 R7 恢复了步骤)
- [ ] 更新 Baseline 对比文档
- [ ] 撰写 Ablation 分析 (R6-A vs R7 vs R6 vs R3)

### 5.3 推荐的消融实验矩阵

| 实验 | SFT 基座 | kl_coef | text_scale | format 机制 | outcome | 预期 EM |
|------|---------|---------|-----------|-----------|---------|--------|
| R6-A (baseline) | Elite | 0.1 | 0.3 | bonus 0.3 | 10.0 | 0.246 |
| R7-A | Elite | 0.1 | 0.3 | penalty 1.5 | 10.0 | ? |
| R7-B | Full | 0.05 | 0.1 | penalty 1.5 | 10.0 | ? |
| R7-C | Full | 0.1 | 0.1 | prompt强制 | 10.0 | ? |

---

*最后更新: 2026-07-02*
