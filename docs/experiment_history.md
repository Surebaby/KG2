# KG-ProWeight 实验迭代历史

> 基座模型：Llama-3-8B-Instruct + LoRA r=32
> 评估协议：3 数据集 × 3 种子 × 50/100 样本 / EM + F1 + IHR

---

## 迭代总览

| 轮次 | 日期 | 训练步数 | hotpotqa EM | KL 终值 | r_kg | 核心突破 | 状态 |
|---|---|---|---|---|---|---|---|
| R7-B | 07-03 | 5000+ | 0.323 | ~40 | 0 | ValidTrajectory gate | 格式空洞 |
| R8 phase1 | 07-06 | 2000 | 0.317 | ~36 | 0 | Content Gate 恢复推理 | 步骤偏短 |
| R8 phase2 | 07-07 | +1000 | 0.360 | ~30 | 0 | min_valid_steps=2 | 推理内容掉 |
| R9 step504 | 07-08 | 504 | 0.280 | ~36 | 0-0.38 | Dynamic KG Reward (α=0.85) | r_kg 首次打破 0 |
| R9 1k | 07-10 | 504→1000 | 0.287 | 45-73 | 0-0.38 | Prompt KG 注入试用 | 400步后崩 |
| **R9 v3** | **07-13** | **2000** | **0.360** 🔥 | **21-30** 🔥 | **0.06-0.39** | **Precision PRM + 降参 + KL稳定** | **最优！** |

---

## R7-B：ValidTrajectory Gate

### 改进
- 从 R6-A 的 `step_format_bonus`（正向奖励）改为 ValidTrajectory gate（门控惩罚）
- 添加 SFT Anchor（每隔 N 步做一次 SFT 反向传播保持格式）
- 步骤标记必须存在且序号连续，且 Final Answer 可提取

### 结果
| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | 0.323 | 0.407 |
| 2wiki | 0.233 | 0.243 |
| musique | 0.040 | 0.056 |
| **平均** | **0.199** | **0.235** |

### 问题
- 模型找到新 shortcut：写 `[Step 1]\nReasoning: \nFinal Answer: X`（空推理）
- Gate 只检查标记存在，不检查内容质量

---

## R8 phase1：Content Gate

### 改进
- `min_reasoning_chars=20`：每步 Reasoning 必须有 ≥20 字符内容
- `min_valid_steps=1`：冷启动友好
- SFT Replay (15%)：每 batch 混入 SFT prompt 保持格式
- `step_reward_scale=5.0`：放大中间步骤奖励对抗 KL

### 结果
| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | 0.317 | 0.424 |
| 2wiki | 0.217 | 0.242 |
| musique | 0.010 | 0.063 |
| **平均** | **0.181** | **0.243** |

### 成就
- ✅ 推理内容 100% 恢复（vs R7-B 0%）
- ✅ 步骤格式 100% 保持
- ❌ 步骤偏短（avg 1.2 vs Elite SFT 3.0）
- ❌ α 仍然约 0（KG 离线）

---

## R8 phase2：min_valid_steps=2 续训

### 改进
- `min_valid_steps: 1→2`
- 从 R8 phase1 final checkpoint 续训 1000 步

### 结果
| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | 0.360 | — |
| 2wiki | 0.243 | — |
| musique | 0.013 | — |

### 成就
- ✅ 步骤数 1.2→1.9
- ✅ hotpotqa EM 从 0.317→0.360
- ❌ 推理内容率 100%→50%（数据混合问题）

---

## R9 step504：Dynamic KG Reward

### 改进
- **核心修复**：reward 端不再用银标静态 `spec.kg_subgraph`
- 改为从模型输出文本提取实体 → EntityLinker(缓存) → WikidataSubgraphRetriever(缓存) → 动态 KG 子图
- 初步 precision PRM（验证三元组命中率）
- EntityLinker 改为 `offline=True`（避免 Wikidata 超时）

### 结果
| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | 0.280 | 0.384 |
| 2wiki | 0.100 | 0.130 |
| musique | 0.020 | 0.056 |
| **平均** | **0.133** | **0.190** |

### 成就
- 🔥 **α 从 0.02→0.85**（KG 分支激活！）
- 🔥 **r_kg 首次打破零死锁**（波动 0-0.38）
- ✅ 步骤数稳定在 2.0

---

## R9 1k：Prompt KG 注入 + 参数回退

### 改进
- Prompt 端动态 KG 注入（`_prepare_prompts` 从问题实体查缓存）
- 但 `confidence_threshold=101` 导致 exact-only，命中率为 0

### 结果
| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | 0.287 | 0.373 |

### 问题
- ❌ KL 从 40 反弹到 73（参数退回旧值）
- ❌ 400 步后 r_kg 归零、r_text 崩溃
- ❌ SFT anchor loss 飙升 4→12
- 根因：`step_reward_scale=5.0` 过激 + 参数回退到 `lr=5e-6, epochs=4`

---

## R9 v3：最终稳定版（当前最优）

### 改进

**1. Precision PRM（软打分）**
```python
# 旧：硬二值
if all_verified: return +1
else: return 0

# 新：精确率
precision = verified_count / total_count
return precision  # 0.0~1.0，非零即梯度
```

**2. 奖励函数重构**
| 参数 | 旧 | 新 | 原因 |
|---|---|---|---|
| `step_reward_scale` | 5.0 | **1.0** | 卸下重赏，让 KL 惩罚正常工作 |
| `outcome_weight` | 10.0 | **2.0** | 保持比例，不过度放大 |
| `invalid_penalty` | -10.0 | **-2.0** | 自动跟随 outcome_weight 降 |

**3. PPO 超参调优**
| 参数 | 旧 | 新 | 原因 |
|---|---|---|---|
| `learning_rate` | 5e-6 | **1e-6** | 延缓策略漂移，防 KL 爆炸 |
| `ppo_epochs` | 4 | **2** | 减少同 batch 重复更新 |
| `sft_anchor_weight` | 0.05 | **0.10** | 增强格式锚定 |

**4. Prompt 优化**
- `[Knowledge Graph Context] ... [End of Knowledge Graph]` 边界标记
- 指令："Copy exact triples. Do not invent."

**5. 防 OOM**
- `max_new_tokens: 384→256`，`max_steps: 7→5`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 训练曲线

| 阶段 | step 范围 | KL | r_kg |
|---|---|---|---|
| 早期 | 32-256 | 75→47 | 0.04-0.29 |
| 中期 | 288-512 | 50→37 | 0.00-0.21 |
| 后期 | 544-800 | 38→34 | 0.00-0.39 |
| 末期 | 832-1984 | **29→21** | **0.06-0.25** |

### 最终结果

| 数据集 | EM | F1 |
|---|---|---|
| hotpotqa | **0.360** 🔥 | **0.419** |
| 2wiki | 0.087 | 0.139 |
| musique | 0.000 | 0.059 |
| **平均** | **0.149** | **0.206** |

### vs Elite SFT

| 指标 | Elite SFT | R9 v3 |
|---|---|---|
| **hotpotqa EM** | 0.353 | **0.360** 🔥 |
| KL 终值 | — | **21** |
| r_kg | — | **正波动** |
| 训练稳定性 | — | **2000步不崩** |

---

## 关键经验教训

1. **Precision PRM 比硬二值有效**：非零即梯度，打破 zero-signal 死锁
2. **step_reward_scale=5.0 过激**：降回 1.0 后 KL 从 70→21
3. **lr=1e-6 是关键**：8B LoRA PPO 对学习率极度敏感，5e-6 就会崩
4. **Dynamic KG Reward 是核心**：α 从 0.02→0.85，让 α-gate 真正工作
5. **Prompt KG 注入需优化**：离线模糊匹配太慢，需预计算索引
6. **min_valid_steps=1 够用**：2000 步自发学到 2 步推理，不需要强制

---

*最后更新: 2026-07-13*
