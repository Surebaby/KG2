# R9 v4 问题诊断报告

> 2026-07-21 | R9 v4 (Precision PRM + Dynamic KG Cache) 训练验证 + 评估结果分析

---

## 一、做了什么

| 阶段 | 内容 | 结果 |
|---|---|---|
| R9 代码重构 | `step_reward_scale`、`invalid_penalty`、precision PRM、`question_kg_index` 缓存 | ✅ 全部生效 |
| 远程训练 | 2000 步 PPO（分三次续训） | α=0.82, r_kg=0.29, KL=14.8 |
| 本地评估 | 3 数据集 × t={0, 0.7}、KG 链路全开 | EM=0.080, 2wiki/musique 提升但 hotpotqa 崩盘 |

---

## 二、核心数据对比

| 指标 | R9 v3 | R9 v4 | Δ |
|---|---|---|---|
| hotpotqa EM | 0.360 | 0.11 | -0.25 ↓ |
| 2wikimultihopqa EM | 0.087 | 0.10 | +0.01 ↑ |
| musique EM | 0.000 | 0.03 | +0.03 ↑ |
| **平均 EM** | **0.149** | **0.080** | -0.07 |
| 训练 α-gate | 0.02 | 0.82 | +40× |
| 训练 r_kg | 0 | 0.29 | ∞ |
| 推理 α-gate | N/A | 0.68 | — |

关键矛盾：**训练指标全面提升，推理时 hotpotqa 答案质量反而下降**。

---

## 三、已排除的原因

| 假设 | 结论 | 证据 |
|---|---|---|
| Wikidata 实时查询失败 | ❌ | RAW 输出有完整 KG 三元组引用，实体链接正常 |
| 推理格式丢失 | ❌ | RAW 输出包含 `[Step N]`、`Reasoning`、`Knowledge Used`、`Conclusion`、`Final Answer` |
| Prompt 训练/推理不一致 | ❌ | `build_rl_messages` = `build_inference_messages` = `build_sft_messages` |
| Temperature 不当 | ❌ | t=0 和 t=0.7 结果几乎一致（0.09 vs 0.11） |
| 续训导致格式退化 | ❌ | 训练末 step_rate=1.00, reasoning_content=1.00 |

---

## 四、确认的根因

### 根因 #1：检索召回率低（主因）

E5+BM25 RRF top-15 在多跳问题上召回不足：

```
5 个样本中只有 1 个答案在检索文档里
→ 模型引用 KG 三元组格式正确但内容无关
→ 推理路径偏离正确答案
```

对比：
- hotpotqa 文本检索质量较好 → R9 v3（无 KG 依赖）EM=0.36
- R9 v4 被训练成依赖 KG → KG 质量差 → EM 跌到 0.11
- 2wiki/musique 文本检索原本就差 → R9 v4 的 KG 至少提供了额外信号 → EM 小幅提升

### 根因 #2：奖励结构失衡

| 参数 | R9 v3 | R9 v4 | 影响 |
|---|---|---|---|
| `outcome_weight` | 10.0 | 2.0 | 正确答案奖励缩水 5× |
| `step_reward_scale` | 无 | 1.0 | 每步 citation 都有独立奖励 |

模型学会了"写 KG 引用 = 拿奖励"，但引用的内容不保证答案正确。**格式得分压过了答案得分**。

### 根因 #3：三次续训，optimizer 状态丢失

```
Run 1: 500步  KL 69→28
Run 2: 500步  KL 27→41  ← optimizer重置，策略震荡
Run 3: 1000步 KL 32→15
```

每次续训 optimizer（动量、学习率衰减）从头开始，训练动力学连续性断裂。

---

## 五、改进方案

### 短期（直接可实验）

| 改进 | 预期效果 | 改动量 |
|---|---|---|
| **retrieval_topk: 15→50** | 更多候选文档，提高答案覆盖率 | 1 行 |
| **换 BGE-large retriever** | MTEB 榜比 E5-base 高 ~5% | 换模型文件 |
| **outcome_weight: 2.0→10.0** | 重罚错误答案 | 1 行 |
| **step_reward_scale: 1.0→0.3** | 降低 citation 奖励 | 1 行 |

### 中期

| 改进 | 预期效果 |
|---|---|
| **查询改写 / 子问题分解** | 多跳问题拆成单跳分别检索 |
| **从头训练 2000 步（不续训）** | 消除 optimizer 状态丢失 |
| **换 E5-mistral-7b retriever** | 更强的语义理解 |

### 长期

| 改进 | 预期效果 |
|---|---|
| **检索+推理联合训练** | 让模型学会在 KG 质量差时 fallback 到文本 |
| **动态 α-gate** | 根据 KG 质量自适应调整 text/KG 权重 |

---

## 六、外部 Baseline 总结

| Baseline | EM avg | 状态 |
|---|---|---|
| Full SFT | 0.291 | ✅ |
| Elite SFT | 0.257 | ✅ |
| CoRAG | 0.167 | ✅ 唯一正常工作的外部模型 |
| R1-Searcher | 0.154 | ✅ 需 `<think>` prompt + max_tokens=1024 |
| R9 v3 | 0.149 | ✅ |
| Zero-shot | 0.103 | ✅ |
| R9 v4 | 0.080 | ⚠️ 训练指标提升但推理退化 |
| Naive RAG | 0.061 | ✅ |
| IRCoT | — | ❌ FlashRAG 兼容性 bug |
| Self-RAG | — | ❌ HF 被墙 |
| ReaRAG | — | ❌ HF 被墙，以 α=0 PPO 替代 |

---

## 七、结论

R9 v4 验证了缓存 + precision PRM 的**训练机制正确**（α=0.82, r_kg=0.29 远超 R9 v3 的 α=0.02, r_kg=0），但推理效果受制于**检索质量**和**奖励结构失衡**。下一步：

1. **提高检索召回率**（top-k 增大或换 retriever）——解决根因 #1
2. **调整奖励权重**（outcome_weight→10.0, step_reward_scale→0.3）——解决根因 #2
3. **从头训练 2000 步**（不续训）——解决根因 #3

---

*文件生成: 2026-07-21*
