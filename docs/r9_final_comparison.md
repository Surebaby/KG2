# R9 系列完整对比与问题总结

> 2026-07-21 | 从 R9 v3 到 R9 v5 的演进

---

## 一、R9 演进

| 版本 | 改动 | hotpotqa | 2wiki | musique | **平均** |
|---|---|---|---|---|---|
| R9 v3 | Precision PRM + Dynamic KG | 0.360 | 0.087 | 0.000 | **0.149** |
| R9 v4 | +缓存 +step_reward_scale +invalid_penalty | 0.11 | 0.10 | 0.03 | **0.080** |
| **R9 v5** | **+outcome=10 +scale=0.3 +relevance** | **0.34** | **0.25** | **0.13** | **0.240** |

---

## 二、当前排名

| Baseline | EM avg |
|---|---|
| Full SFT | 0.291 |
| Elite SFT | 0.257 |
| **R9 v5 (500步)** | **0.240** |
| CoRAG | 0.167 |
| R1-Searcher | 0.154 |
| Zero-shot | 0.103 |
| Naive RAG | 0.061 |

---

## 三、已解决的问题

### ✅ Reward 结构失衡

outcome_weight 2→10, step_reward_scale 1→0.3。EM 0.080→0.240。

### ✅ KG Citation Bias

precision × relevance 替代纯 precision，真实但无关的 triple 不得分。

### ✅ Precision PRM 零信号死锁

verified/total 连续值打破 0 奖励。

### ✅ KG 缓存

8493 entries, 861k triples, 100% 命中。

---

## 四、当前问题

### 问题 #1：KG 三元组噪音

86 万三元组中，`instance of` 11 万次、`subclass of` 5 万次、`has part(s)` 4.5 万次。`country of citizenship`（真正有用）仅 0.7 万次排名第 19。

→ **方向**：重建缓存时加 relation 白名单

### 问题 #2：实体链接错误

"Big Stone Gap" 链到镇而非电影，"Corliss Archer" 链到维基消歧义页。

→ **方向**：LLM 辅助消歧义或在构建阶段验证实体相关性

### 问题 #3：检索召回率

E5+BM25 top-15，5 样本中仅 1 个答案在文档里。

→ **方向**：top-50 + reranker / 换 retriever / 查询改写

### 问题 #4：500 步未收敛

KL=50，2000 步可能进一步提分。

→ **方向**：跑满 2000 步

---

## 五、优先级

| 优先级 | 问题 | 预期提升 | 改动量 |
|---|---|---|---|
| P0 | 跑满 2000 步 | +0.02~0.05 | YAML 一行 |
| P1 | KG relation 白名单 | +0.03~0.08 | 重建缓存 |
| P2 | 检索 reranker | +0.02~0.05 | 新模块 |
| P3 | 实体链接改进 | +0.03~0.05 | 改逻辑 |

---

## 六、一句话

**R9 v5 证明 reward 方向正确（EM 0.080→0.240），瓶颈已从"模型不会用 KG"转移到"KG 本身质量不够"。下一步改善 KG 内容而非继续调 reward。**

---

*文件生成: 2026-07-21*
