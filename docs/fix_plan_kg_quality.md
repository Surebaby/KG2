# KG 质量修复方案

**诊断日期**: 2026-08-08
**症状**: KG 注入零效果（SFT+KG = SFT noKG = 0.340），α 卡在 0.12，PPO ≤ SFT
**根因**: KG 三元组约 70% 是噪声；α gate 正确地学会了忽略它们

---

## 一、系统架构

```
问题 → EntityLinker → Wikidata SPARQL 查询 → kg_filter (硬删除→配额→打分→Top-30) → prompt
                                                                                      ↓
文本检索 → RRF 融合 → rerank top-10 → 截断到 3860 tokens → 段落(950 tok) + KG(300 tok) → α gate → LLM
```

### α Gate 定义

```
α = σ( ( [f_density, f_confidence, f_entropy] · W + b ) / τ )
```

- **W** = [1.0, 1.5, -0.8]：三个特征的权重
- **b** = -2.0：偏置项（强低 α 先验）
- **τ** = 0.5：温度参数（控制 sigmoid 陡峭度）

### 三个输入特征

| 特征 | 含义 | 推理时的问题 |
|------|------|------------|
| f_density | KG 子图密度 = \|E\|/(\|V\|+ε) | 各题差异不大 |
| f_confidence | 实体链接置信度（模糊匹配） | 各题差异不大 |
| f_entropy | 模型预测熵 = -mean(log p_token) | **推理时恒为 1.0**（见根因 1） |

### 奖励函数

```
R_total = (α × R_KG + (1-α) × R_Text × 0.3) × 1.5 + 4.0 × EM
```

---

## 二、六项根因诊断

### 根因 1 🔴🔴🔴：推理时 α gate 是盲的

**位置**: `kgproweight/pipeline/kg_proweight_pipeline.py:288-298`

**问题**: `_compute_alpha_stats()` 调用 `compute_features()` 时传入 `logprobs=None`，导致 `f_entropy` 默认为 1.0。α gate 看到每道题、每一步的熵完全相同。

**影响**: α 仅依赖 graph_density 和 entity_linker 置信度——这两个信号跨题目基本恒定（KG 密度相似，链接置信度相似）。gate 无法感知模型真实的困惑程度，形同盲人。

---

### 根因 2 🔴🔴：b=-2.0 将 α 锁死在 0.12

**位置**: `kgproweight/reward/alpha_gate.py:36`

**问题**: 计算典型特征值下的 α 范围：

```
典型特征 (density=0.1, confidence=0.7, entropy=1.0):
  logit = (0.1×1.0 + 0.7×1.5 + 1.0×(-0.8) + (-2.0)) / 0.5
        = (0.1 + 1.05 - 0.8 - 2.0) / 0.5
        = -3.3
  α = σ(-3.3) ≈ 0.036

最优情况 (density=0.5, confidence=1.0, entropy=1.0):
  logit = (0.5 + 1.5 - 0.8 - 2.0) / 0.5 = -1.6
  α = σ(-1.6) ≈ 0.17

最差情况 (density=0, confidence=0, entropy=1.0):
  α = σ((-2.0)/0.5) = σ(-4.0) ≈ 0.018
```

**无论输入什么特征，α 永远在 [0.018, 0.17] 之间**，gate 根本无法区分"KG 有用"和"KG 无用"。

**影响**: gate 的参数空间在低端饱和，丧失了区分能力。

---

### 根因 3 🔴🔴：实体链接缺少上下文

**位置**: `kgproweight/kg/entity_linker.py:312-400`

**问题**: `link_single()` 接受 `retrieved_titles` 参数用于消歧，但推理 pipeline (`_build_kg_context:152-154`) 调用时没有传入检索到的段落。实体消歧只用问题文本，不用检索证据。

**影响**:
- "What Are Little **Girls Made Of**" → "Made" 被链接到荷兰小镇而非歌曲名
- "Guns N **Roses**" → "Roses" 被链接到 Saint Jhn 的歌曲而非 Guns N Roses
- 产生事实性错误的三元组，模型却信任它们（标注为 "Knowledge Graph"）

---

### 根因 4 🔴：三层过滤后仍有 70% 噪声

**位置**: `kgproweight/kg/kg_filter.py`

**问题**: `score_triple()` 的 `entity_anchor` 权重为 0.30——只要实体名称出现在问题中，triple 就能拿到 ≥0.30 的基础分，无论实际有无价值。

以下三类 PID 靠 entity_anchor 高分存活，占总三元组的 21%：

| PID | 关系名 | 示例 | 为什么是噪声 |
|-----|--------|------|------------|
| P735 | given name | (Ed Wood, given name, Ed) | 纯冗余 |
| P734 | family name | (Ed Wood, family name, Wood) | 纯冗余 |
| P21 | sex or gender | (Ed Wood, sex or gender, male) | 对 QA 无帮助 |

**影响**: 每道题平均 300 个 prompt token 浪费在无意义的三元组上。

---

### 根因 5 🟡：文本奖励被严重压低

**位置**: `kgproweight/training/phase3_ppo.yaml`

**问题**: `text_reward_scale=0.3` 意味着 R_Text 的贡献打了三折，而 R_KG 保持全额。当 R_KG 噪声大时，整体步级奖励被噪声主导。

**影响**: PPO 优化方向偏向"正确引用 KG 格式"而非"正确的文本推理"。

---

### 根因 6 🟡：所有步骤 α 完全相同

**位置**: `kgproweight/reward/alpha_gate.py:50-60`

**问题**: Gate 每步计算一次 α，但由于所有特征跨步相同（同一个 KG 子图、同一个 linker 置信度、同一个熵=1.0），同一题的所有步 α 完全一致。

**影响**: 模型无法表达"第一步需要 KG 查事实，后续步骤靠文本推理"。KG 要么全开要么全关。

---

## 三、修复方案

### Phase A — 即刻生效（改 5 行代码，无需重训）

#### A1. 修复 α gate 的工作范围

**文件**: `kgproweight/reward/alpha_gate.py` 第 36 行
**改动**: `init_bias` 从 -2.0 改为 0.0

```
修复前: α ∈ [0.018, 0.17]  → 永远低 α
修复后: α ∈ [0.12, 0.92]   → 可以表达"KG 有用"和"KG 无用"
```

**原理**: b=0 时，gate 自然 α 范围变为 [0.12, 0.88]。Phase 2 的校准损失（BCE with coverage_target）会自动调整 gate 参数使之匹配实际 KG 质量。不是强制拉高 α，而是给 gate 表达"高 α"的能力。

**注意**: 这需要重新训练 Phase 2，否则已训练的 gate 参数 W 是在 b=-2.0 的饱和区学出来的，不能直接改 bias 用。如果不想重训 Phase 2，可以先单独调 bias 在推理时测试效果。

#### A2. 减少 KG 三元组数量

**文件**: `kgproweight/pipeline/kg_proweight_pipeline.py`（`max_kg_triples` 默认值）
**改动**: `max_kg_triples` 从 30 改为 12

**原理**: 12 个三元组 × 89% KG 覆盖率 = 每道题约 10.7 个三元组。经过 Phase B 的噪声过滤后，真正有用的约 6-7 个。KG 段从 300 token 降到约 130 token，减少模型注意力浪费。

**注意**: 这个改动需要同步到 `configs/training/phase3_ppo.yaml` 中的 `ppo_max_kg_triples`，保持训练和推理一致。

---

### Phase B — 短期改进（改 ~30 行代码，无需重训，但建议重建 KG 缓存）

#### B1. 硬删除无价值的关系类型

**文件**: `kgproweight/kg/kg_filter.py`
**改动**: 在 `_HARD_DELETE_RELATION_LABELS` 集合中添加：

```python
"given name",          # P735 — 对 QA 永远无用
"family name",         # P734 — 对 QA 永远无用
"sex or gender",       # P21  — 对 QA 永远无用
```

**原理**: 这三类 PID 占三元组的 21%。三元组本身是真的（人的 given name 确实是他的名字），但对答题毫无帮助。`entity_anchor`（0.30 权重）因为实体名称出现在问题中而给出高分，使它们存活到最后。

**预期效果**: 每道题直接去除约 4.8 个纯噪声三元组。

#### B2. 增加最低分阈值

**文件**: `kgproweight/kg/kg_filter.py` 中的 `filter_and_rank_triples()` 函数
**改动**: 在打分排序后、Top-K 选取前，增加：

```python
scored = [(s, t) for s, t in scored if s >= 0.25]
```

**原理**: 低于 0.25 分的 triple 与问题几乎无关。一个只拿到 entity_anchor（0.30 × 1.0 = 0.30）但其他维度全为零的 triple，仍有 0.30 分；但 relation、passage、path 全没命中的 triple 对 QA 没有贡献。

**预期效果**: 再去除约 20% 的低相关度三元组。

#### B3. 推理时使用检索上下文做实体消歧

**文件**: `kgproweight/pipeline/kg_proweight_pipeline.py` 中的 `_build_kg_context()` 方法
**改动**: 在检索完成后，提取段落标题列表，传给 `link_single(retrieved_titles=passage_titles)`。

**原理**: Entity Linker 的 `_score_candidates()` 已支持 `retrieved_titles` 参数（提供 0.10 的 title_support 加分），只是推理时没有传入。使用检索到的文档标题作为消歧信号。

**预期效果**: 减少"Made → 荷兰小镇"类实体链接错误——错误实体的名称不会出现在检索段落标题中。

---

### Phase C — 中期改进（改 ~50 行 + 重建 KG 缓存，约 2 小时）

#### C1. 基于检索段落的 KG 验证过滤器

**文件**: `kgproweight/kg/kg_filter.py` — 新增函数
**改动**:

```python
def filter_by_passage_support(triples, passages, min_support=1):
    """只保留至少有一个实体出现在检索段落中的三元组"""
    passage_text = " ".join(
        str(p.get("contents", "") or p.get("text", "")) 
        for p in passages[:10]
    ).lower()
    kept = []
    for triple in triples:
        h_ok = triple[0].lower() in passage_text
        t_ok = triple[2].lower() in passage_text
        if h_ok or t_ok:
            kept.append(triple)
    return kept
```

**原理**: 如果三元组的实体从未出现在任何检索段落中，模型就没有文本证据来理解和利用这个 KG 事实。KG 信息缺少文本锚点，成为孤立噪声。

**预期效果**: 自动消除实体链接错误（错误实体不出现在段落中），只保留有文本支撑的事实。这是对根因 3 的兜底修复。

#### C2. 用严格过滤重建 KG 缓存

**文件**: 新建 `scripts/prepare/rebuild_kg_cache.py`
**改动**: 重新生成 `question_kg_index_v2.json`，使用新的过滤参数：
- `max_kg_triples=15`（替代 30）
- 硬删除 P735 (given name)、P734 (family name)、P21 (sex or gender)
- 评分阈值 ≥ 0.25
- Passage-verified 过滤（需要先对每个问题做检索）

**预期效果**: 清洁的 KG 缓存，每道题约 6-8 个高质量三元组（从当前的 23 个大幅下降）。

---

### Phase D — 深度改进（改 ~50 行 + 完整重训，需数天）

#### D1. 修复推理时的语义熵

**文件**: `kgproweight/pipeline/kg_proweight_pipeline.py:288-298`
**改动**: 生成过程中从 generator 收集每步的 token logprobs，传入 `compute_semantic_entropy()`。需要 generator 在返回 tokens 的同时返回 `scores`。

**原理**: 修复根因 1。使用真实的 per-step 模型困惑度作为 f_entropy，α gate 才能真正感知模型的确定性。

**预期效果**: 模型困惑时（高熵）→ α 升高，多依赖 KG；模型确定时（低熵）→ α 降低，信任文本推理。

#### D2. 每步独立 α 的 Prompt 工程

**文件**: `kgproweight/data/prompts.py` — `SFT_SYSTEM_PROMPT`
**改动**: 添加指令：

```
- In early steps, use the Knowledge Graph to establish background facts.
- In later steps, rely more on the retrieved passages for reasoning.
```

**原理**: 即使架构上所有步骤的 α 相同，prompt 工程可以引导模型在前几步更多引用 KG，后几步更多依赖文本。

#### D3. 奖励函数重平衡

**文件**: `configs/training/phase3_ppo.yaml`
**改动**: `text_reward_scale` 从 0.3 改为 0.7，`step_reward_scale` 从 1.5 改为 1.0

**原理**: 当前缩放使 R_Text 几乎无关（0.3×），PPO 几乎纯粹为 KG 引用格式优化。提高 text_reward_scale 使文本推理质量在奖励中占比更大。

**预期效果**: PPO 学会在 KG 引用和文本推理质量之间取得平衡。

---

## 四、实施顺序

### 第 1 天（即刻验证）

```
A1: bias 改为 0.0（需先重训 Phase 2，或暂时只改推理时参数测试效果）
A2: max_kg_triples 30 → 12
B1: 硬删除 3 个 PID
→ 重新评测 SFT+KG wiki18 n=300（约 21 分钟）
→ 目标：α > 0.15，SFT+KG > SFT noKG ≥ 2pp
```

### 第 2 天（如果第 1 天有效果）

```
B2: 评分阈值 ≥ 0.25
B3: passage 感知实体链接
C1: passage 验证过滤
C2: 重建 KG 缓存（约 2 小时计算）
→ 用新缓存重新评测
```

### 第 3 天+（如需深度改进）

```
D1: 推理时真实 logprobs
D2: per-step prompt 优化
D3: reward 重平衡
→ 完整重训 Phase 2 + Phase 3
```

---

## 五、成功标准

| 指标 | 当前值 | 目标值 | 验证方式 |
|------|--------|--------|---------|
| SFT+KG vs SFT noKG | 0.340 = 0.340（差 0pp） | KG 比 noKG 高 ≥ 2pp | wiki18 n=300 McNemar p<0.05 |
| α 均值 | 0.117 | 0.20–0.35 | alpha_distribution.jsonl |
| α 范围 | [0.02, 0.26] | [0.05, 0.70] | alpha_distribution.jsonl |
| 有用 KG triples 占比 | ~30% | ≥ 60% | 随机抽查 20 道题 |
| 实体链接错误率 | ~10% | ≤ 3% | 随机抽查 50 个 mention |
| PPO vs SFT | 0.327 < 0.340 | PPO ≥ SFT + 1pp | wiki18 n=300 |

---

## 六、风险评估

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 拉高 bias 导致 α 过高（KG 被过度信任） | 中 | Phase 2 重训时校准损失会自动纠正 |
| KG 对 HotpotQA 本质上无帮助（文本已足够） | 低-中 | 在 2wikimultihopqa / musique 上测试（更需要跨文档推理） |
| 清洁 KG 后 PPO 仍不优于 SFT | 中 | PPO 可能存在固有的回归均值现象；可尝试 GRPO 替代 |
| Entity Linker 改进需要 GENRE 模型 | 低 | 当前上下文打分（无需 GENRE）已足够；GENRE 是可选的增强 |
| 重建 KG 缓存需要 Wikidata 在线查询 | 中 | 离线模式下用已有缓存兜底；分批查询避免限流 |
