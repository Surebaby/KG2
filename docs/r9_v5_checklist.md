# R9 v5 修改清单

> 状态: 仅分析，未执行 | 2026-07-21

---

## 修改总览

| 优先级 | 改动 | 文件 | 当前值 | 目标值 |
|---|---|---|---|---|
| P0 | outcome_weight | YAML | 2.0 | **10.0** |
| P0 | step_reward_scale | YAML | 1.0 | **0.3** |
| P1 | KG Relevance | prm_annotator.py | 仅 precision | **precision × relevance** |
| P2 | 检索 Reranker | hybrid.py + 新文件 | top-15 | **top-50 → rerank → top-5** |
| P3 | invalid_penalty | composite_reward.py | -outcome_weight (=-2) | **-5 或 -outcome_weight** |
| P3 | min_valid_steps | YAML | 2 | **2（不改）** |
| P3 | 从头训练 | 训练命令 | 续训 | **2000 步一次性** |

---

## 详细修改清单

### P0-1: outcome_weight 2→10

**文件**: `configs/training/phase3_ppo.yaml` L45

```yaml
# 改前:
    outcome_weight: 2.0
# 改后:
    outcome_weight: 10.0
```

**代码默认值验证**（不用改，确认 YAML 优先级最高）:
- `phase3_ppo.py:93`: `outcome_weight: float = 10.0` ✅ 默认已是 10
- `schemas.py:107`: `outcome_weight: float = 8.0`
- `composite_reward.py:55`: `outcome_weight: float = 1.0`
- `composite_reward.py:194`: `invalid_penalty = -self.outcome_weight` → 自动变为 -10

**影响链**:
```
YAML outcome_weight=10
  → phase3_ppo.py:590 传入 CompositeRewardModel
    → composite_reward.py:188  正确答案: +10×EM
    → composite_reward.py:194  invalid_penalty = -10
    → reward_function.py:284   最终 step 加 outcome
```

---

### P0-2: step_reward_scale 1.0→0.3

**文件**: `configs/training/phase3_ppo.yaml` L84

```yaml
# 改前:
    step_reward_scale: 1.0
# 改后:
    step_reward_scale: 0.3
```

**代码默认值**（不用改）:
- `phase3_ppo.py:105`: `step_reward_scale: float = 5.0` ← 默认偏高但 YAML 覆盖
- `composite_reward.py:58`: `step_reward_scale: float = 1.0`
- `composite_reward.py:101`: `r_total = (alpha * r_kg + (1-alpha) * r_text * text_reward_scale) * step_reward_scale`

**影响**: 每步 composite reward 乘以 0.3，降低 citation 对总 reward 的贡献。

**为什么不是 0**: r_kg 信号仍然需要存在（帮助 α-gate 学习），只是不能被放大到压过 outcome。

---

### P1: KG Relevance（新增功能）

**目标**: `R_KG = Precision × Relevance`

**文件**: `kgproweight/reward/prm_annotator.py`

**需要新增**: 一个 relevance 评分函数，判断 triple 是否支持当前问题的推理。

**方案 A（简单——优先推荐）**: 利用已有的 `_triple_relevant` 逻辑

```python
# prm_annotator.py:204 已有:
if self.require_triple_relevance and not self._triple_relevant(
    step.cited_triples, step.intermediate_conclusion
):
    return NEUTRAL  # ← 当前是二值判断

# 升级: 把 relevance 变成连续分数
# 修改 annotate() 返回 (precision, relevance) 而非仅 label
# R_KG = precision * relevance_score
```

**方案 B（高级——后续迭代）**: 用 LLM 判断 triple 是否支持当前问题

```python
# 新增: prm_annotator.py
def _triple_question_relevance(self, triples, question, conclusion):
    """LLM judge: 这些 triple 是否帮助回答这个问题？"""
    # prompt: "Question: {q}\nConclusion: {c}\nTriples: {t}\n
    #          Do these triples help answer the question? Score 0-1"
```

**推荐**: 先用方案 A（已有基础设施），方案 B 留到 R9 v6。

---

### P2: 检索 Reranker（新增功能）

**目标**: Retriever → Top50 → Reranker → Top5 → Prompt

**文件**:

| 文件 | 改动 |
|---|---|
| `kgproweight/retrieval/hybrid.py` | `DEFAULT_TOPK` 不能直接改（影响所有 baseline），新增独立参数 |
| `configs/training/phase3_ppo.yaml` | 新增 `rerank_topk: 5` 字段 |
| 新文件 `kgproweight/retrieval/reranker.py` | Reranker 实现（可选方案见下） |

**Reranker 方案**:

| 方案 | 实现 | 依赖 |
|---|---|---|
| A. Cross-encoder reranker | 用 sentence-transformers 的 CrossEncoder 对 top-50 重排序 | `pip install sentence-transformers` |
| B. LLM-based reranker | 用 Llama-3-8B 判断 passage 相关性 | 本机已有，但慢 |
| C. BM25 二次排序 | 用 question 对 top-50 做 BM25 重排 | 零依赖，已安装 |

**推荐**: 先用方案 A（Cross-encoder），最成熟。

---

### P3: 其他微调

#### P3-1: invalid_penalty

**文件**: `configs/training/phase3_ppo.yaml` + `composite_reward.py:194`

当前 `composite_reward.py:194`:
```python
invalid_penalty = -self.outcome_weight  # 动态计算
```

outcome_weight 改回 10 后，invalid_penalty 自动变为 -10。

但 R9 v5 建议 -5。需要独立参数:

```yaml
# YAML 新增:
    invalid_penalty: -5.0
```

```python
# composite_reward.py:194 改为:
invalid_penalty = self.invalid_penalty  # 从 YAML 读取
```

**影响**: 低。当前格式稳定性好，不需要强惩罚。

#### P3-2: min_valid_steps

**文件**: `configs/training/phase3_ppo.yaml` L69

```yaml
# 当前:
    min_valid_steps: 2
# R9 v5 建议: 保持 2 或改回 1
```

**判断**: 不改。R9 v4 格式稳定，2 步门槛合理。

#### P3-3: 从头训练

不是代码改动，是运行方式改动。YAML `total_ppo_steps: 2000` 一次跑完，不续训。

---

## 不改的代码（保留项确认）

| 功能 | 位置 | 确认 |
|---|---|---|
| ValidTrajectory | `reward_function.py:268` | ✅ 保留 |
| SFT Anchor | `phase3_ppo.py:595`, YAML L58-60 | ✅ 保留 (0.1) |
| Dynamic KG Cache | `phase3_ppo.py:607-616` | ✅ 保留 |
| Precision PRM | `prm_annotator.py:185-210` | ✅ 保留 |
| α-gate | `composite_reward.py:84-87` | ✅ 保留 |
| Content Gate (min_reasoning) | YAML L76 | ✅ 保留 (20) |
| kl_coef | YAML L28 | ✅ 保留 (0.15) |
| learning_rate | YAML L22 | ✅ 保留 (1e-6) |

---

## 修改文件汇总

| # | 文件 | 改动类型 | 优先级 |
|---|---|---|---|
| 1 | `configs/training/phase3_ppo.yaml` | outcome_weight: 2→10 | P0 |
| 2 | `configs/training/phase3_ppo.yaml` | step_reward_scale: 1→0.3 | P0 |
| 3 | `kgproweight/reward/prm_annotator.py` | 新增 relevance 评分 | P1 |
| 4 | `kgproweight/retrieval/reranker.py` | 新文件，Cross-encoder reranker | P2 |
| 5 | `kgproweight/retrieval/hybrid.py` | 新增 rerank_topk 参数 | P2 |
| 6 | `configs/training/phase3_ppo.yaml` | 新增 rerank_topk 字段 | P2 |
| 7 | `kgproweight/reward/composite_reward.py` | invalid_penalty 独立参数 | P3 |
| 8 | `configs/training/phase3_ppo.yaml` | total_ppo_steps: 500→2000 | P3 |

---

## 预期效果

```
R9 v4:  KG Citation Bias → EM=0.11 (hotpotqa)
R9 v5:  KG Utility导向 → 预期 EM=0.25+ (hotpotqa)
```

核心逻辑：
```
之前: reward = 0.3×(5步×0.5r_kg) + 2×EM = 0.75 + 0  = 0.75 (不回答也赚钱)
之后: reward = 0.09×(5步×0.5r_kg) + 10×EM = 0.23 + 10 = 10.23 (答对才赚钱)
```

---

*文件生成: 2026-07-21 | 等待确认后执行*
