# Legacy KG 修复 - 快速开始

> **5 分钟了解整个系统，15 分钟运行第一次审计**

---

## 一句话总结

**分层诊断 Legacy KG 的信息损失点，精确修复，而非盲目优化。**

---

## 核心问题

当前 Legacy KG 对 HotpotQA/MuSiQue 无效。原因是：
- A) 原始 Wikidata 缓存里**根本没有**目标关系/值？
- B) 有，但被**错误链接**或**错误过滤**了？

**如果是 B**，可以在不增加外部知识的情况下修复。  
**如果是 A**，需要转向 passage-derived edges。

---

## 三步走

```
1️⃣ 运行审计  →  2️⃣ 查看瓶颈  →  3️⃣ 选择修复路线
   (2 小时)        (10 分钟)        (根据结果)
```

---

## Step 1: 运行审计（最快 2 小时）

### 1.1 检查依赖

```bash
python scripts/preflight_audit_check.py
```

**预期输出**:
```
✓ All checks passed (20/20)
Ready to run audit:
  bash scripts/run_legacy_kg_audit.sh
```

如果有失败项，先修复。

---

### 1.2 启动审计

```bash
bash scripts/run_legacy_kg_audit.sh
```

**运行时间**: 约 2 小时（200 题，离线缓存模式）

**进度提示**:
```
Processing dataset: hotpotqa
  Processed 10/100 questions...
  Processed 20/100 questions...
  ...
✓ Completed hotpotqa: 100 audits

Processing dataset: musique
  ...
✓ Completed musique: 100 audits

Generating audit report...
✓ Audit complete: 200 total questions audited
```

---

### 1.3 查看结果

**控制台摘要**:
```
================================================================================
LEGACY KG COVERAGE AUDIT SUMMARY
================================================================================
Datasets: hotpotqa, musique
Sample: 200 questions (seed=46)

Primary bottleneck: L3_RELATION_MISSING

Top 3 bottlenecks:
  1. L3_RELATION_MISSING: 82 (41.0%)
  2. L4_FILTERING_REMOVED_USEFUL: 46 (23.0%)
  3. L2_LOW_LINKING_QUALITY: 30 (15.0%)

Repair feasibility:
  - legacy_rerank_fixable: 46 (23.0%)
  - entity_linking_fixable: 30 (15.0%)
  - passage_derived_required: 82 (41.0%)
================================================================================
```

**详细报告**:
```bash
# Markdown 摘要（易读）
cat reports/legacy_kg_audit/audit_n100_seed46_*.md

# JSON 详情（机器可读）
cat reports/legacy_kg_audit/audit_n100_seed46_*.json
```

---

## Step 2: 解读瓶颈（10 分钟）

### 2.1 瓶颈类型速查

| 瓶颈 | 含义 | 修复策略 |
|------|------|----------|
| **L1_NO_MENTIONS** | 问题中没提取到实体 | 改进 mention extraction |
| **L2_LINKING_FAILURE** | 实体链接全失败 | 修复 entity linker |
| **L2_LOW_LINKING_QUALITY** | 链接成功率 <40% | 用 passage titles 消歧 |
| **L3_ANSWER_NOT_IN_RAW** | 答案实体不在缓存 | Passage-derived required |
| **L3_RELATION_MISSING** | 目标关系不在缓存 | Passage-derived or expand |
| **L4_COMPLETE_FILTERING_LOSS** | 全被过滤掉 | 调整过滤阈值 |
| **L4_FILTERING_REMOVED_USEFUL** | 有用边被过滤 | 修复 reranker |
| **L5_DOWNSTREAM** | KG 可用但下游有问题 | 检查 prompt/model |

### 2.2 修复可行性速查

#### 场景 A: `legacy_rerank_fixable + entity_linking_fixable > 60%`

**瓶颈主要在**: 链接质量差、过滤太严

**修复方向**: **Legacy KG v2**
- 改进实体链接（用 passage context 消歧）
- Query-aware relation ranking
- Precision-first 过滤
- 移除 occupation/instance-of 等泛化边

**预期收益**: +2~5pp EM

---

#### 场景 B: `passage_derived_required > 60%`

**瓶颈主要在**: 原始 Wikidata 缓存里没有目标关系

**修复方向**: **Passage-derived Edges**
- 从同一 top-10 passages 提取结构化关系
- 标记 source=passage
- 不算新知识源（passages 本来就给所有方法）

**预期收益**: +5~8pp EM

---

#### 场景 C: 混合瓶颈

**瓶颈分散**: 各类问题都有

**修复方向**: **Hybrid Evidence Graph**
- Legacy KG v2（修复链接/排序）
- Passage-derived edges
- 统一融合，precision-first 去重

**预期收益**: 综合最优

---

## Step 3: 选择路线（根据结果）

### 路线 A: Legacy KG v2

**适用**: 场景 A

**TODO 脚本**:
```bash
python scripts/prepare/build_legacy_kg_v2.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46.json \
  --improvements entity_linking query_aware_ranking \
  --output indexes/kg_cache/question_kg_index_legacy_v2.json
```

**对照验证**:
```bash
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A B D
```

**判定门**: B.EM ≥ A.EM - 0.02（不退化）

---

### 路线 B: Passage-derived Edges

**适用**: 场景 B

**TODO 脚本**:
```bash
python scripts/prepare/extract_passage_derived_edges.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46.json \
  --passages data/retrievals/rrf100_rerank10/ \
  --output indexes/passage_edges/hotpot_musique.json
```

**对照验证**:
```bash
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A C D
```

**判定门**: C.EM > D.EM + 0.05 AND citation_rate > 0.7

---

### 路线 C: Hybrid

**适用**: 场景 C

**步骤**: 先运行路线 A + B，然后运行完整四臂对照

**对照验证**:
```bash
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A B C D
```

---

## 关键约束（必读）

### ⚠️ 1. Gold 仅用于审计

所有 gold 信息（answer, supporting_facts）**只能在 KG 构建完成后**用于评分。

**不能**用于：
- ❌ 实体链接
- ❌ 关系选择
- ❌ 三元组排序

---

### ⚠️ 2. 版本隔离

修复产物**必须**新命名，不能覆盖原文件：

```bash
# ❌ 错误
output="indexes/kg_cache/question_kg_index_v2.json"

# ✓ 正确
output="indexes/kg_cache/question_kg_index_legacy_v2_hotpot_musique.json"
```

---

### ⚠️ 3. 负结果保留

如果修复无效，**必须如实报告**，不能只发表成功路线。

---

## 常见问题

### Q1: 审计需要多久？

**A**: 约 2 小时（200 题，离线缓存模式）。如果缓存 miss 较多，可能需要 3-4 小时。

---

### Q2: 审计会修改任何数据吗？

**A**: 不会。审计是只读的，只生成报告，不修改任何 KG 索引或数据集。

---

### Q3: 审计结果是否可重复？

**A**: 是。使用固定 seed（默认 46），相同数据和缓存会产生完全相同的结果。

---

### Q4: 如果审计显示"passage_derived_required"占 80%+，怎么办？

**A**: 说明 Legacy Wikidata 路线确实到顶了。应该：
1. 如实报告负结果
2. 转向 passage-derived edges
3. 或改进检索本身（扩展 ProofKG 到其他数据集）

不应该继续优化已经没有信息的 KG。

---

### Q5: 四臂对照需要训练吗？

**A**: 不需要。对照实验是**零训练**的，使用现有 SFT checkpoint，只改变 KG 输入。

只有对照实验通过后，才考虑训练整合。

---

### Q6: 我可以只审计一个数据集吗？

**A**: 可以。修改 `scripts/run_legacy_kg_audit.sh` 中的 `DATASETS` 变量：

```bash
# 只审计 HotpotQA
DATASETS="hotpotqa"

# 只审计 MuSiQue
DATASETS="musique"
```

---

### Q7: 审计的 n_samples 可以调整吗？

**A**: 可以。修改脚本中的 `N_SAMPLES`：

```bash
# 小规模测试（30 题）
N_SAMPLES=30

# 标准（100 题）
N_SAMPLES=100

# 全量（None = 使用整个 dev split）
N_SAMPLES=None
```

---

### Q8: 如何查看某一题的详细诊断？

**A**: 查看 JSON 报告的 `audits` 数组：

```bash
cat reports/legacy_kg_audit/audit_n100_seed46_*.json | \
  python -c "
import json, sys
data = json.load(sys.stdin)
# 查看第一题
audit = data['audits'][0]
print('Question:', audit['question'])
print('Answer:', audit['answer'])
print('Bottleneck:', audit['bottleneck'])
print('Layers:', audit['layers'])
"
```

---

## 下一步

### 立即执行（5 分钟）

1. 运行 preflight check
2. 如果通过，启动审计
3. 泡杯咖啡，等待 2 小时

### 审计完成后（10 分钟）

1. 查看控制台摘要
2. 阅读 Markdown 报告
3. 根据瓶颈分布选择路线

### 修复实现（1-2 周，需要人工批准）

1. 实现对应的修复脚本（legacy_v2 或 passage_derived）
2. 运行四臂对照验证
3. 生成决策报告
4. 根据结果决定是否进入训练

---

## 需要帮助？

### 文档

- **使用指南**: `docs/legacy_kg_audit_guide.md`（详细）
- **执行摘要**: `docs/legacy_kg_repair_executive_summary.md`（概览）
- **交付清单**: `docs/DELIVERY_CHECKLIST.md`（完整系统）

### 配置

- **四臂对照**: `configs/experiments/legacy_kg_repair_comparison.yaml`

### 测试

```bash
# 运行测试验证系统完整性
pytest tests/test_legacy_kg_coverage_audit.py -v
```

---

## 最后检查

在运行审计前，确认：

- ✅ `python scripts/preflight_audit_check.py` 全部通过
- ✅ 了解瓶颈类型和修复策略的对应关系
- ✅ 知道审计结果如何影响下一步路线
- ✅ 理解 Gold 只能用于审计的约束
- ✅ 准备好 2 小时等待审计完成

---

**准备好了？开始吧！**

```bash
bash scripts/run_legacy_kg_audit.sh
```

---

_Good luck! 🚀_
