# Legacy KG Coverage Audit

## 目的

对 HotpotQA 和 MuSiQue 的 legacy KG 进行分层诊断，定位信息损失发生在哪一层，从而决定修复策略。

## 核心问题

当前负结果只证明：

> 现有 legacy KG 构建结果对 HotpotQA/MuSiQue 没有增量效用。

但它**没有**证明原始候选池里完全没有有用信息。问题可能发生在：

```
原始知识源
  → 实体/QID链接         ← L2: QID linking
  → 关系/PID映射         ← L3: Relation coverage
  → 候选三元组获取       ← L1: Raw candidates
  → 过滤与Top-K          ← L4: Top-K filtering
  → prompt注入           ← L5: Prompt injection
  → 模型使用             ← L6: Model utilization
```

只要有用事实存在于原始候选池、但被错误链接或过滤掉，就可以在**不增加外部知识源**的情况下修复。

## 审计层级

### Layer 1: Raw Knowledge Source
**检查**：原始 KG 缓存中是否存在支持答案的实体、relation、value？

**诊断指标**：
- `raw_has_answer_mentions`: 答案实体是否出现在原始三元组中
- `raw_answer_entity_count`: 答案实体在原始三元组中的出现次数
- `raw_total_entities`: 成功链接的实体总数
- `raw_total_triples`: 原始缓存的三元组总数

**瓶颈**：
- `L1_NO_MENTIONS`: 问题中没有提取到实体提及
- `L1_EMPTY_CACHE`: 链接的实体在缓存中没有三元组

**修复策略**：
- `improve_mention_extraction`: 改进实体提及抽取
- `expand_kg_cache_coverage`: 扩展 KG 缓存覆盖

---

### Layer 2: QID Linking
**检查**：实体是否正确链接到 Wikidata QID？

**诊断指标**：
- `mentions_extracted`: 从问题中提取的实体提及数
- `mentions_linked`: 成功链接到 QID 的提及数
- `mentions_abstained`: 链接器拒绝链接的提及数
- `qid_linking_quality`: 链接成功率 (linked / extracted)

**瓶颈**：
- `L2_LINKING_FAILURE`: 所有实体提及链接失败
- `L2_LOW_LINKING_QUALITY`: 链接成功率 < 40%

**修复策略**：
- `fix_entity_linker`: 修复实体链接器
- `improve_entity_disambiguation`: 改进实体消歧（使用 passage titles 作为上下文）

---

### Layer 3: Relation/Value Coverage
**检查**：原始三元组中是否包含目标关系和值？

**诊断指标**：
- `raw_has_target_relations`: 是否包含推断的目标关系类型
- `target_relation_types`: 基于问题推断的目标关系（如 "temporal", "location"）
- `raw_matched_relation_count`: 匹配的目标关系数量

**瓶颈**：
- `L3_ANSWER_NOT_IN_RAW`: 答案实体不在原始缓存中
- `L3_RELATION_MISSING`: 目标关系不在原始缓存中

**修复策略**：
- `passage_derived_required`: Legacy Wikidata 路线到顶，需要从 passages 提取结构化证据
- `passage_derived_or_expand_cache`: 可尝试扩展缓存，或转向 passage-derived

---

### Layer 4: Top-K Filtering
**检查**：有用的边是否在 Top-12 过滤后幸存？

**诊断指标**：
- `top12_triple_count`: 过滤后保留的三元组数量
- `top12_has_useful_edges`: Top-12 中是否包含有用的边
- `top12_answer_mention_count`: Top-12 中答案实体的提及次数

**瓶颈**：
- `L4_COMPLETE_FILTERING_LOSS`: 所有三元组被过滤掉
- `L4_FILTERING_REMOVED_USEFUL`: 有用的边在原始数据中存在但被过滤移除

**修复策略**：
- `fix_filter_threshold`: 调整过滤阈值
- `fix_reranker`: 修复重排序逻辑（query-aware ranking, precision-first）

---

### Layer 5+6: Downstream
**检查**：KG 在 prompt 中可用，但模型未使用？

**瓶颈**：
- `L5_DOWNSTREAM`: KG 在 Top-12 中可用，瓶颈可能在 prompt 构造或模型行为

**修复策略**：
- `investigate_prompt_or_model`: 需要运行实际推理来诊断

---

## 使用方法

### 1. 运行审计

```bash
python scripts/diagnose/legacy_kg_coverage_audit.py \
  --datasets hotpotqa musique \
  --n_samples 100 \
  --seed 46 \
  --split dev \
  --kg_index indexes/kg_cache/question_kg_index_v2.json \
  --output reports/legacy_kg_bottleneck_audit_n100_seed46.json
```

**参数说明**：
- `--datasets`: 要审计的数据集（支持 hotpotqa, 2wikimultihopqa, musique）
- `--n_samples`: 每个数据集采样多少题（None = 全部）
- `--seed`: 随机种子（确保可重复）
- `--split`: 数据集分片（dev/test，**不要用 train 避免泄漏**）
- `--kg_index`: Legacy KG 索引路径
- `--output`: 输出 JSON 报告路径
- `--max_mentions`: 每个问题最多提取多少个实体提及（默认 5）
- `--offline`: 只使用缓存，不调用在线 Wikidata（默认）

### 2. 查看报告

审计完成后会生成两个文件：

**JSON 报告** (`*.json`)：
```json
{
  "manifest": {
    "experiment_id": "LEGACY_KG_AUDIT_HOTPOTQA_MUSIQUE_DEV_N100_SEED46",
    "bottleneck_distribution": {
      "L3_RELATION_MISSING": 41,
      "L4_FILTERING_REMOVED_USEFUL": 23,
      "L2_LOW_LINKING_QUALITY": 15,
      ...
    },
    "repair_strategy_distribution": {
      "passage_derived_required": 41,
      "fix_reranker": 23,
      ...
    },
    "key_findings": {
      "primary_bottleneck": "L3_RELATION_MISSING",
      "repair_feasibility": {
        "legacy_rerank_fixable": 35,
        "entity_linking_fixable": 20,
        "passage_derived_required": 45
      }
    }
  },
  "audits": [
    {
      "dataset": "hotpotqa",
      "qid": "dev_0",
      "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
      "answer": "yes",
      "layers": {
        "mentions_extracted": 2,
        "mentions_linked": 2,
        "raw_total_triples": 35,
        "raw_has_answer_mentions": true,
        "raw_has_target_relations": false,
        "top12_triple_count": 12,
        "top12_has_useful_edges": false
      },
      "bottleneck": "L3_RELATION_MISSING",
      "repair_potential": "passage_derived_required"
    },
    ...
  ]
}
```

**Markdown 摘要** (`*.md`)：
- 瓶颈分布表格
- 修复策略分布
- 各层通过率
- 每个数据集的细分统计

### 3. 解读结果并决定修复方向

根据 `key_findings.repair_feasibility`：

#### 场景 A：瓶颈主要在 Filtering/Linking (>60%)

```
legacy_rerank_fixable: 50+ questions
entity_linking_fixable: 20+ questions
passage_derived_required: <30 questions
```

**结论**：原始候选池里有信息，但被错误过滤或链接错误。

**下一步**：
1. 构建 **Legacy KG v2**：
   - 改进实体链接（使用 passage titles 消歧）
   - Query-aware relation ranking
   - Precision-first 去噪（移除 occupation/instance-of 等泛化边）
2. 运行四臂对照实验（见下文）

#### 场景 B：瓶颈主要在 Relation Missing (>60%)

```
passage_derived_required: 60+ questions
legacy_rerank_fixable: <20 questions
```

**结论**：原始 Wikidata 缓存里根本没有目标 relation/value。

**下一步**：
1. **不要**继续优化 legacy KG 排序
2. 直接开发 **Passage-derived edges**：
   - 从同一 top-10 passages 提取结构化关系
   - 标记 `source=passage`
   - 与 legacy KG 融合（如果有）

#### 场景 C：混合瓶颈

```
passage_derived_required: 40 questions
legacy_rerank_fixable: 30 questions
entity_linking_fixable: 30 questions
```

**结论**：需要 **Hybrid Evidence Graph**。

**下一步**：
1. 构建 Legacy KG v2（修复链接和排序）
2. 添加 Passage-derived edges
3. 统一融合：Wikidata edges + passage edges
4. Precision-first 去重，最多 12 条

---

## 四臂对照实验

审计完成后，**必须**运行零训练的对照实验验证修复有效：

```yaml
arms:
  A_legacy_original:
    # 当前 legacy KG，不做任何修改
    kg_source: "indexes/kg_cache/question_kg_index_v2.json"

  B_legacy_v2_repaired:
    # 只修链接/关系/排序，相同 Wikidata 缓存
    kg_source: "indexes/kg_cache/question_kg_index_legacy_v2.json"

  C_hybrid_evidence:
    # Legacy v2 + passage-derived edges
    kg_source: "combined"

  D_nokg:
    # 无 KG baseline
    kg_source: null

fixed_across_arms:
  - model: "models/sft_legacy_repaired_v2_quota70/final"
  - passages: "RRF@100 + rerank@10"  # 相同检索结果
  - decoding: "temperature=0"
```

**评估指标**：
- Overall: EM / F1 / parse_rate / citation_rate
- Per-bottleneck-type: 按审计发现的瓶颈类型分组统计

**判定门**：
- B.EM >= A.EM - 0.02: v2 不能显著退化
- C.EM > D.EM + 0.05 AND C.citation_rate > 0.7: hybrid 有正向效用

---

## 关键约束

### 1. Gold 仅用于审计

所有 gold 信息（supporting_facts, answer, decomposition）**只能在 KG 构建完成后**用于评分，**不能**参与：
- 实体链接
- 关系选择
- 三元组排序

### 2. 版本隔离

修复后的资产**必须**命名为 `*_v2` 或 `*_hybrid`，不能覆盖旧 legacy KG：

```bash
# ❌ 错误：覆盖原文件
output="indexes/kg_cache/question_kg_index_v2.json"

# ✓ 正确：新版本命名
output="indexes/kg_cache/question_kg_index_legacy_v2_hotpot_musique.json"
```

### 3. 可追溯性

每个产物必须记录：
- `experiment_id`: 唯一实验标识
- `audit_version`: 审计脚本版本
- `builder_version`: KG 构建器版本
- `data_hashes`: 数据文件的 MD5/SHA256
- `code_commit`: Git commit hash

### 4. 失败结果保留

如果 Legacy v2 没有改善，**必须如实报告**，不能只发表成功路线。

---

## 预期结果

### HotpotQA
- **仅修 legacy 排序**：可能只能从负增益修到持平或小幅正增益
- **真正提升**：大概率来自 passage-derived graph

### MuSiQue
- **Canonicalizer**：尚有明确修复空间，可能先让覆盖率和效用转正
- **Subquery dependency graph**：比强制 relation graph 更适合

### 2WikiMultiHopQA
- **继续保留 ProofKG**：大幅增益已验证，不需要改动

---

## 输出示例

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

**解读**：
- 41% 的题目瓶颈在"原始缓存里没有目标关系" → passage-derived 是主要方向
- 23% 的题目有用边被过滤掉 → 修复 reranker 有收益空间
- 15% 的题目实体链接质量差 → 改进消歧有帮助

**建议下一步**：
1. 优先开发 passage-derived edges（覆盖 41%）
2. 同时改进 entity linking + reranker（覆盖另外 38%）
3. 构建 Hybrid Evidence Graph v1 统一两者

---

## 参考

- `RESEARCH_WORKFLOW.md` §8: 当前遇到的问题
- `todo.md` §1: 当前结论摘要
- `AGENTS.md` §3: 数据处理规范
- `retraining_plan.md`: 重训整改方案
