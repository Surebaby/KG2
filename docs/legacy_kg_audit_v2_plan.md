# Legacy KG 审计框架 - v2 修复计划

**日期**: 2026-09-02  
**状态**: 待实现

---

## 🎯 修复目标

修正 v1 的核心判定指标失效问题，实现科学严谨的审计系统。

---

## 📋 v2 修复清单

### 优先级 P0: 必须修复才能运行 n=100

#### 1. 修复 Answer Normalization (15 分钟)

**问题**: `answer_lower.split()` 保留标点，导致匹配失败

**位置**: `scripts/diagnose/legacy_kg_coverage_audit.py:116`

**修复**:
```python
# 当前
answer_words = set(answer_lower.split())

# 修复后
import re
answer_words = set(re.findall(r'\b\w+\b', answer_lower))
```

**验证**: `pytest tests/test_legacy_kg_coverage_audit.py::test_check_answer_in_triples_partial_match -v`

---

#### 2. 实现答案类型分组 (1 小时)

**新增函数**:
```python
def classify_answer_type(question: str, answer: str) -> str:
    """按答案类型分组，决定使用什么指标"""
    answer_lower = answer.lower().strip()
    question_lower = question.lower()
    
    # Boolean
    if answer_lower in ["yes", "no"]:
        return "boolean"
    
    # Numeric
    if re.match(r'^[\d,\.]+$', answer.replace(' ', '')):
        return "numeric"
    
    # Temporal
    if any(word in question_lower for word in ["when", "year", "date"]):
        if re.match(r'^\d{4}', answer) or "year" in answer_lower:
            return "temporal"
    
    # Comparison
    if any(word in question_lower for word in 
           ["compare", "larger", "smaller", "more", "less", "most", "least"]):
        return "comparison"
    
    # Derived phrase (需要组合/计算)
    if any(word in answer_lower for word in 
           ["about", "approximately", "through", "via"]):
        return "derived_phrase"
    
    # Default: entity/span
    return "entity_span"
```

**使用**:
```python
answer_type = classify_answer_type(question, answer)

# 只有 entity_span 才检查 answer surface
if answer_type == "entity_span":
    check_answer_in_raw = check_answer_in_triples(...)
else:
    check_answer_in_raw = None  # 不适用
```

---

#### 3. 实现 Required-Hop Coverage (核心，3-4 小时)

**新增数据类**:
```python
@dataclass
class GoldSupportDiagnostic:
    """Gold support 覆盖诊断（仅用于审计后评估）"""
    
    # Anchor 提取和链接
    gold_anchors: List[str]  # 从 supporting_facts 提取
    anchors_extracted: List[bool]  # 是否被 mention extractor 提取
    anchors_linked_correct: List[bool]  # 是否链接到正确 QID
    
    # Required hops (从 gold supporting_facts 派生)
    required_hops: List[Dict[str, Any]]  # [{"from": QID, "relation": "?", "to": value}]
    hops_in_raw: List[bool]  # 每一跳是否在 raw triples
    hops_in_top12: List[bool]  # 每一跳是否在 Top-12
    
    # Derivation 完整性
    complete_derivation_in_raw: bool
    complete_derivation_in_top12: bool
    
    # 辅助（仅 entity_span）
    answer_surface_in_raw: Optional[bool]
```

**实现 Gold Support 解析**:
```python
def extract_gold_support_hops(
    question: str,
    answer: str,
    supporting_facts: List[Tuple[str, int]],
    answer_type: str
) -> List[Dict[str, Any]]:
    """
    从 supporting_facts 提取必需的推理跳
    
    注意: Gold 仅在 KG 构建完成后用于审计
    """
    if answer_type == "boolean":
        # 需要两边的对比实体
        # 例如: (A, publisher, X), (B, publisher, X) → "yes"
        return extract_comparison_operands(supporting_facts)
    
    elif answer_type == "numeric":
        # 需要数值和单位
        # 例如: (marathon, distance, 42195m)
        return extract_numeric_operands(supporting_facts)
    
    elif answer_type == "temporal":
        # 需要日期
        # 例如: (A, birth_date, 1500), (B, birth_date, 1900)
        return extract_temporal_operands(supporting_facts)
    
    elif answer_type == "comparison":
        # 需要被比较的实体和属性
        return extract_comparison_entities(supporting_facts)
    
    elif answer_type == "entity_span":
        # 需要答案实体和连接路径
        return extract_entity_path(supporting_facts, answer)
    
    else:  # derived_phrase
        return extract_descriptive_path(supporting_facts)
```

**检查 Hop Coverage**:
```python
def check_hop_coverage(
    required_hops: List[Dict[str, Any]],
    raw_triples: List[Dict[str, Any]],
    top12_triples: List[Dict[str, Any]]
) -> Tuple[List[bool], List[bool]]:
    """检查每一跳是否存在"""
    
    hops_in_raw = []
    hops_in_top12 = []
    
    for hop in required_hops:
        # 模糊匹配（考虑别名、单位转换等）
        in_raw = any(
            fuzzy_match_hop(hop, triple)
            for triple in raw_triples
        )
        in_top12 = any(
            fuzzy_match_hop(hop, triple)
            for triple in top12_triples
        )
        
        hops_in_raw.append(in_raw)
        hops_in_top12.append(in_top12)
    
    return hops_in_raw, hops_in_top12
```

---

#### 4. 修复 QID 正确性检查 (2 小时)

**当前问题**: 只统计 non-abstain，不检查正确性

**修复**:
```python
def check_qid_correctness(
    mentions: List[str],
    linked_qids: List[str],
    gold_anchors: List[str],  # 从 supporting_facts 提取
    gold_qids: List[str]      # Gold QID（如果可用）
) -> Tuple[float, List[bool]]:
    """
    检查 QID 链接正确性（不只是成功率）
    
    注意: Gold 仅用于审计后评估
    """
    correctness = []
    
    for mention, linked_qid in zip(mentions, linked_qids):
        if not linked_qid:  # abstain
            correctness.append(False)
            continue
        
        # 检查是否链接到 gold anchor 的正确 QID
        is_correct = False
        for gold_anchor, gold_qid in zip(gold_anchors, gold_qids):
            if mention_matches_anchor(mention, gold_anchor):
                if linked_qid == gold_qid:
                    is_correct = True
                    break
        
        correctness.append(is_correct)
    
    # 正确率 = 正确链接数 / 应该链接的数量
    linking_correctness = sum(correctness) / max(len(gold_anchors), 1)
    
    return linking_correctness, correctness
```

---

#### 5. 修正瓶颈分类逻辑 (1 小时)

**新分类**:
```python
def classify_bottleneck_v2(
    answer_type: str,
    mentions_extracted: int,
    gold_anchors_extracted: List[bool],
    anchors_linked_correct: List[bool],
    required_hops: List[Dict],
    hops_in_raw: List[bool],
    hops_in_top12: List[bool],
    complete_derivation_in_raw: bool,
    complete_derivation_in_top12: bool
) -> Tuple[str, str]:
    """v2 瓶颈分类"""
    
    # L1: Mention extraction
    if mentions_extracted == 0:
        return "L1_NO_MENTIONS", "improve_mention_extraction"
    
    if not any(gold_anchors_extracted):
        return "L1_GOLD_ANCHOR_NOT_EXTRACTED", "improve_mention_extraction"
    
    # L2: QID linking correctness
    if not any(anchors_linked_correct):
        return "L2_QID_LINKING_INCORRECT", "improve_entity_linker"
    
    if sum(anchors_linked_correct) / len(gold_anchors_extracted) < 0.5:
        return "L2_LOW_LINKING_CORRECTNESS", "improve_entity_linker"
    
    # L3: Knowledge coverage
    if not required_hops:
        return "L3_CACHE_EMPTY", "expand_cache_or_check_retrieval"
    
    missing_relations = [
        hop for hop, in_raw in zip(required_hops, hops_in_raw)
        if not in_raw
    ]
    
    if len(missing_relations) > len(required_hops) * 0.5:
        return "L3_REQUIRED_RELATIONS_MISSING", "passage_derived_or_expand_wikidata"
    
    if not complete_derivation_in_raw:
        return "L3_INCOMPLETE_DERIVATION_PATH", "check_hop_connectivity"
    
    # L4: Filtering
    if complete_derivation_in_raw and not complete_derivation_in_top12:
        return "L4_USEFUL_EDGES_FILTERED", "fix_reranker"  # ← 这才是 rerank 可修复
    
    # L5, L6 (待实现)
    return "L5_OR_L6_OR_AVAILABLE", "investigate_downstream"
```

**关键差异**:
- ✅ 检查 gold anchor 提取（不只是任意 mention）
- ✅ 检查 QID 正确性（不只是 non-abstain）
- ✅ 检查 required hops（不只是 answer surface）
- ✅ 只有 derivation 在 raw 但不在 top12 时，才说 "filtering 问题"

---

### 优先级 P1: 提升准确性

#### 6. 实现 Cache 来源诊断 (1 小时)

```python
@dataclass
class CacheSourceDiagnostic:
    """区分不同层级的缺失"""
    in_local_2hop_cache: bool
    in_historical_requests: bool  # 如果有历史查询日志
    attempted_online_query: bool  # 是否尝试在线查询
    in_wikidata_schema: bool      # 关系是否存在于 Wikidata schema
    
    # 只有都是 False 才能说 "Wikidata 确实没有"
```

**使用**:
```python
if not hop_in_raw:
    cache_diag = diagnose_cache_source(hop, kg_retriever)
    
    if not cache_diag.in_local_2hop_cache:
        if cache_diag.in_historical_requests:
            reason = "local_cache_miss_but_historically_available"
        else:
            reason = "not_in_2hop_cache_unknown_if_in_wikidata"
    
    # 不要轻易说 "Wikidata 没有"
```

---

#### 7. 实现 L5: Prompt Injection Check (30 分钟)

```python
def check_prompt_injection(
    top12_triples: List[Dict],
    final_prompt: str,
    max_context_length: int = 6144
) -> Tuple[bool, List[bool]]:
    """检查 Top-12 是否完整进入 prompt"""
    
    # 检查 prompt 长度
    prompt_truncated = len(final_prompt) > max_context_length
    
    # 检查每条三元组是否在 prompt 中
    triples_in_prompt = []
    for triple in top12_triples:
        triple_text = format_triple(triple)
        in_prompt = triple_text in final_prompt
        triples_in_prompt.append(in_prompt)
    
    return prompt_truncated, triples_in_prompt
```

---

#### 8. 实现 L6: Model Utilization Check (需要推理，2 小时)

```python
def check_model_utilization(
    question: str,
    top12_triples: List[Dict],
    model_output: str,
    kg_citations: List[str]
) -> Dict[str, Any]:
    """检查模型是否使用了可用的 KG"""
    
    # 需要实际推理才能获得
    # 1. model_output (生成的推理过程)
    # 2. kg_citations (引用的三元组)
    
    available_useful_triples = [
        t for t in top12_triples
        if is_relevant_to_question(t, question)
    ]
    
    cited_useful_triples = [
        t for t in available_useful_triples
        if any(cite_matches_triple(c, t) for c in kg_citations)
    ]
    
    utilization_rate = len(cited_useful_triples) / max(len(available_useful_triples), 1)
    
    return {
        "available_useful": len(available_useful_triples),
        "cited_useful": len(cited_useful_triples),
        "utilization_rate": utilization_rate
    }
```

**注意**: L6 需要实际运行推理，成本较高。可以：
- 选项 A: 在小样本（n=20）上运行
- 选项 B: 标记为 "需要推理，暂未实现"
- 选项 C: 改名为 "四层审计"（L1-L4）

---

### 优先级 P2: 完善性

#### 9. 更新测试套件 (1 小时)

```python
# 新增测试
def test_classify_answer_type():
    assert classify_answer_type("Is X true?", "yes") == "boolean"
    assert classify_answer_type("How many?", "42") == "numeric"
    assert classify_answer_type("When was X born?", "1900") == "temporal"
    # ...

def test_check_hop_coverage():
    required_hops = [{"from": "Q123", "relation": "P31", "to": "Q456"}]
    raw_triples = [{"head": "Q123", "relation": "P31", "tail": "Q456"}]
    hops_in_raw, _ = check_hop_coverage(required_hops, raw_triples, [])
    assert hops_in_raw == [True]

def test_classify_bottleneck_v2():
    # 测试所有新的瓶颈类型
    pass
```

---

#### 10. 更新文档 (1 小时)

需要更新：
- `docs/legacy_kg_audit_guide.md` - 新的层级定义和指标
- `docs/QUICKSTART.md` - v2 使用方法
- `configs/experiments/legacy_kg_repair_comparison.yaml` - 新的判定标准

---

## ⏱️ 时间估算

| 任务 | 优先级 | 时间 |
|------|--------|------|
| 1. Answer normalization | P0 | 15 分钟 |
| 2. 答案类型分组 | P0 | 1 小时 |
| 3. Required-hop coverage | P0 | 3-4 小时 |
| 4. QID 正确性 | P0 | 2 小时 |
| 5. 瓶颈分类 v2 | P0 | 1 小时 |
| 6. Cache 来源诊断 | P1 | 1 小时 |
| 7. L5 Prompt injection | P1 | 30 分钟 |
| 8. L6 Model utilization | P1 | 2 小时（或标记待实现）|
| 9. 测试套件 | P2 | 1 小时 |
| 10. 文档更新 | P2 | 1 小时 |

**P0 总计**: 7-8 小时  
**P1 总计**: 3.5 小时  
**P2 总计**: 2 小时  

**完整 v2**: 12-13.5 小时  
**最小可行 v2** (P0 only): 7-8 小时

---

## 🎯 v2 验收标准

### 代码质量
- ✅ 所有测试通过（不能有 failed）
- ✅ Answer normalization 正确处理标点
- ✅ 答案类型正确分类
- ✅ Required-hop 覆盖率计算正确

### 科学严谨性
- ✅ 不用 answer surface 作为 boolean/numeric 的主要指标
- ✅ 检查 QID 正确性而非链接成功率
- ✅ 区分 cache miss 和 Wikidata 缺失
- ✅ 只有 derivation 在 raw 不在 top12 时才说 filtering 问题

### 可追溯性
- ✅ v2 协议冻结并版本化
- ✅ 排除 v1 使用的 20 题
- ✅ n=100 使用新的未见题目
- ✅ Gold 仅用于审计后评估

---

## 🚀 执行顺序

### 第一步：实现 P0（最小可行 v2）
```bash
# 1. 修复 answer normalization
# 2. 实现答案类型分组
# 3. 实现 required-hop coverage
# 4. 修复 QID 正确性
# 5. 更新瓶颈分类

# 验证
pytest tests/test_legacy_kg_coverage_audit.py -v
# 预期：所有测试通过

# 小样本测试（n=10，不包含 v1 的题目）
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --datasets hotpotqa musique \
  --n_samples 10 \
  --seed 47 \
  --version v2 \
  --exclude_qids <v1 的 20 个 qid> \
  --output reports/legacy_kg_audit/audit_v2_n10_seed47_test.json
```

### 第二步：小样本验证（n=10）
- 检查新指标是否合理
- 确认瓶颈分类不再全部堆积在一类
- 验证 boolean/numeric 不再误判

### 第三步：冻结 v2 协议
```yaml
version: v2
frozen_date: 2026-09-XX
changes_from_v1:
  - answer_type_grouping
  - required_hop_coverage
  - qid_correctness_check
  - revised_bottleneck_classification
excluded_development_qids: [<v1 的 20 个>]
```

### 第四步：运行 n=100 confirmation
```bash
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --datasets hotpotqa musique \
  --n_samples 100 \
  --seed 48 \
  --version v2 \
  --exclude_qids <v1 的 20 个> \
  --output reports/legacy_kg_audit/audit_v2_n100_seed48_confirmation.json
```

### 第五步：根据 v2 结果决策
只有 v2 显示以下情况，才能做相应决策：

**如果**: `L3_REQUIRED_RELATIONS_MISSING > 60%`  
**则**: 考虑 passage-derived（但仍需检查是否 cache miss）

**如果**: `L4_USEFUL_EDGES_FILTERED > 60%`  
**则**: 开发 Legacy KG v2（修复 reranker）

**如果**: `L2_QID_LINKING_INCORRECT > 60%`  
**则**: 改进 entity linker

---

## 📝 文档更新计划

### 需要创建
- ✅ `V1_CORRECTION_REPORT.md` (已完成)
- ⏳ `V2_IMPLEMENTATION_PLAN.md` (本文件)
- ⏳ `V2_PROTOCOL.md` (v2 协议冻结文档)

### 需要更新
- ⏳ `docs/legacy_kg_audit_guide.md` - v2 指标说明
- ⏳ `docs/QUICKSTART.md` - v2 使用方法
- ⏳ README 文件 - 标注 v1 无效，v2 待实现

---

## 总结

**v1 教训**: 代理指标失效，科学结论无效

**v2 改进**: 
- 按答案类型分组
- 检查 required-hop coverage
- 区分 cache miss 和知识缺失
- 检查 QID 正确性

**时间成本**: 7-13.5 小时（最小 7 小时）

**预期**: v2 能给出科学严谨的瓶颈分布

---

_v2 计划生成时间：2026-09-02_  
_预计开始时间：待批准_
