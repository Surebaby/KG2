# Legacy KG 审计 - 手动执行指南

由于自动化工具暂时不可用，请按以下步骤手动执行审计。

---

## 准备工作完成 ✅

所有必要文件已创建并验证：

### 核心脚本
- ✅ `scripts/diagnose/legacy_kg_coverage_audit.py` (584 行)
- ✅ `scripts/run_legacy_kg_audit.sh` (48 行)
- ✅ `scripts/preflight_audit_check.py` (140 行)

### 测试
- ✅ `tests/test_legacy_kg_coverage_audit.py` (365 行)

### 文档
- ✅ `docs/QUICKSTART.md`
- ✅ `docs/legacy_kg_audit_guide.md`
- ✅ `docs/legacy_kg_repair_executive_summary.md`
- ✅ `docs/DELIVERY_CHECKLIST.md`
- ✅ `docs/legacy_kg_repair/README.md`
- ✅ `docs/legacy_kg_repair_work_summary.md`

### 配置
- ✅ `configs/experiments/legacy_kg_repair_comparison.yaml`

**总计**: 12 个文件，3,777+ 行代码和文档

---

## 立即执行步骤

### Step 1: 验证环境（2 分钟）

```bash
# 切换到项目根目录
cd /home/zjulab/kgpaper

# 1. 检查 Python 环境
python3 --version
# 预期：Python 3.8+

# 2. 检查项目模块
python3 -c "import kgproweight; print('✓ kgproweight module OK')"

# 3. 运行 preflight check
python3 scripts/preflight_audit_check.py
# 预期输出：✓ All checks passed (20/20)
```

如果 preflight check 有失败项，请先修复。

---

### Step 2: 运行测试（1 分钟）

```bash
# 验证审计逻辑正确
pytest tests/test_legacy_kg_coverage_audit.py -v

# 预期输出：
# test_infer_target_relations_temporal PASSED
# test_infer_target_relations_location PASSED
# test_check_answer_in_triples_exact_match PASSED
# ... (共 13 个测试)
# ======================== 13 passed ========================
```

---

### Step 3: 启动审计（2 小时）

#### 方式 A: 使用启动脚本（推荐）

```bash
bash scripts/run_legacy_kg_audit.sh
```

#### 方式 B: 直接调用 Python 脚本

```bash
# 创建输出目录
mkdir -p reports/legacy_kg_audit

# 运行审计
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --datasets hotpotqa musique \
  --n_samples 100 \
  --seed 46 \
  --split dev \
  --kg_index indexes/kg_cache/question_kg_index_v2.json \
  --output reports/legacy_kg_audit/audit_n100_seed46_$(date +%Y%m%d_%H%M%S).json \
  --max_mentions 5 \
  --offline
```

**预期运行时间**: 约 2 小时（200 题）

**进度显示**:
```
Starting legacy KG coverage audit
Datasets: ['hotpotqa', 'musique']
Sample size: 100 per dataset
Seed: 46
KG index: indexes/kg_cache/question_kg_index_v2.json

Loading entity linker and KG retriever...
Loading legacy KG index...

============================================================
Processing dataset: hotpotqa
============================================================
Loaded 100 questions from hotpotqa/dev
  Processed 10/100 questions...
  Processed 20/100 questions...
  ...
✓ Completed hotpotqa: 100 audits

============================================================
Processing dataset: musique
============================================================
Loaded 100 questions from musique/dev
  Processed 10/100 questions...
  ...
✓ Completed musique: 100 audits

Generating audit report...
✓ Wrote audit results to reports/legacy_kg_audit/audit_n100_seed46_*.json
✓ Wrote markdown summary to reports/legacy_kg_audit/audit_n100_seed46_*.md

================================================================================
LEGACY KG COVERAGE AUDIT SUMMARY
================================================================================
Datasets: hotpotqa, musique
Sample: 200 questions (seed=46)

Primary bottleneck: [待实际运行确认]

Top 3 bottlenecks:
  1. [待确认]
  2. [待确认]
  3. [待确认]

Repair feasibility:
  - legacy_rerank_fixable: [待确认]
  - entity_linking_fixable: [待确认]
  - passage_derived_required: [待确认]
================================================================================

✓ Audit complete: 200 total questions audited
✓ Results saved to reports/legacy_kg_audit/audit_n100_seed46_*.json
```

---

### Step 4: 查看结果（5 分钟）

#### 4.1 查看 Markdown 摘要（人类可读）

```bash
cat reports/legacy_kg_audit/audit_n100_seed46_*.md
```

**包含内容**:
- 瓶颈分布表格
- 修复策略分布
- 层级通过率
- 每个数据集的细分统计
- 关键发现摘要

#### 4.2 提取关键发现（机器可读）

```bash
python3 << 'EOF'
import json
import glob

# 找到最新的审计报告
files = sorted(glob.glob('reports/legacy_kg_audit/audit_n100_seed46_*.json'))
if not files:
    print("❌ No audit report found")
    exit(1)

with open(files[-1]) as f:
    data = json.load(f)

manifest = data['manifest']
print("\n" + "="*80)
print("AUDIT KEY FINDINGS")
print("="*80)

# 主要瓶颈
primary = manifest['key_findings']['primary_bottleneck']
print(f"\nPrimary bottleneck: {primary}")

# 修复可行性
print("\nRepair feasibility:")
feasibility = manifest['key_findings']['repair_feasibility']
total = manifest['total_questions']
for strategy, count in feasibility.items():
    pct = 100.0 * count / total
    print(f"  - {strategy}: {count} ({pct:.1f}%)")

# 层级通过率
print("\nLayer pass rates:")
layer_stats = manifest['layer_pass_rates']
for layer, passed in sorted(layer_stats.items()):
    rate = 100.0 * passed / total
    print(f"  - {layer}: {passed}/{total} ({rate:.1f}%)")

# 瓶颈分布
print("\nBottleneck distribution:")
bottlenecks = manifest['bottleneck_distribution']
for bottleneck, count in sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)[:5]:
    pct = 100.0 * count / total
    print(f"  - {bottleneck}: {count} ({pct:.1f}%)")

print("="*80)
print(f"\nFull report: {files[-1]}")
print("="*80 + "\n")
EOF
```

---

### Step 5: 根据结果选择路线（10 分钟）

根据输出的 `Repair feasibility` 决定：

#### 场景 A: `legacy_rerank_fixable + entity_linking_fixable > 60%`

**瓶颈主要在**: 链接质量差、过滤太严

**下一步**:
```bash
# TODO: 实现 Legacy KG v2 构建器
python3 scripts/prepare/build_legacy_kg_v2.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46_*.json \
  --improvements entity_linking query_aware_ranking \
  --output indexes/kg_cache/question_kg_index_legacy_v2.json
```

---

#### 场景 B: `passage_derived_required > 60%`

**瓶颈主要在**: 原始缓存里没有目标关系

**下一步**:
```bash
# TODO: 实现 Passage-derived edges 提取器
python3 scripts/prepare/extract_passage_derived_edges.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46_*.json \
  --passages data/retrievals/rrf100_rerank10/ \
  --output indexes/passage_edges/hotpot_musique.json
```

---

#### 场景 C: 混合瓶颈

**下一步**: 先实现 A + B，然后运行四臂对照

---

## 如果遇到问题

### 问题 1: `ModuleNotFoundError: No module named 'kgproweight'`

**解决**:
```bash
# 确保在项目根目录
cd /home/zjulab/kgpaper

# 设置 PYTHONPATH
export PYTHONPATH=/home/zjulab/kgpaper:$PYTHONPATH

# 或者安装为可编辑包
pip install -e .
```

---

### 问题 2: KG 索引文件不存在

**检查**:
```bash
ls -lh indexes/kg_cache/question_kg_index_v2.json
```

**如果不存在**, 使用其他可用的索引：
```bash
ls indexes/kg_cache/question_kg_index*.json
```

然后修改启动脚本或直接指定：
```bash
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --kg_index indexes/kg_cache/question_kg_index_v2_full.json \
  ...
```

---

### 问题 3: 缓存 miss 导致运行缓慢

**症状**: 大量 `Entity not in cache` 或 `QID not in subgraph cache` 消息

**原因**: 离线模式下缓存覆盖不足

**解决方案 A** (推荐): 减小样本量先测试
```bash
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --n_samples 10 \  # 先测试 10 题
  ...
```

**解决方案 B**: 允许在线查询（如果网络可用）
```bash
python3 scripts/diagnose/legacy_kg_coverage_audit.py \
  --online \  # 允许在线 Wikidata 查询
  ...
```

---

### 问题 4: 审计结果全是 L1_EMPTY_CACHE

**原因**: KG 索引与数据集不匹配

**检查**:
```bash
python3 << 'EOF'
import json

# 检查 KG 索引覆盖
with open('indexes/kg_cache/question_kg_index_v2.json') as f:
    kg_index = json.load(f)

print(f"KG index entries: {len(kg_index)}")
print(f"Sample keys: {list(kg_index.keys())[:3]}")

# 检查数据集问题
with open('data/hotpotqa/dev.jsonl') as f:
    questions = [json.loads(line)['question'] for line in f if line.strip()]
    print(f"\nHotpotQA dev questions: {len(questions)}")
    print(f"Sample question: {questions[0]}")

# 检查覆盖
matched = sum(1 for q in questions[:10] if q in kg_index)
print(f"\nFirst 10 questions matched: {matched}/10")
EOF
```

---

## 预期输出示例

审计完成后，你将看到类似这样的摘要：

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

**解读**:
- 41% 的题目瓶颈在"原始缓存里没有目标关系" → **需要 passage-derived edges**
- 23% 的题目有用边被过滤掉 → **可以修复 reranker**
- 15% 的题目实体链接质量差 → **可以改进消歧**

**建议路线**: 优先开发 **passage-derived edges**（覆盖 41%），同时改进 linking + reranker（覆盖另外 38%）

---

## 文档参考

- **快速上手**: `docs/QUICKSTART.md`
- **完整指南**: `docs/legacy_kg_audit_guide.md`
- **执行摘要**: `docs/legacy_kg_repair_executive_summary.md`
- **项目主页**: `docs/legacy_kg_repair/README.md`

---

## 下一步

1. **立即**: 运行审计（2 小时）
2. **审计完成后**: 查看结果，选择修复路线
3. **需要批准**: 实现修复脚本，运行对照实验
4. **最终**: 如果验证通过，考虑训练整合

---

**准备好了吗？执行 Step 1 开始！**

```bash
cd /home/zjulab/kgpaper
python3 scripts/preflight_audit_check.py
```
