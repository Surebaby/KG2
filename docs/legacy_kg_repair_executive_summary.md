# Legacy KG 修复方案：执行摘要

**日期**: 2026-09-02  
**目标**: 在不增加外部知识源的情况下，提升 HotpotQA 和 MuSiQue 的 Legacy KG 质量

---

## 核心洞察

当前负结果证明：
> 现有 legacy KG 构建结果对 HotpotQA/MuSiQue 没有增量效用。

但**没有**证明原始候选池里完全没有有用信息。

**关键区分**：
1. **保持 Legacy KG 文件不变**，只让模型更会使用 → 已验证无效
2. **使用相同原始资源**，修复实体链接、关系选择和排序 → 可能有效

如果有用事实存在于原始候选池但被错误链接或过滤，就可以修复。  
如果原始池里根本没有目标 relation/value，则需要 passage-derived edges。

---

## 已完成的工作

### 1. 分层覆盖审计脚本

**文件**: `scripts/diagnose/legacy_kg_coverage_audit.py`

**功能**: 诊断信息损失发生在哪一层：

```
Layer 1: Raw Knowledge Source    → 原始缓存是否有答案实体？
Layer 2: QID Linking             → 实体链接是否正确？
Layer 3: Relation Coverage       → 是否有目标关系？
Layer 4: Top-K Filtering         → 有用边是否被过滤掉？
Layer 5: Prompt Injection        → KG 是否进入 prompt？
Layer 6: Model Utilization       → 模型是否使用 KG？
```

**输出**:
- JSON 报告：包含每题的逐层诊断
- Markdown 摘要：瓶颈分布、修复策略、层级通过率
- 控制台输出：关键发现和修复可行性统计

**用法**:
```bash
bash scripts/run_legacy_kg_audit.sh
```

或直接：
```bash
python scripts/diagnose/legacy_kg_coverage_audit.py \
  --datasets hotpotqa musique \
  --n_samples 100 \
  --seed 46 \
  --split dev \
  --kg_index indexes/kg_cache/question_kg_index_v2.json \
  --output reports/legacy_kg_audit/audit_n100_seed46.json
```

### 2. 测试套件

**文件**: `tests/test_legacy_kg_coverage_audit.py`

**覆盖**:
- 目标关系推断（temporal, location, identity）
- 答案在三元组中的检测
- 关系覆盖检查
- 瓶颈分类逻辑
- 数据类完整性

**运行**:
```bash
pytest tests/test_legacy_kg_coverage_audit.py -v
```

### 3. 使用指南

**文件**: `docs/legacy_kg_audit_guide.md`

**内容**:
- 审计层级详细说明
- 瓶颈类型和修复策略
- 结果解读和决策树
- 四臂对照实验设计
- 关键约束和边界条件

### 4. 四臂对照实验配置

**文件**: `configs/experiments/legacy_kg_repair_comparison.yaml`

**四臂**:
- **A**: Legacy Original（当前 baseline）
- **B**: Legacy v2 Repaired（修复链接/排序）
- **C**: Hybrid Evidence（v2 + passage edges）
- **D**: NoKG（对照基线）

**控制变量**:
- 相同模型 checkpoint
- 相同检索结果（passages）
- 相同解码参数

**预注册门**:
- B.EM ≥ A.EM - 0.02（v2 不显著退化）
- C.EM > D.EM + 0.05 AND citation_rate > 0.7（hybrid 有效用）
- citation_precision > 0.9（不幻觉引用）

---

## 执行流程

### 第一步：运行审计（必须）

```bash
# 1. 运行分层审计
bash scripts/run_legacy_kg_audit.sh

# 2. 查看结果
cat reports/legacy_kg_audit/audit_n100_seed46_*.md

# 3. 提取关键发现
python -c "
import json
with open('reports/legacy_kg_audit/audit_n100_seed46_*.json') as f:
    data = json.load(f)
    print('Primary bottleneck:', data['manifest']['key_findings']['primary_bottleneck'])
    print('Repair feasibility:')
    for k, v in data['manifest']['key_findings']['repair_feasibility'].items():
        print(f'  {k}: {v}')
"
```

### 第二步：根据审计结果选择路线

#### 路线 A：瓶颈主要在 Filtering/Linking

**条件**: `legacy_rerank_fixable + entity_linking_fixable > 60%`

**行动**:
```bash
# 1. 构建 Legacy KG v2
python scripts/prepare/build_legacy_kg_v2.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46.json \
  --improvements entity_linking query_aware_ranking \
  --output indexes/kg_cache/question_kg_index_legacy_v2_hotpot_musique.json

# 2. 运行对照实验（Arm A vs B）
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A B D \
  --checkpoint models/sft_legacy_repaired_v2_quota70/final
```

#### 路线 B：瓶颈主要在 Relation Missing

**条件**: `passage_derived_required > 60%`

**行动**:
```bash
# 1. 提取 passage-derived edges
python scripts/prepare/extract_passage_derived_edges.py \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46.json \
  --passages data/retrievals/rrf100_rerank10/ \
  --output indexes/passage_edges/hotpot_musique_n100_seed46.json

# 2. 运行对照实验（Arm A vs C vs D）
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A C D \
  --checkpoint models/sft_legacy_repaired_v2_quota70/final
```

#### 路线 C：混合瓶颈

**条件**: 瓶颈分散

**行动**:
```bash
# 1. 构建 Legacy v2
python scripts/prepare/build_legacy_kg_v2.py ...

# 2. 提取 passage edges
python scripts/prepare/extract_passage_derived_edges.py ...

# 3. 运行完整四臂对照
python scripts/evaluation/run_kg_comparison.py \
  --config configs/experiments/legacy_kg_repair_comparison.yaml \
  --arms A B C D \
  --checkpoint models/sft_legacy_repaired_v2_quota70/final
```

### 第三步：分析结果并做决策

```bash
# 生成决策报告
python scripts/analysis/kg_repair_decision.py \
  --comparison_results outputs/legacy_kg_repair_comparison_n100_seed46/ \
  --audit_report reports/legacy_kg_audit/audit_n100_seed46.json \
  --output reports/legacy_kg_repair_decision.md
```

**决策树**:

```
审计结果
├─ 瓶颈主要在 Filtering/Linking (>60%)
│  ├─ Legacy v2 改善 ≥5pp
│  │  └─ ✓ 用 v2 替换正式 KG，扩展到 full dev，重建 train index
│  └─ Legacy v2 改善 <2pp
│     └─ ✗ Legacy 路线到顶，转向 passage-derived
│
├─ 瓶颈主要在 Relation Missing (>60%)
│  ├─ Hybrid (C) 比 NoKG (D) 高 ≥5pp 且 citation_rate >0.7
│  │  └─ ✓ Passage-derived 有效，设计训练整合
│  └─ Hybrid 无显著改善
│     └─ ✗ 检索本身是瓶颈，优先改进 retrieval
│
└─ 混合瓶颈
   ├─ Hybrid 最优
   │  └─ ✓ 设计统一证据融合协议，进入训练
   └─ 都无改善
      └─ ✗ 如实报告负结果，转向其他方向
```

---

## 关键约束

### 1. 科研诚信

- **Gold 仅用于审计**: 所有 gold 信息（answer, supporting_facts）只能在 KG 构建完成后用于评分
- **版本隔离**: 修复产物命名为 `*_v2`，不覆盖原文件
- **负结果保留**: 如果修复无效，必须如实报告

### 2. 可追溯性

每个产物必须记录：
```yaml
experiment_id: "LEGACY_KG_V2_HOTPOT_MUSIQUE_N100_SEED46"
versions:
  audit: "legacy-kg-audit-v1"
  builder: "legacy-v2-builder-1"
data_hashes:
  source_kg: "bb47221b..."
  output_kg: "a3f9d8e2..."
git_commit: "abc1234"
```

### 3. 预注册门控

对照实验的停止条件必须**提前锁定**：
- B vs A: 不能显著退化（-2pp 容忍）
- C vs D: 必须有正向效用（+5pp 且 citation >0.7）
- 统计显著性：McNemar test (p<0.05)

---

## 预期结果

### HotpotQA
- **Rerank 修复**: 可能从负增益修到持平或 +2~3pp
- **Passage edges**: 大概率主要收益来源，可能 +5~8pp

### MuSiQue  
- **Canonicalizer**: 覆盖率和效用可能转正
- **Subquery graph**: 比强制 relation graph 更适合

### 2WikiMultiHopQA
- **保留 ProofKG**: 已验证大幅增益，不修改

---

## 下一步行动

### 立即可执行（无需人工批准）

1. **运行审计**:
   ```bash
   bash scripts/run_legacy_kg_audit.sh
   ```

2. **运行测试**:
   ```bash
   pytest tests/test_legacy_kg_coverage_audit.py -v
   ```

3. **查看示例输出**:
   ```bash
   cat docs/legacy_kg_audit_guide.md
   ```

### 需要人工确认的后续步骤

1. **审计结果解读**: 研究者确认瓶颈分布和修复方向
2. **构建 Legacy v2**: 如果审计显示 rerank 有修复空间
3. **提取 Passage edges**: 如果审计显示 relation missing
4. **运行四臂对照**: 零训练验证，需要 GPU 资源
5. **训练整合**: 只有对照实验通过后才考虑

---

## 文件清单

```
scripts/
├── diagnose/
│   └── legacy_kg_coverage_audit.py         # 分层审计脚本（主要工作）
├── run_legacy_kg_audit.sh                  # 快速启动脚本
├── prepare/
│   ├── build_legacy_kg_v2.py              # TODO: Legacy v2 构建器
│   └── extract_passage_derived_edges.py   # TODO: Passage 边提取
├── evaluation/
│   └── run_kg_comparison.py               # TODO: 四臂对照实验
└── analysis/
    └── kg_repair_decision.py              # TODO: 决策报告生成

tests/
└── test_legacy_kg_coverage_audit.py        # 审计脚本测试套件

configs/
└── experiments/
    └── legacy_kg_repair_comparison.yaml    # 四臂对照配置

docs/
├── legacy_kg_audit_guide.md               # 使用指南
└── legacy_kg_repair_executive_summary.md  # 本文档

reports/
└── legacy_kg_audit/                       # 审计输出目录
    ├── audit_n100_seed46_*.json           # JSON 详细结果
    └── audit_n100_seed46_*.md             # Markdown 摘要
```

**状态**:
- ✅ **已完成**: 审计脚本、测试、文档、配置模板
- ⏳ **待实现**: Legacy v2 构建器、Passage 边提取器、对照实验运行器
- 🔒 **待批准**: 审计结果解读、修复方向选择、对照实验执行

---

## 与现有工作的关系

### 已有的成功案例：2Wiki ProofKG
- 通过 query-aware proof construction 实现 +31pp EM
- 证明高质量 KG 确实有效
- HotpotQA/MuSiQue 的修复目标：复现类似收益

### 失败的尝试
- Passage-aware KG v2/v3: mention 链接率低，未过门
- 简单扩大三元组数量：无效
- Rule-based planner: 结构门失败

### 本次方案的新颖性
- **分层诊断**: 精确定位瓶颈，而非盲目修复
- **零训练验证**: 先验证 KG 质量，再考虑训练整合
- **版本隔离**: 不覆盖已有资产，保留失败结果
- **预注册门控**: 避免事后调整评估规则

---

## 风险和限制

### 已知风险
1. **审计启发式不完美**: 目标关系推断是粗粒度的
2. **Cache 覆盖**: 离线模式下缓存 miss 会导致空 KG
3. **Passage edges 成本**: 需要额外的关系抽取模型

### 明确限制
1. **不证明训练收益**: 审计和对照只验证 KG 质量，不保证 PPO 会学到
2. **单 seed**: n=100 仍有噪声，需要多 seed 或扩大样本
3. **Dev 泛化**: dev 上的改善不保证 test 上同样有效

### 失败容忍
如果审计显示 `passage_derived_required > 80%`，说明 Legacy Wikidata 路线确实到顶，应**如实报告**并转向其他方向（如改进检索、扩展 ProofKG 到其他数据集）。

---

## 总结

**核心问题**: Legacy KG 对 HotpotQA/MuSiQue 无效，是因为信息不存在，还是存在但被错误处理？

**解决方案**: 分层审计 → 精确定位瓶颈 → 针对性修复 → 零训练验证

**成功标准**:
- Audit 完成并生成可解释的瓶颈分布
- 至少一个修复方向在对照实验中显示 ≥5pp EM 改善
- 修复策略可扩展到 full dev 和 train

**下一步**: 运行 `bash scripts/run_legacy_kg_audit.sh`，查看结果，根据瓶颈分布选择路线 A/B/C。
