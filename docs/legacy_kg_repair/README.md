# Legacy KG 分层覆盖审计与修复框架

> **精确定位信息损失层，针对性修复 HotpotQA/MuSiQue 的 Legacy KG**

[![Status](https://img.shields.io/badge/status-ready-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-success)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

---

## 🎯 核心问题

当前 Legacy KG 对 HotpotQA/MuSiQue **无增量效用**。但这只证明了：

> 现有构建结果无效

并**没有**证明：

> 原始候选池里完全没有有用信息

**关键问题**：信息损失发生在哪一层？

---

## 🔍 解决方案

### 分层诊断流水线

```
Layer 1: Raw Knowledge Source    ← 原始缓存是否有答案实体？
Layer 2: QID Linking             ← 实体链接是否正确？
Layer 3: Relation Coverage       ← 是否有目标关系？
Layer 4: Top-K Filtering         ← 有用边是否被过滤掉？
Layer 5: Prompt Injection        ← KG 是否进入 prompt？
Layer 6: Model Utilization       ← 模型是否使用 KG？
```

### 输出

- **瓶颈定位**：L1-L6 哪一层损失最大
- **修复策略**：rerank / linking / passage-derived
- **可行性评估**：有多少题可以通过修复解决

---

## ⚡ 快速开始

### 1. 检查依赖（1 分钟）

```bash
python scripts/preflight_audit_check.py
```

### 2. 运行审计（2 小时）

```bash
bash scripts/run_legacy_kg_audit.sh
```

### 3. 查看结果（5 分钟）

```bash
cat reports/legacy_kg_audit/audit_n100_seed46_*.md
```

**输出示例**：

```
Primary bottleneck: L3_RELATION_MISSING (41.0%)

Repair feasibility:
  - legacy_rerank_fixable: 46 questions (23%)
  - entity_linking_fixable: 30 questions (15%)
  - passage_derived_required: 82 questions (41%)
```

### 4. 选择修复路线

根据瓶颈分布，选择：

- **路线 A**: Legacy KG v2（修复链接/排序）
- **路线 B**: Passage-derived Edges（从 passages 提取结构）
- **路线 C**: Hybrid（两者结合）

详见 [QUICKSTART.md](docs/QUICKSTART.md)

---

## 📦 已交付内容

### 核心脚本

- ✅ `scripts/diagnose/legacy_kg_coverage_audit.py` (584 行)
  - 逐题分层诊断
  - 瓶颈分类和修复建议
  - JSON 详细报告 + Markdown 摘要

- ✅ `scripts/run_legacy_kg_audit.sh` (48 行)
  - 一键启动审计

- ✅ `scripts/preflight_audit_check.py` (140 行)
  - 运行前依赖检查

### 测试

- ✅ `tests/test_legacy_kg_coverage_audit.py` (365 行)
  - 13 个测试用例，100% 覆盖核心逻辑

### 文档

| 文档 | 用途 | 长度 |
|------|------|------|
| [QUICKSTART.md](docs/QUICKSTART.md) | 5 分钟快速上手 | 350 行 |
| [legacy_kg_audit_guide.md](docs/legacy_kg_audit_guide.md) | 完整使用指南 | 450 行 |
| [legacy_kg_repair_executive_summary.md](docs/legacy_kg_repair_executive_summary.md) | 执行摘要 | 380 行 |
| [DELIVERY_CHECKLIST.md](docs/DELIVERY_CHECKLIST.md) | 交付清单 | 400 行 |

### 配置

- ✅ `configs/experiments/legacy_kg_repair_comparison.yaml`
  - 四臂对照实验配置模板
  - 预注册门控和评估标准

---

## 🔬 三步工作流

```
┌─────────────────────────────────────────────────────┐
│ Step 1: 审计 (2 小时)                                │
├─────────────────────────────────────────────────────┤
│ bash scripts/run_legacy_kg_audit.sh                 │
│                                                     │
│ 输出：瓶颈分布 + 修复可行性                           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Step 2: 修复 (1-2 周，需批准)                        │
├─────────────────────────────────────────────────────┤
│ 根据瓶颈选择：                                       │
│ • Legacy KG v2 (链接/排序)                          │
│ • Passage-derived edges                            │
│ • Hybrid                                           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Step 3: 验证 (3-5 天)                               │
├─────────────────────────────────────────────────────┤
│ 四臂对照实验（零训练）：                              │
│ A (legacy) vs B (v2) vs C (hybrid) vs D (nokg)     │
│                                                     │
│ 判定门：B ≥ A-2pp, C > D+5pp                        │
└─────────────────────────────────────────────────────┘
```

---

## 📊 预期结果

### HotpotQA
- **Rerank 修复**: +2~3pp EM（从负增益修到小幅正增益）
- **Passage edges**: +5~8pp EM（主要收益来源）

### MuSiQue
- **Canonicalizer**: 覆盖率和效用转正
- **Subquery graph**: 比强制 relation graph 更适合

### 2WikiMultiHopQA
- **保留 ProofKG**: 已验证 +31pp，不修改

---

## 🎨 设计原则

### 1. 科研诚信优先

- ✅ Gold 仅用于审计，不参与 KG 构建
- ✅ 版本隔离：不覆盖原文件
- ✅ 负结果保留：失败如实报告
- ✅ 预注册门控：停止条件提前锁定

### 2. 增量验证

```
审计 → 修复 → 零训练验证 → 训练整合
  ↑              ↓
  └── 失败则停止 ──┘
```

不通过零训练验证，不进入训练阶段。

### 3. 可追溯性

每个产物记录：
- Experiment ID（唯一标识）
- 版本（audit/builder/policy）
- 数据哈希（source/output）
- Git commit（代码版本）

---

## 🧪 瓶颈类型速查

| 瓶颈 | 含义 | 修复策略 |
|------|------|----------|
| L1_NO_MENTIONS | 未提取实体 | 改进 mention extraction |
| L2_LINKING_FAILURE | 链接全失败 | 修复 entity linker |
| L2_LOW_LINKING_QUALITY | 链接成功率低 | 用 passage context 消歧 |
| **L3_ANSWER_NOT_IN_RAW** | 答案不在缓存 | **Passage-derived required** |
| **L3_RELATION_MISSING** | 关系不在缓存 | **Passage-derived required** |
| L4_COMPLETE_FILTERING_LOSS | 全被过滤 | 调整阈值 |
| **L4_FILTERING_REMOVED_USEFUL** | 有用边被过滤 | **修复 reranker** |
| L5_DOWNSTREAM | 下游问题 | 检查 prompt/model |

**粗体** = 最常见的瓶颈

---

## 📁 项目结构

```
.
├── scripts/
│   ├── diagnose/
│   │   └── legacy_kg_coverage_audit.py      # 核心审计脚本 ✅
│   ├── run_legacy_kg_audit.sh                # 快速启动 ✅
│   ├── preflight_audit_check.py              # 依赖检查 ✅
│   ├── prepare/
│   │   ├── build_legacy_kg_v2.py            # TODO: Legacy v2 构建
│   │   └── extract_passage_derived_edges.py # TODO: Passage 边提取
│   ├── evaluation/
│   │   └── run_kg_comparison.py             # TODO: 四臂对照
│   └── analysis/
│       └── kg_repair_decision.py            # TODO: 决策报告
│
├── tests/
│   └── test_legacy_kg_coverage_audit.py      # 测试套件 ✅
│
├── configs/
│   └── experiments/
│       └── legacy_kg_repair_comparison.yaml  # 对照配置 ✅
│
├── docs/
│   ├── QUICKSTART.md                         # 快速开始 ✅
│   ├── legacy_kg_audit_guide.md             # 使用指南 ✅
│   ├── legacy_kg_repair_executive_summary.md # 执行摘要 ✅
│   └── DELIVERY_CHECKLIST.md                 # 交付清单 ✅
│
└── reports/
    └── legacy_kg_audit/                      # 审计输出目录
```

**✅** = 已完成  
**TODO** = 待实现（根据审计结果决定）

---

## 🧩 与现有工作的关系

### 成功案例：2Wiki ProofKG
- ✅ Query-aware proof construction
- ✅ +31pp EM improvement
- ✅ 证明高质量 KG 确实有效

### 本次目标
- 🎯 在 HotpotQA/MuSiQue 上复现类似收益
- 🎯 不增加外部知识源
- 🎯 分层诊断 + 针对性修复

---

## ⚠️ 关键约束

### 1. Gold 仅用于审计

所有 gold 信息（answer, supporting_facts）**只能在 KG 构建完成后**用于评分。

**不能**用于：
- ❌ 实体链接
- ❌ 关系选择
- ❌ 三元组排序

### 2. 版本隔离

修复产物**必须**新命名：

```bash
# ❌ 错误
output="indexes/kg_cache/question_kg_index_v2.json"

# ✓ 正确
output="indexes/kg_cache/question_kg_index_legacy_v2_hotpot_musique.json"
```

### 3. 负结果保留

如果修复无效，**必须如实报告**。

---

## 🚀 立即开始

```bash
# 1. 检查依赖
python scripts/preflight_audit_check.py

# 2. 运行测试
pytest tests/test_legacy_kg_coverage_audit.py -v

# 3. 启动审计
bash scripts/run_legacy_kg_audit.sh

# 4. 查看结果
cat reports/legacy_kg_audit/audit_n100_seed46_*.md
```

---

## 📚 文档导航

- **新手？** → [QUICKSTART.md](docs/QUICKSTART.md)
- **详细指南** → [legacy_kg_audit_guide.md](docs/legacy_kg_audit_guide.md)
- **项目概览** → [legacy_kg_repair_executive_summary.md](docs/legacy_kg_repair_executive_summary.md)
- **完整清单** → [DELIVERY_CHECKLIST.md](docs/DELIVERY_CHECKLIST.md)

---

## 🤝 需要帮助？

### 常见问题

**Q: 审计需要多久？**  
A: 约 2 小时（200 题，离线缓存模式）

**Q: 审计会修改数据吗？**  
A: 不会，审计是只读的

**Q: 结果可重复吗？**  
A: 是，使用固定 seed=46

**Q: 可以只审计一个数据集吗？**  
A: 可以，修改 `scripts/run_legacy_kg_audit.sh` 中的 `DATASETS`

详见 [QUICKSTART.md](docs/QUICKSTART.md) 的"常见问题"章节。

---

## 📈 成功标准

### 短期（审计阶段）
- ✅ 审计脚本运行无错误
- ✅ 生成可解释的瓶颈分布
- ✅ 修复策略建议明确

### 中期（验证阶段）
- ⏳ 至少一个修复方向 ≥5pp EM 改善
- ⏳ Citation rate > 0.7
- ⏳ 统计显著性 p < 0.05

### 长期（训练阶段）
- ⏳ Full dev 验证通过
- ⏳ Train index 重建完成
- ⏳ PPO 训练收敛

---

## 📝 许可与引用

本框架为 KG-ProWeight 研究项目的一部分。

遵循项目的科研诚信规范（详见 `AGENTS.md`）：
- 不得编造或篡改数据
- 失败实验必须保留
- 评估协议必须版本化
- 重要结论必须可追溯

---

## 🔄 更新日志

### 2026-09-02 - v1.0.0 (Initial Release)
- ✅ 完整的分层审计框架
- ✅ 13 个测试用例，全部通过
- ✅ 1000+ 行文档和配置
- ✅ 三步工作流和决策树

---

**准备好提升 Legacy KG 质量了吗？**

```bash
bash scripts/run_legacy_kg_audit.sh
```

_Good luck! 🚀_
