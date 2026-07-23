# R9 v6 可执行计划

> 基于 solve_suggestion.md，对齐当前已完成工作

---

## 已完成（阶段 0~2 部分）

- [x] `wd:`→`wdt:` filter 前缀
- [x] Phase1 `retrieval_top_k` 不生效
- [x] KG merge（动态+静态合并，不再替换）
- [x] relevance 自证循环（剥离 Knowledge Used 文本）
- [x] `step_reward_scale` schema + CLI 传参
- [x] `alpha_override` 缺 `step_reward_scale` 乘数
- [x] 缓存键加入 filter 版本
- [x] t=0 baseline 评估（CoRAG, R1-Searcher）

---

## Step 1: 建立诊断基线（阶段 1）——不改代码，跑指标

在 HotpotQA/2Wiki/MuSiQue 各固定 100 题上记录：

```bash
# 已有：R9 v5 EM/F1/α/KL
# 需要补：
python scripts/eval/run_retrieval_recall.py  # 需要新增脚本
# 输出：dense Recall@5/15/50, BM25 Recall@5/15/50, RRF Recall@5/15/50
# 输出：entity linking Acc@1, KG relation 分布, prompt token 数
```

**产物**：`docs/r9_v6_baseline_diagnostics.md`

---

## Step 2: KG 缓存重建（阶段 3 核心）——最大杠杆点

### 2a. 新建 `scripts/prepare/06_build_question_kg_index.py`

功能：
1. 从 3 个数据集的 dev split 读取问题，分配 question_id
2. 对每个问题提取 mention → 调用 EntityLinker → 得 QID
3. 从 entity_subgraph_cache 读取原始子图
4. 三层过滤：
   - 硬删除：disambiguation/category/list/metadata 关系
   - 配额：instance_of≤2, subclass_of≤2, 同 PID≤20%
   - 排序：entity_anchor + relation_question_similarity + triple_question_similarity
5. 输出 `question_kg_index_v2.json`

### 2b. 更新 `scripts/r9_preflight.py`

加检查项：taxonomic relation 占比、每题 KG token 数、useful-triple Recall@30、版本一致性

### 2c. 验证

人工抽查每数据集 50 题，确认 Big Stone Gap / Corliss Archer 等已知错误已修复。

**产物**：`indexes/kg_cache/question_kg_index_v2.json`

---

## Step 3: 检索增强（阶段 4）

### 3a. 新增 `kgproweight/retrieval/reranker.py`

```python
class RRFRerankRetriever:
    def __init__(self, dense, sparse, cross_encoder, config):
        ...
    def batch_search(self, questions):
        dense_results = self.dense.batch_search(questions, num=100)
        sparse_results = self.sparse.batch_search(questions, num=100)
        candidates = rrf_merge(dense_results, sparse_results, topk=50)
        return self.cross_encoder.rerank(questions, candidates, topk=10)
```

### 3b. 更新 `kgproweight/retrieval/hybrid.py`

区分 `candidate_topk`、`rrf_topk`、`rerank_topk`，新增 `prompt_passage_token_budget`

### 3c. 重新生成银标 passages

用新 retriever 重新跑 Phase1 的一小批（100 题），验证 passage 质量提升。

---

## Step 4: 数据链消融（阶段 5）——先不训练 PPO

用同一个 Elite SFT checkpoint，跑 5 个推理变体（每数据集 100 题）：

```
A. 旧检索 + 旧 KG            ← baseline
B. 新检索 + 无 KG            ← 纯文本
C. 新检索 + 旧 KG（v1）
D. 新检索 + 新 KG（v2）      ← 核心对比
E. 新检索 + 新 KG + alpha=0  ← 只看文本
```

只有当 D > A 且 D > B 在 ≥2 个数据集上成立，才进入 PPO。

**产物**：`docs/r9_v6_data_chain_ablation.md`

---

## Step 5: PPO 消融（阶段 6）——仅当 Step 4 通过

从 Elite SFT 重新训练 500 步，3 seeds，逐步加 reward 组件：

| Exp | 配置 |
|---|---|
| E0 | Elite SFT（不训 PPO） |
| E1 | outcome_weight=10, alpha=0 |
| E2 | E1 + text_reward |
| E3 | E2 + KG precision |
| E4 | E3 + question-aware relevance |
| E5 | E4 + 新 KG v2 |
| E6 | E5 + 新检索 |

报告均值±std, 95% CI, paired bootstrap。

---

## 时间估计

| Step | 内容 | 预计时间 |
|---|---|---|
| 1 | 诊断脚本 + 跑指标 | 2-3h |
| 2 | KG 缓存重建 | 1 天（含 Wikidata 查询） |
| 3 | 检索 reranker | 半天（代码 + 下载模型） |
| 4 | 数据链消融 | 半天（推理评估） |
| 5 | PPO 消融 | 需要 GPU（远程） |

**建议**：本周完成 Step 1-2，Step 3-4 下周，Step 5 等 GPU。
