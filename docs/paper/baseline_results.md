# 基线结果（Baseline Results）

> 6 条检索基线（naive_rag / self_rag / trace / r1_searcher / corag / rearag）× 3 数据集，seed=42，n=300。检索 = 两阶段混合 RRF：dense E5 top-100 + sparse BM25 top-100 → RRF k=60 → top-50 → **bge-reranker-v2-m3 → top-10**（与主方法 KG-ProWeight 完全一致的检索条件）。生成 = temperature=0，max_tokens=512。
>
> `zero_shot`（无检索，纯记忆）不在本次 rerun 内，沿用 Stage 2 数值。评估方法与复现命令见 [baseline_eval_method.md](baseline_eval_method.md)。
>
> 核对日期：2026-08-22（rearag 三数据集 rerank-10 EM/F1 + IHR 于本日补齐）。

## 1. EM（ExactMatch，越高越好）

| method | hotpotqa | 2wiki | musique |
|---|---|---|---|
| zero_shot | 0.000 | 0.000 | 0.000 |
| naive_rag | 0.250 | 0.050 | 0.090 |
| self_rag | 0.177 | 0.113 | 0.087 |
| r1_searcher | **0.387** | **0.370** | 0.177 |
| corag | 0.203 | 0.197 | 0.120 |
| trace (IRCOT) | 0.377 | 0.353 | **0.227** |
| rearag | 0.337 | 0.313 | 0.263 |

## 2. F1（token 级，越高越好）

| method | hotpotqa | 2wiki | musique |
|---|---|---|---|
| zero_shot | 0.044 | 0.041 | 0.033 |
| naive_rag | 0.373 | 0.168 | 0.196 |
| self_rag | 0.288 | 0.242 | 0.196 |
| r1_searcher | **0.506** | **0.441** | 0.269 |
| corag | 0.285 | 0.289 | 0.180 |
| trace (IRCOT) | 0.500 | **0.463** | **0.343** |
| rearag | 0.476 | 0.474 | 0.395 |

## 3. IHR（Intermediate Hallucination Rate，越低越好）

仅推理基线适用（无可抽取推理步的基线不适用）。judge = deepseek-v4-pro，n=50，seed=42。**本表在 rerank-10 输出上复测**（trace 旧 top-15 IHR 为 0.380 / 0.403 / 0.468，rerank 后明显下降）。

| method | hotpotqa | 2wiki | musique |
|---|---|---|---|
| trace (IRCOT) | 0.253 | 0.286 | 0.434 |
| r1_searcher | 0.101 | 0.173 | 0.101 |
| rearag | 0.170 | 0.179 | 0.188 |

rearag 旧 top-15 IHR = 0.237（hotpotqa，deepseek-v4-pro），rerank-10 后降为 0.170；2wiki / musique 为本次新增（此前未测）。

## 4. 原始数值（精确，供复现核对）

EM / F1（seed=42，n=300；检索基线为 rerank-10，zero_shot 沿用 Stage 2）：

| method | dataset | EM | F1 |
|---|---|---|---|
| zero_shot | hotpotqa | 0.000000 | 0.044289 |
| zero_shot | 2wiki | 0.000000 | 0.041325 |
| zero_shot | musique | 0.000000 | 0.033340 |
| naive_rag | hotpotqa | 0.250000 | 0.373425 |
| naive_rag | 2wiki | 0.050000 | 0.168376 |
| naive_rag | musique | 0.090000 | 0.195895 |
| self_rag | hotpotqa | 0.176667 | 0.288005 |
| self_rag | 2wiki | 0.113333 | 0.242097 |
| self_rag | musique | 0.086667 | 0.196227 |
| trace | hotpotqa | 0.376667 | 0.499720 |
| trace | 2wiki | 0.353333 | 0.463324 |
| trace | musique | 0.226667 | 0.343407 |
| r1_searcher | hotpotqa | 0.386667 | 0.506149 |
| r1_searcher | 2wiki | 0.370000 | 0.441314 |
| r1_searcher | musique | 0.176667 | 0.268711 |
| corag | hotpotqa | 0.203333 | 0.284926 |
| corag | 2wiki | 0.196667 | 0.288925 |
| corag | musique | 0.120000 | 0.179939 |
| rearag | hotpotqa | 0.336667 | 0.475665 |
| rearag | 2wiki | 0.313333 | 0.474421 |
| rearag | musique | 0.263333 | 0.394518 |

## 5. 数据落盘位置

- **EM/F1**：`outputs/baselines_rerank/<method>/<dataset>/seed_42/<ts>_<method>/metric_score.txt`（rerank-10 的 6 条检索基线）；`zero_shot` 沿用 `outputs/baselines_stage2/zero_shot/...`。
- **IHR**：`.../ihr_result_ircot.json` / `.../ihr_result_r1_searcher.json` / `.../ihr_result_rearag.json`（`mean_ihr` / `judge_model` / `n_items` / `items[].ihr+steps[]`）。
- **逐样本**：`.../intermediate_data.json`（`output.retrieval_result` 10 docs、`output.pred`、`output.metric_score`；r1_searcher 另有 `output.raw_pred` 保存完整原始链供 IHR 抽步骤）。

## 6. 需注意的口径（写论文前定夺）

1. **检索 top-k 已对齐**：基线现为 top-50 → bge-reranker → 10，与主方法一致。上一版（Stage 2）基线用 top-15 无 rerank 的数值作废，跨方法比较一律以本表 rerank-10 为准。
2. **GPU**：评估（主方法 + 基线）均在 **RTX 4090（24GB）** 上跑，两侧 manifest 已核实一致（`gpu_name=NVIDIA GeForce RTX 4090`）。`04_experimental_setup.md` 的「Blackwell 96GB」表述需改为 4090。
3. **IHR judge 模型**：本批基线 IHR 用 deepseek-v4-pro；历史 KG-ProWeight（主方法）IHR 用 deepseek-chat，两者 IHR 数值不可直接对比（基线内部自洽）。
4. **zero_shot EM=0** 是预期：无检索纯记忆，三数据集 EM 均为 0（F1 ~0.03–0.04），作为检索增益的下限参照。
