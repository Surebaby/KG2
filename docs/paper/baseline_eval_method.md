# 基线评估方法（Baseline Evaluation Methodology）

> 本文档描述 `scripts/eval/run_baselines.py` 如何跑 7 条基线、如何从原始生成文本得到 EM/F1，以及 IHR（Intermediate Hallucination Rate）如何评估。对应代码：`kgproweight/eval/baselines.py`、`kgproweight/eval/runner.py`、`flashrag_src/flashrag/evaluator/metrics.py`。
>
> 最后核对日期：2026-08-21（rerank-10 对齐主方法后）。

## 1. 总流程

[run_baselines.py](scripts/eval/run_baselines.py) 解析 `--methods --datasets --seeds --test_sample_num`，做三重循环：

```
for method in methods:
    for ds in datasets:
        for seed in seeds:
            run_evaluation(...)     # kgproweight/eval/runner.py
```

每个 `(method, ds, seed)` 组合独立跑一次，流程固定为：

```
问题 → [检索 top-50 → bge-reranker → top-10] → 模板拼 prompt → 模型生成 pred → pred_process_fun 抽最终答案 → EM/F1 打分
```

`runner.run_evaluation()`（[runner.py:80](kgproweight/eval/runner.py#L80)）负责：设置种子 → 加载数据集（采样 n=300）→ 实例化 pipeline → 调 `pipeline.run(ds, do_eval=True, pred_process_fun=...)` → 写 `metric_score.json` + `manifest.json`。`finally` 块里 `del pipeline + gc.collect() + torch.cuda.empty_cache()`，保证循环跑下一个方法时能重新加载进完整 VRAM（避免 device_map=auto 误判空闲显存而把模型 offload 到 CPU）。

## 2. 基线清单

7 条基线在 [baselines.py](kgproweight/eval/baselines.py) 的 `BASELINES` 注册表里定义，每条只改 pipeline 类和 generator 模型：

| name | pipeline 类 | generator | 说明 |
|---|---|---|---|
| zero_shot | SequentialPipeline（`run_mode="naive"`） | llama3-8B-instruct | **无检索**，纯靠模型记忆，EM/F1 下限 |
| naive_rag | SequentialPipeline | llama3-8B-instruct | 单跳：检索 top-50 → rerank-10 → 拼 prompt → 生成 |
| self_rag | SelfRAGPipeline | selfrag | 自适应判断是否需要检索 |
| trace | IRCOTPipeline（`max_iter=6`） | llama3-8B-instruct | 迭代推理，逐句生成，抽 "So the answer is" |
| r1_searcher | SequentialPipeline | r1-searcher | 生成含 `<answer>` 标签，从中抽取 |
| corag | SequentialPipeline | corag | CoRAG，抽首个内容行 |
| rearag | ReaRAGPipeline（`max_iter_num=15`） | rearag | 显式 Thought/Action 结构 |

`zero_shot` 走 `pipeline.naive_run()`（[pipeline.py:72](flashrag_src/flashrag/pipeline/pipeline.py#L72)），**跳过检索**，直接 `问题 → 生成`。

## 3. 检索（仅 standard 方法；zero_shot 无）

混合 RRF，由 [hybrid.py](kgproweight/retrieval/hybrid.py) 的 `build_flashrag_config` 构造：

- **语料**：全量 Wikipedia wiki18 ≈ 21M 文档（`indexes_wiki18/corpus_flashrag.jsonl`）。
- **dense**：E5-base（768 维），`indexes_wiki18/e5_fp16.dat` memmap，chunked brute-force（CPU faiss）。
- **sparse**：BM25（bm25s 后端）。
- **融合**：dense 与 sparse 各取 top-100 → RRF（`rrf_k=60`）→ **top-50**。
- **两阶段 rerank**：top-50 再经 `bge-reranker-v2-m3`（cross-encoder）重排取 **top-10**。由 `runner._wrap_retriever_rerank`（[runner.py:80](kgproweight/eval/runner.py#L80)）在 `pipeline.retriever.batch_search` 外包一层，复用主方法的 `rerank_passages` 分发器，两侧 reranker 完全一致。
- 命令侧对应 `--retrieval_topk 50 --rerank 10`（[run_baselines.py](scripts/eval/run_baselines.py)）。

> ✅ 已对齐：基线现为 top-50 → rerank-10，与主方法一致。上一版基线 `DEFAULT_TOPK=15`（无 reranker）的数值作废（见 [baseline_results.md](baseline_results.md) 口径说明）。

## 4. 生成

HF `generate`，eval 全程确定性解码：

- `temperature=0.0`，`do_sample=False`（greedy）。
- `max_tokens=512`，`generator_max_input_len=6144`，`generator_batch_size=1`。
- prompt 由模板拼装，如 naive_rag：`"Reference passages:\n<top-10>\n\nQuestion: {q}\nAnswer:"`。

## 5. 答案抽取（`pred_process_fun`）

非推理基线生成的 `pred` 基本就是答案本身；推理基线（trace/rearag）的 `pred` 是**整条推理链**，必须先抽最终答案，否则 EM 恒为 0。

- **trace**：`ircot_pred_parse`（[pred_parse.py:21](flashrag_src/flashrag/utils/pred_parse.py#L21)）取 `"So the answer is[:]"` 之后的内容（容忍无冒号）。
- **r1_searcher**：`r1_extract_answer` 抽 `<answer>…</answer>` 标签内容，兜底 `answer:` 行 / 最后一行。
- **corag**：`corag_extract_answer` 抽首个非 `Question:/Answer:` 前缀且长度 >3 的内容行。
- **zero_shot / naive_rag / rearag**：不做抽取（raw pred 即答案）。

`runner` 会把 `spec.extras["pred_process_fun"]` 传给 `pipeline.run(..., pred_process_fun=...)`，在打分前应用到每个样本（[pipeline.py:27](flashrag_src/flashrag/pipeline/pipeline.py#L27)）。

## 6. EM / F1 计算（[metrics.py](flashrag_src/flashrag/evaluator/metrics.py)）

先做 `normalize_answer`（[utils.py:5](flashrag_src/flashrag/evaluator/utils.py#L5)），顺序为：**小写 → 去标点 → 去冠词 a/an/the → 合并空白**。

**EM（ExactMatch）**：归一化后 `pred == 任一 gold answer` 记 1，否则 0，对 n 个样本求平均。

**F1（token 级）**：归一化后按空格切词，取 pred 与 gold 的**词多重集交集**（`Counter`）：

```
precision = 公共词数 / pred 词数
recall    = 公共词数 / gold 词数
F1        = 2·P·R / (P + R)
```

多个 gold answer 取 max；若 pred 或 gold 是 `yes/no/noanswer` 且两者不等，直接记 0。

**例**：pred = `"President Barack Obama"`，gold = `"Barack Obama"` → 归一化 `"president barack obama"` vs `"barack obama"`；EM = 0（不等）；F1：公共词 {barack, obama}=2，precision=2/3，recall=2/2=1，F1 = 2·(2/3)·1 / (2/3+1) = 0.8。

**配置的指标**：`metrics = ["em", "f1", "input_tokens"]`。`avg_input_tokens` 只是 tiktoken 数 prompt 长度的遥测，与 EM/F1 无关（日志里的 `Error in input_tokens: 'prompt'` 是该遥测对某些样本取不到 prompt 字段，不影响评分）。

## 7. IHR（Intermediate Hallucination Rate）

仅**推理基线**适用（trace / rearag / r1_searcher），用 LLM-as-judge 逐推理步判「是否幻觉」：

- judge 模型：`deepseek-v4-pro`（OpenAI 兼容端点 `https://api.deepseek.com`，key 在 `.env`）。
- 步骤抽取：
  - trace（method=`ircot`）取每个 iteration 的 `new_thought`；
  - rearag（method=`rearag`）取 `output.messages` 里 assistant 轮的 `Thought N:` 段；
  - r1_searcher（method=`r1_searcher`）取 `output.raw_pred` 里第一个 `<think>…</think>` 推理块（`<answer>` 之后的重复段丢弃），按句切分。
- 逐步返回 `{hallucination, confidence, reason}`，`aggregate_ihr = 幻觉步数 / 总步数`。
- 采样 `n=50`，`seed=42`。
- 非推理基线（zero_shot/naive_rag/self_rag/corag）无可抽取推理步，IHR 不适用。

脚本：`scripts/eval/run_baseline_ihr.py --method {ircot,rearag,r1_searcher} --judge_model deepseek-v4-pro`。

## 8. 产出与复现

每个组合在 `outputs/<root>/<method>/<dataset>/seed_<seed>/<时间戳>_<method>/` 下写：

- `metric_score.json` / `metric_score.txt` — 汇总 `{em, f1, avg_input_tokens}`。
- `intermediate_data.json` — 每个样本的 `id/question/golden_answers/output`，其中 `output` 含 `retrieval_result`（10 docs）、`prompt`、`pred`（抽取后）、`metric_score`（逐样本 em/f1/input_tokens）。r1_searcher 额外保留 `output.raw_pred`（完整原始链）供 IHR 抽步骤。
- `manifest.json` — 环境快照（git commit、GPU、包版本）。

**复现命令**（以 Stage 1 trace 为例）：

```bash
export PYTHONPATH="/home/zjulab/kgpaper/flashrag_src:/home/zjulab/kgpaper"
export KGPW_CORPUS_PATH=/home/zjulab/kgpaper/indexes_wiki18/corpus_flashrag.jsonl
export KGPW_DENSE_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/e5_fp16.dat
export KGPW_BM25_INDEX_PATH=/home/zjulab/kgpaper/indexes_wiki18/bm25
export CUDA_VISIBLE_DEVICES=0

python scripts/eval/run_baselines.py \
  --methods trace --datasets hotpotqa --seeds 42 \
  --test_sample_num 300 --save_root outputs/baselines_trace_fix
```

## 9. 口径说明

1. **检索 top-k（已对齐）**：基线现为 `top-50 → bge-reranker → 10`，与主方法一致。上一版基线 `DEFAULT_TOPK=15` 无 rerank 的数值作废，见 [baseline_results.md](baseline_results.md)。
2. **GPU**：基线 `manifest.json` 记录 `gpu_name=NVIDIA GeForce RTX 4090`（24GB），而 `04_experimental_setup.md` 写的是 RTX PRO 6000 Blackwell（96GB）。评估实际均在 4090 上跑，`04_experimental_setup.md` 需改为 4090。
3. **IHR judge 模型**：本批基线 IHR 用 `deepseek-v4-pro`；历史 KG-ProWeight（主方法）IHR 用 `deepseek-chat`，两者 IHR 数值不可直接对比（基线内部自洽）。
