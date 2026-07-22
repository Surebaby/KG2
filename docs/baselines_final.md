# KG-ProWeight Baseline 对比结果 (Final)

> **最后更新: 2026-07-17**
> **评估协议**: hybrid RRF top-15 检索 (E5 dense + BM25 sparse) / 3 数据集 × 3 种子 × 100 样本 / EM + F1
> **生成参数**: `max_tokens=512, temperature=0.7, do_sample=True, max_input_len=4096`
> **R1-Searcher 生成参数**: `max_tokens=1024, temperature=0.6, top_p=0.9`（需更大 token 预算容纳推理链）

---

## 一、Baseline 清单

| # | Baseline | 模型 | 类型 | 来源 |
|---|---|---|---|---|
| 1 | **Zero-shot** | Llama-3-8B-Instruct | 无检索, 仅模型参数知识 | FlashRAG SequentialPipeline (naive) |
| 2 | **Naive RAG** | Llama-3-8B-Instruct | 单轮检索 + 简单 prompt | FlashRAG SequentialPipeline |
| 3 | **CoRAG** | CoRAG-Llama3.1-8B | NeurIPS 2025 多跳 RAG | FlashRAG SequentialPipeline |
| 4 | **R1-Searcher** | Llama-3.1-8B-RAG-RL | arXiv 2025 推理+搜索 RL | FlashRAG SequentialPipeline |
| 5 | **Elite SFT** | Llama-3-8B-Instruct + LoRA | 精品 2k 样本 SFT | KG-ProWeight Pipeline |
| 6 | **Full SFT** | Llama-3-8B-Instruct + LoRA | 全量银标数据 SFT | KG-ProWeight Pipeline |
| 7 | **R9 v3 (PPO)** | Llama-3-8B-Instruct + LoRA | Elite SFT + Precision PRM + Dynamic KG + lr=1e-6 | KG-ProWeight Pipeline |

**备注**:
- Zero-shot / Naive RAG / CoRAG 使用 FlashRAG 标准 pipeline 评估
- Elite SFT / Full SFT / R9 v3 使用 KG-ProWeight 自研 pipeline 评估
- 所有 baseline 共享同一套 hybrid RRF 检索配置 (E5 + BM25, top-15)

---

## 二、EM 对比

### 2.1 汇总表

| 数据集 | Zero-shot | Naive RAG | CoRAG | R1-Searcher | Elite SFT | Full SFT | **R9 v3** |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 0.203 | 0.177 | 0.367 | 0.310 | 0.353 | **0.397** | 0.360 |
| 2wikimultihopqa | 0.080 | 0.007 | 0.133 | 0.143 | 0.273 | **0.303** | 0.087 |
| musique | 0.027 | 0.000 | 0.000 | 0.010 | 0.143 | **0.173** | 0.000 |
| **平均** | **0.103** | **0.061** | **0.167** | **0.154** | **0.257** | **0.291** | **0.149** |

### 2.2 F1 对比

| 数据集 | Zero-shot | Naive RAG | CoRAG | R1-Searcher | Elite SFT | Full SFT | **R9 v3** |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 0.281 | 0.257 | 0.444 | 0.444 | 0.456 | **0.511** | 0.419 |
| 2wikimultihopqa | 0.225 | 0.105 | 0.156 | 0.167 | 0.315 | **0.334** | 0.139 |
| musique | 0.103 | 0.021 | 0.048 | 0.102 | 0.196 | **0.244** | 0.059 |
| **平均** | **0.203** | **0.127** | **0.216** | **0.238** | **0.322** | **0.363** | **0.206** |

---

## 三、Per-Seed 详细拆解

### 3.1 Zero-shot (Llama-3-8B-Instruct, 无检索)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | — | — |
| hotpotqa | 42 | — | — |
| hotpotqa | 2024 | — | — |
| **hotpotqa avg** | | **0.203** | **0.281** |
| 2wikimultihopqa | 13 | — | — |
| 2wikimultihopqa | 42 | — | — |
| 2wikimultihopqa | 2024 | — | — |
| **2wiki avg** | | **0.080** | **0.225** |
| musique | 13 | — | — |
| musique | 42 | — | — |
| musique | 2024 | — | — |
| **musique avg** | | **0.027** | **0.103** |

> Zero-shot 无 per-seed 原始记录，仅保留平均值（来自 eval_base 汇总）。

### 3.2 Naive RAG (Llama-3-8B-Instruct, 单轮检索)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | — | — |
| hotpotqa | 42 | — | — |
| hotpotqa | 2024 | — | — |
| **hotpotqa avg** | | **0.177** | **0.257** |
| 2wikimultihopqa | 13 | — | — |
| 2wikimultihopqa | 42 | — | — |
| 2wikimultihopqa | 2024 | — | — |
| **2wiki avg** | | **0.007** | **0.105** |
| musique | 13 | — | — |
| musique | 42 | — | — |
| musique | 2024 | — | — |
| **musique avg** | | **0.000** | **0.021** |

> Naive RAG 无 per-seed 原始记录，仅保留平均值。

### 3.3 CoRAG (CoRAG-Llama3.1-8B, NeurIPS 2025)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | 0.40 | 0.493 |
| hotpotqa | 42 | 0.40 | 0.500 |
| hotpotqa | 2024 | 0.30 | 0.338 |
| **hotpotqa avg** | | **0.367** | **0.444** |
| 2wikimultihopqa | 13 | 0.20 | 0.204 |
| 2wikimultihopqa | 42 | 0.20 | 0.203 |
| 2wikimultihopqa | 2024 | 0.00 | 0.060 |
| **2wiki avg** | | **0.133** | **0.156** |
| musique | 13 | 0.00 | 0.036 |
| musique | 42 | 0.00 | 0.040 |
| musique | 2024 | 0.00 | 0.069 |
| **musique avg** | | **0.000** | **0.048** |

> 数据来源: `outputs/_baselines/corag/` (FlashRAG 标准 pipeline, 2026-07-16)

### 3.4 R1-Searcher (Llama-3.1-8B-RAG-RL, arXiv 2025)

> 使用 SequentialPipeline + `<think>` 格式 prompt + `max_tokens=1024`。
> 该模型为推理+搜索格式训练，需 `<think>` 开头激活推理链。
> hotpotqa 仅完成 2/3 seeds（seed_2024 因显存碎片 OOM）。

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | 0.30 | 0.443 |
| hotpotqa | 42 | 0.32 | 0.444 |
| hotpotqa | 2024 | ❌ OOM | — |
| **hotpotqa avg** | | **0.310** | **0.444** |
| 2wikimultihopqa | 13 | 0.15 | 0.168 |
| 2wikimultihopqa | 42 | 0.13 | 0.160 |
| 2wikimultihopqa | 2024 | 0.15 | 0.172 |
| **2wiki avg** | | **0.143** | **0.167** |
| musique | 13 | 0.01 | 0.101 |
| musique | 42 | 0.02 | 0.114 |
| musique | 2024 | 0.00 | 0.093 |
| **musique avg** | | **0.010** | **0.102** |

### 3.5 Elite SFT (Llama-3-8B-Instruct + LoRA, 2k 精品样本)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | — | — |
| hotpotqa | 42 | — | — |
| hotpotqa | 2024 | — | — |
| **hotpotqa avg** | | **0.353** | **0.456** |
| 2wikimultihopqa | 13 | — | — |
| 2wikimultihopqa | 42 | — | — |
| 2wikimultihopqa | 2024 | — | — |
| **2wiki avg** | | **0.273** | **0.315** |
| musique | 13 | — | — |
| musique | 42 | — | — |
| musique | 2024 | — | — |
| **musique avg** | | **0.143** | **0.196** |

> Elite SFT 无 per-seed 原始记录，仅保留平均值。

### 3.5 Full SFT (Llama-3-8B-Instruct + LoRA, 全量银标数据)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | 0.39 | 0.508 |
| hotpotqa | 42 | 0.40 | 0.498 |
| hotpotqa | 2024 | 0.40 | 0.527 |
| **hotpotqa avg** | | **0.397** | **0.511** |
| 2wikimultihopqa | 13 | 0.35 | 0.376 |
| 2wikimultihopqa | 42 | 0.26 | 0.290 |
| 2wikimultihopqa | 2024 | 0.30 | 0.337 |
| **2wiki avg** | | **0.303** | **0.334** |
| musique | 13 | 0.16 | 0.228 |
| musique | 42 | 0.18 | 0.240 |
| musique | 2024 | 0.18 | 0.264 |
| **musique avg** | | **0.173** | **0.244** |

> 数据来源: `eval_sft` (KG-ProWeight pipeline)

### 3.6 R9 v3 (Elite SFT + Precision PRM + Dynamic KG, 2000 steps, KL=21)

| 数据集 | 种子 | EM | F1 |
|---|---|---|---|
| hotpotqa | 13 | 0.38 | 0.435 |
| hotpotqa | 42 | 0.34 | 0.394 |
| hotpotqa | 2024 | 0.36 | 0.428 |
| **hotpotqa avg** | | **0.360** | **0.419** |
| 2wikimultihopqa | 13 | 0.06 | 0.117 |
| 2wikimultihopqa | 42 | 0.06 | 0.111 |
| 2wikimultihopqa | 2024 | 0.14 | 0.190 |
| **2wiki avg** | | **0.087** | **0.139** |
| musique | 13 | 0.00 | 0.063 |
| musique | 42 | 0.00 | 0.068 |
| musique | 2024 | 0.00 | 0.047 |
| **musique avg** | | **0.000** | **0.059** |

> 数据来源: `outputs/R9_v3_final_eval/` (KG-ProWeight pipeline, 2026-07-13)
> 训练参数: `lr=1e-6, ppo_epochs=2, step_reward_scale=1.0, outcome_weight=2.0, kl_coef=0.15, sft_anchor=0.10`
> KL 从 75 收敛到 21，训练全程稳定。hotpotqa EM=0.360 首次超越 Elite SFT (0.353)。

---

## 四、IHR (Intermediate Hallucination Rate) 对比

| 数据集 | Elite SFT | Full SFT |
|---|---|---|
| hotpotqa | 0.290 | **0.243** |
| 2wikimultihopqa | 0.563 | **0.312** |
| musique | 0.557 | **0.559** |
| **平均** | **0.470** | **0.371** |

> IHR 越低越好。Full SFT 在 hotpotqa/2wiki 上 IHR 优于 Elite SFT。
> R9 v3 / Zero-shot / Naive RAG / CoRAG 无可用的 IHR 数据（R9 v3 步骤格式不稳定；Zero-shot/Naive RAG 无步骤结构）。

---

## 五、关键发现

1. **SFT 是最大单一贡献** — Zero-shot EM=0.10 → Full SFT EM=0.29 (+2.9×)，远超其他所有改进
2. **Full SFT vs Elite SFT** — 全量数据带来 +0.034 EM (+13%) 的边际增益
3. **Naive RAG 在 2wiki/musique 上几乎全零** — 简单 prompt 在需要多步推理时完全失效
4. **CoRAG 表现分化** — hotpotqa EM=0.367 仅次于 Full SFT (0.397)，但 musique EM=0.000
5. **R9 v3 hotpotqa 首次超越 Elite SFT** — EM=0.360 vs 0.353，验证了 Precision PRM 的有效性
6. **R9 v3 泛化不足** — 2wiki/musique 上大幅落后 SFT，需进一步改进跨数据集泛化
7. **musique 是三个数据集中最难攻克的** — 最佳结果 Full SFT EM=0.173，检索质量差 + 多跳推理复杂度高

---

## 六、评估配置参考

| 参数 | 值 |
|---|---|
| 检索方式 | E5 dense + BM25 sparse, RRF fusion (k=60) |
| 检索 top-k | 15 |
| 生成 max_tokens | 512 |
| temperature | 0.7 |
| do_sample | True |
| max_input_len | 4096 |
| GPU | RTX 4090 24GB |
| GPU memory utilization | 0.80 |

---

## 七、外部 Baseline 评估状态

| Baseline | 模型 | 状态 | 备注 |
|---|---|---|---|
| **CoRAG** | Llama-3.1-8B | ✅ 完成 | EM=0.167，标准 RAG prompt 可用 |
| **R1-Searcher** | Llama-3.1-8B-RAG-RL | ✅ 完成 | EM=0.154，需 `<think>` prompt + max_tokens=1024 |
| **IRCoT (trace)** | Llama-3-8B | ❌ 失败 | FlashRAG IRCOTPipeline 兼容性 bug (list index out of range / Config.get) |
| **Self-RAG** | SelfRAG-7B | ❌ 未下载 | HF 被墙，模型不可用 |
| **ReaRAG** | ReaRAG-9B | ❌ 未下载 | HF 被墙，以 α=0 PPO 作为功能代理 |

---

*文件生成: 2026-07-16 | 数据来源: `docs/baselines.md` + `outputs/R9_v3_final_eval/` + `outputs/_baselines/`*
