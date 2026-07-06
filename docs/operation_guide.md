# KG-ProWeight 实验操作指南（中文 · Python 命令版）

> 目标硬件：**NVIDIA RTX PRO 6000 Blackwell 96GB（bf16）**。  
> 本文是从零复现实验的标准流程，所有步骤均使用 `python` 命令（不使用 `make` / `.sh`）。

---

## 0. 一次性环境准备

```bash
# 进入项目并激活环境
cd /home/ai/flashrag/kgpaper
conda activate kgpw
source .env

# 基础检查（GPU + bf16）
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# 期望输出：NVIDIA RTX PRO 6000 ... True

# 单元测试（建议）
python -m pytest -q
```

### 0.1 必要目录与关键产物

若你从别的机器迁移索引，至少保证 `$KGPW_INDEX_DIR` 下存在：

- `indexes/corpus_flashrag.jsonl`
- `indexes/e5_Flat.index`
- `indexes/bm25/`
- `indexes/kg_cache/` 与 `indexes/entity_cache.jsonl`

---

## 1. 构建语料与索引（Step 0–3）

> 注意：`00_convert_corpus.py` 必须显式传 `--input`。

### 1.1 语料格式转换（raw wiki jsonl -> FlashRAG jsonl）

```bash
python scripts/prepare/00_convert_corpus.py \
  --input /path/to/your/raw_wikipedia.jsonl \
  --output "$KGPW_INDEX_DIR/corpus_flashrag.jsonl"
```

输入格式（每行）：
`{"id": int, "title": "...", "text": "..."}`  
输出格式（每行）：
`{"id": "str", "contents": "title\ntext"}`

### 1.2 构建稠密索引（E5 + FAISS Flat）

```bash
CUDA_VISIBLE_DEVICES=0 python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path "$KGPW_E5_PATH" \
  --corpus_path "$KGPW_INDEX_DIR/corpus_flashrag.jsonl" \
  --save_dir "$KGPW_INDEX_DIR" \
  --use_fp16 \
  --max_length 512 \
  --batch_size 1024 \
  --pooling_method mean \
  --faiss_type Flat
```

### 1.3 构建 BM25 稀疏索引

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path "$KGPW_INDEX_DIR/corpus_flashrag.jsonl" \
  --bm25_backend bm25s \
  --save_dir "$KGPW_INDEX_DIR"
```

### 1.4 下载数据集（HotpotQA / 2Wiki / MuSiQue）

```bash
python scripts/prepare/03_download_datasets.py \
  --datasets hotpotqa 2wikimultihopqa musique \
  --splits train dev
```

---

## 2. Phase 1：银标轨迹蒸馏（Silver Trajectories）

目标：生成 >= 15,000 条可用轨迹（通常从 ~25,000 次 teacher 采样筛选）。

```bash
export DEEPSEEK_API_KEY=sk-...

python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml
```

可选：更细粒度控制

```bash
python scripts/train/phase1_generate_silver.py \
  --dataset hotpotqa \
  --split train \
  --max_queries 25000 \
  --teacher deepseek-v4-flash \
  --max_workers 8 \
  --resume
```

主要输出：

- `data/silver_data/silver_trajectories.jsonl`
- `indexes/entity_cache.jsonl`
- `indexes/kg_cache/kg_subgraph_cache.jsonl`

---

## 3. Phase 2：PRM + α-Gate 训练

```bash
python scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm.yaml \
  --silver_data data/silver_data/silver_trajectories.jsonl
```

默认设置（Pro 6000）：

- 基座：Llama-3-8B（bf16）
- LoRA：`r=32`
- batch：`8`，grad accumulation：`2`（有效 batch=16）
- 训练约 3 小时

主要输出：

- `checkpoints/prm_alpha_gate/best_checkpoint/`
- `data/silver_data/silver_with_logprobs.jsonl`（训练前一遍 forward 生成）

---

## 4. Phase 3a：SFT（PPO 前必须完成）

```bash
python scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft.yaml
```

默认：1 epoch，lr=2e-5，batch=8，约 2 小时。  
主要输出：`checkpoints/sft_student/final/`

---

## 5. Phase 3b：PPO + GAE + Critic（默认主路径）

```bash
python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --student_model "$KGPW_LLAMA3_PATH" \
  --sft_checkpoint checkpoints/sft_student/final \
  --prm_checkpoint checkpoints/prm_alpha_gate/best_checkpoint \
  --text_reward_backend rearag
```

默认（96GB，无量化）：

- batch 64 / mini-batch 8 / ppo_epochs 4
- 约 5000 steps，耗时约 10 小时
- 输出：`checkpoints/kg_proweight_final/final/`

### 5.1 显存受限（24GB 卡）回退 GRPO

```bash
python scripts/train/phase3_grpo.py \
  --config configs/training/phase3_grpo.yaml
```

---

## 6. 构建 D_dropout 鲁棒性数据集（建议在评测前完成）

```bash
python scripts/prepare/05_build_d_dropout.py \
  --sample_size 1000
```

输出：`data/d_dropout/dev.jsonl`  
每条样本包含 `metadata.dropout.modified_kg`。

---

## 7. 基线评测（统一 RRF top-50）


##环境设置

cd /home/ai/flashrag/kgpaper
# 2) 激活 conda 环境（必须用 kgpw，不要用 base）
conda activate kgpw
# 3) 加载项目环境变量
source .env
# 4) 修正 FlashRAG 路径（你机器上 third_party/FlashRAG 不存在，需指向实际目录）
export KGPW_FLASHRAG_ROOT=/home/ai/flashrag/flashrag/FlashRAG-main
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -c "from kgproweight.utils.paths import data_dir,index_dir; print('data',data_dir()); print('index',index_dir())"


```bash
python scripts/eval/run_baselines.py \
  --datasets hotpotqa 2wikimultihopqa musique \
  --split dev \
  --gpu_id 0 \
  --seeds 13 42 2024
```

如需单方法调试：

```bash
python scripts/eval/run_baselines.py \
  --methods rearag \
  --datasets hotpotqa \
  --split dev \
  --gpu_id 0
```

说明：

- 所有 baseline 共用同一套检索配置 `configs/retrieval/hybrid_rrf_top50.yaml`
- baseline 注册表以 `kgproweight/eval/baselines.py` 为准

---

## 7b. 长时间评测前的快速自检（推荐）

CPU 快速检查：

```bash
python scripts/eval/sanity_check.py
```

可选 GPU 小样本冒烟（200 条）：

```bash
python scripts/eval/sanity_check.py \
  --run-eval \
  --kg2-fallback \
  --test_sample_num 200
```

---

## 8. 评测 KG-ProWeight 主模型

可选：先预热 Wikidata 缓存（离线评测更稳）

```bash
python scripts/prepare/04_prewarm_wikidata_cache.py \
  --datasets hotpotqa 2wikimultihopqa musique d_dropout \
  --split dev
```

主评测：

```bash
python scripts/eval/run_kg_proweight.py \
  --checkpoint checkpoints/kg_proweight_final/final \
  --datasets hotpotqa 2wikimultihopqa musique d_dropout \
  --split dev \
  --gpu_id 0 \
  --seeds 13 42 2024
```

输出位于：`outputs/kg_proweight/<dataset>/`，包含 EM/F1、逐步 α 与启发式 IHR 等。

---

## 9. 消融实验（论文 §7）

全量消融：

```bash
python scripts/eval/run_ablations.py \
  --datasets hotpotqa 2wikimultihopqa musique \
  --gpu_id 0 \
  --seeds 13 42 2024
```

只跑指定变体：

```bash
python scripts/eval/run_ablations.py \
  --variants alpha_zero alpha_one alpha_half binary_labels single_retriever
```

---

## 10. Rigour 扩展实验

> 注意：IHR（GPT-4o 评审）不在 Step 7/8 的默认指标中，需单独运行。

```bash
python scripts/eval/run_ihr_judge.py \
  --datasets hotpotqa 2wikimultihopqa musique \
  --split dev
```

```bash
python scripts/eval/run_data_efficiency.py \
  --sizes 1000 2000 5000 10000 15000
```

```bash
python scripts/eval/run_variance_validation.py \
  --max_steps 500
```

输出目录：`outputs/rigor/<name>/summary.json` 与 `plot.png`

---

## 11. 汇总论文表格与图

```bash
python scripts/eval/summarize_results.py \
  --datasets hotpotqa 2wikimultihopqa musique d_dropout
```

主要产物：

- `outputs/summary/table1_main_results.md`
- `outputs/summary/table2_ablations.md`
- `outputs/summary/alpha_distribution.md`
- `outputs/summary/significance.md`

---

## 12. 常见问题排查

| 现象 | 解决方式 |
|---|---|
| PPO OOM | 降低 `mini_batch_size`（如 8 -> 4），或改用 `phase3_grpo.py` |
| Wikidata SPARQL 429 | 先运行 `04_prewarm_wikidata_cache.py`，并降低并发 |
| HuggingFace 下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `KGPW_PROJECT_ROOT` 未设置 | 重新 `source .env` |
| FlashRAG 导入失败 | `pip install -e third_party/FlashRAG` |
| `prepare-corpus` 报 `--input` 缺失 | 改为手动执行 `00_convert_corpus.py --input ...` |

---

## 13. 端到端耗时参考（Pro 6000 96GB）

| 阶段 | 预估时长（小时） | GPU | 主要输出 |
|---|---:|---|---|
| Step 0–3（数据+索引） | 8 | 1 | `indexes/`, `data/` |
| Phase 1（silver） | 6 | 0 | `silver_trajectories.jsonl` |
| Phase 2（PRM） | 3 | 1 | `prm_alpha_gate/` |
| Phase 3a（SFT） | 2 | 1 | `sft_student/` |
| Phase 3b（PPO） | 10 | 1 | `kg_proweight_final/` |
| Baselines（3 数据集） | 24 | 1 | `outputs/baselines/` |
| KG-ProWeight（4 数据集） | 6 | 1 | `outputs/kg_proweight/` |
| 消融（5 组） | 8 | 1 | `outputs/ablation/` |
| Rigour（IHR/效率/方差） | 12 | 1 | `outputs/rigor/` |
| **总计** | **~80** | — | 完整论文实验产物 |
