# KG-ProWeight

> **Knowledge Graph-Anchored Process Rewards for Multi-Hop Retrieval-Augmented Generation**
>
> Three-phase training framework: Teacher Distillation → α-Gate → PPO with KG-verified process rewards.

Reference implementation for the KG-ProWeight paper. Provides an end-to-end pipeline for multi-hop RAG with knowledge-graph-grounded reinforcement learning.

---

## Quickstart

```bash
git clone <this-repo> kgpaper && cd kgpaper
pip install -e flashrag_src
pip install -e ".[dev]"
```

### Environment

```bash
export PYTHONPATH=$(pwd):$(pwd)/flashrag_src
export KGPW_PROJECT_ROOT=$(pwd)
export KGPW_DATA_DIR=$KGPW_PROJECT_ROOT/data
export KGPW_INDEX_DIR=$KGPW_PROJECT_ROOT/indexes
export KGPW_CHECKPOINT_DIR=$KGPW_PROJECT_ROOT/checkpoints
export KGPW_OUTPUT_DIR=$KGPW_PROJECT_ROOT/outputs
export KGPW_FLASHRAG_ROOT=$KGPW_PROJECT_ROOT/flashrag_src

# Model paths (auto-detected from project_root/models/ or env vars)
export KGPW_LLAMA3_PATH=/path/to/llama3-8b
export KGPW_REARAG_PATH=/path/to/rearag-9b
```

Model paths are resolved automatically: checks `$PROJECT_ROOT/models/`, then `/root/autodl-tmp/models/`, then HuggingFace.

---

## Training

### Phase 1: Generate Silver Data (Teacher LLM → Wikidata-verified trajectories)

```bash
python scripts/train/phase1_generate_silver.py \
  --config configs/training/phase1_silver.yaml \
  --max_queries 25000 \
  --teacher_model deepseek-chat
```

Produces `data/silver_data/silver_trajectories.jsonl` with three-valued labels (+1/-1/0).

### Phase 2: Train α-Gate and PRM

```bash
python scripts/train/phase2_train_prm.py \
  --config configs/training/phase2_prm.yaml \
  --silver_path data/silver_data/silver_trajectories.jsonl
```

Trains process reward model and `alpha_gate.pt` in `checkpoints/prm_alpha_gate/`.

### Phase 3a: Supervised Fine-Tuning (SFT)

```bash
python scripts/train/phase3_sft.py \
  --config configs/training/phase3_sft.yaml \
  --silver_path data/silver_data/silver_trajectories.jsonl \
  --output_dir checkpoints/sft_student
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_model` | llama3-8B-instruct | |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | |
| `learning_rate` | 2e-4 | |
| `num_epochs` | 3 | |
| `batch_size` | 4 | per-device |
| `max_seq_length` | 4096 | |
| `grad_accum` | 4 | effective batch = 16 |

**Elite SFT variant** (2,000 quality-filtered samples): `checkpoints/sft_student_elite/final`

**Full SFT** (all 9,839 silver samples): `checkpoints/sft_student/final`

### Phase 3b: PPO with KG-Anchored Process Reward

```bash
python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --sft_checkpoint checkpoints/sft_student_elite/final \
  --output_dir outputs/r8_experiment
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_model` | llama3-8B-instruct | |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | |
| `learning_rate` | 1e-5 | PPO learning rate |
| `batch_size` | 8 | Rollout batch size |
| `mini_batch_size` | 1 | PPO mini-batch |
| `ppo_epochs` | 4 | Epochs per batch |
| `kl_coef` | 0.1 | Initial KL penalty |
| `target_kl` | 8.0 | Adaptive KL controller target |
| `kl_horizon` | 2000.0 | KL controller horizon |
| `gamma` | 0.95 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `cliprange` | 0.2 | PPO clip |
| `max_grad_norm` | 1.0 | |
| `total_ppo_steps` | 2000 | Total trajectories seen |
| `save_every_steps` | 500 | Checkpoint interval |
| `max_new_tokens` | 384 | Generation length |
| `temperature` | 1.0 | Rollout sampling |
| `top_p` | 1.0 | No truncation (must match TRL) |
| `max_input_length` | 6144 | Prompt truncation |

**Reward parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `outcome_weight` | 10.0 | EM bonus for correct answer |
| `text_reward_scale` | 0.3 | Scale down ReaRAG text reward noise |
| `min_valid_steps` | 1 | Min steps for trajectory validity |
| `min_reasoning_chars` | 20 | Content-aware gate (R8) |
| `sft_anchor_weight` | 0.05 | SFT anchor loss weight |
| `sft_anchor_interval` | 10 | Anchor every N PPO steps |
| `sft_replay_ratio` | 0.15 | SFT prompts in PPO batch (R8) |

**Reward formula:**
```
R_t = α_t · R_KG(t) + (1-α_t) · R_text(t) · text_reward_scale

Last step bonus (conditional on ValidTrajectory):
  + outcome_weight × EM(pred, gold)  if trajectory valid
  + 0                                 otherwise
```

**ValidTrajectory criteria (R8):**
- ≥ `min_valid_steps` parsed `[Step N]` blocks
- Extractable `[Final Answer]`
- Sequential step indices
- Non-empty text per step
- **Reasoning content ≥ `min_reasoning_chars` characters per step** (content gate)

---

## Evaluation

```bash
# Single dataset × seed
python scripts/eval/run_kg_proweight.py \
  --checkpoint checkpoints/kg_proweight_R7B/final \
  --datasets hotpotqa --seeds 42 --test_sample_num 100 \
  --save_root outputs/eval --gpu_id 0

# IHR (LLM-as-Judge)
python scripts/eval/run_ihr_judge.py \
  --predictions outputs/eval/<run>/intermediate_data.json \
  --judge_model deepseek-chat --sample 200
```

See `docs/baselines_final.md` for full baseline comparison.

## R9 v5 Results (July 2026)

| Baseline | EM avg | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|---|
| Full SFT | 0.291 | 0.397 | 0.303 | 0.173 |
| Elite SFT | 0.257 | 0.353 | 0.273 | 0.143 |
| **R9 v5 (500 steps)** | **0.240** | 0.34 | 0.25 | 0.13 |
| CoRAG | 0.167 | 0.367 | 0.133 | 0.000 |
| R1-Searcher | 0.154 | 0.310 | 0.143 | 0.010 |
| Zero-shot | 0.103 | 0.203 | 0.080 | 0.027 |
| Naive RAG | 0.061 | 0.177 | 0.007 | 0.000 |

Key R9 v5 innovations:
- **Precision × Relevance**: KG reward filters irrelevant triples via lexical evidence overlap
- **outcome_weight=10.0**: Restores answer correctness as primary training signal
- **step_reward_scale=0.3**: Prevents citation reward from dominating outcome
- **Dynamic KG Cache**: 8493 pre-built Q→KG entries, 100% hit rate

---

## Monitoring (TensorBoard)

```bash
tensorboard --logdir outputs/<run>/tensorboard --port 6006 --bind_all
```

Key metrics:

| Panel | Metric | Healthy range |
|-------|--------|---------------|
| `custom/mean_reward` | Average trajectory reward | 1.0–8.0 |
| `custom/valid_rate` | Trajectory validity rate | 40–80% |
| `custom/kl_divergence` | KL divergence | 0.5–20 |
| `custom/clip_fraction` | PPO clip fraction | 0.05–0.30 |
| `custom/sft_anchor_loss` | SFT anchor CE loss | 1–10 |
| `r8/reasoning_content_rate` | Steps with content | > 60% |
| `r8/step_rate` | Step-structured outputs | > 70% |

---

## Project Structure

```
kgpaper/
├── kgproweight/              # Core Python package
│   ├── config/               # YAML loader + schemas
│   ├── data/                 # prompts, parsers, silver dataset
│   ├── eval/                 # metrics, baselines, stats, IHR
│   ├── kg/                   # entity linker, Wikidata retriever, cache
│   ├── pipeline/             # FlashRAG pipeline subclasses
│   ├── retrieval/            # hybrid RRF top-K
│   ├── reward/               # α-gate, PRM, text reward, IHR judge
│   ├── training/             # phase1/2/3, reward function, PPO trainer
│   └── utils/                # paths, seeds, logging
├── configs/                  # YAML configs (training, eval, ablation)
├── flashrag_src/             # FlashRAG dependency (vendored)
├── scripts/                  # CLI entry points
│   ├── train/                # phase1_silver, phase2_prm, phase3_sft, phase3_ppo
│   ├── eval/                 # run_kg_proweight, run_baselines, run_ihr_judge
│   ├── prepare/              # corpus, indices, datasets, cache
│   └── deploy/               # AutoDL deployment helpers
├── docs/                     # Paper, baselines, architecture, logs
├── tests/                    # pytest suite
└── references/               # Related papers (PDF)
```

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
