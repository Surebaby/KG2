# KG-ProWeight 项目完整概述

> 最后更新：2026-07-09

---

## 一、项目简介

KG-ProWeight 是一个三阶段训练框架，在强化学习中引入外部知识图谱（Wikidata）作为逐步骤的事实性锚点。使用 Llama-3-8B-Instruct 作为基座模型，LoRA 微调。

### 三阶段

| 阶段 | 内容 | 产出 |
|---|---|---|
| Phase 1 | Teacher LLM 生成推理轨迹 + Wikidata 验证 | 25,000 条 silver 数据（含三值标签） |
| Phase 2 | 训练 α-gate + PRM | `alpha_gate.pt` |
| Phase 3a | SFT 监督微调 | Elite SFT / Full SFT checkpoint |
| Phase 3b | PPO 强化学习 | R8/R9 checkpoint |

---

## 二、奖励函数（R9 当前版本）

### 公式

```
R_step(t) = (α_t · R_KG(t) + (1-α_t) · R_text(t) · 0.3) × 5.0

if ValidTrajectory:
  R_outcome = 10.0 × EM(pred, gold)
else:
  R_outcome = -10.0

R_total = Σ R_step(t) + R_outcome
```

### 各分量

| 分量 | 值域 | 来源 | 说明 |
|---|---|---|---|
| `α_t` | 0~1 | α-gate（三特征分类器） | 动态加权 KG 与文本奖励。当前 ≈0.85 |
| `R_KG(t)` | +1/0/−1 | PRMAnnotator 对照 Wikidata 子图 | +1 验证通过，0 无三元组，−1 矛盾 |
| `R_text(t)` | −1~+1 | ReaRAG-9B 冻结模型 | 步骤文本连贯性评分 |
| `step_reward_scale` | 5.0 | 手动设定 | 放大中间步骤奖励以覆盖 KL 成本 |
| `outcome_weight` | 10.0 | 手动设定 | 正确答案奖励倍数 |
| `invalid_penalty` | −10.0 | 手动设定 | 非法轨迹惩罚 |

### ValidTrajectory 判定

```python
1. len(steps) >= min_valid_steps (1)
2. [Final Answer] 可提取
3. 步骤序号连续
4. 每步 raw_text 非空
5. 每步 Reasoning 内容 >= 20 chars
```

### α-gate

```
输入：
  graph_density    — 模型输出实体与 KG 子图的匹配度
  link_confidence  — 实体能查到 Wikidata QID 的比例
  semantic_entropy — token logprobs 的熵

输出：
  α ∈ [0, 1]      — 当前 ≈ 0.85
```

---

## 三、训练参数

### 模型参数

| 参数 | 值 |
|---|---|
| `base_model` | Llama-3-8B-Instruct |
| `dtype` | bf16 |
| `lora_r` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | q/k/v/o_proj |
| `max_input_length` | 6144 |
| `max_new_tokens` | 384 |

### PPO 参数

| 参数 | 值 | 说明 |
|---|---|---|
| `learning_rate` | 5e-6 | |
| `batch_size` | 8 | 每批 rollout |
| `mini_batch_size` | 1 | |
| `ppo_epochs` | 4 | |
| `cliprange` | 0.2 | |
| `kl_coef` | 0.15 | |
| `target_kl` | 8.0 | |
| `kl_horizon` | 2000 | |
| `gamma` | 0.95 | |
| `lam` | 0.95 | GAE λ |
| `max_grad_norm` | 1.0 | |
| `temperature` | 1.0 | |
| `top_p` | 1.0 | |
| `save_every_steps` | 500 | |

### 格式约束

| 参数 | 值 |
|---|---|
| `min_valid_steps` | 1 |
| `min_reasoning_chars` | 20 |
| `sft_anchor_weight` | 0.05 |
| `sft_anchor_interval` | 10 |
| `sft_replay_ratio` | 0.15 |

### KG 配置

| 组件 | 模式 | 数据量 |
|---|---|---|
| prompt 端 KG | 银标静态 | 43% 样本有 |
| reward 端实体链接 | `EntityLinker(offline=True)` | 35K 实体缓存 |
| reward 端子图获取 | `WikidataSubgraphRetriever(offline=True)` | 63K 子图缓存 |

---

## 四、项目结构

```
kgpaper/
├── kgproweight/              # 核心 Python 包
│   ├── config/               # YAML 加载 + pydantic schemas
│   ├── data/                 # prompts, parsers, silver dataset reader
│   ├── eval/                 # metrics, baselines, stats, IHR
│   ├── kg/                   # EntityLinker, WikidataRetriever, cache
│   ├── pipeline/             # FlashRAG pipeline 子类
│   ├── retrieval/            # hybrid RRF top-K
│   ├── reward/               # α-gate, PRM, text reward, IHR judge, composite reward
│   ├── training/             # phase1/2/3, reward_function, PPO trainer
│   └── utils/                # paths, seeds, logging, FlashRAG bootstrap
├── configs/                  # YAML 配置
│   ├── base.yaml
│   ├── training/             # phase1_silver, phase2_prm, phase3_sft, phase3_ppo
│   ├── eval/                 # 各 baseline 评估配置
│   ├── retrieval/            # hybrid RRF 检索配置
│   └── ablation/             # 消融实验配置
├── flashrag_src/             # FlashRAG 依赖（vendored）
├── scripts/
│   ├── train/                # phase1_generate_silver, phase2_train_prm, phase3_sft, phase3_ppo
│   ├── eval/                 # run_kg_proweight, run_baselines, run_ihr_judge
│   ├── prepare/              # corpus, indices, datasets, cache
│   └── deploy/               # AutoDL 部署脚本
├── data/                     # hotpotqa, 2wikimultihopqa, musique, silver_data
├── indexes/                  # e5_Flat.index, bm25/, entity_cache, kg_cache
├── checkpoints/              # SFT, PPO 各版本 checkpoint
├── models/                   # e5, llama3-8b (config files only)
├── docs/                     # paper, baselines, 实验日志, 分析文档
├── tests/                    # pytest
└── references/               # 参考文献 PDF
```

---

## 五、关键文件修改记录

### R9 动态 KG（2026-07-08）

| 文件 | 改动 |
|---|---|
| `kgproweight/training/reward_function.py` | `__call__` 中从模型输出提取实体 → 缓存查子图 → 替代 `spec.kg_subgraph`（+35 行） |
| `kgproweight/training/phase3_ppo.py` | 传入 `WikidataSubgraphRetriever(offline=True)` 到 reward_fn；EntityLinker 改为 `offline=True`；SFT replay 使用真实 spec |
| `kgproweight/reward/composite_reward.py` | `step_reward_scale` 参数 ×5.0；非法轨迹 penalty -10 |
| `configs/training/phase3_ppo.yaml` | `step_reward_scale: 5.0`；`learning_rate: 5e-6`；`kl_coef: 0.15` |

### 已知问题

1. **R_KG 恒为 0**：模型输出自然语言推理，不写三元组格式。银标数据 75% 有三元组但 SFT 没学到引用习惯
2. **步骤偏短**：`min_valid_steps=1` 使模型满足于 2 步，需要提升至 2-3
3. **代理不稳定**：Clash HTTP 代理不支持 SPARQL，搜索 API 也偶尔超时
4. **SPARQL 被封**：Wikidata 子图实时查询不可用，依赖 63K 缓存

---

## 六、训练命令

### 无代理（推荐，当前方案）

```bash
cd /root/autodl-tmp/kgpaper
export PYTHONPATH=/root/autodl-tmp/kgpaper:/root/autodl-tmp/kgpaper/flashrag_src
export KGPW_FLASHRAG_ROOT=/root/autodl-tmp/kgpaper/flashrag_src
export KGPW_PROJECT_ROOT=/root/autodl-tmp/kgpaper
export KGPW_DATA_DIR=/root/autodl-tmp/kgpaper/data
export KGPW_INDEX_DIR=/root/autodl-tmp/kgpaper/indexes

python scripts/train/phase3_ppo.py \
  --config configs/training/phase3_ppo.yaml \
  --sft_checkpoint checkpoints/sft_student_elite/final \
  --total_steps 500 \
  --output_dir outputs/r9_final
```

### 带代理（EntityLinker 走 Wikidata API）

```bash
export HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897
# 其余同上
```

### 评估

```bash
python scripts/eval/run_kg_proweight.py \
  --checkpoint checkpoints/r9_step504 \
  --datasets hotpotqa --seeds 13 --test_sample_num 100 \
  --save_root outputs/R9_eval --gpu_id 0
```

### IHR（LLM-as-Judge）

```bash
python scripts/eval/run_ihr_judge.py \
  --predictions <intermediate_data.json> \
  --judge_model deepseek-chat --sample 50
```

---

## 七、实验结果对比

| | Elite SFT | Full SFT | R7-B final | R8 final | **R9 step504** |
|---|---|---|---|---|---|
| α | — | — | 0.02 | 0.02 | **0.85** |
| n_steps | 3.0 | 3.2 | 1.0 | 1.2 | **2.0** |
| valid_rate | — | — | 70% | 78% | **98%** |
| reasoning_content | 100% | 100% | **0%** | 100% | **51%** |
| EM (hotpotqa) | 0.353 | 0.397 | 0.323 | 0.317 | 待测 |
| F1 (hotpotqa) | 0.456 | 0.511 | 0.407 | 0.424 | 待测 |
| IHR avg | 0.470 | 0.371 | — | 0.327 | 待测 |

---

## 八、下一步

1. **R9 step504 EM/F1 评估**（待跑）
2. **min_valid_steps: 1→2**：提升步骤数到 3+ 
3. **prompt 端注入 KG**：让模型在 prompt 中看到三元组，学会引用
4. **解决代理稳定性**：让 EntityLinker 在线查询 Wikidata
5. **论文完善**：Method / Experiments / Results 章节

---

*生成时间: 2026-07-09*
