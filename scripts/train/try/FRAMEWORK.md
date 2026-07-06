# KG-ProWeight `try` 变体 — 全流程框架与修复总览

> 本文档梳理 `scripts/train/try/` 下的全部实验性实现:它们在**不修改包内
> `kgproweight/` 任何文件**的前提下,复用包内逻辑、只覆盖有缺陷的部分,把
> Phase 1 → 2 → 3a → 3b 整条流水线在单张 24GB 卡(4-bit)上跑通。
>
> 隔离原则:`try` 是 Python 关键字,本目录**不是可导入包**;各 CLI 用
> `sys.path.insert` 把本目录加入路径后平铺导入兄弟模块。以脚本方式运行,勿用 `-m`。

---

## 0. 为什么有 try 变体

包内 `kgproweight/` 是论文的"正式实现",但有两类问题需要在不动它的前提下迭代:

1. **正确性缺陷**:Phase 1 标注器有系统性误判;Phase 3b PPO 有 4 处与论文设计
   (`docs/paper_design.md`,source-of-truth)不符,其中一处(per-step 奖励)是论文
   核心机制 Theorem 2 的代码基础。
2. **显存约束**:包内全部用 bf16,8B 模型在 24GB 卡上 OOM。try 变体加 4-bit QLoRA
   让全流程能在本机(RTX 4090 24GB)冒烟;正式训练仍在 Pro6000 96GB 上用 bf16。

---

## 1. 整体数据流

```
HotpotQA train
   │
   ▼  Phase 1 (try): 蒸馏 + 改进标注
phase1_generate_silver_try.py
   → silver_try_*.jsonl  (每条:question/steps[label±1/0]/cited_triples/
                          kg_subgraph/retrieved_passages/answer/metadata{gold_answer})
   │   ← 同一批 silver 贯穿以下全部
   ▼  Phase 2 (try, 4-bit): PRM head + α-Gate 联合训练
phase2_prm_try.py
   → alpha_gate.pt + prm_head/ + text_reward_head.pt + silver_with_logprobs.jsonl
   │
   ▼  Phase 3a (try, 4-bit + merge): SFT 学格式
phase3_sft_try.py --merge_output
   → final/ (LoRA adapter) + merged/ (完整 bf16 模型,供 PPO 直接加载)
   │
   ▼  Phase 3b (try, 4-bit): PPO,per-step 奖励进 GAE
phase3_ppo_try.py  (StepRewardPPOTrainer + ImprovedRewardFunction)
   → final/ (PPO 后的 LoRA adapter) + history.jsonl
   │
   ▼
评估 / 消融 / Theorem-2 方差验证(尚未在 try 下重写)
```

`run_pipeline_smoke.sh` 把 Phase 2→3a→3b 串成一条龙(连通性冒烟用)。

### 1.1 目录结构(按用途分类)

```
scripts/train/try/
├── shared/            prm_annotator_try.py            ← 跨 Phase 1/3 共享的标注器
├── phase1_distill/    phase1_generate_silver_try.py (CLI 入口)
│                      phase1_distill_try.py / distill_helpers_try.py
├── phase2_prm/        phase2_prm_try.py
├── phase3_sft/        phase3_sft_try.py
├── phase3_ppo/        phase3_ppo_try.py (入口) / ppo_reward_try.py / ppo_trainer_try.py
├── tools/             dump_label_record.py / reannotate_compare.py   ← 离线分析
├── tests/             test_distill_offline.py / test_ppo_offline.py / smoke_ppo_tiny.py
├── outputs/           所有产物(silver_try_*.jsonl / pipeline_smoke/ / label_record_*)
├── *.md               README.md / README_ppo.md / 本文件
└── run_pipeline_smoke.sh
```

**导入机制**:每个可执行脚本顶部把 try 根 + 各 phase 子目录 + shared 一起插入
`sys.path`,所以平铺导入(`from prm_annotator_try import ...`)在分目录后仍然有效。
照常以脚本运行(`python scripts/train/try/<sub>/<file>.py`),勿用 `-m`。
本文档下文提到文件时只写裸文件名,按上表去对应 phase 目录找。


---

## 2. Phase 1 — 改进的轨迹蒸馏

**目标**:把接受率从基线 7.6% 提上来,并修正标注器的系统性误判。

**文件**:
- `phase1_generate_silver_try.py` — CLI 入口 + 冷缓存守卫
- `phase1_distill_try.py` — 编排(`run_phase1`):复用包内 Teacher/检索器,重写
  `_process_one` + 接受循环
- `distill_helpers_try.py` — 改动逻辑:宽松答案匹配、稳健 mention 提取、分层过滤器
- `prm_annotator_try.py` — **改进版 3 分类标注器**(见 §2.1)
- `dump_label_record.py` — 把任意 silver JSONL 导出成逐轨迹标签记录(`--full` 出全文)
- `reannotate_compare.py` / `test_distill_offline.py` — 离线对比 / 单测

**5 处改进**(详见 `README.md`):宽松答案匹配(score≥0.3 即接受)、分层 KG 密度接受
(kg_rich/medium/sparse 配额,不再硬砍低 triple_rate)、稳健 mention 提取(passage
标题锚点)、SPARQL 优雅降级 + 预热守卫、格式重试保留。接受率提到 ~60-69%。

### 2.1 标注器的两个关键修复(ImprovedPRMAnnotator)

原版 `PRMAnnotator` 在实跑中退化成二分类(0% NEUTRAL),且有两类系统性错误。
`prm_annotator_try.py` 修正:

| 问题 | 现象 | 修复 |
|---|---|---|
| **凑数 +1** | 教师挂一条子图里存在但与结论无关的 triple,仍判 +1(实测 24%) | `require_triple_relevance`:验证通过的引用还需至少一条 triple 的实体与结论词面相关,否则降为 0 |
| **-1 误报** | `_contradicts` 把"KG 无此信息/无法确定/X 不是 Y"这类**诚实弃权**判成矛盾(实测 11/16) | `_ABSTENTION_RE` 弃权护栏(`_is_contradiction` 包装),只有真正的反断言才 -1 |

failed 尝试记录(避免重蹈):曾试图用"主语实体重叠"加固矛盾判定,但 `ENTITY_RE`
提取的是短语片段("Rock"/"May")而非语法主语,既没修掉误报又误伤真负样本,已撤销。

**效果**(80 条实测):凑数 +1 24%→2%,-1 误报 11→0,标签分布从二值塌缩恢复成
真三分类(+1:40% / 0:54% / -1:6%)。

### 2.2 已知数据特性(非 bug)

- 0/NEUTRAL 占比偏高(accepted ~48-60%):HotpotQA 上真正"KG 强支撑"的步本就少,
  多数推理靠 passage + 世界知识。写论文解读 α 分布时需说明。
- -1 类很薄(~3-6%):α 的负向极端监督弱,α→0 区域主要靠 0 + kg_sparse 桶支撑。
- ~30% accepted 轨迹零 +1(纯文本/世界知识):PPO 里这些轨迹奖励由 R_Text 主导。

---

## 3. Phase 2 — PRM head + α-Gate 联合训练 (4-bit)

**文件**:`phase2_prm_try.py`

**做法**:不重写包内 230 行的 `run_phase2` 训练循环。改为**运行时 monkeypatch**
包内两个 bf16 加载函数为 4-bit NF4,再调用未改动的 `run_phase2`:
- `phase2_prm.compute_step_logprobs` → 4-bit 版(logprob 预扫)
- `phase2_prm._build_base_model` → 4-bit + `prepare_model_for_kbit_training`(PRM 训练)

两者都是模块级、按裸名调用,patch 生效。

**产出**:`alpha_gate.pt`、`prm_head/`(LoRA adapter + prm_head.pt)、
`text_reward_head.pt`、`silver_with_logprobs.jsonl`(回填了每步 logprob)。

**α-Gate**(贯穿 2/3b/推理的核心):`α_t = σ((W·x_t + b)/τ)`,
`x_t = [图密度, 链接置信度, 语义熵]`。初始 W=(1.0,1.5,-0.8)、b=-2.0。
密度↑置信↑→α→1(信 KG),熵↑→α→0(信文本)。Phase 2 学它的权重。

**修复的历史 bug**:旧版把语义熵硬编码 0.5;现在经 logprob 预扫用真实 token logprob 算熵。

---

## 4. Phase 3a — SFT 学格式 (4-bit + merge)

**文件**:`phase3_sft_try.py`

**为什么需要**:PPO 前先让模型学会 `[Step N]…[Final Answer]` 格式,否则 PPO rollout
产不出可解析轨迹,奖励无从施加。只用 accepted 轨迹,且**丢弃 label=-1 的步**(不教幻觉)。

**try 改动**:包内 `phase3_sft.py` 用 bf16 + max_length 4096,24GB OOM。try 版加:
- 4-bit NF4 QLoRA + `prepare_model_for_kbit_training` + gradient checkpointing
- CLI 可调 `--max_length`(默认 1024)
- 复用包内 `_build_dataset`/`_tokenise`/`_render_assistant_trace`
- **`--merge_output`**:训练后用 bf16 重载基座 + 套 adapter + `merge_and_unload`,
  存成 `merged/` 完整模型

**关键衔接修复**:SFT 产出默认是纯 LoRA adapter(只有 adapter_config.json,无
config.json),PPO 的 `from_pretrained` 无法直接加载。`--merge_output` 产出完整模型,
喂给 PPO 的 `--sft_checkpoint`。注意:不在 4-bit 模型上 merge(有损),而是 bf16 重载基座再合并。

---

## 5. Phase 3b — PPO,per-step 奖励进 GAE (4-bit)

**文件**:`ppo_reward_try.py`(reward fn)、`ppo_trainer_try.py`(trainer)、
`phase3_ppo_try.py`(入口)。详见 `README_ppo.md`。

**4 处修复 + 1 显存优化**(对照包内 `phase3_ppo.py` 的缺陷):

| 编号 | 缺陷 | 修复 |
|---|---|---|
| **P0-1**(论文级) | per-step 奖励被 `.sum()` 成标量,TRL 0.11.4 只放最后一个 token,GAE 退化成 outcome-only → Theorem 2 失去支撑 | `StepRewardPPOTrainer` 只 override `compute_rewards`,把每步 R_total 铺到该步 [Step N] 末 token;GAE/clip/minibatch/KL 全部复用 TRL |
| **P0-2** | R_KG 用原版标注器(凑数/误报回流→reward hacking) | 换 `ImprovedPRMAnnotator` |
| **P0-3** | outcome EM 比教师答案 `traj.answer` | 比 `metadata["gold_answer"]` 真 gold |
| **P1-1** | entropy 恒 1.0 | `generate(output_scores=True)` 取真实 per-step logprob |
| **P1-2** | 独立加载第二个全量 8B ref | LoRA 时 `ref_model=None`(TRL 用禁 adapter 的 policy 当 ref);原 `create_reference_model` 复制全量 base 是 OOM 根因 |

**复合奖励**:`R_total(t) = α_t·R_KG(t) + (1-α_t)·R_Text(t)`,outcome EM 加在最后一步。
已核实 R_KG∈{-1,0,+1} 与 R_text(tanh 到 [-1,1])量纲一致,无需归一化。

**核心设计**(P0-1):TRL `step()` 内部 `compute_rewards → compute_advantages(GAE) →
train_minibatch`。只 override `compute_rewards`,通过 side-channel
(`set_pending_step_rewards`)注入 per-token 奖励张量,KL penalty 仍按 TRL 逐 token 算。
**耦合提示**:override 复制了 TRL 0.11.4 的 `_kl_penalty`/`kl_ctl`,升级 TRL 需重新同步。

---

## 6. 测试与冒烟

| 文件/产物 | 作用 | 状态 |
|---|---|---|
| `test_distill_offline.py` | Phase 1 蒸馏离线单测 | ✅ |
| `test_ppo_offline.py` | PPO reward + compute_rewards override 离线单测(mock) | ✅ 4 测全过 |
| `smoke_ppo_tiny.py` | tiny 模型验证 override 与真实 TRL step() 对接(CPU) | ✅ |
| 真实 8B 4-bit PPO 冒烟 | 端到端 EXIT=0 | ✅ |
| `run_pipeline_smoke.sh` | Phase 2→3a→3b 一条龙(80 条) | ✅ EXIT=0 |

**全流程冒烟结果**(80 条,`outputs/pipeline_smoke/`):三阶段端到端跑通,manifest 确认
Phase 3b 加载的是本次 Phase 2/3a 的产出(数据流真贯通)。**α-gate 权重训完≈初始值
(80 条学不到东西)——本次只验证连通性,不验证效果**。

---

## 7. 显存与运行环境

- 本机 RTX 4090 **24GB**(共享,常被他人 openvla_oft 进程占 ~16GB;跑前必查
  `nvidia-smi --query-compute-apps`)。
- 8B 的任何 bf16 训练都**装不下 24GB**;必须 4-bit + 小 mini_batch + 短 max_length。
  且无法与占 16GB 的进程共存,只能趁 GPU 空闲独占跑。
- 运行环境:conda `kgpw`(python 3.10),TRL 0.11.4(PPOConfig/PPOTrainer 弃用警告无害),
  transformers 4.49,bitsandbytes 0.49.2。
- 跑前:`source .env && export KGPW_FLASHRAG_ROOT=...`,以及
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## 8. 距离正式实验还差什么

1. **扩数据**:用最终 ImprovedPRMAnnotator 生成中/大规模 silver(现仅 50-80 条)。
2. **上 Pro6000 96GB**:bf16(`--no_4bit` 或直接用包内版本)、真实 ReaRAG-9B 或
   llama_head text reward、完整 batch_size/total_steps。
3. **效果验证**:几千条 + 多 epoch SFT + 数百 PPO step 才能看到非随机的 EM/F1 信号。
4. **eval / 消融 / Theorem-2 方差脚本**:尚未在 try 下重写,正式实验时对照包内 `eval/`。
