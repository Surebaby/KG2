# KG-ProWeight 重训与实验整改方案

> 生成日期：2026-08-21
> 目的：针对"KG 定位 / 银标 KG 有效率低 / IHR 口径 / PPO 重训 / 全量重评 / 训练与奖励参数"六类问题，给出待处理清单 + 可执行的详细方案。
> 依据：`RESEARCH_WORKFLOW.md`、`docs/paper/v2_review.md`、`docs/paper/口径统一清单.md`、当前代码（`kgproweight/` + `configs/training/phase3_ppo.yaml`）、训练产物（`checkpoints/*/manifest.json` + `history.jsonl`）、银标数据、KG 索引。

---

## 0. 本次核实中新发现的问题（必须先读，直接改变重训方案）

以下五条是本轮重新核代码/数据时发现的、**现有文档没记录**的事实。它们各自都能让一次重训白跑，所以排在方案最前面。

### 0.1 ⚠️ 训练期 KG 索引命中率 = 0%（训练/推理 KG 分布并未真正对齐）

- `question_kg_index_v2_full.json` 的 22,393 条**全部是 dev 前缀**（`dev_*`），是按**评估集**构建的。
- PPO 训练用银标 `silver_v1_reannotated.jsonl`，其 qid 前缀是 **`train_*`**。
- 实测：用 4,000 条 accepted 训练题去查该索引，**命中率 0/4000 = 0%**。
- 后果：PPO 每次构建 prompt 时 `question_kg_index.get(traj.question)` 永远 miss，**永远走 fallback 分支** —— 对银标自带的 `kg_subgraph`（已按 `max_keep=50` 重标注，均值 41.2 / 中位 50 条）再做一次 `filter_and_rank_triples(max_keep=ppo_max_kg_triples)`。

| 阶段 | KG 来源 | 三元组预算 |
|------|---------|-----------|
| PPO①（`ppo_r10_split`）训练 | 银标 `kg_subgraph` → `filter_and_rank_triples(max_keep=30)` | **30** |
| PPO②（`kg_proweight_ppo_v2`）训练 | 同上，`max_keep=12` | **12** |
| 评估（`kg_proweight_pipeline.py`） | `question_kg_index_v2.json`（dev 索引，`min_keep=5` 构建）→ `pipeline` 再切片 | **12** |

- **结论**：所谓"训练/推理 KG 分布对齐"只在"都过了 `filter_and_rank_triples`"这一层成立，但**训练侧走的是银标原始子图过滤、推理侧走的是 dev 索引过滤**，两个子图的来源、`min_keep`、`max_keep` 都不同。PPO① 用 30、PPO② 用 12，而推理用 12——**PPO① 其实是在训练期喂了比推理多 2.5 倍的 KG**。
- **影响**：这是除 Plan B 权重外，第二个训练/推理错位。`v2_review.md` 把 PPO② 的暴跌全部归因于 Plan B，但 `ppo_max_kg_triples 30→12` 是同期引入的另一个混杂变量。

### 0.2 ⚠️ Teacher 模型身份：数据记录是 `deepseek-chat`，不是文档写的 `deepseek-v4-flash`

- `silver_v1_reannotated.jsonl` 全部 24,998 条的 `teacher_model` 字段都是 **`deepseek-chat`**。
- 文档（`口径统一清单` §④、`RESEARCH_WORKFLOW.md`）统一写 `deepseek-v4-flash`；`phase1_generate_silver.py` 的 `--teacher` 默认值是 `deepseek-v4-flash`，但 `phase1_distill.py:295` 的类默认和 `configs/base.yaml` 都是 `deepseek-chat`。
- **结论**：**实际生成银标用的是 `deepseek-chat`**。论文写 teacher 身份时必须按数据记录写 `deepseek-chat`（或先确认 `deepseek-v4-flash` 与 `deepseek-chat` 是否为同一 API 的两个别名——这需要你向 DeepSeek 侧确认，不能猜）。

### 0.3 ⚠️ V6 银标（含 2wiki/musique 的那批）有双重污染：dev 泄漏 + 冒烟语料检索

- `silver_v6_full_20260801_0136.jsonl`（22,398 条）及所有 `*_20260801_*.jsonl` 的 qid 前缀是 **`dev_*`** —— 是**评估集**生成的（日志开头就有 `EVAL-SPLIT LEAKAGE ACKNOWLEDGED: --split 'dev'` 警告）。
- 且这批的 `retrieved_passages` 的 passage id 全部 `< 989`，即检索自 `indexes_smoke/corpus_flashrag.jsonl`（**989 文档的冒烟语料**），不是 wiki18（21M）。
- **结论**：这批数据**不能用于任何训练**（既泄漏评估集、检索上下文又是垃圾）。文档里 "2wiki/musique 已有银标" 的说法不能直接采用——需要**用 `--split train` + wiki18 语料重新生成**才可用。好消息是 `data/2wikimultihopqa/train.jsonl`（167,454 条）与 `data/musique/train.jsonl`（19,938 条）都在，重生成有原料。

### 0.4 ⚠️ 主方法历史评估用的是 `temperature=0.7`，不是当前基线的 `0.0`

- 主方法历史 run（如 `outputs/R8_final/*/config.yaml`）里 `temperature: 0.7, do_sample: true`，`test_sample_num: 100`。
- 当前基线（`outputs/baselines_rerank/*`）是 `temperature: 0.0, do_sample: false`，`n=300`。
- **结论**：statistics.md 里主方法的 EM/F1 与基线新表**在采样温度、样本量、KG 索引版本三个维度都不同**，不能同表比较。这也解释了 `R8_final` 三 seed（13/42/2024）EM 0.33/0.30/0.32 的差异——是 `temp=0.7` 的采样方差，不是"KG 噪声地板"。

### 0.5 ⚠️ 评估在 `temp=0` 下是确定性的，"多 seed 噪声地板 ~4pp"对 EM/F1 不成立

- 评估用 `random_sample=False`，取 dev **前 N** 条；`temperature=0.0` 时生成完全确定。
- 实测 `baseline_3seed`（temp=0）seed 13 与 2024 的 EM 都是 0.36，完全一致。
- **结论**：EM/F1 在 `temp=0 + first-N` 下**没有 seed 采样方差**，跑多 seed 是浪费。EM/F1 的统计显著性应来自**逐题配对（McNemar）** 和 **bootstrap 置信区间**，不是跨 seed。真正的采样噪声只在 **IHR（n=50 抽样，`random.seed`）** 上——那里多 seed 才有意义。

---

## 1. 待处理问题总清单（按依赖排序）

| # | 问题 | 依赖 | 产物 | 归属 |
|---|------|------|------|------|
| P0 | 定调 KG 定位（训练期正则 vs 训练+推理输入增强） | 无 | 论文措辞 | 你定 |
| P1 | 银标 KG 有效率低 → 重建银标（train-split + wiki18 + 强制 KG 引用） | 无 | 新银标 | 你定是否重生成 |
| P2 | 修训练期 KG 索引 0 命中 + 对齐 `ppo_max_kg_triples=12` | P1 | 训练侧代码/脚本 | 代码 |
| P3 | PPO 重训（回退 Plan B + 对齐 PPO① 的锚点/α-gate/split） | P1, P2 | 新 PPO checkpoint | 训练 |
| P4 | 全量重评（主方法 4 臂 × 3 数据集，temp=0，n=300，与基线同条件） | P3 | EM/F1 主表 | 评估 |
| P5 | IHR 统一 deepseek-v4-pro 重判（主方法 + 补 reaRAG 2wiki/musique） | P4 | IHR 主表 | 评估 |
| P6 | 训练/奖励参数是否需要进一步调 | P3 结果 | 是否加训 | 你定 |

---

## 2. 逐项详细方案

### P0 — KG 定位定调（阻塞 Introduction / Contribution 的写法）

**问题**：文档自相矛盾。`RESEARCH_WORKFLOW.md` §3.4 写"推理时无需外部 KG（KG 只用于训练）"，但代码 `kg_proweight_pipeline.py` 默认 `inject_kg=True`，主消融轴"有 KG / 无 KG"正是推理期注入。当前实际是"训练期正则 + 推理期输入增强"的混合。

**方案（三个可选项，需要你选一个）**：

- **选项 A（推荐，与代码一致）**：定位写成"**训练期过程奖励正则 + 推理期 KG 输入增强**的双重用法"。论文主消融"KG vs noKG"直接对应推理期注入；PPO 的 KG 分支对应训练期正则。诚实、与实验完全吻合，不损失卖点。
- **选项 B（纯训练期正则）**：把推理期注入默认关掉（`inject_kg=False`），则"SFT+KG 优于 SFT"那组结果就不成立了，主消融轴要改成"PPO 有无 KG 奖励分支"。**与现有 EM/F1 主表冲突，需要重新设计所有实验**。
- **选项 C（纯推理期增强）**：砍掉训练期 KG 奖励，只留推理注入。**方法名与论文框架就名不副实**。

**验收**：在论文方法节 + 摘要里，KG 定位的表述与最终选的实验设计一致；`RESEARCH_WORKFLOW.md` §3.4 的"待定调"标注解除。

---

### P1 — 银标 KG 有效率低（核心上游问题）

**问题**：当前银标（`silver_v1_reannotated.jsonl`）KG 引用率低，直接决定 `kg_reward_share` 上限。

**现状（实测）**：

| 指标 | 值 |
|------|-----|
| 轨迹数 | 24,998（全部 hotpotqa **train** split） |
| accepted | 9,839（**39.4%**） |
| 步总数 / 有引用的步 | 80,120 / 24,969（**31.2%** 整体；**51.1%** accepted 内） |
| accepted 步标签分布 | NEU 70.2% / POS 25.5% / NEG 4.3% |
| teacher_model | `deepseek-chat`（见 0.2） |

**拒绝原因分解（24,998 条中 15,159 条被拒）**：

| 拒绝原因 | 条数 | 性质 |
|---------|------|------|
| `answer_score=0.00` | 6,294 | 最终答案错 → 整个轨迹被拒（合理，但说明弱教师产率低） |
| `sparse_quota_full` | 4,726 | KG 稀疏桶配额满 → **人为丢弃** |
| `step_count=2` | 3,811 | 只有 2 步，不满足 `min_steps=3` |
| 其余 | ~328 | 分数偏低等 |

**根因分析**：
1. **弱教师 + 答案错**：39.4% 接受率里最大一坨是"最终答案错"（answer_score=0）。换强教师（DeepSeek-R1 / GPT-4o 级）能直接提高答案命中率，从而抬接受率。
2. **KG 引用靠"自愿"**：teacher 只在 `Knowledge Used` 里写它觉得该引的三元组，约一半 step 不引（accepted 内 51.1% 引）。代码注释已确认 `kg_reward_share` 的杠杆是**引用频率**而非引用精度（引用步的 precision 实测 13/13、relevance 0.556，都很高）。
3. **接受门槛里的 `answer_score=0` 一刀切**：`answer_match_score` 是 lenient 版，0.3 门槛不高，但弱教师大量输出 0 分答案。

**方案（按性价比排序）**：

1. **强制 KG 引用（改动最小，收益最大）**：改 teacher prompt，要求**每个非 discourse step 必须给出 ≥1 条 `Knowledge Used`**，并在 `_needs_format_retry` 里对"某步无引用"做一次 retry。目标：accepted 内引用率 51%→85%+。
2. **换强教师重生成**（需 DeepSeek API 配额 + 成本确认）：`phase1_generate_silver.py --split train --teacher deepseek-v4-pro`（或更强），覆盖 hotpotqa/2wiki/musique 三数据集 train split。产出多数据集、无泄漏、wiki18 检索的新银标。
3. **修接受过滤器**（小改）：把 `sparse_quota_full` 的丢弃从"静默丢"改成"保留并降权"；`step_count=2` 的轨迹若非答案错可放宽 `min_steps=3→2`（这类轨迹在 MuSiQue 常见）。
4. **标签分布再平衡**：accepted 步 NEU 70.2% 太高，PRM 学不到判别信号。对 POS/NEG 步过采样，或 Phase 2 损失改 focal/类别加权（`phase2_prm.py` 已有逆频权重，检查是否真的生效）。

**关键约束（重生成时不可再犯）**：
- 必须 `--split train`（禁止 `--allow_eval_split`，避免 0.3 的泄漏重演）。
- 必须用 **wiki18 语料 + rerank-10** 检索（`indexes_wiki18/` + `--rerank 10`），禁止落到 `indexes_smoke`（989 文档）。
- 记录 teacher 真实身份（0.2）、检索语料版本、rerank 设置进 manifest。

**验收**：新银标 accepted 内引用率 ≥85%；`kg_reward_share`（PPO 训练时）≥10%；无 dev 泄漏；三数据集覆盖。

---

### P2 — 修训练期 KG 索引 0 命中 + 对齐 `ppo_max_kg_triples`

**问题**：见 0.1。训练侧 KG 索引命中 0%，且 `ppo_max_kg_triples` 在 PPO①/PPO② 间从 30 变到 12。

**方案**：

1. **为训练 split 重建 KG 索引**：用 `scripts/prepare/06_build_question_kg_index.py --datasets hotpotqa 2wikimultihopqa musique --split train`（或 `--silver <新银标>`），输出 `question_kg_index_v2_train.json`。让 PPO 的 `_q_kg_path` 能指向训练 split 索引（当前 `phase3_ppo.py:797` 硬编码读 dev 的 `question_kg_index_v2.json`，需要加一个 `--kg_index_path` 参数或按 `split` 自动选文件）。
2. **把 `ppo_max_kg_triples` 钉死在 12**（与推理 `max_kg_triples=12` 一致）：
   - 当前 `phase3_ppo.py:178` 默认已是 `12`，但 `phase3_ppo.yaml` 的注释还写着"fixed at 30"——**注释是错的，要改**。
   - 在 `launch_split_ppo.sh` 显式加 `--ppo_max_kg_triples 12`（若已支持）或在 yaml 里把该值真正透传（当前 yaml 无法设置该值，靠 dataclass 默认，容易漂）。
3. **统一 fallback 语义**：索引 miss 时的 fallback（`phase3_ppo.py:363` 的 `filter_and_rank_triples(max_keep=ppo_max_kg_triples)`）与索引命中时的过滤必须用**同一套 `min_keep`/`max_keep`**，否则又引入一个隐藏错位。

**验收**：新一次 PPO 训练的 `manifest.json` 里 `ppo_max_kg_triples=12`，且训练日志里 `question_kg_index` 命中率 >90%（不再是 0%）；训练 prompt 的 KG 三元组数与推理一致。

---

### P3 — PPO 重训

**问题**：Plan B 已回退（yaml 值 4.0/1.5），但 PPO② 那次把 SFT 锚点、α-gate、fold 三个 artifact 也换成了更旧的，需要一并对齐回 PPO①。

**方案（对齐 PPO① 的五个配置，只保留"回退 Plan B"这一个变量）**：

| 配置项 | 值（= PPO①） | 来源 |
|--------|-------------|------|
| `outcome_weight` | 4.0 | yaml（已回退） |
| `step_reward_scale` | 1.5 | yaml（已回退） |
| `text_reward_scale` | 0.3 | yaml（不变） |
| `sft_checkpoint` | `checkpoints/sft_student_split/final` | PPO① manifest |
| `alpha_gate_path` | `checkpoints/prm_alpha_gate_v1reann_negfix/alpha_gate.pt` | PPO① manifest |
| `split` | `train` | `--split train` |
| `ppo_max_kg_triples` | **12**（新：与推理对齐，见 P2） | 0.1 |
| `kl_coef` / `target_kl` | 0.15 / 40.0 | yaml |
| `total_steps` / `batch_size` | 3000 / 4（=750 updates） | yaml |

**关键点**：
- 训练在 **AutoDL 远端 96GB**（`launch_split_ppo.sh` 的 `REMOTE_ROOT=/root/autodl-tmp/kgpaper`），本机 4090 24GB 跑不了完整 PPO（见 `trl-ppo-logits-oom` 记忆）。重训需在远端执行并回传 `history.jsonl` + `tensorboard` + samples。
- `launch_split_ppo.sh` 已指向 PPO① 的 SFT/α-gate，**基本可以直接用**，但需按 P2 把 `ppo_max_kg_triples` 与训练 KG 索引路径修对。
- 训练期监控 `kg_reward_share`：目标 ≥10%（PPO① 均值 9.0%、末值 17.8%；PPO② 只有 1.59%/2.7%）。若仍 <5%，说明银标引用频率没解决，回 P1。

**验收**：新 checkpoint 的 `manifest.json` 五配置与上表一致；`history.jsonl` 里 `kg_reward_share` 均值 ≥10%、`ppo_mean_kl` 终值 <25（不重演 PPO② 的 29.4）。

---

### P4 — 全量重评（主方法，与基线同条件）

**问题**：主方法历史数字（statistics.md）是 `temp=0.7 / n=100 / 旧 KG 索引` 跑出来的，与 8/21 新基线（`temp=0 / n=300 / rerank-10 / 新 KG 索引`）不同条件。

**方案**：在**回退 Plan B 重训出的新 PPO** 上，重新跑主方法四臂：

```
for arm in {SFT noKG, SFT+KG, PPO noKG, PPO+KG}:
    for ds in {hotpotqa, 2wikimultihopqa, musique}:
        python scripts/eval/run_kg_proweight.py \
            --checkpoint <对应 checkpoint> --datasets $ds \
            --test_sample_num 300 --seeds 42 \
            --rerank 10 --alpha_gate_path <...> \
            [--no_kg 若 noKG 臂]
```

**口径对齐（必须与基线完全一致）**：
- `temp=0.0`、`max_tokens=512`、`n=300`、`--rerank 10`（top-50 → bge-reranker → 10）。
- SFT 臂 checkpoint = `sft_student_split/final`；PPO 臂 = 新重训 checkpoint；noKG 臂加 `--no_kg`。
- 记录 GPU = RTX 4090（评估在本机跑），写进 manifest 供论文 Reproducibility。

**统计口径（修正 0.5 的误区）**：
- EM/F1 在 `temp=0 + first-300` 下**确定**，不需要多 seed。显著性用**逐题 McNemar**（KG 臂 vs noKG 臂配对）+ **bootstrap 95% CI**。
- 若要补样本量：2Wiki 可加 `n=1000`（历史上做过，用于收紧 CI），其余维持 n=300。

**验收**：产出 `outputs/<root>/<arm>/<ds>/seed_42/*/metric_score.json`，四臂 × 三数据集完整；EM/F1 与基线新表可直接同表。

---

### P5 — IHR 统一口径重判

**问题**：
- 主方法历史 IHR（`outputs/ihr_eval/`，12 个文件）全部是 `deepseek-chat` 判的。
- 基线 IHR（`outputs/baselines_rerank/`）是 `deepseek-v4-pro` 判的。
- 两个 judge 不可比，而 IHR 是本文核心主张。

**方案**：
1. **主方法 IHR 用 `deepseek-v4-pro` 重判**：对 P4 重评产出的主方法四臂推理链，跑 `scripts/eval/run_ihr_judge.py --judge_model deepseek-v4-pro --sample 50 --seed 42`（`run_ihr_judge.py` 默认 judge 还是 `gpt-4o-2024-08-06`，必须显式传 `--judge_model deepseek-v4-pro`）。
2. **补 reaRAG 基线**：reaRAG 的 2wiki/musique EM/F1 本批跳过（见 `baseline_results.md`），且 reaRAG rerank-10 的 IHR 只有旧 top-15 的 0.237。需补跑 reaRAG × {hotpotqa, 2wiki, musique} 的 EM/F1 + IHR。
3. **IHR 的统计**：n=50 有真实采样方差（`random.seed(42)`），可加 `seed ∈ {42, 123, 2024}` 三抽样算均值 ± 标准差，或用 bootstrap。这是唯一"多 seed 有意义"的地方（见 0.5）。

**验收**：主方法与基线的 IHR 全部 `judge_model=deepseek-v4-pro`；主表可比。

---

### P6 — 训练/奖励参数是否需要进一步调（在 P3 结果出来后决定）

**问题**：你提到"奖励函数设置等都需要训练"。先明确：**当前奖励结构没有硬 bug**，问题在三个可调的"信号强度"旋钮。是否要调、往哪调，**必须等 P3 重训后的 `history.jsonl` 出来再定**，否则是拍脑袋改核心 reward（AGENTS.md §8 需确认）。

**现有旋钮与当前值（代码实测）**：

| 旋钮 | 当前值 | 作用 | 触发调整的信号 |
|------|--------|------|---------------|
| `outcome_weight` | 4.0 | 最终答案 EM∈{0,1} 的权重 | EM 奖励独大 → 降；EM 信号太弱 → 升 |
| `step_reward_scale` | 1.5 | 过程奖励缩放 | `kg_reward_share` <5% → 升 |
| `text_reward_scale` | 0.3 | 文本流畅奖励缩放 | 文本奖励噪声主导 → 降 |
| `discount` | 0.95 | 回报折现 | 长轨迹过程信号衰减太快 → 升 |
| `min_valid_steps` | 2 | 格式门（无独立格式奖励） | `valid_rate` 崩溃 → 降 |
| `sft_anchor_weight/interval` | 0.10 / 10 | 拉回 SFT 锚 | `ppo_mean_kl` 终值 >25 → 加锚 |

**逐步奖励公式（`composite_reward.py` 实测，写作时照此）**：

$$R_t = \bigl(\alpha_t \cdot R_{KG}(t) + (1-\alpha_t)\cdot R_{text}(t)\cdot 0.3\bigr)\cdot 1.5$$

- 末步：`trajectory_valid` 为真 → $+4.0\cdot\text{EM}$；否则 $-4.0$。
- 回报 $G_t=\sum_{k\ge t} 0.95^{k-t}R_k$。
- `r_kg` 是连续值 `precision × relevance ∈ {-1}∪[0,1]`（规则标注器 `PRMAnnotator.label()`，不是训练出的 PRM head）。

**两个已知的、可以现在修的小问题**：
1. **步数上限**：`reward_function.py:298` 用 `max_steps=7` 截断，但 rollout 实测 `n_steps_sample` 均值 8.87（最大 18）。即 **reward 只算了前 7 步，模型却在生成 8~18 步**——超出部分无监督。建议 `max_steps` 与 `max_new_tokens` 允许的步数上限对齐，或在 reward 里显式惩罚超长。
2. **`kg_reward_share` 的杠杆是引用频率不是精度**（代码注释已确认，见 P1）。调 `step_reward_scale` 只能把"现有引用"放大，抬不了"零引用步"的份额——根子在银标，回 P1。

**验收**：P3 训练曲线出来后，按上表"触发信号"逐项判断是否需要调；任何对核心 reward 的改动单独开一次训练 + 单独 Experiment ID。

---

## 3. 执行顺序（建议）

```
1. [你定] P0 KG 定位（选项 A 推荐）          —— 不阻塞训练，只阻塞写作
2. [你定] P1 是否重建银标                     —— 决定 P3 的训练数据
        ├─ 不重建：用现有 silver_v1（hotpotqa-only），先跑 P3 恢复 KG 正则化
        └─ 重建：换强教师 + 强制引用 + 三数据集 train split + wiki18
3. [代码] P2 修训练 KG 索引 + 钉死 ppo_max_kg_triples=12
4. [训练] P3 远端重训 PPO（对齐 PPO① 五配置）
5. [评估] P4 主方法四臂 × 3 数据集 重评（temp=0, n=300, rerank-10）
6. [评估] P5 IHR 统一 deepseek-v4-pro 重判 + 补 reaRAG 2wiki/musique
7. [决定] P6 看 history.jsonl 决定是否调奖励参数再加训
```

**若想最快拿到一个可用的"PPO 回退重训"结果**：跳过 P1 重建（先用现有 hotpotqa 银标），直接走 P2→P3，4 小时后就有新 PPO checkpoint，然后 P4/P5。P1 银标重建是"抬天花板"的长期项，可以和这条快线并行。

---

## 4. 需要你拍板的决策点（汇总）

| # | 决策 | 我推荐 | 影响 |
|---|------|--------|------|
| D1 | KG 定位（P0） | 选项 A：训练期正则 + 推理期输入增强 | Introduction/Contribution 写法 |
| D2 | 是否重建银标（P1） | 先跑快线，重建作并行长期项 | 训练数据、kg_reward_share 上限 |
| D3 | teacher 真实身份（0.2） | 确认 deepseek-chat 是否 = deepseek-v4-flash | 论文 teacher 表述 |
| D4 | `ppo_max_kg_triples` 是否钉 12（P2） | 钉 12，与推理对齐 | 训练/推理 KG 一致 |
| D5 | P6 是否调奖励参数 | 等 P3 曲线，不预设 | 核心 reward 改动需单开实验 |
| D6 | PPO 重训是否启动 | 需你确认（AGENTS.md §8 大规模训练） | P3 |

---

*本文档基于 2026-08-21 代码/数据核实。所有关键数字都标了来源文件；若你改了其中任一配置，需同步更新本文档对应条目。*
