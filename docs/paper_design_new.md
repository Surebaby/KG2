# KG-ProWeight (修订版): Adaptive Process Supervision for Agentic RAG

> 基于知识图谱约束蒸馏与动态可信度权重的 Agentic RAG 自适应过程监督方法。
>
> **本文档(`paper_design_new.md`)是按当前实际代码修订的方法规范**,取代
> `paper_design.md`(v1.0,2026-05-08)中与实现不符的部分。修订依据是
> `scripts/train/try/` 下经冒烟验证的实现(见 `FRAMEWORK.md` / `PROGRESS_REPORT.md`)。
> 凡与旧版有出入处,均以本文件 + 代码为准,并标注 **[修订]**。

---

## 目录

1. [Abstract](#1-abstract)
2. [Methodology](#2-methodology)
   - 2.1 Phase 1 — 图引导轨迹蒸馏(含三值标注判定标准)
   - 2.2 Phase 2 — 动态可信度门控 α-Gate
   - 2.3 Phase 3 — 自适应过程监督 RL
3. [与旧版 paper_design 的关键差异](#3-与旧版的关键差异)
4. [Experimental Design](#4-experimental-design)
5. [Theoretical Analysis](#5-theoretical-analysis)
6. [Ablation Studies](#6-ablation-studies)
7. [已知数据特性与局限](#7-已知数据特性与局限)

---

## 1. Abstract

KG-ProWeight 用外部知识图谱(Wikidata)作为逻辑锚点解决 Agentic RAG 的两个痛点:
(C1) 过程奖励标签昂贵;(C2) 文本过程监督缺乏每步的事实性锚点,导致"推理幻觉"。
方法auto-构造细粒度三值过程奖励标签(+1/0/−1),并用一个可学习的可信度门控 α
自适应地混合 KG 奖励与文本奖励。

三项贡献:

1. **图引导轨迹蒸馏**:结构化约束 prompt + Wikidata 可达性校验,从 Teacher LLM
   (GPT-4o / DeepSeek-V3)近零成本蒸馏三值步标签,并保留事实可追溯性。
2. **动态可信度门控 α-Gate**:3 特征可学习门控(图密度、链接置信度、语义不确定度)
   逐步混合 KG 与文本过程奖励;密度高、置信高 → α→1(强制 KG),缺失 → α→0(回退文本)。
3. **自适应过程监督 RL**:PPO + GAE,在**逐步**复合奖励上训练 Student
   (Llama-3-8B-Instruct)。

---

## 2. Methodology

### 2.1 Phase 1 — 图引导轨迹蒸馏

四个子步骤:子图锚定 → 约束式思维链生成 → **三值自动 PRM 标注** → 质量过滤(银标接受)。

#### 2.1.1 子图锚定

对每个问题,从 question 抽取实体 mention,链接到 Wikidata QID,取 2-hop 子图
`G_sub`(`K_e=30`,`max_hops=2`)。**[修订]** mention 提取改为多源稳健提取:
spaCy NER(可选)+ 大写短语正则 + **检索 passage 标题**作为锚点(HotpotQA/2Wiki 的
gold 支撑文档通常以关键实体命名)。`coverage` 仅作软信号记录,**不再用于拒绝**。
SPARQL 失败时优雅降级为空子图(该轨迹落入 kg_sparse 桶,而非被丢弃)。

#### 2.1.2 约束式思维链生成

Teacher 按统一 schema 生成轨迹(`kgproweight/data/prompts.py`,所有阶段共用):

```
[Step N]
Reasoning: <自然语言推理>
Knowledge Used: [(head, relation, tail), ...]   # 引用的 KG 三元组
Conclusion: <一句事实结论>
...
[Final Answer]
<最终答案>
```

#### 2.1.3 三值自动 PRM 标注 **[修订 — 这是本版核心改动]**

每一步标 +1 / 0 / −1。标注器为 `ImprovedPRMAnnotator`
(`scripts/train/try/shared/prm_annotator_try.py`)。**判定遵循"默认中性,只在能明确
证实时给 +1、能明确证伪时给 −1"的保守策略**,完整决策树如下(按顺序短路):

**输入**:该步文本、KG 子图、之前各步的结论列表 `prev_conclusions`。
**前置量**:`subgraph_usable = (子图三元组数 ≥ min_subgraph_for_verify=3)`。

```
① 该步是话语/过场句(DISCOURSE_RE 匹配开头)且未引用任何三元组
      → 0 (NEUTRAL)   "纯衔接步,无可验证内容"

② 该步未引用任何三元组(Knowledge Used: [])
      ├─ 与某条 prev_conclusion 构成矛盾(_is_contradiction 为真)
      │     → −1 (NEGATIVE)   "无引用时唯一的负触发"
      └─ 否则
            → 0 (NEUTRAL)   "KG 无法证实也无法证伪(含正确的世界知识步)"

③ 该步引用了三元组
      ├─ 3a) 子图太稀疏(< 3 条,subgraph_usable=False)
      │       → 0 (NEUTRAL)   "子图不足以证伪,对应痛点 C2:不惩罚 KG 缺失"
      ├─ 3b) 逐条用 triple_in_subgraph(fuzzy≥80) 验证;
      │       与某条 prev_conclusion 矛盾(_is_contradiction)
      │       → −1 (NEGATIVE)   "矛盾优先级最高"
      ├─ 3c) 全部三元组验证通过 且 至少一条与本步 Conclusion 词面相关
      │       (_triple_relevant 为真)
      │       → +1 (POSITIVE)   "KG 真正支撑了这步"
      ├─ 3d) 全部验证通过 但 没有一条与 Conclusion 相关(凑数引用)
      │       → 0 (NEUTRAL)   "[修订] 三元组虽真但不支撑本步,降级防 reward hacking"
      └─ 3e) 引用了但子图里查不到
              → 0 (NEUTRAL)   "视为子图不完整,而非已证实的幻觉"
```

**三个判定标准的精确定义**:

- **+1 (POSITIVE)** — 同时满足:(a) 本步引用了 ≥1 条三元组;(b) 子图 ≥3 条;
  (c) 所有引用的三元组都能在子图里(模糊匹配阈值 80)找到;(d) 至少一条引用三元组的
  头或尾实体的内容词出现在本步 Conclusion 里(`require_triple_relevance`);
  (e) 不与任何前序结论矛盾。语义:**KG 明确支撑了这一步的结论**。
- **0 (NEUTRAL)** — 默认标签。涵盖:纯话语步;无引用且无矛盾(典型是"用世界知识/passage
  推理但没引 KG"的正确步);子图太稀疏无法判定;引用了但查不到;**引用真但与结论无关
  (凑数)**。语义:**KG 既不能证实也不能证伪这一步**。
- **−1 (NEGATIVE)** — **唯一触发是与前序结论的硬矛盾**(`_is_contradiction`)。语义:
  **KG 或前序事实明确否定了这一步**。

**`_is_contradiction` 的两道护栏 [修订]**:
1. **弃权护栏**(`_ABSTENTION_RE`,`guard_abstention=True`):若结论是诚实弃权
   ("no evidence that…"/"does not directly link…"/"the KG has no info"/"cannot
   determine"/"X is not identified"等),**一律不判矛盾**——这是模型如实报告 KG 缺口,
   不是幻觉。修复了原版把弃权误判成 −1 的问题(实测 11/16 误报 → 0)。
2. **底层启发式**(`_contradicts`):两句结论共享 ≥2 个长度≥4 的内容词,且一方含否定词
   (not/never/no/cannot/does not)、另一方不含,才算矛盾。
   - **已知局限**:曾尝试加"主语实体一致"约束(要求矛盾双方共享专有名词),但
     `ENTITY_RE` 提取的是短语片段而非语法主语,既没修掉误报又误伤真负样本,**已撤销**。
     残留约 3-4/80 条弃权措辞未被正则覆盖,−1 总量小,影响有限,跑 IHR 前可再收紧。

**`_triple_relevant` 定义 [修订]**:本步引用的任一三元组,其头实体或尾实体的内容词
(长度≥4、非停用词)与 Conclusion 的内容词有交集,即视为相关。无法读取 Conclusion 时
默认相关(不增加假阴性)。此约束把"教师挂任意一条在子图里、但与推理无关的三元组也拿 +1"
的凑数率从实测 24% 降到 2%。

**关键参数**:`min_subgraph_for_verify=3`、`triple_fuzzy_threshold=80.0`、
`require_triple_relevance=True`、`guard_abstention=True`、`neutral_pattern_match=True`。

#### 2.1.4 质量过滤(银标接受)**[修订]**

`StratifiedSilverFilter`(`distill_helpers_try.py`)。**取消原版对 triple_rate/coverage
的硬性拒绝**,改为:

- **普适质量门(所有桶都要过)**:步数 ∈ [min_steps=3, max_steps=7];宽松答案匹配分
  `answer_match_score ≥ min_answer_score=0.3`。
- **按 triple_rate 分桶 + 配额**(triple_rate = 引用了三元组的步数 / 总步数):
  - `kg_rich`(≥0.5):无条件接受;
  - `kg_medium`(≥0.15):接受至多 `medium_quota=35%` 的接受池;
  - `kg_sparse`(<0.15):接受至多 `sparse_quota=25%` 的接受池。
- **保留 kg_sparse 切片是刻意的**:α-Gate 需要低密度/低覆盖样本来学 α→0 回退区域
  (正是 D_dropout 要验证的行为),原版过滤器把这些系统性删掉了。

`answer_match_score` 取四者最大:归一化精确匹配→1.0、gold 是 pred 子串→1.0、
gold 在 pred 中的 token 召回、别名容忍的 token-F1。先 `clean_final_answer` 去掉
"the answer is"前缀和尾随从句。这救回了"正确但啰嗦/别名"的答案。

**接受率**:基线 7.6% → ~60-69%。

---

### 2.2 Phase 2 — 动态可信度门控 α-Gate

#### 2.2.1 三维特征

每步特征 `x_t = [f_density, f_confidence, f_entropy]`:
- `f_density` — KG 子图密度(`graph_density`,边数/(点数+ε))。
- `f_confidence` — 链接置信度(实体链接的模糊匹配分均值;若有 TransE 嵌入则用余弦)。
- `f_entropy` — **语义不确定度,来自该步 token 的真实 logprob**。
  **[修订]** 旧实现把它硬编码(0.5,PPO 期间退化为 1.0);现经 logprob 预扫
  (`compute_step_logprobs`)用真实 token logprob 计算 `entropy_from_logprobs = -mean(logp)`。

#### 2.2.2 门控方程

```
α_t = σ( (W·x_t + b) / τ )
```

初始 `W=(1.0, 1.5, -0.8)`、`b=-2.0`、`τ=0.5`(下限 0.1)。密度↑、置信↑ → α→1(信 KG);
熵↑ → α→0(信文本)。Phase 2 学 W/b/τ。

#### 2.2.3 损失

```
L_gate = L_PRM(对 {+1,0,−1} 的 3 类交叉熵) + λ·L_calibration
```

`L_calibration` 为 BCE(α, coverage>0.5 目标),`λ = calibration_weight = 0.1`。
另可选训练一个辅助 text_reward_head(Linear→Tanh,对 ±1 质量目标做 MSE)。

**产出**:`alpha_gate.pt`、`prm_head/`(LoRA adapter + prm_head.pt)、
`text_reward_head.pt`、`silver_with_logprobs.jsonl`。

**[修订] 实现说明**:24GB 卡上 `phase2_prm_try.py` 在运行时把包内两个 bf16 加载函数
monkeypatch 成 4-bit NF4,再调用未改动的包内 `run_phase2`(复用整个训练循环)。

---

### 2.3 Phase 3 — 自适应过程监督 RL

#### 2.3.1 Phase 3a — SFT(学格式)

PPO 前先 SFT,让 Student 学会 `[Step N]…[Final Answer]` schema,否则 rollout 产不出
可解析轨迹。只用 accepted 轨迹,且**丢弃 label=−1 的步**(不教幻觉)。
**[修订]** `phase3_sft_try.py` 用 4-bit QLoRA + gradient checkpointing + 可调 max_length;
`--merge_output` 训练后用 bf16 重载基座合并 LoRA,产出完整模型供 PPO 直接加载
(解决 SFT 产出是纯 adapter、PPO `from_pretrained` 无法加载的衔接断层)。

#### 2.3.2 复合每步奖励

```
R_total(t) = α_t · R_KG(t) + (1 − α_t) · R_Text(t)
R_outcome  = EM(answer, gold) ∈ {0,1}   # 仅加在最后一步
Return: G_t = Σ_{k≥t} γ^{k−t} R_total(k),  γ = 0.95
```

- `R_KG(t) ∈ {+1, 0, −1}` —— **[修订]** 由 `ImprovedPRMAnnotator` 给(非原版),
  避免凑数 +1 / 弃权 −1 误报作为奖励回流导致 reward hacking。
- `R_Text(t)` —— 冻结的 ReaRAG 式文本奖励模型,或 fallback 的 Llama-3-8B 奖励头;
  输出经 tanh 映射到 **[−1,1]**,与 R_KG 量纲一致(已核实无需额外归一化)。
- `R_outcome` —— **[修订]** EM 比的是 `metadata["gold_answer"]`(真 gold),
  而非教师自己的答案 `traj.answer`(否则会奖励模仿教师的错误答案)。

#### 2.3.3 PPO + GAE **[修订 — 论文核心机制的代码修复]**

超参:lr=1e-5,batch=64,mini_batch=8,ppo_epochs=4,clip ε=0.2,KL β=0.01,
γ=0.95,λ(GAE)=0.95,grad-norm cap=1.0,total steps≈5000。

**P0-1 修复**:每步奖励 `R_total(t)` 必须真正进入 GAE。原实现把每步奖励 `.sum()` 成
一个标量,而 TRL 0.11.4 的 `compute_rewards` 只把标量放在**回答最后一个 token**上,
GAE 因此退化为 outcome-only,**Theorem 2 失去代码基础**。

修复:`StepRewardPPOTrainer(PPOTrainer)` **只 override `compute_rewards`**——
把每步 `R_total(t)` 铺到该步 `[Step N]` 区间的末 token(最后一步再叠加 R_outcome),
通过 side-channel `set_pending_step_rewards` 注入 per-token 奖励张量;TRL 原本的
逐 token KL penalty、`compute_advantages`(GAE)、minibatch、clip 全部复用。
这样 GAE 在真正的逐步信号上做信用分配。Critic 用 PRMValueHead。

**参考模型 [修订]**:LoRA 时 `ref_model=None`,TRL 自动用"禁用 adapter 的 policy"当
参考模型;原 `create_reference_model` 复制第二个全量 8B 是 24GB OOM 的根因。

**耦合提示**:override 复制了 TRL 0.11.4 的 `_kl_penalty`/`kl_ctl` 逻辑;升级 TRL
需重新同步该方法。

---

## 3. 与旧版的关键差异

| 维度 | 旧 paper_design / 包内实现 | 本修订版 / try 实现 |
|---|---|---|
| 步标注 | 原版 PRMAnnotator,实跑退化二分类(0% NEUTRAL),凑数+1 24%、−1误报 11/16 | ImprovedPRMAnnotator:相关性约束 + 弃权护栏,真三分类(+1:40%/0:54%/−1:6%) |
| 银标接受 | 硬性 triple_rate/coverage 阈值拒绝 | 分层分桶 + 配额,保留 kg_sparse(α→0 监督) |
| f_entropy | 硬编码 0.5 / PPO 期 1.0 | 真实 token logprob |
| R_KG 来源 | 原版标注器 | ImprovedPRMAnnotator |
| outcome 对比 | 教师答案 | 真 gold(metadata) |
| per-step→GAE | `.sum()` 成标量(outcome-only) | StepRewardPPOTrainer 逐步进 GAE |
| 参考模型 | 第二个全量 8B | LoRA 禁 adapter 共享 |

---

## 4. Experimental Design

- **语料/检索**:Wiki18 100w(~15M passages);E5(dense)+ BM25s(sparse)RRF(k=60)top-50。
- **KG 检索**:Wikidata SPARQL,K_e=30,2-hop,仅用于 PRM 奖励与 α 特征。
- **模型**:Teacher=GPT-4o/DeepSeek-V3;Student=Llama-3-8B-Instruct;
  PRM=Llama-3-8B+LoRA+value head+α-gate;Text reward=ReaRAG-9B(冻结)或 fallback 头。
- **数据集**:HotpotQA / 2WikiMultiHopQA / MuSiQue;D_dropout(切断答案路径桥接三元组,1000 条)。
- **指标**:EM、F1(FlashRAG);IHR(GPT-4o LLM-as-Judge + Cohen κ);α 分布;数据效率曲线;
  配对 bootstrap(n=10000)显著性,三种子。
- **运行环境 [修订]**:正式训练 Pro6000 96GB bf16;本机 RTX 4090 24GB 用 4-bit 变体冒烟。

---

## 5. Theoretical Analysis

- **Theorem 1(幻觉惩罚界)**:PPO 下 `α_t ≥ α_min > 0` 时,第 t 步关系幻觉概率
  `P_θ ≤ C·exp(−α_min·η·T)`。
- **Theorem 2(优势方差缩减)**:`p_miss` 为 `G_sub=∅` 概率,`Δ_R=E[R_KG−R_text|KG缺失]`,
  则 `V_dynamic ≤ V_fixed − p_miss(1−p_miss)·Δ_R²/4`。
  **[修订]** 此定理的实证依赖 P0-1——只有每步奖励真正进 GAE,advantage variance 才有意义;
  在 `.sum()` 标量化的旧实现下无法验证。

---

## 6. Ablation Studies

| 变体 | 修改 | 预期 |
|---|---|---|
| α=0 | 重训 PPO,α≡0 | IHR↑,EM/F1↓ 2-4 |
| α=1 | 重训 PPO,α≡1 | D_dropout F1↓ >5 |
| α=0.5 | 重训 PPO,α≡0.5 | 中等,逊于动态 |
| binary labels | Phase 2 只用 {+1,−1} 重训 | IHR 略↑ |
| single retriever | 全程仅 E5 | KG 链接覆盖↓,F1↓ |

所有消融均**重训**(经 `--alpha_override` / `--binary_labels_only` 钩子),非推理时打补丁。

---

## 7. 已知数据特性与局限

- **0/NEUTRAL 占比高**(accepted ~54%):HotpotQA 上真正"KG 强支撑"的步本就少,多数推理
  靠 passage + 世界知识。这是数据固有特性,非 bug;解读 α 分布时需说明。
- **−1 类很薄**(~6%):α 负向极端监督弱,α→0 区域主要靠 0 + kg_sparse 桶支撑。
- **~30% accepted 轨迹零 +1**:纯文本/世界知识推理,PPO 里奖励由 R_Text 主导。
- **−1 残留误报**(~3-4/80):弃权护栏正则未覆盖全部措辞;跑 IHR 前可收紧。
- **当前数据规模小**(50-80 条):仅验证管道连通,α-gate 训完权重≈初始值;效果结论需
  扩到 ~15k 接受规模后才有意义。
