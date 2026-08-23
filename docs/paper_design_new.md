# KG-ProWeight (修订版): Adaptive Process Supervision for Agentic RAG

> 基于知识图谱约束蒸馏与动态可信度权重的 Agentic RAG 自适应过程监督方法。
>
> **本文档(`paper_design_new.md`)是按当前实际代码修订的方法规范**,取代
> `paper_design.md`(v1.0,2026-05-08)中与实现不符的部分。修订依据是
> `scripts/train/try/` 下经冒烟验证的实现(见 `FRAMEWORK.md` / `PROGRESS_REPORT.md`)。
> 凡与旧版有出入处,均以本文件 + 代码为准,并标注 **[修订]**。
>
> **[修订 2026-08-19 — 代码一致性审计]** 本轮审计发现本文档若干处描述落后于代码,
> 已按 `kgproweight/` 包内实现校正,逐条标 `[修订 2026-08-19]`:
> (a) §1 / §2.1.3 / §2.3.2 —— 步标签是**连续 KG 效用分**(`precision × relevance`),
> 不是三值;三分类只出现在 Phase 2 训练 PRM head 时的 ±0.5 分桶;
> (b) §2.2.3 —— 校准损失的目标是"该步 KG 是否给出非中性判决",不是 `coverage>0.5`;
> 且 3 类 CE 带逆频类权重(文档原先未提);
> (c) §2.3.2 —— 实现的奖励公式含 `step_reward_scale` / `text_reward_scale` 两个乘子,
> 以及 `trajectory_valid` 门与 `−outcome_weight` 无效惩罚(文档原先未提);
> (d) §2.3.3 —— Critic 是 TRL 的 `AutoModelForCausalLMWithValueHead`,`PRMValueHead`
> 全仓库从未实例化;
> (e) §2.2.3 / §2.3.2 —— Phase 2 的三个产出里,进入 Phase 3 奖励的**只有 `alpha_gate.pt`**。
>
> 尚未处理、需研究者定调的不一致(不在本轮修订范围):§2.3.3 的"参考模型 `ref_model=None`"
> 与代码的 `create_reference_model(policy)` 相反;§4 的"KG 仅用于 PRM 奖励与 α 特征"与推理
> 期 `inject_kg=True` 相反;§2.1.4 / §3 / §7 的接受率与标签分布数字待按实测回填。

---

## 目录

1. [Abstract](#1-abstract)
2. [Methodology](#2-methodology)
   - 2.1 Phase 1 — 图引导轨迹蒸馏(含 KG 效用分标注判定标准)
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
方法自动构造细粒度的**连续 KG 效用分**作为过程奖励标签,并用一个可学习的可信度门控 α
自适应地混合 KG 奖励与文本奖励。

**[修订 2026-08-19]** 旧表述"三值过程奖励标签(+1/0/−1)"与代码不符。
`PRMAnnotator.label()`(`kgproweight/reward/prm_annotator.py:162`)返回的是
`r_KG = precision × relevance ∈ {−1} ∪ [0, 1]` 的**连续值**:−1 只由"与前序结论矛盾"
触发,+1 只在"全部引用三元组均通过验证且均与本步推理相关"时取到,中间是部分分。
三分类只出现在 Phase 2 训练 PRM head 时对该连续值的 ±0.5 分桶
(`phase2_prm._label_to_class`)。详见 §2.1.3。

三项贡献:

1. **图引导轨迹蒸馏**:结构化约束 prompt + Wikidata 可达性校验,从 Teacher LLM
   (deepseek-v4-flash)近零成本蒸馏**连续 KG 效用分**步标签(下游按 ±0.5 分桶为
   POS/NEU/NEG 训练 3 类 PRM head),并保留事实可追溯性。
2. **动态可信度门控 α-Gate**:3 特征可学习门控(图密度、链接置信度、语义不确定度)
   逐步混合 KG 与文本过程奖励;密度高、置信高 → α→1(信 KG),缺失 → α→0(回退文本)。
3. **自适应过程监督 RL**:PPO + GAE,在**逐步**复合奖励上训练 Student
   (Llama-3-8B-Instruct)。

---

## 2. Methodology

### 2.1 Phase 1 — 图引导轨迹蒸馏

四个子步骤:子图锚定 → 约束式思维链生成 → **自动 PRM 标注(连续 KG 效用分)** → 质量过滤(银标接受)。

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

#### 2.1.3 自动 PRM 标注:连续 KG 效用分 **[修订 2026-08-19 — 按训练路径的代码校正]**

**先说清楚标注器有两份实现,本节描述的是真正进训练的那一份**:

| 实现 | 返回类型 | 谁在用 |
|---|---|---|
| `PRMAnnotator`(`kgproweight/reward/prm_annotator.py:162`) | `float`,**连续** | 银标生成(`scripts/train/phase1_generate_silver.py:222`)、PPO 奖励(`kgproweight/training/reward_function.py`)、推理管线(`kg_proweight_pipeline.py:104`) —— **即全部正式实验路径** |
| `ImprovedPRMAnnotator`(`scripts/train/try/shared/prm_annotator_try.py:157`) | `int`,**真三值** | 仅 `scripts/train/try/` 下的 24GB 冒烟脚本与其测试 |

本文档上一版描述的是 try 变体(真三值),但 24,998 条银标、Phase 2 训练样本、PPO 的
每步 `R_KG` 全部由**包内连续版**产生。以下按包内实现重写。
(注:`scripts/train/try/tests/test_annotator_port_parity.py` 断言两者标注一致,该断言在
连续化改动后是否仍成立 **UNKNOWN**,需重跑该测试确认。)

**取值域**:`r_KG = precision × relevance ∈ {−1} ∪ [0, 1]`。
- `−1` 只由"与前序结论硬矛盾"触发(唯一负来源,没有分数化的负值);
- `+1` 只在"全部引用三元组都通过验证 **且** 全部与本步推理相关"时取到;
- `0` 是默认值,也是"引用了但一条都查不到"与"引用真但一条都不相关"的取值;
- `(0, 1)` 开区间是 **[修订 2026-08-19 新增]** 部分分:部分引用可验证、或部分引用相关。

**判定仍遵循"默认中性,只在能明确证实时给正分、能明确证伪时给 −1"的保守策略**,
完整决策树如下(按顺序短路):

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
      ├─ 3b) 逐条用 triple_in_subgraph(fuzzy≥80) 验证,累计
      │       precision = 通过条数 / 引用总条数
      ├─ 3c) 与某条 prev_conclusion 矛盾(_is_contradiction)
      │       → −1 (NEGATIVE)   "矛盾优先级最高,先于任何分数计算返回"
      ├─ 3d) 逐条判 _triple_relevant,累计
      │       relevance = 相关条数 / 引用总条数
      │       r_kg = precision × relevance
      ├─ 3e) precision ≥ 1.0 且 relevance ≥ 1.0
      │       → +1 (POSITIVE)   "全部三元组既真又相关,KG 完整支撑这一步"
      ├─ 3f) precision > 0(含 relevance 为 0 或部分)
      │       → r_kg ∈ [0, 1)   "[修订 2026-08-19] 部分分:按可验证率×相关率给,
      │                          替代旧版的 all-or-nothing。relevance=0 时 r_kg=0,
      │                          即旧版 3d'凑数引用降级'的结果不变"
      └─ 3g) precision == 0(引用了但子图里一条都查不到)
              → 0 (NEUTRAL)   "视为子图不完整,而非已证实的幻觉"
```

> **与旧版三值树的实质差别只有一处**:旧版 `+1` 的条件是"全部验证通过 **且至少一条**
> 相关",现在是"全部验证通过 **且全部**相关";不满足则落到 `precision × relevance`
> 的部分分,而不是一律归 0。旧版 3d(全真但全不相关 → 0)与 3e(全查不到 → 0)的
> **取值结果没变**,只是走的分支不同。

**判定语义的精确定义**:

- **正分(r_KG > 0)** — 前提:(a) 本步引用了 ≥1 条三元组;(b) 子图 ≥3 条;
  (c) 不与任何前序结论矛盾。分值 = (在子图里模糊匹配阈值 80 内找到的引用占比) ×
  (与本步推理/结论词面相关的引用占比)。取到满分 `+1` 需两个占比都为 1。
  语义:**KG 在多大程度上支撑了这一步**。
- **0** — 默认标签。涵盖:纯话语步;无引用且无矛盾(典型是"用世界知识/passage
  推理但没引 KG"的正确步);子图太稀疏无法判定;引用了但一条都查不到;引用真但
  一条都与本步无关(凑数)。语义:**KG 既不能证实也不能证伪这一步**。
- **−1** — **唯一触发是与前序结论的硬矛盾**(`_is_contradiction`)。语义:
  **KG 或前序事实明确否定了这一步**。注意负值不分级,没有 `−0.5` 这类中间负分。

**下游如何消费这个连续值 [修订 2026-08-19 新增]**:

| 消费方 | 做法 |
|---|---|
| Phase 2 训练 3 类 PRM head | `phase2_prm._label_to_class`:`≥ +0.5 → POSITIVE` / `≤ −0.5 → NEGATIVE` / 其余 `NEUTRAL`(`_POSITIVE_THRESHOLD = 0.5`)。**这是"三分类"在本方法中出现的唯一位置。** |
| Phase 3a SFT 样本过滤 | `phase3_sft.py:78` 丢弃 `label ≤ −0.5` 的步。原先写 `== -1` 会漏掉严格介于 −1 与 0 之间的值(当前实现下该区间取不到,但阈值写法更稳健)。 |
| Phase 3b PPO 每步奖励 | **直接用连续值**,不分桶(`composite_reward.py:95` `r_kg = float(self.prm_annotator.label(...))`)。 |

> **不得**用 `int()` 截断该值:`int(0.5)=0`、`int(0.75)=0` 会把全部部分分静默压成
> NEUTRAL,抹掉整个 partial-credit 信号(这正是 `_label_to_class` 的 docstring 记录的
> 已修 bug)。


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

**`_triple_relevant` 定义 [修订 2026-08-19 — 按代码校正]**:**逐条**判定
(`prm_annotator.py:314`,调用点 `:230` 每次只传一条三元组),对每条三元组:

- **比对文本** = 本步**推理正文** + Conclusion。推理正文是 `raw_text` 在
  `"Knowledge Used:"` 处截断后的前半段(`:222-224`)。**这一步剥离是必要的**:三元组本身
  就是从 `Knowledge Used:` 里抽出来的,拿它去匹配含该段的 `raw_text` 会 100% 自命中
  (self-citation),相关性判定退化为恒真。
- **评分**:head 短语命中 +1、tail 短语命中 +1、relation 短语命中 +0.5;**score > 0 即相关**。
- **分类关系一律不相关**:`instance of` / `subclass of` 直接跳过,不论实体是否重叠
  ——`(Ed Wood, instance of, human)` 是真三元组但对任何问题都没用。
- 比对文本为空(读不到推理与结论)时**默认相关**,不增加假阴性。

此约束把"教师挂任意一条在子图里、但与推理无关的三元组也拿 +1"的凑数率从实测 24% 降到 2%。
**[修订 2026-08-19]** 旧描述有三处与代码不符,已更正:(a) 判定粒度是逐条而非"任一条"
(结果要聚合成 `relevance` 比率);(b) 比对文本含推理正文,不只 Conclusion;
(c) 评分是 head/tail/relation 加权短语匹配 + 分类关系黑名单,不是"内容词(长度≥4)交集"。

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
  **[修订 2026-08-19 — 字段名陷阱]** Phase 2 训练时这个量存在样本的 **`coverage` 字段**
  里(`phase2_prm.py:64` 注释、`:174` `coverage=float(link_conf)`),读取处再
  `clamp(0,1)` 后喂给门控(`:726`)。**`coverage` 这个名字是历史残留,装的是步级
  link_confidence,不是"问题级 KG 覆盖度"**;旧版文档里的 `coverage_target` 是另一个量
  (见 §2.2.3)。写作与读码时不要把两者当成同一个东西。
- `f_entropy` — **语义不确定度,来自该步 token 的真实 logprob**。
  **[修订]** 旧实现把它硬编码(0.5,PPO 期间退化为 1.0);现经 logprob 预扫
  (`compute_step_logprobs`)用真实 token logprob 计算 `entropy_from_logprobs = -mean(logp)`。

#### 2.2.2 门控方程

```
α_t = σ( (W·x_t + b) / τ )
```

初始 `W=(1.0, 1.5, -0.8)`、`b=-2.0`、`τ=0.5`(下限 0.1)。密度↑、置信↑ → α→1(信 KG);
熵↑ → α→0(信文本)。Phase 2 学 W/b/τ。

**[修订 — 口径, 2026-08-18]**: α 只在**训练期**对 reward 做 KG/文本重加权
(`R_total = α·R_KG + (1−α)·R_Text`)。**eval 阶段 α 不参与 KG 门控**——仅作 telemetry
记录(`statistics.md` §二),不能靠 α 抬 EM/F1。见 `docs/paper/10_口径统一清单.md` ⑥。

#### 2.2.3 损失 **[修订 2026-08-19 — 按代码校正]**

```
L_gate = L_PRM + λ·L_calibration,   λ = calibration_weight = 0.1
```

**`L_PRM` = 带类权重的 3 类交叉熵**(`phase2_prm.py:681-689`,`:721`):

- 3 类目标由连续 `r_KG` 按 ±0.5 分桶得到(`_label_to_class`,见 §2.1.3)。
- **[修订 2026-08-19 新增]** 权重不是均匀的:`class_weighted_loss=True`(默认)时按
  **逆频率**取权重、对均值归一化、再截到 `max_class_weight=10.0`
  (`w_c = clamp(N/(3·n_c) / mean, ≤10)`)。这是对付 NEUTRAL 占绝对多数(见 §7)的关键
  手段,旧版文档完全没提,而它直接决定 PRM head 是否退化成常数预测器。

**`L_calibration` = BCE(α, kg_has_verdict)**(`phase2_prm.py:728-733`):

- **[修订 2026-08-19]** 目标**不是** `coverage > 0.5`,而是
  `kg_has_verdict = (labels_class != NEUTRAL)` —— "该步 KG 是否给出了非中性判决"。
- 代码里给出的改动理由(fix #2):旧目标 `coverage>0.5` 与门控的输入特征耦合,门控可以
  把某个输入特征直接抄成输出来最小化该项,校准退化为恒等映射。新目标独立于三个门控输入。
- **这改变了 α 的语义**:从"问题级 KG 覆盖度的校准"变成"步级 KG 是否可判决的校准"。
  论文里解读 α 分布时必须按后者表述,不能再写成"α 反映 KG 覆盖度"。

另可选训练一个辅助 text_reward_head(Linear→Tanh,对 ±1 质量目标做 MSE)。

**产出**:`alpha_gate.pt`、`prm_head/`(LoRA adapter + prm_head.pt)、
`text_reward_head.pt`、`silver_with_logprobs.jsonl`。

> **[修订 2026-08-19 — 关键口径] 这三个权重产出里,只有 `alpha_gate.pt` 会进入
> Phase 3 的奖励函数。** `phase3_ppo.py:705-709` 只 `load_state_dict` 了 α-gate;
> `prm_head/` 与 `text_reward_head.pt` 在正式 PPO 运行中都没有被加载
> (`launch_split_ppo.sh:118` 用 `--text_reward_backend rearag`,走 ReaRAG-9B 打分器;
> `--text_reward_fallback_path` 未传)。也就是说训练出的 3 类 PRM head **不参与**
> `R_KG` 的计算,`R_KG` 全程由规则标注器 `PRMAnnotator.label()` 现算。详见 §2.3.2。
> 注:`scripts/deploy/*.sh` 里的 `--text_reward_fallback_path .../prm_head` 是把
> `prm_head/` 当 **文本奖励** 的 fallback 骨干用,即便走这条路它服务的也是 `R_Text`,
> 不是 `R_KG`。

**[修订] 实现说明**:24GB 卡上 `phase2_prm_try.py` 在运行时把包内两个 bf16 加载函数
monkeypatch 成 4-bit NF4,再调用未改动的包内 `run_phase2`(复用整个训练循环)。

---

### 2.3 Phase 3 — 自适应过程监督 RL

#### 2.3.1 Phase 3a — SFT(学格式)

PPO 前先 SFT,让 Student 学会 `[Step N]…[Final Answer]` schema,否则 rollout 产不出
可解析轨迹。只用 accepted 轨迹,且**丢弃 `label ≤ −0.5` 的步**(不教幻觉)。
**[修订 2026-08-19]** 阈值不是 `label == −1`:标签是连续值(§2.1.3),等值比较会漏掉严格
介于 −1 与 0 之间的负步;`phase3_sft.py:78` 用 `float(step.label) <= -0.5`。
**[修订]** `phase3_sft_try.py` 用 4-bit QLoRA + gradient checkpointing + 可调 max_length;
`--merge_output` 训练后用 bf16 重载基座合并 LoRA,产出完整模型供 PPO 直接加载
(解决 SFT 产出是纯 adapter、PPO `from_pretrained` 无法加载的衔接断层)。

#### 2.3.2 复合每步奖励 **[修订 2026-08-19 — 公式按代码补全]**

实现在 `kgproweight/reward/composite_reward.py`。**完整形式如下**,旧版公式缺了两个
缩放乘子和末步的两个门,已补齐:

```
# 每一步（:99-101）
R_total(t) = ( α_t · R_KG(t) + (1 − α_t) · R_Text(t) · c_text ) · c_step
             c_text = text_reward_scale = 0.3
             c_step = step_reward_scale = 1.5

# 末步修正（:178-212）—— 二者互斥
if trajectory_valid:                       # 合法轨迹才有资格拿终局奖励
    R_total(T) += ω_outcome · EM(pred, gold)      # ω_outcome = outcome_weight = 4.0
else:                                      # 不合法则倒扣同样的量
    R_total(T) += −ω_outcome                      # = −4.0

Return: G_t = Σ_{k≥t} γ^{k−t} R_total(k),  γ = 0.95
```

- `R_KG(t) ∈ {−1} ∪ [0, 1]` —— **[修订 2026-08-19]** 是 **连续** KG 效用分,不是 `{+1,0,−1}`;
  见 §2.1.3。由规则标注器 `PRMAnnotator.label()` **在 rollout 时现算**
  (`composite_reward.py:95`),**不是** Phase 2 训出的 3 类 PRM head —— 那个 head 不进
  奖励路径(见 §2.2.3 末尾的口径框)。方法名里的 "PRM" 在 RL 阶段指的是这个规则标注器。
- `R_Text(t)` —— 冻结的 ReaRAG-9B 打分器(正式运行用 `--text_reward_backend rearag`),
  或 fallback 的 Llama-3-8B 奖励头;输出经 tanh 映射到 **[−1,1]**。
  **[修订 2026-08-19]** 旧版此处写"与 R_KG 量纲一致(已核实无需额外归一化)",与同文档
  §2.3.3 列出的 `text_reward_scale=0.3` 自相矛盾。**代码站在 0.3 这一侧**:R_Text 在
  进复合式前额外乘 0.3(注释理由:R5 期 R_text 会盖过 R_KG)。两个量纲**不**一致,
  论文里必须写出这个 0.3。
- `R_outcome` —— **[修订]** EM 比的是 `metadata["gold_answer"]`(真 gold),
  而非教师自己的答案 `traj.answer`(否则会奖励模仿教师的错误答案)。EM ∈ {0,1},
  所以该项要么 0 要么 `ω_outcome`。
- **`trajectory_valid` 门 [修订 2026-08-19 新增]** —— 纯**格式**谓词
  (`reward_function.py:220-273`),不判事实正确性。五个条件全满足才为真:
  ① ≥ `min_valid_steps`(=2)个可解析 `[Step N]`;② 能抽出 `Final Answer`;
  ③ 步号连续 1,2,3…;④ 每步文本非空;⑤ 每步 `Reasoning:` 正文 ≥
  `min_reasoning_chars`(=20)字符(且必须存在 `Reasoning:` 字段)。
  它的作用是让终局奖励**有条件**,防止 PPO 只吐一个裸答案就领走大奖。
  **本方法没有独立的格式奖励项,格式约束完全由这个门承担。**
- **`−ω_outcome` 无效惩罚 [修订 2026-08-19 新增]** —— 代码注释记录的理由:若不合法只是
  "拿不到奖励"(+0),PPO 会发现"闭嘴"最优 —— 短输出 KL 代价低,而合法长输出会因 KL 拿到
  负回报,于是 `invalid → 0 > valid → 负`。所以惩罚必须大于一条合法轨迹的 KL 代价
  (~15/300 token)。若一步都没解析出来,则追加一条只带该惩罚的合成步(`:208-212`)。
- **这两个门会改变最优策略的形状**,不是可以省略的实现细节。论文的奖励公式必须写全,
  否则读者无法复现,消融也无法解释 `valid_rate` 为什么是 PPO 稳定性的金丝雀。

#### 2.3.3 PPO + GAE **[修订 — 论文核心机制的代码修复]**

超参:lr=1e-6,batch=4,mini_batch=1,ppo_epochs=1,clip ε=0.2,γ=0.95,λ(GAE)=0.95,
vf_coef=0.5,kl_coef=0.15,target_kl=40.0(逐 token 序列和),total=3000 trajectories(750 updates)。
奖励权重:outcome_weight=4.0,step_reward_scale=1.5,text_reward_scale=0.3(已从 Plan B 的
8.0/0.5 回退,见 `10_口径统一清单.md` ①)。SFT anchor:weight=0.10,interval=10。

**P0-1 修复**:每步奖励 `R_total(t)` 必须真正进入 GAE。原实现把每步奖励 `.sum()` 成
一个标量,而 TRL 0.11.4 的 `compute_rewards` 只把标量放在**回答最后一个 token**上,
GAE 因此退化为 outcome-only,**Theorem 2 失去代码基础**。

修复:`StepRewardPPOTrainer(PPOTrainer)` **只 override `compute_rewards`**——
把每步 `R_total(t)` 铺到该步 `[Step N]` 区间的末 token(最后一步再叠加 R_outcome),
通过 side-channel `set_pending_step_rewards` 注入 per-token 奖励张量;TRL 原本的
逐 token KL penalty、`compute_advantages`(GAE)、minibatch、clip 全部复用。
这样 GAE 在真正的逐步信号上做信用分配。

**Critic [修订 2026-08-19 — 按代码校正]**:Critic 是 **TRL 自带的
`AutoModelForCausalLMWithValueHead`**(`phase3_ppo.py:232` 导入、`:269` 实例化),
**不是** `PRMValueHead`。`PRMValueHead` 在全仓库**从未被实例化**——只存在类定义
(`kgproweight/reward/prm_value_head.py:13`)、`__init__` 导出,以及
`CompositeRewardModel.__init__` 里一个可选参数 `prm_value_head`,它被存进 `self`
(`composite_reward.py:64`)后**全文再无任何读取**。旧版文档"Critic 用 PRMValueHead"
不成立;若论文想主张"用 PRM 初始化 Critic"这一设计点,需要先把它接进代码并重训,
不能按现状撰写。

**参考模型 [修订]**:LoRA 时 `ref_model=None`,TRL 自动用"禁用 adapter 的 policy"当
参考模型;原 `create_reference_model` 复制第二个全量 8B 是 24GB OOM 的根因。

**耦合提示**:override 复制了 TRL 0.11.4 的 `_kl_penalty`/`kl_ctl` 逻辑;升级 TRL
需重新同步该方法。

---

## 3. 与旧版的关键差异

| 维度 | 旧 paper_design / 早期包内实现 | 当前代码(包内,正式实验路径) |
|---|---|---|
| 步标注 | 原版 PRMAnnotator,实跑退化二分类(0% NEUTRAL),凑数+1 24%、−1误报 11/16 | 相关性约束 + 弃权护栏 + **连续 KG 效用分**(`precision × relevance`,§2.1.3);三分类只在 Phase 2 的 ±0.5 分桶处出现 |
| 步标签分布 | — | `+1:40% / 0:54% / −1:6%` **⚠️ UNKNOWN** —— 该数字与实测的 `80.4% / 15.4% / 4.2%`(全 80,120 步)冲突,可能是 accepted-only 与全量的口径差。**本轮未实跑核对,不得直接引用**,须按 accepted-only 重算后回填 |
| 银标接受 | 硬性 triple_rate/coverage 阈值拒绝 | 分层分桶 + 配额,保留 kg_sparse(α→0 监督) |
| f_entropy | 硬编码 0.5 / PPO 期 1.0 | 真实 token logprob |
| R_KG 来源 | 原版标注器 | **规则标注器** `PRMAnnotator.label()` 现算;Phase 2 训出的 3 类 PRM head **不进奖励**(§2.2.3) |
| PRM 损失 | 无权重 3 类 CE | 3 类 CE + **逆频类权重**(clamp≤10) |
| α 校准目标 | `BCE(α, coverage>0.5)` | `BCE(α, labels_class ≠ NEUTRAL)`(§2.2.3) |
| 奖励公式 | `α·R_KG + (1−α)·R_Text` + EM | 同式再乘 `text_reward_scale=0.3` / `step_reward_scale=1.5`,末步加 `trajectory_valid` 门与 `−outcome_weight` 惩罚(§2.3.2) |
| Critic | PRMValueHead | TRL `AutoModelForCausalLMWithValueHead`;`PRMValueHead` 从未实例化(§2.3.3) |
| outcome 对比 | 教师答案 | 真 gold(metadata) |
| per-step→GAE | `.sum()` 成标量(outcome-only) | StepRewardPPOTrainer 逐步进 GAE |
| 参考模型 | 第二个全量 8B | ⚠️ **待定,见文首"尚未处理"项** —— 本文档 §2.3.3 写 `ref_model=None`(LoRA 禁 adapter 共享),而 `phase3_ppo.py:301-305` 显式 `create_reference_model(policy)` 并注明理由(KL 须锚 SFT 而非裸基座)。两者相反,需研究者定调后再改 |

> **[修订 2026-08-19]** 上表右列原标题是"本修订版 / try 实现"。这是本轮审计发现的
> 根源性混淆:上一版文档描述的是 `scripts/train/try/` 下的冒烟实现,而 24,998 条银标、
> Phase 2 训练、PPO 奖励走的全是**包内** `kgproweight/` 实现,两者在标注返回类型上已经
> 分叉(§2.1.3 的对照表)。右列现已统一改为描述包内代码,即正式实验真正跑的那一份。

---

## 4. Experimental Design

- **语料/检索**:Wiki18 100w(~15M passages);E5(dense)+ BM25s(sparse)RRF(k=60)top-50。
- **KG 检索**:Wikidata SPARQL,K_e=30,2-hop,仅用于 PRM 奖励与 α 特征。
  ⚠️ **[2026-08-19 待定调]** "仅用于 PRM 奖励与 α 特征"与代码不符:
  `kg_proweight_pipeline.py` 默认 `inject_kg=True`,推理时把三元组拼进
  `[Knowledge Graph Context]` 送入 prompt,而主消融轴"有 KG / 无 KG"正建立在推理期
  注入之上。当前实验实际是"训练期正则 + 推理期输入增强"的混合。定调前本条不得按字面
  写入论文(与 `RESEARCH_WORKFLOW.md` §3.4 同一问题)。
- **模型**:Teacher=deepseek-v4-flash;Student=Llama-3-8B-Instruct;
  PRM=Llama-3-8B+LoRA+value head+α-gate;Text reward=ReaRAG-9B(冻结)或 fallback 头。
- **数据集**:HotpotQA / 2WikiMultiHopQA / MuSiQue;D_dropout(切断答案路径桥接三元组,1000 条)。
- **指标**:EM、F1(FlashRAG);IHR(deepseek-v4-pro LLM-as-Judge + Cohen κ);α 分布;数据效率曲线;
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

> **[修订 2026-08-19]** 本节的 `~54%` / `~6%` / `~30%` 三个占比与 §3 表中标 UNKNOWN 的
> 标签分布是同一组待核数字。实测口径(`silver_v1_reannotated.jsonl` 全 80,120 步)给出的是
> `0: 80.4% / +1: 15.4% / −1: 4.2%`,与本节的 accepted-only 口径差距很大。**本轮未实跑
> 核对,以下百分比一律不得直接写入论文**,须先按 accepted-only 重算并记录口径。定性结论
> (NEUTRAL 占多数、−1 很薄)在两种口径下方向一致,可以保留。

- **0/NEUTRAL 占比高**(accepted ~54%,**待核**):HotpotQA 上真正"KG 强支撑"的步本就少,多数推理
  靠 passage + 世界知识。这是数据固有特性,非 bug;解读 α 分布时需说明。
- **−1 类很薄**(~6%,**待核**):α 负向极端监督弱,α→0 区域主要靠 0 + kg_sparse 桶支撑。
- **~30% accepted 轨迹零 +1**(**待核**):纯文本/世界知识推理,PPO 里奖励由 R_Text 主导。
- **−1 残留误报**(~3-4/80):弃权护栏正则未覆盖全部措辞;跑 IHR 前可收紧。
- **数据规模**:银标 24,998 条轨迹 / 80,120 步,accepted 9,839(39.4%)。此前的"50-80 条"
  是冒烟阶段快照,已过时(见 `statistics.md` §十)。
