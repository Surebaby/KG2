# KG-ProWeight 研究工作流速览

> 目的：让一个新会话（或新读者）在几分钟内理清整个项目的"是什么、为什么、做到哪、卡在哪"。
> 最后更新：2026-08-19。

---

## 1. 研究领域

- **检索增强生成（RAG）**，聚焦**多跳问答（Multi-hop QA）**：HotpotQA / 2WikiMultiHopQA / MuSiQue。
- **带过程奖励的强化学习（Process-reward RL）**：PPO 微调 Llama-3-8B。
- **知识图谱事实性锚定（KG-grounded factuality）**：用外部知识图谱（Wikidata）作为推理中间步骤的"事实锚"，抑制幻觉。
- **幻觉抑制（Hallucination reduction）**，尤其关注**中间步骤幻觉率（IHR）**——即使最终答案指标（EM/F1）已被 SFT 封顶。

一句话定位：**"用外部 KG 给 LLM 的每一步推理提供一个可机器验证的事实性锚点，在 RL 阶段把它当作正则化信号，从而降低中间步骤幻觉。"**

---

## 2. 研究核心问题

> **能否用一个外部知识图谱（Wikidata），在强化学习训练时对 LLM 的每一步推理给出可机器验证的奖励信号，从而降低中间步骤幻觉率（IHR）？**

子问题：
1. 如何把"原始 KG 三元组"变成"对问答真正有用的少量三元组"（KG 质量/降噪）？
2. 如何对每一步推理产出可机器验证的**连续**过程奖励（KG 效用分 = 验证精度 × 相关率），并在需要类别标签时按 ±0.5 分桶为 POS/NEU/NEG？
3. 如何自适应地加权"KG 奖励"与"文本奖励"（α-gate），而不是硬编码？
4. 训练与推理的 KG 分布必须对齐，否则 PPO 会学坏（本项目踩过的最大坑）。

**核心主张**：即便 EM/F1 被 SFT 上限封顶（统计不显著），IHR 仍然可以显著下降——这是本工作区别于"单纯刷榜"的地方。

---

## 3. 核心理论

### 3.1 三阶段训练范式
- **Phase 1 — 教师蒸馏 + KG 扎根轨迹**：Teacher（deepseek-v4-flash）生成带 `[Step N] Reasoning / Knowledge Used / Conclusion` 规范的多步推理，通过 Wikidata 拓扑验证自动标注每一步。
- **Phase 2 — PRM head + α-gate**：训练一个三路分类 PRM head（NEG/NEU/POS，带逆频类权重）+ 可学习门控 α。
  **口径（2026-08-19 核对代码）**：Phase 2 的三个权重产出（`prm_head/`、`alpha_gate.pt`、`text_reward_head.pt`）里，
  **只有 `alpha_gate.pt` 进入 Phase 3 的奖励函数**；每步 $r_{KG}$ 由**规则标注器** `PRMAnnotator.label()` 在 rollout 时现算，
  训练出的 PRM head **不参与**奖励计算。方法名里的 "PRM" 在 RL 阶段指的是这个规则标注器。
- **Phase 3 — PPO 强化学习**：在自适应复合奖励下微调学生模型（Llama-3-8B + LoRA）。
  Critic 是 TRL 自带的 `AutoModelForCausalLMWithValueHead`（不是仓库里的 `PRMValueHead`，后者从未实例化）。

### 3.2 过程奖励（PRM）：连续 KG 效用分

原标题写"三值过程奖励"、正文写"验证精度 + 结论相关性"，本身自相矛盾。按代码（`kgproweight/reward/prm_annotator.py:162`）更正如下：

- 每个被引三元组做拓扑验证，**逐条**累计两个比率：
  - 验证精度 $\text{precision}$ = 在 2-hop 子图里（模糊匹配阈值 80）找到的引用占比；
  - 相关率 $\text{relevance}$ = 与本步**推理正文 + 结论**词面相关的引用占比（head/tail 命中 +1、relation +0.5，分类关系 `instance of`/`subclass of` 永不算相关；判定前先把 `Knowledge Used:` 段从文本里剥掉，否则三元组自命中）。
- $r_{KG} = \text{precision} \times \text{relevance} \in \{-1\} \cup [0,1]$：
  - $+1$ = 全部引用都验证通过**且**全部相关；
  - $(0,1)$ = 部分分（部分可验证或部分相关）；
  - $0$ = 中性 / 子图太稀疏无法判定 / 引用全查不到 / 引用真但全不相关（凑数）；
  - $-1$ = 与前序结论硬矛盾（**唯一**负来源，不分级，没有中间负分）。
- **三分类只出现在 Phase 2 训练 PRM head 时**（`_label_to_class`：$\ge +0.5 \to$ POS，$\le -0.5 \to$ NEG，其余 NEU）；Phase 3a SFT 丢弃 $\le -0.5$ 的步；**PPO 直接消费连续值，不分桶**。

### 3.3 α-gate 自适应加权（核心公式）
$$\alpha_t = \sigma\left(\frac{\mathbf{W}^\top \mathbf{x}_t + b}{\tau}\right)$$

门控输入三特征 $\mathbf{x}_t = [f_{density}, f_{confidence}, f_{entropy}]$：
- **图密度** $f_{density} = |E|/(|V|+\epsilon)$；
- **链接置信度** $f_{confidence}$ = 实体链接器模糊匹配置信度均值（注意 Phase 2 里这个量存在样本的 `coverage` 字段，名字是历史残留，装的不是"KG 覆盖度"）；
- **语义熵** $f_{entropy}$ = 该步 token 负均值 log-prob（模型不确定性）。

α 的校准目标是 $\mathrm{BCE}(\alpha,\ \mathbb{1}[\text{该步标签} \ne \text{NEUTRAL}])$，即"KG 是否对该步给出判决"，**不是** KG 覆盖度（旧写法 `coverage>0.5` 与门控输入耦合，门控能直接抄特征，已在代码里换掉）。

**逐步奖励（按 `composite_reward.py` 实测补全，2026-08-19）**：
$$R_t = \bigl(\alpha_t \cdot R_{KG}(t) + (1-\alpha_t) \cdot R_{text}(t) \cdot c_{text}\bigr) \cdot c_{step}, \qquad c_{text}=0.3,\ c_{step}=1.5$$

**末步修正（二者互斥，$\omega_{outcome}=4.0$）**：

- 若 `trajectory_valid` 为真：$R_T \leftarrow R_T + \omega_{outcome}\cdot \text{EM}(\hat{a}, a^*)$，其中 $\text{EM} \in \{0,1\}$；
- 否则：$R_T \leftarrow R_T - \omega_{outcome}$（无效惩罚；若一步都没解析出来，则追加一条只带该惩罚的合成步）。

$$\text{回报：} G_t = \sum_{k \ge t} \gamma^{k-t} R_k, \qquad \gamma = 0.95$$

> 原公式 $R_{total} = \sum \gamma^{t-1} R_t + \omega_{outcome}\cdot\text{EM}$ 缺了两个缩放乘子（$c_{text}$、$c_{step}$）、`trajectory_valid` 门和 $-\omega_{outcome}$ 无效惩罚。后两个是**机制**而非常数：门让终局奖励有条件（防止 PPO 吐个裸答案就领大奖），惩罚防止"闭嘴最优"（不合法只是 +0 时，短输出 KL 代价低于合法长输出）。本方法没有独立的格式奖励项，格式约束全部由这个门承担 —— 这也是 `valid_rate` 成为 PPO 稳定性金丝雀的原因。
>
> `trajectory_valid` 是纯格式谓词（`reward_function.py:220`）：≥ `min_valid_steps`(=2) 个可解析 `[Step N]`、能抽出 `Final Answer`、步号连续、每步文本非空、每步 `Reasoning:` 正文 ≥ `min_reasoning_chars`(=20) 字符。

### 3.4 KG 作为 RL 正则化器
- KG 不是"更多上下文"，而是**训练期的事实性正则信号**；**推理时无需外部 KG 访问**（KG 只用于训练）。

> ⚠️ **该定位与当前代码不符，待研究者定调（2026-08-19 核对）**：`kg_proweight_pipeline.py` 默认 `inject_kg=True`，推理时把三元组拼进 `[Knowledge Graph Context]` 送入 prompt；§4 的主消融轴"有 KG / 无 KG"也正建立在推理期注入之上。也就是说当前实验实际做的是"训练期正则 + 推理期输入增强"的混合，而非纯 training-time regularizer。在定调前，本条不得按字面写入论文。

---

## 4. 变量

### 自变量（被操纵 / 消融）
| 变量 | 取值 | 说明 |
|---|---|---|
| KG 供给 | 有 KG / 无 KG（noKG） | 主要消融轴 |
| outcome_weight $\omega$ | 4.0（当前）/ 8.0（Plan B，已回退） | 最终答案权重 vs 过程信号 |
| step_reward_scale | 1.5（当前）/ 0.5（Plan B） | 过程奖励缩放 |
| KL 系数 | 0.15 | 防止策略漂移 |
| α-gate 偏置校正 | +0.78（一次性） | 恢复 α 操作区间 |

### 因变量（被度量）
| 指标 | 含义 |
|---|---|
| **EM** | 精确匹配（最终答案） |
| **F1** | 词级 F1（最终答案） |
| **IHR** | 中间幻觉率，LLM-as-Judge（deepseek-v4-pro） |
| α 分布 | 门控输出（训练用，评估时仅遥测） |
| valid_rate | 轨迹合法性门通过率（PPO 稳定性的金丝雀） |

### 控制变量（保持一致）
- 检索：混合 RRF（dense E5 + sparse BM25），top-k=10，bge-reranker-v2-m3 重排。
- 评估协议：seed=42，n=300，temperature=0。
- Teacher = deepseek-v4-flash；IHR Judge = deepseek-v4-pro。

---

## 5. 研究方法

1. **数据/训练**：Phase1 蒸馏 9,839 条接受轨迹（来自 HotpotQA 24,998 条，接受率 39.4%）→ Phase2 训练 PRM+α-gate（26,583 步样本）→ Phase3 PPO（3000 步 / 750 次优化更新）。
2. **评估协议**：每个模型在 3 数据集 × n=300 × seed=42 × temp=0 下跑 EM/F1/IHR；与 7 个基线对比。
3. **基线（7）**：zero_shot、naive_rag、trace（IRCOT/llama3-8B）、self_rag（SelfRAG-7B）、r1_searcher（Qwen-2.5-7B-RAG-RL）、corag（CoRAG-8B）、rearag（ReaRAG-9B ChatGLM）。
4. **消融**：KG vs noKG（验证 KG 贡献）；reward 权重（4.0/1.5 vs 8.0/0.5）；α-gate 消融。
5. **IHR 判定**：LLM-as-Judge（deepseek-v4-pro）逐步判定是否幻觉，聚合为平均 IHR；对 reasoning 基线（rearag/trace）做同口径抽取。

---

## 6. 目标期刊

**ICLR 2027**（投稿目标）。当前处于"把统计口径/基线搞干净"的阶段，为投稿做诚实、可复现的对比统计。

---

## 7. 研究进度

**已完成：**
- R9 v6 KG 质量修复（噪声审计 + 四层过滤 + α-gate 偏置校正 + KG key strip bug + 训练/推理对齐）。KG 从 22.8 → 10.3 三元组/题（-55%），覆盖 98%。
- SFT+KG 优于 SFT noKG：HotpotQA +1.3pp / 2Wiki +2.7pp / MuSiQue +2.3pp（**均不显著**，~4pp 噪声地板，单 seed）。
- 7 基线评估跑通（修复了 rearag 的 generation_params 被 clobber 导致 max_length=20 截断的问题）。
- **修复 MuSiQue/2Wiki 0-KG 注入**（2026-08-19）：`question_kg_index_v2.json` 原本只覆盖 hotpotqa，musique/2wiki 全部 miss 索引回退实时 Wikidata → 断网期 = 0 KG。已离线重建索引覆盖三数据集（symlink → `question_kg_index_v2_full.json`，22393 条）。
- **对齐 min_keep=5**（2026-08-19）：索引构建原先 `min_keep=0`（严格），推理 fallback 用 `min_keep=5`（放宽），同一题"索引空但 fallback 有"。已把构建脚本 `06_build_question_kg_index.py` 改成 `--min_keep 5`（默认）并离线重建。索引命中率：hotpotqa 94.5%（不变）/ 2wiki 91.7%→**96.9%** / musique 80.1%→**87.6%**。平均三元组/题 10.3→15.9。旧 min_keep=0 版备份为 `question_kg_index_v2_full.minkeep0.bak.json`。

**进行中：**
- 完整基线 EM/F1 评估（7 方法 × 3 数据集 × seed 42，n=300）在后台运行。
- 基线 IHR 判定（rearag + trace，deepseek-v4-pro judge）待 EM/F1 完成后执行。

**待办 / 已回退待重训：**
- PPO Plan B（outcome 8.0 / step 0.5）已回退到 4.0 / 1.5，**待重新训练**。
- ~~MuSiQue PPO+KG 注入 0 KG~~ 已修复（索引覆盖三数据集）；剩余 ~20% musique 空 KG 是实体抽取/过滤质量上限，非缓存问题（见 §8.9）。

---

## 8. 所遇到的问题（关键结论 / 教训）

1. **PPO Plan B 训练跑坏**：outcome 8.0 / step 0.5 使模型退化到 SFT 以下（HotpotQA -14pp；MuSiQue -17pp OOD 崩塌，IHR 31.9%→52.0%）。已回退 4.0/1.5。**教训**：过度放大 outcome 权重会杀死过程信号。
2. **训练-推理 KG 分布不匹配**：训练用原始 ~105 三元组（噪声），推理用 ~10 高质量三元组 → PPO 学会忽略 KG，2Wiki -4.6pp（p=0.005）。已通过统一过滤修复。
3. **KG 噪声高达 70%**：审计发现 ~70% 三元组对 QA 无贡献（~48% 噪声/错误 + ~21% 平凡无用）。已用四层过滤修复。
4. **银色数据质量是上游瓶颈**：Teacher=deepseek-v4-flash，仅 31% 步骤引用 KG，接受率 39.4%，PRM 80% 中性，kg_reward_share 仅 2.7% → KG 奖励信号弱。
5. **单 seed + n=300 的噪声地板 ~4pp**：所有 SFT+KG 增益（+1.3~2.7pp）都不显著，需要多 seed 才能下结论。
6. **α-gate 是评估期 no-op**：α 只在训练期参与奖励，评估时是纯遥测，不能移动 EM/F1——评审若问"α 是否有效"需谨慎表述。
7. **实体链接器覆盖缺口 ~4%**：离线缓存（35K+ 实体）+ 模糊匹配仍漏掉部分实体。
8. **KG key strip bug**：builder 剥离了 key、lookup 没剥离 → ~10% 精选 KG 丢失，修复 +1pp EM（0.245→0.255）。
9. **MuSiQue 0 KG**（已修复根因）：`question_kg_index_v2.json` 只覆盖 hotpotqa（7405 条），musique/2wiki 从未进索引 → 全部回退实时 Wikidata → 断网期 = 0 KG。实体/子图缓存其实已离线覆盖 90%+，缺口纯粹是"索引没重建到另两个数据集"。已离线重建合并（22393 条）。**剩余 ~20% musique 空 KG 的构成**：287 条无 QID（实体抽取失败，如 "UHF" 缩略词、"Caroline LeRoy" 多词名）、187 条有 QID 但 `filter_and_rank_triples`(min_keep=0) 分数阈值砍为 0、仅 6 条是真正的缓存 miss。索引构建原用 `min_keep=0`（严格），推理 fallback 用 `min_keep=5`（放宽），两者不一致——**已对齐**：构建脚本默认 `min_keep=5` 并重建，musique 命中率 80.1%→87.6%、2wiki 91.7%→96.9%（hotpotqa 不变）。剩余空 KG（musique 12.4%）只剩无 QID 与缓存 miss，在线补齐收益极小。
10. **rearag 生成截断**：baseline 的 `extras` 浅拷贝 clobber 了 `generation_params.max_tokens` → HF generate 回退 max_length=20 → "No valid answer found"。已深合并修复。

---

## 附：关键口径（诚实版）

- Teacher = **deepseek-v4-flash**（不是 GPT-4o）。
- IHR Judge = **deepseek-v4-pro**。
- PPO：lr=1e-6 / batch=4 / mini_batch=1 / ppo_epochs=1 / kl_coef=0.15 / target_kl=40.0。
- outcome_weight=**4.0** / step_reward_scale=**1.5** / text_reward_scale=**0.3**（Plan B 8.0/0.5 已回退）。
- α 是**训练期专用**；评估时仅遥测。
- 无格式奖励（用 trajectory_valid 门代替）；不合法轨迹倒扣 **−outcome_weight = −4.0**。
- 24,998 条银色轨迹，接受率 39.4%（9,839 条）。

**以下三条为 2026-08-19 代码核对后补充，写作时必须照此表述：**

- 每步 $r_{KG}$ 是**连续** KG 效用分（`precision × relevance ∈ {−1} ∪ [0,1]`），不是三值；三分类只出现在 Phase 2 训 PRM head 的 ±0.5 分桶处（见 §3.2）。
- PPO 的 $R_{KG}$ 由**规则标注器** `PRMAnnotator.label()` 现算；Phase 2 训出的 3 类 PRM head **没有进奖励路径**（Phase 3 只加载 `alpha_gate.pt`）。
- Critic = TRL `AutoModelForCausalLMWithValueHead`；仓库里的 `PRMValueHead` 从未被实例化。
- α 的校准目标是"该步 KG 是否给出非中性判决"，**不是** KG 覆盖度（见 §3.3）。

> **待研究者定调的两处不一致**（不在本轮文档修订范围）：
> ① §3.4 "推理时无需外部 KG 访问（KG 只用于训练）" 与代码 `kg_proweight_pipeline.py` 默认 `inject_kg=True`（KG 进 prompt）相反，而 §4 的主消融轴正是"有 KG / 无 KG"；
> ② `paper_design_new.md` §2.3.3 写 `ref_model=None`，代码 `phase3_ppo.py:301-305` 显式 `create_reference_model(policy)` 以把 KL 锚在 SFT 而非裸基座。
