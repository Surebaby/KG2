# KG-ProWeight 项目进度报告
> 截至 2026-06-11。代码位于 `scripts/train/try/`(实验性变体,不动包内 `kgproweight/`)。
> 配套技术文档见 `scripts/train/try/FRAMEWORK.md`。

---

## 一、一句话总结

整条流水线(Phase 1 蒸馏 → Phase 2 PRM/α-Gate → Phase 3a SFT → Phase 3b PPO)
**代码层面已全部打通,并在 24GB 卡上用 80 条数据端到端冒烟通过**。期间修复了
Phase 1 标注器的系统性误判、Phase 3b PPO 的 4 处与论文不符之处、以及阶段间的衔接缺陷。
**当前瓶颈是数据规模**(仅 50-80 条),距离"能看效果"的正式实验还需扩数据 + 上大显存机。

---

## 二、整体进度状态

| 阶段 | 代码 | 冒烟验证 | 备注 |
|---|---|---|---|
| Phase 1 蒸馏(改进标注) | ✅ | ✅ 80 条 | 接受率 7.6%→69%,标签三分类恢复 |
| Phase 2 PRM + α-Gate | ✅ 4-bit 变体 | ✅ 80 条 | monkeypatch 复用包内训练循环 |
| Phase 3a SFT(+merge) | ✅ 4-bit 变体 | ✅ 80 条 | 产出完整模型供 PPO 加载 |
| Phase 3b PPO(4 修复) | ✅ 4-bit 变体 | ✅ tiny+8B+全流程 | per-step 奖励真正进 GAE |
| 全流程一条龙 | ✅ | ✅ EXIT=0 | 同一批 80 条贯穿,数据流真贯通 |
| 评估 / 消融 / 定理验证 | ❌ 未在 try 重写 | — | 正式实验时对照包内 eval/ |

---

## 三、已解决的问题

### 3.1 Phase 1:接受率过低 + 标注器系统性误判

**问题**:基线 250 次尝试仅 19 条接受(7.6%);标注器实跑退化成二分类(0% NEUTRAL)。

**已解决**:
- **接受率 7.6% → 69%**:宽松答案匹配(substring/recall/别名容忍 F1,score≥0.3)、
  分层 KG 密度接受(rich/medium/sparse 配额,不再硬砍低 triple_rate)、稳健 mention
  提取(passage 标题锚点)、SPARQL 优雅降级。
- **标注器二值塌缩 → 真三分类**(80 条实测 +1:40% / 0:54% / -1:6%):
  - 凑数 +1(教师挂无关 triple 仍判+1):实测 24% → **2%**。修法:验证通过的引用还需
    至少一条 triple 实体与结论词面相关。
  - -1 误报(把"KG无此信息/无法确定"判成矛盾):实测 11/16 → **0**。修法:弃权护栏,
    只有真正反断言才 -1。

### 3.2 Phase 3b PPO:4 处与论文设计不符(对照 docs/paper_design.md)

| 编号 | 缺陷 | 影响 | 已解决 |
|---|---|---|---|
| **P0-1** | per-step 奖励被 `.sum()` 成标量;TRL 0.11.4 只把它放最后一个 token | GAE 退化成 outcome-only,**论文核心机制 Theorem 2 失去代码支撑** | `StepRewardPPOTrainer` override `compute_rewards`,每步奖励铺到该步末 token,GAE 在真 per-step 信号上跑;tiny+8B 双重冒烟验证与 TRL 对接无误 |
| **P0-2** | R_KG 用原版标注器 | 凑数/误报回流 → reward hacking | 换 ImprovedPRMAnnotator |
| **P0-3** | outcome EM 比的是教师答案 | 奖励模仿教师的错误答案 | 改比 metadata 里的真 gold |
| **P1-1** | 语义熵恒为 1.0 | α 三维特征废掉一维 | 从 generate 取真实 per-step logprob |

### 3.3 衔接与显存缺陷

- **SFT→PPO 格式断层**:SFT 产出是纯 LoRA adapter(无 config.json),PPO 的
  `from_pretrained` 无法直接加载。**已解决**:SFT 加 `--merge_output`,bf16 重载基座
  合并成完整模型再喂 PPO。
- **24GB 显存 OOM**:8B 任何 bf16 训练都装不下。**已解决**:Phase 2/3a/3b 全部加
  4-bit QLoRA 变体(`ref_model=None` 共享参考模型是 PPO OOM 的根因修复)。

### 3.4 代码审查发现的正确性缺陷(2026-06-11 修复,保留原版可对比)

四路并行代码审查后,修复了 5 个不报错但会**静默污染结果**的问题。修复都在 try
变体里(复用包内 helper,只覆盖出错的逻辑),包内文件保持不动:

| 编号 | 缺陷 | 位置 | 修复(try) |
|---|---|---|---|
| **#1** | Phase 2 在用**被拒绝的轨迹**当负样本训练(`for traj in reader` 没过滤 accepted),把"质量没问题、只是被配额挤掉"的轨迹当 `quality=-1` 喂给 α-gate | `phase2_prm.py:60` | `run_phase2_fixed` 从 `reader.accepted()` 建样本(80条→只用55条accepted的179步) |
| **#2** | α-gate 校准**退化**:同一个 `coverage>0.5` 既当输入特征又当 BCE 目标,门控只要复制一维就能让 loss→0,学不到密度/熵 | `phase2_prm.py:327,329` | 目标改为"KG 对该步是否有裁决"(label≠0),与三维输入无关;链接置信度用连续 coverage |
| **#3** | SFT 用教师答案 `traj.answer` 当 `[Final Answer]`,但 PPO 的 EM 比 `gold_answer`——两阶段优化不同目标(55条里14条不一致) | `phase3_sft.py:57` | 覆盖 `_render_assistant_trace`,优先用 `metadata["gold_answer"]` |
| **#4** | PRM 取 `last_hidden_state[:,-1,:]`,右 padding 下是 PAD 位;batch 里较短样本在 PAD 表示上训练 | `phase2_prm.py:322` | `_last_nonpad_hidden` 按 attention mask 取最后一个真 token |
| **#5** | logprob 写回循环只跳空文本不跳 label==0,开 `--binary_labels_only`(消融用)必崩 IndexError | `phase2_prm.py:261` | provenance 索引写回,binary 模式样本数 97 精确对齐 |
| #7 | SFT 丢 -1 步后 `[Step N]` 编号不连续(1,3,4) | `phase3_sft.py:53` | 连续重编号(随 #3 一起修) |

验证:#3/#7 在真实数据上单测通过(14/14 用 gold、8 条连续编号);#1/#2/#4/#5 逻辑层
在真实数据上验证通过(179样本全来自 accepted、binary 97 对齐、last-nonpad 正确)。

**GPU 端到端验证(2026-06-15,80 条 silver,epochs=1)**:`run_phase2_fixed` 跑通
**EXIT=0**,四项修复全部确认:
- **#1**:日志 `55/80 accepted, 179 steps`;`--legacy` 对照是 `257 steps`(全 80 条)——
  直接证明 fixed 不再把 25 条 rejected 当负样本。
- **#2 非退化**:GPU run 的 cal loss 持续在 0.2~0.5 波动**不坍缩**。CPU 合成对照锁死结论——
  退化版(coverage 既做特征又做目标)final cal_loss=**0.00003**(门控复制一维即可);
  fixed 版(目标=label≠neutral,与三输入独立)final cal_loss=**0.615**(无法靠复制达成)。
- **#4**:跑通无 index 错误,PRM loss 正常下降(0.72→0.14 量级)。
- **#5**:179 步 logprob 写回无越界,`silver_with_logprobs.jsonl` 正常产出。
- **如实记录**:α-gate 权重在 80 条上几乎没动(W 仅 ~0.002)。这不是 #2 没修好——cal loss
  行为已证明校准正常工作;权重不动纯因 `calibration_weight=0.1`×1epoch×80条梯度太小,
  即 §4.1 的"80 条学不到东西"老问题。**机制已修对,权重移动等正式大数据(~15k)。**

`--legacy` 可回退原版对比(注:legacy 处理 257 步,显存峰值更高,紧接 fixed 跑会因残留
显存 OOM,需间隔或单独跑)。

**误报排除**:审查曾报 `_tokenise` 的 ragged labels 会崩 collation,但 SFT 冒烟实际
已成功跑过(transformers 4.49 collator 正常 pad),判定误报。

### 3.5 PPO 奖励对齐 + 护栏修复(2026-06-11,正式训练前收尾)

在 §3.4 五个修复之后,把审查里"degrade 但不崩"的三项也在 try 里修完,确保正式训练前
全部到位。都在 try 文件,包内不动:

| 编号 | 缺陷 | 位置 | 修复 |
|---|---|---|---|
| **#6** | PPO 每步奖励 + 熵 logprob 的 token 区间建在 `decode(skip_special)` 再 re-tokenize 的坐标上,与 trainer scatter 用的 `response_ids` 不对齐,每步奖励偏几个 token——**抹糊逐步信用分配(Theorem 2 的代码基础)** | `ppo_reward_try.py` `phase3_ppo_try.py` | 新增 `step_spans_over_ids`:直接在 `response_ids` 坐标上(prefix-decode 二分定位 `[Step N]`)算区间,同一份 spans 同时喂奖励放置和熵分桶;去掉 pad/truncate hack |
| **#6b** | rollout 无可解析 `[Step N]` 时(早期常见),`records` 为空,包内 `if records:` 把 EM outcome 整个丢掉——正确答案拿零任务信号 | `ppo_reward_try.py` | 无步时把 EM outcome 落到末 token |
| **#8** | `set_pending_step_rewards` 漏设/batch 长度不符时,fallback 静默加零占位(注释误称 fail loudly),KL-only 训练不报错 | `ppo_trainer_try.py` | `_require_step_rewards` 默认严格:漏设直接 raise;`set_pending` 校验 len==batch_size |
| 弃权正则 | `_ABSTENTION_RE` 裸 `is not/was not` 把真矛盾("X is not a scientist")也当弃权,削薄本就 6% 的 -1 类 | `prm_annotator_try.py` | 收紧:copula 否定需带弃权宾语(available/found/identified/the same/in the graph…),裸否定不再命中 |

验证(全部 CPU,无需 GPU):
- #6 用**真实 Llama tokenizer** 对照——新 spans `(0,24)(24,52)` 覆盖全 52 token、末步奖励落在真末 token;旧路径 `(0,23)(24,50)` 末步早 2 个 token(即错位量),证实修复。
- #6b 单测:坏格式+正确答案→EM=1(落末 token)、坏格式+错答案→0。
- #8:漏设 pending 现在 raise;离线 fallback 测试仍通过(用 `__new__` 不触发严格模式)。
- 弃权正则:4/4 真矛盾保留、7/7 真弃权命中。
- 全部离线测试(`test_ppo_offline.py` / `test_distill_offline.py`)+ 四个改动文件 import 全绿。

### 3.6 二轮深度审查修复(2026-06-15,GPU 验证后)

GPU 跑通 fixed Phase 2 后,又派两路 agent 复查"phase 间契约 + Phase1 distill + PPO 数据准备"
这些一轮审查覆盖较弱的面,发现并修复 5 个(都已亲自核实非误报):

| 编号 | 严重度 | 缺陷 | 修复 |
|---|---|---|---|
| **B5** | critical(必崩) | §3.5 #6 修复**自身引入的回归**:`cfg.max_steps` 在 `Phase3PPOTryConfig` 未定义,真实 PPO 循环第一个 batch 就 `AttributeError`。测试只测组件、没跑训练循环故漏网 | dataclass 加 `max_steps:int=7` |
| **B3** | critical(静默) | prompt 模板顺序 Question→Passages→KG,tokenizer **右截断**;50 passages≈10k+ token 把 KG 块(252 三元组)整个截掉——**PPO rollout 时模型根本看不到 KG,静默废掉 KG-grounding 核心论点** | `_prepare_prompts` 加 `ppo_max_passages=8`/`ppo_max_kg_triples=50` 上限;`_generate` 截断前检查超长就 **warn**(不再静默) |
| **A4** | low(reward-hack 面) | PPO 在 gold 缺失时回退用 **teacher 答案**当 EM 目标——会教 PPO 去匹配 teacher 的错误 | 改为**跳过**无 gold 的轨迹(warn 计数),不回退 |
| **A2b** | low | `_ABSTENTION_RE` 的 `identified in`/`is identified` 正向断言("Einstein is identified as...")被误当弃权,否决真矛盾 | 删掉两个正向 alternative,只留否定形式 `not identified`(独立 + copula 双覆盖) |
| **A2c** | medium(削 +1 类) | `_content_tokens` 的 `len>=4` 过滤把短专名(Ulm/USA/UK)和数字(年份)从两侧都丢掉,verified+relevant 的步因无 token 重叠被 POSITIVE→NEUTRAL 降级,削薄 +1 类、污染 reward 分布 | 阈值降到 `len>=3`;`_triple_relevant` 加全实体**词边界短语匹配**兜住 len<3 尾部 |

验证:B5 字段就位、phase3_ppo import 通过;A2b 2/2 正向断言保留 + 2/2 否定命中;A2c 短/数字实体
4/4 relevance 正确;离线两套测试全绿。

**两个 design-level 决策(不是 bug,需你拍板,见正文讨论)**:Finding 2(α-gate 的
`link_confidence` 特征:Phase 2 训练用轨迹级 coverage,PPO 推理用步级 entity-linker 模糊匹配,
两者分布不一致)、A2a(伪造引用 vs 真实 KG 缺口都标 NEUTRAL,无法区分)。

### 3.7 Finding 2 修复:link_confidence 步级对齐(2026-06-15,走 A 路线)

用户定走 **A 路线**(让 Phase 2 训练对齐 PPO 推理)。实现发现这是**两部分**问题:

**(a) 计算对齐**:Phase 2 训练原本把 `_StepSample.coverage` 填成轨迹级
`metadata['coverage']`(每条轨迹一个常数);PPO 推理用步级
`compute_link_confidence(parse_steps(resp)[i].mentioned_entities, EntityLinker)`。
改:Phase 2 改用 `parsed_step_from_silver_dict(step).mentioned_entities` +
同一个 `compute_link_confidence` —— 与 PPO **同源 parser、同函数**。无需改 silver schema
(步实体本就由 parser 从 step 文本派生,不是存储字段)。

**(b) 关键二次发现 —— cache 为空,特征本是死的**:对齐后测出两端都是 **0.0000**。
根因:PPO 的 `ImprovedPRMAnnotator()` 和我初版 Phase 2 都用**空 cache** 的默认
`EntityLinker()` → `link_confidence` 横竖恒 0,α-gate 第二维(init 权重 1.5,最大)
形同废掉。改:**两端都**用 `EntityLinker(cache_path=resolve_entity_cache_path())`
加载真实 cache(`indexes/entity_cache.jsonl`,34807 条)。

验证(`tests/test_phase2_linkconf_align.py`,CPU):
- 真实 cache 下,Phase 2 与 PPO 对同一 step 文本算出的 link_confidence **逐位相等**
  (0.8052 / 0.8583 / 0.9267 / 0.8167,4/4 非零),解析出的实体集也完全一致。
- 离线两套(`test_ppo_offline` / `test_distill_offline`)+ phase2/phase3_ppo import 全绿。

改动文件(仍只在 try):`phase2_prm_try.py`(import + `_build_samples_accepted_only` 改算 +
`run_phase2_fixed` 建 linker)、`phase3_ppo_try.py`(annotator 注入带 cache 的 linker)、
`tests/test_phase2_linkconf_align.py`(新)。

> 副作用备注:实体抽取器把 "Reasoning"/"Conclusion" 这类大写词也当实体,会给抽不到
> 真实体的步(如纯弃权步)算出非零 link_confidence。这是 Phase 2/PPO **共有**的既有行为,
> 不破坏对齐(两端一致),属 entity-extractor 噪声,不在 Finding 2 范围内;若要清理需另开
> 任务且会同等影响两端。

### 3.8 实体脚手架剥离(2026-06-15,上条副作用的收尾)

用户确认收紧实体抽取。新建 `shared/entity_filter_try.py::clean_entities`,剥掉
ENTITY_RE 误抓的推理脚手架词("Reasoning"/"Conclusion"/"Therefore"/"Step"/"Final
Answer" 等;仅当整条 mention 的小写形 == 脚手架词才丢,故 "Albert Einstein" 不受影响)。

**两端同步应用**(否则刚建的 Finding 2 对齐会破):
- Phase 2:`_build_samples_accepted_only` 在 `parsed_step_from_silver_dict` 后 `clean_entities`。
- PPO:`ppo_reward_try` 在 `parse_steps` 后对每步 `mentioned_entities` `clean_entities`。
- 单一真源,两处共用同一函数。

效果(对齐仍逐位相等):Einstein 步 0.81→0.80、Nolan 步 0.93→**1.00**(只剩真实体)、
**纯弃权步 0.82→0.00**(脚手架剥光→无真实体→正确归零)。测试加
`test_abstention_step_has_no_entity` 锁死弃权步=0、`test_entities_identical` 加断言确保
脚手架不泄漏。离线两套 + 入口 import 全绿。

### 决策记录:A2a 不处理(2026-06-15)
用户定**不强调抓幻觉**,故 A2a(伪造引用 vs 真实 KG 缺口都标 NEUTRAL)**保持现状**,
不加候选 -1 区分。仅此一条说明,无代码改动。

---

## 四、仍存在 / 需要决策的问题

### 4.1 数据规模太小(最大瓶颈)

现仅 50-80 条 silver。冒烟实测 Phase 2 训完 α-gate 权重 ≈ 初始值
(W=[1.004,1.504,-0.796] vs 初始 [1.0,1.5,-0.8])——**80 条学不到东西**。
所有冒烟只验证了"管道通不通",**没有任何效果意义**。

→ 需生成中/大规模 silver(目标论文级 ~25k 尝试、~15k 接受)。

### 4.2 数据本身的结构特性(非 bug,但要在论文里说明)

- **0/NEUTRAL 占比高**(accepted ~54%):HotpotQA 上真正"KG 强支撑"的步本就少,
  多数推理靠 passage + 世界知识。解读 α 分布时需说明。
- **-1 类很薄**(~6%):α 的负向极端监督弱,α→0 区域主要靠 0 + kg_sparse 桶撑。
  若 IHR 分析依赖区分幻觉,-1 这端数据质量是短板。
- **~30% accepted 轨迹零 +1**(纯文本):PPO 里这些轨迹奖励由 R_Text 主导,
  需确保论文叙述与此一致。

### 4.3 Phase 2 当前用的是旧标注数据(衔接断点)

`checkpoints/prm_alpha_gate/alpha_gate.pt`(6/5 产出)是用**旧 PRMAnnotator** 的
silver 训的。要让 ImprovedPRMAnnotator 的干净标签贯通到 α-gate,需用新 silver 重训
Phase 2。冒烟已验证 `phase2_prm_try.py` 能跑,但需要足量数据才有意义。

### 4.4 仍有 ~3 个 -1 弃权误报(长尾)

弃权护栏的正则没覆盖全部措辞变体,50 条里残留 ~4 个、80 条里 ~3 个。-1 总量本就小,
影响有限,**暂不修**(继续堆正则有过拟合风险)。跑 IHR 分析前可再收紧。

### 4.5 TRL 版本耦合(技术债)

P0-1 的 `compute_rewards` override 复制了 TRL 0.11.4 的私有 KL 逻辑
(`_kl_penalty`/`kl_ctl`)。**升级 TRL 必须重新同步此方法**,否则 KL 通道会失配。

### 4.6 text reward model 用的是 dummy

所有冒烟用 `--text_reward_backend dummy`(返回 0)。正式实验需接真实 ReaRAG-9B
或包内已训的 llama_head text reward。

---

## 五、待完成的实验

### 5.1 立即可做(数据生成,CPU/API 为主)

- [ ] **扩 silver 数据**:用最终 ImprovedPRMAnnotator 生成中规模(先 ~2k)再大规模
      (~15k 接受)。这是解锁后续一切效果实验的前提。
- [ ] 生成后用 `tools/dump_label_record.py` 复核大样本上的凑数率/误报率
      (现 24%→2% / 11→0 是小样本结论,需在大样本确认)。

### 5.2 正式训练(需 Pro6000 96GB,bf16)

- [ ] Phase 2 用新 silver 重训(bf16 或 `--no_4bit`),产出干净 α-gate。
- [ ] Phase 3a SFT:多 epoch、完整 max_length 4096、真实数据量。
- [ ] Phase 3b PPO:数百~5000 step、真实 text reward、完整 batch_size=64。
- [ ] 三阶段用同一批大规模 silver 贯穿。

### 5.3 论文级评估与分析(需在 try 下补齐或对照包内 eval/)

- [ ] **主结果**:KG-ProWeight vs 6 个基线(HotpotQA/2Wiki/MuSiQue),统一 RRF top-50,EM/F1。
- [ ] **消融**(§7):α=0/0.5/1、binary labels、single retriever —— 均**重训**而非推理打补丁。
- [ ] **Theorem 2 方差验证**:动态 α vs 固定 α 的 advantage variance(P0-1 修复后此实验才有意义)。
- [ ] **IHR**:GPT-4o LLM-as-Judge + Cohen κ(跑前先收紧 -1 误报)。
- [ ] **α 分布分析**:D_std vs D_dropout 的 per-step α 均值/方差。
- [ ] **数据效率曲线**:F1 vs silver 规模 {1k,2k,5k,10k,15k}。

### 5.4 鲁棒性

- [ ] **D_dropout**:切断答案路径 KG 边,验证 α→0 回退(论文核心卖点之一)。

---

## 六、关键风险提示(供讨论)

1. **GPU 资源**:本机 24GB 4090 共享,常被他人占 16GB,8B 训练只能趁空跑。正式实验
   依赖 Pro6000 96GB 的可用性。
2. **效果未知**:目前 0 效果数据。方法是否有效,要等大数据训练后才能下结论。
   凑数 +1 / -1 误报已修,理论上奖励信号更干净,但需实证。
3. **数据特性可能影响卖点**:0 占比高、-1 薄是 HotpotQA 固有的;若 α 分布不够"动态",
   需考虑换/加数据集(2Wiki/MuSiQue 多跳更重,可能 KG 支撑更强)。

### 3.9 修复回灌包内 + 规模问题修复(2026-06-15,全量训练前)

**背景**:全量训练走文档化的 `make phase1..phase3-ppo`→包内 `kgproweight.training.*`,
而此前所有修复都在 `scripts/train/try/`。若不回灌,全量等于跑未修代码。用户定:
明天从 Phase1 生成 silver,且规模问题一并修。**全部已回灌包内并差分验证**(包内不再是
未修版本;try 变体保留作参照)。

**回灌清单(均对照 try 逐项 file:line 核实,差分测试通过):**
- **键石**:`kgproweight/reward/prm_annotator.py` 原地重写为改进版逻辑(保留类名
  `PRMAnnotator`,所有 import 自动生效):filler+1→0、漂移不再触发-1、缺口/稀疏→0、
  弃权护栏 `_ABSTENTION_RE`、`len>=3` 内容词 + 短实体短语匹配。差分:9 label 用例 +
  稀疏 + 轨迹全与 try `ImprovedPRMAnnotator` 逐位一致。
- **键石**:`kgproweight/data/entity_filter.py` 新增 `clean_entities`(Phase2 + PPO 共用)。
- **Phase 1**:`phase1_distill.py` 加 `StratifiedSilverFilter`/`answer_match_score`/
  `extract_mentions_robust`/`_Candidate`/`_decide_and_write`,`Phase1Config.accept_filter`
  默认改 Stratified,写全部候选;并修 `scripts/train/phase1_generate_silver.py` 调用方
  (旧 `SilverFilter` 会因新 run_phase1 调 `.decide()/.stats()` 而崩)。
- **Phase 2**:`phase2_prm.py` 改 #1 accepted-only / #2 非退化校准 / #4 last-nonpad /
  #5 provenance 写回 / Finding2 步级 link_confidence + cache。差分:179 样本、link_confidence、
  label 全与 try 一致;cache 命中 179/179 非零。
- **Phase 3 SFT**:`phase3_sft.py` `_render_assistant_trace` 改 gold 来源 + 连续重编号。
- **Phase 3 PPO**:新增 `kgproweight/training/step_reward_ppo_trainer.py`
  (`StepRewardPPOTrainer`,与 try 字节级一致);`reward_function.py` 加
  `step_spans_over_ids` + clean_entities + #6 坐标对齐 + #6b outcome fallback +
  signature 扩展;`phase3_ppo.py` 用新 trainer(去掉 `.sum()`)、gold 跳过无 gold、
  B3 passage/triple 上限 + 截断告警、B5 max_steps、P1-1 真 logprobs、P1-2 ref_model=None。
  差分:compute_rewards 数学([-0.04,-0.04,1.26,2.06,-0.04])、缺 pending 抛错、span 对齐均通过。

**规模问题(全量才暴露,均已修)**:
- **Phase 2 logprob 预pass 批处理**:原 1 forward/step(15k 步要数小时)→ 批处理 + mask
  按行均值。数值验证:批处理 vs 逐行参照 diff=0.0。
- **PPO scores OOM**:原整批累积 `out.scores`(batch64×512×128k vocab≈8GB)→ 每 prompt
  立即转 per-step logprob 后 `del out`。
- **total_steps 覆盖**:不静默改语义,加 log 报 `len(samples)/batch_size/全覆盖所需步数`。

**入口确认**:`make phase1/phase2/phase3-sft/phase3-ppo` 全部 import 包内已修 `run_*`;
PPO 入口以 kwargs 构造 config,新字段走默认,兼容。GRPO 未在范围内,但因共用包内
`PRMAnnotator` 自动获得标注器修复(无 PPO 专属 #6/cache 注入)。

**未在本机做的**:包内 Phase2 是 bf16(目标 96GB Pro6000),24GB 4090 装不下,故未在本机
跑包内 Phase2 端到端;但逻辑已与 try(已 GPU EXIT=0)差分一致,差异仅 bf16 显存(96GB 问题,
非正确性)。
