# 多步检索本地文献核验笔记

## 1. 范围与引用口径

本文记录 `docs/paper/references/` 中与当前多步检索路线直接相关的七份 PDF，以及可从原文直接核验的机制。
下文页码均指 PDF 阅读器显示的页码；这些论文的正文印刷页码与对应 PDF 页码一致。

这些笔记只支持“某项设计受到既有机制启发”之类的定位，不自动支持以下主张：

- v6 复现了任一论文的完整方法；
- v6 的效果已经优于单轮检索或上述方法；
- 某篇论文的训练结果可以迁移到本项目的模型、语料或评估协议；
- 七份本地论文足以证明普遍性的 novelty 或 “first” claim。

## 2. 本地论文与可核验证据

### 2.1 CoRAG

- 本地文件：[corag.pdf](paper/references/corag.pdf)
- 标题：*Chain-of-Retrieval Augmented Generation*
- 原文证据：
  - PDF p.3，§3.1：第 `i` 个子查询写为
    `Q_i = LLM(Q_<i, A_<i, Q)`，即同时依赖原问题、历史子查询和历史子答案。
  - PDF p.4，§3.1–3.2：每个子答案由该子查询检索到的 top-k 文档生成；训练同时学习
    next sub-query、sub-answer 和 final answer prediction。
  - PDF p.5，§3.3：给出 greedy、best-of-N 和 breadth-first tree search 三种检索链解码。
  - PDF p.9，§5.4：额外训练 Yes/No stop prediction；作者同时报告提前停止节省 token
    可能以性能下降为代价。
  - PDF p.10，§5.5：在检索召回分析中，以 RRF 合并 retrieval chain 的多轮检索结果。
  - PDF pp.14–15，Appendix A：链生成有最大长度；相同的重复子查询会被丢弃。
- 可借鉴：原问题持续锚定、由已有检索状态产生下一查询、多候选查询分支、硬步数预算、
  重复查询抑制，以及多查询结果融合的思想。
- 边界：训练链的终止和筛选使用正确答案匹配或正确答案条件似然，不能进入 v6 的
  Gold-free 物化。§5.5 的 RRF 是论文的 retrieval-recall 分析方式，不能表述成 CoRAG
  最终回答阶段采用了与 v6 相同的 A8/CE 文档替换策略。

### 2.2 ReaRAG

- 本地文件：[ReaRAG.pdf](paper/references/ReaRAG.pdf)
- 标题：*ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models
  with Iterative Retrieval Augmented Generation*
- 原文证据：
  - PDF p.3，§3.1：reasoning chain 由 Thought–Action–Observation 三元组构成，action space
    为 `search()` 与 `finish()`，并受 `T_max` 约束。
  - PDF p.4，Algorithm 1 和 §3.2：每轮模型输入为 `prompt + question + prior chain`；检索
    observation 追加到 chain 后参与下一轮 thought/action/query。
  - PDF p.5，Algorithm 2：另维护 `Sum`，逐轮追加 `(query, observation)`；选择 Finish 后，
    独立 Answer LLM 基于问题和 `Sum` 生成答案。
  - PDF pp.7–8，§4.5：分析信息抽取失败、错误传播、过度检索和后续自我修正。
  - PDF p.9，Limitations：明确列出有限 action space、数据构造成本和迭代推理延迟。
- 可借鉴：显式保存 query/observation 状态与 provenance、让上一轮 observation 约束下一轮
  query、使用硬迭代上限，并区分检索轨迹与最终回答输入。
- 边界：Algorithm 1 的数据构造输入包括 gold documents，并用 ground-truth answer F1
  过滤轨迹；这些步骤不得用于 v6 dev/test 检索。ReaRAG 还包含 LRM 轨迹生成、SFT 蒸馏及
  独立 Answer LLM，不是当前零训练 v6 的组成部分。原文算法未明确给出达到 `T_max`
  但未选择 Finish 时的最终答案 fallback，因此该行为应记为 `UNKNOWN`，不能自行补写。

### 2.3 R1-Searcher

- 本地文件：[R1-Searcher.pdf](paper/references/R1-Searcher.pdf)
- 标题：*R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning*
- 原文证据：
  - PDF p.3，§2.1–2.2：模型在 reasoning 中用特殊 token 产生 search query；提示要求查询
    只含关键词。
  - PDF p.5，§2.2.2：提示进一步规定“一次 query 只涉及一个 triple”；检索文档插回
    reasoning context，使后续生成可以利用 observation。
  - PDF pp.4–5，§2.2.1–2.2.2：采用两阶段 outcome-based RL；训练时 mask 检索文档 token，
    避免把环境 observation 当成策略生成 token 优化。
  - PDF p.6，§3.3：实现中的最大检索次数为 8。
- 可借鉴：relation-scoped、近似单事实的原子查询，以及未来若训练交错检索策略时对环境
  token 做 loss masking。
- 边界：论文没有提出 A8 保护、固定 10 篇文档 merge 或完整问题 CE 严格替换。Stage-1
  对“至少调用一次检索”给奖励，不评价 query 或 evidence 本身的质量，不能作为当前 v6
  bridge admission 的依据。两阶段 RL 也不属于本轮 v6 组合。

### 2.4 Search-R1

- 本地文件：[Search-R1.pdf](paper/references/Search-R1.pdf)
- 标题：*Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement
  Learning*
- 原文证据：
  - PDF p.4，§3.1：PPO/GRPO rollout 中 mask retrieved tokens，只优化 LLM 生成 token。
  - PDF p.5，§3.2：模型生成 `<search>query</search>`，检索结果以 `<information>` 追加到
    累计 rollout，下一轮生成以整个已有序列为条件。
  - PDF p.6，Algorithm 1：遇到 answer token 或达到 maximum action budget 时停止；非法
    action 会追加 rethink 提示。
  - PDF p.6，§3.4：训练奖励仅为最终答案 EM。
  - PDF pp.20–21，Appendix I：案例中模型在已有足够信息后仍继续一次检索作自验证，说明
    自由多轮策略也可能产生冗余调用。
- 可借鉴：明确的 query/observation 交互协议、累计状态、硬 action budget、异常处理，以及
  未来 RL 中的 retrieved-token mask。
- 边界：Search-R1 是自由生成 query 的多轮 RL，并以 Gold answer 的最终 EM 为训练信号；
  它没有 v6 的冻结 query plan、bridge provenance、A8 保护或 CE 严格替换。不能把 v6
  称为 Search-R1 训练或 Search-R1 的直接实现。

### 2.5 Self-RAG

- 本地文件：[self-rag.pdf](paper/references/self-rag.pdf)
- 标题：*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- 原文证据：
  - PDF p.4，Table 1 和 Algorithm 1：模型基于 `(input, preceding generation)` 预测是否
    检索；retriever 的输入为 `(x, y_{t-1})`。
  - PDF p.4，Algorithm 1：对多个 passage 分别预测 relevance、support 和 utility critique。
  - PDF p.5，§3.2：critic 产生的 reflection token 离线插入训练数据，再训练 generator。
  - PDF p.6，§3.3：每次需要检索时并行处理 K 个 passage，并以 critique score 做
    segment-level beam search；也可用阈值控制 retrieval。
- 可借鉴：Gold-free passage relevance/support gate、并行保留多个候选、在接纳证据前显式
  检查相关性和支持性。
- 边界：Self-RAG 没有显式生成分解后的文本 query，其 retriever 接收原输入与上一生成
  segment；它也没有全局文档去重/merge 或明确的多跳 action budget。critic、reflection
  token 训练和 beam decoding 都是新的模型变量，不应并入当前 v6。

### 2.6 IRCoT

- 本地文件：[IRCOT.pdf](paper/references/IRCOT.pdf)
- 标题：*Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*
- 原文证据：
  - PDF pp.3–4，§3.1：先用完整问题检索基础段落；之后交替执行 Reason 与 Retrieve。Reason 每次读取
    原问题、截至当前累积的全部段落和已有 CoT，只保留新生成的第一句；Retrieve 直接用这句作为查询。
  - PDF p.4，Fig.2 与 §3.1：当新句包含 `answer is:` 或达到最大步数即终止；实验最大 8 步、总段落
    预算 15。
  - PDF p.3，§3.1：few-shot demonstrations 包含人工标注 CoT 和共同支持答案的段落；test 实例本身
    只读取当前累计检索结果，不读取其 Gold supporting paragraphs。
  - PDF p.4，§3.2：检索完成后，QA reader 读取收集到的段落回答原始问题。
- 可借鉴：不在检索前锁死完整语义 plan；每得到一批新证据才生成下一句局部推理；让原问题、历史推理和
  累积 observation 共同决定下一查询；旧证据累积保留，而不是用 full-question 相关性分数立即替换。
- 边界：IRCoT 不是显式 `Q1/A1/Q2/A2` 的 subquestion executor，而是“自然语言 CoT 句→检索 query”的
  交替控制器；它依赖少量人工 CoT demonstrations，且没有 v7 的严格 JSON、单 citation、answer-type 和
  B/C 配对 activation 门。直接移除这些门会同时改变多个变量，必须另立协议，不能称为 v7 的小修。

### 2.7 Decomposed Prompting

- 本地文件：[Decomposed Prompting.pdf](paper/references/Decomposed%20Prompting.pdf)
- 标题：*Decomposed Prompting: A Modular Approach for Solving Complex Tasks*
- 原文证据：
  - PDF pp.3–4，§3–3.2：decomposer 逐轮产生 `sub-task + sub-question` 程序；controller 把问题交给
    专门 handler，并将上一步答案写回 history 以生成下一问题，直到 `[EOQ]`。
  - PDF p.8，§4.4 与 Fig.11：开放域多跳 QA 中，decomposer 提出单跳问题，`retrieve_odqa` 同时返回
    answer 与 documents；下一子问题显式使用前一答案，最后 `multihop_rcqa` 基于累计文档回答原问题。
  - PDF pp.8–9，§4.4：作者为 20 个训练问题人工标注 CoT/decomposition，每个 prompt 抽 15 例；其
    注释还说明结构化 decomposition 使用 GPT3 级模型，因为较小模型不能可靠地产生所需结构。
- 可借鉴：把 planner、single-hop reader、retriever 和 final reader 拆成可独立训练/验证的模块；用真正
  的单跳自然语言问题作为 handler 接口，而不是只用 `subject + PID`；为 decomposer 和 subanswer handler
  提供任务专属示范或监督；上一答案经过 handler 验证后再绑定到下一问题。
- 边界：论文使用人工 decomposition demonstrations 和更大 decomposer；不能据其结果推断当前 8B 强 SFT
  会零样本掌握新 JSON 接口。其资源、模型和检索预算与本项目 baseline 也不同，需另设 matched controls。

## 3. 冻结 v6 与上述工作的关系

当前 v6 development combination 的项目定义见
[`retraining_plan.md` §33](retraining_plan.md)：
“原始完整问题锚定 + 冻结子问题/关系 + 最多两个候选 bridge”的多查询扩展；保留 Arm-A
前 8 篇、相同完整问题 CE 严格替换和总 10 篇预算。

当前独立 merge 包装见
[`dependent_merge_v6.py`](../kgproweight/retrieval/dependent_merge_v6.py)：logical hop 内保留
`query_variants`，每个 variant 独立提供至多 top-2 候选，再复用 v5 的严格 merge；最终最多
替换 2 篇，并在去重后保留 `logical_hop_id`、`query_variant_id`、`query` 和 `hint`
provenance。这里描述的是工程合同，不是效果结论。

### 3.1 相似处

- 与 CoRAG 相似：下一跳检索不丢掉原问题语义，并允许多个 query branch；同时使用硬预算
  和重复结果去重。
- 与 ReaRAG/Search-R1 相似：后续查询包含由上一轮检索产生的 observation。v6 中该
  observation 是具有来源记录的 bridge hint，而不是自由生成的 thought。
- 与 R1-Searcher 相似：冻结 relation/subquery 约束查询保持原子化，避免只使用宽泛完整问题。
- 与 Self-RAG 相似：候选不能仅因被检索到就进入最终上下文，还要经过与完整问题相关的评分门。

### 3.2 关键不同处

- v6 是 Gold-free、零训练、确定性的检索物化组合；上述 CoRAG/ReaRAG/R1-Searcher/
  Search-R1 的答案监督、轨迹蒸馏或 RL 均未引入。
- v6 不让 LLM 在线自由决定下一 query 或 Finish；query 来自冻结计划与最多两个 bridge hint，
  停止由固定计划、query/document budget 和 exact fallback 决定。
- v6 不把全部 observation 原样累积进 prompt。各 variant 的文档先按 ID 去重，再用同一冻结的
  full-question score 比较；只有严格胜过可替换的 Arm-A 尾部文档才进入最终 top-10，平分时
  Arm A 获胜。
- v6 的多 variant 不是 CoRAG 的 best-of-N chain、tree search，也不是 Self-RAG 的
  segment-level beam search；它们的候选单位、评分信号和训练方式均不同。
- v6 当前只能表述为受多步检索、observation-conditioned query 和保守 evidence selection
  启发的开发组合。通过预注册的 Gold-free 机制门、答案效用门和 fresh confirmation 之前，
  不得声称稳定提升；confirmation 还需保留检索调用数匹配的 generic expansion 对照，以区分
  结构化查询收益与单纯增加检索次数的收益。

## 4. v7“回答子问题再检索”为何未通过，以及文献真正指出的修复方向

v7并不是“没有拆出步骤”：冻结planner在HotpotQA20和MuSiQue20上均达到20/20 schema-valid且可执行。
失败发生在语义执行接口。depth-1共41个reader任务，严格JSON parse 30个，但机械验证仅6个；24个回答
被自报`answer_type=other`拒绝，10个把`abstain`写成非布尔值，另1个citation数量不符。更重要的是，
`schema-valid`不等于语义正确。例如Hotpot `dev_1107`的问题需要先找“Barrie Ciliberti任教的大学”，
但冻结step写成`educated at/P69`；`dev_1670`把“担任哪座城市的市长”写成subject人物的`position held/P39`，
无法直接得到城市。错误的冻结step会把single-hop reader导向错误的中间变量。

这与IRCoT/DecomP的关键差异是：

1. v7先一次性冻结最多4步的relation/PID plan，再看证据；IRCoT每轮看完新段落后才生成下一句推理，
   DecomP也用完整Q/A history生成下一子问题；
2. v7让一个未为该接口训练的强SFT同时遵守answer、citation、type、boolean四字段，type或JSON形式错即整跳
   失效；DecomP为各handler提供独立few-shot examples，IRCoT以自然语言单句作为宽松接口；
3. v7要求B和C在同一hop同时可用才共同激活，任一root失败会阻断后代，这是因果评估的严格控制，而非最佳
   在线系统策略；IRCoT/DecomP不会为了matched control让另一个arm的失败阻断当前系统；
4. v7最终仍保护A前8篇、总计10篇，并以完整问题CE决定替换；IRCoT累计最多15篇，DecomP把各subanswer
   对应documents显式交给最终reader，降低“补到第二跳却挤掉第一跳”的风险。

因此，当前结果不能写成“问题无法拆解”。准确结论是：冻结relation plan可以被解析，但其语义角色未经
验证；强SFT的subanswer接口契约不匹配；paired fail-closed执行又将局部失败递归放大。正式Gold-free上界
见`outputs/audits/subquestion_dependent_retrieval_v7_depth1_monotonic_upper_bound_retry1/report.json`：在锁定的
single-generation、观测到`retry_count=0`的v7轨迹中，Hotpot/MuSiQue最乐观机械验证率仅`.158/.174`，
低于`.40`门，故未打开Gold且没有EM/F1。

如果继续，不应直接“照搬IRCoT”或降低v7门。最小、可归因的顺序是：先对冻结raw responses做Gold-free
接口反事实审计（由程序推断type、严格布尔归一化是否足以越过`.40`）；若通过，再另冻新版本，从
“evidence-conditioned生成下一自然语言子问题/推理句”与“专门single-hop handler”二者中只选一个主变量。
随后必须加入相同检索调用预算的generic expansion控制，并保留canonical A臂，才能区分结构化拆解收益、
额外检索次数收益与模型调用收益。

另需修正历史比较口径：当前FlashRAG `trace`仅能称为IRCoT-style历史系统参考。对三份n=300
`intermediate_data.json`只读复核，最终累计`retrieval_result`篇数分别为Hotpot 11–67（均值26.19）、
2Wiki 11–69（28.74）、MuSiQue 12–67（28.64）；超过IRCoT论文15篇总cap的题分别为271/300、
266/300、280/300。它与本项目固定top-10实验并不预算匹配，也不是论文原算法的忠实复现。正式动态实验
必须另做matched controller；如果做论文faithful reproduction，应单独使用其语料、demonstrations、15篇cap
和独立final reader，放在不同结果表中。
