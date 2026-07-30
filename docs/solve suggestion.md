
一、总判断与执行优先级
======================

当前 R9 v5（500 步）平均 EM=0.240，低于 Elite SFT 的 0.257。现有结果只能说明 outcome_weight=10、step_reward_scale=0.3 等改动使 R9 v4 的严重退化得到恢复，不能证明 KG reward 已经产生稳定增益。

必须按以下依赖顺序处理：

P0-A  修复实体链接：错误 QID 会同时污染 KG prompt、KG reward 和 alpha-gate。
P0-B  阻断 reward 的“模型输出自证”循环：不能让模型先生成错误实体，再用该实体的真实 Wikidata 三元组给自己得分。
P1-A  重建问题相关、可排序、可版本化的 KG 缓存。
P1-B  将检索改为“top-50 候选池 → reranker → top-10/15 prompt”，统一训练与推理。
P2    重新生成 question_kg_index 和银标 passage，先做 SFT 推理消融。
P3    在数据链确认改善后，从同一 Elite SFT 起点重新训练 PPO；KL 稳定后才扩到 2000 步。

不要先做的事情：

1. 不要只把 prompt 的 retrieval_topk 从 15 改为 50。50 篇 passage 会再次造成上下文截断。
2. 不要只加一个全局 relation 白名单。高频 relation 不等于无用 relation。
3. 不要直接继续现有 R9 v5 checkpoint 跑 2000 步。当前数据链存在系统性错误，继续训练会固化错误策略。
4. 不要用 5 个案例推断全数据集错误率或召回率。


二、问题一：KG 缓存噪声
========================

2.1 已确认的代码原因
------------------

1. scripts/prepare/04_prewarm_wikidata_cache.py 创建 WikidataSubgraphRetriever 时没有传 relation_filter。
2. kgproweight/kg/wikidata_retriever.py 的 SPARQL 查询使用 LIMIT 30，但没有 ORDER BY，返回的不是“最相关 30 条”。
3. relation_filter 只作用于 1-hop；2-hop 查询完全未过滤。
4. relation_filter 当前生成 wd:Pxx，直接属性 URI 应使用 wdt:Pxx。
5. SubgraphCache 的键只有“QID_hops”，没有包含过滤策略版本、max_neighbors 和排序版本。修改规则后旧缓存仍会命中。
6. PPO prompt 最多只输入前 30 条 KG；推理最多输入前 50 条。如果缓存无排序，有用事实可能在截断位置以后。
7. 当前 preflight 只检查缓存是否存在、命中率、平均三元组数量，没有检查三元组是否正确或与问题相关。

2.2 正确解决方案：原始图缓存与问题级缓存分离
----------------------------------------

建立两级缓存：

A. entity_subgraph_cache
   key = qid + hops + retrieval_policy_version + max_neighbors
   value = 经过基础清洗的实体子图

B. question_kg_index
   key = question_id（优先）或标准化 question hash
   value = 已链接实体、候选分数、排序后的 top-K 三元组、构建版本和诊断信息

禁止继续使用纯问题字符串作为唯一主键。推荐格式：

{
  "question_id": "hotpotqa_xxx",
  "question": "...",
  "linked_entities": [
    {"mention": "Big Stone Gap", "qid": "...", "score": 0.91,
     "margin": 0.34, "type": "film"}
  ],
  "triples": [
    {"h": "...", "pid": "P57", "r": "director", "t": "...",
     "score": 0.88, "hop": 1, "source_qid": "..."}
  ],
  "builder_version": "r9v6-kg-1",
  "relation_policy_version": "rel-1"
}

2.3 Relation 策略
-----------------

采用三层策略，而非单一白名单：

第一层：硬删除

- Wikimedia category、topic's main category
- 页面维护、模板、外部数据库 ID
- Wikimedia disambiguation/list/category 等元数据关系
- 无英文 label、空实体、self-loop

第二层：配额限制

- instance of：每个源实体最多 2 条
- subclass of：每个源实体最多 2 条
- has part/part of：每个源实体最多 3 条
- 同一 PID：最多占最终 top-K 的 20%

第三层：问题相关性排序

每个三元组评分：

score = 0.30 * entity_anchor
      + 0.25 * relation_question_similarity
      + 0.25 * triple_question_similarity
      + 0.10 * path_connectivity
      + 0.10 * retrieved_evidence_support

其中：

- entity_anchor：head/tail 是否来自问题中的高置信实体；
- relation_question_similarity：关系是否符合问题类型，如 nationality→P27；
- triple_question_similarity：question 与“head relation tail”的语义相似度；
- path_connectivity：是否连接两个问题实体或位于有效两跳路径上；
- retrieved_evidence_support：top passage 是否同时出现相关实体/关系表述。

最终 prompt 建议保留 20–30 条，而不是 80–100 条。

2.4 必须修改的代码
------------------

文件：kgproweight/kg/wikidata_retriever.py

1. 1-hop filter 中将 wd:Pxx 改为 wdt:Pxx。
2. 2-hop 对 ?p1 和 ?p2 都应用 relation policy。
3. 原始查询结果先去重，再进行基础清洗。
4. 缓存键加入 policy_version 与 max_neighbors。
5. 不依赖 SPARQL 原始顺序决定 prompt 顺序。

文件：scripts/prepare/04_prewarm_wikidata_cache.py

1. 预热阶段只构建 entity_subgraph_cache。
2. 记录成功、失败、歧义、abstain、每种 PID 频次和版本信息。
3. 不再把“缓存命中”当成“缓存质量”。

新增文件：scripts/prepare/06_build_question_kg_index.py

职责：

1. 读取数据集问题与 question_id；
2. 调用上下文实体链接器；
3. 读取实体级原始子图；
4. 根据问题进行三元组排序和 relation 配额控制；
5. 输出 question_kg_index_v2.jsonl；
6. 输出 kg_build_report.json。

文件：scripts/r9_preflight.py

新增检查：

- entity linker 高置信率、abstain 率；
- taxonomic relation 占比；
- 每题 KG token 数；
- useful-triple Recall@30；
- builder/policy 版本一致性；
- question_id 覆盖率；
- 禁止读取旧版无版本缓存。

2.5 验收标准
------------

- instance of + subclass of 占最终 prompt KG 的比例低于 25%，但不要求为 0。
- 人工标注 100 题上，top-30 KG 中至少一条关键支持事实的覆盖率较当前提升 15 个百分点以上。
- KG prompt P95 token 数受控，不发生 question、passage 或 KG 静默截断。
- 旧缓存与新策略不会混用。


三、问题二：实体链接偏移
========================

3.1 已确认的代码原因
------------------

当前 EntityLinker 的 Wikidata fallback：

1. 输入只有 mention，没有完整 question；
2. Wikidata Search 返回 top-5，但代码直接选择第一名；
3. exact cache 使用全局 mention→QID；
4. fuzzy cache 也可能复用近似名称的错误 QID；
5. exact cache hit 的 link_confidence 返回 1.0，即使 QID 本身是错的；
6. GENRE 也仅处理 mention，并最终通过同一个 Wikidata Search 映射 QID。

因此 Big Stone Gap 会优先链接为城镇，Corliss Archer 会链接为消歧义页面。

3.2 正确解决方案：候选生成、上下文排序、可拒答
------------------------------------------

实体链接 API 改为：

link_single(
    mention,
    question,
    retrieved_titles=None,
    expected_types=None
) -> LinkResult

LinkResult 至少包含：

- selected_qid
- selected_label
- description
- instance_of
- score
- second_score
- margin
- abstained
- candidate_list
- linker_version

候选生成：

1. Wikidata wbsearchentities top-10；
2. Wikipedia/语料标题 exact match 候选；
3. alias match；
4. 可选 GENRE 候选。

候选排序：

score = 0.30 * mention_match
      + 0.30 * context_description_similarity
      + 0.20 * type_compatibility
      + 0.10 * retrieved_title_support
      + 0.10 * entity_coherence

负向规则：

- Wikimedia disambiguation page：强惩罚；
- Wikimedia category/list：强惩罚；
- 问题明确说 film，但候选类型为 town：类型冲突惩罚；
- 问题明确说 person，但候选为 page/category：类型冲突惩罚。

abstain 规则建议初值：

- top1 score < 0.65 → 不链接；
- top1-top2 margin < 0.10 → 不链接；
- 明确类型冲突 → 不链接。

无 KG 好于错误 KG。abstain 时 alpha-gate 应降低 KG 权重、回退文本证据。

3.3 缓存修复
------------

禁止继续使用全局裸 mention→QID 作为最终决策缓存。

候选缓存：

candidate_cache_key = normalized_mention + search_language + linker_version

决策缓存：

decision_cache_key = normalized_mention + question_id + linker_version

缓存中同时保留 score、margin、type 和候选列表，便于审计。

对已知错误可加入临时人工修正表，但人工表只能作为紧急补丁，不能作为论文中的主要方法。

3.4 阻断“模型输出自证”奖励
--------------------------

当前 reward_function 会从模型输出提取实体、获取其 Wikidata 子图，并在 dynamic_kg 非空时完全替换问题 KG。这会产生如下 reward hacking：

模型生成错误实体
→ 系统检索错误实体的真实 KG
→ 模型引用该真实三元组
→ PRM 判定 citation verified
→ 错误推理得到正奖励

必须改为：

question_kg = 来自 question_kg_index 的锚定 KG
generated_expansion = 模型输出实体扩展 KG

只有满足下列条件之一的 generated entity 才能进入 reward 图：

1. 与 question entity 在 2-hop 内连通；
2. 出现在检索 top passage；
3. 上下文实体链接得分和 margin 均达标。

最终：

kg_for_reward = question_kg + allowed_generated_expansion

禁止 dynamic_kg 完全替换 question_kg。

3.5 修复 R9 v5 的 relevance
---------------------------

现有 _triple_relevant() 没有 question 参数，并将 step.raw_text 作为 reasoning。raw_text 包含 Knowledge Used，引用三元组本身通常已经出现在文本中，因此 lexical relevance 很容易恒为真。

应将 ParsedStep 拆成：

- reasoning_text
- cited_triples
- conclusion_text

relevance 输入必须是：

- question；
- reasoning_text（不含 Knowledge Used）；
- conclusion_text；
- retrieved evidence。

推荐公式：

r_kg = verification_precision
     * question_relevance
     * evidence_grounding

如果三元组真实但与问题无关，应得到 0 或 neutral；如果实体本身不属于问题证据链，不得仅因为其在 Wikidata 中真实就获得正奖励。

3.6 必须修改的代码
------------------

- kgproweight/kg/entity_linker.py
  返回候选、上下文分数、margin 和 abstain，而不是第一个 QID。

- kgproweight/kg/cache.py
  区分候选缓存与问题上下文决策缓存；缓存格式版本化。

- kgproweight/training/reward_function.py
  reward KG 以 question KG 为锚，不允许生成实体无条件替换。

- kgproweight/reward/prm_annotator.py
  label() 接收 question 和 evidence；relevance 排除 citation 文本。

- kgproweight/pipeline/kg_proweight_pipeline.py
  推理时传入完整问题、检索标题和预期类型。

3.7 验收标准
------------

- 人工标注至少 200 个歧义 mention，报告 Acc@1、错误率、abstain rate。
- 目标 Acc@1 ≥ 90%；高置信错误率 ≤ 3%。
- Big Stone Gap、Corliss Archer 两个已知案例必须通过回归测试。
- disambiguation/category/list 类型不得作为高置信最终实体。
- 错误实体生成不得通过动态 KG 获得正 R_KG。


四、问题三：文本检索召回不足
============================

4.1 已确认的配置问题
------------------

1. kgproweight/retrieval/hybrid.py 的 DEFAULT_TOPK=15。
2. configs/retrieval/hybrid_rrf_top50.yaml 写的是 50，但只有显式加载该配置才生效。
3. README 的评测命令没有传 --config，因此按默认 top-15 运行。
4. Phase 1 的 retrieval_top_k 虽配置为 50，但 _build_retriever() 没把它传给 build_flashrag_config，底层仍只返回默认 15。
5. PPO 不会重新检索，而是读取银标文件中已固化的 retrieved_passages。只改 YAML 不会改善现有 PPO 数据。
6. 直接将 50 篇文档放入 prompt 会超过上下文预算，仓库此前已经因这一问题将默认值降到 15。

4.2 正确架构
------------

阶段一：候选召回

- E5 dense top-100
- BM25 top-100
- 通过 doc id 去重
- RRF 融合，保留 top-50

阶段二：精排

- Cross-encoder 对 question-passage 成对打分
- top-50 → top-10，或在 token 预算允许时 top-15
- 推荐先试 BAAI/bge-reranker-v2-m3

阶段三：多跳查询扩展

- 原问题查询一次；
- 生成 1–2 个子问题；
- 每个子问题分别检索；
- 合并候选后统一 rerank；
- 保证不同子问题至少各保留 2 篇文档，防止一个 hop 占满结果。

阶段四：prompt 预算

不要固定只按文档数量截断，应按 token 预算打包：

max_input_length = 6144
reserved_generation = 384
reserved_instruction_and_question = 700
reserved_kg = 1200
passage_budget ≈ 3860 tokens

按 rerank 顺序加入 passage，直到达到 passage_budget；记录实际加入数量和是否截断。

4.3 Reranker 的正确接入方式
---------------------------

FlashRAG 已有 reranker 实现，但 merge_method=rerank 会直接重排多个 retriever 的候选并集，不等价于严格的“RRF top-50 → rerank”。

建议新增明确的两阶段封装：

class RRFRerankRetriever:
    def batch_search(questions):
        dense = dense_retriever.batch_search(questions, num=100)
        sparse = bm25_retriever.batch_search(questions, num=100)
        candidates = rrf_merge(dense, sparse, topk=50)
        return cross_encoder.rerank(questions, candidates, topk=10)

配置字段必须区分：

- dense_candidate_topk: 100
- sparse_candidate_topk: 100
- rrf_candidate_topk: 50
- rerank_topk: 10
- prompt_passage_token_budget: 3860

不要继续使用一个 retrieval_topk 同时表示候选数和 prompt 文档数。

4.4 必须修改的代码
------------------

- kgproweight/retrieval/hybrid.py
  区分 candidate_topk、rrf_topk、rerank_topk；显式实现 RRF 后重排。

- scripts/train/phase1_generate_silver.py
  _build_retriever(dataset_name, retrieval_top_k)，并将 top-k 真正传入 build_flashrag_config。

- scripts/eval/run_kg_proweight.py
  默认加载 configs/eval/kg_proweight.yaml，或增加明确的候选/精排 CLI 参数；日志打印最终生效配置。

- configs/retrieval/hybrid_rrf_top50.yaml
  加入 reranker 模型、候选 top-k、最终 top-k 和 token budget。

- prompt 构造模块
  按 token budget 打包 passage，不按固定数量盲截断。

- 银标数据
  使用新 retriever 重新生成 retrieved_passages；不得继续使用旧银标 passage 做 PPO。

4.5 是否更换 Dense Retriever 的判断规则
------------------------------------

先测 Recall，再决定是否换模型：

1. supporting-fact Recall@50 < 80%
   → 候选召回不足；尝试 BGE-M3/E5-mistral、查询改写、子问题分解。

2. Recall@50 ≥ 80%，但 Recall@10 明显低
   → 候选足够，主要问题是排序；优先上 reranker。

3. Recall@10 已高，但 EM/F1 仍低
   → 检索不是主因；继续检查 KG 干扰、生成和 reward。

不要只使用“答案字符串是否在文档中”作为召回指标。yes/no 问题往往不会在文档中直接出现 yes/no。应使用：

- HotpotQA supporting-fact title Recall@K；
- 两个 supporting entity 的联合覆盖率；
- answer entity Recall@K；
- MuSiQue/2Wiki 的分 hop supporting evidence 覆盖率。

4.6 验收标准
------------

- supporting-fact Recall@50 相对当前提升至少 10 个百分点。
- rerank 后 Recall@10 接近候选 Recall@50，下降不超过 5 个百分点。
- prompt 截断率为 0；KG block 保留率为 100%。
- Phase 1、SFT、PPO、eval 使用同一套最终 passage 预算。
- 每次运行日志必须打印 dense/sparse/RRF/rerank/最终 prompt 的实际 top-k。


五、完整实施步骤
================

阶段 0：修复公开仓库可运行性
--------------------------

1. 恢复缺失的 kgproweight/data 目录。当前 .gitignore 的 data/ 会连包内目录一起忽略，应改成根目录规则 /data/。
2. pyproject.toml 使用 setuptools package discovery，确保 kgproweight.* 子包被打包。
3. 增加 question_kg_index 的正式构建脚本。
4. 固定依赖版本并运行全部离线单元测试。
5. 每个缓存和模型输出记录 git commit、config hash、数据 hash。

阶段 1：建立诊断基线
------------------

从 HotpotQA、2Wiki、MuSiQue 各固定抽取至少 100 题，保存 question_id，后续所有消融使用完全相同的题目顺序。

记录当前：

- dense Recall@5/15/50；
- BM25 Recall@5/15/50；
- RRF Recall@5/15/50；
- entity linking Acc@1；
- KG useful-triple Recall@30；
- KG relation 分布；
- prompt token 数与截断率；
- Elite SFT EM/F1；
- R9 v5 EM/F1、IHR、alpha、KL。

阶段 2：实体链接与 reward 锚定
----------------------------

1. 实现上下文候选排序和 abstain。
2. 清理/版本化实体缓存。
3. 修复模型输出动态 KG 自证问题。
4. 将 PRM relevance 改成 question-aware。
5. 添加至少以下回归测试：

- Big Stone Gap 在 film 上下文中不能链接为 town；
- Big Stone Gap 在 Virginia town 上下文中应链接为 town；
- Corliss Archer 不得链接为 disambiguation page；
- 问题无支持时允许 abstain；
- 错误生成实体的真实三元组不得产生正 reward；
- Knowledge Used 中重复三元组不能单独令 relevance=1。

阶段 3：KG 重建
--------------

1. 修复两跳 relation filtering。
2. 构建新 entity_subgraph_cache。
3. 构建 question_kg_index_v2。
4. 生成 KG 质量报告。
5. 人工检查每数据集随机 50 题和全部已知错误题。

阶段 4：检索扩召回与精排
------------------------

1. 先离线生成 dense/BM25 top-100。
2. RRF 合并到 top-50。
3. Cross-encoder 重排到 top-10/15。
4. 对比是否需要子问题检索。
5. 按 token budget 组装 prompt。
6. 重新生成银标 retrieved_passages。

阶段 5：先评估数据链，不训练 PPO
-------------------------------

使用同一个 Elite SFT checkpoint 做以下推理消融：

A. 旧检索 + 旧 KG
B. 新实体链接 + 旧检索 + 新 KG
C. 新检索 + 无 KG
D. 新检索 + 新 KG
E. 新检索 + 新 KG，但 alpha=0

只有当 D 相比 A/C 在至少两个数据集上稳定改善，且错误 KG 案例显著减少，才进入 PPO。

阶段 6：PPO 因果消融
------------------

所有实验从完全相同的 Elite SFT checkpoint 开始，先统一训练 500 trajectories/steps，并运行 3 个 seed：

E0  Elite SFT，不训练 PPO
E1  PPO，仅 outcome_weight=10，alpha=0
E2  E1 + text reward
E3  E2 + verified KG precision
E4  E3 + question-aware relevance
E5  E4 + 新实体链接和新 KG
E6  E5 + 新检索/reranker

必须报告均值、标准差、95% CI 和 paired bootstrap。R9 v5 的三项同时修改不能代替此消融。

阶段 7：决定是否跑 2000 步
-------------------------

每 250 步评估一次固定 dev 子集。只有同时满足以下条件才继续：

- KL 未持续超过 20；
- valid_rate 没有崩塌；
- dev EM/F1 没有连续两个 checkpoint 下降；
- citation 数和输出长度没有异常上升；
- reward 改善与 EM/F1 改善同方向。

如果 KL≈50，不应继续盲跑。应先检查：

- init_kl_coef 是否足够；
- adaptive KL controller 状态是否在续训时恢复；
- optimizer/scheduler 是否完整恢复；
- rollout sampling 与 TRL logprob 重算分布是否一致；
- 是否发生 reward hacking。


六、推荐配置初值
================

retrieval:

dense_candidate_topk: 100
sparse_candidate_topk: 100
rrf_k: 60
rrf_candidate_topk: 50
rerank_model: BAAI/bge-reranker-v2-m3
rerank_topk: 10
prompt_passage_token_budget: 3860
query_rewrite: false（先做基线；Recall@50 不足再打开）

KG：

max_question_entities: 5
entity_link_min_score: 0.65
entity_link_min_margin: 0.10
max_hops: 2
raw_max_neighbors: 100
prompt_max_kg_triples: 30
max_same_relation_ratio: 0.20
max_instance_of_per_entity: 2
max_subclass_of_per_entity: 2

PPO 初始实验：

total_ppo_steps: 500
outcome_weight: 10.0
step_reward_scale: 0.3
text_reward_scale: 0.3
kl_coef: 0.15（若 KL 持续过高，再做独立调整）
target_kl: 8.0
save_every_steps: 250
seed: 13/42/2024 分别训练


七、论文中必须补充的指标
========================

检索：

- supporting-fact Recall@5/10/15/50
- answer-entity Recall@K
- 分 hop evidence coverage
- reranker 前后 Recall 和 nDCG/MRR

实体链接：

- Acc@1
- candidate Recall@10
- high-confidence error rate
- abstain rate
- disambiguation-page error rate

KG：

- useful-triple Recall@K
- question-relevant triple precision
- relation entropy
- taxonomic relation ratio
- 每题 KG token 数
- KG 错误导致的答案翻转率

训练：

- EM/F1 mean±std
- 95% CI 与 paired bootstrap
- KL、valid_rate、step_rate
- alpha 分布及 D_dropout 变化
- R_KG 与最终答对率的相关性
- reward hacking 案例数


八、最终验收门槛
================

只有同时满足以下条件，才能宣称问题得到解决：

1. 实体链接高置信错误率 ≤ 3%。
2. KG top-30 useful-triple Recall 相对旧缓存提高 ≥ 15 个百分点。
3. supporting-fact Recall@10 相对旧 top-15 提高 ≥ 10 个百分点。
4. prompt 截断率为 0，KG block 保留率为 100%。
5. 500 步、3 seeds 的 PPO 平均 EM 至少高于 Elite SFT，并且 95% CI/paired bootstrap 支持改善，不只是单次高分。
6. Full SFT 仍是强基线；若 PPO 未超过 Full SFT，应如实报告“KG process reward 提高可解释性/稳健性，但未提高主任务 EM”，不能只按排名描述。
7. R9 v6 在 D_dropout 上 alpha 应下降，且文本 fallback 后性能退化显著小于固定 alpha=1。





最终结论
========

截图中的三个问题不是彼此独立的：错误实体首先生成错误 KG；无问题约束的 KG 和 reward 又会奖励真实但答非所问的三元组；文本召回不足使模型缺少纠正错误 KG 的证据。正确的修复顺序必须是：

上下文实体链接
→ 问题锚定的 KG 与 reward
→ 问题相关 KG 排序和版本化缓存
→ top-50 候选召回与 top-10/15 精排
→ 重建银标上下文
→ 再训练和评估 PPO。

仅调整 reward 权重或增加训练步数无法从根本上解决这三个瓶颈。
