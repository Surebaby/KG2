1、银标数据
  使用新 retriever 重新生成 retrieved_passages；不得继续使用旧银标 passage 做 PPO。
2、


完整进度报告
阶段	内容	状态
0	公开仓库可运行性	⚠️ 3/5：构建脚本✅, gitignore❌, pyproject❌, 测试❌
1	诊断基线	⚠️ answer Recall@K 可测，supporting-fact Recall❌ (HF被墙)
2	实体链接 + reward 锚定	✅ 核心完成：LinkResult, abstain, KG合并, 4/4回归
3	KG 重建	⚠️ 5/5 v2 缓存完成，但 entity_subgraph_cache❌
4	检索扩召回	⚠️ BM25 reranker✅, token budget✅, cross-encoder❌, 银标重建❌
5	数据链消融	✅ 5-way ablation 完成
6	PPO 因果消融	❌ 未开始
7	2000 步判定	❌ 未开始
核心未完成项：

优先级	项目	阻塞原因
P0	更换 Cross-encoder	需下载 BGE-reranker-v2-m3
P0	重新生成银标 passages	需 GPU + 新 retriever
P1	Supporting-fact Recall	HF 被墙，需代理或离线数据
P1	PPO 因果消融 (E0-E6)	需远程 GPU
P2	Cross-encoder 替代 BM25	同上
P2	entity_subgraph_cache 重建	需 Wikidata 在线查询
要继续推进哪个？



银标数据生成现存问题
#	严重度	问题
1	Critical	Reranker 从未在 Phase 1 使用——50 篇 passages 只按 RRF 排序，没经过 cross-encoder
2	Critical	Passage 不存储 quality score——无法事后过滤或重排
3	High	Teacher 看 50 篇，PPO 只取前 15 篇——训练/蒸馏分布不匹配
4	High	无 passage 质量过滤——60 字符的短文档也存进去了
5	Medium	PPO 不重新检索——passages 冻结在 Phase 1 生成时
6	Low	Teacher prompt 无 token budget 保护
核心问题：Phase 1 生成的 passages 排序质量差（只有 RRF），PPO 取前 15 篇时拿到的不是最好的。要修——在 Phase 1 管线中接入 cross-encoder reranker，存储排序后的 passages。


结构
每条银标轨迹包含：


{
  "question": "...",
  "answer": "...",
  "gold_answer": "...",
  "retrieved_passages": [50篇文档],  ← 检索结果（冻结）
  "kg_subgraph": [三元组列表],         ← Wikidata 子图
  "trajectory": "[Step 1]...",        ← Teacher 模型生成的推理链
  "metadata": {"qid": "...", "dataset": "hotpotqa"}
}
下游用途

Phase 1 (生成)
  Teacher LLM + 检索 → 推理链 + passages + KG → 写入 JSONL
  ↓
Phase 2 (PRM 训练)
  读取 JSONL → 用推理链训练 Process Reward Model + α-gate
  ↓
Phase 3a (SFT)
  读取 JSONL → 用 question + passages + KG + 推理链 做监督微调
  ↓
Phase 3b (PPO)
  读取 JSONL → 用冻结的 passages + KG 做 prompt → PPO 训练
                         ↑
                    这里不会重新检索！
passages 一旦在 Phase 1 写入 JSONL，后续所有阶段都用这一份冻结数据。 这就是为什么 Phase 1 的检索质量如此关键——它决定了整个训练链的 passage 质量上限。



是的。分两步：


Step A: 重建 v2 缓存 (本地 CPU，不需要 Teacher API)
  1. 运行 06_build_question_kg_index.py
  2. 这会从旧 v1 数据出发，经 Python QA filter + v2 scoring
  3. 输出 question_kg_index_v2.json → PPO prompt KG 升级

Step B: 重建银标 (需要 Teacher API + GPU)
  1. 运行 phase1_generate_silver.py --rerank 10 --offline auto
  2. Cross-encoder 重排 passages + QA filtered KG
  3. Teacher 看到更干净的 KG → 生成更多有用引用
  4. 输出新 silver_trajectories.jsonl → SFT + PPO 数据升级
Step A 立刻能做。Step B 需要 Teacher API key 和大量 GPU 时间。要先做 A 吗？

V6 银标 50 题测试结果 ✅
指标	旧银标	新 V6	改善
KG 噪音	46%	15%	-67%
Traj 用 KG	79%	100%	+21pp
Avg KG/题	30	10	更精选
Avg KG 引用/题	—	2.0	每篇 2 个有用引用
Passages	50 (RRF)	10 (cross-encoder)	质量 ↑
Accept rate	56%	48%	略降(正常)
全部 24 条 accepted 轨迹都使用了 KG 引用！ 旧版只有 79%。噪音降了 67%。


已知但暂缓：SFT replay 随机配对（复杂重构）、实体缓存跨问题污染（需上下文感知缓存）、Alpha Gate 训推偏移（推理缺 logprob）。这些不影响 P0 训练链路。