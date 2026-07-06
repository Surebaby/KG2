# KG-ProWeight: Knowledge Graph-Anchored Process Rewards for Multi-Hop Retrieval-Augmented Generation

---

## Abstract

检索增强生成(RAG)使大语言模型(LLMs)能够将多跳推理锚定于外部证据。现有方法要么通过仅评估最终答案质量的结果奖励信号训练大语言模型，要么采用来自冻结文本评分模型的逐步骤过程奖励提供更密集的监督。两种范式均未提供显式的、机器可验证的机制来验证中间推理步骤的事实正确性。我们提出 KG-ProWeight，一个三阶段训练框架，在强化学习中引入外部知识图谱(Wikidata)作为逐步骤的事实性锚点。第一阶段通过对推理轨迹进行 Wikidata 拓扑验证自动构建三值过程奖励标签，免除人工标注。第二阶段训练可学习的 α-门控，每一步动态加权 KG 衍生与文本衍生的奖励信号，在 KG 覆盖稀疏时自适应过渡至文本监督。第三阶段在此自适应复合奖励下施加强化学习。在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上，KG-ProWeight 通过监督微调取得对基座模型的显著提升，在 PPO 下保持强劲性能，并降低了中间幻觉率。消融实验证实 KG 事实性锚点对于长推理链上的非退化性能是必要的。*(实验数据持续更新中)*

Retrieval-augmented generation (RAG) enables large language models (LLMs) to ground multi-hop reasoning in external evidence. Existing methods either train LLMs through outcome-based reward signals that evaluate only the final answer quality, or employ per-step process rewards from frozen text-scoring models to provide denser supervision. Neither paradigm provides an explicit, machine-verifiable mechanism to verify the factual correctness of intermediate reasoning steps. We propose KG-ProWeight, a three-phase training framework that introduces an external knowledge graph (Wikidata) as a per-step factuality anchor in reinforcement learning. Phase 1 auto-constructs three-valued process-reward labels through topological verification of reasoning trajectories against Wikidata, eliminating manual annotation. Phase 2 trains a learned α-gate that dynamically weights KG-derived and text-derived reward signals per step, enabling adaptive transition to text-based signals where KG coverage is sparse. Phase 3 applies reinforcement learning under this adaptive composite reward. On HotpotQA, 2WikiMultiHopQA, and MuSiQue, KG-ProWeight achieves substantial gains over the base model through supervised fine-tuning, preserves strong performance under PPO, and reduces intermediate hallucination rate. Ablation confirms that the KG factuality anchor is essential for non-degenerate performance on long reasoning chains. *(Experimental results being updated.)*

---

## 1. Introduction

大语言模型在各类自然语言处理任务中取得了卓越的性能表现 [1, 2, 3]。尽管其庞大的参数量使其能在预训练期间学习丰富知识，大语言模型仍可能生成幻觉性、过时或不准确的内容，尤其在需要长尾或领域特定知识的场景中 [4, 5]。为解决这一问题，检索增强生成 (RAG) 已成为一项关键策略。通过将知识检索与骨干大语言模型显式解耦，此类架构实现了更准确、更可靠的内容生成，并在开放域问答等知识密集型任务上展现出尤为增强的性能 [6, 7, 8]。

将强化学习应用于 RAG 的现有努力可大致分为两类。第一类采用基于结果的奖励——仅评估最终答案正确性的二元或标量信号——通过 PPO 或 GRPO 引导策略优化。以 Search-R1 [9] 和 R1-Searcher [10] 为先驱，这一范式已被证明极为有效：仅通过稀疏的终端奖励训练，大语言模型即可学会自主生成搜索查询、解析检索文档并将其推理结构化为可解释的链条，在多跳 QA 基准上取得了相对于基于提示和基于 SFT 的基线的显著提升。仅靠结果奖励的成功与深度推理领域的更广泛发现一致，即即使是最小的奖励信号，结合充分的探索，也能诱导出诸如自我验证和纠错等复杂的涌现行为 [11, 12]。第二类受长程强化学习中信用分配问题的驱动，采用过程奖励——评估每个推理子单元质量的逐步骤信号——以提供更密集的监督。结果奖励模型 (ORM) 与过程奖励模型 (PRM) 的区分由 Lightman et al. [13] 在数学推理领域确立，其 PRM 在评估步骤级逻辑有效性方面显著优于仅检查最终答案的 ORM。在 RAG 领域，ReaRAG [14] 及相关方法采用冻结的文本评分模型——它们自身往往也是大型语言模型——对每一步的连贯性进行评分，作为替代过程奖励。然而，两种范式均共享一个根本性局限：均未提供显式的、机器可验证的机制来验证中间推理步骤的事实正确性。结果奖励过于稀疏，无法为单个步骤分配信用；而基于文本的过程奖励评估的是流畅性和合理性，而非事实真相。在知识稀疏或跨文档场景中，文本奖励可能主动产生误导——奖励那些风格优美但包含未获检索证据支持的虚构关系的中间结论。这一局限在多跳推理场景中尤为严重：任何中间步骤的事实错误都可能沿推理链传播并污染最终答案 [15]；我们将此现象称为**中间幻觉 (Intermediate Hallucination)**，其隐蔽之处恰在于出错的步骤可能读起来流畅且看似有理有据，使其对仅检查最终输出的评估指标不可见。

与这些 RL 发展并行，知识图谱 (KG) 作为提升大语言模型事实性的互补机制也得到了探索。现有的 KG-LLM 集成遵循三种范式：输入增强——将 KG 三元组拼接到提示中 [16]；结构感知编码——图神经网络将 KG 拓扑编码进大语言模型的隐状态中 [17]；以及事后验证——在推理时对照 KG 检查生成的声明 [18]。近期工作已开始探索将 KG 纳入 RL 训练循环——例如，使用 FactAlign 奖励在 GRPO 微调中对生成答案与 ground-truth 子图进行评分对比 [19]——但这些方法将事实性约束施加于输出层面（对最终答案进行评分）而非过程层面（对每个推理步骤单独评分）。由此产生了一个关键研究缺口：**是否可能在 RL 训练期间为每个单独的推理步骤提供显式的、机器可验证的事实性信号，而不对部署模型施加运行时 KG 依赖？**

我们认为 KG 恰好具备填补这一缺口的独特条件。KG 以确定性真值对关系事实进行编码——一个三元组要么存在于 KG 中，要么不存在——使其天然适合充当训练时的事实性校验源。我们提出的框架 KG-ProWeight 将 KG 作为 RL 奖励中的逐步骤事实性锚点引入，通过三个一体化阶段实现这一思想。首先，我们提示教师大语言模型生成带显式三元组引用的多步推理轨迹，然后对照 2-hop Wikidata 子图对每个引用的三元组进行拓扑验证，以几乎零人工标注成本生成三值过程奖励标注（验证通过为 +1，矛盾为 −1，不可验证为 0）。其次，我们训练一个可学习的 α-门控——以图谱密度、链接置信度和语义熵为条件的轻量级三特征分类器——为每一步动态加权 KG 衍生与文本衍生的奖励分量，在 KG 覆盖稀疏区域自适应过渡至文本监督。第三，我们通过监督微调后在门控的自适应复合奖励 $R_t = \alpha_t \cdot R_{\text{KG}}(t) + (1-\alpha_t) \cdot R_{\text{text}}(t)$ 下进行近端策略优化，微调 Llama-3-8B 学生模型。KG 仅在训练期间用作奖励信号；推理时，部署模型不需要任何 KG 访问。

在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上，我们进行了涵盖多个模型变体的系统性实证研究——基座大语言模型、仅 SFT、带完整 α-门控的 PPO、纯文本奖励 PPO ($\alpha \equiv 0$) 和纯结果奖励 PPO。我们的核心发现有三个方面。第一，α-门控的 KG 分支对 PPO 保持 SFT 级性能是必要的：移除它会导致策略在长推理链上退化至接近随机准确率。第二，KG 锚定奖励降低了中间幻觉率——通过 LLM-as-judge 的逐步骤评估测量——即使在最终答案指标受限于 SFT 的情况下，证明了过程级事实性约束以 EM/F1 无法捕获的方式提升了推理质量。第三，纯结果奖励在需要四跳或更多推理的任务上显著落后于 α-门控变体，揭示了稀疏终端奖励在深度多跳链上的结构性局限。总体而言，KG-ProWeight 展现出四个主要特征：

1. **KG 作为训练时锚点**：与先前将 KG 用于输入增强或事后验证的工作不同，KG-ProWeight 将 KG 仅作为 RL 训练期间的奖励信号使用——塑造策略学会重视的内容，而不约束推理；
2. **自适应事实性门控**：α-门控为每一步动态加权 KG 与文本奖励，在稀疏区域防止噪声 KG 信号破坏梯度，同时在 KG 验证可靠的区域利用它；
3. **零人工标注**：第一阶段通过对 Wikidata 的拓扑验证自动构建三值逐步骤标签，以极低的成本生成训练数据；
4. **过程级评估**：我们引入 IHR 作为核心指标，并证明 KG 锚定过程奖励即使在最终答案指标受限于 SFT 时仍能减少中间幻觉。

---

Large language models (LLMs) have achieved remarkable performance across various natural language processing tasks [1, 2, 3]. Despite their extensive parameters enabling them to learn rich knowledge during pre-training, LLMs may still generate hallucinated, outdated, or inaccurate content, especially in scenarios requiring long-tail or domain-specific knowledge [4, 5]. To address this problem, retrieval-augmented generation (RAG) has emerged as a pivotal strategy. By explicitly decoupling knowledge retrieval from the backbone LLMs, such architectures have achieved more accurate and reliable content generation and shown particularly enhanced performance on knowledge-intensive tasks such as open-domain question answering [6, 7, 8].

Existing efforts in applying reinforcement learning to RAG can be roughly categorized into two groups. The first group employs outcome-based rewards—binary or scalar signals that evaluate only the correctness of the final answer—to guide policy optimization via PPO or GRPO. Pioneered by Search-R1 [9] and R1-Searcher [10], this paradigm has proven remarkably effective: trained with only sparse terminal rewards, LLMs learn to autonomously generate search queries, parse retrieved documents, and structure their reasoning into interpretable chains, achieving substantial gains over prompting-based and SFT-based baselines on multi-hop QA benchmarks. The success of outcome-only rewards aligns with broader findings in deep reasoning, where even minimal reward signals, combined with sufficient exploration, can induce complex emergent behaviors such as self-verification and error correction [11, 12]. The second group, motivated by the credit assignment challenge in long-horizon RL, employs process rewards—per-step signals that evaluate the quality of each reasoning sub-unit—to provide denser supervision. The distinction between outcome reward models (ORMs) and process reward models (PRMs) was established by Lightman et al. [13], where PRMs evaluating step-level logical validity substantially outperformed ORMs. In the RAG domain, ReaRAG [14] and related methods employ frozen text-scoring models—often large LMs themselves—to evaluate the coherence of each step as surrogate process rewards. While these dense signals accelerate convergence, both paradigms share a fundamental limitation: neither provides an explicit, machine-verifiable mechanism to verify the factual correctness of intermediate reasoning steps. Outcome rewards are too sparse to assign credit to individual steps, and text-based process rewards assess fluency and plausibility, not factual truth. In knowledge-sparse or cross-document settings, text rewards can be actively misleading—rewarding stylistically polished intermediate conclusions that contain fabricated relations unsupported by retrieved evidence. This limitation becomes particularly severe in multi-hop reasoning scenarios, where factual errors in any intermediate step can propagate through the reasoning chain and contaminate the final answer [15]; the phenomenon, which we term **intermediate hallucination**, is insidious precisely because the offending step may read fluently and appear well-grounded, making it invisible to metrics that only inspect the final output.

Parallel to these RL developments, knowledge graphs (KGs) have been investigated as a complementary mechanism for improving LLM factuality. Existing KG-LLM integration follows three paradigms: input augmentation, where KG triples are appended to the prompt [16]; structure-aware encoding, where graph neural networks embed KG topology into the LLM's hidden states [17]; and post-hoc verification, where generated claims are checked against a KG at inference time [18]. Recent work has begun exploring KGs within RL training loops—for instance, using FactAlign rewards that score generated answers against ground-truth subgraphs within GRPO fine-tuning [19]—but these approaches apply factuality constraints at the output level rather than at the per-step process level. A critical research gap thus arises: **is it possible to provide an explicit, machine-verifiable factuality signal for each individual reasoning step during RL training, without imposing runtime KG dependencies on the deployed model?**

We argue that KGs are uniquely positioned to fill this gap. KGs encode relational facts with deterministic truth values—a triple is either present in the KG or it is not—making them natural training-time verifiable factuality sources. Our proposed framework, KG-ProWeight, introduces KGs as a per-step factuality anchor in the RL reward, operationalized through three integrated phases. First, we prompt a Teacher LLM to generate multi-step reasoning trajectories with explicit triple citations, then topologically verify each cited triple against a 2-hop Wikidata subgraph, producing three-valued process-reward annotations (+1 for verified, −1 for contradicted, 0 for non-falsifiable steps) at near-zero human labeling cost. Second, we train a learned α-gate—a lightweight three-feature classifier conditioned on graph density, link confidence, and semantic entropy—that dynamically weights KG-derived and text-derived reward signals per step, enabling adaptive transition to text supervision in KG-sparse regions. Third, we fine-tune a student model (Llama-3-8B) via supervised fine-tuning followed by proximal policy optimization under the gate's adaptive composite reward $R_t = \alpha_t \cdot R_{\text{KG}}(t) + (1-\alpha_t) \cdot R_{\text{text}}(t)$. The KG is used exclusively during training as a reward signal; at inference time, the deployed model requires no KG access.

Across HotpotQA, 2WikiMultiHopQA, and MuSiQue, we conduct a systematic empirical study spanning multiple model variants—base LLM, SFT only, PPO with full α-gate, PPO with pure text reward ($\alpha \equiv 0$), and PPO with pure outcome reward. Our key findings are threefold. First, the α-gate's KG branch is necessary for PPO to retain SFT-level performance: removing it causes the policy to collapse to near-random accuracy on long chains. Second, KG-anchored rewards reduce intermediate hallucination rate—measured by LLM-as-judge per-step evaluation—even when final-answer metrics are bounded by SFT, demonstrating that process-level factuality constraints improve reasoning quality in ways not captured by EM/F1. Third, pure outcome-based rewards underperform the α-gate variant by a substantial margin on tasks requiring four or more reasoning hops, revealing a structural limitation of sparse terminal rewards for deep multi-hop chains. Overall, KG-ProWeight exhibits four main characteristics:

1. **KG as training-time anchor**: unlike prior work that uses KGs for input augmentation or post-hoc verification, KG-ProWeight leverages KGs exclusively as a reward signal during RL—shaping what the policy learns to value without constraining inference;
2. **Adaptive factuality gating**: the α-gate dynamically weights KG and text rewards per step, preventing noisy KG signals in sparse regions from corrupting gradients while exploiting KG verification where it is reliable;
3. **Zero human annotation**: Phase 1 auto-constructs three-valued per-step labels through topological verification against Wikidata, producing training data at minimal cost;
4. **Process-level evaluation**: we introduce IHR as a core metric and show that KG-anchored process rewards reduce intermediate hallucinations even when final-answer metrics are bounded by SFT.

---

## 2. Related Work

### 2.1 Agentic RAG and Multi-Hop Reasoning | 智能体 RAG 与多跳推理

早期检索增强生成系统遵循单次检索范式 [25, 26]，不足以应对需要跨多文档证据合成的多跳问题。迭代检索方法如 IRCoT [22]、Self-Ask [23] 和 Iter-RetGen [24] 将检索与推理交织进行，逐步将问题分解为子查询并累积证据。Self-RAG [21] 引入反思 token，允许模型批评自己的生成并在适当时机决定检索。然而，这些方法依赖提示策略，并未通过训练优化模型内部的检索推理策略，限制了其对新领域的适应能力。大型推理模型 (LRM) 的出现，如 OpenAI-o1、DeepSeek-R1 [11] 和 QwQ-32B，已证明通过扩展链式思维推理来扩大测试时计算可在数学和编程任务上取得显著增益。然而，这些模型仍受其参数化知识的限制，其在开放域 QA 上的应用需要与检索系统进行显式集成。Search-o1 [27] 将 LRM 范式扩展为 Reason-in-Documents 模块，但在多跳场景中面临过度思考和信息提取失败的问题。Li et al. [28] 提供了 RAG-推理系统的全面综述。

Early retrieval-augmented generation systems follow a single-retrieval paradigm [25, 26], which is insufficient for multi-hop questions requiring evidence synthesis across multiple documents. Iterative retrieval methods such as IRCoT [22], Self-Ask [23], and Iter-RetGen [24] interleave retrieval with reasoning, progressively decomposing questions into sub-queries and accumulating evidence. Self-RAG [21] introduces reflection tokens that allow the model to critique its own generations and decide when to retrieve. These methods, however, rely on prompting strategies that do not optimize the model's internal retrieval-and-reasoning policy through training, limiting their adaptability to novel domains. The emergence of Large Reasoning Models (LRMs) such as OpenAI-o1, DeepSeek-R1 [11], and QwQ-32B has demonstrated that scaling test-time compute through extended chain-of-thought reasoning yields substantial gains on mathematics and coding tasks. However, these models remain constrained by their parametric knowledge, and their application to open-domain QA requires explicit integration with retrieval systems. Search-o1 [27] extends the LRM paradigm with a Reason-in-Documents module, but suffers from overthinking and information extraction failures in multi-hop settings. A comprehensive survey of RAG-reasoning systems is provided by Li et al. [28].

### 2.2 RL for Retrieval-Augmented Reasoning | 检索增强推理的强化学习

一项快速发展的研究方向将 RL 应用于训练 LLM 将推理与搜索引擎调用交错进行。Search-R1 [9] 将搜索引擎建模为环境的一部分，使用简单的结果奖励——最终答案的二元正确性——配合 PPO 或 GRPO，证明即使最小奖励信号也足以学习有意义的搜索行为。R1-Searcher [10] 提出两阶段 RL 框架：检索奖励教会模型正确调用搜索，随后答案奖励优化最终准确率。AutoRefine [29] 在搜索调用之间引入显式知识精炼步骤，通过 GRPO 将检索特定奖励与答案正确性相结合，在复杂多跳查询上取得显著增益。O2-Searcher [30] 在统一训练机制中为开放性和封闭性问题设计了独立的奖励分支。来自 SimpleDeepSearcher [31] 的不同观点认为，战略性数据工程——从实时网页搜索合成高质量推理轨迹并通过 SFT 微调——能以远低于 RL 方法的计算成本超越它们。仅用 871 条精心筛选的样本，其 SFT 方法就超越了 RL 基线 24.9%，质疑了 RL 训练对于检索增强任务的复杂性是否合理。ReaRAG [14] 是与我方工作最接近的先前研究，它在 Thought-Action-Observation 范式下对 9B 模型进行微调，使用 LRM 生成审慎思考轨迹，再通过 SFT 蒸馏到学生模型中。ReaRAG 明确回避了 RL 训练，认为策略蒸馏可达到相当的性能。我方工作与 ReaRAG 存在两点根本性差异：(i) 我们确实采用 RL (PPO)，但通过 KG 验证的过程奖励而非单纯依赖结果信号对其进行增强；(ii) 我们的奖励是自适应的——α-门控动态决定每一步是信任 KG 还是文本评分器，这一机制在 ReaRAG 的固定管线中不存在。所有这些方法的共性：除 AutoRefine 的检索质量奖励部分例外，每项先前工作使用的奖励函数评估的要么是最终答案（基于结果），要么是中间步骤的文本连贯性（通过冻结评判器基于过程）。无一者在逐步骤层面纳入外部的、机器可验证的事实性信号。**据我们所知，我方工作是首次将知识图谱约束推进到 RL-for-RAG 管线的训练时过程奖励中。**

A rapidly growing line of work applies RL to train LLMs to interleave reasoning with search engine calls. Search-R1 [9] models the search engine as part of the environment and uses a straightforward outcome-based reward—binary correctness of the final answer—with PPO or GRPO, demonstrating that even minimal reward signals suffice for learning meaningful search behaviors. R1-Searcher [10] proposes a two-stage RL framework: a retrieve-reward teaches the model to correctly invoke search, followed by an answer-reward that optimizes final accuracy. AutoRefine [29] introduces explicit knowledge refinement steps between search calls and combines retrieval-specific rewards with answer correctness via GRPO, achieving particular gains on complex multi-hop queries. O2-Searcher [30] designs separate reward strands for open-ended and closed-ended questions within a unified training mechanism. A dissenting perspective comes from SimpleDeepSearcher [31], which argues that strategic data engineering—synthesizing high-quality reasoning trajectories from live web search and fine-tuning via SFT—can outperform RL-based methods with substantially less computational cost. Using only 871 curated samples, their SFT approach surpasses RL baselines by 24.9%, questioning whether the complexity of RL training is justified for retrieval-augmented tasks. ReaRAG [14], the closest prior work to ours, fine-tunes a 9B model under a Thought-Action-Observation paradigm, using an LRM to generate deliberate thinking trajectories that are then distilled into the student via SFT. ReaRAG explicitly avoids RL training, arguing that strategic distillation achieves comparable performance. Our work differs from ReaRAG in two fundamental ways: (i) we do employ RL (PPO), but augment it with KG-verified process rewards rather than relying solely on outcome signals; and (ii) our reward is adaptive—the α-gate dynamically determines per-step whether to trust the KG or the text scorer, a mechanism absent in ReaRAG's fixed pipeline. Commonality across all these methods: with the partial exception of AutoRefine's retrieval-quality reward, every prior work uses reward functions that evaluate either the final answer (outcome-based) or the text coherence of intermediate steps (process-based via a frozen judge). None incorporate an external, machine-verifiable factuality signal at the per-step level. **Our work is, to our knowledge, the first to push knowledge-graph constraints into the training-time process reward of an RL-for-RAG pipeline.**

### 2.3 Process Reward Models and Hallucination | 过程奖励模型与幻觉

结果奖励模型 (ORM) 与过程奖励模型 (PRM) 的区分由 Lightman et al. [13] 在数学推理的背景下确立，其中评估每一步逻辑有效性的 PRM 显著优于仅检查最终答案的 ORM。OmegaPRM [32] 通过蒙特卡洛树搜索降低了标注成本，但尚未在开放域 QA 上验证。后续工作通过 Q 值排序 [35]、熵正则化 [33] 以及证明 GRPO 隐含地充当 PRM 的理论分析 [36] 推进了 PRM 设计。Zheng et al. [34] 提供了全面的综述。ReaRAG 和 R1-Searcher 为检索增强任务训练了 PRM 风格的组件，但这些 PRM 评判的是文本质量和检索相关性，而非事实真相。RAG 中的幻觉问题已得到广泛研究。KG 已被用于输入增强 [16]、结构感知编码 [17] 以及生成声明的事后验证 [18]。KG-ProWeight 是首次将 KG 用作训练时过程奖励——事实性信号在 RL 期间塑造策略梯度，而不仅仅是在推理时过滤其输出。

The distinction between outcome reward models (ORMs) and process reward models (PRMs) was established by Lightman et al. [13] in the context of mathematical reasoning, where PRMs that evaluate each step's logical validity substantially outperform ORMs that only check the final answer. OmegaPRM [32] reduces labeling cost via Monte Carlo tree search, but has not been validated on open-domain QA. Subsequent work has advanced PRM design through Q-value ranking [35], entropy regularization [33], and theoretical analysis showing that GRPO implicitly functions as a PRM [36]. A comprehensive survey is provided by Zheng et al. [34]. ReaRAG and R1-Searcher train PRM-style components for retrieval-augmented tasks, but these PRMs judge text quality and retrieval relevance, not factual truth. The hallucination problem in RAG has been studied extensively. KGs have been used for input augmentation [16], structure-aware encoding [17], and post-hoc verification of generated claims [18]. KG-ProWeight is the first to use KGs as a training-time process reward—the factuality signal shapes the policy's gradient during RL, rather than merely filtering its outputs at inference time.

### 2.4 Knowledge Graph Augmentation | 知识图谱增强

KG-LLM 融合存在三条路线：(a) 输入增强，将 KG 三元组拼接到提示中 [16]；(b) 结构感知编码，图神经网络将 KG 拓扑编码进 LLM 的隐状态中 [17]；(c) 事后约束，如 KGTraceRefiner [39] 在推理时对照 KG 验证生成的声明。更近期的工作已开始探索将 KG 纳入 RL 训练循环——GRPO 微调中的 FactAlign 奖励 [19]、用于 KG 增强 LLM 训练的 ground-truth 子图 [37]、以及用于事实验证的 KG 软提示 [38]。KG-ProWeight 代表了第四种范式：**训练时过程约束**。KG 既非模型输入的一部分，也非其架构的一部分——它是一个奖励信号，在 RL 期间塑造策略学会重视的内容。这一设计选择使推理管线保持轻量（测试时不需要 KG 调用），同时确保训练目标编码了对于事实扎根的推理步骤的偏好。

Three lines of KG-LLM integration exist: (a) input augmentation, where KG triples are appended to the prompt [16]; (b) structure-aware encoding, where graph neural networks encode KG topology into the LLM's hidden states [17]; and (c) post-hoc constraints, such as KGTraceRefiner [39], which verifies generated claims against a KG at inference time. More recent work has explored KGs within RL training loops—FactAlign rewards within GRPO fine-tuning [19], ground-truth subgraphs for KG-augmented LLM training [37], and KG soft prompts for fact-checking [38]. KG-ProWeight represents a fourth paradigm: **training-time process constraints**. The KG is not part of the model's input or architecture—it is a reward signal that shapes what the policy learns to value during RL. This design choice keeps the inference pipeline lightweight (no KG calls at test time) while ensuring that the training objective encodes a preference for factually grounded reasoning steps.

### 2.5 Intermediate Hallucination Rate (IHR) | 中间幻觉率

中间幻觉率 (IHR) 量化了推理步骤中包含无事实支撑声明的比例，由外部 LLM（GPT-4o 或等效模型）评判。IHR 通过评估推理的过程而非结果，补充了标准的 EM/F1 指标。我们采用 IHR 作为核心评估指标，并在 RL-for-RAG 文献中首次证明，即使最终答案指标受限于 SFT，过程级 KG 奖励依然可以降低 IHR。

The Intermediate Hallucination Rate (IHR), as defined in this work, quantifies the fraction of reasoning steps that contain factually unsupported claims, as judged by an external LLM (GPT-4o or equivalent). IHR complements standard EM/F1 metrics by evaluating the process rather than the outcome of reasoning. We adopt IHR as a core evaluation metric and demonstrate, for the first time in the RL-for-RAG literature, that process-level KG rewards can reduce IHR even when final-answer metrics are bounded by SFT.

---

## 3. Method *(待写)*

## 4. Experimental Setup *(待写)*

## 5. Results and Analysis *(待写)*

## 6. Conclusion *(待写)*

---

## References

[1] Brown, T. et al. Language models are few-shot learners. *NeurIPS*, 2020.

[2] OpenAI. GPT-4 technical report. *arXiv:2303.08774*, 2023.

[3] Touvron, H. et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*, 2023.

[4] Ji, Z. et al. Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12):1–38, 2023.

[5] Zhang, Y. et al. Siren's song in the AI ocean: A survey on hallucination in large language models. *arXiv:2309.01219*, 2023.

[6] Petroni, F. et al. KILT: A benchmark for knowledge intensive language tasks. *NAACL*, 2021.

[7] Tan, C. et al. Retrieval-augmented generation for AI-generated content: A survey. *arXiv:2402.19473*, 2024.

[8] Jin, C. et al. Retrieval-augmented generation for large language models: A survey. *arXiv:2312.10997*, 2024.

[9] Jin, P. et al. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. *COLM*, 2025.

[10] Song, H. et al. R1-Searcher: Incentivizing the search capability in LLMs via reinforcement learning. *arXiv:2503.05592*, 2025.

[11] DeepSeek-AI et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv:2501.12948*, 2025.

[12] Kumar, A. et al. Training language models to self-correct via reinforcement learning. *arXiv:2409.12917*, 2024.

[13] Lightman, H. et al. Let's verify step by step. *ICLR*, 2024.

[14] Lee, Z. et al. ReaRAG: Knowledge-guided reasoning enhances factuality of large reasoning models with iterative retrieval augmented generation. *arXiv:2503.21729*, 2025.

[15] Cao, S. et al. Error propagation in multi-hop QA with retrieval. *EMNLP*, 2023.

[16] Baek, J. et al. KAPING: Knowledge-augmented language model prompting. *EMNLP*, 2023.

[17] Tian, Y. et al. SubgraphRAG: Retrieving subgraphs for knowledge-grounded generation. *arXiv*, 2024.

[18] Xu, X. et al. SearChain: Global and local search over knowledge graphs for factual error correction. *ACL*, 2024.

[19] IEEE Access. Can LLMs perform RAG as multi-hop reasoning over knowledge graphs? *IEEE Access*, 2025.

[20] Lewis, P. et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*, 2020.

[21] Asai, A. et al. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *ICLR*, 2024.

[22] Trivedi, H. et al. IRCoT: Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. *ACL*, 2023.

[23] Press, O. et al. Self-Ask: Measuring and narrowing the compositionality gap in language models. *EMNLP*, 2023.

[24] Shao, Z. et al. Iter-RetGen: Enhancing retrieval-augmented generation with iterative retrieval. *EMNLP*, 2023.

[25] Borgeaud, S. et al. Improving language models by retrieving from trillions of tokens. *ICML*, 2022.

[26] Izacard, G. et al. Atlas: Few-shot learning with retrieval augmented models. *JMLR*, 2023.

[27] Li, Z. et al. Search-o1: Enhancing large reasoning models with retrieval-augmented reasoning. *arXiv*, 2025.

[28] Li, Y. et al. A survey of RAG-reasoning systems in large language models. *EMNLP Findings*, 2025.

[29] Shi, Y. et al. Search and refine during think: Facilitating knowledge refinement for improved retrieval-augmented reasoning. *NeurIPS*, 2025.

[30] Mei, J. et al. O2-Searcher: A searching-based agent model for open-domain open-ended question answering. *arXiv:2505.16582*, 2025.

[31] Sun, S. et al. SimpleDeepSearcher: Deep information seeking via web-powered reasoning trajectory synthesis. *arXiv:2505.16834*, 2025.

[32] Liao, G. et al. OmegaPRM: Automated process reward modeling via Monte Carlo tree search. *arXiv*, 2024.

[33] Setlur, A. et al. Rewarding progress: Scaling automated process verifiers for LLM reasoning. *ICLR*, 2025.

[34] Zheng, Y. et al. A survey of process reward models. *arXiv:2510.08049*, 2025.

[35] Lee, W. et al. Process reward model with Q-value rankings. *ICLR*, 2025.

[36] Sullivan, Z. GRPO is secretly a process reward model. *arXiv:2509.21154*, 2025.

[37] Cattaneo, A. et al. Ground-truth subgraphs for better training and evaluation of KG-augmented LLMs. *NeurIPS*, 2025.

[38] Yang, R. et al. GraphCheck: Fact-checking via knowledge graph soft prompts. *ACL*, 2025.

[39] FlashRAG Contributors. KGTraceRefiner: Integrating KG trace into FlashRAG, 2024.
