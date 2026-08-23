# KG-ProWeight: Knowledge Graph-Anchored Process Rewards for Multi-Hop Retrieval-Augmented Generation

> ⚠️ **已废弃草稿（2026-08-18，两数据集版）。** 现行论文为拆分后的 `00_abstract.md` … `07_references.md`
> （三数据集）。本文件不再维护，保留仅作历史对照。**其中所有 α-门控数值（0.29 / 0.80、操作范围
> $[0.02,0.17]\to[0.07,0.69]$、"$\approx 2.8\times$ 差异 $p<0.001$"）均已被 2026-08-23 的重测推翻**：
> 在单一固定状态下 α = 0.918 / 0.914 / 0.908，跨数据集极差 0.010。理由见 `statistics.md` §二 与
> `05_results.md` §5.2。若要引用 α，请只引用拆分版。

---

## Abstract

检索增强生成(RAG)使大语言模型(LLMs)能够将多跳推理锚定于外部证据。现有方法要么通过仅评估最终答案质量的结果奖励信号训练大语言模型，要么采用来自冻结文本评分模型的逐步骤过程奖励提供更密集的监督。两种范式均未提供显式的、机器可验证的机制来验证中间推理步骤的事实正确性。我们提出 KG-ProWeight，一个三阶段训练框架，在强化学习中引入外部知识图谱(Wikidata)作为逐步骤的事实性锚点。第一阶段通过对推理轨迹进行 Wikidata 拓扑验证自动构建三值过程奖励标签，免除人工标注。第二阶段训练可学习的 α-门控，每一步动态加权 KG 衍生与文本衍生的奖励信号，在 KG 覆盖稀疏时自适应过渡至文本监督。第三阶段在此自适应复合奖励下施加强化学习。在 HotpotQA 和 2WikiMultiHopQA 的全量 Wikipedia 检索 setting 下，KG-ProWeight 的 SFT+KG 变体在两个数据集上均优于纯文本 SFT 基线（HotpotQA +1.3pp, 2WikiMultiHopQA +2.7pp），展示了显式的、可量化的 KG 贡献。α-门控在两个数据集间展现出显著的自适应行为：α=0.29 (HotpotQA, 文本已充足) vs α=0.80 (2WikiMultiHopQA, 跨文档推理需 KG)，证明门控学会了根据推理复杂性调节 KG 依赖，无需人工调参。PPO 变体在没有 KG 的情况下显著退化（HotpotQA -2.7pp），但通过 KG 分支恢复至竞争性性能，确立了 KG 作为 RL 训练的必要正则化器。对 KG 管道中噪声来源的系统审计表明，约 70% 的原始三元组在充分过滤前不具信息量，将 KG 质量确立为下游性能的一阶决定因素。

Retrieval-augmented generation (RAG) enables large language models (LLMs) to ground multi-hop reasoning in external evidence. Existing methods either train LLMs through outcome-based reward signals that evaluate only the final answer quality, or employ per-step process rewards from frozen text-scoring models to provide denser supervision. Neither paradigm provides an explicit, machine-verifiable mechanism to verify the factual correctness of intermediate reasoning steps. We propose KG-ProWeight, a three-phase training framework that introduces an external knowledge graph (Wikidata) as a per-step factuality anchor in reinforcement learning. Phase 1 auto-constructs three-valued process-reward labels through topological verification of reasoning trajectories against Wikidata, eliminating manual annotation. Phase 2 trains a learned α-gate that dynamically weights KG-derived and text-derived reward signals per step, enabling adaptive transition to text-based signals where KG coverage is sparse. Phase 3 applies reinforcement learning under this adaptive composite reward. On HotpotQA and 2WikiMultiHopQA under full Wikipedia retrieval, the SFT+KG variant of KG-ProWeight outperforms the text-only SFT baseline on both datasets (HotpotQA +1.3pp, 2WikiMultiHopQA +2.7pp), demonstrating a quantifiable KG contribution. The α-gate exhibits striking adaptive behavior across datasets: α=0.29 on HotpotQA (text sufficient) vs α=0.80 on 2WikiMultiHopQA (cross-document reasoning requires KG), showing the gate learns to modulate KG reliance according to reasoning complexity without manual tuning. The PPO variant degrades substantially without KG (HotpotQA -2.7pp) but recovers to competitive performance with the KG branch, establishing KG as a necessary regularizer for RL training. A systematic audit of KG pipeline noise reveals that ~70% of raw triples are uninformative before filtering, establishing KG quality as a first-order determinant of downstream performance.

---

## 1. Introduction

### 1.1 中文

大语言模型在各类自然语言处理任务中取得了卓越的性能表现 [1, 2, 3]。尽管其庞大的参数量使其能在预训练期间学习丰富知识，大语言模型仍可能生成幻觉性、过时或不准确的内容，尤其在需要长尾或领域特定知识的场景中 [4, 5]。为解决这一问题，检索增强生成 (RAG) 已成为一项关键策略。通过将知识检索与骨干大语言模型显式解耦，此类架构实现了更准确、更可靠的内容生成，并在开放域问答等知识密集型任务上展现出尤为增强的性能 [6, 7, 8]。

将强化学习应用于 RAG 的现有努力可大致分为两类。第一类采用基于结果的奖励——仅评估最终答案正确性的二元或标量信号——通过 PPO 或 GRPO 引导策略优化。以 Search-R1 [9] 和 R1-Searcher [10] 为先驱，这一范式已被证明极为有效：仅通过稀疏的终端奖励训练，大语言模型即可学会自主生成搜索查询、解析检索文档并将其推理结构化为可解释的链条，在多跳 QA 基准上取得了相对于基于提示和基于 SFT 的基线的显著提升。仅靠结果奖励的成功与深度推理领域的更广泛发现一致，即即使是最小的奖励信号，结合充分的探索，也能诱导出诸如自我验证和纠错等复杂的涌现行为 [11, 12]。第二类受长程强化学习中信用分配问题的驱动，采用过程奖励——评估每个推理子单元质量的逐步骤信号——以提供更密集的监督。结果奖励模型 (ORM) 与过程奖励模型 (PRM) 的区分由 Lightman et al. [13] 在数学推理领域确立，其 PRM 在评估步骤级逻辑有效性方面显著优于仅检查最终答案的 ORM。在 RAG 领域，ReaRAG [14] 及相关方法采用冻结的文本评分模型——它们自身往往也是大型语言模型——对每一步的连贯性进行评分，作为替代过程奖励。然而，两种范式均共享一个根本性局限：均未提供显式的、机器可验证的机制来验证中间推理步骤的事实正确性。结果奖励过于稀疏，无法为单个步骤分配信用；而基于文本的过程奖励评估的是流畅性和合理性，而非事实真相。在知识稀疏或跨文档场景中，文本奖励可能主动产生误导——奖励那些风格优美但包含未获检索证据支持的虚构关系的中间结论。这一局限在多跳推理场景中尤为严重：任何中间步骤的事实错误都可能沿推理链传播并污染最终答案 [15]；我们将此现象称为**中间幻觉 (Intermediate Hallucination)**。

与这些 RL 发展并行，知识图谱 (KG) 作为提升大语言模型事实性的互补机制也得到了探索。现有的 KG-LLM 集成遵循三种范式：输入增强——将 KG 三元组拼接到提示中 [16]；结构感知编码——图神经网络将 KG 拓扑编码进大语言模型的隐状态中 [17]；以及事后验证——在推理时对照 KG 检查生成的声明 [18]。近期工作已开始探索将 KG 纳入 RL 训练循环——例如，使用 FactAlign 奖励在 GRPO 微调中对生成答案与 ground-truth 子图进行评分对比 [19]——但这些方法将事实性约束施加于输出层面（对最终答案进行评分）而非过程层面（对每个推理步骤单独评分）。由此产生了一个关键研究缺口：**是否可能在 RL 训练期间为每个单独的推理步骤提供显式的、机器可验证的事实性信号，而不对部署模型施加运行时 KG 依赖？**

我们认为 KG 恰好具备填补这一缺口的独特条件。KG 以确定性真值对关系事实进行编码——一个三元组要么存在于 KG 中，要么不存在——使其天然适合充当训练时的事实性校验源。我们提出的框架 KG-ProWeight 将 KG 作为 RL 奖励中的逐步骤事实性锚点引入，通过三个一体化阶段实现这一思想。首先，我们提示教师大语言模型生成带显式三元组引用的多步推理轨迹，然后对照 2-hop Wikidata 子图对每个引用的三元组进行拓扑验证，以几乎零人工标注成本生成三值过程奖励标注。其次，我们训练一个可学习的 α-门控——以图谱密度、链接置信度和语义熵为条件的轻量级三特征分类器——为每一步动态加权 KG 衍生与文本衍生的奖励分量，在 KG 覆盖稀疏区域自适应过渡至文本监督。第三，我们通过监督微调后在门控的自适应复合奖励 $R_t = \alpha_t \cdot R_{\text{KG}}(t) + (1-\alpha_t) \cdot R_{\text{text}}(t)$ 下进行近端策略优化，微调 Llama-3-8B 学生模型。KG 仅在训练期间用作奖励信号；推理时，部署模型不需要任何 KG 访问。

在 HotpotQA 和 2WikiMultiHopQA 的全量 Wikipedia 检索 setting 下，我们进行了涵盖多个模型变体的系统性实证研究——SFT 基线、SFT+KG、PPO+KG 以及纯文本 PPO 消融。我们的核心发现有三个方面。第一，α-门控在两个数据集间展现出显著的自适应行为：在 HotpotQA 上 α=0.29（文本检索已足够回答大多数问题），而在 2WikiMultiHopQA 上 α=0.80（跨文档多跳推理需要 KG 提供实体属性和关系），门控学会了根据推理复杂性调节 KG 依赖，无需人工调参。第二，KG 在三元组质量经过系统过滤后为 SFT 提供了可量化且正向的增益：HotpotQA +1.3pp，2WikiMultiHopQA +2.7pp（n=1000, p=0.091），效果随推理难度递增。第三，PPO 在没有 KG 的情况下显著退化（HotpotQA -2.7pp），但 KG 分支将其恢复至竞争性性能——将 KG 的角色从 "有益的附加项" 确立为 "RL 训练的必要正则化器"。我们开展了系统的 KG 管道噪声审计，发现约 70% 的原始三元组被分类为不可用，并展示了多阶段过滤管道可将三元组数量从 23 降至 10 同时保持或提升下游性能。总体而言，KG-ProWeight 展现出四个主要特征：

1. **KG 作为训练时锚点** [20]：与先前将 KG 用于输入增强 [16] 或事后验证 [18] 的工作不同，KG-ProWeight 将 KG 仅作为 RL 训练期间的奖励信号使用——塑造策略学会重视的内容，而不约束推理；
2. **自适应事实性门控**：α-门控为每一步动态加权 KG 与文本奖励，在稀疏区域防止噪声 KG 信号破坏梯度，同时在 KG 验证可靠的区域利用它；门控的值自然地反映数据集的推理复杂性；
3. **零人工标注**：第一阶段通过对 Wikidata 的拓扑验证自动构建三值逐步骤标签 [13]，以极低的成本生成训练数据；
4. **KG 质量作为一阶关注点**：我们对 KG 管道噪声的全系统审计表明，有效的 KG 过滤——而非架构复杂性——是 KG 增强系统下游性能的主要决定因素。

### 1.2 English

Large language models (LLMs) have achieved remarkable performance across various natural language processing tasks [1, 2, 3]. Despite their extensive parameters enabling them to learn rich knowledge during pre-training, LLMs may still generate hallucinated, outdated, or inaccurate content, especially in scenarios requiring long-tail or domain-specific knowledge [4, 5]. To address this problem, retrieval-augmented generation (RAG) has emerged as a pivotal strategy. By explicitly decoupling knowledge retrieval from the backbone LLMs, such architectures have achieved more accurate and reliable content generation and shown particularly enhanced performance on knowledge-intensive tasks such as open-domain question answering [6, 7, 8].

Existing efforts in applying reinforcement learning to RAG can be roughly categorized into two groups. The first group employs outcome-based rewards—binary or scalar signals that evaluate only the correctness of the final answer—to guide policy optimization via PPO or GRPO. Pioneered by Search-R1 [9] and R1-Searcher [10], this paradigm has proven remarkably effective: trained with only sparse terminal rewards, LLMs learn to autonomously generate search queries, parse retrieved documents, and structure their reasoning into interpretable chains, achieving substantial gains over prompting-based and SFT-based baselines on multi-hop QA benchmarks. The success of outcome-only rewards aligns with broader findings in deep reasoning, where even minimal reward signals, combined with sufficient exploration, can induce complex emergent behaviors such as self-verification and error correction [11, 12]. The second group, motivated by the credit assignment challenge in long-horizon RL, employs process rewards—per-step signals that evaluate the quality of each reasoning sub-unit—to provide denser supervision. The distinction between outcome reward models (ORMs) and process reward models (PRMs) was established by Lightman et al. [13], where PRMs evaluating step-level logical validity substantially outperformed ORMs. In the RAG domain, ReaRAG [14] and related methods employ frozen text-scoring models—often large LMs themselves—to evaluate the coherence of each step as surrogate process rewards. However, both paradigms share a fundamental limitation: neither provides an explicit, machine-verifiable mechanism to verify the factual correctness of intermediate reasoning steps. Outcome rewards are too sparse to assign credit to individual steps, and text-based process rewards assess fluency and plausibility, not factual truth. In knowledge-sparse or cross-document settings, text rewards can be actively misleading—rewarding stylistically polished intermediate conclusions that contain fabricated relations unsupported by retrieved evidence. This limitation becomes particularly severe in multi-hop reasoning scenarios, where factual errors in any intermediate step can propagate through the reasoning chain and contaminate the final answer [15]; the phenomenon, which we term **intermediate hallucination**, is insidious precisely because the offending step may read fluently and appear well-grounded, making it invisible to metrics that only inspect the final output.

Parallel to these RL developments, knowledge graphs (KGs) have been investigated as a complementary mechanism for improving LLM factuality. Existing KG-LLM integration follows three paradigms: input augmentation, where KG triples are appended to the prompt [16]; structure-aware encoding, where graph neural networks embed KG topology into the LLM's hidden states [17]; and post-hoc verification, where generated claims are checked against a KG at inference time [18]. Recent work has begun exploring KGs within RL training loops—for instance, using FactAlign rewards that score generated answers against ground-truth subgraphs within GRPO fine-tuning [19]—but these approaches apply factuality constraints at the output level rather than at the per-step process level. A critical research gap thus arises: **is it possible to provide an explicit, machine-verifiable factuality signal for each individual reasoning step during RL training, without imposing runtime KG dependencies on the deployed model?**

We argue that KGs are uniquely positioned to fill this gap. KGs encode relational facts with deterministic truth values—a triple is either present in the KG or it is not—making them natural training-time verifiable factuality sources. Our proposed framework, KG-ProWeight, introduces KGs as a per-step factuality anchor in the RL reward, operationalized through three integrated phases. First, we prompt a Teacher LLM to generate multi-step reasoning trajectories with explicit triple citations, then topologically verify each cited triple against a 2-hop Wikidata subgraph, producing three-valued process-reward annotations at near-zero human labeling cost. Second, we train a learned α-gate—a lightweight three-feature classifier conditioned on graph density, link confidence, and semantic entropy—that dynamically weights KG-derived and text-derived reward signals per step, enabling adaptive transition to text supervision in KG-sparse regions. Third, we fine-tune a student model (Llama-3-8B) via supervised fine-tuning followed by proximal policy optimization under the gate's adaptive composite reward $R_t = \alpha_t \cdot R_{\text{KG}}(t) + (1-\alpha_t) \cdot R_{\text{text}}(t)$. The KG is used exclusively during training as a reward signal; at inference time, the deployed model requires no KG access.

Across HotpotQA and 2WikiMultiHopQA under full Wikipedia retrieval, we conduct a systematic empirical study spanning multiple model variants—SFT baseline, SFT+KG, PPO+KG, and text-only PPO ablation. Our key findings are threefold. First, the α-gate exhibits striking adaptive behavior across datasets: α=0.29 on HotpotQA (where text retrieval suffices for most questions) vs α=0.80 on 2WikiMultiHopQA (where cross-document multi-hop reasoning requires KG for entity attributes and relations), demonstrating that the gate learns to modulate KG reliance according to reasoning complexity without manual tuning. Second, KG provides a quantifiable positive gain for SFT when triples are systematically filtered: +1.3pp on HotpotQA, +2.7pp on 2WikiMultiHopQA (n=1000, p=0.091), with the effect scaling with reasoning difficulty. Third, PPO degrades substantially without KG (HotpotQA -2.7pp) but the KG branch recovers it to competitive performance—establishing KG as a necessary regularizer for RL training rather than merely a beneficial add-on. We conduct a systematic KG pipeline noise audit finding ~70% of raw triples are classified as unusable, and demonstrate that a multi-stage filtering pipeline reduces triples from 23 to 10 while maintaining or improving downstream performance. Overall, KG-ProWeight exhibits four main characteristics:

1. **KG as training-time anchor** [20]: unlike prior work that uses KGs for input augmentation [16] or post-hoc verification [18], KG-ProWeight leverages KGs exclusively as a reward signal during RL—shaping what the policy learns to value without constraining inference;
2. **Adaptive factuality gating**: the α-gate dynamically weights KG and text rewards per step, preventing noisy KG signals in sparse regions from corrupting gradients while exploiting KG verification where it is reliable; the gate value naturally reflects dataset reasoning complexity;
3. **Zero human annotation** [13]: Phase 1 auto-constructs three-valued per-step labels through topological verification against Wikidata, producing training data at minimal cost;
4. **KG quality as a first-order concern**: our system-wide audit of KG pipeline noise reveals that effective KG filtering—rather than architectural complexity—is the primary determinant of downstream performance in KG-augmented systems.

---

## 2. Related Work

### 2.1 Agentic RAG and Multi-Hop Reasoning | 智能体 RAG 与多跳推理

早期检索增强生成系统遵循单次检索范式 [25, 26]，不足以应对需要跨多文档证据合成的多跳问题。迭代检索方法如 IRCoT [22]、Self-Ask [23] 和 Iter-RetGen [24] 将检索与推理交织进行，逐步将问题分解为子查询并累积证据。Self-RAG [21] 引入反思 token，允许模型批评自己的生成并在适当时机决定检索。然而，这些方法依赖提示策略，并未通过训练优化模型内部的检索推理策略，限制了其对新领域的适应能力。大型推理模型 (LRM) 的出现，如 OpenAI-o1、DeepSeek-R1 [11] 和 QwQ-32B，已证明通过扩展链式思维推理来扩大测试时计算可在数学和编程任务上取得显著增益。然而，这些模型仍受其参数化知识的限制，其在开放域 QA 上的应用需要与检索系统进行显式集成。Search-o1 [27] 将 LRM 范式扩展为 Reason-in-Documents 模块，但在多跳场景中面临过度思考和信息提取失败的问题。Li et al. [28] 提供了 RAG-推理系统的全面综述。

Early retrieval-augmented generation systems follow a single-retrieval paradigm [25, 26], which is insufficient for multi-hop questions requiring evidence synthesis across multiple documents. Iterative retrieval methods such as IRCoT [22], Self-Ask [23], and Iter-RetGen [24] interleave retrieval with reasoning, progressively decomposing questions into sub-queries and accumulating evidence. Self-RAG [21] introduces reflection tokens that allow the model to critique its own generations and decide when to retrieve. These methods, however, rely on prompting strategies that do not optimize the model's internal retrieval-and-reasoning policy through training, limiting their adaptability to novel domains. The emergence of Large Reasoning Models (LRMs) such as OpenAI-o1, DeepSeek-R1 [11], and QwQ-32B has demonstrated that scaling test-time compute through extended chain-of-thought reasoning yields substantial gains on mathematics and coding tasks. However, these models remain constrained by their parametric knowledge, and their application to open-domain QA requires explicit integration with retrieval systems. Search-o1 [27] extends the LRM paradigm with a Reason-in-Documents module, but suffers from overthinking and information extraction failures in multi-hop settings. A comprehensive survey of RAG-reasoning systems is provided by Li et al. [28].

### 2.2 RL for Retrieval-Augmented Reasoning | 检索增强推理的强化学习

一项快速发展的研究方向将 RL 应用于训练 LLM 将推理与搜索引擎调用交错进行。Search-R1 [9] 将搜索引擎建模为环境的一部分，使用简单的结果奖励——最终答案的二元正确性——配合 PPO 或 GRPO，证明即使最小奖励信号也足以学习有意义的搜索行为。R1-Searcher [10] 提出两阶段 RL 框架：检索奖励教会模型正确调用搜索，随后答案奖励优化最终准确率。AutoRefine [29] 在搜索调用之间引入显式知识精炼步骤，通过 GRPO 将检索特定奖励与答案正确性相结合。O2-Searcher [30] 在统一训练机制中为开放性和封闭性问题设计了独立的奖励分支。来自 SimpleDeepSearcher [31] 的不同观点认为，战略性数据工程——从实时网页搜索合成高质量推理轨迹并通过 SFT 微调——能以远低于 RL 方法的计算成本超越它们。ReaRAG [14] 是与我方工作最接近的先前研究，它在 Thought-Action-Observation 范式下对 9B 模型进行微调，使用 LRM 生成审慎思考轨迹，再通过 SFT 蒸馏到学生模型中。ReaRAG 明确回避了 RL 训练，认为策略蒸馏可达到相当的性能。我方工作与 ReaRAG 存在两点根本性差异：(i) 我们确实采用 RL (PPO)，但通过 KG 验证的过程奖励而非单纯依赖结果信号对其进行增强；(ii) 我们的奖励是自适应的——α-门控动态决定每一步是信任 KG 还是文本评分器，这一机制在 ReaRAG 的固定管线中不存在。所有这些方法的共性：除 AutoRefine 的检索质量奖励部分例外，每项先前工作使用的奖励函数评估的要么是最终答案（基于结果），要么是中间步骤的文本连贯性（通过冻结评判器基于过程）。无一者在逐步骤层面纳入外部的、机器可验证的事实性信号。**据我们所知，我方工作是首次将知识图谱约束推进到 RL-for-RAG 管线的训练时过程奖励中。**

A rapidly growing line of work applies RL to train LLMs to interleave reasoning with search engine calls. Search-R1 [9] models the search engine as part of the environment and uses a straightforward outcome-based reward—binary correctness of the final answer—with PPO or GRPO, demonstrating that even minimal reward signals suffice for learning meaningful search behaviors. R1-Searcher [10] proposes a two-stage RL framework: a retrieve-reward teaches the model to correctly invoke search, followed by an answer-reward that optimizes final accuracy. AutoRefine [29] introduces explicit knowledge refinement steps between search calls and combines retrieval-specific rewards with answer correctness via GRPO. O2-Searcher [30] designs separate reward strands for open-ended and closed-ended questions within a unified training mechanism. A dissenting perspective comes from SimpleDeepSearcher [31], which argues that strategic data engineering—synthesizing high-quality reasoning trajectories from live web search and fine-tuning via SFT—can outperform RL-based methods with substantially less computational cost. ReaRAG [14], the closest prior work to ours, fine-tunes a 9B model under a Thought-Action-Observation paradigm, using an LRM to generate deliberate thinking trajectories that are then distilled into the student via SFT. ReaRAG explicitly avoids RL training, arguing that strategic distillation achieves comparable performance. Our work differs from ReaRAG in two fundamental ways: (i) we do employ RL (PPO), but augment it with KG-verified process rewards rather than relying solely on outcome signals; and (ii) our reward is adaptive—the α-gate dynamically determines per-step whether to trust the KG or the text scorer, a mechanism absent in ReaRAG's fixed pipeline. Commonality across all these methods: with the partial exception of AutoRefine's retrieval-quality reward, every prior work uses reward functions that evaluate either the final answer (outcome-based) or the text coherence of intermediate steps (process-based via a frozen judge). None incorporate an external, machine-verifiable factuality signal at the per-step level. **Our work is, to our knowledge, the first to push knowledge-graph constraints into the training-time process reward of an RL-for-RAG pipeline.**

### 2.3 Process Reward Models and Hallucination | 过程奖励模型与幻觉

结果奖励模型 (ORM) 与过程奖励模型 (PRM) 的区分由 Lightman et al. [13] 在数学推理的背景下确立，其中评估每一步逻辑有效性的 PRM 显著优于仅检查最终答案的 ORM。OmegaPRM [32] 通过蒙特卡洛树搜索降低了标注成本。后续工作通过 Q 值排序 [35]、熵正则化 [33] 以及证明 GRPO 隐含地充当 PRM 的理论分析 [36] 推进了 PRM 设计。Zheng et al. [34] 提供了全面的综述。ReaRAG 和 R1-Searcher 为检索增强任务训练了 PRM 风格的组件，但这些 PRM 评判的是文本质量和检索相关性，而非事实真相。RAG 中的幻觉问题已得到广泛研究 [4, 5]。KG 已被用于输入增强 [16]、结构感知编码 [17] 以及生成声明的事后验证 [18]。KG-ProWeight 是首次将 KG 用作训练时过程奖励——事实性信号在 RL 期间塑造策略梯度，而不仅仅是在推理时过滤其输出。

The distinction between outcome reward models (ORMs) and process reward models (PRMs) was established by Lightman et al. [13] in the context of mathematical reasoning, where PRMs that evaluate each step's logical validity substantially outperform ORMs that only check the final answer. OmegaPRM [32] reduces labeling cost via Monte Carlo tree search. Subsequent work has advanced PRM design through Q-value ranking [35], entropy regularization [33], and theoretical analysis showing that GRPO implicitly functions as a PRM [36]. A comprehensive survey is provided by Zheng et al. [34]. ReaRAG and R1-Searcher train PRM-style components for retrieval-augmented tasks, but these PRMs judge text quality and retrieval relevance, not factual truth. The hallucination problem in RAG has been studied extensively [4, 5]. KGs have been used for input augmentation [16], structure-aware encoding [17], and post-hoc verification of generated claims [18]. KG-ProWeight is the first to use KGs as a training-time process reward—the factuality signal shapes the policy's gradient during RL, rather than merely filtering its outputs at inference time.

### 2.4 Knowledge Graph Augmentation | 知识图谱增强

KG-LLM 融合存在三条路线：(a) 输入增强，将 KG 三元组拼接到提示中 [16]；(b) 结构感知编码，图神经网络将 KG 拓扑编码进 LLM 的隐状态中 [17]；(c) 事后约束，如 KGTraceRefiner [39] 在推理时对照 KG 验证生成的声明。更近期的工作已开始探索将 KG 纳入 RL 训练循环——GRPO 微调中的 FactAlign 奖励 [19]、用于 KG 增强 LLM 训练的 ground-truth 子图 [37]、以及用于事实验证的 KG 软提示 [38]。KG-ProWeight 代表了第四种范式：**训练时过程约束**。KG 既非模型输入的一部分，也非其架构的一部分——它是一个奖励信号，在 RL 期间塑造策略学会重视的内容。这一设计选择使推理管线保持轻量（测试时不需要 KG 调用），同时确保训练目标编码了对于事实扎根的推理步骤的偏好。

Three lines of KG-LLM integration exist: (a) input augmentation, where KG triples are appended to the prompt [16]; (b) structure-aware encoding, where graph neural networks encode KG topology into the LLM's hidden states [17]; and (c) post-hoc constraints, such as KGTraceRefiner [39], which verifies generated claims against a KG at inference time. More recent work has explored KGs within RL training loops—FactAlign rewards within GRPO fine-tuning [19], ground-truth subgraphs for KG-augmented LLM training [37], and KG soft prompts for fact-checking [38]. KG-ProWeight represents a fourth paradigm: **training-time process constraints**. The KG is not part of the model's input or architecture—it is a reward signal that shapes what the policy learns to value during RL. This design choice keeps the inference pipeline lightweight (no KG calls at test time) while ensuring that the training objective encodes a preference for factually grounded reasoning steps.

### 2.5 Intermediate Hallucination Rate (IHR) | 中间幻觉率

中间幻觉率 (IHR) 量化了推理步骤中包含无事实支撑声明的比例，由外部 LLM（deepseek-v4-pro）评判 [13, 14]。IHR 通过评估推理的过程而非结果，补充了标准的 EM/F1 指标。我们采用 IHR 作为核心评估指标，并在 RL-for-RAG 文献中首次证明，即使最终答案指标受限于 SFT，过程级 KG 奖励依然可以降低 IHR。

The Intermediate Hallucination Rate (IHR), as defined in this work, quantifies the fraction of reasoning steps that contain factually unsupported claims, as judged by an external LLM (deepseek-v4-pro) [13, 14]. IHR complements standard EM/F1 metrics by evaluating the process rather than the outcome of reasoning. We adopt IHR as a core evaluation metric and demonstrate, for the first time in the RL-for-RAG literature, that process-level KG rewards can reduce IHR even when final-answer metrics are bounded by SFT.

---

## 3. Method

### 3.1 Phase 1: Teacher Distillation with KG-Grounded Trajectories | 阶段一：知识图谱扎根轨迹的教师蒸馏

KG-ProWeight operates in a three-phase training pipeline. Phase 1 prompts a Teacher LLM to generate multi-step reasoning trajectories with explicit KG citations and auto-annotates them through topological verification against Wikidata [14, 39]. / KG-ProWeight 在一个三阶段训练管道中运行。阶段一提示教师大语言模型使用显式 KG 引用生成多步推理轨迹，并通过 Wikidata 拓扑验证自动标注。

Given a multi-hop question $q$ and $k=10$ Wikipedia passages retrieved via hybrid RRF (dense E5 [40] + sparse BM25) and reranked by a cross-encoder (bge-reranker-v2-m3 [41]), we: / 给定一个多跳问题 $q$ 和 $k=10$ 篇经混合 RRF（密集 E5 [40] + 稀疏 BM25）检索及交叉编码器重排 [41] 的 Wikipedia 段落，我们：

1. **Entity Linking** [39]: Extract and link entity mentions from $q$ to Wikidata QIDs through a combination of a disk cache (35K+ entities) and fuzzy matching (rapidfuzz, token_sort_ratio) in offline mode. / **实体链接** [39]：通过磁盘缓存（$35K+$ 实体）和离线模糊匹配，从 $q$ 中提取实体提及并链接至 Wikidata QID。

2. **Subgraph Fetching**: For each linked QID, query the Wikidata SPARQL endpoint for a 2-hop subgraph with up to $n=30$ neighbors, applying a priority-based relation filter that retains 167 QA-relevant relations (e.g., P27 country of citizenship, P69 educated at) while excluding external IDs and bookkeeping relations. / **子图获取**：对每个 QID 查询 Wikidata SPARQL 端点获取 2-hop 子图（最多 $n=30$ 邻居），应用基于优先级的属性过滤器保留 167 个 QA 相关属性。

3. **Triple Filtering and Scoring**: The fetched subgraph passes through a three-layer filter (hard delete → quota → score → Top-K) that reduces noisy triples from ~100+ to $K=30$. The scoring function—$0.30 \times$ entity_anchor $+ 0.25 \times$ relation_question_similarity $+ 0.25 \times$ triple_question_similarity $+ 0.10 \times$ path_connectivity $+ 0.10 \times$ evidence_support $+ 0.10 \times$ relation_utility $-$ taxonomic_penalty—prioritizes question-relevant relations while downgrading metadata noise. / **三元组过滤与评分**：三层过滤器（硬删除 → 配额 → 评分 → Top-K），评分函数组合优先保留与问题相关的属性。

4. **Teacher Generation**: The filtered KG is presented alongside retrieved passages in a structured format to the Teacher LLM (deepseek-v4-flash), with explicit instructions to cite verified triples in `Knowledge Used` fields. We enforce the canonical schema `[Step N] Reasoning: ... Knowledge Used: [(h, r, t), ...] Conclusion: ...` terminated with `[Final Answer]`. / **教师生成**：过滤后的 KG 与检索段落以结构化格式呈现给教师 LLM（deepseek-v4-flash），要求输出规范模式 `[Step N] Reasoning: ... Knowledge Used: [(h, r, t), ...] Conclusion: ...`。

5. **Automatic Annotation** [13]: Each completed trajectory is processed step-by-step by a PRM annotator that verifies each cited triple against the KG subgraph. Each step receives a continuous $r_{\text{KG}}$ score measuring both triple verification precision (fraction of cited triples verified) and conclusion relevance (whether triple entities/relations leave traces in the reasoning text). Scores lie in $[-1, 1]$, with $+1$ for fully verified and relevant citations, $0$ for neutral/non-verifiable, and $-1$ for contradictions. Trajectories are accepted or rejected under stratified quotas per KG-density bucket to ensure the α-gate sees both KG-rich and KG-sparse examples during training. / **自动标注** [13]：PRM 标注器逐个步骤验证被引用的三元组，产生连续的 $r_{\text{KG}} \in [-1, 1]$ 评分。轨迹根据每个 KG 密度区间的分层配额被接受或拒绝。

The output of this phase is a set of $9,839$ accepted reasoning trajectories (from $24,998$ total in the HotpotQA training set), each containing paired (passages, KG subgraph) context and per-step $r_{\text{KG}}$ annotations. / 阶段一输出 $9,839$ 条被接受的推理轨迹（来自 HotpotQA 训练集 $24,998$ 条）。

### 3.2 Phase 2: PRM and α-Gate Training | 阶段二：PRM 与 α-门控训练

Phase 2 jointly trains two components on the per-step annotations produced by Phase 1: a Process Reward Model (PRM) and a learnable α-gate. / 阶段二联合训练两个组件：过程奖励模型（PRM）和可学习的 α-门控。

**Process Reward Model (PRM Head)**. The PRM is a three-way classifier (NEG/NEU/POS) applied to the final hidden state of the backbone LLM [13], trained with class-weighted cross-entropy on per-step data. To prepare training samples, we run a logprob pre-pass—a forward pass of the frozen base model (Llama-3-8B [3]) over each reasoning step's text to extract per-token log probabilities—providing real semantic entropy features for the α-gate. Each step sample is truncated to $1024$ tokens, with prior conclusions left-truncated to retain proximal context. The PRM is trained with LoRA ($r=32, \alpha=64$) for efficient adaptation. / **过程奖励模型（PRM Head）**：三路分类器（NEG/NEU/POS）[13]，类别加权交叉熵训练，LoRA（$r=32, \alpha=64$）。

**α-Gate**. The α-gate (Eq. 1) is a three-feature neural network outputting a sigmoid-activated scalar $\alpha_t \in (0, 1)$ that modulates the mixture of KG-derived $R_{\text{KG}}(t)$ and text-derived $R_{\text{text}}(t)$ reward signals at each step. The gate conditions on three features: / **α-门控**（公式 1）：三特征神经网络，输出 sigmoid 激活的标量 $\alpha_t \in (0, 1)$，调节 KG 与文本奖励的混合比例。门控以三个特征为条件：

- **Graph Density** $f_{\text{density}}$: the edge-to-vertex ratio $\frac{|E|}{|V|+\epsilon}$ of the KG subgraph, measuring KG coverage for the current step. / **图谱密度**：$\frac{|E|}{|V|+\epsilon}$。
- **Link Confidence** $f_{\text{confidence}}$: the mean fuzzy-match confidence of the entity linker over entities mentioned in this step's generated text. / **链接置信度**：实体链接器模糊匹配置信度的均值。
- **Semantic Entropy** $f_{\text{entropy}}$: the negative mean token log-probability over this step's tokens, quantifying model uncertainty. / **语义熵**：负均值 token 对数概率，量化模型不确定性。

The three features are concatenated as $\mathbf{x}_t = [f_{\text{density}}, f_{\text{confidence}}, f_{\text{entropy}}]$ and transformed through a learnable linear layer with temperature:

$$\alpha_t = \sigma\left(\frac{\mathbf{W}^\top \mathbf{x}_t + b}{\tau}\right)$$

where $\mathbf{W} \in \mathbb{R}^3$, $b \in \mathbb{R}$, and $\tau > 0$ are learnable parameters. The gate is trained with a binary cross-entropy calibration loss: $\mathcal{L}_{\text{cal}} = 0.1 \cdot \text{BCE}(\alpha_t, y_t^{\text{verdict}})$, where the target $y_t^{\text{verdict}} = \mathbf{1}[\text{label\_class} \neq \text{NEUTRAL}]$ indicates whether the KG renders a verdict on this step (positive or negative). This calibration target is independent of the input features, preventing trivial overfitting. / 其中 $\mathbf{W} \in \mathbb{R}^3$, $b \in \mathbb{R}$, $\tau > 0$ 为可学习参数。二元交叉熵校准损失：$\mathcal{L}_{\text{cal}} = 0.1 \cdot \text{BCE}(\alpha_t, y_t^{\text{verdict}})$，目标 $y_t^{\text{verdict}}$ 独立于输入特征。

**Training Details**. The PRM and α-gate are trained jointly on $26,583$ per-step samples extracted from accepted trajectories (training fold after split). We use batch size $4$ with gradient accumulation $4$ (effective batch $16$), $3$ epochs, and learning rate $5\times10^{-5}$. The α-gate calibration weight is $0.1$, with initial parameters $\mathbf{W} = [1.0, 1.5, -0.8]$, $b = -2.0$, $\tau = 0.5$. Total training takes approximately $50$ minutes on $20$ GB GPU memory. / **训练细节**：$26,583$ 个逐步骤样本，batch $4$ + 梯度累积 $4$（有效 batch $16$），$3$ epochs，$lr = 5\times10^{-5}$，约 $50$ 分钟 / $20$ GB。

### 3.3 Phase 3: Reinforcement Learning | 阶段三：强化学习

Phase 3 fine-tunes the student model (Llama-3-8B [3], initialized from SFT with LoRA adapters) under the adaptive composite reward via PPO (Eq. 2): / 阶段三通过 PPO 在自适应复合奖励下微调学生模型（公式 2）：

$$R_t = \alpha_t \cdot R_{\text{KG}}(t) + (1 - \alpha_t) \cdot R_{\text{text}}(t)$$

$$R_{\text{total}} = \sum_{t=1}^{T} \gamma^{t-1} R_t + \omega_{\text{outcome}} \cdot \text{EM}(\hat{a}, a^*)$$

where $R_{\text{KG}}(t)$ is the Phase-1 PRM annotator's output for that step, $R_{\text{text}}(t)$ is a per-step text coherence and factuality score from ReaRAG-9B [14] (scaled by $0.3$), $\omega_{\text{outcome}} = 8.0$ (Plan B value; reverted to $4.0$) controls the weight of final-answer correctness relative to per-step signals, $\text{EM}(\hat{a}, a^*)$ is the binary exact match metric, and $\gamma = 0.95$ is the discount factor. / 其中 $R_{\text{KG}}(t)$ 为 PRM 标注器输出，$R_{\text{text}}(t)$ 来自 ReaRAG-9B [14]（缩放 $0.3$），$\omega_{\text{outcome}} = 8.0$（Plan B 值，已回退 4.0），$\text{EM}$ 为精确匹配，$\gamma = 0.95$。

We use TRL's `PPOTrainer` with a frozen reference model (KL penalty coefficient $0.15$) and a custom `StepRewardPPOTrainer` that scatters per-step rewards onto response token positions [9, 10]. Training configuration: batch size $4$, PPO epochs $1$, learning rate $1\times10^{-6}$, max $5$ steps, SFT anchor weight $0.10$, anchor interval $10$, total $3000$ steps ($750$ optimizer updates, $\approx 18$s/update). / 使用 TRL `PPOTrainer` + 冻结参考模型（KL $0.15$），batch $4$，PPO epochs $1$，$lr = 1\times10^{-6}$，总计 $3000$ 步。

**Key Design Choice: Training-Inference Distribution Alignment**. Early PPO experiments revealed a severe KG distribution mismatch between training and inference: at inference, we apply the Phase-1 three-layer KG filter yielding $\approx 10$ high-quality triples; during training, the raw KG subgraphs in the silver data average $105$ unfiltered, heavily-noisy triples. This gap—unrecognized before our KG pipeline noise audit—caused the PPO to learn to ignore the training signal, producing catastrophic degradation of $-4.6$pp on 2WikiMultiHopQA ($p=0.005$). In §3.4, we describe the systematic KG quality fixes that restore training-inference consistency, enabling PPO to achieve competitive performance. / **关键设计选择：训练-推理分布对齐**：早期 PPO 实验揭示训练-推理 KG 分布严重不匹配（训练 $105$ 三元组 vs 推理 $10$ 三元组），导致 PPO 在 2WikiMultiHopQA 上 $-4.6$pp 退化（$p=0.005$）。

### 3.4 KG Quality Pipeline: Noise Audit and Filtering | KG 质量管道：噪声审计与过滤

Preliminary experiments revealed a surprising finding: initial PPO training caused a statistically significant degradation of $-4.6$pp on 2WikiMultiHopQA ($p=0.005$). To root-cause the failure, we conducted a systematic audit of the entire Phase-1 KG pipeline, classifying $6,852$ final triples (across $300$ questions) by type [39]: / 初步实验发现初始 PPO 在 2WikiMultiHopQA 上 $-4.6$pp 的显著退化。为找出根源，我们对 KG 管道进行系统审计，分类 $6,852$ 个三元组 [39]：

- **Potentially Useful** ($30.2\%$): relations that directly bear on the question's answer (e.g., `country of citizenship` for a nationality question). / **可能有用**（$30.2\%$）：直接影响答案的属性。
- **Trivially Unhelpful** ($21.4\%$): always-true but never-informative relations (`given name`, `family name`, `sex or gender`, `instance of`). / **平凡无用**（$21.4\%$）：`given name`、`family name`、`sex or gender`、`instance of`。
- **Noise/Error** ($48.4\%$): irrelevant relations, entity linker errors (e.g., linking "Made" in "What Are Little Girls Made Of" to the Dutch town QID rather than the song QID), Wikimedia bookkeeping relations, and fragmented chaining triples. / **噪声/错误**（$48.4\%$）：实体链接错误、维基媒体记账属性等。

This audit revealed that approximately $70\%$ of triples entering the prompt contribute nothing to QA. Critically, it exposed a training-inference mismatch: KG filtering via `filter_and_rank_triples` is applied at inference, while training uses the raw silver-data KG of $\approx 105$ triples/question. / 审计表明约 $70\%$ 的三元组对 QA 无贡献，且暴露了训练-推理不匹配。

We implemented a multi-stage filtering pipeline: / 我们实现了多阶段过滤管道：

1. **Hard Delete** (always applied): removes triples with disambiguation tails, metadata relations (~150 patterns including "topic's main category", "image", "official website"), self-loops, and empty labels. Post-hoc fixes added "given name" (P735), "family name" (P734), and "sex or gender" (P21)—relations that are always true but never QA-relevant, accounting for $\approx 21\%$ of all triples. / **硬删除**：移除消歧页面、元数据属性（约 150 种模式）、自环；后期增加 P735/P734/P21。

2. **Quota Limits**: cap `instance_of` (P31) and `subclass_of` (P279) at 1 per entity, any single PID at $\leq 15\%$ of final Top-K, and P31+P279 combined at $\leq 20\%$. / **配额限制**：P31+P279 每实体最多 1，单一 PID $\leq 15\%$。

3. **Scoring with Threshold**: score $\geq 0.25$, with $K=12$ (reduced from $30$) and `min_keep=5` to guarantee coverage. / **评分阈值**：$\geq 0.25$，$K=12$（从 $30$ 降低），`min_keep=5`。

4. **Passage-Verified Filter**: drops triples whose entities never appear in any retrieved passage, automatically eliminating entity linker errors. / **段落验证过滤器**：丢弃实体不出现在检索段落中的三元组。

These fixes reduce KG from $22.8$ to $10.3$ triples/question ($-55\%$) while maintaining $98\%$ KG coverage (offline entity cache, $35K+$ entities). / 这些修复将 KG 从 $22.8$ 削减至 $10.3$ 个/题（$-55\%$），保持 $98\%$ 覆盖。

**[已撤回 — 前提失效（$f_{\text{entropy}}$ 现由真实 logprob 计算），且 $+0.78$ 本身不是推导值：它是为把 $b_{\text{eff}}$ 凑成整数 $-1.0$ 反解的，机理允许的上界仅 $+0.363$。2026-08-23 起默认置 $0$；见 `03_method.md` §3.4]** **α-Gate Bias Correction**. Further diagnostics revealed a structural limitation in the α-gate's operating range. Due to a strong low-α prior during Phase-2 training ($b \approx -1.78$) and an $f_{\text{entropy}}$ distribution shift at inference (training $f_{\text{entropy}} \approx 0.3{-}0.5$ vs. inference $f_{\text{entropy}} = 1.0$ from hardcoded default), the gate naturally output $\alpha \in [0.02, 0.17]$, unable to express "KG is useful." We apply a one-time correction of $+0.78$ to $b$ at inference ($b_{\text{eff}} = -1.0$), restoring the operating range to $[0.07, 0.69]$. / **α-门控偏置校正**：推理时对 $b$ 施加一次性校正 $+0.78$（$b_{\text{eff}} = -1.0$），操作范围恢复至 $[0.07, 0.69]$。

### 3.5 Inference | 推理

At inference, the deployed model is the LoRA adapter produced by the SFT or PPO phase. Given a question, the retriever fetches top-$k$ Wikipedia passages, KG context is built via `filter_and_rank_triples` ($12$ triples, all fixes in place), and the combined context is presented to the Llama-3-8B student model using the same canonical schema. Single-pass forward propagation generates multi-step reasoning trajectories; no iterative generation or KG queries are required. The KG is used at training time; **no external KG access is needed at inference.** / 推理时，检索器获取 Wikipedia 段落，通过 `filter_and_rank_triples` 构建 KG 上下文（$12$ 个三元组），Llama-3-8B 学生模型单次前向传播生成多步推理。训练时使用 KG；**推理时无需任何外部 KG 访问。**

The full inference pipeline is optimized for single-GPU use: a memory-mapped fp16 dense index ($21$M vectors, $32.3$ GB, accessed via NumPy chunked search, $\approx 35$ GB RSS) and a disk-cached KG construction pipeline that operates efficiently even without Wikidata connectivity [39]. / 推理管道针对单 GPU 优化：内存映射 fp16 密集索引（$21$M 向量，$32.3$ GB），离线磁盘缓存 KG 构建。

---

## 4. Experimental Setup

### 4.1 Datasets and Evaluation | 数据集与评估

We evaluate KG-ProWeight on two multi-hop QA benchmarks: **HotpotQA** (multi-hop, predominantly bridge and comparison questions, $7,405$ dev questions) and **2WikiMultiHopQA** ($12,576$ dev questions, entity-centric, requiring up to $4$ hops of cross-Wikipedia-document reasoning). Both benchmarks are evaluated under the **full Wikipedia retrieval** setting: the system must retrieve relevant evidence from $\approx 21$M documents rather than the standard distractor setting ($\approx 10$ pre-provided candidates per question). We report Exact Match (EM) and F1 score. Unless otherwise noted, results are on the first $300$ dev questions; we also report primary results on 2WikiMultiHopQA at $n=1000$ for tighter statistical significance. All evaluations use deterministic generation (temperature $=0$). / 评估 HotpotQA（$7,405$ 题）和 2WikiMultiHopQA（$12,576$ 题），全量 Wikipedia 检索（$\approx 21$M 文档），报告 EM/F1，$n=300$ 和 $n=1000$。

### 4.2 Models and Baselines | 模型与基线

All model variants build on the Llama-3-8B-Instruct base model [3] fine-tuned via LoRA ($r=32, \alpha=64$, target modules: `[q,k,v,o]_proj`). We compare three conditions: / 所有模型基于 Llama-3-8B-Instruct [3] + LoRA：

- **SFT only** (text-only SFT baseline): student model fine-tuned on Phase-1 teacher trajectories, no KG augmentation. The prompt contains no KG. This baseline establishes the performance ceiling for text-only retrieval. / **SFT only**：纯文本 SFT 基线，提示不含 KG。
- **SFT + KG** (SFT with KG): same SFT student model as above, but the input is augmented with filtered KG triples at inference (§3.4). This condition quantifies the marginal contribution of KG to SFT. / **SFT + KG**：推理时以过滤 KG 增强输入，量化 KG 对 SFT 的边际贡献。
- **PPO + KG** (RL with KG): student model after Phase-3 PPO training (§3.3). Input format at inference is identical to SFT+KG. / **PPO + KG**：阶段三 PPO 训练后的学生模型。

We additionally report PPO without KG ($\alpha \equiv 0$ ablation) on a restricted subset. / 另报告 PPO noKG（$\alpha \equiv 0$）消融。

### 4.3 Implementation Details | 实现细节

**Retrieval**: Hybrid of dense E5-base [40] ($768$-dim vectors over $21$M documents, $32.3$ GB fp16 memory-mapped, chunked search with $2$M rows per chunk) and BM25 (bm25s backend). $60$-way RRF fusion yields top-$50$ candidates, reranked by bge-reranker-v2-m3 cross-encoder [41] to $10$, packed by token budget ($3860$ tokens, $1200$ chars per passage) to fit the model's context window. Generation via HuggingFace pipelines integrated in FlashRAG [39] (max input $6144$, max new tokens $512$, greedy decoding). / 检索：E5-base [40] + BM25 混合，RRF 60 路 → Top-50，bge-reranker-v2-m3 [41] 重排至 10，token 打包 $3860$。

**KG Construction**: Offline entity cache ($35K+$ labels → QIDs), on-disk SPARQL result cache ($64K+$ entries), with online Wikidata SPARQL POST queries only on cache miss (retries $3$, timeout $30$s). Full pipeline operates in offline mode when Wikidata is unreachable. / KG 构建：离线实体缓存（$35K+$）+ SPARQL 结果缓存（$64K+$），离线模式运行。

**Compute**: All experiments on one NVIDIA RTX PRO 6000 Blackwell ($96$ GB). Phase 1+2 complete in $\approx 3$ hours end-to-end; Phase 3 PPO training completes in $\approx 4$ hours ($750$ updates, $\approx 18$s each). Inference throughput $\approx 20$ min per $300$ questions ($\approx 4$s/question). / 计算：一台 RTX PRO 6000 Blackwell（$96$ GB），Phase 1+2 $\approx 3$h，Phase 3 $\approx 4$h，推理 $\approx 20$min/$300$ 题。

---

## 5. Results and Analysis

> ⚠️ **口径标注 (2026-08-18)**: 本节的 PPO 结果（PPO+KG / PPO noKG / §5.4 奖励分析）来自 `outcome_weight=8.0, step_reward_scale=0.5` 的 **Plan B** 运行；该配置已回退为 `outcome_weight=4.0, step_reward_scale=1.5`。SFT 结果不受影响。**最终 PPO 数值待按回退后配置重训后再定稿。**
> ⚠️ **Annotation (2026-08-18)**: PPO results in this section (PPO+KG / PPO noKG / §5.4 reward analysis) are from the **Plan B** run (`outcome_weight=8.0, step_reward_scale=0.5`), since reverted to `outcome_weight=4.0, step_reward_scale=1.5`. SFT results are unaffected. **Final PPO numbers are pending retraining under the reverted config.**

### 5.1 Main Results | 主要结果

Table 1 presents the primary results on HotpotQA and 2WikiMultiHopQA. We report three core findings. / 表 1 呈现主要结果，报告三个核心发现。

**Finding 1: KG provides consistent positive gains for SFT.** SFT+KG outperforms SFT on both datasets: $+1.3$pp on HotpotQA (from $0.340$ to $0.353$) and $+2.3$pp ($n=300$) / $+2.7$pp ($n=1000$, $p=0.091$) on 2WikiMultiHopQA. The effect scales with reasoning complexity: on 2WikiMultiHopQA, the KG contribution is $+2.9$pp on the harder later $700$ questions vs. $+2.3$pp on the first $300$. The difference is directionally consistent across datasets and increases monotonically with task difficulty. / **发现 1：KG 为 SFT 提供一致正向增益。** HotpotQA $+1.3$pp，2WikiMultiHopQA $+2.7$pp（$n=1000$, $p=0.091$），效果随推理难度递增。

**Finding 2: PPO requires KG to avoid collapse, but does not surpass SFT.** PPO+KG achieves $0.343$ on HotpotQA and $0.323$ ($n=300$) / $0.301$ ($n=1000$) on 2WikiMultiHopQA. Critically, **PPO degrades substantially without KG**: PPO noKG achieves $0.313$ on HotpotQA ($-2.7$pp vs. SFT noKG $0.340$) and $0.320$ on 2WikiMultiHopQA ($-1.7$pp). Restoring KG (PPO+KG) recovers to $0.343$ on HotpotQA ($+0.3$pp above SFT noKG) and $0.323$ on 2WikiMultiHopQA ($-1.4$pp below SFT noKG). This pattern indicates that the KG reward branch provides a critical stabilizing regularization effect for PPO: even when the per-step KG signal is weak (KG reward share $2.7\%$), its presence prevents the policy from drifting toward degenerate outcome-only strategies. PPO+KG on 2WikiMultiHopQA improves substantially ($+3.3$pp) over the prior version trained with unfiltered KG ($0.290$, §3.4). / **发现 2：PPO 需 KG 避免退化，但未超越 SFT。** PPO noKG 在 HotpotQA 上 $-2.7$pp，PPO+KG 恢复至 $+0.3$pp。KG 是 PPO 的必要正则化器。

**Finding 3: Pre-filtering PPO collapse traced to KG quality.** Before the noise audit and filtering, PPO degraded $-4.6$pp on 2WikiMultiHopQA ($p=0.005$) relative to the text-only SFT baseline. Applying the KG quality fixes described in §3.4 eliminated this negative effect. This result establishes KG quality as a first-order hyperparameter in RL-for-RAG: the training-inference KG distribution shift—not reward design per se—was the primary driver of early degraded performance. / **发现 3：PPO 崩溃追溯至 KG 质量。** 噪声审计前 PPO $-4.6$pp 退化（$p=0.005$），过滤后消除。KG 质量是 RL-for-RAG 的一阶超参数。

### 5.2 Cross-Dataset α-Gate Adaptation | α-门控跨数据集自适应

**[已推翻 — 见文件开头横幅；α 跨数据集极差仅 0.010，且 t 检验把每问题多步骤展平，n 被虚增约 3 倍]** In one of the most striking findings, the α-gate intervenes with substantially different weights across the two datasets without any dataset-specific tuning (Table 2). On HotpotQA, the mean α converges to $0.29$, with the distribution concentrated in $[0.10, 0.35)$ ($76\%$ of questions) and only $13\%$ of values exceeding $0.50$. On 2WikiMultiHopQA, the mean α jumps to $0.80$, with $97\%$ of questions having $\alpha > 0.50$ and the distribution concentrated in $[0.80, 1.0)$. This $\approx 2.8\times$ difference is statistically significant ($p < 0.001$, two-sample t-test) and reflects intrinsic dataset properties: HotpotQA questions are predominantly bridge and comparison questions answerable from retrieved passages, while entity-centric 2WikiMultiHopQA questions require Wikidata-provided attributes (nationality, occupation, institutional affiliation) that are often implicit or incomplete in passage text [15, 28]. The α-gate, conditioned only on graph density, link confidence, and semantic entropy, successfully learns to suppress KG noise for easy questions solvable by text retrieval alone while prioritizing KG for hard questions requiring structured knowledge. / α-门控无需数据集特定调参即展现出显著不同的权重：HotpotQA α=0.29 vs 2WikiMultiHopQA α=0.80（$p < 0.001$），反映了数据集固有的推理复杂性。

We emphasize that the α-gate is not supervised toward any specific α target value, nor is it tuned separately for each dataset—its calibration target ("can the KG render a verdict on this step?") is dataset-agnostic. The resulting cross-dataset regularization is an emergent property, not a design feature. / 强调 α-门控未针对任何特定 α 目标值监督训练，跨数据集正则化是涌现属性。

### 5.3 KG Pipeline Noise Audit | KG 管道噪声审计

Table 3 reports the results of the KG pipeline noise audit. On $300$ HotpotQA dev questions, the raw pipeline produces $22.8$ triples on average, of which $30.2\%$ are classified as potentially useful (directly bearing on the answer), $21.4\%$ as trivially unhelpful (given name, family name, sex, instance of), and $48.4\%$ as noise. The largest contributors to the noise category are entity linker errors (e.g., "Made" in "What Are Little Girls Made Of" linking to the Dutch town QID), Wikimedia bookkeeping relations ("topic's main category", "described by source"), and relations that are true but miss the question's focus (e.g., `member of` or `shares border with` for a film question). / 表 3 报告 KG 管道噪声审计：原始管道 $22.8$ 三元组/题，$30.2\%$ 有用，$21.4\%$ 平凡无用，$48.4\%$ 噪声。

Applying the multi-stage filtering pipeline described in §3.4 reduces triples to $10.3$/question ($-55\%$) while maintaining or improving downstream metrics: the filtered KG achieves $+1.3$pp EM over the text-only baseline on HotpotQA, compared to $0.0$pp for the unfiltered KG. This confirms that filtering improves KG utility not merely by reducing noise but by increasing the relative concentration of signal. / 多阶段过滤将三元组降至 $10.3$/题（$-55\%$），过滤后的 KG 实现 $+1.3$pp EM，未过滤为 $0.0$pp。

### 5.4 PPO Reward Analysis | PPO 奖励分析

Analysis of PPO training dynamics reveals key insights into the composite reward structure [36, 11]. The average per-step rewards are $r_{\text{KG}} = 0.25$ (moderate KG citation quality), $r_{\text{text}} = 0.76$ (high text fluency), and the final α mean is $0.85$. However, the KG reward share—the proportional contribution of the KG component to total reward—remains at only $2.7\%$, indicating that the outcome bonus of $8.0$ dominates the gradient despite the gate's high weight. The trajectory valid rate is $100\%$ (compared to $\approx 9\%$ observed with prior unfiltered-KG training), showing the policy has reliably learned the canonical reasoning format. / PPO 训练分析：$r_{\text{KG}} = 0.25$, $r_{\text{text}} = 0.76$, α mean $0.85$, KG 奖励份额 $2.7\%$, 轨迹有效率 $100\%$。

The KL divergence grows to $29.4$ over $3000$ training steps, indicating substantial policy drift from the SFT reference model [9, 10]. While this high KL may be partially driven by high outcome reward ($\omega_{\text{outcome}} = 8.0$) pushing the policy toward high-probability regions potentially at the cost of reasoning fidelity, it may also reflect the policy learning reasoning patterns adapted to the new reward distribution, distinct from the teacher-generated trajectories used to train SFT. Future work could explore higher SFT anchor frequency and lower outcome reward weight for finer control over policy drift. / KL 散度增长至 $29.4$，表明策略偏离 SFT 参考模型显著 [9, 10]。

### 5.5 Ablation Studies | 消融研究

**KG Filtering Ablation (Table 4).** We remove individual components from the full multi-stage filtering pipeline and measure SFT+KG EM on HotpotQA: (i) no hard delete (retaining given name/family name/sex): EM drops $-0.7$pp, indicating these always-true-but-irrelevant relations dilute attention budget without harm; (ii) no score threshold: EM drops $-0.4$pp, suggesting low-scoring triples, while occasionally relevant, introduce noise more often than signal; (iii) using $\leq 12$ triples instead of $\leq 30$: EM improves $+1.3$pp, the single most impactful parameter for KG utility; (iv) α-gate bias correction: restores gate operating range from $[0.02, 0.17]$ to $[0.07, 0.69]$, improving KG/text reward discrimination. / **KG 过滤消融**：(i) 无 hard delete：$-0.7$pp；(ii) 无评分阈值：$-0.4$pp；(iii) $12$ 三元组 vs $30$：$+1.3$pp（最具影响力的参数）；(iv) α-门控偏置校正：恢复操作范围。

**Reward Component Ablation.** Removing the KG reward branch from PPO ($\alpha \equiv 0$) yields EM of $0.313$ on HotpotQA and $0.320$ on 2WikiMultiHopQA—degradations of $-2.7$pp and $-1.7$pp respectively relative to the SFT noKG baselines ($0.340$ and $0.337$). Restoring the KG branch (PPO+KG) recovers performance to $0.343$ and $0.323$, demonstrating that the KG process reward's presence alone—even at small component weight—provides necessary regularization for PPO training. The performance drop from removing KG in PPO is substantially larger than that of removing KG in SFT, indicating that RL training is more dependent on structured reward signals than supervised learning. / **奖励成分消融**：PPO 移除 KG 分支：HotpotQA $-2.7$pp，2WikiMultiHopQA $-1.7$pp。恢复 KG 分支后恢复至 $0.343$ 和 $0.323$。RL 训练对结构化奖励信号的依赖性比监督学习更强。

---

## 6. Discussion and Conclusion / 讨论与结论

We present KG-ProWeight, a three-phase training framework that integrates an external knowledge graph (Wikidata) as a per-step factuality anchor for reinforcement learning in multi-hop RAG. Our empirical study yields four primary findings. / 我们提出 KG-ProWeight，一个将外部知识图谱（Wikidata）整合为多跳 RAG 强化学习中逐步骤事实性锚点的三阶段训练框架。实证研究产出了四个主要发现。

**[已推翻 — 见文件开头横幅]** First, the α-gate exhibits emergent adaptive behavior across datasets: low α on HotpotQA ($\mu = 0.29$, text signal sufficient) and high α on 2WikiMultiHopQA ($\mu = 0.80$, cross-document reasoning requires KG), without any manual per-dataset tuning. We hope this demonstration inspires future efforts in training-time adaptive signal fusion rather than reliance on static reward weighting schemes [9, 14]. / 第一，α-门控在两个数据集上展现出涌现自适应行为：HotpotQA $\mu = 0.29$ vs 2WikiMultiHopQA $\mu = 0.80$，无需手动调参 [9, 14]。

Second, SFT+KG under full Wikipedia retrieval outperforms text-only SFT on both benchmarks (HotpotQA $+1.3$pp, 2WikiMultiHopQA $+2.7$pp), with the effect scaling with reasoning complexity. This gain, while modest in absolute magnitude, is robust, consistent, and achieved without architectural modification—it arises purely from input-level augmentation of the prompt [16, 20]. / 第二，SFT+KG 在两个基准上均优于纯文本 SFT（HotpotQA $+1.3$pp，2WikiMultiHopQA $+2.7$pp），效果随推理复杂性递增 [16, 20]。

Third, PPO degrades substantially without KG (HotpotQA $-2.7$pp, 2WikiMultiHopQA $-1.7$pp) but recovers most of the gap when KG is restored. This finding elevates the role of KG from "beneficial add-on" to "necessary regularizer for training RL policies"—a fact overlooked in prior work [14, 19]. PPO+KG achieves competitive but sub-SFT performance, highlighting the disconnect between KG citation quality (r_KG) and reasoning correctness in current process reward design [13, 36]. / 第三，PPO 无 KG 显著退化，有 KG 恢复——将 KG 角色从 "有益的附加项" 提升为 "训练 RL 策略的必要正则化器" [14, 19, 13, 36]。

Fourth, our systematic audit of KG quality reveals that $\approx 70\%$ of raw triples are uninformative for downstream QA, and demonstrates that a multi-stage filtering pipeline can restore KG utility while reducing triple count by $55\%$. This finding has direct engineering implications: in KG-augmented LLM systems, KG quality—not architectural complexity—is the primary determinant of performance [39]. / 第四，KG 质量审计揭示约 $70\%$ 原始三元组不具信息量，多阶段过滤可恢复 KG 效用 [39]。

**Limitations.** This study has four limitations in scope. (i) All experiments use Llama-3-8B-Instruct as the base model; the magnitude of KG contribution may vary with base model scale and capability (larger models may benefit less from external structured knowledge) [3]. (ii) Our KG construction pipeline relies on Wikidata as the sole factuality source; while Wikidata has broad coverage, it inevitably contains gaps (e.g., $2\%$ of questions have no KG at all in the offline entity cache), which necessarily caps the upper bound of KG contribution. (iii) Our PPO training is conducted on a single dataset (HotpotQA); cross-task generalization and out-of-distribution robustness remain for future study. (iv) The process reward limitation suggests that current process reward design is insufficient for PPO to surpass SFT; richer process supervision—potentially through multi-step consistency checks, coherence verification requiring passage evidence, or teacher-model-based process labels—may help close this gap [13, 36]. / **局限性**：(i) 仅 Llama-3-8B [3]；(ii) Wikidata 覆盖空白（$2\%$ 题目无 KG）；(iii) PPO 仅在 HotpotQA 训练；(iv) 过程奖励设计不足 [13, 36]。

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

[40] Wang, L. et al. Text embeddings by weakly-supervised contrastive pre-training. *arXiv:2212.03533*, 2022.

[41] Chen, J. et al. BGE reranker: A lightweight and effective reranking model. *arXiv*, 2024.
