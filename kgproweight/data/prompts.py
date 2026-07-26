"""Canonical prompt schema — the *only* place prompts are defined.

Every Teacher / SFT / RL / inference / PRM-annotation call must build its
prompt through one of the ``build_*_messages`` factories below. This fixes
bug #6: the legacy code had three drifting schemas (``[Step N]`` for the
Teacher, ``<|begin_of_query|>`` for RL, FlashRAG's default for inference)
and PRMAnnotator's regex parser became fragile as a result.

All factories return a ``list[ChatMessage]`` ready for
``tokenizer.apply_chat_template`` or the OpenAI chat completions API.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

from kgproweight.retrieval.hybrid import DEFAULT_TOPK

# ---------------------------------------------------------------------------
# Step / answer markers — single source of truth
# ---------------------------------------------------------------------------

STEP_OPEN = "[Step {n}]"
STEP_CLOSE = "[/Step]"
FINAL_ANSWER_OPEN = "[Final Answer]"


# ---------------------------------------------------------------------------
# Teacher prompts (Phase 1)
# ---------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = """You are a careful multi-hop reasoning assistant.

For every question, you MUST emit your reasoning trace using the exact schema below.
There are between 3 and 7 steps. Each step contains a small set of factual claims
expressed both as natural language and as KG triples drawn from the supplied
Knowledge Graph context. The Final Answer is a short string (entity, year, name, etc.).

Schema (use brackets EXACTLY as written):

[Step 1]
Reasoning: <natural-language reasoning for this step>
Knowledge Used: [(head_entity, relation, tail_entity), ...]
Conclusion: <one short factual conclusion>

[Step 2]
...

[Final Answer]
<final concise answer>

Hard requirements:
- Cite ONLY triples that are present in the supplied Knowledge Graph. If the
  Knowledge Graph is empty, write ``Knowledge Used: []``.
- If the Knowledge Graph is NOT empty, each step must include at least one
  triple in ``Knowledge Used``.
- The trace must contain 3 to 7 steps (inclusive).
- Do NOT include any text after [Final Answer].
- Keep each Reasoning sentence under 50 words.
- Do not rely on world knowledge when KG has relevant facts; prefer KG-backed
  reasoning and copy triple surface forms exactly from the KG block.

Output quality checklist before you answer:
1) Step count is between 3 and 7.
2) Every step has at least one KG triple when KG is non-empty.
3) Final answer is short and directly answers the question.
"""


TEACHER_USER_TEMPLATE = """Question: {question}

Retrieved Passages (top-{top_k}):
{retrieved_block}

Knowledge Graph (2-hop):
{kg_block}

Generate the multi-step reasoning trace and Final Answer following the schema.
"""


# ---------------------------------------------------------------------------
# SFT / inference prompts (Phase 3a + eval)
# ---------------------------------------------------------------------------

SFT_SYSTEM_PROMPT = """You are a multi-hop reasoning assistant.

Always answer using the schema:

[Step 1]
Reasoning: <natural-language reasoning for this step>
Knowledge Used: [(head_entity, relation, tail_entity), ...]
Conclusion: <one short factual conclusion>

[Step 2]
...

[Final Answer]
<answer>

Strict formatting rules:
- Knowledge Used MUST use the EXACT format: [(h, r, t), (h, r, t), ...]
  with square brackets around the list and parentheses around each triple.
- Copy triples VERBATIM from the Knowledge Graph block when available.
- Do NOT invent relations. If the Knowledge Graph is empty, write [].
- Every step MUST include both a Conclusion and Knowledge Used field.
- If you cite no triples, write: Knowledge Used: []
- Stop generating after [Final Answer].
"""


SFT_USER_TEMPLATE = """Question: {question}

Retrieved Passages:
{retrieved_block}

[Knowledge Graph Context]
{kg_block}
[End of Knowledge Graph]

Use the Knowledge Graph as a reference. When citing facts, copy the exact
(head, relation, tail) triples from the Knowledge Graph into Knowledge Used.
Do not invent relations or entities that are not present in the Knowledge Graph.
"""


INFERENCE_USER_TEMPLATE = SFT_USER_TEMPLATE


# ---------------------------------------------------------------------------
# RL prompts (Phase 3b PPO)
# ---------------------------------------------------------------------------

RL_SYSTEM_PROMPT = SFT_SYSTEM_PROMPT


RL_USER_TEMPLATE = SFT_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Block formatters
# ---------------------------------------------------------------------------

KG_BLOCK_TEMPLATE = "  ({head}, {relation}, {tail})"
RETRIEVED_BLOCK_TEMPLATE = "[{rank}] {text}"


def format_kg_block(triples: Sequence[Sequence[str]], max_triples: int = 50) -> str:
    """Render a triple list as one ``(h, r, t)`` per line."""
    if not triples:
        return "  (empty)"
    lines: List[str] = []
    for triple in list(triples)[:max_triples]:
        if len(triple) != 3:
            continue
        h, r, t = (str(x).strip() for x in triple)
        if not (h and r and t):
            continue
        lines.append(KG_BLOCK_TEMPLATE.format(head=h, relation=r, tail=t))
    return "\n".join(lines) if lines else "  (empty)"


def format_retrieved_block(
    passages: Sequence[Mapping | str],
    max_passages: int = DEFAULT_TOPK,
    max_tokens: int = 0,
    chars_per_token: int = 4,
) -> str:
    """Render a list of FlashRAG-style passage dicts (or raw strings).

    When ``max_tokens > 0``, passages are packed by token budget instead of
    fixed count. Passages are assumed pre-sorted by relevance (reranker output).
    Each passage is truncated to 1200 chars (~300 tokens) max.
    """
    if not passages:
        return "  (no passages retrieved)"

    passage_list = list(passages)
    out: List[str] = []
    budget_used = 0
    n_skipped = 0

    for rank, p in enumerate(passage_list, start=1):
        if isinstance(p, Mapping):
            text = str(p.get("contents") or p.get("text") or "").strip()
        else:
            text = str(p).strip()
        if not text:
            n_skipped += 1
            continue

        # Trim to 1200 chars max (~300 tokens at 4 chars/token)
        if len(text) > 1200:
            text = text[:1200] + " …"

        tok_est = len(text) // chars_per_token

        # Token budget check
        if max_tokens > 0:
            if budget_used + tok_est > max_tokens:
                # Try to fit a truncated version
                remaining = max_tokens - budget_used
                if remaining > 200:  # at least ~50 tokens
                    text = text[:remaining * chars_per_token] + " …"
                    tok_est = len(text) // chars_per_token
                else:
                    n_skipped += 1
                    continue
            budget_used += tok_est

        out.append(RETRIEVED_BLOCK_TEMPLATE.format(rank=rank, text=text))

        # Fixed-count path: stop at max_passages
        if max_tokens == 0 and len(out) >= max_passages:
            n_skipped = len(passage_list) - rank
            break

    if n_skipped > 0:
        logger.debug("Passage budget: %d included, %d skipped (budget=%s)",
                     len(out), n_skipped,
                     f"{max_tokens}tok" if max_tokens > 0 else f"{max_passages}cnt")

    return "\n".join(out) if out else "  (no passages retrieved)"


# ---------------------------------------------------------------------------
# Message-builders
# ---------------------------------------------------------------------------

ChatMessage = dict


def build_teacher_messages(
    question: str,
    retrieved_passages: Sequence[Mapping | str],
    kg_triples: Sequence[Sequence[str]],
    top_k: int = DEFAULT_TOPK,
    max_kg_triples: int = 50,
) -> List[ChatMessage]:
    user = TEACHER_USER_TEMPLATE.format(
        question=question.strip(),
        top_k=top_k,
        retrieved_block=format_retrieved_block(retrieved_passages, max_passages=top_k),
        kg_block=format_kg_block(kg_triples, max_triples=max_kg_triples),
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_sft_messages(
    question: str,
    retrieved_passages: Sequence[Mapping | str],
    kg_triples: Sequence[Sequence[str]],
    answer_trace: Optional[str] = None,
    top_k: int = DEFAULT_TOPK,
    max_kg_triples: int = 50,
) -> List[ChatMessage]:
    user = SFT_USER_TEMPLATE.format(
        question=question.strip(),
        retrieved_block=format_retrieved_block(retrieved_passages, max_passages=top_k),
        kg_block=format_kg_block(kg_triples, max_triples=max_kg_triples),
    )
    msgs: List[ChatMessage] = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if answer_trace is not None:
        msgs.append({"role": "assistant", "content": answer_trace})
    return msgs


def build_rl_messages(
    question: str,
    retrieved_passages: Sequence[Mapping | str],
    kg_triples: Sequence[Sequence[str]],
    top_k: int = DEFAULT_TOPK,
    max_kg_triples: int = 50,
) -> List[ChatMessage]:
    return build_sft_messages(
        question=question,
        retrieved_passages=retrieved_passages,
        kg_triples=kg_triples,
        answer_trace=None,
        top_k=top_k,
        max_kg_triples=max_kg_triples,
    )


def build_inference_messages(
    question: str,
    retrieved_passages: Sequence[Mapping | str],
    kg_triples: Sequence[Sequence[str]],
    top_k: int = DEFAULT_TOPK,
    max_kg_triples: int = 50,
) -> List[ChatMessage]:
    return build_sft_messages(
        question=question,
        retrieved_passages=retrieved_passages,
        kg_triples=kg_triples,
        answer_trace=None,
        top_k=top_k,
        max_kg_triples=max_kg_triples,
    )
