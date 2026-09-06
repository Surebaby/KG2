"""Gold-free helpers for the canonical-subquestion QA v9 diagnostic.

The v8 experiment asked the strong SFT checkpoint to emit a bespoke one-line
subanswer.  This module changes only that interface: a subquestion is presented
through the same :func:`build_inference_messages` schema used by ordinary SFT
evaluation, and the concise value is read from ``[Final Answer]``.  Provenance
admission remains the frozen v8 lexical binder, so a prompt change cannot be
mistaken for a new evidence verifier.

This module does not load datasets, retrieve documents, access Gold labels, or
score answers.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping, Sequence

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_inference_messages
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    EXPECTED_TOP_K,
    parse_and_bind_subanswer,
    project_top10_passages_for_prompt,
)


CANONICAL_SUBQA_VERSION = "canonical-subquestion-qa-v9-phase0-1"
PARSER_VERSION = "canonical-final-answer-parser-v9-phase0-1"
MAX_KG_TRIPLES = 0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_canonical_subqa_messages(
    *,
    subquestion: str,
    retrieved_passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the exact legacy-SFT inference prompt for one subquestion.

    ``project_top10_passages_for_prompt`` validates the same ten-document
    projection used by v8's final reader.  The KG block is deliberately empty:
    phase 0 isolates the answer-interface mismatch using the already frozen q1
    retrieval output.
    """

    if not isinstance(subquestion, str) or not subquestion.strip():
        raise ValueError("subquestion must be non-empty text")
    if subquestion != subquestion.strip() or "\n" in subquestion or "\r" in subquestion:
        raise ValueError("subquestion must be one unpadded line")
    passages = project_top10_passages_for_prompt(
        retrieved_passages,
        role="canonical_subqa_v9_q1",
    )
    if len(passages) != EXPECTED_TOP_K:
        raise ValueError("canonical subquestion QA requires exactly ten passages")
    return build_inference_messages(
        question=subquestion,
        retrieved_passages=passages,
        kg_triples=[],
        top_k=EXPECTED_TOP_K,
        max_kg_triples=MAX_KG_TRIPLES,
    )


def parse_and_bind_canonical_subanswer(
    generation: str,
    *,
    subquestion: str,
    retrieved_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract ``[Final Answer]`` and apply the unchanged v8 provenance binder."""

    if not isinstance(generation, str):
        raise TypeError("generation must be text")
    answer = extract_final_answer(generation)
    steps = parse_steps(generation, known_kg=[])
    binding = parse_and_bind_subanswer(
        answer if answer is not None else "",
        q1_query=subquestion,
        q1_passages=retrieved_passages,
    )
    return {
        "schema_version": CANONICAL_SUBQA_VERSION,
        "parser_version": PARSER_VERSION,
        "gold_access": False,
        "generation_sha256": _sha256_text(generation),
        "final_answer_parsed": answer is not None,
        "final_answer": answer,
        "final_answer_sha256": _sha256_text(answer) if answer is not None else None,
        "parsed_step_count": len(steps),
        "has_step_trace": bool(steps),
        "binding": deepcopy(binding),
    }


__all__ = [
    "CANONICAL_SUBQA_VERSION",
    "MAX_KG_TRIPLES",
    "PARSER_VERSION",
    "build_canonical_subqa_messages",
    "parse_and_bind_canonical_subanswer",
]
