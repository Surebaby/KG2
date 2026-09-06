"""Gold-free rank-first provenance binding for canonical sub-QA v9.1.

This module changes exactly one admission rule relative to v9 phase 0.  An
answer surface may occur in more than one of the already retrieved top-10
documents; when it does, the document with the smallest retrieval rank is
selected deterministically.  All lexical matching, Unicode normalization,
numeric-boundary, evidence-unit, excerpt, and offset behavior remains owned by
the frozen v8 binder.

The resulting binding proves only that the parsed answer occurs locally in a
model-visible document.  It is not evidence that the document semantically
supports the answer.  This module never loads a dataset, reads Gold labels, or
scores an answer.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping, Sequence

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.retrieval.canonical_subqa_v9 import (
    PARSER_VERSION,
    build_canonical_subqa_messages,
)
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    DynamicDecompositionV8Error,
    parse_and_bind_subanswer,
    project_top10_passages_for_prompt,
)


CANONICAL_SUBQA_VERSION = "canonical-subquestion-qa-v9.1-rank-first-binder-1"
PROVENANCE_BINDER_VERSION = "dynamic-decomposition-v9.1-rank-first-surface-binder-1"
VERIFICATION_SCOPE = "lexical_surface_locality_only_not_semantic_support"
SELECTION_POLICY = "minimum_retrieval_rank_among_surface_matching_documents"

# A non-lexical placeholder cannot be accepted by the frozen v8 subanswer
# parser, so it cannot accidentally match any admissible answer surface.
_MASKED_DOCUMENT_TEXT = "∅"
_RUNTIME_PROJECTION_FIELDS = (
    "binder_version",
    "gold_access",
    "verified",
    "reason",
    "verification_scope",
    "selection_policy",
    "selection_reason",
    "verified_answer",
    "supporting_document_key",
    "supporting_doc_id",
    "supporting_doc_rank",
    "supporting_sentence",
    "supporting_sentence_sha256",
    "support_location",
    "support_unit_index",
    "bound_evidence_excerpt",
    "bound_evidence_excerpt_sha256",
    "supporting_document_prompt_sha256",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _upgrade_binding(
    binding: Mapping[str, Any],
    *,
    selection_reason: str,
) -> dict[str, Any]:
    """Attach the v9.1 contract without mutating a v8 binding."""

    result = deepcopy(dict(binding))
    result["binder_version"] = PROVENANCE_BINDER_VERSION
    result["verification_scope"] = VERIFICATION_SCOPE
    result["selection_policy"] = SELECTION_POLICY
    result["selection_reason"] = selection_reason
    result.setdefault("selected_document_matching_unit_count", 0)
    return result


def _masked_probe_passages(
    passages: Sequence[Mapping[str, Any]],
    *,
    visible_rank: int,
) -> list[dict[str, str]]:
    """Keep one prompt-visible document and mask the other nine.

    Calling the frozen v8 binder on this projection tells us whether the
    answer occurs in the selected document while retaining v8's exact
    evidence-unit and source-offset implementation.
    """

    projected = project_top10_passages_for_prompt(
        passages,
        role="canonical_subqa_v9_1_rank_probe",
    )
    if not 1 <= visible_rank <= len(projected):
        raise DynamicDecompositionV8Error("visible_rank is outside the top-10")
    probes: list[dict[str, str]] = []
    for rank, document in enumerate(projected, start=1):
        if rank == visible_rank:
            probes.append(deepcopy(document))
        else:
            probes.append(
                {
                    "doc_id": str(document["doc_id"]),
                    "title": _MASKED_DOCUMENT_TEXT,
                    "text": _MASKED_DOCUMENT_TEXT,
                }
            )
    return probes


def bind_subanswer_rank_first(
    answer: str,
    *,
    subquestion: str,
    retrieved_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind one concise answer to the highest-ranked matching document.

    The frozen v8 binder is called first, so null-like values, booleans,
    subquestion echoes, absent surfaces, unsafe inputs, and exact lexical
    boundaries retain their prior behavior.  Only v8's
    ``answer_surface_ambiguous_across_documents`` outcome is relaxed.
    """

    base = parse_and_bind_subanswer(
        answer,
        q1_query=subquestion,
        q1_passages=retrieved_passages,
    )
    if base.get("verified") is True:
        result = _upgrade_binding(base, selection_reason="only_surface_matching_document")
        result["selected_document_matching_unit_count"] = int(
            base.get("matching_unit_count", 0)
        )
        result["reason"] = "verified_rank_first_document_surface"
        return result

    if base.get("reason") != "answer_surface_ambiguous_across_documents":
        return _upgrade_binding(
            base,
            selection_reason=f"fail_closed:{base.get('reason', 'unknown_reason')}",
        )

    matching_probes: list[dict[str, Any]] = []
    # Input order is the retrieval-rank contract.  Iteration therefore makes
    # the first verified probe the deterministic minimum-rank document.
    for rank in range(1, 11):
        probe = parse_and_bind_subanswer(
            answer,
            q1_query=subquestion,
            q1_passages=_masked_probe_passages(
                retrieved_passages,
                visible_rank=rank,
            ),
        )
        if probe.get("verified") is True:
            matching_probes.append(probe)

    expected_document_count = int(base.get("matching_document_count", 0))
    if len(matching_probes) != expected_document_count or not matching_probes:
        raise DynamicDecompositionV8Error(
            "rank-first probes disagree with the frozen v8 document count"
        )
    total_unit_count = sum(int(row.get("matching_unit_count", 0)) for row in matching_probes)
    if total_unit_count != int(base.get("matching_unit_count", 0)):
        raise DynamicDecompositionV8Error(
            "rank-first probes disagree with the frozen v8 evidence-unit count"
        )

    selected = min(
        matching_probes,
        key=lambda row: int(row["supporting_doc_rank"]),
    )
    result = _upgrade_binding(
        selected,
        selection_reason="minimum_rank_selected_from_multiple_surface_matching_documents",
    )
    result["reason"] = "verified_rank_first_document_surface"
    result["matching_document_count"] = expected_document_count
    result["matching_unit_count"] = total_unit_count
    result["selected_document_matching_unit_count"] = int(
        selected.get("matching_unit_count", 0)
    )
    return result


def parse_and_bind_canonical_subanswer(
    generation: str,
    *,
    subquestion: str,
    retrieved_passages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Parse canonical ``[Final Answer]`` output and apply v9.1 binding."""

    if not isinstance(generation, str):
        raise TypeError("generation must be text")
    answer = extract_final_answer(generation)
    steps = parse_steps(generation, known_kg=[])
    binding = bind_subanswer_rank_first(
        answer if answer is not None else "",
        subquestion=subquestion,
        retrieved_passages=retrieved_passages,
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
        "binding": binding,
    }


def project_rank_first_binding_for_runtime(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the allowlisted evidence fields needed by a v9.1 runtime.

    The projection deliberately retains the *new* binder version.  It must not
    be passed to a frozen v8 helper whose version guard expects v8 semantics;
    a v9.1 runner can use this projection to build its own dynamic state and
    prioritize the bound q1 document during passage merging.
    """

    if not isinstance(binding, Mapping):
        raise DynamicDecompositionV8Error("rank-first binding must be an object")
    if binding.get("binder_version") != PROVENANCE_BINDER_VERSION:
        raise DynamicDecompositionV8Error("rank-first binding has an unexpected binder version")
    if binding.get("gold_access") is not False or binding.get("verified") is not True:
        raise DynamicDecompositionV8Error(
            "runtime projection requires a verified Gold-free rank-first binding"
        )
    missing = sorted(field for field in _RUNTIME_PROJECTION_FIELDS if binding.get(field) is None)
    if missing:
        raise DynamicDecompositionV8Error(
            f"verified rank-first binding lacks runtime fields: {missing}"
        )
    return {field: deepcopy(binding[field]) for field in _RUNTIME_PROJECTION_FIELDS}


__all__ = [
    "CANONICAL_SUBQA_VERSION",
    "PARSER_VERSION",
    "PROVENANCE_BINDER_VERSION",
    "SELECTION_POLICY",
    "VERIFICATION_SCOPE",
    "bind_subanswer_rank_first",
    "build_canonical_subqa_messages",
    "parse_and_bind_canonical_subanswer",
    "project_rank_first_binding_for_runtime",
]
