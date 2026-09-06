"""Six structural source-use features; no semantic or answer verification.

The two v2 additions measure distinct visible-edge coverage and the weakest
step's citation precision.  They are invariant to repeated citation occurrences
and exact step duplication.  A changed final answer or changed free text with
unchanged citations deliberately has the same representation.

Neither addition reads execution-hop coverage, a ProofKG score/component, a
derived answer, a ReaRAG score, or a gold label.  Coverage can correlate with the
structural Graph target; fitting that heuristic target is not independent
evidence of source reliability.  This module preserves the legacy hard gate;
the caller must subsequently apply its frozen source-credit mask.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES as V1_FEATURE_NAMES,
    compute_gate_features,
)


FEATURE_VERSION = "source-quality-trajectory-features-v2"
FEATURE_NAMES = V1_FEATURE_NAMES + (
    "source_edge_coverage",
    "min_step_citation_precision",
)


def compute_gate_features_v2(
    spec: Any, steps: Sequence[Any], proof_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the unchanged hard gate with the registered six feature values.

    For visible unique triples ``G`` and unique cited triples ``C_s`` at step
    ``s``, coverage is ``|G intersect union(C_s)| / |G|``.  Per-step precision is
    ``|G intersect C_s| / (|C_s| + |U_s| + malformed_s)`` where ``U_s`` is the
    set of nonempty unknown citation surfaces.  Empty denominators yield zero;
    the second feature is the minimum over steps, or zero for no steps.

    Only ``scorer_version`` is read from a nonempty proof result.  An empty
    result retains v1's pre-scoring/invalid-trajectory contract.  These are
    learned inputs with no hand-assigned positive reward or monotonicity claim.
    """
    version_only = (
        {"scorer_version": proof_result.get("scorer_version")}
        if proof_result else {}
    )
    result = compute_gate_features(spec, steps, version_only)
    visible = {
        tuple(str(value).strip() for value in triple)
        for triple in (getattr(spec, "kg_subgraph", []) or [])
        if isinstance(triple, (list, tuple)) and len(triple) == 3
    }
    cited_union: set[tuple[str, str, str]] = set()
    step_precisions = []
    for step in steps:
        cited = {
            tuple(str(value).strip() for value in triple)
            for triple in (getattr(step, "cited_triples", []) or [])
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        }
        unknown = {
            str(surface).strip()
            for surface in (getattr(step, "unknown_citation_surfaces", []) or [])
            if str(surface).strip()
        }
        malformed = int(bool(getattr(step, "knowledge_used_malformed_content", False)))
        denominator = len(cited) + len(unknown) + malformed
        step_precisions.append(len(cited & visible) / denominator if denominator else 0.0)
        cited_union.update(cited)
    result["feature_version"] = FEATURE_VERSION
    result["values"].update({
        "source_edge_coverage": len(cited_union & visible) / len(visible) if visible else 0.0,
        "min_step_citation_precision": min(step_precisions, default=0.0),
    })
    result["telemetry"].update({
        "trajectory_feature_scope": "visible_citation_structure_only",
        "free_text_semantics_verified": False,
        "proof_score_or_components_used": False,
        "text_score_used": False,
        "gold_answer_used": False,
        "source_unique_edge_count": len(visible),
        "source_unique_cited_edge_count": len(cited_union & visible),
    })
    return result
