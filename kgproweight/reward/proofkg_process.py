"""Gold-free trajectory reward for automatically executed ProofKG paths."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Any, Mapping, Sequence

from kgproweight.kg.question_kg import question_key


COMPARISON_CUES = re.compile(
    r"\b(which|who)\b.*\b(first|earlier|later|older|younger|more|less|higher|lower|"
    r"longer|shorter|before|after)\b|\b(first|earlier|later|older|younger)\b",
    re.IGNORECASE,
)


def _norm(value: object) -> str:
    # Keep this byte-for-byte equivalent in meaning to the frozen confirmation
    # scorer's phrase matching.  In particular, articles are significant here:
    # this is path/node matching, not answer-metric normalisation.
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _answer_norm(value: object) -> str:
    text = _norm(value)
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def _phrase_in(phrase: object, text: object) -> bool:
    needle, haystack = _norm(phrase), _norm(text)
    return bool(needle and re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def token_f1(prediction: str, gold: str) -> float:
    predicted, target = _answer_norm(prediction).split(), _answer_norm(gold).split()
    if not predicted or not target:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def canonical_answer_normalize(value: object) -> str:
    """Mirror FlashRAG's answer normalizer without importing its runtime."""

    text = str(value or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonical_exact_match(prediction: str, gold: str) -> float:
    """FlashRAG-compatible normalized exact match for one answer surface."""

    predicted = canonical_answer_normalize(prediction)
    target = canonical_answer_normalize(gold)
    return float(predicted == target)


def canonical_token_f1(prediction: str, gold: str) -> float:
    """FlashRAG-compatible token F1, including its yes/no guard.

    The historical ``token_f1`` above deliberately remains unchanged because
    existing automatic-ProofKG runs used it.  Mixed PPO opts into this canonical
    variant so a prediction such as ``"not yes"`` cannot receive partial F1
    against the boolean Gold ``"yes"`` while canonical evaluation assigns 0.
    """

    predicted = canonical_answer_normalize(prediction)
    target = canonical_answer_normalize(gold)
    special = {"yes", "no", "noanswer"}
    if predicted in special and predicted != target:
        return 0.0
    if target in special and predicted != target:
        return 0.0
    predicted_tokens, target_tokens = predicted.split(), target.split()
    if not predicted_tokens or not target_tokens:
        return 0.0
    overlap = sum((Counter(predicted_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted_tokens), overlap / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


def is_automatic_proofkg(record: Mapping[str, Any] | None, kg: Sequence[Sequence[str]]) -> bool:
    """Strict eligibility: nonempty executed plan whose builder asserts no Gold."""

    if not record or not kg:
        return False
    provenance = record.get("provenance") or {}
    hops = (record.get("query_plan") or {}).get("hops") or []
    # Partial executions must fall back to the legacy branch.  A plan with two
    # hops but only one materialised edge is not a complete proof path.
    complete = provenance.get("complete_plan_execution")
    # Older frozen confirmation artifacts predate the explicit flag.  Their
    # conservative compatibility check is edge-count >= planned-hop count; all
    # newly built training artifacts carry the exact executor decision.
    enough_edges = len(kg) >= len(hops)
    complete_enough = (
        complete is True and enough_edges if complete is not None else enough_edges
    )
    return provenance.get("gold_access") is False and bool(hops) and complete_enough


def is_identity_safe_automatic_proofkg(
    record: Mapping[str, Any] | None,
    kg: Sequence[Sequence[str]],
    *,
    dataset: object,
    qid: object,
) -> bool:
    """Return whether a complete automatic proof came from the exact row join.

    ``is_automatic_proofkg`` intentionally preserves the historical structural
    predicate.  Mixed-dataset PPO needs one additional guard: a structurally
    valid-looking runtime record must have been joined through the versioned
    ``dataset::qid`` contract.  The training loader validates the question text
    and SHA256 before it writes ``question_key`` into runtime metadata; checking
    that key here prevents stale/pre-existing silver metadata from opting an
    unrelated row into process reward.
    """

    if not record:
        return False
    try:
        expected = question_key(str(dataset or ""), str(qid or ""))
    except ValueError:
        return False
    return (
        str(record.get("question_key") or "") == expected
        and is_automatic_proofkg(record, kg)
    )


def required_steps(record: Mapping[str, Any]) -> int:
    planned = len((record.get("query_plan") or {}).get("hops") or [])
    return max(2, min(3, planned or 2))


def _reachable_edges(question: str, kg, cited_by_step):
    triples = [tuple(str(value).strip() for value in edge) for edge in kg if len(edge) == 3]
    reachable_nodes = {
        node
        for head, _, tail in triples
        for node in (head, tail)
        if _phrase_in(node, question)
    }
    reachable_edges = set()
    for citations in cited_by_step:
        changed = True
        while changed:
            changed = False
            for edge in map(tuple, citations):
                if edge not in reachable_edges and edge[0] in reachable_nodes:
                    reachable_edges.add(edge)
                    reachable_nodes.add(edge[2])
                    changed = True
    return triples, reachable_edges, reachable_nodes


def score_grounded_process(*, question: str, kg, steps, predicted_answer: str) -> dict[str, float]:
    """Score citations/path structure without reading a Gold answer."""

    triples, reachable, reachable_nodes = _reachable_edges(
        question, kg, [step.cited_triples for step in steps]
    )
    known = [triple for step in steps for triple in step.cited_triples]
    unknown = sum(len(step.unknown_citation_surfaces) for step in steps)
    attempts = len(known) + unknown
    citation_precision = len(known) / attempts if attempts else 0.0
    grounded = sum(
        int(_phrase_in(head, step.intermediate_conclusion or "") or
            _phrase_in(tail, step.intermediate_conclusion or ""))
        for step in steps for head, _, tail in step.cited_triples
    )
    conclusion_grounding = grounded / len(known) if known else 0.0
    edge_coverage = len(reachable) / len(set(triples)) if triples else 0.0
    outgoing = {head for head, _, _ in triples}
    terminals = {tail for _, _, tail in triples if tail not in outgoing}
    supported = (
        {head for head, _, _ in reachable if _phrase_in(head, question)}
        if COMPARISON_CUES.search(question)
        else terminals.intersection(reachable_nodes)
    )
    answer_alignment = float(
        bool(predicted_answer) and any(
            _phrase_in(predicted_answer, value) or _phrase_in(value, predicted_answer)
            for value in supported
        )
    )
    unknown_ratio = unknown / attempts if attempts else 0.0
    duplicate_ratio = (len(known) - len(set(known))) / len(known) if known else 0.0
    score = (
        0.25 * citation_precision
        + 0.25 * conclusion_grounding
        + 0.30 * edge_coverage
        + 0.20 * answer_alignment
        - 0.50 * unknown_ratio
        - 0.15 * duplicate_ratio
    )
    return {
        "score": float(score),
        "citation_precision": float(citation_precision),
        "conclusion_grounding": float(conclusion_grounding),
        "reachable_edge_coverage": float(edge_coverage),
        "answer_path_alignment": float(answer_alignment),
        "unknown_citation_ratio": float(unknown_ratio),
        "duplicate_citation_ratio": float(duplicate_ratio),
    }
