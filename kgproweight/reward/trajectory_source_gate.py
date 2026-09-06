"""Dataset-agnostic hard gate for trajectory-level Graph credit.

The gate answers only whether a per-question graph is safe enough to enter the
Graph reward branch.  It never chooses a dataset, never reads a gold answer,
and never estimates the learned alpha value.  ``m_graph`` is therefore a hard
fail-closed mask; the learned/fixed source credit is a separate experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Dict, Mapping, Sequence

from kgproweight.kg.question_kg import (
    question_key,
    question_sha256,
    validate_question_kg_record,
)


SOURCE_GATE_SCHEMA_VERSION = "trajectory-source-gate-v1"
SOURCE_GATE_VERSION = "trajectory-source-gate-hard-mask-v1"
_PID = re.compile(r"^P\d+$")
_QID = re.compile(r"^Q\d+$")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalised_triples(values: Sequence[Sequence[Any]]) -> list[tuple[str, str, str]]:
    triples = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        triple = tuple(str(part).strip() for part in value)
        if all(triple):
            triples.append(triple)
    return triples


@dataclass(frozen=True)
class GraphGateDecision:
    m_graph: int
    graph_eligible: bool
    routing_reason: str
    checks: Dict[str, bool]
    kg_sha256: str
    execution_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_graph_gate(
    record: Mapping[str, Any],
    *,
    dataset: str,
    qid: str,
    question: str,
    historical_cutoff: str,
) -> GraphGateDecision:
    """Return a strict Graph-credit decision for one question-KG record.

    ``historical_cutoff`` is supplied by the versioned evidence-store manifest;
    old per-question records did not repeat it.  Requiring it here ensures a new
    release explicitly binds the shared upstream provenance instead of falsely
    claiming that the old rows already contained a cutoff.
    """

    checks: Dict[str, bool] = {}
    try:
        validate_question_kg_record(record)
        checks["schema_valid"] = True
    except (TypeError, ValueError):
        checks["schema_valid"] = False

    expected_key = question_key(dataset, qid)
    checks["identity_match"] = (
        str(record.get("question_key") or "") == expected_key
        and str(record.get("dataset") or "").strip().lower()
        == str(dataset).strip().lower()
        and str(record.get("qid") or "").strip() == str(qid).strip()
        and str(record.get("question") or "").strip() == str(question).strip()
        and str(record.get("question_sha256") or "")
        == question_sha256(question)
    )

    triples = _normalised_triples(record.get("kg_subgraph") or [])
    checks["graph_nonempty"] = bool(triples)
    checks["graph_components_nonempty"] = bool(triples) and all(
        all(part for part in triple) for triple in triples
    )

    provenance = record.get("provenance") or {}
    execution = record.get("execution") or {}
    plan = record.get("query_plan") or {}
    checks["gold_access_false"] = provenance.get("gold_access") is False
    checks["runtime_error_zero"] = record.get("runtime_error") in (None, "")
    checks["cutoff_bound"] = bool(str(historical_cutoff or "").strip())
    checks["provenance_bound"] = bool(
        str(provenance.get("builder_version") or "").strip()
    )
    checks["plan_recognized"] = bool(plan.get("recognized"))
    checks["complete_declared"] = (
        provenance.get("complete_plan_execution") is True
        and execution.get("complete_plan_execution") is True
    )

    planned_hops = list(plan.get("hops") or [])
    executed_hops = list(execution.get("hops") or [])
    executed_by_index = {
        int(hop.get("hop_index", -1)): hop
        for hop in executed_hops
        if isinstance(hop, Mapping)
    }
    trace_triples: set[tuple[str, str, str]] = set()
    hop_contract_ok = bool(planned_hops) and len(executed_by_index) >= len(planned_hops)
    for index, planned in enumerate(planned_hops, start=1):
        executed = executed_by_index.get(index)
        pids = [str(pid).strip() for pid in (planned.get("pids") or [])]
        matches = _normalised_triples((executed or {}).get("matches") or [])
        input_entities = list((executed or {}).get("input_entities") or [])
        inputs_have_qid = bool(input_entities) and all(
            isinstance(entity, Mapping)
            and bool(_QID.fullmatch(str(entity.get("qid") or "")))
            for entity in input_entities
        )
        if (
            executed is None
            or not pids
            or not all(_PID.fullmatch(pid) for pid in pids)
            or not matches
            or not inputs_have_qid
        ):
            hop_contract_ok = False
        trace_triples.update(matches)
    checks["all_hops_executed_with_qid_pid_tail"] = hop_contract_ok
    checks["retained_edges_traceable"] = bool(triples) and set(triples).issubset(
        trace_triples
    )
    checks["no_duplicate_edges"] = len(triples) == len(set(triples))

    graph_eligible = all(checks.values())
    if graph_eligible:
        reason = "identity_safe_complete_traceable_graph"
    elif not checks["graph_nonempty"]:
        reason = "no_trusted_graph"
    else:
        failed = [name for name, passed in checks.items() if not passed]
        reason = "failed:" + ",".join(failed)
    return GraphGateDecision(
        m_graph=int(graph_eligible),
        graph_eligible=graph_eligible,
        routing_reason=reason,
        checks=checks,
        kg_sha256=_canonical_sha256(record.get("kg_subgraph") or []),
        execution_sha256=_canonical_sha256(execution),
    )


def make_source_gate_record(
    record: Mapping[str, Any],
    *,
    dataset: str,
    qid: str,
    question: str,
    text_evidence_available: bool,
    historical_cutoff: str,
) -> Dict[str, Any]:
    decision = evaluate_graph_gate(
        record,
        dataset=dataset,
        qid=qid,
        question=question,
        historical_cutoff=historical_cutoff,
    )
    provenance = record.get("provenance") or {}
    return {
        "schema_version": SOURCE_GATE_SCHEMA_VERSION,
        "gate_version": SOURCE_GATE_VERSION,
        "question_key": question_key(dataset, qid),
        "dataset": str(dataset).strip().lower(),
        "qid": str(qid).strip(),
        "question_sha256": question_sha256(question),
        "text_evidence_available": bool(text_evidence_available),
        "graph_eligible": decision.graph_eligible,
        "m_graph": decision.m_graph,
        "proof_source": (
            str(provenance.get("builder_version") or "")
            if decision.graph_eligible
            else "none"
        ),
        "routing_reason": decision.routing_reason,
        "eligibility_checks": decision.checks,
        "kg_sha256": decision.kg_sha256,
        "execution_sha256": decision.execution_sha256,
        "historical_cutoff": historical_cutoff if decision.graph_eligible else None,
    }

