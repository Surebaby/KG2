#!/usr/bin/env python
"""Freeze a Gold-free failure decomposition for dependent retrieval v5.

The v5 materialization stopped before answer attachment because its mechanism
gates failed.  This script only summarizes the already-frozen retrieval trace;
it neither reads Gold labels nor changes a retrieval or evaluation decision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "musique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "median": median,
        "max": ordered[-1],
    }


def _selector_rows(row: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for hop in row.get("hops") or []:
        selector = hop.get("bridge_selector")
        if selector:
            yield selector


def _classify(row: Mapping[str, Any]) -> str:
    status = str(row["execution_status"])
    if status == "fallback_no_dependent_step":
        return "no_declared_dependency"
    if status == "fallback_no_candidate_strictly_better":
        return "dependent_retrieved_but_full_question_ce_rejected"
    if status == "executed_changed":
        return "dependent_document_retained"
    if status != "fallback_bridge_abstain":
        return f"other:{status}"
    selectors = list(_selector_rows(row))
    if any(selector.get("profile", {}).get("profile_conflict") for selector in selectors):
        return "bridge_abstain_relation_type_profile_conflict"
    if any(
        "producer_consumer_type_conflict" in decision.get("reasons", [])
        or "high_confidence_type_conflict" in decision.get("reasons", [])
        for selector in selectors
        for decision in selector.get("candidate_decisions") or []
    ):
        return "bridge_abstain_candidate_type_conflict"
    return "bridge_abstain_no_supported_candidate"


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    admission_bases: Counter[str] = Counter()
    accepted_types: Counter[str] = Counter()
    merge_decisions: Counter[str] = Counter()
    rejected_margins: list[float] = []
    selected_margins: list[float] = []
    required_producers = accepted_producers = 0
    dependent_eligible = dependent_query_nonempty = changed = 0

    for row in rows:
        status_counts[str(row["execution_status"])] += 1
        stage_counts[_classify(row)] += 1
        required_producers += int(row.get("selector_summary", {}).get("required_producers", 0))
        accepted_producers += int(row.get("selector_summary", {}).get("accepted_producers", 0))
        if row.get("has_dependent_step"):
            dependent_eligible += 1
            dependent_query_nonempty += int(int(row.get("dependent_query_count", 0)) > 0)
        merge = row.get("merge") or {}
        changed += int(bool(merge.get("changed")))
        for selector in _selector_rows(row):
            for decision in selector.get("candidate_decisions") or []:
                admission_bases[str(decision.get("admission_basis", "UNKNOWN"))] += 1
                for reason in decision.get("reasons") or []:
                    rejection_reasons[str(reason)] += 1
                if decision.get("decision") == "accept":
                    types = decision.get("candidate_type", {}).get("types") or ["unknown"]
                    for candidate_type in types:
                        accepted_types[str(candidate_type)] += 1
        original_scores = {
            str(item["document_key"]): float(item["score"])
            for item in merge.get("evicted_originals") or []
        }
        for item in merge.get("candidate_inventory") or []:
            merge_decisions[str(item.get("decision", "UNKNOWN"))] += 1
        for item in merge.get("rejected_not_strictly_better") or []:
            rejected_margins.append(float(item["score"]) - float(item["compared_original_score"]))
        for item in merge.get("evicted_originals") or []:
            selected_margins.append(float(item["replacement_score"]) - float(item["score"]))

    n = len(rows)
    return {
        "n": n,
        "execution_status_counts": dict(sorted(status_counts.items())),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "dependent_step_eligible": dependent_eligible,
        "dependent_query_nonempty": dependent_query_nonempty,
        "dependent_hop_query_nonempty_rate": dependent_query_nonempty / max(1, dependent_eligible),
        "changed_questions": changed,
        "retained_new_dependent_document_question_rate": changed / max(1, n),
        "bridge_required_producers": required_producers,
        "bridge_accepted_producers": accepted_producers,
        "bridge_producer_acceptance_rate": accepted_producers / max(1, required_producers),
        "selector_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "selector_admission_bases": dict(sorted(admission_bases.items())),
        "accepted_candidate_types": dict(sorted(accepted_types.items())),
        "merge_candidate_decisions": dict(sorted(merge_decisions.items())),
        "rejected_candidate_minus_compared_a_margin": _percentiles(rejected_margins),
        "selected_candidate_minus_evicted_a_margin": _percentiles(selected_margins),
    }


def _compact_row(row: Mapping[str, Any], question: str) -> dict[str, Any]:
    selectors = list(_selector_rows(row))
    accepted_surfaces = [
        surface
        for selector in selectors
        for surface in selector.get("accepted_surfaces") or []
    ]
    expected_types = sorted({
        str(value)
        for selector in selectors
        for value in selector.get("profile", {}).get("expected_types") or []
    })
    rejection_counts: Counter[str] = Counter(
        str(reason)
        for selector in selectors
        for decision in selector.get("candidate_decisions") or []
        for reason in decision.get("reasons") or []
    )
    merge = row.get("merge") or {}
    rejected_margins = [
        float(item["score"]) - float(item["compared_original_score"])
        for item in merge.get("rejected_not_strictly_better") or []
    ]
    selected_margins = [
        float(item["replacement_score"]) - float(item["score"])
        for item in merge.get("evicted_originals") or []
    ]
    dependent_queries = [
        query["query"]
        for hop in row.get("hops") or []
        if hop.get("is_dependent")
        for query in hop.get("queries") or []
    ]
    return {
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question_sha256": row["question_sha256"],
        "question": question,
        "execution_status": row["execution_status"],
        "failure_stage": _classify(row),
        "has_dependent_step": bool(row.get("has_dependent_step")),
        "dependent_queries": dependent_queries,
        "accepted_bridge_surfaces": accepted_surfaces,
        "expected_bridge_types": expected_types,
        "selector_rejection_reasons": dict(sorted(rejection_counts.items())),
        "new_dependent_candidate_count": int(row.get("new_dependent_candidate_count", 0)),
        "selected_new_document_count": len(merge.get("selected_new") or []),
        "rejected_margin_summary": _percentiles(rejected_margins),
        "selected_margin_summary": _percentiles(selected_margins),
        "fallback_exact": bool(row.get("fallback_exact")),
        "gold_access": bool(row.get("gold_access")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", type=Path,
        default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_v5"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_v5_failure_decomposition"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")

    report_path = args.run_dir / "report.json"
    details_path = args.run_dir / "execution_details.jsonl"
    arm_a_path = args.run_dir / "arm_a.jsonl"
    arm_b_path = args.run_dir / "arm_b.jsonl"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol_path = Path(source_report["preregistration"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _sha256(protocol_path) != source_report["preregistration"]["sha256"]:
        raise ValueError("preregistration hash mismatch")
    for key, path in (("arm_a", arm_a_path), ("arm_b", arm_b_path), ("execution_details", details_path)):
        if _sha256(path) != source_report["outputs"][key]["sha256"]:
            raise ValueError(f"source output hash mismatch: {key}")
    if source_report["status"] != "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED":
        raise ValueError("unexpected source materialization status")

    rows = _read_jsonl(details_path)
    arm_a = {(str(row["dataset"]), str(row["qid"])): row for row in _read_jsonl(arm_a_path)}
    arm_b = {(str(row["dataset"]), str(row["qid"])): row for row in _read_jsonl(arm_b_path)}
    keys = [(str(row["dataset"]), str(row["qid"])) for row in rows]
    if len(rows) != 60 or len(set(keys)) != 60 or set(keys) != set(arm_a) or set(keys) != set(arm_b):
        raise ValueError("expected an exact 60-row identity join across v5 artifacts")
    if any(bool(row.get("gold_access")) for row in [*rows, *arm_a.values(), *arm_b.values()]):
        raise ValueError("Gold access detected in a Gold-free audit")

    by_dataset = {
        dataset: _aggregate([row for row in rows if row["dataset"] == dataset])
        for dataset in DATASETS
    }
    mechanism_thresholds = protocol["decision_gates"]["mechanism"]
    mechanism_checks: dict[str, dict[str, bool]] = {}
    for dataset in DATASETS:
        source = source_report["by_dataset"][dataset]
        mechanism_checks[dataset] = {
            "plan_executable_rate": source["plan_executable_rate"]
            >= mechanism_thresholds["plan_executable_rate_min_each_dataset"],
            "dependent_hop_query_nonempty_rate": source["dependent_hop_query_nonempty_rate"]
            >= mechanism_thresholds["dependent_hop_query_nonempty_rate_min_each_dataset"],
            "retained_new_dependent_document_question_rate": source["retained_new_dependent_document_question_rate"]
            >= mechanism_thresholds["retained_new_dependent_document_question_rate_min_each_dataset"],
        }

    args.out.mkdir(parents=True)
    compact_path = args.out / "per_question_gold_free.jsonl"
    with compact_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = (str(row["dataset"]), str(row["qid"]))
            handle.write(json.dumps(_compact_row(row, str(arm_a[key]["question"])), ensure_ascii=False) + "\n")

    all_pass = all(all(checks.values()) for checks in mechanism_checks.values())
    report = {
        "schema_version": "dependent-retrieval-v5-gold-free-failure-decomposition-1",
        "experiment_id": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V5-GOLD-FREE-DIAGNOSIS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL_STOP_GOLD_FREE_MECHANISM_GATE" if not all_pass else "UNEXPECTED_MECHANISM_PASS",
        "development_only": True,
        "gold_access": False,
        "answer_generation_run": False,
        "em_f1_available": False,
        "source_materialization": _source(report_path),
        "protocol": _source(protocol_path),
        "inputs": {
            "arm_a": _source(arm_a_path),
            "arm_b": _source(arm_b_path),
            "execution_details": _source(details_path),
        },
        "safety_summary": source_report["safety_summary"],
        "mechanism_gates": {
            "thresholds": mechanism_thresholds,
            "checks": mechanism_checks,
            "all_pass": all_pass,
        },
        "by_dataset": by_dataset,
        "diagnosis": {
            "bridge_admission_bottleneck": (
                "A substantial fraction of eligible questions never formed a dependent query because no bridge "
                "candidate passed the frozen evidence/type checks."
            ),
            "candidate_utility_bottleneck": (
                "When dependent retrieval did run, the frozen full-question cross-encoder usually scored its "
                "top candidates below the two replaceable Arm-A tail documents, so exact fallback was triggered."
            ),
            "interpretation": (
                "v5 removed forced harmful injection but became too conservative to satisfy its preregistered "
                "mechanism coverage. This is a Gold-free mechanism diagnosis, not an EM/F1 result."
            ),
        },
        "decision": (
            "Stop v5 before Gold attachment and answer evaluation. Do not lower the frozen gates or open a fresh "
            "confirmation set. Any successor must be a separately versioned development experiment."
        ),
        "scientific_boundary": (
            "Only frozen retrieval traces and questions were read. No answers, support labels, model outputs, "
            "EM/F1 scores, or confirmation data were accessed."
        ),
        "outputs": {"per_question_gold_free": _source(compact_path)},
    }
    report_out = args.out / "report.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        args.out,
        extra={"phase": "dependent_retrieval_v5_gold_free_failure_decomposition", **report},
        status=report["status"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if all_pass:
        raise SystemExit("unexpected pass: this script is intended to freeze the observed v5 stop")


if __name__ == "__main__":
    main()
