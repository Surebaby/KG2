#!/usr/bin/env python
"""Execute frozen AI-authored QID/PID plans, then audit against dataset evidence.

The construction phase never loads dataset source rows.  It writes and hashes
its runtime-only artifacts before the post-build audit is allowed to read Gold
evidence.  This is an engineering pilot: AI-authored plans are not human Gold.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.kg.question_kg import make_question_kg_record, validate_question_kg_record
from kgproweight.kg.wikidata_property_retriever import WikidataPropertyRetriever
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir
from scripts.pilot.audit_query_aware_kg_coverage import _chain_summary, _reference_hops


Triple = Tuple[str, str, str]
BUILDER_VERSION = "ai-plan-proof-kg-1"
_QID = re.compile(r"^Q[1-9][0-9]*$")
_PID = re.compile(r"^P[1-9][0-9]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _validate_decision(decision: Mapping[str, Any]) -> None:
    anchors = list(decision.get("anchors") or [])
    steps = list(decision.get("steps") or [])
    if not decision.get("row_id") or not anchors or not steps:
        raise ValueError("decision requires row_id, anchors, and steps")
    for anchor in anchors:
        if not _QID.fullmatch(str(anchor.get("qid") or "")):
            raise ValueError(f"invalid anchor QID in {decision['row_id']}: {anchor!r}")
    available = {f"anchor_{index}" for index in range(1, len(anchors) + 1)}
    slots: set[str] = set()
    for step in steps:
        subject = str(step.get("subject_ref") or "")
        pid = str(step.get("pid") or "")
        output = str(step.get("output_slot") or "")
        dependencies = {str(value) for value in step.get("dependencies") or []}
        subject_slot = subject[1:] if subject.startswith("$") else None
        if subject not in available and subject_slot not in slots:
            raise ValueError(f"forward/unknown subject_ref in {decision['row_id']}: {subject}")
        if not _PID.fullmatch(pid):
            raise ValueError(f"invalid PID in {decision['row_id']}: {pid}")
        if not output or output in slots:
            raise ValueError(f"invalid/duplicate output slot in {decision['row_id']}: {output}")
        if not dependencies.issubset(slots):
            raise ValueError(f"forward/unknown dependency in {decision['row_id']}: {dependencies}")
        slots.add(output)


def execute_ai_plan(
    decision: Mapping[str, Any],
    retriever: Any,
    *,
    max_values_per_hop: int = 3,
    max_total_triples: int = 12,
) -> Tuple[List[Triple], Dict[str, Any]]:
    """Execute a validated plan with exact tail-QID propagation."""
    _validate_decision(decision)
    anchors = {
        f"anchor_{index}": {
            "qid": str(anchor["qid"]),
            "label": str(anchor.get("title") or anchor.get("surface") or anchor["qid"]),
            "surface": str(anchor.get("surface") or ""),
        }
        for index, anchor in enumerate(decision["anchors"], start=1)
    }
    slots: Dict[str, List[Dict[str, str]]] = {}
    triples: List[Triple] = []
    seen: set[Triple] = set()
    hop_rows: List[Dict[str, Any]] = []

    for hop_index, step in enumerate(decision["steps"], start=1):
        subject_ref = str(step["subject_ref"])
        subjects = (
            list(slots.get(subject_ref[1:], []))
            if subject_ref.startswith("$")
            else [anchors[subject_ref]]
        )
        pid = str(step["pid"])
        matched_edges: List[Dict[str, str | None]] = []
        seen_edges: set[Tuple[str, str, str, str | None]] = set()
        for subject in subjects:
            for edge in retriever.fetch_edges(str(subject["qid"]), [pid]):
                if str(edge.get("pid") or "") != pid:
                    continue
                key = (
                    str(edge.get("head_qid") or subject["qid"]),
                    str(edge.get("relation") or pid),
                    str(edge.get("tail_value") or ""),
                    str(edge["tail_qid"]) if edge.get("tail_qid") else None,
                )
                if not key[2] or key in seen_edges:
                    continue
                seen_edges.add(key)
                matched_edges.append(dict(edge))
                if len(matched_edges) >= max_values_per_hop:
                    break
            if len(matched_edges) >= max_values_per_hop:
                break

        output_entities: List[Dict[str, str]] = []
        for edge in matched_edges:
            triple = (
                str(edge.get("head_label") or edge.get("head_qid") or ""),
                str(edge.get("relation") or pid),
                str(edge.get("tail_value") or ""),
            )
            if all(triple) and triple not in seen and len(triples) < max_total_triples:
                triples.append(triple)
                seen.add(triple)
            tail_qid = str(edge.get("tail_qid") or "")
            if tail_qid and _QID.fullmatch(tail_qid):
                entity = {"qid": tail_qid, "label": triple[2], "surface": triple[2]}
                if entity not in output_entities:
                    output_entities.append(entity)
        slots[str(step["output_slot"])] = output_entities
        hop_rows.append(
            {
                "hop_index": hop_index,
                "subject_ref": subject_ref,
                "pid": pid,
                "output_slot": str(step["output_slot"]),
                "dependencies": list(step.get("dependencies") or []),
                "input_entities": subjects,
                "matched_edges": matched_edges,
                "output_entities": output_entities,
                "has_property_value": bool(matched_edges),
            }
        )

    return triples, {
        "anchors": anchors,
        "hops": hop_rows,
        "n_triples": len(triples),
        "complete_plan_execution": bool(hop_rows)
        and all(row["has_property_value"] for row in hop_rows),
    }


def _plan_dict(decision: Mapping[str, Any]) -> Dict[str, Any]:
    consumed_slots = {
        str(step["subject_ref"])[1:]
        for step in decision["steps"]
        if str(step["subject_ref"]).startswith("$")
    }
    steps = []
    for step in decision["steps"]:
        value = dict(step)
        value["relation_role"] = (
            "bridge" if str(step["output_slot"]) in consumed_slots else "answer_operand"
        )
        steps.append(value)
    return {
        "planner_version": "ai-prereview-v1",
        "recognized": str(decision.get("should_abstain") or "").upper() != "YES",
        "operation": str(decision.get("operation") or "unknown"),
        "anchors": list(decision["anchors"]),
        "steps": steps,
        "confidence": str(decision.get("confidence") or "UNKNOWN"),
        "should_abstain": str(decision.get("should_abstain") or "UNKNOWN"),
        "notes": str(decision.get("notes") or ""),
    }


def _relation_plan_audit(decision: Mapping[str, Any], references: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    remaining = Counter(str(step["pid"]) for step in decision["steps"])
    evaluable = [row for row in references if row.get("target", {}).get("pids")]
    hits = 0
    for row in evaluable:
        match = next((pid for pid in row["target"]["pids"] if remaining[pid] > 0), None)
        if match:
            remaining[match] -= 1
            hits += 1
    return {
        "evaluable_reference_hops": len(evaluable),
        "hit_reference_hops": hits,
        "recall": hits / len(evaluable) if evaluable else None,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    evaluable = [row for row in rows if row["proof_chain_audit"]["evaluable"]]
    return {
        "counts": {
            "questions": n,
            "anchor_qid_nonempty": sum(
                bool(row["execution"]["anchors"])
                and all(anchor.get("qid") for anchor in row["execution"]["anchors"].values())
                for row in rows
            ),
            "nonempty_proof_kg": sum(bool(row["kg_subgraph"]) for row in rows),
            "complete_plan_execution": sum(
                bool(row["execution"]["complete_plan_execution"]) for row in rows
            ),
            "proof_evaluable": len(evaluable),
            "full_relation_value_chain": sum(
                row["proof_chain_audit"]["all_relation_value_hit"] is True for row in evaluable
            ),
            "runtime_errors": sum(bool(row.get("runtime_error")) for row in rows),
        },
        "rates": {
            "anchor_qid_nonempty": sum(
                bool(row["execution"]["anchors"])
                and all(anchor.get("qid") for anchor in row["execution"]["anchors"].values())
                for row in rows
            ) / max(1, n),
            "nonempty_proof_kg": sum(bool(row["kg_subgraph"]) for row in rows) / max(1, n),
            "complete_plan_execution": sum(
                bool(row["execution"]["complete_plan_execution"]) for row in rows
            ) / max(1, n),
            "full_relation_value_chain_rate_evaluable": sum(
                row["proof_chain_audit"]["all_relation_value_hit"] is True for row in evaluable
            ) / max(1, len(evaluable)),
        },
        "mean_triples": sum(len(row["kg_subgraph"]) for row in rows) / max(1, n),
    }


def _gate_results(protocol: Mapping[str, Any], aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    gates = protocol["engineering_acceptance_gates"]
    rates = aggregate["rates"]
    counts = aggregate["counts"]
    values = {
        "input_hashes_match": True,
        "runtime_gold_access": False,
        "anchor_qid_nonempty_rate": rates["anchor_qid_nonempty"],
        "nonempty_proof_kg_rate": rates["nonempty_proof_kg"],
        "complete_plan_execution_rate": rates["complete_plan_execution"],
        "full_relation_value_chain_rate_evaluable": rates[
            "full_relation_value_chain_rate_evaluable"
        ],
        "uncaught_runtime_exception_count": counts["runtime_errors"],
    }
    checks = {
        "input_hashes_match": values["input_hashes_match"] is gates["input_hashes_match"],
        "runtime_gold_access": values["runtime_gold_access"] is gates["runtime_gold_access"],
        "anchor_qid_nonempty_rate": values["anchor_qid_nonempty_rate"]
        >= gates["anchor_qid_nonempty_rate_min"],
        "nonempty_proof_kg_rate": values["nonempty_proof_kg_rate"]
        >= gates["nonempty_proof_kg_rate_min"],
        "complete_plan_execution_rate": values["complete_plan_execution_rate"]
        >= gates["complete_plan_execution_rate_min"],
        "full_relation_value_chain_rate_evaluable": values[
            "full_relation_value_chain_rate_evaluable"
        ] >= gates["full_relation_value_chain_rate_evaluable_min"],
        "uncaught_runtime_exception_count": values["uncaught_runtime_exception_count"]
        <= gates["uncaught_runtime_exception_count_max"],
    }
    return {"thresholds": gates, "values": values, "checks": checks, "all_pass": all(checks.values())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--online_properties", action="store_true")
    parser.add_argument("--request_delay", type=float, default=0.1)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort).resolve()
    decisions_path = Path(args.decisions).resolve()
    protocol_path = Path(args.protocol).resolve()
    run_path = Path(args.run_dir).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    expected = protocol["inputs"]
    actual_hashes = {
        "cohort": _sha256(cohort_path),
        "ai_plan_decisions": _sha256(decisions_path),
    }
    for name, actual in actual_hashes.items():
        if actual != expected[name]["sha256"]:
            raise SystemExit(f"frozen input hash mismatch for {name}: {actual}")
    cohort = _read_jsonl(cohort_path)
    decisions = _read_jsonl(decisions_path)
    if len(cohort) != int(protocol["n"]) or len(decisions) != len(cohort):
        raise SystemExit("cohort/decision count differs from frozen protocol")
    decision_by_id = {str(row["row_id"]): row for row in decisions}
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != {
        str(row["row_id"]) for row in cohort
    }:
        raise SystemExit("cohort and decision row IDs do not form a one-to-one join")
    for decision in decisions:
        _validate_decision(decision)

    run_dir, experiment_id = prepare_new_run_dir(
        run_path,
        experiment_id=args.experiment_id,
        extra={
            "phase": "ai_plan_zero_training_proof_kg",
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "scientific_boundary": protocol["scientific_boundary"],
        },
    )
    runtime_kg_path = run_dir / "runtime_question_kg.jsonl"
    runtime_details_path = run_dir / "runtime_details.jsonl"
    runtime_freeze_path = run_dir / "runtime_freeze.json"
    report_path = run_dir / "report.json"
    property_cache = run_dir / "wikidata_property_cache.jsonl"
    edge_cache = run_dir / "wikidata_property_edges.jsonl"

    try:
        retriever = WikidataPropertyRetriever(
            cache_path=property_cache,
            edge_cache_path=edge_cache,
            offline=not args.online_properties,
            request_delay=args.request_delay,
            max_retries=args.max_retries,
        )
        runtime_records: List[Dict[str, Any]] = []
        runtime_details: List[Dict[str, Any]] = []
        for index, frozen in enumerate(cohort, start=1):
            decision = decision_by_id[str(frozen["row_id"])]
            triples, execution = execute_ai_plan(decision, retriever)
            record = make_question_kg_record(
                dataset=str(frozen["dataset"]),
                qid=str(frozen["qid"]),
                question=str(frozen["question"]),
                triples=triples,
                query_plan=_plan_dict(decision),
                provenance={
                    "builder_version": BUILDER_VERSION,
                    "source": "AI pre-review candidate; not human Gold",
                    "property_backend": "current Wikidata wbgetentities",
                    "gold_used_for_build": False,
                },
            )
            validate_question_kg_record(record)
            runtime_records.append(record)
            runtime_details.append({**record, "row_id": frozen["row_id"], "execution": execution})
            print(f"runtime build {index}/{len(cohort)}", flush=True)

        # This write-and-hash boundary is deliberately before any source train
        # row (and therefore before any Gold evidence) is loaded.
        _write_jsonl(runtime_kg_path, runtime_records)
        _write_jsonl(runtime_details_path, runtime_details)
        runtime_freeze = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RUNTIME_ARTIFACTS_FROZEN_BEFORE_GOLD_AUDIT",
            "gold_used_for_build": False,
            "runtime_question_kg": {
                "path": str(runtime_kg_path),
                "sha256": _sha256(runtime_kg_path),
            },
            "runtime_details": {
                "path": str(runtime_details_path),
                "sha256": _sha256(runtime_details_path),
            },
        }
        runtime_freeze_path.write_text(
            json.dumps(runtime_freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        # Post-build Gold audit begins only below this point.
        selected: Dict[str, set[str]] = defaultdict(set)
        for row in cohort:
            selected[str(row["dataset"])].add(str(row["qid"]))
        source_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        source_inputs: Dict[str, Dict[str, str]] = {}
        for dataset, ids in selected.items():
            source_path = Path(args.data_root).resolve() / dataset / "train.jsonl"
            for source in _read_jsonl(source_path):
                source_id = str(source.get("id") or "")
                if source_id in ids:
                    source_rows[(dataset, source_id)] = source
            source_inputs[dataset] = {"path": str(source_path), "sha256": _sha256(source_path)}
        if len(source_rows) != len(cohort):
            raise RuntimeError("selected source rows are missing from dataset train files")

        audited: List[Dict[str, Any]] = []
        for runtime in runtime_details:
            source = source_rows[(str(runtime["dataset"]), str(runtime["qid"]))]
            decision = decision_by_id[str(runtime["row_id"])]
            references = _reference_hops(str(runtime["dataset"]), source)
            triples = [tuple(value) for value in runtime["kg_subgraph"]]
            audited.append(
                {
                    **runtime,
                    "reference_hops": references,
                    "relation_plan_audit": _relation_plan_audit(decision, references),
                    "proof_chain_audit": _chain_summary(references, triples),
                }
            )
        audited_path = run_dir / "postbuild_audit_details.jsonl"
        _write_jsonl(audited_path, audited)
        aggregate = _aggregate(audited)
        gate_results = _gate_results(protocol, aggregate)
        status = "PASS_AI_ENGINEERING_ONLY" if gate_results["all_pass"] else "FAIL_STOP"
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "status": status,
            "scope": protocol["scope"],
            "scientific_boundary": protocol["scientific_boundary"],
            "protocol": {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
                "runtime_gold_access": False,
                "gold_used_for_post_build_audit": True,
                "online_properties": bool(args.online_properties),
            },
            "inputs": {
                "cohort": {"path": str(cohort_path), "sha256": actual_hashes["cohort"]},
                "ai_plan_decisions": {
                    "path": str(decisions_path),
                    "sha256": actual_hashes["ai_plan_decisions"],
                },
                "datasets_postbuild_only": source_inputs,
            },
            "runtime_freeze": runtime_freeze,
            "overall": aggregate,
            "operation_counts": dict(Counter(row["query_plan"]["operation"] for row in audited)),
            "required_plan_pids": dict(
                Counter(step["pid"] for row in decisions for step in row["steps"])
            ),
            "gates": gate_results,
            "outputs": {
                "runtime_question_kg": {
                    "path": str(runtime_kg_path), "sha256": _sha256(runtime_kg_path)
                },
                "runtime_details": {
                    "path": str(runtime_details_path), "sha256": _sha256(runtime_details_path)
                },
                "runtime_freeze": {
                    "path": str(runtime_freeze_path), "sha256": _sha256(runtime_freeze_path)
                },
                "postbuild_audit_details": {
                    "path": str(audited_path), "sha256": _sha256(audited_path)
                },
                "property_cache": {
                    "path": str(property_cache),
                    "sha256": _sha256(property_cache) if property_cache.exists() else None,
                },
                "property_edge_cache": {
                    "path": str(edge_cache),
                    "sha256": _sha256(edge_cache) if edge_cache.exists() else None,
                },
            },
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "ai_plan_zero_training_proof_kg",
                "result_status": status,
                "scope": protocol["scope"],
                "inputs": report["inputs"],
                "outputs": report["outputs"],
                "gates": gate_results,
            },
            status=status,
        )
        print(json.dumps({"status": status, "overall": aggregate, "gates": gate_results}, indent=2))
    except Exception as exc:
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": experiment_id,
                "phase": "ai_plan_zero_training_proof_kg",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
            status="FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
