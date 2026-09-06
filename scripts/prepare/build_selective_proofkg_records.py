#!/usr/bin/env python
"""Build versioned selective-ProofKG records from an execution-level runtime trace.

Offline semantic validation: for each question, validate every Proof edge against
its execution.hops (head QID, PID, tail, plan-step mapping), emit only trusted
edges, and record partial/complete eligibility plus every rejected edge.

Never reads gold / answer / supporting facts / model correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

VALIDATOR_VERSION = "selective-proofkg-validator-1"
RECORD_SCHEMA = "selective-proofkg-record-v1"
_QID = __import__("re").compile(r"^Q[1-9][0-9]*$")
_PID = __import__("re").compile(r"^P[1-9][0-9]*$")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _tail_qid(tail: object) -> str | None:
    t = str(tail).strip()
    return t if _QID.match(t) else None


def validate_edges(plan: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
    """Per-edge execution-level validation.  Returns trusted_edges + rejected."""
    plan_hops = list(plan.get("hops") or [])
    exec_hops = {str(h.get("hop_index") or ""): h for h in execution.get("hops") or []}
    trusted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen = set()

    for step_idx, ph in enumerate(plan_hops, start=1):
        slot = str(ph.get("output_slot") or "")
        hop_no = __import__("re").search(r"(\d+)$", slot)
        hop_no = hop_no.group(1) if hop_no else str(step_idx)
        eh = exec_hops.get(hop_no)
        if eh is None:
            continue
        pids = list(ph.get("pids") or [])
        pid = pids[0] if pids else None
        head_qids = [str(e.get("qid") or "") for e in eh.get("input_entities") or [] if e.get("qid")]
        head_qid = head_qids[0] if head_qids else None
        sources = eh.get("match_sources") or []
        for mi, m in enumerate(eh.get("matches") or []):
            if len(m) != 3:
                rejected.append({"reason": "bad_triple_arity", "step": step_idx, "triple": m})
                continue
            head, relation, tail = str(m[0]).strip(), str(m[1]).strip(), str(m[2]).strip()
            source = sources[mi] if mi < len(sources) else ""
            problems = []
            if not head_qid or not _QID.match(head_qid):
                problems.append("missing_head_qid")
            if not pid or not _PID.match(pid):
                problems.append("missing_pid")
            if not tail:
                problems.append("missing_tail")
            key = (head_qid or head, pid or "", tail)
            dup = key in seen
            edge = {
                "head": head, "head_qid": head_qid, "relation": relation, "pid": pid,
                "tail": tail, "tail_qid": _tail_qid(tail),
                "plan_step_index": step_idx, "provenance": source,
            }
            if problems or dup:
                rejected.append({"reason": "invalid_edge", "step": step_idx, "edge": edge, "problems": problems, "duplicate": dup})
                continue
            seen.add(key)
            trusted.append(edge)
    return {"trusted_edges": trusted, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question_kg", required=True)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--historical_cutoff", default="2020-12-09T23:59:59Z")
    args = parser.parse_args()

    kg_rows = _read_jsonl(Path(args.question_kg))
    detail_rows = {str(r["qid"]): r for r in _read_jsonl(Path(args.runtime_details))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    records: List[Dict[str, Any]] = []
    gate_details: List[Dict[str, Any]] = []
    for kg in kg_rows:
        qid = str(kg["qid"])
        detail = detail_rows.get(qid, {})
        plan = kg.get("query_plan") or {}
        execution = detail.get("execution") or {}
        runtime_error = detail.get("runtime_error")
        gold_access = (kg.get("provenance") or {}).get("gold_access", False)
        complete = bool((kg.get("provenance") or {}).get("complete_plan_execution"))

        ed = validate_edges(plan, execution)
        trusted = ed["trusted_edges"]
        rejected = ed["rejected"]

        # trusted gate (record-level, mechanical)
        reasons: List[str] = []
        if (kg.get("schema_version") or "") != "question-kg-by-dataset-qid-1":
            reasons.append("bad_schema_version")
        if str(kg.get("question_key") or "") != f"{args.dataset}::{qid}":
            reasons.append("question_key_mismatch")
        if gold_access is not False:
            reasons.append("gold_access_not_false")
        if runtime_error:
            reasons.append(f"runtime_error:{runtime_error}")
        if not trusted:
            reasons.append("no_trusted_edge")

        partial_eligible = bool(trusted) and not reasons
        complete_eligible = partial_eligible and complete

        records.append({
            "schema_version": RECORD_SCHEMA,
            "dataset": args.dataset, "qid": qid,
            "question_sha256": kg.get("question_sha256"),
            "query_plan_sha256": _sha(plan),
            "runtime_details_sha256": _sha(execution),
            "validator_version": VALIDATOR_VERSION,
            "historical_cutoff": args.historical_cutoff,
            "partial_eligible": partial_eligible,
            "complete_eligible": complete_eligible,
            "routing_reasons": reasons,
            "trusted_edges": trusted,
        })
        gate_details.append({
            "qid": qid, "partial_eligible": partial_eligible, "complete_eligible": complete_eligible,
            "routing_reasons": reasons, "trusted_edges": trusted, "rejected_edges": rejected,
            "complete_plan_execution": complete, "runtime_error": runtime_error, "gold_access": gold_access,
        })

    def wr(p, rows):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    wr(out / "selective_question_kg_records.jsonl", records)
    wr(out / "gate_details.jsonl", gate_details)

    report = {
        "schema_version": "selective-proofkg-report-v1",
        "validator_version": VALIDATOR_VERSION,
        "dataset": args.dataset,
        "n": len(records),
        "partial_eligible": sum(1 for r in records if r["partial_eligible"]),
        "complete_eligible": sum(1 for r in records if r["complete_eligible"]),
        "nonempty": sum(1 for r in records if r["trusted_edges"]),
        "total_trusted_edges": sum(len(r["trusted_edges"]) for r in records),
        "total_rejected_edges": sum(len(g["rejected_edges"]) for g in gate_details),
        "historical_cutoff": args.historical_cutoff,
        "gold_access": False,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(out, extra={"experiment_id": out.name, "phase": "build_selective_proofkg_records", "dataset": args.dataset, **report}, status="COMPLETE")
    logger.info("selective records: %d partial / %d complete / %d edges", report["partial_eligible"], report["complete_eligible"], report["total_trusted_edges"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
