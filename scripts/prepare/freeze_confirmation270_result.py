#!/usr/bin/env python
"""Freeze the 2Wiki confirmation270 result, final KG data, and semantic-audit sample.

Three append-only artifacts (never overwriting existing runs):
1. result record   — PASS_STRUCTURAL_CONFIRMATION + two honest metadata addenda;
2. final KG        — per-question records with triples + per-triple source (store
                     / historical_fallback), complete_plan_execution, versions,
                     closure-cache hash, gold_access=false; identity join = 1.0
                     INCLUDING empty-KG questions (all 270, not just 228);
3. semantic audit  — a hash-selected sample of 50 of the 200 complete questions,
                     frozen BEFORE any model generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIRM_DIR = ROOT / "data/derived/inference_proofkg_v1_pilot30/2wikimultihopqa/confirmation270_v3"
ROUND2 = CONFIRM_DIR / "round_2" / "runtime"
DEFAULT_KG_OUT = ROOT / "data/derived/inference_proofkg_v1_confirmation270_2wiki_v3"
DEFAULT_RESULT_OUT = ROOT / "outputs/audits/2wiki_confirmation270_v3"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple_sources(details_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[Tuple, str]]:
    """Map each question_key -> {triple: source} from the hop matches."""
    out: Dict[str, Dict[Tuple, str]] = {}
    for key, row in details_rows.items():
        sources: Dict[Tuple, str] = {}
        for hop in (row.get("execution") or {}).get("hops") or []:
            matches = hop.get("matches") or []
            match_sources = hop.get("match_sources") or []
            for triple, src in zip(matches, match_sources):
                t = tuple(str(x) for x in triple)
                if t not in sources:
                    sources[t] = str(src)
        out[key] = sources
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg_out", type=Path, default=DEFAULT_KG_OUT)
    parser.add_argument("--result_out", type=Path, default=DEFAULT_RESULT_OUT)
    args = parser.parse_args()

    kg_rows = _read_jsonl(ROUND2 / "runtime_question_kg.jsonl")
    details_rows = {str(r["question_key"]): r for r in _read_jsonl(ROUND2 / "runtime_details.jsonl")}
    assert len(kg_rows) == 270 and len(details_rows) == 270

    sources_by_key = _triple_sources(details_rows)
    closure_report = json.loads((CONFIRM_DIR / "closure_report.json").read_text(encoding="utf-8"))
    closure_cache_sha = closure_report["closure_cache_sha256"]

    # ---- Step 2: final KG (all 270, incl. empty-KG; identity join = 1.0) ----
    if args.kg_out.exists():
        raise SystemExit(f"refusing to overwrite: {args.kg_out}")
    args.kg_out.mkdir(parents=True)

    frozen_records: List[Dict[str, Any]] = []
    for row in kg_rows:
        key = str(row["question_key"])
        triples = row.get("kg_subgraph") or []
        sources = sources_by_key.get(key, {})
        frozen_records.append({
            "schema_version": "inference-proofkg-confirmation270-1",
            "question_key": key,
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question": row["question"],
            "question_sha256": row["question_sha256"],
            "kg_subgraph": triples,
            "triple_sources": [sources.get(tuple(str(x) for x in t), "") for t in triples],
            "complete_plan_execution": bool((row.get("provenance") or {}).get("complete_plan_execution")),
            "planner_version": (row.get("query_plan") or {}).get("planner_version"),
            "builder_version": (row.get("provenance") or {}).get("builder_version"),
            "closure_cache_sha256": closure_cache_sha,
            "gold_access": (row.get("provenance") or {}).get("gold_access", False),
        })

    _write_jsonl(args.kg_out / "question_kg_records.jsonl", frozen_records)
    n_complete = sum(1 for r in frozen_records if r["complete_plan_execution"])
    n_nonempty = sum(1 for r in frozen_records if r["kg_subgraph"])
    kg_report = {
        "schema_version": "inference-proofkg-confirmation270-report-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(frozen_records),
        "identity_join": 1.0,
        "nonempty": n_nonempty,
        "complete_plan_execution": n_complete,
        "empty_kg_included": True,
        "closure_cache_sha256": closure_cache_sha,
        "records_sha256": _sha256(args.kg_out / "question_kg_records.jsonl"),
        "gold_access": False,
    }
    (args.kg_out / "report.json").write_text(json.dumps(kg_report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(args.kg_out, extra={"experiment_id": args.kg_out.name, "phase": "freeze_confirmation270_kg", "n": len(frozen_records)}, status="COMPLETE")
    logger.info("froze %d KG records (%d nonempty, %d complete)", len(frozen_records), n_nonempty, n_complete)

    # ---- Step 1: result record + two metadata addenda ----
    result = {
        "schema_version": "confirmation270-result-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_STRUCTURAL_CONFIRMATION",
        "semantic_correctness": "NOT_YET_EVALUATED",
        "sft_utility": "NOT_YET_EVALUATED",
        "final_counts": {
            "plan_recognized": "270/270",
            "nonempty": f"{n_nonempty}/270",
            "complete_plan_execution": f"{n_complete}/270",
            "runtime_errors": 0,
            "gold_access": False,
        },
        "closure": {
            "last_materialized_round": closure_report["last_materialized_round"],
            "convergence_check_round": closure_report["convergence_check_round"],
            "requested_total_2wiki": closure_report["requested_total_2wiki"],
        },
        "final_kg_dir": str(args.kg_out),
    }
    (args.result_out / "result_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # planner scope addendum (report says n100, actually n=270)
    plans_report = json.loads((args.result_out / "plans" / "report.json").read_text(encoding="utf-8"))
    scope_addendum = {
        "schema_version": "planner-report-metadata-addendum-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "METADATA_CORRECTION_ONLY",
        "corrects": {"original_scope": plans_report.get("scope"), "corrected_scope": "2wiki confirmation270 n=270 (greedy, zero-shot learned planner v1.1)"},
        "unchanged": {"counts": plans_report.get("counts"), "rates": plans_report.get("rates")},
    }
    (args.result_out / "plans" / "metadata_addendum.json").write_text(json.dumps(scope_addendum, ensure_ascii=False, indent=2), encoding="utf-8")

    # retrospective derivation addendum (270 SHA256 not pre-bound)
    inputs_270 = _sha256(args.result_out / "planner_inputs.confirmation.jsonl")
    inputs_810 = _sha256(ROOT / "outputs/audits/inference_proofkg_v1_pilot30x3_execution_v1/planner_inputs.confirmation.jsonl")
    derivation_addendum = {
        "schema_version": "confirmation270-derivation-addendum-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RETROSPECTIVE_DERIVATION",
        "fact": "the 270 2wiki inputs are a deterministic per-row subset of the pre-frozen 810 confirmation inputs (pilot overlap = 0)",
        "inputs_270_sha256": inputs_270,
        "inputs_810_sha256": inputs_810,
        "honest_note": "the pre-run protocol froze the 810 inputs but did NOT bind the independent 270-subset SHA256; this addendum records it retrospectively, not as a pre-bound quantity.",
    }
    (args.result_out / "derivation_addendum.json").write_text(json.dumps(derivation_addendum, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Step 3: semantic audit sample (50 of the 200 complete, hash-selected) ----
    audit_out = ROOT / "outputs/audits/2wiki_confirmation270_semantic_audit"
    audit_out.mkdir(parents=True, exist_ok=True)
    complete = [r for r in frozen_records if r["complete_plan_execution"] and r["kg_subgraph"]]
    assert len(complete) == 200, f"expected 200 complete, got {len(complete)}"
    complete_sorted = sorted(complete, key=lambda r: r["question_sha256"])
    # deterministic hash-selection of 50: sort by sha256, take every 4th (200/50).
    sample = complete_sorted[::4][:50]
    assert len(sample) == 50
    audit_rows = [
        {
            "question_key": r["question_key"],
            "qid": r["qid"],
            "question": r["question"],
            "question_sha256": r["question_sha256"],
            "kg_subgraph": r["kg_subgraph"],
            "triple_sources": r["triple_sources"],
            "complete_plan_execution": True,
            "gold_access": False,
        }
        for r in sample
    ]
    _write_jsonl(audit_out / "audit_sample.jsonl", audit_rows)
    audit_report = {
        "schema_version": "semantic-audit-sample-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_SAMPLE_FROZEN_BEFORE_GENERATION",
        "n_sample": len(audit_rows),
        "n_complete_pool": len(complete),
        "selection": "hash-selected (sort by question_sha256, every 4th of 200 complete)",
        "audit_sample_sha256": _sha256(audit_out / "audit_sample.jsonl"),
        "audit_checks": [
            "anchor linked to correct entity in question",
            "pid matches question intent",
            "each hop head/tail continuous",
            "triples support the required operation",
            "entity name collision / direction error / irrelevant-but-complete chain",
        ],
        "note": "Gold may be used to EVALUATE this sample after freezing, but must NOT modify the ProofKG.",
    }
    (audit_out / "audit_report.json").write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("froze %d-question semantic-audit sample", len(audit_rows))
    print(json.dumps({"result": result["status"], "kg": kg_report, "audit_sample": len(audit_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
