#!/usr/bin/env python3
"""Static, append-only audit for the ProofKG process-v2.2 derivation repair.

Gold answers are read only after the Gold-free scorer derives its answer, and
are used solely for retrospective agreement telemetry.  This is not a
rankability test and cannot authorize PPO training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from kgproweight.reward.proofkg_process_v2 import (
    _answer_consistency as answer_consistency_v2_1,
    build_execution_trace,
)
from kgproweight.reward.proofkg_process_v2_2 import (
    SCORER_VERSION,
    _answer_consistency_v2_2,
    build_execution_trace_v2_2,
)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, required=True)
    parser.add_argument("--question-kg-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    silver = {
        f"{row['dataset']}::{row['qid']}": row for row in _read_jsonl(args.silver)
    }
    records = [
        row for row in _read_jsonl(args.question_kg_records)
        if row.get("process_reward_eligible") is True
    ]
    if len(records) != 400:
        raise ValueError(f"expected frozen Proof400, found {len(records)} eligible records")

    outcomes = Counter()
    operator_counts = Counter()
    v21_temporal_mask = 0
    order_invariant = True
    examples = []
    for record in records:
        key = str(record["question_key"])
        gold = str((silver[key].get("metadata") or {}).get("gold_answer") or "")
        plan = record.get("query_plan") or {}
        execution = record.get("execution") or {}
        trace = build_execution_trace_v2_2(plan, execution)
        score, mask, operator, derived, status = _answer_consistency_v2_2(
            str(record["question"]), trace, gold
        )
        operator_counts[operator] += 1
        outcomes[(operator, status, int(mask), int(score))] += 1

        if operator == "temporal":
            old_score = answer_consistency_v2_1(
                str(record["question"]),
                record.get("kg_subgraph") or [],
                build_execution_trace(plan, execution),
                gold,
            )
            v21_temporal_mask += int(old_score[1])

        reversed_execution = {
            **execution,
            "hops": [
                {**hop, "matches": list(reversed(hop.get("matches") or []))}
                for hop in reversed(execution.get("hops") or [])
            ],
        }
        reversed_result = _answer_consistency_v2_2(
            str(record["question"]),
            build_execution_trace_v2_2(plan, reversed_execution),
            gold,
        )
        order_invariant = order_invariant and reversed_result == (
            score, mask, operator, derived, status
        )
        if len(examples) < 12 and (operator == "temporal" or mask == 0):
            examples.append(
                {
                    "question_key": key,
                    "operator": operator,
                    "status": status,
                    "mask": int(mask),
                    "derived_answer": derived,
                    "gold_agreement_after_derivation": int(score),
                }
            )

    compact_outcomes = [
        {
            "operator": key[0],
            "status": key[1],
            "m_A": key[2],
            "gold_agreement_after_derivation": key[3],
            "count": count,
        }
        for key, count in sorted(outcomes.items())
    ]
    temporal_determined = sum(
        row["count"] for row in compact_outcomes
        if row["operator"] == "temporal" and row["m_A"] == 1
    )
    temporal_agree = sum(
        row["count"] for row in compact_outcomes
        if row["operator"] == "temporal" and row["gold_agreement_after_derivation"] == 1
    )
    report = {
        "schema_version": "proofkg-process-v2-2-static-audit-v1",
        "experiment_id": args.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_IMPLEMENTATION_STATIC_AUDIT_RANKABILITY_PENDING",
        "scorer_version": SCORER_VERSION,
        "scope": {
            "eligible_records": len(records),
            "operator_counts": dict(operator_counts),
            "gold_use": "post-derivation retrospective agreement only",
            "gold_enters_scorer": False,
        },
        "results": {
            "v2_1_temporal_deterministic_mask_count": v21_temporal_mask,
            "v2_2_temporal_determined": temporal_determined,
            "v2_2_temporal_gold_agreement_among_determined": temporal_agree,
            "input_order_invariant_400_of_400": bool(order_invariant),
            "outcomes": compact_outcomes,
        },
        "gates": {
            "all_400_executed_without_exception": True,
            "v2_1_historical_bug_reproduced_temporal_mask_zero": v21_temporal_mask == 0,
            "v2_2_temporal_45_of_46_determined": temporal_determined == 45,
            "v2_2_temporal_determined_all_agree": temporal_agree == temporal_determined,
            "input_order_invariant": bool(order_invariant),
        },
        "scientific_boundary": (
            "Static derivation agreement is not reward rankability. Existing candidate "
            "pools must be re-scored append-only, followed by family-disjoint confirmation."
        ),
        "training_started": False,
        "examples": examples,
    }
    if not all(report["gates"].values()):
        report["status"] = "FAIL_STATIC_AUDIT"
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root = Path.cwd().resolve()
    manifest = {
        "schema_version": "proofkg-process-v2-2-static-audit-manifest-v1",
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "inputs": {
            "silver": _identity(args.silver.resolve()),
            "question_kg_records": _identity(args.question_kg_records.resolve()),
        },
        "code": {
            "v2_1_frozen": _identity(root / "kgproweight/reward/proofkg_process_v2.py"),
            "v2_2": _identity(root / "kgproweight/reward/proofkg_process_v2_2.py"),
            "audit": _identity(root / "scripts/diagnose/audit_proofkg_process_v2_2_static.py"),
        },
        "outputs": {"report": _identity(report_path)},
        "training_started": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
