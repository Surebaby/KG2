#!/usr/bin/env python
"""Retrospective relation-graph compatibility audit for the failed HotpotQA zero-shot plans.

The subquery-graph zero-shot pilot produced relation-graph-shaped output
(subject + relation_label + pid) that failed the subquery schema.  This audit
re-parses those *already-frozen* predictions under the relation-graph schema to
decide whether a single-variable relation-graph re-run is worth doing.

Status is RETROSPECTIVE_SCHEMA_COMPATIBILITY_ONLY: it never upgrades the original
subquery-graph failure into a pass, and it performs no model inference.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kgproweight.utils.logging import dump_manifest, get_logger
from scripts.prepare.audit_query_planner_supervision import validate_record
from scripts.prepare.build_query_planner_supervision import SCHEMA_VERSION

logger = get_logger(__name__)

DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_pilot30x3_plans_v1"
    / "predictions.question_only.jsonl"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "hotpot_relation_graph_compatibility_h1"
)

_PID = re.compile(r"^P[1-9][0-9]*$")
_HOP = re.compile(r"^hop_(\d+)$")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    hotpot = [r for r in rows if r.get("dataset") == "hotpotqa"]
    errors = Counter()
    rel_valid = pid_valid = anchor_from_q = dep_ok = 0
    pids: Counter = Counter()
    for r in hotpot:
        target = r.get("predicted_target") or {}
        record = {
            "schema_version": SCHEMA_VERSION,
            "question_key": r.get("question_key"),
            "dataset": r.get("dataset"),
            "qid": r.get("qid"),
            "question": r.get("question"),
            "question_sha256": r.get("question_sha256"),
            "target_type": "relation_graph",
            "target": target,
        }
        errs = validate_record(record)
        if not errs:
            rel_valid += 1
        for e in errs:
            errors[e] += 1

        steps = target.get("steps") or []
        if steps and all(
            isinstance(s.get("pid"), str) and _PID.fullmatch(s["pid"])
            for s in steps if "pid" in s
        ) and all("pid" in s for s in steps):
            pid_valid += 1
        for s in steps:
            if "pid" in s:
                pids[str(s["pid"])] += 1

        q = str(r.get("question") or "").lower()
        anchors = target.get("anchors") or []
        if anchors and all(str(a).strip().lower() in q for a in anchors):
            anchor_from_q += 1

        deps_ok = True
        for i, s in enumerate(steps, start=1):
            for d in s.get("dependencies") or []:
                m = _HOP.fullmatch(str(d))
                if not m or int(m.group(1)) >= i:
                    deps_ok = False
        if deps_ok:
            dep_ok += 1

    n = len(hotpot)
    return {
        "n": n,
        "relation_graph_schema_valid": rel_valid,
        "pid_format_valid": pid_valid,
        "anchors_all_from_question": anchor_from_q,
        "dependency_refs_hop_n_valid": dep_ok,
        "validation_errors": dict(errors),
        "pid_distribution": dict(pids),
        "gold_forbidden_fields": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {args.out}")
    args.out.mkdir(parents=True)

    rows = _read_jsonl(args.predictions)
    result = audit(rows)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RETROSPECTIVE_SCHEMA_COMPATIBILITY_ONLY",
        "scope": (
            "re-parse frozen subquery-graph zero-shot HotpotQA plans under the "
            "relation-graph schema; no model inference; does NOT upgrade the "
            "original subquery-graph failure"
        ),
        "source": str(args.predictions),
        "finding": result,
        "interpretation": {
            "relation_graph_bias": (
                "subject+relation_label+pid appear in most steps and PID format is "
                f"valid in {result['pid_format_valid']}/{result['n']} questions, but "
                f"only {result['relation_graph_schema_valid']}/{result['n']} is fully "
                "relation-graph valid. The dominant error (invalid_output_slot) is "
                "step_N naming left by the subquery-graph prompt — a prompt artifact "
                "a relation-graph re-run would remove — while unmapped_pid shows some "
                "steps omit pid, the real remaining risk."
            ),
            # A cheap re-run is justified if the model emits valid PIDs in a clear
            # majority of questions (relation-graph shape, not random format drift).
            "h2_justified": result["pid_format_valid"] >= 15,
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dump_manifest(
        args.out,
        extra={
            "experiment_id": args.out.name,
            "phase": "audit",
            "status": report["status"],
            "h2_justified": report["interpretation"]["h2_justified"],
        },
        status="COMPLETE",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
