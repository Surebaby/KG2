#!/usr/bin/env python
"""Integrity-check and score a fixed-input Teacher swap pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List

from scripts.pilot.score_paired_bridge_teacher_pilot import _candidate_metrics, _paired
from scripts.pilot.score_silver_regeneration import _md5, _read
from kgproweight.training.phase1_distill import _quota_selection_counts


def _qid(row: Dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("id") or "")


def _api_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls = [call for row in rows for call in ((row.get("metadata") or {}).get("api_calls") or [])]
    latencies = [float(call["latency_seconds"]) for call in calls if call.get("latency_seconds") is not None]
    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cache_hit_tokens")
    return {
        "n_api_calls": len(calls),
        "primary_calls": sum(call.get("purpose") == "primary" for call in calls),
        "format_retry_calls": sum(call.get("purpose") == "format_retry" for call in calls),
        "tokens": {
            key: sum(int(call.get(key) or 0) for call in calls) for key in token_keys
        },
        "latency_seconds_mean": statistics.mean(latencies) if latencies else None,
        "latency_seconds_median": statistics.median(latencies) if latencies else None,
        "finish_reasons": {
            reason: sum(call.get("finish_reason") == reason for call in calls)
            for reason in sorted({str(call.get("finish_reason")) for call in calls})
        },
        "response_models": sorted({str(call.get("response_model")) for call in calls}),
    }


def _quota_projection(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def one(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = {"kg_rich": 0, "kg_medium": 0, "kg_sparse": 0}
        for row in subset:
            md = row.get("metadata") or {}
            if md.get("quality_pass"):
                bucket = str(md.get("kg_bucket") or "")
                if bucket in counts:
                    counts[bucket] += 1
        selected = _quota_selection_counts(
            counts["kg_rich"], counts["kg_medium"], counts["kg_sparse"], 0.35, 0.25
        )
        return {"quality_bucket_counts": counts, "selected_bucket_counts": selected,
                "selected_total": sum(selected.values())}

    per_dataset = {
        dataset: one([row for row in rows if row.get("dataset") == dataset])
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
    }
    return {
        "global_pool": one(rows),
        "per_dataset": per_dataset,
        "per_dataset_selected_total": sum(
            value["selected_total"] for value in per_dataset.values()
        ),
        "note": "count projection only; production currently applies quotas per dataset run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--control_teacher", default="deepseek-chat")
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--pilot_teacher", default=None)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--expected", type=int, default=90)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    control_paths = [Path(path).resolve() for path in args.control]
    pilot_path = Path(args.pilot).resolve()
    failure_path = Path(args.failures).resolve()
    control = [row for path in control_paths for row in _read(path)]
    pilot = _read(pilot_path)
    failures = _read(failure_path)
    pilot_models = sorted({str(row.get("teacher_model") or "") for row in pilot})
    pilot_teacher = args.pilot_teacher or (pilot_models[0] if len(pilot_models) == 1 else "UNKNOWN")
    errors: List[str] = []
    if len(control) != args.expected:
        errors.append(f"control count {len(control)} != {args.expected}")
    if len(pilot) + len(failures) != args.expected:
        errors.append(f"pilot accounting {len(pilot)}+{len(failures)} != {args.expected}")
    if failures:
        errors.append(f"pilot has {len(failures)} failed qids")
    if [_qid(row) for row in control] != [_qid(row) for row in pilot]:
        errors.append("control/pilot qid order differs")

    control_by_qid = {_qid(row): row for row in control}
    for lineno, row in enumerate(pilot, 1):
        source = control_by_qid.get(_qid(row))
        if source is None:
            continue
        if row.get("kg_subgraph") != source.get("kg_subgraph"):
            errors.append(f"pilot:{lineno}: KG differs from fixed control")
        if row.get("retrieved_passages") != source.get("retrieved_passages"):
            errors.append(f"pilot:{lineno}: passages differ from fixed control")
        md = row.get("metadata") or {}
        if md.get("teacher_thinking") != "disabled":
            errors.append(f"pilot:{lineno}: thinking is not disabled")
        if md.get("teacher_temperature") != 0.0:
            errors.append(f"pilot:{lineno}: temperature is not zero")

    by_dataset: Dict[str, Any] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        c_rows = [row for row in control if row.get("dataset") == dataset]
        p_rows = [row for row in pilot if row.get("dataset") == dataset]
        by_dataset[dataset] = {
            "control": _candidate_metrics(c_rows),
            "pilot": _candidate_metrics(p_rows),
            "paired_pilot_minus_control": _paired(c_rows, p_rows),
        }

    report = {
        "report_schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_pass": not errors,
        "integrity_errors": errors,
        "protocol": {
            "design": "fixed passages and filtered KG; Teacher-only swap",
            "control_teacher": args.control_teacher,
            "pilot_teacher": pilot_teacher,
            "thinking": "disabled",
            "temperature": 0.0,
            "expected_qids": args.expected,
        },
        "source_files": [
            {"path": str(path), "records": len(_read(path)), "md5": _md5(path)}
            for path in [*control_paths, pilot_path, failure_path]
        ],
        "aggregate": {
            "control": _candidate_metrics(control),
            "pilot": _candidate_metrics(pilot),
            "paired_pilot_minus_control": _paired(control, pilot),
        },
        "datasets": by_dataset,
        "api": _api_metrics(pilot),
        "quota_projection": {
            "control": _quota_projection(control),
            "pilot": _quota_projection(pilot),
        },
        "failures": failures,
        "scientific_verdict": "RESEARCHER_DECISION_REQUIRED",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
