#!/usr/bin/env python
"""Score paired Phase-1 control/additive-v3 Teacher generations.

This report deliberately leaves the scientific verdict to the researcher.  It
checks protocol integrity and reports paired deltas on the immutable candidate
sidecars as well as quota-selected outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.pilot.score_silver_regeneration import _md5, _metric_bundle, _metrics, _read


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
ARMS = ("control", "additive_v3")


def _qid(row: Dict[str, Any]) -> str:
    return str(row.get("qid") or row.get("id") or "")


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _candidate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = [row.get("metadata") or {} for row in rows]
    quality_passed = sum(bool(md.get("quality_pass")) for md in metadata)
    return {
        "trajectory": _metrics(rows),
        "quality_passed": quality_passed,
        "quality_pass_rate_pct": 100.0 * quality_passed / len(rows) if rows else None,
        "quality_reject_reasons": dict(
            sorted(
                Counter(
                    str(md.get("quality_reject_reason") or "UNKNOWN")
                    for md in metadata
                    if not md.get("quality_pass")
                ).items()
            )
        ),
        "kg_bucket_counts": dict(
            sorted(Counter(str(md.get("kg_bucket") or "UNKNOWN") for md in metadata).items())
        ),
        "format_retry_count": sum(bool(md.get("format_retried")) for md in metadata),
        "retry_success_count": sum(bool(md.get("retry_succeeded")) for md in metadata),
        "citation_contract_error_count": sum(
            len(md.get("citation_contract_errors") or []) for md in metadata
        ),
    }


def _paired(candidate_control: List[Dict[str, Any]], candidate_v3: List[Dict[str, Any]]) -> Dict[str, Any]:
    left = {_qid(row): row for row in candidate_control}
    right = {_qid(row): row for row in candidate_v3}
    common = sorted(set(left) & set(right))
    quality_up = quality_down = 0
    answer_deltas: List[float] = []
    label_fraction_deltas: List[float] = []
    for qid in common:
        c_row, v_row = left[qid], right[qid]
        c_md, v_md = c_row.get("metadata") or {}, v_row.get("metadata") or {}
        c_quality, v_quality = bool(c_md.get("quality_pass")), bool(v_md.get("quality_pass"))
        quality_up += int(v_quality and not c_quality)
        quality_down += int(c_quality and not v_quality)
        answer_deltas.append(float(v_md.get("answer_score", 0.0)) - float(c_md.get("answer_score", 0.0)))

        def fractional_rate(row: Dict[str, Any]) -> float:
            labels = [float(step.get("label", 0.0)) for step in row.get("steps") or []]
            return sum(value not in {-1.0, 0.0, 1.0} for value in labels) / max(1, len(labels))

        label_fraction_deltas.append(fractional_rate(v_row) - fractional_rate(c_row))
    return {
        "n_pairs": len(common),
        "control_only_qids": sorted(set(left) - set(right)),
        "additive_v3_only_qids": sorted(set(right) - set(left)),
        "quality_pass_up": quality_up,
        "quality_pass_down": quality_down,
        "quality_pass_net": quality_up - quality_down,
        "answer_match_delta_mean": _mean(answer_deltas),
        "answer_match_delta_counts": {
            "up": sum(value > 0 for value in answer_deltas),
            "down": sum(value < 0 for value in answer_deltas),
            "tie": sum(value == 0 for value in answer_deltas),
        },
        "per_trajectory_fractional_label_rate_delta_mean": _mean(label_fraction_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Experiment directory containing arm subdirectories")
    parser.add_argument("--expected_per_dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    integrity_errors: List[str] = []
    files: List[Dict[str, Any]] = []
    arm_rows: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for arm in ARMS:
        arm_rows[arm] = {}
        expected_mode = "off" if arm == "control" else "additive_v3"
        for dataset in DATASETS:
            selected_path = root / arm / f"{dataset}.jsonl"
            candidate_path = root / arm / f"{dataset}.candidates.jsonl"
            try:
                selected = _read(selected_path)
                candidates = _read(candidate_path)
            except (FileNotFoundError, ValueError) as exc:
                integrity_errors.append(str(exc))
                selected, candidates = [], []
            arm_rows[arm][dataset] = {"selected": selected, "candidates": candidates}
            for kind, path, rows in (
                ("selected", selected_path, selected),
                ("candidates", candidate_path, candidates),
            ):
                if path.exists():
                    files.append(
                        {"arm": arm, "dataset": dataset, "kind": kind, "path": str(path),
                         "records": len(rows), "md5": _md5(path)}
                    )
                if len(rows) != args.expected_per_dataset:
                    integrity_errors.append(
                        f"{path}: {len(rows)} records, expected {args.expected_per_dataset}"
                    )
            if [_qid(row) for row in selected] != [_qid(row) for row in candidates]:
                integrity_errors.append(f"{dataset}/{arm}: selected and candidate qid order differs")
            for lineno, row in enumerate(candidates, 1):
                md = row.get("metadata") or {}
                extra = md.get("extra") or {}
                if extra.get("source_split") != "train":
                    integrity_errors.append(f"{candidate_path}:{lineno}: source_split is not train")
                if extra.get("sample_seed") != args.seed:
                    integrity_errors.append(f"{candidate_path}:{lineno}: sample_seed != {args.seed}")
                if extra.get("retrieval_bridge_mode") != expected_mode:
                    integrity_errors.append(
                        f"{candidate_path}:{lineno}: bridge mode is not {expected_mode}"
                    )
                if len(row.get("kg_subgraph") or []) > 12:
                    integrity_errors.append(f"{candidate_path}:{lineno}: stored KG exceeds 12")
                if md.get("n_triples_teacher") != len(row.get("kg_subgraph") or []):
                    integrity_errors.append(f"{candidate_path}:{lineno}: Teacher/stored KG count differs")

    datasets: Dict[str, Any] = {}
    for dataset in DATASETS:
        control = arm_rows["control"][dataset]
        v3 = arm_rows["additive_v3"][dataset]
        if [_qid(row) for row in control["candidates"]] != [_qid(row) for row in v3["candidates"]]:
            integrity_errors.append(f"{dataset}: paired arm qid order differs")
        datasets[dataset] = {
            "control": {
                "candidates": _candidate_metrics(control["candidates"]),
                "posthoc_selected_output": _metric_bundle(control["selected"]),
            },
            "additive_v3": {
                "candidates": _candidate_metrics(v3["candidates"]),
                "posthoc_selected_output": _metric_bundle(v3["selected"]),
            },
            "paired_additive_v3_minus_control": _paired(
                control["candidates"], v3["candidates"]
            ),
        }

    aggregate_candidates = {
        arm: [
            row
            for dataset in DATASETS
            for row in arm_rows[arm][dataset]["candidates"]
        ]
        for arm in ARMS
    }
    aggregate_selected = {
        arm: [
            row
            for dataset in DATASETS
            for row in arm_rows[arm][dataset]["selected"]
        ]
        for arm in ARMS
    }

    report = {
        "report_schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root),
        "protocol": {
            "design": "paired control vs additive_v3",
            "datasets": list(DATASETS),
            "split": "train",
            "seed": args.seed,
            "questions_per_dataset": args.expected_per_dataset,
            "primary_teacher_generations_expected": args.expected_per_dataset * len(DATASETS) * len(ARMS),
            "teacher_temperature": 0.0,
            "rrf_candidate_topk": 100,
            "rerank_topk": 10,
            "additive_v3": {"first_round_topk": 5, "max_bridge_queries": 2, "bridge_only_k": 50},
            "teacher_kg": {"max_triples": 12, "min_keep": 5},
        },
        "integrity_pass": not integrity_errors,
        "integrity_errors": integrity_errors,
        "source_files": files,
        "metric_semantics": {
            "candidate_quality": "intrinsic quality before post-hoc KG-bucket quotas",
            "posthoc_selected_output": "same candidate rows with accepted/selection fields finalized",
            "citation_metric": "exact parsed citation membership in the filtered KG shown to Teacher",
            "fractional_label": "step label not exactly one of {-1, 0, 1}",
        },
        "datasets": datasets,
        "aggregate": {
            "control": {
                "candidates": _candidate_metrics(aggregate_candidates["control"]),
                "posthoc_selected_output": _metric_bundle(aggregate_selected["control"]),
            },
            "additive_v3": {
                "candidates": _candidate_metrics(aggregate_candidates["additive_v3"]),
                "posthoc_selected_output": _metric_bundle(aggregate_selected["additive_v3"]),
            },
            "paired_additive_v3_minus_control": _paired(
                aggregate_candidates["control"], aggregate_candidates["additive_v3"]
            ),
        },
        "scientific_verdict": "RESEARCHER_DECISION_REQUIRED",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if integrity_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
