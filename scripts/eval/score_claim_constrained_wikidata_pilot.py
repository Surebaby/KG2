#!/usr/bin/env python
"""Freeze the two-dataset result of the claim-constrained Wikidata pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from scripts.eval.evaluate_a1_fixed_context_kg import _aggregate
from scripts.pilot.score_a1_fixed_context_kg import paired_metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build_dir", type=Path, required=True)
    parser.add_argument("--hotpot_predictions", type=Path, required=True)
    parser.add_argument("--musique_predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")

    build_report = json.loads((args.build_dir / "report.json").read_text(encoding="utf-8"))
    detail_rows = _read_jsonl(args.build_dir / "runtime_details.jsonl")
    injected = {
        dataset: {str(row["qid"]) for row in detail_rows if row["dataset"] == dataset and row["kg_subgraph"]}
        for dataset in ("hotpotqa", "musique")
    }
    prediction_paths = {
        "hotpotqa": args.hotpot_predictions,
        "musique": args.musique_predictions,
    }
    results: dict[str, Any] = {}
    all_pass = True
    for dataset, path in prediction_paths.items():
        rows = _read_jsonl(path)
        if len(rows) != 60 or {row["arm"] for row in rows} != {"legacy", "proof"}:
            raise ValueError(f"incomplete paired predictions: {path}")
        by_arm = {
            arm: _aggregate([row for row in rows if row["arm"] == arm])
            for arm in ("legacy", "proof")
        }
        paired = paired_metrics(rows, "sft")
        per_qid = {}
        for row in rows:
            per_qid.setdefault(str(row["qid"]), {})[str(row["arm"])] = row
        fallback_qids = sorted(set(per_qid) - injected[dataset])
        fallback_identical = sum(
            per_qid[qid]["legacy"]["prediction"] == per_qid[qid]["proof"]["prediction"]
            for qid in fallback_qids
        ) / max(1, len(fallback_qids))
        utility_gates = {
            "net_correct_at_least_1": paired["net_correct"] >= 1,
            "lost_correct_at_most_1": paired["lost_correct"] <= 1,
            "f1_delta_nonnegative": paired["delta_f1"] >= 0,
            "fallback_prediction_identity": fallback_identical == 1.0,
        }
        passed = all(utility_gates.values())
        all_pass = all_pass and passed
        structure = build_report["by_dataset"][dataset]
        results[dataset] = {
            "structural": {
                **structure,
                "nonempty_rate": structure["nonempty"] / structure["n"],
                "complete_rate": structure["complete"] / structure["n"],
                "old_full_replacement_nonempty_gate_pass": structure["nonempty"] / structure["n"] >= 0.8,
                "old_full_replacement_complete_gate_pass": structure["complete"] / structure["n"] >= 0.7,
            },
            "by_arm": by_arm,
            "paired": paired,
            "injected_qids": len(injected[dataset]),
            "fallback_qids": len(fallback_qids),
            "fallback_prediction_identity_rate": fallback_identical,
            "utility_gates": utility_gates,
            "utility_gate_pass": passed,
            "predictions": {"path": str(path), "sha256": _sha256(path)},
        }

    args.out.mkdir(parents=True)
    report = {
        "schema_version": "claim-constrained-wikidata-pilot-result-1",
        "experiment_id": "CLAIM-CONSTRAINED-WIKIDATA-HOTPOT-MUSIQUE-PILOT30-V1",
        "status": "FAIL_STOP_WIKIDATA_ONLY_FOR_HOTPOT_AND_MUSIQUE",
        "development_only": True,
        "confirmation_opened": False,
        "gold_access_during_kg_build": False,
        "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "structural_integrity": build_report["structural_gate"],
        "results": results,
        "all_utility_gates_pass": all_pass,
        "metadata_incident": {
            "dataset": "hotpotqa",
            "event": "The first frozen A/B protocol omitted the report-only scope field. All 60 predictions were written before the evaluator raised KeyError('scope').",
            "impact": "No prompt, model, decoding, prediction, Gold value, or scoring rule changed. Inputs in the metadata-corrected v2 directory are byte-identical. The complete predictions were scored without regeneration and the failed manifest is preserved.",
            "failed_run_preserved": "outputs/validation/claim_constrained_wikidata_hotpot_pilot30_ab_v1",
        },
        "conclusion": {
            "supported": [
                "Every retained edge is a standard claim from a historical Wikidata entity revision and has a tail supported by the frozen passages.",
                "Coverage remains far below the previously frozen full-replacement gates on both datasets.",
                "Selective injection fails the zero-training utility gate on both datasets: each loses one correct answer and gains none.",
            ],
            "decision": "Stop Wikidata-only development for HotpotQA and MuSiQue. Do not add these branches to SAEG-v1 training or evaluation.",
            "not_claimed": [
                "Passage-derived subject-relation-object graphs are ineffective.",
                "DBpedia augmentation is ineffective.",
                "All possible external knowledge sources are ineffective.",
            ],
        },
    }
    report_path = args.out / "result_record.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, status=report["status"], extra={
        "experiment_id": report["experiment_id"],
        "phase": "claim_constrained_wikidata_pilot_result",
        "result_record_sha256": _sha256(report_path),
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
