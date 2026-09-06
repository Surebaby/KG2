#!/usr/bin/env python
"""Audit whether existing paper baselines can be reused in the QPEG main table."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from flashrag.evaluator.utils import normalize_answer


METHODS = ("naive_rag", "self_rag", "r1_searcher", "corag", "trace", "rearag")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _canonical_em(prediction: str, golden_answers: list[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(answer) for answer in golden_answers))


def _canonical_f1(prediction: str, golden_answers: list[str]) -> float:
    """Mirror FlashRAG F1_Score, including its yes/no/noanswer guard."""
    normalized_prediction = normalize_answer(prediction)
    best = 0.0
    for ground_truth in golden_answers:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        if not prediction_tokens or not ground_truth_tokens:
            continue
        num_same = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
        if num_same == 0:
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _passage_ids(row: dict[str, Any]) -> list[str]:
    passages = (row.get("output") or {}).get("retrieval_result") or row.get("passages") or []
    return [str(passage.get("id")) for passage in passages]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final_contexts", type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.retrieval_contexts.jsonl"),
    )
    parser.add_argument("--baseline_root", type=Path, default=Path("outputs/baselines_rerank"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v1_external_baseline_compatibility_v2"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")
    args.out.mkdir(parents=True)

    contexts = _read_jsonl(args.final_contexts)
    final = {
        dataset: {str(row["qid"]): row for row in contexts if row["dataset"] == dataset}
        for dataset in DATASETS
    }
    details: dict[str, Any] = {}
    for method in METHODS:
        details[method] = {}
        for dataset in DATASETS:
            pattern = str(args.baseline_root / method / dataset / "seed_42" / "*" / "intermediate_data.json")
            paths = [Path(value) for value in glob.glob(pattern)]
            if len(paths) != 1:
                raise ValueError(f"{method}/{dataset}: expected one intermediate_data.json, got {paths}")
            path = paths[0]
            rows = json.loads(path.read_text(encoding="utf-8"))
            qid_match = 0
            question_match = 0
            passage_order_match = 0
            passage_set_match = 0
            row_em: list[float] = []
            row_f1: list[float] = []
            canonical_em: list[float] = []
            canonical_f1: list[float] = []
            stored_vs_canonical_exact = 0
            for row in rows:
                qid = str(row.get("id"))
                target = final[dataset].get(qid)
                if target is None:
                    continue
                qid_match += 1
                question_match += str(row.get("question") or "").strip() == str(target["question"]).strip()
                actual_ids = _passage_ids(row)
                target_ids = [str(passage.get("id")) for passage in target["passages"]]
                passage_order_match += actual_ids == target_ids
                passage_set_match += set(actual_ids) == set(target_ids)
                metric = (row.get("output") or {}).get("metric_score") or {}
                prediction = str((row.get("output") or {}).get("pred") or "")
                golden_answers = [str(value) for value in row.get("golden_answers") or []]
                recomputed_em = _canonical_em(prediction, golden_answers)
                recomputed_f1 = _canonical_f1(prediction, golden_answers)
                canonical_em.append(recomputed_em)
                canonical_f1.append(recomputed_f1)
                if "em" in metric:
                    row_em.append(float(metric["em"]))
                if "f1" in metric:
                    row_f1.append(float(metric["f1"]))
                if (
                    "em" in metric and "f1" in metric
                    and abs(float(metric["em"]) - recomputed_em) <= 1e-12
                    and abs(float(metric["f1"]) - recomputed_f1) <= 1e-12
                ):
                    stored_vs_canonical_exact += 1
            config_path = path.with_name("config.yaml")
            config_text = config_path.read_text(encoding="utf-8", errors="ignore") if config_path.exists() else ""
            same_final_identity = len(rows) == 300 and qid_match == question_match == 300
            same_resource = "indexes_wiki18/corpus_flashrag.jsonl" in config_text
            deterministic = "temperature: 0.0" in config_text and "do_sample: false" in config_text
            if passage_order_match == 300:
                comparison_scope = "same_final_qids_and_same_top10_document_order"
            elif passage_set_match == 300:
                comparison_scope = "same_final_qids_and_same_top10_document_set"
            else:
                comparison_scope = "same_final_qids_and_wiki18_resource_end_to_end_only"
            details[method][dataset] = {
                "run_dir": str(path.parent),
                "n": len(rows),
                "qid_match": qid_match,
                "question_match": question_match,
                "passage_id_order_match": passage_order_match,
                "passage_id_set_match": passage_set_match,
                "same_wiki18_corpus": same_resource,
                "temperature_zero_and_no_sampling": deterministic,
                "stored_metric_rows": {"em": len(row_em), "f1": len(row_f1)},
                "stored_metrics": {
                    "em": sum(row_em) / len(row_em) if row_em else None,
                    "f1": sum(row_f1) / len(row_f1) if row_f1 else None,
                },
                "canonical_rescore": {
                    "em": sum(canonical_em) / len(canonical_em) if canonical_em else None,
                    "f1": sum(canonical_f1) / len(canonical_f1) if canonical_f1 else None,
                    "rows": len(canonical_em),
                },
                "stored_vs_canonical_exact_rows": stored_vs_canonical_exact,
                "main_table_identity_gate": same_final_identity and same_resource and deterministic,
                "comparison_scope": comparison_scope,
            }

    gates = {
        "all_methods_qid_question_match_300": all(
            payload["qid_match"] == payload["question_match"] == 300
            for method in details.values() for payload in method.values()
        ),
        "all_methods_same_wiki18_resource": all(
            payload["same_wiki18_corpus"] for method in details.values() for payload in method.values()
        ),
        "all_methods_deterministic_decoding": all(
            payload["temperature_zero_and_no_sampling"] for method in details.values() for payload in method.values()
        ),
        "all_stored_metric_rows_complete": all(
            payload["stored_metric_rows"] == {"em": 300, "f1": 300}
            for method in details.values() for payload in method.values()
        ),
        "all_predictions_canonical_rescore_exact": all(
            payload["stored_vs_canonical_exact_rows"] == 300
            for method in details.values() for payload in method.values()
        ),
    }
    report = {
        "schema_version": "qpeg-external-baseline-compatibility-v2",
        "experiment_id": "QPEG-V1-EXTERNAL-BASELINE-COMPATIBILITY-V2-CANONICAL-RESCORE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_MAIN_TABLE_REUSE" if all(gates.values()) else "FAIL_REQUIRES_RERUN",
        "interpretation": {
            "main_table": "native end-to-end systems; same final qids, Wiki18 resource, seed42, deterministic decode, canonical rescored EM/F1",
            "strict_qpeg_attribution": "not provided by external baselines; use internal frozen-passage A-F arms",
            "rearag": "native retrieval differs and is end-to-end-only, not a matched-passage causal control",
            "proofkg": "extra-Wikidata result remains outside the same-resource table",
        },
        "gates": gates,
        "details": details,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra={"phase": "baseline_compatibility", **report}, status=report["status"])
    print(json.dumps({"status": report["status"], "gates": gates}, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
