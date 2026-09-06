#!/usr/bin/env python
"""A0: test whether question-surface planner anchors resolve in full Wiki18."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.entity_linker import passage_title
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.audit_iterative_bridge_retrieval import (
    _build_retriever,
    _validate_full_wiki18_assets,
)
from scripts.prepare.build_query_planner_supervision import _norm


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _titles(documents: Sequence[Mapping[str, Any]], topk: int) -> list[str]:
    return [passage_title(document) for document in documents[:topk] if passage_title(document)]


def _coverage(
    target_titles: Sequence[str],
    query_anchors: Sequence[str],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    topk: int,
) -> dict[str, Any]:
    retrieved: list[str] = []
    for anchor in query_anchors:
        retrieved.extend(_titles(results[anchor], topk))
    retrieved_keys = {_norm(title) for title in retrieved}
    target_keys = [_norm(title) for title in target_titles]
    hits = [title for title, key in zip(target_titles, target_keys) if key in retrieved_keys]
    return {
        "target_n": len(target_titles),
        "target_hit_n": len(hits),
        "recall": len(hits) / len(target_titles) if target_titles else 0.0,
        "any_hit": bool(hits),
        "complete": bool(target_titles) and len(hits) == len(target_titles),
        "hit_titles": hits,
        "retrieved_titles": list(dict.fromkeys(retrieved)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], source: str, topk: int) -> dict[str, float]:
    values = [row[f"{source}_at_{topk}"] for row in rows]
    return {
        "anchor_recall": sum(value["recall"] for value in values) / len(values),
        "question_any": sum(value["any_hit"] for value in values) / len(values),
        "question_complete": sum(value["complete"] for value in values) / len(values),
    }


def _evaluate_gates(metrics: Mapping[str, Any], gates: Mapping[str, float]) -> dict[str, Any]:
    predicted_20 = metrics["predicted_surface"]["20"]["question_complete"]
    oracle_20 = metrics["gold_alias_oracle"]["20"]["question_complete"]
    ratio = predicted_20 / oracle_20 if oracle_20 else 0.0
    checks = {
        "oracle_complete_at_20": oracle_20 >= gates["oracle_complete_at_20_min"],
        "predicted_complete_at_5": (
            metrics["predicted_surface"]["5"]["question_complete"]
            >= gates["predicted_complete_at_5_min"]
        ),
        "predicted_complete_at_20": predicted_20 >= gates["predicted_complete_at_20_min"],
        "predicted_to_oracle_ratio_at_20": ratio >= gates["predicted_to_oracle_ratio_at_20_min"],
    }
    return {"pass": all(checks.values()), "checks": checks, "predicted_to_oracle_ratio_at_20": ratio}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--dense_index_path", required=True)
    parser.add_argument("--bm25_index_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    if args.topk != int(protocol["retrieval"]["topk"]):
        raise SystemExit("runtime topk differs from frozen protocol")
    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "planner_v2_a0_title_retrieval",
            "cohort": artifact_identity(args.cohort),
            "protocol": artifact_identity(args.protocol),
        },
    )
    assets = _validate_full_wiki18_assets(
        args.corpus_path,
        args.dense_index_path,
        args.bm25_index_path,
        expected_docs=int(protocol["retrieval"]["expected_docs"]),
    )
    cohort = list(_read_jsonl(args.cohort))
    queries = sorted({
        anchor
        for row in cohort
        for anchor in row["predicted_anchors"] + row["gold_alias_anchors"]
    })
    retriever = _build_retriever(
        "2wikimultihopqa",
        args.topk,
        corpus_path=args.corpus_path,
        dense_index_path=args.dense_index_path,
        bm25_index_path=args.bm25_index_path,
    )
    retrieved = retriever.batch_search(queries)
    results = {query: list(documents) for query, documents in zip(queries, retrieved)}

    detail_rows: list[dict[str, Any]] = []
    for row in cohort:
        detail = dict(row)
        for topk in (5, 10, 20):
            detail[f"predicted_surface_at_{topk}"] = _coverage(
                row["gold_alias_anchors"], row["predicted_anchors"], results, topk
            )
            detail[f"gold_alias_oracle_at_{topk}"] = _coverage(
                row["gold_alias_anchors"], row["gold_alias_anchors"], results, topk
            )
        detail_rows.append(detail)

    metrics = {
        source: {
            str(topk): _aggregate(detail_rows, source, topk) for topk in (5, 10, 20)
        }
        for source in ("predicted_surface", "gold_alias_oracle")
    }
    gate_result = _evaluate_gates(metrics, protocol["gates"])
    details_path = output_dir / "details.jsonl"
    with details_path.open("x", encoding="utf-8") as fh:
        for row in detail_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "PASS" if gate_result["pass"] else "FAIL_STOP",
        "scope": "seen_confirmation_diagnostics_only",
        "n": len(detail_rows),
        "retrieval": {
            "method": "full_wiki18_e5_bm25_rrf",
            "topk": args.topk,
            "rrf_k": 60,
            "per_retriever_candidate_topk": 100,
        },
        "metrics": metrics,
        "gates": gate_result,
        "assets": assets,
        "inputs": {
            "cohort": artifact_identity(args.cohort),
            "protocol": artifact_identity(args.protocol),
        },
        "details": artifact_identity(details_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
