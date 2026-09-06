#!/usr/bin/env python
"""Gold-free one-round bridge retrieval audit on deterministic train samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.retrieval.bridge import (
    additive_bridge_candidates,
    bridge_v2_rejection_reason,
    extract_bridge_queries,
    filter_bridge_queries_v2,
    reciprocal_rank_fuse,
)
from kgproweight.retrieval.hybrid import build_flashrag_config
from kgproweight.retrieval.reranker import rerank_passages
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.paths import data_dir

from scripts.pilot.audit_retrieval_topk import _aggregate, _row_metrics


_WIKI18_DOCS = 21_015_324


def _normalise_question(value: Any) -> str:
    return " ".join(re.sub(r"\s+", " ", str(value or "").strip()).split()).casefold()


def _read_source(
    dataset: str,
    split: str,
    n: int,
    seed: int,
    *,
    selection_jsonl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = Path(data_dir()) / dataset / f"{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if selection_jsonl:
        with Path(selection_jsonl).open(encoding="utf-8") as fh:
            selected = [json.loads(line) for line in fh if line.strip()]
        by_question: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_question.setdefault(_normalise_question(row.get("question")), []).append(row)
        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []
        ambiguous: List[str] = []
        for item in selected:
            question = str(item.get("question") or "")
            matches = by_question.get(_normalise_question(question), [])
            if not matches:
                missing.append(str(item.get("qid") or question))
                continue
            if len(matches) != 1:
                ambiguous.append(str(item.get("qid") or question))
                continue
            row = dict(matches[0])
            # Preserve the diagnostic set's stable replay qid while retaining
            # raw HotpotQA gold/support annotations for post-retrieval scoring.
            row["source_id"] = row.get("id")
            row["id"] = str(item.get("qid") or item.get("id") or row.get("id") or "")
            resolved.append(row)
        if missing or ambiguous:
            raise ValueError(
                f"selection resolution failed: missing={missing}, ambiguous={ambiguous}"
            )
        return resolved
    return rows if n >= len(rows) else random.Random(seed).sample(rows, n)


def _apply_explicit_retrieval_assets(
    config: Dict[str, Any],
    *,
    corpus_path: Optional[str],
    dense_index_path: Optional[str],
    bm25_index_path: Optional[str],
) -> Dict[str, Any]:
    """Override every FlashRAG retrieval path, including nested RRF entries."""
    if corpus_path:
        config["corpus_path"] = corpus_path
    if dense_index_path:
        config["index_path"] = dense_index_path
    if bm25_index_path:
        config["bm25_index_path"] = bm25_index_path
    for retriever in config.get("multi_retriever_setting", {}).get("retriever_list", []):
        if corpus_path:
            retriever["corpus_path"] = corpus_path
        method = str(retriever.get("retrieval_method") or "").lower()
        if method == "e5" and dense_index_path:
            retriever["index_path"] = dense_index_path
        elif method == "bm25" and bm25_index_path:
            retriever["index_path"] = bm25_index_path
    return config


def _validate_full_wiki18_assets(
    corpus_path: str,
    dense_index_path: str,
    bm25_index_path: str,
    *,
    expected_docs: int = _WIKI18_DOCS,
    embedding_dim: int = 768,
) -> Dict[str, Any]:
    """Fail before retrieval unless all three explicit assets are aligned."""
    from bm25s.utils.corpus import JsonlCorpus

    corpus = Path(corpus_path).resolve()
    dense = Path(dense_index_path).resolve()
    bm25 = Path(bm25_index_path).resolve()
    for path in (corpus, dense, bm25):
        if not path.exists():
            raise ValueError(f"missing retrieval asset: {path}")
    if corpus.suffix != ".jsonl":
        raise ValueError(f"full Wiki18 corpus must be JSONL: {corpus}")

    row_bytes = embedding_dim * 2
    if dense.stat().st_size % row_bytes:
        raise ValueError(f"dense index byte size is not fp16 x {embedding_dim}: {dense}")
    dense_rows = dense.stat().st_size // row_bytes
    params = json.loads((bm25 / "params.index.json").read_text(encoding="utf-8"))
    bm25_rows = int(params.get("num_docs", -1))
    corpus_view = JsonlCorpus(corpus, show_progress=False, save_index=False, verbosity=0)
    corpus_rows = len(corpus_view)
    first_id = str(corpus_view[0].get("id")) if corpus_rows else ""
    last_id = str(corpus_view[-1].get("id")) if corpus_rows else ""
    counts = {"corpus": corpus_rows, "dense": dense_rows, "bm25": bm25_rows}
    if set(counts.values()) != {expected_docs}:
        raise ValueError(f"Wiki18 asset count mismatch: {counts}, expected {expected_docs}")
    if first_id != "0" or last_id != str(expected_docs - 1):
        raise ValueError(
            f"Wiki18 corpus id boundary mismatch: first={first_id!r}, last={last_id!r}"
        )
    return {
        "status": "PASS",
        "expected_docs": expected_docs,
        "counts": counts,
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float16",
        "paths": {
            "corpus": str(corpus),
            "dense": str(dense),
            "bm25": str(bm25),
        },
        "corpus_boundary_ids": [first_id, last_id],
    }


def _build_retriever(
    dataset: str,
    topk: int,
    *,
    corpus_path: Optional[str] = None,
    dense_index_path: Optional[str] = None,
    bm25_index_path: Optional[str] = None,
):
    setup_flashrag()
    from flashrag.utils import get_retriever

    config = build_flashrag_config(
        dataset_name=dataset,
        save_note="phase1_iterative_bridge_audit",
        save_dir=str(data_dir() / "silver_data" / "_runtime"),
        split="train",
        topk=topk,
        corpus_path=corpus_path,
    )
    config = _apply_explicit_retrieval_assets(
        config,
        corpus_path=corpus_path,
        dense_index_path=dense_index_path,
        bm25_index_path=bm25_index_path,
    )
    return get_retriever(flashrag_config(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique"]
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--n_per_dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_jsonl",
        help=(
            "Optional fixed diagnostic set containing qid/question fields. "
            "Questions are resolved to raw dataset rows so gold/support labels are "
            "used only for post-retrieval scoring."
        ),
    )
    parser.add_argument("--rrf_candidate_k", type=int, default=100)
    parser.add_argument("--first_round_topk", type=int, default=5)
    parser.add_argument("--max_bridges", type=int, default=2)
    parser.add_argument(
        "--compare_abstention_v2",
        action="store_true",
        help="Also evaluate the frozen abstention-only bridge-v2 filter.",
    )
    parser.add_argument(
        "--compare_additive_v3",
        action="store_true",
        help="Also evaluate original-preserving additive bridge candidates.",
    )
    parser.add_argument(
        "--additive_bridge_only_k",
        type=int,
        default=50,
        help="Maximum bridge-only documents appended by additive v3.",
    )
    parser.add_argument("--rerank_k", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--corpus_path")
    parser.add_argument("--dense_index_path")
    parser.add_argument("--bm25_index_path")
    parser.add_argument(
        "--require_full_wiki18",
        action="store_true",
        help="Require explicit, aligned 21,015,324-document Wiki18 assets.",
    )
    parser.add_argument(
        "--passages_output",
        help=(
            "Optional JSONL containing the final ranked passages for zero-training "
            "paired evaluation. Requires a fixed --selection_jsonl."
        ),
    )
    parser.add_argument(
        "--passages_view",
        choices=("auto", "control"),
        default="auto",
        help=(
            "Passage view written by --passages_output. 'control' freezes the "
            "standard question-only RRF plus cross-encoder rerank output; "
            "'auto' preserves the historical bridge-view behavior."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.compare_abstention_v2 and args.compare_additive_v3:
        raise SystemExit("compare only one bridge research variable per audit")
    rerank_ks = sorted(set(args.rerank_k))
    if args.rrf_candidate_k <= 0 or rerank_ks[-1] > args.rrf_candidate_k:
        raise SystemExit("rerank cutoffs must fit inside the RRF candidate pool")
    if args.first_round_topk <= 0 or args.max_bridges <= 0:
        raise SystemExit("first_round_topk and max_bridges must be positive")
    if args.additive_bridge_only_k < 0:
        raise SystemExit("additive_bridge_only_k must be non-negative")
    if args.passages_output and not args.selection_jsonl:
        raise SystemExit("--passages_output requires --selection_jsonl")

    retrieval_assets = None
    explicit_paths = [args.corpus_path, args.dense_index_path, args.bm25_index_path]
    if args.require_full_wiki18:
        if not all(explicit_paths):
            raise SystemExit(
                "--require_full_wiki18 requires --corpus_path, --dense_index_path, "
                "and --bm25_index_path"
            )
        try:
            retrieval_assets = _validate_full_wiki18_assets(
                args.corpus_path,
                args.dense_index_path,
                args.bm25_index_path,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    report: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": (
            "phase1_gold_free_bridge_v1_vs_abstention_v2_audit"
            if args.compare_abstention_v2
            else (
                "phase1_gold_free_bridge_v1_vs_additive_v3_audit"
                if args.compare_additive_v3
                else "phase1_gold_free_one_round_bridge_retrieval_audit"
            )
        ),
        "split": args.split,
        "sample_strategy": "fixed_selection" if args.selection_jsonl else "random",
        "seed": args.seed,
        "n_per_dataset": args.n_per_dataset,
        "selection_jsonl": (
            str(Path(args.selection_jsonl).resolve()) if args.selection_jsonl else None
        ),
        "selection_sha256": (
            hashlib.sha256(Path(args.selection_jsonl).read_bytes()).hexdigest()
            if args.selection_jsonl
            else None
        ),
        "rrf_candidate_k": args.rrf_candidate_k,
        "first_round_topk": args.first_round_topk,
        "max_bridges": args.max_bridges,
        "bridge_source": "top passage titles + title-cased body mentions; no dataset labels",
        "bridge_v2": (
            "abstention only: reject bare generic singleton categories and repeated extraction fragments"
            if args.compare_abstention_v2
            else None
        ),
        "bridge_v3": (
            "preserve original RRF candidates; append bridge-only candidates ranked by equal-weight RRF"
            if args.compare_additive_v3
            else None
        ),
        "additive_bridge_only_k": (
            args.additive_bridge_only_k if args.compare_additive_v3 else None
        ),
        "query_fusion": "equal-weight RRF(k=60)",
        "rerank_k": rerank_ks,
        "retrieval_assets": retrieval_assets,
        "passages_view": args.passages_view if args.passages_output else None,
        "datasets": {},
    }
    passage_override_rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        try:
            rows = _read_source(
                dataset,
                args.split,
                args.n_per_dataset,
                args.seed,
                selection_jsonl=args.selection_jsonl,
            )
        except ValueError as exc:
            raise SystemExit(f"{dataset}: {exc}") from exc
        questions = [str(row.get("question") or "") for row in rows]
        retriever = _build_retriever(
            dataset,
            args.rrf_candidate_k,
            corpus_path=args.corpus_path,
            dense_index_path=args.dense_index_path,
            bm25_index_path=args.bm25_index_path,
        )

        original_candidates = [list(result) for result in retriever.batch_search(questions)]
        first_round = rerank_passages(
            questions,
            original_candidates,
            topk=max(args.first_round_topk, rerank_ks[-1]),
            method="cross-encoder",
        )
        control_only_output = bool(
            args.passages_output and args.passages_view == "control"
        )
        bridges = (
            [[] for _ in questions]
            if control_only_output
            else [
                extract_bridge_queries(
                    question,
                    docs[: args.first_round_topk],
                    max_docs=args.first_round_topk,
                    max_bridges=args.max_bridges,
                )
                for question, docs in zip(questions, first_round)
            ]
        )
        bridges_v2 = (
            [filter_bridge_queries_v2(values) for values in bridges]
            if args.compare_abstention_v2
            else []
        )

        flat_queries: List[str] = []
        owners: List[int] = []
        for owner, values in enumerate(bridges):
            for value in values:
                flat_queries.append(value)
                owners.append(owner)
        flat_results = (
            [list(result) for result in retriever.batch_search(flat_queries)]
            if flat_queries
            else []
        )
        by_owner: List[List[List[Dict[str, Any]]]] = [[] for _ in rows]
        for owner, results in zip(owners, flat_results):
            by_owner[owner].append(results)
        fused_candidates_v1 = [
            reciprocal_rank_fuse(
                [original_candidates[i], *by_owner[i]],
                topk=args.rrf_candidate_k,
            )
            for i in range(len(rows))
        ]
        bridge_reranked_v1 = (
            first_round
            if control_only_output
            else rerank_passages(
                questions,
                fused_candidates_v1,
                topk=rerank_ks[-1],
                method="cross-encoder",
            )
        )

        fused_candidates_v2: List[List[Dict[str, Any]]] = []
        bridge_reranked_v2: List[List[Dict[str, Any]]] = []
        if args.compare_abstention_v2:
            for i, (queries_v1, results_v1, queries_v2) in enumerate(
                zip(bridges, by_owner, bridges_v2)
            ):
                kept = set(queries_v2)
                results_v2 = [
                    results for query, results in zip(queries_v1, results_v1) if query in kept
                ]
                fused_candidates_v2.append(
                    reciprocal_rank_fuse(
                        [original_candidates[i], *results_v2],
                        topk=args.rrf_candidate_k,
                    )
                )
            bridge_reranked_v2 = rerank_passages(
                questions,
                fused_candidates_v2,
                topk=rerank_ks[-1],
                method="cross-encoder",
            )

        fused_candidates_v3: List[List[Dict[str, Any]]] = []
        bridge_reranked_v3: List[List[Dict[str, Any]]] = []
        if args.compare_additive_v3:
            fused_candidates_v3 = [
                additive_bridge_candidates(
                    original_candidates[i],
                    by_owner[i],
                    max_bridge_only=args.additive_bridge_only_k,
                )
                for i in range(len(rows))
            ]
            bridge_reranked_v3 = rerank_passages(
                questions,
                fused_candidates_v3,
                topk=rerank_ks[-1],
                method="cross-encoder",
            )

        views: Dict[str, Any] = {}
        detail_views: Dict[str, Any] = {}
        comparisons = [("control_candidate", original_candidates)]
        if args.compare_abstention_v2:
            comparisons.extend(
                [
                    ("bridge_v1_candidate", fused_candidates_v1),
                    ("bridge_v2_candidate", fused_candidates_v2),
                ]
            )
        elif args.compare_additive_v3:
            comparisons.extend(
                [
                    ("bridge_v1_candidate", fused_candidates_v1),
                    ("bridge_v3_candidate", fused_candidates_v3),
                ]
            )
        else:
            comparisons.append(("bridge_candidate", fused_candidates_v1))
        comparisons.extend(
            [(f"control_rerank_{k}", [docs[:k] for docs in first_round]) for k in rerank_ks]
        )
        if args.compare_abstention_v2:
            comparisons.extend(
                [
                    (f"bridge_v1_rerank_{k}", [docs[:k] for docs in bridge_reranked_v1])
                    for k in rerank_ks
                ]
            )
            comparisons.extend(
                [
                    (f"bridge_v2_rerank_{k}", [docs[:k] for docs in bridge_reranked_v2])
                    for k in rerank_ks
                ]
            )
        elif args.compare_additive_v3:
            comparisons.extend(
                [
                    (f"bridge_v1_rerank_{k}", [docs[:k] for docs in bridge_reranked_v1])
                    for k in rerank_ks
                ]
            )
            comparisons.extend(
                [
                    (f"bridge_v3_rerank_{k}", [docs[:k] for docs in bridge_reranked_v3])
                    for k in rerank_ks
                ]
            )
        else:
            comparisons.extend(
                [
                    (f"bridge_rerank_{k}", [docs[:k] for docs in bridge_reranked_v1])
                    for k in rerank_ks
                ]
            )
        for label, docs_by_row in comparisons:
            details = [_row_metrics(row, docs) for row, docs in zip(rows, docs_by_row)]
            views[label] = _aggregate(details)
            detail_views[label] = details

        qid_blob = "\n".join(str(row.get("id") or row.get("qid") or "") for row in rows)
        report["datasets"][dataset] = {
            "sample_qid_sha256": hashlib.sha256(qid_blob.encode("utf-8")).hexdigest(),
            "n_bridge_queries": len(flat_queries),
            "mean_bridge_queries": len(flat_queries) / max(1, len(rows)),
            "bridge_queries": [
                {"qid": str(row.get("id") or row.get("qid") or ""), "queries": values}
                for row, values in zip(rows, bridges)
            ],
            "metrics": views,
            "details": detail_views,
        }
        if args.compare_abstention_v2:
            rejected = [
                {"query": query, "reason": bridge_v2_rejection_reason(query)}
                for values in bridges
                for query in values
                if bridge_v2_rejection_reason(query) is not None
            ]
            n_v2 = sum(len(values) for values in bridges_v2)
            report["datasets"][dataset].update(
                {
                    "n_bridge_v1_queries": len(flat_queries),
                    "mean_bridge_v1_queries": len(flat_queries) / max(1, len(rows)),
                    "n_bridge_v2_queries": n_v2,
                    "mean_bridge_v2_queries": n_v2 / max(1, len(rows)),
                    "n_v2_full_abstain_questions": sum(not values for values in bridges_v2),
                    "v2_rejected_by_reason": dict(
                        sorted(Counter(item["reason"] for item in rejected).items())
                    ),
                    "v2_rejected_queries": rejected,
                    "bridge_query_comparison": [
                        {
                            "qid": str(row.get("id") or row.get("qid") or ""),
                            "v1": values_v1,
                            "v2": values_v2,
                        }
                        for row, values_v1, values_v2 in zip(rows, bridges, bridges_v2)
                    ],
                }
            )

        if args.passages_output:
            if args.passages_view == "control":
                final_view = first_round
                final_label = "control"
            else:
                final_view = (
                    bridge_reranked_v3
                    if args.compare_additive_v3
                    else (
                        bridge_reranked_v2
                        if args.compare_abstention_v2
                        else bridge_reranked_v1
                    )
                )
                final_label = (
                    "bridge_v3"
                    if args.compare_additive_v3
                    else ("bridge_v2" if args.compare_abstention_v2 else "bridge_v1")
                )
            for row, docs in zip(rows, final_view):
                passage_override_rows.append(
                    {
                        "qid": str(row.get("id") or row.get("qid") or ""),
                        "source_id": str(row.get("source_id") or ""),
                        "dataset": dataset,
                        "question": str(row.get("question") or "").strip(),
                        "retrieval_view": final_label,
                        "retrieved_passages": docs[: rerank_ks[-1]],
                    }
                )
        if args.compare_additive_v3:
            additions = [
                len(v3_docs) - len(original_docs)
                for original_docs, v3_docs in zip(original_candidates, fused_candidates_v3)
            ]
            report["datasets"][dataset].update(
                {
                    "n_bridge_v1_queries": len(flat_queries),
                    "mean_bridge_v1_queries": len(flat_queries) / max(1, len(rows)),
                    "additive_v3_candidate_sizes": [len(docs) for docs in fused_candidates_v3],
                    "additive_v3_bridge_only_added": additions,
                    "mean_additive_v3_bridge_only_added": sum(additions)
                    / max(1, len(additions)),
                    "min_additive_v3_bridge_only_added": min(additions, default=0),
                    "max_additive_v3_bridge_only_added": max(additions, default=0),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.passages_output:
        passage_output = Path(args.passages_output)
        passage_output.parent.mkdir(parents=True, exist_ok=True)
        with passage_output.open("w", encoding="utf-8") as fh:
            for row in passage_override_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {dataset: payload["metrics"] for dataset, payload in report["datasets"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
