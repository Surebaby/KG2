#!/usr/bin/env python
"""No-Teacher retrieval audit for deterministic Phase-1 source questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.retrieval.hybrid import build_flashrag_config
from kgproweight.retrieval.reranker import rerank_passages
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.paths import data_dir


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _doc_title(doc: Dict[str, Any]) -> str:
    title = str(doc.get("title") or "").strip()
    if title:
        return title
    contents = str(doc.get("contents") or doc.get("text") or "").strip()
    return contents.splitlines()[0].strip() if contents else ""


def _supporting_titles(row: Dict[str, Any]) -> List[str]:
    metadata = row.get("metadata") or {}
    support = metadata.get("supporting_facts") or {}
    titles = list(support.get("title") or [])
    if not titles:
        nested = metadata.get("metadata") or {}
        for hop in nested.get("question_decomposition") or []:
            paragraph = hop.get("support_paragraph") or {}
            title = paragraph.get("title")
            if title:
                titles.append(title)
    return list(dict.fromkeys(str(title).strip() for title in titles if str(title).strip()))


def _gold_answers(row: Dict[str, Any]) -> List[str]:
    return [str(value) for value in (row.get("golden_answers") or []) if str(value).strip()]


def _row_metrics(row: Dict[str, Any], docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    retrieved_titles = {_normalise(_doc_title(doc)) for doc in docs if _doc_title(doc)}
    support_titles = _supporting_titles(row)
    support_keys = [_normalise(title) for title in support_titles]
    n_support = len(support_keys)
    n_support_hit = sum(key in retrieved_titles for key in support_keys)
    passage_blob = _normalise(" ".join(str(doc.get("contents") or doc.get("text") or "") for doc in docs))
    golds = [_normalise(value) for value in _gold_answers(row)]
    gold_literal_hit = any(value and value in passage_blob for value in golds)
    return {
        "qid": str(row.get("id") or row.get("qid") or ""),
        "n_support_titles": n_support,
        "n_support_titles_hit": n_support_hit,
        "any_support_hit": bool(n_support_hit),
        "all_support_hit": bool(n_support and n_support_hit == n_support),
        "gold_literal_hit": gold_literal_hit,
        "supporting_titles": support_titles,
    }


def _aggregate(details: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    details = list(details)
    n = len(details)
    support_total = sum(row["n_support_titles"] for row in details)
    support_hit = sum(row["n_support_titles_hit"] for row in details)
    return {
        "n_questions": n,
        "questions_with_support_annotations": sum(row["n_support_titles"] > 0 for row in details),
        "any_support_recall_pct": 100.0 * sum(row["any_support_hit"] for row in details) / max(1, n),
        "all_support_recall_pct": 100.0 * sum(row["all_support_hit"] for row in details) / max(1, n),
        "support_title_micro_recall_pct": 100.0 * support_hit / max(1, support_total),
        "gold_literal_hit_pct": 100.0 * sum(row["gold_literal_hit"] for row in details) / max(1, n),
        "support_titles_hit": support_hit,
        "support_titles_total": support_total,
    }


def _validate_cutoffs(rrf_candidate_k: int, rerank_k: Sequence[int]) -> List[int]:
    rerank_ks = sorted(set(int(value) for value in rerank_k))
    if rrf_candidate_k <= 0 or rrf_candidate_k > 200:
        raise ValueError("rrf_candidate_k must be in [1, 200]")
    if not rerank_ks or rerank_ks[0] <= 0 or rerank_ks[-1] > rrf_candidate_k:
        raise ValueError("rerank_k must be positive and no larger than rrf_candidate_k")
    return rerank_ks


def _read_source(dataset: str, split: str, n: int, seed: int) -> List[Dict[str, Any]]:
    path = Path(data_dir()) / dataset / f"{split}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


def _build_retriever(dataset: str, topk: int):
    setup_flashrag()
    from flashrag.utils import get_retriever

    raw_config = build_flashrag_config(
        dataset_name=dataset,
        save_note="phase1_retrieval_topk_audit",
        save_dir=str(data_dir() / "silver_data" / "_runtime"),
        split="train",
        topk=topk,
    )
    return get_retriever(flashrag_config(raw_config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wikimultihopqa", "musique"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--n_per_dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rrf_candidate_k", type=int, default=50)
    parser.add_argument("--rerank_k", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        rerank_ks = _validate_cutoffs(args.rrf_candidate_k, args.rerank_k)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "phase1_retrieval_topk_audit_no_teacher",
        "split": args.split,
        "sample_strategy": "random",
        "seed": args.seed,
        "n_per_dataset": args.n_per_dataset,
        "rrf_candidate_k": args.rrf_candidate_k,
        "rerank_k": rerank_ks,
        "datasets": {},
    }
    for dataset in args.datasets:
        rows = _read_source(dataset, args.split, args.n_per_dataset, args.seed)
        questions = [str(row.get("question") or "") for row in rows]
        retriever = _build_retriever(dataset, topk=args.rrf_candidate_k)
        candidates = [list(result) for result in retriever.batch_search(questions)]
        if len(candidates) != len(rows):
            raise RuntimeError(f"{dataset}: retriever returned {len(candidates)}/{len(rows)} rows")
        reranked = rerank_passages(questions, candidates, topk=rerank_ks[-1], method="cross-encoder")

        views: Dict[str, Any] = {}
        detail_views: Dict[str, Any] = {}
        for label, docs_by_row in [
            (f"rrf_{args.rrf_candidate_k}", candidates),
            *[(f"rerank_{k}", [docs[:k] for docs in reranked]) for k in rerank_ks],
        ]:
            details = [_row_metrics(row, docs) for row, docs in zip(rows, docs_by_row)]
            views[label] = _aggregate(details)
            detail_views[label] = details

        qid_blob = "\n".join(str(row.get("id") or row.get("qid") or "") for row in rows)
        report["datasets"][dataset] = {
            "sample_qid_sha256": hashlib.sha256(qid_blob.encode("utf-8")).hexdigest(),
            "metrics": views,
            "details": detail_views,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        dataset: payload["metrics"]
        for dataset, payload in report["datasets"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
