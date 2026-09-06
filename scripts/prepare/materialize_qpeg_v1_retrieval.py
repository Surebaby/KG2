#!/usr/bin/env python
"""Materialise frozen QPEG pilot/confirmation retrieval contexts.

Uses the canonical two-stage retrieval stack and writes only question + top-10
passages.  The input cohort is already frozen and contains no gold fields.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.retrieval.hybrid import build_flashrag_config
from kgproweight.retrieval.reranker import pack_passages_by_token_budget, rerank_passages
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import dump_manifest


SCHEMA_VERSION = "qpeg-retrieval-context-v1"
EXPERIMENT_ID = "QPEG-V1-PILOT50-CONFIRMATION100-RETRIEVAL-SEED42"
FORBIDDEN_FIELDS = {
    "golden_answers", "answer", "answers", "supporting_facts", "support",
    "decomposition", "question_decomposition", "evidence", "reasoning", "sp",
}


def _sha_json(value: Any) -> str:
    # Keep passage identity compatible with the historical canonical n=300
    # retrieval freeze.
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _build_retriever(dataset: str):
    setup_flashrag()
    from flashrag.utils import get_retriever

    config = build_flashrag_config(
        dataset_name=dataset,
        save_note="qpeg_v1_retrieval",
        save_dir="outputs/_qpeg_retrieval_runtime",
        split="dev",
        topk=50,
        seed=42,
    )
    return get_retriever(flashrag_config(config))


def materialize_dataset(dataset: str, rows: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    retriever = _build_retriever(dataset)
    contexts: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        questions = [str(row["question"]) for row in batch]
        candidates = [list(values) for values in retriever.batch_search(questions)]
        if len(candidates) != len(batch):
            raise RuntimeError(f"{dataset}: retrieval returned {len(candidates)}/{len(batch)}")
        reranked = rerank_passages(
            questions,
            candidates,
            topk=10,
            method="cross-encoder",
            cross_encoder_model="models/bge-reranker-v2-m3",
        )
        reranked = [pack_passages_by_token_budget(values, 3860) for values in reranked]
        for row, passages in zip(batch, reranked):
            if len(passages) != 10:
                raise ValueError(f"{dataset}/{row['qid']}: expected 10 passages, got {len(passages)}")
            contexts.append({
                "schema_version": SCHEMA_VERSION,
                "question_key": row["question_key"],
                "dataset": dataset,
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "family_sha256": row["family_sha256"],
                "role": row["role"],
                "gold_access": False,
                "passages": passages,
                "passages_sha256": _sha_json(passages),
                "retrieval_source": "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860",
            })
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path("outputs/audits/qpeg_v1_n1350_seed42_preregistration/retrieval_requests.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/audits/qpeg_v1_pilot_confirmation_retrieval_seed42"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite retrieval output: {args.out}")
    if args.batch_size < 1:
        raise SystemExit("batch_size must be positive")

    rows = _read_jsonl(args.requests)
    if len(rows) != 450:
        raise ValueError(f"expected 450 frozen requests, got {len(rows)}")
    for row in rows:
        present = FORBIDDEN_FIELDS & set(row)
        if present:
            raise ValueError(f"{row.get('question_key')}: forbidden fields: {sorted(present)}")
        if row.get("gold_access") is not False:
            raise ValueError(f"{row.get('question_key')}: gold_access must be false")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    args.out.mkdir(parents=True)
    all_contexts: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        dataset_rows = grouped[dataset]
        if len(dataset_rows) != 150:
            raise ValueError(f"{dataset}: expected 150 requests, got {len(dataset_rows)}")
        contexts = materialize_dataset(dataset, dataset_rows, args.batch_size)
        _write_jsonl(args.out / f"{dataset}.retrieval_contexts.jsonl", contexts)
        all_contexts.extend(contexts)
        counts[dataset] = len(contexts)
    _write_jsonl(args.out / "retrieval_contexts.jsonl", all_contexts)

    report = {
        "schema_version": "qpeg-retrieval-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": counts,
        "total": len(all_contexts),
        "requests_sha256": _sha_file(args.requests),
        "contexts_sha256": _sha_file(args.out / "retrieval_contexts.jsonl"),
        "retrieval": "E5@100 + BM25@100 -> RRF(k=60)@50 -> bge-reranker-v2-m3@10 -> pack3860",
        "gold_access": False,
        "all_have_10_passages": all(len(row["passages"]) == 10 for row in all_contexts),
        "identity_unique": len({row["question_key"] for row in all_contexts}) == len(all_contexts),
    }
    if not report["all_have_10_passages"] or not report["identity_unique"]:
        raise SystemExit(f"retrieval gates failed: {report}")
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        args.out,
        extra={"phase": "qpeg_retrieval", **report},
        status="COMPLETE",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
