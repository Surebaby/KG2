#!/usr/bin/env python
"""Derive canonical prompt passages and hashes from a QPEG retrieval run.

Document IDs/order are copied unchanged.  The first materialisation omitted
the pipeline's deterministic 3860-token passage packing and encoded hashes
with compact separators.  This utility applies the same packing function as
the canonical pipeline and then writes the historical default-separator hash.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from kgproweight.utils.logging import dump_manifest
from kgproweight.retrieval.reranker import pack_passages_by_token_budget


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _hash(value: Any, *, compact: bool) -> str:
    kwargs = {"sort_keys": True, "ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    blob = json.dumps(value, **kwargs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite normalised output: {args.out}")
    args.out.mkdir(parents=True)

    combined: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    old_compact = 0
    already_canonical = 0
    passage_objects_changed = 0
    for dataset in DATASETS:
        source_file = args.source / f"{dataset}.retrieval_contexts.jsonl"
        source_hashes[str(source_file)] = _sha_file(source_file)
        rows = _read(source_file)
        converted: list[dict[str, Any]] = []
        for row in rows:
            supplied = str(row.get("passages_sha256") or "")
            source_passages = row["passages"]
            compact = _hash(source_passages, compact=True)
            canonical_source = _hash(source_passages, compact=False)
            if supplied == compact:
                old_compact += 1
            elif supplied == canonical_source:
                already_canonical += 1
            else:
                raise ValueError(f"{row.get('question_key')}: passages hash matches neither known algorithm")
            packed = pack_passages_by_token_budget(source_passages, 3860)
            if packed != source_passages:
                passage_objects_changed += 1
            canonical = _hash(packed, compact=False)
            converted_row = dict(row)
            converted_row["source_passages_sha256"] = supplied
            converted_row["passages"] = packed
            converted_row["passages_sha256"] = canonical
            converted_row["passages_hash_algorithm"] = "json.dumps(sort_keys=True, ensure_ascii=False, default_separators)"
            converted_row["retrieval_source"] = "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
            converted.append(converted_row)
        _write(args.out / f"{dataset}.retrieval_contexts.jsonl", converted)
        combined.extend(converted)
        counts[dataset] = len(converted)
    _write(args.out / "retrieval_contexts.jsonl", combined)

    report = {
        "schema_version": "qpeg-retrieval-hash-normalization-v1",
        "experiment_id": "QPEG-V1-PILOT-CONFIRMATION-RETRIEVAL-CANONICAL-HASH-V2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DERIVED_NO_RETRIEVAL_CHANGE",
        "source": str(args.source),
        "source_hashes": source_hashes,
        "counts": counts,
        "total": len(combined),
        "source_compact_hash_rows": old_compact,
        "source_already_canonical_rows": already_canonical,
        "document_id_or_order_changed": False,
        "passage_objects_changed_by_canonical_pack3860": passage_objects_changed,
        "output_sha256": _sha_file(args.out / "retrieval_contexts.jsonl"),
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(args.out, extra={"phase": "qpeg_retrieval_hash_normalization", **report}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
