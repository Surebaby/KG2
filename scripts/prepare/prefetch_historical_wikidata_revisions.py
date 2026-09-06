#!/usr/bin/env python
"""Concurrently prefetch historical Wikidata revisions from Gold-free runtime QIDs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Mapping

from kgproweight.kg.historical_wikidata_retriever import (
    HISTORICAL_CACHE_VERSION,
    HistoricalWikidataPropertyRetriever,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest


PREFETCH_VERSION = "historical-wikidata-runtime-prefetch-1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def runtime_qids(rows: Iterable[Mapping[str, Any]]) -> tuple[set[str], Dict[str, set[str]]]:
    qids: set[str] = set()
    qid_pids: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        execution = row.get("execution") or {}
        for entity in (execution.get("anchor_entities") or {}).values():
            if entity.get("qid"):
                qids.add(str(entity["qid"]))
        for hop in execution.get("hops") or []:
            for entity in hop.get("input_entities") or []:
                if entity.get("qid"):
                    qid = str(entity["qid"])
                    qids.add(qid)
                    qid_pids[qid].update(str(pid) for pid in hop.get("pids") or [])
            for entity in hop.get("output_entities") or []:
                if entity.get("qid"):
                    qids.add(str(entity["qid"]))
    return qids, qid_pids


def _seed_cache(target: Path, seeds: list[Path], cutoff: str) -> int:
    seen: set[str] = set()
    rows: list[str] = []
    for seed in seeds:
        for row in _read_jsonl(seed):
            if row.get("schema_version") != HISTORICAL_CACHE_VERSION or row.get("cutoff") != cutoff:
                continue
            key = str(row.get("key") or "")
            if key and key not in seen:
                seen.add(key)
                rows.append(json.dumps(row, ensure_ascii=False))
    if rows:
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--seed_cache", action="append", default=[])
    parser.add_argument("--cutoff", default="2020-12-09T23:59:59Z")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.2,
        help="Per-worker delay after a successful Wikidata request; use low concurrency to avoid HTTP 429.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be in [1, 32]")
    if args.request_delay < 0:
        raise SystemExit("--request_delay must be >= 0")

    runtime_path = Path(args.runtime_details).resolve()
    seed_paths = [Path(value).resolve() for value in args.seed_cache]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    for path in (runtime_path, *seed_paths):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    output_dir.mkdir(parents=True)
    cache_path = output_dir / "historical_entity_cache.jsonl"
    seeded = _seed_cache(cache_path, seed_paths, args.cutoff)
    qids, qid_pids = runtime_qids(_read_jsonl(runtime_path))
    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=cache_path,
        cutoff=args.cutoff,
        offline=False,
        timeout=args.timeout,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
    )
    missing = sorted(qid for qid in qids if retriever.cache_key(qid) not in retriever._cache)
    counts: Counter[str] = Counter()

    def fetch(qid: str) -> tuple[str, str | None]:
        try:
            entity, _ = retriever._entity(qid)
            return qid, None if entity is not None else "missing_before_cutoff"
        except Exception as exc:  # retained in report; transient errors are not cached
            return qid, f"{type(exc).__name__}: {exc}"

    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, qid): qid for qid in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            qid, error = future.result()
            if error:
                errors[qid] = error
                counts["failed_or_missing"] += 1
            else:
                counts["fetched"] += 1
            if index % 20 == 0 or index == len(futures):
                print(f"prefetched {index}/{len(futures)}", flush=True)

    counts["runtime_qids"] = len(qids)
    counts["runtime_qid_pid_pairs"] = sum(len(pids) for pids in qid_pids.values())
    counts["seeded_cache_rows"] = seeded
    counts["requested_missing_qids"] = len(missing)
    counts["final_cache_rows"] = sum(1 for _ in _read_jsonl(cache_path)) if cache_path.exists() else 0
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "prefetch_version": PREFETCH_VERSION,
        "status": "COMPLETE" if not errors else "COMPLETE_WITH_MISSES",
        "cutoff": args.cutoff,
        "workers": args.workers,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "request_delay": args.request_delay,
        "scientific_boundary": {
            "runtime_details_gold_access": False,
            "dataset_annotations_access": False,
            "model_training": False,
        },
        "counts": dict(counts),
        "errors": errors,
        "inputs": {
            "runtime_details": artifact_identity(runtime_path),
            "seed_caches": [artifact_identity(path) for path in seed_paths],
        },
        "outputs": {"historical_entity_cache": artifact_identity(cache_path)},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status=report["status"], extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
