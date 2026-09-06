#!/usr/bin/env python
"""Targeted, low-concurrency prefetch for the inference Proof-KG pilot.

Reads the machine-executable missing-request list from step 6B and, using the
same retriever classes the final offline rebuild will read, fetches ONLY:

- anchor title/QID resolution (WikipediaTitleResolver);
- exact (entity_qid, pid) property values at the frozen historical cutoff
  (HistoricalWikidataPropertyRetriever).

Writes into a NEW isolated cache dir (never touches existing caches) and emits a
request log + cache manifest.  No Gold field is read or written.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.historical_wikidata_retriever import HistoricalWikidataPropertyRetriever
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REQUESTS = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_pilot30_missing_requests"
    / "missing_requests.jsonl"
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "indexes" / "inference_proofkg_v1_pilot30"

CUTOFF = "2020-12-09T23:59:59Z"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cutoff", default=CUTOFF)
    args = parser.parse_args()

    if args.cache_dir.exists() and any(args.cache_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty cache dir: {args.cache_dir}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    requests = _read_jsonl(args.requests)
    title_reqs = [r for r in requests if r["request_type"] == "title_or_qid_resolution"]
    prop_reqs = [r for r in requests if r["request_type"] == "historical_property"]

    title_cache = args.cache_dir / "title_cache.jsonl"
    hist_cache = args.cache_dir / "historical_property_cache.jsonl"

    title_resolver = WikipediaTitleResolver(
        cache_path=title_cache, offline=False, timeout=args.timeout, max_retries=args.retries,
    )
    hist_retriever = HistoricalWikidataPropertyRetriever(
        cache_path=hist_cache, cutoff=args.cutoff, offline=False,
        timeout=args.timeout, request_delay=args.delay, max_retries=args.retries,
    )

    log_rows: List[Dict[str, Any]] = []
    lock = __import__("threading").Lock()

    def resolve_title(req: Dict[str, Any]) -> Dict[str, Any]:
        surface = req["anchor_surface"]
        result = title_resolver.resolve(surface)
        outcome = (
            "success" if (result.selected_qid and not result.abstained)
            else "abstain" if result.abstained else "success"
        )
        return {
            "dataset": req["dataset"], "qid": req["qid"], "request_type": "title_or_qid_resolution",
            "surface": surface, "qid_resolved": result.selected_qid, "outcome": outcome,
            "abstain_reason": result.abstain_reason if result.abstained else "",
        }

    def fetch_prop(req: Dict[str, Any]) -> Dict[str, Any]:
        qid, pid = req["entity_qid"], req["pid"]
        edges = hist_retriever.fetch_edges(qid, [pid])
        return {
            "dataset": req["dataset"], "qid": req["qid"], "request_type": "historical_property",
            "entity_qid": qid, "pid": pid, "n_edges": len(edges),
            "outcome": "success" if edges else "abstain",
            "abstain_reason": "" if edges else "no_edges_at_cutoff",
        }

    # Deduplicate by (qid) for properties and by surface for titles.
    seen_prop = set()
    prop_jobs = []
    for r in prop_reqs:
        key = (r["entity_qid"], r["pid"])
        if key in seen_prop:
            continue
        seen_prop.add(key)
        prop_jobs.append(r)
    seen_title = set()
    title_jobs = []
    for r in title_reqs:
        if r["anchor_surface"] in seen_title:
            continue
        seen_title.add(r["anchor_surface"])
        title_jobs.append(r)

    total_jobs = len(title_jobs) + len(prop_jobs)
    done = 0

    def run(job, fn):
        nonlocal done
        row = fn(job)
        with lock:
            log_rows.append(row)
            done += 1
        return row

    logger.info("prefetch: %d title + %d property unique jobs (workers=%d, delay=%s)",
                len(title_jobs), len(prop_jobs), args.workers, args.delay)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for r in title_jobs:
            futures.append(pool.submit(run, r, resolve_title))
        for r in prop_jobs:
            futures.append(pool.submit(run, r, fetch_prop))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # keep the run alive on a single request failure
                with lock:
                    log_rows.append({"outcome": "fail", "error": f"{type(exc).__name__}: {exc}"})
                    done += 1
            print(f"prefetched {done}/{total_jobs}", flush=True)

    # Persist the request log, then record cache file hashes.
    log_path = args.cache_dir / "request_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for row in sorted(log_rows, key=lambda r: json.dumps(r, sort_keys=True)):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {"success": 0, "abstain": 0, "fail": 0}
    for r in log_rows:
        counts[r.get("outcome", "fail")] = counts.get(r.get("outcome", "fail"), 0) + 1

    manifest = {
        "schema_version": "inference-proofkg-prefetch-manifest-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TARGETED_PREFETCH_COMPLETE",
        "cutoff": args.cutoff,
        "policy": {"workers": args.workers, "delay": args.delay, "timeout": args.timeout, "retries": args.retries},
        "api_endpoint": {
            "title_resolution": "wikipedia action=query prop=pageprops (wikibase_item)",
            "property": "wikidata wbgetentities / action=query revisions before cutoff",
        },
        "counts": {
            "unique_jobs": total_jobs,
            "requested": len(requests),
            **counts,
        },
        "files": {
            "title_cache": {"path": str(title_cache), "sha256": _sha256(title_cache) if title_cache.exists() else None},
            "historical_property_cache": {"path": str(hist_cache), "sha256": _sha256(hist_cache) if hist_cache.exists() else None},
            "request_log": {"path": str(log_path), "sha256": _sha256(log_path)},
        },
        "source_requests_sha256": _sha256(args.requests),
        "note": "raw revision values retained alongside display values inside the historical cache; no existing cache was modified.",
    }
    (args.cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
