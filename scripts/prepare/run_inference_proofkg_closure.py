#!/usr/bin/env python
"""Bounded iterative exact-prefetch closure for the 2Wiki Proof-KG pilot.

Offline execution -> extract newly-exposed exact (QID, PID) -> targeted
historical prefetch (real N-thread, deterministic) -> offline execution again,
until no new request or max_rounds.  Every round saves its requests, its cache
*increment* (not just the growing total) and its runtime separately, each with a
SHA256; the v2 cache and v2 failed result are never overwritten.

Reproducibility note: after the parallel prefetch the cache files are re-sorted
by QID so their bytes (and therefore SHA256) are run-to-run identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from kgproweight.kg.historical_wikidata_retriever import HistoricalWikidataPropertyRetriever
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANS = DEFAULT_ROOT / "outputs/audits/inference_proofkg_v1_pilot30_offline_diag/2wikimultihopqa/plans.question_only.jsonl"
DEFAULT_PROTO = DEFAULT_ROOT / "outputs/audits/inference_proofkg_v1_pilot30x3_execution_v1/execution_protocol.json"
DEFAULT_ENTITY_INDEX = DEFAULT_ROOT / "data/silver_data/pilots/entity_candidates_local_roots_v2_20260828/entity_desc_index.expanded_local_roots.json"
DEFAULT_ENTITY_CACHE = DEFAULT_ROOT / "indexes/entity_cache.jsonl"
DEFAULT_TITLE_CACHE = DEFAULT_ROOT / "indexes/inference_proofkg_v1_pilot30/title_cache.jsonl"
DEFAULT_BASE_HIST = DEFAULT_ROOT / "indexes/inference_proofkg_v1_pilot30/historical_property_cache.jsonl"
DEFAULT_VERSIONED_ALIAS = DEFAULT_ROOT / "indexes/versioned_2wiki_evidence_store_v2_seed20260902"
DEFAULT_OUT = DEFAULT_ROOT / "data/derived/inference_proofkg_v1_pilot30/2wikimultihopqa/closure_v3b"
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


def _seed_requested(missing_requests_path: Path, dataset: str) -> Set[Tuple[str, str]]:
    """Seed the requested set from the single-round prefetch, scoped to one dataset."""
    requested: Set[Tuple[str, str]] = set()
    for r in _read_jsonl(missing_requests_path):
        if r["request_type"] == "historical_property" and r["dataset"] == dataset:
            requested.add((str(r["entity_qid"]), str(r["pid"])))
    return requested


def _cache_qids(path: Path) -> Set[str]:
    return {str(r.get("qid") or "") for r in _read_jsonl(path) if r.get("qid")}


def _sort_cache(path: Path) -> None:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: str(r.get("qid") or r.get("key") or ""))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _run_executor(round_dir: Path, historical_cache: Path, args) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    store = bool(args.versioned_alias_store) and str(args.versioned_alias_store) != "none"
    backend = "store-first + historical-fallback" if store else "historical-only"
    cmd = [
        sys.executable, str(DEFAULT_ROOT / "scripts/pilot/build_automatic_proofkg_from_plans.py"),
        "--plans", str(args.plans),
        "--protocol", str(args.protocol),
        "--entity_index", str(args.entity_index),
        "--entity_cache", str(args.entity_cache),
        "--title_cache", str(args.title_cache),
        "--property_cache", str(round_dir / "_property_cache.jsonl"),
        "--historical_property_cache", str(historical_cache),
        "--output_dir", str(round_dir / "runtime"),
        "--experiment_id", f"{args.experiment_id}-{round_dir.name.upper()}",
        "--scope", f"{args.dataset} closure round ({backend}, offline)",
    ]
    if store:
        cmd += ["--versioned_alias_store", str(args.versioned_alias_store)]
    if args.exact_entity_cache_only:
        cmd += ["--exact_entity_cache_only"]
    subprocess.run(cmd, check=True, cwd=str(DEFAULT_ROOT))


def _extract_new_property_requests(
    runtime_details: Path, requested: Set[Tuple[str, str]], dataset: str
) -> List[Dict[str, Any]]:
    new: List[Dict[str, Any]] = []
    for row in _read_jsonl(runtime_details):
        qid = str(row.get("qid") or "")
        for hop in (row.get("execution") or {}).get("hops") or []:
            ins = hop.get("input_entities") or []
            resolved = [e for e in ins if e.get("qid") and not e.get("abstained")]
            if not resolved or hop.get("matches"):
                continue
            entity_qid = str(resolved[0]["qid"])
            for pid in hop.get("pids") or []:
                key = (entity_qid, str(pid))
                if key in requested:
                    continue
                requested.add(key)
                new.append({
                    "dataset": dataset,
                    "qid": qid,
                    "entity_qid": entity_qid,
                    "pid": str(pid),
                    "hop": int(hop.get("hop_index") or 0),
                    "request_type": "historical_property",
                })
    return new


def _prefetch(new_requests: List[Dict[str, Any]], closure_cache: Path, args) -> Dict[str, int]:
    before = _cache_qids(closure_cache)
    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=closure_cache, cutoff=args.cutoff, offline=False,
        timeout=args.timeout, request_delay=args.delay, max_retries=args.retries,
    )
    # one job per distinct QID (the retriever fetches the whole entity revision).
    jobs: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for req in new_requests:
        if req["entity_qid"] not in seen:
            seen.add(req["entity_qid"])
            jobs.append(req)

    def fetch_one(req):
        edges = retriever.fetch_edges(req["entity_qid"], [req["pid"]])
        return bool(edges)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        positives = list(pool.map(fetch_one, jobs))
    _sort_cache(closure_cache)

    outcomes = {"positive": sum(1 for p in positives if p), "no_edges_at_cutoff": sum(1 for p in positives if not p)}
    after = _cache_qids(closure_cache)
    new_qids = after - before
    return {**outcomes, "fetched_qids": sorted(new_qids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTO)
    parser.add_argument("--entity_index", type=Path, default=DEFAULT_ENTITY_INDEX)
    parser.add_argument("--entity_cache", type=Path, default=DEFAULT_ENTITY_CACHE)
    parser.add_argument("--title_cache", type=Path, default=DEFAULT_TITLE_CACHE)
    parser.add_argument(
        "--exact_entity_cache_only",
        action="store_true",
        help="Pass fail-closed exact-cache entity linking to every executor round.",
    )
    parser.add_argument("--base_historical_cache", type=Path, default=DEFAULT_BASE_HIST)
    parser.add_argument("--dataset", default="2wikimultihopqa",
                        help="Dataset tag for requests / experiment id (default 2wikimultihopqa).")
    parser.add_argument("--versioned_alias_store", type=str, default=str(DEFAULT_VERSIONED_ALIAS),
                        help="Versioned store for store-first combined retriever; pass 'none' for historical-only (e.g. HotpotQA).")
    parser.add_argument("--seed_requests", default=None,
                        help="Optional missing-requests file to seed the requested set (default: empty).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--experiment_id",
        default=None,
        help="Stable experiment id prefix; defaults to the output directory name.",
    )
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.experiment_id is None:
        args.experiment_id = args.out.name

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {args.out}")
    args.out.mkdir(parents=True)

    # Dataset-scoped seed: only 2Wiki property requests seed the requested set
    # (the single-round prefetch also contained 7 musique requests, which must
    # not contaminate 2Wiki's closure bookkeeping).  For a fresh cohort (e.g.
    # train-only), pass --seed_requests "" to start from an empty requested set.
    if args.seed_requests:
        requested = _seed_requested(Path(args.seed_requests), args.dataset)
    else:
        requested = set()
    seed_count = len(requested)

    # Round 0 uses the base v2 cache untouched.
    round0 = args.out / "round_0"
    _run_executor(round0, args.base_historical_cache, args)
    logger.info("round_0 done (base cache)")

    closure_cache = args.out / "closure_historical_property_cache.jsonl"
    closure_cache.write_bytes(args.base_historical_cache.read_bytes())
    _sort_cache(closure_cache)

    rounds: List[Dict[str, Any]] = []
    last_materialized_round = 0
    convergence_check_round: int | None = None
    stop_reason = "max_rounds"

    for r in range(1, args.max_rounds + 1):
        prev_runtime = args.out / f"round_{r-1}" / "runtime" / "runtime_details.jsonl"
        new_requests = _extract_new_property_requests(prev_runtime, requested, args.dataset)
        if not new_requests:
            convergence_check_round = r
            stop_reason = "no_new_requests"
            rounds.append({"round": r, "n_new": 0, "stop_reason": stop_reason, "note": "convergence check only, no runtime materialized"})
            break

        outcomes = _prefetch(new_requests, closure_cache, args)

        # Cache increment = the QIDs added this round (sorted, deterministic).
        new_qids = set(outcomes["fetched_qids"])
        inc_rows = sorted(
            (x for x in _read_jsonl(closure_cache) if str(x.get("qid") or "") in new_qids),
            key=lambda r: str(r.get("qid") or ""),
        )
        inc_path = args.out / f"round_{r}_cache_increment.jsonl"
        _write_jsonl(inc_path, inc_rows)
        req_path = args.out / f"round_{r}_requests.jsonl"
        _write_jsonl(req_path, new_requests)

        _run_executor(args.out / f"round_{r}", closure_cache, args)
        last_materialized_round = r
        stop_reason = "max_rounds"
        rounds.append({
            "round": r, "n_new": len(new_requests),
            "outcomes": {k: outcomes[k] for k in ("positive", "no_edges_at_cutoff")},
            "fetched_qids": outcomes["fetched_qids"],
            "cache_increment_sha256": _sha256(inc_path),
            "requests_sha256": _sha256(req_path),
            "stop_reason": "continued",
        })
        logger.info("round_%d: %d new requests, %s", r, len(new_requests), outcomes)

    report = {
        "schema_version": "inference-proofkg-closure-v3b-1",
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "max_rounds": args.max_rounds,
        "cutoff": args.cutoff,
        "policy": {"workers": args.workers, "delay": args.delay, "timeout": args.timeout, "retries": args.retries},
        "seed_requests_2wiki": seed_count,
        "new_closure_requests_total": sum(rd.get("n_new", 0) for rd in rounds),
        "requested_total_2wiki": len(requested),
        "last_materialized_round": last_materialized_round,
        "convergence_check_round": convergence_check_round,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "closure_cache_sha256": _sha256(closure_cache),
        "exact_entity_cache_only": bool(args.exact_entity_cache_only),
    }
    (args.out / "closure_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(args.out, extra={
        "experiment_id": args.experiment_id,
        "phase": "iterative_prefetch_closure",
        "last_materialized_round": last_materialized_round,
        "convergence_check_round": convergence_check_round,
        "stop_reason": stop_reason,
        "requested_total_2wiki": len(requested),
    }, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
