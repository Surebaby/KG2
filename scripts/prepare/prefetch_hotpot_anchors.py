#!/usr/bin/env python
"""Targeted HotpotQA anchor prefetch (title/QID resolution only, no properties).

Resolves the 33 frozen missing anchor surfaces online via the Wikipedia title
resolver, writing into a NEW isolated title cache.  No gold, no broad
neighbourhood, no property fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--title_cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    requests = _read_jsonl(Path(args.requests))
    # dedup by surface
    seen = set()
    jobs = []
    for r in requests:
        s = str(r["anchor_surface"]).strip()
        if s and s not in seen:
            seen.add(s)
            jobs.append(r)

    resolver = WikipediaTitleResolver(
        cache_path=args.title_cache, offline=False,
        timeout=args.timeout, max_retries=args.retries, request_delay=args.delay,
    )
    log_rows: List[Dict[str, Any]] = []

    def resolve_one(r):
        surface = str(r["anchor_surface"]).strip()
        res = resolver.resolve(surface)
        outcome = "positive" if (res.selected_qid and not res.abstained) else ("abstain" if res.abstained else "fail")
        return {"qid_req": r["qid"], "surface": surface, "resolved_qid": res.selected_qid,
                "outcome": outcome, "abstain_reason": res.abstain_reason if res.abstained else ""}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(resolve_one, jobs):
            log_rows.append(row)
            print(f"resolved {len(log_rows)}/{len(jobs)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    log_path = out / "anchor_prefetch_log.jsonl"
    log_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log_rows), encoding="utf-8")

    counts = {"positive": 0, "abstain": 0, "fail": 0}
    for x in log_rows:
        counts[x["outcome"]] = counts.get(x["outcome"], 0) + 1

    report = {
        "schema_version": "hotpot-anchor-prefetch-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANCHOR_PREFETCH_COMPLETE",
        "counts": {"unique_surfaces": len(jobs), **counts},
        "policy": {"workers": args.workers, "delay": args.delay, "timeout": args.timeout, "retries": args.retries},
        "title_cache_sha256": _sha256(Path(args.title_cache)),
        "requests_sha256": _sha256(Path(args.requests)),
        "log_sha256": _sha256(log_path),
    }
    (out / "anchor_prefetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from kgproweight.utils.logging import dump_manifest
    dump_manifest(out, extra={"phase": "prefetch_hotpot_anchors", **report}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
