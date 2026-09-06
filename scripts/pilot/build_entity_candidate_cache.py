#!/usr/bin/env python
"""Build a versioned Wikidata candidate index for unresolved pilot mentions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List

import requests

from kgproweight.kg.entity_linker import (
    DEFAULT_PROXY_HEADERS,
    WIKIDATA_SEARCH_URL,
    WIKIDATA_USER_AGENT,
)
from kgproweight.utils.logging import dump_manifest


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_rows", required=True)
    parser.add_argument("--base_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    replay_path = Path(args.replay_rows).resolve()
    base_index_path = Path(args.base_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    partial_path = output_dir / "candidate_queries.partial.jsonl"
    final_path = output_dir / "candidate_queries.jsonl"
    expanded_path = output_dir / "entity_desc_index.expanded.json"

    replay_rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mentions = sorted(
        {
            mention
            for row in replay_rows
            for mention, reason in (row.get("link_failures") or {}).items()
            if str(reason).startswith("no candidates")
        },
        key=lambda value: (_clean(value), value),
    )
    write_lock = threading.Lock()

    def query(mention: str) -> Dict[str, Any]:
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            started = time.monotonic()
            try:
                response = requests.get(
                    WIKIDATA_SEARCH_URL,
                    params={
                        "action": "wbsearchentities",
                        "search": mention,
                        "language": "en",
                        "format": "json",
                        "limit": 10,
                        "props": "",
                    },
                    headers={"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS},
                    timeout=args.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                candidates = [
                    {
                        "qid": item["id"],
                        "label": item.get("label", mention),
                        "description": item.get("description", ""),
                    }
                    for item in payload.get("search") or []
                    if item.get("id")
                ]
                return {
                    "mention": mention,
                    "key": _clean(mention),
                    "status": "ok",
                    "attempts": attempt,
                    "latency_seconds": time.monotonic() - started,
                    "candidates": candidates,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(attempt)
        return {
            "mention": mention,
            "key": _clean(mention),
            "status": "error",
            "attempts": args.max_retries,
            "error": last_error,
            "candidates": [],
        }

    results: Dict[str, Dict[str, Any]] = {}
    with partial_path.open("w", encoding="utf-8") as partial, ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {executor.submit(query, mention): mention for mention in mentions}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result["mention"]] = result
            with write_lock:
                partial.write(json.dumps(result, ensure_ascii=False) + "\n")
                partial.flush()
            if completed % 25 == 0:
                print(f"candidate cache progress: {completed}/{len(mentions)}", flush=True)

    ordered = [results[mention] for mention in mentions]
    with final_path.open("w", encoding="utf-8") as fh:
        for result in ordered:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    expanded: Dict[str, List[Dict[str, Any]]] = json.loads(
        base_index_path.read_text(encoding="utf-8")
    )
    added_keys = added_candidates = 0
    for result in ordered:
        if result["status"] != "ok" or not result["candidates"]:
            continue
        key = result["key"]
        existing = expanded.setdefault(key, [])
        if not existing:
            added_keys += 1
        seen = {str(item.get("qid")) for item in existing}
        for candidate in result["candidates"]:
            if candidate["qid"] not in seen:
                existing.append(candidate)
                seen.add(candidate["qid"])
                added_candidates += 1
    expanded_path.write_text(
        json.dumps(expanded, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    ok = sum(result["status"] == "ok" for result in ordered)
    with_candidates = sum(bool(result["candidates"]) for result in ordered)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "replay_rows": str(replay_path),
            "replay_rows_md5": _md5(replay_path),
            "base_index": str(base_index_path),
            "base_index_md5": _md5(base_index_path),
        },
        "protocol": {
            "endpoint": WIKIDATA_SEARCH_URL,
            "language": "en",
            "limit": 10,
            "gold_used": False,
            "max_workers": args.max_workers,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
        },
        "accounting": {
            "unique_mentions": len(mentions),
            "successful_requests": ok,
            "failed_requests": len(mentions) - ok,
            "mentions_with_candidates": with_candidates,
            "mentions_without_candidates": len(mentions) - with_candidates,
            "added_index_keys": added_keys,
            "added_candidates": added_candidates,
        },
        "outputs": {
            "raw_completion_order": str(partial_path),
            "ordered_queries": str(final_path),
            "expanded_index": str(expanded_path),
            "expanded_index_md5": _md5(expanded_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir / "run",
        extra={
            "experiment": "wikidata_candidate_index_expansion",
            "report": str(report_path),
            "expanded_index": str(expanded_path),
            "gold_used": False,
            "unique_mentions": len(mentions),
            "successful_requests": ok,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if ok != len(mentions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
