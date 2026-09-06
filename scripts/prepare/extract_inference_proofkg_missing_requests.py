#!/usr/bin/env python
"""Extract machine-executable missing-request lists from the offline diagnostics.

Reads the OFFLINE_COVERAGE_DIAGNOSTIC ``runtime_details.jsonl`` for 2Wiki and
MuSiQue and emits two request kinds for the targeted prefetch (step 6C):

- ``title_or_qid_resolution`` — an anchor surface that abstained (no QID) in the
  offline pass;
- ``historical_property`` — a resolved ``(entity_qid, pid)`` whose property value
  was absent from the offline cache.

Requests are deduplicated, traceable to dataset/qid/hop, and never request
broad-neighbourhood lookups.  No Gold field is read or emitted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "audits" / "inference_proofkg_v1_pilot30_offline_diag"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "audits"
    / "inference_proofkg_v1_pilot30_missing_requests"
)

DATASETS = ("2wikimultihopqa", "musique")

FORBIDDEN = {"golden_answers", "answer", "answers", "supporting_facts", "target", "evidence"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(dataset: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("qid") or "")
        execution = row.get("execution") or {}

        # 1) title/QID resolution: anchors that abstained or produced no QID.
        for surface, anchor in (execution.get("anchor_entities") or {}).items():
            if anchor.get("abstained") or not anchor.get("qid"):
                requests.append({
                    "dataset": dataset,
                    "qid": qid,
                    "anchor_surface": str(surface),
                    "request_type": "title_or_qid_resolution",
                    "trace": {"hop": None, "reason": anchor.get("abstain_reason") or "missing_qid"},
                })

        # 2) historical property: a hop with a resolved input QID but no match.
        for hop in execution.get("hops") or []:
            ins = hop.get("input_entities") or []
            resolved = [e for e in ins if e.get("qid") and not e.get("abstained")]
            if resolved and not hop.get("matches"):
                entity_qid = resolved[0]["qid"]
                for pid in hop.get("pids") or []:
                    requests.append({
                        "dataset": dataset,
                        "qid": qid,
                        "entity_qid": str(entity_qid),
                        "pid": str(pid),
                        "hop": int(hop.get("hop_index") or 0),
                        "request_type": "historical_property",
                        "trace": {"hop": int(hop.get("hop_index") or 0), "reason": "property_value_missing_offline"},
                    })
    return requests


def _dedupe(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in requests:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {args.out}")
    args.out.mkdir(parents=True)

    all_requests: List[Dict[str, Any]] = []
    source_hashes: Dict[str, str] = {}
    for ds in DATASETS:
        details_path = args.root / ds / "runtime" / "runtime_details.jsonl"
        if not details_path.is_file():
            raise FileNotFoundError(f"missing diagnostic runtime_details: {details_path}")
        rows = _read_jsonl(details_path)
        # No gold fields may appear anywhere in the request input.
        for row in rows:
            if FORBIDDEN.intersection(str(k) for k in row):
                raise ValueError(f"forbidden field in {ds} diagnostic row")
        all_requests.extend(extract(ds, rows))
        source_hashes[ds] = _sha256(details_path)

    deduped = _dedupe(all_requests)
    by_type: Dict[str, int] = {}
    for r in deduped:
        by_type[r["request_type"]] = by_type.get(r["request_type"], 0) + 1

    out_path = args.out / "missing_requests.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in deduped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MISSING_REQUESTS_FROZEN",
        "scope": "targeted prefetch inputs derived from offline coverage diagnostics",
        "counts": {"total": len(deduped), "by_type": by_type},
        "sources": {
            ds: {"runtime_details": str(args.root / ds / "runtime" / "runtime_details.jsonl"), "sha256": h}
            for ds, h in source_hashes.items()
        },
        "forbidden_fields": 0,
        "no_broad_neighbourhood": True,
        "requests_file": str(out_path),
        "requests_sha256": _sha256(out_path),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(
        args.out,
        extra={
            "experiment_id": args.out.name,
            "phase": "extract_missing_requests",
            "status": report["status"],
            "counts": report["counts"],
        },
        status="COMPLETE",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
