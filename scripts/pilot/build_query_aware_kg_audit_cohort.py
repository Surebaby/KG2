#!/usr/bin/env python
"""Freeze a stratified, train-only cohort for query-aware KG coverage audit.

The cohort contains 50 questions from each of HotpotQA, 2WikiMultiHopQA and
MuSiQue.  It never reads dev/test and excludes questions used by the existing
hidden33 and hard25 diagnostics.  Selection is deterministic and occurs before
any KG coverage result is computed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASET_QUOTAS = {
    "hotpotqa": {"comparison": 25, "bridge": 25},
    "2wikimultihopqa": {
        "compositional": 13,
        "comparison": 13,
        "bridge_comparison": 12,
        "inference": 12,
    },
    "musique": {"2hop": 20, "3hop": 20, "4hop": 10},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rank(seed: int, dataset: str, qid: str) -> str:
    return hashlib.sha256(f"{seed}\0{dataset}\0{qid}".encode()).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _stratum(dataset: str, row: Dict[str, Any]) -> str:
    if dataset == "musique":
        hops = len(
            row.get("metadata", {}).get("metadata", {}).get("question_decomposition", [])
        )
        return f"{hops}hop"
    return str(row.get("metadata", {}).get("type") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    for target in (output, manifest):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")

    excluded_questions: set[str] = set()
    exclusion_sources: List[Dict[str, str]] = []
    for raw in args.exclude:
        path = Path(raw).resolve()
        for row in _iter_jsonl(path):
            question = str(row.get("question") or "").strip()
            if question:
                excluded_questions.add(question)
        exclusion_sources.append({"path": str(path), "sha256": _sha256(path)})

    data_root = Path(args.data_root).resolve()
    source_records: List[Dict[str, str]] = []
    selected: List[Dict[str, Any]] = []
    availability: Dict[str, Counter[str]] = {}
    for dataset, quotas in DATASET_QUOTAS.items():
        source = data_root / dataset / "train.jsonl"
        candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in _iter_jsonl(source):
            question = str(row.get("question") or "").strip()
            qid = str(row.get("id") or "").strip()
            if not qid or not question or question in excluded_questions:
                continue
            candidates[_stratum(dataset, row)].append(row)
        availability[dataset] = Counter({key: len(value) for key, value in candidates.items()})
        for stratum, quota in quotas.items():
            rows = candidates.get(stratum, [])
            if len(rows) < quota:
                raise SystemExit(
                    f"insufficient {dataset}/{stratum}: {len(rows)} available, {quota} required"
                )
            rows.sort(key=lambda row: _stable_rank(args.seed, dataset, str(row["id"])))
            for row in rows[:quota]:
                selected.append(
                    {
                        "dataset": dataset,
                        "source_split": "train",
                        "source_id": str(row["id"]),
                        "question": str(row["question"]).strip(),
                        "gold_answers": [str(value) for value in row.get("golden_answers") or []],
                        "stratum": stratum,
                        "selection_seed": args.seed,
                    }
                )
        source_records.append({"path": str(source), "sha256": _sha256(source)})

    selected.sort(key=lambda row: (row["dataset"], row["stratum"], row["source_id"]))
    if len(selected) != 150 or len({(r["dataset"], r["source_id"]) for r in selected}) != 150:
        raise SystemExit("cohort must contain exactly 150 unique dataset/id pairs")
    if any(row["question"] in excluded_questions for row in selected):
        raise SystemExit("excluded diagnostic question leaked into cohort")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    cohort_sha = _sha256(output)
    counts = Counter((row["dataset"], row["stratum"]) for row in selected)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FROZEN_BEFORE_KG_AUDIT",
        "scope": "train-only; 50 questions per dataset; no dev/test",
        "selection": {
            "seed": args.seed,
            "method": "sha256(seed,dataset,source_id) rank within fixed strata",
            "quotas": DATASET_QUOTAS,
            "counts": {f"{ds}/{stratum}": count for (ds, stratum), count in sorted(counts.items())},
            "excluded_unique_questions": len(excluded_questions),
        },
        "sources": source_records,
        "exclusions": exclusion_sources,
        "availability": {dataset: dict(counts) for dataset, counts in availability.items()},
        "cohort": {"path": str(output), "sha256": cohort_sha, "rows": len(selected)},
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cohort": payload["cohort"], "counts": payload["selection"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
