#!/usr/bin/env python
"""Freeze a stratified, disjoint 2Wiki train confirmation cohort and gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


DEFAULT_QUOTAS = {
    "compositional": 38,
    "comparison": 38,
    "bridge_comparison": 37,
    "inference": 37,
}
PREREGISTERED_GATES = {
    "plan_recognized_rate_min": 0.90,
    "reference_relation_recall_min": 0.90,
    "expected_explicit_anchor_in_plan_recall_min": 0.90,
    "expected_explicit_anchor_linked_from_plan_recall_min": 0.85,
    "complete_plan_execution_rate_min": 0.80,
    "full_relation_value_chain_rate_evaluable_min": 0.70,
    "per_stratum_plan_recognized_rate_min": 0.80,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def exclusion_ids(paths: Sequence[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            for key in ("source_id", "qid", "id"):
                if row.get(key) is not None:
                    result.add(str(row[key]))
                    break
    return result


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    excluded: set[str],
    seed: int,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> List[Dict[str, Any]]:
    pools: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("id") or "")
        stratum = str((row.get("metadata") or {}).get("type") or "")
        if qid and qid not in excluded and stratum in quotas:
            pools[stratum].append(row)
    selected: List[Dict[str, Any]] = []
    for offset, (stratum, quota) in enumerate(quotas.items()):
        candidates = sorted(pools[stratum], key=lambda row: str(row["id"]))
        rng = random.Random(seed + offset * 1009)
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"not enough rows for {stratum}: {len(candidates)} < {quota}")
        for row in candidates[:quota]:
            selected.append(
                {
                    "dataset": "2wikimultihopqa",
                    "source_id": str(row["id"]),
                    "source_split": "train",
                    "question": str(row["question"]),
                    "gold_answers": list(row.get("golden_answers") or []),
                    "stratum": stratum,
                    "selection_seed": seed,
                }
            )
    selected.sort(key=lambda row: (row["stratum"], row["source_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/2wikimultihopqa/train.jsonl")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True)
    source = Path(args.source).resolve()
    excludes = [Path(value).resolve() for value in args.exclude]
    excluded = exclusion_ids(excludes)
    selected = select_rows(
        _read_jsonl(source), excluded=excluded, seed=args.seed, quotas=DEFAULT_QUOTAS
    )
    if len(selected) != 150 or any(row["source_id"] in excluded for row in selected):
        raise RuntimeError("confirmation cohort size/disjointness invariant failed")
    cohort_path = output_dir / "cohort.jsonl"
    with cohort_path.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "FROZEN_BEFORE_KG_BUILD",
        "scope": "2WikiMultiHopQA train-only independent structural confirmation; no model inference",
        "selection": {
            "seed": args.seed,
            "strategy": "stratified_without_replacement",
            "quotas": DEFAULT_QUOTAS,
            "counts": dict(Counter(row["stratum"] for row in selected)),
        },
        "source": {"path": str(source), "sha256": _sha256(source)},
        "exclusions": [
            {"path": str(path), "sha256": _sha256(path)} for path in excludes
        ],
        "n_excluded_ids_union": len(excluded),
        "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path), "n": len(selected)},
        "preregistered_gates": PREREGISTERED_GATES,
        "gold_boundary": "gold/evidence may be used only after gold-free KG construction for structural audit",
    }
    manifest_path = output_dir / "cohort_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir / "run", extra=manifest)
    print(json.dumps({"cohort": manifest["cohort"], "counts": manifest["selection"]["counts"], "gates": PREREGISTERED_GATES}, indent=2))


if __name__ == "__main__":
    main()
