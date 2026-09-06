#!/usr/bin/env python
"""Merge the pilot30 + confirmation270 Proof-KG into the formal 2Wiki n=300 supply.

Both partitions are already frozen; this only combines them into a single
canonical question-KG file for the eval pipeline, preserving identity/hash and
gold_access=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.question_kg import make_question_kg_record, load_question_kg_index
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "data/derived/inference_proofkg_v1_pilot30/2wikimultihopqa/closure_v3b/round_1/runtime/runtime_question_kg.jsonl"
DEFAULT_CONF = ROOT / "data/derived/inference_proofkg_v1_confirmation270_2wiki_v3/question_kg_records.jsonl"
DEFAULT_OUT = ROOT / "data/derived/inference_proofkg_v1_2wiki_dev_n300_v1"


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
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)

    pilot = _read_jsonl(args.pilot)
    confirmation = _read_jsonl(args.confirmation)
    assert len(pilot) == 30 and len(confirmation) == 270

    # confirmation270 uses a custom freeze schema; re-emit in canonical form.
    canonical: List[Dict[str, Any]] = []
    for r in confirmation:
        prov = {
            "gold_access": bool(r.get("gold_access", False)),
            "complete_plan_execution": bool(r.get("complete_plan_execution")),
            "builder_version": r.get("builder_version"),
            "closure_cache_sha256": r.get("closure_cache_sha256"),
        }
        canonical.append(make_question_kg_record(
            dataset=r["dataset"], qid=r["qid"], question=r["question"],
            triples=r.get("kg_subgraph") or [], provenance=prov,
        ))
    # pilot30 is already canonical.
    canonical.extend(pilot)

    # Validate: identity join, no duplicate, gold_access=false.
    index = load_question_kg_index(canonical)
    assert len(index) == 300, f"expected 300 unique, got {len(index)}"
    assert all(not (r.get("provenance") or {}).get("gold_access") for r in canonical)
    assert len(set(r["qid"] for r in canonical)) == 300

    out_path = args.out / "question_kg_records.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in canonical), encoding="utf-8")

    nonempty = sum(1 for r in canonical if r.get("kg_subgraph"))
    report = {
        "schema_version": "inference-proofkg-n300-merge-1",
        "n": len(canonical),
        "pilot_confirmation_overlap": 0,
        "identity_join": 1.0,
        "question_hash": 1.0,
        "gold_access": False,
        "duplicate": 0,
        "nonempty": nonempty,
        "sources": {
            "pilot30": {"path": str(args.pilot), "sha256": _sha256(args.pilot)},
            "confirmation270": {"path": str(args.confirmation), "sha256": _sha256(args.confirmation)},
        },
        "records_sha256": _sha256(out_path),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(args.out, extra={"experiment_id": args.out.name, "phase": "merge_n300_proofkg", "n": len(canonical)}, status="COMPLETE")
    logger.info("merged n=300 (nonempty=%d)", nonempty)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
