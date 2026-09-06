#!/usr/bin/env python
"""Lexically audit whether ProofKG answers are visible in KG/passages."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _norm(value: object) -> str:
    return re.sub(r"[^\w\s]", " ", str(value).lower()).strip()


def _hit(gold: str, values: list[object]) -> bool:
    target = _norm(gold)
    return bool(target) and any(
        target == _norm(value) or (len(target) >= 4 and target in _norm(value))
        for value in values
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir, experiment_id = prepare_new_run_dir(args.output_dir)
    rows = [
        row for row in SilverDatasetReader(args.silver).accepted()
        if bool(row.metadata.get("gold_derived", False))
    ]
    if not rows:
        raise ValueError("no gold-derived ProofKG rows found")

    totals = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    hop_hist = Counter()
    for row in rows:
        gold = str(row.metadata.get("gold_answer") or row.answer or "")
        components = [value for triple in row.kg_subgraph for value in triple]
        passages = [
            (item.get("contents") or item.get("text") or "")
            if isinstance(item, dict) else str(item)
            for item in row.retrieved_passages[:15]
        ]
        kg_hit = _hit(gold, components)
        passage_hit = _hit(gold, passages)
        qtype = str(row.metadata.get("question_type") or "UNKNOWN")
        for target in (totals, by_type[qtype]):
            target["n"] += 1
            target["kg_visible"] += kg_hit
            target["passage_visible"] += passage_hit
            target["both_visible"] += kg_hit and passage_hit
            target["neither_visible"] += not kg_hit and not passage_hit
        hop_hist[len(row.kg_subgraph)] += 1

    def render(counter: Counter) -> dict:
        n = counter["n"]
        return {
            "n": n,
            **{
                key: {"count": counter[key], "rate": counter[key] / n}
                for key in (
                    "kg_visible", "passage_visible", "both_visible", "neither_visible"
                )
            },
        }

    report = {
        "experiment_id": experiment_id,
        "status": "DIAGNOSTIC_COMPLETE",
        "metric": "normalized lexical exact-or-substring answer exposure",
        "overall": render(totals),
        "by_question_type": {
            key: render(value) for key, value in sorted(by_type.items())
        },
        "kg_hops": dict(sorted(hop_hist.items())),
        "interpretation_boundary": (
            "Lexical presence does not prove copying or reasoning; this diagnostic "
            "cannot replace hidden-answer model evaluation."
        ),
        "input": artifact_identity(args.silver),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dump_manifest(
        out_dir,
        extra={
            "experiment_id": experiment_id,
            "phase": "proofkg_answer_exposure_audit",
            "report": artifact_identity(report_path),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
