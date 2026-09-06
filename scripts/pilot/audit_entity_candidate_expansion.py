#!/usr/bin/env python
"""Audit a candidate-index replay against a fixed-input baseline.

This is a gold-free screening audit.  Confidence thresholds select a human
review queue; they must not be interpreted as correctness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from rapidfuzz import fuzz

from kgproweight.utils.logging import dump_manifest


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_rows", required=True)
    parser.add_argument("--expanded_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_score", type=float, default=0.35)
    parser.add_argument("--min_margin", type=float, default=0.10)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_rows).resolve()
    expanded_path = Path(args.expanded_rows).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_rows = _read(baseline_path)
    expanded_rows = _read(expanded_path)
    baseline = {str(row["qid"]): row for row in baseline_rows}
    if set(baseline) != {str(row["qid"]) for row in expanded_rows}:
        raise SystemExit("baseline and expanded qid sets differ")

    assignments: List[Dict[str, Any]] = []
    for row in expanded_rows:
        old = baseline[str(row["qid"])]
        all_linked_mentions = list((row.get("link_diagnostics") or {}).keys())
        for mention, diag in row.get("link_diagnostics", {}).items():
            if old.get("linked_entities", {}).get(mention) == diag.get("qid"):
                continue
            score = float(diag.get("score") or 0.0)
            margin = float(diag.get("margin") or 0.0)
            has_graph = int(row.get("qid_raw_counts", {}).get(diag.get("qid"), 0)) > 0
            in_question = bool(diag.get("appears_in_question"))
            surface_similarity = fuzz.ratio(mention.casefold(), str(diag.get("label") or "").casefold()) / 100.0
            mention_surface = " ".join(mention.casefold().split())
            nested_in_longer = any(
                mention_surface != " ".join(other.casefold().split())
                and f" {mention_surface} " in f" {' '.join(other.casefold().split())} "
                for other in all_linked_mentions
            )
            assignments.append({
                "qid": row["qid"],
                "dataset": row.get("dataset"),
                "question": row.get("question"),
                "mention": mention,
                "selected_qid": diag.get("qid"),
                "selected_label": diag.get("label"),
                "description": diag.get("description"),
                "score": score,
                "margin": margin,
                "candidate_count": diag.get("candidate_count"),
                "appears_in_question": in_question,
                "local_raw_triples": int(row.get("qid_raw_counts", {}).get(diag.get("qid"), 0)),
                "surface_similarity": surface_similarity,
                "nested_in_longer_linked_mention": nested_in_longer,
                "screen_high_confidence": score >= args.min_score and margin >= args.min_margin,
                "human_correct_qid": None,
                "human_notes": None,
            })

    review = [
        item for item in assignments
        if item["appears_in_question"] and item["local_raw_triples"] > 0
    ]
    question_grounded = [item for item in assignments if item["appears_in_question"]]
    question_path = output_dir / "all_question_grounded_links_for_human_review.jsonl"
    with question_path.open("w", encoding="utf-8") as fh:
        for item in question_grounded:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    review_path = output_dir / "question_grounded_links_for_human_review.jsonl"
    with review_path.open("w", encoding="utf-8") as fh:
        for item in review:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    all_path = output_dir / "all_new_link_assignments.jsonl"
    with all_path.open("w", encoding="utf-8") as fh:
        for item in assignments:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def nonempty(rows: List[Dict[str, Any]], field: str) -> int:
        return sum(int(row.get(field, 0) > 0) for row in rows)

    report = {
        "status": "RESEARCHER_REVIEW_REQUIRED",
        "integrity": {
            "paired_qids": len(expanded_rows),
            "gold_used": False,
            "baseline_md5": _md5(baseline_path),
            "expanded_md5": _md5(expanded_path),
        },
        "fixed_input_question_coverage": {
            "baseline_entity_linked": nonempty(baseline_rows, "new_n_linked"),
            "expanded_entity_linked": nonempty(expanded_rows, "new_n_linked"),
            "baseline_raw_kg_nonempty": nonempty(baseline_rows, "new_n_raw_triples"),
            "expanded_raw_kg_nonempty": nonempty(expanded_rows, "new_n_raw_triples"),
            "baseline_filtered_kg_nonempty": nonempty(baseline_rows, "new_n_filtered_triples"),
            "expanded_filtered_kg_nonempty": nonempty(expanded_rows, "new_n_filtered_triples"),
        },
        "new_assignment_screen": {
            "total": len(assignments),
            "question_surface": sum(item["appears_in_question"] for item in assignments),
            "local_graph_available": sum(item["local_raw_triples"] > 0 for item in assignments),
            "question_surface_and_local_graph": len(review),
            "screen_high_confidence": sum(item["screen_high_confidence"] for item in assignments),
            "question_surface_graph_and_high_confidence": sum(
                item["screen_high_confidence"] for item in review
            ),
            "question_surface_nested_in_longer_link": sum(
                item["nested_in_longer_linked_mention"] for item in question_grounded
            ),
            "low_surface_similarity_below_0_7": sum(
                item["surface_similarity"] < 0.7 for item in assignments
            ),
            "warning": "These are screening proxies, not correctness labels.",
        },
        "promotion_decision": "DO_NOT_PROMOTE_BEFORE_HUMAN_REVIEW",
        "outputs": {
            "all_assignments": str(all_path),
            "all_question_grounded_review": str(question_path),
            "priority_existing_graph_review": str(review_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir / "run",
        extra={
            "experiment": "entity_candidate_expansion_gold_free_audit",
            "report": str(report_path),
            "status": report["status"],
            "gold_used": False,
            "new_assignments": len(assignments),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
