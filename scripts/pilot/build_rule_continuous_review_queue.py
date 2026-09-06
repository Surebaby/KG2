#!/usr/bin/env python
"""Build blinded human-review queues for rule-continuous positive steps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SourceSpec = Tuple[str, Path, Path]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalise_triple(value: Sequence[Any]) -> Tuple[str, str, str]:
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _matching_passage_excerpts(
    passages: Iterable[Any], cited_triples: Sequence[Tuple[str, str, str]]
) -> List[Dict[str, Any]]:
    needles = {
        surface.casefold()
        for triple in cited_triples
        for surface in (triple[0], triple[2])
        if surface.strip()
    }
    ranked: List[Tuple[int, int, Dict[str, Any]]] = []
    for position, passage in enumerate(passages):
        if isinstance(passage, dict):
            contents = str(passage.get("contents") or passage.get("text") or "")
            passage_id = passage.get("id")
        else:
            contents = str(passage)
            passage_id = None
        folded = contents.casefold()
        hits = sum(needle in folded for needle in needles)
        if not hits:
            continue
        excerpt = " ".join(contents.split())[:1200]
        ranked.append((hits, -position, {"passage_id": passage_id, "excerpt": excerpt}))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:3]]


def _review_id(teacher: str, qid: str, step_position: int) -> str:
    raw = f"{teacher}|{qid}|{step_position}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _split_for_qid(qid: str) -> str:
    # QID-level split prevents the same question leaking across fit/eval Teachers.
    bucket = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "fit" if bucket < 7 else "heldout_eval"


def _human_fields(cited_triples: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    return {
        "triple_ratings": [
            {
                "triple": list(triple),
                "factual_accuracy_0_2": None,
                "step_support_0_4": None,
                "question_relevance_0_4": None,
                "notes": None,
            }
            for triple in cited_triples
        ],
        "step_overall_utility_0_4": None,
        "explicit_abstention": None,
        "misleading_or_contradictory": None,
        "reviewer_confidence_0_2": None,
        "notes": None,
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("TEACHER", "CANDIDATES", "COMPONENTS"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected", type=int, default=54)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sources: List[SourceSpec] = [
        (teacher, Path(candidate).resolve(), Path(components).resolve())
        for teacher, candidate, components in args.source
    ]

    blind_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    source_manifest: List[Dict[str, Any]] = []
    seen_ids = set()

    for teacher, candidate_path, component_path in sources:
        candidates = _read_jsonl(candidate_path)
        components = _read_jsonl(component_path)
        row_by_qid = {str(row["qid"]): row for row in candidates}
        selected = [row for row in components if float(row.get("old_label", 0.0)) == 1.0]
        source_manifest.append({
            "teacher": teacher,
            "candidate_path": str(candidate_path),
            "candidate_md5": _md5(candidate_path),
            "component_path": str(component_path),
            "component_md5": _md5(component_path),
            "selected_old_positive_steps": len(selected),
        })

        for component in selected:
            qid = str(component["qid"])
            step_position = int(component["step_position"])
            source_row = row_by_qid[qid]
            source_steps = source_row.get("steps") or []
            if step_position >= len(source_steps):
                raise SystemExit(f"step_position out of range: {teacher} {qid} {step_position}")
            step = source_steps[step_position]
            cited_triples = [
                _normalise_triple(value)
                for value in step.get("cited_triples") or []
                if len(value) == 3
            ]
            review_id = _review_id(teacher, qid, step_position)
            if review_id in seen_ids:
                raise SystemExit(f"duplicate review_id: {review_id}")
            seen_ids.add(review_id)

            blind_rows.append({
                "review_id": review_id,
                "dataset": source_row.get("dataset"),
                "question": source_row.get("question"),
                "step_text": step.get("text"),
                "cited_triples": [list(triple) for triple in cited_triples],
                "teacher_visible_kg": source_row.get("kg_subgraph") or [],
                "matching_passage_excerpts": _matching_passage_excerpts(
                    source_row.get("retrieved_passages") or [], cited_triples
                ),
                "human": _human_fields(cited_triples),
            })
            key_rows.append({
                "review_id": review_id,
                "teacher": teacher,
                "qid": qid,
                "dataset": component.get("dataset"),
                "step_position": step_position,
                "step_index": component.get("step_index"),
                "split": _split_for_qid(qid),
                "old_label": component.get("old_label"),
                "rule_continuous_v1_1": component.get("rule_continuous_v1"),
                "rule_branch": component.get("branch"),
                "rule_components": component.get("components"),
            })

    if len(blind_rows) != args.expected:
        raise SystemExit(f"selected {len(blind_rows)} rows != expected {args.expected}")

    reviewer_a = json.loads(json.dumps(blind_rows))
    reviewer_b = json.loads(json.dumps(blind_rows))
    random.Random(args.seed).shuffle(reviewer_a)
    random.Random(args.seed + 1).shuffle(reviewer_b)
    for row in reviewer_a:
        row["reviewer_id"] = "A"
    for row in reviewer_b:
        row["reviewer_id"] = "B"

    _write_jsonl(output_dir / "reviewer_a.blind.jsonl", reviewer_a)
    _write_jsonl(output_dir / "reviewer_b.blind.jsonl", reviewer_b)
    _write_jsonl(output_dir / "review_key.do_not_open_before_review.jsonl", key_rows)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HUMAN_REVIEW_TEMPLATE_LABELS_NULL",
        "protocol": {
            "unit": "old-positive cited step",
            "teacher_and_scores_blinded": True,
            "gold_answer_included": False,
            "reviewers": 2,
            "review_order_seeds": [args.seed, args.seed + 1],
            "split": "deterministic QID-level 70/30 hash split",
        },
        "sources": source_manifest,
        "counts": {
            "review_items": len(blind_rows),
            "unique_qids": len({row["qid"] for row in key_rows}),
            "fit": sum(row["split"] == "fit" for row in key_rows),
            "heldout_eval": sum(row["split"] == "heldout_eval" for row in key_rows),
        },
        "outputs": {
            "reviewer_a": "reviewer_a.blind.jsonl",
            "reviewer_b": "reviewer_b.blind.jsonl",
            "key": "review_key.do_not_open_before_review.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
