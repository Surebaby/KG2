#!/usr/bin/env python
"""Post-failure near-exact diagnostic for SAEG P-alignment v2.

The frozen exact-match data gate remains failed.  This diagnostic estimates
how many exact failures are merely Wiki text/format drift by using a strict,
fixed title-equal token-F1 matcher.  It must not relabel or resample training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.utils.logging import dump_manifest


THRESHOLD = 0.90
EXPERIMENT_ID = "SAEG-P-ALIGNMENT-V2-NEAR-EXACT-DIAGNOSTIC-V1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def canonical_sentence_tokens(title: object, sentence: object) -> list[str]:
    title_tokens = tokens(title)
    sentence_tokens = tokens(sentence)
    # Runtime Wiki passages sometimes repeat the article title at the start of
    # the first sentence. Remove repeated exact title prefixes only.
    while title_tokens and sentence_tokens[: len(title_tokens)] == title_tokens:
        sentence_tokens = sentence_tokens[len(title_tokens):]
    return sentence_tokens


def token_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_count, right_count = Counter(left), Counter(right)
    overlap = sum((left_count & right_count).values())
    return 2.0 * overlap / max(1, sum(left_count.values()) + sum(right_count.values()))


def near_exact_match(selected: Mapping[str, Any], required: Mapping[str, Any]) -> float:
    if tokens(selected.get("title")) != tokens(required.get("title")):
        return 0.0
    return token_f1(
        canonical_sentence_tokens(selected.get("title"), selected.get("sentence")),
        canonical_sentence_tokens(required.get("title"), required.get("sentence")),
    )


def classify(row: Mapping[str, Any]) -> tuple[str, set[int], list[dict[str, Any]]]:
    required = list(row.get("required_support_units") or [])
    selected = list(row.get("selected_edges") or [])
    matched: set[int] = set()
    edge_diagnostics = []
    for edge in selected:
        scored = [(near_exact_match(edge, unit), index) for index, unit in enumerate(required)]
        best_score, best_index = max(scored, default=(0.0, -1))
        is_match = best_score >= THRESHOLD
        if is_match:
            matched.add(best_index)
        edge_diagnostics.append({
            "passage_id": edge["passage_id"],
            "best_required_index": best_index if is_match else None,
            "best_token_f1": best_score,
            "near_exact_match": is_match,
        })
    if not selected:
        quality = "empty"
    elif not matched:
        quality = "misleading"
    elif len(matched) == len(required):
        quality = "complete"
    else:
        quality = "partial"
    return quality, matched, edge_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, default=Path(
        "data/silver_data/saeg_p_alignment_v2_train1781_candidates_seed42/"
        "evidence_quality.train_gold_only.jsonl"))
    parser.add_argument("--exact_report", type=Path, default=Path(
        "data/silver_data/saeg_p_alignment_v2_train1781_candidates_seed42/report.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/audits/saeg_p_alignment_v2_near_exact_diagnostic_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite diagnostic: {args.out}")
    exact_report = json.loads(args.exact_report.read_text(encoding="utf-8"))
    if exact_report.get("status") != "FAIL_STOP_DATA_GATES":
        raise ValueError("near-exact diagnostic is only valid after the frozen exact gate fails")
    rows = read_jsonl(args.quality)
    details = []
    counters: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    for row in rows:
        quality, matched, edge_diagnostics = classify(row)
        dataset = str(row["dataset"])
        selected = len(row.get("selected_edges") or [])
        required = len(row.get("required_support_units") or [])
        matched_edges = sum(bool(value["near_exact_match"]) for value in edge_diagnostics)
        counters[f"{dataset}::{quality}"] += 1
        sums[f"{dataset}::selected"] += selected
        sums[f"{dataset}::matched_selected"] += matched_edges
        sums[f"{dataset}::required"] += required
        sums[f"{dataset}::matched_required"] += len(matched)
        details.append({
            "question_key": row["question_key"],
            "dataset": dataset,
            "qid": row["qid"],
            "exact_quality_class": row["quality_class"],
            "near_exact_quality_class": quality,
            "matched_required_indices": sorted(matched),
            "selected_edge_diagnostics": edge_diagnostics,
        })
    per_dataset = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        classes = {
            name: counters[f"{dataset}::{name}"]
            for name in ("complete", "partial", "misleading", "empty")
        }
        per_dataset[dataset] = {
            "rows": sum(classes.values()),
            "classes": classes,
            "selected_edge_near_exact_precision": sums[f"{dataset}::matched_selected"] / max(
                1, sums[f"{dataset}::selected"]
            ),
            "required_unit_near_exact_recall": sums[f"{dataset}::matched_required"] / max(
                1, sums[f"{dataset}::required"]
            ),
        }
    args.out.mkdir(parents=True, exist_ok=False)
    details_path = args.out / "details.jsonl"
    write_jsonl(details_path, details)
    report = {
        "schema_version": "saeg-p-alignment-near-exact-diagnostic-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_EXACT_DATA_GATE_REMAINS_FAILED",
        "matcher": {
            "title": "exact normalized token sequence",
            "sentence": "strip repeated exact leading title then multiset token-F1",
            "threshold": THRESHOLD,
            "semantic_entailment_proven": False,
        },
        "per_dataset": per_dataset,
        "inputs": {
            "quality": {"path": str(args.quality), "sha256": sha256_file(args.quality)},
            "exact_report": {"path": str(args.exact_report), "sha256": sha256_file(args.exact_report)},
        },
        "output": {"details": {"path": str(details_path), "sha256": sha256_file(details_path)}},
        "scientific_boundary": (
            "Post-failure development diagnostic only. It detects near-verbatim Wiki text drift, not general "
            "semantic entailment. It does not overwrite exact labels, reverse the failed gate, create a train "
            "schedule, or authorize model updates."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "saeg_p_alignment_v2_near_exact_diagnostic", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
