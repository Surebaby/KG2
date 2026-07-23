#!/usr/bin/env python
"""Build question_kg_index_v2 from existing cache with 3-layer filtering.

Usage:
  python scripts/prepare/06_build_question_kg_index.py \
    --input indexes/kg_cache/question_kg_index.json \
    --output indexes/kg_cache/question_kg_index_v2.json \
    --report docs/kg_build_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from kgproweight.kg.kg_filter import (
    _pid_for_triple,
    extract_question_entities,
    filter_and_rank_triples,
    hard_delete_triple,
    make_question_id,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to v1 question_kg_index.json")
    p.add_argument("--output", required=True, help="Path for v2 output .json")
    p.add_argument("--report", default="docs/kg_build_report.md", help="Report output")
    p.add_argument("--max_keep", type=int, default=30, help="Max triples per question")
    return p.parse_args()


pid_for_triple = _pid_for_triple  # use kg_filter's version with label→PID mapping


def main():
    args = parse_args()
    t0 = time.time()

    # Load existing cache
    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} entries from {args.input}")

    # ── Stats: before ──
    all_triples_before = sum(len(e["t"]) for e in raw)
    all_relations_before = Counter()
    for entry in raw:
        for t in entry["t"]:
            all_relations_before[t[1]] += 1

    # ── Build new index ──
    v2_entries = []
    total_hard_deleted = 0
    total_quota_dropped = 0
    total_kept = 0
    per_dataset: Dict[str, dict] = defaultdict(lambda: {"count": 0, "triples_before": 0, "triples_after": 0})

    for entry in raw:
        q = entry["q"]
        triples = [tuple(t) for t in entry["t"]]
        n_before = len(triples)

        pid_map = {t: pid_for_triple(t) for t in triples}

        # Count hard deletes
        hard_del = sum(1 for t in triples if hard_delete_triple(t, pid=pid_map.get(t, "")))
        total_hard_deleted += hard_del

        # Filter and rank (rich format)
        filtered_rich = filter_and_rank_triples(
            triples, q, pid_map=pid_map, max_keep=args.max_keep, rich=True,
        )
        n_after = len(filtered_rich)
        total_quota_dropped += (n_before - hard_del - n_after)
        total_kept += n_after

        # Extract linked entities from question
        entities = extract_question_entities(q)

        # Generate question_id
        qid = make_question_id(q)

        # Build v2 entry (rich format)
        v2_entries.append({
            "question_id": qid,
            "question": q,
            "linked_entities": entities,
            "triples": filtered_rich,
            "n_before": n_before,
            "n_after": n_after,
            "builder_version": "r9v6-kg-1",
            "relation_policy_version": "rel-1",
        })
        per_dataset["all"]["count"] += 1
        per_dataset["all"]["triples_before"] += n_before
        per_dataset["all"]["triples_after"] += n_after

    # ── Stats: after ──
    all_relations_after = Counter()
    for entry in v2_entries:
        for t in entry["triples"]:
            all_relations_after[t["r"]] += 1

    # ── Write output ──
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(v2_entries, f, ensure_ascii=False)
    print(f"Wrote {len(v2_entries)} entries to {args.output}")

    # ── Generate report ──
    report_lines = [
        "# KG Build Report — question_kg_index_v2",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Builder: r9v6-kg-1 | Relation policy: rel-1",
        "",
        "## Summary",
        f"| Metric | Before | After | Change |",
        f"|---|---|---|---|",
        f"| Entries | {len(raw)} | {len(v2_entries)} | — |",
        f"| Total triples | {all_triples_before} | {total_kept} | {all_triples_before - total_kept} removed ({(all_triples_before - total_kept)/max(1,all_triples_before)*100:.1f}%) |",
        f"| Hard deleted | — | {total_hard_deleted} | — |",
        f"| Quota/score dropped | — | {total_quota_dropped} | — |",
        f"| Avg triples/question | {all_triples_before/max(1,len(raw)):.1f} | {total_kept/max(1,len(v2_entries)):.1f} | — |",
        "",
        "## Top 20 Relations: Before",
        "| Relation | Count |",
        "|---|---|",
    ]
    for rel, count in all_relations_before.most_common(20):
        report_lines.append(f"| {rel} | {count} |")

    report_lines += [
        "",
        "## Top 20 Relations: After",
        "| Relation | Count |",
        "|---|---|",
    ]
    for rel, count in all_relations_after.most_common(20):
        report_lines.append(f"| {rel} | {count} |")

    # Taxonomic ratio — only count instance_of + subclass_of
    taxonomic = ["instance of", "subclass of"]
    tax_before = sum(all_relations_before.get(r, 0) for r in taxonomic)
    tax_after = sum(all_relations_after.get(r, 0) for r in taxonomic)
    report_lines += [
        "",
        "## Taxonomic Relation Ratio",
        f"| Metric | Before | After |",
        f"|---|---|---|",
        f"| instance_of + subclass_of + has part(s) + part of | {tax_before/all_triples_before*100:.1f}% | {tax_after/max(1,total_kept)*100:.1f}% |",
        f"| Target | — | < 25% |",
    ]

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    elapsed = time.time() - t0
    print(f"Report → {args.report}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Taxonomic ratio: {tax_before/all_triples_before*100:.1f}% → {tax_after/max(1,total_kept)*100:.1f}%")


if __name__ == "__main__":
    main()
