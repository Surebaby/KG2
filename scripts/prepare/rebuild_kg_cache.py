#!/usr/bin/env python3
"""Rebuild question_kg_index with stricter v3 filtering.

Reads the existing v2 cache, re-filters each question's triples through the
updated pipeline (hard-delete given_name/family_name/sex, score threshold,
reduced max_keep), and writes a v3 cache.

Passage-verified filtering is NOT applied here (requires retrieval for every
question — ~1.6 h of FAISS). It runs at inference time for cache-miss paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kgproweight.kg.kg_filter import (
    filter_and_rank_triples,
    _pid_for_triple,
)


def rebuild_cache(
    v2_path: str,
    v3_path: str,
    max_keep: int = 12,
    min_keep: int = 5,
    dry_run: bool = False,
) -> None:
    """Re-filter v2 cache → v3 cache with stricter rules."""

    with open(v2_path, encoding="utf-8") as f:
        v2_data = json.load(f)

    print(f"Loaded {len(v2_data)} entries from {v2_path}")

    v3_data: List[dict] = []
    total_before = 0
    total_after = 0
    n_removed = 0
    removed_rels: Dict[str, int] = {}

    for entry in v2_data:
        question = entry.get("question", entry.get("q", ""))
        triples_raw = entry.get("triples", entry.get("t", []))

        # Convert to (h, r, t) tuple list
        if triples_raw and isinstance(triples_raw[0], dict):
            triples = [(t["h"], t["r"], t["t"]) for t in triples_raw]
        elif triples_raw and isinstance(triples_raw[0], list):
            triples = [tuple(t) for t in triples_raw]
        else:
            triples = []

        total_before += len(triples)

        # Build pid_map
        pid_map = {t: _pid_for_triple(t) for t in triples}

        # Re-filter with stricter rules.
        # Stage 1: full filter (hard-delete + quota + score≥0.25 + top-K).
        filtered = filter_and_rank_triples(
            triples,
            question=question,
            pid_map=pid_map,
            max_keep=max_keep,
        )
        # Stage 2: minimum guarantee. If we dropped below min_keep, relax
        # the score threshold so every question keeps at least min_keep triples
        # (the hard-delete and quota filters are NOT relaxed — they remove
        # guaranteed noise like given_name/sex/disambiguation).
        if len(filtered) < min_keep and len(triples) > len(filtered):
            # Re-rank all surviving triples by score, keep top-N without threshold
            from kgproweight.kg.kg_filter import score_triple, hard_delete_triple, quota_filter
            surviving = [t for t in triples if not hard_delete_triple(t, pid=pid_map.get(t, ""))]
            surviving = quota_filter(surviving, pid_map)
            scored = [(score_triple(t, question, pid=pid_map.get(t, "")), t) for t in surviving]
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [t for _, t in scored[:max_keep]]
            # Ensure min_keep is met (even with low-score triples)
            if len(filtered) < min_keep:
                filtered = [t for _, t in scored[:max(min_keep, max_keep)]]

        total_after += len(filtered)

        # Track removed relations
        kept_set = set(filtered)
        for t in triples:
            if t not in kept_set:
                rel = t[1]
                removed_rels[rel] = removed_rels.get(rel, 0) + 1
                n_removed += 1

        # Build v3 entry in same format as v2
        v3_entry = {
            "question_id": entry.get("question_id", ""),
            "question": question,
            "linked_entities": entry.get("linked_entities", []),
            "triples": [
                {"h": t[0], "pid": pid_map.get(t, ""), "r": t[1], "t": t[2],
                 "score": 0.0, "hop": 1}
                for t in filtered
            ],
            "builder_version": "r9v6-kg-3",
            "relation_policy_version": "rel-2",
        }
        v3_data.append(v3_entry)

    reduction = 100 * (1 - total_after / max(1, total_before))
    print(f"\nTriples: {total_before} → {total_after} (-{reduction:.0f}%)")
    print(f"Removed: {n_removed} triples")
    print(f"\nTop removed relations:")
    for rel, count in sorted(removed_rels.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:>5d}  {rel}")

    if dry_run:
        print(f"\n[DRY RUN] Would write {len(v3_data)} entries to {v3_path}")
        return

    Path(v3_path).parent.mkdir(parents=True, exist_ok=True)
    with open(v3_path, "w", encoding="utf-8") as f:
        json.dump(v3_data, f, ensure_ascii=False)
    print(f"\nWritten {len(v3_data)} entries to {v3_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-path", default="indexes/kg_cache/question_kg_index_v2.json")
    p.add_argument("--v3-path", default="indexes/kg_cache/question_kg_index_v3.json")
    p.add_argument("--max-keep", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rebuild_cache(args.v2_path, args.v3_path, max_keep=args.max_keep, dry_run=args.dry_run)
