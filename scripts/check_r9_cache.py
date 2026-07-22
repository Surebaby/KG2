#!/usr/bin/env python
"""R9 cache integrity check — verify question→KG index before training."""
from __future__ import annotations

import json
import random
from pathlib import Path

from kgproweight.utils.paths import index_dir


def main():
    cache_path = Path(index_dir()) / "kg_cache" / "question_kg_index.json"

    # 1. Check file exists and size
    if not cache_path.exists():
        print(f"❌ Cache NOT FOUND: {cache_path}")
        return 1
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"✅ Cache exists: {cache_path}")
    print(f"   Size: {size_mb:.1f} MB")

    # 2. Load and check structure
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"   Entries: {len(raw)}")

    # 3. Sample entry
    for entry in raw[:3]:
        q = entry.get("q", "")[:80]
        t = entry.get("t", [])
        print(f"   Q: {q}...")
        print(f"   Triples: {len(t)}, first 3: {t[:3]}")
        print()

    # 4. Check for duplicates / coverage
    questions = [e["q"] for e in raw]
    unique_q = set(questions)
    print(f"   Unique questions: {len(unique_q)} / {len(questions)} ({len(unique_q)/max(1,len(questions))*100:.1f}%)")
    if len(unique_q) != len(questions):
        dupes = len(questions) - len(unique_q)
        print(f"   ⚠️  {dupes} duplicate questions found")

    # 5. Triple stats
    total_triples = sum(len(e.get("t", [])) for e in raw)
    avg_triples = total_triples / max(1, len(raw))
    print(f"   Total triples: {total_triples}")
    print(f"   Avg triples/question: {avg_triples:.1f}")

    # 6. Index build timing
    import time
    t0 = time.time()
    q_kg_index = {e["q"]: e["t"] for e in raw}
    elapsed = time.time() - t0
    print(f"\n✅ Index build: {elapsed:.3f}s ({len(q_kg_index)} entries)")

    # 7. Random lookup timing
    sample_qs = random.sample(list(q_kg_index.keys()), min(100, len(q_kg_index)))
    t0 = time.time()
    hits = 0
    for q in sample_qs:
        if q_kg_index.get(q):
            hits += 1
    elapsed = time.time() - t0
    print(f"✅ Random lookup: {hits}/{len(sample_qs)} hits in {elapsed:.4f}s ({elapsed/len(sample_qs)*1000:.2f}ms/query)")

    # 8. Check overlap with silver data questions
    from kgproweight.utils.paths import data_dir
    silver_path = Path(data_dir()) / "silver_data" / "all.jsonl"
    if silver_path.exists():
        silver_data = [json.loads(l) for l in silver_path.read_text(encoding="utf-8").strip().split("\n")]
        silver_qs = {item["question"] for item in silver_data if "question" in item}
        cache_qs = set(q_kg_index.keys())
        overlap = silver_qs & cache_qs
        print(f"\n✅ Silver data overlap: {len(overlap)}/{len(silver_qs)} ({len(overlap)/max(1,len(silver_qs))*100:.1f}%)")
        missing = silver_qs - cache_qs
        if missing:
            print(f"   ⚠️  {len(missing)} silver questions NOT in cache")
            for mq in list(missing)[:3]:
                print(f"      {mq[:80]}...")

    print("\n✅ Cache validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
