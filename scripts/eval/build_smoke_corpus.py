#!/usr/bin/env python
"""Build a tiny smoke-test corpus + dataset from HotpotQA dev distractor context.

Each dev item bundles ~10 paragraphs (2 gold + 8 distractors). We pool those
paragraphs across the first N questions into a small FlashRAG corpus so dense+bm25
retrieval can actually surface gold passages -> SFT vs PPO comparison is meaningful.
"""
import json
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROOT = Path("/home/ai/flashrag/kgpaper")
SRC = ROOT / "data/hotpotqa/dev.jsonl"
OUT_DATA = ROOT / "data/hotpotqa_smoke"
OUT_IDX = ROOT / "indexes_smoke"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_IDX.mkdir(parents=True, exist_ok=True)

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        rows.append(json.loads(line))

# Pool paragraphs, dedup by (title, text)
seen = {}
corpus = []
for r in rows:
    ctx = r["metadata"]["metadata"]["context"]
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        text = " ".join(s.strip() for s in sents).strip()
        key = (title, text)
        if key in seen or not text:
            continue
        seen[key] = len(corpus)
        corpus.append({"id": str(len(corpus)), "contents": f"{title}\n{text}"})

# Write small corpus
corpus_path = OUT_IDX / "corpus_flashrag.jsonl"
with open(corpus_path, "w") as f:
    for c in corpus:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# Write small dev set (strip heavy context to keep dataset lean; keep QA fields)
dev_path = OUT_DATA / "dev.jsonl"
with open(dev_path, "w") as f:
    for r in rows:
        slim = {"id": r["id"], "question": r["question"],
                "golden_answers": r["golden_answers"], "metadata": {}}
        f.write(json.dumps(slim, ensure_ascii=False) + "\n")

print(f"questions: {len(rows)}")
print(f"corpus passages (deduped): {len(corpus)}")
print(f"corpus -> {corpus_path}  ({corpus_path.stat().st_size/1024:.0f} KB)")
print(f"dev    -> {dev_path}  ({dev_path.stat().st_size/1024:.0f} KB)")
