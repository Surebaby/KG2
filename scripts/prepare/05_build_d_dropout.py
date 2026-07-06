#!/usr/bin/env python
"""Step 5 — Build the D_dropout robustness benchmark.

Procedure:
  1. Sample N items from HotpotQA dev.
  2. For each item, link query entities and fetch the 2-hop subgraph.
  3. Identify triples that bridge the gold answer entity and *sever* them
     (replace with random noise triples).
  4. Persist the resulting items to ``$KGPW_DATA_DIR/d_dropout/dev.jsonl``.
     Each record carries ``metadata.dropout.modified_kg`` which the
     KG-ProWeight pipeline consumes (bug-fix #5).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

from rapidfuzz import fuzz

from kgproweight.kg.coverage import coverage_score
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import data_dir

configure_logging("INFO")
logger = get_logger(__name__)

_NOISE_ENTITIES = [
    "Random Organization",
    "Unknown Person",
    "Fictional Place",
    "Unrelated Entity",
    "Placeholder Entity",
]
_NOISE_RELATIONS = [
    "unrelated_to",
    "does_not_affect",
    "is_irrelevant_to",
    "has_no_connection_with",
    "is_distinct_from",
]


def sever_answer_path(
    kg_subgraph: List[Tuple[str, str, str]],
    gold_answer: str,
    threshold: float = 70.0,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    answer = gold_answer.lower().strip()
    modified: List[Tuple[str, str, str]] = []
    severed: List[Tuple[str, str, str]] = []
    for triple in kg_subgraph:
        h, r, t = triple
        if fuzz.token_sort_ratio(h.lower(), answer) >= threshold or fuzz.token_sort_ratio(t.lower(), answer) >= threshold:
            severed.append(triple)
            modified.append(
                (
                    random.choice(_NOISE_ENTITIES),
                    random.choice(_NOISE_RELATIONS),
                    random.choice(_NOISE_ENTITIES),
                )
            )
        else:
            modified.append(triple)
    return modified, severed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hotpotqa_dev", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--sample_size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entity_cache", default=None)
    p.add_argument("--kg_cache_dir", default=None)
    p.add_argument("--max_modified_triples", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    hotpot_path = Path(args.hotpotqa_dev) if args.hotpotqa_dev else Path(data_dir()) / "hotpotqa" / "dev.jsonl"
    out_path = Path(args.output) if args.output else Path(data_dir()) / "d_dropout" / "dev.jsonl"
    entity_cache = Path(args.entity_cache) if args.entity_cache else Path(resolve_entity_cache_path())
    kg_cache_dir = Path(args.kg_cache_dir) if args.kg_cache_dir else Path(resolve_kg_cache_dir())

    if not hotpot_path.exists():
        sys.exit(f"HotpotQA dev not found at {hotpot_path}. Run scripts/prepare/03_download_datasets.py first.")

    items = []
    with open(hotpot_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    n = min(args.sample_size, len(items))
    sampled = random.sample(items, n)
    logger.info("Sampled %d items from %s", n, hotpot_path)

    linker = EntityLinker(cache_path=str(entity_cache) if entity_cache.exists() else None)
    kg_retr = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=str(kg_cache_dir))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_severed_total = 0
    n_no_triples = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, item in enumerate(sampled):
            question = item.get("question", "")
            gold = (item.get("golden_answers") or [""])[0]

            mentions = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", question)
            mentions = list(dict.fromkeys(mentions))[:5]
            linked = linker.link(mentions)
            qids = [q for q in linked.values() if q]
            triples = kg_retr.fetch(qids) if qids else []
            if not triples:
                n_no_triples += 1
                modified = []
                severed = []
            else:
                modified, severed = sever_answer_path(triples, gold)
            n_severed_total += len(severed)
            time.sleep(0.1)

            record = {
                "id": item.get("id", str(i)),
                "question": question,
                "golden_answers": item.get("golden_answers", []),
                "metadata": {
                    **(item.get("metadata") or {}),
                    "dropout": {
                        "original_kg_size": len(triples),
                        "modified_kg": [list(t) for t in modified[: args.max_modified_triples]],
                        "original_kg": [list(t) for t in triples[: args.max_modified_triples]],
                        "severed_triples": [list(t) for t in severed],
                        "n_severed": len(severed),
                        "kg_coverage": coverage_score(mentions, linked),
                    },
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                logger.info("processed %d/%d  severed=%d  empty_kg=%d", i + 1, n, n_severed_total, n_no_triples)

    logger.info(
        "Done. items=%d items_without_kg=%d severed=%d → %s",
        n,
        n_no_triples,
        n_severed_total,
        out_path,
    )


if __name__ == "__main__":
    main()
