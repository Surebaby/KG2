#!/usr/bin/env python
"""Step 4 — Pre-warm Wikidata caches (entities + 2-hop subgraphs).

Mirrors the mention-extraction logic of :class:`KGProWeightPipeline` so that
the eval pipeline can run fully offline. By default scans HotpotQA /
2WikiMultiHopQA / MuSiQue / D_dropout dev splits.

Usage::

    python scripts/prepare/04_prewarm_wikidata_cache.py \
        --datasets hotpotqa 2wikimultihopqa musique d_dropout --split dev
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from kgproweight.kg.cache import EntityCache, SubgraphCache
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import data_dir, output_dir

configure_logging("INFO")
logger = get_logger(__name__)

ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
DEFAULT_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "d_dropout"]


def extract_entities(question: str, max_entities: int = 5) -> List[str]:
    entities = ENTITY_PATTERN.findall(question or "")
    return list(dict.fromkeys(entities))[:max_entities]


def dataset_path(data_root: Path, name: str, split: str) -> Path:
    return data_root / name / f"{split}.jsonl"


def iter_questions(path: Path, limit: Optional[int]):
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield str(obj.get("id", idx)), obj.get("question", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--split", default="dev")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--limit_per_dataset", type=int, default=None)
    p.add_argument("--max_entities_per_question", type=int, default=5)
    p.add_argument("--entity_cache", default=None)
    p.add_argument("--kg_cache_dir", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--max_mentions", type=int, default=None)
    p.add_argument("--offline_check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir) if args.data_dir else Path(data_dir())
    entity_cache_path = Path(args.entity_cache) if args.entity_cache else Path(resolve_entity_cache_path())
    kg_cache_dir = Path(args.kg_cache_dir) if args.kg_cache_dir else Path(resolve_kg_cache_dir())
    report_path = Path(args.report) if args.report else Path(output_dir()) / "cache_prewarm_report.json"

    mention_counts: Counter = Counter()
    dataset_sizes: Dict[str, int] = {}

    for ds in args.datasets:
        path = dataset_path(data_root, ds, args.split)
        if not path.exists():
            logger.warning("Missing dataset split: %s", path)
            continue
        n = 0
        for _, question in iter_questions(path, args.limit_per_dataset):
            n += 1
            for m in extract_entities(question, args.max_entities_per_question):
                mention_counts[m] += 1
        dataset_sizes[ds] = n
        logger.info("Scanned %s/%s: %d items", ds, args.split, n)

    mentions = [m for m, _ in mention_counts.most_common()]
    if args.max_mentions is not None:
        mentions = mentions[: args.max_mentions]

    entity_cache = EntityCache(entity_cache_path)
    sub_cache = SubgraphCache(kg_cache_dir / "kg_subgraph_cache.jsonl")
    logger.info(
        "Mentions=%d entity_cache=%d kg_cache=%d",
        len(mentions),
        len(entity_cache),
        len(sub_cache),
    )

    if args.offline_check:
        cached_mentions = sum(1 for m in mentions if entity_cache.get(m) is not None)
        cached_qids = {entity_cache.get(m) for m in mentions if entity_cache.get(m) is not None}
        cached_kg = sum(1 for q in cached_qids if f"{q}_2" in sub_cache)
        report = {
            "offline_check": True,
            "datasets": args.datasets,
            "split": args.split,
            "dataset_sizes": dataset_sizes,
            "mentions_total": len(mentions),
            "linked_in_cache": cached_mentions,
            "qids_with_subgraph": cached_kg,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Offline check report → %s", report_path)
        return

    linker = EntityLinker(cache_path=str(entity_cache_path))
    kg_retriever = WikidataSubgraphRetriever(max_hops=2, max_neighbors=30, cache_dir=str(kg_cache_dir))

    failed: List[str] = []
    for idx, mention in enumerate(mentions, start=1):
        if entity_cache.get(mention) is not None:
            continue
        qid = linker.link_single(mention)
        if qid is None:
            failed.append(mention)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if idx % 50 == 0:
            logger.info("linked %d/%d  failed=%d", idx, len(mentions), len(failed))

    qids: Set[str] = {q for q in (entity_cache.get(m) for m in mentions) if q}
    missing_kg = []
    for idx, qid in enumerate(sorted(qids), start=1):
        if f"{qid}_2" in sub_cache:
            continue
        triples = kg_retriever.fetch([qid])
        if not triples:
            missing_kg.append(qid)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if idx % 25 == 0:
            logger.info("fetched %d/%d QIDs  missing=%d", idx, len(qids), len(missing_kg))

    report = {
        "offline_check": False,
        "datasets": args.datasets,
        "split": args.split,
        "dataset_sizes": dataset_sizes,
        "mentions_total": len(mentions),
        "linked_qids": len(qids),
        "missing_qids": missing_kg,
        "failed_mentions": failed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Prewarm report → %s", report_path)


if __name__ == "__main__":
    main()
