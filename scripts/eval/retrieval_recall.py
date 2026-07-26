#!/usr/bin/env python
"""Measure supporting-fact retrieval recall on HotpotQA, 2Wiki, MuSiQue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

from kgproweight.retrieval.hybrid import DEFAULT_TOPK, build_flashrag_config
from kgproweight.data.flashrag_loader import flashrag_config
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import data_dir

configure_logging("INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["hotpotqa"])
    p.add_argument("--split", default="dev")
    p.add_argument("--k_values", type=int, nargs="+", default=[5, 15, 50])
    p.add_argument("--max_queries", type=int, default=100)
    return p.parse_args()


def _get_supporting_titles_hotpotqa(item: dict) -> Set[str]:
    """Extract supporting fact titles from HotpotQA item."""
    titles = set()
    for fact in item.get("supporting_facts", []):
        if isinstance(fact, list) and len(fact) >= 1:
            titles.add(fact[0].strip().lower())
    return titles


def _get_supporting_titles_2wiki(item: dict) -> Set[str]:
    """Extract supporting evidence titles from 2WikiMultihopQA."""
    titles = set()
    for ev in item.get("evidence", []):
        if isinstance(ev, list):
            for e in ev:
                if isinstance(e, dict) and "title" in e:
                    titles.add(e["title"].strip().lower())
    return titles


def _get_supporting_titles_musique(item: dict) -> Set[str]:
    """Extract supporting paragraph titles from MuSiQue."""
    titles = set()
    for qa in item.get("question_decomposition", []):
        for sp in qa.get("paragraphs", []):
            if isinstance(sp, dict) and "title" in sp:
                titles.add(sp["title"].strip().lower())
    return titles


def _check_answer_in_passages(answer: str, passages: list) -> bool:
    """Check if any gold answer appears in retrieved passage text."""
    answer_clean = answer.lower().strip().rstrip('.?!,;:')
    if not answer_clean or len(answer_clean) < 2:
        return False
    for doc in passages:
        text = (doc.get("contents", "") or doc.get("text", "")).lower()
        if answer_clean in text:
            return True
    return False


def main():
    args = parse_args()

    setup_flashrag()
    from flashrag.utils import get_retriever

    for ds_name in args.datasets:
        ds_path = Path(data_dir()) / ds_name / f"{args.split}.jsonl"
        if not ds_path.exists():
            logger.warning("Dataset not found: %s", ds_path)
            continue

        items = [json.loads(l) for l in ds_path.read_text(encoding="utf-8").strip().split("\n")]
        if args.max_queries and len(items) > args.max_queries:
            items = items[:args.max_queries]

        questions = [item["question"] for item in items]

        max_k = max(args.k_values)
        flashrag_cfg = build_flashrag_config(
            dataset_name=ds_name,
            save_note="recall_test",
            save_dir="/tmp/recall_test",
            split=args.split,
            topk=max_k,
        )
        cfg = flashrag_config(flashrag_cfg)
        retriever = get_retriever(cfg)

        logger.info("Retrieving top-%d for %d queries...", max_k, len(questions))
        all_results = retriever.batch_search(questions)

        print(f"\n{'='*60}")
        print(f"  {ds_name} — Answer Recall ({len(items)} queries)")
        print(f"{'='*60}")
        print(f"{'K':>6}  {'Recall':>8}  {'Hits':>6}/{'>Total':>6}")
        print(f"{'-'*6}  {'-'*8}  {'-'*12}")

        for k in args.k_values:
            total = 0
            hit = 0
            for item, results in zip(items, all_results):
                answers = item.get("golden_answers", [])
                if not answers:
                    continue
                total += 1
                docs = results[:k]
                if any(_check_answer_in_passages(ans, docs) for ans in answers):
                    hit += 1

            recall = hit / max(1, total)
            print(f"{k:>6}  {recall:>8.3f}  {hit:>4}/{total}")


if __name__ == "__main__":
    main()
