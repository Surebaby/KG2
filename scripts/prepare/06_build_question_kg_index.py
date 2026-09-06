#!/usr/bin/env python
"""Build ``question_kg_index_v2`` with the three-layer relation policy.

Two input modes:

``--datasets`` (preferred)
    Read questions straight from the dataset splits, link their mentions, pull
    each QID's subgraph from ``kg_subgraph_cache``, then filter/rank. This is
    the only mode that can cover the EVAL splits — the v1 index was built from
    training questions only, so at eval time every question missed the index
    and silently fell back to raw SPARQL-order triples, which is why the
    measured "74.6% noise removed" never reached inference.

``--input``
    Legacy: re-filter an existing v1 ``question_kg_index.json`` in place. Kept
    so the old artifact can still be converted, but it cannot add coverage.

``--silver``
    Read questions from a silver ``.jsonl``. This is the only mode that covers
    the PPO PROMPT set: the questions PPO rolls out on are the silver file's,
    and they are disjoint from the dev-split questions ``--datasets --split dev``
    produces (silver qids are ``train_*``, the dev index's are ``dev_*``), which
    is why the shipped ``question_kg_index_v2.json`` misses 100% of PPO prompts.
    Use the SAME file passed to ``--silver_data``, and ``--max_keep`` equal to
    ``ppo_max_kg_triples`` (12), or teacher and student see different KG budgets.

Usage::

    # Cover the eval splits (this is what inference reads)
    python scripts/prepare/06_build_question_kg_index.py \
      --datasets hotpotqa 2wikimultihopqa musique --split dev \
      --output indexes/kg_cache/question_kg_index_v2.json

    # Cover the PPO prompt set (this is what phase3_ppo.py reads)
    python scripts/prepare/06_build_question_kg_index.py \
      --silver data/silver_data/silver_v1_reannotated.jsonl \
      --min_keep 5 --max_keep 12 \
      --output indexes/kg_cache/question_kg_index_v2_train.json

    # Convert the legacy artifact
    python scripts/prepare/06_build_question_kg_index.py \
      --input indexes/kg_cache/question_kg_index.json \
      --output indexes/kg_cache/question_kg_index_v2.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.kg_filter import (
    _pid_for_triple,
    filter_and_rank_triples,
    hard_delete_triple,
    make_question_id,
)
from kgproweight.kg.wikidata_retriever import (
    _QA_RELATION_FILTER,
    WikidataSubgraphRetriever,
)
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.paths import data_dir

BUILDER_VERSION = "r9v6-kg-2"
RELATION_POLICY_VERSION = "rel-2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None, help="Legacy v1 question_kg_index.json")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset names to read questions from (e.g. hotpotqa musique)")
    p.add_argument("--split", default="dev", help="Split to read when --datasets is used")
    p.add_argument("--silver", default=None,
                   help="Silver .jsonl to take questions from (covers the PPO prompt set)")
    p.add_argument("--merge_into", default=None,
                   help="Existing v2 index to merge with (entries are keyed by question)")
    p.add_argument("--output", required=True, help="Path for v2 output .json")
    p.add_argument("--report", default="docs/kg_build_report.md", help="Report output")
    # 12, was 30. The docstring above already says --max_keep MUST equal
    # ppo_max_kg_triples (12), but the default contradicted it, so an index built
    # without the flag gave the student a 30-triple view of a 12-triple budget.
    p.add_argument("--max_keep", type=int, default=12, help="Max triples per question")
    p.add_argument("--min_keep", type=int, default=5,
                   help="Min triples per question; relaxes the score threshold to match "
                        "the inference fallback (which uses min_keep=5). 0 = strict.")
    p.add_argument("--max_mentions", type=int, default=5)
    p.add_argument("--offline", action="store_true", default=True,
                   help="Cache-only: never call Wikidata (default)")
    p.add_argument("--online", dest="offline", action="store_false",
                   help="Allow live Wikidata calls for cache misses")
    return p.parse_args()


pid_for_triple = _pid_for_triple  # kg_filter's version (has the label→PID map)


def _load_from_index(path: str) -> List[Dict[str, Any]]:
    """Legacy mode: ``[{"q": ..., "t": [[h, r, t], ...]}, ...]``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for e in raw:
        out.append({
            "question": e.get("question", e.get("q", "")),
            "triples": [tuple(t) for t in e.get("triples", e.get("t", [])) if len(t) == 3],
            "linked_entities": [],
        })
    return out


def _build_components(offline: bool):
    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=offline)
    kg = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=resolve_kg_cache_dir(),
        offline=offline, relation_filter=_QA_RELATION_FILTER,
    )
    return linker, kg


def _resolve_one(q: str, ds: str, qid: str, linker, kg, max_mentions: int) -> Dict[str, Any]:
    """Link a question's mentions and pull their subgraphs from the raw cache."""
    linked: List[Dict[str, Any]] = []
    qids: List[str] = []
    for m in extract_mentions(q, max_n=max_mentions):
        r = linker.link_single(m, question=q)
        linked.append({
            "mention": m,
            "qid": r.selected_qid,
            "label": r.selected_label,
            "description": r.description,
            "score": round(float(r.score), 4),
            "margin": round(float(r.margin), 4),
            "abstained": bool(r.abstained),
            "abstain_reason": r.abstain_reason,
        })
        if r.selected_qid and not r.abstained:
            qids.append(r.selected_qid)
    return {
        "question": q,
        "dataset": ds,
        "qid": qid,
        "triples": kg.fetch(qids) if qids else [],
        "linked_entities": linked,
    }


def _load_from_datasets(
    datasets: List[str], split: str, max_mentions: int, offline: bool,
) -> List[Dict[str, Any]]:
    linker, kg = _build_components(offline)
    out: List[Dict[str, Any]] = []
    for ds in datasets:
        src = Path(data_dir()) / ds / f"{split}.jsonl"
        if not src.exists():
            print(f"  ! {src} missing — skipped")
            continue
        n = 0
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = str(obj.get("question", "")).strip()
                if not q:
                    continue
                out.append(_resolve_one(
                    q, ds, str(obj.get("id", "")), linker, kg, max_mentions,
                ))
                n += 1
        print(f"  {ds}/{split}: {n} questions")
    return out


def _load_from_silver(
    silver_path: str, max_mentions: int, offline: bool,
) -> List[Dict[str, Any]]:
    """Cover the PPO prompt set by re-resolving each silver question.

    Re-resolving from the RAW subgraph cache — rather than re-filtering the v1
    index's already-degraded triples — is what removes the pre-fix noise (30.6%
    of legacy entries carried unmapped relations that reached the prompt).
    """
    linker, kg = _build_components(offline)
    out: List[Dict[str, Any]] = []
    seen = set()
    with open(silver_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(obj.get("question", "")).strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append(_resolve_one(
                q, str(obj.get("dataset", "silver")), str(obj.get("qid", "")),
                linker, kg, max_mentions,
            ))
    print(f"  silver {Path(silver_path).name}: {len(out)} unique questions")
    return out


def main():
    args = parse_args()
    t0 = time.time()

    if not args.input and not args.datasets and not args.silver:
        raise SystemExit("Provide --datasets and/or --silver (preferred), or --input")

    records: List[Dict[str, Any]] = []
    if args.datasets:
        print(f"Reading questions from datasets={args.datasets} split={args.split} "
              f"(offline={args.offline})")
        records += _load_from_datasets(
            args.datasets, args.split, args.max_mentions, args.offline,
        )
    if args.silver:
        print(f"Reading questions from silver {args.silver} (offline={args.offline})")
        records += _load_from_silver(args.silver, args.max_mentions, args.offline)
    if args.input:
        print(f"Reading legacy index {args.input}")
        records += _load_from_index(args.input)
    print(f"Total input records: {len(records)}")

    # ── Stats: before ──
    all_triples_before = sum(len(r["triples"]) for r in records)
    all_relations_before = Counter()
    for r in records:
        for t in r["triples"]:
            all_relations_before[t[1]] += 1

    # ── Build new index ──
    v2_entries: List[Dict[str, Any]] = []
    total_hard_deleted = 0
    total_quota_dropped = 0
    total_kept = 0
    per_dataset: Dict[str, dict] = defaultdict(
        lambda: {"count": 0, "triples_before": 0, "triples_after": 0, "empty": 0}
    )

    for rec in records:
        q = rec["question"]
        triples = rec["triples"]
        n_before = len(triples)
        pid_map = {t: pid_for_triple(t) for t in triples}

        hard_del = sum(1 for t in triples if hard_delete_triple(t, pid=pid_map.get(t, "")))
        total_hard_deleted += hard_del

        # Anchor scoring on the ACTUAL linked mentions when we have them; the
        # legacy path re-derives them from the question text.
        q_entities = [
            e["mention"] for e in rec.get("linked_entities", [])
            if e.get("qid") and not e.get("abstained")
        ] or None

        filtered_rich = filter_and_rank_triples(
            triples, q, pid_map=pid_map, max_keep=args.max_keep,
            min_keep=args.min_keep, rich=True, question_entities=q_entities,
        )
        n_after = len(filtered_rich)
        total_quota_dropped += (n_before - hard_del - n_after)
        total_kept += n_after

        ds = rec.get("dataset", "legacy")
        v2_entries.append({
            "question_id": rec.get("qid") or make_question_id(q, ds if ds != "legacy" else ""),
            "question": q,
            "dataset": ds,
            "linked_entities": rec.get("linked_entities", []),
            "triples": filtered_rich,
            "n_before": n_before,
            "n_after": n_after,
            "builder_version": BUILDER_VERSION,
            "relation_policy_version": RELATION_POLICY_VERSION,
        })
        per_dataset[ds]["count"] += 1
        per_dataset[ds]["triples_before"] += n_before
        per_dataset[ds]["triples_after"] += n_after
        if n_after == 0:
            per_dataset[ds]["empty"] += 1

    # Deduplicate on question — the consumers build a dict keyed by question, so
    # duplicates waste memory and make "N entries" misleading. They arise when a
    # question appears in more than one source (e.g. a silver question that is
    # also in a dataset split). Keep the entry with more triples.
    if len({e["question"] for e in v2_entries}) != len(v2_entries):
        best: Dict[str, Dict[str, Any]] = {}
        for e in v2_entries:
            prev = best.get(e["question"])
            if prev is None or len(e["triples"]) > len(prev["triples"]):
                best[e["question"]] = e
        print(f"Deduplicated {len(v2_entries) - len(best)} repeated questions")
        v2_entries = list(best.values())

    # ── Merge with an existing index (question is the lookup key) ──
    if args.merge_into and Path(args.merge_into).exists():
        with open(args.merge_into, encoding="utf-8") as f:
            existing = json.load(f)
        by_q = {e.get("question", e.get("q", "")): e for e in existing}
        for e in v2_entries:
            by_q[e["question"]] = e          # new build wins
        v2_entries = list(by_q.values())
        print(f"Merged with {args.merge_into}: {len(v2_entries)} total entries")

    # ── Stats: after ──
    all_relations_after = Counter()
    for entry in v2_entries:
        for t in entry["triples"]:
            all_relations_after[t["r"]] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(v2_entries, f, ensure_ascii=False)
    print(f"Wrote {len(v2_entries)} entries to {out_path}")

    # ── Report ──
    link_total = sum(len(e["linked_entities"]) for e in v2_entries)
    link_ok = sum(
        1 for e in v2_entries for x in e["linked_entities"]
        if x.get("qid") and not x.get("abstained")
    )
    link_abstain = sum(
        1 for e in v2_entries for x in e["linked_entities"] if x.get("abstained")
    )

    report_lines = [
        "# KG Build Report — question_kg_index_v2",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Builder: {BUILDER_VERSION} | Relation policy: {RELATION_POLICY_VERSION}",
        "",
        "## Summary",
        "| Metric | Before | After | Change |",
        "|---|---|---|---|",
        f"| Entries | {len(records)} | {len(v2_entries)} | — |",
        f"| Total triples | {all_triples_before} | {total_kept} | "
        f"{all_triples_before - total_kept} removed "
        f"({(all_triples_before - total_kept)/max(1,all_triples_before)*100:.1f}%) |",
        f"| Hard deleted | — | {total_hard_deleted} | — |",
        f"| Quota/score dropped | — | {total_quota_dropped} | — |",
        f"| Avg triples/question | {all_triples_before/max(1,len(records)):.1f} | "
        f"{total_kept/max(1,len(v2_entries)):.1f} | — |",
        "",
        "## Coverage per dataset",
        "| Dataset | Questions | Triples before | Triples after | Empty KG |",
        "|---|---|---|---|---|",
    ]
    for ds, st in sorted(per_dataset.items()):
        report_lines.append(
            f"| {ds} | {st['count']} | {st['triples_before']} | {st['triples_after']} "
            f"| {st['empty']} ({st['empty']/max(1,st['count'])*100:.1f}%) |"
        )

    report_lines += [
        "",
        "## Entity linking",
        "| Metric | Value |",
        "|---|---|",
        f"| Mentions processed | {link_total} |",
        f"| Linked (high confidence) | {link_ok} ({link_ok/max(1,link_total)*100:.1f}%) |",
        f"| Abstained | {link_abstain} ({link_abstain/max(1,link_total)*100:.1f}%) |",
        "",
        "## Top 20 Relations: Before",
        "| Relation | Count |",
        "|---|---|",
    ]
    for rel, count in all_relations_before.most_common(20):
        report_lines.append(f"| {rel} | {count} |")

    report_lines += ["", "## Top 20 Relations: After", "| Relation | Count |", "|---|---|"]
    for rel, count in all_relations_after.most_common(20):
        report_lines.append(f"| {rel} | {count} |")

    taxonomic = ["instance of", "subclass of"]
    tax_before = sum(all_relations_before.get(r, 0) for r in taxonomic)
    tax_after = sum(all_relations_after.get(r, 0) for r in taxonomic)
    report_lines += [
        "",
        "## Taxonomic Relation Ratio",
        "| Metric | Before | After |",
        "|---|---|---|",
        f"| instance_of + subclass_of | {tax_before/max(1,all_triples_before)*100:.1f}% "
        f"| {tax_after/max(1,total_kept)*100:.1f}% |",
        "| Target | — | < 25% |",
    ]

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Report → {report_path}")
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"Taxonomic ratio: {tax_before/max(1,all_triples_before)*100:.1f}% → "
          f"{tax_after/max(1,total_kept)*100:.1f}%")
    print(f"Entity linking: {link_ok}/{link_total} high-confidence, {link_abstain} abstained")


if __name__ == "__main__":
    main()
