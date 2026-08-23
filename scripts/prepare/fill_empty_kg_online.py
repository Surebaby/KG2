#!/usr/bin/env python
"""Fill empty-KG index entries with live Wikidata (rate-limited).

The offline index rebuild (``06_build_question_kg_index.py --offline``) leaves a
fraction of questions with empty KG because their entities/subgraphs are missing
from the offline caches. This script re-resolves ONLY those empty entries against
live Wikidata, with a conservative inter-request delay to stay under the SPARQL
``429`` rate limit, then re-applies the exact same ``filter_and_rank_triples``
policy so the filled entries are indistinguishable from the offline build.

Every successful link/fetch is persisted to ``entity_cache.jsonl`` /
``kg_subgraph_cache.jsonl`` (the normal ``EntityLinker`` / retriever behaviour),
so an interrupted run is naturally resumable: a re-run skips what was already
fetched.

Usage::

    python scripts/prepare/fill_empty_kg_online.py \\
      --input indexes/kg_cache/question_kg_index_v2_full.json \\
      --dataset musique \\
      --output indexes/kg_cache/question_kg_index_v2_full.json \\
      --delay 3.0 \\
      --report docs/kg_fill_report.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.entity_linker import EntityLinker, extract_mentions
from kgproweight.kg.kg_filter import (
    _pid_for_triple,
    filter_and_rank_triples,
)
from kgproweight.kg.wikidata_retriever import (
    _QA_RELATION_FILTER,
    WikidataSubgraphRetriever,
)
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Index .json with empty entries to fill.")
    p.add_argument("--output", required=True, help="Where to write the updated index.")
    p.add_argument("--dataset", default=None,
                   help="Only fill empties for this dataset (default: all datasets).")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds to sleep after each network miss (rate-limit guard).")
    p.add_argument("--max_mentions", type=int, default=5)
    p.add_argument("--max_keep", type=int, default=30)
    p.add_argument("--min_keep", type=int, default=5,
                   help="Matches the index build + inference fallback (min_keep=5).")
    p.add_argument("--report", default="docs/kg_fill_report.md")
    return p.parse_args()


def _resolve_online(question: str, linker, kg, max_mentions: int) -> Dict[str, Any]:
    """Re-link + re-fetch one question against live Wikidata, mirroring
    ``06_build_question_kg_index._resolve_one`` but online."""
    linked: List[Dict[str, Any]] = []
    qids: List[str] = []
    for m in extract_mentions(question, max_n=max_mentions):
        r = linker.link_single(m, question=question)
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
        "triples": kg.fetch(qids) if qids else [],
        "linked_entities": linked,
    }


def main():
    args = parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))

    linker = EntityLinker(
        cache_path=resolve_entity_cache_path(), offline=False,
        request_delay=args.delay,
    )
    kg = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=resolve_kg_cache_dir(),
        offline=False, request_delay=args.delay, relation_filter=_QA_RELATION_FILTER,
    )

    empty = [
        e for e in raw
        if not e.get("triples")
        and (args.dataset is None or e.get("dataset") == args.dataset)
    ]
    print(f"Empty entries to fill: {len(empty)} "
          f"(dataset={args.dataset or 'all'}, delay={args.delay}s)")

    filled = 0
    t0 = time.time()
    for i, e in enumerate(empty):
        q = e.get("question", "")
        result = _resolve_online(q, linker, kg, args.max_mentions)
        e["linked_entities"] = result["linked_entities"]

        triples = result["triples"]
        if triples:
            pid_map = {t: _pid_for_triple(t) for t in triples}
            q_entities = [
                x["mention"] for x in result["linked_entities"]
                if x.get("qid") and not x.get("abstained")
            ] or None
            filtered = filter_and_rank_triples(
                triples, q, pid_map=pid_map, max_keep=args.max_keep,
                min_keep=args.min_keep, rich=True, question_entities=q_entities,
            )
            e["triples"] = filtered
            if filtered:
                filled += 1
        else:
            e["triples"] = []

        if (i + 1) % 25 == 0:
            rate = (i + 1) / max(1, time.time() - t0)
            print(f"  progress {i + 1}/{len(empty)}  filled={filled}  "
                  f"({rate:.2f} q/s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    report = [
        "# KG Fill Report — empty entries (online)",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Input: {args.input}  Output: {args.output}",
        f"Dataset: {args.dataset or 'all'}  Delay: {args.delay}s",
        "",
        f"- Empty entries processed: {len(empty)}",
        f"- Filled (non-empty KG after re-resolve): {filled}",
        f"- Still empty: {len(empty) - filled}",
        f"- Elapsed: {time.time() - t0:.1f}s",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report), encoding="utf-8")

    print(f"Filled {filled}/{len(empty)} empty entries; wrote {len(raw)} entries → {out_path}")
    print(f"Report → {args.report}")


if __name__ == "__main__":
    main()
