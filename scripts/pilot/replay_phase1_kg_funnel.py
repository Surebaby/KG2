#!/usr/bin/env python
"""Replay fixed passages through the current Phase-1 entity/KG funnel.

The source entity cache is copied into the experiment directory before use, so
this diagnostic cannot mutate the existing cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from kgproweight.kg.entity_linker import EntityLinker, build_passage_text, build_passage_titles
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever, _QA_RELATION_FILTER
from kgproweight.retrieval.bootstrap import (
    resolve_entity_cache_path,
    resolve_kg_cache_dir,
)
from kgproweight.training.phase1_distill import extract_mentions_robust
from kgproweight.utils.logging import dump_manifest


def _read(paths: List[Path]) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_counts(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, int]:
    return {
        "mentions_found": sum(int(row[f"{prefix}_n_mentions"] > 0) for row in rows),
        "entity_linked": sum(int(row[f"{prefix}_n_linked"] > 0) for row in rows),
        "raw_kg_nonempty": sum(int(row[f"{prefix}_n_raw_triples"] > 0) for row in rows),
        "filtered_kg_nonempty": sum(int(row[f"{prefix}_n_filtered_triples"] > 0) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected", type=int, default=90)
    parser.add_argument("--max_mentions", type=int, default=8)
    parser.add_argument("--max_keep", type=int, default=12)
    parser.add_argument("--min_keep", type=int, default=5)
    parser.add_argument("--entity_index", default=None)
    args = parser.parse_args()

    source_paths = [Path(path).resolve() for path in args.source]
    source_rows = _read(source_paths)
    if len(source_rows) != args.expected:
        raise SystemExit(f"source count {len(source_rows)} != {args.expected}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    source_cache = Path(resolve_entity_cache_path()).resolve()
    experiment_cache = output_dir / "entity_cache.snapshot.jsonl"
    shutil.copyfile(source_cache, experiment_cache)
    linker = EntityLinker(
        cache_path=str(experiment_cache),
        offline=True,
        entity_index_path=args.entity_index,
    )
    kg_retriever = WikidataSubgraphRetriever(
        max_hops=2,
        max_neighbors=30,
        cache_dir=resolve_kg_cache_dir(),
        offline=True,
        relation_filter=_QA_RELATION_FILTER,
    )

    output_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        passages = list(source.get("retrieved_passages") or [])
        question = str(source.get("question") or "")
        mentions = extract_mentions_robust(
            question, passages=passages, max_n=args.max_mentions
        )
        titles = build_passage_titles(passages)
        passage_text = build_passage_text(passages)
        linked: Dict[str, str] = {}
        link_diagnostics: Dict[str, Dict[str, Any]] = {}
        failures: Dict[str, str] = {}
        for mention in mentions:
            result = linker.link_single(
                mention,
                question=question,
                retrieved_titles=titles,
                passage_text=passage_text,
            )
            if result.selected_qid and not result.abstained:
                linked[mention] = result.selected_qid
                link_diagnostics[mention] = {
                    "qid": result.selected_qid,
                    "label": result.selected_label,
                    "description": result.description,
                    "score": result.score,
                    "margin": result.margin,
                    "appears_in_question": mention.casefold() in question.casefold(),
                    "candidate_count": len(result.candidates),
                }
            else:
                failures[mention] = result.abstain_reason or "UNKNOWN"
        qids = list(dict.fromkeys(linked.values()))
        qid_raw_counts = {
            entity_qid: len(kg_retriever.fetch([entity_qid])) for entity_qid in qids
        }
        raw = kg_retriever.fetch(qids) if qids else []
        filtered = filter_and_rank_triples(
            raw,
            question=question,
            max_keep=args.max_keep,
            min_keep=args.min_keep,
        )
        old_md = source.get("metadata") or {}
        output_rows.append(
            {
                "qid": source.get("qid"),
                "dataset": source.get("dataset"),
                "question": question,
                "passage_titles": titles,
                "mentions": mentions,
                "linked_entities": linked,
                "link_diagnostics": link_diagnostics,
                "qid_raw_counts": qid_raw_counts,
                "link_failures": failures,
                "filtered_kg": [list(triple) for triple in filtered],
                "old_n_mentions": int(old_md.get("n_mentions") or 0),
                "old_n_linked": len(old_md.get("linked_entities") or {}),
                "old_n_raw_triples": int(old_md.get("n_triples_prefilter") or 0),
                "old_n_filtered_triples": len(source.get("kg_subgraph") or []),
                "new_n_mentions": len(mentions),
                "new_n_linked": len(linked),
                "new_n_raw_triples": len(raw),
                "new_n_filtered_triples": len(filtered),
            }
        )

    rows_path = output_dir / "replay_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        old = _stage_counts(rows, "old")
        new = _stage_counts(rows, "new")
        return {
            "n_questions": len(rows),
            "old": old,
            "new": new,
            "delta_new_minus_old": {key: new[key] - old[key] for key in old},
            "new_link_failure_reasons": dict(
                sorted(
                    Counter(
                        reason.split(" (")[0]
                        for row in rows
                        for reason in row["link_failures"].values()
                    ).items()
                )
            ),
        }

    report = {
        "integrity_pass": len(output_rows) == args.expected,
        "protocol": {
            "fixed_source_passages": True,
            "offline": True,
            "max_mentions": args.max_mentions,
            "max_keep": args.max_keep,
            "min_keep": args.min_keep,
            "source_cache_read_only": str(source_cache),
            "experiment_cache": str(experiment_cache),
            "entity_index": str(Path(args.entity_index).resolve()) if args.entity_index else "default",
        },
        "aggregate": group(output_rows),
        "datasets": {
            dataset: group([row for row in output_rows if row["dataset"] == dataset])
            for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
        },
        "source_files": [
            {"path": str(path), "records": len(_read([path])), "md5": _md5(path)}
            for path in source_paths
        ],
        "rows": str(rows_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir / "run",
        extra={
            "experiment": "phase1_kg_funnel_replay",
            "report": str(report_path),
            "rows": str(rows_path),
            "submitted": len(source_rows),
            "source_cache_md5_before": _md5(source_cache),
            "experiment_cache_md5_after": _md5(experiment_cache),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
