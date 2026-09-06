#!/usr/bin/env python
"""Build offline passage-aware KG overrides for a fixed zero-training cohort.

The build mirrors Phase 1's entity-linking path: mentions come from the question
and top passage titles, candidate disambiguation sees passage titles/bodies, and
only cached Wikidata subgraphs are used.  It never reads gold answers and never
rewrites the source silver file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kgproweight.kg.entity_linker import (
    EntityLinker,
    build_passage_text,
    build_passage_titles,
)
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.training.phase1_distill import extract_mentions_robust
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "passage-aware-kg-override-1"
Triple = Tuple[str, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--silver", required=True, help="Read-only source of old KG by qid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--prompt_passages", type=int, default=15)
    parser.add_argument("--max_mentions", type=int, default=8)
    parser.add_argument("--min_keep", type=int, default=5)
    parser.add_argument("--max_keep", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passages_path = Path(args.passages).resolve()
    silver_path = Path(args.silver).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for path in (output_path, report_path, run_dir):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing path: {path}")
    if args.prompt_passages != 15 or args.max_keep != 12 or args.min_keep != 5:
        raise SystemExit("E2 protocol is frozen at passages=15, min_keep=5, max_keep=12")

    passage_rows = _read_jsonl(passages_path)
    passage_by_qid = {str(row.get("qid") or ""): row for row in passage_rows}
    if not passage_by_qid or "" in passage_by_qid:
        raise SystemExit("passage overrides require non-empty qids")
    if len(passage_by_qid) != len(passage_rows):
        raise SystemExit("duplicate qids in passage overrides")

    silver_by_qid: Dict[str, Dict[str, Any]] = {}
    with silver_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or row.get("id") or "")
            if qid in passage_by_qid:
                silver_by_qid[qid] = row
    missing = sorted(set(passage_by_qid) - set(silver_by_qid))
    if missing:
        raise SystemExit(f"passage qids absent from silver: {missing}")

    linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
    kg = WikidataSubgraphRetriever(
        max_hops=2,
        max_neighbors=30,
        cache_dir=resolve_kg_cache_dir(),
        offline=True,
        relation_filter=_QA_RELATION_FILTER,
    )

    output_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for passage_row in passage_rows:
        qid = str(passage_row["qid"])
        source = silver_by_qid[qid]
        question = str(source.get("question") or "").strip()
        if question != str(passage_row.get("question") or "").strip():
            raise SystemExit(f"question mismatch for {qid}")
        passages = list(passage_row.get("retrieved_passages") or [])[: args.prompt_passages]
        mentions = extract_mentions_robust(question, passages=passages, max_n=args.max_mentions)
        titles = build_passage_titles(passages)
        passage_text = build_passage_text(passages)

        linked: List[Dict[str, Any]] = []
        linked_qids: List[str] = []
        for mention in mentions:
            result = linker.link_single(
                mention,
                question=question,
                retrieved_titles=titles,
                passage_text=passage_text,
            )
            linked.append(
                {
                    "mention": mention,
                    "qid": result.selected_qid,
                    "label": result.selected_label,
                    "score": round(float(result.score), 4),
                    "margin": round(float(result.margin), 4),
                    "abstained": bool(result.abstained),
                    "abstain_reason": result.abstain_reason,
                }
            )
            if result.selected_qid and not result.abstained:
                linked_qids.append(result.selected_qid)
        linked_qids = list(dict.fromkeys(linked_qids))
        raw_triples = kg.fetch(linked_qids) if linked_qids else []
        new_kg: List[Triple] = filter_and_rank_triples(
            raw_triples,
            question=question,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
        )
        old_kg = [
            tuple(str(part) for part in value)
            for value in (source.get("kg_subgraph") or [])
            if isinstance(value, (list, tuple)) and len(value) == 3
        ]

        counts["questions"] += 1
        counts["mentions"] += len(mentions)
        counts["linked_mentions"] += sum(
            bool(row["qid"] and not row["abstained"]) for row in linked
        )
        counts["old_triples"] += len(old_kg)
        counts["new_raw_triples"] += len(raw_triples)
        counts["new_triples"] += len(new_kg)
        counts["old_empty"] += int(not old_kg)
        counts["new_empty"] += int(not new_kg)
        counts["kg_changed"] += int(old_kg != new_kg)

        output_rows.append(
            {
                "qid": qid,
                "question": question,
                "retrieval_view": passage_row.get("retrieval_view"),
                "retrieved_passages": passages,
                "kg_subgraph": [list(value) for value in new_kg],
            }
        )
        details.append(
            {
                "qid": qid,
                "mentions": mentions,
                "linked_entities": linked,
                "linked_qids": linked_qids,
                "n_old_triples": len(old_kg),
                "n_raw_cached_triples": len(raw_triples),
                "n_new_triples": len(new_kg),
                "old_empty": not old_kg,
                "new_empty": not new_kg,
                "kg_changed": old_kg != new_kg,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "BUILT_NOT_EVALUATED",
        "builder_version": BUILDER_VERSION,
        "protocol": {
            "gold_used_for_build": False,
            "offline_only": True,
            "prompt_passages": args.prompt_passages,
            "max_mentions": args.max_mentions,
            "min_keep": args.min_keep,
            "max_keep": args.max_keep,
            "mention_source": "question + top-5 retrieved passage titles",
            "linker_context": "question + top-15 passage titles/bodies",
        },
        "inputs": {
            "passages": str(passages_path),
            "passages_sha256": _sha256(passages_path),
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
            "silver_read_only": True,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
        },
        "counts": dict(counts),
        "details": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "zero_training_e2_input_build",
            "builder_version": BUILDER_VERSION,
            "passages": str(passages_path),
            "passages_sha256": report["inputs"]["passages_sha256"],
            "silver": str(silver_path),
            "silver_sha256": report["inputs"]["silver_sha256"],
            "output": str(output_path),
            "output_sha256": report["output"]["sha256"],
            "counts": dict(counts),
        },
    )
    print(json.dumps({"counts": dict(counts), "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
