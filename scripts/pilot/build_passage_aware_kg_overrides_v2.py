#!/usr/bin/env python
"""Build additive, passage-aware KG overrides for a frozen zero-training cohort.

Version 2 differs from the earlier pilot in three research-relevant ways:

* the offline entity-description index is explicit and checksummed;
* the source entity cache is copied before linking, so the source stays read-only;
* old question KG and new passage-derived KG are merged, with a non-empty old-KG
  fallback and an optional chain-aware 12-triple selector.

The builder never reads gold answers and refuses to overwrite any artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kgproweight.kg.entity_linker import (
    EntityLinker,
    build_passage_text,
    build_passage_titles,
)
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.kg.wikidata_retriever import (
    _QA_RELATION_FILTER,
    WikidataSubgraphRetriever,
)
from kgproweight.retrieval.bootstrap import (
    resolve_entity_cache_path,
    resolve_kg_cache_dir,
)
from kgproweight.training.phase1_distill import extract_mentions_robust
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "passage-aware-kg-override-2"
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


def _triple(value: object) -> Triple | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _dedupe_triples(values: Iterable[object]) -> List[Triple]:
    result: List[Triple] = []
    seen: set[Triple] = set()
    for value in values:
        item = _triple(value)
        if item is None or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _endpoint_matches(endpoint: str, anchors: Sequence[str]) -> bool:
    endpoint_norm = _norm(endpoint)
    if not endpoint_norm:
        return False
    for anchor in anchors:
        anchor_norm = _norm(anchor)
        if not anchor_norm:
            continue
        if endpoint_norm == anchor_norm:
            return True
        if len(endpoint_norm) >= 4 and endpoint_norm in anchor_norm:
            return True
        if len(anchor_norm) >= 4 and anchor_norm in endpoint_norm:
            return True
    return False


def _touches(triple: Triple, anchors: Sequence[str]) -> bool:
    return _endpoint_matches(triple[0], anchors) or _endpoint_matches(triple[2], anchors)


def select_additive_kg(
    old_kg: Sequence[Triple],
    passage_kg: Sequence[Triple],
    *,
    question: str,
    question_mentions: Sequence[str],
    passage_titles: Sequence[str],
    max_keep: int = 12,
    min_keep: int = 5,
    selection_policy: str = "chain_aware",
) -> Tuple[List[Triple], Dict[str, Any]]:
    """Merge and select KG triples without allowing a new empty result.

    ``chain_aware`` reserves up to four slots each for question anchors,
    passage-title anchors, and triples connected to already selected nodes.
    Remaining slots follow the existing question-aware ranking.  This is a
    selection policy only; it does not invent edges or use gold labels.
    """
    old = _dedupe_triples(old_kg)
    passage = _dedupe_triples(passage_kg)
    merged = _dedupe_triples([*old, *passage])
    old_set = set(old)
    passage_set = set(passage)

    pool_rich = filter_and_rank_triples(
        merged,
        question=question,
        max_keep=max(max_keep * 4, max_keep),
        min_keep=min_keep,
        rich=True,
        question_entities=list(question_mentions),
    )
    ranked: List[Triple] = [
        (str(row["h"]), str(row["r"]), str(row["t"])) for row in pool_rich
    ]

    if selection_policy not in {"ranked", "chain_aware"}:
        raise ValueError(f"unsupported selection policy: {selection_policy}")

    if selection_policy == "ranked":
        selected = ranked[:max_keep]
        bucket_counts = {"question_anchor": 0, "passage_anchor": 0, "connected": 0}
    else:
        selected: List[Triple] = []
        selected_set: set[Triple] = set()
        bucket_counts: Counter[str] = Counter()

        def take(bucket: str, predicate, limit: int) -> None:
            for item in ranked:
                if len(selected) >= max_keep or bucket_counts[bucket] >= limit:
                    break
                if item in selected_set or not predicate(item):
                    continue
                selected.append(item)
                selected_set.add(item)
                bucket_counts[bucket] += 1

        take("question_anchor", lambda item: _touches(item, question_mentions), 4)
        take("passage_anchor", lambda item: _touches(item, passage_titles), 4)

        selected_nodes = {part for item in selected for part in (item[0], item[2])}
        take("connected", lambda item: _touches(item, tuple(selected_nodes)), 4)
        take("ranked_fill", lambda item: True, max_keep)

    fallback_used = False
    if not selected and old:
        # Monotonic non-empty guarantee.  Old KG is already part of the source
        # silver artifact; v2 must never replace it with an empty generated KG.
        selected = old[:max_keep]
        fallback_used = True

    provenance = []
    for item in selected:
        sources = []
        if item in old_set:
            sources.append("old_question_kg")
        if item in passage_set:
            sources.append("passage_kg")
        provenance.append({"triple": list(item), "sources": sources})

    diagnostics = {
        "n_old_unique": len(old),
        "n_passage_unique": len(passage),
        "n_merged_unique": len(merged),
        "n_ranked_pool": len(ranked),
        "n_selected": len(selected),
        "n_selected_old": sum(item in old_set for item in selected),
        "n_selected_passage": sum(item in passage_set for item in selected),
        "fallback_to_old": fallback_used,
        "bucket_counts": dict(bucket_counts),
        "provenance": provenance,
    }
    return selected, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--silver", required=True, help="Read-only old KG source")
    parser.add_argument("--entity_index", required=True)
    parser.add_argument(
        "--selection_jsonl",
        default=None,
        help="Optional qid-bearing JSONL used only to freeze a subset of passages",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--prompt_passages", type=int, default=15)
    parser.add_argument("--max_mentions", type=int, default=8)
    parser.add_argument("--min_keep", type=int, default=5)
    parser.add_argument("--max_keep", type=int, default=12)
    parser.add_argument(
        "--selection_policy", choices=("ranked", "chain_aware"), default="chain_aware"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passages_path = Path(args.passages).resolve()
    silver_path = Path(args.silver).resolve()
    entity_index_path = Path(args.entity_index).resolve()
    selection_path = Path(args.selection_jsonl).resolve() if args.selection_jsonl else None
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for source in (passages_path, silver_path, entity_index_path):
        if not source.is_file():
            raise SystemExit(f"missing input file: {source}")
    if selection_path is not None and not selection_path.is_file():
        raise SystemExit(f"missing selection file: {selection_path}")
    for target in (output_path, report_path, run_dir):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")
    if args.prompt_passages != 15 or args.max_keep != 12 or args.min_keep != 5:
        raise SystemExit("v2 pilot is frozen at passages=15, min_keep=5, max_keep=12")

    passage_rows = _read_jsonl(passages_path)
    if selection_path is not None:
        selection_rows = _read_jsonl(selection_path)
        selected_qids = [str(row.get("qid") or row.get("id") or "") for row in selection_rows]
        if not selected_qids or "" in selected_qids or len(set(selected_qids)) != len(selected_qids):
            raise SystemExit("selection JSONL requires unique non-empty qids")
        passage_all = {str(row.get("qid") or ""): row for row in passage_rows}
        missing_selection = sorted(set(selected_qids) - set(passage_all))
        if missing_selection:
            raise SystemExit(f"selected qids absent from passages: {missing_selection}")
        passage_rows = [passage_all[qid] for qid in selected_qids]
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

    run_dir.mkdir(parents=True, exist_ok=False)
    source_cache = Path(resolve_entity_cache_path()).resolve()
    experiment_cache = run_dir / "entity_cache.snapshot.jsonl"
    if source_cache.is_file():
        shutil.copyfile(source_cache, experiment_cache)
    else:
        experiment_cache.touch()
    linker = EntityLinker(
        cache_path=str(experiment_cache),
        offline=True,
        entity_index_path=str(entity_index_path),
    )
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
    abstentions: Counter[str] = Counter()
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
            if result.abstained:
                abstentions[result.abstain_reason.split(" (")[0] or "unknown"] += 1
            if result.selected_qid and not result.abstained:
                linked_qids.append(result.selected_qid)
        linked_qids = list(dict.fromkeys(linked_qids))

        raw_triples = kg.fetch(linked_qids) if linked_qids else []
        passage_kg: List[Triple] = filter_and_rank_triples(
            raw_triples,
            question=question,
            min_keep=args.min_keep,
            max_keep=args.max_keep * 4,
            question_entities=mentions,
        )
        old_kg = _dedupe_triples(source.get("kg_subgraph") or [])
        final_kg, selection = select_additive_kg(
            old_kg,
            passage_kg,
            question=question,
            question_mentions=mentions,
            passage_titles=titles,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
            selection_policy=args.selection_policy,
        )

        counts["questions"] += 1
        counts["mentions"] += len(mentions)
        counts["linked_mentions"] += sum(
            bool(row["qid"] and not row["abstained"]) for row in linked
        )
        counts["old_triples"] += len(old_kg)
        counts["raw_triples"] += len(raw_triples)
        counts["passage_triples"] += len(passage_kg)
        counts["final_triples"] += len(final_kg)
        counts["old_empty"] += int(not old_kg)
        counts["passage_empty"] += int(not passage_kg)
        counts["final_empty"] += int(not final_kg)
        counts["fallback_to_old"] += int(selection["fallback_to_old"])
        counts["kg_changed"] += int(old_kg != final_kg)
        counts["selected_old"] += int(selection["n_selected_old"])
        counts["selected_passage"] += int(selection["n_selected_passage"])

        output_rows.append(
            {
                "qid": qid,
                "question": question,
                "retrieval_view": passage_row.get("retrieval_view"),
                "retrieved_passages": passages,
                "kg_subgraph": [list(value) for value in final_kg],
                "kg_builder_version": BUILDER_VERSION,
                "kg_selection_policy": args.selection_policy,
            }
        )
        details.append(
            {
                "qid": qid,
                "mentions": mentions,
                "linked_entities": linked,
                "linked_qids": linked_qids,
                "n_raw_cached_triples": len(raw_triples),
                "selection": selection,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "BUILT_NOT_MODEL_EVALUATED",
        "builder_version": BUILDER_VERSION,
        "protocol": {
            "training": "none",
            "gold_used_for_build": False,
            "offline_only": True,
            "source_entity_cache_read_only": True,
            "prompt_passages": args.prompt_passages,
            "max_mentions": args.max_mentions,
            "min_keep": args.min_keep,
            "max_keep": args.max_keep,
            "merge_policy": "additive_old_union_passage_with_old_nonempty_fallback",
            "selection_policy": args.selection_policy,
            "mention_source": "question + top-5 retrieved passage titles",
            "linker_context": "question + top-15 passage titles/bodies",
        },
        "inputs": {
            "passages": str(passages_path),
            "passages_sha256": _sha256(passages_path),
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
            "silver_read_only": True,
            "source_entity_cache": str(source_cache),
            "experiment_entity_cache": str(experiment_cache),
            "entity_index": str(entity_index_path),
            "entity_index_sha256": _sha256(entity_index_path),
            "selection_jsonl": str(selection_path) if selection_path else None,
            "selection_jsonl_sha256": _sha256(selection_path) if selection_path else None,
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "counts": dict(counts),
        "link_rate": counts["linked_mentions"] / max(1, counts["mentions"]),
        "abstention_reasons": dict(abstentions),
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
            "phase": "zero_training_passage_aware_kg_v2_build",
            "builder_version": BUILDER_VERSION,
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "output": report["output"],
            "counts": dict(counts),
            "link_rate": report["link_rate"],
            "abstention_reasons": report["abstention_reasons"],
        },
    )
    print(
        json.dumps(
            {
                "counts": dict(counts),
                "link_rate": report["link_rate"],
                "abstention_reasons": report["abstention_reasons"],
                "output": report["output"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
