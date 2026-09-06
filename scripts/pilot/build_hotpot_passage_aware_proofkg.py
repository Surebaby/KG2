#!/usr/bin/env python
"""HotpotQA planner-free passage-aware Proof-KG builder.

Fallback after the zero-shot planner failed.  Builds a precision-first KG from
the question + frozen passages only (no learned planner, no gold):
  mention extraction -> high-confidence linking -> deterministic relation-PID
  detection -> exact historical property fetch -> precision-first assembly.
Gold answers are never read.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from kgproweight.kg.entity_linker import EntityLinker, build_passage_titles, extract_mentions
from kgproweight.kg.historical_wikidata_retriever import HistoricalWikidataPropertyRetriever
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir, get_logger

logger = get_logger(__name__)

BUILDER_VERSION = "hotpot-passage-aware-proofkg-1"
_MAX_TRIPLES = 12

# Deterministic relation-intent PID rules (final-clause only, conservative).
_DIRECT_INTENT_RULES = (
    (r"who directed|which director|directed by|who .* director", {"P57"}),
    (r"birthplace|born in|place of birth|where .* born", {"P19"}),
    (r"birth date|date of birth|born on|when .* born", {"P569"}),
    (r"occupation|profession|what .* do|what capacity", {"P106"}),
    (r"country of citizenship|nationality|citizen of", {"P27"}),
    (r"country\b|which country|what country", {"P17"}),
    (r"screenwriter|wrote the screenplay", {"P58"}),
    (r"producer of|produced by|who produced", {"P162"}),
    (r"cast member|who played|who plays|who portrayed|starring", {"P161"}),
    (r"author of|who wrote|who authored", {"P50"}),
    (r"spouse|married to|who .* married", {"P26"}),
    (r"father of|whose father|who .* father", {"P22"}),
    (r"mother of|whose mother|who .* mother", {"P25"}),
)


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def detect_relation_pids(question: str) -> set:
    q = _norm(question)
    for pattern, pids in _DIRECT_INTENT_RULES:
        if re.search(pattern, q):
            return set(pids)
    return set()


def high_confidence_qids(linked: Sequence[Dict[str, Any]], min_score: float = 0.55, min_margin: float = 0.10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in linked:
        if row.get("abstained") or not row.get("qid"):
            continue
        if float(row.get("score") or 0.0) < min_score or float(row.get("margin") or 0.0) < min_margin:
            continue
        qid = str(row["qid"])
        if qid in seen:
            continue
        seen.add(qid)
        out.append(row)
    return out


def build_hotpot_passage_aware_kg(
    question: str,
    retrieved_passages: Sequence[Dict[str, Any]],
    linker: EntityLinker,
    retriever: HistoricalWikidataPropertyRetriever,
    title_resolver: WikipediaTitleResolver | None = None,
) -> tuple[List[list], Dict[str, Any]]:
    pids = detect_relation_pids(question)
    passage_titles = build_passage_titles(list(retrieved_passages))
    passage_text = " ".join(str(p.get("contents") or "") for p in retrieved_passages)

    mentions = extract_mentions(question, max_n=6)
    linked: List[Dict[str, Any]] = []
    for m in mentions:
        r = title_resolver.resolve(m) if title_resolver else None
        if not r or r.abstained or not r.selected_qid:
            r = linker.link_single(m, question=question, retrieved_titles=passage_titles, passage_text=passage_text)
        if r and not r.abstained and r.selected_qid:
            linked.append({"qid": r.selected_qid, "label": getattr(r, "selected_label", m), "score": getattr(r, "score", 1.0), "margin": getattr(r, "margin", 1.0)})

    anchors = high_confidence_qids(linked)
    if not pids or not anchors:
        return [], {"recognized": False, "reason": "no_intent_or_anchor", "pids": sorted(pids), "n_anchors": len(anchors)}

    triples: List[list] = []
    seen = set()
    q_norm = _norm(question)
    for anchor in anchors:
        edges = retriever.fetch_edges(str(anchor["qid"]), sorted(pids))
        for e in edges:
            t = (str(e["head_label"]), str(e["relation"]), str(e["tail_value"]))
            # anchor head must be in the question or passage evidence
            if _norm(t[0]) not in q_norm and _norm(t[0]) not in _norm(passage_text):
                continue
            if t in seen:
                continue
            seen.add(t)
            triples.append(list(t))
        if len(triples) >= _MAX_TRIPLES:
            break

    return triples[: _MAX_TRIPLES], {
        "recognized": bool(triples),
        "reason": "" if triples else "no_intent_matched_triples",
        "pids": sorted(pids),
        "n_anchors": len(anchors),
        "n_triples": len(triples),
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--entity_index", required=True)
    parser.add_argument("--entity_cache", required=True)
    parser.add_argument("--title_cache", required=True)
    parser.add_argument("--historical_property_cache", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--cutoff", default="2020-12-09T23:59:59Z")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.proof_input))
    run_dir, experiment_id = prepare_new_run_dir(args.run_dir, experiment_id=args.experiment_id,
                                                 extra={"phase": "build_hotpot_passage_aware_proofkg", "n": len(rows)})
    linker = EntityLinker(cache_path=args.entity_cache, offline=True, entity_index_path=args.entity_index)
    title_resolver = WikipediaTitleResolver(cache_path=args.title_cache, offline=True)
    retriever = HistoricalWikidataPropertyRetriever(cache_path=args.historical_property_cache, cutoff=args.cutoff, offline=True)

    records = []
    diag = []
    for r in rows:
        triples, d = build_hotpot_passage_aware_kg(
            question=r["question"], retrieved_passages=r["retrieved_passages"],
            linker=linker, retriever=retriever, title_resolver=title_resolver,
        )
        records.append(make_question_kg_record(
            dataset="hotpotqa", qid=r["qid"], question=r["question"], triples=triples,
            provenance={"builder_version": BUILDER_VERSION, "gold_access": False, "complete_plan_execution": d["recognized"]},
        ))
        diag.append({**r, "diagnostic": d, "n_triples": len(triples)})
        print(f"built {len(records)}/{len(rows)}", flush=True)

    (run_dir / "question_kg_records.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    (run_dir / "runtime_details.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in diag), encoding="utf-8")
    n = len(records)
    nonempty = sum(1 for x in records if x["kg_subgraph"])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "RUNTIME_KG_FROZEN_BEFORE_GOLD_AUDIT",
        "builder_version": BUILDER_VERSION,
        "counts": {"n": n, "recognized": nonempty, "nonempty": nonempty, "runtime_errors": 0},
        "rates": {"recognized": nonempty / n, "nonempty": nonempty / n},
        "gold_access": False,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(run_dir, extra={"experiment_id": experiment_id, "phase": "build_hotpot_passage_aware_proofkg", **report}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
