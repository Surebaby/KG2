#!/usr/bin/env python
"""Distil passage-aware KG v2 into a precision-first, additive KG v3.

V3 preserves the stored question KG byte-for-byte and appends at most four
new triples, never exceeding twelve total.  A delta triple must be anchored by
a high-confidence linked entity and its relation must directly match the
question intent.  Gold answers are never read by this builder.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kgproweight.kg.entity_linker import build_passage_titles
from kgproweight.kg.kg_filter import (
    _pid_for_triple,
    score_triple,
)
from kgproweight.utils.logging import dump_manifest


BUILDER_VERSION = "passage-aware-kg-override-3-precision-delta"
Triple = Tuple[str, str, str]
_GENERIC_ANCHORS = {
    "all", "american", "british", "canadian", "english", "experience",
    "film", "german", "germany", "king", "love", "president", "star",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _triples(values: Iterable[object]) -> List[Triple]:
    result: List[Triple] = []
    seen: set[Triple] = set()
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        item = tuple(str(part).strip() for part in value)
        if item not in seen:
            seen.add(item)
            result.append(item)  # type: ignore[arg-type]
    return result


def high_confidence_anchors(
    linked_entities: Sequence[Dict[str, Any]],
    *,
    min_score: float = 0.55,
    min_margin: float = 0.10,
) -> List[str]:
    anchors: List[str] = []
    for row in linked_entities:
        if row.get("abstained") or not row.get("qid"):
            continue
        if float(row.get("score") or 0.0) < min_score:
            continue
        if float(row.get("margin") or 0.0) < min_margin:
            continue
        for value in (row.get("mention"), row.get("label")):
            clean = _norm(value)
            if clean and clean not in _GENERIC_ANCHORS and clean not in anchors:
                anchors.append(clean)
    return anchors


def direct_answer_intent_pids(question: str) -> set[str]:
    """Conservative PID set for the relation requested by the final clause.

    The broad v2 scorer reacts to any keyword anywhere in a multi-hop question
    (for example ``starred``), which promotes bridge facts even when the final
    request is ``what capacity``.  V3 only recognises explicit answer intents;
    unknown forms abstain and preserve stored KG.
    """
    q = _norm(question)
    ordered_rules = (
        (r"what capacity|started (?:his|her|their) career|occupation|profession", {"P106", "P39"}),
        (r"based on what|based upon what", {"P144"}),
        (r"what .*award|which .*award|what award|which award", {"P166", "P1346"}),
        (r"who directed|which director|who was the director", {"P57"}),
        (r"bordered by|borders which|shares a border", {"P47"}),
        (r"are .* both writers|both .* writers", {"P106"}),
        (r"under which world leader", {"P6", "P35"}),
        (r"who was the father|whose father|father of", {"P22"}),
        (r"born first|born earlier|born later|older|younger", {"P19", "P569"}),
        (r"what country|which country|claimed by what country", {"P17", "P27", "P131"}),
        (r"in what .*film|which .*film .*star", {"P161"}),
    )
    for pattern, pids in ordered_rules:
        if re.search(pattern, q):
            return set(pids)
    return set()


def select_precision_delta(
    old_kg: Sequence[Triple],
    v2_kg: Sequence[Triple],
    *,
    question: str,
    linked_entities: Sequence[Dict[str, Any]],
    passage_titles: Sequence[str],
    max_delta: int = 4,
    max_total: int = 12,
) -> Tuple[List[Triple], Dict[str, Any]]:
    """Return stored KG plus a small, query-intent-matched delta."""
    old = _triples(old_kg)
    old_set = set(old)
    anchors = high_confidence_anchors(linked_entities)
    question_norm = _norm(question)
    direct_pids = direct_answer_intent_pids(question)
    budget = min(max_delta, max(0, max_total - len(old)))
    accepted: List[Tuple[float, Triple, Dict[str, Any]]] = []
    rejected: Counter[str] = Counter()

    for triple in _triples(v2_kg):
        if triple in old_set:
            rejected["already_in_stored_kg"] += 1
            continue
        head_norm = _norm(triple[0])
        if head_norm not in anchors:
            rejected["head_not_high_confidence_anchor"] += 1
            continue
        if head_norm not in question_norm:
            rejected["anchor_not_explicit_in_question"] += 1
            continue
        pid = _pid_for_triple(triple)
        if not direct_pids:
            rejected["unrecognized_direct_answer_intent"] += 1
            continue
        if pid not in direct_pids:
            rejected["relation_not_direct_question_intent"] += 1
            continue
        intent = 1.0
        relevance = score_triple(
            triple,
            question,
            pid=pid,
            question_entities=[value for value in anchors],
        )
        accepted.append(
            (
                float(intent) + float(relevance),
                triple,
                {
                    "triple": list(triple),
                    "pid": pid,
                    "intent_score": intent,
                    "relevance_score": round(float(relevance), 4),
                    "anchor": triple[0],
                },
            )
        )
    accepted.sort(key=lambda item: (-item[0], item[1]))
    selected = accepted[:budget]
    delta = [item[1] for item in selected]
    final = [*old, *delta]
    return final, {
        "n_stored": len(old),
        "delta_budget": budget,
        "n_v2": len(_triples(v2_kg)),
        "n_delta_candidates": len(accepted),
        "n_delta_selected": len(delta),
        "high_confidence_anchors": anchors,
        "direct_answer_intent_pids": sorted(direct_pids),
        "selected_delta": [item[2] for item in selected],
        "rejected": dict(rejected),
        "stored_prefix_preserved": final[: len(old)] == old,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2_overrides", required=True)
    parser.add_argument("--v2_report", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_delta", type=int, default=4)
    parser.add_argument("--max_total", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2_path = Path(args.v2_overrides).resolve()
    v2_report_path = Path(args.v2_report).resolve()
    silver_path = Path(args.silver).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    run_dir = Path(args.run_dir).resolve()
    for source in (v2_path, v2_report_path, silver_path):
        if not source.is_file():
            raise SystemExit(f"missing input file: {source}")
    for target in (output_path, report_path, run_dir):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")
    if args.max_delta != 4 or args.max_total != 12:
        raise SystemExit("v3 pilot is frozen at max_delta=4 and max_total=12")

    v2_rows = _read_jsonl(v2_path)
    v2_by_qid = {str(row.get("qid") or ""): row for row in v2_rows}
    v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    detail_by_qid = {str(row["qid"]): row for row in v2_report["details"]}
    if set(v2_by_qid) != set(detail_by_qid):
        raise SystemExit("v2 override/report qids differ")

    silver_by_qid: Dict[str, Dict[str, Any]] = {}
    with silver_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or row.get("id") or "")
            if qid in v2_by_qid:
                silver_by_qid[qid] = row
    if set(silver_by_qid) != set(v2_by_qid):
        raise SystemExit("some v2 qids are absent from silver")

    counts: Counter[str] = Counter()
    output_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for row in v2_rows:
        qid = str(row["qid"])
        source = silver_by_qid[qid]
        question = str(source.get("question") or "").strip()
        if question != str(row.get("question") or "").strip():
            raise SystemExit(f"question mismatch for {qid}")
        old_kg = _triples(source.get("kg_subgraph") or [])
        v2_kg = _triples(row.get("kg_subgraph") or [])
        passages = list(row.get("retrieved_passages") or [])[:15]
        final, selection = select_precision_delta(
            old_kg,
            v2_kg,
            question=question,
            linked_entities=detail_by_qid[qid].get("linked_entities") or [],
            passage_titles=build_passage_titles(passages),
            max_delta=args.max_delta,
            max_total=args.max_total,
        )
        if final[: len(old_kg)] != old_kg:
            raise SystemExit(f"stored KG prefix changed for {qid}")
        if len(final) > args.max_total or len(final) - len(old_kg) > args.max_delta:
            raise SystemExit(f"KG budget violation for {qid}")
        counts["questions"] += 1
        counts["stored_triples"] += len(old_kg)
        counts["v2_triples"] += len(v2_kg)
        counts["v3_triples"] += len(final)
        counts["delta_triples"] += selection["n_delta_selected"]
        counts["changed_questions"] += int(final != old_kg)
        counts["stored_empty"] += int(not old_kg)
        counts["v3_empty"] += int(not final)
        output_rows.append(
            {
                "qid": qid,
                "question": question,
                "retrieval_view": row.get("retrieval_view"),
                "retrieved_passages": passages,
                "kg_subgraph": [list(value) for value in final],
                "kg_builder_version": BUILDER_VERSION,
                "kg_selection_policy": "stored_prefix_plus_precision_delta",
            }
        )
        details.append({"qid": qid, "selection": selection})

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
            "stored_kg_preserved_as_prefix": True,
            "max_delta": args.max_delta,
            "max_total": args.max_total,
            "min_link_score": 0.55,
            "min_link_margin": 0.10,
            "delta_head_must_be_explicit_in_question": True,
            "relation_intent_required": 1.0,
        },
        "inputs": {
            "v2_overrides": str(v2_path),
            "v2_overrides_sha256": _sha256(v2_path),
            "v2_report": str(v2_report_path),
            "v2_report_sha256": _sha256(v2_report_path),
            "silver": str(silver_path),
            "silver_sha256": _sha256(silver_path),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
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
            "phase": "zero_training_passage_aware_kg_v3_build",
            "builder_version": BUILDER_VERSION,
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "output": report["output"],
            "counts": dict(counts),
        },
    )
    print(json.dumps({"counts": dict(counts), "output": report["output"]}, indent=2))


if __name__ == "__main__":
    main()
