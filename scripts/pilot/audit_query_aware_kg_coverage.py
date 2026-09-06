#!/usr/bin/env python
"""Audit query-specific KG relation/value/proof-chain coverage on a frozen cohort.

The current KG is built from the question only, with the project's offline
entity linker, subgraph cache and existing Top-12 policy.  Dataset gold
evidence is read only after that build and is used solely for post-build audit.
No model inference or parameter update is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kgproweight.kg.kg_filter import _pid_for_triple, filter_and_rank_triples
from kgproweight.utils.logging import dump_manifest


_question_kg_builder = importlib.import_module("scripts.prepare.06_build_question_kg_index")
_build_components = _question_kg_builder._build_components
_resolve_one = _question_kg_builder._resolve_one


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


def _norm(value: object) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.lower()))


def _surface_match(left: object, right: object) -> bool:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    return a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a)


_VALUE_ALIASES = {
    "american": "united states", "america": "united states", "u s": "united states",
    "us": "united states", "british": "united kingdom", "english": "united kingdom",
    "french": "france", "italian": "italy", "indian": "india", "german": "germany",
    "canadian": "canada", "australian": "australia", "spanish": "spain",
    "austrian": "austria", "swedish": "sweden", "dutch": "netherlands",
    "argentine": "argentina", "chinese": "china", "russian": "russia",
    "mexican": "mexico", "hungarian": "hungary", "danish": "denmark",
    "norwegian": "norway", "polish": "poland", "czech": "czech republic",
    "brazilian": "brazil", "belgian": "belgium", "irish": "ireland",
    "swiss": "switzerland", "turkish": "turkey", "finnish": "finland",
    "filipino": "philippines", "greek": "greece", "portuguese": "portugal",
    "israeli": "israel", "romanian": "romania", "serbian": "serbia",
    "croatian": "croatia", "ukrainian": "ukraine", "chilean": "chile",
    "pakistani": "pakistan", "nigerian": "nigeria", "egyptian": "egypt",
    "bangladeshi": "bangladesh", "indonesian": "indonesia", "soviet": "soviet union",
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _date_signature(value: object) -> Tuple[int, int | None, int | None] | None:
    text = _norm(value)
    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
    if not year_match:
        return None
    year = int(year_match.group(1))
    month = next((number for name, number in _MONTHS.items() if name in text.split()), None)
    day = None
    if month is not None:
        numbers = [int(value) for value in re.findall(r"\b([0-9]{1,2})\b", text)]
        day = next((value for value in numbers if 1 <= value <= 31), None)
    return year, month, day


def _value_match(left: object, right: object) -> bool:
    if _surface_match(left, right):
        return True
    a = _VALUE_ALIASES.get(_norm(left), _norm(left))
    b = _VALUE_ALIASES.get(_norm(right), _norm(right))
    if a and a == b:
        return True
    date_a, date_b = _date_signature(left), _date_signature(right)
    if date_a and date_b:
        # Match at the precision shared by both values; do not invent a missing
        # month/day when an asset stores only a year.
        return all(x == y for x, y in zip(date_a, date_b) if x is not None and y is not None)
    return False


def _triple_pid(triple: Triple) -> str:
    return _pid_for_triple(triple)


def _relation_target(label: str) -> Dict[str, List[str]]:
    clean = _norm(label)
    pid = _pid_for_triple(("head", label, "tail"))
    aliases = {clean}
    alias_groups = {
        "owned by": {"owned by", "owner"},
        "author": {"author", "written by"},
        "has part": {"has part", "has part s"},
        "located in the administrative territorial entity": {
            "located in the administrative territorial entity",
            "located in administrative territorial entity",
        },
    }
    aliases.update(alias_groups.get(clean, set()))
    return {"pids": [pid] if pid else [], "labels": sorted(aliases)}


def _infer_question_relation(question: str) -> Dict[str, List[str]]:
    """Conservative intent mapping; unknown wording remains unevaluable."""
    q = _norm(question)
    rules: Sequence[Tuple[str, Sequence[str], Sequence[str]]] = (
        (r"when .* born|what year .* born", ("P569",), ("date of birth",)),
        (r"born first|born earlier|born later|older|younger", ("P569",), ("date of birth",)),
        (r"where .* born|place of birth|what city .* born", ("P19",), ("place of birth",)),
        (r"when .* died|date of death", ("P570",), ("date of death",)),
        (r"where .* died|place of death|city .* die", ("P20",), ("place of death",)),
        (r"when .* founded|when .* formed|year .* formation|started first|founded first", ("P571",), ("inception",)),
        (r"who .* mother|mother of|whose mother", ("P25",), ("mother",)),
        (r"who .* father|father of|whose father|fathered", ("P22",), ("father",)),
        (r"spouse|married|wife|husband", ("P26",), ("spouse",)),
        (r"who directed|director of|which director", ("P57",), ("director",)),
        (r"who wrote|author of|which author", ("P50",), ("author",)),
        (r"composer|composed", ("P86",), ("composer",)),
        (r"performer|performed|sang", ("P175",), ("performer",)),
        (r"country of citizenship|nationality|citizen", ("P27",), ("country of citizenship",)),
        (r"what country|which country|in which country", ("P17", "P27", "P495"), ("country", "country of citizenship", "country of origin")),
        (r"owned by|owner of", ("P127",), ("owned by",)),
        (r"founded by|founder", ("P112",), ("founded by",)),
        (r"headquarters", ("P159",), ("headquarters location",)),
        (r"what city|where is .* located|located in", ("P131", "P276"), ("located in the administrative territorial entity", "location")),
        (r"capital of|what is the capital", ("P36", "P1376"), ("capital", "capital of")),
        (r"population|how many .* live", ("P1082",), ("population",)),
        (r"educated at|college did|university did|school did", ("P69",), ("educated at",)),
        (r"employer|worked for", ("P108",), ("employer",)),
        (r"occupation|profession|career|what capacity", ("P106", "P39"), ("occupation", "position held")),
        (r"record label", ("P264",), ("record label",)),
        (r"member of sports team|which team|what club", ("P54",), ("member of sports team",)),
        (r"shares border|bordered by|borders", ("P47",), ("shares border with",)),
        (r"award", ("P166",), ("award received",)),
        (r"cause of death", ("P509",), ("cause of death",)),
        (r"publisher|published by", ("P123",), ("publisher",)),
        (r"producer|produced by", ("P162", "P176"), ("producer", "manufacturer")),
        (r"cast member|starring|starred", ("P161",), ("cast member",)),
    )
    for pattern, pids, labels in rules:
        if re.search(pattern, q):
            return {"pids": list(pids), "labels": [_norm(label) for label in labels]}
    return {"pids": [], "labels": []}


def _replace_placeholders(text: str, prior_answers: Sequence[str]) -> str:
    result = str(text)
    for index, answer in enumerate(prior_answers, start=1):
        result = result.replace(f"#{index}", answer)
    return result


def _reference_hops(dataset: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if dataset == "2wikimultihopqa":
        evidence = row.get("metadata", {}).get("evidences", {})
        heads = evidence.get("fact") or []
        relations = evidence.get("relation") or []
        tails = evidence.get("entity") or []
        return [
            {
                "head": str(head),
                "tail": str(tail),
                "target": _relation_target(str(relation)),
                "relation_source": "dataset_evidence_exact",
            }
            for head, relation, tail in zip(heads, relations, tails)
        ]
    if dataset == "musique":
        decomposition = row.get("metadata", {}).get("metadata", {}).get("question_decomposition", [])
        prior_answers: List[str] = []
        hops: List[Dict[str, Any]] = []
        for item in decomposition:
            raw_question = str(item.get("question") or "")
            resolved = _replace_placeholders(raw_question, prior_answers)
            answer = str(item.get("answer") or "")
            if ">>" in resolved:
                head, relation = (part.strip() for part in resolved.split(">>", 1))
                target = _relation_target(relation)
                source = "dataset_decomposition_exact_relation"
            else:
                head = resolved
                target = _infer_question_relation(resolved)
                source = "dataset_decomposition_rule_relation"
            hops.append(
                {
                    "head": head,
                    "tail": answer,
                    "target": target,
                    "relation_source": source,
                }
            )
            prior_answers.append(answer)
        return hops
    return []


def _expected_explicit_anchors(
    dataset: str,
    row: Dict[str, Any],
    question: str,
    reference_hops: Sequence[Dict[str, Any]],
) -> List[str]:
    """Gold-side audit anchors that are explicitly named in the question."""
    candidates: List[str] = []
    if dataset == "hotpotqa":
        candidates.extend(row.get("metadata", {}).get("supporting_facts", {}).get("title") or [])
    elif dataset == "2wikimultihopqa":
        for hop in reference_hops:
            candidates.extend([hop.get("head", ""), hop.get("tail", "")])
    elif dataset == "musique":
        candidates.extend(
            hop.get("head", "")
            for hop in reference_hops
            if hop.get("relation_source") == "dataset_decomposition_exact_relation"
        )
    anchors: List[str] = []
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if clean and _surface_match(clean, question) and clean not in anchors:
            anchors.append(clean)
    return anchors


def _relation_matches(triple: Triple, target: Dict[str, List[str]]) -> bool:
    pid = _triple_pid(triple)
    if pid and pid in target.get("pids", []):
        return True
    return _norm(triple[1]) in set(target.get("labels", []))


def _hop_metrics(hop: Dict[str, Any], triples: Sequence[Triple]) -> Dict[str, Any]:
    target = hop["target"]
    evaluable = bool(target.get("pids") or target.get("labels")) and bool(_norm(hop.get("tail")))
    relation_rows = [triple for triple in triples if _relation_matches(triple, target)]
    value_rows = [triple for triple in triples if _value_match(triple[2], hop.get("tail"))]
    relation_value_rows = [
        triple for triple in relation_rows if _value_match(triple[2], hop.get("tail"))
    ]
    exact_rows = [
        triple for triple in relation_value_rows
        if _surface_match(triple[0], hop.get("head"))
    ]
    return {
        "evaluable": evaluable,
        "relation_hit": bool(relation_rows) if evaluable else None,
        "value_hit": bool(value_rows) if evaluable else None,
        "relation_value_hit": bool(relation_value_rows) if evaluable else None,
        "exact_hop_hit": bool(exact_rows) if evaluable else None,
        "relation_value_matches": [list(triple) for triple in relation_value_rows[:3]],
        "exact_matches": [list(triple) for triple in exact_rows[:3]],
    }


def _chain_summary(hops: Sequence[Dict[str, Any]], triples: Sequence[Triple]) -> Dict[str, Any]:
    metrics = [_hop_metrics(hop, triples) for hop in hops]
    evaluable = bool(metrics) and all(item["evaluable"] for item in metrics)
    return {
        "evaluable": evaluable,
        "n_hops": len(metrics),
        "n_evaluable_hops": sum(bool(item["evaluable"]) for item in metrics),
        "all_relation_hit": all(bool(item["relation_hit"]) for item in metrics) if evaluable else None,
        "all_value_hit": all(bool(item["value_hit"]) for item in metrics) if evaluable else None,
        "all_relation_value_hit": all(bool(item["relation_value_hit"]) for item in metrics) if evaluable else None,
        "all_exact_hop_hit": all(bool(item["exact_hop_hit"]) for item in metrics) if evaluable else None,
        "hops": metrics,
    }


def _answer_component_hit(answers: Sequence[str], triples: Sequence[Triple]) -> bool | None:
    targets = [answer for answer in answers if _norm(answer) not in {"", "yes", "no"}]
    if not targets:
        return None
    return any(_surface_match(part, answer) for triple in triples for part in triple for answer in targets)


def _failure_class(
    linked_ok: int,
    raw_chain: Dict[str, Any],
    selected_chain: Dict[str, Any],
) -> str:
    if not raw_chain["evaluable"]:
        return "REFERENCE_RELATION_UNKNOWN"
    if selected_chain["all_relation_value_hit"]:
        return "SELECTED_FULL_RELATION_VALUE_CHAIN"
    if raw_chain["all_relation_value_hit"]:
        return "SELECTOR_DROPPED_REQUIRED_HOP"
    if linked_ok == 0:
        return "ENTITY_LINK_FAILURE"
    if not raw_chain["all_relation_hit"]:
        return "RAW_MISSING_REQUIRED_RELATION"
    return "RAW_RELATION_PRESENT_VALUE_MISSING"


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    raw_triples = selected_triples = linked_mentions = linked_ok = 0
    expected_anchors = linked_expected_anchors = 0
    for row in rows:
        counts["questions"] += 1
        counts[f"failure::{row['failure_class']}"] += 1
        counts["raw_empty"] += int(row["n_raw_triples"] == 0)
        counts["selected_empty"] += int(row["n_selected_triples"] == 0)
        counts["intent_recognized"] += int(row["final_intent_recognized"])
        counts["raw_intent_relation_hit"] += int(row["raw_intent_relation_hit"] is True)
        counts["selected_intent_relation_hit"] += int(row["selected_intent_relation_hit"] is True)
        counts["raw_answer_component_hit"] += int(row["raw_answer_component_hit"] is True)
        counts["selected_answer_component_hit"] += int(row["selected_answer_component_hit"] is True)
        expected_anchors += row["n_expected_explicit_anchors"]
        linked_expected_anchors += row["n_linked_expected_explicit_anchors"]
        if row["raw_chain"]["evaluable"]:
            counts["proof_evaluable"] += 1
            counts["raw_full_relation_value_chain"] += int(row["raw_chain"]["all_relation_value_hit"] is True)
            counts["selected_full_relation_value_chain"] += int(row["selected_chain"]["all_relation_value_hit"] is True)
            counts["raw_full_exact_chain"] += int(row["raw_chain"]["all_exact_hop_hit"] is True)
            counts["selected_full_exact_chain"] += int(row["selected_chain"]["all_exact_hop_hit"] is True)
            for prefix in ("raw", "selected"):
                for hop in row[f"{prefix}_chain"]["hops"]:
                    if not hop["evaluable"]:
                        continue
                    counts[f"{prefix}_evaluable_hops"] += 1
                    counts[f"{prefix}_hop_relation_hit"] += int(hop["relation_hit"] is True)
                    counts[f"{prefix}_hop_value_hit"] += int(hop["value_hit"] is True)
                    counts[f"{prefix}_hop_relation_value_hit"] += int(hop["relation_value_hit"] is True)
                    counts[f"{prefix}_hop_exact_hit"] += int(hop["exact_hop_hit"] is True)
        raw_triples += row["n_raw_triples"]
        selected_triples += row["n_selected_triples"]
        linked_mentions += row["n_linked_mentions"]
        linked_ok += row["n_linked_ok"]
    n = max(1, counts["questions"])
    proof_n = max(1, counts["proof_evaluable"])
    return {
        "counts": dict(counts),
        "rates": {
            "linked_mention_rate": linked_ok / max(1, linked_mentions),
            "explicit_gold_anchor_link_recall": linked_expected_anchors / max(1, expected_anchors),
            "raw_empty_rate": counts["raw_empty"] / n,
            "selected_empty_rate": counts["selected_empty"] / n,
            "intent_recognized_rate": counts["intent_recognized"] / n,
            "raw_intent_relation_hit_rate_all": counts["raw_intent_relation_hit"] / n,
            "selected_intent_relation_hit_rate_all": counts["selected_intent_relation_hit"] / n,
            "raw_answer_component_hit_rate": counts["raw_answer_component_hit"] / n,
            "selected_answer_component_hit_rate": counts["selected_answer_component_hit"] / n,
            "proof_evaluable_rate": counts["proof_evaluable"] / n,
            "raw_full_relation_value_chain_rate_evaluable": counts["raw_full_relation_value_chain"] / proof_n,
            "selected_full_relation_value_chain_rate_evaluable": counts["selected_full_relation_value_chain"] / proof_n,
            "raw_full_exact_chain_rate_evaluable": counts["raw_full_exact_chain"] / proof_n,
            "selected_full_exact_chain_rate_evaluable": counts["selected_full_exact_chain"] / proof_n,
            "raw_hop_relation_hit_rate": counts["raw_hop_relation_hit"] / max(1, counts["raw_evaluable_hops"]),
            "raw_hop_relation_value_hit_rate": counts["raw_hop_relation_value_hit"] / max(1, counts["raw_evaluable_hops"]),
            "selected_hop_relation_hit_rate": counts["selected_hop_relation_hit"] / max(1, counts["selected_evaluable_hops"]),
            "selected_hop_relation_value_hit_rate": counts["selected_hop_relation_value_hit"] / max(1, counts["selected_evaluable_hops"]),
        },
        "means": {
            "raw_triples": raw_triples / n,
            "selected_triples": selected_triples / n,
            "linked_mentions": linked_mentions / n,
            "linked_ok": linked_ok / n,
            "expected_explicit_gold_anchors": expected_anchors / n,
            "linked_expected_explicit_gold_anchors": linked_expected_anchors / n,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort_manifest", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--kg_overrides", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_mentions", type=int, default=5)
    parser.add_argument("--max_keep", type=int, default=12)
    parser.add_argument("--min_keep", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = Path(args.cohort).resolve()
    cohort_manifest_path = Path(args.cohort_manifest).resolve()
    output_path = Path(args.output).resolve()
    details_path = Path(args.details).resolve()
    overrides_path = Path(args.kg_overrides).resolve()
    run_dir = Path(args.run_dir).resolve()
    for target in (output_path, details_path, overrides_path, run_dir):
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing path: {target}")
    cohort = _read_jsonl(cohort_path)
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    if _sha256(cohort_path) != cohort_manifest["cohort"]["sha256"]:
        raise SystemExit("cohort hash differs from its frozen manifest")
    if len(cohort) != 150:
        raise SystemExit(f"expected frozen n=150 cohort, got {len(cohort)}")

    by_dataset_id: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_hashes: Dict[str, Dict[str, str]] = {}
    selected_ids: Dict[str, set[str]] = defaultdict(set)
    for row in cohort:
        selected_ids[row["dataset"]].add(row["source_id"])
    data_root = Path(args.data_root).resolve()
    for dataset, ids in selected_ids.items():
        path = data_root / dataset / "train.jsonl"
        for row in _read_jsonl(path):
            qid = str(row.get("id") or "")
            if qid in ids:
                by_dataset_id[(dataset, qid)] = row
        source_hashes[dataset] = {"path": str(path), "sha256": _sha256(path)}
    if len(by_dataset_id) != len(cohort):
        raise SystemExit("some frozen cohort ids are absent from source train JSONL")

    linker, kg = _build_components(offline=True)
    details: List[Dict[str, Any]] = []
    override_rows: List[Dict[str, Any]] = []
    for index, frozen in enumerate(cohort, start=1):
        dataset = frozen["dataset"]
        source_id = frozen["source_id"]
        source = by_dataset_id[(dataset, source_id)]
        question = frozen["question"]

        # Gold-free construction boundary: only question/dataset/id enter here.
        resolved = _resolve_one(question, dataset, source_id, linker, kg, args.max_mentions)
        raw_triples: List[Triple] = [tuple(value) for value in resolved["triples"] if len(value) == 3]
        pid_map = {triple: _triple_pid(triple) for triple in raw_triples}
        question_entities = [
            item["mention"] for item in resolved.get("linked_entities", [])
            if item.get("qid") and not item.get("abstained")
        ] or None
        rich = filter_and_rank_triples(
            raw_triples,
            question,
            pid_map=pid_map,
            max_keep=args.max_keep,
            min_keep=args.min_keep,
            rich=True,
            question_entities=question_entities,
        )
        selected: List[Triple] = [(item["h"], item["r"], item["t"]) for item in rich]

        # Dataset evidence is deliberately read only after KG construction.
        reference_hops = _reference_hops(dataset, source)
        expected_anchors = _expected_explicit_anchors(dataset, source, question, reference_hops)
        raw_chain = _chain_summary(reference_hops, raw_triples)
        selected_chain = _chain_summary(reference_hops, selected)
        final_intent = _infer_question_relation(question)
        intent_recognized = bool(final_intent["pids"] or final_intent["labels"])
        raw_intent_hit = any(_relation_matches(triple, final_intent) for triple in raw_triples) if intent_recognized else None
        selected_intent_hit = any(_relation_matches(triple, final_intent) for triple in selected) if intent_recognized else None
        linked_mentions = resolved.get("linked_entities", [])
        linked_ok = sum(bool(item.get("qid") and not item.get("abstained")) for item in linked_mentions)
        linked_expected_anchors = [
            anchor for anchor in expected_anchors
            if any(
                item.get("qid")
                and not item.get("abstained")
                and _surface_match(item.get("label"), anchor)
                for item in linked_mentions
            )
        ]
        failure = _failure_class(linked_ok, raw_chain, selected_chain)
        detail = {
            **frozen,
            "n_linked_mentions": len(linked_mentions),
            "n_linked_ok": linked_ok,
            "linked_entities": linked_mentions,
            "expected_explicit_anchors": expected_anchors,
            "linked_expected_explicit_anchors": linked_expected_anchors,
            "n_expected_explicit_anchors": len(expected_anchors),
            "n_linked_expected_explicit_anchors": len(linked_expected_anchors),
            "n_raw_triples": len(raw_triples),
            "n_selected_triples": len(selected),
            "final_intent": final_intent,
            "final_intent_recognized": intent_recognized,
            "raw_intent_relation_hit": raw_intent_hit,
            "selected_intent_relation_hit": selected_intent_hit,
            "raw_answer_component_hit": _answer_component_hit(frozen["gold_answers"], raw_triples),
            "selected_answer_component_hit": _answer_component_hit(frozen["gold_answers"], selected),
            "reference_hops": reference_hops,
            "raw_chain": raw_chain,
            "selected_chain": selected_chain,
            "failure_class": failure,
        }
        details.append(detail)
        override_rows.append(
            {
                "dataset": dataset,
                "qid": source_id,
                "question": question,
                "linked_entities": linked_mentions,
                "kg_subgraph": [list(triple) for triple in selected],
                "kg_rich": rich,
                "builder": "current-offline-question-kg-rel2-top12-min5",
            }
        )
        if index % 25 == 0:
            print(f"audited {index}/150")

    by_dataset = {
        dataset: _aggregate([row for row in details if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in details})
    }
    overall = _aggregate(details)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE_ZERO_TRAIN_AUDIT",
        "scope": "150 train questions; 50 per dataset; no dev/test; no model inference",
        "protocol": {
            "gold_used_for_kg_build": False,
            "gold_used_for_post_build_audit": True,
            "current_builder": "offline entity link + cached 2-hop subgraph + rel2 Top-12 min-5",
            "max_mentions": args.max_mentions,
            "max_keep": args.max_keep,
            "min_keep": args.min_keep,
            "proof_chain_metric": "all annotated hops require matching relation and tail value",
            "exact_chain_metric": "relation+tail plus annotated head surface",
            "hotpot_reference_limit": "no structured relation chain; intent-only metrics",
            "musique_reference_limit": "exact relation after >>; conservative rules for natural-language decomposition hops",
        },
        "inputs": {
            "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path)},
            "cohort_manifest": {"path": str(cohort_manifest_path), "sha256": _sha256(cohort_manifest_path)},
            "datasets": source_hashes,
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "failure_classes": dict(Counter(row["failure_class"] for row in details)),
        "interpretation_guard": "raw proof miss conflates entity-link and offline subgraph-asset miss; no causal efficacy claim",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as fh:
        for row in details:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with overrides_path.open("w", encoding="utf-8") as fh:
        for row in override_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["outputs"] = {
        "details": {"path": str(details_path), "sha256": _sha256(details_path)},
        "kg_overrides": {"path": str(overrides_path), "sha256": _sha256(overrides_path)},
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        run_dir,
        extra={
            "experiment_id": args.experiment_id,
            "phase": "query_aware_kg_relation_coverage_audit",
            "scope": report["scope"],
            "protocol": report["protocol"],
            "inputs": report["inputs"],
            "outputs": report["outputs"],
        },
    )
    print(json.dumps({"overall": overall, "by_dataset": by_dataset, "failure_classes": report["failure_classes"]}, indent=2))


if __name__ == "__main__":
    main()
