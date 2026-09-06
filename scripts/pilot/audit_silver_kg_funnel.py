#!/usr/bin/env python
"""Audit Phase-1 KG availability and usage as a stage-wise funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import string
from typing import Any, Dict, List


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def _read_many(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def _pct(count: int, total: int) -> float | None:
    return 100.0 * count / total if total else None


def _source_funnel(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    stages = {
        "mentions_found": 0,
        "entity_linked": 0,
        "raw_kg_nonempty": 0,
        "filtered_teacher_kg_nonempty": 0,
    }
    failures = {
        "no_mentions": 0,
        "mentions_but_no_entity_link": 0,
        "entity_linked_but_raw_kg_empty": 0,
        "raw_kg_but_filtered_empty": 0,
    }
    prefilter_total = teacher_total = linked_total = mentions_total = 0
    for row in rows:
        md = row.get("metadata") or {}
        mentions = int(md.get("n_mentions") or 0)
        linked = len(md.get("linked_entities") or {})
        prefilter = int(md.get("n_triples_prefilter") or 0)
        teacher = len(row.get("kg_subgraph") or [])
        mentions_total += mentions
        linked_total += linked
        prefilter_total += prefilter
        teacher_total += teacher
        stages["mentions_found"] += int(mentions > 0)
        stages["entity_linked"] += int(linked > 0)
        stages["raw_kg_nonempty"] += int(prefilter > 0)
        stages["filtered_teacher_kg_nonempty"] += int(teacher > 0)
        failures["no_mentions"] += int(mentions == 0)
        failures["mentions_but_no_entity_link"] += int(mentions > 0 and linked == 0)
        failures["entity_linked_but_raw_kg_empty"] += int(linked > 0 and prefilter == 0)
        failures["raw_kg_but_filtered_empty"] += int(prefilter > 0 and teacher == 0)
    return {
        "n_questions": n,
        "stage_counts": stages,
        "stage_rates_pct_of_all": {key: _pct(value, n) for key, value in stages.items()},
        "failure_counts": failures,
        "means_per_question": {
            "mentions": mentions_total / n if n else None,
            "linked_entities": linked_total / n if n else None,
            "raw_triples": prefilter_total / n if n else None,
            "filtered_teacher_triples": teacher_total / n if n else None,
        },
        "conditional_retention_pct": {
            "entity_link_given_mentions": _pct(stages["entity_linked"], stages["mentions_found"]),
            "raw_kg_given_entity_link": _pct(stages["raw_kg_nonempty"], stages["entity_linked"]),
            "filtered_nonempty_given_raw_kg": _pct(
                stages["filtered_teacher_kg_nonempty"], stages["raw_kg_nonempty"]
            ),
        },
    }


def _teacher_usage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    kg_rows = [row for row in rows if row.get("kg_subgraph")]
    any_cited = 0
    any_positive = 0
    all_steps_cited = 0
    cited_steps = total_steps = 0
    for row in rows:
        steps = row.get("steps") or []
        flags = [bool(step.get("cited_triples")) for step in steps]
        total_steps += len(steps)
        cited_steps += sum(flags)
        any_cited += int(any(flags))
        any_positive += int(any(float(step.get("label", 0.0)) > 0 for step in steps))
        all_steps_cited += int(bool(steps) and all(flags))
    kg_n = len(kg_rows)
    kg_any_cited = sum(
        any(bool(step.get("cited_triples")) for step in row.get("steps") or [])
        for row in kg_rows
    )
    kg_any_positive = sum(
        any(float(step.get("label", 0.0)) > 0 for step in row.get("steps") or [])
        for row in kg_rows
    )
    return {
        "n_questions": n,
        "kg_nonempty_questions": kg_n,
        "questions_with_any_citation": any_cited,
        "questions_with_any_positive_kg_label": any_positive,
        "questions_with_all_steps_cited": all_steps_cited,
        "step_citation_rate_pct": _pct(cited_steps, total_steps),
        "usage_given_nonempty_kg_pct": _pct(kg_any_cited, kg_n),
        "positive_signal_given_nonempty_kg_pct": _pct(kg_any_positive, kg_n),
        "note": "citation/positive-label are current-policy proxies for KG use/effectiveness, not gold KG recall",
    }


def _normalise(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _kg_content_proxies(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gold-aware diagnostics; these are proxies, not a new acceptance rule."""
    from kgproweight.kg.kg_filter import filter_by_passage_support

    kg_rows = [row for row in rows if row.get("kg_subgraph")]
    question_grounded = gold_present = passage_supported = 0
    passage_supported_counts: List[int] = []
    for row in kg_rows:
        triples = [tuple(triple) for triple in row.get("kg_subgraph") or []]
        question = _normalise(row.get("question"))
        gold = _normalise((row.get("metadata") or {}).get("gold_answer"))
        question_grounded += int(
            any(_normalise(h) in question or _normalise(t) in question for h, _, t in triples)
        )
        gold_present += int(
            bool(gold)
            and any(
                gold == _normalise(h)
                or gold == _normalise(t)
                or gold in _normalise(h)
                or gold in _normalise(t)
                for h, _, t in triples
            )
        )
        supported = filter_by_passage_support(triples, row.get("retrieved_passages") or [])
        passage_supported_counts.append(len(supported))
        passage_supported += int(bool(supported))
    n = len(kg_rows)
    return {
        "kg_nonempty_questions": n,
        "question_entity_present_in_kg": question_grounded,
        "question_entity_present_rate_pct": _pct(question_grounded, n),
        "gold_answer_surface_present_in_kg": gold_present,
        "gold_answer_surface_present_rate_pct": _pct(gold_present, n),
        "passage_supported_kg_nonempty": passage_supported,
        "passage_supported_kg_rate_pct": _pct(passage_supported, n),
        "passage_supported_triples_mean": (
            sum(passage_supported_counts) / n if n else None
        ),
        "note": "gold-answer surface presence is diagnostic only and must not be used to build/filter training KG",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--teacher_arm", action="append", default=[], help="LABEL=JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = _read_many(args.source)
    teacher_arms: Dict[str, List[Dict[str, Any]]] = {"control": source}
    for item in args.teacher_arm:
        label, path = item.split("=", 1)
        teacher_arms[label] = _read_many([path])

    report: Dict[str, Any] = {
        "report_schema_version": 2,
        "metric_semantics": {
            "coverage": "engineering funnel availability, not gold supporting-triple recall",
            "effectiveness": "exact Teacher citation/current positive-label proxy, not causal answer gain",
        },
        "source_funnel": {
            "aggregate": _source_funnel(source),
            "datasets": {
                dataset: _source_funnel([row for row in source if row.get("dataset") == dataset])
                for dataset in DATASETS
            },
        },
        "kg_content_proxies": {
            "aggregate": _kg_content_proxies(source),
            "datasets": {
                dataset: _kg_content_proxies(
                    [row for row in source if row.get("dataset") == dataset]
                )
                for dataset in DATASETS
            },
        },
        "teacher_usage": {},
    }
    for label, rows in teacher_arms.items():
        report["teacher_usage"][label] = {
            "aggregate": _teacher_usage(rows),
            "datasets": {
                dataset: _teacher_usage([row for row in rows if row.get("dataset") == dataset])
                for dataset in DATASETS
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
