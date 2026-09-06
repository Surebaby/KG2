#!/usr/bin/env python
"""Audit a now-seen query-planner confirmation run without changing its score."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from kgproweight.eval.query_planner import (
    _dependency_edges,
    _norm,
    _operator_text,
    _pid_sequence,
    _token_f1,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _anchors(target: Mapping[str, Any]) -> list[str]:
    return sorted(_norm(value) for value in target.get("anchors") or [])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=25)
    args = parser.parse_args()

    output_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        extra={
            "phase": "seen_confirmation_error_audit",
            "predictions": artifact_identity(args.predictions),
            "records": artifact_identity(args.records),
            "score_mutation": False,
        },
    )
    rows = list(_read_jsonl(args.predictions))
    records = {row["question_key"]: row for row in _read_jsonl(args.records)}

    two_wiki = [row for row in rows if row["dataset"] == "2wikimultihopqa"]
    exact_counts = Counter()
    anchor_error = Counter()
    anchor_question_surface = Counter()
    question_surface_executable_graph = 0
    two_wiki_examples: list[dict[str, Any]] = []
    for row in two_wiki:
        gold = row.get("gold_target") or {}
        predicted = row.get("predicted_target") or {}
        pid_exact = _pid_sequence(gold) == _pid_sequence(predicted)
        dependency_exact = _dependency_edges(gold) == _dependency_edges(predicted)
        gold_anchors, predicted_anchors = _anchors(gold), _anchors(predicted)
        anchor_exact = gold_anchors == predicted_anchors
        exact_counts["pid"] += pid_exact
        exact_counts["dependency"] += dependency_exact
        exact_counts["anchor"] += anchor_exact
        exact_counts["pid_dependency"] += pid_exact and dependency_exact
        exact_counts["graph"] += pid_exact and dependency_exact and anchor_exact
        gold_counter, predicted_counter = Counter(gold_anchors), Counter(predicted_anchors)
        missing = list((gold_counter - predicted_counter).elements())
        extra = list((predicted_counter - gold_counter).elements())
        if anchor_exact:
            anchor_error["exact"] += 1
        elif missing and extra:
            anchor_error["missing_and_extra"] += 1
        elif missing:
            anchor_error["missing_only"] += 1
        elif extra:
            anchor_error["extra_only"] += 1
        else:
            anchor_error["other"] += 1
        if not anchor_exact:
            source = records.get(row["question_key"]) or {}
            normalized_question = _norm(source.get("question") or "")
            gold_all_in_question = bool(gold_anchors) and all(
                anchor in normalized_question for anchor in gold_anchors
            )
            predicted_all_in_question = bool(predicted_anchors) and all(
                anchor in normalized_question for anchor in predicted_anchors
            )
            anchor_question_surface["mismatch_rows"] += 1
            anchor_question_surface["gold_all_in_question"] += gold_all_in_question
            anchor_question_surface["predicted_all_in_question"] += predicted_all_in_question
            anchor_question_surface["predicted_in_gold_out"] += (
                predicted_all_in_question and not gold_all_in_question
            )
            anchor_question_surface["both_all_in"] += (
                predicted_all_in_question and gold_all_in_question
            )
            anchor_question_surface["neither_all_in"] += (
                not predicted_all_in_question and not gold_all_in_question
            )
        normalized_question = _norm((records.get(row["question_key"]) or {}).get("question") or "")
        question_surface_executable_graph += (
            pid_exact
            and dependency_exact
            and bool(predicted_anchors)
            and all(anchor in normalized_question for anchor in predicted_anchors)
        )
        if not (pid_exact and dependency_exact and anchor_exact):
            source = records.get(row["question_key"]) or {}
            two_wiki_examples.append({
                "question_key": row["question_key"],
                "question": source.get("question"),
                "pid_exact": pid_exact,
                "dependency_exact": dependency_exact,
                "anchor_exact": anchor_exact,
                "missing_anchors": missing,
                "extra_anchors": extra,
                "gold_target": gold,
                "predicted_target": predicted,
                "schema_valid": row.get("schema_valid"),
                "validation_errors": row.get("validation_errors") or [],
            })

    musique = [row for row in rows if row["dataset"] == "musique"]
    operator_scores: list[float] = []
    scores_by_step: dict[int, list[float]] = defaultdict(list)
    style_groups: dict[str, list[float]] = defaultdict(list)
    musique_length_mismatch = 0
    row_optimal_permutation: list[float] = []
    graph_isomorphic_best_scores: list[float] = []
    graph_isomorphic_formal_scores: list[float] = []
    graph_isomorphic_rows = 0
    musique_examples: list[dict[str, Any]] = []
    for row in musique:
        gold_steps = (row.get("gold_target") or {}).get("steps") or []
        predicted_steps = (row.get("predicted_target") or {}).get("steps") or []
        musique_length_mismatch += len(gold_steps) != len(predicted_steps)
        n_steps = max(len(gold_steps), len(predicted_steps))
        gold_ops: list[str] = []
        predicted_ops: list[str] = []
        row_scores: list[float] = []
        row_style_mismatch = 0
        for index in range(n_steps):
            gold_raw = gold_steps[index].get("subquery_template", "") if index < len(gold_steps) else ""
            predicted_raw = (
                predicted_steps[index].get("subquery_template", "")
                if index < len(predicted_steps) else ""
            )
            gold_op, predicted_op = _operator_text(gold_raw), _operator_text(predicted_raw)
            score = _token_f1(predicted_op, gold_op)
            operator_scores.append(score)
            row_scores.append(score)
            scores_by_step[index + 1].append(score)
            gold_style = "arrow" if ">>" in gold_raw else "natural_language"
            predicted_style = "arrow" if ">>" in predicted_raw else "natural_language"
            style_groups[f"gold_{gold_style}__pred_{predicted_style}"].append(score)
            row_style_mismatch += gold_style != predicted_style
            gold_ops.append(gold_op)
            predicted_ops.append(predicted_op)
        if n_steps:
            best = max(
                sum(_token_f1(predicted_ops[pred_index], gold_ops[gold_index])
                    for gold_index, pred_index in enumerate(permutation))
                for permutation in itertools.permutations(range(n_steps))
            )
            row_optimal_permutation.append(best / n_steps)
        if gold_steps and len(gold_steps) == len(predicted_steps):
            def dependency_indices(steps: list[dict[str, Any]], index: int) -> set[int]:
                result: set[int] = set()
                for value in steps[index].get("dependencies") or []:
                    match = re.search(r"(\d+)$", str(value))
                    if match:
                        result.add(int(match.group(1)) - 1)
                return result

            gold_dependencies = [
                dependency_indices(gold_steps, index) for index in range(len(gold_steps))
            ]
            predicted_dependencies = [
                dependency_indices(predicted_steps, index)
                for index in range(len(predicted_steps))
            ]
            best_isomorphic: list[float] | None = None
            for permutation in itertools.permutations(range(len(gold_steps))):
                isomorphic = all(
                    predicted_dependencies[permutation[index]]
                    == {permutation[dependency] for dependency in gold_dependencies[index]}
                    for index in range(len(gold_steps))
                )
                if not isomorphic:
                    continue
                candidate = [
                    _token_f1(
                        _operator_text(predicted_steps[permutation[index]]["subquery_template"]),
                        _operator_text(gold_steps[index]["subquery_template"]),
                    )
                    for index in range(len(gold_steps))
                ]
                if best_isomorphic is None or sum(candidate) > sum(best_isomorphic):
                    best_isomorphic = candidate
            if best_isomorphic is not None:
                graph_isomorphic_rows += 1
                graph_isomorphic_best_scores.extend(best_isomorphic)
                graph_isomorphic_formal_scores.extend(row_scores)
        if _mean(row_scores) < 0.5 or row_style_mismatch or len(gold_steps) != len(predicted_steps):
            source = records.get(row["question_key"]) or {}
            musique_examples.append({
                "question_key": row["question_key"],
                "question": source.get("question"),
                "operator_f1": _mean(row_scores),
                "style_mismatch_steps": row_style_mismatch,
                "gold_step_count": len(gold_steps),
                "predicted_step_count": len(predicted_steps),
                "gold_target": row.get("gold_target") or {},
                "predicted_target": row.get("predicted_target") or {},
                "schema_valid": row.get("schema_valid"),
                "validation_errors": row.get("validation_errors") or [],
            })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "COMPLETE_SEEN_DIAGNOSTICS_ONLY",
        "scope": {
            "n": len(rows),
            "by_dataset": {"2wikimultihopqa": len(two_wiki), "musique": len(musique)},
            "formal_score_changed": False,
            "confirmation_reusable": False,
        },
        "2wikimultihopqa": {
            "exact_rates": {key: value / len(two_wiki) for key, value in exact_counts.items()},
            "anchor_error_counts": dict(anchor_error),
            "anchor_question_surface_diagnostic": dict(anchor_question_surface),
            "question_surface_executable_graph_exact_diagnostic": (
                question_surface_executable_graph / len(two_wiki)
            ),
            "graph_error_count": len(two_wiki_examples),
        },
        "musique": {
            "operator_step_n": len(operator_scores),
            "operator_macro_f1_recomputed": _mean(operator_scores),
            "operator_exact_rate": sum(score == 1.0 for score in operator_scores) / len(operator_scores),
            "operator_zero_count": sum(score == 0.0 for score in operator_scores),
            "step_count_mismatch_rows": musique_length_mismatch,
            "scores_by_step": {
                str(step): {"n": len(values), "mean_f1": _mean(values)}
                for step, values in sorted(scores_by_step.items())
            },
            "style_groups": {
                key: {"n": len(values), "mean_f1": _mean(values)}
                for key, values in sorted(style_groups.items())
            },
            "style_mismatch_steps": sum(
                len(values) for key, values in style_groups.items()
                if "gold_arrow__pred_natural_language" in key
                or "gold_natural_language__pred_arrow" in key
            ),
            "row_macro_optimal_permutation_f1_diagnostic": _mean(row_optimal_permutation),
            "graph_isomorphic_operator_diagnostic": {
                "rows": graph_isomorphic_rows,
                "steps": len(graph_isomorphic_best_scores),
                "formal_f1_on_same_rows": _mean(graph_isomorphic_formal_scores),
                "best_isomorphic_f1_on_same_rows": _mean(graph_isomorphic_best_scores),
                "note": "conditional seen-diagnostics only; excludes non-isomorphic and step-count-mismatch rows"
            },
        },
        "inputs": {
            "predictions": artifact_identity(args.predictions),
            "records": artifact_identity(args.records),
        },
    }
    examples_path = output_dir / "error_examples.jsonl"
    with examples_path.open("x", encoding="utf-8") as fh:
        for example in sorted(two_wiki_examples, key=lambda row: row["question_key"])[:args.max_examples]:
            fh.write(json.dumps({"dataset": "2wikimultihopqa", **example}, ensure_ascii=False) + "\n")
        for example in sorted(musique_examples, key=lambda row: (row["operator_f1"], row["question_key"]))[:args.max_examples]:
            fh.write(json.dumps({"dataset": "musique", **example}, ensure_ascii=False) + "\n")
    report["error_examples"] = artifact_identity(examples_path)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir, status="COMPLETE", extra=report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
