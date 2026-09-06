"""Strict structural evaluation for the answer-free learned query planner."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from scripts.prepare.audit_query_planner_supervision import validate_record
from scripts.prepare.build_query_planner_supervision import _norm, _record_leaks_source_values


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def parse_plan(text: str) -> tuple[Dict[str, Any] | None, str | None]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"json_decode:{exc.msg}"
    if not isinstance(value, dict):
        return None, "top_level_not_object"
    return value, None


def plan_validation_errors(record: Mapping[str, Any], predicted_target: Mapping[str, Any]) -> list[str]:
    candidate = dict(record)
    candidate["target"] = dict(predicted_target)
    return validate_record(candidate)


def _f1_counts(gold: Sequence[Any], predicted: Sequence[Any]) -> tuple[int, int, int]:
    gold_counter, predicted_counter = Counter(gold), Counter(predicted)
    true_positive = sum((gold_counter & predicted_counter).values())
    return true_positive, sum(predicted_counter.values()), sum(gold_counter.values())


def _f1(true_positive: int, predicted: int, gold: int) -> float:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _dependency_edges(target: Mapping[str, Any]) -> list[tuple[int, str]]:
    return [
        (int(step.get("step") or index), str(dependency))
        for index, step in enumerate((target.get("steps") or []), start=1)
        for dependency in (step.get("dependencies") or [])
    ]


def _pid_sequence(target: Mapping[str, Any]) -> list[str]:
    return [str(step.get("pid") or "") for step in target.get("steps") or []]


def _operator_text(template: object) -> str:
    raw = str(template or "")
    if ">>" in raw:
        return _norm(raw.partition(">>")[2])
    return _norm(re.sub(r"#\d+", " REFERENCE ", raw))


def _token_f1(left: str, right: str) -> float:
    left_tokens, right_tokens = left.split(), right.split()
    tp, predicted, gold = _f1_counts(right_tokens, left_tokens)
    return _f1(tp, predicted, gold)


def score_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    pid_tp = pid_predicted = pid_gold = 0
    dependency: Dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    operator_scores: list[float] = []
    for row in rows:
        dataset = str(row["dataset"])
        gold = row.get("gold_target") or {}
        predicted = row.get("predicted_target") or {}
        counts["n"] += 1
        counts[f"n::{dataset}"] += 1
        valid = bool(row.get("schema_valid"))
        counts["schema_valid"] += int(valid)
        counts[f"schema_valid::{dataset}"] += int(valid)
        counts["leakage"] += int(bool(row.get("source_value_leakage")))
        counts[f"leakage::{dataset}"] += int(bool(row.get("source_value_leakage")))
        if dataset == "2wikimultihopqa":
            gold_pids, predicted_pids = _pid_sequence(gold), _pid_sequence(predicted)
            tp, pred_n, gold_n = _f1_counts(gold_pids, predicted_pids)
            pid_tp += tp
            pid_predicted += pred_n
            pid_gold += gold_n
            counts["2wiki_pid_sequence_exact"] += int(predicted_pids == gold_pids)
            gold_edges, predicted_edges = _dependency_edges(gold), _dependency_edges(predicted)
            edge_counts = _f1_counts(gold_edges, predicted_edges)
            dependency[dataset] = [
                dependency[dataset][index] + edge_counts[index] for index in range(3)
            ]
            gold_anchors = sorted(_norm(value) for value in gold.get("anchors") or [])
            predicted_anchors = sorted(_norm(value) for value in predicted.get("anchors") or [])
            counts["2wiki_graph_exact"] += int(
                predicted_pids == gold_pids
                and predicted_edges == gold_edges
                and predicted_anchors == gold_anchors
            )
        elif dataset == "musique":
            gold_edges, predicted_edges = _dependency_edges(gold), _dependency_edges(predicted)
            edge_counts = _f1_counts(gold_edges, predicted_edges)
            dependency[dataset] = [
                dependency[dataset][index] + edge_counts[index] for index in range(3)
            ]
            counts["musique_dependency_graph_exact"] += int(predicted_edges == gold_edges)
            gold_steps, predicted_steps = gold.get("steps") or [], predicted.get("steps") or []
            for index in range(max(len(gold_steps), len(predicted_steps))):
                gold_operator = _operator_text(
                    gold_steps[index].get("subquery_template") if index < len(gold_steps) else ""
                )
                predicted_operator = _operator_text(
                    predicted_steps[index].get("subquery_template") if index < len(predicted_steps) else ""
                )
                operator_scores.append(_token_f1(predicted_operator, gold_operator))

    n, n_2wiki, n_musique = counts["n"], counts["n::2wikimultihopqa"], counts["n::musique"]
    return {
        "n": n,
        "by_dataset": {"2wikimultihopqa": n_2wiki, "musique": n_musique},
        "schema_valid_rate": counts["schema_valid"] / n if n else 0.0,
        "schema_valid_rate_by_dataset": {
            dataset: counts[f"schema_valid::{dataset}"] / counts[f"n::{dataset}"]
            if counts[f"n::{dataset}"] else 0.0
            for dataset in ("2wikimultihopqa", "musique")
        },
        "answer_or_evidence_tail_leakage_rate": counts["leakage"] / n if n else 0.0,
        "2wikimultihopqa": {
            "pid_micro_f1": _f1(pid_tp, pid_predicted, pid_gold),
            "pid_sequence_exact": counts["2wiki_pid_sequence_exact"] / n_2wiki if n_2wiki else 0.0,
            "dependency_edge_f1": _f1(*dependency["2wikimultihopqa"]),
            "graph_exact": counts["2wiki_graph_exact"] / n_2wiki if n_2wiki else 0.0,
        },
        "musique": {
            "dependency_edge_f1": _f1(*dependency["musique"]),
            "dependency_graph_exact": counts["musique_dependency_graph_exact"] / n_musique if n_musique else 0.0,
            "operator_macro_f1": sum(operator_scores) / len(operator_scores) if operator_scores else 0.0,
        },
    }


def evaluate_gates(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, bool] = {
        "schema_valid_rate": metrics["schema_valid_rate"] >= gates["schema_valid_rate_min"],
        "leakage_rate": metrics["answer_or_evidence_tail_leakage_rate"]
        <= gates["answer_or_evidence_tail_leakage_rate_max"],
    }
    for dataset in ("2wikimultihopqa", "musique"):
        for metric, threshold in gates[dataset].items():
            if not metric.endswith("_min"):
                continue
            metric_name = metric[:-4]
            checks[f"{dataset}.{metric_name}"] = metrics[dataset][metric_name] >= threshold
    return {"pass": all(checks.values()), "checks": checks}


def resolve_dev_gates(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read dev gates from either the v1 or the explicit v1.1 protocol key."""

    for key in ("dev_gates", "smoke_dev_gates"):
        gates = protocol.get(key)
        if isinstance(gates, Mapping):
            return gates
    raise KeyError("protocol has neither dev_gates nor smoke_dev_gates")


def load_source_rows(data_root: str | Path) -> Dict[str, Dict[str, Any]]:
    from kgproweight.kg.question_kg import question_key

    root = Path(data_root)
    result: Dict[str, Dict[str, Any]] = {}
    for dataset in ("2wikimultihopqa", "musique"):
        for row in _read_jsonl(root / dataset / "train.jsonl"):
            result[question_key(dataset, str(row["id"]))] = row
    return result


def build_scored_row(
    record: Mapping[str, Any],
    generated_text: str,
    source_row: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    parsed, parse_error = parse_plan(generated_text)
    errors = plan_validation_errors(record, parsed) if parsed is not None else [str(parse_error)]
    leakage = False
    if parsed is not None and source_row is not None:
        candidate = dict(record)
        candidate["target"] = parsed
        leakage = _record_leaks_source_values(candidate, source_row)
    return {
        "question_key": record["question_key"],
        "dataset": record["dataset"],
        "qid": record["qid"],
        "generated_text": generated_text,
        "predicted_target": parsed,
        "gold_target": record["target"],
        "schema_valid": not errors,
        "validation_errors": errors,
        "source_value_leakage": leakage,
    }
