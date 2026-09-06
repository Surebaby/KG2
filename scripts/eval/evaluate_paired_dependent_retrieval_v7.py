#!/usr/bin/env python
"""Evaluate frozen v7 A/B/C scorer inputs with one strong-SFT load.

This is an intention-to-treat development analysis: all forty frozen rows,
including exact-A fallbacks, remain in both comparisons.  The preregistered
primary contrast is C (mechanically verified subanswer) minus B (deterministic
entity hint); C minus canonical one-shot A is secondary.

The module keeps validation and statistics CPU-testable.  The production CLI
alone imports the ML stack, requires CUDA/BF16, loads the strong-SFT adapter
once, and greedily generates all three arms.  Identical serialized prompts
reuse the first generation byte-for-byte and are nevertheless scored against
each row's own frozen Gold projection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.parsers import extract_final_answer, parse_steps
from kgproweight.data.prompts import build_rl_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.pilot.score_a1_fixed_context_kg import _bootstrap_ci, _mcnemar_exact
from scripts.prepare import freeze_dependent_retrieval_v7 as v7_freeze
from scripts.prepare import freeze_dependent_retrieval_v7_implementation as v7_implementation
from scripts.prepare import finalize_paired_dependent_retrieval_v7 as v7_finalizer


EVALUATOR_VERSION = "paired-dependent-retrieval-v7-evaluator-1"
RESULT_SCHEMA = "paired-dependent-retrieval-v7-result-1"
ARMS = v7_finalizer.ARMS
ARM_KEYS = ("A", "B", "C")
DATASETS = v7_finalizer.DATASETS
PROTOCOL_INPUT_KEYS = {"A": "arm_a", "B": "arm_b", "C": "arm_c"}
COMMON_FIELDS = (
    "row_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "family_sha256",
    "role",
    "gold_answers",
    "gold_attachment",
)


class V7EvaluationError(ValueError):
    """A frozen evaluator input or analysis contract was violated."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V7EvaluationError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(
                V7EvaluationError(f"non-finite JSON constant: {token}")
            ))
            if not isinstance(value, dict):
                raise V7EvaluationError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def _resolved_locked_file(
    protocol: Mapping[str, Any], name: str, cli_value: str | None
) -> Path:
    lock = (protocol.get("inputs") or {}).get(name)
    if not isinstance(lock, Mapping):
        raise V7EvaluationError(f"protocol lacks input lock: {name}")
    locked = Path(str(lock.get("path") or "")).expanduser().resolve()
    path = Path(cli_value).expanduser().resolve() if cli_value else locked
    if path != locked:
        raise V7EvaluationError(f"{name} path differs from frozen protocol")
    current = v7_finalizer._file_lock(path)
    if not v7_finalizer._lock_equal(current, lock):
        raise V7EvaluationError(f"{name} bytes differ from frozen protocol")
    return path


def _validate_model_tree(path: Path, lock: Mapping[str, Any], *, label: str) -> None:
    expected_path = Path(str(lock.get("path") or "")).expanduser().resolve()
    if path != expected_path:
        raise V7EvaluationError(f"{label} path differs from frozen protocol")
    if v7_freeze.tree_lock(path) != dict(lock):
        raise V7EvaluationError(f"{label} content tree differs from frozen protocol")


def _common_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in COMMON_FIELDS if field not in row]
    if missing:
        raise V7EvaluationError(f"scored row lacks common fields: {missing}")
    return {field: row[field] for field in COMMON_FIELDS}


def validate_scored_inputs(
    protocol: Mapping[str, Any],
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Validate the strict, complete ITT population before loading a model."""

    if set(arms) != set(ARM_KEYS):
        raise V7EvaluationError("evaluator requires exactly A/B/C scored inputs")
    expected_n = int(protocol.get("n", -1))
    if expected_n <= 0 or any(len(arms[key]) != expected_n for key in ARM_KEYS):
        raise V7EvaluationError(
            "strict ITT population broken: "
            + ", ".join(f"{key}={len(arms[key])}" for key in ARM_KEYS)
        )
    if protocol.get("arms") != list(ARMS):
        raise V7EvaluationError("protocol arm order/names differ from frozen A/B/C")

    keys: list[str] = []
    qids: list[str] = []
    dataset_counts: Counter[str] = Counter()
    for index, triple in enumerate(zip(arms["A"], arms["B"], arms["C"])):
        a, b, c = (dict(row) for row in triple)
        if _common_projection(a) != _common_projection(b) or _common_projection(a) != _common_projection(c):
            raise V7EvaluationError(f"A/B/C scored identities or Gold differ at row {index}")
        key = str(a["question_key"])
        if key != f"{a['dataset']}::{a['qid']}" or key in keys:
            raise V7EvaluationError(f"invalid/duplicate question_key: {key}")
        keys.append(key)
        qids.append(str(a["qid"]))
        dataset_counts[str(a["dataset"])] += 1
        if [a.get("arm"), b.get("arm"), c.get("arm")] != list(ARMS):
            raise V7EvaluationError(f"scored arm labels differ: {key}")
        if a.get("gold_attachment") != "SCORER_ONLY_AFTER_ALL_GOLD_FREE_GATES":
            raise V7EvaluationError(f"Gold attachment boundary marker differs: {key}")
        golds = a.get("gold_answers")
        if not isinstance(golds, list) or not golds or any(not str(value).strip() for value in golds):
            raise V7EvaluationError(f"missing/invalid scorer Gold: {key}")
        question = str(a["question"])
        if a.get("question_sha256") != _sha256_text(question):
            raise V7EvaluationError(f"question hash mismatch: {key}")
        for arm_key, row in zip(ARM_KEYS, (a, b, c)):
            passages = row.get("retrieved_passages")
            if not isinstance(passages, list) or len(passages) != 10:
                raise V7EvaluationError(f"{key}::{arm_key} is not Top-10")
            if row.get("kg_subgraph") != []:
                raise V7EvaluationError(f"{key}::{arm_key} has non-empty KG input")
        for arm_key, row in (("B", b), ("C", c)):
            equal_a = row["retrieved_passages"] == a["retrieved_passages"]
            if bool(row.get("fallback_to_a")) != equal_a:
                raise V7EvaluationError(
                    f"{key}::{arm_key} fallback flag differs from exact A equality"
                )
    if _sha256_text("\n".join(qids)) != str(protocol.get("qid_order_sha256") or ""):
        raise V7EvaluationError("qid order differs from frozen evaluation protocol")
    if _sha256_text("\n".join(keys)) != str(
        protocol.get("question_key_order_sha256") or ""
    ):
        raise V7EvaluationError("question-key order differs from frozen evaluation protocol")
    expected_counts = Counter(
        {name: int(value) for name, value in (protocol.get("by_dataset") or {}).items()}
    )
    if dataset_counts != expected_counts:
        raise V7EvaluationError("per-dataset population differs from frozen protocol")


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _visible(golds: Sequence[str], values: Sequence[str]) -> bool:
    blob = _norm(" ".join(values))
    return any(_norm(gold) and _norm(gold) in blob for gold in golds)


def score_generation(
    *,
    row: Mapping[str, Any],
    generation: str,
    prompt_sha256: str,
    prompt_tokens: int,
    arm: str,
    input_sha256: str,
    reused_identical_prompt_from: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Apply the canonical final-answer parser/EM/F1 scoring projection."""

    steps = parse_steps(generation, known_kg=[])
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()
    golds = [str(value) for value in row["gold_answers"]]
    indices = [step.index for step in steps]
    passages = [
        str(value.get("contents") or value.get("text") or value.get("title") or "")
        for value in row["retrieved_passages"]
    ]
    return {
        "row_id": row["row_id"],
        "question_key": row["question_key"],
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question": row["question"],
        "gold_answers": golds,
        "model_label": "strong_sft",
        "arm": arm,
        "input_sha256": input_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "reused_identical_prompt_from": dict(reused_identical_prompt_from)
        if reused_identical_prompt_from
        else None,
        "fallback_to_a": bool(row.get("fallback_to_a", False)),
        "prediction": answer,
        "em": compute_em(answer, golds) if answer else 0.0,
        "f1": compute_f1(answer, golds) if answer else 0.0,
        "well_formed": bool(steps and answer),
        "n_steps": len(steps),
        "contiguous": indices == list(range(1, len(indices) + 1)),
        "known_citation_count": sum(len(step.cited_triples) for step in steps),
        "unknown_citation_count": sum(
            len(step.unknown_citation_surfaces) for step in steps
        ),
        "citation_contract_error": any(step.citation_contract_errors for step in steps),
        "gold_in_passages": _visible(golds, passages),
        "generation": generation,
    }


def aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "em": sum(float(row["em"]) for row in rows) / max(1, n),
        "f1": sum(float(row["f1"]) for row in rows) / max(1, n),
        "parse_count": sum(bool(row["well_formed"]) for row in rows),
        "parse_rate": sum(bool(row["well_formed"]) for row in rows) / max(1, n),
        "contiguous_rate": sum(bool(row["contiguous"]) for row in rows) / max(1, n),
        "gold_in_passages_rate": sum(bool(row["gold_in_passages"]) for row in rows)
        / max(1, n),
        "prediction_empty_count": sum(not str(row["prediction"]).strip() for row in rows),
        "step_histogram": dict(Counter(int(row["n_steps"]) for row in rows)),
    }


def paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    treatment_arm: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute a complete paired ITT contrast, never an activated-hop subset."""

    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    question_order: list[str] = []
    for row in rows:
        identity = (str(row["question_key"]), str(row["arm"]))
        if identity in by_key:
            raise V7EvaluationError(f"duplicate prediction identity: {identity}")
        by_key[identity] = row
        if row["arm"] == baseline_arm:
            question_order.append(str(row["question_key"]))
    arm_rows = [row for row in rows if row["arm"] in {baseline_arm, treatment_arm}]
    if len(arm_rows) != 2 * len(question_order):
        raise V7EvaluationError("paired comparison is incomplete")
    try:
        pairs = [
            (by_key[(key, baseline_arm)], by_key[(key, treatment_arm)])
            for key in question_order
        ]
    except KeyError as exc:
        raise V7EvaluationError(f"paired comparison is missing {exc.args[0]}") from exc
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    gained = sum(float(right["em"]) > float(left["em"]) for left, right in pairs)
    lost = sum(float(right["em"]) < float(left["em"]) for left, right in pairs)
    baseline_parse = sum(bool(left["well_formed"]) for left, _ in pairs)
    treatment_parse = sum(bool(right["well_formed"]) for _, right in pairs)
    n = len(pairs)
    baseline_em = sum(float(left["em"]) for left, _ in pairs) / max(1, n)
    treatment_em = sum(float(right["em"]) for _, right in pairs) / max(1, n)
    baseline_f1 = sum(float(left["f1"]) for left, _ in pairs) / max(1, n)
    treatment_f1 = sum(float(right["f1"]) for _, right in pairs) / max(1, n)
    return {
        "analysis_population": "ITT_ALL_ROWS_WITH_FALLBACKS_INCLUDED",
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "n": n,
        "baseline_em": baseline_em,
        "treatment_em": treatment_em,
        "delta_em": treatment_em - baseline_em,
        "delta_em_bootstrap_95ci": _bootstrap_ci(
            em_diffs, seed=bootstrap_seed, draws=10000
        ),
        "baseline_f1": baseline_f1,
        "treatment_f1": treatment_f1,
        "delta_f1": treatment_f1 - baseline_f1,
        "delta_f1_bootstrap_95ci": _bootstrap_ci(
            f1_diffs, seed=bootstrap_seed + 1, draws=10000
        ),
        "gained_correct": gained,
        "lost_correct": lost,
        "tied_correctness": n - gained - lost,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "prediction_changed": sum(
            str(left["prediction"]).strip() != str(right["prediction"]).strip()
            for left, right in pairs
        ),
        "baseline_parse_count": baseline_parse,
        "treatment_parse_count": treatment_parse,
        "baseline_parse_rate": baseline_parse / max(1, n),
        "treatment_parse_rate": treatment_parse / max(1, n),
        "parse_count_delta": treatment_parse - baseline_parse,
    }


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build pooled and per-dataset A/B/C metrics and both frozen contrasts."""

    by_arm = {
        arm: aggregate_arm([row for row in predictions if row["arm"] == arm])
        for arm in ARMS
    }
    primary = paired_comparison(
        predictions,
        baseline_arm=ARMS[1],
        treatment_arm=ARMS[2],
        bootstrap_seed=20260904,
    )
    secondary = paired_comparison(
        predictions,
        baseline_arm=ARMS[0],
        treatment_arm=ARMS[2],
        bootstrap_seed=20260906,
    )
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        selected = [row for row in predictions if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "arms": {
                arm: aggregate_arm([row for row in selected if row["arm"] == arm])
                for arm in ARMS
            },
            "C_minus_B": paired_comparison(
                selected,
                baseline_arm=ARMS[1],
                treatment_arm=ARMS[2],
                bootstrap_seed=20260914 if dataset == "hotpotqa" else 20260915,
            ),
            "C_minus_A": paired_comparison(
                selected,
                baseline_arm=ARMS[0],
                treatment_arm=ARMS[2],
                bootstrap_seed=20260916 if dataset == "hotpotqa" else 20260917,
            ),
        }
    return {
        "by_arm": by_arm,
        "overall": {"C_minus_B": primary, "C_minus_A": secondary},
        "by_dataset": by_dataset,
    }


def evaluate_utility_gates(
    utility_gates: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the frozen primary and secondary gates without reinterpretation."""

    if dict(utility_gates) != v7_freeze.UTILITY_GATES:
        raise V7EvaluationError("development utility gates differ from preregistration")
    primary = metrics["overall"]["C_minus_B"]
    secondary = metrics["overall"]["C_minus_A"]
    by_dataset = metrics["by_dataset"]
    primary_checks = {
        "pooled_net_correct": int(primary["net_correct"])
        >= int(utility_gates["C_minus_B_pooled_net_correct_min"]),
        "pooled_delta_f1_strictly_positive": float(primary["delta_f1"])
        > float(utility_gates["C_minus_B_pooled_delta_f1_gt"]),
        "max_net_correct_loss_per_dataset": all(
            int(values["C_minus_B"]["net_correct"])
            >= -int(utility_gates["C_minus_B_max_net_correct_loss_per_dataset"])
            for values in by_dataset.values()
        ),
        "parse_count_not_degraded": int(primary["parse_count_delta"])
        >= int(utility_gates["C_minus_B_parse_count_delta_min"]),
    }
    secondary_checks = {
        "pooled_net_correct": int(secondary["net_correct"])
        >= int(utility_gates["C_minus_A_pooled_net_correct_min"]),
        "pooled_delta_f1_strictly_positive": float(secondary["delta_f1"])
        > float(utility_gates["C_minus_A_pooled_delta_f1_gt"]),
        "max_net_correct_loss_per_dataset": all(
            int(values["C_minus_A"]["net_correct"])
            >= -int(utility_gates["C_minus_A_max_net_correct_loss_per_dataset"])
            for values in by_dataset.values()
        ),
        "parse_count_not_degraded": int(secondary["parse_count_delta"])
        >= int(utility_gates["C_minus_A_parse_count_delta_min"]),
    }
    return {
        "primary_comparison": utility_gates["primary_comparison"],
        "primary": {
            "checks": primary_checks,
            "all_pass": all(primary_checks.values()),
        },
        "secondary_standard_baseline_comparison": utility_gates[
            "secondary_standard_baseline_comparison"
        ],
        "secondary": {
            "checks": secondary_checks,
            "all_pass": all(secondary_checks.values()),
        },
        "all_pass": all(primary_checks.values()) and all(secondary_checks.values()),
    }


def validate_fallback_prediction_reuse(
    predictions: Sequence[Mapping[str, Any]],
    scored_arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_prediction = {
        (str(row["question_key"]), str(row["arm"])): row for row in predictions
    }
    checked = 0
    for arm_key, arm_name in (("B", ARMS[1]), ("C", ARMS[2])):
        for row in scored_arms[arm_key]:
            if not row.get("fallback_to_a"):
                continue
            key = str(row["question_key"])
            left = by_prediction[(key, ARMS[0])]
            right = by_prediction[(key, arm_name)]
            checked += 1
            if (
                left["prompt_sha256"] != right["prompt_sha256"]
                or left["generation"] != right["generation"]
                or left["prediction"] != right["prediction"]
            ):
                raise V7EvaluationError(f"fallback generation was not byte-reused: {key}::{arm_key}")
    return {"fallback_arm_rows_checked": checked, "prompt_generation_prediction_exact": True}


def _current_evaluator_import_closure() -> dict[str, dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    paths = v7_implementation.local_import_closure([Path(__file__).resolve()], project_root)
    return {
        path.relative_to(project_root).as_posix(): v7_implementation.file_lock(
            path, allow_empty=True
        )
        for path in paths
    }


def validate_evaluator_dependency_closure(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any scoring-code drift, including indirect local imports."""

    code = protocol.get("code")
    if not isinstance(code, Mapping):
        raise V7EvaluationError("evaluation protocol lacks code authorization")
    frozen = code.get("evaluator_import_closure")
    if not isinstance(frozen, Mapping) or not frozen:
        raise V7EvaluationError("evaluation protocol lacks evaluator import closure")
    required = code.get("required_evaluator_dependencies")
    if required != sorted(v7_finalizer.REQUIRED_EVALUATOR_DEPENDENCIES):
        raise V7EvaluationError("required evaluator dependency inventory differs")
    observed = _current_evaluator_import_closure()
    if set(observed) != set(frozen):
        missing = sorted(set(frozen) - set(observed))
        extra = sorted(set(observed) - set(frozen))
        raise V7EvaluationError(
            f"evaluator import closure role set differs; missing={missing}, extra={extra}"
        )
    for name in sorted(observed):
        expected = frozen[name]
        if not isinstance(expected, Mapping) or not v7_finalizer._lock_equal(
            observed[name], expected
        ):
            raise V7EvaluationError(f"evaluator dependency content differs: {name}")
    for role, relative in (
        ("evaluator", "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"),
        ("gold_finalizer", "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"),
    ):
        lock = code.get(role)
        if not isinstance(lock, Mapping) or not v7_finalizer._lock_equal(
            observed[relative], lock
        ):
            raise V7EvaluationError(f"{role} lock differs from full import closure")
    return {
        "validated": True,
        "file_count": len(observed),
        "paths": sorted(observed),
    }


def _validate_finalizer_manifest(
    protocol_path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the append-only manifest emitted beside the protocol."""

    manifest_path = protocol_path.with_name("manifest.json")
    manifest = _read_json(manifest_path)
    run = manifest.get("run")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "FROZEN_READY_FOR_V7_ANSWER_EVALUATION"
        or not isinstance(run, Mapping)
        or run.get("phase") != "paired_dependent_retrieval_v7_gold_attachment"
        or run.get("experiment_id") != protocol.get("gold_attachment_experiment_id")
        or run.get("evaluation_experiment_id") != protocol.get("experiment_id")
        or run.get("protocol_sha256") != _sha256_file(protocol_path)
        or run.get("gold_opened_only_after_gold_free_gates") is not True
        or run.get("answer_evaluation_authorized") is not True
        or run.get("evaluation_authorization_schema")
        != v7_finalizer.EVAL_AUTHORIZATION_SCHEMA
    ):
        raise V7EvaluationError("finalizer manifest does not authorize this protocol")
    return v7_finalizer._file_lock(manifest_path)


def _require_input_lock(
    protocol: Mapping[str, Any], name: str
) -> tuple[Path, Mapping[str, Any]]:
    inputs = protocol.get("inputs")
    lock = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(lock, Mapping):
        raise V7EvaluationError(f"evaluation protocol lacks parent lock: {name}")
    try:
        path = v7_finalizer._assert_current_lock(lock, label=f"eval.inputs.{name}")
    except (OSError, ValueError) as exc:
        raise V7EvaluationError(str(exc)) from exc
    return path, lock


def validate_evaluation_authorization(
    protocol_path: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the finalizer issuer, immutable ancestry, and model attestation."""

    if protocol.get("experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS["evaluation"]:
        raise V7EvaluationError("evaluation Experiment ID is not the preregistered ID")
    if protocol.get("gold_attachment_experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS[
        "gold_attachment"
    ]:
        raise V7EvaluationError("Gold-attachment Experiment ID differs")
    if protocol.get("materialization_experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS[
        "materialization"
    ]:
        raise V7EvaluationError("materialization Experiment ID differs")

    manifest_lock = _validate_finalizer_manifest(protocol_path, protocol)
    authorization = protocol.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get(
        "schema_version"
    ) != v7_finalizer.EVAL_AUTHORIZATION_SCHEMA:
        raise V7EvaluationError("missing finalizer evaluation authorization")
    expected_flags = {
        "status": "AUTHORIZED_FOR_FROZEN_V7_ANSWER_EVALUATION_ONLY",
        "issuer": "paired-dependent-retrieval-v7-gold-finalizer",
        "gold_attachment_complete": True,
        "answer_evaluation": True,
        "training": False,
        "evaluation_experiment_id": protocol["experiment_id"],
        "gold_attachment_experiment_id": protocol["gold_attachment_experiment_id"],
    }
    for name, expected in expected_flags.items():
        if authorization.get(name) != expected:
            raise V7EvaluationError(f"evaluation authorization differs: {name}")

    dependency = validate_evaluator_dependency_closure(protocol)
    issuer = authorization.get("issuer_code")
    finalizer_lock = (protocol.get("code") or {}).get("gold_finalizer")
    if (
        not isinstance(issuer, Mapping)
        or not isinstance(finalizer_lock, Mapping)
        or not v7_finalizer._lock_equal(issuer, finalizer_lock)
    ):
        raise V7EvaluationError("authorization issuer is not the frozen finalizer")

    parent_names = {
        "preregistration",
        "truncation_addendum",
        "trajectory_semantics_addendum",
        "trajectory_semantics_addendum_manifest",
        "implementation_lock",
        "plan_lock",
        "materialization_report",
    }
    parent_chain = authorization.get("parent_chain")
    if not isinstance(parent_chain, Mapping) or set(parent_chain) != parent_names:
        raise V7EvaluationError("evaluation authorization parent-chain roles differ")
    input_alias = {
        "materialization_report": "retrieval_report",
        **{name: name for name in parent_names - {"materialization_report"}},
    }
    input_paths: dict[str, Path] = {}
    input_locks: dict[str, Mapping[str, Any]] = {}
    for parent_name, input_name in input_alias.items():
        path, lock = _require_input_lock(protocol, input_name)
        input_paths[input_name] = path
        input_locks[input_name] = lock
        parent = parent_chain.get(parent_name)
        if not isinstance(parent, Mapping) or not v7_finalizer._lock_equal(parent, lock):
            raise V7EvaluationError(f"authorization parent differs: {parent_name}")

    if input_locks["preregistration"].get("sha256") != (
        v7_implementation.EXPECTED_PREREGISTRATION_SHA256
    ):
        raise V7EvaluationError("evaluation does not descend from canonical v7 preregistration")
    if input_locks["truncation_addendum"].get("sha256") != (
        v7_implementation.EXPECTED_ADDENDUM_SHA256
    ):
        raise V7EvaluationError("evaluation does not descend from canonical v7 truncation addendum")

    prereg, prereg_lock = v7_finalizer._validate_preregistration(
        input_paths["preregistration"]
    )
    _, addendum_lock = v7_finalizer._validate_addendum(
        input_paths["truncation_addendum"], preregistration_lock=prereg_lock
    )
    _, trajectory_lock, trajectory_manifest_lock = v7_finalizer._validate_trajectory_addendum(
        input_paths["trajectory_semantics_addendum"],
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
    )
    if not v7_finalizer._lock_equal(
        trajectory_manifest_lock, input_locks["trajectory_semantics_addendum_manifest"]
    ):
        raise V7EvaluationError("trajectory addendum manifest parent differs")
    implementation, implementation_lock = v7_finalizer._validate_implementation_lock(
        input_paths["implementation_lock"],
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
        trajectory_addendum_lock=trajectory_lock,
        trajectory_addendum_manifest_lock=trajectory_manifest_lock,
    )
    plan, plan_lock = v7_finalizer._validate_plan_lock(
        input_paths["plan_lock"],
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
        implementation=implementation,
        implementation_lock=implementation_lock,
        preregistration=prereg,
        trajectory_addendum_lock=trajectory_lock,
        trajectory_addendum_manifest_lock=trajectory_manifest_lock,
    )

    report = _read_json(input_paths["retrieval_report"])
    if (
        report.get("schema_version") != v7_finalizer.EXPECTED_REPORT_SCHEMA
        or report.get("status") != v7_finalizer.EXPECTED_REPORT_STATUS
        or report.get("experiment_id") != protocol["materialization_experiment_id"]
        or report.get("gate_decision") != "PASS_READY_FOR_SEPARATE_GOLD_FINALIZER"
        or (report.get("materialization_gate") or {}).get("passed") is not True
        or (report.get("gold_free_mechanism_gate") or {}).get("passed") is not True
    ):
        raise V7EvaluationError("materialization report does not authorize Gold attachment")
    report_refs = {
        "preregistration": prereg_lock,
        "truncation_addendum": addendum_lock,
        "trajectory_semantics_addendum": trajectory_lock,
        "implementation_lock": implementation_lock,
        "plan_lock": plan_lock,
    }
    for name, expected in report_refs.items():
        observed = report.get(name)
        if not isinstance(observed, Mapping) or not v7_finalizer._lock_equal(
            observed, expected
        ):
            raise V7EvaluationError(f"materialization report parent differs: {name}")
    output_map = {
        "arm_a": "retrieval_arm_a_no_gold",
        "arm_b": "retrieval_arm_b_no_gold",
        "arm_c": "retrieval_arm_c_no_gold",
        "execution_details": "execution_details_no_gold",
        "budget_ledger": "budget_ledger_no_gold",
    }
    report_outputs = report.get("outputs") or {}
    for report_name, input_name in output_map.items():
        _, frozen_input = _require_input_lock(protocol, input_name)
        observed = report_outputs.get(report_name)
        if not isinstance(observed, Mapping) or not v7_finalizer._lock_equal(
            observed, frozen_input
        ):
            raise V7EvaluationError(
                f"materialization output differs from scorer parent: {report_name}"
            )
    scorer_gold = (protocol.get("inputs") or {}).get("scorer_gold")
    if not isinstance(scorer_gold, Mapping) or set(scorer_gold) != set(DATASETS):
        raise V7EvaluationError("scorer-Gold lock inventory differs")
    for dataset, lock in scorer_gold.items():
        if not isinstance(lock, Mapping):
            raise V7EvaluationError(f"invalid scorer-Gold lock: {dataset}")
        try:
            v7_finalizer._assert_current_lock(lock, label=f"scorer_gold.{dataset}")
        except (OSError, ValueError) as exc:
            raise V7EvaluationError(str(exc)) from exc

    models = protocol.get("models") or {}
    if (
        models.get("strong_sft") != prereg["models"]["subanswer_and_final_strong_sft"]
        or models.get("base_model") != prereg["models"]["base_model"]
        or (models.get("content_locks") or {}).get("strong_sft")
        != prereg["models"]["inherited_content_locks"]["strong_sft"]
        or (models.get("content_locks") or {}).get("base_model")
        != prereg["models"]["inherited_content_locks"]["base_model"]
    ):
        raise V7EvaluationError("final answer model identity differs from preregistration")
    attestation = (protocol.get("gold_free_materialization_observed") or {}).get(
        "subanswer_model_identity"
    )
    if not isinstance(attestation, Mapping) or attestation.get("validated") is not True:
        raise V7EvaluationError("protocol lacks validated C-subanswer model identity")
    if attestation.get("strong_sft_tree_sha256") != models["content_locks"][
        "strong_sft"
    ].get("tree_sha256"):
        raise V7EvaluationError("C-subanswer strong-SFT attestation differs")

    return {
        "validated": True,
        "finalizer_manifest": manifest_lock,
        "dependency_closure": dependency,
        "parent_count": len(parent_names),
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != v7_finalizer.EVAL_PROTOCOL_SCHEMA:
        raise V7EvaluationError("unexpected v7 evaluation-protocol schema")
    if protocol.get("status") != v7_finalizer.EVAL_PROTOCOL_STATUS:
        raise V7EvaluationError("v7 evaluation protocol is not frozen pre-generation")
    if protocol.get("experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS["evaluation"]:
        raise V7EvaluationError("v7 evaluation protocol Experiment ID differs")
    if (
        protocol.get("development_only") is not True
        or protocol.get("globally_fresh") is not False
        or protocol.get("independent_confirmation") is not False
    ):
        raise V7EvaluationError("v7 development-only scientific boundary differs")
    if protocol.get("gold_access_during_planning_retrieval_subanswer_and_merge") is not False:
        raise V7EvaluationError("protocol does not attest Gold-free materialization")
    estimand = protocol.get("estimand") or {}
    if (
        estimand.get("analysis_population") != "ITT_ALL_40_WITH_FALLBACKS_INCLUDED"
        or estimand.get("primary") != v7_freeze.UTILITY_GATES["primary_comparison"]
        or estimand.get("secondary")
        != v7_freeze.UTILITY_GATES["secondary_standard_baseline_comparison"]
        or estimand.get("subgroup_selection") is not False
    ):
        raise V7EvaluationError("v7 ITT estimand differs from preregistration")
    generation = protocol.get("generation") or {}
    expected = {
        "prompt": "canonical legacy build_rl_messages",
        "kg_subgraph": [],
        "top_k_passages": 10,
        "decode": "greedy",
        "do_sample": False,
        "max_new_tokens": 512,
        "seed": 42,
        "identical_prompt_reuse": "byte-exact generation and parsed score reuse within question",
        "single_model_load_for_all_three_arms": True,
    }
    for name, value in expected.items():
        if generation.get(name) != value:
            raise V7EvaluationError(f"frozen final-generation setting differs: {name}")
    gates = protocol.get("decision_gates") or {}
    if gates.get("development_utility") != v7_freeze.UTILITY_GATES:
        raise V7EvaluationError("protocol utility gates differ from preregistration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm_a")
    parser.add_argument("--arm_b")
    parser.add_argument("--arm_c")
    parser.add_argument("--adapter")
    parser.add_argument("--base_model")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--experiment_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    authorization_attestation = validate_evaluation_authorization(
        protocol_path, protocol
    )
    experiment_id = str(protocol["experiment_id"])
    if args.experiment_id and args.experiment_id != experiment_id:
        raise V7EvaluationError("Experiment ID differs from frozen protocol")

    paths = {
        key: _resolved_locked_file(
            protocol, PROTOCOL_INPUT_KEYS[key], getattr(args, f"arm_{key.lower()}")
        )
        for key in ARM_KEYS
    }
    arms = {key: _read_jsonl(path) for key, path in paths.items()}
    validate_scored_inputs(protocol, arms)

    models = protocol.get("models") or {}
    identities = {
        "strong_sft": models.get("strong_sft") or {},
        "base_model": models.get("base_model") or {},
    }
    content_locks = models.get("content_locks") or {}
    adapter_path = Path(
        args.adapter or str(identities["strong_sft"].get("path") or "")
    ).expanduser().resolve()
    base_path = Path(
        args.base_model or str(identities["base_model"].get("path") or "")
    ).expanduser().resolve()
    _validate_model_tree(adapter_path, content_locks["strong_sft"], label="strong_sft")
    _validate_model_tree(base_path, content_locks["base_model"], label="base_model")
    evaluator_lock = (protocol.get("code") or {}).get("evaluator")
    if not isinstance(evaluator_lock, Mapping) or not v7_finalizer._lock_equal(
        v7_finalizer._file_lock(Path(__file__)), evaluator_lock
    ):
        raise V7EvaluationError("evaluator code differs from the pre-execution implementation lock")

    generation_cfg = protocol["generation"]
    seed = int(generation_cfg["seed"])
    max_new_tokens = int(generation_cfg["max_new_tokens"])
    run_dir, reserved_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=experiment_id,
        extra={
            "phase": "paired_dependent_retrieval_v7_answer_evaluation",
            "evaluator_version": EVALUATOR_VERSION,
            "protocol_sha256": _sha256_file(protocol_path),
            "analysis_population": "ITT_ALL_40_WITH_FALLBACKS_INCLUDED",
        },
    )
    try:
        # Delayed imports guarantee that all frozen inputs and model hashes are
        # checked before CUDA initialization.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model loaded on {device}; refusing CPU/disk fallback")

        input_hashes = {key: _sha256_file(path) for key, path in paths.items()}
        prompt_cache: dict[str, dict[str, Any]] = {}
        predictions: list[dict[str, Any]] = []
        for index, triple in enumerate(zip(arms["A"], arms["B"], arms["C"]), start=1):
            for arm_key, arm_name, row in zip(ARM_KEYS, ARMS, triple):
                messages = build_rl_messages(
                    question=str(row["question"]),
                    retrieved_passages=list(row["retrieved_passages"]),
                    kg_triples=[],
                    top_k=10,
                )
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_hash = _sha256_text(prompt)
                cached = prompt_cache.get(prompt_hash)
                if cached is None:
                    encoded = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    ).to(device)
                    prompt_tokens = int(encoded["input_ids"].shape[1])
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    generation = tokenizer.decode(
                        output[0, prompt_tokens:], skip_special_tokens=True
                    )
                    cached = {
                        "generation": generation,
                        "prompt_tokens": prompt_tokens,
                        "first_question_key": str(row["question_key"]),
                        "first_arm": arm_name,
                    }
                    prompt_cache[prompt_hash] = cached
                    reused = None
                else:
                    generation = str(cached["generation"])
                    prompt_tokens = int(cached["prompt_tokens"])
                    reused = {
                        "question_key": str(cached["first_question_key"]),
                        "arm": str(cached["first_arm"]),
                    }
                predictions.append(
                    score_generation(
                        row=row,
                        generation=generation,
                        prompt_sha256=prompt_hash,
                        prompt_tokens=prompt_tokens,
                        arm=arm_name,
                        input_sha256=input_hashes[arm_key],
                        reused_identical_prompt_from=reused,
                    )
                )
            print(f"v7 paired A/B/C inference {index}/{len(arms['A'])}", flush=True)

        if len(predictions) != 3 * int(protocol["n"]):
            raise V7EvaluationError("prediction population is incomplete")
        predictions_path = run_dir / "predictions.jsonl"
        _write_jsonl_exclusive(predictions_path, predictions)
        metrics = summarize_predictions(predictions)
        fallback = validate_fallback_prediction_reuse(predictions, arms)
        decision = evaluate_utility_gates(
            protocol["decision_gates"]["development_utility"], metrics
        )
        status = (
            "PASS_DEVELOPMENT_FEASIBILITY_ADVANCE_TO_GENUINELY_UNSEEN_CONFIRMATION"
            if decision["all_pass"]
            else "FAIL_STOP_DEVELOPMENT_FEASIBILITY"
        )
        report = {
            "schema_version": RESULT_SCHEMA,
            "experiment_id": reserved_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "scope": protocol["scope"],
            "evaluator_version": EVALUATOR_VERSION,
            "analysis_population": "ITT_ALL_40_WITH_FALLBACKS_INCLUDED",
            "primary_comparison": "C_minus_B",
            "secondary_comparison": "C_minus_A",
            "protocol": {
                "path": str(protocol_path),
                "sha256": _sha256_file(protocol_path),
            },
            "evaluation_authorization": authorization_attestation,
            "generation": generation_cfg,
            **metrics,
            "fallback_reuse": fallback,
            "decision": decision,
            "inputs": {
                "arms": {
                    key: {"path": str(paths[key]), "sha256": input_hashes[key]}
                    for key in ARM_KEYS
                },
                "strong_sft": artifact_identity(adapter_path),
                "base_model": artifact_identity(base_path),
            },
            "outputs": {
                "predictions": {
                    "path": str(predictions_path),
                    "sha256": _sha256_file(predictions_path),
                }
            },
            "scientific_boundary": (
                "Globally consumed development rows only. Passing these frozen gates is "
                "feasibility evidence, not independent confirmation or a paper claim."
            ),
        }
        report_path = run_dir / "report.json"
        _write_json_exclusive(report_path, report)
        dump_manifest(
            run_dir,
            status=status,
            extra={
                "experiment_id": reserved_id,
                "phase": "paired_dependent_retrieval_v7_answer_evaluation",
                "report_sha256": _sha256_file(report_path),
                "decision": decision,
            },
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "overall": metrics["overall"],
                    "decision": decision,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except BaseException as exc:
        dump_manifest(
            run_dir,
            status="ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_RUNTIME",
            extra={
                "experiment_id": reserved_id,
                "phase": "paired_dependent_retrieval_v7_answer_evaluation",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
