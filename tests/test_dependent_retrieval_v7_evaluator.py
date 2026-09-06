from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.eval import evaluate_paired_dependent_retrieval_v7 as evaluator
from scripts.prepare import freeze_dependent_retrieval_v7 as freeze


def _passages(label: str) -> list[dict[str, str]]:
    return [
        {
            "id": f"{label}-{index}",
            "title": f"Title {label} {index}",
            "contents": f"Title {label} {index}\nSynthetic evidence {index}.",
        }
        for index in range(10)
    ]


def _scored_inputs() -> tuple[dict, dict[str, list[dict]]]:
    arms = {"A": [], "B": [], "C": []}
    for dataset in evaluator.DATASETS:
        for index in range(2):
            qid = f"{dataset}-{index}"
            key = f"{dataset}::{qid}"
            question = f"Synthetic question {key}?"
            passages_a = _passages(f"{qid}-a")
            passages_b = passages_a if index == 0 else _passages(f"{qid}-b")
            passages_c = passages_a if index == 0 else _passages(f"{qid}-c")
            common = {
                "schema_version": "paired-dependent-retrieval-v7-arm-1",
                "row_id": f"dependent-retrieval-v7::{key}",
                "question_key": key,
                "dataset": dataset,
                "qid": qid,
                "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "family_sha256": hashlib.sha256(f"family:{key}".encode()).hexdigest(),
                "role": "development_consumed",
                "gold_answers": [f"answer-{qid}"],
                "gold_attachment": "SCORER_ONLY_AFTER_ALL_GOLD_FREE_GATES",
                "kg_subgraph": [],
                "gold_access": False,
            }
            for arm_key, arm_name, passages in zip(
                evaluator.ARM_KEYS, evaluator.ARMS, (passages_a, passages_b, passages_c)
            ):
                row = {
                    **common,
                    "arm": arm_name,
                    "retrieved_passages": deepcopy(passages),
                    "passages_sha256": "unused-by-evaluator",
                }
                if arm_key != "A":
                    row["fallback_to_a"] = index == 0
                    row["retrieval_trace"] = {"gold_access": False}
                arms[arm_key].append(row)
    keys = [row["question_key"] for row in arms["A"]]
    qids = [row["qid"] for row in arms["A"]]
    protocol = {
        "n": 4,
        "by_dataset": {"hotpotqa": 2, "musique": 2},
        "arms": list(evaluator.ARMS),
        "qid_order_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
        "question_key_order_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }
    return protocol, arms


def _prediction(
    key: str,
    dataset: str,
    arm: str,
    em: float,
    *,
    f1: float | None = None,
    parsed: bool = True,
    fallback: bool = False,
) -> dict:
    return {
        "question_key": key,
        "dataset": dataset,
        "qid": key.split("::", 1)[1],
        "arm": arm,
        "em": float(em),
        "f1": float(em if f1 is None else f1),
        "prediction": "correct" if em else "wrong",
        "well_formed": parsed,
        "contiguous": parsed,
        "gold_in_passages": False,
        "n_steps": 1 if parsed else 0,
        "fallback_to_a": fallback,
        "prompt_sha256": f"prompt::{key}::{arm}",
        "generation": f"generation::{key}::{arm}",
    }


def _passing_predictions() -> list[dict]:
    rows: list[dict] = []
    # C gains two Hotpot questions over both A and B; all MuSiQue questions tie.
    values = {
        "hotpotqa::h0": (0, 0, 1),
        "hotpotqa::h1": (0, 0, 1),
        "musique::m0": (1, 1, 1),
        "musique::m1": (0, 0, 0),
    }
    for key, scores in values.items():
        dataset = key.split("::", 1)[0]
        for arm, score in zip(evaluator.ARMS, scores):
            rows.append(_prediction(key, dataset, arm, score))
    return rows


def test_validate_scored_inputs_requires_complete_strict_itt_population() -> None:
    protocol, arms = _scored_inputs()
    evaluator.validate_scored_inputs(protocol, arms)
    incomplete = deepcopy(arms)
    incomplete["C"].pop()
    with pytest.raises(evaluator.V7EvaluationError, match="strict ITT population"):
        evaluator.validate_scored_inputs(protocol, incomplete)


def test_validate_scored_inputs_rejects_nonempty_kg_confound() -> None:
    protocol, arms = _scored_inputs()
    arms["C"][0]["kg_subgraph"] = [["x", "r", "y"]]
    with pytest.raises(evaluator.V7EvaluationError, match="non-empty KG"):
        evaluator.validate_scored_inputs(protocol, arms)


def test_score_generation_uses_canonical_parser_and_gold() -> None:
    _, arms = _scored_inputs()
    row = deepcopy(arms["A"][0])
    row["gold_answers"] = ["Synthetic Place"]
    generation = """[Step 1]\nReasoning: Read the evidence.\nKnowledge Used: []\nConclusion: The place is Synthetic Place.\n\n[Final Answer] Synthetic Place\n"""
    scored = evaluator.score_generation(
        row=row,
        generation=generation,
        prompt_sha256="a" * 64,
        prompt_tokens=100,
        arm=evaluator.ARMS[0],
        input_sha256="b" * 64,
        reused_identical_prompt_from=None,
    )
    assert scored["prediction"] == "Synthetic Place"
    assert scored["em"] == 1.0
    assert scored["f1"] == 1.0
    assert scored["well_formed"] is True


def test_paired_metrics_report_ci_gained_lost_and_parse() -> None:
    rows = []
    scores = {
        "hotpotqa::q0": (0, 1),
        "hotpotqa::q1": (1, 0),
        "hotpotqa::q2": (0, 1),
    }
    for key, (baseline, treatment) in scores.items():
        rows.append(_prediction(key, "hotpotqa", evaluator.ARMS[1], baseline))
        rows.append(
            _prediction(
                key,
                "hotpotqa",
                evaluator.ARMS[2],
                treatment,
                parsed=key != "hotpotqa::q1",
            )
        )
    result = evaluator.paired_comparison(
        rows,
        baseline_arm=evaluator.ARMS[1],
        treatment_arm=evaluator.ARMS[2],
        bootstrap_seed=7,
    )
    assert result["analysis_population"] == "ITT_ALL_ROWS_WITH_FALLBACKS_INCLUDED"
    assert result["gained_correct"] == 2
    assert result["lost_correct"] == 1
    assert result["net_correct"] == 1
    assert result["parse_count_delta"] == -1
    assert len(result["delta_em_bootstrap_95ci"]) == 2
    assert len(result["delta_f1_bootstrap_95ci"]) == 2


def test_summary_and_frozen_primary_secondary_gates_pass() -> None:
    predictions = _passing_predictions()
    metrics = evaluator.summarize_predictions(predictions)
    assert metrics["overall"]["C_minus_B"]["n"] == 4
    assert metrics["overall"]["C_minus_B"]["net_correct"] == 2
    assert metrics["overall"]["C_minus_A"]["net_correct"] == 2
    decision = evaluator.evaluate_utility_gates(freeze.UTILITY_GATES, metrics)
    assert decision["primary"]["all_pass"] is True
    assert decision["secondary"]["all_pass"] is True
    assert decision["all_pass"] is True


def test_primary_failure_cannot_be_rescued_by_secondary() -> None:
    metrics = evaluator.summarize_predictions(_passing_predictions())
    metrics["overall"]["C_minus_B"]["net_correct"] = 1
    decision = evaluator.evaluate_utility_gates(freeze.UTILITY_GATES, metrics)
    assert decision["primary"]["all_pass"] is False
    assert decision["secondary"]["all_pass"] is True
    assert decision["all_pass"] is False


def test_fallback_prediction_reuse_must_be_byte_exact() -> None:
    _, arms = _scored_inputs()
    predictions: list[dict] = []
    for triple in zip(arms["A"], arms["B"], arms["C"]):
        key = triple[0]["question_key"]
        for arm_name, row in zip(evaluator.ARMS, triple):
            fallback = bool(row.get("fallback_to_a"))
            prompt = f"prompt::{key}"
            generation = f"generation::{key}"
            predictions.append(
                {
                    "question_key": key,
                    "arm": arm_name,
                    "prompt_sha256": prompt if fallback or arm_name == evaluator.ARMS[0] else prompt + arm_name,
                    "generation": generation if fallback or arm_name == evaluator.ARMS[0] else generation + arm_name,
                    "prediction": "answer" if fallback or arm_name == evaluator.ARMS[0] else arm_name,
                }
            )
    result = evaluator.validate_fallback_prediction_reuse(predictions, arms)
    assert result["fallback_arm_rows_checked"] == 4
    broken = deepcopy(predictions)
    target = next(
        row
        for row in broken
        if row["question_key"].endswith("-0") and row["arm"] == evaluator.ARMS[2]
    )
    target["generation"] = "different"
    with pytest.raises(evaluator.V7EvaluationError, match="not byte-reused"):
        evaluator.validate_fallback_prediction_reuse(broken, arms)


def test_utility_gate_configuration_cannot_drift() -> None:
    metrics = evaluator.summarize_predictions(_passing_predictions())
    gates = dict(freeze.UTILITY_GATES)
    gates["C_minus_B_pooled_net_correct_min"] = 1
    with pytest.raises(evaluator.V7EvaluationError, match="differ from preregistration"):
        evaluator.evaluate_utility_gates(gates, metrics)


def test_evaluator_rejects_tampered_indirect_scoring_dependency() -> None:
    closure = evaluator._current_evaluator_import_closure()
    protocol = {
        "code": {
            "evaluator_import_closure": deepcopy(closure),
            "required_evaluator_dependencies": sorted(
                evaluator.v7_finalizer.REQUIRED_EVALUATOR_DEPENDENCIES
            ),
            "evaluator": closure[
                "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"
            ],
            "gold_finalizer": closure[
                "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"
            ],
        }
    }
    protocol["code"]["evaluator_import_closure"][
        "kgproweight/eval/metrics.py"
    ]["sha256"] = "0" * 64
    with pytest.raises(evaluator.V7EvaluationError, match="dependency content differs"):
        evaluator.validate_evaluator_dependency_closure(protocol)


def test_evaluator_rejects_manual_protocol_without_complete_parent_chain(
    tmp_path: Path,
) -> None:
    closure = evaluator._current_evaluator_import_closure()
    protocol = {
        "experiment_id": freeze.FUTURE_EXPERIMENT_IDS["evaluation"],
        "gold_attachment_experiment_id": freeze.FUTURE_EXPERIMENT_IDS[
            "gold_attachment"
        ],
        "materialization_experiment_id": freeze.FUTURE_EXPERIMENT_IDS[
            "materialization"
        ],
        "code": {
            "evaluator_import_closure": closure,
            "required_evaluator_dependencies": sorted(
                evaluator.v7_finalizer.REQUIRED_EVALUATOR_DEPENDENCIES
            ),
            "evaluator": closure[
                "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"
            ],
            "gold_finalizer": closure[
                "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"
            ],
        },
        "authorization": {
            "schema_version": evaluator.v7_finalizer.EVAL_AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED_FOR_FROZEN_V7_ANSWER_EVALUATION_ONLY",
            "issuer": "paired-dependent-retrieval-v7-gold-finalizer",
            "issuer_code": closure[
                "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"
            ],
            "gold_attachment_complete": True,
            "answer_evaluation": True,
            "training": False,
            "evaluation_experiment_id": freeze.FUTURE_EXPERIMENT_IDS["evaluation"],
            "gold_attachment_experiment_id": freeze.FUTURE_EXPERIMENT_IDS[
                "gold_attachment"
            ],
            "parent_chain": {},
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "status": "FROZEN_READY_FOR_V7_ANSWER_EVALUATION",
        "run": {
            "phase": "paired_dependent_retrieval_v7_gold_attachment",
            "experiment_id": freeze.FUTURE_EXPERIMENT_IDS["gold_attachment"],
            "evaluation_experiment_id": freeze.FUTURE_EXPERIMENT_IDS["evaluation"],
            "protocol_sha256": evaluator._sha256_file(protocol_path),
            "gold_opened_only_after_gold_free_gates": True,
            "answer_evaluation_authorized": True,
            "evaluation_authorization_schema": (
                evaluator.v7_finalizer.EVAL_AUTHORIZATION_SCHEMA
            ),
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(evaluator.V7EvaluationError, match="parent-chain roles"):
        evaluator.validate_evaluation_authorization(protocol_path, protocol)
