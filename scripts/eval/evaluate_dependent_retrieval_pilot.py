#!/usr/bin/env python
"""Evaluate frozen question-only vs dependency-aware retrieval arms.

This evaluator is intentionally independent of the canonical KG-ProWeight
pipeline.  Retrieval has already happened: it consumes two versioned JSONL
arms, verifies that question, Gold and legacy-KG fields are paired, and runs
the same strong-SFT adapter with greedy decoding on both passage views.

Gold answers may be present in the frozen arm files only because the retrieval
runner appends them *after* both passage arms have been materialised.  The
protocol must explicitly attest ``gold_access_during_retrieval=false``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch

from kgproweight.data.prompts import build_rl_messages
from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.eval.evaluate_a1_fixed_context_kg import _aggregate, _score_generation
from scripts.pilot.score_a1_fixed_context_kg import _bootstrap_ci, _mcnemar_exact


EVALUATOR_VERSION = "dependent-retrieval-pilot-eval-1"
ARMS = ("A_question_only", "B_dependent")
COMMON_FIELDS = (
    "row_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "gold_answers",
    "kg_subgraph",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _resolved_protocol_input(
    protocol: Mapping[str, Any], name: str, cli_value: str | None
) -> tuple[Path, Mapping[str, Any]]:
    lock = protocol["inputs"][name]
    locked_path = Path(str(lock["path"])).expanduser().resolve()
    path = Path(cli_value).expanduser().resolve() if cli_value else locked_path
    if path != locked_path:
        raise ValueError(f"{name} path differs from frozen protocol")
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != str(lock["sha256"]):
        raise ValueError(f"{name} hash differs from frozen protocol")
    return path, lock


def _verify_model_lock(root: Path, lock: Mapping[str, Any], *, kind: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    locked_path = lock.get("path")
    if locked_path is not None and root != Path(str(locked_path)).expanduser().resolve():
        raise ValueError(f"{kind} path differs from frozen protocol")

    # New protocols may store the complete artifact_identity dictionary.
    identity = lock.get("artifact_identity")
    if identity is not None and artifact_identity(root) != identity:
        raise ValueError(f"{kind} artifact identity differs from frozen protocol")
    if all(key in lock for key in ("files", "inventory_sha256")):
        if artifact_identity(root) != dict(lock):
            raise ValueError(f"{kind} artifact identity differs from frozen protocol")

    expected_files = {
        "adapter_config_sha256": "adapter_config.json",
        "adapter_model_sha256": "adapter_model.safetensors",
        "config_sha256": "config.json",
        "model_index_sha256": "model.safetensors.index.json",
    }
    verified = bool(identity is not None or "inventory_sha256" in lock)
    for key, filename in expected_files.items():
        if key not in lock:
            continue
        file_path = root / filename
        if not file_path.is_file() or _sha256_file(file_path) != str(lock[key]):
            raise ValueError(f"{kind} {filename} differs from frozen protocol")
        verified = True
    if not verified:
        raise ValueError(f"{kind} protocol lock contains no recognised content hash")


def _common_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in COMMON_FIELDS if key not in row]
    if missing:
        raise ValueError(f"row is missing paired fields: {missing}")
    return {key: row[key] for key in COMMON_FIELDS}


def _validate_rows(
    protocol: Mapping[str, Any],
    arm_a: Sequence[Mapping[str, Any]],
    arm_b: Sequence[Mapping[str, Any]],
) -> None:
    expected_n = int(protocol["n"])
    if len(arm_a) != expected_n or len(arm_b) != expected_n:
        raise ValueError(
            f"strict paired population broken: A={len(arm_a)}, B={len(arm_b)}, n={expected_n}"
        )
    if [_common_projection(row) for row in arm_a] != [
        _common_projection(row) for row in arm_b
    ]:
        raise ValueError("A/B arms differ outside retrieved passages and arm telemetry")

    question_keys = [str(row["question_key"]) for row in arm_a]
    if len(set(question_keys)) != expected_n:
        raise ValueError("duplicate question_key in paired input")
    locked_qid_sha = protocol.get("qid_order_sha256")
    if locked_qid_sha is not None:
        current = _sha256_text("\n".join(str(row["qid"]) for row in arm_a))
        if current != str(locked_qid_sha):
            raise ValueError("qid order hash differs from frozen protocol")

    top_k = int(protocol["generation"]["top_k_passages"])
    for index, (left, right) in enumerate(zip(arm_a, arm_b)):
        for arm, row in (("A", left), ("B", right)):
            if not isinstance(row.get("retrieved_passages"), list):
                raise ValueError(f"{arm}[{index}] retrieved_passages is not a list")
            if len(row["retrieved_passages"]) != top_k:
                raise ValueError(
                    f"{arm}[{index}] passage count={len(row['retrieved_passages'])}, expected {top_k}"
                )
            # Project question identities intentionally strip outer whitespace;
            # two frozen Hotpot questions carry a trailing space in the raw row.
            if question_sha256(str(row["question"])) != str(row["question_sha256"]):
                raise ValueError(f"{arm}[{index}] question hash mismatch")
        if bool(right.get("fallback_to_a")) and right["retrieved_passages"] != left["retrieved_passages"]:
            raise ValueError(f"B[{index}] claims fallback_to_a but passages differ")


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(row["question_key"]), str(row["arm"])): row
        for row in rows
    }
    keys = [
        str(row["question_key"])
        for row in rows
        if row["arm"] == ARMS[0]
    ]
    pairs = [(by_key[(key, ARMS[0])], by_key[(key, ARMS[1])]) for key in keys]
    em_diffs = [float(right["em"]) - float(left["em"]) for left, right in pairs]
    f1_diffs = [float(right["f1"]) - float(left["f1"]) for left, right in pairs]
    gained = sum(float(right["em"]) > float(left["em"]) for left, right in pairs)
    lost = sum(float(right["em"]) < float(left["em"]) for left, right in pairs)
    tied = len(pairs) - gained - lost
    changed = sum(
        str(left["prediction"]).strip() != str(right["prediction"]).strip()
        for left, right in pairs
    )
    a_em = sum(float(left["em"]) for left, _ in pairs) / max(1, len(pairs))
    b_em = sum(float(right["em"]) for _, right in pairs) / max(1, len(pairs))
    a_f1 = sum(float(left["f1"]) for left, _ in pairs) / max(1, len(pairs))
    b_f1 = sum(float(right["f1"]) for _, right in pairs) / max(1, len(pairs))
    a_parse = sum(bool(left["well_formed"]) for left, _ in pairs)
    b_parse = sum(bool(right["well_formed"]) for _, right in pairs)
    return {
        "n": len(pairs),
        "arm_a_em": a_em,
        "arm_b_em": b_em,
        "delta_em": b_em - a_em,
        "delta_em_bootstrap_95ci": _bootstrap_ci(em_diffs, seed=20260903),
        "arm_a_f1": a_f1,
        "arm_b_f1": b_f1,
        "delta_f1": b_f1 - a_f1,
        "delta_f1_bootstrap_95ci": _bootstrap_ci(f1_diffs, seed=20260904),
        "gained_correct": gained,
        "lost_correct": lost,
        "tied_correctness": tied,
        "net_correct": gained - lost,
        "mcnemar_exact_p": _mcnemar_exact(gained, lost),
        "prediction_changed": changed,
        "arm_a_parse_rate": a_parse / max(1, len(pairs)),
        "arm_b_parse_rate": b_parse / max(1, len(pairs)),
        "parse_count_delta": b_parse - a_parse,
    }


def _trace_number(row: Mapping[str, Any], name: str) -> int | None:
    trace = row.get("retrieval_trace")
    if not isinstance(trace, Mapping) or name not in trace:
        return None
    value = trace[name]
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mechanism(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "plan_executable": [],
        "has_dependent_step": [],
        "dependent_query_count": [],
        "second_hop_query_count": [],
        "new_dependent_candidate_count": [],
    }
    for row in rows:
        for name in fields:
            value = _trace_number(row, name)
            if value is not None:
                fields[name].append(value)
    n = len(rows)
    eligible_rows = [
        row for row in rows
        if _trace_number(row, "has_dependent_step") == 1
    ]
    eligible_second_hop_values = [
        value
        for row in eligible_rows
        if (value := _trace_number(row, "second_hop_query_count")) is not None
    ]
    dependent_eligible_n = len(eligible_rows)
    return {
        "n": n,
        "plan_executable_observed_n": len(fields["plan_executable"]),
        "plan_executable_rate": (
            sum(value > 0 for value in fields["plan_executable"]) / n if n else 0.0
        ),
        "has_dependent_step_observed_n": len(fields["has_dependent_step"]),
        "dependent_step_eligible_n": dependent_eligible_n,
        "second_hop_query_observed_n": len(eligible_second_hop_values),
        "second_hop_query_nonempty_rate": (
            sum(value > 0 for value in eligible_second_hop_values) / dependent_eligible_n
            if dependent_eligible_n else None
        ),
        "new_dependent_candidate_observed_n": len(fields["new_dependent_candidate_count"]),
        "new_dependent_candidate_question_rate": (
            sum(value > 0 for value in fields["new_dependent_candidate_count"]) / n if n else 0.0
        ),
        "mean_dependent_queries": (
            sum(fields["dependent_query_count"]) / n if n else 0.0
        ),
        "fallback_count": sum(bool(row.get("fallback_to_a")) for row in rows),
        "fallback_reasons": dict(
            Counter(
                str((row.get("retrieval_trace") or {}).get("fallback_reason") or "unspecified")
                for row in rows
                if row.get("fallback_to_a")
            )
        ),
    }


def _require_gate(gates: Mapping[str, Any], name: str) -> Any:
    if name not in gates:
        raise ValueError(f"protocol decision_gates is missing {name}")
    return gates[name]


def _decision(
    protocol: Mapping[str, Any],
    overall: Mapping[str, Any],
    by_dataset: Mapping[str, Mapping[str, Any]],
    mechanism: Mapping[str, Mapping[str, Any]],
    fallback_exact: bool,
) -> dict[str, Any]:
    cfg = protocol["decision_gates"]
    checks: dict[str, bool] = {
        "pooled_net_correct_gain": overall["net_correct"]
        >= int(_require_gate(cfg, "pooled_net_correct_gain_min")),
        "pooled_f1_positive": overall["delta_f1"] > 0.0,
        "no_dataset_net_loss_over_limit": all(
            value["net_correct"] >= -int(_require_gate(cfg, "max_net_correct_loss_per_dataset"))
            for value in by_dataset.values()
        ),
        "parse_not_degraded": overall["parse_count_delta"]
        >= int(_require_gate(cfg, "parse_count_delta_min")),
        "fallback_predictions_exact": fallback_exact,
    }

    optional_mechanism_gates = {
        "plan_executable_rate_min_each_dataset": (
            "plan_executable_observed_n",
            "plan_executable_rate",
            "plan_executable_rate",
        ),
        "second_hop_query_nonempty_rate_min_each_dataset": (
            "second_hop_query_observed_n",
            "second_hop_query_nonempty_rate",
            "second_hop_query_nonempty_rate",
        ),
        "new_dependent_candidate_question_rate_min_each_dataset": (
            "new_dependent_candidate_observed_n",
            "new_dependent_candidate_question_rate",
            "new_dependent_candidate_question_rate",
        ),
    }
    for gate_name, (observed_name, metric_name, check_name) in optional_mechanism_gates.items():
        if gate_name not in cfg:
            continue
        threshold = float(cfg[gate_name])
        if gate_name == "second_hop_query_nonempty_rate_min_each_dataset":
            # Only questions whose frozen plan actually contains a dependency
            # are eligible.  Parallel independent roots are not second hops.
            incomplete = [
                dataset
                for dataset, values in mechanism.items()
                if int(values.get("has_dependent_step_observed_n", 0)) != int(values["n"])
                or int(values[observed_name]) != int(values["dependent_step_eligible_n"])
            ]
            if incomplete:
                raise ValueError(
                    f"protocol declares {gate_name}, but eligible retrieval_trace "
                    f"is incomplete for {incomplete}"
                )
            no_eligible = [
                dataset for dataset, values in mechanism.items()
                if int(values.get("dependent_step_eligible_n", 0)) <= 0
            ]
            if no_eligible:
                checks[check_name] = False
                continue
        else:
            missing = [
                dataset
                for dataset, values in mechanism.items()
                if int(values[observed_name]) != int(values["n"])
            ]
            if missing:
                raise ValueError(
                    f"protocol declares {gate_name}, but retrieval_trace is "
                    f"incomplete for {missing}"
                )
        checks[check_name] = all(
            values[metric_name] is not None and float(values[metric_name]) >= threshold
            for values in mechanism.values()
        )
    return {"checks": checks, "all_pass": all(checks.values())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--arm_a", help="Defaults to protocol.inputs.arm_a.path")
    parser.add_argument("--arm_b", help="Defaults to protocol.inputs.arm_b.path")
    parser.add_argument("--adapter", help="Defaults to protocol.models.strong_sft.path")
    parser.add_argument("--base_model", help="Defaults to protocol.base_model.path")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", help="Must equal the frozen protocol ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = str(protocol.get("status") or "")
    if "FROZEN" not in status or status.startswith("FAIL"):
        raise ValueError(f"protocol is not a frozen pre-generation protocol: {status!r}")
    if protocol.get("gold_access_during_retrieval") is not False:
        raise ValueError("protocol must attest gold_access_during_retrieval=false")

    experiment_id = str(protocol["experiment_id"])
    if args.experiment_id and args.experiment_id != experiment_id:
        raise ValueError("experiment_id differs from frozen protocol")
    arm_a_path, _ = _resolved_protocol_input(protocol, "arm_a", args.arm_a)
    arm_b_path, _ = _resolved_protocol_input(protocol, "arm_b", args.arm_b)
    arm_a = _read_jsonl(arm_a_path)
    arm_b = _read_jsonl(arm_b_path)
    _validate_rows(protocol, arm_a, arm_b)

    strong_sft_lock = protocol["models"]["strong_sft"]
    adapter_path = Path(args.adapter or strong_sft_lock["path"]).expanduser().resolve()
    base_lock = protocol["base_model"]
    base_path = Path(args.base_model or base_lock["path"]).expanduser().resolve()
    _verify_model_lock(adapter_path, strong_sft_lock, kind="strong_sft")
    _verify_model_lock(base_path, base_lock, kind="base_model")

    generation_cfg = protocol["generation"]
    seed = int(generation_cfg["seed"])
    max_new_tokens = int(generation_cfg["max_new_tokens"])
    if max_new_tokens != 512 or bool(generation_cfg.get("do_sample", False)):
        raise ValueError("pilot requires frozen greedy max_new_tokens=512")

    run_dir, frozen_experiment_id = prepare_new_run_dir(
        args.run_dir,
        experiment_id=experiment_id,
        extra={
            "phase": "dependent_retrieval_zero_training_pilot",
            "evaluator_version": EVALUATOR_VERSION,
            "protocol_sha256": _sha256_file(protocol_path),
        },
    )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        if not str(model.device).startswith("cuda"):
            raise RuntimeError(f"model loaded on {model.device}; refusing silent CPU fallback")
        print(f"dependent-retrieval evaluator device={model.device}", flush=True)

        input_hashes = {
            ARMS[0]: _sha256_file(arm_a_path),
            ARMS[1]: _sha256_file(arm_b_path),
        }
        predictions: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(arm_a, arm_b), start=1):
            prompt_cache: dict[str, dict[str, Any]] = {}
            for arm, row in ((ARMS[0], left), (ARMS[1], right)):
                messages = build_rl_messages(
                    question=str(row["question"]),
                    retrieved_passages=list(row["retrieved_passages"]),
                    kg_triples=list(row["kg_subgraph"]),
                    top_k=int(generation_cfg["top_k_passages"]),
                )
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_sha = _sha256_text(prompt)
                if prompt_sha in prompt_cache:
                    scored = deepcopy(prompt_cache[prompt_sha])
                    scored.update(
                        {
                            "arm": arm,
                            "input_sha256": input_hashes[arm],
                            "fallback_to_a": bool(row.get("fallback_to_a")),
                            "retrieval_trace": row.get("retrieval_trace"),
                            "reused_identical_prompt_from_arm": prompt_cache[prompt_sha]["arm"],
                        }
                    )
                    predictions.append(scored)
                    continue

                encoded = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generation = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
                )
                scored = _score_generation(
                    row=row,
                    generation=generation,
                    prompt_sha256=prompt_sha,
                    prompt_tokens=int(encoded["input_ids"].shape[1]),
                    model_label="strong_sft",
                    arm=arm,
                    input_sha256=input_hashes[arm],
                )
                scored.update(
                    {
                        "question_key": str(row["question_key"]),
                        "fallback_to_a": bool(row.get("fallback_to_a")),
                        "retrieval_trace": row.get("retrieval_trace"),
                        "reused_identical_prompt_from_arm": None,
                    }
                )
                predictions.append(scored)
                prompt_cache[prompt_sha] = scored
            print(f"dependent-retrieval paired inference {index}/{len(arm_a)}", flush=True)

        predictions_path = run_dir / "predictions.jsonl"
        _write_jsonl(predictions_path, predictions)
        expected = len(arm_a) * 2
        if len(predictions) != expected:
            raise ValueError(f"prediction count {len(predictions)} != {expected}")

        by_arm = {
            arm: _aggregate([row for row in predictions if row["arm"] == arm])
            for arm in ARMS
        }
        overall = _paired(predictions)
        datasets = sorted({str(row["dataset"]) for row in arm_a})
        by_dataset: dict[str, Any] = {}
        mechanism: dict[str, Any] = {}
        for dataset in datasets:
            current_predictions = [row for row in predictions if row["dataset"] == dataset]
            current_b = [row for row in arm_b if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "arms": {
                    arm: _aggregate([row for row in current_predictions if row["arm"] == arm])
                    for arm in ARMS
                },
                "B_minus_A": _paired(current_predictions),
            }
            mechanism[dataset] = _mechanism(current_b)

        by_prediction = {
            (str(row["question_key"]), str(row["arm"])): row for row in predictions
        }
        fallback_pairs = [
            (
                by_prediction[(str(row["question_key"]), ARMS[0])],
                by_prediction[(str(row["question_key"]), ARMS[1])],
            )
            for row in arm_b
            if row.get("fallback_to_a")
        ]
        fallback_prompt_exact = all(
            left["prompt_sha256"] == right["prompt_sha256"] for left, right in fallback_pairs
        )
        fallback_prediction_exact = all(
            left["prediction"] == right["prediction"]
            and left["generation"] == right["generation"]
            for left, right in fallback_pairs
        )
        fallback_exact = fallback_prompt_exact and fallback_prediction_exact
        decision = _decision(protocol, overall, {
            dataset: values["B_minus_A"] for dataset, values in by_dataset.items()
        }, mechanism, fallback_exact)
        result_status = (
            "PASS_DEVELOPMENT_FEASIBILITY_ADVANCE_TO_FRESH_CONFIRMATION"
            if decision["all_pass"]
            else "FAIL_STOP_DEVELOPMENT_FEASIBILITY"
        )
        report = {
            "schema_version": "dependent-retrieval-pilot-result-1",
            "experiment_id": frozen_experiment_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": result_status,
            "scope": protocol.get("scope", "DEVELOPMENT_FEASIBILITY_ONLY"),
            "evaluator_version": EVALUATOR_VERSION,
            "protocol": {"path": str(protocol_path), "sha256": _sha256_file(protocol_path)},
            "generation": generation_cfg,
            "by_arm": by_arm,
            "overall": {"B_minus_A": overall},
            "by_dataset": by_dataset,
            "mechanism": mechanism,
            "fallback": {
                "n": len(fallback_pairs),
                "prompt_exact": fallback_prompt_exact,
                "prediction_exact": fallback_prediction_exact,
            },
            "decision_gates": decision,
            "inputs": {
                "arm_a": {"path": str(arm_a_path), "sha256": input_hashes[ARMS[0]]},
                "arm_b": {"path": str(arm_b_path), "sha256": input_hashes[ARMS[1]]},
                "adapter": artifact_identity(adapter_path),
                "base_model": artifact_identity(base_path),
            },
            "outputs": {
                "predictions": {
                    "path": str(predictions_path),
                    "sha256": _sha256_file(predictions_path),
                }
            },
            "scientific_boundary": (
                "Development-only, zero-training paired evidence. Gold was appended only after "
                "retrieval materialisation. A pass advances to a fresh family/QID-disjoint "
                "confirmation; it is not itself confirmation evidence."
            ),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            run_dir,
            extra={"phase": "dependent_retrieval_zero_training_pilot", **report},
            status=result_status,
        )
        print(
            json.dumps(
                {
                    "status": result_status,
                    "overall": overall,
                    "by_dataset": {
                        dataset: values["B_minus_A"] for dataset, values in by_dataset.items()
                    },
                    "fallback": report["fallback"],
                    "decision_gates": decision,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except BaseException as exc:
        dump_manifest(
            run_dir,
            extra={
                "experiment_id": frozen_experiment_id,
                "phase": "dependent_retrieval_zero_training_pilot",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
            status="ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_RUNTIME",
        )
        raise


if __name__ == "__main__":
    main()
