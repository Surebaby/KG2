#!/usr/bin/env python
"""Attach scorer-only Gold after retrieval and freeze the paired eval protocol.

The retrieval process and this finalizer are intentionally separate.  This
program refuses to run unless the upstream report is complete and explicitly
attests ``gold_access=false``.  It then joins raw dev answers by dataset::qid,
verifies question hashes and A/B pairing, and writes new append-only evaluation
inputs plus a protocol frozen before answer generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


FINALIZER_VERSION = "dependent-retrieval-pilot-finalizer-1"
EXPECTED_RETRIEVAL_STATUS = "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED"
DATASETS = ("hotpotqa", "musique")
PAIR_COMMON_FIELDS = (
    "row_id", "question_key", "dataset", "qid", "question", "question_sha256",
    "split", "gold_access", "kg_subgraph", "legacy_kg_sha256",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolve_report_artifact(report: Mapping[str, Any], name: str) -> Path:
    lock = report["outputs"][name]
    path = Path(str(lock["path"])).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != str(lock["sha256"]):
        raise ValueError(f"upstream {name} artifact differs from retrieval report")
    return path


def _enforce_retrieval_completion_gates(report: Mapping[str, Any]) -> None:
    """Do not freeze scorer inputs from a partially failed retrieval run."""

    by_dataset = report.get("by_dataset")
    if not isinstance(by_dataset, Mapping):
        raise ValueError("retrieval report is missing by_dataset telemetry")
    for dataset in DATASETS:
        values = by_dataset.get(dataset)
        if not isinstance(values, Mapping):
            raise ValueError(f"retrieval report is missing telemetry for {dataset}")
        if int(values.get("n", -1)) != 30:
            raise ValueError(f"{dataset} retrieval population is not the frozen n=30")
        if int(values.get("runtime_errors", -1)) != 0:
            raise ValueError(f"{dataset} retrieval has runtime_errors; freeze gate requires zero")
        if int(values.get("fallback_execution_error", -1)) != 0:
            raise ValueError(
                f"{dataset} retrieval has fallback_execution_error; freeze gate requires zero"
            )
        if values.get("fallback_exact") is not True:
            raise ValueError(f"{dataset} retrieval does not attest exact fallback")


def _index_raw_gold(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        dataset = path.parent.name.casefold()
        if dataset not in DATASETS:
            raise ValueError(f"cannot infer an allowed dataset from {path}")
        for row in _read_jsonl(path):
            qid = str(row.get("id") or row.get("qid") or "").strip()
            key = question_key(dataset, qid)
            if key in indexed:
                raise ValueError(f"duplicate scorer Gold key: {key}")
            golds = [str(value).strip() for value in (row.get("golden_answers") or []) if str(value).strip()]
            if not golds:
                raise ValueError(f"raw scorer row has no golden_answers: {key}")
            indexed[key] = {
                "question": str(row.get("question") or "").strip(),
                "question_sha256": question_sha256(str(row.get("question") or "")),
                "gold_answers": golds,
            }
    return indexed


def _common(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in PAIR_COMMON_FIELDS if field not in row]
    if missing:
        raise ValueError(f"retrieval arm row missing common fields: {missing}")
    return {field: row[field] for field in PAIR_COMMON_FIELDS}


def _validate_and_enrich(
    arm_a: Sequence[Mapping[str, Any]],
    arm_b: Sequence[Mapping[str, Any]],
    gold_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not arm_a or len(arm_a) != len(arm_b):
        raise ValueError(f"strict paired population broken: A={len(arm_a)}, B={len(arm_b)}")
    enriched_a: list[dict[str, Any]] = []
    enriched_b: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (left_raw, right_raw) in enumerate(zip(arm_a, arm_b)):
        left, right = dict(left_raw), dict(right_raw)
        if _common(left) != _common(right):
            raise ValueError(f"A/B common fields differ at row {index}")
        key = str(left["question_key"])
        expected_key = question_key(str(left["dataset"]), str(left["qid"]))
        if key != expected_key or key in seen:
            raise ValueError(f"invalid or duplicate question key: {key}")
        seen.add(key)
        if key not in gold_index:
            raise ValueError(f"scorer Gold identity join is missing {key}")
        scorer = gold_index[key]
        if str(left["question_sha256"]) != str(scorer["question_sha256"]):
            raise ValueError(f"scorer Gold question hash mismatch for {key}")
        if str(left["question"]).strip() != str(scorer["question"]):
            raise ValueError(f"scorer Gold question mismatch for {key}")
        if "gold_answers" in left or "gold_answers" in right:
            raise ValueError(f"upstream retrieval arm already contains Gold for {key}")
        left["gold_answers"] = list(scorer["gold_answers"])
        right["gold_answers"] = list(scorer["gold_answers"])
        enriched_a.append(left)
        enriched_b.append(right)
    return enriched_a, enriched_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_report", type=Path, required=True)
    parser.add_argument("--hotpot_dev", type=Path, default=Path("data/hotpotqa/dev.jsonl"))
    parser.add_argument("--musique_dev", type=Path, default=Path("data/musique/dev.jsonl"))
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--evaluator", type=Path,
        default=Path("scripts/eval/evaluate_dependent_retrieval_pilot.py"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--experiment_id", required=True,
        help="Unique Experiment ID for this CPU-only freeze/finalization run.",
    )
    parser.add_argument(
        "--evaluation_experiment_id", required=True,
        help="Distinct unique Experiment ID reserved for the later GPU answer evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k_passages", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment_id == args.evaluation_experiment_id:
        raise ValueError(
            "finalizer and answer evaluation require distinct Experiment IDs"
        )
    report_path = args.retrieval_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != EXPECTED_RETRIEVAL_STATUS:
        raise ValueError(
            f"retrieval report status must be {EXPECTED_RETRIEVAL_STATUS}, got {report.get('status')!r}"
        )
    if report.get("gold_access") is not False:
        raise ValueError("retrieval report does not attest gold_access=false")
    _enforce_retrieval_completion_gates(report)
    arm_a_path = _resolve_report_artifact(report, "arm_a")
    arm_b_path = _resolve_report_artifact(report, "arm_b")
    arm_a, arm_b = _read_jsonl(arm_a_path), _read_jsonl(arm_b_path)
    gold_paths = [args.hotpot_dev.expanduser().resolve(), args.musique_dev.expanduser().resolve()]
    gold_index = _index_raw_gold(gold_paths)
    final_a, final_b = _validate_and_enrich(arm_a, arm_b, gold_index)
    if len(final_a) != 60:
        raise ValueError(f"frozen pilot requires 60 paired questions, found {len(final_a)}")

    out, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id,
        extra={
            "phase": "dependent_retrieval_post_materialisation_freeze",
            "gold_access_during_retrieval": False,
        },
    )
    try:
        final_a_path, final_b_path = out / "arm_a.scored.jsonl", out / "arm_b.scored.jsonl"
        _write_jsonl(final_a_path, final_a)
        _write_jsonl(final_b_path, final_b)
        qids = [str(row["qid"]) for row in final_a]
        question_keys = [str(row["question_key"]) for row in final_a]
        adapter_path = args.adapter.expanduser().resolve()
        base_path = args.base_model.expanduser().resolve()
        evaluator_path = args.evaluator.expanduser().resolve()
        runner_path = Path("scripts/pilot/audit_plan_once_dependent_retrieval.py").resolve()
        dependent_path = Path("kgproweight/retrieval/dependent.py").resolve()
        finalizer_path = Path(__file__).resolve()
        for path in (evaluator_path, runner_path, dependent_path, finalizer_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        protocol = {
            "schema_version": "dependent-retrieval-pilot-protocol-1",
            # This ID is reserved for the later evaluator.  The current CPU
            # finalizer has its own ID in the run manifest below.
            "experiment_id": str(args.evaluation_experiment_id),
            "freeze_experiment_id": experiment_id,
            "status": "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION",
            "scope": "development-only HotpotQA30 + MuSiQue30; paired passage retrieval",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access_during_retrieval": False,
            "single_variable": "canonical full-question passages -> plan-once dependency-aware passages",
            "n": len(final_a),
            "qid_order_sha256": _sha256_text("\n".join(qids)),
            "question_key_order_sha256": _sha256_text("\n".join(question_keys)),
            "inputs": {
                "arm_a": {"path": str(final_a_path.resolve()), "sha256": _sha256_file(final_a_path)},
                "arm_b": {"path": str(final_b_path.resolve()), "sha256": _sha256_file(final_b_path)},
                "retrieval_report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
                "retrieval_arm_a_no_gold": {"path": str(arm_a_path), "sha256": _sha256_file(arm_a_path)},
                "retrieval_arm_b_no_gold": {"path": str(arm_b_path), "sha256": _sha256_file(arm_b_path)},
                "scorer_gold": {
                    dataset: {"path": str(path), "sha256": _sha256_file(path)}
                    for dataset, path in zip(DATASETS, gold_paths)
                },
            },
            "models": {"strong_sft": artifact_identity(adapter_path)},
            "base_model": artifact_identity(base_path),
            "generation": {
                "seed": int(args.seed), "decode": "greedy", "do_sample": False,
                "temperature": None, "top_p": None,
                "max_new_tokens": int(args.max_new_tokens),
                "top_k_passages": int(args.top_k_passages),
            },
            "decision_gates": {
                "pooled_net_correct_gain_min": 3,
                "max_net_correct_loss_per_dataset": 1,
                "parse_count_delta_min": 0,
                "plan_executable_rate_min_each_dataset": 0.80,
                "second_hop_query_nonempty_rate_min_each_dataset": 0.80,
                "new_dependent_candidate_question_rate_min_each_dataset": 0.50,
            },
            "code": {
                "retrieval_runner": {"path": str(runner_path), "sha256": _sha256_file(runner_path)},
                "dependent_helpers": {"path": str(dependent_path), "sha256": _sha256_file(dependent_path)},
                "finalizer": {"path": str(finalizer_path), "sha256": _sha256_file(finalizer_path)},
                "evaluator": {"path": str(evaluator_path), "sha256": _sha256_file(evaluator_path)},
            },
            "scientific_boundary": (
                "Pilot development evidence only. Gold was joined after retrieval solely for scoring; "
                "the unopened confirmation population is not part of this protocol."
            ),
        }
        protocol_path = out / "protocol.json"
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            out, status="FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION",
            extra={
                "experiment_id": experiment_id,
                "phase": "dependent_retrieval_post_materialisation_freeze",
                "protocol_sha256": _sha256_file(protocol_path),
                "gold_access_during_retrieval": False,
            },
        )
        print(json.dumps({
            "status": "FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION",
            "n": len(final_a), "protocol": str(protocol_path),
            "protocol_sha256": _sha256_file(protocol_path),
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            out, status="FAILED_RUNTIME",
            extra={
                "experiment_id": experiment_id,
                "phase": "dependent_retrieval_post_materialisation_freeze",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
