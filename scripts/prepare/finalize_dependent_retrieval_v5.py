#!/usr/bin/env python
"""Freeze scorer inputs for the adaptive v5 dependent-retrieval combination.

This finalizer is intentionally independent from the v4 finalizer and the
answer evaluator.  It first verifies the completed, Gold-free v5 retrieval
materialisation and recomputes its fixed-budget/prefix/displacement safety
invariants.  Only after every retrieval gate passes does it open raw dev rows
to attach ``gold_answers`` for scoring.

The generated protocol labels v5 as an adaptive development *combination*
experiment (typed bridge admission + guarded merge).  It therefore supports
only a claim about the combined system, never either component in isolation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.retrieval.dependent_merge_v5 import POLICY_VERSION, passage_score_key
from kgproweight.retrieval.dependent_v5 import SELECTOR_VERSION
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from scripts.prepare.finalize_dependent_retrieval_pilot import (
    DATASETS,
    _index_raw_gold,
    _read_jsonl,
    _resolve_report_artifact,
    _sha256_file,
    _sha256_text,
    _validate_and_enrich,
    _write_jsonl,
)
from scripts.prepare import freeze_dependent_retrieval_v5 as v5_freeze


FINALIZER_VERSION = "dependent-retrieval-v5-finalizer-1"
EXPECTED_REPORT_SCHEMA = "plan-once-dependent-retrieval-v5-report-1"
EXPECTED_REPORT_STATUS = "COMPLETE_INPUTS_NOT_ANSWER_EVALUATED"
EXPECTED_ROWS_PER_DATASET = 30
EXPECTED_TOTAL = 60
EXPECTED_PASSAGES = 10
PROTECTED_PREFIX = 8

# These are the v4 evaluator gates.  Do not relax them for v5.
EVALUATOR_DECISION_GATES = {
    "pooled_net_correct_gain_min": 3,
    "max_net_correct_loss_per_dataset": 1,
    "parse_count_delta_min": 0,
    "plan_executable_rate_min_each_dataset": 0.80,
    "second_hop_query_nonempty_rate_min_each_dataset": 0.80,
    "new_dependent_candidate_question_rate_min_each_dataset": 0.50,
}
ANSWER_UTILITY_GATES = {
    "pooled_net_correct_gain_min": 3,
    "pooled_delta_f1_gt": 0.0,
    "max_net_correct_loss_per_dataset": 1,
    "parse_count_delta_min": 0,
}

FORBIDDEN_UPSTREAM_KEYS = frozenset({
    "answer", "answers", "gold_answer", "gold_answers", "golden_answers",
    "supporting_facts", "supporting_titles", "question_decomposition",
    "decomposition", "evidence", "evidences", "target",
})


def _json_sha256(value: Any) -> str:
    import hashlib

    blob = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _assert_gold_free(value: Any, *, where: str) -> None:
    """Reject Gold/support/decomposition fields without inspecting text."""

    if isinstance(value, Mapping):
        bad = {str(key).casefold() for key in value} & FORBIDDEN_UPSTREAM_KEYS
        if bad:
            raise ValueError(f"upstream retrieval contains forbidden fields at {where}: {sorted(bad)}")
        for key, child in value.items():
            _assert_gold_free(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_gold_free(child, where=f"{where}[{index}]")


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('dataset') or '').strip()}::{str(row.get('qid') or '').strip()}"


def _index_details(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = _row_key(row)
        if key == "::" or key in indexed:
            raise ValueError(f"invalid or duplicate execution detail key: {key}")
        indexed[key] = row
    return indexed


def _hop_dependency_map(detail: Mapping[str, Any]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for hop in detail.get("hops") or []:
        if not isinstance(hop, Mapping):
            raise ValueError("execution detail contains a non-object hop")
        hop_id = str(hop.get("hop_id") or "").strip()
        if not hop_id or hop_id in result:
            raise ValueError(f"invalid or duplicate hop_id: {hop_id!r}")
        result[hop_id] = bool(hop.get("dependencies"))
    return result


def _audit_materialization_safety(
    arm_a: Sequence[Mapping[str, Any]],
    arm_b: Sequence[Mapping[str, Any]],
    execution_details: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute v5 safety invariants from frozen rows and merge telemetry."""

    if len(arm_a) != EXPECTED_TOTAL or len(arm_b) != EXPECTED_TOTAL:
        raise ValueError(f"v5 requires {EXPECTED_TOTAL} paired rows, A={len(arm_a)} B={len(arm_b)}")
    details = _index_details(execution_details)
    if len(details) != EXPECTED_TOTAL:
        raise ValueError(f"v5 requires {EXPECTED_TOTAL} execution details, found {len(details)}")

    counts = Counter(str(row.get("dataset") or "") for row in arm_a)
    if counts != Counter({dataset: EXPECTED_ROWS_PER_DATASET for dataset in DATASETS}):
        raise ValueError(f"v5 dataset population differs from same60: {dict(counts)}")

    aggregate = Counter()
    per_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    question_keys: list[str] = []
    qids: list[str] = []
    for index, (left_raw, right_raw) in enumerate(zip(arm_a, arm_b)):
        left, right = dict(left_raw), dict(right_raw)
        _assert_gold_free(left, where=f"arm_a[{index}]")
        _assert_gold_free(right, where=f"arm_b[{index}]")
        if left.get("arm") != "A_question_only" or right.get("arm") != "B_dependent":
            raise ValueError(f"arm labels differ from evaluator schema at row {index}")
        key = _row_key(left)
        if key != _row_key(right) or key not in details:
            raise ValueError(f"A/B/detail identity join differs at row {index}")
        if key in question_keys:
            raise ValueError(f"duplicate paired key: {key}")
        question_keys.append(key)
        qids.append(str(left.get("qid") or ""))
        dataset = str(left.get("dataset") or "")
        if dataset not in per_dataset:
            raise ValueError(f"unexpected dataset: {dataset}")

        # _validate_and_enrich later checks all scorer-common fields.  Before
        # Gold is opened, freeze the key prompt identities explicitly.
        for field in (
            "row_id", "question_key", "dataset", "qid", "question",
            "question_sha256", "split", "gold_access", "kg_subgraph",
            "legacy_kg_sha256",
        ):
            if field not in left or field not in right or left[field] != right[field]:
                raise ValueError(f"A/B common field {field} differs at row {index}")
        if left["gold_access"] is not False:
            raise ValueError(f"upstream row does not attest gold_access=false: {key}")

        a_passages = left.get("retrieved_passages")
        b_passages = right.get("retrieved_passages")
        if not isinstance(a_passages, list) or not isinstance(b_passages, list):
            raise ValueError(f"retrieved_passages must be lists: {key}")
        if len(a_passages) != EXPECTED_PASSAGES or len(b_passages) != EXPECTED_PASSAGES:
            raise ValueError(f"all-top10 gate failed for {key}")
        if str(left.get("passages_sha256") or "") != _json_sha256(a_passages):
            raise ValueError(f"Arm A passage hash mismatch: {key}")
        if str(right.get("passages_sha256") or "") != _json_sha256(b_passages):
            raise ValueError(f"Arm B passage hash mismatch: {key}")
        a_keys = [passage_score_key(row) for row in a_passages]
        b_keys = [passage_score_key(row) for row in b_passages]
        if len(set(a_keys)) != EXPECTED_PASSAGES or len(set(b_keys)) != EXPECTED_PASSAGES:
            raise ValueError(f"fixed-budget deduplication failed for {key}")
        if a_passages[:PROTECTED_PREFIX] != b_passages[:PROTECTED_PREFIX]:
            raise ValueError(f"prefix8 exact gate failed for {key}")

        detail = details[key]
        if detail.get("gold_access") is not False:
            raise ValueError(f"execution detail does not attest gold_access=false: {key}")
        if detail.get("error") is not None or detail.get("execution_status") == "fallback_execution_error":
            raise ValueError(f"runtime/execution error present for {key}")
        exact = a_passages == b_passages
        if bool(right.get("fallback_to_a")) != exact:
            raise ValueError(f"fallback_to_a/exact passage mismatch for {key}")
        merge = detail.get("merge")
        removed = set(a_keys) - set(b_keys)
        added = set(b_keys) - set(a_keys)
        # Pre-merge abstentions (invalid/no-dependent/bridge-rejected) never
        # invoke the merge policy.  They are admissible only as exact A
        # fallbacks with no document-set change.  Any changed row must carry
        # the complete guarded-merge telemetry below.
        if merge is None:
            if not exact or added or removed:
                raise ValueError(f"missing v5 merge telemetry for changed row {key}")
            safety = detail.get("safety")
            if not isinstance(safety, Mapping) or safety.get("fallback_exact") is not True:
                raise ValueError(f"pre-merge fallback lacks exact safety telemetry for {key}")
            selected, evicted = [], []
        else:
            if not isinstance(merge, Mapping):
                raise ValueError(f"invalid v5 merge telemetry for {key}")
            if int(merge.get("protected_originals", -1)) != PROTECTED_PREFIX:
                raise ValueError(f"merge did not use protected_originals={PROTECTED_PREFIX}: {key}")
            if int(merge.get("total_budget", -1)) != EXPECTED_PASSAGES:
                raise ValueError(f"merge total budget differs for {key}")
            selected = list(merge.get("selected_new") or [])
            evicted = list(merge.get("evicted_originals") or [])
        selected_keys = {str(row.get("document_key") or "") for row in selected}
        evicted_keys = {str(row.get("document_key") or "") for row in evicted}
        if added != selected_keys or removed != evicted_keys or len(added) != len(removed):
            raise ValueError(f"unauthorized original displacement inventory for {key}")
        if len(selected) != len(selected_keys) or len(evicted) != len(evicted_keys):
            raise ValueError(f"duplicate v5 selection/eviction telemetry for {key}")

        dependency_by_hop = _hop_dependency_map(detail)
        selected_by_key = {str(row["document_key"]): row for row in selected}
        for eviction in evicted:
            original_rank = int(eviction.get("original_rank", -1))
            original_key = str(eviction.get("document_key") or "")
            replacement_key = str(eviction.get("replaced_by") or "")
            if original_rank <= PROTECTED_PREFIX or original_rank > EXPECTED_PASSAGES:
                raise ValueError(f"protected/invalid original displacement for {key}")
            if a_keys[original_rank - 1] != original_key:
                raise ValueError(f"evicted original rank/key mismatch for {key}")
            if replacement_key not in selected_by_key:
                raise ValueError(f"replacement is absent from selected_new for {key}")
            old_score = float(eviction.get("score"))
            new_score = float(eviction.get("replacement_score"))
            if not new_score > old_score:
                raise ValueError(f"unauthorized non-improving displacement for {key}")
            if float(selected_by_key[replacement_key].get("score")) != new_score:
                raise ValueError(f"selected/replacement score mismatch for {key}")
        for selected_row in selected:
            hop_id = str(selected_row.get("hop_id") or "")
            if dependency_by_hop.get(hop_id) is not True:
                raise ValueError(f"root or unknown-hop passage injected for {key}: {hop_id}")

        if isinstance(merge, Mapping) and bool(merge.get("fallback_exact")) != exact:
            raise ValueError(f"merge fallback_exact mismatch for {key}")
        aggregate.update({
            "n": 1, "top10": 1, "prefix8_exact": 1,
            "changed": int(not exact), "fallback_exact": int(exact),
            "selected_new": len(added), "evicted_originals": len(removed),
        })
        per_dataset[dataset].update({
            "n": 1, "top10": 1, "prefix8_exact": 1,
            "changed": int(not exact), "fallback_exact": int(exact),
            "selected_new": len(added), "evicted_originals": len(removed),
            "plan_executable": int(bool(detail.get("plan_executable"))),
            "dependent_eligible": int(bool(detail.get("has_dependent_step"))),
            "dependent_nonempty": int(
                bool(detail.get("has_dependent_step"))
                and int(detail.get("second_hop_query_count") or 0) > 0
            ),
        })

    result_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset, values in per_dataset.items():
        item = dict(values)
        n = int(values["n"])
        eligible = int(values["dependent_eligible"])
        item.update({
            "plan_executable_rate": (
                int(values["plan_executable"]) / n if n else 0.0
            ),
            "dependent_hop_query_nonempty_rate": (
                int(values["dependent_nonempty"]) / eligible if eligible else None
            ),
            "retained_new_dependent_document_question_rate": (
                int(values["changed"]) / n if n else 0.0
            ),
        })
        result_by_dataset[dataset] = item
    return {
        "n": EXPECTED_TOTAL,
        "all_top10": aggregate["top10"] == EXPECTED_TOTAL,
        "prefix8_exact": aggregate["prefix8_exact"] == EXPECTED_TOTAL,
        "unauthorized_original_displacements": 0,
        "root_passages_injected": 0,
        "fallback_exact": True,
        "changed_questions": aggregate["changed"],
        "fallback_questions": aggregate["fallback_exact"],
        "selected_new_documents": aggregate["selected_new"],
        "evicted_original_documents": aggregate["evicted_originals"],
        "qid_order_sha256": _sha256_text("\n".join(qids)),
        "question_key_order_sha256": _sha256_text("\n".join(question_keys)),
        "by_dataset": result_by_dataset,
    }


def _enforce_report_safety(report: Mapping[str, Any], computed: Mapping[str, Any]) -> None:
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA:
        raise ValueError(f"unexpected v5 report schema: {report.get('schema_version')!r}")
    if report.get("status") != EXPECTED_REPORT_STATUS:
        raise ValueError(f"retrieval report status must be {EXPECTED_REPORT_STATUS}")
    if report.get("gold_access") is not False:
        raise ValueError("retrieval report does not attest gold_access=false")
    if report.get("development_only") is not True:
        raise ValueError("v5 report must be explicitly marked development_only=true")
    if report.get("canonical_pipeline_modified") is not False:
        raise ValueError("v5 report must attest canonical_pipeline_modified=false")
    if str(report.get("selector_version") or "") != SELECTOR_VERSION:
        raise ValueError("retrieval report selector_version differs from dependent_v5")
    if str(report.get("merge_policy_version") or "") != POLICY_VERSION:
        raise ValueError("retrieval report merge_policy_version differs from dependent_merge_v5")
    if not str(report.get("runner_version") or "").strip():
        raise ValueError("retrieval report is missing runner_version")
    settings = report.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("v5 report is missing settings")
    expected_settings = {
        "protected_originals": PROTECTED_PREFIX,
        "total_passages": EXPECTED_PASSAGES,
        "root_hop_injection": False,
    }
    for name, value in expected_settings.items():
        if settings.get(name) != value:
            raise ValueError(
                f"v5 report materialization setting differs: "
                f"{name}={settings.get(name)!r}, expected {value!r}"
            )
    by_dataset = report.get("by_dataset")
    if not isinstance(by_dataset, Mapping):
        raise ValueError("v5 report is missing by_dataset telemetry")
    for dataset in DATASETS:
        values = by_dataset.get(dataset)
        if not isinstance(values, Mapping) or int(values.get("n", -1)) != EXPECTED_ROWS_PER_DATASET:
            raise ValueError(f"v5 report population mismatch for {dataset}")
        for name in ("runtime_errors", "fallback_execution_error", "unauthorized_original_displacements", "root_passages_injected"):
            if int(values.get(name, -1)) != 0:
                raise ValueError(f"{dataset} safety gate {name}=0 failed")
        if values.get("all_top10") is not True:
            raise ValueError(f"{dataset} all_top10 gate failed")
        if values.get("prefix8_exact") is not True:
            raise ValueError(f"{dataset} prefix8_exact gate failed")
        if values.get("fallback_exact") is not True:
            raise ValueError(f"{dataset} fallback_exact gate failed")
        mechanism_gates = {
            "plan_executable_rate": 0.80,
            "dependent_hop_query_nonempty_rate": 0.80,
            "retained_new_dependent_document_question_rate": 0.50,
        }
        for name, minimum in mechanism_gates.items():
            observed = values.get(name)
            recomputed = (computed.get("by_dataset") or {}).get(dataset, {}).get(name)
            if observed != recomputed:
                raise ValueError(
                    f"{dataset} mechanism telemetry mismatch: "
                    f"{name} report={observed!r} recomputed={recomputed!r}"
                )
            if observed is None or float(observed) < minimum:
                raise ValueError(
                    f"{dataset} Gold-free mechanism gate failed: "
                    f"{name}={observed!r} < {minimum}"
                )

    summary = report.get("safety_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("v5 report is missing safety_summary")
    expected = {
        "all_top10": True,
        "prefix8_exact": True,
        "unauthorized_original_displacements": 0,
        "root_passages_injected": 0,
        "fallback_exact": True,
        "runtime_errors": 0,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"global materialization safety gate failed: {name}={summary.get(name)!r}")
    for name in ("all_top10", "prefix8_exact", "unauthorized_original_displacements", "root_passages_injected", "fallback_exact"):
        if computed[name] != expected[name]:
            raise ValueError(f"recomputed materialization safety gate failed: {name}")


def _validate_v4_gate_identity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    observed = protocol.get("decision_gates")
    if observed != EVALUATOR_DECISION_GATES:
        raise ValueError("v4 evaluator gates differ from the required unchanged v5 gates")
    return protocol, {"path": str(path), "sha256": _sha256_file(path)}


def _validate_v4_controls(
    protocol: Mapping[str, Any],
    *,
    arm_a_sha256: str,
    qid_order_sha256: str,
    question_key_order_sha256: str,
    adapter_identity: Mapping[str, Any],
    base_identity: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> None:
    """Lock population, Arm A, model and decoding to the completed v4 test."""

    if str(protocol.get("qid_order_sha256") or "") != qid_order_sha256:
        raise ValueError("v5 qid order differs from consumed v4 same60")
    if str(protocol.get("question_key_order_sha256") or "") != question_key_order_sha256:
        raise ValueError("v5 question-key order differs from consumed v4 same60")
    v4_a = (protocol.get("inputs") or {}).get("retrieval_arm_a_no_gold") or {}
    if str(v4_a.get("sha256") or "") != arm_a_sha256:
        raise ValueError("v5 Arm A bytes differ from frozen v4 Arm A")
    if (protocol.get("models") or {}).get("strong_sft") != dict(adapter_identity):
        raise ValueError("v5 strong-SFT identity differs from v4")
    if protocol.get("base_model") != dict(base_identity):
        raise ValueError("v5 base-model identity differs from v4")
    if protocol.get("generation") != dict(generation):
        raise ValueError("v5 generation controls differ from v4")


def _validate_v4_scorer_gold_locks(
    protocol: Mapping[str, Any],
    gold_paths: Sequence[Path],
) -> dict[str, dict[str, str]]:
    """Verify scorer-only files are byte-identical to the completed v4 test.

    This check happens only after all Gold-free retrieval and materialisation
    gates pass, but before raw rows are parsed or labels are joined.
    """

    if len(gold_paths) != len(DATASETS):
        raise ValueError("v5 scorer Gold requires exactly one path per frozen dataset")
    locks = (protocol.get("inputs") or {}).get("scorer_gold")
    if not isinstance(locks, Mapping):
        raise ValueError("v4 protocol is missing scorer_gold locks")
    verified: dict[str, dict[str, str]] = {}
    for dataset, raw_path in zip(DATASETS, gold_paths):
        path = raw_path.expanduser().resolve()
        lock = locks.get(dataset)
        if not isinstance(lock, Mapping):
            raise ValueError(f"v4 protocol is missing scorer Gold lock for {dataset}")
        locked_path = Path(str(lock.get("path") or "")).expanduser().resolve()
        if path != locked_path:
            raise ValueError(f"v5 scorer Gold path differs from v4 for {dataset}")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256_file(path)
        if digest != str(lock.get("sha256") or ""):
            raise ValueError(f"v5 scorer Gold SHA256 differs from v4 for {dataset}")
        verified[dataset] = {"path": str(path), "sha256": digest}
    return verified


def _validate_preregistration(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != v5_freeze.SCHEMA_VERSION:
        raise ValueError("unexpected v5 preregistration schema")
    status = str(protocol.get("status") or "")
    scope = str(protocol.get("scope") or "")
    if status != v5_freeze.STATUS:
        raise ValueError("v5 preregistration is not frozen")
    if scope != v5_freeze.SCOPE:
        raise ValueError("v5 preregistration does not declare the adaptive combination scope")
    experiment_ids = protocol.get("experiment_ids")
    if experiment_ids != v5_freeze.EXPERIMENT_IDS:
        raise ValueError("v5 preregistration Experiment IDs differ")
    if str(experiment_ids["materialization"]) != str(report.get("experiment_id") or ""):
        raise ValueError("v5 preregistration/materialization Experiment IDs differ")
    lock = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if report.get("preregistration") != lock:
        raise ValueError("retrieval report preregistration lock differs")
    runtime = report.get("runtime_locks")
    if not isinstance(runtime, Mapping):
        raise ValueError("retrieval report is missing runtime_locks")
    if runtime.get("preregistration") != lock:
        raise ValueError("runtime preregistration lock differs")
    for name in ("inputs", "code", "models", "settings"):
        if runtime.get(name) != protocol.get(name):
            raise ValueError(f"runtime {name} locks differ from preregistration")
    for section in ("inputs", "code"):
        for name, raw_lock in (protocol.get(section) or {}).items():
            if not isinstance(raw_lock, Mapping):
                raise ValueError(f"invalid preregistered {section} lock: {name}")
            current_path = Path(str(raw_lock.get("path") or "")).expanduser().resolve()
            if not current_path.is_file():
                raise FileNotFoundError(current_path)
            current_lock = {
                "path": str(current_path),
                "size_bytes": current_path.stat().st_size,
                "sha256": _sha256_file(current_path),
            }
            if current_lock != dict(raw_lock):
                raise ValueError(f"current {section} bytes drifted after preregistration: {name}")
    if report.get("settings") != protocol.get("settings"):
        raise ValueError("retrieval report settings differ from preregistration")
    actual_assets = report.get("retrieval_assets")
    locked_assets = protocol.get("retrieval_assets")
    if not isinstance(actual_assets, Mapping) or not isinstance(locked_assets, Mapping):
        raise ValueError("retrieval asset locks are missing")
    if int(actual_assets.get("expected_docs", -1)) != int(
        locked_assets.get("expected_documents", -2)
    ):
        raise ValueError("runtime Wiki18 document count differs from preregistration")
    if actual_assets.get("counts") != locked_assets.get("counts"):
        raise ValueError("runtime Wiki18 asset counts differ from preregistration")
    runtime_paths = actual_assets.get("paths") or {}
    for lock_name, runtime_name in (
        ("corpus", "corpus"),
        ("dense_index", "dense"),
        ("bm25_index", "bm25"),
    ):
        locked_item = locked_assets.get(lock_name) or {}
        if Path(str(runtime_paths.get(runtime_name) or "")).resolve() != Path(
            str(locked_item.get("path") or "")
        ).resolve():
            raise ValueError(f"runtime Wiki18 {lock_name} path differs")
    return lock


def _code_lock(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _validate_experiment_ids(materialization: object, freeze: object, evaluation: object) -> None:
    values = [str(value or "").strip() for value in (materialization, freeze, evaluation)]
    if any(not value for value in values) or len(set(values)) != 3:
        raise ValueError(
            "materialization, freeze and evaluation require three non-empty distinct Experiment IDs"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_report", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--v4_frozen_protocol", type=Path,
        default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_freeze_v4/protocol.json"),
    )
    parser.add_argument("--hotpot_dev", type=Path, default=Path("data/hotpotqa/dev.jsonl"))
    parser.add_argument("--musique_dev", type=Path, default=Path("data/musique/dev.jsonl"))
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    parser.add_argument(
        "--runner", type=Path,
        default=Path("scripts/pilot/audit_plan_once_dependent_retrieval_v5.py"),
    )
    parser.add_argument(
        "--evaluator", type=Path,
        default=Path("scripts/eval/evaluate_dependent_retrieval_pilot.py"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True, help="Unique CPU freeze Experiment ID")
    parser.add_argument(
        "--evaluation_experiment_id", required=True,
        help="Distinct Experiment ID reserved for the later GPU evaluator",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k_passages", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.retrieval_report.expanduser().resolve()
    prereg_path = args.preregistration.expanduser().resolve()
    v4_protocol_path = args.v4_frozen_protocol.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Everything through this point is Gold-free.  Raw dev rows are not opened
    # until the report, artifacts and recomputed materialization gates pass.
    _validate_experiment_ids(
        report.get("experiment_id"), args.experiment_id, args.evaluation_experiment_id
    )
    prereg_lock = _validate_preregistration(prereg_path, report)
    prereg_document = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg_ids = prereg_document["experiment_ids"]
    if str(args.experiment_id) != str(prereg_ids["post_materialization_freeze"]):
        raise ValueError("freeze Experiment ID differs from preregistration")
    if str(args.evaluation_experiment_id) != str(prereg_ids["answer_evaluation"]):
        raise ValueError("evaluation Experiment ID differs from preregistration")
    v4_protocol, v4_gate_lock = _validate_v4_gate_identity(v4_protocol_path)
    arm_a_path = _resolve_report_artifact(report, "arm_a")
    arm_b_path = _resolve_report_artifact(report, "arm_b")
    execution_path = _resolve_report_artifact(report, "execution_details")
    arm_a, arm_b = _read_jsonl(arm_a_path), _read_jsonl(arm_b_path)
    execution_details = _read_jsonl(execution_path)
    computed_safety = _audit_materialization_safety(arm_a, arm_b, execution_details)
    _enforce_report_safety(report, computed_safety)
    adapter_identity = artifact_identity(args.adapter.expanduser().resolve())
    base_identity = artifact_identity(args.base_model.expanduser().resolve())
    generation = {
        "seed": int(args.seed), "decode": "greedy", "do_sample": False,
        "temperature": None, "top_p": None,
        "max_new_tokens": int(args.max_new_tokens),
        "top_k_passages": int(args.top_k_passages),
    }
    _validate_v4_controls(
        v4_protocol,
        arm_a_sha256=_sha256_file(arm_a_path),
        qid_order_sha256=str(computed_safety["qid_order_sha256"]),
        question_key_order_sha256=str(computed_safety["question_key_order_sha256"]),
        adapter_identity=adapter_identity,
        base_identity=base_identity,
        generation=generation,
    )

    # Only byte identities are checked here.  Gold rows/labels are parsed and
    # joined only after every Gold-free check above succeeds and the files are
    # proven identical to the completed v4 evaluation inputs.
    gold_paths = [args.hotpot_dev.expanduser().resolve(), args.musique_dev.expanduser().resolve()]
    scorer_gold_locks = _validate_v4_scorer_gold_locks(v4_protocol, gold_paths)
    gold_index = _index_raw_gold(gold_paths)
    final_a, final_b = _validate_and_enrich(arm_a, arm_b, gold_index)
    if len(final_a) != EXPECTED_TOTAL:
        raise ValueError(f"v5 frozen pilot requires {EXPECTED_TOTAL} paired questions")

    out, freeze_experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "dependent_retrieval_v5_post_materialisation_freeze",
            "scope": "ADAPTIVE_DEVELOPMENT_COMBINATION_SAME60_CONSUMED",
            "gold_access_during_retrieval": False,
        },
    )
    try:
        final_a_path = out / "arm_a.scored.jsonl"
        final_b_path = out / "arm_b.scored.jsonl"
        _write_jsonl(final_a_path, final_a)
        _write_jsonl(final_b_path, final_b)

        project_root = Path(__file__).resolve().parents[2]
        code = {
            "retrieval_runner_v5": _code_lock(args.runner),
            "dependent_v5": _code_lock(project_root / "kgproweight/retrieval/dependent_v5.py"),
            "dependent_merge_v5": _code_lock(project_root / "kgproweight/retrieval/dependent_merge_v5.py"),
            "evaluator": _code_lock(args.evaluator),
            "finalizer_v5": _code_lock(Path(__file__)),
            "gold_join_helpers": _code_lock(
                project_root / "scripts/prepare/finalize_dependent_retrieval_pilot.py"
            ),
        }
        qids = [str(row["qid"]) for row in final_a]
        question_keys = [str(row["question_key"]) for row in final_a]
        protocol = {
            "schema_version": "dependent-retrieval-v5-eval-protocol-1",
            "experiment_id": str(args.evaluation_experiment_id),
            "freeze_experiment_id": freeze_experiment_id,
            "materialization_experiment_id": str(report["experiment_id"]),
            "status": "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION",
            "scope": (
                "ADAPTIVE_DEVELOPMENT_COMBINATION; SAME60_CONSUMED; "
                "HotpotQA30 + MuSiQue30; confirmation remains unopened"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gold_access_during_retrieval": False,
            "experimental_design": {
                "primary_comparison": "A_question_only vs B_dependent",
                "combined_variable": [
                    "typed bridge admission v5",
                    "precision-first prefix8 guarded merge v5",
                ],
                "component_attribution_allowed": False,
                "population_reselection_after_v4": False,
                "same_consumed_question_keys_as_v4": True,
                "selector_version": SELECTOR_VERSION,
                "merge_policy_version": POLICY_VERSION,
                "runner_version": str(report["runner_version"]),
            },
            "n": EXPECTED_TOTAL,
            "qid_order_sha256": _sha256_text("\n".join(qids)),
            "question_key_order_sha256": _sha256_text("\n".join(question_keys)),
            "inputs": {
                "arm_a": {"path": str(final_a_path.resolve()), "sha256": _sha256_file(final_a_path)},
                "arm_b": {"path": str(final_b_path.resolve()), "sha256": _sha256_file(final_b_path)},
                "retrieval_report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
                "retrieval_arm_a_no_gold": {"path": str(arm_a_path), "sha256": _sha256_file(arm_a_path)},
                "retrieval_arm_b_no_gold": {"path": str(arm_b_path), "sha256": _sha256_file(arm_b_path)},
                "execution_details_no_gold": {"path": str(execution_path), "sha256": _sha256_file(execution_path)},
                "preregistration": prereg_lock,
                "v4_frozen_protocol_gate_identity": v4_gate_lock,
                "scorer_gold": scorer_gold_locks,
            },
            "models": {"strong_sft": adapter_identity},
            "base_model": base_identity,
            "generation": generation,
            "decision_gates": dict(EVALUATOR_DECISION_GATES),
            "answer_utility_gates_unchanged_from_v4": dict(ANSWER_UTILITY_GATES),
            "materialization_safety_gates": {
                "all_top10": True,
                "protected_prefix": PROTECTED_PREFIX,
                "prefix8_exact": True,
                "unauthorized_original_displacements": 0,
                "root_passages_injected": 0,
                "runtime_errors": 0,
                "fallback_execution_error": 0,
                "fallback_exact": True,
            },
            "materialization_safety_observed": computed_safety,
            "code": code,
            "scientific_boundary": (
                "Adaptive development evidence on the same 60 consumed questions only. "
                "The combined bridge-plus-merge system may be assessed, but neither component "
                "may be credited separately. Gold was joined only after retrieval and safety "
                "freeze. A pass can only authorize a fresh family/QID-disjoint confirmation."
            ),
        }
        protocol_path = out / "protocol.json"
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            out,
            status="FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION",
            extra={
                "experiment_id": freeze_experiment_id,
                "phase": "dependent_retrieval_v5_post_materialisation_freeze",
                "protocol_sha256": _sha256_file(protocol_path),
                "gold_access_during_retrieval": False,
                "scope": "ADAPTIVE_DEVELOPMENT_COMBINATION_SAME60_CONSUMED",
            },
        )
        print(json.dumps({
            "status": "FROZEN_READY_FOR_LOCAL_ANSWER_EVALUATION",
            "n": EXPECTED_TOTAL,
            "protocol": str(protocol_path),
            "protocol_sha256": _sha256_file(protocol_path),
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(
            out,
            status="FAILED_RUNTIME",
            extra={
                "experiment_id": freeze_experiment_id,
                "phase": "dependent_retrieval_v5_post_materialisation_freeze",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
