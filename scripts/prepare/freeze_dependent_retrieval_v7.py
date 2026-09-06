#!/usr/bin/env python
"""Freeze the answer-free v7 development cohort and preregistration assets.

This command is deliberately CPU-only.  It selects HotpotQA20 + MuSiQue20
from an already frozen question-only SAEG confirmation pool, reclassifies only
those rows as development/consumed in a new append-only ledger, and commits to
the unselected remainder without materialising it.  It never imports a model,
opens a raw dataset, runs retrieval, or reads scorer Gold.

The resulting protocol is *not* an execution authorization.  Runtime runner,
finalizer, and evaluator hashes must be added in a separate append-only lock
before any v7 planner, model, retrieval, or scoring call.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import artifact_identity


SCHEMA_VERSION = "subquestion-dependent-retrieval-v7-preregistration-1"
STATUS = "FROZEN_COHORT_AND_RULES_BLOCKED_UNTIL_IMPLEMENTATION_HASH_LOCK"
SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
EXPERIMENT_ID = "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-PREREGISTRATION"
FUTURE_EXPERIMENT_IDS = {
    "implementation_lock": "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-IMPLEMENTATION-LOCK",
    "planner": "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-PLANS",
    "materialization": "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-MATERIALIZE",
    "gold_attachment": "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-GOLD-ATTACH",
    "evaluation": "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-EVAL",
}

DATASETS = ("hotpotqa", "musique")
SOURCE_COUNTS = {
    "2wikimultihopqa": 100,
    "hotpotqa": 100,
    "musique": 100,
}
SELECTED_PER_DATASET = 20
SELECTION_SALT = "subquestion-dependent-retrieval-v7-development-seed-20260904"
TARGET_TYPES = {"hotpotqa": "relation_graph", "musique": "subquery_graph"}

EXPECTED_SOURCE_SHA256 = "609bb0c8b78ea10a9a4b69283c0310a57e7a4f2534f1338502e31c2c5550299f"
EXPECTED_PARENT_PROTOCOL_SHA256 = "5d30398d01667c43a7dc854eadec7c28efa5774e24e142ab3ed6ba36132e69fd"
EXPECTED_DESIGN_SHA256 = "a11babfa816ea8373d1e2725d115dd3ce85028af8986a2b00fed1ca1efd54d7e"
EXPECTED_DESIGN_MANIFEST_SHA256 = "39f880a818df8353857ed357c7d856b12940af4f905869a03fca7a5110f3b657"
EXPECTED_EXPOSURE_AUDIT_SHA256 = "6dfd02c69d95baf895c2004430f58f7d3f5c60f4c212bf3a1cd0343c0e2aef00"
EXPECTED_V6_PROTOCOL_SHA256 = "182a812d06c5014a7a0e7d7546f8dd9eaef845e5d4020830fe612dafb955a69b"
EXPECTED_PLANNER_CONFIG_SHA256 = "05aa129699c00f6de8a8a23dcf4934d611f03ea88070077a4507371e0c1968c3"
EXPECTED_WIKI18_DOCUMENTS = 21_015_324

DEFAULT_PATHS = {
    "source": Path("outputs/audits/saeg_v1_evaluation_protocol_v1/confirmation.question_only.jsonl"),
    "parent_protocol": Path("outputs/audits/saeg_v1_evaluation_protocol_v1/protocol.json"),
    "contexts": Path("outputs/audits/qpeg_v4_schema_adaptation_eval_retrieval_v1/retrieval_contexts.jsonl"),
    "design": Path("outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/protocol.json"),
    "design_manifest": Path("outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/manifest.json"),
    "exposure_audit": Path("outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/historical_identity_audit.json"),
    "v6_protocol": Path("outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v6_preregistration/protocol.json"),
    "planner_adapter": Path("checkpoints/query_planner_learned_scale_v1_1_seed42/final"),
    "planner_config": Path("configs/training/query_planner_learned_scale_v1_1_seed42.yaml"),
    "planner_generator": Path("scripts/eval/generate_query_plans_unseen.py"),
    "subanswer_module": Path("kgproweight/retrieval/subanswer_v7.py"),
    "dependent_helpers": Path("kgproweight/retrieval/dependent.py"),
    "query_renderer": Path("kgproweight/retrieval/dependent_v6.py"),
    "merge_helper": Path("kgproweight/retrieval/dependent_merge_v6.py"),
    "prompt_factory": Path("kgproweight/data/prompts.py"),
    "answer_parser": Path("kgproweight/data/parsers.py"),
}
DEFAULT_OUT = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration"
)

FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "decomposition",
        "evidence",
        "evidences",
        "gold_answer",
        "gold_answers",
        "gold_target",
        "golden_answers",
        "paragraph_text",
        "question_decomposition",
        "reasoning",
        "sp",
        "supporting_facts",
        "target",
    }
)

MATERIALIZATION_GATES = {
    "runtime_errors": 0,
    "identity_join_rate": 1.0,
    "recursive_forbidden_input_fields": 0,
    "gold_access": False,
    "all_rows_and_arms_top10": True,
    "duplicate_output_documents": 0,
    "unauthorized_A_prefix_displacements": 0,
    "root_only_documents_injected": 0,
    "all_dependent_queries_start_with_exact_original_question": True,
    "all_final_CE_pairs_use_exact_original_question": True,
    "B_C_query_budget_equal_every_question_depth_and_hop": True,
    "budget_padding_queries": 0,
    "unverified_subanswers_used": 0,
    "fallback_pair_and_A_byte_exact": True,
}
MECHANISM_GATES = {
    "plan_executable_rate_min_each_dataset": 0.8,
    "strict_subanswer_json_parse_rate_min_each_dataset": 0.5,
    "mechanically_verified_subanswer_rate_min_each_dataset": 0.4,
    "paired_dependent_hop_activation_rate_min_each_dataset": 0.4,
    "retained_new_dependent_document_question_rate_min_each_dataset_each_of_B_and_C": 0.25,
}
UTILITY_GATES = {
    "primary_comparison": "C_verified_subanswer minus B_entity_hint_top1",
    "C_minus_B_pooled_net_correct_min": 2,
    "C_minus_B_pooled_delta_f1_gt": 0.0,
    "C_minus_B_max_net_correct_loss_per_dataset": 1,
    "C_minus_B_parse_count_delta_min": 0,
    "secondary_standard_baseline_comparison": "C_verified_subanswer minus A_canonical_one_shot",
    "C_minus_A_pooled_net_correct_min": 2,
    "C_minus_A_pooled_delta_f1_gt": 0.0,
    "C_minus_A_max_net_correct_loss_per_dataset": 1,
    "C_minus_A_parse_count_delta_min": 0,
}

PLANNED_RUNTIME_PATHS = {
    "retrieval_runner": "scripts/pilot/materialize_paired_dependent_retrieval_v7.py",
    "subanswer_generator": "scripts/pilot/generate_grounded_subanswers_v7.py",
    "gold_finalizer": "scripts/prepare/finalize_paired_dependent_retrieval_v7.py",
    "evaluator": "scripts/eval/evaluate_paired_dependent_retrieval_v7.py",
    "dependent_v7_helper": "kgproweight/retrieval/dependent_v7.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def tree_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required directory is missing: {resolved}")
    inventory = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for child in sorted(value for value in resolved.rglob("*") if value.is_file()):
        relative = child.relative_to(resolved).as_posix()
        digest = sha256_file(child)
        size = child.stat().st_size
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        inventory.update(relative.encode("utf-8") + b"\0")
        inventory.update(str(size).encode("ascii") + b"\0")
        inventory.update(digest.encode("ascii") + b"\n")
    if not files:
        raise ValueError(f"required directory contains no files: {resolved}")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "size_bytes": sum(int(row["size_bytes"]) for row in files),
        "tree_sha256": inventory.hexdigest(),
        "files": files,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def assert_answer_free(value: Any, *, location: str = "row") -> None:
    if isinstance(value, Mapping):
        bad = FORBIDDEN_KEYS.intersection(str(key) for key in value)
        if bad:
            raise ValueError(f"forbidden Gold/answer fields at {location}: {sorted(bad)}")
        for key, child in value.items():
            assert_answer_free(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_answer_free(child, location=f"{location}[{index}]")


def _expect_hash(lock: Mapping[str, Any], expected: str, label: str) -> None:
    if str(lock.get("sha256") or "") != expected:
        raise ValueError(f"{label} SHA256 drift")


def _selection_digest(row: Mapping[str, Any], *, salt: str = SELECTION_SALT) -> str:
    fields = (
        salt,
        str(row["dataset"]),
        str(row["qid"]),
        str(row["question_sha256"]),
        str(row["family_sha256"]),
    )
    return sha256_text("\0".join(fields))


def validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_counts: Mapping[str, int] = SOURCE_COUNTS,
) -> None:
    if Counter(str(row.get("dataset")) for row in rows) != Counter(expected_counts):
        raise ValueError("source dataset counts differ from frozen pool")
    question_keys: list[str] = []
    question_hashes: list[str] = []
    families: list[str] = []
    for index, row in enumerate(rows):
        assert_answer_free(row, location=f"source[{index}]")
        if row.get("role") != "confirmation" or row.get("gold_access") is not False:
            raise ValueError(f"source[{index}] is not answer-free confirmation")
        required = (
            "question_key",
            "dataset",
            "qid",
            "question",
            "question_sha256",
            "family_sha256",
            "passages_sha256",
        )
        missing = [name for name in required if not str(row.get(name) or "")]
        if missing:
            raise ValueError(f"source[{index}] missing identity fields: {missing}")
        dataset, qid = str(row["dataset"]), str(row["qid"])
        if str(row["question_key"]) != f"{dataset}::{qid}":
            raise ValueError(f"source[{index}] question_key mismatch")
        if question_sha256(str(row["question"])) != str(row["question_sha256"]):
            raise ValueError(f"source[{index}] question SHA256 mismatch")
        question_keys.append(str(row["question_key"]))
        question_hashes.append(str(row["question_sha256"]))
        families.append(str(row["family_sha256"]))
    for label, values in (
        ("question_key", question_keys),
        ("question_sha256", question_hashes),
        ("family_sha256", families),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate source {label}")


def select_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_per_dataset: int = SELECTED_PER_DATASET,
    salt: str = SELECTION_SALT,
) -> list[dict[str, Any]]:
    if n_per_dataset <= 0:
        raise ValueError("n_per_dataset must be positive")
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidates = [dict(row) for row in rows if str(row["dataset"]) == dataset]
        if len(candidates) < n_per_dataset:
            raise ValueError(f"{dataset} has {len(candidates)} rows, need {n_per_dataset}")
        candidates.sort(
            key=lambda row: (
                _selection_digest(row, salt=salt),
                str(row["question_key"]),
            )
        )
        for rank, source in enumerate(candidates[:n_per_dataset], start=1):
            selected.append(
                {
                    "schema_version": "dependent-retrieval-v7-development-cohort-1",
                    "row_id": f"dependent-retrieval-v7::{source['question_key']}",
                    "question_key": str(source["question_key"]),
                    "dataset": dataset,
                    "qid": str(source["qid"]),
                    "question": str(source["question"]),
                    "question_sha256": str(source["question_sha256"]),
                    "family_sha256": str(source["family_sha256"]),
                    "source_passages_sha256": str(source["passages_sha256"]),
                    "source_role": "confirmation",
                    "role": "development_consumed",
                    "target_type": TARGET_TYPES[dataset],
                    "selection_rank_within_dataset": rank,
                    "selection_digest": _selection_digest(source, salt=salt),
                    "globally_fresh": False,
                    "independent_confirmation": False,
                    "gold_access": False,
                }
            )
    return selected


def planner_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "query-planner-supervision-1",
            "row_id": str(row["row_id"]),
            "question_key": str(row["question_key"]),
            "dataset": str(row["dataset"]),
            "qid": str(row["qid"]),
            "question": str(row["question"]),
            "question_sha256": str(row["question_sha256"]),
            "target_type": str(row["target_type"]),
            "role": "development_consumed",
            "gold_access": False,
        }
        for row in selected
    ]


def reclassification_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "dependent-retrieval-v7-reclassification-ledger-1",
            "question_key": str(row["question_key"]),
            "dataset": str(row["dataset"]),
            "qid": str(row["qid"]),
            "question_sha256": str(row["question_sha256"]),
            "family_sha256": str(row["family_sha256"]),
            "previous_role": "confirmation",
            "new_role": "development_consumed",
            "experiment_id": EXPERIMENT_ID,
            "globally_fresh": False,
            "independent_confirmation": False,
            "gold_access": False,
        }
        for row in selected
    ]


def _identity_payload(rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "\0".join(
            (
                str(row["dataset"]),
                str(row["question_key"]),
                str(row["question_sha256"]),
                str(row["family_sha256"]),
            )
        )
        for row in rows
    ]
    return "\n".join(sorted(lines))


def build_remainder_commitment(
    source_rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_keys = {str(row["question_key"]) for row in selected}
    remaining = [row for row in source_rows if str(row["question_key"]) not in selected_keys]
    counts = Counter(str(row["dataset"]) for row in remaining)
    expected = {"2wikimultihopqa": 100, "hotpotqa": 80, "musique": 80}
    if dict(counts) != expected:
        raise ValueError(f"unexpected unselected counts: {dict(counts)}")
    return {
        "schema_version": "dependent-retrieval-v7-unselected-commitment-1",
        "status": "UNMATERIALIZED_AND_NOT_AUTHORIZED_FOR_V7_EXECUTION",
        "counts": expected,
        "identity_commitment_sha256": sha256_text(_identity_payload(remaining)),
        "selected_identity_commitment_sha256": sha256_text(_identity_payload(selected)),
        "unselected_rows_materialized_here": False,
        "questions_or_passages_copied_here": False,
        "parent_pool_mutated": False,
        "future_consumer_rule": (
            "Subtract reclassification_ledger.question_only.jsonl from the immutable "
            "parent pool. No remaining row is automatically authorized as v7 confirmation."
        ),
    }


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _output_lock(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_design(design: Mapping[str, Any]) -> None:
    if design.get("status") != "RULES_AND_SELECTION_ALGORITHM_FROZEN_BEFORE_V7_GPU_OR_RETRIEVAL":
        raise ValueError("v7 design status drift")
    if design.get("scope") != "DEVELOPMENT_ONLY_FEASIBILITY_COMPARISON_ON_GLOBALLY_CONSUMED_ROWS":
        raise ValueError("v7 design scope drift")
    population = design.get("population") or {}
    if population.get("datasets_in_order") != list(DATASETS):
        raise ValueError("v7 dataset order drift")
    if int(population.get("selected_per_dataset", -1)) != SELECTED_PER_DATASET:
        raise ValueError("v7 selected count drift")
    reclassification = population.get("reclassification") or {}
    if (
        reclassification.get("v7_role") != "development_consumed"
        or reclassification.get("globally_fresh") is not False
        or reclassification.get("independent_confirmation") is not False
    ):
        raise ValueError("v7 reclassification boundary drift")
    paired = design.get("paired_execution") or {}
    if "either 0 or 1" not in str(paired.get("budget_invariant") or ""):
        raise ValueError("v7 B/C hop budget drift")
    generation = design.get("generation") or {}
    if int((generation.get("subanswer") or {}).get("max_new_tokens", -1)) != 96:
        raise ValueError("v7 subanswer generation budget drift")
    gates = design.get("decision_gates") or {}
    if gates.get("materialization") != MATERIALIZATION_GATES:
        raise ValueError("v7 materialization gates drift")
    if gates.get("gold_free_mechanism") != MECHANISM_GATES:
        raise ValueError("v7 mechanism gates drift")
    if gates.get("development_utility") != UTILITY_GATES:
        raise ValueError("v7 utility gates drift")


def _validate_parent(
    parent: Mapping[str, Any], source_lock: Mapping[str, Any], contexts_lock: Mapping[str, Any]
) -> None:
    if parent.get("status") != "FROZEN_ANSWER_FREE_BEFORE_SAEG_DEVELOPMENT":
        raise ValueError("parent SAEG protocol status drift")
    integrity = parent.get("integrity") or {}
    if integrity.get("confirmation_opened") is not False or integrity.get(
        "gold_in_protocol_or_cohorts"
    ) is not False:
        raise ValueError("parent SAEG confirmation is not answer-free/unopened")
    confirmation = (parent.get("outputs") or {}).get("confirmation") or {}
    if str(confirmation.get("sha256") or "") != str(source_lock["sha256"]):
        raise ValueError("source differs from parent confirmation lock")
    fresh_contexts = (parent.get("inputs") or {}).get("fresh_contexts") or {}
    if str(fresh_contexts.get("sha256") or "") != str(contexts_lock["sha256"]):
        raise ValueError("canonical A contexts differ from parent lock")


def _validate_v6_assets(
    v6: Mapping[str, Any],
    *,
    verify_local_artifact_identity: bool,
) -> None:
    if v6.get("status") != "FROZEN_BEFORE_RETRIEVAL":
        raise ValueError("inherited v6 asset protocol status drift")
    if int((v6.get("retrieval_assets") or {}).get("expected_documents", -1)) != EXPECTED_WIKI18_DOCUMENTS:
        raise ValueError("inherited Wiki18 document count drift")
    settings = v6.get("settings") or {}
    expected_settings = {
        "network_access": False,
        "rrf_candidate_k": 100,
        "retrieval_query_max_length": 128,
        "step_rerank_topk": 10,
        "protected_originals": 8,
        "total_passages": 10,
        "ce_max_chars": 1200,
        "root_hop_injection": False,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"inherited v6 setting drift: {key}")
    models = v6.get("models") or {}
    content_locks = v6.get("model_content_locks") or {}
    for name in ("retrieval_encoder", "cross_encoder", "strong_sft", "base_model"):
        if name not in models or name not in content_locks:
            raise ValueError(f"v6 model lock missing: {name}")
        if verify_local_artifact_identity:
            path = Path(str(models[name].get("path") or "")).expanduser().resolve()
            if artifact_identity(path) != models[name]:
                raise ValueError(f"local model artifact identity drift: {name}")


def build_freeze_bundle(
    *,
    source_path: Path,
    parent_protocol_path: Path,
    contexts_path: Path,
    design_path: Path,
    design_manifest_path: Path,
    exposure_audit_path: Path,
    v6_protocol_path: Path,
    planner_adapter_path: Path,
    planner_config_path: Path,
    code_paths: Mapping[str, Path],
    output_dir: Path,
    expected_source_sha256: str = EXPECTED_SOURCE_SHA256,
    expected_parent_protocol_sha256: str = EXPECTED_PARENT_PROTOCOL_SHA256,
    expected_design_sha256: str = EXPECTED_DESIGN_SHA256,
    expected_design_manifest_sha256: str = EXPECTED_DESIGN_MANIFEST_SHA256,
    expected_exposure_audit_sha256: str = EXPECTED_EXPOSURE_AUDIT_SHA256,
    expected_v6_protocol_sha256: str = EXPECTED_V6_PROTOCOL_SHA256,
    expected_planner_config_sha256: str = EXPECTED_PLANNER_CONFIG_SHA256,
    expected_source_counts: Mapping[str, int] = SOURCE_COUNTS,
    n_per_dataset: int = SELECTED_PER_DATASET,
    verify_local_artifact_identity: bool = True,
) -> dict[str, Any]:
    locks = {
        "source_question_only_pool": file_lock(source_path),
        "parent_saeg_protocol": file_lock(parent_protocol_path),
        "canonical_A_contexts": file_lock(contexts_path),
        "v7_design": file_lock(design_path),
        "v7_design_manifest": file_lock(design_manifest_path),
        "historical_identity_audit": file_lock(exposure_audit_path),
        "v6_asset_protocol": file_lock(v6_protocol_path),
        "planner_config": file_lock(planner_config_path),
    }
    for name, expected in (
        ("source_question_only_pool", expected_source_sha256),
        ("parent_saeg_protocol", expected_parent_protocol_sha256),
        ("v7_design", expected_design_sha256),
        ("v7_design_manifest", expected_design_manifest_sha256),
        ("historical_identity_audit", expected_exposure_audit_sha256),
        ("v6_asset_protocol", expected_v6_protocol_sha256),
        ("planner_config", expected_planner_config_sha256),
    ):
        _expect_hash(locks[name], expected, name)

    design = read_json(design_path)
    design_manifest = read_json(design_manifest_path)
    exposure = read_json(exposure_audit_path)
    parent = read_json(parent_protocol_path)
    v6 = read_json(v6_protocol_path)
    _validate_design(design)
    if design_manifest.get("status") != design.get("status"):
        raise ValueError("v7 design manifest status drift")
    if exposure.get("status") != "ADVISORY_IDENTITY_SCAN_NOT_A_FRESHNESS_PROOF":
        raise ValueError("historical exposure audit status drift")
    if (exposure.get("matching") or {}).get("known_scored_identity_overlap") != {
        "hotpotqa": 0,
        "musique": 0,
    }:
        raise ValueError("historical exposure audit result drift")
    _validate_parent(parent, locks["source_question_only_pool"], locks["canonical_A_contexts"])
    _validate_v6_assets(v6, verify_local_artifact_identity=verify_local_artifact_identity)

    source_rows = read_jsonl(source_path)
    validate_source_rows(source_rows, expected_counts=expected_source_counts)
    selected = select_development_rows(
        source_rows, n_per_dataset=n_per_dataset, salt=SELECTION_SALT
    )
    expected_selected = n_per_dataset * len(DATASETS)
    if len(selected) != expected_selected:
        raise RuntimeError("deterministic selection count mismatch")
    planners = planner_rows(selected)
    ledger = reclassification_rows(selected)
    for label, rows in (("selected", selected), ("planner", planners), ("ledger", ledger)):
        for index, row in enumerate(rows):
            assert_answer_free(row, location=f"{label}[{index}]")
    remainder = build_remainder_commitment(source_rows, selected)

    code_locks = {name: file_lock(path) for name, path in sorted(code_paths.items())}
    code_locks["cohort_freezer"] = file_lock(Path(__file__))
    planner_lock = tree_lock(planner_adapter_path)
    inherited_models = dict(v6["models"])
    inherited_content_locks = dict(v6["model_content_locks"])
    retrieval_assets = dict(v6["retrieval_assets"])
    retrieval_asset_content_locks = dict(v6.get("retrieval_asset_content_locks") or {})

    output_paths = {
        "development": output_dir / "development.question_only.jsonl",
        "planner": output_dir / "planner.question_only.jsonl",
        "reclassification_ledger": output_dir / "reclassification_ledger.question_only.jsonl",
        "unselected_commitment": output_dir / "unselected_commitment.json",
    }
    output_payloads = {
        "development": _canonical_jsonl_bytes(selected),
        "planner": _canonical_jsonl_bytes(planners),
        "reclassification_ledger": _canonical_jsonl_bytes(ledger),
        "unselected_commitment": _canonical_json_bytes(remainder),
    }
    output_locks = {
        name: _output_lock(output_paths[name], payload)
        for name, payload in output_payloads.items()
    }

    selection_order = [str(row["question_key"]) for row in selected]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "future_experiment_ids": dict(FUTURE_EXPERIMENT_IDS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
        "population": {
            "n": len(selected),
            "by_dataset": dict(Counter(str(row["dataset"]) for row in selected)),
            "datasets_in_order": list(DATASETS),
            "source_role": "confirmation",
            "new_role": "development_consumed",
            "new_to_dependent_retrieval_line": True,
            "globally_fresh": False,
            "independent_confirmation": False,
            "selection_salt": SELECTION_SALT,
            "selection_formula": (
                "sha256(salt + NUL + dataset + NUL + qid + NUL + "
                "question_sha256 + NUL + family_sha256)"
            ),
            "question_key_order_sha256": sha256_text("\n".join(selection_order)),
            "selected_identity_commitment_sha256": sha256_text(
                _identity_payload(selected)
            ),
            "known_stored_gold_scored_identity_overlap": 0,
            "known_overlap_is_not_freshness_proof": True,
        },
        "unselected": remainder,
        "inputs": locks,
        "outputs": output_locks,
        "models": {
            "query_planner": {
                "path": str(planner_adapter_path.expanduser().resolve()),
                "content_lock": planner_lock,
                "base_model": inherited_models["base_model"],
            },
            "subanswer_and_final_strong_sft": inherited_models["strong_sft"],
            "base_model": inherited_models["base_model"],
            "retrieval_encoder": inherited_models["retrieval_encoder"],
            "cross_encoder": inherited_models["cross_encoder"],
            "inherited_content_locks": inherited_content_locks,
        },
        "retrieval_assets": retrieval_assets,
        "retrieval_asset_content_locks": retrieval_asset_content_locks,
        "arms": design["arms"],
        "planner": design["planner"],
        "paired_execution": design["paired_execution"],
        "retrieval_and_merge": design["retrieval_and_merge"],
        "generation": design["generation"],
        "fallback": design["fallback"],
        "gold_policy": design["gold_policy"],
        "decision_gates": design["decision_gates"],
        "anti_p_hacking": design["anti_p_hacking"],
        "required_telemetry": design["required_telemetry"],
        "code_interfaces_locked_now": code_locks,
        "planned_runtime_paths_not_yet_implemented_or_locked": dict(
            PLANNED_RUNTIME_PATHS
        ),
        "required_next_lock": (
            "Before execution, hash every runtime path above and reverify this "
            "protocol, every model/retrieval content lock, and every output cohort hash."
        ),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "parent_pool_mutated": False,
        "scientific_boundary": design["scientific_boundary"],
    }
    return {
        "selected": selected,
        "planners": planners,
        "ledger": ledger,
        "remainder": remainder,
        "output_paths": output_paths,
        "output_payloads": output_payloads,
        "protocol": protocol,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def write_freeze_bundle(bundle: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen v7 assets: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = bundle["output_payloads"]
    paths = bundle["output_paths"]
    for name in ("development", "planner", "reclassification_ledger", "unselected_commitment"):
        _write_exclusive(paths[name], payloads[name])
    protocol_path = output_dir / "protocol.json"
    protocol_payload = _canonical_json_bytes(bundle["protocol"])
    _write_exclusive(protocol_path, protocol_payload)
    output_artifacts = {
        name: file_lock(path) for name, path in {**paths, "protocol": protocol_path}.items()
    }
    manifest = {
        "schema_version": "dependent-retrieval-v7-preregistration-manifest-1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "artifacts": output_artifacts,
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "parent_pool_mutated": False,
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
        "scientific_boundary": (
            "Question-only cohort/rules freeze only. No planner output, retrieval, "
            "subanswer, final answer, Gold score, training, or utility claim exists."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _write_exclusive(manifest_path, _canonical_json_bytes(manifest))
    return {"protocol": file_lock(protocol_path), "manifest": file_lock(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_names = (
        "planner_generator",
        "subanswer_module",
        "dependent_helpers",
        "query_renderer",
        "merge_helper",
        "prompt_factory",
        "answer_parser",
    )
    bundle = build_freeze_bundle(
        source_path=args.source,
        parent_protocol_path=args.parent_protocol,
        contexts_path=args.contexts,
        design_path=args.design,
        design_manifest_path=args.design_manifest,
        exposure_audit_path=args.exposure_audit,
        v6_protocol_path=args.v6_protocol,
        planner_adapter_path=args.planner_adapter,
        planner_config_path=args.planner_config,
        code_paths={name: getattr(args, name) for name in code_names},
        output_dir=args.out,
    )
    result = write_freeze_bundle(bundle, args.out)
    print(json.dumps({"status": STATUS, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
