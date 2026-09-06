#!/usr/bin/env python
"""Audit identity capacity for a possible subquestion-decomposition v8 cohort.

This is an append-only capacity audit, not a cohort freeze.  The raw train/dev
sources may contain Gold.  The audit opens those sources but projects only
``id``/``qid`` and ``question``; it neither uses nor emits Gold fields, and it
does not emit any individual question identity.

The command-line path uses an explicit, frozen inventory of 58 historical
evaluation/protocol identity registries plus a separately evidenced local
training-input inventory.  Neither inventory is claimed to be a complete
historical training ledger. Runtime discovery (including glob/rglob) is
deliberately absent so that later files cannot silently change the exclusion
set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from scripts.prepare.freeze_qpeg_v1_protocol import (  # noqa: E402
    FAMILY_VERSION,
    family_sha256,
)


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SPLITS = ("train", "dev")
SCHEMA_VERSION = "subquestion-v8-cohort-capacity-audit-v1"
MANIFEST_SCHEMA_VERSION = "subquestion-v8-cohort-capacity-manifest-v1"
EXPERIMENT_ID = "SUBQUESTION-DECOMPOSITION-V8-COHORT-CAPACITY-AUDIT-V1"

DEVELOPMENT_PER_DATASET = 30
PROSPECTIVE_PER_DATASET = 300
PAPER_CONFIRMATION_RESERVE_PER_DATASET = 1000
DEVELOPMENT_PROSPECTIVE_GATE = DEVELOPMENT_PER_DATASET + PROSPECTIVE_PER_DATASET
BALANCED_RESERVE_GATE = (
    DEVELOPMENT_PER_DATASET
    + PROSPECTIVE_PER_DATASET
    + PAPER_CONFIRMATION_RESERVE_PER_DATASET
)

LIMITED_PASS_STATUS = (
    "SCOPE_A_ONLY_PASS_DEV30_PROSPECTIVE300_FAIL_BALANCED_RESERVE1000"
)
FULL_PASS_STATUS = "SCOPE_A_ONLY_PASS_DEV30_PROSPECTIVE300_AND_BALANCED_RESERVE1000"
FAIL_STATUS = "SCOPE_A_ONLY_FAIL_DEV30_PROSPECTIVE300_CAPACITY"
TRAINING_LEDGER_BLOCKED_STATUS = "BLOCKED_TRAINING_INPUT_OUTSIDE_RAW_TRAIN"
SCOPE_B_INVALID_STATUS = (
    "INVALID_FOR_FREEZE_INCOMPLETE_TRAINING_LEDGER_DIAGNOSTIC_ONLY"
)


# Frozen once from the four explicitly documented discovery classes:
#   outputs/audits/**/*question_only*.jsonl
#   outputs/audits/**/cohort*.jsonl
#   outputs/audits/**/*answer_free*.jsonl
#   outputs/audits/**/planner_inputs*.jsonl
# Files without at least one valid dataset + qid + question projection were
# excluded at freeze time.  The formal audit below never repeats discovery.
HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS: tuple[str, ...] = (
    "outputs/audits/2wiki_confirmation270_v3/planner_inputs.confirmation.jsonl",
    "outputs/audits/2wiki_confirmation270_v3/plans/predictions.question_only.jsonl",
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_confirmation.question_only.jsonl",
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_dev.question_only.jsonl",
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_fresh221.question_only.jsonl",
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_reserve.question_only.jsonl",
    "outputs/audits/2wiki_learned_verifier_l0_cohort_freeze/verifier_train.question_only.jsonl",
    "outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/cohort.question_only.jsonl",
    "outputs/audits/2wiki_train_only_rankability_confirmation_n100_v1/plans.question_only.jsonl",
    "outputs/audits/2wiki_train_only_rankability_n150_v1/cohort.question_only.jsonl",
    "outputs/audits/2wiki_train_only_rankability_n150_v1/plans.question_only.jsonl",
    "outputs/audits/automatic_proofkg_2wiki_train_k4_v1_n1500_seed42_preregistration/cohort.question_only.jsonl",
    "outputs/audits/automatic_proofkg_2wiki_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl",
    "outputs/audits/automatic_proofkg_2wiki_v2_independent_n100_seed20260830_preregistration/cohort.question_only.jsonl",
    "outputs/audits/automatic_proofkg_2wiki_v3_independent_n100_seed20260831_preregistration/cohort.question_only.jsonl",
    "outputs/audits/automatic_proofkg_unseen_n100_seed20260830_preregistration/cohort.question_only.jsonl",
    "outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2/planner_inputs.hotpot.jsonl",
    "outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/predictions.question_only.jsonl",
    "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/confirmation.question_only.jsonl",
    "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/pilot.question_only.jsonl",
    "outputs/audits/inference_proofkg_v1_pilot30_offline_diag/2wikimultihopqa/plans.question_only.jsonl",
    "outputs/audits/inference_proofkg_v1_pilot30_offline_diag/musique/plans.question_only.jsonl",
    "outputs/audits/inference_proofkg_v1_pilot30x3_execution_v1/planner_inputs.confirmation.jsonl",
    "outputs/audits/inference_proofkg_v1_pilot30x3_execution_v1/planner_inputs.pilot.jsonl",
    "outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/predictions.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol/population.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v1_n1799_k4_seed42_protocol/prompt_groups.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/ordinary200.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/population.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/prompt_groups.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/proof400.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protected_a_canonical_main.question_only.jsonl",
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protected_a_unopened_confirmation.question_only.jsonl",
    "outputs/audits/musique_relation_graph_pilot30_v1/planner_inputs.musique.jsonl",
    "outputs/audits/musique_relation_graph_pilot30_v1/plans/predictions.question_only.jsonl",
    "outputs/audits/passage_sro_llama_pilot30_v1_protocol/cohort.answer_free.jsonl",
    "outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v1_preregistration/cohort.question_only.jsonl",
    "outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v2_preregistration/cohort.question_only.jsonl",
    "outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration/cohort.question_only.jsonl",
    "outputs/audits/proofkg_dynamic_validity_confirmation_n100_seed20260902_preregistration/cohort.question_only.jsonl",
    "outputs/audits/qpeg_v1_n1350_seed42_preregistration/confirmation.question_only.jsonl",
    "outputs/audits/qpeg_v1_n1350_seed42_preregistration/final.question_only.jsonl",
    "outputs/audits/qpeg_v1_n1350_seed42_preregistration/pilot.question_only.jsonl",
    "outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/confirmation.question_only.jsonl",
    "outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/development.question_only.jsonl",
    "outputs/audits/qpeg_v4_schema_adaptation_protocol_v1/train.question_only.jsonl",
    "outputs/audits/query_aware_kg_relation_coverage_train150_seed20260828_v1/cohort.jsonl",
    "outputs/audits/query_aware_proof_kg_2wiki_train150_seed20260829_confirmation_v1/cohort.jsonl",
    "outputs/audits/query_planner_v2_a1_single_review_n30_seed20260829/cohort.jsonl",
    "outputs/audits/query_planner_v2_a1_single_review_n30_seed20260829_retry1/cohort.jsonl",
    "outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum/effective_train.question_only.jsonl",
    "outputs/audits/saeg_v1_evaluation_protocol_v1/2wiki_dev_confirmation_planner.question_only.jsonl",
    "outputs/audits/saeg_v1_evaluation_protocol_v1/canonical_reporting.question_only.jsonl",
    "outputs/audits/saeg_v1_evaluation_protocol_v1/confirmation.question_only.jsonl",
    "outputs/audits/saeg_v1_evaluation_protocol_v1/development.question_only.jsonl",
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/development.question_only.jsonl",
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/planner.question_only.jsonl",
    "outputs/audits/versioned_2wiki_store_v1_independent_n100_seed20260901_preregistration/cohort.question_only.jsonl",
)

if len(HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS) != 58:
    raise RuntimeError(
        "the historical evaluation/protocol registry inventory must contain exactly 58 paths"
    )
if len(set(HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS)) != 58:
    raise RuntimeError("the evaluation/protocol registry inventory contains duplicate paths")
if (
    tuple(sorted(HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS))
    != HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
):
    raise RuntimeError("the evaluation/protocol registry inventory must remain sorted")


@dataclass(frozen=True)
class TrainingInputSpec:
    """Static local training-input evidence; row identities are never emitted."""

    path: str
    evidence_path: str
    ledger_state: str
    dataset_hint: str | None = None
    qid_alias: str = "qid"


COMPLETED_TRAINING = "completed_training_or_runtime_consumption"
PREPARED_CONSERVATIVE = "prepared_not_trained_conservative_exclusion"

# Every path below is declared by the paired manifest/config in evidence_path.
# SAEG and mixed-PPO inputs are deliberately included as conservative exclusions
# but are not described as completed training (their local states are
# PASS_NOT_TRAINED / CONFIGURED_NOT_STARTED / PREPARED_BLOCKED_GPU_PROBE).
LOCAL_TRAINING_INPUT_SPECS: tuple[TrainingInputSpec, ...] = (
    TrainingInputSpec(
        "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl",
        "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/question_kg_records.jsonl",
        "outputs/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/automatic_proofkg_2wiki_train_k4_v1/silver_train.jsonl",
        "outputs/ppo_automatic_proofkg_2wiki_k4_smoke600_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/question_kg_records.jsonl",
        "outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v2/ppo_t.lock.json",
        PREPARED_CONSERVATIVE,
    ),
    TrainingInputSpec(
        "data/silver_data/mixed_ppo_three_dataset_v1_n1799_k4_seed42/silver_train.jsonl",
        "outputs/audits/mixed3_rearag_ppo_pair_7200_seed42_v2/ppo_t.lock.json",
        PREPARED_CONSERVATIVE,
    ),
    TrainingInputSpec(
        "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/prompt_groups.jsonl",
        "outputs/audits/mixed3_rearag_proof400_ppo_pair_7200_seed42_v2/ppo_t.lock.json",
        PREPARED_CONSERVATIVE,
    ),
    TrainingInputSpec(
        "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/question_kg_records.jsonl",
        "outputs/audits/mixed3_rearag_proof400_ppo_pair_7200_seed42_v2/ppo_t.lock.json",
        PREPARED_CONSERVATIVE,
    ),
    TrainingInputSpec(
        "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42/silver_train.jsonl",
        "outputs/audits/mixed3_rearag_proof400_ppo_pair_7200_seed42_v2/ppo_t.lock.json",
        PREPARED_CONSERVATIVE,
    ),
    TrainingInputSpec(
        "data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/old10_bridge5_v3.jsonl",
        "outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/manifest.json",
        COMPLETED_TRAINING,
        dataset_hint="hotpotqa",
    ),
    TrainingInputSpec(
        "data/silver_data/pilots/ppo_smoke_hybrid_train_seed42_20260828/schedule600.jsonl",
        "outputs/ppo_legacy_repaired_v2_quota70_hard_seed42_no_text_head_smoke600_hybrid_old10_bridge5_v3/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/question_kg_records.jsonl",
        "checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/proofkg_curriculum_light20_v2_n5000_seed42_20260831/silver_curriculum.jsonl",
        "checkpoints/sft_proofkg_curriculum_light20_v2_n5000_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/question_kg_records.jsonl",
        "checkpoints/sft_proofkg_curriculum_mix_v1_n8000_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829/silver_curriculum.jsonl",
        "checkpoints/sft_proofkg_curriculum_mix_v1_n8000_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl",
        "checkpoints/sft_qpeg_v4_schema_adaptation_n2400_seed42/manifest.json",
        COMPLETED_TRAINING,
        qid_alias="metadata.source_qid",
    ),
    TrainingInputSpec(
        "data/silver_data/query_planner_supervision_split_v1_seed20260829/dev.jsonl",
        "checkpoints/query_planner_learned_scale_v1_1_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/query_planner_supervision_split_v1_seed20260829/train.jsonl",
        "checkpoints/query_planner_learned_scale_v1_1_seed42/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/saeg_v1_sft_balanced_epoch4860_seed42_v1/silver_train.jsonl",
        "configs/training/phase3_sft_saeg_v1_balanced_epoch4860_seed42.yaml",
        PREPARED_CONSERVATIVE,
        qid_alias="source_qid",
    ),
    TrainingInputSpec(
        "data/silver_data/silver_legacy_repaired_v2_quota70_hotpot_train.jsonl",
        "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/manifest.json",
        COMPLETED_TRAINING,
    ),
    TrainingInputSpec(
        "data/silver_data/silver_v1_reannotated.jsonl",
        "checkpoints/sft_student_split/manifest.json",
        COMPLETED_TRAINING,
    ),
)

if len({spec.path for spec in LOCAL_TRAINING_INPUT_SPECS}) != len(LOCAL_TRAINING_INPUT_SPECS):
    raise RuntimeError("local training input inventory contains duplicate paths")
if tuple(sorted(LOCAL_TRAINING_INPUT_SPECS, key=lambda spec: spec.path)) != LOCAL_TRAINING_INPUT_SPECS:
    raise RuntimeError("local training input inventory must remain sorted")


@dataclass(frozen=True)
class Identity:
    """In-memory identity projection.  Instances are never serialized."""

    dataset: str
    qid: str
    question_sha256: str
    family_sha256: str


@dataclass
class RawSplitProjection:
    identities: dict[str, Identity]
    source_identity: dict[str, str]
    stats: dict[str, int]


@dataclass
class HistoricalProjection:
    qids: set[tuple[str, str]]
    families: set[tuple[str, str]]
    question_hashes: set[tuple[str, str]]
    inventory: list[dict[str, str]]
    stats: dict[str, Any]


@dataclass
class TrainingProjection:
    qids: set[tuple[str, str]]
    families: set[tuple[str, str]]
    inventory: list[dict[str, str]]
    evidence_inventory: list[dict[str, str]]
    stats: dict[str, Any]
    raw_train_containment_pass: bool


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _raw_qid(row: Mapping[str, Any]) -> tuple[str, bool]:
    """Read only raw ``id``/``qid`` and reject an ambiguous dual identity."""

    row_id = _clean(row.get("id"))
    row_qid = _clean(row.get("qid"))
    if row_id and row_qid and row_id != row_qid:
        return "", True
    return row_id or row_qid, False


def _historical_qid(row: Mapping[str, Any]) -> str:
    # Historical schemas are heterogeneous.  source_id is an identity alias in
    # two old train-only cohort registries; no content/Gold field is accessed.
    for field in ("qid", "id", "source_id"):
        value = _clean(row.get(field))
        if value:
            return value
    return ""


def _training_qid(row: Mapping[str, Any], alias: str) -> str:
    if alias == "qid":
        return _clean(row.get("qid"))
    if alias == "source_qid":
        return _clean(row.get("source_qid"))
    if alias == "metadata.source_qid":
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            return ""
        return _clean(metadata.get("source_qid"))
    raise ValueError(f"unsupported static training qid alias: {alias!r}")


def _safe_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"inventory path must be a safe relative path: {relative_path!r}")
    return path


def _json_row(raw_line: bytes, *, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON at {path}:{line_number}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
    return value


def _project_raw_split(path: Path, dataset: str) -> RawSplitProjection:
    if not path.is_file():
        raise FileNotFoundError(f"missing raw capacity source: {path}")

    digest = hashlib.sha256()
    identities: dict[str, Identity] = {}
    stats: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                stats["blank_rows"] += 1
                continue
            stats["rows_parsed"] += 1
            row = _json_row(raw_line, path=path, line_number=line_number)
            qid, conflict = _raw_qid(row)
            question = _clean(row.get("question"))
            if conflict:
                stats["conflicting_id_qid_rows"] += 1
                continue
            if not qid:
                stats["missing_identity_rows"] += 1
                continue
            if not question:
                stats["missing_question_rows"] += 1
                continue

            identity = Identity(
                dataset=dataset,
                qid=qid,
                question_sha256=question_sha256(question),
                family_sha256=family_sha256(question),
            )
            stats["question_hashes_recomputed_rows"] += 1
            stats["family_hashes_recomputed_rows"] += 1
            previous = identities.get(qid)
            if previous is None:
                identities[qid] = identity
                continue
            if previous != identity:
                raise ValueError(
                    f"raw source has one dataset-scoped identity with conflicting question text: "
                    f"{path}:{line_number}"
                )
            stats["duplicate_identical_identity_rows"] += 1

    stats["valid_unique_qids"] = len(identities)
    stats["valid_unique_question_hashes"] = len(
        {identity.question_sha256 for identity in identities.values()}
    )
    stats["valid_unique_dataset_scoped_families"] = len(
        {identity.family_sha256 for identity in identities.values()}
    )
    return RawSplitProjection(
        identities=identities,
        source_identity={"path": path.as_posix(), "sha256": digest.hexdigest()},
        stats=dict(sorted(stats.items())),
    )


def _project_historical_registries(
    project_root: Path,
    relative_paths: Sequence[str],
) -> HistoricalProjection:
    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    question_hashes: set[tuple[str, str]] = set()
    inventory: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    rows_by_dataset: Counter[str] = Counter()
    qids_by_dataset: dict[str, set[str]] = defaultdict(set)
    families_by_dataset: dict[str, set[str]] = defaultdict(set)
    qid_to_question_hash: dict[tuple[str, str], str] = {}

    if not relative_paths:
        raise ValueError("historical evaluation/protocol registry inventory must not be empty")
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("historical evaluation/protocol registry inventory has duplicate paths")

    for value in relative_paths:
        relative = _safe_relative_path(value)
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"missing historical evaluation/protocol registry: {relative.as_posix()}"
            )

        digest = hashlib.sha256()
        qualified_in_file = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    counts["blank_rows"] += 1
                    continue
                counts["rows_parsed"] += 1
                row = _json_row(raw_line, path=path, line_number=line_number)
                dataset = _clean(row.get("dataset")).lower()
                qid = _historical_qid(row)
                question = _clean(row.get("question"))
                if dataset not in DATASETS:
                    counts["invalid_or_unsupported_dataset_rows"] += 1
                    continue
                if not qid:
                    counts["missing_identity_rows"] += 1
                    continue
                if not question:
                    counts["missing_question_rows"] += 1
                    continue

                recomputed_question_hash = question_sha256(question)
                recomputed_family_hash = family_sha256(question)
                key = (dataset, qid)
                previous = qid_to_question_hash.get(key)
                if previous is not None and previous != recomputed_question_hash:
                    raise ValueError(
                        "historical registries disagree on question text for one "
                        f"dataset-scoped identity at {relative.as_posix()}:{line_number}"
                    )
                qid_to_question_hash[key] = recomputed_question_hash
                qids.add(key)
                families.add((dataset, recomputed_family_hash))
                question_hashes.add((dataset, recomputed_question_hash))
                qids_by_dataset[dataset].add(qid)
                families_by_dataset[dataset].add(recomputed_family_hash)
                rows_by_dataset[dataset] += 1
                counts["qualified_identity_rows"] += 1
                counts["question_hashes_recomputed_rows"] += 1
                counts["family_hashes_recomputed_rows"] += 1
                qualified_in_file += 1

        if qualified_in_file == 0:
            raise ValueError(
                "historical evaluation/protocol registry has no valid dataset/qid/question "
                f"projection: {relative.as_posix()}"
            )
        inventory.append({"path": relative.as_posix(), "sha256": digest.hexdigest()})

    counts["source_file_count"] = len(inventory)
    counts["qualified_unique_dataset_scoped_qids"] = len(qids)
    counts["qualified_unique_dataset_scoped_families"] = len(families)
    counts["qualified_unique_dataset_scoped_question_hashes"] = len(question_hashes)
    stats: dict[str, Any] = dict(sorted(counts.items()))
    stats["by_dataset"] = {
        dataset: {
            "qualified_rows": rows_by_dataset[dataset],
            "qualified_unique_qids": len(qids_by_dataset[dataset]),
            "qualified_unique_dataset_scoped_families": len(families_by_dataset[dataset]),
        }
        for dataset in DATASETS
    }
    return HistoricalProjection(
        qids=qids,
        families=families,
        question_hashes=question_hashes,
        inventory=inventory,
        stats=stats,
    )


def _project_training_inputs(
    project_root: Path,
    specs: Sequence[TrainingInputSpec],
    raw: Mapping[str, Mapping[str, RawSplitProjection]],
) -> TrainingProjection:
    if not specs:
        raise ValueError("local training input inventory must not be empty")
    if len({spec.path for spec in specs}) != len(specs):
        raise ValueError("local training input inventory contains duplicate paths")

    raw_train_qids = {
        dataset: set(raw[dataset]["train"].identities) for dataset in DATASETS
    }
    raw_train_families = {
        dataset: {
            identity.family_sha256
            for identity in raw[dataset]["train"].identities.values()
        }
        for dataset in DATASETS
    }
    raw_train_question_qids: dict[str, dict[str, set[str]]] = {
        dataset: defaultdict(set) for dataset in DATASETS
    }
    for dataset in DATASETS:
        for qid, identity in raw[dataset]["train"].identities.items():
            raw_train_question_qids[dataset][identity.question_sha256].add(qid)

    evidence_text: dict[str, str] = {}
    evidence_inventory: list[dict[str, str]] = []
    for evidence_value in sorted({spec.evidence_path for spec in specs}):
        evidence_relative = _safe_relative_path(evidence_value)
        evidence_path = project_root / evidence_relative
        if not evidence_path.is_file():
            raise FileNotFoundError(
                f"missing training-input provenance evidence: {evidence_relative.as_posix()}"
            )
        payload = evidence_path.read_bytes()
        evidence_text[evidence_value] = payload.decode("utf-8")
        evidence_inventory.append(
            {
                "path": evidence_relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    canonical_qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    outside_qid_identities: set[tuple[str, str]] = set()
    outside_families: set[tuple[str, str]] = set()
    inventory: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    rows_by_dataset: Counter[str] = Counter()
    files_by_state: Counter[str] = Counter()
    rows_by_state: Counter[str] = Counter()

    for spec in specs:
        relative = _safe_relative_path(spec.path)
        if spec.path not in evidence_text[spec.evidence_path]:
            raise ValueError(
                "training input is not declared by its frozen manifest/config evidence: "
                f"{spec.path}"
            )
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing local training input: {relative.as_posix()}")
        if spec.dataset_hint is not None and spec.dataset_hint not in DATASETS:
            raise ValueError(f"invalid static dataset hint for {spec.path}")

        digest = hashlib.sha256()
        qualified_in_file = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    counts["blank_rows"] += 1
                    continue
                counts["rows_parsed"] += 1
                row = _json_row(raw_line, path=path, line_number=line_number)
                row_dataset = _clean(row.get("dataset")).lower()
                if row_dataset and spec.dataset_hint and row_dataset != spec.dataset_hint:
                    raise ValueError(
                        f"training input conflicts with its static dataset hint: {spec.path}"
                    )
                dataset = row_dataset or _clean(spec.dataset_hint).lower()
                source_qid = _training_qid(row, spec.qid_alias)
                question = _clean(row.get("question"))
                if dataset not in DATASETS:
                    counts["invalid_or_unsupported_dataset_rows"] += 1
                    continue
                if not source_qid:
                    counts["missing_identity_rows"] += 1
                    continue
                if not question:
                    counts["missing_question_rows"] += 1
                    continue

                question_hash = question_sha256(question)
                family_hash = family_sha256(question)
                raw_question_matches = raw_train_question_qids[dataset].get(
                    question_hash, set()
                )
                if source_qid in raw_train_qids[dataset]:
                    counts["source_qid_direct_raw_train_match_rows"] += 1
                elif raw_question_matches:
                    counts["source_qid_resolved_by_exact_question_hash_rows"] += 1
                if len(raw_question_matches) > 1:
                    counts["exact_question_hash_ambiguous_raw_train_rows"] += 1
                if not raw_question_matches:
                    outside_qid_identities.add((dataset, source_qid))
                else:
                    # Duplicate raw questions are resolved conservatively: all
                    # matching raw-train qids enter the exclusion ledger.
                    canonical_qids.update((dataset, qid) for qid in raw_question_matches)
                if family_hash not in raw_train_families[dataset]:
                    outside_families.add((dataset, family_hash))
                families.add((dataset, family_hash))
                counts["qualified_identity_rows"] += 1
                counts["question_hashes_recomputed_rows"] += 1
                counts["family_hashes_recomputed_rows"] += 1
                rows_by_dataset[dataset] += 1
                rows_by_state[spec.ledger_state] += 1
                qualified_in_file += 1

        if qualified_in_file == 0:
            raise ValueError(
                "local training input has no valid dataset/qid/question projection: "
                f"{relative.as_posix()}"
            )
        inventory.append({"path": relative.as_posix(), "sha256": digest.hexdigest()})
        files_by_state[spec.ledger_state] += 1

    counts["source_file_count"] = len(inventory)
    counts["provenance_evidence_file_count"] = len(evidence_inventory)
    counts["canonical_raw_train_unique_dataset_scoped_qids"] = len(canonical_qids)
    counts["unique_dataset_scoped_families"] = len(families)
    counts["outside_raw_train_unique_dataset_scoped_qids"] = len(outside_qid_identities)
    counts["outside_raw_train_unique_dataset_scoped_families"] = len(outside_families)
    containment_pass = not outside_qid_identities and not outside_families
    stats: dict[str, Any] = dict(sorted(counts.items()))
    stats["by_dataset"] = {
        dataset: {"qualified_rows": rows_by_dataset[dataset]} for dataset in DATASETS
    }
    stats["by_ledger_state"] = {
        state: {"source_files": files_by_state[state], "qualified_rows": rows_by_state[state]}
        for state in (COMPLETED_TRAINING, PREPARED_CONSERVATIVE)
    }
    stats["raw_train_containment_gate"] = {
        "outside_raw_train_unique_dataset_scoped_qids": len(outside_qid_identities),
        "outside_raw_train_unique_dataset_scoped_families": len(outside_families),
        "pass": containment_pass,
        "qid_resolution": (
            "Use the declared source qid alias when it directly matches raw train; otherwise "
            "resolve by exact recomputed question hash and conservatively include every raw-train "
            "qid when duplicate raw questions exist."
        ),
    }
    return TrainingProjection(
        qids=canonical_qids,
        families=families,
        inventory=inventory,
        evidence_inventory=evidence_inventory,
        stats=stats,
        raw_train_containment_pass=containment_pass,
    )


def _scope_counts(
    *,
    dataset: str,
    dev: Mapping[str, Identity],
    historical: HistoricalProjection,
    training: TrainingProjection,
    raw_train_qids: set[str],
    raw_train_families: set[str],
    exclude_training_ledger: bool,
    require_raw_train_family_isolation: bool,
    gate_330: int,
    gate_1330: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    historical_qid_hits = {
        qid for qid in dev if (dataset, qid) in historical.qids
    }
    historical_family_hits = {
        qid
        for qid, identity in dev.items()
        if (dataset, identity.family_sha256) in historical.families
    }
    training_qid_hits = {qid for qid in dev if (dataset, qid) in training.qids}
    training_family_hits = {
        qid
        for qid, identity in dev.items()
        if (dataset, identity.family_sha256) in training.families
    }
    raw_train_family_hits = {
        qid for qid, identity in dev.items() if identity.family_sha256 in raw_train_families
    }
    raw_train_qid_hits = set(dev) & raw_train_qids

    eligible: dict[str, Identity] = {}
    for qid, identity in dev.items():
        if qid in historical_qid_hits or qid in historical_family_hits:
            continue
        if exclude_training_ledger and (
            qid in training_qid_hits or qid in training_family_hits
        ):
            continue
        if qid in raw_train_qid_hits:
            continue
        if require_raw_train_family_isolation and qid in raw_train_family_hits:
            continue
        eligible[qid] = identity

    eligible_qids = set(eligible)
    eligible_families = {identity.family_sha256 for identity in eligible.values()}
    freezable_capacity = min(len(eligible_qids), len(eligible_families))
    historical_qid_overlap = sum(
        (dataset, qid) in historical.qids for qid in eligible_qids
    )
    historical_family_overlap = sum(
        (dataset, family) in historical.families for family in eligible_families
    )
    raw_train_qid_overlap = len(eligible_qids & raw_train_qids)
    training_qid_overlap = sum((dataset, qid) in training.qids for qid in eligible_qids)
    training_family_overlap = sum(
        (dataset, family) in training.families for family in eligible_families
    )
    counts = {
        "raw_dev_valid_unique_qids": len(dev),
        "raw_dev_unique_dataset_scoped_families": len(
            {identity.family_sha256 for identity in dev.values()}
        ),
        "historical_qid_hit_unique_qids": len(historical_qid_hits),
        "historical_family_hit_unique_qids": len(historical_family_hits),
        "historical_qid_or_family_hit_unique_qids": len(
            historical_qid_hits | historical_family_hits
        ),
        "training_ledger_qid_hit_unique_qids": len(training_qid_hits),
        "training_ledger_family_hit_unique_qids": len(training_family_hits),
        "training_ledger_qid_or_family_hit_unique_qids": len(
            training_qid_hits | training_family_hits
        ),
        "raw_train_qid_hit_unique_qids": len(raw_train_qid_hits),
        "raw_train_family_hit_unique_qids": len(raw_train_family_hits),
        "eligible_unique_qids": len(eligible_qids),
        "eligible_unique_dataset_scoped_families": len(eligible_families),
        "exact_freezable_capacity_one_per_dataset_scoped_family": freezable_capacity,
        "eligible_historical_registry_qid_overlap": historical_qid_overlap,
        "eligible_historical_registry_family_overlap": historical_family_overlap,
        "eligible_raw_train_qid_overlap": raw_train_qid_overlap,
        "eligible_training_ledger_qid_overlap": training_qid_overlap,
        "eligible_training_ledger_family_overlap": training_family_overlap,
        "gate_dev30_plus_prospective300_n330": freezable_capacity >= gate_330,
        "gate_dev30_plus_prospective300_plus_reserve1000_n1330": (
            freezable_capacity >= gate_1330
        ),
    }
    return counts, eligible_qids, eligible_families


def _scope_report(
    *,
    raw: Mapping[str, Mapping[str, RawSplitProjection]],
    historical: HistoricalProjection,
    training: TrainingProjection,
    exclude_training_ledger: bool,
    require_raw_train_family_isolation: bool,
    gate_330: int,
    gate_1330: int,
) -> tuple[dict[str, Any], bool, bool]:
    by_dataset: dict[str, Any] = {}
    all_330 = True
    all_1330 = True
    for dataset in DATASETS:
        train_families = {
            identity.family_sha256
            for identity in raw[dataset]["train"].identities.values()
        }
        counts, _, _ = _scope_counts(
            dataset=dataset,
            dev=raw[dataset]["dev"].identities,
            historical=historical,
            training=training,
            raw_train_qids=set(raw[dataset]["train"].identities),
            raw_train_families=train_families,
            exclude_training_ledger=exclude_training_ledger,
            require_raw_train_family_isolation=require_raw_train_family_isolation,
            gate_330=gate_330,
            gate_1330=gate_1330,
        )
        by_dataset[dataset] = counts
        all_330 = all_330 and counts["gate_dev30_plus_prospective300_n330"]
        all_1330 = (
            all_1330
            and counts["gate_dev30_plus_prospective300_plus_reserve1000_n1330"]
        )
    return (
        {
            "by_dataset": by_dataset,
            "balanced_all_datasets_gate_n330": all_330,
            "balanced_all_datasets_gate_n1330": all_1330,
        },
        all_330,
        all_1330,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _implementation_inventory(project_root: Path) -> list[dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        Path(question_sha256.__code__.co_filename).resolve(),
        Path(family_sha256.__code__.co_filename).resolve(),
    )
    unique_paths = sorted(set(paths), key=lambda value: value.as_posix())
    return [
        {
            "path": _display_path(path, project_root),
            "sha256": _sha256_file(path),
        }
        for path in unique_paths
    ]


def run_capacity_audit(
    *,
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
    historical_registry_paths: Sequence[str] | None = None,
    training_input_specs: Sequence[TrainingInputSpec] | None = None,
    development_prospective_gate: int = DEVELOPMENT_PROSPECTIVE_GATE,
    balanced_reserve_gate: int = BALANCED_RESERVE_GATE,
) -> dict[str, Any]:
    """Run the aggregate-only capacity audit and create one new output directory.

    The two optional inventories exist only to permit isolated fixture tests.
    The command-line entry point always uses both frozen static constants.
    """

    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only capacity audit directory: {output_dir}"
        )
    if not _clean(experiment_id):
        raise ValueError("experiment_id must be non-empty")
    if development_prospective_gate <= 0 or balanced_reserve_gate < development_prospective_gate:
        raise ValueError("capacity gates must be positive and ordered")

    registry_paths = tuple(
        HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
        if historical_registry_paths is None
        else historical_registry_paths
    )
    if historical_registry_paths is None and len(registry_paths) != 58:
        raise RuntimeError("formal audit must use the frozen 58-path inventory")

    selected_training_specs = tuple(
        LOCAL_TRAINING_INPUT_SPECS if training_input_specs is None else training_input_specs
    )
    raw: dict[str, dict[str, RawSplitProjection]] = {}
    raw_source_inventory: list[dict[str, str]] = []
    for dataset in DATASETS:
        raw[dataset] = {}
        for split in SPLITS:
            projection = _project_raw_split(data_root / dataset / f"{split}.jsonl", dataset)
            # Store project-relative paths when possible.  Only path/hash enter
            # an output inventory; no row identity is serialized.
            projection.source_identity["path"] = _display_path(
                data_root / dataset / f"{split}.jsonl", project_root
            )
            raw[dataset][split] = projection
            raw_source_inventory.append(dict(projection.source_identity))
    historical = _project_historical_registries(project_root, registry_paths)
    training = _project_training_inputs(project_root, selected_training_specs, raw)

    scope_a, scope_a_330, scope_a_1330 = _scope_report(
        raw=raw,
        historical=historical,
        training=training,
        exclude_training_ledger=True,
        require_raw_train_family_isolation=True,
        gate_330=development_prospective_gate,
        gate_1330=balanced_reserve_gate,
    )
    scope_b, _, _ = _scope_report(
        raw=raw,
        historical=historical,
        training=training,
        exclude_training_ledger=False,
        require_raw_train_family_isolation=False,
        gate_330=development_prospective_gate,
        gate_1330=balanced_reserve_gate,
    )

    if not training.raw_train_containment_pass:
        status = TRAINING_LEDGER_BLOCKED_STATUS
    elif scope_a_1330:
        status = FULL_PASS_STATUS
    elif scope_a_330:
        status = LIMITED_PASS_STATUS
    else:
        status = FAIL_STATUS

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    raw_projection_stats = {
        dataset: {
            split: raw[dataset][split].stats
            for split in SPLITS
        }
        for dataset in DATASETS
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": status,
        "audit_kind": "capacity_only_not_cohort_freeze",
        "source_files_opened": True,
        "source_may_contain_gold": True,
        "gold_fields_used": False,
        "gold_fields_emitted": False,
        "source_access_disclosure": {
            "source_files_opened": True,
            "source_may_contain_gold": True,
            "gold_fields_used": False,
            "gold_fields_emitted": False,
            "raw_source_fields_used": ["id", "qid", "question"],
            "historical_registry_fields_used": [
                "dataset",
                "id",
                "qid",
                "source_id",
                "question",
            ],
            "training_input_fields_used": [
                "dataset",
                "qid_or_declared_source_qid_alias",
                "question",
            ],
            "full_source_bytes_hashed_for_provenance_only": True,
            "source_file_hashes_used_for_capacity_or_selection": False,
            "stored_question_or_family_hashes_trusted": False,
        },
        "identity_derivation": {
            "question_hash": "sha256(str(question).strip().encode('utf-8'))",
            "question_hash_recomputed_for_every_qualified_row": True,
            "family_version": FAMILY_VERSION,
            "family_hash_recomputed_for_every_qualified_row": True,
            "family_scope": "dataset-scoped (dataset, family_sha256)",
            "family_semantics": "lexical-family proxy; not a semantic-family claim",
        },
        "budgets": {
            "development_per_dataset": DEVELOPMENT_PER_DATASET,
            "prospective_validation_per_dataset": PROSPECTIVE_PER_DATASET,
            "paper_confirmation_reserve_per_dataset": PAPER_CONFIRMATION_RESERVE_PER_DATASET,
            "gate_n330": development_prospective_gate,
            "gate_n1330": balanced_reserve_gate,
        },
        "historical_evaluation_protocol_registry_projection": historical.stats,
        "local_training_input_projection": training.stats,
        "training_ledger_completeness": {
            "complete_historical_training_ledger_available": False,
            "missing_old_checkpoint_input_ledgers": "UNKNOWN",
            "local_inventory_scope": (
                "Static locally verifiable completed/runtime-consumed inputs plus explicitly "
                "identified prepared-but-not-trained SAEG and mixed-PPO inputs included only "
                "as conservative exclusions."
            ),
            "conservative_coverage": (
                "Scope A additionally excludes every family in the corresponding raw-train "
                "source, so locally missing old checkpoint ledgers are not represented as "
                "known-complete training provenance."
            ),
            "saeg_training_status": "PASS_NOT_TRAINED",
            "mixed_ppo_v1_training_status": "CONFIGURED_NOT_STARTED",
            "mixed_ppo_v2_training_status": "PREPARED_BLOCKED_GPU_PROBE",
        },
        "existing_materialized_identity_source_evidence": {
            "frozen_registry_file_count": len(historical.inventory),
            "qualified_unique_dataset_scoped_qids": len(historical.qids),
            "classified_consumed_or_protected_unique_dataset_scoped_qids": len(
                historical.qids
            ),
            "existing_answer_free_unused_pool": 0,
            "definition": (
                "Every qualified dataset::qid in the frozen 58-file historical evaluation/"
                "protocol registry inventory belongs to the consumed/protected union; "
                "there is no unclassified existing identity row available to freeze "
                "directly as a fresh v8 cohort. This is not a complete training ledger."
            ),
        },
        "raw_source_projection": raw_projection_stats,
        "scope_a_strict": {
            "definition": (
                "Candidate dataset::qid must not match any historical consumed/protected "
                "dataset::qid, any locally verifiable training-ledger qid, or a raw-train "
                "dataset::qid; candidate dataset-scoped lexical family must not match any "
                "historical consumed/protected family, any locally verifiable training-ledger "
                "family, or any family in the same dataset's raw-train source. Capacity "
                "permits at most one candidate per dataset-scoped family. Scope A alone "
                "controls the reported capacity decision."
            ),
            "used_for_capacity_decision": True,
            **scope_a,
        },
        "scope_b_relaxed_raw_train_family_isolation": {
            "decision_status": SCOPE_B_INVALID_STATUS,
            "used_for_capacity_decision": False,
            "definition": (
                "Candidate dataset::qid must not match any historical consumed/protected "
                "dataset::qid or a raw-train dataset::qid, and its family must not match a "
                "historical consumed/protected family. Unlike Scope A, raw-train families and "
                "the separately reconstructed local training ledger are not complete exclusion "
                "grounds here. Because old training ledgers are incomplete, Scope B is a "
                "diagnostic only and cannot support any freeze or capacity decision."
            ),
            **scope_b,
        },
        "checks": {
            "formal_inventory_has_58_explicit_paths": len(historical.inventory) == 58,
            "formal_training_inventory_is_static": training_input_specs is None,
            "training_inputs_within_corresponding_raw_train": (
                training.raw_train_containment_pass
            ),
            "capacity_decision_scope_a_only": True,
            "scope_b_invalid_for_freeze": True,
            "no_runtime_registry_discovery": (
                historical_registry_paths is None and training_input_specs is None
            ),
            "scope_a_eligible_exclusion_overlaps_zero": all(
                values["eligible_historical_registry_qid_overlap"] == 0
                and values["eligible_historical_registry_family_overlap"] == 0
                and values["eligible_raw_train_qid_overlap"] == 0
                and values["eligible_training_ledger_qid_overlap"] == 0
                and values["eligible_training_ledger_family_overlap"] == 0
                for values in scope_a["by_dataset"].values()
            ),
            "individual_question_identity_rows_emitted": False,
            "fresh_identity_rows_generated_or_frozen": False,
            "fresh_answer_free_rows_generated_or_frozen": False,
            "this_is_a_cohort_freeze": False,
        },
        "scientific_boundary": (
            "This audit measures identity capacity only. It opens raw sources that may "
            "contain Gold, uses only the declared identity projection, and does not select "
            "or freeze any v8 row. It does not authorize retrieval, inference, scoring, "
            "training, or use of the paper-confirmation reserve."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = output_dir / "inventory.json"
    report_path = output_dir / "report.json"
    inventory = {
        "historical_evaluation_protocol_registries": historical.inventory,
        "local_training_inputs": training.inventory,
        "training_input_manifest_config_evidence": training.evidence_inventory,
    }
    _write_json(inventory_path, inventory)
    _write_json(report_path, report)

    implementation_inventory = _implementation_inventory(project_root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": status,
        "python_version": platform.python_version(),
        "inputs": {
            "raw_source_inventory": raw_source_inventory,
            "historical_evaluation_protocol_registry_count": len(historical.inventory),
            "historical_evaluation_protocol_registry_set_sha256": _sha256_json(
                historical.inventory
            ),
            "local_training_input_count": len(training.inventory),
            "local_training_input_set_sha256": _sha256_json(training.inventory),
            "training_input_manifest_config_evidence_count": len(
                training.evidence_inventory
            ),
            "training_input_manifest_config_evidence_set_sha256": _sha256_json(
                training.evidence_inventory
            ),
        },
        "implementation_inventory": implementation_inventory,
        "outputs": [
            {"path": "inventory.json", "sha256": _sha256_file(inventory_path)},
            {"path": "report.json", "sha256": _sha256_file(report_path)},
        ],
        "source_inventory_artifact": {
            "path": "inventory.json",
            "sha256": _sha256_file(inventory_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project_root", type=Path, default=default_root)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", default=EXPERIMENT_ID)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or (project_root / "data")).resolve()
    report = run_capacity_audit(
        project_root=project_root,
        data_root=data_root,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        # Explicit on purpose: the CLI can never substitute runtime discovery.
        historical_registry_paths=None,
        training_input_specs=None,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(args.output_dir),
                "cohort_frozen": False,
            },
            ensure_ascii=False,
        )
    )
    if report["status"] in {FAIL_STATUS, TRAINING_LEDGER_BLOCKED_STATUS}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
