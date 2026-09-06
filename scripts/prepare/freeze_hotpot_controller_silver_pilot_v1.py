#!/usr/bin/env python
"""Freeze a HotpotQA train-side Controller silver-label pilot.

This freezer performs no query generation, retrieval, model inference, or
training.  It reads the immutable converted HotpotQA train file, applies a
strict support-chain screen, merges the frozen consumed-identity registries,
and writes exactly one identity-only pilot.  The selected identities are the
fixed denominator for a later, separately implemented silver q1/q2 generation
and review stage; a failed generation or review may not be replaced in this
pilot.

The structural screen deliberately reads train-side final-answer annotations.
They are used to reject boolean questions, orient the answer-bearing support
document, and rule out bridge/final-answer leakage from the original question.
Consequently this is a Gold-screened *silver-label engineering cohort*, not a
Gold-free evaluation cohort and not an exact decomposition Gold set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from kgproweight.data import hotpot_controller_silver as hotpot_silver  # noqa: E402
from scripts.prepare import audit_subquestion_v8_cohort_capacity as capacity  # noqa: E402
from scripts.prepare.freeze_qpeg_v1_protocol import (  # noqa: E402
    FAMILY_VERSION,
    family_sha256,
)


DATASET = "hotpotqa"
SOURCE_SPLIT = "train"
PILOT_SIZE = 30
PILOT_ACCEPTED_MIN = 24
LEVEL_ORDER = ("easy", "medium", "hard")
LEVEL_QUOTAS = {"easy": 10, "medium": 10, "hard": 10}
FULL_RELEASE_SIZES = {"train": 600, "dev": 60, "confirmation": 30}
FULL_RELEASE_ACCEPTED_MIN = sum(FULL_RELEASE_SIZES.values())
EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-SILVER-LABEL-COVERAGE-"
    "PILOT30-SEED20260904-V1"
)
SELECTION_SALT = EXPERIMENT_ID
SCHEMA_VERSION = "hotpot-controller-silver-pilot-freeze-1"
PROTOCOL_SCHEMA_VERSION = "hotpot-controller-silver-pilot-protocol-1"
MANIFEST_SCHEMA_VERSION = "hotpot-controller-silver-pilot-manifest-1"
STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_SILVER_PILOT30_NOT_GENERATED_NOT_TRAINED"
TEST_STATUS = "COMPLETE_TEST_SIZED_IDENTITY_ONLY_SILVER_PILOT_NOT_FORMAL"
OUTPUT_ROW_FIELDS = ("dataset", "qid", "question")

RAW_TRAIN_PATH = Path("data/hotpotqa/train.jsonl")
EXPECTED_RAW_TRAIN_SHA256 = (
    "47444e1f8ccfd9c5f4001cc1252f99abbb0e07edc770bba7daac06d1cc17a9f6"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_hotpot_silver_label_coverage_"
    "pilot30_seed20260904_v1"
)

CAPACITY_INVENTORY_PATH = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1/"
    "inventory.json"
)
EXPECTED_CAPACITY_INVENTORY_SHA256 = (
    "5f1ea159bd2eeaff2fa185c20f5106f5c133f5740d0d04e802a03d9d77cff696"
)

SEALED_PARENT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
    "dev30_prospective300_seed20260904_v1"
)
SEALED_PARENT_REPORT_PATH = SEALED_PARENT_DIR / "report.json"
SEALED_PARENT_MANIFEST_PATH = SEALED_PARENT_DIR / "manifest.json"
SEALED_PROSPECTIVE_FILENAME = "prospective.identity_only.jsonl"
EXPECTED_SEALED_PARENT_HASHES = {
    "report": "233b931716d96e0a6e40e0cb2c0e961a5c79c04884d6cac584c301e9ce9fe4b7",
    "manifest": "cda6525e1562697c31e17cb457280fe272de039ebebee23a2ddcabaa942730e6",
}

# Post-capacity-audit consumed/protected identity registries.  The v7 identity
# source is already one of the frozen 58 historical registries, but keeping it
# explicit makes the Phase-0 lineage visible.  Query Controller v4.4 has no
# Hotpot rows; it is included so this union remains suitable as the parent of a
# future append-only three-dataset release.
EXPLICIT_CONSUMED_IDENTITY_PATHS: tuple[Path, ...] = (
    Path(
        "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
        "dev30_prospective300_seed20260904_v1/development.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_"
        "seed20260904_v1/smoke.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
        "pilot30x3_seed20260904_v1/pilot.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_development_"
        "preregistration/development.question_only.jsonl"
    ),
    Path(
        "outputs/audits/query_controller_v1_exact_text_pilot_seed42_"
        "protocol_v4_4/train.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/query_controller_v1_exact_text_pilot_seed42_"
        "protocol_v4_4/dev.identity_only.jsonl"
    ),
    Path(
        "outputs/audits/query_controller_v1_exact_text_pilot_seed42_"
        "protocol_v4_4/confirmation.identity_only.jsonl"
    ),
)
EXPECTED_EXPLICIT_CONSUMED_SHA256 = {
    EXPLICIT_CONSUMED_IDENTITY_PATHS[0]: (
        "dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[1]: (
        "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[2]: (
        "7f4c63eb5589ce342a59e9942d986992ce357eb2888a46a0d6b04dc38794a9d8"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[3]: (
        "7fd01236609ed010a42bb92d41a0e978323035ee715b1cc7a2047a1eebd2a8bc"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[4]: (
        "2a31e10a1d37e2090e9909fe05975fba3179c1f4301b8fe0a3e1c94192da2da3"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[5]: (
        "5c78577ccf5bea18e401f1863580871b09d09d1a0483adcdf1a9816172dc9a07"
    ),
    EXPLICIT_CONSUMED_IDENTITY_PATHS[6]: (
        "b88bbae2c0758e5f7ffc77ff4aefff4dae6b822c967ba77108229d38b5e4b46b"
    ),
}

# These thresholds are normative and are serialized into the protocol before
# any future q1/q2 producer is allowed to run.
STRUCTURAL_THRESHOLDS: dict[str, Any] = {
    "authoritative_extractor": (
        "kgproweight.data.hotpot_controller_silver.extract_hotpot_support_chain"
    ),
    "authoritative_extractor_version": hotpot_silver.BUILDER_VERSION,
    "metadata_type_exact": "bridge",
    "metadata_level_allowlist_exact": list(LEVEL_ORDER),
    "supporting_fact_entries_exact": 2,
    "distinct_support_titles_exact": 2,
    "support_titles_must_not_overlap_or_alias": True,
    "question_support_title_hits_exact": 1,
    "question_hit_title_role": "root",
    "question_bridge_title_hits_max": 0,
    "context_title_matches_per_support_title_exact": 1,
    "support_pointer_duplicates_max": 0,
    "root_and_bridge_support_documents_nonempty": True,
    "root_support_sentences_binding_bridge_surface_exact": 1,
    "bridge_surface_occurrences_in_bound_root_sentence_exact": 1,
    "bridge_support_sentences_containing_root_title_max": 0,
    "final_answers_required": True,
    "forbidden_final_answers_exact": ["yes", "no", "unknown", "none"],
    "final_answer_surface_characters_min": 2,
    "final_alias_in_question_support_titles_or_root_support_max": 0,
    "final_alias_equivalent_to_root_bridge_or_intermediate_max": 0,
    "bridge_support_sentences_containing_final_surface_exact": 1,
    "second_hop_support_sentences_led_by_final_alias_max": 0,
    "matching_contract": (
        "authoritative extractor NFKC+casefold word-boundary surfaces; support-title "
        "aliases include one trailing-parenthetical-stripped form; identity-alias "
        "comparison additionally NFKD-folds diacritics; final-secret screen also "
        "catches conservative ordered-token subsequences"
    ),
}

FUTURE_SILVER_GENERATION_GATES: dict[str, Any] = {
    "authorization_granted_by_this_freeze": False,
    "producer_identity_and_prompt_hash_required_before_generation": True,
    "producer_candidates_per_identity_per_slot_exact": 1,
    "best_of_or_manual_rewrite_allowed": False,
    "source_action_exact": "text",
    "pid_exact": None,
    "q1_bridge_surface_mentions_max": 0,
    "q1_final_answer_surface_mentions_max": 0,
    "q2_final_answer_surface_mentions_max": 0,
    "support_binding_rate_required_for_accepted": 1.0,
    "hotpot_companion_action_schema_rate_required_for_accepted": 1.0,
    "query_nonrepeat_rate_required_for_accepted": 1.0,
    "placeholder_free_rate_required_for_accepted": 1.0,
    "dependency_closed_rate_required_for_accepted": 1.0,
    "q1_answerable_from_first_hop_support_rate_required_for_accepted": 1.0,
    "q2_answerable_from_second_hop_support_rate_required_for_accepted": 1.0,
    "retrieval_top_k": 10,
    "q1_first_hop_support_and_bridge_retrieval_rate_required_for_accepted": 1.0,
    "q2_second_hop_support_and_final_retrieval_rate_required_for_accepted": 1.0,
    "training_q2_observation_source_exact": (
        "train_annotation_intermediate_bound_to_first_hop_support"
    ),
    "runtime_q2_observation_source_exact": (
        "strong_sft_reader_prediction_bound_to_retrieved_passage"
    ),
    "gold_bridge_injected_as_runtime_observation_allowed": False,
    "context_isolated_ai_review_calls_exact": 2,
    "statistically_independent_reviewer_claim_allowed": False,
    "ai_unanimous_all_fields_pass_rate_required_for_accepted": 1.0,
    "ai_disagreement_or_unknown_policy": "reject_no_repair",
    "pilot_fixed_denominator": PILOT_SIZE,
    "pilot_accepted_min": PILOT_ACCEPTED_MIN,
    "pilot_accepted_rate_min": PILOT_ACCEPTED_MIN / PILOT_SIZE,
    "pilot_failed_identity_replacement_allowed": False,
    "pilot_to_full_release_rule": (
        "advance only if accepted>=24/30, every accepted item passes every mechanical, "
        "support, retrieval, and unanimous-AI gate, and identity/family/Gold boundaries "
        "have zero violations"
    ),
    "future_release_source_split_exact": SOURCE_SPLIT,
    "future_release_sizes": dict(FULL_RELEASE_SIZES),
    "future_release_fixed_denominator_total": FULL_RELEASE_ACCEPTED_MIN,
    "future_release_accepted_unique_families_min": FULL_RELEASE_ACCEPTED_MIN,
    "future_release_cross_role_family_overlap_max": 0,
    "future_release_consumed_qid_and_family_overlap_max": 0,
    "future_release_all_mechanical_support_retrieval_ai_gates_rate_required": 1.0,
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    qid: str
    question: str
    question_sha256: str
    family_sha256: str
    level: str


@dataclass(frozen=True)
class RawProjection:
    candidates: Mapping[str, Candidate]
    identities: Mapping[str, Candidate]
    question_hash_to_qids: Mapping[str, frozenset[str]]
    source_identity: Mapping[str, Any]
    funnel: Mapping[str, Any]


@dataclass(frozen=True)
class ConsumedProjection:
    qids: frozenset[tuple[str, str]]
    families: frozenset[tuple[str, str]]
    inventory: tuple[Mapping[str, Any], ...]
    stats: Mapping[str, Any]


class CandidateReject(ValueError):
    """Expected, counted rejection under the frozen structural rules."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_SPACE = re.compile(r"\s+")
def _clean(value: object) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _sha256_file(path: Path) -> str:
    if path.name == SEALED_PROSPECTIVE_FILENAME:
        raise PermissionError("sealed prospective identity content must not be opened or hashed")
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


def _load_json(path: Path) -> dict[str, Any]:
    if path.name == SEALED_PROSPECTIVE_FILENAME:
        raise PermissionError("sealed prospective identity content must not be opened")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_row(raw_line: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
    return value


def _resolve(project_root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _raw_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    raw_id = _clean(row.get("id"))
    raw_qid = _clean(row.get("qid"))
    if raw_id and raw_qid and raw_id != raw_qid:
        raise ValueError("raw Hotpot row has conflicting id and qid")
    qid = raw_id or raw_qid
    question = _clean(row.get("question"))
    if not qid or not question:
        raise ValueError("raw Hotpot row lacks id/qid or question")
    return qid, question


def _raw_level(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return _clean(metadata.get("level")).casefold()


def _strict_candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    """Project exactly the authoritative strict Hotpot support-chain pool."""

    try:
        chain = hotpot_silver.extract_hotpot_support_chain(row)
    except hotpot_silver.HotpotSilverReject as exc:
        raise CandidateReject(exc.code) from exc
    level = _raw_level(row)
    if level not in LEVEL_ORDER:
        raise CandidateReject("metadata_level_not_allowed")

    return Candidate(
        dataset=DATASET,
        qid=chain.qid,
        question=chain.question,
        question_sha256=question_sha256(chain.question),
        family_sha256=family_sha256(chain.question),
        level=level,
    )


def _project_raw_train(
    path: Path, *, expected_sha256: str | None
) -> RawProjection:
    if path.name == SEALED_PROSPECTIVE_FILENAME:
        raise PermissionError("sealed prospective file cannot be a raw-train source")
    digest = hashlib.sha256()
    identities: dict[str, Candidate] = {}
    candidates: dict[str, Candidate] = {}
    question_hash_to_qids: dict[str, set[str]] = defaultdict(set)
    rejection_reasons: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                counts["blank_rows"] += 1
                continue
            counts["raw_rows"] += 1
            row = _json_row(raw_line, path=path, line_number=line_number)
            qid, question = _raw_identity(row)
            identity = Candidate(
                dataset=DATASET,
                qid=qid,
                question=question,
                question_sha256=question_sha256(question),
                family_sha256=family_sha256(question),
                level=_raw_level(row),
            )
            if qid in identities:
                raise ValueError(f"duplicate raw Hotpot qid: {qid}")
            identities[qid] = identity
            question_hash_to_qids[identity.question_sha256].add(qid)
            try:
                candidate = _strict_candidate_from_row(row)
            except CandidateReject as exc:
                rejection_reasons[exc.reason] += 1
                continue
            candidates[qid] = candidate
            counts["strict_candidates_before_consumed_exclusion"] += 1
            counts[f"strict_candidates_level::{candidate.level}"] += 1
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("raw Hotpot train SHA256 drift")
    return RawProjection(
        candidates=candidates,
        identities=identities,
        question_hash_to_qids={
            key: frozenset(values) for key, values in question_hash_to_qids.items()
        },
        source_identity={
            "path": str(path),
            "sha256": actual_sha256,
            "rows": counts["raw_rows"],
        },
        funnel={
            "raw_rows": counts["raw_rows"],
            "blank_rows": counts["blank_rows"],
            "strict_candidates_before_consumed_exclusion": counts[
                "strict_candidates_before_consumed_exclusion"
            ],
            "strict_candidates_by_level": {
                level: counts[f"strict_candidates_level::{level}"]
                for level in LEVEL_ORDER
            },
            "rejection_reasons_first_failed_gate": dict(sorted(rejection_reasons.items())),
        },
    )


def _capacity_source_hashes(
    *,
    project_root: Path,
    inventory_path: Path | None,
    expected_inventory_sha256: str | None,
    historical_paths: Sequence[str],
    training_specs: Sequence[capacity.TrainingInputSpec],
    enforce_exact_inventory: bool,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if inventory_path is None:
        return {}, None
    path = _resolve(project_root, inventory_path)
    digest = _sha256_file(path)
    if expected_inventory_sha256 is not None and digest != expected_inventory_sha256:
        raise ValueError("capacity consumed-inventory SHA256 drift")
    value = _load_json(path)
    historical = value.get("historical_evaluation_protocol_registries")
    training = value.get("local_training_inputs")
    if not isinstance(historical, list) or not isinstance(training, list):
        raise ValueError("capacity inventory lacks historical/training source arrays")
    if any(not isinstance(item, Mapping) for item in [*historical, *training]):
        raise ValueError("capacity inventory source entry is not an object")
    historical_map = {str(item.get("path")): str(item.get("sha256")) for item in historical}
    training_map = {str(item.get("path")): str(item.get("sha256")) for item in training}
    if enforce_exact_inventory:
        if tuple(historical_map) != tuple(historical_paths):
            raise ValueError("frozen historical registry path inventory drift")
        if tuple(training_map) != tuple(spec.path for spec in training_specs):
            raise ValueError("frozen training-input path inventory drift")
    source_hashes = {**historical_map, **training_map}
    if len(source_hashes) != len(historical_map) + len(training_map):
        raise ValueError("capacity inventory contains duplicate source paths")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in source_hashes.values()):
        raise ValueError("capacity inventory contains malformed source SHA256")
    return source_hashes, {
        "path": _display_path(path, project_root),
        "sha256": digest,
        "historical_registry_paths": len(historical_map),
        "training_input_paths": len(training_map),
    }


def _historical_qid(row: Mapping[str, Any]) -> str:
    for field in ("qid", "id", "source_id"):
        value = _clean(row.get(field))
        if value:
            return value
    return ""


def _read_consumed_source(
    *,
    path: Path,
    project_root: Path,
    kind: str,
    raw: RawProjection,
    expected_sha256: str | None,
    dataset_hint: str | None = None,
    qid_alias: str = "qid",
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, Any]]:
    if path.name == SEALED_PROSPECTIVE_FILENAME:
        raise PermissionError("sealed prospective file cannot enter consumed union")
    digest = hashlib.sha256()
    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                counts["blank_rows"] += 1
                continue
            counts["rows"] += 1
            row = _json_row(raw_line, path=path, line_number=line_number)
            row_dataset = _clean(row.get("dataset")).casefold()
            dataset = row_dataset or _clean(dataset_hint).casefold()
            if dataset != DATASET:
                continue
            question = _clean(row.get("question"))
            if not question:
                raise ValueError(f"Hotpot consumed row lacks question: {path}:{line_number}")
            if kind == "training_input":
                source_qid = capacity._training_qid(row, qid_alias)
                matches = raw.question_hash_to_qids.get(question_sha256(question), frozenset())
                if not matches:
                    raise ValueError(
                        "Hotpot training consumed identity is outside the frozen raw train: "
                        f"{path}:{line_number}:{source_qid}"
                    )
                qids.update((DATASET, qid) for qid in matches)
            else:
                source_qid = _historical_qid(row)
                if not source_qid:
                    raise ValueError(
                        f"Hotpot consumed row lacks qid/id/source_id: {path}:{line_number}"
                    )
                qids.add((DATASET, source_qid))
            families.add((DATASET, family_sha256(question)))
            counts["hotpot_rows"] += 1
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"consumed source SHA256 drift: {path}")
    return qids, families, {
        "kind": kind,
        "path": _display_path(path, project_root),
        "sha256": actual_sha256,
        "rows": counts["rows"],
        "hotpot_rows": counts["hotpot_rows"],
        "hotpot_unique_qids_contributed": len(qids),
        "hotpot_unique_families_contributed": len(families),
    }


def _merge_consumed_union(
    *,
    project_root: Path,
    raw: RawProjection,
    historical_paths: Sequence[str],
    training_specs: Sequence[capacity.TrainingInputSpec],
    explicit_paths: Sequence[Path],
    capacity_source_hashes: Mapping[str, str],
    expected_explicit_hashes: Mapping[Path, str] | None,
) -> ConsumedProjection:
    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    inventory: list[Mapping[str, Any]] = []

    for relative in historical_paths:
        path = _resolve(project_root, relative)
        source_qids, source_families, source_inventory = _read_consumed_source(
            path=path,
            project_root=project_root,
            kind="historical_registry",
            raw=raw,
            expected_sha256=capacity_source_hashes.get(relative),
        )
        qids.update(source_qids)
        families.update(source_families)
        inventory.append(source_inventory)

    for spec in training_specs:
        path = _resolve(project_root, spec.path)
        source_qids, source_families, source_inventory = _read_consumed_source(
            path=path,
            project_root=project_root,
            kind="training_input",
            raw=raw,
            expected_sha256=capacity_source_hashes.get(spec.path),
            dataset_hint=spec.dataset_hint,
            qid_alias=spec.qid_alias,
        )
        qids.update(source_qids)
        families.update(source_families)
        inventory.append(source_inventory)

    for relative in explicit_paths:
        path = _resolve(project_root, relative)
        expected = None
        if expected_explicit_hashes is not None:
            if relative not in expected_explicit_hashes:
                raise ValueError(f"missing explicit consumed SHA256 lock: {relative}")
            expected = expected_explicit_hashes[relative]
        source_qids, source_families, source_inventory = _read_consumed_source(
            path=path,
            project_root=project_root,
            kind="post_capacity_explicit_registry",
            raw=raw,
            expected_sha256=expected,
        )
        qids.update(source_qids)
        families.update(source_families)
        inventory.append(source_inventory)

    return ConsumedProjection(
        qids=frozenset(qids),
        families=frozenset(families),
        inventory=tuple(inventory),
        stats={
            "source_inventory_entries": len(inventory),
            "unique_source_files": len(
                {str(item.get("path")) for item in inventory}
            ),
            "historical_registry_files": len(historical_paths),
            "training_input_files": len(training_specs),
            "post_capacity_explicit_registry_files": len(explicit_paths),
            "unique_hotpot_qids": len(qids),
            "unique_hotpot_families": len(families),
            "complete_historical_training_ledger_available": False,
            "missing_old_checkpoint_input_ledgers": "UNKNOWN",
        },
    )


def _validate_sealed_parent_metadata(
    *,
    project_root: Path,
    report_path: Path,
    manifest_path: Path,
    expected_hashes: Mapping[str, str] | None,
) -> dict[str, Any]:
    report_path = _resolve(project_root, report_path)
    manifest_path = _resolve(project_root, manifest_path)
    report_sha256 = _sha256_file(report_path)
    manifest_sha256 = _sha256_file(manifest_path)
    if expected_hashes is not None:
        if report_sha256 != expected_hashes.get("report"):
            raise ValueError("sealed-parent report SHA256 drift")
        if manifest_sha256 != expected_hashes.get("manifest"):
            raise ValueError("sealed-parent manifest SHA256 drift")
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    expected_status = "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE"
    if report.get("status") != expected_status or manifest.get("status") != expected_status:
        raise ValueError("sealed-parent terminal status drift")
    checks = report.get("checks") or {}
    seal = report.get("prospective_seal") or {}
    if (
        checks.get("all_freeze_gates_pass") is not True
        or checks.get("raw_train_qid_overlap") != 0
        or checks.get("raw_train_family_overlap") != 0
    ):
        raise ValueError("sealed parent does not prove raw-train disjointness")
    if seal.get("status") != "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT":
        raise ValueError("sealed prospective status drift")
    outputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
    }
    if outputs.get("report.json") != report_sha256:
        raise ValueError("sealed-parent manifest does not bind report.json")
    prospective_sha256 = outputs.get(SEALED_PROSPECTIVE_FILENAME, "")
    if re.fullmatch(r"[0-9a-f]{64}", prospective_sha256) is None:
        raise ValueError("sealed-parent manifest lacks prospective declared SHA256")
    return {
        "report_path": _display_path(report_path, project_root),
        "report_sha256": report_sha256,
        "manifest_path": _display_path(manifest_path, project_root),
        "manifest_sha256": manifest_sha256,
        "status": expected_status,
        "prospective_relative_path_declared_only": _display_path(
            manifest_path.parent / SEALED_PROSPECTIVE_FILENAME,
            project_root,
        ),
        "prospective_declared_sha256_not_rehashed": prospective_sha256,
        "prospective_content_opened": False,
        "prospective_content_hashed": False,
        "raw_train_qid_overlap_reported": 0,
        "raw_train_family_overlap_reported": 0,
    }


def select_pilot(
    *,
    candidates: Mapping[str, Candidate],
    consumed: ConsumedProjection,
    salt: str,
    n: int,
    level_quotas: Mapping[str, int],
) -> tuple[list[Candidate], dict[str, Any]]:
    if not _clean(salt) or type(n) is not int or n <= 0:
        raise ValueError("selection salt must be nonempty and n must be a positive integer")
    if (
        set(level_quotas) != set(LEVEL_ORDER)
        or any(type(level_quotas[level]) is not int or level_quotas[level] < 0 for level in LEVEL_ORDER)
        or sum(level_quotas.values()) != n
    ):
        raise ValueError("level quotas must cover easy/medium/hard exactly and sum to n")
    eligible: list[Candidate] = []
    qid_hits = 0
    family_hits = 0
    for candidate in candidates.values():
        qid_hit = (DATASET, candidate.qid) in consumed.qids
        family_hit = (DATASET, candidate.family_sha256) in consumed.families
        qid_hits += int(qid_hit)
        family_hits += int(family_hit)
        if not qid_hit and not family_hit:
            eligible.append(candidate)
    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in eligible:
        by_family[candidate.family_sha256].append(candidate)
    if len(by_family) < n:
        raise ValueError(
            f"Hotpot strict eligible family capacity {len(by_family)} is below pilot size {n}"
        )
    by_level_family: dict[str, dict[str, list[Candidate]]] = {
        level: defaultdict(list) for level in LEVEL_ORDER
    }
    for candidate in eligible:
        if candidate.level not in by_level_family:
            raise ValueError(f"candidate has non-frozen Hotpot level: {candidate.level!r}")
        by_level_family[candidate.level][candidate.family_sha256].append(candidate)

    selected: list[Candidate] = []
    selected_families: set[str] = set()
    selected_by_level: Counter[str] = Counter()
    for level in LEVEL_ORDER:
        if level_quotas[level] == 0:
            continue
        ordered_families = sorted(
            by_level_family[level],
            key=lambda family: hashlib.sha256(
                f"{salt}\0{DATASET}\0{level}\0{family}".encode("utf-8")
            ).hexdigest(),
        )
        for family in ordered_families:
            if family in selected_families:
                continue
            selected.append(
                min(
                    by_level_family[level][family],
                    key=lambda candidate: (candidate.question_sha256, candidate.qid),
                )
            )
            selected_families.add(family)
            selected_by_level[level] += 1
            if selected_by_level[level] == level_quotas[level]:
                break
        if selected_by_level[level] != level_quotas[level]:
            raise ValueError(
                "Hotpot strict eligible family capacity after cross-level disjointness "
                f"is below quota for {level}: "
                f"{selected_by_level[level]} < {level_quotas[level]}"
            )
    return selected, {
        "strict_candidate_qids_before_consumed_exclusion": len(candidates),
        "consumed_qid_hits_not_mutually_exclusive": qid_hits,
        "consumed_family_hits_not_mutually_exclusive": family_hits,
        "eligible_qids_after_consumed_exclusion": len(eligible),
        "eligible_unique_dataset_scoped_families": len(by_family),
        "eligible_unique_families_by_level_before_cross_level_selection": {
            level: len(by_level_family[level]) for level in LEVEL_ORDER
        },
        "level_quotas": {level: level_quotas[level] for level in LEVEL_ORDER},
        "selected_rows_by_level": {
            level: selected_by_level[level] for level in LEVEL_ORDER
        },
        "selected_rows": len(selected),
        "remaining_unique_family_capacity_after_pilot": len(by_family) - n,
    }


def _public_rows(selected: Sequence[Candidate]) -> list[dict[str, str]]:
    return [
        {"dataset": candidate.dataset, "qid": candidate.qid, "question": candidate.question}
        for candidate in selected
    ]


def _jsonl_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if tuple(row) != OUTPUT_ROW_FIELDS or set(row) != set(OUTPUT_ROW_FIELDS):
            raise ValueError("pilot identity row violates exact field/order allowlist")
        digest.update(
            (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if tuple(row) != OUTPUT_ROW_FIELDS or set(row) != set(OUTPUT_ROW_FIELDS):
                raise ValueError("pilot identity row violates exact field/order allowlist")
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _implementation_inventory(project_root: Path) -> list[dict[str, str]]:
    paths = {
        Path(__file__).resolve(),
        Path(hotpot_silver.__file__).resolve(),
        Path(capacity.__file__).resolve(),
        Path(question_sha256.__code__.co_filename).resolve(),
        Path(family_sha256.__code__.co_filename).resolve(),
    }
    return [
        {"path": _display_path(path, project_root), "sha256": _sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]


def run_freeze(
    *,
    project_root: Path = PROJECT_ROOT,
    raw_train_path: Path = RAW_TRAIN_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiment_id: str = EXPERIMENT_ID,
    selection_salt: str = SELECTION_SALT,
    pilot_size: int = PILOT_SIZE,
    level_quotas: Mapping[str, int] = LEVEL_QUOTAS,
    historical_registry_paths: Sequence[str] = capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS,
    training_input_specs: Sequence[capacity.TrainingInputSpec] = capacity.LOCAL_TRAINING_INPUT_SPECS,
    explicit_consumed_identity_paths: Sequence[Path] = EXPLICIT_CONSUMED_IDENTITY_PATHS,
    capacity_inventory_path: Path | None = CAPACITY_INVENTORY_PATH,
    expected_capacity_inventory_sha256: str | None = EXPECTED_CAPACITY_INVENTORY_SHA256,
    expected_explicit_consumed_hashes: Mapping[Path, str] | None = EXPECTED_EXPLICIT_CONSUMED_SHA256,
    expected_raw_train_sha256: str | None = EXPECTED_RAW_TRAIN_SHA256,
    sealed_parent_report_path: Path = SEALED_PARENT_REPORT_PATH,
    sealed_parent_manifest_path: Path = SEALED_PARENT_MANIFEST_PATH,
    expected_sealed_parent_hashes: Mapping[str, str] | None = EXPECTED_SEALED_PARENT_HASHES,
    enforce_formal_locks: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Freeze the pilot and return its in-memory protocol/report/manifest."""

    project_root = Path(project_root).resolve()
    raw_train_path = _resolve(project_root, raw_train_path)
    output_dir = _resolve(project_root, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite append-only pilot: {output_dir}")
    if enforce_formal_locks and (
        experiment_id != EXPERIMENT_ID
        or selection_salt != SELECTION_SALT
        or pilot_size != PILOT_SIZE
        or dict(level_quotas) != LEVEL_QUOTAS
        or raw_train_path != _resolve(project_root, RAW_TRAIN_PATH)
        or output_dir != _resolve(project_root, DEFAULT_OUTPUT_DIR)
        or tuple(historical_registry_paths)
        != tuple(capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS)
        or tuple(training_input_specs) != tuple(capacity.LOCAL_TRAINING_INPUT_SPECS)
        or tuple(explicit_consumed_identity_paths) != EXPLICIT_CONSUMED_IDENTITY_PATHS
        or capacity_inventory_path != CAPACITY_INVENTORY_PATH
        or expected_capacity_inventory_sha256 != EXPECTED_CAPACITY_INVENTORY_SHA256
        or dict(expected_explicit_consumed_hashes or {})
        != EXPECTED_EXPLICIT_CONSUMED_SHA256
        or expected_raw_train_sha256 != EXPECTED_RAW_TRAIN_SHA256
        or sealed_parent_report_path != SEALED_PARENT_REPORT_PATH
        or sealed_parent_manifest_path != SEALED_PARENT_MANIFEST_PATH
        or dict(expected_sealed_parent_hashes or {}) != EXPECTED_SEALED_PARENT_HASHES
    ):
        raise ValueError(
            "formal pilot identity, sources, hashes, thresholds, and output path are immutable"
        )

    sealed_parent = _validate_sealed_parent_metadata(
        project_root=project_root,
        report_path=sealed_parent_report_path,
        manifest_path=sealed_parent_manifest_path,
        expected_hashes=expected_sealed_parent_hashes,
    )
    raw = _project_raw_train(raw_train_path, expected_sha256=expected_raw_train_sha256)
    capacity_hashes, capacity_inventory = _capacity_source_hashes(
        project_root=project_root,
        inventory_path=capacity_inventory_path,
        expected_inventory_sha256=expected_capacity_inventory_sha256,
        historical_paths=historical_registry_paths,
        training_specs=training_input_specs,
        enforce_exact_inventory=enforce_formal_locks,
    )
    consumed = _merge_consumed_union(
        project_root=project_root,
        raw=raw,
        historical_paths=historical_registry_paths,
        training_specs=training_input_specs,
        explicit_paths=explicit_consumed_identity_paths,
        capacity_source_hashes=capacity_hashes,
        expected_explicit_hashes=expected_explicit_consumed_hashes,
    )
    selected, selection = select_pilot(
        candidates=raw.candidates,
        consumed=consumed,
        salt=selection_salt,
        n=pilot_size,
        level_quotas=level_quotas,
    )
    rows = _public_rows(selected)
    selected_qids = {(row.dataset, row.qid) for row in selected}
    selected_families = {(row.dataset, row.family_sha256) for row in selected}
    checks = {
        "exact_pilot_rows": len(selected) == pilot_size,
        "unique_dataset_scoped_qids": len(selected_qids) == len(selected),
        "one_qid_per_dataset_scoped_family": len(selected_families) == len(selected),
        "exact_level_quotas": Counter(row.level for row in selected)
        == Counter(level_quotas),
        "consumed_qid_overlap": len(selected_qids & set(consumed.qids)),
        "consumed_family_overlap": len(selected_families & set(consumed.families)),
        "all_selected_are_strict_candidates": all(
            row.qid in raw.candidates and raw.candidates[row.qid] == row for row in selected
        ),
        "output_rows_exact_field_allowlist": all(
            tuple(row) == OUTPUT_ROW_FIELDS and set(row) == set(OUTPUT_ROW_FIELDS)
            for row in rows
        ),
        "q1_q2_records_generated": 0,
        "retrieval_calls": 0,
        "model_calls": 0,
        "sealed_prospective_content_opened": False,
        "sealed_prospective_content_hashed": False,
        "existing_v4_4_modified": False,
    }
    checks["all_freeze_gates_pass"] = bool(
        checks["exact_pilot_rows"]
        and checks["unique_dataset_scoped_qids"]
        and checks["one_qid_per_dataset_scoped_family"]
        and checks["exact_level_quotas"]
        and checks["consumed_qid_overlap"] == 0
        and checks["consumed_family_overlap"] == 0
        and checks["all_selected_are_strict_candidates"]
        and checks["output_rows_exact_field_allowlist"]
        and checks["q1_q2_records_generated"] == 0
        and checks["retrieval_calls"] == 0
        and checks["model_calls"] == 0
    )
    if not checks["all_freeze_gates_pass"]:
        raise ValueError("Hotpot silver pilot failed one or more identity freeze gates")

    generated_at = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    canonical = bool(
        enforce_formal_locks
        and experiment_id == EXPERIMENT_ID
        and selection_salt == SELECTION_SALT
        and pilot_size == PILOT_SIZE
        and dict(level_quotas) == LEVEL_QUOTAS
    )
    terminal_status = STATUS if canonical else TEST_STATUS
    cohort_sha256 = _jsonl_rows_sha256(rows)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": "FROZEN_BEFORE_ANY_SILVER_Q1_Q2_GENERATION",
        "scope": "HOTPOTQA_RAW_TRAIN_SIDE_SILVER_LABEL_COVERAGE_PILOT_ONLY",
        "cohort": {
            "dataset": DATASET,
            "source_split": SOURCE_SPLIT,
            "rows": pilot_size,
            "fixed_denominator_no_replacement": True,
            "identity_fields_exact": list(OUTPUT_ROW_FIELDS),
            "identity_sha256": cohort_sha256,
            "family_version": FAMILY_VERSION,
            "family_scope": "dataset-scoped lexical-family proxy",
            "level_order": list(LEVEL_ORDER),
            "level_quotas": {level: level_quotas[level] for level in LEVEL_ORDER},
            "selection_salt": selection_salt,
            "selection_algorithm": (
                "in frozen easy/medium/hard order, take each level's required quota "
                "by sha256(salt\\0hotpotqa\\0level\\0family_sha256), skipping any "
                "family already selected by an earlier level, then minimum "
                "(question_sha256,qid) within the level-family"
            ),
        },
        "structural_candidate_thresholds_frozen_before_generation": dict(
            STRUCTURAL_THRESHOLDS
        ),
        "future_silver_generation_and_review_gates_frozen_before_generation": dict(
            FUTURE_SILVER_GENERATION_GATES
        ),
        "consumed_union": {
            "capacity_inventory": capacity_inventory,
            "source_inventory": list(consumed.inventory),
            "stats": dict(consumed.stats),
            "qid_and_family_exclusion_both_required": True,
            "dataset_scoped": True,
        },
        "gold_boundary": {
            "source_may_contain_gold": True,
            "final_answer_gold_accessed": True,
            "supporting_facts_gold_accessed": True,
            "supporting_facts_gold_uses": [
                "require exactly two supporting-fact pointers on two distinct titles",
                "join each support title and sentence index to one context sentence",
                "orient the root-to-bridge chain and bind first-hop and second-hop evidence",
            ],
            "candidate_selection_is_gold_screened": True,
            "final_answer_gold_uses": [
                "require nonempty answers and exclude yes/no/unknown/none or one-character aliases",
                "reject final-answer secret occurrence in the original question",
                "reject final-answer identity aliases of the root, bridge, or intermediate",
                "reject final-answer secret occurrence in either support title",
                "reject final-answer secret occurrence in the first-hop support sentence",
                "require exactly one second-hop support sentence containing a final surface",
                "reject a second-hop support sentence led by the final-answer surface",
            ],
            "final_answer_or_support_text_emitted": False,
            "gold_fields_in_identity_output": False,
            "label_status": "SILVER_CANDIDATE_IDENTITY_ONLY_NOT_YET_GENERATED",
            "gold_or_exact_decomposition_claim_allowed": False,
            "outcome_evaluation_use_allowed": False,
        },
        "freshness_scope": {
            "claim": "DISJOINT_FROM_ENUMERATED_CONSUMED_UNION_ONLY",
            "complete_historical_training_ledger_available": False,
            "missing_old_checkpoint_input_ledgers": "UNKNOWN",
            "global_never_seen_claim_allowed": False,
        },
        "sealed_prospective_parent_metadata_only": sealed_parent,
        "authorization": {
            "identity_freeze": True,
            "q1_q2_generation": False,
            "retrieval": False,
            "model_inference": False,
            "training": False,
            "em_f1_ihr": False,
            "modify_existing_v4_4": False,
        },
        "scientific_boundary": (
            "This append-only artifact freezes a Gold-screened HotpotQA train-side "
            "identity denominator for a later silver-label coverage audit. It is not "
            "Gold-free, does not contain q1/q2 labels, does not validate retrieval or "
            "QA utility, and does not authorize Controller training."
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": terminal_status,
        "canonical_formal_freeze": canonical,
        "raw_train": {
            **dict(raw.source_identity),
            "path": _display_path(raw_train_path, project_root),
        },
        "structural_candidate_funnel": dict(raw.funnel),
        "consumed_union": {
            "stats": dict(consumed.stats),
            "source_inventory_sha256": _sha256_json(list(consumed.inventory)),
        },
        "selection": selection,
        "cohort": {
            "path": "pilot.identity_only.jsonl",
            "sha256": cohort_sha256,
            "rows": len(rows),
        },
        "checks": checks,
        "gold_screening_disclosure": protocol["gold_boundary"],
        "generation_thresholds_frozen": True,
        "q1_q2_generated": False,
        "data_raw_modified": False,
        "scientific_boundary": protocol["scientific_boundary"],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = output_dir / "pilot.identity_only.jsonl"
    protocol_path = output_dir / "protocol.json"
    report_path = output_dir / "report.json"
    _write_jsonl(cohort_path, rows)
    _write_json(protocol_path, protocol)
    _write_json(report_path, report)
    if _sha256_file(cohort_path) != cohort_sha256:
        raise ValueError("written pilot identity hash differs from in-memory lock")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": terminal_status,
        "python_version": platform.python_version(),
        "inputs": {
            "raw_train": report["raw_train"],
            "capacity_consumed_inventory": capacity_inventory,
            "consumed_source_inventory": list(consumed.inventory),
            "sealed_prospective_parent_metadata_only": sealed_parent,
        },
        "implementation_inventory": _implementation_inventory(project_root),
        "outputs": [
            {"path": cohort_path.name, "sha256": _sha256_file(cohort_path)},
            {"path": protocol_path.name, "sha256": _sha256_file(protocol_path)},
            {"path": report_path.name, "sha256": _sha256_file(report_path)},
        ],
        "output_row_field_allowlist": list(OUTPUT_ROW_FIELDS),
        "gold_final_answer_accessed_for_screening": True,
        "gold_fields_emitted": False,
        "q1_q2_generated": False,
        "retrieval_calls": 0,
        "model_calls": 0,
        "training_started": False,
        "existing_v4_4_modified": False,
        "sealed_prospective_content_opened": False,
        "sealed_prospective_content_hashed": False,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {"protocol": protocol, "report": report, "manifest": manifest}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw_train", type=Path, default=RAW_TRAIN_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_freeze(
        project_root=args.project_root,
        raw_train_path=args.raw_train,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["report"]["status"],
                "output_dir": str(_resolve(args.project_root.resolve(), args.output_dir)),
                "pilot_rows": result["report"]["cohort"]["rows"],
                "strict_candidate_families": result["report"]["selection"][
                    "eligible_unique_dataset_scoped_families"
                ],
                "q1_q2_generated": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
