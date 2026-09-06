#!/usr/bin/env python
"""Freeze the fresh train-side canonical-subQA v9 pilot30x3 cohort.

This is an identity-only custodian export.  The source train JSONL files can
contain Gold fields, but this script projects only ``id``/``qid`` and
``question``.  The frozen cohort contains exactly ``dataset``, ``qid``, and
``question``.

The formal command is deliberately bound to the static 58-file historical
registry and 20-file local training-input ledger used by the v8 Scope-A audit.
It also excludes the consumed v8 development90 and engineering smoke12.  The
sealed v8 prospective900 is *not* opened or hashed: disjointness follows from
the locked v8 freeze report, which proves that the prospective cohort has zero
qid/family overlap with every corresponding raw-train split.
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
from typing import Any, Iterable, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.prepare import audit_subquestion_v8_cohort_capacity as capacity  # noqa: E402


DATASETS = capacity.DATASETS
PILOT_PER_DATASET = 30
OUTPUT_ROW_FIELDS = ("dataset", "qid", "question")
EXPERIMENT_ID = "SUBQUESTION-DECOMPOSITION-V9-CANONICAL-SUBQA-PILOT-SEED20260904-V1"
SELECTION_SALT = EXPERIMENT_ID
SCHEMA_VERSION = "canonical-subqa-v9-fresh-pilot-identity-freeze-1"
PROTOCOL_SCHEMA_VERSION = "canonical-subqa-v9-fresh-pilot-selection-protocol-1"
MANIFEST_SCHEMA_VERSION = "canonical-subqa-v9-fresh-pilot-identity-manifest-1"
STATUS = "COMPLETE_FROZEN_FRESH_TRAIN_SIDE_IDENTITY_ONLY_PILOT30X3_NOT_RUN"

CAPACITY_AUDIT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1"
)
V8_COHORT_FREEZE_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_"
    "seed20260904_v1"
)
V8_DEVELOPMENT_PATH = V8_COHORT_FREEZE_DIR / "development.identity_only.jsonl"
V8_SMOKE_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_"
    "seed20260904_v1"
)
V8_SMOKE_PATH = V8_SMOKE_DIR / "smoke.identity_only.jsonl"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
    "pilot30x3_seed20260904_v1"
)

# These are metadata locks only.  In particular, the prospective JSONL hash
# declared inside the v8 manifest is never recomputed by this script.
EXPECTED_PARENT_HASHES = {
    "capacity_inventory": "5f1ea159bd2eeaff2fa185c20f5106f5c133f5740d0d04e802a03d9d77cff696",
    "capacity_report": "2551f26b6c40658cd4f57e47886d26c9dd16c6f5ee90dfdc3402e8f2a647ee6f",
    "capacity_manifest": "52334cd13ffe958cf239d4af2759bdc69ca4c34502820258a6858220f67e871f",
    "v8_freeze_report": "233b931716d96e0a6e40e0cb2c0e961a5c79c04884d6cac584c301e9ce9fe4b7",
    "v8_freeze_manifest": "cda6525e1562697c31e17cb457280fe272de039ebebee23a2ddcabaa942730e6",
    "v8_development": "dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2",
    "v8_smoke_report": "2156f59aebaeca8e34fbf0b1e40d46f66d54aa8349aef0af46c0b6ff8baa683b",
    "v8_smoke_manifest": "7eed188667cf8f9436c002705d3d11e0181c95f13b4986320c5a16695ddbf30b",
    "v8_smoke": "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606",
}


@dataclass(frozen=True)
class Candidate:
    """Private selection record; hashes are never emitted in the cohort."""

    dataset: str
    qid: str
    question: str
    question_sha256: str
    family_sha256: str


@dataclass(frozen=True)
class IdentityProjection:
    qids: frozenset[tuple[str, str]]
    families: frozenset[tuple[str, str]]
    inventory: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if tuple(row) != OUTPUT_ROW_FIELDS or set(row) != set(OUTPUT_ROW_FIELDS):
                raise ValueError(f"identity row violates field allowlist: {sorted(row)}")
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _candidate_from_identity_fields(
    row: Mapping[str, Any], *, dataset: str
) -> Candidate | None:
    """Project only id/qid/question; no content or Gold field is accessed."""

    qid, conflict = capacity._raw_qid(row)
    question = capacity._clean(row.get("question"))
    if conflict or not qid or not question:
        return None
    return Candidate(
        dataset=dataset,
        qid=qid,
        question=question,
        question_sha256=capacity.question_sha256(question),
        family_sha256=capacity.family_sha256(question),
    )


def _load_train_candidates(
    path: Path,
    *,
    dataset: str,
    projection: capacity.RawSplitProjection,
) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = capacity._json_row(raw_line, path=path, line_number=line_number)
            candidate = _candidate_from_identity_fields(row, dataset=dataset)
            if candidate is None:
                continue
            expected = projection.identities.get(candidate.qid)
            if expected is None or (
                expected.question_sha256 != candidate.question_sha256
                or expected.family_sha256 != candidate.family_sha256
            ):
                raise ValueError(
                    f"train identity projection drift: {dataset}/{candidate.qid}"
                )
            previous = candidates.get(candidate.qid)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"conflicting duplicate train identity: {dataset}/{candidate.qid}"
                )
            candidates[candidate.qid] = candidate
    if set(candidates) != set(projection.identities):
        raise ValueError(f"identity-only train projection is incomplete for {dataset}")
    return candidates


def _load_identity_only(path: Path, *, project_root: Path) -> IdentityProjection:
    """Load an explicitly permitted consumed identity-only file."""

    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    qid_questions: dict[tuple[str, str], str] = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                counts["blank_rows"] += 1
                continue
            row = capacity._json_row(raw_line, path=path, line_number=line_number)
            if set(row) != set(OUTPUT_ROW_FIELDS):
                raise ValueError(f"consumed identity file is not identity-only: {path}")
            dataset = capacity._clean(row.get("dataset")).lower()
            qid = capacity._clean(row.get("qid"))
            question = capacity._clean(row.get("question"))
            if dataset not in DATASETS or not qid or not question:
                raise ValueError(f"invalid consumed identity at {path}:{line_number}")
            key = (dataset, qid)
            qhash = capacity.question_sha256(question)
            previous = qid_questions.get(key)
            if previous is not None and previous != qhash:
                raise ValueError(f"conflicting consumed identity at {path}:{line_number}")
            qid_questions[key] = qhash
            qids.add(key)
            families.add((dataset, capacity.family_sha256(question)))
            counts[dataset] += 1
            counts["rows"] += 1
    return IdentityProjection(
        qids=frozenset(qids),
        families=frozenset(families),
        inventory={
            "path": _display_path(path, project_root),
            "sha256": digest.hexdigest(),
            "rows": counts["rows"],
            "per_dataset": {dataset: counts[dataset] for dataset in DATASETS},
        },
    )


def validate_v8_parent_metadata(
    *,
    project_root: Path,
    expected_report_sha256: str | None = EXPECTED_PARENT_HASHES["v8_freeze_report"],
    expected_manifest_sha256: str | None = EXPECTED_PARENT_HASHES["v8_freeze_manifest"],
) -> dict[str, Any]:
    """Validate v8 report/manifest only, never the prospective cohort bytes."""

    parent_dir = project_root / V8_COHORT_FREEZE_DIR
    report_path = parent_dir / "report.json"
    manifest_path = parent_dir / "manifest.json"
    report_sha = _sha256_file(report_path)
    manifest_sha = _sha256_file(manifest_path)
    if expected_report_sha256 is not None and report_sha != expected_report_sha256:
        raise ValueError("v8 cohort-freeze report metadata lock drift")
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("v8 cohort-freeze manifest metadata lock drift")
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    checks = report.get("checks", {})
    seal = report.get("prospective_seal", {})
    if report.get("status") != "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE":
        raise ValueError("v8 cohort parent does not have the expected complete status")
    if checks.get("all_freeze_gates_pass") is not True:
        raise ValueError("v8 cohort parent did not pass all freeze gates")
    if checks.get("raw_train_qid_overlap") != 0:
        raise ValueError("v8 parent does not prove prospective/raw-train qid disjointness")
    if checks.get("raw_train_family_overlap") != 0:
        raise ValueError("v8 parent does not prove prospective/raw-train family disjointness")
    if seal.get("status") != "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT":
        raise ValueError("v8 prospective seal status drift")
    if manifest.get("status") != report.get("status"):
        raise ValueError("v8 cohort report/manifest status mismatch")
    outputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
    }
    if outputs.get("report.json") != report_sha:
        raise ValueError("v8 manifest does not bind its report")
    declared_prospective_sha = outputs.get("prospective.identity_only.jsonl", "")
    if len(declared_prospective_sha) != 64:
        raise ValueError("v8 manifest lacks a declared prospective seal hash")
    return {
        "directory": V8_COHORT_FREEZE_DIR.as_posix(),
        "report_sha256": report_sha,
        "manifest_sha256": manifest_sha,
        "status": report["status"],
        "prospective_status": seal["status"],
        "prospective_declared_sha256_from_parent_manifest": declared_prospective_sha,
        "prospective_content_opened": False,
        "prospective_content_hashed": False,
        "raw_train_qid_overlap_reported": 0,
        "raw_train_family_overlap_reported": 0,
    }


def _verify_parent_capacity_hashes(capacity_dir: Path) -> None:
    paths = {
        "capacity_inventory": capacity_dir / "inventory.json",
        "capacity_report": capacity_dir / "report.json",
        "capacity_manifest": capacity_dir / "manifest.json",
    }
    for key, path in paths.items():
        if _sha256_file(path) != EXPECTED_PARENT_HASHES[key]:
            raise ValueError(f"parent capacity metadata lock drift: {key}")


def _verify_current_inventory(
    *,
    project_root: Path,
    parent_lock: Mapping[str, Any],
    raw_train_inventory: Sequence[Mapping[str, str]],
    historical: capacity.HistoricalProjection,
    training: capacity.TrainingProjection,
) -> None:
    if historical.inventory != parent_lock["historical_inventory"]:
        raise ValueError("static 58-file historical registry inventory drift")
    if training.inventory != parent_lock["training_inventory"]:
        raise ValueError("static 20-file training-input ledger drift")
    if training.evidence_inventory != parent_lock["evidence_inventory"]:
        raise ValueError("training-input provenance evidence drift")

    frozen_train = [
        item
        for item in parent_lock["raw_source_inventory"]
        if isinstance(item, Mapping) and str(item.get("path", "")).endswith("/train.jsonl")
    ]
    if list(raw_train_inventory) != frozen_train:
        raise ValueError("raw-train source inventory drift")

    # The parent manifest binds the precise implementation used to construct
    # the static inventories.  Refuse selection after implementation drift.
    for item in parent_lock["implementation_inventory"]:
        if not isinstance(item, Mapping):
            raise ValueError("invalid parent implementation inventory")
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = project_root / path
        if _sha256_file(path) != str(item.get("sha256")):
            raise ValueError(f"capacity implementation drift: {item.get('path')}")


def select_fresh_train_pilot(
    *,
    dataset: str,
    candidates: Mapping[str, Candidate],
    excluded_qids: set[tuple[str, str]],
    excluded_families: set[tuple[str, str]],
    salt: str,
    n: int,
) -> tuple[list[Candidate], dict[str, int]]:
    """Select one eligible qid per dataset-scoped family deterministically."""

    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not salt.strip() or n <= 0:
        raise ValueError("selection salt must be non-empty and n must be positive")

    eligible: list[Candidate] = []
    qid_excluded = 0
    family_excluded = 0
    for candidate in candidates.values():
        qid_hit = (dataset, candidate.qid) in excluded_qids
        family_hit = (dataset, candidate.family_sha256) in excluded_families
        qid_excluded += int(qid_hit)
        family_excluded += int(family_hit)
        if not qid_hit and not family_hit:
            eligible.append(candidate)

    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in eligible:
        by_family[candidate.family_sha256].append(candidate)
    if len(by_family) < n:
        raise ValueError(
            f"{dataset}: only {len(by_family)} eligible families remain; need {n}"
        )

    ordered_families = sorted(
        by_family,
        key=lambda family: hashlib.sha256(
            f"{salt}\0{dataset}\0{family}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[Candidate] = []
    for family in ordered_families[:n]:
        selected.append(
            min(
                by_family[family],
                key=lambda item: (item.question_sha256, item.qid),
            )
        )
    stats = {
        "raw_train_unique_qids": len(candidates),
        "qid_exclusion_hits_not_mutually_exclusive": qid_excluded,
        "family_exclusion_hits_not_mutually_exclusive": family_excluded,
        "eligible_unique_qids": len(eligible),
        "eligible_unique_dataset_scoped_families": len(by_family),
        "selected_rows": len(selected),
        "remaining_one_per_family_capacity_after_selection": len(by_family) - n,
    }
    return selected, stats


def _public_rows(candidates: Sequence[Candidate]) -> list[dict[str, str]]:
    return [
        {"dataset": row.dataset, "qid": row.qid, "question": row.question}
        for row in candidates
    ]


def _selection_checks(
    *,
    selected: Sequence[Candidate],
    historical: capacity.HistoricalProjection,
    training: capacity.TrainingProjection,
    v8_development: IdentityProjection,
    v8_smoke: IdentityProjection,
    raw: Mapping[str, capacity.RawSplitProjection],
    per_dataset: int,
    parent_v8_lock: Mapping[str, Any],
) -> dict[str, Any]:
    qids = {(row.dataset, row.qid) for row in selected}
    families = {(row.dataset, row.family_sha256) for row in selected}
    raw_train_qids = {
        (dataset, qid) for dataset in DATASETS for qid in raw[dataset].identities
    }
    by_dataset = {
        dataset: sum(row.dataset == dataset for row in selected) for dataset in DATASETS
    }
    checks: dict[str, Any] = {
        "total_rows": len(selected),
        "expected_total_rows": per_dataset * len(DATASETS),
        "per_dataset_counts": by_dataset,
        "each_dataset_exactly_30": all(value == per_dataset for value in by_dataset.values()),
        "unique_dataset_scoped_qids": len(qids),
        "unique_dataset_scoped_families": len(families),
        "one_qid_per_dataset_scoped_family": len(selected) == len(families),
        "all_rows_are_from_corresponding_raw_train": len(qids - raw_train_qids) == 0,
        "historical_58_registry_qid_overlap": len(qids & historical.qids),
        "historical_58_registry_family_overlap": len(families & historical.families),
        "training_20_ledger_qid_overlap": len(qids & training.qids),
        "training_20_ledger_family_overlap": len(families & training.families),
        "v8_development90_qid_overlap": len(qids & v8_development.qids),
        "v8_development90_family_overlap": len(families & v8_development.families),
        "v8_smoke12_qid_overlap": len(qids & v8_smoke.qids),
        "v8_smoke12_family_overlap": len(families & v8_smoke.families),
        "sealed_prospective900_content_opened": False,
        "sealed_prospective900_content_hashed": False,
        "sealed_prospective900_disjoint_by_locked_raw_train_qid_gate": (
            parent_v8_lock["raw_train_qid_overlap_reported"] == 0
        ),
        "sealed_prospective900_disjoint_by_locked_raw_train_family_gate": (
            parent_v8_lock["raw_train_family_overlap_reported"] == 0
        ),
        "output_row_field_allowlist": list(OUTPUT_ROW_FIELDS),
        "output_rows_exactly_match_field_allowlist": all(
            tuple(row) == OUTPUT_ROW_FIELDS and set(row) == set(OUTPUT_ROW_FIELDS)
            for row in _public_rows(selected)
        ),
    }
    zero_overlap_keys = (
        "historical_58_registry_qid_overlap",
        "historical_58_registry_family_overlap",
        "training_20_ledger_qid_overlap",
        "training_20_ledger_family_overlap",
        "v8_development90_qid_overlap",
        "v8_development90_family_overlap",
        "v8_smoke12_qid_overlap",
        "v8_smoke12_family_overlap",
    )
    checks["all_freeze_gates_pass"] = bool(
        len(selected) == checks["expected_total_rows"]
        and checks["each_dataset_exactly_30"]
        and len(selected) == len(qids) == len(families)
        and checks["all_rows_are_from_corresponding_raw_train"]
        and all(checks[key] == 0 for key in zero_overlap_keys)
        and checks["sealed_prospective900_disjoint_by_locked_raw_train_qid_gate"]
        and checks["sealed_prospective900_disjoint_by_locked_raw_train_family_gate"]
        and checks["output_rows_exactly_match_field_allowlist"]
    )
    return checks


def _implementation_inventory(project_root: Path) -> list[dict[str, str]]:
    paths = {
        Path(__file__).resolve(),
        Path(capacity.__file__).resolve(),
        Path(capacity.question_sha256.__code__.co_filename).resolve(),
        Path(capacity.family_sha256.__code__.co_filename).resolve(),
    }
    return [
        {"path": _display_path(path, project_root), "sha256": _sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]


def run_freeze(
    *,
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    capacity_audit_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
    selection_salt: str = SELECTION_SALT,
    pilot_per_dataset: int = PILOT_PER_DATASET,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir)
    capacity_audit_dir = Path(capacity_audit_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only v9 pilot directory: {output_dir}"
        )
    if experiment_id != EXPERIMENT_ID or selection_salt != SELECTION_SALT:
        raise ValueError("formal v9 pilot Experiment ID/selection salt is immutable")
    if pilot_per_dataset != PILOT_PER_DATASET:
        raise ValueError("formal v9 pilot requires exactly 30 rows per dataset")

    _verify_parent_capacity_hashes(capacity_audit_dir)
    parent_capacity_lock = __import__(
        "scripts.prepare.freeze_dynamic_decomposition_v8",
        fromlist=["_capacity_artifact_lock"],
    )._capacity_artifact_lock(capacity_audit_dir)
    parent_v8_lock = validate_v8_parent_metadata(project_root=project_root)

    raw: dict[str, capacity.RawSplitProjection] = {}
    candidates: dict[str, dict[str, Candidate]] = {}
    raw_train_inventory: list[dict[str, str]] = []
    raw_for_training: dict[str, dict[str, capacity.RawSplitProjection]] = {}
    for dataset in DATASETS:
        path = data_root / dataset / "train.jsonl"
        projection = capacity._project_raw_split(path, dataset)
        projection.source_identity["path"] = _display_path(path, project_root)
        raw[dataset] = projection
        raw_for_training[dataset] = {"train": projection}
        raw_train_inventory.append(dict(projection.source_identity))
        candidates[dataset] = _load_train_candidates(
            path, dataset=dataset, projection=projection
        )

    historical = capacity._project_historical_registries(
        project_root, capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
    )
    training = capacity._project_training_inputs(
        project_root, capacity.LOCAL_TRAINING_INPUT_SPECS, raw_for_training
    )
    if not training.raw_train_containment_pass:
        raise ValueError("static training ledger escaped corresponding raw-train source")
    _verify_current_inventory(
        project_root=project_root,
        parent_lock=parent_capacity_lock,
        raw_train_inventory=raw_train_inventory,
        historical=historical,
        training=training,
    )

    v8_development = _load_identity_only(
        project_root / V8_DEVELOPMENT_PATH, project_root=project_root
    )
    v8_smoke = _load_identity_only(
        project_root / V8_SMOKE_PATH, project_root=project_root
    )
    if v8_development.inventory["sha256"] != EXPECTED_PARENT_HASHES["v8_development"]:
        raise ValueError("v8 development90 identity lock drift")
    if v8_smoke.inventory["sha256"] != EXPECTED_PARENT_HASHES["v8_smoke"]:
        raise ValueError("v8 smoke12 identity lock drift")
    smoke_report_path = project_root / V8_SMOKE_DIR / "report.json"
    smoke_manifest_path = project_root / V8_SMOKE_DIR / "manifest.json"
    if _sha256_file(smoke_report_path) != EXPECTED_PARENT_HASHES["v8_smoke_report"]:
        raise ValueError("v8 smoke report lock drift")
    if _sha256_file(smoke_manifest_path) != EXPECTED_PARENT_HASHES["v8_smoke_manifest"]:
        raise ValueError("v8 smoke manifest lock drift")

    excluded_qids = set(historical.qids) | set(training.qids)
    excluded_families = set(historical.families) | set(training.families)
    excluded_qids |= set(v8_development.qids) | set(v8_smoke.qids)
    excluded_families |= set(v8_development.families) | set(v8_smoke.families)

    selected: list[Candidate] = []
    capacity_stats: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        dataset_selected, stats = select_fresh_train_pilot(
            dataset=dataset,
            candidates=candidates[dataset],
            excluded_qids=excluded_qids,
            excluded_families=excluded_families,
            salt=selection_salt,
            n=pilot_per_dataset,
        )
        selected.extend(dataset_selected)
        capacity_stats[dataset] = stats

    checks = _selection_checks(
        selected=selected,
        historical=historical,
        training=training,
        v8_development=v8_development,
        v8_smoke=v8_smoke,
        raw=raw,
        per_dataset=pilot_per_dataset,
        parent_v8_lock=parent_v8_lock,
    )
    if not checks["all_freeze_gates_pass"]:
        raise ValueError("fresh v9 train-side pilot failed one or more freeze gates")

    generated_at = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": "FROZEN_BEFORE_RETRIEVAL_OR_MODEL_INFERENCE",
        "purpose": (
            "Fresh Gold-free mechanism pilot for the approved canonical-subanswer v9 "
            "binding change; this file freezes identities, not outcome results."
        ),
        "cohort": {
            "source_split": "raw train",
            "datasets": list(DATASETS),
            "per_dataset": pilot_per_dataset,
            "total": pilot_per_dataset * len(DATASETS),
            "row_fields_exact": list(OUTPUT_ROW_FIELDS),
            "selection_salt": selection_salt,
            "selection_algorithm": (
                "sha256(salt\\0dataset\\0dataset_scoped_family_sha256), then "
                "minimum(question_sha256,qid) within family"
            ),
            "family_version": capacity.FAMILY_VERSION,
            "family_scope": "dataset-scoped lexical-family proxy",
            "one_qid_per_family": True,
        },
        "mandatory_exclusions": {
            "historical_registry_paths": 58,
            "local_training_input_paths": 20,
            "v8_development_rows": 90,
            "v8_engineering_smoke_rows": 12,
            "v8_phase0_rows": "same v8 development90 identities; covered by that exclusion",
            "sealed_v8_prospective900": (
                "never opened or hashed; disjointness inherited from locked zero "
                "raw-train qid/family overlap gates"
            ),
        },
        "approved_single_variable": (
            "When a non-null/non-boolean extracted subanswer appears in multiple retrieved "
            "documents, bind it deterministically to the highest-ranked exact surface match "
            "instead of rejecting it solely for multi-document ambiguity."
        ),
        "not_authorized_by_this_freeze": [
            "Gold access or EM/F1/IHR scoring",
            "opening or hashing the sealed prospective900",
            "large-scale training",
            "changing the frozen v8/v9 results",
        ],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "freeze_kind": "fresh_train_side_identity_only_pilot30x3",
        "selection_implementation": (
            "Reran the frozen static 58-registry/20-ledger projections and selected "
            "from raw train; no separate v9 cohort-audit artifact exists."
        ),
        "parent_capacity_lock": {
            key: parent_capacity_lock[key]
            for key in (
                "directory",
                "manifest_sha256",
                "report_sha256",
                "inventory_sha256",
                "status",
            )
        },
        "parent_v8_cohort_metadata_lock": parent_v8_lock,
        "selection_capacity": capacity_stats,
        "two_wiki_capacity_boundary": {
            "eligible_unique_qids_before_one_per_family_selection": capacity_stats[
                "2wikimultihopqa"
            ]["eligible_unique_qids"],
            "eligible_unique_families_before_selection": capacity_stats[
                "2wikimultihopqa"
            ]["eligible_unique_dataset_scoped_families"],
            "selected_families": pilot_per_dataset,
            "remaining_one_per_family_capacity": capacity_stats[
                "2wikimultihopqa"
            ]["remaining_one_per_family_capacity_after_selection"],
            "interpretation": (
                "Only one additional eligible 2Wiki lexical family remains under the "
                "same frozen exclusion ledger; this is a hard capacity warning, not an "
                "outcome or difficulty statement."
            ),
        },
        "checks": checks,
        "source_access_disclosure": {
            "raw_train_source_files_opened": True,
            "raw_train_source_may_contain_gold": True,
            "source_fields_accessed_for_selection": ["id", "qid", "question"],
            "gold_fields_accessed_for_selection": False,
            "gold_fields_emitted": False,
            "historical_and_training_full_bytes_hashed_for_locking": True,
            "sealed_prospective900_content_opened": False,
            "sealed_prospective900_content_hashed": False,
            "data_raw_modified": False,
        },
        "scientific_boundary": (
            "This artifact proves only identity selection and disjointness. It does not "
            "show that decomposition, retrieval, SFT, PPO, EM/F1, or IHR improves."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = output_dir / "pilot.identity_only.jsonl"
    protocol_path = output_dir / "protocol.json"
    report_path = output_dir / "report.json"
    _write_jsonl(cohort_path, _public_rows(selected))
    _write_json(protocol_path, protocol)
    _write_json(report_path, report)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "python_version": platform.python_version(),
        "inputs": {
            "capacity_parent_metadata": {
                "inventory_sha256": EXPECTED_PARENT_HASHES["capacity_inventory"],
                "report_sha256": EXPECTED_PARENT_HASHES["capacity_report"],
                "manifest_sha256": EXPECTED_PARENT_HASHES["capacity_manifest"],
            },
            "v8_parent_metadata": parent_v8_lock,
            "raw_train_inventory": raw_train_inventory,
            "historical_registry_count": len(historical.inventory),
            "historical_registry_set_sha256": _sha256_json(historical.inventory),
            "training_input_count": len(training.inventory),
            "training_input_set_sha256": _sha256_json(training.inventory),
            "training_evidence_set_sha256": _sha256_json(training.evidence_inventory),
            "v8_development_identity": dict(v8_development.inventory),
            "v8_smoke_identity": dict(v8_smoke.inventory),
        },
        "implementation_inventory": _implementation_inventory(project_root),
        "outputs": [
            {"path": cohort_path.name, "sha256": _sha256_file(cohort_path)},
            {"path": protocol_path.name, "sha256": _sha256_file(protocol_path)},
            {"path": report_path.name, "sha256": _sha256_file(report_path)},
        ],
        "output_row_field_allowlist": list(OUTPUT_ROW_FIELDS),
        "gold_access": False,
        "gold_fields_emitted": False,
        "retrieval_or_model_inference": False,
        "sealed_prospective900_content_opened": False,
        "sealed_prospective900_content_hashed": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--capacity_audit_dir", type=Path, default=CAPACITY_AUDIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or project_root / "data").resolve()
    capacity_audit_dir = args.capacity_audit_dir
    if not capacity_audit_dir.is_absolute():
        capacity_audit_dir = project_root / capacity_audit_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    report = run_freeze(
        project_root=project_root,
        data_root=data_root,
        output_dir=output_dir,
        capacity_audit_dir=capacity_audit_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "rows": report["checks"]["total_rows"],
                "two_wiki_remaining_family_capacity": report[
                    "two_wiki_capacity_boundary"
                ]["remaining_one_per_family_capacity"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
