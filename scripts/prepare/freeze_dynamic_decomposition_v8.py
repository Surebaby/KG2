#!/usr/bin/env python
"""Freeze the strict-Scope-A dynamic-decomposition v8 identity cohorts.

This is an identity-only custodian export.  Source JSONL files may contain
Gold fields, but selection reads only ``dataset`` (where applicable),
``id``/``qid``, and ``question``.  Frozen rows contain exactly
``dataset``, ``qid``, and ``question``.  Per-question hashes, family labels,
Gold, metadata, retrieval results, and model outputs are never emitted.

The command-line path is deliberately tied to the append-only v8 capacity
audit.  It reuses that audit's static 58-file historical registry and static
20-file local training-input ledger, reruns the same strict Scope-A logic,
and refuses to freeze if any capacity-audit input has drifted.
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
SCHEMA_VERSION = "subquestion-v8-identity-cohort-freeze-v1"
MANIFEST_SCHEMA_VERSION = "subquestion-v8-identity-cohort-freeze-manifest-v1"
EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-COHORT-FREEZE-"
    "DEV30-PROSPECTIVE300-SEED20260904-V1"
)
SELECTION_SEED = 20260904
DEVELOPMENT_PER_DATASET = 30
PROSPECTIVE_PER_DATASET = 300
OUTPUT_ROW_FIELDS = ("dataset", "qid", "question")
STATUS = "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE"

DEFAULT_CAPACITY_AUDIT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_capacity_audit_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_"
    "seed20260904_v1"
)


@dataclass(frozen=True)
class Candidate:
    """Private in-memory identity projection; hashes are never serialized."""

    dataset: str
    qid: str
    question: str
    question_sha256: str
    family_sha256: str


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
                raise ValueError(f"frozen row violates field allowlist: {sorted(row)}")
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _capacity_artifact_lock(capacity_audit_dir: Path) -> dict[str, Any]:
    """Validate the prior append-only capacity artifact and return its lock."""

    manifest_path = capacity_audit_dir / "manifest.json"
    report_path = capacity_audit_dir / "report.json"
    inventory_path = capacity_audit_dir / "inventory.json"
    for path in (manifest_path, report_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing capacity-audit artifact: {path}")

    manifest = _load_json(manifest_path)
    report = _load_json(report_path)
    inventory = _load_json(inventory_path)
    if manifest.get("experiment_id") != capacity.EXPERIMENT_ID:
        raise ValueError("capacity-audit Experiment ID mismatch")
    if report.get("experiment_id") != capacity.EXPERIMENT_ID:
        raise ValueError("capacity-audit report Experiment ID mismatch")
    if report.get("status") not in {capacity.LIMITED_PASS_STATUS, capacity.FULL_PASS_STATUS}:
        raise ValueError("capacity audit did not pass the strict n=330 Scope-A gate")
    scope_a = report.get("scope_a_strict")
    if not isinstance(scope_a, Mapping) or scope_a.get(
        "balanced_all_datasets_gate_n330"
    ) is not True:
        raise ValueError("capacity audit lacks a passing strict Scope-A n=330 gate")
    if report.get("checks", {}).get("scope_a_eligible_exclusion_overlaps_zero") is not True:
        raise ValueError("capacity audit exclusion-overlap gate did not pass")
    if report.get("checks", {}).get("training_inputs_within_corresponding_raw_train") is not True:
        raise ValueError("capacity audit training-input containment gate did not pass")

    declared_outputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
    }
    for name, path in (("report.json", report_path), ("inventory.json", inventory_path)):
        if declared_outputs.get(name) != _sha256_file(path):
            raise ValueError(f"capacity-audit {name} hash mismatch")

    historical_inventory = inventory.get("historical_evaluation_protocol_registries")
    training_inventory = inventory.get("local_training_inputs")
    evidence_inventory = inventory.get("training_input_manifest_config_evidence")
    if not isinstance(historical_inventory, list):
        raise ValueError("capacity inventory lacks historical registry list")
    if not isinstance(training_inventory, list):
        raise ValueError("capacity inventory lacks local training-input list")
    if not isinstance(evidence_inventory, list):
        raise ValueError("capacity inventory lacks training evidence list")
    if [item.get("path") for item in historical_inventory] != list(
        capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
    ):
        raise ValueError("capacity audit does not bind the current static 58-path registry")
    if [item.get("path") for item in training_inventory] != [
        spec.path for spec in capacity.LOCAL_TRAINING_INPUT_SPECS
    ]:
        raise ValueError("capacity audit does not bind the current static training ledger")

    return {
        "directory": capacity_audit_dir.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "report_sha256": _sha256_file(report_path),
        "inventory_sha256": _sha256_file(inventory_path),
        "status": str(report["status"]),
        "historical_inventory": historical_inventory,
        "training_inventory": training_inventory,
        "evidence_inventory": evidence_inventory,
        "raw_source_inventory": manifest.get("inputs", {}).get(
            "raw_source_inventory", []
        ),
        "implementation_inventory": manifest.get("implementation_inventory", []),
    }


def _candidate_from_identity_fields(
    row: Mapping[str, Any], *, dataset: str
) -> Candidate | None:
    """Project only id/qid/question; no other source field is accessed."""

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


def _load_dev_candidates(
    path: Path,
    *,
    dataset: str,
    projection: capacity.RawSplitProjection,
) -> dict[str, Candidate]:
    """Load the answer-free identity projection and cross-check the audit view."""

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
            if expected is None:
                raise ValueError(
                    f"identity-only projection disagrees with capacity projection: "
                    f"{dataset}/{candidate.qid}"
                )
            if (
                expected.question_sha256 != candidate.question_sha256
                or expected.family_sha256 != candidate.family_sha256
            ):
                raise ValueError(
                    f"identity hash drift while loading dev source: "
                    f"{dataset}/{candidate.qid}"
                )
            previous = candidates.get(candidate.qid)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"conflicting duplicate identity in dev source: "
                    f"{dataset}/{candidate.qid}"
                )
            candidates[candidate.qid] = candidate
    if set(candidates) != set(projection.identities):
        raise ValueError(f"identity-only dev projection is incomplete for {dataset}")
    return candidates


def select_scope_a_cohorts(
    *,
    dataset: str,
    candidates: Mapping[str, Candidate],
    eligible_qids: set[str],
    development_n: int,
    prospective_n: int,
    seed: int,
) -> tuple[list[Candidate], list[Candidate]]:
    """Select one identity per family with a fixed answer-free ordering."""

    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if development_n <= 0 or prospective_n <= 0:
        raise ValueError("development and prospective sizes must be positive")
    missing = eligible_qids - set(candidates)
    if missing:
        raise ValueError(f"{dataset}: eligible qids missing from dev projection")

    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for qid in eligible_qids:
        candidate = candidates[qid]
        by_family[candidate.family_sha256].append(candidate)
    needed = development_n + prospective_n
    if len(by_family) < needed:
        raise ValueError(
            f"{dataset}: only {len(by_family)} strict Scope-A families; need {needed}"
        )

    ordered_families = sorted(
        by_family,
        key=lambda family: hashlib.sha256(
            f"{seed}\0{dataset}\0{family}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[Candidate] = []
    for family in ordered_families[:needed]:
        within_family = sorted(
            by_family[family], key=lambda item: (item.question_sha256, item.qid)
        )
        selected.append(within_family[0])
    return selected[:development_n], selected[development_n:]


def _public_rows(candidates: Sequence[Candidate]) -> list[dict[str, str]]:
    rows = [
        {
            "dataset": candidate.dataset,
            "qid": candidate.qid,
            "question": candidate.question,
        }
        for candidate in candidates
    ]
    if any(tuple(row) != OUTPUT_ROW_FIELDS for row in rows):
        raise AssertionError("internal output-field order drift")
    return rows


def _partition_checks(
    *,
    development: Sequence[Candidate],
    prospective: Sequence[Candidate],
    historical: capacity.HistoricalProjection,
    training: capacity.TrainingProjection,
    raw: Mapping[str, Mapping[str, capacity.RawSplitProjection]],
    development_n: int,
    prospective_n: int,
) -> dict[str, Any]:
    def scoped_qids(rows: Sequence[Candidate]) -> set[tuple[str, str]]:
        return {(row.dataset, row.qid) for row in rows}

    def scoped_families(rows: Sequence[Candidate]) -> set[tuple[str, str]]:
        return {(row.dataset, row.family_sha256) for row in rows}

    dev_qids = scoped_qids(development)
    prospective_qids = scoped_qids(prospective)
    dev_families = scoped_families(development)
    prospective_families = scoped_families(prospective)
    all_rows = list(development) + list(prospective)
    all_qids = dev_qids | prospective_qids
    all_families = dev_families | prospective_families
    expected_dev = development_n * len(DATASETS)
    expected_prospective = prospective_n * len(DATASETS)

    raw_train_qids = {
        (dataset, qid)
        for dataset in DATASETS
        for qid in raw[dataset]["train"].identities
    }
    raw_train_families = {
        (dataset, identity.family_sha256)
        for dataset in DATASETS
        for identity in raw[dataset]["train"].identities.values()
    }
    by_dataset = {
        dataset: {
            "development_rows": sum(row.dataset == dataset for row in development),
            "prospective_rows": sum(row.dataset == dataset for row in prospective),
            "union_rows": sum(row.dataset == dataset for row in all_rows),
            "union_unique_dataset_scoped_qids": len(
                {(row.dataset, row.qid) for row in all_rows if row.dataset == dataset}
            ),
            "union_unique_dataset_scoped_families": len(
                {
                    (row.dataset, row.family_sha256)
                    for row in all_rows
                    if row.dataset == dataset
                }
            ),
        }
        for dataset in DATASETS
    }
    checks = {
        "development_total_rows": len(development),
        "prospective_total_rows": len(prospective),
        "by_dataset": by_dataset,
        "development_expected_count": len(development) == expected_dev,
        "prospective_expected_count": len(prospective) == expected_prospective,
        "each_dataset_exact_development_and_prospective_count": all(
            values["development_rows"] == development_n
            and values["prospective_rows"] == prospective_n
            for values in by_dataset.values()
        ),
        "development_prospective_dataset_scoped_qid_overlap": len(
            dev_qids & prospective_qids
        ),
        "development_prospective_dataset_scoped_family_overlap": len(
            dev_families & prospective_families
        ),
        "one_qid_per_dataset_scoped_family": len(all_rows) == len(all_families),
        "historical_registry_qid_overlap": len(all_qids & historical.qids),
        "historical_registry_family_overlap": len(all_families & historical.families),
        "training_ledger_qid_overlap": len(all_qids & training.qids),
        "training_ledger_family_overlap": len(all_families & training.families),
        "raw_train_qid_overlap": len(all_qids & raw_train_qids),
        "raw_train_family_overlap": len(all_families & raw_train_families),
        "output_row_field_allowlist": list(OUTPUT_ROW_FIELDS),
        "output_rows_exactly_match_field_allowlist": all(
            tuple(row) == OUTPUT_ROW_FIELDS
            for row in _public_rows(all_rows)
        ),
    }
    pass_conditions = (
        checks["development_expected_count"]
        and checks["prospective_expected_count"]
        and checks["each_dataset_exact_development_and_prospective_count"]
        and checks["development_prospective_dataset_scoped_qid_overlap"] == 0
        and checks["development_prospective_dataset_scoped_family_overlap"] == 0
        and checks["one_qid_per_dataset_scoped_family"]
        and checks["historical_registry_qid_overlap"] == 0
        and checks["historical_registry_family_overlap"] == 0
        and checks["training_ledger_qid_overlap"] == 0
        and checks["training_ledger_family_overlap"] == 0
        and checks["raw_train_qid_overlap"] == 0
        and checks["raw_train_family_overlap"] == 0
        and checks["output_rows_exactly_match_field_allowlist"]
    )
    checks["all_freeze_gates_pass"] = bool(pass_conditions)
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


def _verify_current_inputs_match_capacity_lock(
    *,
    lock: Mapping[str, Any],
    raw_source_inventory: Sequence[Mapping[str, str]],
    historical: capacity.HistoricalProjection,
    training: capacity.TrainingProjection,
) -> None:
    comparisons = (
        (list(raw_source_inventory), lock["raw_source_inventory"], "raw source"),
        (historical.inventory, lock["historical_inventory"], "historical registry"),
        (training.inventory, lock["training_inventory"], "training input"),
        (training.evidence_inventory, lock["evidence_inventory"], "training evidence"),
    )
    for current, frozen, label in comparisons:
        if current != frozen:
            raise ValueError(
                f"{label} inventory drifted after the capacity audit; refuse to freeze"
            )
    capacity_impl = {
        str(item.get("path")): str(item.get("sha256"))
        for item in lock["implementation_inventory"]
        if isinstance(item, Mapping)
    }
    for path_value, expected_hash in capacity_impl.items():
        path = Path(path_value)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(
                "capacity-audit implementation drifted before cohort freeze: "
                f"{path_value}"
            )


def run_freeze(
    *,
    project_root: Path,
    data_root: Path,
    capacity_audit_dir: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID,
    seed: int = SELECTION_SEED,
    development_per_dataset: int = DEVELOPMENT_PER_DATASET,
    prospective_per_dataset: int = PROSPECTIVE_PER_DATASET,
    historical_registry_paths: Sequence[str] | None = None,
    training_input_specs: Sequence[capacity.TrainingInputSpec] | None = None,
    verify_formal_capacity_audit: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Freeze two answer-free identity cohorts without touching source data."""

    project_root = Path(project_root).resolve()
    data_root = Path(data_root).resolve()
    capacity_audit_dir = Path(capacity_audit_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite append-only cohort freeze directory: {output_dir}"
        )
    if not str(experiment_id).strip():
        raise ValueError("experiment_id must be non-empty")
    if development_per_dataset <= 0 or prospective_per_dataset <= 0:
        raise ValueError("cohort sizes must be positive")

    registry_paths = tuple(
        capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
        if historical_registry_paths is None
        else historical_registry_paths
    )
    selected_training_specs = tuple(
        capacity.LOCAL_TRAINING_INPUT_SPECS
        if training_input_specs is None
        else training_input_specs
    )
    if verify_formal_capacity_audit:
        if historical_registry_paths is not None or training_input_specs is not None:
            raise ValueError("formal freeze cannot substitute either static inventory")
        lock = _capacity_artifact_lock(capacity_audit_dir)
    else:
        lock = {
            "directory": "TEST_ONLY_NO_FORMAL_CAPACITY_LOCK",
            "manifest_sha256": "TEST_ONLY",
            "report_sha256": "TEST_ONLY",
            "inventory_sha256": "TEST_ONLY",
            "status": "TEST_ONLY",
        }

    raw: dict[str, dict[str, capacity.RawSplitProjection]] = {}
    raw_source_inventory: list[dict[str, str]] = []
    for dataset in DATASETS:
        raw[dataset] = {}
        for split in capacity.SPLITS:
            source_path = data_root / dataset / f"{split}.jsonl"
            projection = capacity._project_raw_split(source_path, dataset)
            projection.source_identity["path"] = _display_path(source_path, project_root)
            raw[dataset][split] = projection
            raw_source_inventory.append(dict(projection.source_identity))

    historical = capacity._project_historical_registries(project_root, registry_paths)
    training = capacity._project_training_inputs(
        project_root, selected_training_specs, raw
    )
    if not training.raw_train_containment_pass:
        raise ValueError("training-input projection escaped corresponding raw train")
    if verify_formal_capacity_audit:
        _verify_current_inputs_match_capacity_lock(
            lock=lock,
            raw_source_inventory=raw_source_inventory,
            historical=historical,
            training=training,
        )

    development: list[Candidate] = []
    prospective: list[Candidate] = []
    strict_capacity: dict[str, int] = {}
    needed = development_per_dataset + prospective_per_dataset
    for dataset in DATASETS:
        train_families = {
            identity.family_sha256
            for identity in raw[dataset]["train"].identities.values()
        }
        scope_counts, eligible_qids, _ = capacity._scope_counts(
            dataset=dataset,
            dev=raw[dataset]["dev"].identities,
            historical=historical,
            training=training,
            raw_train_qids=set(raw[dataset]["train"].identities),
            raw_train_families=train_families,
            exclude_training_ledger=True,
            require_raw_train_family_isolation=True,
            gate_330=needed,
            gate_1330=needed,
        )
        strict_capacity[dataset] = int(
            scope_counts["exact_freezable_capacity_one_per_dataset_scoped_family"]
        )
        if strict_capacity[dataset] < needed:
            raise ValueError(
                f"{dataset}: strict Scope-A capacity {strict_capacity[dataset]} < {needed}"
            )
        candidates = _load_dev_candidates(
            data_root / dataset / "dev.jsonl",
            dataset=dataset,
            projection=raw[dataset]["dev"],
        )
        selected_development, selected_prospective = select_scope_a_cohorts(
            dataset=dataset,
            candidates=candidates,
            eligible_qids=eligible_qids,
            development_n=development_per_dataset,
            prospective_n=prospective_per_dataset,
            seed=seed,
        )
        development.extend(selected_development)
        prospective.extend(selected_prospective)

    checks = _partition_checks(
        development=development,
        prospective=prospective,
        historical=historical,
        training=training,
        raw=raw,
        development_n=development_per_dataset,
        prospective_n=prospective_per_dataset,
    )
    if not checks["all_freeze_gates_pass"]:
        raise ValueError("identity cohort failed one or more strict Scope-A freeze gates")

    generated_at = generated_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "freeze_kind": "identity_only_strict_scope_a",
        "capacity_audit_lock": {
            key: lock[key]
            for key in (
                "directory",
                "manifest_sha256",
                "report_sha256",
                "inventory_sha256",
                "status",
            )
        },
        "selection_contract": {
            "seed": seed,
            "algorithm": "sha256-seeded-dataset-scoped-family-order-v1",
            "family_version": capacity.FAMILY_VERSION,
            "family_scope": "dataset-scoped",
            "one_qid_per_family": True,
            "development_per_dataset": development_per_dataset,
            "prospective_per_dataset": prospective_per_dataset,
            "reserve_per_dataset": 0,
            "selection_features": [
                "dataset",
                "id_or_qid",
                "question",
                "question-derived lexical family",
                "fixed seed",
            ],
            "model_or_retrieval_used": False,
            "gold_used_for_selection": False,
        },
        "strict_scope_a_capacity_before_selection": strict_capacity,
        "checks": checks,
        "source_access_disclosure": {
            "source_files_opened": True,
            "source_may_contain_gold": True,
            "gold_fields_accessed_for_selection": False,
            "gold_fields_emitted": False,
            "raw_source_fields_accessed": ["id", "qid", "question"],
            "historical_registry_fields_accessed": [
                "dataset",
                "id",
                "qid",
                "source_id",
                "question",
            ],
            "training_input_fields_accessed": [
                "dataset",
                "qid_or_declared_source_qid_alias",
                "question",
            ],
            "per_question_hash_or_family_emitted": False,
            "output_row_fields": list(OUTPUT_ROW_FIELDS),
            "data_raw_modified": False,
        },
        "prospective_seal": {
            "status": "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT",
            "permitted_before_formal_evaluation": [
                "file-integrity verification",
                "aggregate count verification",
                "field-allowlist verification",
            ],
            "forbidden_before_formal_evaluation": [
                "manual inspection for method tuning",
                "retrieval or model inference",
                "Gold scoring",
                "selection based on expected difficulty or outcome",
            ],
        },
        "training_ledger_boundary": {
            "complete_historical_training_ledger_available": False,
            "missing_old_checkpoint_input_ledgers": "UNKNOWN",
            "conservative_control": (
                "Strict Scope A excludes every qid and lexical family in each corresponding "
                "raw-train source in addition to the static local training ledger."
            ),
        },
        "scientific_boundary": (
            "This artifact freezes answer-free identities only. It does not run retrieval, "
            "generate subquestions, inspect Gold, score EM/F1/IHR, modify an evaluation "
            "protocol, or authorize training. The prospective split remains sealed."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    development_path = output_dir / "development.identity_only.jsonl"
    prospective_path = output_dir / "prospective.identity_only.jsonl"
    report_path = output_dir / "report.json"
    _write_jsonl(development_path, _public_rows(development))
    _write_jsonl(prospective_path, _public_rows(prospective))
    _write_json(report_path, report)

    output_artifacts = [
        {"path": development_path.name, "sha256": _sha256_file(development_path)},
        {"path": prospective_path.name, "sha256": _sha256_file(prospective_path)},
        {"path": report_path.name, "sha256": _sha256_file(report_path)},
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at_utc": generated_at,
        "status": STATUS,
        "python_version": platform.python_version(),
        "capacity_audit_lock": report["capacity_audit_lock"],
        "inputs": {
            "raw_source_inventory": raw_source_inventory,
            "historical_registry_count": len(historical.inventory),
            "historical_registry_set_sha256": _sha256_json(historical.inventory),
            "training_input_count": len(training.inventory),
            "training_input_set_sha256": _sha256_json(training.inventory),
            "training_evidence_count": len(training.evidence_inventory),
            "training_evidence_set_sha256": _sha256_json(
                training.evidence_inventory
            ),
        },
        "implementation_inventory": _implementation_inventory(project_root),
        "outputs": output_artifacts,
        "output_row_field_allowlist": list(OUTPUT_ROW_FIELDS),
        "selection_contains_gold": False,
        "per_question_hash_or_family_emitted": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument(
        "--capacity_audit_dir", type=Path, default=DEFAULT_CAPACITY_AUDIT_DIR
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment_id", default=EXPERIMENT_ID)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
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
        capacity_audit_dir=capacity_audit_dir,
        output_dir=output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        # Explicitly fixed: no CLI can replace the two static inventories.
        historical_registry_paths=None,
        training_input_specs=None,
        verify_formal_capacity_audit=True,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "development_rows": report["checks"]["development_total_rows"],
                "prospective_rows": report["checks"]["prospective_total_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
