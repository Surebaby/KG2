"""Strict, development-only loader for the frozen v8 identity cohort.

The prospective cohort is sealed.  This module intentionally implements no
unlock flag, token, environment-variable override, or permissive alternate
path.  A later prospective run requires a separate append-only authorization
artifact and a new loader/protocol version.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any


COHORT_LOADER_VERSION = "dynamic-decomposition-v8-development-cohort-loader-1"
FREEZE_MANIFEST_SCHEMA_VERSION = "subquestion-v8-identity-cohort-freeze-manifest-v1"
FREEZE_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-COHORT-FREEZE-"
    "DEV30-PROSPECTIVE300-SEED20260904-V1"
)
FREEZE_STATUS = "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_DIRECTORY_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_cohort_freeze_dev30_prospective300_"
    "seed20260904_v1"
)
MANIFEST_RELATIVE_PATH = FREEZE_DIRECTORY_RELATIVE / "manifest.json"
DEVELOPMENT_RELATIVE_PATH = FREEZE_DIRECTORY_RELATIVE / "development.identity_only.jsonl"
PROSPECTIVE_RELATIVE_PATH = FREEZE_DIRECTORY_RELATIVE / "prospective.identity_only.jsonl"

EXPECTED_MANIFEST_SHA256 = "cda6525e1562697c31e17cb457280fe272de039ebebee23a2ddcabaa942730e6"
EXPECTED_DEVELOPMENT_SHA256 = "dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2"
SEALED_PROSPECTIVE_SHA256 = "36b680cabef059dae7370bb131b1bafc0f120baf372f4e7666aa0e2d13b13c99"

DEVELOPMENT_ROLE = "development"
SEALED_PROSPECTIVE_ROLE = "prospective"
ROW_FIELDS = ("dataset", "qid", "question")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEVELOPMENT_PER_DATASET = 30


class FrozenCohortError(ValueError):
    """The frozen cohort lock or an identity row is invalid."""


class SealedProspectiveCohortError(PermissionError):
    """Prospective identities were requested without append-only authorization."""


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FrozenCohortError(f"cannot hash frozen artifact: {path}") from exc
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except _DuplicateJSONKey as exc:
        raise FrozenCohortError(f"duplicate JSON key in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FrozenCohortError(f"cannot read valid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise FrozenCohortError(f"frozen manifest must be a JSON object: {path}")
    return value


def _safe_identity_text(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FrozenCohortError(
            f"line {line_number} {field} must be a non-empty unpadded string"
        )
    if any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value):
        raise FrozenCohortError(f"line {line_number} {field} contains a control character")
    return value


def _resolve_locked_path(
    supplied: str | Path | None,
    *,
    expected_relative: Path,
    label: str,
) -> Path:
    expected = (PROJECT_ROOT / expected_relative).resolve()
    actual = expected if supplied is None else Path(supplied).resolve()
    if actual != expected:
        if actual.name == "prospective.identity_only.jsonl" or "prospective" in actual.name.casefold():
            raise SealedProspectiveCohortError(
                "prospective cohort is sealed; no unlock authorization is implemented"
            )
        raise FrozenCohortError(f"{label} path is not the frozen path: {actual}")
    return actual


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != FREEZE_MANIFEST_SCHEMA_VERSION:
        raise FrozenCohortError("frozen cohort manifest schema mismatch")
    if manifest.get("experiment_id") != FREEZE_EXPERIMENT_ID:
        raise FrozenCohortError("frozen cohort Experiment ID mismatch")
    if manifest.get("status") != FREEZE_STATUS:
        raise FrozenCohortError("frozen cohort status is not complete")
    if manifest.get("selection_contains_gold") is not False:
        raise FrozenCohortError("frozen cohort manifest does not assert Gold-free selection")
    if tuple(manifest.get("output_row_field_allowlist") or ()) != ROW_FIELDS:
        raise FrozenCohortError("frozen cohort field allowlist mismatch")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise FrozenCohortError("frozen cohort manifest outputs must be a list")
    declared: dict[str, str] = {}
    for item in outputs:
        if not isinstance(item, dict):
            raise FrozenCohortError("frozen cohort output lock must be an object")
        path, digest = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise FrozenCohortError("frozen cohort output lock is malformed")
        if path in declared:
            raise FrozenCohortError(f"duplicate frozen output lock: {path}")
        declared[path] = digest
    if declared.get("development.identity_only.jsonl") != EXPECTED_DEVELOPMENT_SHA256:
        raise FrozenCohortError("manifest development cohort SHA mismatch")
    if declared.get("prospective.identity_only.jsonl") != SEALED_PROSPECTIVE_SHA256:
        raise FrozenCohortError("manifest sealed prospective cohort SHA mismatch")


def _load_identity_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise FrozenCohortError(f"cannot open frozen development cohort: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith("\n"):
                raise FrozenCohortError(f"line {line_number} lacks the frozen newline terminator")
            if not raw_line.strip():
                raise FrozenCohortError(f"blank line in frozen cohort at {line_number}")
            try:
                row = json.loads(raw_line, object_pairs_hook=_unique_object)
            except _DuplicateJSONKey as exc:
                raise FrozenCohortError(
                    f"duplicate JSON key in cohort line {line_number}: {exc}"
                ) from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise FrozenCohortError(f"invalid JSON at cohort line {line_number}") from exc
            if not isinstance(row, dict):
                raise FrozenCohortError(f"cohort line {line_number} must be an object")
            if tuple(row) != ROW_FIELDS or set(row) != set(ROW_FIELDS):
                raise FrozenCohortError(
                    f"cohort line {line_number} violates the exact identity field allowlist"
                )
            dataset = _safe_identity_text(row["dataset"], field="dataset", line_number=line_number)
            qid = _safe_identity_text(row["qid"], field="qid", line_number=line_number)
            question = _safe_identity_text(
                row["question"], field="question", line_number=line_number
            )
            if dataset not in DATASETS:
                raise FrozenCohortError(
                    f"cohort line {line_number} has unsupported dataset {dataset!r}"
                )
            identity = f"{dataset}::{qid}"
            if identity in seen:
                raise FrozenCohortError(f"duplicate frozen cohort identity: {identity}")
            seen.add(identity)
            counts[dataset] += 1
            rows.append({"dataset": dataset, "qid": qid, "question": question})
    expected_counts = {dataset: DEVELOPMENT_PER_DATASET for dataset in DATASETS}
    if dict(counts) != expected_counts:
        raise FrozenCohortError(
            f"development cohort counts mismatch: observed={dict(counts)}, expected={expected_counts}"
        )
    return rows


def load_frozen_v8_cohort(
    *,
    role: str = DEVELOPMENT_ROLE,
    manifest_path: str | Path | None = None,
    cohort_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load only the locked development cohort and return its provenance lock.

    ``role='prospective'`` and any prospective path are rejected before the
    prospective cohort file is hashed or opened.  There is deliberately no
    authorization or override parameter in this loader version.
    """

    if role != DEVELOPMENT_ROLE:
        if role == SEALED_PROSPECTIVE_ROLE or "prospective" in str(role).casefold():
            raise SealedProspectiveCohortError(
                "prospective cohort is sealed; append-only unlock authorization is required"
            )
        raise FrozenCohortError(f"unsupported frozen cohort role: {role!r}")
    manifest = _resolve_locked_path(
        manifest_path,
        expected_relative=MANIFEST_RELATIVE_PATH,
        label="manifest",
    )
    cohort = _resolve_locked_path(
        cohort_path,
        expected_relative=DEVELOPMENT_RELATIVE_PATH,
        label="cohort",
    )
    if _sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise FrozenCohortError("frozen cohort manifest file SHA mismatch")
    manifest_value = _load_json_object(manifest)
    _validate_manifest(manifest_value)
    if _sha256_file(cohort) != EXPECTED_DEVELOPMENT_SHA256:
        raise FrozenCohortError("frozen development cohort file SHA mismatch")
    rows = _load_identity_rows(cohort)
    return {
        "loader_version": COHORT_LOADER_VERSION,
        "role": DEVELOPMENT_ROLE,
        "gold_access": False,
        "manifest_path": MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "cohort_path": DEVELOPMENT_RELATIVE_PATH.as_posix(),
        "cohort_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "row_count": len(rows),
        "per_dataset_counts": {dataset: DEVELOPMENT_PER_DATASET for dataset in DATASETS},
        "prospective_unlocked": False,
        "rows": rows,
    }


__all__ = [
    "COHORT_LOADER_VERSION",
    "DEVELOPMENT_RELATIVE_PATH",
    "EXPECTED_DEVELOPMENT_SHA256",
    "EXPECTED_MANIFEST_SHA256",
    "FrozenCohortError",
    "MANIFEST_RELATIVE_PATH",
    "SEALED_PROSPECTIVE_SHA256",
    "SealedProspectiveCohortError",
    "load_frozen_v8_cohort",
]
