#!/usr/bin/env python
"""Freeze strict-eligible Query Controller v1 train-side cohorts.

The freezer consumes *validated q1/q2 action candidate pools*, never a random
sample of raw identities. A selected identity is therefore guaranteed to have
a passage-bound, linear two-step target before it is assigned to train, dev, or
confirmation. Downstream materialization must consume the emitted locks
exactly and may not replace an identity that later fails.

The confirmation cohort is a family-disjoint holdout from the old planner's
train split. It is a train-side mechanism confirmation, not an external test.
The separately sealed prospective900 remains unopened and unhashed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.eval.query_controller_v1 import (  # noqa: E402
    OBSERVATION_ANNOTATION_PATHS,
    OBSERVATION_BINDING_METHODS,
    OBSERVATION_FIELDS,
    OBSERVATION_PROVENANCE_FIELDS,
    PREVIOUS_ACTION_FIELDS,
    validate_action_record,
)
from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from scripts.prepare.build_query_controller_action_supervision_v1 import (  # noqa: E402
    canonical_action_pair_sha256,
)
from scripts.prepare.freeze_qpeg_v1_protocol import (  # noqa: E402
    FAMILY_VERSION,
    family_sha256,
)


EXPERIMENT_ID = "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_4"
SCHEMA_VERSION = "query-controller-v1-pilot-protocol-4.4"
MANIFEST_SCHEMA_VERSION = "query-controller-v1-pilot-manifest-4.4"
STATUS = (
    "FROZEN_V4_4_PEFT_DEFAULT_FP32_CLEAN_RELOAD_"
    "SAME_IDENTITIES_2DATASET_ACTIONS_NOT_TRAINED"
)
# v4.4 changes only clean-reload dtype handling and telemetry. Keeping the
# original v4.2 selection salt is deliberate: every train/dev/confirmation
# identity must be byte-identical to the already-frozen v4.3 locks.
SELECTION_SALT = "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_2"

ENABLED_DATASETS = ("2wikimultihopqa", "musique")
SPLIT_SIZES = {"train": 600, "dev": 60, "confirmation": 30}
SOURCE_SPLIT_FOR_ROLE = {
    "train": "train",
    "dev": "dev",
    "confirmation": "train",
}
IDENTITY_FIELDS = (
    "dataset",
    "qid",
    "question_key",
    "question",
    "question_sha256",
    "family_sha256",
    "split",
    "action_pair_sha256",
)

PAIR_HASH_INDEX_FIELDS = frozenset(
    {
        "dataset",
        "qid",
        "source_split",
        "family_sha256",
        "action_pair_canonical_sha256",
    }
)

IMPLEMENTATION_PATHS = {
    "central_action_validator": Path("kgproweight/eval/query_controller_v1.py"),
    "protocol_freezer": Path("scripts/prepare/freeze_query_controller_v1_protocol.py"),
    "action_builder": Path(
        "scripts/prepare/build_query_controller_action_supervision_v1.py"
    ),
    "controller_trainer": Path("kgproweight/training/query_controller.py"),
    "controller_train_cli": Path("scripts/train/query_controller.py"),
    "controller_greedy_runner": Path("kgproweight/eval/query_controller_runner.py"),
    "controller_generate_cli": Path("scripts/eval/generate_query_controller_actions.py"),
    "controller_mechanism_scorer": Path(
        "scripts/eval/evaluate_query_controller_actions.py"
    ),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

DEFAULT_CANDIDATE_DIR = Path(
    "data/silver_data/query_controller_action_candidates_v1_seed42_v4_1"
)
EXPECTED_CANDIDATE_SHA256 = {
    "train": "67cdb276c02a42ecd76febae947a1d0ded64c182b30ca21ad1d02e8ce83dc281",
    "dev": "a74a24e92120537562b63f277f0c220291c4355e22e90ea156873b7b2aa3fa83",
    "pair_hashes": "3669a4e788ef306ee86f3d80fe40c70bc8fbc72d8aa6ec74eeac3447484d5b00",
    "rejected": "cfcae01be261a9584b8207fa511cde3de6ae310f64d1ddd7bdee6a1dff6d9b74",
    "report": "5a1f5f37f645055a75e01091f4f46a3edbb4f742985235966a2ac4ad3709a423",
    "manifest": "80ea6c84df9264b8017ab677cddc68448961b76d4b11acafb289c91e8fd9f0a8",
    "hardening_audit": "e499056466956fbd510fdc30d4beff216348999a18ef7443b8624b46ccc8a05d",
    "metadata_addendum": "7946d97720c1afcadd1e618735edf843e758c0512283c380b90f78f48c236531",
}
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4"
)

EXPECTED_V4_3_IDENTITY_SHA256 = {
    "train": "2a31e10a1d37e2090e9909fe05975fba3179c1f4301b8fe0a3e1c94192da2da3",
    "dev": "5c78577ccf5bea18e401f1863580871b09d09d1a0483adcdf1a9816172dc9a07",
    "confirmation": "b88bbae2c0758e5f7ffc77ff4aefff4dae6b822c967ba77108229d38b5e4b46b",
}
V4_3_FAILURE_LINEAGE = {
    "protocol": {
        "path": Path(
            "outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_3/"
            "protocol.json"
        ),
        "sha256": "8ba09daae131c3a2a2860bbc7f28dfb4e71fb46bc03585921a47e0c4aa92a998",
    },
    "protocol_supersession_addendum": {
        "path": Path(
            "outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_3/"
            "metadata_addendum.json"
        ),
        "sha256": "11349678124371a42b32f8542b5a0cc5a672ff31398d6693e9ab29114734b73a",
    },
    "probe_manifest": {
        "path": Path(
            "outputs/probes/query_controller_action_v1_probe20_seed42_v4_3/manifest.json"
        ),
        "sha256": "65f9d7593116cb1bed13b336118355432458b3cd32dd014874d3dc47c9002eb1",
    },
    "probe_failure_addendum": {
        "path": Path(
            "outputs/probes/query_controller_action_v1_probe20_seed42_v4_3/"
            "metadata_addendum.json"
        ),
        "sha256": "8ee7b3dfdb5de3b051fbd26e80ff416f3b7d5b4ce47b628c74b8d7d8d4cb35c1",
    },
    "training_history": {
        "path": Path(
            "outputs/probes/query_controller_action_v1_probe20_seed42_v4_3/"
            "training_history.jsonl"
        ),
        "sha256": "1206cbc7af5ca9d54a701cd16b142320f7a81d4c65d482adcbf88a6d6bd8d113",
    },
}

SEALED_PARENT_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_cohort_freeze_"
    "dev30_prospective300_seed20260904_v1"
)
EXPECTED_SEALED_PARENT_SHA256 = {
    "report.json": "233b931716d96e0a6e40e0cb2c0e961a5c79c04884d6cac584c301e9ce9fe4b7",
    "manifest.json": "cda6525e1562697c31e17cb457280fe272de039ebebee23a2ddcabaa942730e6",
}
EXPECTED_SEALED_PROSPECTIVE_DECLARED_SHA256 = (
    "36b680cabef059dae7370bb131b1bafc0f120baf372f4e7666aa0e2d13b13c99"
)

CONSUMED_IDENTITY_PATHS = (
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
)
EXPECTED_CONSUMED_SHA256 = {
    CONSUMED_IDENTITY_PATHS[0]: "dedb1f90f815ca21efdb6980be37d4775c72d7c79812038e78bce1ecef4c0cb2",
    CONSUMED_IDENTITY_PATHS[1]: "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606",
    CONSUMED_IDENTITY_PATHS[2]: "7f4c63eb5589ce342a59e9942d986992ce357eb2888a46a0d6b04dc38794a9d8",
}

# Phase-0 has 41 tasks over 37 unique qids. Its task rows omit question text,
# so the versioned v7 identity file supplies the matching family projection.
PHASE0_TASK_PATH = Path(
    "outputs/audits/subquestion_decomposition_v8_phase0_v7_contract_"
    "counterfactual_v1/per_task_counterfactual.jsonl"
)
PHASE0_IDENTITY_SOURCE_PATH = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration/development.question_only.jsonl"
)
EXPECTED_PHASE0_SHA256 = {
    "tasks": "097e064500d8a540a5c9d3aeccc34b9e7a501537d97e42963b3b2b1584cfbd55",
    "identities": "7fd01236609ed010a42bb92d41a0e978323035ee715b1cc7a2047a1eebd2a8bc",
}


def _sha256_file(path: Path) -> str:
    if path.name == "prospective.identity_only.jsonl":
        raise PermissionError("sealed prospective900 must not be opened or hashed")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _implementation_locks(project_root: Path) -> dict[str, dict[str, str]]:
    locks: dict[str, dict[str, str]] = {}
    for role, relative in IMPLEMENTATION_PATHS.items():
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Controller implementation missing: {path}")
        locks[role] = {
            "path": relative.as_posix(),
            "sha256": _sha256_file(path),
        }
    return locks


def _validate_v4_3_failure_lineage(project_root: Path) -> dict[str, Any]:
    """Bind v4.4 to the preserved v4.3 FAIL_STOP without opening held-out data."""

    artifacts: dict[str, dict[str, str]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for role, lock in V4_3_FAILURE_LINEAGE.items():
        relative = Path(lock["path"])
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required v4.3 failure artifact missing: {path}")
        digest = _sha256_file(path)
        if digest != lock["sha256"]:
            raise ValueError(f"v4.3 failure-lineage hash drift: {role}")
        artifacts[role] = {"path": relative.as_posix(), "sha256": digest}
        if path.suffix == ".json":
            documents[role] = _load_object(path)

    protocol_addendum = documents["protocol_supersession_addendum"]
    probe_manifest = documents["probe_manifest"]
    probe_addendum = documents["probe_failure_addendum"]
    if (
        protocol_addendum.get("status")
        != "SUPERSEDED_AFTER_FAIL_STOP_CLEAN_RELOAD_DTYPE_COERCION_VALIDATOR_BUG"
        or (protocol_addendum.get("usage_policy") or {}).get("new_training_allowed")
        is not False
    ):
        raise ValueError("v4.3 protocol supersession record is not fail-closed")
    if probe_manifest.get("status") != "FAIL_STOP":
        raise ValueError("v4.3 probe manifest is not the preserved FAIL_STOP")
    if (
        probe_addendum.get("status")
        != "FAIL_STOP_PRESERVED_CLEAN_RELOAD_DTYPE_COERCION_VALIDATOR_BUG"
        or probe_addendum.get("original_result_upgraded_to_success") is not False
        or (probe_addendum.get("scientific_decision") or {}).get(
            "training_success_claim_allowed"
        )
        is not False
    ):
        raise ValueError("v4.3 probe failure addendum does not preserve the failure")

    history_path = project_root / V4_3_FAILURE_LINEAGE["training_history"]["path"]
    history = list(_read_jsonl(history_path))
    logged_steps = [int(row["step"]) for row in history if "loss" in row]
    if logged_steps != list(range(1, 21)):
        raise ValueError("v4.3 diagnostic history does not contain exact steps 1..20")
    return {
        "predecessor_protocol_version": "v4_3",
        "predecessor_terminal_status": "FAIL_STOP",
        "failure_class": "clean_reload_adapter_dtype_coercion_fp32_to_bf16",
        "failure_preserved_not_upgraded": True,
        "same_identity_selection_required": True,
        "same_action_record_bytes_required": True,
        "successor_single_implementation_variable": (
            "PEFT default FP32 autocast for clean single-adapter reload with "
            "torch.equal zero tolerance and saved/live/reloaded dtype telemetry"
        ),
        "confirmation_action_records_opened": False,
        "prospective_content_opened_or_hashed": False,
        "artifacts": artifacts,
    }


def _load_object(path: Path) -> dict[str, Any]:
    if path.name == "prospective.identity_only.jsonl":
        raise PermissionError("sealed prospective900 must not be opened")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.name == "prospective.identity_only.jsonl":
        raise PermissionError("sealed prospective900 must not be opened")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object at {path}:{line_number}")
            yield row


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if tuple(row) != IDENTITY_FIELDS or set(row) != set(IDENTITY_FIELDS):
                raise ValueError("identity row violates the exact field/order contract")
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _identity_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if tuple(row) != IDENTITY_FIELDS or set(row) != set(IDENTITY_FIELDS):
            raise ValueError("identity row violates the exact field/order contract")
        digest.update(
            (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _validate_sealed_parent(
    project_root: Path,
    *,
    parent_dir: Path,
    expected_hashes: Mapping[str, str] | None,
    expected_declared_sha256: str,
) -> dict[str, Any]:
    parent = project_root / parent_dir
    report_path = parent / "report.json"
    manifest_path = parent / "manifest.json"
    report_sha = _sha256_file(report_path)
    manifest_sha = _sha256_file(manifest_path)
    if expected_hashes is not None:
        if report_sha != expected_hashes["report.json"]:
            raise ValueError("sealed parent report lock drift")
        if manifest_sha != expected_hashes["manifest.json"]:
            raise ValueError("sealed parent manifest lock drift")
    report = _load_object(report_path)
    manifest = _load_object(manifest_path)
    checks = report.get("checks") or {}
    seal = report.get("prospective_seal") or {}
    if report.get("status") != "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE":
        raise ValueError("sealed parent status mismatch")
    if checks.get("raw_train_qid_overlap") != 0 or checks.get("raw_train_family_overlap") != 0:
        raise ValueError("parent does not prove prospective/raw-train disjointness")
    if seal.get("status") != "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT":
        raise ValueError("prospective seal is not closed")
    outputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
    }
    declared = outputs.get("prospective.identity_only.jsonl")
    if declared != expected_declared_sha256:
        raise ValueError("declared prospective SHA mismatch")
    return {
        "parent_directory": parent_dir.as_posix(),
        "parent_report_sha256": report_sha,
        "parent_manifest_sha256": manifest_sha,
        "prospective_relative_path": (
            parent_dir / "prospective.identity_only.jsonl"
        ).as_posix(),
        "prospective_declared_sha256": declared,
        "prospective_content_opened": False,
        "prospective_content_hashed": False,
        "prospective_unlocked": False,
        "raw_train_qid_overlap_reported": 0,
        "raw_train_family_overlap_reported": 0,
    }


def _identity_from_record(
    record: Mapping[str, Any],
    *,
    role: str,
    action_pair_sha256: str,
) -> dict[str, str]:
    question = str(record["state"]["original_question"])
    return {
        "dataset": str(record["dataset"]),
        "qid": str(record["qid"]),
        "question_key": str(record["question_key"]),
        "question": question,
        "question_sha256": str(record["question_sha256"]),
        "family_sha256": str(record["family_sha256"]),
        "split": role,
        "action_pair_sha256": action_pair_sha256,
    }


def _load_pair_hash_index(
    path: Path,
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, Any]]:
    """Load the candidate-side exact q1/q2 hash inventory."""

    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    counts = Counter()
    for row in _read_jsonl(path):
        if set(row) != PAIR_HASH_INDEX_FIELDS:
            raise ValueError(f"candidate pair-hash row has invalid schema: {path}")
        dataset = str(row["dataset"])
        qid = str(row["qid"])
        source_split = str(row["source_split"])
        family_hash = str(row["family_sha256"])
        pair_hash = str(row["action_pair_canonical_sha256"])
        if dataset not in ENABLED_DATASETS or source_split not in {"train", "dev"}:
            raise ValueError(f"candidate pair-hash identity invalid: {row}")
        if not qid or _SHA256_RE.fullmatch(family_hash) is None:
            raise ValueError(f"candidate pair-hash family/qid invalid: {row}")
        if _SHA256_RE.fullmatch(pair_hash) is None:
            raise ValueError(f"candidate action-pair hash malformed: {row}")
        key = (source_split, dataset, qid)
        if key in rows:
            raise ValueError(f"duplicate candidate action-pair hash: {key}")
        rows[key] = {str(key): str(value) for key, value in row.items()}
        counts[source_split] += 1
    if not rows:
        raise ValueError("candidate action-pair hash index is empty")
    return rows, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "rows": len(rows),
        "per_source_split": dict(sorted(counts.items())),
    }


def _load_action_candidates(
    path: Path,
    *,
    source_split: str,
    expected_pair_hashes: Mapping[tuple[str, str, str], Mapping[str, str]] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load an exact valid q1/q2 pool and project one identity per qid."""

    digest = _sha256_file(path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_examples: set[str] = set()
    rows = 0
    for row in _read_jsonl(path):
        validate_action_record(row, expected_split=source_split)
        dataset = str(row["dataset"])
        qid = str(row["qid"])
        if dataset not in ENABLED_DATASETS:
            raise ValueError(f"candidate pool contains disabled dataset: {dataset}")
        example_id = str(row["example_id"])
        if example_id in seen_examples:
            raise ValueError(f"duplicate candidate example_id: {example_id}")
        seen_examples.add(example_id)
        grouped[(dataset, qid)].append(row)
        rows += 1

    identities: dict[tuple[str, str], dict[str, Any]] = {}
    per_dataset = Counter()
    for key, pair in grouped.items():
        if len(pair) != 2 or {row["slot"] for row in pair} != {"q1", "q2_dynamic"}:
            raise ValueError(f"candidate qid does not have exact q1/q2 pair: {key}")
        by_slot = {str(row["slot"]): row for row in pair}
        q1, q2 = by_slot["q1"], by_slot["q2_dynamic"]
        identity_fields = (
            "dataset",
            "qid",
            "question_key",
            "question_sha256",
            "family_sha256",
        )
        if any(q1[field] != q2[field] for field in identity_fields):
            raise ValueError(f"candidate pair identity mismatch: {key}")
        if q1["state"]["original_question"] != q2["state"]["original_question"]:
            raise ValueError(f"candidate pair question mismatch: {key}")
        previous = q2["state"]["previous_actions"]
        if len(previous) != 1 or previous[0].get("query") != q1["target"]["query"]:
            raise ValueError(f"candidate pair q1/q2 linkage mismatch: {key}")
        observation = q2["state"]["verified_observations"][0]
        provenance = observation.get("provenance") or {}
        if (
            provenance.get("source") != "train_annotation_support"
            or not provenance.get("annotation_path")
            or not provenance.get("binding_method")
        ):
            raise ValueError(f"q2 intermediate is not passage-bound annotation: {key}")
        source_pair_hash = canonical_action_pair_sha256(pair)
        if expected_pair_hashes is not None:
            pair_lock = expected_pair_hashes.get((source_split, key[0], key[1]))
            if pair_lock is None:
                raise ValueError(f"candidate action-pair hash is missing: {key}")
            if pair_lock["family_sha256"] != q1["family_sha256"]:
                raise ValueError(f"candidate action-pair family lock mismatch: {key}")
            if pair_lock["action_pair_canonical_sha256"] != source_pair_hash:
                raise ValueError(f"candidate action-pair content hash mismatch: {key}")
        role_pair_hashes = {source_split: source_pair_hash}
        if source_split == "train":
            role_pair_hashes["confirmation"] = canonical_action_pair_sha256(
                pair, split_override="confirmation"
            )
        identity = _identity_from_record(
            q1, role=source_split, action_pair_sha256=source_pair_hash
        )
        identity["_action_pair_sha256_by_role"] = role_pair_hashes
        if identity["question_sha256"] != question_sha256(identity["question"]):
            raise ValueError(f"candidate question hash mismatch: {key}")
        if identity["family_sha256"] != family_sha256(identity["question"]):
            raise ValueError(f"candidate family hash mismatch: {key}")
        identities[key] = identity
        per_dataset[identity["dataset"]] += 1
    return identities, {
        "path": str(path),
        "sha256": digest,
        "source_planner_split": source_split,
        "action_rows": rows,
        "strict_valid_qid_pairs": len(identities),
        "strict_valid_families_per_dataset": {
            dataset: len(
                {
                    row["family_sha256"]
                    for row in identities.values()
                    if row["dataset"] == dataset
                }
            )
            for dataset in ENABLED_DATASETS
        },
        "strict_valid_qids_per_dataset": {
            dataset: per_dataset[dataset] for dataset in ENABLED_DATASETS
        },
    }


def _validate_candidate_v4_1_evidence(
    candidate_dir: Path, *, expected_hashes: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the hardened, pair-hashed candidate release and its audits."""

    paths = {
        "report": candidate_dir / "report.json",
        "manifest": candidate_dir / "manifest.json",
        "hardening_audit": candidate_dir / "hardening_audit.json",
        "metadata_addendum": candidate_dir / "metadata_addendum.json",
    }
    digests = {role: _sha256_file(path) for role, path in paths.items()}
    for role, digest in digests.items():
        if digest != expected_hashes[role]:
            raise ValueError(f"candidate v4_1 evidence lock drift: {role}")

    report = _load_object(paths["report"])
    manifest = _load_object(paths["manifest"])
    audit = _load_object(paths["hardening_audit"])
    addendum = _load_object(paths["metadata_addendum"])
    if (
        report.get("schema_version")
        != "query-controller-action-candidate-pool-v2-pair-hashed"
        or report.get("status")
        != "COMPLETE_V4_HARDENED_ELIGIBLE_CANDIDATES_NOT_COHORT_NOT_TRAINED"
        or report.get("experiment_id")
        != "QUERY-CONTROLLER-ACTION-V1-SEED42-ELIGIBLE-CANDIDATES-V4-1"
    ):
        raise ValueError("candidate v4_1 report identity/status mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status")
        != "COMPLETE_V4_HARDENED_ELIGIBLE_CANDIDATES_NOT_COHORT_NOT_TRAINED"
    ):
        raise ValueError("candidate v4_1 manifest status mismatch")
    report_outputs = report.get("outputs") or {}
    output_roles = {
        "train.jsonl": "train",
        "dev.jsonl": "dev",
        "pair_hashes.jsonl": "pair_hashes",
        "rejected.jsonl": "rejected",
    }
    if any(
        (report_outputs.get(filename) or {}).get("sha256")
        != expected_hashes[role]
        for filename, role in output_roles.items()
    ):
        raise ValueError("candidate v4_1 report does not bind every output hash")
    record_validation = audit.get("record_validation") or {}
    future_policy = audit.get("future_secret_policy") or {}
    duplicate_article = audit.get("adjacent_duplicate_article_audit") or {}
    pair_audit = audit.get("pair_hash_audit") or {}
    exception_audit = audit.get("exception_policy_audit") or {}
    if (
        audit.get("schema_version")
        != "query-controller-action-candidate-pool-v4-1-hardening-audit-1"
        or audit.get("status")
        != "PASS_HARDENING_AUDIT_ELIGIBLE_CANDIDATES_NOT_COHORT_NOT_TRAINED"
        or audit.get("overall_gate") != "PASS"
        or audit.get("selection_authority")
        != "NONE_PROTOCOL_FREEZER_MUST_SELECT_IDENTITIES"
        or audit.get("training_started") is not False
        or record_validation.get("action_rows") != 21286
        or record_validation.get("action_pairs") != 10643
        or record_validation.get("canonical_schema_errors") != 0
        or record_validation.get("invalid_pair_shapes") != 0
        or future_policy.get("recomputed_selected_leaks") != 0
        or duplicate_article.get("residual_controller_generated_or_dynamic_state_fields")
        != 0
        or duplicate_article.get("gate") != "PASS"
        or pair_audit.get("index_exact_five_field_schema_errors") != 0
        or pair_audit.get("duplicate_keys") != 0
        or pair_audit.get("missing_pair_keys") != 0
        or pair_audit.get("extra_pair_keys") != 0
        or pair_audit.get("canonical_full_record_recompute_mismatches") != 0
        or pair_audit.get("provenance_included_in_hash") is not True
        or pair_audit.get("gate") != "PASS"
        or exception_audit.get("unknown_exception_behavior") != "FAIL_STOP_PROPAGATE"
        or exception_audit.get("gate") != "PASS"
    ):
        raise ValueError("candidate v4_1 hardening audit gate failed")
    if (
        addendum.get("status")
        != "PASS_V4_1_HARDENED_PAIR_HASHED_ELIGIBLE_CANDIDATES_NOT_COHORT_NOT_TRAINED"
        or addendum.get("base_manifest_sha256") != expected_hashes["manifest"]
        or addendum.get("base_report_sha256") != expected_hashes["report"]
        or (addendum.get("hardening_audit") or {}).get("sha256")
        != expected_hashes["hardening_audit"]
        or (addendum.get("hardening_audit") or {}).get("overall_gate") != "PASS"
        or addendum.get("canonical_pair_hash_recompute_mismatches") != 0
        or addendum.get("selected_final_or_future_secret_leaks") != 0
        or addendum.get("unknown_exception_policy") != "FAIL_STOP_PROPAGATE"
        or addendum.get("candidate_pool_remains_unselected") is not True
        or addendum.get("protocol_freeze_required_before_release") is not True
        or addendum.get("training_started") is not False
    ):
        raise ValueError("candidate v4_1 addendum gate failed")
    return {
        "candidate_directory": str(candidate_dir),
        "report": {"path": str(paths["report"]), "sha256": digests["report"]},
        "manifest": {
            "path": str(paths["manifest"]),
            "sha256": digests["manifest"],
        },
        "hardening_audit": {
            "path": str(paths["hardening_audit"]),
            "sha256": digests["hardening_audit"],
            "status": audit["status"],
            "canonical_action_rows": 21286,
            "canonical_action_pairs": 10643,
            "selected_final_or_future_secret_leaks": 0,
            "pair_hash_recompute_mismatches": 0,
            "generated_or_dynamic_state_duplicate_articles": 0,
            "gate": "PASS",
        },
        "append_only_validation": {
            "path": str(paths["metadata_addendum"]),
            "sha256": digests["metadata_addendum"],
            "status": addendum["status"],
            "gate": "PASS",
        },
    }


def _load_identity_registry(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    identities: dict[tuple[str, str], dict[str, str]] = {}
    rows = 0
    for row in _read_jsonl(path):
        dataset = str(row.get("dataset") or "").strip()
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        if not dataset or not qid or not question:
            raise ValueError(f"incomplete consumed identity: {path}")
        question_hash = question_sha256(question)
        if row.get("question_sha256") not in (None, question_hash):
            raise ValueError(f"consumed question hash mismatch: {dataset}::{qid}")
        family_hash = family_sha256(question)
        if row.get("family_sha256") not in (None, family_hash):
            raise ValueError(f"consumed family hash mismatch: {dataset}::{qid}")
        item = {
            "dataset": dataset,
            "qid": qid,
            "question_sha256": question_hash,
            "family_sha256": family_hash,
        }
        key = (dataset, qid)
        if key in identities and identities[key] != item:
            raise ValueError(f"conflicting consumed identity: {dataset}::{qid}")
        identities[key] = item
        rows += 1
    return identities, rows


def _load_consumed(
    project_root: Path,
    *,
    identity_paths: Sequence[Path],
    expected_hashes: Mapping[Path, str] | None,
    phase0_task_path: Path,
    phase0_identity_source_path: Path,
    expected_phase0_hashes: Mapping[str, str] | None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[dict[str, Any]]]:
    qids: set[tuple[str, str]] = set()
    families: set[tuple[str, str]] = set()
    inventory: list[dict[str, Any]] = []
    for relative in identity_paths:
        path = project_root / relative
        digest = _sha256_file(path)
        if expected_hashes is not None and digest != expected_hashes[relative]:
            raise ValueError(f"consumed identity lock drift: {relative}")
        identities, rows = _load_identity_registry(path)
        qids.update(identities)
        families.update(
            (item["dataset"], item["family_sha256"]) for item in identities.values()
        )
        inventory.append(
            {
                "role": "consumed_identity_registry",
                "path": relative.as_posix(),
                "sha256": digest,
                "rows": rows,
                "unique_qids": len(identities),
            }
        )

    task_path = project_root / phase0_task_path
    source_path = project_root / phase0_identity_source_path
    task_sha = _sha256_file(task_path)
    source_sha = _sha256_file(source_path)
    if expected_phase0_hashes is not None:
        if task_sha != expected_phase0_hashes["tasks"]:
            raise ValueError("Phase0 task registry drift")
        if source_sha != expected_phase0_hashes["identities"]:
            raise ValueError("Phase0 identity projection drift")
    phase0_keys: dict[tuple[str, str], str] = {}
    phase0_rows = 0
    for row in _read_jsonl(task_path):
        key = (str(row.get("dataset") or ""), str(row.get("qid") or ""))
        declared = str(row.get("question_sha256") or "")
        if not all(key) or not declared:
            raise ValueError("invalid Phase0 task identity")
        old = phase0_keys.get(key)
        if old is not None and old != declared:
            raise ValueError(f"conflicting Phase0 task identity: {key}")
        phase0_keys[key] = declared
        phase0_rows += 1
    projected, _ = _load_identity_registry(source_path)
    missing = sorted(set(phase0_keys) - set(projected))
    if missing:
        raise ValueError(f"Phase0 identity projection missing qids: {missing[:3]}")
    if phase0_rows != 41 or len(phase0_keys) != 37:
        raise ValueError(
            f"Phase0 registry shape drift: rows={phase0_rows}, unique={len(phase0_keys)}"
        )
    for key, declared in phase0_keys.items():
        identity = projected[key]
        if identity["question_sha256"] != declared:
            raise ValueError(f"Phase0 question hash mismatch: {key}")
        qids.add(key)
        families.add((identity["dataset"], identity["family_sha256"]))
    inventory.append(
        {
            "role": "consumed_phase0_37_projection",
            "task_path": phase0_task_path.as_posix(),
            "task_sha256": task_sha,
            "identity_source_path": phase0_identity_source_path.as_posix(),
            "identity_source_sha256": source_sha,
            "task_rows": phase0_rows,
            "unique_qids": len(phase0_keys),
            "unique_dataset_scoped_families": len(
                {
                    (projected[key]["dataset"], projected[key]["family_sha256"])
                    for key in phase0_keys
                }
            ),
        }
    )
    return qids, families, inventory


def _selection_rank(row: Mapping[str, Any], *, role: str) -> str:
    payload = "\0".join(
        (SELECTION_SALT, role, row["dataset"], row["family_sha256"], row["qid"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_for_role(row: Mapping[str, Any], *, role: str) -> dict[str, str]:
    role_hashes = row.get("_action_pair_sha256_by_role")
    if not isinstance(role_hashes, Mapping) or role not in role_hashes:
        raise ValueError(f"missing action-pair hash for role={role}")
    return {
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question_key": row["question_key"],
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "family_sha256": row["family_sha256"],
        "split": role,
        "action_pair_sha256": str(role_hashes[role]),
    }


def _select_role(
    pool: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    dataset: str,
    role: str,
    quota: int,
    excluded_qids: set[tuple[str, str]],
    excluded_families: set[tuple[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for row in pool.values():
        if row["dataset"] != dataset:
            continue
        qid_key = (dataset, row["qid"])
        family_key = (dataset, row["family_sha256"])
        if qid_key in excluded_qids:
            stats["excluded_qid"] += 1
            continue
        if family_key in excluded_families:
            stats["excluded_family"] += 1
            continue
        by_family[row["family_sha256"]].append(row)
    one_per_family = [
        min(values, key=lambda item: _selection_rank(item, role=role))
        for _, values in sorted(by_family.items())
    ]
    ordered = sorted(one_per_family, key=lambda item: _selection_rank(item, role=role))
    if len(ordered) < quota:
        raise ValueError(
            f"{dataset}/{role}: only {len(ordered)} strict valid families; need {quota}"
        )
    selected = [_identity_for_role(item, role=role) for item in ordered[:quota]]
    stats["eligible_unique_families_before_quota"] = len(ordered)
    stats["selected"] = len(selected)
    return selected, dict(stats)


def _protocol_body(
    *,
    sealed_lock: Mapping[str, Any],
    cohort_locks: Mapping[str, Any],
    candidate_inventory: Sequence[Mapping[str, Any]],
    candidate_v4_1_evidence: Mapping[str, Any] | None,
    consumed_inventory: Sequence[Mapping[str, Any]],
    implementation_locks: Mapping[str, Mapping[str, str]],
    predecessor_failure_lineage: Mapping[str, Any],
    identity_continuity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "scope": "2DATASET_EXACT_SUPERVISION_DEVELOPMENT_AND_PROBE_NO_OUTCOME_GOLD",
        "enabled_training_datasets": list(ENABLED_DATASETS),
        "dataset_readiness": {
            "2wikimultihopqa": {
                "status": "AUTHORIZED_EXACT_TEXT_SUPERVISION_ENGINEERING_RELEASE",
                "source": "train_metadata.evidences",
            },
            "musique": {
                "status": "AUTHORIZED_EXACT_TEXT_SUPERVISION_ENGINEERING_RELEASE",
                "source": "train_metadata.question_decomposition",
            },
            "hotpotqa": {
                "status": "NOT_INCLUDED_LABEL_COVERAGE_UNKNOWN",
                "source": "supporting_facts_do_not_supply_an_exact_subquery_chain",
                "claim_boundary": "No three-dataset Controller release or Hotpot generality claim.",
            },
        },
        "cohort": {
            "eligibility_before_selection": True,
            "eligibility": (
                "central-validator-valid exact q1+q2_dynamic pair; strict linear two-step "
                "old-planner row; q2 intermediate annotation uniquely passage-bound"
            ),
            "selection": "deterministic_SHA256_one_qid_per_dataset_scoped_lexical_family",
            "selection_order": [
                "dev60 from old planner dev",
                "confirmation30 from old planner train, held out first",
                "train600 from old planner train after dev/confirmation family exclusion",
            ],
            "source_split_for_role": dict(SOURCE_SPLIT_FOR_ROLE),
            "confirmation_role": (
                "train-side family-disjoint mechanism confirmation; not external/final"
            ),
            "family_version": FAMILY_VERSION,
            "split_sizes_per_enabled_dataset": dict(SPLIT_SIZES),
            "identity_fields": list(IDENTITY_FIELDS),
            "qid_overlap_between_splits": 0,
            "family_overlap_between_splits": 0,
            "cohort_locks": dict(cohort_locks),
            "v4_3_identity_continuity": dict(identity_continuity),
        },
        "inputs": {
            "strict_valid_action_candidate_pools": list(candidate_inventory),
            "candidate_v4_1_hardening_evidence": (
                dict(candidate_v4_1_evidence)
                if candidate_v4_1_evidence is not None
                else None
            ),
            "consumed_controller_cohorts": list(consumed_inventory),
            "sealed_prospective_parent_metadata_only": dict(sealed_lock),
            "v4_3_failure_lineage": dict(predecessor_failure_lineage),
        },
        "implementation_locks": {
            key: dict(value) for key, value in sorted(implementation_locks.items())
        },
        "gold_boundary": {
            "source_train_files_may_contain_gold": True,
            "freezer_direct_gold_final_answer_accessed": False,
            "upstream_candidate_builder_gold_final_answer_use": "leakage_exclusion_only",
            "candidate_eligibility_is_gold_screened": True,
            "construction_train_intermediate_annotation_used": True,
            "q1_train_intermediate_annotation_used": False,
            "q2_train_intermediate_annotation_used": True,
            "training_q2_model_visible_intermediate": (
                "annotation-derived_but_passage-bound; includes exact supporting excerpt "
                "and provenance; this is not annotation-free supervision"
            ),
            "training_model_visible_forbidden": [
                "gold final answer",
                "future-hop answer/tail",
                "unbound intermediate annotation",
                "raw decomposition/evidence objects",
                "evaluation Gold",
            ],
            "runtime_q2_model_visible_intermediate": (
                "Reader-predicted and retrieved-passage-bound; train annotation forbidden"
            ),
            "gold_final_answer_visible_rate": 0.0,
            "evaluation_gold_access_rate": 0.0,
            "confirmation_selection_reads_mechanical_label_eligibility": True,
            "confirmation_identity_output_contains_targets": False,
            "answer_scoring_authorized": False,
        },
        "action_contract": {
            "schema_version": "query-controller-action-v1",
            "allowed_release_splits": ["train", "dev", "confirmation"],
            "trainer_allowed_splits": ["train", "dev"],
            "source_action": "text",
            "dual_source_routing": False,
            "anchor_nullable": True,
            "pid_nullable": True,
            "q1_previous_actions": 0,
            "q1_verified_observations": 0,
            "q2_previous_actions": 1,
            "q2_verified_observations": 1,
            "target_format": "exact JSON object; retrieve action only",
            "identity_lock_binds_canonical_q1_q2_action_pair_sha256": True,
            "nested_model_visible_state_fields_are_exact": True,
            "previous_action_fields": sorted(PREVIOUS_ACTION_FIELDS),
            "observation_fields": sorted(OBSERVATION_FIELDS),
            "observation_provenance_fields": sorted(
                OBSERVATION_PROVENANCE_FIELDS
            ),
            "observation_provenance_source": "train_annotation_support",
            "observation_annotation_paths": sorted(OBSERVATION_ANNOTATION_PATHS),
            "observation_binding_methods": sorted(OBSERVATION_BINDING_METHODS),
        },
        "data_release_gates": {
            "exact_train_qids_per_enabled_dataset": 600,
            "exact_dev_qids_per_enabled_dataset": 60,
            "exact_confirmation_qids_per_enabled_dataset": 30,
            "exact_actions_per_qid": 2,
            "eligibility_applied_before_identity_selection": True,
            "action_schema_valid_rate": 1.0,
            "question_identity_join_rate": 1.0,
            "query_nonrepeat_rate": 1.0,
            "placeholder_free_rate": 1.0,
            "dependency_closed_rate": 1.0,
            "source_action_text_rate": 1.0,
            "state_use_valid_rate": 1.0,
            "gold_boundary_valid_rate": 1.0,
            "duplicate_example_id_count": 0,
            "lock_consumption_policy": "exact join; construction failure aborts; no replacement",
            "action_pair_hash_match_rate": 1.0,
            "v4_3_identity_lock_byte_match_rate": 1.0,
            "v4_3_action_record_bytes_required": True,
            "v4_3_fail_stop_preserved": True,
            "hotpot_policy": (
                "report UNKNOWN label coverage; absence does not fail this two-dataset release"
            ),
        },
        "training_contract": {
            "authorization": "20-step probe only after every data-release gate passes",
            "initialization": "base Llama-3-8B instruct; independent Controller LoRA",
            "forbidden_initialization": [
                "Strong-SFT Reader adapter",
                "historical query-planner adapter",
            ],
            "trainer_input_splits": ["train", "dev"],
            "confirmation_read_by_trainer": False,
            "objective": "assistant target JSON tokens only; all state/observation tokens masked",
            "target_truncation": "forbidden",
            "checkpoint_selection": "fixed final probe step; no confirmation access",
            "probe_optimizer_steps": 20,
            "runtime_config": {
                "phase": "probe",
                "experiment_id": "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4",
                "model": {
                    "base_model": "llama3-8B-instruct",
                    "method": "qlora",
                    "initialization": "base_instruct",
                    "init_adapter_path": None,
                    "load_in_4bit": True,
                    "dtype": "bf16",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "data": {
                    "allowed_source_actions": ["text"],
                    "train_quotas": {
                        dataset: {"q1": 32, "q2_dynamic": 32}
                        for dataset in ENABLED_DATASETS
                    },
                    "dev_quotas": {
                        dataset: {"q1": 16, "q2_dynamic": 16}
                        for dataset in ENABLED_DATASETS
                    },
                },
                "training": {
                    "seed": 42,
                    "max_seq_length": 1024,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "learning_rate": 0.0001,
                    "warmup_ratio": 0.05,
                    "lr_scheduler_type": "cosine",
                    "num_train_epochs": 1,
                    "max_steps": 20,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "logging_steps": 1,
                    "eval_strategy": "no",
                    "eval_steps": 20,
                    "save_strategy": "no",
                    "save_steps": 20,
                    "save_total_limit": 1,
                    "verify_saved_adapter_reload": True,
                },
            },
            "probe_gates": {
                "cuda_runtime_available": True,
                "finite_loss_rate": 1.0,
                "nonzero_supervised_token_rate": 1.0,
                "nonzero_trainable_gradient_observed": True,
                "adapter_save_reload_exact": True,
                "adapter_save_fidelity_exact": True,
                "adapter_clean_reload_single_adapter": True,
                "adapter_clean_reload_tensor_exact": True,
                "adapter_dtype_inventories_recorded": True,
                "adapter_saved_live_clean_reload_dtype_inventories_equal": True,
                "oom_count": 0,
                "nan_or_inf_count": 0,
                "dev_only_validation": True,
                "confirmation_open_count": 0,
                "prospective_open_or_hash_count": 0,
            },
        },
        "future_gold_free_mechanism_gate": {
            "authorization": (
                "not granted by this protocol; requires frozen adapter and a new append-only "
                "runtime protocol"
            ),
            "cohort": "train-side confirmation30 per enabled dataset",
            "runtime_intermediate_source": "Reader-predicted and passage-bound",
            "annotation_derived_intermediate_visible_at_runtime": False,
            "gold_access_count": 0,
            "runtime_error_count": 0,
            "active_adapter_hash_match_rate": 1.0,
            "cache_and_call_accounting_rate": 1.0,
            "q1_schema_valid_rate_min_each_dataset": 0.97,
            "q2_dynamic_schema_valid_rate_min_each_dataset": 0.95,
            "query_nonrepeat_rate": 1.0,
            "placeholder_free_rate": 1.0,
            "a1_admissible_rate_min_each_dataset": 0.40,
            "dynamic_state_binding_integrity_rate": 1.0,
            "dynamic_transition_rate_min_each_dataset": 0.50,
            "final_passage_budget_and_unique_rate": 1.0,
            "fallback_byte_identity_rate": 1.0,
            "reader_one_shot_regression_identity_rate": 1.0,
        },
        "probe_evaluation_contract": {
            "authorization": "post_probe_dev_teacher_forced_only",
            "cohort_role": "dev",
            "input_split": "dev",
            "datasets": list(ENABLED_DATASETS),
            "slots": ["q1", "q2_dynamic"],
            "confirmation_access": False,
            "prospective_access": False,
            "exact_qids_per_enabled_dataset": 60,
            "exact_action_rows_per_enabled_dataset": 120,
            "exact_actions": 240,
            "state_source": "annotation_derived_but_passage_bound",
            "runtime_reader_predicted": False,
            "required_probe_artifacts": {
                "experiment_id": "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4",
                "manifest_status": "COMPLETE",
                "completed_optimizer_steps": 20,
                "training_manifest_sha256": "required_external_eval_lock",
                "adapter_sha256": "required_external_eval_lock",
                "asset_lock_lineage_match": True,
            },
            "decoding": {
                "strategy": "greedy",
                "do_sample": False,
                "temperature": 0.0,
                "seed": 42,
                "batch_size": 4,
                "max_input_tokens": 1024,
                "max_new_tokens": 192,
                "base_model": "llama3-8B-instruct",
                "dtype": "bf16",
                "load_in_4bit": True,
            },
            "mechanism_gates": {
                "identity_join_rate": 1.0,
                "query_nonrepeat_rate": 1.0,
                "placeholder_free_rate": 1.0,
                "dependency_closed_rate_min_each_dataset_slot": {
                    "q1": 0.97,
                    "q2_dynamic": 0.95,
                },
                "state_use_valid_rate_min_each_dataset_slot": {
                    "q1": 0.97,
                    "q2_dynamic": 0.95,
                },
                "schema_valid_rate_min_each_dataset_slot": {
                    "q1": 0.97,
                    "q2_dynamic": 0.95,
                },
            },
            "outcome_metrics_authorized": {
                "em": False,
                "f1": False,
                "ihr": False,
            },
            "status_semantics": {
                "generation_complete_status": "COMPLETE_GENERATION_NOT_MECHANISM_PASS",
                "mechanism_pass_requires_separate_scorer_gate": True,
                "generation_complete_implies_mechanism_pass": False,
            },
            "scientific_boundary": (
                "Teacher-forced action mechanics on the frozen release dev split only; "
                "not Reader-predicted runtime and not retrieval or QA utility."
            ),
        },
        "outcome_unlock_rule": {
            "em_f1_authorized_now": False,
            "required_sequence": [
                "pass every data-release gate",
                "pass the 20-step training probe gates",
                "train and freeze a separately authorized Controller adapter",
                "freeze a new Gold-free runtime protocol",
                "pass every train-side confirmation Gold-free mechanism gate",
                "freeze prediction bytes and hashes",
                "obtain separate authorization for independent outcome scoring",
            ],
            "sealed_prospective900": (
                "remains unopened and unhashed; this protocol implements no unlock"
            ),
            "ihr": "not authorized before frozen prospective EM/F1",
        },
        "scientific_boundary": (
            "This is a strict-eligible, two-dataset exact-text Controller engineering "
            "protocol. The train-side confirmation is not an external test. This artifact "
            "is not a trained model, an EM/F1 result, a three-dataset release, or evidence "
            "that the Controller improves retrieval."
        ),
    }


def run_freeze(
    *,
    project_root: Path = PROJECT_ROOT,
    candidate_paths: Mapping[str, Path] | None = None,
    expected_candidate_hashes: Mapping[str, str] | None = EXPECTED_CANDIDATE_SHA256,
    output_dir: Path | None = None,
    split_sizes: Mapping[str, int] = SPLIT_SIZES,
    consumed_identity_paths: Sequence[Path] = CONSUMED_IDENTITY_PATHS,
    expected_consumed_hashes: Mapping[Path, str] | None = EXPECTED_CONSUMED_SHA256,
    phase0_task_path: Path = PHASE0_TASK_PATH,
    phase0_identity_source_path: Path = PHASE0_IDENTITY_SOURCE_PATH,
    expected_phase0_hashes: Mapping[str, str] | None = EXPECTED_PHASE0_SHA256,
    sealed_parent_dir: Path = SEALED_PARENT_DIR,
    expected_parent_hashes: Mapping[str, str] | None = EXPECTED_SEALED_PARENT_SHA256,
    expected_sealed_declared_sha256: str = EXPECTED_SEALED_PROSPECTIVE_DECLARED_SHA256,
    require_v4_3_failure_lineage: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    destination = (output_dir or project_root / DEFAULT_OUTPUT_DIR).resolve()
    if destination.exists():
        raise FileExistsError(f"append-only output already exists: {destination}")
    if set(split_sizes) != set(SPLIT_SIZES) or any(
        type(value) is not int or value <= 0 for value in split_sizes.values()
    ):
        raise ValueError("split_sizes must define positive train/dev/confirmation quotas")
    paths = candidate_paths or {
        "train": DEFAULT_CANDIDATE_DIR / "train.jsonl",
        "dev": DEFAULT_CANDIDATE_DIR / "dev.jsonl",
    }
    if set(paths) != {"train", "dev"}:
        raise ValueError("candidate_paths must contain exactly train and dev")
    predecessor_failure_lineage = (
        _validate_v4_3_failure_lineage(project_root)
        if require_v4_3_failure_lineage
        else {
            "predecessor_protocol_version": "TEST_ONLY",
            "failure_preserved_not_upgraded": True,
            "same_identity_selection_required": False,
            "confirmation_action_records_opened": False,
            "prospective_content_opened_or_hashed": False,
            "artifacts": {},
        }
    )
    resolved_candidate_paths = {
        split: path if path.is_absolute() else project_root / path
        for split, path in paths.items()
    }
    if len({path.parent.resolve() for path in resolved_candidate_paths.values()}) != 1:
        raise ValueError("train/dev candidate pools must share one versioned directory")
    candidate_dir = resolved_candidate_paths["train"].parent
    pair_hashes: dict[tuple[str, str, str], dict[str, str]] | None = None
    pair_hash_inventory: dict[str, Any] | None = None
    if expected_candidate_hashes is not None and "pair_hashes" in expected_candidate_hashes:
        pair_hashes, pair_hash_inventory = _load_pair_hash_index(
            candidate_dir / "pair_hashes.jsonl"
        )
        if pair_hash_inventory["sha256"] != expected_candidate_hashes["pair_hashes"]:
            raise ValueError("candidate action-pair hash index lock drift")

    sealed_lock = _validate_sealed_parent(
        project_root,
        parent_dir=sealed_parent_dir,
        expected_hashes=expected_parent_hashes,
        expected_declared_sha256=expected_sealed_declared_sha256,
    )
    consumed_qids, consumed_families, consumed_inventory = _load_consumed(
        project_root,
        identity_paths=consumed_identity_paths,
        expected_hashes=expected_consumed_hashes,
        phase0_task_path=phase0_task_path,
        phase0_identity_source_path=phase0_identity_source_path,
        expected_phase0_hashes=expected_phase0_hashes,
    )
    pools: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    candidate_inventory: list[dict[str, Any]] = []
    for source_split in ("train", "dev"):
        path = resolved_candidate_paths[source_split]
        pool, inventory = _load_action_candidates(
            path,
            source_split=source_split,
            expected_pair_hashes=pair_hashes,
        )
        if (
            expected_candidate_hashes is not None
            and inventory["sha256"] != expected_candidate_hashes[source_split]
        ):
            raise ValueError(f"strict candidate pool lock drift: {source_split}")
        pools[source_split] = pool
        candidate_inventory.append(inventory)
    if pair_hashes is not None:
        expected_pair_rows = sum(len(pool) for pool in pools.values())
        if len(pair_hashes) != expected_pair_rows:
            raise ValueError(
                "candidate action-pair hash index has missing or extra identities: "
                f"index={len(pair_hashes)}, candidates={expected_pair_rows}"
            )
        candidate_inventory.append(
            {"role": "candidate_action_pair_hash_index", **dict(pair_hash_inventory or {})}
        )
    candidate_v4_1_evidence = None
    if expected_candidate_hashes is not None:
        candidate_v4_1_evidence = _validate_candidate_v4_1_evidence(
            resolved_candidate_paths["train"].parent,
            expected_hashes=expected_candidate_hashes,
        )
    # ``project_root`` is injectable for synthetic data fixtures.  Code locks
    # always bind the actual repository containing this freezer, never a test
    # fixture's temporary data root.
    implementation_locks = _implementation_locks(PROJECT_ROOT)

    selected: dict[str, list[dict[str, str]]] = {role: [] for role in SPLIT_SIZES}
    selection_stats: dict[str, Any] = {}
    selected_qids = set(consumed_qids)
    selected_families = set(consumed_families)

    # Preserve original planner split. Confirmation is held out from train
    # before selecting train because old dev has only 61 strict-valid 2Wiki
    # families and cannot supply dev60 + confirmation30.
    for role in ("dev", "confirmation", "train"):
        source_split = SOURCE_SPLIT_FOR_ROLE[role]
        for dataset in ENABLED_DATASETS:
            rows, stats = _select_role(
                pools[source_split],
                dataset=dataset,
                role=role,
                quota=int(split_sizes[role]),
                excluded_qids=selected_qids,
                excluded_families=selected_families,
            )
            selected[role].extend(rows)
            selected_qids.update((row["dataset"], row["qid"]) for row in rows)
            selected_families.update(
                (row["dataset"], row["family_sha256"]) for row in rows
            )
            selection_stats[f"{dataset}::{role}"] = {
                **stats,
                "source_planner_split": source_split,
            }

    for role in selected:
        selected[role].sort(key=lambda row: (row["dataset"], row["qid"]))
    all_rows = [row for rows in selected.values() for row in rows]
    qid_keys = [(row["dataset"], row["qid"]) for row in all_rows]
    family_keys = [(row["dataset"], row["family_sha256"]) for row in all_rows]
    if len(qid_keys) != len(set(qid_keys)):
        raise ValueError("selected splits have qid overlap")
    if len(family_keys) != len(set(family_keys)):
        raise ValueError("selected splits have family overlap")
    if any(key in consumed_qids for key in qid_keys):
        raise ValueError("selected qid overlaps a consumed controller cohort")
    if any(key in consumed_families for key in family_keys):
        raise ValueError("selected family overlaps a consumed controller cohort")

    selected_identity_hashes = {
        role: _identity_rows_sha256(rows) for role, rows in selected.items()
    }
    canonical_successor = require_v4_3_failure_lineage and dict(split_sizes) == SPLIT_SIZES
    if canonical_successor and selected_identity_hashes != EXPECTED_V4_3_IDENTITY_SHA256:
        raise ValueError(
            "v4.4 identity selection differs from frozen v4.3 identities: "
            f"actual={selected_identity_hashes}"
        )
    identity_continuity = {
        "selection_salt": SELECTION_SALT,
        "predecessor_identity_sha256": dict(EXPECTED_V4_3_IDENTITY_SHA256),
        "successor_identity_sha256": selected_identity_hashes,
        "byte_identical_to_v4_3": canonical_successor,
        "comparison_used_identity_hashes_only": True,
        "confirmation_action_records_opened": False,
    }

    destination.mkdir(parents=True, exist_ok=False)
    cohort_locks: dict[str, Any] = {}
    for role, rows in selected.items():
        path = destination / f"{role}.identity_only.jsonl"
        _write_jsonl(path, rows)
        counts = Counter(row["dataset"] for row in rows)
        cohort_locks[role] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "rows": len(rows),
            "per_dataset": {dataset: counts[dataset] for dataset in ENABLED_DATASETS},
            "source_planner_split": SOURCE_SPLIT_FOR_ROLE[role],
            "gold_fields_emitted": False,
        }

    protocol = _protocol_body(
        sealed_lock=sealed_lock,
        cohort_locks=cohort_locks,
        candidate_inventory=candidate_inventory,
        candidate_v4_1_evidence=candidate_v4_1_evidence,
        consumed_inventory=consumed_inventory,
        implementation_locks=implementation_locks,
        predecessor_failure_lineage=predecessor_failure_lineage,
        identity_continuity=identity_continuity,
    )
    # Test-sized freezes retain the algorithm but must not masquerade as the
    # canonical 600/60/30 release.
    protocol["cohort"]["materialized_split_sizes_per_enabled_dataset"] = dict(
        split_sizes
    )
    protocol["protocol_body_canonical_sha256"] = _canonical_sha256(protocol)
    protocol_path = destination / "protocol.json"
    _write_json(protocol_path, protocol)
    report = {
        "schema_version": "query-controller-v1-pilot-freeze-report-4.4",
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "enabled_training_datasets": list(ENABLED_DATASETS),
        "hotpotqa_status": "NOT_INCLUDED_LABEL_COVERAGE_UNKNOWN",
        "checks": {
            "eligibility_before_selection": True,
            "exact_pair_candidate_validation": True,
            "qid_overlap_between_splits": 0,
            "family_overlap_between_splits": 0,
            "consumed_qid_overlap": 0,
            "consumed_family_overlap": 0,
            "phase0_unique_qids_registered": 37,
            "freezer_direct_gold_final_answer_accessed": False,
            "upstream_candidate_builder_gold_final_answer_use": "leakage_exclusion_only",
            "candidate_eligibility_is_gold_screened": True,
            "v4_3_failure_preserved_not_upgraded": True,
            "identity_bytes_equal_v4_3": canonical_successor,
            "selection_gold_fields_emitted": False,
            "q2_intermediate_annotation_eligibility_accessed": True,
            "sealed_prospective_content_opened": False,
            "sealed_prospective_content_hashed": False,
            "em_f1_ihr_authorized": False,
        },
        "selection_stats": selection_stats,
        "cohorts": cohort_locks,
        "scientific_boundary": protocol["scientific_boundary"],
    }
    report_path = destination / "report.json"
    _write_json(report_path, report)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "inputs": {
            "candidate_pools": candidate_inventory,
            "consumed_registries": consumed_inventory,
            "implementation_locks": implementation_locks,
        },
        "outputs": {
            "protocol.json": _sha256_file(protocol_path),
            "report.json": _sha256_file(report_path),
            **{
                f"{role}.identity_only.jsonl": values["sha256"]
                for role, values in cohort_locks.items()
            },
        },
        "prospective_opened_or_hashed": False,
        "action_data_built": False,
        "training_started": False,
        "answer_scoring_performed": False,
    }
    _write_json(destination / "manifest.json", manifest)
    return {"protocol": protocol, "report": report, "manifest": manifest}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidate_dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.project_root.resolve()
    candidate_dir = args.candidate_dir
    if not candidate_dir.is_absolute():
        candidate_dir = root / candidate_dir
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    result = run_freeze(
        project_root=root,
        candidate_paths={
            "train": candidate_dir / "train.jsonl",
            "dev": candidate_dir / "dev.jsonl",
        },
        output_dir=output,
    )
    print(
        json.dumps(
            {
                "status": result["report"]["status"],
                "output_dir": str(output),
                "enabled_training_datasets": list(ENABLED_DATASETS),
                "hotpotqa_status": result["report"]["hotpotqa_status"],
                "prospective_opened_or_hashed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
