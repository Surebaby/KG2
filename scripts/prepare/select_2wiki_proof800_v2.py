#!/usr/bin/env python3
"""Append-only Proof800 selector v2 for the official-raw closure-v3 path.

This module supersedes the *interface* of ``select_2wiki_proof800_v1.py``
without changing that file or any already frozen v1 artifact.  It deliberately
separates two states:

``waiting``
    Record that the selection policy is ready but the official-raw unified
    source contract/implementation has not yet been frozen.  A waiting record
    is never accepted by ``select``.

``freeze-after-code``
    Freeze a runnable v2 protocol only after (a) a passing closure-v3 release
    exists and (b) the final official-raw unified source contract binds its
    implementation SHA256.  The protocol binds both exact releases/contracts.

``select``
    Revalidate those bindings, join the frozen n=1500 population, and select
    exactly 200 strict candidates from each of the four 2Wiki question types.

No stage reads an answer for filtering/ranking.  ``select`` may transport the
outcome label already present in the unified training supply, but it is not a
selection feature.  Failed/superseded closure releases are rejected; their
schema is never rewritten or presented as closure-v3.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import (
    question_key,
    question_sha256,
    validate_question_kg_record,
)
from kgproweight.reward.trajectory_source_gate import (
    SOURCE_GATE_SCHEMA_VERSION,
    SOURCE_GATE_VERSION,
    make_source_gate_record,
)
from kgproweight.utils.logging import dump_manifest
from scripts.prepare import select_2wiki_proof800_v1 as shared
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    PROTECTED_LEDGER_SCHEMA,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


ROOT = Path(__file__).resolve().parents[2]
DATASET = shared.DATASET
SEED = shared.SEED
QTYPES = shared.QTYPES
TARGET_BY_TYPE = dict(shared.TARGET_BY_TYPE)
TOTAL_TARGET = shared.TOTAL_TARGET
HISTORICAL_CUTOFF = shared.HISTORICAL_CUTOFF
HEX64 = shared.HEX64
QID = shared.QID
PID = shared.PID

CLOSURE_REPORT_SCHEMA = "2wiki-official-raw-clean-closure-v3"
CLOSURE_PASS_STATUS = (
    "COMPLETE_DIAGNOSTIC_CLEAN_CLOSURE_NOT_SELECTED_NOT_TRAINED"
)
CLOSURE_DECISION = "CONTINUE_TO_PROOF800_SELECTION"

UNIFIED_CONTRACT_SCHEMA = "2wiki-unified-proofkg-official-raw-contract-v3"
UNIFIED_CONTRACT_STATUS = (
    "FROZEN_OFFICIAL_RAW_SOURCE_CONTRACT_NOT_MATERIALIZED_NOT_TRAINED"
)
UNIFIED_RELEASE_SCHEMA = "2wiki-unified-proofkg-official-raw-candidate-supply-v3"
UNIFIED_RELEASE_STATUS = (
    "COMPLETE_STRICT_OFFICIAL_RAW_CANDIDATE_SUPPLY_NOT_SELECTED_NOT_TRAINED"
)
UNIFIED_WRAPPER_SCHEMA = "2wiki-unified-proofkg-official-raw-candidate-wrapper-v3"
UNIFIED_REQUIRED_OUTPUTS = (
    "silver_train",
    "question_kg_records",
    "source_gate_records",
    "proof_candidates",
)

PROTOCOL_SCHEMA = "2wiki-proof800-strict-selection-protocol-v2"
WAITING_STATUS = "WAITING_FOR_FINAL_UNIFIED_CONTRACT_NOT_RUN_NOT_TRAINED"
FROZEN_STATUS = "FROZEN_CLOSURE_V3_AND_UNIFIED_V3_NOT_RUN_NOT_TRAINED"
RESULT_SCHEMA = "2wiki-proof800-strict-selection-result-v2"
RESULT_STATUS = "COMPLETE_STRICT_PROOF800_NOT_TRAINED"
SELECTION_RECORD_SCHEMA = "2wiki-proof800-selection-record-v2"
EXPERIMENT_ID_WAITING = "2WIKI-PROOF800-STRICT-SELECTION-V2-SEED42-WAITING"
EXPERIMENT_ID_PROTOCOL = "2WIKI-PROOF800-STRICT-SELECTION-V2-SEED42-PROTOCOL"
EXPERIMENT_ID_RESULT = "2WIKI-PROOF800-STRICT-SELECTION-V2-SEED42-RESULT"

DEFAULT_COHORT_RELEASE = shared.DEFAULT_COHORT_RELEASE
DEFAULT_PLANNER_POSTFLIGHT = shared.DEFAULT_PLANNER_POSTFLIGHT
DEFAULT_LEDGER = shared.DEFAULT_LEDGER
DEFAULT_REPLAY = shared.DEFAULT_REPLAY
DEFAULT_ORDINARY_PROTOCOL = shared.DEFAULT_ORDINARY_PROTOCOL
DEFAULT_UNIFIED_CONTRACT = ROOT / (
    "outputs/audits/2wiki_unified_proofkg_official_raw_v3_contract/"
    "unified_contract.json"
)
DEFAULT_WAITING_DIR = ROOT / (
    "outputs/audits/2wiki_proof800_strict_selection_v2_seed42_waiting"
)
DEFAULT_PROTOCOL_DIR = ROOT / (
    "outputs/audits/2wiki_proof800_strict_selection_v2_seed42_preregistration"
)
DEFAULT_RESULT_DIR = ROOT / (
    "outputs/audits/2wiki_proof800_strict_selection_v2_seed42_result"
)


# Reuse only stable mechanics from v1.  The frozen v1 module itself is not
# modified, and its closure/unified validators are intentionally not reused.
_read_json = shared._read_json
_read_jsonl = shared._read_jsonl
_write_jsonl = shared._write_jsonl
_sha256 = shared._sha256
_canonical_sha256 = shared._canonical_sha256
_passages_sha256 = shared._passages_sha256
_identity = shared._identity
_resolve_identity = shared._resolve_identity
_index = shared._index
_forbidden_present = shared._forbidden_present
_load_release_file = shared._load_release_file
_validate_replay_release = shared._validate_replay_release
_validate_ordinary_release = shared._validate_ordinary_release
_cohort_rows = shared._cohort_rows
_validate_planner_postflight = shared._validate_planner_postflight
_identity_triplets = shared._identity_triplets
_root_anchors_resolved = shared._root_anchors_resolved
_hops_complete_and_traceable = shared._hops_complete_and_traceable


def choose_exact_proof800(
    admitted: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve the frozen four-by-200 deterministic selection mechanics."""

    return shared.choose_exact_proof800(admitted)


def _same_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(name) == expected.get(name) for name in ("sha256", "size_bytes"))


def _assert_binding_equal(
    actual: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label}: bound file set drifted")
    for name in expected:
        if not _same_identity(actual[name], expected[name]):
            raise ValueError(f"{label}: SHA/size drift for {name}")


def _superseded(directory: Path, report: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    """Fail closed on explicit in-record or append-only supersession markers."""

    statuses = [str(report.get("status") or ""), str(manifest.get("status") or "")]
    for path in directory.glob("*.json"):
        if path.name in {"report.json", "manifest.json", "closure_report.json"}:
            continue
        try:
            value = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        statuses.append(str(value.get("status") or ""))
    return any(
        status.startswith("SUPERSEDED") or "NOT_CONSUMABLE" in status
        for status in statuses
    )


def validate_closure_v3_release(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate and SHA-bind the authoritative closure-v3 execution release."""

    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if _superseded(directory, report, manifest):
        raise ValueError("closure release is explicitly superseded/not consumable")
    if (
        report.get("schema_version") != CLOSURE_REPORT_SCHEMA
        or report.get("status") != CLOSURE_PASS_STATUS
        or manifest.get("status") != CLOSURE_PASS_STATUS
        or report.get("all_pass") is not True
        or report.get("decision") != CLOSURE_DECISION
        or report.get("gold_access") is not False
        or report.get("training_started") is not False
        or (manifest.get("run") or {}).get("training_started") is not False
        or not (report.get("gates") or {})
        or not all(value is True for value in (report.get("gates") or {}).values())
    ):
        raise ValueError("closure-v3 release status/schema/gates boundary failed")
    boundary = report.get("scientific_boundary") or {}
    if not (
        boundary.get("structural_and_source_eligibility_only") is True
        and boundary.get("passages_or_answers_read") is False
        and boundary.get("proof800_selected") is False
        and boundary.get("training_started") is False
    ):
        raise ValueError("closure-v3 scientific boundary failed")

    output_paths = {
        name: _load_release_file(report, name, release_dir=directory)
        for name in (
            "runtime_report",
            "runtime_details",
            "strict_eligibility_telemetry",
        )
    }
    telemetry_identity = (report.get("outputs") or {}).get(
        "strict_eligibility_telemetry"
    ) or {}
    if int(telemetry_identity.get("rows", -1)) != 1500:
        raise ValueError("closure-v3 telemetry row binding is not 1500")

    input_paths = {
        name: _resolve_identity(value, label=f"closure-v3 input {name}")
        for name, value in (report.get("inputs") or {}).items()
        if name in {"execution_lock", "closure_report"}
        and isinstance(value, Mapping)
    }
    if set(input_paths) != {"execution_lock", "closure_report"}:
        raise ValueError("closure-v3 does not bind execution lock/report hashes")

    manifest_run = manifest.get("run") or {}
    manifest_report = manifest_run.get("report") or {}
    if not isinstance(manifest_report, Mapping) or not _same_identity(
        manifest_report, _identity(report_path)
    ):
        raise ValueError("closure-v3 manifest/report SHA binding drifted")
    for name in ("runtime_details", "strict_eligibility_telemetry"):
        value = manifest_run.get(name) or {}
        if not isinstance(value, Mapping) or not _same_identity(
            value, _identity(output_paths[name])
        ):
            raise ValueError(f"closure-v3 manifest/{name} SHA binding drifted")

    runtime = _index(_read_jsonl(output_paths["runtime_details"]), label="closure-v3 runtime")
    telemetry = _index(
        _read_jsonl(output_paths["strict_eligibility_telemetry"]),
        label="closure-v3 telemetry",
    )
    if set(runtime) != set(telemetry):
        raise ValueError("closure-v3 runtime/telemetry identity join is not exact")
    expected_types = Counter(
        {
            "bridge_comparison": 390,
            "comparison": 390,
            "compositional": 389,
            "inference": 331,
        }
    )
    if len(telemetry) != 1500 or Counter(
        str(row.get("question_type") or "") for row in telemetry.values()
    ) != expected_types:
        raise ValueError("closure-v3 n1500/question-type population drifted")
    for key, trace in runtime.items():
        row = telemetry[key]
        if (
            row.get("schema_version")
            != "2wiki-official-raw-strict-eligibility-telemetry-v1"
            or row.get("gold_access_false") is not True
            or _forbidden_present(row)
            or str(row.get("runtime_record_sha256") or "")
            != _canonical_sha256(trace)
            or str(row.get("kg_sha256") or "")
            != _canonical_sha256(trace.get("kg_subgraph") or [])
            or str(row.get("execution_sha256") or "")
            != _canonical_sha256(trace.get("execution") or {})
        ):
            raise ValueError(f"closure-v3 runtime/telemetry hash mismatch: {key}")

    binding = {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        **{name: _identity(path) for name, path in output_paths.items()},
        **{name: _identity(path) for name, path in input_paths.items()},
    }
    return telemetry, binding


def _materializer_identity(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = contract.get("implementation")
    nested = (contract.get("code") or {}).get("materializer")
    choices = [value for value in (direct, nested) if isinstance(value, Mapping)]
    if len(choices) != 1:
        raise ValueError("unified contract must bind exactly one materializer implementation")
    return choices[0]


def validate_unified_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the finalized official-raw source contract and code SHA."""

    contract = _read_json(path)
    if (
        contract.get("schema_version") != UNIFIED_CONTRACT_SCHEMA
        or contract.get("status") != UNIFIED_CONTRACT_STATUS
        or contract.get("release_schema_version") != UNIFIED_RELEASE_SCHEMA
        or contract.get("release_status") != UNIFIED_RELEASE_STATUS
        or contract.get("candidate_wrapper_schema_version") != UNIFIED_WRAPPER_SCHEMA
        or tuple(contract.get("required_outputs") or ()) != UNIFIED_REQUIRED_OUTPUTS
        or contract.get("training_started") is not False
    ):
        raise ValueError("official-raw unified source contract is incomplete or drifted")
    implementation = _materializer_identity(contract)
    implementation_path = _resolve_identity(
        implementation, label="official-raw unified materializer"
    )
    return contract, {
        "contract": _identity(path),
        "materializer": _identity(implementation_path),
    }


def _selection_policy() -> dict[str, Any]:
    return {
        "seed": SEED,
        "target_total": TOTAL_TARGET,
        "target_by_question_type": TARGET_BY_TYPE,
        "candidate_universe": (
            "exact dataset::qid/question-hash subset of the frozen official-raw "
            "n1500 cohort"
        ),
        "hard_candidate_checks": [
            "identity_and_current_family_hash_exact",
            "canonical_question_kg_schema_valid",
            "planner_schema_valid_true",
            "gold_access_false",
            "runtime_error_zero",
            "provenance_and_historical_cutoff_bound",
            "all_root_anchors_resolved_to_nonabstained_qids",
            "all_planned_hops_executed_with_exact_pids_qid_inputs_and_nonempty_matches",
            "complete_plan_execution_true",
            "nonempty_unique_retained_edges_all_traceable_to_execution",
            "source_gate_recomputed_exact_and_m_graph_one",
            "ten_nonempty_frozen_passages_and_sha256_exact",
            "unified_record_wrapper_gate_silver_hash_join_exact",
            "protected_replay_ordinary_qid_hash_family_overlap_zero",
        ],
        "ranking": {
            "uses_gold_or_model_correctness": False,
            "mechanics": "unchanged_from_selector_v1",
            "primary": (
                "one deterministic representative per current lexical family "
                "within question type"
            ),
            "secondary": (
                "deterministic repeat-family rows only after all distinct families "
                "in that type"
            ),
            "within_family_rank": (
                "sha256(seed, proof800-v1-within-family, dataset, qid)"
            ),
            "family_rank": "sha256(seed, proof800-v1-family, dataset, family_sha256)",
            "repeat_rank": "sha256(seed, proof800-v1-repeat, dataset, qid)",
            "question_type_order": list(QTYPES),
            "cross_type_qid_and_question_hash_reuse": "forbidden",
            "cross_type_family_reuse": (
                "allowed_but_reported; family is not an outcome label"
            ),
        },
        "failure_policy": (
            "if any type has fewer than 200 strict rows, write no result release "
            "and fail; thresholds are not lowered"
        ),
    }


def write_waiting_record(*, output_dir: Path) -> dict[str, Any]:
    """Write a non-runnable append-only record while unified-v3 is unfinished."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite waiting record: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID_WAITING,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": WAITING_STATUS,
        "reason": (
            "A runnable selector protocol may be frozen only after the final "
            "official-raw unified source contract binds its materializer SHA256."
        ),
        "closure_contract": {
            "report_schema": CLOSURE_REPORT_SCHEMA,
            "pass_status": CLOSURE_PASS_STATUS,
            "decision": CLOSURE_DECISION,
            "execution_report_and_runtime_hashes_required": True,
        },
        "expected_unified_contract": {
            "path": str(DEFAULT_UNIFIED_CONTRACT),
            "schema_version": UNIFIED_CONTRACT_SCHEMA,
            "status": UNIFIED_CONTRACT_STATUS,
            "release_schema_version": UNIFIED_RELEASE_SCHEMA,
            "release_status": UNIFIED_RELEASE_STATUS,
            "candidate_wrapper_schema_version": UNIFIED_WRAPPER_SCHEMA,
            "required_outputs": list(UNIFIED_REQUIRED_OUTPUTS),
        },
        "selection": _selection_policy(),
        "code": {
            "selector_v2": _identity(Path(__file__)),
            "shared_v1_mechanics": _identity(Path(shared.__file__)),
        },
        "runnable": False,
        "proof800_selected": False,
        "training_started": False,
    }
    record_path = output_dir / "waiting.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=WAITING_STATUS,
        extra={
            "phase": "proof800_selector_v2_waiting_for_unified_contract",
            "experiment_id": EXPERIMENT_ID_WAITING,
            "waiting": _identity(record_path),
            "runnable": False,
            "proof800_selected": False,
            "training_started": False,
        },
    )
    return record


def freeze_protocol_after_code(
    *,
    cohort_release: Path,
    planner_postflight: Path,
    protected_ledger_dir: Path,
    replay_dir: Path,
    ordinary_protocol: Path,
    closure_dir: Path,
    unified_contract_path: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID_PROTOCOL,
) -> dict[str, Any]:
    """Freeze the runnable protocol after both external contracts are final."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v2 protocol: {output_dir}")
    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        raise ValueError("Proof800 v2 protocol requires a non-empty Experiment ID")
    cohort, cohort_binding = _cohort_rows(cohort_release)
    planner_binding = _validate_planner_postflight(planner_postflight)
    ledger_path, ledger_binding = validate_protected_ledger_release(
        protected_ledger_dir
    )
    replay_path, replay_binding = _validate_replay_release(replay_dir)
    ordinary_path, ordinary_binding = _validate_ordinary_release(ordinary_protocol)
    _closure, closure_binding = validate_closure_v3_release(closure_dir)
    unified_contract, unified_contract_binding = validate_unified_contract(
        unified_contract_path
    )

    blocked_rows = [
        *_read_jsonl(ledger_path),
        *_read_jsonl(replay_path),
        *_read_jsonl(ordinary_path),
    ]
    blocked_qids, blocked_hashes, blocked_families = _identity_triplets(blocked_rows)
    overlaps = Counter()
    for row in cohort:
        overlaps["qid"] += str(row["qid"]) in blocked_qids
        overlaps["question_sha256"] += (
            str(row["question_sha256"]) in blocked_hashes
        )
        overlaps["family_sha256"] += str(row["family_sha256"]) in blocked_families
    expected_counts = Counter(
        {
            "bridge_comparison": 390,
            "comparison": 390,
            "compositional": 389,
            "inference": 331,
        }
    )
    gates = {
        "candidate_cohort_n1500_exact": len(cohort) == 1500,
        "candidate_question_type_counts_exact": Counter(
            str(row["question_type"]) for row in cohort
        )
        == expected_counts,
        "candidate_protected_replay_ordinary_qid_hash_family_overlap_zero": not any(
            overlaps.values()
        ),
        "planner_postflight_pass": True,
        "closure_v3_release_pass_and_hash_bound": True,
        "unified_v3_contract_and_implementation_hash_bound": True,
        "four_question_type_targets_exactly_200": TARGET_BY_TYPE
        == {qtype: 200 for qtype in QTYPES},
        "answer_blind_selection_fields_only": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"Proof800 v2 preregistration gates failed: {gates}; "
            f"overlap={dict(overlaps)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": FROZEN_STATUS,
        "scope": (
            "strict Proof800 selection from the frozen official-raw n1500 "
            "2Wiki train-only cohort"
        ),
        "historical_cutoff": HISTORICAL_CUTOFF,
        "selection": _selection_policy(),
        "contracts": {
            "closure_v3": {
                "report_schema": CLOSURE_REPORT_SCHEMA,
                "pass_status": CLOSURE_PASS_STATUS,
                "decision": CLOSURE_DECISION,
                "release": closure_binding,
            },
            "unified_v3": {
                "contract_values": {
                    "schema_version": unified_contract["schema_version"],
                    "status": unified_contract["status"],
                    "release_schema_version": unified_contract[
                        "release_schema_version"
                    ],
                    "release_status": unified_contract["release_status"],
                    "candidate_wrapper_schema_version": unified_contract[
                        "candidate_wrapper_schema_version"
                    ],
                    "required_outputs": list(unified_contract["required_outputs"]),
                },
                "binding": unified_contract_binding,
            },
        },
        "inputs": {
            "candidate_cohort_release": cohort_binding,
            "planner_postflight": planner_binding,
            "protected_ledger_release": ledger_binding,
            "clean_replay_release": replay_binding,
            "ordinary200_release": ordinary_binding,
        },
        "code": {
            "selector_v2": _identity(Path(__file__)),
            "shared_v1_selection_mechanics": _identity(Path(shared.__file__)),
            "source_gate": _identity(
                ROOT / "kgproweight/reward/trajectory_source_gate.py"
            ),
            "v4_freezer": _identity(
                ROOT
                / "scripts/prepare/freeze_mixed_ppo_three_dataset_v4_proof800.py"
            ),
            "v4_materializer": _identity(
                ROOT
                / "scripts/prepare/materialize_mixed_ppo_three_dataset_v4_proof800.py"
            ),
        },
        "gates": gates,
        "scientific_boundary": {
            "train_only": True,
            "answer_values_read_for_selection": False,
            "gold_support_or_source_steps_read_for_selection": False,
            "semantic_correctness": "UNKNOWN_NOT_USED_FOR_SELECTION",
            "selection_started": False,
            "training_started": False,
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": FROZEN_STATUS,
        "candidate_counts": {
            "total": len(cohort),
            "by_question_type": dict(
                sorted(Counter(str(row["question_type"]) for row in cohort).items())
            ),
            "unique_current_families": len(
                {str(row["family_sha256"]) for row in cohort}
            ),
            "blocked_overlap": dict(overlaps),
        },
        "contracts": protocol["contracts"],
        "gates": gates,
        "protocol": _identity(protocol_path),
        "selection_started": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=FROZEN_STATUS,
        extra={
            "phase": "proof800_strict_selection_v2_preregistration",
            "experiment_id": experiment_id,
            "protocol": _identity(protocol_path),
            "report": _identity(report_path),
            "selection_started": False,
            "training_started": False,
        },
    )
    return report


def _validate_protocol(
    protocol_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_path = protocol_dir / "protocol.json"
    report_path = protocol_dir / "report.json"
    manifest_path = protocol_dir / "manifest.json"
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    protocol_experiment_id = str(protocol.get("experiment_id") or "").strip()
    manifest_run = manifest.get("run") or {}
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("status") != FROZEN_STATUS
        or not protocol_experiment_id
        or report.get("schema_version") != PROTOCOL_SCHEMA
        or report.get("status") != FROZEN_STATUS
        or report.get("experiment_id") != protocol_experiment_id
        or manifest.get("status") != FROZEN_STATUS
        or manifest_run.get("phase")
        != "proof800_strict_selection_v2_preregistration"
        or manifest_run.get("experiment_id") != protocol_experiment_id
        or manifest_run.get("selection_started") is not False
        or manifest_run.get("training_started") is not False
        or not (protocol.get("gates") or {})
        or not all(value is True for value in (protocol.get("gates") or {}).values())
        or (protocol.get("scientific_boundary") or {}).get("training_started")
        is not False
    ):
        raise ValueError("Proof800 v2 protocol is waiting, superseded, or invalid")
    if not _same_identity(
        manifest_run.get("protocol") or {}, _identity(protocol_path)
    ):
        raise ValueError("Proof800 v2 protocol manifest binding drifted")
    if not _same_identity(
        manifest_run.get("report") or {}, _identity(report_path)
    ) or not _same_identity(report.get("protocol") or {}, _identity(protocol_path)):
        raise ValueError("Proof800 v2 protocol report/manifest binding drifted")
    for section in (protocol.get("inputs") or {}).values():
        if not isinstance(section, Mapping):
            raise ValueError("Proof800 v2 protocol input binding malformed")
        for label, value in section.items():
            if isinstance(value, Mapping) and "path" in value:
                _resolve_identity(value, label=f"protocol input {label}")
    for label, value in (protocol.get("code") or {}).items():
        _resolve_identity(value, label=f"protocol code {label}")

    unified = ((protocol.get("contracts") or {}).get("unified_v3") or {})
    binding = unified.get("binding") or {}
    contract_path = _resolve_identity(
        binding.get("contract") or {}, label="frozen unified-v3 contract"
    )
    contract, current_binding = validate_unified_contract(contract_path)
    _assert_binding_equal(current_binding, binding, label="unified-v3 contract")
    values = unified.get("contract_values") or {}
    if values != {
        "schema_version": contract["schema_version"],
        "status": contract["status"],
        "release_schema_version": contract["release_schema_version"],
        "release_status": contract["release_status"],
        "candidate_wrapper_schema_version": contract[
            "candidate_wrapper_schema_version"
        ],
        "required_outputs": list(contract["required_outputs"]),
    }:
        raise ValueError("embedded unified-v3 contract values drifted")
    cohort_identity = (
        ((protocol.get("inputs") or {}).get("candidate_cohort_release") or {}).get(
            "cohort"
        )
    )
    if not isinstance(cohort_identity, Mapping):
        raise ValueError("Proof800 v2 protocol does not bind the cohort")
    cohort = _read_jsonl(_resolve_identity(cohort_identity, label="protocol cohort"))
    return protocol, cohort


def validate_unified_v3_supply(
    directory: Path,
    *,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Validate a release against the exact unified-v3 contract in protocol."""

    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if _superseded(directory, report, manifest):
        raise ValueError("unified-v3 supply is explicitly superseded/not consumable")
    values = (
        (((protocol.get("contracts") or {}).get("unified_v3") or {}).get(
            "contract_values"
        ))
        or {}
    )
    if (
        report.get("schema_version") != values.get("release_schema_version")
        or report.get("status") != values.get("release_status")
        or manifest.get("status") != values.get("release_status")
        or not (report.get("checks") or {})
        or not all(value is True for value in (report.get("checks") or {}).values())
        or report.get("training_started") is not False
        or (manifest.get("run") or {}).get("training_started") is not False
    ):
        raise ValueError("unified-v3 supply status/schema/checks failed")

    unified_contract_binding = (
        (((protocol.get("contracts") or {}).get("unified_v3") or {}).get("binding"))
        or {}
    )
    source_contract = report.get("source_contract") or {}
    expected_contract = unified_contract_binding.get("contract") or {}
    if not isinstance(source_contract, Mapping) or not _same_identity(
        source_contract, expected_contract
    ):
        raise ValueError("unified-v3 release is not bound to the frozen source contract")
    _resolve_identity(source_contract, label="unified-v3 release source contract")

    ledger_expected = (protocol.get("inputs") or {}).get(
        "protected_ledger_release"
    ) or {}
    supply_ledger = report.get("protected_ledger") or {}
    if not (
        supply_ledger.get("version") == PROTECTED_LEDGER_SCHEMA
        and supply_ledger.get("complete") is True
        and supply_ledger.get("current_family_recomputed") is True
    ):
        raise ValueError("unified-v3 supply lacks complete protected-ledger binding")
    for name in ("ledger", "report", "manifest"):
        actual = supply_ledger.get(name) or {}
        expected = ledger_expected.get(name) or {}
        if not isinstance(actual, Mapping) or not _same_identity(actual, expected):
            raise ValueError(f"unified-v3 protected-ledger mismatch: {name}")
        _resolve_identity(actual, label=f"unified-v3 protected ledger {name}")

    required_outputs = tuple(values.get("required_outputs") or ())
    if required_outputs != UNIFIED_REQUIRED_OUTPUTS:
        raise ValueError("unified-v3 required output contract drifted")
    paths = {
        name: _load_release_file(report, name, release_dir=directory)
        for name in required_outputs
    }
    manifest_report = (manifest.get("run") or {}).get("report") or {}
    if not isinstance(manifest_report, Mapping) or not _same_identity(
        manifest_report, _identity(report_path)
    ):
        raise ValueError("unified-v3 manifest/report SHA binding drifted")

    rows = {
        "silver": _index(_read_jsonl(paths["silver_train"]), label="unified-v3 silver"),
        "records": _index(
            _read_jsonl(paths["question_kg_records"]), label="unified-v3 KG"
        ),
        "gates": _index(
            _read_jsonl(paths["source_gate_records"]), label="unified-v3 gate"
        ),
        "wrappers": _index(
            _read_jsonl(paths["proof_candidates"]), label="unified-v3 wrappers"
        ),
    }
    key_sets = {name: set(index) for name, index in rows.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError(
            "unified-v3 four-way identity join failed: "
            f"{ {name: len(keys) for name, keys in key_sets.items()} }"
        )
    wrapper_schema = str(values.get("candidate_wrapper_schema_version") or "")
    if any(
        row.get("schema_version") != wrapper_schema
        for row in rows["wrappers"].values()
    ):
        raise ValueError("unified-v3 candidate wrapper schema drifted")
    return rows, {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "source_contract": _identity(
            _resolve_identity(source_contract, label="unified-v3 source contract")
        ),
        **{name: _identity(path) for name, path in paths.items()},
    }


def _silver_identity_exact(
    silver: Mapping[str, Any],
    *,
    dataset: str,
    qid: str,
    question: str,
    question_hash: str,
) -> bool:
    """Validate silver identity while supporting its pre-hash storage schema."""

    silver_question = str(silver.get("question") or "").strip()
    return bool(
        str(silver.get("dataset") or "").strip().lower() == dataset
        and str(silver.get("qid") or "").strip() == qid
        and silver_question == question
        and question_sha256(silver_question) == question_hash
        and (
            "question_sha256" not in silver
            or str(silver.get("question_sha256") or "") == question_hash
        )
    )


def assess_candidate(
    *,
    cohort: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    silver: Mapping[str, Any],
    record: Mapping[str, Any],
    gate: Mapping[str, Any],
    closure_telemetry: Mapping[str, Any],
    historical_cutoff: str,
    planner_predictions_sha256: str,
    wrapper_schema_version: str = UNIFIED_WRAPPER_SCHEMA,
) -> tuple[bool, dict[str, bool]]:
    """Re-run every strict v1 gate under the explicit unified-v3 schema."""

    dataset = str(cohort.get("dataset") or "").strip().lower()
    qid = str(cohort.get("qid") or "").strip()
    question = str(cohort.get("question") or "").strip()
    key = question_key(dataset, qid)
    qhash = question_sha256(question)
    family = family_sha256(question)
    passages = list(silver.get("retrieved_passages") or [])
    passages_hash = _passages_sha256(passages)
    metadata = silver.get("metadata") or {}

    checks: dict[str, bool] = {
        "candidate_universe_identity_exact": all(
            str(row.get("dataset") or "").strip().lower() == dataset
            and str(row.get("qid") or "").strip() == qid
            and str(row.get("question") or "").strip() == question
            and str(row.get("question_sha256") or "") == qhash
            for row in (wrapper, record)
        )
        # The training-silver schema predates the redundant stored hash.  Keep
        # the same exact-identity predicate by recomputing it from the live
        # question; if a stored hash is present it must also agree.
        and _silver_identity_exact(
            silver,
            dataset=dataset,
            qid=qid,
            question=question,
            question_hash=qhash,
        )
        and str(wrapper.get("question_key") or "") == key
        and str(record.get("question_key") or "") == key
        and str(gate.get("question_key") or "") == key
        and str(closure_telemetry.get("question_key") or "") == key
        and str(closure_telemetry.get("dataset") or "").strip().lower() == dataset
        and str(closure_telemetry.get("qid") or "").strip() == qid
        and str(closure_telemetry.get("question_sha256") or "") == qhash,
        "question_type_exact": str(wrapper.get("question_type") or "")
        == str(cohort.get("question_type") or "")
        == str(metadata.get("question_type") or ""),
        "family_exact": wrapper.get("family_version") == FAMILY_VERSION
        and str(wrapper.get("family_sha256") or "") == family
        and str(closure_telemetry.get("family_sha256") or "") == family,
        "wrapper_schema_and_answer_blind": wrapper.get("schema_version")
        == wrapper_schema_version
        and wrapper.get("gold_access") is False
        and not _forbidden_present(
            {key: value for key, value in wrapper.items() if key != "question_kg_record"}
        ),
        "wrapper_record_hash_exact": _canonical_sha256(
            wrapper.get("question_kg_record") or {}
        )
        == _canonical_sha256(record),
        "record_schema_valid": True,
        "planner_schema_valid": record.get("planner_schema_valid") is True
        and (record.get("query_plan") or {}).get("recognized") is True,
        "gold_access_false": (record.get("provenance") or {}).get("gold_access")
        is False
        and closure_telemetry.get("gold_access_false") is True,
        "runtime_error_zero": record.get("runtime_error") in (None, "")
        and closure_telemetry.get("runtime_error_zero") is True,
        "provenance_complete": bool(
            str((record.get("provenance") or {}).get("builder_version") or "").strip()
        )
        and str((record.get("provenance") or {}).get("historical_cutoff") or "")
        == historical_cutoff
        and str(
            (record.get("provenance") or {}).get("planner_predictions_sha256")
            or ""
        )
        == planner_predictions_sha256
        and (record.get("provenance") or {}).get("complete_plan_execution") is True,
        "all_root_anchors_resolved": _root_anchors_resolved(record)
        and closure_telemetry.get("all_root_anchors_resolved") is True,
        "all_hops_complete_and_traceable": _hops_complete_and_traceable(record)
        and (record.get("execution") or {}).get("complete_plan_execution") is True
        and closure_telemetry.get("all_hops_complete") is True
        and closure_telemetry.get("retained_edges_traceable") is True,
        "graph_nonempty": bool(record.get("kg_subgraph"))
        and closure_telemetry.get("graph_nonempty") is True,
        "passages_complete_and_hash_bound": len(passages) == 10
        and all(
            isinstance(passage, Mapping)
            and bool(str(passage.get("contents") or "").strip())
            for passage in passages
        )
        and str(metadata.get("retrieved_passages_sha256") or "") == passages_hash
        and str(wrapper.get("proof_passages_sha256") or "") == passages_hash,
        "outcome_label_present_not_used_for_ranking": bool(
            str(metadata.get("gold_answer") or silver.get("answer") or "").strip()
        ),
        "source_steps_absent": silver.get("steps") == []
        and str(silver.get("teacher_output") or "") == "",
    }
    try:
        validate_question_kg_record(record)
    except (TypeError, ValueError):
        checks["record_schema_valid"] = False

    recomputed = make_source_gate_record(
        record,
        dataset=dataset,
        qid=qid,
        question=question,
        text_evidence_available=True,
        historical_cutoff=historical_cutoff,
    )
    checks["source_gate_exact_and_eligible"] = (
        gate.get("schema_version") == SOURCE_GATE_SCHEMA_VERSION
        and gate.get("gate_version") == SOURCE_GATE_VERSION
        and gate.get("m_graph") == 1
        and gate.get("graph_eligible") is True
        and all(bool(value) for value in (gate.get("eligibility_checks") or {}).values())
        and _canonical_sha256(gate) == _canonical_sha256(recomputed)
    )
    checks["closure_hash_attestation_exact"] = (
        str(closure_telemetry.get("kg_sha256") or "")
        == _canonical_sha256(record.get("kg_subgraph") or [])
        and str(closure_telemetry.get("execution_sha256") or "")
        == _canonical_sha256(record.get("execution") or {})
        and closure_telemetry.get("m_graph") == 1
    )
    return all(checks.values()), checks


def select_release(
    *,
    protocol_dir: Path,
    closure_dir: Path,
    unified_supply_dir: Path,
    output_dir: Path,
    experiment_id: str = EXPERIMENT_ID_RESULT,
) -> dict[str, Any]:
    """Select Proof800 only from the exact closure-v3/unified-v3 bindings."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Proof800 v2 result: {output_dir}")
    experiment_id = str(experiment_id or "").strip()
    if not experiment_id:
        raise ValueError("Proof800 v2 result requires a non-empty Experiment ID")
    protocol, cohort_rows = _validate_protocol(protocol_dir)
    cohort = _index(cohort_rows, label="frozen n1500 cohort")
    closure, actual_closure_binding = validate_closure_v3_release(closure_dir)
    expected_closure_binding = (
        (((protocol.get("contracts") or {}).get("closure_v3") or {}).get("release"))
        or {}
    )
    _assert_binding_equal(
        actual_closure_binding,
        expected_closure_binding,
        label="closure-v3 release",
    )
    supply, supply_binding = validate_unified_v3_supply(
        unified_supply_dir, protocol=protocol
    )

    blocked_sections = protocol["inputs"]
    ledger_path = _resolve_identity(
        blocked_sections["protected_ledger_release"]["ledger"],
        label="protected ledger",
    )
    replay_path = _resolve_identity(
        blocked_sections["clean_replay_release"]["selection_records"],
        label="clean replay identities",
    )
    ordinary_path = _resolve_identity(
        blocked_sections["ordinary200_release"]["ordinary200"],
        label="ordinary200 identities",
    )
    blocked_qids, blocked_hashes, blocked_families = _identity_triplets(
        [
            *_read_jsonl(ledger_path),
            *_read_jsonl(replay_path),
            *_read_jsonl(ordinary_path),
        ]
    )

    funnel = Counter()
    failed_check_counts = Counter()
    admitted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidate_keys = set(cohort)
    supply_keys = set(supply["wrappers"])
    if set(closure) != candidate_keys:
        raise ValueError(
            "closure-v3 telemetry does not exactly cover frozen n1500 cohort: "
            f"closure={len(closure)} cohort={len(candidate_keys)}"
        )
    planner_predictions_sha256 = str(
        protocol["inputs"]["planner_postflight"]["predictions"]["sha256"]
    )
    wrapper_schema = str(
        protocol["contracts"]["unified_v3"]["contract_values"][
            "candidate_wrapper_schema_version"
        ]
    )
    for key in sorted(candidate_keys):
        funnel["frozen_candidate_universe"] += 1
        base = cohort[key]
        missing = [
            name
            for name in ("wrappers", "silver", "records", "gates")
            if key not in supply[name]
        ]
        if missing:
            funnel["missing_join"] += 1
            audit_rows.append(
                {
                    "schema_version": SELECTION_RECORD_SCHEMA,
                    "question_key": key,
                    "dataset": DATASET,
                    "qid": base["qid"],
                    "question_sha256": base["question_sha256"],
                    "family_sha256": base["family_sha256"],
                    "question_type": base["question_type"],
                    "admitted": False,
                    "selected": False,
                    "failure_reasons": [f"missing:{name}" for name in missing],
                }
            )
            continue
        wrapper = supply["wrappers"][key]
        _ok, checks = assess_candidate(
            cohort=base,
            wrapper=wrapper,
            silver=supply["silver"][key],
            record=supply["records"][key],
            gate=supply["gates"][key],
            closure_telemetry=closure[key],
            historical_cutoff=str(protocol["historical_cutoff"]),
            planner_predictions_sha256=planner_predictions_sha256,
            wrapper_schema_version=wrapper_schema,
        )
        reasons = [name for name, passed in checks.items() if not passed]
        if str(base["qid"]) in blocked_qids:
            reasons.append("blocked_qid")
        if str(base["question_sha256"]) in blocked_hashes:
            reasons.append("blocked_question_sha256")
        if str(base["family_sha256"]) in blocked_families:
            reasons.append("blocked_family_sha256")
        if reasons:
            failed_check_counts.update(reasons)
            funnel["strict_rejected"] += 1
        else:
            funnel["strict_admitted"] += 1
            item = dict(wrapper)
            item["question_type"] = str(base["question_type"])
            item["family_version"] = FAMILY_VERSION
            item["family_sha256"] = str(base["family_sha256"])
            item["proof_record_sha256"] = _canonical_sha256(
                supply["records"][key]
            )
            admitted.append(item)
        audit_rows.append(
            {
                "schema_version": SELECTION_RECORD_SCHEMA,
                "question_key": key,
                "dataset": DATASET,
                "qid": base["qid"],
                "question_sha256": base["question_sha256"],
                "family_sha256": base["family_sha256"],
                "question_type": base["question_type"],
                "admitted": not reasons,
                "selected": False,
                "failure_reasons": sorted(reasons),
                "checks": checks,
                "proof_record_sha256": _canonical_sha256(supply["records"][key]),
                "proof_passages_sha256": str(
                    wrapper.get("proof_passages_sha256") or ""
                ),
                "closure_runtime_record_sha256": str(
                    closure[key].get("runtime_record_sha256") or ""
                ),
            }
        )

    selected, selection_stats = choose_exact_proof800(admitted)
    selected_keys = {str(row["question_key"]) for row in selected}
    for row in audit_rows:
        if row["question_key"] in selected_keys:
            row["selected"] = True
    selected_by_type = Counter(str(row["question_type"]) for row in selected)
    gates = {
        "selected_exactly_800": len(selected) == TOTAL_TARGET,
        "selected_200_each_question_type": selected_by_type
        == Counter(TARGET_BY_TYPE),
        "selected_subset_of_frozen_n1500": selected_keys.issubset(candidate_keys),
        "selected_all_strict_admitted": selected_keys.issubset(
            {str(row["question_key"]) for row in admitted}
        ),
        "selected_qid_unique": len({str(row["qid"]) for row in selected})
        == TOTAL_TARGET,
        "selected_question_hash_unique": len(
            {str(row["question_sha256"]) for row in selected}
        )
        == TOTAL_TARGET,
        "selected_protected_replay_ordinary_overlap_zero": all(
            str(row["qid"]) not in blocked_qids
            and str(row["question_sha256"]) not in blocked_hashes
            and str(row["family_sha256"]) not in blocked_families
            for row in selected
        ),
        "selected_gold_access_false": all(
            row.get("gold_access") is False for row in selected
        ),
        "selection_answer_blind": True,
        "closure_v3_exact_release_bound": True,
        "unified_v3_exact_contract_bound": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Proof800 v2 final gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    proof_path = output_dir / "proof_candidates.jsonl"
    audit_path = output_dir / "selection_records.question_only.jsonl"
    selected_records_path = output_dir / "question_kg_records.jsonl"
    selected_gates_path = output_dir / "source_gate_records.jsonl"
    _write_jsonl(proof_path, selected)
    _write_jsonl(audit_path, audit_rows)
    _write_jsonl(
        selected_records_path,
        (supply["records"][key] for key in sorted(selected_keys)),
    )
    _write_jsonl(
        selected_gates_path,
        (supply["gates"][key] for key in sorted(selected_keys)),
    )
    outputs = {
        "proof_candidates": _identity(proof_path),
        "selection_records": _identity(audit_path),
        "question_kg_records": _identity(selected_records_path),
        "source_gate_records": _identity(selected_gates_path),
    }
    report = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": RESULT_STATUS,
        "funnel": {
            **dict(sorted(funnel.items())),
            "unified_supply_outside_frozen_cohort_ignored": len(
                supply_keys - candidate_keys
            ),
            "failed_check_counts": dict(sorted(failed_check_counts.items())),
            "strict_admitted_by_question_type": dict(
                sorted(
                    Counter(str(row["question_type"]) for row in admitted).items()
                )
            ),
        },
        "selection": {
            **selection_stats,
            "selected_by_question_type": dict(sorted(selected_by_type.items())),
            "family_policy": protocol["selection"]["ranking"],
        },
        "gates": gates,
        "inputs": {
            "protocol": _identity(protocol_dir / "protocol.json"),
            "closure_v3_release": actual_closure_binding,
            "unified_v3_supply": supply_binding,
        },
        "outputs": outputs,
        "scientific_boundary": {
            "train_only": True,
            "answer_values_read_for_selection": False,
            "answer_correctness_used_for_selection": False,
            "gold_support_or_source_steps_used_for_selection": False,
            "semantic_correctness": "UNKNOWN_NOT_USED_FOR_SELECTION",
            "training_started": False,
        },
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=RESULT_STATUS,
        extra={
            "phase": "proof800_strict_selection_v2",
            "experiment_id": experiment_id,
            "report": _identity(report_path),
            "outputs": outputs,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    waiting = subparsers.add_parser(
        "waiting", help="write a non-runnable WAITING contract record"
    )
    waiting.add_argument("--output-dir", type=Path, default=DEFAULT_WAITING_DIR)

    freeze = subparsers.add_parser(
        "freeze-after-code",
        help="freeze only after closure-v3 and unified-v3 implementation are final",
    )
    freeze.add_argument("--cohort-release", type=Path, default=DEFAULT_COHORT_RELEASE)
    freeze.add_argument(
        "--planner-postflight", type=Path, default=DEFAULT_PLANNER_POSTFLIGHT
    )
    freeze.add_argument("--protected-ledger-dir", type=Path, default=DEFAULT_LEDGER)
    freeze.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    freeze.add_argument(
        "--ordinary-protocol", type=Path, default=DEFAULT_ORDINARY_PROTOCOL
    )
    freeze.add_argument("--closure-dir", type=Path, required=True)
    freeze.add_argument(
        "--unified-contract", type=Path, default=DEFAULT_UNIFIED_CONTRACT
    )
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    freeze.add_argument("--experiment-id", default=EXPERIMENT_ID_PROTOCOL)

    select = subparsers.add_parser("select", help="materialize exact strict Proof800")
    select.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    select.add_argument("--closure-dir", type=Path, required=True)
    select.add_argument("--unified-supply-dir", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    select.add_argument("--experiment-id", default=EXPERIMENT_ID_RESULT)

    args = parser.parse_args()
    if args.command == "waiting":
        report = write_waiting_record(output_dir=args.output_dir)
    elif args.command == "freeze-after-code":
        report = freeze_protocol_after_code(
            cohort_release=args.cohort_release,
            planner_postflight=args.planner_postflight,
            protected_ledger_dir=args.protected_ledger_dir,
            replay_dir=args.replay_dir,
            ordinary_protocol=args.ordinary_protocol,
            closure_dir=args.closure_dir,
            unified_contract_path=args.unified_contract,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
    else:
        report = select_release(
            protocol_dir=args.protocol_dir,
            closure_dir=args.closure_dir,
            unified_supply_dir=args.unified_supply_dir,
            output_dir=args.output_dir,
            experiment_id=args.experiment_id,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
