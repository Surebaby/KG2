#!/usr/bin/env python3
"""Strict CPU preflight for the final mixed3-v4 Proof800 PPO release.

This command is a read-only release audit.  It reopens every materialised
asset, verifies the frozen identities and hashes, exercises the production
question-KG join, and checks the isolation/gold-use boundaries required before
remote PPO training.  It does not freeze data, materialise data, or start
training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.question_kg import (
    load_question_kg_index,
    question_key,
    question_sha256,
    validate_question_kg_record,
)
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.reward.trajectory_source_gate import (
    SOURCE_GATE_SCHEMA_VERSION,
    SOURCE_GATE_VERSION,
    make_source_gate_record,
)
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    DEFAULT_PROTECTED_LEDGER_DIR,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_mixed_ppo_three_dataset_v4_proof800 import (
    DATASETS,
    EXPECTED_PROOF_SUPPLY_SCHEMA,
    EXPECTED_PROOF_SUPPLY_STATUS,
    EXPECTED_PROOF_SUPPLY_OUTPUTS,
    EXPECTED_PROTOCOL_EXPERIMENT,
    EXPECTED_PROTOCOL_SCHEMA,
    EXPECTED_PROTOCOL_STATUS,
    FORBIDDEN_PROMPT_FIELDS,
    REPORT_SCHEMA,
    STATUS as DATA_STATUS,
    _canonical_sha256,
    _identity,
    _load_protocol,
    _read_jsonl,
    _resolve_bound_file,
    _sha256,
    _ten_safe_passages,
    _validate_parent_release,
    _validate_proof_supply_release,
    _validate_replay_release,
    identity_overlap_counts,
    validate_schedule_assets,
)


PREFLIGHT_SCHEMA = "source-gated-mixed3-v4-proof800-preflight-v1"
PREFLIGHT_STATUS_PASS = "PASS_NOT_TRAINED"
PREFLIGHT_STATUS_FAIL = "FAIL_NOT_TRAINED"
DEFAULT_DATA_DIR = Path(
    "data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42"
)
DEFAULT_REPLAY_DIR = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_preflight"
)
DEFAULT_EXPERIMENT_ID = (
    "MIXED-PPO-THREE-DATASET-V4-PROOF800-N3000-K4-SEED42-PREFLIGHT"
)
REQUIRED_DATA_FILES = {
    "silver_train": "silver_train.jsonl",
    "question_kg_records": "question_kg_records.jsonl",
    "source_gate_records": "source_gate_records.jsonl",
    "sampling_weights": "sampling_weights.jsonl",
    "prompt_groups": "prompt_groups.jsonl",
    "fixed_rollout_schedule": "fixed_rollout_schedule.jsonl",
}
REQUIRED_REPLAY_FILES = (
    "silver_train.jsonl",
    "selection_records.jsonl",
    "report.json",
    "manifest.json",
)


def _identity_matches(identity: Any, path: Path) -> bool:
    if not isinstance(identity, Mapping) or not path.is_file():
        return False
    raw = str(identity.get("path") or "").strip()
    digest = str(identity.get("sha256") or "").strip()
    if not raw or not digest:
        return False
    bound = Path(raw)
    if not bound.is_absolute():
        bound = Path.cwd() / bound
    try:
        same_path = bound.resolve() == path.resolve()
    except OSError:
        return False
    size_ok = identity.get("size_bytes") is None or int(identity["size_bytes"]) == path.stat().st_size
    return same_path and size_ok and _sha256(path) == digest


def _safe_bound_path(identity: Any) -> Path | None:
    if not isinstance(identity, Mapping):
        return None
    try:
        return _resolve_bound_file(identity, label="preflight binding")
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if FORBIDDEN_PROMPT_FIELDS & {str(key) for key in value}:
            return True
        return any(_has_forbidden_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_key(item) for item in value)
    return False


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, require_question_key: bool
) -> tuple[dict[str, Mapping[str, Any]], bool]:
    output: dict[str, Mapping[str, Any]] = {}
    valid = True
    for row in rows:
        dataset = str(row.get("dataset") or "").strip().lower()
        qid = str(row.get("qid") or "").strip()
        if dataset not in DATASETS or not qid:
            valid = False
            continue
        expected = question_key(dataset, qid)
        supplied = str(row.get("question_key") or "").strip()
        if require_question_key and supplied != expected:
            valid = False
        if expected in output:
            valid = False
        output[expected] = row
    return output, valid


def audit_core_assets(
    *,
    silver: Sequence[Mapping[str, Any]],
    question_kg: Sequence[Mapping[str, Any]],
    source_gates: Sequence[Mapping[str, Any]],
    weights: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    protected: Sequence[Mapping[str, Any]],
    historical_cutoff: str = "2020-12-09T23:59:59Z",
) -> dict[str, bool]:
    """Return the fixed scientific gates over already parsed rows."""

    silver_by_key, silver_identity_valid = _index_rows(
        silver, require_question_key=False
    )
    kg_by_key, kg_identity_valid = _index_rows(
        question_kg, require_question_key=True
    )
    gate_by_key, gate_identity_valid = _index_rows(
        source_gates, require_question_key=True
    )
    group_by_key, group_identity_valid = _index_rows(
        groups, require_question_key=True
    )
    keys = set(silver_by_key)

    dataset_counts = Counter(
        str(row.get("dataset") or "").strip().lower() for row in silver
    )
    scoped_qids = {
        (str(row.get("dataset") or "").strip().lower(), str(row.get("qid") or "").strip())
        for row in silver
    }
    scoped_hashes = {
        (
            str(row.get("dataset") or "").strip().lower(),
            question_sha256(str(row.get("question") or "").strip()),
        )
        for row in silver
    }
    identity_join = (
        silver_identity_valid
        and kg_identity_valid
        and gate_identity_valid
        and group_identity_valid
        and keys == set(kg_by_key) == set(gate_by_key) == set(group_by_key)
    )

    current_hashes_ok = identity_join and all(
        str(group_by_key[key].get("question") or "").strip()
        == str(silver_by_key[key].get("question") or "").strip()
        and str(group_by_key[key].get("question_sha256") or "")
        == question_sha256(str(silver_by_key[key].get("question") or "").strip())
        and str(kg_by_key[key].get("question_sha256") or "")
        == str(group_by_key[key].get("question_sha256") or "")
        and str(gate_by_key[key].get("question_sha256") or "")
        == str(group_by_key[key].get("question_sha256") or "")
        for key in keys
    )
    current_families_ok = identity_join and all(
        str(group_by_key[key].get("family_version") or "") == FAMILY_VERSION
        and str(group_by_key[key].get("family_sha256") or "")
        == family_sha256(str(silver_by_key[key].get("question") or "").strip())
        and str((silver_by_key[key].get("metadata") or {}).get("family_version") or "")
        == FAMILY_VERSION
        and str((silver_by_key[key].get("metadata") or {}).get("family_sha256") or "")
        == str(group_by_key[key].get("family_sha256") or "")
        for key in keys
    )

    kg_schema_ok = True
    for record in question_kg:
        try:
            validate_question_kg_record(record)
        except (TypeError, ValueError):
            kg_schema_ok = False
            break

    graph_keys = {
        key for key, row in gate_by_key.items() if int(row.get("m_graph", -1)) == 1
    }
    eligible_alignment = identity_join and all(
        bool(group_by_key[key].get("process_reward_eligible"))
        == bool((silver_by_key[key].get("metadata") or {}).get("process_reward_eligible"))
        == bool(kg_by_key[key].get("process_reward_eligible"))
        == bool(gate_by_key[key].get("graph_eligible"))
        == (key in graph_keys)
        for key in keys
    )
    source_gate_strict = identity_join and all(
        (
            int(gate_by_key[key].get("m_graph", -1)) == 1
            and all(bool(value) for value in (gate_by_key[key].get("eligibility_checks") or {}).values())
            and str(gate_by_key[key].get("proof_source") or "") not in ("", "none")
            and bool(str(gate_by_key[key].get("historical_cutoff") or "").strip())
            and bool(kg_by_key[key].get("kg_subgraph"))
        )
        if key in graph_keys
        else (
            int(gate_by_key[key].get("m_graph", -1)) == 0
            and gate_by_key[key].get("graph_eligible") is False
            and str(gate_by_key[key].get("proof_source") or "") == "none"
            and not kg_by_key[key].get("kg_subgraph")
        )
        for key in keys
    )
    source_gate_hashes = identity_join and all(
        str(gate_by_key[key].get("kg_sha256") or "")
        == _canonical_sha256(kg_by_key[key].get("kg_subgraph") or [])
        and str(gate_by_key[key].get("execution_sha256") or "")
        == _canonical_sha256(kg_by_key[key].get("execution") or {})
        for key in keys
    )
    source_gate_recomputed = identity_join and all(
        gate_by_key[key].get("schema_version") == SOURCE_GATE_SCHEMA_VERSION
        and gate_by_key[key].get("gate_version") == SOURCE_GATE_VERSION
        and dict(gate_by_key[key])
        == make_source_gate_record(
            kg_by_key[key],
            dataset=str(silver_by_key[key].get("dataset") or ""),
            qid=str(silver_by_key[key].get("qid") or ""),
            question=str(silver_by_key[key].get("question") or ""),
            text_evidence_available=True,
            historical_cutoff=historical_cutoff,
        )
        for key in keys
    )

    prompt_gold_boundary = identity_join and all(
        row.get("steps") == []
        and not str(row.get("teacher_output") or "").strip()
        and not row.get("passage_evidence")
        and row.get("evidence_mode") is None
        and str((row.get("metadata") or {}).get("gold_answer") or "").strip()
        and (row.get("metadata") or {}).get("gold_use") == "outcome_reward_label_only"
        and (row.get("metadata") or {}).get("source_gold_trace_removed") is True
        and (row.get("metadata") or {}).get("failed_qpeg_or_saeg_p_edges_included")
        is False
        and not _has_forbidden_key(row.get("retrieved_passages") or [])
        for row in silver
    ) and all(
        (row.get("provenance") or {}).get("gold_access") is False
        and not _has_forbidden_key(row)
        for row in question_kg
    ) and all(
        row.get("gold_access") is False and not _has_forbidden_key(row)
        for row in groups
    ) and all(not _has_forbidden_key(row) for row in source_gates)

    schedule_gates = validate_schedule_assets(groups, weights, groups, schedule)
    scheduled_graph = sum(
        int(gate_by_key.get(question_key(row.get("dataset"), row.get("qid")), {}).get("m_graph", 0))
        for row in schedule
    )
    rollout_indices_exact = [int(row.get("rollout_index", -1)) for row in schedule] == list(
        range(1, len(schedule) + 1)
    )

    population_replay_overlap = identity_overlap_counts(silver, replay)
    population_protected_overlap = identity_overlap_counts(silver, protected)
    replay_protected_overlap = identity_overlap_counts(replay, protected)

    checks = {
        "population_3000": len(silver) == len(keys) == 3000,
        "dataset_1000_each": dataset_counts
        == Counter({dataset: 1000 for dataset in DATASETS}),
        "dataset_scoped_qid_unique": len(scoped_qids) == 3000,
        "dataset_scoped_question_hash_unique": len(scoped_hashes) == 3000,
        "question_kg_source_gate_group_identity_join_1": identity_join,
        "current_question_hashes_recomputed": current_hashes_ok,
        "current_family_hashes_recomputed": current_families_ok,
        "question_kg_schema_valid": kg_schema_ok,
        "proof800_graph_exact_and_2wiki_only": len(graph_keys) == 800
        and all(key.startswith("2wikimultihopqa::") for key in graph_keys),
        "ordinary_non_graph_2200": len(keys - graph_keys) == 2200,
        "eligible_flag_alignment": eligible_alignment,
        "source_gate_strict_and_fail_closed": source_gate_strict,
        "source_gate_payload_hashes_exact": source_gate_hashes,
        "source_gate_schema_version_and_recomputation_exact": source_gate_recomputed,
        "all_exactly_ten_safe_passages": all(
            _ten_safe_passages(row.get("retrieved_passages")) for row in silver
        ),
        "gold_forbidden_from_prompt_evidence_and_traces": prompt_gold_boundary,
        "sampling_weights_3000": len(weights) == 3000,
        "prompt_groups_3000": len(groups) == 3000,
        "fixed_rollout_schedule_k4_12000": len(schedule) == 12000
        and bool(schedule_gates.get("schedule_k4_identity_exact"))
        and rollout_indices_exact,
        "scheduled_graph_trajectories_3200": scheduled_graph == 3200,
        "weights_identity_join_and_sum_one": bool(
            schedule_gates.get("weights_identity_join_1")
        )
        and bool(schedule_gates.get("weights_sum_1")),
        "groups_identity_join_once": bool(
            schedule_gates.get("groups_identity_join_1_and_once")
        ),
        "replay_exact_2000": len(replay) == 2000,
        "population_replay_qid_hash_current_family_overlap_zero": not any(
            population_replay_overlap.values()
        ),
        "population_protected_qid_hash_current_family_overlap_zero": not any(
            population_protected_overlap.values()
        ),
        "replay_protected_qid_hash_current_family_overlap_zero": not any(
            replay_protected_overlap.values()
        ),
    }
    return checks


def _validate_report_and_manifest(
    *,
    data_dir: Path,
    replay_dir: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    protected_binding: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    run = manifest.get("run") or {}
    output_bindings = report.get("outputs") or {}
    manifest_outputs = run.get("outputs") or {}
    report_path = data_dir / "report.json"
    protocol_identity = (report.get("inputs") or {}).get("protocol") or {}
    protocol_path = _safe_bound_path(protocol_identity) or Path("/__missing__")
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.is_file()
        else {}
    )
    protocol_manifest_identity = (report.get("inputs") or {}).get(
        "protocol_manifest"
    ) or {}
    protocol_manifest_path = _safe_bound_path(protocol_manifest_identity)
    protocol_manifest = (
        json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
        if protocol_manifest_path is not None
        else {}
    )

    replay_bindings = (report.get("inputs") or {}).get("replay") or {}
    replay_files = {name: replay_dir / name for name in REQUIRED_REPLAY_FILES}
    replay_bound = all(
        _identity_matches(replay_bindings.get(name), path)
        for name, path in replay_files.items()
    )
    manifest_replay = run.get("replay") or {}
    proof_binding = run.get("proof_supply") or {}
    report_proof_payloads = (report.get("inputs") or {}).get("proof_supply") or {}
    report_proof_release = (
        ((report.get("inputs") or {}).get("release_metadata") or {}).get("proof_supply")
        or {}
    )
    proof_paths = {
        name: _safe_bound_path(identity)
        for name, identity in report_proof_payloads.items()
    }
    proof_release_paths = {
        name: _safe_bound_path(identity)
        for name, identity in report_proof_release.items()
    }
    proof_files_bound = (
        set(report_proof_payloads) == set(EXPECTED_PROOF_SUPPLY_OUTPUTS)
        and set(report_proof_release) == {"report.json", "manifest.json"}
        and all(path is not None for path in proof_paths.values())
        and all(path is not None for path in proof_release_paths.values())
    )
    proof_report_identity = report_proof_release.get("report.json") or {}
    proof_report_path = _safe_bound_path(proof_report_identity)
    proof_report = (
        json.loads(proof_report_path.read_text(encoding="utf-8"))
        if proof_report_path is not None
        else {}
    )
    proof_manifest_path = _safe_bound_path(
        report_proof_release.get("manifest.json") or {}
    )
    proof_manifest = (
        json.loads(proof_manifest_path.read_text(encoding="utf-8"))
        if proof_manifest_path is not None
        else {}
    )
    proof_manifest_run = proof_manifest.get("run") or {}
    proof_checks = proof_report.get("checks") or {}
    proof_report_outputs = proof_report.get("outputs") or {}
    proof_output_chain = proof_files_bound and all(
        _identity_matches(
            proof_report_outputs.get(name),
            proof_paths[name] or Path("/__missing__"),
        )
        and report_proof_payloads.get(name) == proof_report_outputs.get(name)
        for name in EXPECTED_PROOF_SUPPLY_OUTPUTS
    )

    frozen_outputs = protocol.get("outputs") or {}
    copied_assets = ("sampling_weights", "prompt_groups", "fixed_rollout_schedule")
    copied_match = all(
        str((frozen_outputs.get(name) or {}).get("sha256") or "")
        == _sha256(paths[name])
        for name in copied_assets
    )
    release_metadata = ((report.get("inputs") or {}).get("release_metadata") or {})
    report_protected = release_metadata.get("protected_ledger") or {}
    protocol_run = protocol_manifest.get("run") or {}

    checks = {
        "report_schema_status_complete_not_trained": report.get("schema_version")
        == REPORT_SCHEMA
        and report.get("status") == DATA_STATUS
        and report.get("training_started") is False,
        "manifest_status_complete_not_trained": manifest.get("status") == DATA_STATUS
        and run.get("training_started") is False
        and run.get("phase") == "mixed_ppo_v4_proof800_data_materialization"
        and run.get("experiment_id") == report.get("experiment_id"),
        "materialization_gates_all_pass": bool(report.get("gates"))
        and all(bool(value) for value in (report.get("gates") or {}).values()),
        "report_output_hashes_exact": all(
            _identity_matches(output_bindings.get(name), path)
            for name, path in paths.items()
        ),
        "manifest_output_hashes_exact": all(
            _identity_matches(manifest_outputs.get(name), path)
            for name, path in paths.items()
        ),
        "manifest_report_hash_exact": _identity_matches(run.get("report"), report_path),
        "protocol_identity_hash_exact": _identity_matches(protocol_identity, protocol_path),
        "protocol_manifest_identity_hash_exact": protocol_manifest_path
        == protocol_path.parent / "manifest.json"
        and _identity_matches(protocol_manifest_identity, protocol_manifest_path)
        and run.get("protocol_manifest") == protocol_manifest_identity,
        "protocol_schema_status_and_gates": protocol.get("schema_version")
        == EXPECTED_PROTOCOL_SCHEMA
        and protocol.get("status") == EXPECTED_PROTOCOL_STATUS
        and protocol.get("experiment_id") == EXPECTED_PROTOCOL_EXPERIMENT
        and bool(protocol.get("gates"))
        and all(bool(value) for value in (protocol.get("gates") or {}).values())
        and protocol_manifest.get("status") == EXPECTED_PROTOCOL_STATUS
        and protocol_run.get("phase") == "mixed_ppo_v4_answer_free_protocol_freeze"
        and protocol_run.get("experiment_id") == EXPECTED_PROTOCOL_EXPERIMENT
        and protocol_run.get("protocol_sha256") == _sha256(protocol_path)
        and protocol_run.get("training_started") is False,
        "frozen_schedule_assets_byte_identical": copied_match,
        "replay_release_files_bound": replay_bound and manifest_replay == replay_bindings,
        "protected_ledger_live_hashes_bound": all(
            (report_protected.get(name) or {}).get("sha256")
            == identity.get("sha256")
            and _safe_bound_path(report_protected.get(name))
            == _safe_bound_path(identity)
            for name, identity in protected_binding.items()
        ),
        "official_unified_v3_payload_and_release_hashes_bound": proof_output_chain
        and proof_binding.get("schema_version") == EXPECTED_PROOF_SUPPLY_SCHEMA
        and proof_binding.get("status") == EXPECTED_PROOF_SUPPLY_STATUS
        and proof_binding.get("payloads") == report_proof_payloads
        and proof_binding.get("release_metadata") == report_proof_release
        and proof_report.get("schema_version") == EXPECTED_PROOF_SUPPLY_SCHEMA
        and proof_report.get("status") == EXPECTED_PROOF_SUPPLY_STATUS
        and proof_report.get("training_started") is False
        and bool(proof_checks)
        and all(bool(value) for value in proof_checks.values())
        and proof_manifest.get("status") == EXPECTED_PROOF_SUPPLY_STATUS
        and proof_manifest_run.get("phase")
        == "unified_2wiki_proofkg_official_raw_v3_candidate_supply"
        and proof_manifest_run.get("experiment_id")
        == proof_report.get("experiment_id")
        and proof_manifest_run.get("training_started") is False
        and _identity_matches(
            proof_manifest_run.get("report"),
            proof_report_path or Path("/__missing__"),
        ),
    }
    return checks


def run_preflight(
    *, data_dir: Path, replay_dir: Path, protected_ledger_dir: Path
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    replay_dir = replay_dir.resolve()
    protected_ledger_dir = protected_ledger_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not replay_dir.is_dir():
        raise FileNotFoundError(replay_dir)

    paths = {name: data_dir / filename for name, filename in REQUIRED_DATA_FILES.items()}
    missing = [str(path) for path in [*paths.values(), data_dir / "report.json", data_dir / "manifest.json"] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    replay_paths = [replay_dir / name for name in REQUIRED_REPLAY_FILES]
    missing_replay = [str(path) for path in replay_paths if not path.is_file()]
    if missing_replay:
        raise FileNotFoundError(missing_replay)

    report = json.loads((data_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    silver = _read_jsonl(paths["silver_train"])
    question_kg = _read_jsonl(paths["question_kg_records"])
    source_gates = _read_jsonl(paths["source_gate_records"])
    weights = _read_jsonl(paths["sampling_weights"])
    groups = _read_jsonl(paths["prompt_groups"])
    schedule = _read_jsonl(paths["fixed_rollout_schedule"])
    ledger_path, _ledger_report_path, _ledger_manifest_path, _ledger_report = (
        validate_protected_ledger_release(protected_ledger_dir)
    )
    protected = _read_jsonl(ledger_path)
    protected_binding = {
        "ledger": _identity(ledger_path),
        "report": _identity(_ledger_report_path),
        "manifest": _identity(_ledger_manifest_path),
    }
    protocol_identity = (report.get("inputs") or {}).get("protocol") or {}
    protocol_path = _resolve_bound_file(
        protocol_identity, label="final data protocol"
    )
    protocol, _frozen_paths, protocol_manifest_path = _load_protocol(protocol_path)

    parent_bindings = (report.get("inputs") or {}).get("parent") or {}
    parent_report_path = _resolve_bound_file(
        parent_bindings.get("report.json") or {}, label="final data parent report"
    )
    parent_files, _parent_rows = _validate_parent_release(
        parent_report_path.parent, final_protocol=protocol
    )
    parent_bound = set(parent_bindings) == set(parent_files) and all(
        _identity_matches(parent_bindings.get(name), path)
        for name, path in parent_files.items()
    )

    replay_files, replay = _validate_replay_release(
        replay_dir, protected_ledger_binding=protected_binding
    )
    replay_bindings = (report.get("inputs") or {}).get("replay") or {}
    replay_bound = set(replay_bindings) == set(replay_files) and all(
        _identity_matches(replay_bindings.get(name), path)
        for name, path in replay_files.items()
    )

    proof_bindings = (report.get("inputs") or {}).get("proof_supply") or {}
    if set(proof_bindings) != set(EXPECTED_PROOF_SUPPLY_OUTPUTS):
        raise ValueError("final data does not bind all official unified-v3 payloads")
    proof_files = {
        name: _resolve_bound_file(identity, label=f"final data proof supply {name}")
        for name, identity in proof_bindings.items()
    }
    proof_metadata = _validate_proof_supply_release(
        next(iter(proof_files.values())).parent,
        proof_files=proof_files,
        protected_ledger_binding=protected_binding,
    )
    proof_release_bindings = (
        ((report.get("inputs") or {}).get("release_metadata") or {}).get(
            "proof_supply"
        )
        or {}
    )
    proof_bound = set(proof_release_bindings) == set(proof_metadata) and all(
        _identity_matches(proof_release_bindings.get(name), path)
        for name, path in proof_metadata.items()
    )
    historical_cutoff = str(
        (report.get("gate_semantics") or {}).get("historical_cutoff") or ""
    ).strip()

    checks = audit_core_assets(
        silver=silver,
        question_kg=question_kg,
        source_gates=source_gates,
        weights=weights,
        groups=groups,
        schedule=schedule,
        replay=replay,
        protected=protected,
        historical_cutoff=historical_cutoff,
    )
    checks.update(
        {
            "historical_cutoff_exact": historical_cutoff
            == "2020-12-09T23:59:59Z",
            "protocol_source_release_strict": _identity_matches(
                (report.get("inputs") or {}).get("protocol_manifest"),
                protocol_manifest_path,
            ),
            "parent_source_release_strict": parent_bound,
            "replay_source_release_strict": replay_bound,
            "official_unified_v3_source_release_strict": proof_bound,
        }
    )
    checks.update(
        _validate_report_and_manifest(
            data_dir=data_dir,
            replay_dir=replay_dir,
            report=report,
            manifest=manifest,
            paths=paths,
            protected_binding=protected_binding,
        )
    )

    # Exercise the production reader/applicator after the independent identity
    # checks; a local set equality is not enough for the training path.
    reader = SilverDatasetReader(paths["silver_train"])
    records = read_question_kg_records(paths["question_kg_records"])
    join = apply_training_question_kg(
        reader.accepted(), records, min_coverage=1.0, require_nonempty=False
    ).to_dict()
    checks["production_question_kg_join_1"] = (
        join.get("coverage_rate") == 1.0
        and int(join.get("covered", -1)) == 3000
        and int(join.get("trajectories", -1)) == 3000
    )
    recorded_join = report.get("question_kg_join_stats") or {}
    checks["recorded_question_kg_join_matches_live"] = (
        recorded_join.get("coverage_rate") == join.get("coverage_rate")
        and int(recorded_join.get("covered", -1)) == int(join.get("covered", -2))
        and int(recorded_join.get("trajectories", -1))
        == int(join.get("trajectories", -2))
    )

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PREFLIGHT_STATUS_PASS if all(checks.values()) else PREFLIGHT_STATUS_FAIL,
        "counts": {
            "population": len(silver),
            "by_dataset": dict(sorted(Counter(row["dataset"] for row in silver).items())),
            "graph": sum(int(row.get("m_graph", -1)) == 1 for row in source_gates),
            "ordinary_non_graph": sum(int(row.get("m_graph", -1)) == 0 for row in source_gates),
            "prompt_groups": len(groups),
            "scheduled_trajectories": len(schedule),
            "replay": len(replay),
        },
        "checks": checks,
        "question_kg_join_stats": join,
        "inputs": {
            "data_report": _identity(data_dir / "report.json"),
            "data_manifest": _identity(data_dir / "manifest.json"),
            "replay_report": _identity(replay_dir / "report.json"),
            "replay_manifest": _identity(replay_dir / "manifest.json"),
            "protected_ledger": protected_binding,
        },
        "scientific_boundary": {
            "evaluation_protocol_modified": False,
            "reward_or_loss_modified": False,
            "training_started": False,
        },
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument(
        "--protected-ledger-dir", type=Path, default=DEFAULT_PROTECTED_LEDGER_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v4 preflight: {output_dir}")

    result = run_preflight(
        data_dir=args.data_dir,
        replay_dir=args.replay_dir,
        protected_ledger_dir=args.protected_ledger_dir,
    )
    result["experiment_id"] = str(args.experiment_id).strip()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=result["status"],
        extra={
            "phase": "mixed_ppo_v4_proof800_preflight",
            "experiment_id": result["experiment_id"],
            "report": _identity(report_path),
            "inputs": result["inputs"],
            "training_started": False,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != PREFLIGHT_STATUS_PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
