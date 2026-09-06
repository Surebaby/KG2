#!/usr/bin/env python3
"""Freeze and execute the strict, answer-blind 2Wiki Proof800 selection.

The command has two append-only stages:

``freeze``
    Bind the already frozen 1,500-question identity cohort, protected identity
    ledger, clean replay, ordinary-2Wiki arm, planner postflight, code versions,
    and the deterministic selection rule.  No ProofKG execution result is read.

``select``
    Join a completed clean-closure attestation to a completed unified ProofKG
    supply, re-run all structural/source/passage checks, and select exactly 200
    rows from each frozen 2Wiki question type.  The selector never reads an
    answer value or uses answer correctness in filtering/ranking.

The output ``proof_candidates.jsonl`` preserves the wrapper contract consumed
by ``freeze_mixed_ppo_three_dataset_v4_proof800.py``.  The full unified supply
remains the source of passages/outcome labels for the final v4 materializer;
this release merely freezes which 800 identities it may consume.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import rank
from scripts.prepare.freeze_mixed_ppo_three_dataset_v4_proof800 import (
    HISTORICAL_CUTOFF,
    PROTECTED_LEDGER_SCHEMA,
    PROTECTED_LEDGER_STATUS,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.materialize_2wiki_proofkg_unified_v2 import (
    CANDIDATE_SCHEMA_VERSION,
    SCHEMA_VERSION as UNIFIED_SUPPLY_SCHEMA,
    STATUS as UNIFIED_SUPPLY_STATUS,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = "2wikimultihopqa"
SEED = 42
QTYPES = ("bridge_comparison", "comparison", "compositional", "inference")
TARGET_BY_TYPE = {qtype: 200 for qtype in QTYPES}
TOTAL_TARGET = 800
PROTOCOL_SCHEMA = "2wiki-proof800-strict-selection-protocol-v1"
PROTOCOL_STATUS = "FROZEN_ANSWER_BLIND_SELECTION_POLICY_NOT_RUN_NOT_TRAINED"
RESULT_SCHEMA = "2wiki-proof800-strict-selection-result-v1"
RESULT_STATUS = "COMPLETE_STRICT_PROOF800_NOT_TRAINED"
SELECTION_RECORD_SCHEMA = "2wiki-proof800-selection-record-v1"
EXPERIMENT_ID_PROTOCOL = "2WIKI-PROOF800-STRICT-SELECTION-V1-SEED42-PROTOCOL"
EXPERIMENT_ID_RESULT = "2WIKI-PROOF800-STRICT-SELECTION-V1-SEED42-RESULT"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
QID = re.compile(r"^Q\d+$")
PID = re.compile(r"^P\d+$")

DEFAULT_COHORT_RELEASE = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_preregistration"
)
DEFAULT_PLANNER_POSTFLIGHT = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "seed42_plans_v1_postflight"
)
DEFAULT_LEDGER = ROOT / "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
DEFAULT_REPLAY = ROOT / (
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_"
    "seed42_v2"
)
DEFAULT_ORDINARY_PROTOCOL = ROOT / (
    "outputs/audits/2wiki_ordinary200_full_ledger_v2_seed42_preregistration/"
    "protocol.json"
)
DEFAULT_PROTOCOL_DIR = ROOT / (
    "outputs/audits/2wiki_proof800_strict_selection_v1_seed42_preregistration"
)
DEFAULT_RESULT_DIR = ROOT / (
    "outputs/audits/2wiki_proof800_strict_selection_v1_seed42_result"
)

FORBIDDEN_SELECTION_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "golden_answers",
    "support",
    "supporting_facts",
    "steps",
    "teacher_output",
    "target",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _passages_sha256(value: Any) -> str:
    # Match the unified supply's canonical retrieval hash contract.
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_identity(value: Mapping[str, Any], *, label: str) -> Path:
    raw = str(value.get("path") or "").strip()
    expected = str(value.get("sha256") or "").strip()
    if not raw or not HEX64.fullmatch(expected):
        raise ValueError(f"{label}: incomplete file identity")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label}: missing file or SHA256 drift")
    return path


def _index(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        dataset = str(row.get("dataset") or DATASET).strip().lower()
        qid = str(row.get("qid") or "").strip()
        key = str(row.get("question_key") or question_key(dataset, qid))
        if not qid or key in output:
            raise ValueError(f"{label}: empty/duplicate key {key!r}")
        output[key] = row
    return output


def _forbidden_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().lower() in FORBIDDEN_SELECTION_FIELDS:
                return True
            if _forbidden_present(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_forbidden_present(item) for item in value)
    return False


def _load_release_file(
    report: Mapping[str, Any], name: str, *, release_dir: Path
) -> Path:
    value = (report.get("outputs") or {}).get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"release does not bind output {name}")
    return _resolve_identity(value, label=f"release output {name}")


def _validate_replay_release(directory: Path) -> tuple[Path, dict[str, Any]]:
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        report.get("schema_version") != "sft-replay-rendered-3to5-v2-clean-isolated"
        or report.get("status") != "COMPLETE_DATA_NOT_TRAINED"
        or manifest.get("status") != "COMPLETE_DATA_NOT_TRAINED"
        or not all(bool(v) for v in (report.get("gates") or {}).values())
        or (manifest.get("run") or {}).get("training_started") is not False
    ):
        raise ValueError("clean replay release status/schema/gates failed")
    rows_path = _load_release_file(report, "selection_records", release_dir=directory)
    if (manifest.get("run") or {}).get("selection_records_sha256") != _sha256(rows_path):
        raise ValueError("clean replay manifest/output binding drifted")
    return rows_path, {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "selection_records": _identity(rows_path),
    }


def _validate_ordinary_release(protocol_path: Path) -> tuple[Path, dict[str, Any]]:
    protocol = _read_json(protocol_path)
    if (
        protocol.get("schema_version") != "2wiki-ordinary200-full-ledger-protocol-v2"
        or protocol.get("status")
        != "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED"
        or not all(
            bool(v)
            for v in (
                ((protocol.get("selection") or {}).get("counts") or {}).get("gates")
                or {}
            ).values()
        )
    ):
        raise ValueError("ordinary200 release status/schema/gates failed")
    rows_path = _load_release_file(protocol, "ordinary200", release_dir=protocol_path.parent)
    rows = _read_jsonl(rows_path)
    if len(rows) != 200:
        raise ValueError("ordinary200 release is not exactly 200 rows")
    return rows_path, {
        "protocol": _identity(protocol_path),
        "ordinary200": _identity(rows_path),
    }


def _cohort_rows(release_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = release_dir / "report.json"
    protocol_path = release_dir / "protocol.json"
    manifest_path = release_dir / "manifest.json"
    report = _read_json(report_path)
    protocol = _read_json(protocol_path)
    manifest = _read_json(manifest_path)
    status = "FROZEN_GOLD_FREE_BEFORE_PLANNER_NOT_MATERIALIZED_NOT_TRAINED"
    if report.get("status") != status or protocol.get("status") != status or manifest.get("status") != status:
        raise ValueError("n1500 cohort release status failed")
    cohort_path = _load_release_file(report, "cohort", release_dir=release_dir)
    rows = _read_jsonl(cohort_path)
    counts = Counter(str(row.get("question_type") or "") for row in rows)
    expected = Counter(
        {"bridge_comparison": 390, "comparison": 390, "compositional": 389, "inference": 331}
    )
    keys: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        question = str(row.get("question") or "").strip()
        key = question_key(str(row.get("dataset") or ""), str(row.get("qid") or ""))
        if (
            row.get("schema_version") != "2wiki-proofkg-official-raw-question-only-v2"
            or str(row.get("dataset") or "").strip().lower() != DATASET
            or row.get("question_key") not in (None, key)
            or str(row.get("question_sha256") or "") != question_sha256(question)
            or str(row.get("family_sha256") or "") != family_sha256(question)
            or row.get("family_version") != FAMILY_VERSION
            or row.get("gold_access") is not False
            or _forbidden_present(row)
            or key in keys
            or str(row.get("question_sha256")) in hashes
        ):
            raise ValueError(f"n1500 cohort identity/gold boundary failed: {key}")
        keys.add(key)
        hashes.add(str(row["question_sha256"]))
    if len(rows) != 1500 or counts != expected:
        raise ValueError(f"n1500 cohort counts drifted: n={len(rows)}, counts={counts}")
    return rows, {
        "cohort": _identity(cohort_path),
        "protocol": _identity(protocol_path),
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
    }


def _validate_planner_postflight(directory: Path) -> dict[str, Any]:
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    status = "PASS_PLANNER_STRUCTURAL_NOT_PROOFKG_MATERIALIZED_NOT_TRAINED"
    if (
        report.get("schema_version") != "2wiki-official-raw-plans-postflight-v1"
        or report.get("status") != status
        or manifest.get("status") != status
        or not all(bool(v) for v in (report.get("gates") or {}).values())
        or ((report.get("scientific_boundary") or {}).get("training_started") is not False)
    ):
        raise ValueError("planner postflight status/schema/gates failed")
    predictions = ((report.get("inputs") or {}).get("predictions"))
    if not isinstance(predictions, Mapping):
        raise ValueError("planner postflight does not bind predictions")
    # This historical postflight records MD5, while this selector freezes a
    # fresh SHA256 identity for the exact same file.
    prediction_path = Path(str(predictions.get("path") or ""))
    if not prediction_path.is_absolute():
        prediction_path = ROOT / prediction_path
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    return {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "predictions": _identity(prediction_path),
    }


def _identity_triplets(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    qids: set[str] = set()
    hashes: set[str] = set()
    families: set[str] = set()
    for row in rows:
        if str(row.get("dataset") or DATASET).strip().lower() != DATASET:
            continue
        qid = str(row.get("qid") or "").strip()
        question = str(row.get("question") or "").strip()
        qhash = str(row.get("question_sha256") or question_sha256(question))
        family = str(row.get("family_sha256") or family_sha256(question))
        if not qid or not question or qhash != question_sha256(question) or family != family_sha256(question):
            raise ValueError("blocked identity is incomplete or hash-drifted")
        qids.add(qid)
        hashes.add(qhash)
        families.add(family)
    return qids, hashes, families


def freeze_protocol(
    *,
    cohort_release: Path,
    planner_postflight: Path,
    protected_ledger_dir: Path,
    replay_dir: Path,
    ordinary_protocol: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite selection protocol: {output_dir}")
    cohort, cohort_binding = _cohort_rows(cohort_release)
    planner_binding = _validate_planner_postflight(planner_postflight)
    ledger_path, ledger_binding = validate_protected_ledger_release(protected_ledger_dir)
    replay_path, replay_binding = _validate_replay_release(replay_dir)
    ordinary_path, ordinary_binding = _validate_ordinary_release(ordinary_protocol)

    blocked_rows = [
        *_read_jsonl(ledger_path),
        *_read_jsonl(replay_path),
        *_read_jsonl(ordinary_path),
    ]
    blocked_qids, blocked_hashes, blocked_families = _identity_triplets(blocked_rows)
    overlaps = Counter()
    for row in cohort:
        overlaps["qid"] += str(row["qid"]) in blocked_qids
        overlaps["question_sha256"] += str(row["question_sha256"]) in blocked_hashes
        overlaps["family_sha256"] += str(row["family_sha256"]) in blocked_families
    gates = {
        "candidate_cohort_n1500_exact": len(cohort) == 1500,
        "candidate_question_type_counts_exact": Counter(
            str(row["question_type"]) for row in cohort
        )
        == Counter(
            {"bridge_comparison": 390, "comparison": 390, "compositional": 389, "inference": 331}
        ),
        "candidate_protected_replay_ordinary_qid_hash_family_overlap_zero": not any(overlaps.values()),
        "planner_postflight_pass": True,
        "answer_blind_selection_fields_only": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Proof800 selection preregistration gates failed: {gates}; overlap={dict(overlaps)}")

    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID_PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PROTOCOL_STATUS,
        "scope": "strict Proof800 selection from the frozen official-raw n1500 2Wiki train-only cohort",
        "historical_cutoff": HISTORICAL_CUTOFF,
        "selection": {
            "seed": SEED,
            "target_total": TOTAL_TARGET,
            "target_by_question_type": TARGET_BY_TYPE,
            "candidate_universe": "exact dataset::qid/question-hash subset of frozen official-raw n1500 cohort",
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
                "primary": "one deterministic representative per current lexical family within question type",
                "secondary": "deterministic repeat-family rows only after all distinct families in that type",
                "within_family_rank": "sha256(seed, proof800-v1-within-family, dataset, qid)",
                "family_rank": "sha256(seed, proof800-v1-family, dataset, family_sha256)",
                "repeat_rank": "sha256(seed, proof800-v1-repeat, dataset, qid)",
                "question_type_order": list(QTYPES),
                "cross_type_qid_and_question_hash_reuse": "forbidden",
                "cross_type_family_reuse": "allowed_but_reported; family is not an outcome label",
            },
            "failure_policy": "if any type has fewer than 200 strict rows, write no result release and fail; thresholds are not lowered",
        },
        "required_future_inputs": {
            "closure_release": "versioned complete clean closure with runtime and strict telemetry SHA bindings",
            "unified_supply": {
                "schema_version": UNIFIED_SUPPLY_SCHEMA,
                "status": UNIFIED_SUPPLY_STATUS,
                "required_outputs": [
                    "silver_train",
                    "question_kg_records",
                    "source_gate_records",
                    "proof_candidates",
                ],
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
            "selector": _identity(Path(__file__)),
            "source_gate": _identity(ROOT / "kgproweight/reward/trajectory_source_gate.py"),
            "unified_materializer": _identity(ROOT / "scripts/prepare/materialize_2wiki_proofkg_unified_v2.py"),
            "v4_freezer": _identity(ROOT / "scripts/prepare/freeze_mixed_ppo_three_dataset_v4_proof800.py"),
            "v4_materializer": _identity(ROOT / "scripts/prepare/materialize_mixed_ppo_three_dataset_v4_proof800.py"),
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
        "experiment_id": EXPERIMENT_ID_PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": PROTOCOL_STATUS,
        "candidate_counts": {
            "total": len(cohort),
            "by_question_type": dict(
                sorted(Counter(str(row["question_type"]) for row in cohort).items())
            ),
            "unique_current_families": len({str(row["family_sha256"]) for row in cohort}),
            "blocked_overlap": dict(overlaps),
        },
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
        status=PROTOCOL_STATUS,
        extra={
            "phase": "proof800_strict_selection_preregistration",
            "experiment_id": EXPERIMENT_ID_PROTOCOL,
            "protocol": _identity(protocol_path),
            "report": _identity(report_path),
            "selection_started": False,
            "training_started": False,
        },
    )
    return report


def _validate_protocol(protocol_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_path = protocol_dir / "protocol.json"
    report_path = protocol_dir / "report.json"
    manifest_path = protocol_dir / "manifest.json"
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or report.get("status") != PROTOCOL_STATUS
        or manifest.get("status") != PROTOCOL_STATUS
        or not all(bool(v) for v in (protocol.get("gates") or {}).values())
        or (protocol.get("scientific_boundary") or {}).get("training_started") is not False
    ):
        raise ValueError("Proof800 selection protocol status/schema/gates failed")
    if _sha256(protocol_path) != ((manifest.get("run") or {}).get("protocol") or {}).get("sha256"):
        raise ValueError("Proof800 protocol manifest binding drifted")
    for section in (protocol.get("inputs") or {}).values():
        if not isinstance(section, Mapping):
            raise ValueError("Proof800 protocol input binding malformed")
        for label, value in section.items():
            if isinstance(value, Mapping) and "path" in value:
                _resolve_identity(value, label=f"protocol input {label}")
    for label, value in (protocol.get("code") or {}).items():
        _resolve_identity(value, label=f"protocol code {label}")
    cohort_identity = ((protocol.get("inputs") or {}).get("candidate_cohort_release") or {}).get("cohort")
    if not isinstance(cohort_identity, Mapping):
        raise ValueError("Proof800 protocol does not bind the cohort")
    cohort = _read_jsonl(_resolve_identity(cohort_identity, label="protocol cohort"))
    return protocol, cohort


def _validate_closure_release(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a closure attestation by contract, without guessing absent data."""

    report_path = directory / "report.json"
    if not report_path.is_file():
        # Some closure runners use closure_report.json but must still bind it
        # from their manifest; accept the name while keeping hash validation.
        report_path = directory / "closure_report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        report.get("schema_version") != "2wiki-official-raw-clean-closure-v2"
        or report.get("status")
        != "COMPLETE_DIAGNOSTIC_CLEAN_CLOSURE_NOT_SELECTED_NOT_TRAINED"
        or report.get("all_pass") is not True
        or report.get("decision") != "CONTINUE_TO_PROOF800_SELECTION"
        or report.get("gold_access") is not False
        or report.get("training_started") is not False
        or manifest.get("status") != report.get("status")
        or not all(bool(v) for v in (report.get("gates") or {}).values())
    ):
        raise ValueError("closure release status/gates/training boundary failed")
    boundary = report.get("scientific_boundary") or {}
    if not (
        boundary.get("structural_and_source_eligibility_only") is True
        and boundary.get("passages_or_answers_read") is False
        and boundary.get("proof800_selected") is False
        and boundary.get("training_started") is False
    ):
        raise ValueError("closure release scientific boundary failed")
    runtime_path = _load_release_file(report, "runtime_details", release_dir=directory)
    telemetry_path = _load_release_file(report, "strict_eligibility_telemetry", release_dir=directory)
    telemetry_identity = (report.get("outputs") or {}).get(
        "strict_eligibility_telemetry"
    ) or {}
    if int(telemetry_identity.get("rows", -1)) != 1500:
        raise ValueError("closure telemetry report row binding is not 1500")
    manifest_report = (manifest.get("run") or {}).get("report") or {}
    if (
        not isinstance(manifest_report, Mapping)
        or manifest_report.get("sha256") != _sha256(report_path)
        or (manifest.get("run") or {}).get("training_started") is not False
    ):
        raise ValueError("closure manifest/report binding drifted")
    runtime = _index(_read_jsonl(runtime_path), label="closure runtime")
    telemetry = _index(_read_jsonl(telemetry_path), label="closure telemetry")
    if set(runtime) != set(telemetry):
        raise ValueError("closure runtime/telemetry identity join is not exact")
    expected_types = Counter(
        {"bridge_comparison": 390, "comparison": 390, "compositional": 389, "inference": 331}
    )
    if len(telemetry) != 1500 or Counter(
        str(row.get("question_type") or "") for row in telemetry.values()
    ) != expected_types:
        raise ValueError("closure telemetry n1500/question-type population drifted")
    for key, trace in runtime.items():
        row = telemetry[key]
        if (
            row.get("schema_version")
            != "2wiki-official-raw-strict-eligibility-telemetry-v1"
            or row.get("gold_access_false") is not True
            or _forbidden_present(row)
            or str(row.get("runtime_record_sha256") or "") != _canonical_sha256(trace)
            or str(row.get("kg_sha256") or "")
            != _canonical_sha256(trace.get("kg_subgraph") or [])
            or str(row.get("execution_sha256") or "")
            != _canonical_sha256(trace.get("execution") or {})
        ):
            raise ValueError(f"closure runtime/telemetry hash mismatch: {key}")
    return telemetry, {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        "runtime_details": _identity(runtime_path),
        "strict_eligibility_telemetry": _identity(telemetry_path),
    }


def _validate_unified_supply(
    directory: Path,
    *,
    protected_ledger_binding: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        report.get("schema_version") != UNIFIED_SUPPLY_SCHEMA
        or report.get("status") != UNIFIED_SUPPLY_STATUS
        or manifest.get("status") != UNIFIED_SUPPLY_STATUS
        or not all(bool(v) for v in (report.get("checks") or {}).values())
        or report.get("training_started") is not False
        or (manifest.get("run") or {}).get("training_started") is not False
    ):
        raise ValueError("unified supply status/schema/checks failed")
    supply_ledger = report.get("protected_ledger") or {}
    if not (
        supply_ledger.get("version") == PROTECTED_LEDGER_SCHEMA
        and supply_ledger.get("complete") is True
        and supply_ledger.get("current_family_recomputed") is True
    ):
        raise ValueError("unified supply lacks complete protected-ledger binding")
    for name in ("ledger", "report", "manifest"):
        actual = supply_ledger.get(name)
        expected = protected_ledger_binding.get(name)
        if (
            not isinstance(actual, Mapping)
            or not isinstance(expected, Mapping)
            or actual.get("sha256") != expected.get("sha256")
        ):
            raise ValueError(f"unified supply protected-ledger mismatch: {name}")
    paths = {
        name: _load_release_file(report, name, release_dir=directory)
        for name in (
            "silver_train",
            "question_kg_records",
            "source_gate_records",
            "proof_candidates",
        )
    }
    if _sha256(report_path) != ((manifest.get("run") or {}).get("report") or {}).get("sha256"):
        raise ValueError("unified supply manifest/report binding drifted")
    rows = {
        "silver": _index(_read_jsonl(paths["silver_train"]), label="unified silver"),
        "records": _index(_read_jsonl(paths["question_kg_records"]), label="unified KG"),
        "gates": _index(_read_jsonl(paths["source_gate_records"]), label="unified gate"),
        "wrappers": _index(_read_jsonl(paths["proof_candidates"]), label="unified wrappers"),
    }
    sets = {name: set(value) for name, value in rows.items()}
    if len({frozenset(value) for value in sets.values()}) != 1:
        raise ValueError(f"unified supply four-way identity join failed: { {k: len(v) for k, v in sets.items()} }")
    return rows, {
        "report": _identity(report_path),
        "manifest": _identity(manifest_path),
        **{name: _identity(path) for name, path in paths.items()},
    }


def _root_anchors_resolved(record: Mapping[str, Any]) -> bool:
    plan = record.get("query_plan") or {}
    execution = record.get("execution") or {}
    anchors = list(plan.get("anchors") or [])
    resolved = execution.get("anchor_entities") or {}
    return bool(anchors) and isinstance(resolved, Mapping) and all(
        isinstance(resolved.get(str(anchor)), Mapping)
        and QID.fullmatch(str(resolved[str(anchor)].get("qid") or "")) is not None
        and resolved[str(anchor)].get("abstained") is False
        for anchor in anchors
    )


def _hops_complete_and_traceable(record: Mapping[str, Any]) -> bool:
    plan = record.get("query_plan") or {}
    execution = record.get("execution") or {}
    planned = list(plan.get("hops") or [])
    executed = list(execution.get("hops") or [])
    by_index: dict[int, Mapping[str, Any]] = {}
    for hop in executed:
        if not isinstance(hop, Mapping):
            return False
        try:
            index = int(hop.get("hop_index", -1))
        except (TypeError, ValueError):
            return False
        if index in by_index:
            return False
        by_index[index] = hop
    if not planned or set(by_index) != set(range(1, len(planned) + 1)):
        return False
    trace: set[tuple[str, str, str]] = set()
    for index, planned_hop in enumerate(planned, start=1):
        if not isinstance(planned_hop, Mapping):
            return False
        actual = by_index[index]
        planned_pids = [str(pid).strip() for pid in (planned_hop.get("pids") or [])]
        actual_pids = [str(pid).strip() for pid in (actual.get("pids") or [])]
        inputs = list(actual.get("input_entities") or [])
        matches = list(actual.get("matches") or [])
        if (
            not planned_pids
            or not all(PID.fullmatch(pid) for pid in planned_pids)
            or actual_pids != planned_pids
            or not inputs
            or not all(
                isinstance(entity, Mapping)
                and QID.fullmatch(str(entity.get("qid") or "")) is not None
                and entity.get("abstained") in (None, False)
                for entity in inputs
            )
            or not matches
        ):
            return False
        for value in matches:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return False
            triple = tuple(str(part).strip() for part in value)
            if not all(triple):
                return False
            trace.add(triple)
    retained = [
        tuple(str(part).strip() for part in value)
        for value in (record.get("kg_subgraph") or [])
        if isinstance(value, (list, tuple)) and len(value) == 3
    ]
    return (
        bool(retained)
        and all(all(triple) for triple in retained)
        and len(retained) == len(set(retained))
        and set(retained).issubset(trace)
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
) -> tuple[bool, dict[str, bool]]:
    """Return a fully mechanical strict-admission decision for one row."""

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
            for row in (wrapper, silver, record)
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
        == CANDIDATE_SCHEMA_VERSION
        and wrapper.get("gold_access") is False
        and not _forbidden_present({k: v for k, v in wrapper.items() if k != "question_kg_record"}),
        "record_schema_valid": True,
        "planner_schema_valid": record.get("planner_schema_valid") is True
        and (record.get("query_plan") or {}).get("recognized") is True,
        "gold_access_false": (record.get("provenance") or {}).get("gold_access") is False
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
        and all(bool(v) for v in (gate.get("eligibility_checks") or {}).values())
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


def choose_exact_proof800(
    admitted: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose exact per-type quotas with deterministic family diversity first."""

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in admitted:
        row = dict(raw)
        qtype = str(row.get("question_type") or "")
        if qtype not in TARGET_BY_TYPE:
            raise ValueError(f"admitted candidate has invalid question type {qtype!r}")
        by_type[qtype].append(row)

    selected: list[dict[str, Any]] = []
    type_stats: dict[str, Any] = {}
    used_qids: set[str] = set()
    used_hashes: set[str] = set()
    for qtype in QTYPES:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_type[qtype]:
            grouped[str(row["family_sha256"])].append(row)
        for family, rows in grouped.items():
            rows.sort(
                key=lambda row: (
                    rank("proof800-v1-within-family", DATASET, str(row["qid"]), seed=SEED),
                    str(row["qid"]),
                )
            )
        families = sorted(
            grouped,
            key=lambda family: (
                rank("proof800-v1-family", DATASET, family, seed=SEED),
                family,
            ),
        )
        distinct = [grouped[family][0] for family in families]
        repeats = sorted(
            [row for family in families for row in grouped[family][1:]],
            key=lambda row: (
                rank("proof800-v1-repeat", DATASET, str(row["qid"]), seed=SEED),
                str(row["qid"]),
            ),
        )
        ordered = [*distinct, *repeats]
        chosen: list[dict[str, Any]] = []
        for row in ordered:
            qid = str(row["qid"])
            qhash = str(row["question_sha256"])
            if qid in used_qids or qhash in used_hashes:
                continue
            chosen.append(dict(row))
            used_qids.add(qid)
            used_hashes.add(qhash)
            if len(chosen) == TARGET_BY_TYPE[qtype]:
                break
        if len(chosen) != TARGET_BY_TYPE[qtype]:
            raise RuntimeError(
                f"Proof800/{qtype}: only {len(chosen)}/{TARGET_BY_TYPE[qtype]} strict candidates; selection aborted without lowering gates"
            )
        selected.extend(chosen)
        type_stats[qtype] = {
            "admitted": len(by_type[qtype]),
            "admitted_unique_families": len(grouped),
            "selected": len(chosen),
            "selected_unique_families": len({str(row["family_sha256"]) for row in chosen}),
            "selected_repeated_family_rows": len(chosen)
            - len({str(row["family_sha256"]) for row in chosen}),
        }
    selected.sort(
        key=lambda row: (
            QTYPES.index(str(row["question_type"])),
            rank("proof800-v1-output", DATASET, str(row["qid"]), seed=SEED),
            str(row["qid"]),
        )
    )
    return selected, {
        "by_question_type": type_stats,
        "selected_total": len(selected),
        "selected_unique_families": len({str(row["family_sha256"]) for row in selected}),
        "selected_repeated_family_rows_global": len(selected)
        - len({str(row["family_sha256"]) for row in selected}),
    }


def select_release(
    *,
    protocol_dir: Path,
    closure_dir: Path,
    unified_supply_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Proof800 result: {output_dir}")
    protocol, cohort_rows = _validate_protocol(protocol_dir)
    cohort = _index(cohort_rows, label="frozen n1500 cohort")
    closure, closure_binding = _validate_closure_release(closure_dir)
    protected_ledger_binding = protocol["inputs"]["protected_ledger_release"]
    supply, supply_binding = _validate_unified_supply(
        unified_supply_dir,
        protected_ledger_binding=protected_ledger_binding,
    )

    blocked_sections = protocol["inputs"]
    ledger_path = _resolve_identity(
        blocked_sections["protected_ledger_release"]["ledger"], label="protected ledger"
    )
    replay_path = _resolve_identity(
        blocked_sections["clean_replay_release"]["selection_records"], label="clean replay identities"
    )
    ordinary_path = _resolve_identity(
        blocked_sections["ordinary200_release"]["ordinary200"], label="ordinary200 identities"
    )
    blocked_qids, blocked_hashes, blocked_families = _identity_triplets(
        [*_read_jsonl(ledger_path), *_read_jsonl(replay_path), *_read_jsonl(ordinary_path)]
    )

    funnel = Counter()
    failed_check_counts = Counter()
    admitted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidate_keys = set(cohort)
    supply_keys = set(supply["wrappers"])
    if set(closure) != candidate_keys:
        raise ValueError(
            "closure telemetry does not exactly cover frozen n1500 cohort: "
            f"closure={len(closure)} cohort={len(candidate_keys)}"
        )
    planner_predictions_sha256 = str(
        protocol["inputs"]["planner_postflight"]["predictions"]["sha256"]
    )
    for key in sorted(candidate_keys):
        funnel["frozen_candidate_universe"] += 1
        base = cohort[key]
        missing = [name for name in ("wrappers", "silver", "records", "gates") if key not in supply[name]]
        if key not in closure:
            missing.append("closure_telemetry")
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
        ok, checks = assess_candidate(
            cohort=base,
            wrapper=wrapper,
            silver=supply["silver"][key],
            record=supply["records"][key],
            gate=supply["gates"][key],
            closure_telemetry=closure[key],
            historical_cutoff=str(protocol["historical_cutoff"]),
            planner_predictions_sha256=planner_predictions_sha256,
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
            item["proof_record_sha256"] = _canonical_sha256(supply["records"][key])
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
                "proof_passages_sha256": str(wrapper.get("proof_passages_sha256") or ""),
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
        "selected_200_each_question_type": selected_by_type == Counter(TARGET_BY_TYPE),
        "selected_subset_of_frozen_n1500": selected_keys.issubset(candidate_keys),
        "selected_all_strict_admitted": selected_keys.issubset(
            {str(row["question_key"]) for row in admitted}
        ),
        "selected_qid_unique": len({str(row["qid"]) for row in selected}) == TOTAL_TARGET,
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
        "selected_gold_access_false": all(row.get("gold_access") is False for row in selected),
        "selection_answer_blind": True,
        "training_not_started": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Proof800 final gates failed: {gates}")

    output_dir.mkdir(parents=True, exist_ok=False)
    proof_path = output_dir / "proof_candidates.jsonl"
    audit_path = output_dir / "selection_records.question_only.jsonl"
    selected_records_path = output_dir / "question_kg_records.jsonl"
    selected_gates_path = output_dir / "source_gate_records.jsonl"
    _write_jsonl(proof_path, selected)
    _write_jsonl(audit_path, audit_rows)
    _write_jsonl(
        selected_records_path, (supply["records"][key] for key in sorted(selected_keys))
    )
    _write_jsonl(
        selected_gates_path, (supply["gates"][key] for key in sorted(selected_keys))
    )
    outputs = {
        "proof_candidates": _identity(proof_path),
        "selection_records": _identity(audit_path),
        "question_kg_records": _identity(selected_records_path),
        "source_gate_records": _identity(selected_gates_path),
    }
    report = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID_RESULT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": RESULT_STATUS,
        "funnel": {
            **dict(sorted(funnel.items())),
            "unified_supply_outside_frozen_cohort_ignored": len(supply_keys - candidate_keys),
            "failed_check_counts": dict(sorted(failed_check_counts.items())),
            "strict_admitted_by_question_type": dict(
                sorted(Counter(str(row["question_type"]) for row in admitted).items())
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
            "closure_release": closure_binding,
            "unified_supply": supply_binding,
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
            "phase": "proof800_strict_selection",
            "experiment_id": EXPERIMENT_ID_RESULT,
            "report": _identity(report_path),
            "outputs": outputs,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze answer-blind selection policy")
    freeze.add_argument("--cohort-release", type=Path, default=DEFAULT_COHORT_RELEASE)
    freeze.add_argument("--planner-postflight", type=Path, default=DEFAULT_PLANNER_POSTFLIGHT)
    freeze.add_argument("--protected-ledger-dir", type=Path, default=DEFAULT_LEDGER)
    freeze.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    freeze.add_argument("--ordinary-protocol", type=Path, default=DEFAULT_ORDINARY_PROTOCOL)
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)

    select = subparsers.add_parser("select", help="materialize exact strict Proof800")
    select.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    select.add_argument("--closure-dir", type=Path, required=True)
    select.add_argument("--unified-supply-dir", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    if args.command == "freeze":
        report = freeze_protocol(
            cohort_release=args.cohort_release,
            planner_postflight=args.planner_postflight,
            protected_ledger_dir=args.protected_ledger_dir,
            replay_dir=args.replay_dir,
            ordinary_protocol=args.ordinary_protocol,
            output_dir=args.output_dir,
        )
    else:
        report = select_release(
            protocol_dir=args.protocol_dir,
            closure_dir=args.closure_dir,
            unified_supply_dir=args.unified_supply_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
