"""Gold-free greedy generation and mechanical scoring for Controller actions."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

from kgproweight.eval.query_controller_v1 import (
    audit_action_record,
    parse_target_response,
    validate_action_record,
)
from kgproweight.training.query_controller import (
    _canonical_target,
    _verify_implementation_locks,
    controller_messages,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed


PREDICTION_SCHEMA_VERSION = "query-controller-greedy-prediction-v1"
PROTOCOL_SCHEMA_VERSION = "query-controller-v1-pilot-protocol-4.4"
PROTOCOL_REPORT_SCHEMA_VERSION = "query-controller-v1-pilot-freeze-report-4.4"
PROTOCOL_MANIFEST_SCHEMA_VERSION = "query-controller-v1-pilot-manifest-4.4"
PROTOCOL_STATUS = (
    "FROZEN_V4_4_PEFT_DEFAULT_FP32_CLEAN_RELOAD_SAME_IDENTITIES_"
    "2DATASET_ACTIONS_NOT_TRAINED"
)
PROTOCOL_EXPERIMENT_ID = "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_4"
RELEASE_SCHEMA_VERSION = "query-controller-action-release-v4.4-pair-locked-same-identities"
RELEASE_STATUS = "COMPLETE_V4_4_PAIR_LOCKED_SAME_IDENTITIES_NOT_TRAINED"
RELEASE_EXPERIMENT_ID = "QUERY-CONTROLLER-ACTION-V1-SEED42-V4-4"
PROBE_EXPERIMENT_ID = "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4"
EVAL_PROTOCOL_SCHEMA_VERSION = "query-controller-v1-eval-hotfix-protocol-1.0"
EVAL_PROTOCOL_REPORT_SCHEMA_VERSION = (
    "query-controller-v1-eval-hotfix-freeze-report-1.0"
)
EVAL_PROTOCOL_MANIFEST_SCHEMA_VERSION = "query-controller-v1-eval-hotfix-manifest-1.0"
EVAL_PROTOCOL_EXPERIMENT_ID = "QUERY-CONTROLLER-V1-V4-4-DEV-EVAL-E1-PROTOCOL"
EVAL_PROTOCOL_STATUS = (
    "FROZEN_EVAL_ONLY_PARENT_V4_4_COMPLETE_HASH_FORWARDING_HOTFIX_NOT_RUN"
)
EVAL_GENERATION_EXPERIMENT_ID = (
    "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4-DEV-EVAL-E1"
)
EVAL_GENERATION_OUTPUT_DIR = (
    "outputs/validation/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1"
)
EVAL_PREDECESSOR_FAILURE_PATH = (
    "outputs/audits/query_controller_v4_4_dev_eval_e0_pre_cuda_failure/"
    "metadata_addendum.json"
)
EVAL_PREDECESSOR_FAILURE_SHA256 = (
    "51f78375cff772f4dcdc166a712660a29c0d9a1bfb2ff380ea37809aa3a0fa34"
)
EVAL_IMPLEMENTATION_ROLES = frozenset(
    {
        "eval_protocol_freezer",
        "controller_greedy_runner",
        "controller_generate_cli",
        "controller_mechanism_scorer",
    }
)
ENABLED_DATASETS = ("2wikimultihopqa", "musique")
CONTROLLER_SLOTS = ("q1", "q2_dynamic")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_external_sha256(value: str, *, role: str) -> str:
    import re

    if re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is None:
        raise ValueError(f"a valid external expected SHA256 is required for {role}")
    return str(value)


def _load_json_object(path: str | Path, *, role: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{role} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{role} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return value


def _manifest_run(document: Mapping[str, Any]) -> Mapping[str, Any]:
    run = document.get("run")
    return run if isinstance(run, Mapping) else document


def _identity_lock(document: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    locks = document.get("identity_locks")
    if not isinstance(locks, list):
        raise ValueError("release identity_locks must be a list")
    matches = [row for row in locks if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"release requires exactly one identity lock for role={role}")
    return matches[0]


def _resolve_protocol_asset(protocol_path: Path, relative: Any) -> Path:
    value = Path(str(relative or ""))
    return value.resolve() if value.is_absolute() else (protocol_path.parent / value).resolve()


def _canonical_action_pair_sha256(pair: Sequence[Mapping[str, Any]]) -> str:
    if len(pair) != 2:
        raise ValueError("Controller action pair must contain exactly two records")
    canonical = [validate_action_record(dict(row), expected_split=str(row["split"])) for row in pair]
    canonical.sort(key=lambda row: int(row["turn_index"]))
    if [row["slot"] for row in canonical] != list(CONTROLLER_SLOTS):
        raise ValueError("Controller action pair must contain q1 then q2_dynamic")
    if len({(row["dataset"], row["qid"]) for row in canonical}) != 1:
        raise ValueError("Controller action pair identity mismatch")
    blob = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol_body_sha256(protocol: Mapping[str, Any]) -> str:
    body = dict(protocol)
    declared = body.pop("protocol_body_canonical_sha256", None)
    actual = _canonical_sha256(body)
    if declared != actual:
        raise ValueError("Controller protocol canonical body hash mismatch")
    return actual


def read_reference_records(path: str | Path) -> list[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Controller evaluation input is missing: {path}")
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            validated = validate_action_record(row)
            example_id = str(validated["example_id"])
            if example_id in seen:
                raise ValueError(f"duplicate Controller example_id: {example_id}")
            seen.add(example_id)
            if validated["gold_boundary"]["gold_final_answer_visible"] is not False:
                raise ValueError(f"QA final Gold is visible for {example_id}")
            if validated["gold_boundary"]["evaluation_gold_access"] is not False:
                raise ValueError(f"evaluation Gold access is enabled for {example_id}")
            rows.append(validated)
    if not rows:
        raise ValueError("Controller evaluation input must be non-empty")
    return rows


def _verify_protocol_bundle(
    protocol_path: Path,
    *,
    expected_protocol_sha256: str,
    verify_current_implementation_locks: bool = True,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    expected = _require_external_sha256(
        expected_protocol_sha256, role="Controller protocol"
    )
    actual = _sha256_file(protocol_path)
    if actual != expected:
        raise ValueError(
            f"Controller protocol external hash mismatch: expected={expected}, actual={actual}"
        )
    protocol = _load_json_object(protocol_path, role="Controller protocol")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
    ):
        raise ValueError("Controller v4.4 protocol identity/status mismatch")
    body_sha256 = _protocol_body_sha256(protocol)
    report_path = protocol_path.parent / "report.json"
    manifest_path = protocol_path.parent / "manifest.json"
    report = _load_json_object(report_path, role="Controller protocol report")
    manifest = _load_json_object(manifest_path, role="Controller protocol manifest")
    if (
        report.get("schema_version") != PROTOCOL_REPORT_SCHEMA_VERSION
        or report.get("status") != PROTOCOL_STATUS
        or report.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
    ):
        raise ValueError("Controller protocol report identity/status mismatch")
    outputs = manifest.get("outputs") or {}
    if (
        manifest.get("schema_version") != PROTOCOL_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != PROTOCOL_STATUS
        or manifest.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
        or manifest.get("training_started") is not False
        or outputs.get("protocol.json") != actual
        or outputs.get("report.json") != _sha256_file(report_path)
    ):
        raise ValueError("Controller protocol manifest/body binding mismatch")
    implementation = (
        _verify_implementation_locks(protocol)
        if verify_current_implementation_locks
        else {
            "status": "HISTORICAL_LOCKS_PRESERVED_FOR_EVAL_SUCCESSOR",
            "verified": dict(protocol.get("implementation_locks") or {}),
        }
    )
    return protocol, {
        "protocol_sha256": actual,
        "protocol_report_sha256": _sha256_file(report_path),
        "protocol_manifest_sha256": _sha256_file(manifest_path),
        "protocol_body_canonical_sha256": body_sha256,
        "implementation_locks": implementation,
    }


def _verify_eval_implementation_locks(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    locks = protocol.get("implementation_locks")
    if not isinstance(locks, Mapping) or set(locks) != EVAL_IMPLEMENTATION_ROLES:
        raise ValueError(
            "eval-e1 implementation locks must contain the exact evaluation roles"
        )
    project_root = Path(__file__).resolve().parents[2]
    verified: Dict[str, Dict[str, str]] = {}
    for role in sorted(EVAL_IMPLEMENTATION_ROLES):
        raw = locks.get(role)
        if not isinstance(raw, Mapping):
            raise ValueError(f"eval-e1 implementation lock is malformed: role={role}")
        raw_path = str(raw.get("path") or "")
        if not raw_path:
            raise ValueError(f"eval-e1 implementation lock has no path: role={role}")
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else project_root / candidate
        path = path.resolve()
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"eval-e1 implementation lock escapes project root: role={role}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"eval-e1 implementation file is missing: role={role}, path={path}"
            )
        expected = _require_external_sha256(
            str(raw.get("sha256") or ""), role=f"eval-e1 implementation:{role}"
        )
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"eval-e1 implementation hash mismatch: role={role}, "
                f"expected={expected}, actual={actual}"
            )
        verified[role] = {"path": relative, "sha256": actual}
    return {"status": "PASS", "verified": verified}


def _verify_eval_successor_bundle(
    eval_protocol_path: Path,
    *,
    expected_eval_protocol_sha256: str,
    parent_protocol_path: Path,
    parent_protocol: Mapping[str, Any],
    parent_bundle: Mapping[str, Any],
    generation_experiment_id: str,
    generation_output_dir: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Authorize current eval code while preserving the completed v4.4 training lineage."""

    expected = _require_external_sha256(
        expected_eval_protocol_sha256, role="Controller eval-e1 protocol"
    )
    actual = _sha256_file(eval_protocol_path)
    if actual != expected:
        raise ValueError(
            "Controller eval-e1 protocol external hash mismatch: "
            f"expected={expected}, actual={actual}"
        )
    protocol = _load_json_object(eval_protocol_path, role="Controller eval-e1 protocol")
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "status",
        "scope",
        "parent_training_lineage",
        "predecessor_eval_failure",
        "change_control",
        "evaluation_contract",
        "implementation_locks",
        "authorized_generation",
        "scientific_boundary",
        "protocol_body_canonical_sha256",
    }
    if set(protocol) != expected_top_level:
        raise ValueError("Controller eval-e1 protocol top-level schema mismatch")
    if (
        protocol.get("schema_version") != EVAL_PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != EVAL_PROTOCOL_STATUS
        or protocol.get("experiment_id") != EVAL_PROTOCOL_EXPERIMENT_ID
    ):
        raise ValueError("Controller eval-e1 protocol identity/status mismatch")
    body_sha256 = _protocol_body_sha256(protocol)
    report_path = eval_protocol_path.parent / "report.json"
    manifest_path = eval_protocol_path.parent / "manifest.json"
    report = _load_json_object(report_path, role="Controller eval-e1 protocol report")
    manifest = _load_json_object(manifest_path, role="Controller eval-e1 protocol manifest")
    outputs = manifest.get("outputs") or {}
    if (
        report.get("schema_version") != EVAL_PROTOCOL_REPORT_SCHEMA_VERSION
        or report.get("status") != EVAL_PROTOCOL_STATUS
        or report.get("experiment_id") != EVAL_PROTOCOL_EXPERIMENT_ID
        or manifest.get("schema_version") != EVAL_PROTOCOL_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != EVAL_PROTOCOL_STATUS
        or manifest.get("experiment_id") != EVAL_PROTOCOL_EXPERIMENT_ID
        or manifest.get("training_started") is not False
        or manifest.get("generation_started") is not False
        or manifest.get("confirmation_opened_or_hashed") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
        or manifest.get("answer_scoring_performed") is not False
        or outputs.get("protocol.json") != actual
        or outputs.get("report.json") != _sha256_file(report_path)
    ):
        raise ValueError("Controller eval-e1 report/manifest binding mismatch")

    parent = protocol.get("parent_training_lineage") or {}
    parent_protocol_lock = parent.get("protocol") or {}
    if (
        _resolve_project_lineage_path(parent_protocol_lock.get("path"))
        != parent_protocol_path.resolve()
        or _resolve_project_lineage_path(parent_protocol_lock.get("report_path"))
        != (parent_protocol_path.parent / "report.json").resolve()
        or _resolve_project_lineage_path(parent_protocol_lock.get("manifest_path"))
        != (parent_protocol_path.parent / "manifest.json").resolve()
    ):
        raise ValueError("Controller eval-e1 parent v4.4 protocol path lineage mismatch")
    if (
        parent_protocol_lock.get("sha256") != parent_bundle.get("protocol_sha256")
        or parent_protocol_lock.get("report_sha256")
        != parent_bundle.get("protocol_report_sha256")
        or parent_protocol_lock.get("manifest_sha256")
        != parent_bundle.get("protocol_manifest_sha256")
        or parent_protocol_lock.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or parent_protocol_lock.get("status") != PROTOCOL_STATUS
        or parent_protocol_lock.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
        or parent_protocol_lock.get("protocol_body_canonical_sha256")
        != parent_protocol.get("protocol_body_canonical_sha256")
        or parent_protocol_lock.get("implementation_locks")
        != parent_protocol.get("implementation_locks")
    ):
        raise ValueError("Controller eval-e1 parent v4.4 protocol lineage mismatch")
    if protocol.get("evaluation_contract") != parent_protocol.get(
        "probe_evaluation_contract"
    ):
        raise ValueError("Controller eval-e1 changed the frozen evaluation contract")

    predecessor = protocol.get("predecessor_eval_failure") or {}
    predecessor_path = _resolve_project_lineage_path(predecessor.get("path"))
    expected_predecessor = {
        "path": EVAL_PREDECESSOR_FAILURE_PATH,
        "sha256": EVAL_PREDECESSOR_FAILURE_SHA256,
        "schema_version": "query-controller-v1-eval-attempt-failure-addendum-1.0",
        "status": "FAIL_PRE_CUDA_NO_GENERATION_NO_PREDICTIONS_RECORDED",
        "attempt_id": (
            "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4-DEV-EVAL-E0"
        ),
        "stage": (
            "CPU asset preflight before CUDA import/query and before output "
            "run-directory creation"
        ),
        "cuda_queried": False,
        "generation_started": False,
        "prediction_rows_written": 0,
    }
    if predecessor != expected_predecessor:
        raise ValueError("Controller eval-e1 predecessor failure lock mismatch")
    if (
        not predecessor_path.is_file()
        or _sha256_file(predecessor_path) != EVAL_PREDECESSOR_FAILURE_SHA256
    ):
        raise ValueError("Controller eval-e1 predecessor failure artifact mismatch")
    predecessor_record = _load_json_object(
        predecessor_path, role="Controller eval-e0 failure addendum"
    )
    if (
        predecessor_record.get("schema_version") != expected_predecessor["schema_version"]
        or predecessor_record.get("status") != expected_predecessor["status"]
        or predecessor_record.get("attempt_id") != expected_predecessor["attempt_id"]
        or (predecessor_record.get("failure") or {}).get("stage")
        != expected_predecessor["stage"]
        or (predecessor_record.get("side_effect_accounting") or {}).get("cuda_queried")
        is not False
        or (predecessor_record.get("side_effect_accounting") or {}).get(
            "generation_started"
        )
        is not False
        or (predecessor_record.get("side_effect_accounting") or {}).get(
            "prediction_rows_written"
        )
        != 0
    ):
        raise ValueError("Controller eval-e0 failure addendum content mismatch")
    if (
        (predecessor_record.get("failure") or {}).get(
            "original_failure_log_or_manifest_available"
        )
        is not False
    ):
        raise ValueError("Controller eval-e0 failure evidence boundary mismatch")

    expected_change_control = {
        "retraining": False,
        "checkpoint_reused": True,
        "data_changed": False,
        "model_changed": False,
        "decoding_changed": False,
        "scorer_changed": False,
        "single_implementation_fix": (
            "forward_verified_parent_protocol_report_and_manifest_sha256_"
            "at_all_formal_evaluation_call_sites"
        ),
        "authorization_wiring_added": True,
    }
    if protocol.get("change_control") != expected_change_control:
        raise ValueError("Controller eval-e1 change-control boundary mismatch")
    parent_contract = parent_protocol.get("probe_evaluation_contract") or {}
    expected_authorized_generation = {
        "experiment_id": EVAL_GENERATION_EXPERIMENT_ID,
        "output_dir": EVAL_GENERATION_OUTPUT_DIR,
        "cohort_role": "dev",
        "exact_actions": 240,
        "decoding": parent_contract.get("decoding"),
        "outcome_metrics_authorized": {"em": False, "f1": False, "ihr": False},
    }
    if (
        generation_experiment_id != EVAL_GENERATION_EXPERIMENT_ID
        or generation_output_dir != EVAL_GENERATION_OUTPUT_DIR
        or protocol.get("authorized_generation") != expected_authorized_generation
    ):
        raise ValueError("Controller eval-e1 generation authorization mismatch")
    implementation = _verify_eval_implementation_locks(protocol)
    return dict(protocol), {
        "status": "PASS",
        "eval_protocol_sha256": actual,
        "eval_protocol_report_sha256": _sha256_file(report_path),
        "eval_protocol_manifest_sha256": _sha256_file(manifest_path),
        "eval_protocol_body_canonical_sha256": body_sha256,
        "implementation_locks": implementation,
    }


def _resolve_project_lineage_path(raw: Any) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate = Path(str(raw or ""))
    return (candidate if candidate.is_absolute() else project_root / candidate).resolve()


def _verify_eval_parent_asset_lineage(
    eval_protocol: Mapping[str, Any],
    *,
    input_path: Path,
    adapter_path: Path,
    training_manifest_path: Path,
    release_report: Mapping[str, Any],
    training_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind e1 to the already-completed v4.4 checkpoint and unchanged dev release."""

    parent = eval_protocol.get("parent_training_lineage") or {}
    release = parent.get("release") or {}
    probe = parent.get("probe") or {}
    release_report_path = input_path.parent / "report.json"
    release_manifest_path = input_path.parent / "manifest.json"
    adapter_config_path = adapter_path / "adapter_config.json"
    throughput_path = training_manifest_path.parent / "throughput.json"
    history_path = training_manifest_path.parent / "training_history.jsonl"
    path_pairs = (
        (release.get("dev_path"), input_path),
        (release.get("report_path"), release_report_path),
        (release.get("manifest_path"), release_manifest_path),
        (probe.get("manifest_path"), training_manifest_path),
        (probe.get("adapter_path"), adapter_path),
        (probe.get("throughput_path"), throughput_path),
        (probe.get("training_history_path"), history_path),
    )
    if any(
        _resolve_project_lineage_path(recorded) != actual.resolve()
        for recorded, actual in path_pairs
    ):
        raise ValueError("Controller eval-e1 parent asset path lineage mismatch")
    if (
        release.get("dev_sha256") != release_report.get("input_sha256")
        or int(release.get("dev_rows", -1)) != 240
        or release.get("report_sha256") != release_report.get("release_report_sha256")
        or release.get("manifest_sha256")
        != release_report.get("release_manifest_sha256")
        or release.get("schema_version") != RELEASE_SCHEMA_VERSION
        or release.get("status") != RELEASE_STATUS
        or release.get("experiment_id") != RELEASE_EXPERIMENT_ID
        or probe.get("manifest_sha256")
        != training_report.get("training_manifest_sha256")
        or probe.get("status") != "COMPLETE"
        or probe.get("experiment_id") != PROBE_EXPERIMENT_ID
        or int(probe.get("global_steps", -1)) != 20
        or probe.get("adapter_sha256") != training_report.get("adapter_sha256")
        or not adapter_config_path.is_file()
        or probe.get("adapter_config_sha256") != _sha256_file(adapter_config_path)
        or not throughput_path.is_file()
        or probe.get("throughput_sha256") != _sha256_file(throughput_path)
        or not history_path.is_file()
        or probe.get("training_history_sha256") != _sha256_file(history_path)
    ):
        raise ValueError("Controller eval-e1 parent completed-asset lineage mismatch")
    return {
        "status": "PASS",
        "checkpoint_reused": True,
        "retraining": False,
        "training_manifest_sha256": str(probe["manifest_sha256"]),
        "adapter_sha256": str(probe["adapter_sha256"]),
        "dev_sha256": str(release["dev_sha256"]),
    }


def _verify_dev_release_and_pairs(
    input_path: Path,
    references: Sequence[Mapping[str, Any]],
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    protocol_report_sha256: str,
    protocol_manifest_sha256: str,
) -> Dict[str, Any]:
    if input_path.name != "dev.jsonl":
        raise ValueError("probe evaluation input must be the exact release dev.jsonl")
    report_path = input_path.parent / "report.json"
    manifest_path = input_path.parent / "manifest.json"
    report = _load_json_object(report_path, role="Controller action release report")
    manifest = _load_json_object(manifest_path, role="Controller action release manifest")
    run = _manifest_run(manifest)
    if (
        report.get("schema_version") != RELEASE_SCHEMA_VERSION
        or report.get("status") != RELEASE_STATUS
        or report.get("experiment_id") != RELEASE_EXPERIMENT_ID
        or report.get("all_release_gates_pass") is not True
    ):
        raise ValueError("Controller v4.4 action release identity/status mismatch")
    if (
        manifest.get("status") != RELEASE_STATUS
        or run.get("schema_version") != RELEASE_SCHEMA_VERSION
        or run.get("status") != RELEASE_STATUS
        or run.get("experiment_id") != RELEASE_EXPERIMENT_ID
        or run.get("outputs") != report.get("outputs")
        or run.get("identity_locks") != report.get("identity_locks")
        or run.get("all_release_gates_pass") is not True
    ):
        raise ValueError("Controller action release manifest/report binding mismatch")
    dev_output = (report.get("outputs") or {}).get("dev.jsonl") or {}
    input_sha256 = _sha256_file(input_path)
    if dev_output.get("sha256") != input_sha256 or dev_output.get("rows") != len(references):
        raise ValueError("Controller action release does not bind exact dev rows/hash")
    protocol_lock = _identity_lock(report, "protocol")
    if protocol_lock.get("sha256") != protocol_sha256:
        raise ValueError("Controller action release protocol lineage mismatch")
    lineage = report.get("protocol_lineage") or {}
    if (
        lineage.get("sha256") != protocol_sha256
        or lineage.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or lineage.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
        or lineage.get("status") != PROTOCOL_STATUS
        or lineage.get("protocol_body_canonical_sha256")
        != protocol.get("protocol_body_canonical_sha256")
        or lineage.get("implementation_locks") != protocol.get("implementation_locks")
        or lineage.get("probe_evaluation_contract")
        != protocol.get("probe_evaluation_contract")
    ):
        raise ValueError("Controller release protocol body/implementation lineage mismatch")

    cohort_locks = ((protocol.get("cohort") or {}).get("cohort_locks") or {})
    dev_lock = cohort_locks.get("dev")
    if not isinstance(dev_lock, Mapping):
        raise ValueError("Controller protocol lacks the frozen dev cohort lock")
    if (lineage.get("cohort_locks") or {}).get("dev") != dev_lock:
        raise ValueError("Controller release protocol dev cohort lineage mismatch")
    identity_path = _resolve_protocol_asset(protocol_path, dev_lock.get("path"))
    if (
        not identity_path.is_file()
        or _sha256_file(identity_path) != dev_lock.get("sha256")
    ):
        raise ValueError("Controller protocol dev identity lock hash mismatch")
    release_dev_lock = _identity_lock(report, "identity_lock_dev")
    if (
        release_dev_lock.get("sha256") != dev_lock.get("sha256")
        or int(release_dev_lock.get("rows", -1)) != int(dev_lock.get("rows", -2))
    ):
        raise ValueError("Controller release dev identity lock differs from protocol")

    identity_rows: list[Dict[str, Any]] = []
    with identity_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid dev identity row at line {line_number}")
            identity_rows.append(value)
    if len(identity_rows) != int(dev_lock.get("rows", -1)):
        raise ValueError("Controller protocol dev identity row count mismatch")
    expected_by_key = {
        (str(row["dataset"]), str(row["qid"])): row for row in identity_rows
    }
    if len(expected_by_key) != len(identity_rows):
        raise ValueError("duplicate dataset::qid in Controller dev identity lock")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in references:
        grouped[(str(row["dataset"]), str(row["qid"]))].append(row)
    if set(grouped) != set(expected_by_key):
        raise ValueError("Controller release dev qids differ from frozen protocol identities")
    for key, pair in grouped.items():
        lock = expected_by_key[key]
        q1 = min(pair, key=lambda row: int(row["turn_index"]))
        for field in (
            "dataset", "qid", "question_key", "question_sha256", "family_sha256", "split"
        ):
            if q1.get(field) != lock.get(field):
                raise ValueError(f"Controller dev identity field mismatch: {key}, field={field}")
        if _canonical_action_pair_sha256(pair) != lock.get("action_pair_sha256"):
            raise ValueError(f"Controller dev action-pair hash mismatch: {key}")
    return {
        "status": "PASS",
        "input_sha256": input_sha256,
        "release_report_sha256": _sha256_file(report_path),
        "release_manifest_sha256": _sha256_file(manifest_path),
        "identity_lock_sha256": str(dev_lock.get("sha256")),
        "identity_qids": len(identity_rows),
        "action_pair_hash_match_rate": 1.0,
    }


def _runtime_config_from_training_manifest(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "phase": "probe",
        "experiment_id": config.get("experiment_id"),
        "model": {
            key: config.get(key)
            for key in (
                "base_model", "method", "initialization", "init_adapter_path",
                "load_in_4bit", "dtype", "lora_r", "lora_alpha", "lora_dropout",
                "target_modules",
            )
        },
        "data": {
            "allowed_source_actions": config.get("allowed_source_actions"),
            "train_quotas": config.get("train_quotas"),
            "dev_quotas": config.get("dev_quotas"),
        },
        "training": {
            key: config.get(source)
            for key, source in (
                ("seed", "seed"),
                ("max_seq_length", "max_seq_length"),
                ("per_device_train_batch_size", "batch_size"),
                ("per_device_eval_batch_size", "eval_batch_size"),
                ("gradient_accumulation_steps", "grad_accum"),
                ("learning_rate", "learning_rate"),
                ("warmup_ratio", "warmup_ratio"),
                ("lr_scheduler_type", "lr_scheduler_type"),
                ("num_train_epochs", "epochs"),
                ("max_steps", "max_steps"),
                ("weight_decay", "weight_decay"),
                ("max_grad_norm", "max_grad_norm"),
                ("logging_steps", "logging_steps"),
                ("eval_strategy", "eval_strategy"),
                ("eval_steps", "eval_steps"),
                ("save_strategy", "save_strategy"),
                ("save_steps", "save_steps"),
                ("save_total_limit", "save_total_limit"),
                ("verify_saved_adapter_reload", "verify_saved_adapter_reload"),
            )
        },
    }


def _verify_probe_training_and_adapter(
    training_manifest_path: Path,
    adapter_path: Path,
    *,
    expected_training_manifest_sha256: str,
    expected_adapter_sha256: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    protocol_report_sha256: str,
    protocol_manifest_sha256: str,
    release_report_sha256: str,
    release_manifest_sha256: str,
    input_sha256: str,
) -> Dict[str, Any]:
    expected_manifest = _require_external_sha256(
        expected_training_manifest_sha256, role="probe training manifest"
    )
    actual_manifest = _sha256_file(training_manifest_path)
    if actual_manifest != expected_manifest:
        raise ValueError("probe training manifest external hash mismatch")
    expected_adapter = _require_external_sha256(
        expected_adapter_sha256, role="Controller adapter"
    )
    weights = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if not weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError(f"Controller adapter is incomplete: {adapter_path}")
    actual_adapter = _sha256_file(weights)
    if actual_adapter != expected_adapter:
        raise ValueError("Controller adapter external hash mismatch")

    manifest = _load_json_object(training_manifest_path, role="probe training manifest")
    run = _manifest_run(manifest)
    contract = protocol.get("probe_evaluation_contract") or {}
    required = contract.get("required_probe_artifacts") or {}
    throughput = run.get("throughput") or {}
    config = run.get("config") or {}
    log_gates = throughput.get("log_gates") or {}
    saved_adapter = throughput.get("saved_adapter") or {}
    dtype_inventories = [
        saved_adapter.get("saved_dtype_inventory"),
        saved_adapter.get("live_dtype_inventory"),
        saved_adapter.get("clean_reload_dtype_inventory"),
    ]
    if (
        manifest.get("status") != required.get("manifest_status")
        or run.get("phase") != "query_controller_sft_probe"
        or run.get("experiment_id") != required.get("experiment_id")
        or int(throughput.get("global_steps", -1))
        != int(required.get("completed_optimizer_steps", -2))
        or throughput.get("exact_optimizer_steps_pass") is not True
        or (throughput.get("optimizer_step_gate") or {}).get("exact_optimizer_steps_pass")
        is not True
        or saved_adapter.get("tensor_reload_exact") is not True
        or saved_adapter.get("save_fidelity_exact") is not True
        or saved_adapter.get("clean_reload_single_adapter") is not True
        or saved_adapter.get("clean_reload_tensor_exact") is not True
        or saved_adapter.get("dtype_inventories_recorded") is not True
        or saved_adapter.get("saved_live_clean_reload_dtype_inventories_equal")
        is not True
        or not all(isinstance(value, Mapping) and bool(value) for value in dtype_inventories)
        or not (dtype_inventories[0] == dtype_inventories[1] == dtype_inventories[2])
        or log_gates.get("finite_train_loss") is not True
        or log_gates.get("finite_logged_losses") is not True
        or log_gates.get("finite_gradient_norms") is not True
        or log_gates.get("nonzero_trainable_gradient_observed") is not True
        or int(throughput.get("trainable_parameters", 0)) <= 0
    ):
        raise ValueError("probe training manifest completion/gate mismatch")
    frozen_runtime = (protocol.get("training_contract") or {}).get("runtime_config")
    if not isinstance(frozen_runtime, Mapping) or _runtime_config_from_training_manifest(config) != dict(frozen_runtime):
        raise ValueError("probe training manifest runtime config differs from protocol")
    if (
        config.get("expected_protocol_sha256") != protocol_sha256
        or config.get("expected_protocol_report_sha256") != protocol_report_sha256
        or config.get("expected_protocol_manifest_sha256") != protocol_manifest_sha256
        or config.get("expected_dev_sha256") != input_sha256
        or config.get("expected_release_report_sha256") != release_report_sha256
        or config.get("expected_release_manifest_sha256") != release_manifest_sha256
    ):
        raise ValueError("probe training manifest asset-lock lineage mismatch")
    data_asset_lock = ((run.get("data_report") or {}).get("asset_lock") or {})
    if (
        data_asset_lock.get("status") != "PASS"
        or (data_asset_lock.get("hashes") or {}).get("protocol") != protocol_sha256
        or (data_asset_lock.get("hashes") or {}).get("protocol_report")
        != protocol_report_sha256
        or (data_asset_lock.get("hashes") or {}).get("protocol_manifest")
        != protocol_manifest_sha256
        or (data_asset_lock.get("hashes") or {}).get("dev") != input_sha256
    ):
        raise ValueError("probe data report asset-lock lineage mismatch")
    recorded_final = (run.get("output_artifacts") or {}).get("final") or {}
    current_final = artifact_identity(adapter_path)
    if (
        recorded_final.get("inventory_sha256") != current_final.get("inventory_sha256")
        or recorded_final.get("files") != current_final.get("files")
    ):
        raise ValueError("Controller adapter directory differs from training manifest")
    return {
        "status": "PASS",
        "training_manifest_sha256": actual_manifest,
        "adapter_sha256": actual_adapter,
        "experiment_id": str(run.get("experiment_id")),
        "optimizer_steps": int(throughput["global_steps"]),
        "adapter_dtype_inventory": dict(dtype_inventories[0]),
        "adapter_dtype_inventories_equal": True,
        "asset_lock_lineage_match": True,
    }


def verify_probe_evaluation_assets(
    *,
    input_path: str | Path,
    adapter_path: str | Path,
    protocol_path: str | Path,
    eval_protocol_path: str | Path,
    training_manifest_path: str | Path,
    expected_protocol_sha256: str,
    expected_eval_protocol_sha256: str,
    expected_training_manifest_sha256: str,
    expected_adapter_sha256: str,
    generation_experiment_id: str,
    generation_output_dir: str,
    cohort_role: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
    base_model: str,
    dtype: str,
    load_in_4bit: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Fail closed on all frozen assets before CUDA is queried."""

    if cohort_role != "dev":
        raise ValueError("v4.4 probe evaluation authorizes teacher-forced dev only")
    input_path = Path(input_path)
    adapter_path = Path(adapter_path)
    protocol_path = Path(protocol_path)
    eval_protocol_path = Path(eval_protocol_path)
    training_manifest_path = Path(training_manifest_path)
    protocol, protocol_report = _verify_protocol_bundle(
        protocol_path,
        expected_protocol_sha256=expected_protocol_sha256,
        verify_current_implementation_locks=False,
    )
    eval_protocol, eval_protocol_report = _verify_eval_successor_bundle(
        eval_protocol_path,
        expected_eval_protocol_sha256=expected_eval_protocol_sha256,
        parent_protocol_path=protocol_path,
        parent_protocol=protocol,
        parent_bundle=protocol_report,
        generation_experiment_id=generation_experiment_id,
        generation_output_dir=generation_output_dir,
    )
    contract = protocol.get("probe_evaluation_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Controller protocol lacks probe_evaluation_contract")
    if (
        contract.get("authorization") != "post_probe_dev_teacher_forced_only"
        or contract.get("cohort_role") != "dev"
        or contract.get("input_split") != "dev"
        or contract.get("datasets") != list(ENABLED_DATASETS)
        or contract.get("slots") != list(CONTROLLER_SLOTS)
        or contract.get("confirmation_access") is not False
        or contract.get("prospective_access") is not False
        or contract.get("exact_qids_per_enabled_dataset") != 60
        or contract.get("exact_action_rows_per_enabled_dataset") != 120
        or contract.get("exact_actions") != 240
        or contract.get("state_source") != "annotation_derived_but_passage_bound"
        or contract.get("runtime_reader_predicted") is not False
        or contract.get("outcome_metrics_authorized")
        != {"em": False, "f1": False, "ihr": False}
    ):
        raise ValueError("Controller probe evaluation contract identity/scope drifted")
    required = contract.get("required_probe_artifacts") or {}
    if (
        required.get("experiment_id") != PROBE_EXPERIMENT_ID
        or required.get("manifest_status") != "COMPLETE"
        or required.get("completed_optimizer_steps") != 20
        or required.get("training_manifest_sha256") != "required_external_eval_lock"
        or required.get("adapter_sha256") != "required_external_eval_lock"
        or required.get("asset_lock_lineage_match") is not True
    ):
        raise ValueError("Controller probe evaluation artifact contract drifted")
    expected_decoding = {
        "strategy": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "seed": seed,
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "base_model": base_model,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
    }
    if contract.get("decoding") != expected_decoding:
        raise ValueError(
            "Controller generation settings differ from frozen probe evaluation decoding"
        )
    semantics = contract.get("status_semantics") or {}
    if (
        semantics.get("generation_complete_status")
        != "COMPLETE_GENERATION_NOT_MECHANISM_PASS"
        or semantics.get("mechanism_pass_requires_separate_scorer_gate") is not True
        or semantics.get("generation_complete_implies_mechanism_pass") is not False
    ):
        raise ValueError("Controller generation/mechanism status semantics drifted")

    references = read_reference_records(input_path)
    if {str(row["split"]) for row in references} != {"dev"}:
        raise ValueError("Controller probe evaluation input contains a non-dev record")
    counts = Counter((str(row["dataset"]), str(row["slot"])) for row in references)
    expected_counts = {
        (dataset, slot): 60 for dataset in ENABLED_DATASETS for slot in CONTROLLER_SLOTS
    }
    if len(references) != 240 or counts != Counter(expected_counts):
        raise ValueError(
            "Controller probe evaluation cohort cardinality mismatch: "
            f"n={len(references)}, counts={dict(counts)}"
        )
    release_report = _verify_dev_release_and_pairs(
        input_path,
        references,
        protocol_path=protocol_path,
        protocol=protocol,
        protocol_sha256=protocol_report["protocol_sha256"],
        protocol_report_sha256=protocol_report["protocol_report_sha256"],
        protocol_manifest_sha256=protocol_report["protocol_manifest_sha256"],
    )
    training_report = _verify_probe_training_and_adapter(
        training_manifest_path,
        adapter_path,
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        expected_adapter_sha256=expected_adapter_sha256,
        protocol=protocol,
        protocol_sha256=protocol_report["protocol_sha256"],
        protocol_report_sha256=protocol_report["protocol_report_sha256"],
        protocol_manifest_sha256=protocol_report["protocol_manifest_sha256"],
        release_report_sha256=release_report["release_report_sha256"],
        release_manifest_sha256=release_report["release_manifest_sha256"],
        input_sha256=release_report["input_sha256"],
    )
    parent_asset_lineage = _verify_eval_parent_asset_lineage(
        eval_protocol,
        input_path=input_path,
        adapter_path=adapter_path,
        training_manifest_path=training_manifest_path,
        release_report=release_report,
        training_report=training_report,
    )
    return references, dict(protocol), {
        "status": "PASS",
        "protocol": protocol_report,
        "eval_protocol": eval_protocol_report,
        "release": release_report,
        "probe": training_report,
        "parent_asset_lineage": parent_asset_lineage,
        "cohort_role": "dev",
        "exact_actions": 240,
        "generation": expected_decoding,
    }


def score_response(record: Mapping[str, Any], response_text: str) -> Dict[str, Any]:
    """Score only syntax/state mechanics; never score the downstream QA answer."""

    parsed: Mapping[str, Any] | None = None
    candidate = dict(record)
    try:
        parsed = parse_target_response(response_text, reference_record=record)
        candidate["target"] = parsed
    except Exception:
        # Replace the frozen target before auditing, otherwise an unparsable
        # response would accidentally inherit the reference action's metrics.
        candidate["target"] = {}
    audit = audit_action_record(candidate, expected_split=str(record["split"]))
    reference_target = dict(record["target"])
    return {
        "valid": bool(audit["valid"]),
        "checks": dict(audit["checks"]),
        "error_codes": list(audit["errors"]),
        "parsed_target": dict(parsed) if parsed is not None else None,
        "structured_target_exact": parsed == reference_target,
        "canonical_text_exact": response_text == _canonical_target(reference_target),
    }


def build_prediction_record(
    reference: Mapping[str, Any],
    response_text: str,
    *,
    prompt_tokens: int,
    generated_tokens: int,
) -> Dict[str, Any]:
    score = score_response(reference, response_text)
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "example_id": reference["example_id"],
        "dataset": reference["dataset"],
        "qid": reference["qid"],
        "question_key": reference["question_key"],
        "question_sha256": reference["question_sha256"],
        "family_sha256": reference["family_sha256"],
        "split": reference["split"],
        "slot": reference["slot"],
        "input_record_sha256": _canonical_sha256(reference),
        "response_text": response_text,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_tokens),
        **score,
    }


def aggregate_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    check_names = (
        "schema_valid",
        "query_contract_valid",
        "query_nonrepeat",
        "placeholder_free",
        "dependency_closed",
        "source_action_valid",
        "state_use_valid",
        "gold_boundary_valid",
    )

    def summarise(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        return {
            "n": n,
            "valid_rate": sum(bool(row["valid"]) for row in items) / n if n else 0.0,
            "structured_target_exact_rate": (
                sum(bool(row["structured_target_exact"]) for row in items) / n if n else 0.0
            ),
            "canonical_text_exact_rate": (
                sum(bool(row["canonical_text_exact"]) for row in items) / n if n else 0.0
            ),
            **{
                f"{name}_rate": (
                    sum(bool(row.get("checks", {}).get(name)) for row in items) / n
                    if n else 0.0
                )
                for name in check_names
            },
        }

    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_slot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dataset_slot: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    error_counts: Counter[str] = Counter()
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
        by_slot[str(row["slot"])].append(row)
        by_dataset_slot[(str(row["dataset"]), str(row["slot"]))].append(row)
        error_counts.update(str(value) for value in row.get("error_codes", []))
    return {
        "schema_version": "query-controller-greedy-mechanism-report-v1",
        "scope": "controller_action_only_no_retrieval_no_qa_gold_no_em",
        "identity_join_rate": 1.0,
        "overall": summarise(rows),
        "by_dataset": {key: summarise(value) for key, value in sorted(by_dataset.items())},
        "by_slot": {key: summarise(value) for key, value in sorted(by_slot.items())},
        "by_dataset_slot": {
            f"{dataset}::{slot}": summarise(value)
            for (dataset, slot), value in sorted(by_dataset_slot.items())
        },
        "error_counts": dict(sorted(error_counts.items())),
    }


def evaluate_teacher_forced_mechanism_gate(
    report: Mapping[str, Any], protocol: Mapping[str, Any], *, cohort_role: str
) -> Dict[str, Any]:
    """Evaluate a versioned action-only gate separately from generation status.

    This gate is intentionally narrower than the protocol's future online
    runtime gate.  In particular, q2 observations in these release records are
    annotation-derived but passage-bound; they are not Reader predictions.
    """

    if cohort_role != "dev":
        raise ValueError("v4.4 teacher-forced mechanism gate authorizes dev only")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("experiment_id") != PROTOCOL_EXPERIMENT_ID
    ):
        raise ValueError("unrecognized or unfrozen Controller v4.4 protocol")
    contract = protocol.get("probe_evaluation_contract") or {}
    if (
        contract.get("authorization") != "post_probe_dev_teacher_forced_only"
        or contract.get("cohort_role") != "dev"
        or contract.get("input_split") != "dev"
        or contract.get("datasets") != list(ENABLED_DATASETS)
        or contract.get("slots") != list(CONTROLLER_SLOTS)
        or contract.get("exact_actions") != 240
        or contract.get("exact_action_rows_per_enabled_dataset") != 120
        or contract.get("exact_qids_per_enabled_dataset") != 60
        or contract.get("state_source") != "annotation_derived_but_passage_bound"
        or contract.get("runtime_reader_predicted") is not False
    ):
        raise ValueError("Controller probe evaluation scope/cardinality contract drifted")
    required_thresholds = contract.get("mechanism_gates") or {}
    if required_thresholds.get("identity_join_rate") != 1.0:
        raise ValueError("protocol lacks exact identity-join gate")
    overall = report.get("overall") or {}
    groups = report.get("by_dataset_slot") or {}
    if overall.get("n") != 240:
        raise ValueError("teacher-forced mechanism report must contain exactly 240 actions")
    expected_group_keys = {
        f"{dataset}::{slot}" for dataset in ENABLED_DATASETS for slot in CONTROLLER_SLOTS
    }
    if set(groups) != expected_group_keys or any(
        (groups.get(key) or {}).get("n") != 60 for key in expected_group_keys
    ):
        raise ValueError("teacher-forced mechanism report dataset/slot cardinality mismatch")
    checks: dict[str, Dict[str, Any]] = {}
    identity_actual = report.get("identity_join_rate")
    checks["overall:identity_join_rate"] = {
        "actual": identity_actual,
        "operator": "==",
        "threshold": 1.0,
        "passed": identity_actual == 1.0,
    }
    for dataset in ENABLED_DATASETS:
        for slot in CONTROLLER_SLOTS:
            key = f"{dataset}::{slot}"
            group = groups.get(key) or {}
            for metric, threshold_group in (
                ("schema_valid_rate", "schema_valid_rate_min_each_dataset_slot"),
                ("dependency_closed_rate", "dependency_closed_rate_min_each_dataset_slot"),
                ("state_use_valid_rate", "state_use_valid_rate_min_each_dataset_slot"),
            ):
                threshold_map = required_thresholds.get(threshold_group) or {}
                threshold = threshold_map.get(slot)
                actual = group.get(metric)
                checks[f"{key}:{metric}"] = {
                    "actual": actual,
                    "operator": ">=",
                    "threshold": threshold,
                    "passed": (
                        isinstance(actual, (int, float))
                        and isinstance(threshold, (int, float))
                        and actual >= threshold
                    ),
                }
    for metric in ("query_nonrepeat_rate", "placeholder_free_rate"):
        actual = overall.get(metric)
        threshold = required_thresholds.get(metric)
        checks[f"overall:{metric}"] = {
            "actual": actual,
            "operator": ">=",
            "threshold": threshold,
            "passed": (
                isinstance(actual, (int, float))
                and isinstance(threshold, (int, float))
                and actual >= threshold
            ),
        }
    passed = bool(checks) and all(row["passed"] for row in checks.values())
    return {
        "schema_version": "query-controller-teacher-forced-mechanism-gate-v1",
        "status": "PASS_TEACHER_FORCED_ACTION_MECHANICS" if passed else "FAIL_STOP_ACTION_MECHANICS",
        "passed": passed,
        "cohort_role": cohort_role,
        "exact_actions": 240,
        "state_source": "annotation_derived_but_passage_bound",
        "runtime_reader_predicted": False,
        "scientific_boundary": (
            "Action syntax/state-use diagnostic only. This is not the online Reader-predicted "
            "dynamic runtime gate and does not measure retrieval utility, QA EM/F1, or IHR."
        ),
        "checks": checks,
        "excluded_online_runtime_gates": [
            "a1_admissible_rate",
            "dynamic_transition_rate",
            "cache_and_call_accounting_rate",
            "fallback_byte_identity_rate",
            "final_passage_budget_and_unique_rate",
            "reader_one_shot_regression_identity_rate",
        ],
    }


def run_greedy_controller(
    *,
    input_path: str,
    adapter_path: str,
    protocol_path: str,
    eval_protocol_path: str,
    training_manifest_path: str,
    expected_protocol_sha256: str,
    expected_eval_protocol_sha256: str,
    expected_training_manifest_sha256: str,
    expected_adapter_sha256: str,
    cohort_role: str,
    output_dir: str,
    experiment_id: str,
    base_model: str = "llama3-8B-instruct",
    batch_size: int = 4,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 192,
    seed: int = 42,
    dtype: str = "bf16",
    load_in_4bit: bool = True,
) -> Dict[str, Any]:
    if batch_size <= 0 or max_input_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("batch size and token limits must be positive")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    # Resolve and hash every protocol/release/cohort/training/adapter asset
    # before importing the GPU stack or querying CUDA state.
    references, protocol, evaluation_preflight = verify_probe_evaluation_assets(
        input_path=input_path,
        adapter_path=adapter_path,
        protocol_path=protocol_path,
        eval_protocol_path=eval_protocol_path,
        training_manifest_path=training_manifest_path,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_eval_protocol_sha256=expected_eval_protocol_sha256,
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        expected_adapter_sha256=expected_adapter_sha256,
        generation_experiment_id=experiment_id,
        generation_output_dir=output_dir,
        cohort_role=cohort_role,
        batch_size=batch_size,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
        base_model=base_model,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Controller greedy generation requires a CUDA-visible process")
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("active GPU does not support bf16")
    adapter = Path(adapter_path)

    base_id = model_path(base_model)
    out_dir, frozen_id = prepare_new_run_dir(
        output_dir,
        experiment_id=experiment_id,
        extra={
            "phase": "query_controller_greedy_mechanism_eval",
            "scope": "no_retrieval_no_qa_gold_no_em",
            "input_artifacts": {
                "input": artifact_identity(input_path),
                "adapter": artifact_identity(adapter_path),
                "base_model": artifact_identity(base_id),
                "protocol": artifact_identity(protocol_path),
                "eval_protocol": artifact_identity(eval_protocol_path),
                "training_manifest": artifact_identity(training_manifest_path),
            },
            "evaluation_preflight": evaluation_preflight,
            "generation": {
                "decode": "greedy",
                "batch_size": batch_size,
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "dtype": dtype,
                "load_in_4bit": load_in_4bit,
            },
        },
    )
    prediction_path = out_dir / "predictions.jsonl"
    try:
        set_seed(seed)
        torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch_dtype,
            quantization_config=quantization,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        model.eval()

        predictions: list[Dict[str, Any]] = []
        started = time.perf_counter()
        with prediction_path.open("w", encoding="utf-8") as sink, torch.inference_mode():
            for start in range(0, len(references), batch_size):
                chunk = references[start : start + batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        controller_messages(row, include_target=False),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in chunk
                ]
                encoded = tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                prompt_width = int(encoded["input_ids"].shape[1])
                actual_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                if max(actual_lengths) > max_input_tokens:
                    offender = chunk[actual_lengths.index(max(actual_lengths))]["example_id"]
                    raise ValueError(
                        f"Controller eval prompt exceeds max_input_tokens={max_input_tokens}: "
                        f"{offender} has {max(actual_lengths)}; truncation is forbidden"
                    )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                continuations = generated[:, prompt_width:]
                for index, reference in enumerate(chunk):
                    token_ids = continuations[index].tolist()
                    # Preserve the tokenizer's exact decoded text.  Parsing may
                    # tolerate surrounding whitespace, but response_text is not
                    # post-processed or silently repaired.
                    response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    generated_tokens = len(token_ids)
                    while generated_tokens and token_ids[generated_tokens - 1] == tokenizer.pad_token_id:
                        generated_tokens -= 1
                    row = build_prediction_record(
                        reference,
                        response_text,
                        prompt_tokens=int(actual_lengths[index]),
                        generated_tokens=generated_tokens,
                    )
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    predictions.append(row)
        elapsed = time.perf_counter() - started
        report = aggregate_predictions(predictions)
        report["generation_status"] = "COMPLETE_GENERATION_NOT_MECHANISM_PASS"
        report["mechanism_gate"] = {
            "status": "NOT_EVALUATED_REQUIRES_SEPARATE_SCORER",
            "passed": None,
            "scientific_boundary": (
                "Generation completion does not imply a mechanism pass. Run the "
                "versioned CPU scorer over the immutable full prediction rows."
            ),
        }
        report["elapsed_seconds"] = elapsed
        report["seconds_per_action"] = elapsed / len(predictions)
        report_path = out_dir / "mechanism_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dump_manifest(
            out_dir,
            status="COMPLETE_GENERATION_NOT_MECHANISM_PASS",
            extra={
                "phase": "query_controller_greedy_mechanism_eval",
                "experiment_id": frozen_id,
                "scope": "no_retrieval_no_qa_gold_no_em",
                "input_artifacts": {
                    "input": artifact_identity(input_path),
                    "adapter": artifact_identity(adapter_path),
                    "base_model": artifact_identity(base_id),
                    "protocol": artifact_identity(protocol_path),
                    "eval_protocol": artifact_identity(eval_protocol_path),
                    "training_manifest": artifact_identity(training_manifest_path),
                },
                "evaluation_preflight": evaluation_preflight,
                "output_artifacts": {
                    "predictions": artifact_identity(prediction_path),
                    "mechanism_report": artifact_identity(report_path),
                },
                "report": report,
            },
        )
        return {"output_dir": str(out_dir), "predictions": str(prediction_path), **report}
    except Exception as exc:
        dump_manifest(
            out_dir,
            status="FAIL_STOP",
            extra={
                "phase": "query_controller_greedy_mechanism_eval",
                "experiment_id": frozen_id,
                "scope": "no_retrieval_no_qa_gold_no_em",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "partial_predictions": artifact_identity(prediction_path),
            },
        )
        raise


__all__ = [
    "PREDICTION_SCHEMA_VERSION",
    "aggregate_predictions",
    "build_prediction_record",
    "evaluate_teacher_forced_mechanism_gate",
    "read_reference_records",
    "run_greedy_controller",
    "score_response",
]
