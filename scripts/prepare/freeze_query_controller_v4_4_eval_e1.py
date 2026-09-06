#!/usr/bin/env python
"""Freeze the eval-only v4.4-e1 Query Controller successor.

This protocol does not create or alter training data, retrain the Controller,
or open confirmation/prospective examples.  It authorizes one deterministic
teacher-forced dev generation using the already completed v4.4 checkpoint.
The only implementation correction is forwarding the already verified parent
protocol report/manifest hashes at every formal evaluation call site.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "query-controller-v1-eval-hotfix-protocol-1.0"
REPORT_SCHEMA_VERSION = "query-controller-v1-eval-hotfix-freeze-report-1.0"
MANIFEST_SCHEMA_VERSION = "query-controller-v1-eval-hotfix-manifest-1.0"
EXPERIMENT_ID = "QUERY-CONTROLLER-V1-V4-4-DEV-EVAL-E1-PROTOCOL"
STATUS = "FROZEN_EVAL_ONLY_PARENT_V4_4_COMPLETE_HASH_FORWARDING_HOTFIX_NOT_RUN"
GENERATION_EXPERIMENT_ID = (
    "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4-DEV-EVAL-E1"
)
GENERATION_OUTPUT_DIR = (
    "outputs/validation/query_controller_action_v1_probe20_seed42_v4_4_dev_eval_e1"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/query_controller_v4_4_dev_eval_e1_protocol"
)

PARENT_PROTOCOL = Path(
    "outputs/audits/query_controller_v1_exact_text_pilot_seed42_protocol_v4_4/"
    "protocol.json"
)
PARENT_PROTOCOL_REPORT = PARENT_PROTOCOL.with_name("report.json")
PARENT_PROTOCOL_MANIFEST = PARENT_PROTOCOL.with_name("manifest.json")
PARENT_RELEASE_DIR = Path("data/silver_data/query_controller_actions_v1_seed42_v4_4")
PARENT_TRAINING_DIR = Path(
    "outputs/probes/query_controller_action_v1_probe20_seed42_v4_4"
)
PREDECESSOR_FAILURE = Path(
    "outputs/audits/query_controller_v4_4_dev_eval_e0_pre_cuda_failure/"
    "metadata_addendum.json"
)

EXPECTED = {
    "parent_protocol": "be9eb2cf1fc00b6ca61fb0f4af4edc6075ef3f3e0aab916555eae9c9e263d55b",
    "parent_protocol_report": "bda7b0e4c033a98cb7e3a37820aa587750f0ea185fde603af7f4776571188743",
    "parent_protocol_manifest": "91c6df5711654129688bfbe39ef965e2ad1fe7f77b80cbded445fde5fb2fb9e1",
    "release_train_declared": "2ded8b3a8bd9fa42eba657307e8cdf816297d58542f2abba0c8014b9b93789ca",
    "release_dev": "86b405599d845b2b35a5e1e3e9557fce9bd19a85530db7ebc67d96050e59ad19",
    "release_confirmation_declared": "87336a5ff0efe5ac90b51ea616a61cd1734dc22dfc8eb93a3a3d3fd06dcb796c",
    "release_report": "eb300ed783cdac546ac6988d223adbc489f7e0c7d429e073e6c01535d73d568d",
    "release_manifest": "a5cb632f353adad6be12481bc125658f23afc7743efde93a3869b593c98975ca",
    "training_manifest": "23d8d2df11e923d24e3f9326374a475d53da9c5e2363e663c1aae8d8848b1dad",
    "adapter": "b3bae36afd770eba4e2d144ed6d07e6ae07c3bc73d09e526294686dcd669f88e",
    "adapter_config": "e3b28fd44ce371dacf1a8091873afc07151aeed6a2c308627dde25f6679524f1",
    "throughput": "5a56a5e58dfef70344b149fa51bb9be7be3179333db716bfb72e30cd9a4040cc",
    "training_history": "67b678b93f4c81c9f97ce8c434b575fb79698450529ea37df110f86d7c94d3dd",
    "data_report": "029e966b34cd94908a1bd0d6c4edc381c4caf6020320d852bfcd5da8302bf3bc",
    "training_log": "628a92a21e2f6e291ca4836680c851f5f9b5f8a9093736feb1eb539bf7190573",
    "predecessor_failure": "51f78375cff772f4dcdc166a712660a29c0d9a1bfb2ff380ea37809aa3a0fa34",
}

PARENT_PROTOCOL_ID = "QUERY-CONTROLLER-V1-EXACT-TEXT-PILOT-SEED42-PROTOCOL-V4_4"
PARENT_PROTOCOL_SCHEMA = "query-controller-v1-pilot-protocol-4.4"
PARENT_PROTOCOL_REPORT_SCHEMA = "query-controller-v1-pilot-freeze-report-4.4"
PARENT_PROTOCOL_MANIFEST_SCHEMA = "query-controller-v1-pilot-manifest-4.4"
PARENT_PROTOCOL_STATUS = (
    "FROZEN_V4_4_PEFT_DEFAULT_FP32_CLEAN_RELOAD_"
    "SAME_IDENTITIES_2DATASET_ACTIONS_NOT_TRAINED"
)
PARENT_RELEASE_ID = "QUERY-CONTROLLER-ACTION-V1-SEED42-V4-4"
PARENT_RELEASE_SCHEMA = "query-controller-action-release-v4.4-pair-locked-same-identities"
PARENT_RELEASE_STATUS = "COMPLETE_V4_4_PAIR_LOCKED_SAME_IDENTITIES_NOT_TRAINED"
PARENT_PROBE_ID = "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4"
PARENT_PROTOCOL_BODY_SHA256 = (
    "c5dc9f5a7e793952120535c6508bdee7d7573889bc40dbbfd676c80445ff12d5"
)

IMPLEMENTATION_PATHS = {
    "eval_protocol_freezer": Path(
        "scripts/prepare/freeze_query_controller_v4_4_eval_e1.py"
    ),
    "controller_greedy_runner": Path("kgproweight/eval/query_controller_runner.py"),
    "controller_generate_cli": Path(
        "scripts/eval/generate_query_controller_actions.py"
    ),
    "controller_mechanism_scorer": Path(
        "scripts/eval/evaluate_query_controller_actions.py"
    ),
}

_FORBIDDEN_TO_OPEN = frozenset(
    {"confirmation.jsonl", "confirmation.identity_only.jsonl", "prospective.identity_only.jsonl"}
)


def _sha256_file(path: Path) -> str:
    if path.name in _FORBIDDEN_TO_OPEN:
        raise PermissionError(f"eval-e1 must not open held-out asset: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_hash(path: Path, expected: str, *, role: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} SHA256 drift: expected={expected}, actual={actual}")
    return actual


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _protocol_body_hash(protocol: Mapping[str, Any]) -> str:
    body = dict(protocol)
    declared = body.pop("protocol_body_canonical_sha256", None)
    actual = _canonical_sha256(body)
    if declared != actual:
        raise ValueError("parent protocol canonical body hash mismatch")
    return actual


def _manifest_run(document: Mapping[str, Any]) -> Mapping[str, Any]:
    run = document.get("run")
    return run if isinstance(run, Mapping) else document


def _identity_lock(document: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    locks = document.get("identity_locks")
    if not isinstance(locks, list):
        raise ValueError("release identity_locks must be a list")
    matches = [row for row in locks if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one release identity lock: {role}")
    return matches[0]


def _validate_parent_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_sha = _require_hash(
        PROJECT_ROOT / PARENT_PROTOCOL, EXPECTED["parent_protocol"], role="parent protocol"
    )
    report_sha = _require_hash(
        PROJECT_ROOT / PARENT_PROTOCOL_REPORT,
        EXPECTED["parent_protocol_report"],
        role="parent protocol report",
    )
    manifest_sha = _require_hash(
        PROJECT_ROOT / PARENT_PROTOCOL_MANIFEST,
        EXPECTED["parent_protocol_manifest"],
        role="parent protocol manifest",
    )
    protocol = _load_object(PROJECT_ROOT / PARENT_PROTOCOL)
    report = _load_object(PROJECT_ROOT / PARENT_PROTOCOL_REPORT)
    manifest = _load_object(PROJECT_ROOT / PARENT_PROTOCOL_MANIFEST)
    if (
        protocol.get("schema_version") != PARENT_PROTOCOL_SCHEMA
        or protocol.get("status") != PARENT_PROTOCOL_STATUS
        or protocol.get("experiment_id") != PARENT_PROTOCOL_ID
        or _protocol_body_hash(protocol) != PARENT_PROTOCOL_BODY_SHA256
    ):
        raise ValueError("parent v4.4 protocol identity/body mismatch")
    if (
        report.get("schema_version") != PARENT_PROTOCOL_REPORT_SCHEMA
        or report.get("status") != PARENT_PROTOCOL_STATUS
        or report.get("experiment_id") != PARENT_PROTOCOL_ID
    ):
        raise ValueError("parent v4.4 protocol report identity mismatch")
    outputs = manifest.get("outputs") or {}
    if (
        manifest.get("schema_version") != PARENT_PROTOCOL_MANIFEST_SCHEMA
        or manifest.get("status") != PARENT_PROTOCOL_STATUS
        or manifest.get("experiment_id") != PARENT_PROTOCOL_ID
        or manifest.get("training_started") is not False
        or manifest.get("prospective_opened_or_hashed") is not False
        or outputs.get("protocol.json") != protocol_sha
        or outputs.get("report.json") != report_sha
    ):
        raise ValueError("parent v4.4 protocol manifest binding mismatch")
    contract = protocol.get("probe_evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError("parent v4.4 probe evaluation contract missing")
    if (
        contract.get("authorization") != "post_probe_dev_teacher_forced_only"
        or contract.get("cohort_role") != "dev"
        or contract.get("confirmation_access") is not False
        or contract.get("prospective_access") is not False
        or contract.get("exact_actions") != 240
        or contract.get("outcome_metrics_authorized")
        != {"em": False, "f1": False, "ihr": False}
    ):
        raise ValueError("parent v4.4 probe evaluation scope drift")
    return protocol, {
        "path": PARENT_PROTOCOL.as_posix(),
        "sha256": protocol_sha,
        "report_path": PARENT_PROTOCOL_REPORT.as_posix(),
        "report_sha256": report_sha,
        "manifest_path": PARENT_PROTOCOL_MANIFEST.as_posix(),
        "manifest_sha256": manifest_sha,
        "schema_version": PARENT_PROTOCOL_SCHEMA,
        "status": PARENT_PROTOCOL_STATUS,
        "experiment_id": PARENT_PROTOCOL_ID,
        "protocol_body_canonical_sha256": PARENT_PROTOCOL_BODY_SHA256,
        "implementation_locks": protocol.get("implementation_locks"),
    }


def _validate_dev_rows(path: Path) -> dict[str, Any]:
    digest = _require_hash(path, EXPECTED["release_dev"], role="release dev")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"release dev row is not an object: line={line_number}")
            rows.append(value)
    counts = Counter((row.get("dataset"), row.get("slot")) for row in rows)
    expected = Counter(
        {
            (dataset, slot): 60
            for dataset in ("2wikimultihopqa", "musique")
            for slot in ("q1", "q2_dynamic")
        }
    )
    if len(rows) != 240 or counts != expected or {row.get("split") for row in rows} != {"dev"}:
        raise ValueError("release dev exact cardinality/split mismatch")
    if len({row.get("example_id") for row in rows}) != 240:
        raise ValueError("release dev example_id uniqueness mismatch")
    return {"path": _relative(path), "sha256": digest, "rows": 240}


def _validate_parent_release(parent_protocol: Mapping[str, Any]) -> dict[str, Any]:
    report_path = PROJECT_ROOT / PARENT_RELEASE_DIR / "report.json"
    manifest_path = PROJECT_ROOT / PARENT_RELEASE_DIR / "manifest.json"
    dev_path = PROJECT_ROOT / PARENT_RELEASE_DIR / "dev.jsonl"
    report_sha = _require_hash(
        report_path, EXPECTED["release_report"], role="parent release report"
    )
    manifest_sha = _require_hash(
        manifest_path, EXPECTED["release_manifest"], role="parent release manifest"
    )
    report = _load_object(report_path)
    manifest = _load_object(manifest_path)
    run = _manifest_run(manifest)
    if (
        report.get("schema_version") != PARENT_RELEASE_SCHEMA
        or report.get("status") != PARENT_RELEASE_STATUS
        or report.get("experiment_id") != PARENT_RELEASE_ID
        or report.get("all_release_gates_pass") is not True
        or run.get("schema_version") != PARENT_RELEASE_SCHEMA
        or run.get("status") != PARENT_RELEASE_STATUS
        or run.get("experiment_id") != PARENT_RELEASE_ID
        or run.get("all_release_gates_pass") is not True
    ):
        raise ValueError("parent v4.4 release identity/gate mismatch")
    outputs = report.get("outputs") or {}
    expected_declared = {
        "train.jsonl": (2400, EXPECTED["release_train_declared"]),
        "dev.jsonl": (240, EXPECTED["release_dev"]),
        "confirmation.jsonl": (120, EXPECTED["release_confirmation_declared"]),
    }
    for name, (rows, digest) in expected_declared.items():
        record = outputs.get(name) or {}
        if record.get("rows") != rows or record.get("sha256") != digest:
            raise ValueError(f"parent release declared output drift: {name}")
    if run.get("outputs") != outputs or run.get("identity_locks") != report.get("identity_locks"):
        raise ValueError("parent release manifest/report binding mismatch")
    for role, digest in (
        ("protocol", EXPECTED["parent_protocol"]),
        ("protocol_report", EXPECTED["parent_protocol_report"]),
        ("protocol_manifest", EXPECTED["parent_protocol_manifest"]),
    ):
        if _identity_lock(report, role).get("sha256") != digest:
            raise ValueError(f"parent release {role} lock drift")
    lineage = report.get("protocol_lineage") or {}
    if (
        lineage.get("sha256") != EXPECTED["parent_protocol"]
        or lineage.get("protocol_body_canonical_sha256") != PARENT_PROTOCOL_BODY_SHA256
        or lineage.get("implementation_locks") != parent_protocol.get("implementation_locks")
        or lineage.get("probe_evaluation_contract")
        != parent_protocol.get("probe_evaluation_contract")
    ):
        raise ValueError("parent release protocol lineage mismatch")
    dev = _validate_dev_rows(dev_path)
    return {
        "dev_path": dev["path"],
        "dev_sha256": dev["sha256"],
        "dev_rows": dev["rows"],
        "report_path": _relative(report_path),
        "report_sha256": report_sha,
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": manifest_sha,
        "schema_version": PARENT_RELEASE_SCHEMA,
        "status": PARENT_RELEASE_STATUS,
        "experiment_id": PARENT_RELEASE_ID,
        "train_sha256_declared_not_reopened": EXPECTED["release_train_declared"],
        "confirmation_sha256_declared_not_opened": EXPECTED[
            "release_confirmation_declared"
        ],
    }


def _validate_parent_probe() -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / PARENT_TRAINING_DIR / "manifest.json"
    throughput_path = PROJECT_ROOT / PARENT_TRAINING_DIR / "throughput.json"
    history_path = PROJECT_ROOT / PARENT_TRAINING_DIR / "training_history.jsonl"
    data_report_path = PROJECT_ROOT / PARENT_TRAINING_DIR / "data_report.json"
    adapter_path = PROJECT_ROOT / PARENT_TRAINING_DIR / "final"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    log_path = PROJECT_ROOT / "logs/training/query_controller_action_v1_probe20_seed42_v4_4.log"
    hashes = {
        "manifest": _require_hash(
            manifest_path, EXPECTED["training_manifest"], role="v4.4 training manifest"
        ),
        "throughput": _require_hash(
            throughput_path, EXPECTED["throughput"], role="v4.4 throughput"
        ),
        "history": _require_hash(
            history_path, EXPECTED["training_history"], role="v4.4 training history"
        ),
        "data_report": _require_hash(
            data_report_path, EXPECTED["data_report"], role="v4.4 data report"
        ),
        "adapter": _require_hash(adapter_weights, EXPECTED["adapter"], role="v4.4 adapter"),
        "adapter_config": _require_hash(
            adapter_config, EXPECTED["adapter_config"], role="v4.4 adapter config"
        ),
        "training_log": _require_hash(
            log_path, EXPECTED["training_log"], role="v4.4 training log"
        ),
    }
    manifest = _load_object(manifest_path)
    run = _manifest_run(manifest)
    throughput = run.get("throughput") or {}
    saved = throughput.get("saved_adapter") or {}
    log_gates = throughput.get("log_gates") or {}
    dtype_inventories = (
        saved.get("saved_dtype_inventory"),
        saved.get("live_dtype_inventory"),
        saved.get("clean_reload_dtype_inventory"),
    )
    if (
        manifest.get("status") != "COMPLETE"
        or run.get("phase") != "query_controller_sft_probe"
        or run.get("experiment_id") != PARENT_PROBE_ID
        or throughput.get("global_steps") != 20
        or throughput.get("exact_optimizer_steps_pass") is not True
        or saved.get("adapter_tensor_count") != 256
        or saved.get("save_fidelity_exact") is not True
        or saved.get("clean_reload_single_adapter") is not True
        or saved.get("clean_reload_tensor_exact") is not True
        or saved.get("tensor_reload_exact") is not True
        or saved.get("dtype_inventories_recorded") is not True
        or saved.get("saved_live_clean_reload_dtype_inventories_equal") is not True
        or dtype_inventories
        != ({"torch.float32": 256}, {"torch.float32": 256}, {"torch.float32": 256})
        or log_gates.get("finite_train_loss") is not True
        or log_gates.get("finite_logged_losses") is not True
        or log_gates.get("finite_gradient_norms") is not True
        or log_gates.get("nonzero_trainable_gradient_observed") is not True
    ):
        raise ValueError("v4.4 completed training gates drifted")
    config = run.get("config") or {}
    if (
        config.get("expected_protocol_sha256") != EXPECTED["parent_protocol"]
        or config.get("expected_protocol_report_sha256")
        != EXPECTED["parent_protocol_report"]
        or config.get("expected_protocol_manifest_sha256")
        != EXPECTED["parent_protocol_manifest"]
        or config.get("expected_dev_sha256") != EXPECTED["release_dev"]
        or config.get("expected_release_report_sha256") != EXPECTED["release_report"]
        or config.get("expected_release_manifest_sha256") != EXPECTED["release_manifest"]
    ):
        raise ValueError("v4.4 training asset-lock config drifted")
    asset_lock = (run.get("data_report") or {}).get("asset_lock") or {}
    lock_hashes = asset_lock.get("hashes") or {}
    if (
        asset_lock.get("status") != "PASS"
        or lock_hashes.get("protocol") != EXPECTED["parent_protocol"]
        or lock_hashes.get("protocol_report") != EXPECTED["parent_protocol_report"]
        or lock_hashes.get("protocol_manifest") != EXPECTED["parent_protocol_manifest"]
        or lock_hashes.get("dev") != EXPECTED["release_dev"]
    ):
        raise ValueError("v4.4 training data-report lineage drifted")
    return {
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": hashes["manifest"],
        "status": "COMPLETE",
        "experiment_id": PARENT_PROBE_ID,
        "global_steps": 20,
        "adapter_path": _relative(adapter_path),
        "adapter_sha256": hashes["adapter"],
        "adapter_config_sha256": hashes["adapter_config"],
        "throughput_path": _relative(throughput_path),
        "throughput_sha256": hashes["throughput"],
        "training_history_path": _relative(history_path),
        "training_history_sha256": hashes["history"],
        "data_report_path": _relative(data_report_path),
        "data_report_sha256": hashes["data_report"],
        "training_log_path": _relative(log_path),
        "training_log_sha256": hashes["training_log"],
        "adapter_tensor_count": 256,
        "adapter_dtype_inventory": {"torch.float32": 256},
        "adapter_save_and_clean_reload_exact": True,
    }


def _validate_predecessor_failure() -> dict[str, Any]:
    path = PROJECT_ROOT / PREDECESSOR_FAILURE
    digest = _require_hash(
        path, EXPECTED["predecessor_failure"], role="eval-e0 failure addendum"
    )
    value = _load_object(path)
    side = value.get("side_effect_accounting") or {}
    if (
        value.get("schema_version")
        != "query-controller-v1-eval-attempt-failure-addendum-1.0"
        or value.get("status")
        != "FAIL_PRE_CUDA_NO_GENERATION_NO_PREDICTIONS_RECORDED"
        or value.get("attempt_id")
        != "QUERY-CONTROLLER-ACTION-V1-PROBE20-SEED42-V4-4-DEV-EVAL-E0"
        or side.get("cuda_queried") is not False
        or side.get("generation_started") is not False
        or side.get("prediction_rows_written") != 0
        or side.get("confirmation_opened") is not False
        or side.get("prospective_opened_or_hashed") is not False
    ):
        raise ValueError("eval-e0 failure addendum identity/scope mismatch")
    return {
        "path": PREDECESSOR_FAILURE.as_posix(),
        "sha256": digest,
        "schema_version": value["schema_version"],
        "status": value["status"],
        "attempt_id": value["attempt_id"],
        "stage": side and value["failure"]["stage"],
        "cuda_queried": False,
        "generation_started": False,
        "prediction_rows_written": 0,
    }


def _implementation_locks() -> dict[str, dict[str, str]]:
    result = {}
    for role, relative in IMPLEMENTATION_PATHS.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing eval-e1 implementation: {path}")
        result[role] = {"path": relative.as_posix(), "sha256": _sha256_file(path)}
    return result


def build_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    parent_protocol, protocol_lineage = _validate_parent_protocol()
    release_lineage = _validate_parent_release(parent_protocol)
    probe_lineage = _validate_parent_probe()
    predecessor = _validate_predecessor_failure()
    evaluation_contract = parent_protocol["probe_evaluation_contract"]
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "scope": {
            "kind": "eval_only_successor",
            "parent_training_version": "v4.4",
            "cohort_role": "dev",
            "teacher_forced_q2_state": True,
            "runtime_reader_predicted": False,
            "training_authorized": False,
        },
        "parent_training_lineage": {
            "protocol": protocol_lineage,
            "release": release_lineage,
            "probe": probe_lineage,
        },
        "predecessor_eval_failure": predecessor,
        "change_control": {
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
        },
        "evaluation_contract": evaluation_contract,
        "implementation_locks": _implementation_locks(),
        "authorized_generation": {
            "experiment_id": GENERATION_EXPERIMENT_ID,
            "output_dir": GENERATION_OUTPUT_DIR,
            "cohort_role": "dev",
            "exact_actions": 240,
            "decoding": evaluation_contract["decoding"],
            "outcome_metrics_authorized": {"em": False, "f1": False, "ihr": False},
        },
        "scientific_boundary": {
            "parent_v4_4_training_remains_complete": True,
            "predecessor_e0_is_not_upgraded": True,
            "generation_complete_does_not_imply_mechanism_pass": True,
            "separate_mechanism_scorer_required": True,
            "confirmation_access": False,
            "prospective_access": False,
            "qa_em_f1_ihr_authorized": False,
            "claim": (
                "This protocol authorizes only the frozen dev240 teacher-forced action-"
                "mechanics evaluation of the unchanged v4.4 checkpoint."
            ),
        },
    }
    protocol["protocol_body_canonical_sha256"] = _canonical_sha256(protocol)
    validation = {
        "parent_protocol_bundle_exact": True,
        "parent_release_bundle_exact": True,
        "parent_dev_240_exact": True,
        "parent_training_complete_20_steps": True,
        "parent_adapter_exact": True,
        "e0_failure_preserved": True,
        "implementation_locks_current": True,
        "evaluation_contract_byte_semantics_unchanged": True,
        "retraining_authorized": False,
        "confirmation_opened_or_hashed": False,
        "prospective_opened_or_hashed": False,
        "em_f1_ihr_authorized": False,
    }
    return protocol, validation


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def freeze(output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"append-only eval protocol already exists: {output_dir}")
    protocol, checks = build_protocol()
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": created_at,
        "checks": checks,
        "parent_training_manifest_sha256": EXPECTED["training_manifest"],
        "parent_adapter_sha256": EXPECTED["adapter"],
        "authorized_generation_experiment_id": GENERATION_EXPERIMENT_ID,
        "scientific_boundary": protocol["scientific_boundary"],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ) as tmp_name:
        tmp = Path(tmp_name)
        _write_json(tmp / "protocol.json", protocol)
        _write_json(tmp / "report.json", report)
        protocol_sha = _sha256_file(tmp / "protocol.json")
        report_sha = _sha256_file(tmp / "report.json")
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "training_started": False,
            "generation_started": False,
            "confirmation_opened_or_hashed": False,
            "prospective_opened_or_hashed": False,
            "answer_scoring_performed": False,
            "outputs": {
                "protocol.json": protocol_sha,
                "report.json": report_sha,
            },
        }
        _write_json(tmp / "manifest.json", manifest)
        os.replace(tmp, output_dir)
    return {
        "protocol": _sha256_file(output_dir / "protocol.json"),
        "report": _sha256_file(output_dir / "report.json"),
        "manifest": _sha256_file(output_dir / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Validate all inputs and print the prospective protocol without writing.",
    )
    args = parser.parse_args()
    if args.check_only:
        protocol, checks = build_protocol()
        print(
            json.dumps(
                {"status": "PASS_CHECK_ONLY", "checks": checks, "protocol": protocol},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return
    hashes = freeze(args.output_dir)
    print(
        json.dumps(
            {
                "status": STATUS,
                "output_dir": str(args.output_dir),
                "hashes": hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
