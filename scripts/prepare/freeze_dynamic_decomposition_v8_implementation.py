#!/usr/bin/env python
"""Freeze the v8 implementation/runtime and its append-only run semantics.

This command is deliberately CPU-only and Gold-free.  It validates the
approved two-call design, the consumed engineering-smoke cohort, and the
locked development cohort; extracts the production runtime contract from the
runner; re-verifies the complete model and Wiki18 content against the prior v7
full-content lock; and writes a new append-only implementation lock.

The command never opens or hashes the sealed prospective cohort.  It also does
not create a smoke/development run directory, load a model, retrieve a
document, generate a prediction, attach Gold, or score an answer.

The small lifecycle helpers in this module are the normative append-only
contract for later materializers: a crashed/failed attempt is retained and a
retry must use a new attempt directory.  Partial JSONL files are never resumed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.retrieval.dynamic_decomposition_v8_cohort import (  # noqa: E402
    COHORT_LOADER_VERSION,
    DEVELOPMENT_ROLE,
    EXPECTED_DEVELOPMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    SEALED_PROSPECTIVE_SHA256,
    load_frozen_v8_cohort,
)


SCHEMA_VERSION = "dynamic-decomposition-v8-implementation-runtime-freeze-2"
MANIFEST_SCHEMA_VERSION = (
    "dynamic-decomposition-v8-implementation-runtime-freeze-manifest-2"
)
STATUS = "AUTHORIZED_SMOKE_THEN_CONDITIONAL_GOLD_FREE_DEVELOPMENT90"
EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-DEVELOPMENT90-"
    "IMPLEMENTATION-FREEZE-RRF100-SEED42-V2"
)

PARENT_IMPLEMENTATION_V1_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_development90_implementation_freeze_"
    "rrf100_seed42_v1"
)
PARENT_IMPLEMENTATION_V1_PROTOCOL_RELATIVE = (
    PARENT_IMPLEMENTATION_V1_RELATIVE / "protocol.json"
)
PARENT_IMPLEMENTATION_V1_MANIFEST_RELATIVE = (
    PARENT_IMPLEMENTATION_V1_RELATIVE / "manifest.json"
)
EXPECTED_PARENT_IMPLEMENTATION_V1_PROTOCOL_SHA256 = (
    "2dcea4c74fa0abbb5f36bb24533594da885dea89ecaa2b41bf4e3027a41e3bbc"
)
EXPECTED_PARENT_IMPLEMENTATION_V1_MANIFEST_SHA256 = (
    "62ad42062e85bc971570db26f3ce13c2a0c5a3aab50fd5621b7883d12c78e471"
)

DESIGN_PROTOCOL_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_two_call_design_dev90_seed20260904_v1/"
    "protocol.json"
)
DESIGN_MANIFEST_RELATIVE = DESIGN_PROTOCOL_RELATIVE.with_name("manifest.json")
EXPECTED_DESIGN_PROTOCOL_SHA256 = (
    "296907e6cbd308270b53f85287327c98bbb92d0173cde7a20135e6cac2545d92"
)
EXPECTED_DESIGN_MANIFEST_SHA256 = (
    "8175df949882096d5afec6f53d05cb7d917d9c14abac27c92b4ed3247040b5ed"
)

SMOKE_DIRECTORY_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1"
)
SMOKE_COHORT_RELATIVE = SMOKE_DIRECTORY_RELATIVE / "smoke.identity_only.jsonl"
SMOKE_REPORT_RELATIVE = SMOKE_DIRECTORY_RELATIVE / "report.json"
SMOKE_MANIFEST_RELATIVE = SMOKE_DIRECTORY_RELATIVE / "manifest.json"
EXPECTED_SMOKE_COHORT_SHA256 = (
    "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606"
)
EXPECTED_SMOKE_REPORT_SHA256 = (
    "2156f59aebaeca8e34fbf0b1e40d46f66d54aa8349aef0af46c0b6ff8baa683b"
)
EXPECTED_SMOKE_MANIFEST_SHA256 = (
    "7eed188667cf8f9436c002705d3d11e0181c95f13b4986320c5a16695ddbf30b"
)

PARENT_CONTENT_LOCK_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_dependent_retrieval_v7_development_implementation_lock_v1_retry1/"
    "protocol.json"
)
PARENT_CONTENT_MANIFEST_RELATIVE = PARENT_CONTENT_LOCK_RELATIVE.with_name(
    "manifest.json"
)
EXPECTED_PARENT_CONTENT_LOCK_SHA256 = (
    "47259e05cebd1771da3022c5ae79f25214ae5010a3bdc834075ecb47fc576bdc"
)
EXPECTED_PARENT_CONTENT_MANIFEST_SHA256 = (
    "7f41e7f3eed499d21953eeeb1e480d6decc0033b56f8539f93180559bad3a1f2"
)

DEFAULT_RUNTIME_PATHS = {
    "runner": Path("scripts/pilot/materialize_dynamic_decomposition_v8.py"),
    "production_driver": Path("scripts/pilot/run_dynamic_decomposition_v8.py"),
    "core": Path("kgproweight/retrieval/dynamic_decomposition_v8.py"),
    "cohort_loader": Path(
        "kgproweight/retrieval/dynamic_decomposition_v8_cohort.py"
    ),
}
RUNTIME_CONTRACT_CONSTANT = "PRODUCTION_RUNTIME_CONTRACT"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_development90_implementation_freeze_"
    "rrf100_seed42_v2"
)
SMOKE_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-RRF100-ENGINEERING-SMOKE4X3-"
    "SEED42-ATTEMPT001"
)
SMOKE_EXPERIMENT_ID_ATTEMPT002 = SMOKE_EXPERIMENT_ID[: -len("001")] + "002"
DEVELOPMENT_EXPERIMENT_ID = (
    "SUBQUESTION-DECOMPOSITION-V8-RRF100-DEVELOPMENT90-SEED42-ATTEMPT001"
)
SMOKE_ATTEMPT001 = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_engineering_smoke4x3_seed42_attempt001"
)
SMOKE_ATTEMPT002 = SMOKE_ATTEMPT001.with_name(
    SMOKE_ATTEMPT001.name[: -len("001")] + "002"
)
SMOKE_ATTEMPT001_RUNNING_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_engineering_smoke4x3_seed42_attempt001/"
    "manifest.running.json"
)
SMOKE_ATTEMPT001_FAILED_RELATIVE = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_engineering_smoke4x3_seed42_attempt001/"
    "manifest.failed.json"
)
EXPECTED_SMOKE_ATTEMPT001_RUNNING_SHA256 = (
    "7b0c745a61e43bdcfa63e6496b6ba706a953e2726d70dce6fc382fc4fe29ae60"
)
EXPECTED_SMOKE_ATTEMPT001_FAILED_SHA256 = (
    "69309de3edcc8d45d3a95193b35a19079867dc8564fd3e4eaa1e08bf6ccb30f3"
)
DEVELOPMENT_ATTEMPT001 = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_development90_seed42_attempt001"
)

EXPECTED_WIKI18_DOCUMENTS = 21_015_324
IDENTITY_FIELDS = ("dataset", "qid", "question")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
MODEL_ROLES = (
    "q1_controller",
    "q2_controller",
    "q1_subanswer_reader",
    "final_reader",
)
RUNTIME_MODEL_ROLES = ("controller", "subanswer_reader", "final_reader")

RUNNING_MANIFEST = "manifest.running.json"
COMPLETE_MANIFEST = "manifest.complete.json"
FAILED_MANIFEST = "manifest.failed.json"


class V8FreezeError(ValueError):
    """The frozen implementation or an append-only run violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_lock(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or (resolved.stat().st_size == 0 and not allow_empty):
        raise FileNotFoundError(f"required file is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def tree_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required directory is missing: {resolved}")
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix()
        child_lock = file_lock(child, allow_empty=True)
        item = {
            "path": relative,
            "size_bytes": child_lock["size_bytes"],
            "sha256": child_lock["sha256"],
        }
        files.append(item)
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii") + b"\0")
        digest.update(str(item["sha256"]).encode("ascii") + b"\n")
    if not files:
        raise V8FreezeError(f"required directory contains no files: {resolved}")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def concise_tree_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(lock["path"]),
        "file_count": int(lock["file_count"]),
        "size_bytes": int(lock["size_bytes"]),
        "tree_sha256": str(lock["tree_sha256"]),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V8FreezeError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise V8FreezeError(f"invalid JSONL framing at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V8FreezeError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _assert_exact_lock(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    lock = file_lock(path)
    if lock["sha256"] != expected_sha256:
        raise V8FreezeError(f"{label} SHA256 drift")
    return lock


def _verify_prior_tree_lock(lock: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Fully re-hash a tree and compare it with a prior full inventory."""

    current = tree_lock(Path(str(lock.get("path") or "")))
    for key in ("file_count", "size_bytes", "tree_sha256", "files"):
        if current[key] != lock.get(key):
            raise V8FreezeError(f"{label} full-content drift: {key}")
    return concise_tree_lock(current)


def _verify_prior_file_lock(lock: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    current = file_lock(Path(str(lock.get("path") or "")))
    for key in ("size_bytes", "sha256"):
        if current[key] != lock.get(key):
            raise V8FreezeError(f"{label} full-content drift: {key}")
    return current


def _module_to_path(module: str, project_root: Path) -> Path | None:
    if not (
        module == "kgproweight"
        or module.startswith("kgproweight.")
        or module == "scripts"
        or module.startswith("scripts.")
    ):
        return None
    relative = Path(*module.split("."))
    module_path = project_root / relative.with_suffix(".py")
    if module_path.is_file():
        return module_path.resolve()
    package_path = project_root / relative / "__init__.py"
    return package_path.resolve() if package_path.is_file() else None


def _local_imports(path: Path, project_root: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            resolved = _module_to_path(module, project_root)
            if resolved is not None:
                imports.add(resolved)
    return sorted(imports)


def local_import_closure(roots: Sequence[Path], project_root: Path) -> list[Path]:
    pending = [path.expanduser().resolve() for path in roots]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"runtime import root missing: {path}")
        visited.add(path)
        pending.extend(
            dependency
            for dependency in _local_imports(path, project_root)
            if dependency not in visited
        )
    return sorted(visited)


def literal_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise V8FreezeError(
                    f"{path}:{name} must be a pure literal for independent freezing"
                ) from exc
    raise V8FreezeError(f"{path}: missing literal constant {name}")


def _value_at(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise V8FreezeError(f"runtime contract missing {dotted}")
        current = current[key]
    return current


def validate_runtime_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the final runner's literal production contract.

    The exact nested shape is intentionally frozen here rather than inferred
    from CLI defaults.  It will be finalized only after the production runner
    exposes ``PRODUCTION_RUNTIME_CONTRACT``.
    """

    if not isinstance(contract, Mapping):
        raise V8FreezeError("production runtime contract must be an object")
    expected = {
        "schema_version": "dynamic-decomposition-v8-production-runtime-contract-1",
        "runtime_version": "dynamic-decomposition-v8-production-runtime-1",
        "gold_access": False,
        "prospective_unlocked": False,
        "seed": 42,
        "production_staged": True,
        "staged_retrieval_contract.stable_deduplicate_cache_misses": True,
        "staged_retrieval_contract.backend_batch_stages": [
            "root_all",
            "q1_all",
            "q2_BC_all",
        ],
        "staged_retrieval_contract.engineering_smoke_logical_retrieval_requests": 84,
        "staged_retrieval_contract.maximum_full_index_passes_per_attempt": 3,
        "shared_hf_runtime.base_model_path": "models/llama3-8b",
        "shared_hf_runtime.strong_sft_adapter_path": (
            "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_"
            "no_text_head/final"
        ),
        "shared_hf_runtime.tokenizer_path": "models/llama3-8b",
        "shared_hf_runtime.tokenizer_source": (
            "base_model_tokenizer_matching_legacy_SFT_evaluation"
        ),
        "shared_hf_runtime.pad_token_policy": "set_to_eos_when_missing",
        "shared_hf_runtime.one_physical_model_instance_for_roles": list(
            RUNTIME_MODEL_ROLES
        ),
        "shared_hf_runtime.torch_dtype": "bfloat16",
        "shared_hf_runtime.local_files_only": True,
        "shared_hf_runtime.peft_is_trainable": False,
        "shared_hf_runtime.chat_template_source": "base_tokenizer.chat_template",
        "shared_hf_runtime.chat_template_add_generation_prompt": True,
        "shared_hf_runtime.model_input_truncation": False,
        "shared_hf_runtime.max_input_tokens_fail_closed": 6144,
        "shared_hf_runtime.role_max_new_tokens.controller": 96,
        "shared_hf_runtime.role_max_new_tokens.subanswer_reader": 96,
        "shared_hf_runtime.role_max_new_tokens.final_reader": 512,
        "shared_hf_runtime.decoding.do_sample": False,
        "shared_hf_runtime.decoding.temperature": None,
        "shared_hf_runtime.decoding.top_p": None,
        "shared_hf_runtime.decoding.seed": 42,
        "canonical_retrieval.dense_top_k": 100,
        "canonical_retrieval.bm25_top_k": 100,
        "canonical_retrieval.rrf_k": 60,
        "canonical_retrieval.rrf_output_k": 100,
        "canonical_retrieval.bge_top_k": 10,
        "canonical_retrieval.query_max_tokens": 128,
        "canonical_retrieval.rerank_text_chars": 1200,
        "canonical_retrieval.model_visible_passage_chars": 1200,
        "canonical_retrieval.expected_documents": EXPECTED_WIKI18_DOCUMENTS,
        "canonical_retrieval.corpus_path": "indexes_wiki18/corpus_flashrag.jsonl",
        "canonical_retrieval.dense_index_path": "indexes_wiki18/e5_fp16.dat",
        "canonical_retrieval.bm25_index_path": "indexes_wiki18/bm25",
        "canonical_retrieval.e5_model_path": "models/e5-base-v2",
        "canonical_retrieval.bge_model_path": "models/bge-reranker-v2-m3",
        "canonical_retrieval.silent_fallback_allowed": False,
        "first_attempts.engineering_smoke.experiment_id": SMOKE_EXPERIMENT_ID,
        "first_attempts.engineering_smoke.output_dir": SMOKE_ATTEMPT001.as_posix(),
        "first_attempts.engineering_smoke.cohort_path": (
            SMOKE_COHORT_RELATIVE.as_posix()
        ),
        "first_attempts.engineering_smoke.cohort_sha256": (
            EXPECTED_SMOKE_COHORT_SHA256
        ),
        "first_attempts.development.experiment_id": DEVELOPMENT_EXPERIMENT_ID,
        "first_attempts.development.output_dir": DEVELOPMENT_ATTEMPT001.as_posix(),
    }
    for dotted, expected_value in expected.items():
        observed = _value_at(contract, dotted)
        if observed != expected_value:
            raise V8FreezeError(
                f"runtime contract drift at {dotted}: "
                f"observed={observed!r}, expected={expected_value!r}"
            )
    logical = _value_at(contract, "logical_budget_by_arm")
    expected_logical = {
        "A_canonical_one_shot": {
            "retrieval": 1,
            "controller": 0,
            "subanswer_reader": 0,
            "final_reader": 1,
        },
        "B_observation_blind": {
            "retrieval": 3,
            "controller": 2,
            "subanswer_reader": 1,
            "final_reader": 1,
        },
        "C_answer_conditioned": {
            "retrieval": 3,
            "controller": 2,
            "subanswer_reader": 1,
            "final_reader": 1,
        },
    }
    if logical != expected_logical:
        raise V8FreezeError("runtime logical per-arm budget drift")
    if (
        12 * sum(values["retrieval"] for values in logical.values())
        != _value_at(
            contract,
            "staged_retrieval_contract.engineering_smoke_logical_retrieval_requests",
        )
    ):
        raise V8FreezeError("staged smoke logical retrieval budget is not 84")
    cache = _value_at(contract, "cache_contract")
    if (
        cache.get("scope") != "in_memory_for_one_locked_materialization_attempt"
        or cache.get("persistent_cache_in_this_runner") is not False
        or cache.get("outer_append_only_resume_required") is not True
        or cache.get("logical_requests_equal_cache_hits_plus_cache_misses") is not True
        or cache.get("physical_executions_equal_cache_misses") is not True
        or set(cache.get("key_forbidden_fields") or ())
        != {"arm_label", "outcome", "gold"}
    ):
        raise V8FreezeError("runtime content-cache contract drift")
    return json.loads(json.dumps(dict(contract), allow_nan=False))


def _validate_design_and_smoke(project_root: Path) -> dict[str, Any]:
    design_protocol = _assert_exact_lock(
        project_root / DESIGN_PROTOCOL_RELATIVE,
        EXPECTED_DESIGN_PROTOCOL_SHA256,
        label="two-call design protocol",
    )
    design_manifest = _assert_exact_lock(
        project_root / DESIGN_MANIFEST_RELATIVE,
        EXPECTED_DESIGN_MANIFEST_SHA256,
        label="two-call design manifest",
    )
    design = read_json(project_root / DESIGN_PROTOCOL_RELATIVE)
    authorization = design.get("researcher_authorization") or {}
    if (
        design.get("status") != "FROZEN_APPROVED_BEFORE_V8_RUNNER_EXECUTION"
        or authorization.get("evaluation_protocol_change_approved") is not True
        or authorization.get("rrf_output_top100_approved") is not True
        or authorization.get(
            "single_strong_sft_model_for_all_generation_roles_approved"
        )
        is not True
        or authorization.get("prospective_validation_authorized") is not False
        or authorization.get("gold_attachment_authorized") is not False
    ):
        raise V8FreezeError("two-call design authorization drift")

    smoke_locks = {
        "cohort": _assert_exact_lock(
            project_root / SMOKE_COHORT_RELATIVE,
            EXPECTED_SMOKE_COHORT_SHA256,
            label="consumed smoke cohort",
        ),
        "report": _assert_exact_lock(
            project_root / SMOKE_REPORT_RELATIVE,
            EXPECTED_SMOKE_REPORT_SHA256,
            label="consumed smoke report",
        ),
        "manifest": _assert_exact_lock(
            project_root / SMOKE_MANIFEST_RELATIVE,
            EXPECTED_SMOKE_MANIFEST_SHA256,
            label="consumed smoke manifest",
        ),
    }
    smoke_rows = read_jsonl(project_root / SMOKE_COHORT_RELATIVE)
    if len(smoke_rows) != 12 or Counter(row.get("dataset") for row in smoke_rows) != Counter(
        {dataset: 4 for dataset in DATASETS}
    ):
        raise V8FreezeError("consumed smoke cohort cardinality drift")
    if any(tuple(row) != IDENTITY_FIELDS or set(row) != set(IDENTITY_FIELDS) for row in smoke_rows):
        raise V8FreezeError("consumed smoke cohort is not identity-only")
    smoke_manifest = read_json(project_root / SMOKE_MANIFEST_RELATIVE)
    if (
        smoke_manifest.get("gold_access") is not False
        or smoke_manifest.get("prospective_opened_or_hashed") is not False
    ):
        raise V8FreezeError("consumed smoke boundary drift")
    return {
        "design_protocol": design_protocol,
        "design_manifest": design_manifest,
        "smoke": smoke_locks,
    }


def _validate_v1_failed_smoke_retry_parent(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Bind v2 to the retained v1 freeze and failed attempt001 terminal."""

    v1_protocol_path = project_root / PARENT_IMPLEMENTATION_V1_PROTOCOL_RELATIVE
    v1_manifest_path = project_root / PARENT_IMPLEMENTATION_V1_MANIFEST_RELATIVE
    running_path = project_root / SMOKE_ATTEMPT001_RUNNING_RELATIVE
    failed_path = project_root / SMOKE_ATTEMPT001_FAILED_RELATIVE
    locks = {
        "implementation_v1_protocol": _assert_exact_lock(
            v1_protocol_path,
            EXPECTED_PARENT_IMPLEMENTATION_V1_PROTOCOL_SHA256,
            label="v1 implementation protocol",
        ),
        "implementation_v1_manifest": _assert_exact_lock(
            v1_manifest_path,
            EXPECTED_PARENT_IMPLEMENTATION_V1_MANIFEST_SHA256,
            label="v1 implementation manifest",
        ),
        "attempt001_running": _assert_exact_lock(
            running_path,
            EXPECTED_SMOKE_ATTEMPT001_RUNNING_SHA256,
            label="smoke attempt001 running manifest",
        ),
        "attempt001_failed": _assert_exact_lock(
            failed_path,
            EXPECTED_SMOKE_ATTEMPT001_FAILED_SHA256,
            label="smoke attempt001 failed manifest",
        ),
    }
    v1_protocol = read_json(v1_protocol_path)
    v1_manifest = read_json(v1_manifest_path)
    running = read_json(running_path)
    failed = read_json(failed_path)
    if (
        v1_protocol.get("schema_version")
        != "dynamic-decomposition-v8-implementation-runtime-freeze-1"
        or v1_protocol.get("experiment_id")
        != "SUBQUESTION-DECOMPOSITION-V8-DEVELOPMENT90-IMPLEMENTATION-FREEZE-RRF100-SEED42-V1"
        or not verify_self_commitment(
            v1_protocol, field="protocol_body_canonical_sha256"
        )
    ):
        raise V8FreezeError("v1 implementation protocol boundary drift")
    if (
        v1_manifest.get("protocol") != locks["implementation_v1_protocol"]
        or not verify_self_commitment(
            v1_manifest, field="manifest_body_canonical_sha256"
        )
    ):
        raise V8FreezeError("v1 implementation manifest boundary drift")
    if (
        running.get("status") != "RUNNING_NEW_ATTEMPT_NO_IN_PLACE_RESUME"
        or running.get("experiment_id") != SMOKE_EXPERIMENT_ID
        or running.get("attempt_id") != "attempt001"
        or running.get("implementation_protocol")
        != locks["implementation_v1_protocol"]
        or running.get("gold_access") is not False
        or running.get("prospective_opened_or_hashed") is not False
        or not verify_self_commitment(
            running, field="manifest_body_canonical_sha256"
        )
    ):
        raise V8FreezeError("smoke attempt001 running boundary drift")
    if (
        failed.get("status") != "FAILED_RETAINED_APPEND_ONLY"
        or failed.get("experiment_id") != SMOKE_EXPERIMENT_ID
        or failed.get("attempt_id") != "attempt001"
        or failed.get("complete_stage_descriptors") != {}
        or failed.get("running_manifest") != locks["attempt001_running"]
        or failed.get("gold_access") is not False
        or failed.get("prospective_opened_or_hashed") is not False
        or not verify_self_commitment(
            failed, field="manifest_body_canonical_sha256"
        )
    ):
        raise V8FreezeError("smoke attempt001 FAILED terminal boundary drift")
    return locks


def _validate_development_cohort() -> dict[str, Any]:
    cohort = load_frozen_v8_cohort(role=DEVELOPMENT_ROLE)
    rows = cohort.get("rows")
    if (
        cohort.get("loader_version") != COHORT_LOADER_VERSION
        or cohort.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or cohort.get("cohort_sha256") != EXPECTED_DEVELOPMENT_SHA256
        or cohort.get("prospective_unlocked") is not False
        or cohort.get("gold_access") is not False
        or not isinstance(rows, list)
        or len(rows) != 90
    ):
        raise V8FreezeError("locked development cohort boundary drift")
    if any(tuple(row) != IDENTITY_FIELDS or set(row) != set(IDENTITY_FIELDS) for row in rows):
        raise V8FreezeError("development cohort is not identity-only")
    if Counter(row["dataset"] for row in rows) != Counter({dataset: 30 for dataset in DATASETS}):
        raise V8FreezeError("development cohort per-dataset counts drift")
    return {
        "loader_version": COHORT_LOADER_VERSION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "development_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "row_count": 90,
        "per_dataset_counts": {dataset: 30 for dataset in DATASETS},
        "prospective_sha256_declared_from_loader_constant_not_rehashed": (
            SEALED_PROSPECTIVE_SHA256
        ),
        "prospective_opened_or_hashed_by_this_command": False,
    }


def _verify_parent_content(
    project_root: Path, *, verify_large_content: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = project_root / PARENT_CONTENT_LOCK_RELATIVE
    manifest_path = project_root / PARENT_CONTENT_MANIFEST_RELATIVE
    protocol_lock = _assert_exact_lock(
        protocol_path,
        EXPECTED_PARENT_CONTENT_LOCK_SHA256,
        label="parent full-content protocol",
    )
    manifest_lock = _assert_exact_lock(
        manifest_path,
        EXPECTED_PARENT_CONTENT_MANIFEST_SHA256,
        label="parent full-content manifest",
    )
    parent = read_json(protocol_path)
    content = parent.get("content_reverification") or {}
    if content.get("full_hash_verification_performed") is not True:
        raise V8FreezeError("parent lock did not perform full-content verification")
    verified = content.get("verified") or {}
    models = verified.get("models") or {}
    wiki18 = verified.get("wiki18") or {}
    required_models = ("base_model", "retrieval_encoder", "cross_encoder", "strong_sft")
    if any(not isinstance(models.get(name), Mapping) for name in required_models):
        raise V8FreezeError("parent model content lock incomplete")
    if any(not isinstance(wiki18.get(name), Mapping) for name in ("corpus", "dense_index", "bm25_index")):
        raise V8FreezeError("parent Wiki18 content lock incomplete")
    if verify_large_content:
        current = {
            "models": {
                name: _verify_prior_tree_lock(models[name], label=f"models.{name}")
                for name in required_models
            },
            "wiki18": {
                "corpus": _verify_prior_file_lock(wiki18["corpus"], label="wiki18.corpus"),
                "dense_index": _verify_prior_file_lock(
                    wiki18["dense_index"], label="wiki18.dense_index"
                ),
                "bm25_index": _verify_prior_tree_lock(
                    wiki18["bm25_index"], label="wiki18.bm25_index"
                ),
            },
        }
    else:
        # Unit tests may inject this branch; the public CLI never exposes it.
        current = {
            "models": {name: concise_tree_lock(models[name]) for name in required_models},
            "wiki18": {
                "corpus": dict(wiki18["corpus"]),
                "dense_index": dict(wiki18["dense_index"]),
                "bm25_index": concise_tree_lock(wiki18["bm25_index"]),
            },
        }
    expected_paths = {
        "base_model": project_root / "models/llama3-8b",
        "retrieval_encoder": project_root / "models/e5-base-v2",
        "cross_encoder": project_root / "models/bge-reranker-v2-m3",
        "strong_sft": project_root
        / "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final",
    }
    for name, expected in expected_paths.items():
        if Path(str(current["models"][name]["path"])).resolve() != expected.resolve():
            raise V8FreezeError(f"locked model path drift: {name}")
    expected_wiki_paths = {
        "corpus": project_root / "indexes_wiki18/corpus_flashrag.jsonl",
        "dense_index": project_root / "indexes_wiki18/e5_fp16.dat",
        "bm25_index": project_root / "indexes_wiki18/bm25",
    }
    for name, expected in expected_wiki_paths.items():
        if Path(str(current["wiki18"][name]["path"])).resolve() != expected.resolve():
            raise V8FreezeError(f"locked Wiki18 path drift: {name}")
    if int(current["wiki18"]["corpus"]["size_bytes"]) != 14_393_573_105:
        raise V8FreezeError("Wiki18 corpus size drift")
    if int(current["wiki18"]["dense_index"]["size_bytes"]) != 32_279_537_664:
        raise V8FreezeError("Wiki18 dense index size drift")
    return current, {"protocol": protocol_lock, "manifest": manifest_lock}


def _tokenizer_and_template_lock(content: Mapping[str, Any]) -> dict[str, Any]:
    # The historical canonical SFT evaluator loads the base tokenizer and sets
    # pad=eos at runtime.  The adapter tree is still separately locked, but its
    # copied tokenizer.json is intentionally not the runtime tokenizer source.
    tokenizer_root = Path(str(content["models"]["base_model"]["path"])).resolve()
    tokenizer_config_path = tokenizer_root / "tokenizer_config.json"
    tokenizer_path = tokenizer_root / "tokenizer.json"
    config = read_json(tokenizer_config_path)
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise V8FreezeError("strong-SFT tokenizer has no non-empty chat_template")
    return {
        "source": str(tokenizer_root),
        "source_policy": "base_model_tokenizer_matching_legacy_SFT_evaluation",
        "pad_token_policy": "set_to_eos_when_missing",
        "tokenizer_config": file_lock(tokenizer_config_path),
        "tokenizer_json": file_lock(tokenizer_path),
        "chat_template_utf8_sha256": sha256_text(template),
        "chat_template_unicode_char_count": len(template),
        "all_generation_roles_use_this_exact_tokenizer_and_template": list(MODEL_ROLES),
    }


def _self_committed_payload(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in value:
        raise V8FreezeError(f"payload already contains self-commitment field {field}")
    result = dict(value)
    result[field] = sha256_bytes(canonical_json_bytes(dict(value)))
    return result


def verify_self_commitment(value: Mapping[str, Any], *, field: str) -> bool:
    digest = value.get(field)
    body = {key: child for key, child in value.items() if key != field}
    return isinstance(digest, str) and digest == sha256_bytes(canonical_json_bytes(body))


def build_implementation_protocol(
    *,
    project_root: Path,
    runtime_paths: Mapping[str, Path],
    generated_at_utc: str,
    verify_large_content: bool = True,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    if set(runtime_paths) != set(DEFAULT_RUNTIME_PATHS):
        raise V8FreezeError("implementation freeze requires the exact runtime path roles")
    resolved_runtime = {
        name: (path if path.is_absolute() else root / path).resolve()
        for name, path in runtime_paths.items()
    }
    for name, expected in DEFAULT_RUNTIME_PATHS.items():
        if resolved_runtime[name] != (root / expected).resolve():
            raise V8FreezeError(f"runtime path differs from canonical path: {name}")

    parents = _validate_design_and_smoke(root)
    batch_retry_parent = _validate_v1_failed_smoke_retry_parent(root)
    cohort = _validate_development_cohort()
    content, parent_content = _verify_parent_content(
        root, verify_large_content=verify_large_content
    )
    contract = validate_runtime_contract(
        literal_constant(resolved_runtime["runner"], RUNTIME_CONTRACT_CONSTANT)
    )
    runtime_locks = {
        name: file_lock(path) for name, path in sorted(resolved_runtime.items())
    }
    closure = local_import_closure(
        [*resolved_runtime.values(), Path(__file__).resolve()], root
    )
    closure_locks = {
        path.relative_to(root).as_posix(): file_lock(path, allow_empty=True)
        for path in closure
    }
    tokenizer = _tokenizer_and_template_lock(content)
    protocol_body = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": generated_at_utc,
        "scope": "CONSUMED_SMOKE4X3_THEN_FROZEN_DEVELOPMENT90_ONLY",
        "parents": {
            **parents,
            "full_content_lock": parent_content,
            "v1_failed_smoke_retry_parent": batch_retry_parent,
        },
        "cohort": cohort,
        "runtime_code": runtime_locks,
        "local_import_closure": closure_locks,
        "runtime_contract": contract,
        "content_reverification": {
            "full_hash_verification_performed_by_this_command": bool(
                verify_large_content
            ),
            "verified_against_parent_protocol_sha256": (
                EXPECTED_PARENT_CONTENT_LOCK_SHA256
            ),
            "content": content,
        },
        "tokenizer_and_chat_template": tokenizer,
        "generation_role_identity": {
            "roles": list(MODEL_ROLES),
            "same_base_tree_sha256": content["models"]["base_model"][
                "tree_sha256"
            ],
            "same_adapter_tree_sha256": content["models"]["strong_sft"][
                "tree_sha256"
            ],
            "same_tokenizer_json_sha256": tokenizer["tokenizer_json"]["sha256"],
            "same_chat_template_sha256": tokenizer[
                "chat_template_utf8_sha256"
            ],
            "runtime_same_python_object_identity_required": True,
        },
        "run_registry": {
            "smoke": {
                "experiment_id": SMOKE_EXPERIMENT_ID,
                "cohort_n": 12,
                "cohort_sha256": EXPECTED_SMOKE_COHORT_SHA256,
                "first_attempt_dir": str((root / SMOKE_ATTEMPT001).resolve()),
                "retry_path_rule": "replace terminal _attemptNNN with a larger NNN",
                "scientific_role": "consumed_engineering_only",
                "next_authorized_attempt_id": "attempt002",
                "next_authorized_experiment_id": SMOKE_EXPERIMENT_ID_ATTEMPT002,
                "next_authorized_attempt_dir": str(
                    (root / SMOKE_ATTEMPT002).resolve()
                ),
                "attempt002_retry_parent": batch_retry_parent[
                    "attempt001_failed"
                ],
            },
            "development": {
                "experiment_id": DEVELOPMENT_EXPERIMENT_ID,
                "cohort_n": 90,
                "cohort_sha256": EXPECTED_DEVELOPMENT_SHA256,
                "first_attempt_dir": str((root / DEVELOPMENT_ATTEMPT001).resolve()),
                "retry_path_rule": "replace terminal _attemptNNN with a larger NNN",
                "scientific_role": "development_ITT",
                "condition": "engineering_smoke_all_runtime_and_mechanism_gates_pass",
            },
        },
        "append_only_execution": {
            "attempt_directory_created_with_exist_ok_false": True,
            "running_manifest_created_exclusive": True,
            "partial_jsonl_resume_allowed": False,
            "in_place_retry_allowed": False,
            "retry_requires_new_attempt_directory_and_id": True,
            "failed_attempt_retained": True,
            "attempt002_must_bind_attempt001_failed_manifest": True,
            "completed_stage_reuse": (
                "only after descriptor and every artifact hash revalidate"
            ),
            "partial_or_uncommitted_stage_reuse_allowed": False,
            "terminal_manifests_mutually_exclusive": [
                COMPLETE_MANIFEST,
                FAILED_MANIFEST,
            ],
            "predictions_frozen_before_separate_gold_join": True,
            "normative_helpers": {
                "reserve": "reserve_attempt_directory",
                "stage_commit": "commit_stage_boundary",
                "reuse_validation": "validate_reusable_stage",
                "terminal": "finalize_attempt",
            },
        },
        "execution_sequence": [
            "retain failed sequential smoke attempt001 and its v1 implementation lock",
            "reserve batched-equivalent consumed smoke4x3 attempt002 with retry_of=attempt001 FAILED manifest",
            "freeze smoke Gold-free report and terminal manifest",
            "require all smoke runtime/mechanism gates",
            "reserve and run development90 attempt001",
            "freeze development predictions before any Gold process",
            "run a separately authorized scorer only after Gold-free gates pass",
        ],
        "authorization": {
            "engineering_smoke_gold_free_materialization": True,
            "development90_gold_free_materialization_after_smoke_pass": True,
            "development_predictions": True,
            "gold_attachment": False,
            "answer_scoring": False,
            "ihr_judging": False,
            "prospective_open_or_hash": False,
            "training": False,
            "reward_or_loss_change": False,
        },
        "gates": {
            "smoke": {
                "row_count": 12,
                "logical_retrieval_requests": 84,
                "production_staged": True,
                "retrieval_batch_stage_order": [
                    "root_all",
                    "q1_all",
                    "q2_BC_all",
                ],
                "full_index_passes_max": 3,
                "runtime_error_count": 0,
                "gold_or_prospective_access_count": 0,
                "all_three_arms_present_rate": 1.0,
                "logical_budget_exact_rate": 1.0,
                "cache_accounting_conservation_rate": 1.0,
                "final_10_unique_rate": 1.0,
            },
            "development_gold_free": {
                "itt_cardinality": 90,
                "per_dataset": 30,
                "logical_retrieval_requests": 630,
                "production_staged": True,
                "retrieval_batch_stage_order": [
                    "root_all",
                    "q1_all",
                    "q2_BC_all",
                ],
                "full_index_passes_max": 3,
                "q1_schema_valid_rate_min_each_dataset": 0.95,
                "B_q2_static_valid_rate_min": 0.90,
                "a1_admissible_rate_min_each_dataset": 0.40,
                "C_dynamic_transition_rate_all_itt_min_each_dataset": 0.32,
                "empty_repeat_padding_query_rate_max": 0.05,
                "logical_ledger_exact_rate": 1.0,
                "logical_B_C_budget_identity_rate": 1.0,
                "cache_accounting_conservation_rate": 1.0,
                "B_static_allowlist_rate": 1.0,
                "C_dynamic_state_binding_integrity_rate": 1.0,
                "a1_ineligible_full_content_identity_rate": 1.0,
                "eligible_dynamic_invalid_original_Q_no_third_call_rate": 1.0,
                "root_and_q1_shared_byte_identity_rate": 1.0,
                "final_10_unique_rate": 1.0,
                "runtime_error_count": 0,
                "gold_or_forbidden_recursive_field_access_count": 0,
            },
            "development_outcome_after_separate_authorization": {
                "primary_C_minus_B_pooled_EM_min": 0.05,
                "primary_C_minus_B_pooled_F1_strictly_positive": True,
                "gained_minus_lost_min_each_dataset": -1,
                "a1_ineligible_prediction_byte_identity_rate": 1.0,
            },
        },
        "scientific_boundary": (
            "This lock proves code/content/config/cohort identity and authorizes "
            "only Gold-free smoke followed conditionally by Gold-free development90. "
            "V2 changes only physical retrieval scheduling from sequential to stable "
            "staged batching; the 84-query smoke logical budget and all model-visible "
            "retrieval/model inputs remain governed by the approved v8 contract. "
            "It is not a retrieval, prediction, score, utility, prospective, or "
            "training result. The official support-source version remains UNKNOWN."
        ),
        "gold_access": False,
        "prospective_opened_or_hashed_by_this_command": False,
        "gpu_calls_by_this_command": 0,
        "retrieval_calls_by_this_command": 0,
    }
    return _self_committed_payload(
        protocol_body, field="protocol_body_canonical_sha256"
    )


def write_implementation_freeze(
    protocol: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite implementation freeze: {output}")
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = output / "protocol.json"
    _write_bytes_exclusive(protocol_path, canonical_json_bytes(dict(protocol)))
    protocol_lock = file_lock(protocol_path)
    manifest_body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": protocol["created_at_utc"],
        "protocol": protocol_lock,
        "authorization": dict(protocol["authorization"]),
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
    }
    manifest = _self_committed_payload(
        manifest_body, field="manifest_body_canonical_sha256"
    )
    manifest_path = output / "manifest.json"
    _write_bytes_exclusive(manifest_path, canonical_json_bytes(manifest))
    return {
        "output_dir": str(output),
        "protocol": protocol_lock,
        "manifest": file_lock(manifest_path),
    }


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _safe_relative_artifact(attempt_dir: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise V8FreezeError(f"unsafe attempt artifact path: {relative!r}")
    resolved_attempt = attempt_dir.resolve()
    resolved = (resolved_attempt / raw).resolve()
    if resolved_attempt not in resolved.parents:
        raise V8FreezeError(f"attempt artifact escapes directory: {relative!r}")
    return resolved


def reserve_attempt_directory(
    *,
    attempt_dir: Path,
    experiment_id: str,
    attempt_id: str,
    implementation_protocol: Mapping[str, Any],
    cohort_lock: Mapping[str, Any],
    created_at_utc: str,
    retry_of: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically reserve a new attempt; an existing directory is never resumed."""

    if not attempt_id.startswith("attempt") or not attempt_id[7:].isdigit():
        raise V8FreezeError("attempt_id must be attemptNNN")
    destination = attempt_dir.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    body = {
        "schema_version": "dynamic-decomposition-v8-run-attempt-running-1",
        "status": "RUNNING_NEW_ATTEMPT_NO_IN_PLACE_RESUME",
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "created_at_utc": created_at_utc,
        "implementation_protocol": dict(implementation_protocol),
        "cohort": dict(cohort_lock),
        "retry_of": dict(retry_of) if retry_of is not None else None,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "partial_jsonl_resume_allowed": False,
    }
    manifest = _self_committed_payload(
        body, field="manifest_body_canonical_sha256"
    )
    _write_bytes_exclusive(
        destination / RUNNING_MANIFEST, canonical_json_bytes(manifest)
    )
    return manifest


def _assert_active_attempt(attempt_dir: Path) -> dict[str, Any]:
    attempt = attempt_dir.expanduser().resolve()
    running_path = attempt / RUNNING_MANIFEST
    if not running_path.is_file():
        raise V8FreezeError("attempt has no running manifest")
    if (attempt / COMPLETE_MANIFEST).exists() or (attempt / FAILED_MANIFEST).exists():
        raise V8FreezeError("attempt is terminal and cannot be modified")
    running = read_json(running_path)
    if not verify_self_commitment(running, field="manifest_body_canonical_sha256"):
        raise V8FreezeError("running manifest self-commitment mismatch")
    return running


def commit_stage_boundary(
    *,
    attempt_dir: Path,
    stage_name: str,
    artifact_paths: Sequence[str],
    row_count: int,
    stage_config_sha256: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    """Commit already-written, complete artifacts as one reusable stage boundary."""

    if not stage_name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in stage_name):
        raise V8FreezeError("invalid stage_name")
    if row_count < 0 or len(stage_config_sha256) != 64:
        raise V8FreezeError("invalid stage row count/config hash")
    attempt = attempt_dir.expanduser().resolve()
    _assert_active_attempt(attempt)
    locks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative in artifact_paths:
        if relative in seen:
            raise V8FreezeError("duplicate stage artifact path")
        seen.add(relative)
        path = _safe_relative_artifact(attempt, relative)
        lock = file_lock(path)
        lock["path"] = relative
        locks.append(lock)
    if not locks:
        raise V8FreezeError("stage boundary requires at least one complete artifact")
    body = {
        "schema_version": "dynamic-decomposition-v8-stage-boundary-1",
        "status": "COMPLETE_HASH_VALIDATED_STAGE_BOUNDARY",
        "stage": stage_name,
        "completed_at_utc": completed_at_utc,
        "row_count": row_count,
        "stage_config_sha256": stage_config_sha256,
        "artifacts": locks,
        "partial_rows_reusable": False,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
    }
    descriptor = _self_committed_payload(
        body, field="descriptor_body_canonical_sha256"
    )
    descriptor_path = attempt / "stages" / f"{stage_name}.complete.json"
    _write_bytes_exclusive(descriptor_path, canonical_json_bytes(descriptor))
    return descriptor


def validate_reusable_stage(attempt_dir: Path, stage_name: str) -> dict[str, Any]:
    """Return a reusable stage only after its descriptor and every file revalidate."""

    attempt = attempt_dir.expanduser().resolve()
    descriptor_path = attempt / "stages" / f"{stage_name}.complete.json"
    if not descriptor_path.is_file():
        raise V8FreezeError("no complete stage descriptor; partial output is not reusable")
    descriptor = read_json(descriptor_path)
    if (
        descriptor.get("status") != "COMPLETE_HASH_VALIDATED_STAGE_BOUNDARY"
        or descriptor.get("partial_rows_reusable") is not False
        or descriptor.get("gold_access") is not False
        or not verify_self_commitment(
            descriptor, field="descriptor_body_canonical_sha256"
        )
    ):
        raise V8FreezeError("stage descriptor boundary mismatch")
    artifacts = descriptor.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise V8FreezeError("stage descriptor has no artifacts")
    for index, lock in enumerate(artifacts):
        if not isinstance(lock, Mapping):
            raise V8FreezeError(f"stage artifact lock {index} is invalid")
        path = _safe_relative_artifact(attempt, str(lock.get("path") or ""))
        current = file_lock(path)
        if (
            current["sha256"] != lock.get("sha256")
            or current["size_bytes"] != lock.get("size_bytes")
        ):
            raise V8FreezeError(f"stage artifact drift: {lock.get('path')}")
    return {
        "descriptor": file_lock(descriptor_path),
        "stage": descriptor,
    }


def finalize_attempt(
    *,
    attempt_dir: Path,
    success: bool,
    reason: str,
    completed_at_utc: str,
    required_complete_stages: Sequence[str] = (),
) -> dict[str, Any]:
    """Write exactly one terminal manifest; never remove the running manifest."""

    attempt = attempt_dir.expanduser().resolve()
    running = _assert_active_attempt(attempt)
    stages = {
        stage: validate_reusable_stage(attempt, stage)["descriptor"]
        for stage in required_complete_stages
    }
    if success and not required_complete_stages:
        raise V8FreezeError("successful attempt requires a complete stage boundary")
    terminal_name = COMPLETE_MANIFEST if success else FAILED_MANIFEST
    other_name = FAILED_MANIFEST if success else COMPLETE_MANIFEST
    if (attempt / other_name).exists():
        raise V8FreezeError("opposite terminal manifest already exists")
    body = {
        "schema_version": "dynamic-decomposition-v8-run-attempt-terminal-1",
        "status": "COMPLETE" if success else "FAILED_RETAINED_APPEND_ONLY",
        "experiment_id": running["experiment_id"],
        "attempt_id": running["attempt_id"],
        "completed_at_utc": completed_at_utc,
        "reason": reason,
        "running_manifest": file_lock(attempt / RUNNING_MANIFEST),
        "complete_stage_descriptors": stages,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
    }
    terminal = _self_committed_payload(
        body, field="manifest_body_canonical_sha256"
    )
    _write_bytes_exclusive(attempt / terminal_name, canonical_json_bytes(terminal))
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for name, default in DEFAULT_RUNTIME_PATHS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    # Deliberately no prospective, Gold, alternate cohort, or skip-content flag.
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = build_implementation_protocol(
        project_root=args.project_root,
        runtime_paths={name: getattr(args, name) for name in DEFAULT_RUNTIME_PATHS},
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        verify_large_content=True,
    )
    output = args.output_dir
    if not output.is_absolute():
        output = args.project_root / output
    result = write_implementation_freeze(protocol, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
