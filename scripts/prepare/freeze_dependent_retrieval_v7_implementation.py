#!/usr/bin/env python
"""Freeze the complete v7 implementation and authorize *planner only*.

This is a CPU-only, Gold-free integrity command.  It revalidates the frozen
v7 preregistration, the producer-passage truncation addendum and the recursive
trajectory-semantics addendum, re-hashes the question-only cohort, every
already-frozen code/model/Wiki18 content lock, and the now-complete runtime
implementation.  Its only authorization is the question-only planner call.
Retrieval/materialisation, Gold attachment and answer evaluation remain
blocked until a separate post-plan lock is written.

The preregistration anticipated ``kgproweight/retrieval/dependent_v7.py``.
The final implementation intentionally has no such module: the single passage
projection lives in the materialisation runner; the generator verifies its
byte commitment and passes the exact same list object to the reader prompt and
the verifier.  This resolution is checked and recorded explicitly rather than
silently dropping the planned path.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.paths import model_path
from scripts.prepare import freeze_dependent_retrieval_v7 as v7_freeze


SCHEMA_VERSION = "subquestion-dependent-retrieval-v7-implementation-lock-1"
STATUS = "AUTHORIZED_PLANNER_ONLY"
SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-IMPLEMENTATION-LOCK"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "c7f1674f62a191671a22844e5589c3f9b80a990aae3c0344cd4001e47a50395d"
)
EXPECTED_ADDENDUM_SHA256 = (
    "0c93f29ab9356b0a818ce398482f096a67533c487e9bdca44271eb8c9a8ecf78"
)
EXPECTED_TRAJECTORY_ADDENDUM_SHA256 = (
    "53738e0474e677af89a08ba2cc16e98f6b0ecd3613dbd45608566065e46bfe2d"
)
TRAJECTORY_ADDENDUM_SCHEMA = (
    "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
)
TRAJECTORY_ADDENDUM_STATUS = "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
TRAJECTORY_ADDENDUM_SCOPE = (
    "RECURSIVE_ARM_SPECIFIC_PRODUCER_CONTEXT_ESTIMAND_CLARIFICATION"
)
TRAJECTORY_INVARIANTS = {
    "shared_root_identical_query_requires_identical_producer_passages": True,
    "divergent_upstream_bridges_may_induce_arm_specific_producer_passages": True,
    "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
    "per_hop_logical_retrieval_budget_equal": True,
}
EXPECTED_N = 40
EXPECTED_COUNTS = {"hotpotqa": 20, "musique": 20}
EXPECTED_TARGET_TYPES = {"hotpotqa": "relation_graph", "musique": "subquery_graph"}
EXPECTED_TASK_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step",
        "step_sha256",
        "producer_passages",
        "producer_passages_sha256",
        "gold_access",
    }
)
EXPECTED_ANSWER_KEYS = frozenset(
    {
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step_sha256",
        "producer_passages_sha256",
        "verified",
        "verified_answer",
        "telemetry",
        "gold_access",
    }
)

DEFAULT_PREREGISTRATION = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/protocol.json"
)
DEFAULT_PREREGISTRATION_MANIFEST = DEFAULT_PREREGISTRATION.with_name("manifest.json")
DEFAULT_ADDENDUM = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration_addendum_producer_truncation_v1/protocol.json"
)
DEFAULT_ADDENDUM_MANIFEST = DEFAULT_ADDENDUM.with_name("manifest.json")
DEFAULT_TRAJECTORY_ADDENDUM = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_recursive_trajectory_v1/protocol.json"
)
DEFAULT_TRAJECTORY_ADDENDUM_MANIFEST = DEFAULT_TRAJECTORY_ADDENDUM.with_name(
    "manifest.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_implementation_lock_v1"
)
DEFAULT_RUNTIME_PATHS = {
    "retrieval_runner": Path(
        "scripts/pilot/materialize_paired_dependent_retrieval_v7.py"
    ),
    "subanswer_generator": Path("scripts/pilot/generate_grounded_subanswers_v7.py"),
    "gold_finalizer": Path(
        "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"
    ),
    "evaluator": Path("scripts/eval/evaluate_paired_dependent_retrieval_v7.py"),
}
PLANNED_OPTIONAL_HELPER = Path("kgproweight/retrieval/dependent_v7.py")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_lock(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or (resolved.stat().st_size <= 0 and not allow_empty):
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def tree_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required directory is missing: {resolved}")
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix()
        size = child.stat().st_size
        child_hash = sha256_file(child)
        files.append({"path": relative, "size_bytes": size, "sha256": child_hash})
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(child_hash.encode("ascii") + b"\n")
    if not files:
        raise ValueError(f"required directory has no files: {resolved}")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _path_from_lock(lock: Mapping[str, Any], label: str) -> Path:
    raw = str(lock.get("path") or "")
    if not raw:
        raise ValueError(f"{label} has no path")
    return Path(raw).expanduser().resolve()


def verify_file_lock(lock: Mapping[str, Any], label: str) -> dict[str, Any]:
    current = file_lock(
        _path_from_lock(lock, label), allow_empty=int(lock.get("size_bytes", -1)) == 0
    )
    if current["size_bytes"] != int(lock.get("size_bytes", -1)):
        raise ValueError(f"{label} size drift")
    if current["sha256"] != str(lock.get("sha256") or ""):
        raise ValueError(f"{label} SHA256 drift")
    return current


def verify_tree_lock(lock: Mapping[str, Any], label: str) -> dict[str, Any]:
    current = tree_lock(_path_from_lock(lock, label))
    for key in ("file_count", "size_bytes", "tree_sha256", "files"):
        if current[key] != lock.get(key):
            raise ValueError(f"{label} tree content drift: {key}")
    return current


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_lock: Mapping[str, Any],
    expected_status: str,
    label: str,
) -> None:
    if manifest.get("status") != expected_status:
        raise ValueError(f"{label} status drift")
    if manifest.get("gold_access") is not False:
        raise ValueError(f"{label} Gold boundary drift")
    recorded = manifest.get("protocol")
    if not isinstance(recorded, Mapping):
        recorded = (manifest.get("artifacts") or {}).get("protocol")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{label} has no protocol lock")
    if str(recorded.get("sha256") or "") != str(protocol_lock["sha256"]):
        raise ValueError(f"{label} protocol SHA256 drift")


def _validate_cohorts(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    outputs = preregistration.get("outputs") or {}
    for name in (
        "development",
        "planner",
        "reclassification_ledger",
        "unselected_commitment",
    ):
        lock = outputs.get(name)
        if not isinstance(lock, Mapping):
            raise ValueError(f"preregistration output lock missing: {name}")
        verify_file_lock(lock, f"preregistration.outputs.{name}")

    development = read_jsonl(_path_from_lock(outputs["development"], "development"))
    planner = read_jsonl(_path_from_lock(outputs["planner"], "planner"))
    if len(development) != EXPECTED_N or len(planner) != EXPECTED_N:
        raise ValueError("v7 cohorts must each contain exactly 40 rows")
    if Counter(str(row.get("dataset")) for row in development) != Counter(EXPECTED_COUNTS):
        raise ValueError("v7 development cohort dataset counts drifted")
    if v7_freeze.planner_rows(development) != planner:
        raise ValueError("planner cohort is not the exact projection of development")

    keys: list[str] = []
    for index, row in enumerate(development):
        v7_freeze.assert_answer_free(row, location=f"development[{index}]")
        dataset = str(row.get("dataset") or "")
        qid = str(row.get("qid") or "")
        question = str(row.get("question") or "")
        if str(row.get("question_key") or "") != f"{dataset}::{qid}":
            raise ValueError(f"development[{index}] identity mismatch")
        if question_sha256(question) != str(row.get("question_sha256") or ""):
            raise ValueError(f"development[{index}] question hash mismatch")
        if row.get("target_type") != EXPECTED_TARGET_TYPES.get(dataset):
            raise ValueError(f"development[{index}] target_type drift")
        if row.get("role") != "development_consumed" or row.get("gold_access") is not False:
            raise ValueError(f"development[{index}] role/Gold boundary drift")
        keys.append(str(row["question_key"]))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate v7 question identity")
    expected_order_hash = str(
        (preregistration.get("population") or {}).get("question_key_order_sha256") or ""
    )
    if sha256_text("\n".join(keys)) != expected_order_hash:
        raise ValueError("v7 question order commitment drift")
    return {
        "n": len(development),
        "by_dataset": dict(Counter(str(row["dataset"]) for row in development)),
        "question_key_order_sha256": expected_order_hash,
    }


def _validate_preregistration_semantics(preregistration: Mapping[str, Any]) -> None:
    if preregistration.get("schema_version") != v7_freeze.SCHEMA_VERSION:
        raise ValueError("v7 preregistration schema drift")
    if preregistration.get("status") != v7_freeze.STATUS:
        raise ValueError("v7 preregistration status drift")
    if preregistration.get("execution_authorization") != (
        "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise ValueError("v7 preregistration authorization drift")
    if preregistration.get("scope") != SCOPE:
        raise ValueError("v7 preregistration scope drift")
    if preregistration.get("gold_access") is not False:
        raise ValueError("v7 preregistration Gold boundary drift")
    if (preregistration.get("future_experiment_ids") or {}).get(
        "implementation_lock"
    ) != EXPERIMENT_ID:
        raise ValueError("implementation Experiment ID drift")
    if preregistration.get("decision_gates") != {
        "materialization": v7_freeze.MATERIALIZATION_GATES,
        "gold_free_mechanism": v7_freeze.MECHANISM_GATES,
        "development_utility": v7_freeze.UTILITY_GATES,
    }:
        raise ValueError("v7 decision gates drift")


def _validate_addendum_semantics(
    addendum: Mapping[str, Any], preregistration_lock: Mapping[str, Any]
) -> None:
    if addendum.get("schema_version") != "subquestion-dependent-retrieval-v7-effective-addendum-1":
        raise ValueError("v7 truncation addendum schema drift")
    if addendum.get("status") != "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL":
        raise ValueError("v7 truncation addendum status drift")
    if addendum.get("execution_authorization") != (
        "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise ValueError("v7 truncation addendum authorization drift")
    parent = (addendum.get("parents") or {}).get("parent_preregistration") or {}
    if str(parent.get("sha256") or "") != str(preregistration_lock["sha256"]):
        raise ValueError("truncation addendum parent preregistration drift")
    invariants = addendum.get("effective_invariants") or {}
    expected = {
        "producer_passages_max": 10,
        "producer_text_unicode_chars_max_each": 1200,
        "python_slice": "text[:1200]",
        "projection_fields": ["doc_id", "title", "text"],
        "reader_and_verifier_projection_hash_equal": True,
        "answer_in_unseen_suffix_never_verified": True,
    }
    if invariants != expected:
        raise ValueError("truncation addendum effective invariants drift")
    if addendum.get("gold_access") is not False:
        raise ValueError("truncation addendum Gold boundary drift")


def _verify_addendum_references(
    addendum: Mapping[str, Any],
    *,
    preregistration_lock: Mapping[str, Any],
    preregistration_manifest_lock: Mapping[str, Any],
) -> None:
    parents = addendum.get("parents") or {}
    expected = {
        "parent_preregistration": preregistration_lock,
        "parent_preregistration_manifest": preregistration_manifest_lock,
    }
    for name, current in expected.items():
        recorded = parents.get(name)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"truncation addendum parent lock missing: {name}")
        if (
            str(recorded.get("sha256") or "") != str(current["sha256"])
            or int(recorded.get("size_bytes", -1)) != int(current["size_bytes"])
        ):
            raise ValueError(f"truncation addendum parent lock drift: {name}")
    for name in ("design_addendum", "design_addendum_manifest"):
        recorded = parents.get(name)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"truncation addendum parent lock missing: {name}")
        verify_file_lock(recorded, f"truncation_addendum.parents.{name}")
    verified = addendum.get("verified_parent_artifacts") or {}
    for name, recorded in verified.items():
        if not isinstance(recorded, Mapping):
            raise ValueError(f"invalid addendum verified artifact: {name}")
        verify_file_lock(recorded, f"truncation_addendum.verified.{name}")


def _validate_trajectory_addendum_semantics(
    trajectory_addendum: Mapping[str, Any],
) -> None:
    if trajectory_addendum.get("schema_version") != TRAJECTORY_ADDENDUM_SCHEMA:
        raise ValueError("v7 recursive trajectory addendum schema drift")
    if trajectory_addendum.get("status") != TRAJECTORY_ADDENDUM_STATUS:
        raise ValueError("v7 recursive trajectory addendum status drift")
    if trajectory_addendum.get("scope") != TRAJECTORY_ADDENDUM_SCOPE:
        raise ValueError("v7 recursive trajectory addendum scope drift")
    if trajectory_addendum.get("effective_invariants") != TRAJECTORY_INVARIANTS:
        raise ValueError("v7 recursive trajectory effective invariants drift")
    if trajectory_addendum.get("execution_authorization") != (
        "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise ValueError("v7 recursive trajectory authorization drift")
    if trajectory_addendum.get("gold_access") is not False:
        raise ValueError("v7 recursive trajectory Gold boundary drift")


def _verify_trajectory_addendum_references(
    trajectory_addendum: Mapping[str, Any],
    *,
    preregistration_lock: Mapping[str, Any],
    preregistration_manifest_lock: Mapping[str, Any],
    truncation_addendum_lock: Mapping[str, Any],
    truncation_addendum_manifest_lock: Mapping[str, Any],
) -> None:
    parents = trajectory_addendum.get("parents") or {}
    expected = {
        "parent_preregistration": preregistration_lock,
        "parent_preregistration_manifest": preregistration_manifest_lock,
        "producer_truncation_addendum": truncation_addendum_lock,
        "producer_truncation_addendum_manifest": truncation_addendum_manifest_lock,
    }
    for name, current in expected.items():
        recorded = parents.get(name)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"recursive trajectory addendum parent lock missing: {name}")
        if (
            str(recorded.get("sha256") or "") != str(current["sha256"])
            or int(recorded.get("size_bytes", -1)) != int(current["size_bytes"])
        ):
            raise ValueError(f"recursive trajectory addendum parent lock drift: {name}")
    for name in (
        "design_protocol",
        "design_manifest",
        "design_trajectory_addendum",
        "design_trajectory_addendum_manifest",
    ):
        recorded = parents.get(name)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"recursive trajectory addendum parent lock missing: {name}")
        verify_file_lock(recorded, f"recursive_trajectory_addendum.parents.{name}")


def _verify_frozen_inputs_and_code(preregistration: Mapping[str, Any]) -> None:
    for group_name in ("inputs", "code_interfaces_locked_now"):
        group = preregistration.get(group_name) or {}
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"empty frozen lock group: {group_name}")
        for name, lock in group.items():
            if not isinstance(lock, Mapping):
                raise ValueError(f"invalid lock: {group_name}.{name}")
            verify_file_lock(lock, f"{group_name}.{name}")


def _verify_model_content(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    models = preregistration.get("models") or {}
    planner = (models.get("query_planner") or {}).get("content_lock")
    inherited = models.get("inherited_content_locks") or {}
    if not isinstance(planner, Mapping):
        raise ValueError("query planner content lock missing")
    verified = {"query_planner": verify_tree_lock(planner, "models.query_planner")}
    for name in ("retrieval_encoder", "cross_encoder", "strong_sft", "base_model"):
        lock = inherited.get(name)
        if not isinstance(lock, Mapping):
            raise ValueError(f"inherited model content lock missing: {name}")
        verified[name] = verify_tree_lock(lock, f"models.{name}")
    exposed_paths = {
        "retrieval_encoder": (models.get("retrieval_encoder") or {}).get("path"),
        "cross_encoder": (models.get("cross_encoder") or {}).get("path"),
        "strong_sft": (models.get("subanswer_and_final_strong_sft") or {}).get("path"),
        "base_model": (models.get("base_model") or {}).get("path"),
    }
    for name, exposed in exposed_paths.items():
        if Path(str(exposed or "")).expanduser().resolve() != _path_from_lock(
            inherited[name], f"models.{name}"
        ):
            raise ValueError(f"model public path/content-lock mismatch: {name}")

    locked_base = _path_from_lock(inherited["base_model"], "models.base_model")
    planner_config_lock = (preregistration.get("inputs") or {}).get("planner_config")
    if not isinstance(planner_config_lock, Mapping):
        raise ValueError("planner config lock missing")
    planner_config = yaml.safe_load(
        _path_from_lock(planner_config_lock, "inputs.planner_config").read_text(
            encoding="utf-8"
        )
    )
    configured_base = str(((planner_config or {}).get("model") or {}).get("base_model") or "")
    if not configured_base:
        raise ValueError("planner config has no model.base_model")
    resolved_config_base = Path(model_path(configured_base)).expanduser().resolve()
    if resolved_config_base != locked_base:
        raise ValueError("planner config base model differs from locked base model")

    adapter_config_path = _path_from_lock(planner, "models.query_planner") / (
        "adapter_config.json"
    )
    adapter_config = read_json(adapter_config_path)
    adapter_base = Path(
        str(adapter_config.get("base_model_name_or_path") or "")
    ).expanduser().resolve()
    if adapter_base != locked_base:
        raise ValueError("planner adapter base model differs from locked base model")
    return verified


def _verify_wiki18_content(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    assets = preregistration.get("retrieval_assets") or {}
    locks = preregistration.get("retrieval_asset_content_locks") or {}
    if int(assets.get("expected_documents", -1)) != v7_freeze.EXPECTED_WIKI18_DOCUMENTS:
        raise ValueError("Wiki18 expected document count drift")
    if assets.get("counts") != {
        "corpus": v7_freeze.EXPECTED_WIKI18_DOCUMENTS,
        "dense": v7_freeze.EXPECTED_WIKI18_DOCUMENTS,
        "bm25": v7_freeze.EXPECTED_WIKI18_DOCUMENTS,
    }:
        raise ValueError("Wiki18 count lock drift")
    preflight_lock = assets.get("preflight")
    if not isinstance(preflight_lock, Mapping):
        raise ValueError("Wiki18 preflight lock missing")
    verify_file_lock(preflight_lock, "Wiki18 preflight")
    preflight = read_json(_path_from_lock(preflight_lock, "Wiki18 preflight"))
    if preflight.get("status") != "PASS" or int(preflight.get("expected_docs", -1)) != (
        v7_freeze.EXPECTED_WIKI18_DOCUMENTS
    ):
        raise ValueError("Wiki18 preflight result drift")
    verified: dict[str, Any] = {}
    for name in ("corpus", "dense_index"):
        lock = locks.get(name)
        if not isinstance(lock, Mapping):
            raise ValueError(f"Wiki18 content lock missing: {name}")
        verified[name] = verify_file_lock(lock, f"Wiki18.{name}")
    bm25 = locks.get("bm25_index")
    if not isinstance(bm25, Mapping):
        raise ValueError("Wiki18 BM25 content lock missing")
    verified["bm25_index"] = verify_tree_lock(bm25, "Wiki18.bm25_index")
    return verified


def _module_to_path(module: str, project_root: Path) -> Path | None:
    if not (module == "kgproweight" or module.startswith("kgproweight.") or module == "scripts" or module.startswith("scripts.")):
        return None
    relative = Path(*module.split("."))
    module_file = project_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_file = project_root / relative / "__init__.py"
    if package_file.is_file():
        return package_file.resolve()
    return None


def _local_imports(path: Path, project_root: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module:
                modules.append(node.module)
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            resolved = _module_to_path(module, project_root)
            if resolved is not None:
                result.add(resolved)
    return sorted(result)


def local_import_closure(root_paths: Sequence[Path], project_root: Path) -> list[Path]:
    pending = [path.expanduser().resolve() for path in root_paths]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"runtime import root missing: {path}")
        visited.add(path)
        for imported in _local_imports(path, project_root):
            if imported not in visited:
                pending.append(imported)
    return sorted(visited)


def _literal_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset":
            if len(value.args) != 1:
                break
            return frozenset(ast.literal_eval(value.args[0]))
        return ast.literal_eval(value)
    raise ValueError(f"{path}: literal constant not found: {name}")


def _call_uses_passages(function: ast.FunctionDef, callee: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name != callee:
            continue
        if any(isinstance(arg, ast.Name) and arg.id == "passages" for arg in node.args):
            return True
        if any(
            keyword.arg == "passages"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "passages"
            for keyword in node.keywords
        ):
            return True
    return False


def validate_runtime_contract(runtime_paths: Mapping[str, Path]) -> dict[str, Any]:
    runner = runtime_paths["retrieval_runner"].expanduser().resolve()
    generator = runtime_paths["subanswer_generator"].expanduser().resolve()
    runner_task_keys = frozenset(_literal_constant(runner, "C_TASK_KEYS"))
    generator_task_keys = frozenset(_literal_constant(generator, "TASK_KEYS"))
    runner_answer_keys = frozenset(_literal_constant(runner, "C_ANSWER_KEYS"))
    generator_identity = tuple(_literal_constant(generator, "OUTPUT_IDENTITY_KEYS"))
    generated_answer_keys = frozenset(
        {*generator_identity, "verified", "verified_answer", "telemetry", "gold_access"}
    )
    if runner_task_keys != EXPECTED_TASK_KEYS or generator_task_keys != EXPECTED_TASK_KEYS:
        raise ValueError("runner/generator C-task exact schema mismatch")
    if runner_answer_keys != EXPECTED_ANSWER_KEYS or generated_answer_keys != EXPECTED_ANSWER_KEYS:
        raise ValueError("runner/generator C-answer exact schema mismatch")
    if int(_literal_constant(generator, "MAX_NEW_TOKENS")) != 96:
        raise ValueError("subanswer max_new_tokens drift")
    if int(_literal_constant(generator, "MAX_PRODUCER_PASSAGES")) != 10:
        raise ValueError("producer passage count drift")
    if int(_literal_constant(generator, "MAX_PASSAGE_TEXT_CHARS")) != 1200:
        raise ValueError("producer passage truncation drift")
    if int(_literal_constant(runner, "BRIDGE_MAX_DOCS")) != 10 or int(
        _literal_constant(runner, "BRIDGE_MAX_BODY_CHARS")
    ) != 1200:
        raise ValueError("runner producer projection limits drift")
    if _literal_constant(runner, "TRAJECTORY_SEMANTICS_ADDENDUM_SCHEMA") != (
        TRAJECTORY_ADDENDUM_SCHEMA
    ) or _literal_constant(runner, "TRAJECTORY_SEMANTICS_ADDENDUM_STATUS") != (
        TRAJECTORY_ADDENDUM_STATUS
    ):
        raise ValueError("runner recursive trajectory addendum contract drift")

    generator_tree = ast.parse(generator.read_text(encoding="utf-8"), filename=str(generator))
    generate = next(
        (
            node
            for node in generator_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_subanswer_rows"
        ),
        None,
    )
    if generate is None:
        raise ValueError("generator has no generate_subanswer_rows function")
    assignment_ok = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "passages" for target in node.targets)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "task"
        and isinstance(node.value.slice, ast.Constant)
        and node.value.slice.value == "producer_passages"
        for node in ast.walk(generate)
    )
    if not assignment_ok:
        raise ValueError("generator does not bind exact task producer_passages")
    for callee in ("build_subanswer_reader_messages", "parse_and_verify_subanswer"):
        if not _call_uses_passages(generate, callee):
            raise ValueError(f"generator does not pass the same passages object to {callee}")

    runner_tree = ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    projection_count = sum(
        isinstance(node, ast.FunctionDef) and node.name == "_producer_passage_projection"
        for node in runner_tree.body
    )
    if projection_count != 1:
        raise ValueError("runner must contain exactly one producer-passage projection helper")
    if "def _producer_passage_projection" in generator.read_text(encoding="utf-8"):
        raise ValueError("generator must not independently re-project producer passages")
    return {
        "task_schema_exact_and_equal": True,
        "answer_schema_exact_and_equal": True,
        "subanswer_max_new_tokens": 96,
        "producer_passages_max": 10,
        "producer_text_unicode_chars_max_each": 1200,
        "trajectory_semantics_addendum_schema": TRAJECTORY_ADDENDUM_SCHEMA,
        "trajectory_semantics_addendum_status": TRAJECTORY_ADDENDUM_STATUS,
        "single_projection_implementation": f"{runner}::_producer_passage_projection",
        "generator_reprojects": False,
        "generator_prompt_and_verifier_receive_same_passages_binding": True,
        "sharing_mode": "shared_by_immutable_task_byte_commitment_and_same_object_binding",
    }


def build_implementation_protocol(
    *,
    preregistration_path: Path,
    preregistration_manifest_path: Path,
    addendum_path: Path,
    addendum_manifest_path: Path,
    trajectory_addendum_path: Path,
    trajectory_addendum_manifest_path: Path,
    runtime_paths: Mapping[str, Path],
    project_root: Path,
    expected_preregistration_sha256: str = EXPECTED_PREREGISTRATION_SHA256,
    expected_addendum_sha256: str = EXPECTED_ADDENDUM_SHA256,
    expected_trajectory_addendum_sha256: str = EXPECTED_TRAJECTORY_ADDENDUM_SHA256,
    verify_large_content: bool = True,
    enforce_planned_runtime_paths: bool = True,
) -> dict[str, Any]:
    if set(runtime_paths) != set(DEFAULT_RUNTIME_PATHS):
        raise ValueError("implementation lock requires the exact four runtime roles")
    preregistration_lock = file_lock(preregistration_path)
    addendum_lock = file_lock(addendum_path)
    trajectory_addendum_lock = file_lock(trajectory_addendum_path)
    if preregistration_lock["sha256"] != expected_preregistration_sha256:
        raise ValueError("v7 preregistration SHA256 drift")
    if addendum_lock["sha256"] != expected_addendum_sha256:
        raise ValueError("v7 truncation addendum SHA256 drift")
    if trajectory_addendum_lock["sha256"] != expected_trajectory_addendum_sha256:
        raise ValueError("v7 recursive trajectory addendum SHA256 drift")
    preregistration = read_json(preregistration_path)
    addendum = read_json(addendum_path)
    trajectory_addendum = read_json(trajectory_addendum_path)
    _validate_preregistration_semantics(preregistration)
    _validate_addendum_semantics(addendum, preregistration_lock)
    _validate_trajectory_addendum_semantics(trajectory_addendum)

    preregistration_manifest_lock = file_lock(preregistration_manifest_path)
    addendum_manifest_lock = file_lock(addendum_manifest_path)
    trajectory_addendum_manifest_lock = file_lock(trajectory_addendum_manifest_path)
    _validate_manifest(
        read_json(preregistration_manifest_path),
        protocol_lock=preregistration_lock,
        expected_status=v7_freeze.STATUS,
        label="v7 preregistration manifest",
    )
    _validate_manifest(
        read_json(addendum_manifest_path),
        protocol_lock=addendum_lock,
        expected_status="FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL",
        label="v7 truncation addendum manifest",
    )
    _validate_manifest(
        read_json(trajectory_addendum_manifest_path),
        protocol_lock=trajectory_addendum_lock,
        expected_status=TRAJECTORY_ADDENDUM_STATUS,
        label="v7 recursive trajectory addendum manifest",
    )
    _verify_addendum_references(
        addendum,
        preregistration_lock=preregistration_lock,
        preregistration_manifest_lock=preregistration_manifest_lock,
    )
    _verify_trajectory_addendum_references(
        trajectory_addendum,
        preregistration_lock=preregistration_lock,
        preregistration_manifest_lock=preregistration_manifest_lock,
        truncation_addendum_lock=addendum_lock,
        truncation_addendum_manifest_lock=addendum_manifest_lock,
    )
    population = _validate_cohorts(preregistration)
    _verify_frozen_inputs_and_code(preregistration)

    # Full model/index verification is mandatory in the CLI.  The switch is
    # injectable only for small synthetic unit fixtures.
    verified_content = {
        "models": _verify_model_content(preregistration) if verify_large_content else "TEST_FIXTURE_SKIPPED",
        "wiki18": _verify_wiki18_content(preregistration) if verify_large_content else "TEST_FIXTURE_SKIPPED",
    }

    runtime_contract = validate_runtime_contract(runtime_paths)
    runtime_locks = {
        name: file_lock(path) for name, path in sorted(runtime_paths.items())
    }
    planner_generator_path = _path_from_lock(
        preregistration["code_interfaces_locked_now"]["planner_generator"],
        "code_interfaces_locked_now.planner_generator",
    )
    roots = [planner_generator_path, *runtime_paths.values()]
    closure_paths = local_import_closure(roots, project_root.expanduser().resolve())
    closure_locks = {
        str(path.relative_to(project_root.expanduser().resolve())): file_lock(
            path, allow_empty=True
        )
        for path in closure_paths
    }

    planned = preregistration.get("planned_runtime_paths_not_yet_implemented_or_locked") or {}
    for name, path in DEFAULT_RUNTIME_PATHS.items():
        if str(planned.get(name) or "") != path.as_posix():
            raise ValueError(f"planned runtime path drift: {name}")
        if enforce_planned_runtime_paths and runtime_paths[name].expanduser().resolve() != (
            project_root.expanduser().resolve() / path
        ).resolve():
            raise ValueError(f"actual runtime path differs from preregistered path: {name}")
    helper_planned = str(planned.get("dependent_v7_helper") or "")
    if helper_planned != PLANNED_OPTIONAL_HELPER.as_posix():
        raise ValueError("planned dependent_v7 helper path drift")
    helper_resolved = (project_root / PLANNED_OPTIONAL_HELPER).resolve()
    imported_helper = any(path == helper_resolved for path in closure_paths)
    if helper_resolved.exists() or imported_helper:
        raise ValueError(
            "dependent_v7.py unexpectedly exists/is imported; update the explicit runtime resolution"
        )
    planned_resolution = {
        "planned_path": str(helper_resolved),
        "state": "ABSENT_AND_NOT_USED",
        "silently_ignored": False,
        "present_on_disk": False,
        "present_in_actual_import_closure": False,
        "reason": (
            "The sole producer projection is retrieval_runner::_producer_passage_projection. "
            "The generator validates the immutable task commitment and binds that exact "
            "producer_passages object to both prompt construction and verification."
        ),
    }

    planner_cohort = (preregistration.get("outputs") or {})["planner"]
    canonical_contexts = (preregistration.get("inputs") or {})["canonical_A_contexts"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "parents": {
            "preregistration": preregistration_lock,
            "preregistration_manifest": preregistration_manifest_lock,
            "truncation_addendum": addendum_lock,
            "truncation_addendum_manifest": addendum_manifest_lock,
            "trajectory_semantics_addendum": trajectory_addendum_lock,
            "trajectory_semantics_addendum_manifest": trajectory_addendum_manifest_lock,
        },
        "inputs": {
            "development": (preregistration.get("outputs") or {})["development"],
            "planner_cohort": planner_cohort,
            "canonical_A_contexts": canonical_contexts,
        },
        "population": population,
        "planner_contract": {
            "experiment_id": (preregistration["future_experiment_ids"])["planner"],
            "cohort": planner_cohort,
            "adapter": preregistration["models"]["query_planner"]["content_lock"],
            "base_model": preregistration["models"]["inherited_content_locks"][
                "base_model"
            ],
            "config": preregistration["inputs"]["planner_config"],
            "generator": preregistration["code_interfaces_locked_now"]["planner_generator"],
            "target_types": EXPECTED_TARGET_TYPES,
            "greedy": True,
            "do_sample": False,
            "max_new_tokens": 512,
            "batch_size_is_non_scientific_runtime_parameter": True,
            "gold_access": False,
        },
        "runtime_code": runtime_locks,
        "lock_issuer": file_lock(Path(__file__)),
        "actual_local_import_closure": closure_locks,
        "actual_import_closure_roots": {
            "planner_generator": file_lock(planner_generator_path),
            **runtime_locks,
        },
        "runtime_contract": runtime_contract,
        "planned_runtime_resolution": {
            "dependent_v7_helper": planned_resolution,
            "all_required_runtime_roles_present": True,
        },
        "content_reverification": {
            "full_hash_verification_performed": bool(verify_large_content),
            "verified": verified_content,
        },
        "authorization": {
            "planner": True,
            "gold_free_materialization": False,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "next_required_lock": (
            "After planner generation, freeze_dependent_retrieval_v7_plans.py must "
            "validate and hash all 40 question-only predictions before materialization."
        ),
        "gold_access": False,
        "gpu_calls_by_this_command": 0,
        "retrieval_calls_by_this_command": 0,
        "scientific_boundary": (
            "This lock proves implementation/input/content identity and authorizes only "
            "the frozen question-only planner. It is not a plan, retrieval, subanswer, "
            "answer, Gold score, confirmation result, or utility claim."
        ),
    }


def write_protocol(protocol: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite implementation lock: {resolved}")
    resolved.mkdir(parents=True, exist_ok=False)
    protocol_path = resolved / "protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(dict(protocol)))
    protocol_lock = file_lock(protocol_path)
    manifest = {
        "schema_version": "subquestion-dependent-retrieval-v7-implementation-lock-manifest-1",
        "experiment_id": protocol["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "protocol": protocol_lock,
        "authorization": dict(protocol["authorization"]),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
    }
    manifest_path = resolved / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return {"protocol": protocol_lock, "manifest": file_lock(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--preregistration_manifest", type=Path, default=DEFAULT_PREREGISTRATION_MANIFEST
    )
    parser.add_argument("--truncation_addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument(
        "--truncation_addendum_manifest", type=Path, default=DEFAULT_ADDENDUM_MANIFEST
    )
    parser.add_argument(
        "--trajectory_addendum", type=Path, default=DEFAULT_TRAJECTORY_ADDENDUM
    )
    parser.add_argument(
        "--trajectory_addendum_manifest",
        type=Path,
        default=DEFAULT_TRAJECTORY_ADDENDUM_MANIFEST,
    )
    for name, default in DEFAULT_RUNTIME_PATHS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_paths = {name: getattr(args, name) for name in DEFAULT_RUNTIME_PATHS}
    protocol = build_implementation_protocol(
        preregistration_path=args.preregistration,
        preregistration_manifest_path=args.preregistration_manifest,
        addendum_path=args.truncation_addendum,
        addendum_manifest_path=args.truncation_addendum_manifest,
        trajectory_addendum_path=args.trajectory_addendum,
        trajectory_addendum_manifest_path=args.trajectory_addendum_manifest,
        runtime_paths=runtime_paths,
        project_root=args.project_root,
        verify_large_content=True,
    )
    result = write_protocol(protocol, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
