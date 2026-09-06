#!/usr/bin/env python
"""Attach scorer Gold only after every v7 Gold-free gate has passed.

The retrieval runner emits three answer-free passage arms plus execution and
query-budget ledgers.  This finalizer treats its command-line Gold paths as
opaque until all of the following have succeeded:

* preregistration, append-only addendum, implementation, and artifact hashes;
* a strict A/B/C identity join and recursive Gold-field audit;
* independently recomputed materialization and mechanism gates; and
* agreement between recomputed telemetry and the runner report.

Only then is ``_index_raw_gold`` called.  Keeping that call below the explicit
``GOLD BOUNDARY`` makes the ordering testable without opening the real dev
files.  This program is CPU-only and never imports a model or retriever.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_key
from kgproweight.retrieval.dependent_merge_v6 import passage_score_key
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir
from scripts.prepare import freeze_dependent_retrieval_v7 as v7_freeze
from scripts.prepare import freeze_dependent_retrieval_v7_implementation as v7_implementation


FINALIZER_VERSION = "paired-dependent-retrieval-v7-finalizer-1"
EXPECTED_REPORT_SCHEMA = "paired-dependent-retrieval-v7-report-1"
EXPECTED_REPORT_STATUS = "COMPLETE_GOLD_FREE_MATERIALIZATION"
EXPECTED_ARM_SCHEMA = "paired-dependent-retrieval-v7-arm-1"
EXPECTED_BUDGET_SCHEMA = "paired-dependent-retrieval-v7-budget-1"
EXPECTED_IMPLEMENTATION_SCHEMA = "subquestion-dependent-retrieval-v7-implementation-lock-1"
EXPECTED_IMPLEMENTATION_STATUS = "AUTHORIZED_PLANNER_ONLY"
EXPECTED_PLAN_LOCK_SCHEMA = "subquestion-dependent-retrieval-v7-plan-lock-1"
EXPECTED_PLAN_LOCK_STATUS = "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
EXPECTED_PLAN_LOCK_EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-PLAN-LOCK-V1"
)
EVAL_PROTOCOL_SCHEMA = "paired-dependent-retrieval-v7-eval-protocol-1"
EVAL_PROTOCOL_STATUS = "FROZEN_AFTER_GOLD_FREE_GATES_AND_GOLD_ATTACHMENT_BEFORE_ANSWER_GENERATION"
EVAL_AUTHORIZATION_SCHEMA = "paired-dependent-retrieval-v7-evaluation-authorization-1"
EXPECTED_TRAJECTORY_ADDENDUM_SCHEMA = (
    "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
)
EXPECTED_TRAJECTORY_ADDENDUM_STATUS = "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
EXPECTED_TRAJECTORY_ADDENDUM_SCOPE = (
    "RECURSIVE_ARM_SPECIFIC_PRODUCER_CONTEXT_ESTIMAND_CLARIFICATION"
)
EXPECTED_TRAJECTORY_ADDENDUM_SHA256 = (
    "53738e0474e677af89a08ba2cc16e98f6b0ecd3613dbd45608566065e46bfe2d"
)
EXPECTED_TRAJECTORY_ADDENDUM_MANIFEST_SHA256 = (
    "a51d05348753e747a9d1955642b1dedefaa26a8bb8ceedf335f043e28d4801d2"
)
EXPECTED_TRAJECTORY_INVARIANTS = {
    "shared_root_identical_query_requires_identical_producer_passages": True,
    "divergent_upstream_bridges_may_induce_arm_specific_producer_passages": True,
    "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
    "per_hop_logical_retrieval_budget_equal": True,
}
DEFAULT_TRAJECTORY_ADDENDUM = Path(
    "outputs/audits/"
    "subquestion_dependent_retrieval_v7_development_preregistration_"
    "addendum_recursive_trajectory_v1/protocol.json"
)

DATASETS = ("hotpotqa", "musique")
EXPECTED_PER_DATASET = 20
EXPECTED_TOTAL = 40
EXPECTED_PASSAGES = 10
PROTECTED_A_PREFIX = 8
ARMS = (
    "A_canonical_one_shot",
    "B_entity_hint_top1",
    "C_verified_subanswer",
)
REPORT_OUTPUTS = (
    "arm_a",
    "arm_b",
    "arm_c",
    "execution_details",
    "budget_ledger",
)
REQUIRED_IMPLEMENTATION_FILES = frozenset(
    {
        "retrieval_runner",
        "subanswer_generator",
        "gold_finalizer",
        "evaluator",
    }
)

# These paths are an explicit lower bound on the dynamically discovered local
# import closure.  The exact closure is also frozen, so adding another local
# scoring dependency later changes the protocol instead of silently escaping
# the hash boundary.
REQUIRED_EVALUATOR_DEPENDENCIES = frozenset(
    {
        "scripts/eval/evaluate_paired_dependent_retrieval_v7.py",
        "scripts/prepare/finalize_paired_dependent_retrieval_v7.py",
        "scripts/prepare/freeze_dependent_retrieval_v7.py",
        "scripts/pilot/score_a1_fixed_context_kg.py",
        "kgproweight/data/prompts.py",
        "kgproweight/data/parsers.py",
        "kgproweight/eval/metrics.py",
        "kgproweight/retrieval/bootstrap.py",
        "kgproweight/utils/logging.py",
    }
)

# Match the frozen preregistration exactly.  ``verified_answer`` and
# ``answer_type`` are intentionally not members: those are Gold-free reader
# telemetry, not scorer labels.
FORBIDDEN_KEYS = v7_freeze.FORBIDDEN_KEYS

COMMON_ARM_FIELDS = (
    "row_id",
    "question_key",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "family_sha256",
    "role",
    "gold_access",
)


class V7FinalizationError(ValueError):
    """A frozen contract or pre-Gold gate was violated."""


def _reject_json_constant(value: str) -> None:
    raise V7FinalizationError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V7FinalizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(value, dict):
        raise V7FinalizationError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, V7FinalizationError) as exc:
                raise V7FinalizationError(
                    f"{path}:{line_number}: invalid strict JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise V7FinalizationError(
                    f"{path}:{line_number}: JSONL row is not an object"
                )
            rows.append(value)
    return rows


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _passages_sha256(value: Any) -> str:
    """Match the immutable canonical-context/historical runner identity.

    Passage-arm hashes intentionally retain ``json.dumps``' default separators.
    C producer-task projections are a distinct compact-JSON contract and are
    only compared to their own prompt/verifier commitments below.
    """

    blob = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": v7_freeze.sha256_file(resolved),
    }


def _lock_equal(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare content locks while normalising legacy relative paths."""

    try:
        paths_equal = Path(str(observed.get("path") or "")).expanduser().resolve() == Path(
            str(expected.get("path") or "")
        ).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return paths_equal and all(
        observed.get(key) == expected.get(key) for key in ("size_bytes", "sha256")
    )


def _assert_current_lock(lock: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(lock, Mapping):
        raise V7FinalizationError(f"{label} is not a file lock")
    path = Path(str(lock.get("path") or "")).expanduser().resolve()
    current = _file_lock(path)
    if not _lock_equal(current, lock):
        raise V7FinalizationError(f"{label} content differs from its frozen lock")
    return path


def _tree_member_lock(tree: Mapping[str, Any], relative: str, *, label: str) -> dict[str, Any]:
    """Project one required file from a frozen directory tree lock."""

    root = Path(str(tree.get("path") or "")).expanduser().resolve()
    members = tree.get("files")
    if not isinstance(members, list):
        raise V7FinalizationError(f"{label} has no frozen file inventory")
    matches = [row for row in members if isinstance(row, Mapping) and row.get("path") == relative]
    if len(matches) != 1:
        raise V7FinalizationError(f"{label} lacks exactly one {relative} member")
    member = matches[0]
    return {
        "path": str((root / relative).resolve()),
        "size_bytes": int(member.get("size_bytes", -1)),
        "sha256": str(member.get("sha256") or ""),
    }


def _validate_subanswer_model_artifacts(
    detail_rows: Sequence[Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that every actual C-reader call used the frozen strong SFT.

    The materializer preserves the generator's content-bearing model artifact
    in every subanswer telemetry row.  This validation happens before the Gold
    boundary and compares both public identities and critical SHA256 locks to
    the full preregistered model trees.
    """

    models = preregistration.get("models") or {}
    inherited = models.get("inherited_content_locks") or {}
    strong_tree = inherited.get("strong_sft")
    base_tree = inherited.get("base_model")
    if not isinstance(strong_tree, Mapping) or not isinstance(base_tree, Mapping):
        raise V7FinalizationError("preregistration lacks strong-SFT/base content locks")
    expected_public = {
        "base_model": models.get("base_model"),
        "strong_sft_adapter": models.get("subanswer_and_final_strong_sft"),
    }
    if not all(isinstance(value, Mapping) for value in expected_public.values()):
        raise V7FinalizationError("preregistration lacks public strong-SFT/base identities")
    expected_critical = {
        "base_model": {
            "config": _tree_member_lock(base_tree, "config.json", label="base_model"),
            "weight_index": _tree_member_lock(
                base_tree, "model.safetensors.index.json", label="base_model"
            ),
        },
        "strong_sft_adapter": {
            "config": _tree_member_lock(
                strong_tree, "adapter_config.json", label="strong_sft"
            ),
            "weights": _tree_member_lock(
                strong_tree, "adapter_model.safetensors", label="strong_sft"
            ),
        },
    }
    expected_load = {
        "torch_dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "separate_process_required": True,
    }
    artifacts: list[Mapping[str, Any]] = []
    for detail_index, detail in enumerate(detail_rows):
        attempts = detail.get("subanswer_telemetry")
        if not isinstance(attempts, list):
            raise V7FinalizationError(
                f"execution detail {detail_index} lacks subanswer telemetry"
            )
        for attempt_index, attempt in enumerate(attempts):
            telemetry = attempt.get("telemetry") if isinstance(attempt, Mapping) else None
            artifact = telemetry.get("model_artifact") if isinstance(telemetry, Mapping) else None
            if not isinstance(artifact, Mapping):
                raise V7FinalizationError(
                    "C subanswer telemetry lacks its content-bearing model_artifact"
                )
            artifacts.append(artifact)
            if set(artifact) != {"base_model", "strong_sft_adapter", "load_contract"}:
                raise V7FinalizationError("C subanswer model_artifact field set differs")
            for role in ("base_model", "strong_sft_adapter"):
                observed_role = artifact.get(role)
                if not isinstance(observed_role, Mapping):
                    raise V7FinalizationError(f"C subanswer model artifact lacks {role}")
                if observed_role.get("identity") != expected_public[role]:
                    raise V7FinalizationError(
                        f"C subanswer {role} identity differs from frozen model"
                    )
                for field, expected_lock in expected_critical[role].items():
                    observed_lock = observed_role.get(field)
                    if not isinstance(observed_lock, Mapping) or not _lock_equal(
                        observed_lock, expected_lock
                    ):
                        raise V7FinalizationError(
                            f"C subanswer {role}.{field} content lock differs"
                        )
            if artifact.get("load_contract") != expected_load:
                raise V7FinalizationError("C subanswer model load contract differs")
    if not artifacts:
        raise V7FinalizationError("no actual C subanswer model artifact was recorded")
    canonical = _canonical_json(artifacts[0])
    if any(_canonical_json(value) != canonical for value in artifacts[1:]):
        raise V7FinalizationError("C subanswer calls used more than one model artifact")
    return {
        "validated": True,
        "call_count": len(artifacts),
        "model_artifact_sha256": _sha256_json(artifacts[0]),
        "strong_sft_tree_sha256": str(strong_tree.get("tree_sha256") or ""),
        "base_model_tree_sha256": str(base_tree.get("tree_sha256") or ""),
    }


def _evaluator_import_closure_locks() -> dict[str, dict[str, Any]]:
    """Hash the complete current local import closure of the evaluator."""

    project_root = Path(__file__).resolve().parents[2]
    evaluator = project_root / "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"
    paths = v7_implementation.local_import_closure([evaluator], project_root)
    locks = {
        path.relative_to(project_root).as_posix(): v7_implementation.file_lock(
            path, allow_empty=True
        )
        for path in paths
    }
    missing = sorted(REQUIRED_EVALUATOR_DEPENDENCIES - set(locks))
    if missing:
        raise V7FinalizationError(
            f"evaluator import closure lacks required scoring dependencies: {missing}"
        )
    return dict(sorted(locks.items()))


def _assert_gold_free(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in FORBIDDEN_KEYS:
                raise V7FinalizationError(
                    f"forbidden Gold/support field before Gold boundary: {location}.{key}"
                )
            if key.casefold() == "gold_access" and child is not False:
                raise V7FinalizationError(
                    f"gold_access is not false before Gold boundary: {location}.{key}"
                )
            _assert_gold_free(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_gold_free(child, location=f"{location}[{index}]")


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('dataset') or '').strip().casefold()}::{str(row.get('qid') or '').strip()}"


def _common_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in COMMON_ARM_FIELDS if field not in row]
    if missing:
        raise V7FinalizationError(f"arm row lacks common fields: {missing}")
    return {field: row[field] for field in COMMON_ARM_FIELDS}


def _passage_keys(passages: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [passage_score_key(passage) for passage in passages]
    if any(not value for value in result):
        raise V7FinalizationError("empty passage identity")
    return result


def _detail_index(rows: Sequence[Mapping[str, Any]], *, expected_total: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = _row_key(row)
        if key == "::" or key in result:
            raise V7FinalizationError(f"duplicate/invalid execution-detail identity at row {index}")
        result[key] = dict(row)
    if len(result) != expected_total:
        raise V7FinalizationError(
            f"execution-detail population={len(result)}, expected {expected_total}"
        )
    return result


def _budget_identity(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("question_key") or ""),
        int(row.get("dependency_depth") or 0),
        str(row.get("logical_hop_sha256") or ""),
    )


def _validate_budget_row(
    row: Mapping[str, Any],
    *,
    question: str,
    verified_slots: set[str],
    dependencies: Sequence[str],
) -> dict[str, Any]:
    if row.get("schema_version") != EXPECTED_BUDGET_SCHEMA:
        raise V7FinalizationError("unexpected budget-ledger schema")
    if row.get("gold_access") is not False:
        raise V7FinalizationError("budget-ledger gold_access is not false")
    b = row.get("B")
    c = row.get("C")
    if not isinstance(b, Mapping) or not isinstance(c, Mapping):
        raise V7FinalizationError("budget row lacks B/C slot objects")
    try:
        b_count = int(b["logical_query_count"])
        c_count = int(c["logical_query_count"])
        b_physical = int(b["physical_slot_count"])
        c_physical = int(c["physical_slot_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V7FinalizationError("invalid B/C budget counts") from exc
    if b_count != c_count or b_count not in {0, 1}:
        raise V7FinalizationError("B/C logical query budgets are unequal or outside {0,1}")
    if b_physical != b_count or c_physical != c_count:
        raise V7FinalizationError("logical and physical B/C query slots differ")
    active = bool(row.get("paired_active"))
    if active != (b_count == 1) or row.get("budget_equal") is not True:
        raise V7FinalizationError("paired_active/budget_equal telemetry differs from counts")
    if active and row.get("paired_skip_reason") is not None:
        raise V7FinalizationError("active paired query carries a skip reason")
    is_root = bool(row.get("is_root"))
    padding = 0
    for arm, slot in (("B", b), ("C", c)):
        query = slot.get("query")
        query_hash = slot.get("query_sha256")
        physical_id = slot.get("physical_slot_id")
        logical_id = slot.get("logical_slot_id")
        if not isinstance(logical_id, str) or not logical_id.startswith(arm + "::"):
            raise V7FinalizationError("invalid arm-tagged logical query slot")
        if active:
            if not isinstance(query, str) or not query:
                raise V7FinalizationError("active query slot has no query")
            if query_hash != _sha256_text(query):
                raise V7FinalizationError("query SHA256 mismatch")
            if not isinstance(physical_id, str) or not physical_id:
                raise V7FinalizationError("active query slot has no physical id")
            if not is_root and not query.startswith(question + "\n"):
                raise V7FinalizationError(
                    "dependent query does not start with the exact original question"
                )
        else:
            if any(value is not None for value in (query, query_hash, physical_id)):
                padding += 1
    if active and is_root:
        if b.get("query") != c.get("query") or b.get("physical_slot_id") != c.get("physical_slot_id"):
            raise V7FinalizationError("root query was not one shared physical search")
        root_owner_count = int(row.get("actual_shared_physical_search_count", -1))
        if root_owner_count not in {0, 1}:
            raise V7FinalizationError("root physical-search owner count is outside {0,1}")
        if int(row.get("actual_independent_physical_search_count", -1)) != 0:
            raise V7FinalizationError("root independent physical-search count differs")
        if row.get("cross_arm_query_strings_identical") is not True:
            raise V7FinalizationError("shared root query identity telemetry differs")
    elif active:
        if b.get("physical_slot_id") == c.get("physical_slot_id"):
            raise V7FinalizationError("active dependent B/C searches share a physical slot")
        if int(row.get("actual_shared_physical_search_count", -1)) != 0:
            raise V7FinalizationError("dependent shared physical-search count differs")
        if int(row.get("actual_independent_physical_search_count", -1)) != 2:
            raise V7FinalizationError("dependent physical-search count is not two")
        if row.get("cross_arm_query_strings_identical") is not (
            b.get("query") == c.get("query")
        ):
            raise V7FinalizationError("dependent cross-arm query equality telemetry differs")
    else:
        if not isinstance(row.get("paired_skip_reason"), str) or not row["paired_skip_reason"]:
            raise V7FinalizationError("paired skip lacks a reason")
        if int(row.get("actual_shared_physical_search_count", -1)) != 0 or int(
            row.get("actual_independent_physical_search_count", -1)
        ) != 0:
            raise V7FinalizationError("paired skip consumed a physical query")
    unverified_used = 0
    if active and not is_root:
        unverified_used = sum(str(slot) not in verified_slots for slot in dependencies)
        if unverified_used:
            raise V7FinalizationError("active C dependent query used an unverified dependency")
    return {
        "padding": padding,
        "unverified_used": unverified_used,
        "active_dependent": int(active and not is_root),
        "root_physical_slot_id": b.get("physical_slot_id") if active and is_root else None,
        "root_physical_owner_count": (
            int(row.get("actual_shared_physical_search_count", 0))
            if active and is_root
            else 0
        ),
    }


def _validate_subanswer_telemetry(
    rows: Sequence[Mapping[str, Any]], *, question_key_value: str
) -> tuple[int, int, set[str]]:
    parsed = 0
    verified = 0
    verified_slots: set[str] = set()
    task_ids: set[str] = set()
    producer_slots: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or "")
        slot = str(row.get("producer_slot") or "")
        if not task_id or task_id in task_ids or not slot or slot in producer_slots:
            raise V7FinalizationError(
                f"duplicate/invalid subanswer telemetry for {question_key_value} at {index}"
            )
        task_ids.add(task_id)
        producer_slots.add(slot)
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, Mapping):
            raise V7FinalizationError("subanswer telemetry lacks generator/verifier object")
        strict_parse = telemetry.get("strict_parse")
        if not isinstance(strict_parse, Mapping) or not isinstance(strict_parse.get("valid"), bool):
            raise V7FinalizationError("subanswer telemetry lacks strict parse result")
        parse_valid = bool(strict_parse["valid"])
        is_verified = row.get("verified")
        if not isinstance(is_verified, bool):
            raise V7FinalizationError("subanswer verified flag is not boolean")
        if is_verified and not parse_valid:
            raise V7FinalizationError("unparseable subanswer was marked verified")
        producer_hash = str(row.get("producer_passages_sha256") or "")
        if not producer_hash or any(
            telemetry.get(name) != producer_hash
            for name in ("prompt_passages_sha256", "verifier_passages_sha256")
        ):
            raise V7FinalizationError("reader/verifier producer-passage hashes differ")
        if telemetry.get("same_passage_bytes_for_prompt_and_verifier") is not True:
            raise V7FinalizationError("reader/verifier did not consume identical passage bytes")
        verifier = telemetry.get("verification")
        if (
            not isinstance(verifier, Mapping)
            or verifier.get("verification_scope")
            != "surface_locality_not_semantic_entailment"
            or verifier.get("verified") is not is_verified
        ):
            raise V7FinalizationError(
                "subanswer verifier verdict/scope differs from frozen surface gate"
            )
        if is_verified:
            value = row.get("promoted_value")
            if not isinstance(value, str) or not value.strip():
                raise V7FinalizationError("verified subanswer lacks a promoted value")
            if verifier.get("verified_answer") != value:
                raise V7FinalizationError("verified subanswer surface differs from verifier")
            verified_slots.add(slot)
        elif (
            row.get("promoted_value") not in {None, ""}
            or verifier.get("verified_answer") not in {None, ""}
        ):
            raise V7FinalizationError("unverified subanswer carries a promoted value")
        parsed += int(parse_valid)
        verified += int(is_verified)
    return parsed, verified, verified_slots


def _validate_final_ce_trace(
    arm_row: Mapping[str, Any], *, question: str, changed: bool
) -> int:
    trace = arm_row.get("retrieval_trace")
    if not isinstance(trace, Mapping):
        raise V7FinalizationError("B/C arm lacks retrieval_trace")
    rows = trace.get("final_ce_trace")
    if not isinstance(rows, list):
        raise V7FinalizationError("B/C arm lacks final CE trace")
    if changed and not rows:
        raise V7FinalizationError("changed arm has no final full-question CE trace")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise V7FinalizationError("final CE trace row is not an object")
        if row.get("question") != question or row.get("uses_exact_original_question") is not True:
            raise V7FinalizationError("final CE pair did not use exact original question")
        if row.get("question_sha256") != _sha256_text(question):
            raise V7FinalizationError("final CE question hash mismatch")
        key = str(row.get("document_key") or "")
        score = row.get("score")
        if not key or key in seen:
            raise V7FinalizationError("missing/duplicate final CE document key")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise V7FinalizationError("missing/non-finite final CE score")
        seen.add(key)
    return len(rows)


def _validate_merge(
    detail: Mapping[str, Any],
    *,
    arm: str,
    a_passages: Sequence[Mapping[str, Any]],
    output: Sequence[Mapping[str, Any]],
    dependent_candidate_keys: set[str],
) -> tuple[bool, int]:
    a_keys = _passage_keys(a_passages)
    output_keys = _passage_keys(output)
    changed = list(output) != list(a_passages)
    added = set(output_keys) - set(a_keys)
    removed = set(a_keys) - set(output_keys)
    if changed != bool(added):
        raise V7FinalizationError(
            f"{arm} changed passage bytes without retaining a new document identity"
        )
    if len(added) != len(removed) or len(added) > EXPECTED_PASSAGES - PROTECTED_A_PREFIX:
        raise V7FinalizationError(f"{arm} replacement inventory exceeds frozen budget")
    if not added.issubset(dependent_candidate_keys):
        raise V7FinalizationError(f"{arm} injected a root-only/untraced document")
    merge_by_arm = detail.get("merge")
    if not isinstance(merge_by_arm, Mapping) or arm not in merge_by_arm:
        raise V7FinalizationError(f"execution detail lacks merge telemetry for {arm}")
    merge = merge_by_arm.get(arm)
    if merge is None:
        if changed or added or removed:
            raise V7FinalizationError(f"changed {arm} output has no merge telemetry")
        return changed, 0
    if not isinstance(merge, Mapping):
        raise V7FinalizationError(f"invalid merge telemetry for {arm}")
    selected = merge.get("selected_new") or []
    evicted = merge.get("evicted_originals") or []
    if not isinstance(selected, list) or not isinstance(evicted, list):
        raise V7FinalizationError(f"invalid merge inventory for {arm}")
    selected_keys = {str(row.get("document_key") or "") for row in selected if isinstance(row, Mapping)}
    evicted_keys = {str(row.get("document_key") or "") for row in evicted if isinstance(row, Mapping)}
    if (
        selected_keys != added
        or evicted_keys != removed
        or len(selected_keys) != len(selected)
        or len(evicted_keys) != len(evicted)
    ):
        raise V7FinalizationError(f"reported merge inventory differs for {arm}")
    for row in evicted:
        rank = int(row.get("original_rank", -1))
        if rank <= PROTECTED_A_PREFIX or rank > EXPECTED_PASSAGES:
            raise V7FinalizationError(f"{arm} displaced a protected/invalid A passage")
        if not float(row.get("replacement_score")) > float(row.get("score")):
            raise V7FinalizationError(f"{arm} accepted a non-strict CE replacement")
    return changed, len(added)


def audit_materialization(
    arm_a_rows: Sequence[Mapping[str, Any]],
    arm_b_rows: Sequence[Mapping[str, Any]],
    arm_c_rows: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    expected_per_dataset: int = EXPECTED_PER_DATASET,
) -> dict[str, Any]:
    """Recompute all frozen Gold-free safety and mechanism measurements."""

    expected_total = expected_per_dataset * len(DATASETS)
    if any(
        len(rows) != expected_total
        for rows in (arm_a_rows, arm_b_rows, arm_c_rows, detail_rows)
    ):
        raise V7FinalizationError(
            "strict A/B/C/detail population differs from the frozen cohort"
        )
    counts = Counter(str(row.get("dataset") or "") for row in arm_a_rows)
    if counts != Counter({dataset: expected_per_dataset for dataset in DATASETS}):
        raise V7FinalizationError("population is not HotpotQA/MuSiQue at frozen counts")
    details = _detail_index(detail_rows, expected_total=expected_total)
    external_budget_by_identity: dict[tuple[str, int, str], dict[str, Any]] = {}
    for index, row in enumerate(budget_rows):
        _assert_gold_free(row, location=f"budget_ledger[{index}]")
        identity = _budget_identity(row)
        if not all(identity) or identity in external_budget_by_identity:
            raise V7FinalizationError("duplicate/invalid external budget-ledger identity")
        external_budget_by_identity[identity] = dict(row)

    question_keys: list[str] = []
    qids: list[str] = []
    detail_budgets: list[Mapping[str, Any]] = []
    per_dataset: dict[str, Counter[str]] = {dataset: Counter() for dataset in DATASETS}
    aggregate = Counter()
    prompt_input_hashes: dict[str, dict[str, str]] = {}
    root_physical_owners: Counter[str] = Counter()

    for index, triple in enumerate(zip(arm_a_rows, arm_b_rows, arm_c_rows)):
        a, b, c = (dict(value) for value in triple)
        for label, row in (("A", a), ("B", b), ("C", c)):
            _assert_gold_free(row, location=f"arm_{label.lower()}[{index}]")
            if row.get("schema_version") != EXPECTED_ARM_SCHEMA:
                raise V7FinalizationError(f"unexpected {label} arm schema")
        if _common_projection(a) != _common_projection(b) or _common_projection(a) != _common_projection(c):
            raise V7FinalizationError(f"A/B/C identity fields differ at row {index}")
        key = _row_key(a)
        if key != str(a.get("question_key")) or key in question_keys or key not in details:
            raise V7FinalizationError(f"A/B/C/detail identity join failed at row {index}")
        question_keys.append(key)
        qids.append(str(a["qid"]))
        dataset = str(a["dataset"])
        if a.get("role") != "development_consumed":
            raise V7FinalizationError(f"row is not development_consumed: {key}")
        if [a.get("arm"), b.get("arm"), c.get("arm")] != list(ARMS):
            raise V7FinalizationError(f"arm labels differ from frozen A/B/C: {key}")
        if a.get("kg_subgraph") != [] or b.get("kg_subgraph") != [] or c.get("kg_subgraph") != []:
            raise V7FinalizationError(f"non-empty KG input would confound retrieval arms: {key}")
        question = str(a["question"])
        if not question or a.get("question_sha256") != _sha256_text(question):
            raise V7FinalizationError(f"question identity hash differs: {key}")

        passages_by_arm: dict[str, list[Mapping[str, Any]]] = {}
        for arm, row in zip(("A", "B", "C"), (a, b, c)):
            passages = row.get("retrieved_passages")
            if not isinstance(passages, list) or len(passages) != EXPECTED_PASSAGES:
                raise V7FinalizationError(f"{key}::{arm} is not Top-{EXPECTED_PASSAGES}")
            if not all(isinstance(passage, Mapping) for passage in passages):
                raise V7FinalizationError(f"{key}::{arm} passage row is not an object")
            passage_keys = _passage_keys(passages)
            if len(set(passage_keys)) != len(passage_keys):
                raise V7FinalizationError(f"{key}::{arm} contains duplicate documents")
            if row.get("passages_sha256") != _passages_sha256(passages):
                raise V7FinalizationError(f"{key}::{arm} passage hash mismatch")
            passages_by_arm[arm] = passages
        if passages_by_arm["B"][:PROTECTED_A_PREFIX] != passages_by_arm["A"][:PROTECTED_A_PREFIX]:
            raise V7FinalizationError(f"{key}::B changed protected A prefix")
        if passages_by_arm["C"][:PROTECTED_A_PREFIX] != passages_by_arm["A"][:PROTECTED_A_PREFIX]:
            raise V7FinalizationError(f"{key}::C changed protected A prefix")

        detail = details[key]
        _assert_gold_free(detail, location=f"execution_details[{key}]")
        for field in ("question_key", "dataset", "qid", "question", "question_sha256", "family_sha256"):
            if detail.get(field) != a.get(field):
                raise V7FinalizationError(f"detail identity field differs: {key}::{field}")
        if detail.get("gold_access") is not False:
            raise V7FinalizationError(f"detail Gold boundary differs: {key}")
        if "error" in str(detail.get("execution_status") or "").casefold():
            raise V7FinalizationError(f"runtime error status in completed materialization: {key}")
        if detail.get("arm_a_passages_sha256") != a["passages_sha256"] or detail.get(
            "arm_b_passages_sha256"
        ) != b["passages_sha256"] or detail.get("arm_c_passages_sha256") != c["passages_sha256"]:
            raise V7FinalizationError(f"detail/arm passage commitments differ: {key}")

        subanswers = detail.get("subanswer_telemetry")
        if not isinstance(subanswers, list):
            raise V7FinalizationError(f"detail lacks subanswer telemetry: {key}")
        parsed, verified, verified_slots = _validate_subanswer_telemetry(
            subanswers, question_key_value=key
        )
        hops = detail.get("hop_telemetry")
        if not isinstance(hops, list):
            raise V7FinalizationError(f"detail lacks hop telemetry: {key}")
        hops_by_identity: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        dependent_candidate_keys = {"B": set(), "C": set()}
        eligible_dependent = 0
        active_dependent = 0
        for hop in hops:
            if not isinstance(hop, Mapping):
                raise V7FinalizationError(f"hop telemetry is not an object: {key}")
            hop_identity = (
                key,
                int(hop.get("dependency_depth") or 0),
                str(hop.get("logical_hop_sha256") or ""),
            )
            if not all(hop_identity) or hop_identity in hops_by_identity:
                raise V7FinalizationError(f"duplicate/invalid logical hop: {key}")
            hops_by_identity[hop_identity] = hop
            dependencies = hop.get("dependencies")
            if not isinstance(dependencies, list):
                raise V7FinalizationError(f"hop dependencies are not a list: {key}")
            if dependencies:
                eligible_dependent += 1
                active_dependent += int(bool(hop.get("paired_active")))
                for arm in ("B", "C"):
                    arm_hop = hop.get(arm)
                    if arm_hop is None:
                        continue
                    if not isinstance(arm_hop, Mapping):
                        raise V7FinalizationError(f"invalid {arm} hop telemetry: {key}")
                    rerank = arm_hop.get("rerank")
                    if not isinstance(rerank, Mapping):
                        raise V7FinalizationError(f"active {arm} hop lacks rerank telemetry: {key}")
                    for pair in rerank.get("ce_pairs") or []:
                        if isinstance(pair, Mapping) and pair.get("selected_rank") is not None:
                            dependent_candidate_keys[arm].add(str(pair.get("document_key") or ""))

        embedded_budgets = detail.get("budget_ledger")
        if not isinstance(embedded_budgets, list):
            raise V7FinalizationError(f"detail lacks embedded budget ledger: {key}")
        if set(_budget_identity(row) for row in embedded_budgets) != set(hops_by_identity):
            raise V7FinalizationError(f"hop/budget identity inventory differs: {key}")
        for row in embedded_budgets:
            identity = _budget_identity(row)
            if identity not in external_budget_by_identity:
                raise V7FinalizationError(f"embedded budget absent from external ledger: {identity}")
            if _canonical_json(row) != _canonical_json(external_budget_by_identity[identity]):
                raise V7FinalizationError(f"embedded/external budget bytes differ: {identity}")
            hop = hops_by_identity[identity]
            stats = _validate_budget_row(
                row,
                question=question,
                verified_slots=verified_slots,
                dependencies=[str(value) for value in hop["dependencies"]],
            )
            if bool(hop.get("dependencies")) and bool(hop.get("paired_active")) != bool(
                row.get("paired_active")
            ):
                raise V7FinalizationError(f"hop/budget paired activation differs: {identity}")
            aggregate.update(
                {
                    "padding": stats["padding"],
                    "unverified_used": stats["unverified_used"],
                }
            )
            if stats["root_physical_slot_id"] is not None:
                root_physical_owners[str(stats["root_physical_slot_id"])] += int(
                    stats["root_physical_owner_count"]
                )
            detail_budgets.append(row)
        if int(detail.get("successful_paired_dependent_hops", -1)) != active_dependent:
            raise V7FinalizationError(f"successful paired-hop count differs: {key}")

        changed_by_arm: dict[str, bool] = {}
        replacement_count = 0
        for arm, row in (("B", b), ("C", c)):
            changed, replacements = _validate_merge(
                detail,
                arm=arm,
                a_passages=passages_by_arm["A"],
                output=passages_by_arm[arm],
                dependent_candidate_keys=dependent_candidate_keys[arm],
            )
            if bool(row.get("fallback_to_a")) != (not changed):
                raise V7FinalizationError(f"{key}::{arm} fallback flag differs from exact equality")
            _validate_final_ce_trace(row, question=question, changed=changed)
            changed_by_arm[arm] = changed
            replacement_count += replacements

        successful = int(detail["successful_paired_dependent_hops"])
        fallback_exact = True
        if successful == 0:
            fallback_exact = (
                passages_by_arm["B"] == passages_by_arm["A"]
                and passages_by_arm["C"] == passages_by_arm["A"]
            )
            if not fallback_exact:
                raise V7FinalizationError(f"zero-hop B/C fallback is not byte-exact A: {key}")
        prompt_input_hashes[key] = {
            arm: _sha256_json(
                {
                    "question": question,
                    "passages_sha256": _passages_sha256(passages_by_arm[arm]),
                    "kg_subgraph": [],
                }
            )
            for arm in ("A", "B", "C")
        }
        if successful == 0 and len(set(prompt_input_hashes[key].values())) != 1:
            raise V7FinalizationError(f"zero-hop final prompt inputs differ: {key}")

        current = per_dataset[dataset]
        current.update(
            {
                "n": 1,
                "plan_executable": int(bool(detail.get("plan_executable"))),
                "subanswer_tasks": len(subanswers),
                "strict_parse_valid": parsed,
                "mechanically_verified": verified,
                "dependent_hops_eligible": eligible_dependent,
                "dependent_hops_active": active_dependent,
                "B_changed": int(changed_by_arm["B"]),
                "C_changed": int(changed_by_arm["C"]),
            }
        )
        aggregate.update(
            {
                "n": 1,
                "fallback_exact": int(successful == 0 and fallback_exact),
                "zero_hop_fallback": int(successful == 0),
                "replacements": replacement_count,
            }
        )

    if len(detail_budgets) != len(budget_rows) or set(
        _budget_identity(row) for row in detail_budgets
    ) != set(external_budget_by_identity):
        raise V7FinalizationError("external budget ledger has missing/extra rows")
    invalid_root_owners = {
        physical_id: owners
        for physical_id, owners in root_physical_owners.items()
        if owners != 1
    }
    if invalid_root_owners:
        raise V7FinalizationError(
            f"shared root physical slots do not have exactly one execution owner: {invalid_root_owners}"
        )

    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset, values in per_dataset.items():
        n = int(values["n"])
        tasks = int(values["subanswer_tasks"])
        eligible = int(values["dependent_hops_eligible"])
        by_dataset[dataset] = {
            **dict(values),
            "plan_executable_rate": int(values["plan_executable"]) / n,
            "strict_subanswer_json_parse_rate": (
                int(values["strict_parse_valid"]) / tasks if tasks else None
            ),
            "mechanically_verified_subanswer_rate": (
                int(values["mechanically_verified"]) / tasks if tasks else None
            ),
            "paired_dependent_hop_activation_rate": (
                int(values["dependent_hops_active"]) / eligible if eligible else None
            ),
            "retained_new_dependent_document_question_rate_B": int(values["B_changed"]) / n,
            "retained_new_dependent_document_question_rate_C": int(values["C_changed"]) / n,
        }

    return {
        "n": expected_total,
        "identity_join_rate": 1.0,
        "runtime_errors": 0,
        "recursive_forbidden_input_fields": 0,
        "gold_access": False,
        "all_rows_and_arms_top10": True,
        "duplicate_output_documents": 0,
        "unauthorized_A_prefix_displacements": 0,
        "root_only_documents_injected": 0,
        "all_dependent_queries_start_with_exact_original_question": True,
        "all_final_CE_pairs_use_exact_original_question": True,
        "B_C_query_budget_equal_every_question_depth_and_hop": True,
        "budget_padding_queries": int(aggregate["padding"]),
        "unverified_subanswers_used": int(aggregate["unverified_used"]),
        "fallback_pair_and_A_byte_exact": True,
        "zero_successful_hop_fallback_questions": int(aggregate["zero_hop_fallback"]),
        "replacement_count_across_B_and_C": int(aggregate["replacements"]),
        "qid_order_sha256": _sha256_text("\n".join(qids)),
        "question_key_order_sha256": _sha256_text("\n".join(question_keys)),
        "final_prompt_input_sha256_by_question": prompt_input_hashes,
        "by_dataset": by_dataset,
    }


def _require_exact_report_metric(
    report_values: Mapping[str, Any], observed_values: Mapping[str, Any], name: str
) -> None:
    if name not in report_values:
        raise V7FinalizationError(f"runner report is missing recomputable metric: {name}")
    left, right = report_values[name], observed_values[name]
    if isinstance(right, float):
        if left is None or not math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12):
            raise V7FinalizationError(f"runner/recomputed metric differs: {name}")
    elif left != right:
        raise V7FinalizationError(f"runner/recomputed metric differs: {name}")


def enforce_gold_free_gates(
    report: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    materialization_gates: Mapping[str, Any],
    mechanism_gates: Mapping[str, Any],
    expected_per_dataset: int = EXPECTED_PER_DATASET,
) -> None:
    """Require both runner-declared and independently observed gate success."""

    summary = report.get("safety_summary")
    if not isinstance(summary, Mapping):
        raise V7FinalizationError("runner report lacks safety_summary")
    for name, expected in materialization_gates.items():
        if name not in observed or observed[name] != expected:
            raise V7FinalizationError(f"recomputed Gold-free materialization gate failed: {name}")
        _require_exact_report_metric(summary, observed, name)
    materialization_attestation = report.get("materialization_gate")
    if (
        not isinstance(materialization_attestation, Mapping)
        or materialization_attestation.get("passed") is not True
        or materialization_attestation.get("observed") != dict(summary)
    ):
        raise V7FinalizationError("runner materialization-gate attestation is not a passing exact copy")

    report_by_dataset = report.get("by_dataset")
    if not isinstance(report_by_dataset, Mapping):
        raise V7FinalizationError("runner report lacks by_dataset")
    mechanism_metric_names = (
        "plan_executable_rate",
        "strict_subanswer_json_parse_rate",
        "mechanically_verified_subanswer_rate",
        "paired_dependent_hop_activation_rate",
        "retained_new_dependent_document_question_rate_B",
        "retained_new_dependent_document_question_rate_C",
    )
    for dataset in DATASETS:
        computed = observed["by_dataset"][dataset]
        reported = report_by_dataset.get(dataset)
        if not isinstance(reported, Mapping):
            raise V7FinalizationError(f"runner report lacks dataset metrics: {dataset}")
        if int(reported.get("n", -1)) != expected_per_dataset:
            raise V7FinalizationError(f"runner report dataset population differs: {dataset}")
        for name in mechanism_metric_names:
            _require_exact_report_metric(reported, computed, name)
        thresholds = {
            "plan_executable_rate": mechanism_gates[
                "plan_executable_rate_min_each_dataset"
            ],
            "strict_subanswer_json_parse_rate": mechanism_gates[
                "strict_subanswer_json_parse_rate_min_each_dataset"
            ],
            "mechanically_verified_subanswer_rate": mechanism_gates[
                "mechanically_verified_subanswer_rate_min_each_dataset"
            ],
            "paired_dependent_hop_activation_rate": mechanism_gates[
                "paired_dependent_hop_activation_rate_min_each_dataset"
            ],
            "retained_new_dependent_document_question_rate_B": mechanism_gates[
                "retained_new_dependent_document_question_rate_min_each_dataset_each_of_B_and_C"
            ],
            "retained_new_dependent_document_question_rate_C": mechanism_gates[
                "retained_new_dependent_document_question_rate_min_each_dataset_each_of_B_and_C"
            ],
        }
        for name, minimum in thresholds.items():
            value = computed[name]
            if value is None or float(value) < float(minimum):
                raise V7FinalizationError(
                    f"Gold-free mechanism gate failed: {dataset}::{name}"
                )
    mechanism_attestation = report.get("gold_free_mechanism_gate")
    if not isinstance(mechanism_attestation, Mapping) or mechanism_attestation.get(
        "passed"
    ) is not True:
        raise V7FinalizationError("runner mechanism-gate attestation is not passing")
    if report.get("gate_decision") != "PASS_READY_FOR_SEPARATE_GOLD_FINALIZER":
        raise V7FinalizationError("runner did not authorize the separate Gold finalizer")


def _artifact_from_report(report: Mapping[str, Any], name: str) -> Path:
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(name), Mapping):
        raise V7FinalizationError(f"runner report lacks content lock for output: {name}")
    return _assert_current_lock(outputs[name], label=f"report.outputs.{name}")


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(path)
    if protocol.get("schema_version") != v7_freeze.SCHEMA_VERSION:
        raise V7FinalizationError("unexpected v7 preregistration schema")
    if protocol.get("status") != v7_freeze.STATUS or protocol.get("scope") != v7_freeze.SCOPE:
        raise V7FinalizationError("v7 preregistration is not the frozen development cohort")
    population = protocol.get("population") or {}
    if int(population.get("n", -1)) != EXPECTED_TOTAL or population.get("by_dataset") != {
        "hotpotqa": EXPECTED_PER_DATASET,
        "musique": EXPECTED_PER_DATASET,
    }:
        raise V7FinalizationError("v7 preregistered population differs")
    if (
        population.get("globally_fresh") is not False
        or population.get("independent_confirmation") is not False
        or population.get("new_role") != "development_consumed"
    ):
        raise V7FinalizationError("v7 development/consumed scientific boundary drifted")
    gates = protocol.get("decision_gates") or {}
    if gates.get("materialization") != v7_freeze.MATERIALIZATION_GATES:
        raise V7FinalizationError("v7 materialization gates drifted")
    if gates.get("gold_free_mechanism") != v7_freeze.MECHANISM_GATES:
        raise V7FinalizationError("v7 mechanism gates drifted")
    if gates.get("development_utility") != v7_freeze.UTILITY_GATES:
        raise V7FinalizationError("v7 utility gates drifted")
    return protocol, _file_lock(path)


def _validate_addendum(
    path: Path, *, preregistration_lock: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    addendum = _read_json(path)
    if addendum.get("schema_version") != "subquestion-dependent-retrieval-v7-effective-addendum-1":
        raise V7FinalizationError("unexpected v7 truncation addendum schema")
    if addendum.get("status") != "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL":
        raise V7FinalizationError("v7 truncation addendum is not frozen")
    parent = (addendum.get("parents") or {}).get("parent_preregistration")
    if not isinstance(parent, Mapping) or not _lock_equal(parent, preregistration_lock):
        raise V7FinalizationError("truncation addendum parent preregistration differs")
    effective = addendum.get("effective_invariants") or {}
    if (
        int(effective.get("producer_passages_max", -1)) != 10
        or int(effective.get("producer_text_unicode_chars_max_each", -1)) != 1200
        or effective.get("reader_and_verifier_projection_hash_equal") is not True
    ):
        raise V7FinalizationError("truncation addendum producer-view invariants drifted")
    return addendum, _file_lock(path)


def _validate_frozen_manifest(
    path: Path,
    *,
    protocol_lock: Mapping[str, Any],
    expected_status: str,
    label: str,
) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("status") != expected_status or manifest.get("gold_access") is not False:
        raise V7FinalizationError(f"{label} status/Gold boundary differs")
    recorded = manifest.get("protocol")
    if not isinstance(recorded, Mapping):
        recorded = (manifest.get("artifacts") or {}).get("protocol")
    if not isinstance(recorded, Mapping) or not _lock_equal(recorded, protocol_lock):
        raise V7FinalizationError(f"{label} does not lock its protocol")
    return _file_lock(path)


def _validate_trajectory_addendum(
    path: Path,
    *,
    preregistration_lock: Mapping[str, Any],
    addendum_lock: Mapping[str, Any],
    enforce_canonical_hash: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the recursive-arm estimand clarification and its full ancestry."""

    trajectory = _read_json(path)
    if trajectory.get("schema_version") != EXPECTED_TRAJECTORY_ADDENDUM_SCHEMA:
        raise V7FinalizationError("unexpected v7 recursive-trajectory addendum schema")
    if trajectory.get("status") != EXPECTED_TRAJECTORY_ADDENDUM_STATUS:
        raise V7FinalizationError("v7 recursive-trajectory addendum is not frozen")
    if trajectory.get("scope") != EXPECTED_TRAJECTORY_ADDENDUM_SCOPE:
        raise V7FinalizationError("v7 recursive-trajectory addendum scope differs")
    if trajectory.get("gold_access") is not False:
        raise V7FinalizationError("v7 recursive-trajectory addendum is not Gold-free")
    if trajectory.get("effective_invariants") != EXPECTED_TRAJECTORY_INVARIANTS:
        raise V7FinalizationError("v7 recursive-trajectory invariants differ")
    parents = trajectory.get("parents")
    required = {
        "design_protocol",
        "design_manifest",
        "parent_preregistration",
        "parent_preregistration_manifest",
        "producer_truncation_addendum",
        "producer_truncation_addendum_manifest",
        "design_trajectory_addendum",
        "design_trajectory_addendum_manifest",
    }
    if not isinstance(parents, Mapping) or set(parents) != required:
        raise V7FinalizationError("recursive-trajectory addendum parent role set differs")
    for name, expected in (
        ("parent_preregistration", preregistration_lock),
        ("producer_truncation_addendum", addendum_lock),
    ):
        observed = parents.get(name)
        if not isinstance(observed, Mapping) or not _lock_equal(observed, expected):
            raise V7FinalizationError(f"recursive-trajectory parent differs: {name}")
    for name in required:
        if not isinstance(parents.get(name), Mapping):
            raise V7FinalizationError(
                f"recursive-trajectory parent is not a content lock: {name}"
            )
        _assert_current_lock(parents[name], label=f"trajectory.parents.{name}")
    prereg_manifest = _validate_frozen_manifest(
        Path(str(parents["parent_preregistration_manifest"]["path"])).resolve(),
        protocol_lock=preregistration_lock,
        expected_status=v7_freeze.STATUS,
        label="recursive-trajectory preregistration manifest",
    )
    if not _lock_equal(parents["parent_preregistration_manifest"], prereg_manifest):
        raise V7FinalizationError("recursive-trajectory preregistration manifest lock differs")
    trunc_manifest = _validate_frozen_manifest(
        Path(str(parents["producer_truncation_addendum_manifest"]["path"])).resolve(),
        protocol_lock=addendum_lock,
        expected_status="FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL",
        label="recursive-trajectory truncation manifest",
    )
    if not _lock_equal(parents["producer_truncation_addendum_manifest"], trunc_manifest):
        raise V7FinalizationError("recursive-trajectory truncation manifest lock differs")

    design_lock = parents["design_protocol"]
    design_manifest_lock = parents["design_manifest"]
    if not isinstance(design_lock, Mapping) or not isinstance(design_manifest_lock, Mapping):
        raise V7FinalizationError("recursive-trajectory design parents are not locks")
    design_path = _assert_current_lock(design_lock, label="trajectory.parents.design_protocol")
    design = _read_json(design_path)
    design_status = "RULES_AND_SELECTION_ALGORITHM_FROZEN_BEFORE_V7_GPU_OR_RETRIEVAL"
    if (
        design.get("schema_version") != "subquestion-dependent-retrieval-v7-design-freeze-1"
        or design.get("status") != design_status
    ):
        raise V7FinalizationError("recursive-trajectory design protocol differs")
    current_design_manifest = _validate_frozen_manifest(
        Path(str(design_manifest_lock.get("path") or "")).resolve(),
        protocol_lock=design_lock,
        expected_status=design_status,
        label="recursive-trajectory design manifest",
    )
    if not _lock_equal(design_manifest_lock, current_design_manifest):
        raise V7FinalizationError("recursive-trajectory design manifest lock differs")

    design_addendum_lock = parents["design_trajectory_addendum"]
    design_addendum_manifest_lock = parents["design_trajectory_addendum_manifest"]
    design_addendum_path = Path(str(design_addendum_lock["path"])).resolve()
    design_addendum = _read_json(design_addendum_path)
    if (
        design_addendum.get("schema_version")
        != "subquestion-dependent-retrieval-v7-recursive-trajectory-design-addendum-1"
        or design_addendum.get("status") != EXPECTED_TRAJECTORY_ADDENDUM_STATUS
        or design_addendum.get("gold_access") is not False
    ):
        raise V7FinalizationError("recursive-trajectory design addendum differs")
    design_addendum_manifest = _read_json(
        Path(str(design_addendum_manifest_lock["path"])).resolve()
    )
    recorded_addendum = design_addendum_manifest.get("addendum")
    if (
        design_addendum_manifest.get("status") != EXPECTED_TRAJECTORY_ADDENDUM_STATUS
        or design_addendum_manifest.get("gold_access") is not False
        or not isinstance(recorded_addendum, Mapping)
        or str(recorded_addendum.get("sha256") or "")
        != str(design_addendum_lock.get("sha256") or "")
    ):
        raise V7FinalizationError("recursive-trajectory design addendum manifest differs")

    trajectory_lock = _file_lock(path)
    if enforce_canonical_hash and trajectory_lock["sha256"] != EXPECTED_TRAJECTORY_ADDENDUM_SHA256:
        raise V7FinalizationError("recursive-trajectory addendum bytes differ from freeze")
    manifest_path = path.with_name("manifest.json")
    trajectory_manifest_lock = _validate_frozen_manifest(
        manifest_path,
        protocol_lock=trajectory_lock,
        expected_status=EXPECTED_TRAJECTORY_ADDENDUM_STATUS,
        label="recursive-trajectory addendum manifest",
    )
    if enforce_canonical_hash and (
        trajectory_manifest_lock["sha256"]
        != EXPECTED_TRAJECTORY_ADDENDUM_MANIFEST_SHA256
    ):
        raise V7FinalizationError("recursive-trajectory addendum manifest bytes differ")
    return trajectory, trajectory_lock, trajectory_manifest_lock


def _validate_implementation_lock(
    path: Path,
    *,
    preregistration_lock: Mapping[str, Any],
    addendum_lock: Mapping[str, Any],
    trajectory_addendum_lock: Mapping[str, Any],
    trajectory_addendum_manifest_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _read_json(path)
    if lock.get("schema_version") != EXPECTED_IMPLEMENTATION_SCHEMA:
        raise V7FinalizationError("unexpected v7 implementation-lock schema")
    if lock.get("status") != EXPECTED_IMPLEMENTATION_STATUS:
        raise V7FinalizationError("v7 implementation is not frozen before execution")
    if lock.get("experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS["implementation_lock"]:
        raise V7FinalizationError("v7 implementation-lock Experiment ID differs")
    if lock.get("gold_access") is not False:
        raise V7FinalizationError("implementation lock is not Gold-free")
    authorization = lock.get("authorization") or {}
    if authorization != {
        "planner": True,
        "gold_free_materialization": False,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }:
        raise V7FinalizationError("implementation lock grants an unexpected authorization")
    if (lock.get("content_reverification") or {}).get(
        "full_hash_verification_performed"
    ) is not True:
        raise V7FinalizationError("implementation lock did not fully re-hash model/Wiki18 content")
    issuer = lock.get("lock_issuer")
    if not isinstance(issuer, Mapping):
        raise V7FinalizationError("implementation lock lacks its CPU lock-issuer hash")
    issuer_path = _assert_current_lock(issuer, label="implementation.lock_issuer")
    expected_issuer = Path(
        "scripts/prepare/freeze_dependent_retrieval_v7_implementation.py"
    ).resolve()
    if issuer_path != expected_issuer:
        raise V7FinalizationError("implementation lock issuer path differs")
    parents = lock.get("parents") or {}
    if not isinstance(parents, Mapping):
        raise V7FinalizationError("implementation lock lacks parents")
    for name, expected in (
        ("preregistration", preregistration_lock),
        ("truncation_addendum", addendum_lock),
        ("trajectory_semantics_addendum", trajectory_addendum_lock),
        (
            "trajectory_semantics_addendum_manifest",
            trajectory_addendum_manifest_lock,
        ),
    ):
        observed = parents.get(name)
        if not isinstance(observed, Mapping) or not _lock_equal(observed, expected):
            raise V7FinalizationError(f"implementation-lock parent differs: {name}")
    files = lock.get("runtime_code")
    if not isinstance(files, Mapping) or set(files) != REQUIRED_IMPLEMENTATION_FILES:
        raise V7FinalizationError("implementation lock lacks required runtime files")
    for name, file_lock in files.items():
        _assert_current_lock(file_lock, label=f"implementation.runtime_code.{name}")
    closure = lock.get("actual_local_import_closure")
    if not isinstance(closure, Mapping) or not closure:
        raise V7FinalizationError("implementation lock lacks its local import closure")
    for name, file_lock in closure.items():
        _assert_current_lock(file_lock, label=f"implementation.imports.{name}")
    return lock, _file_lock(path)


def _validate_plan_lock(
    path: Path,
    *,
    preregistration_lock: Mapping[str, Any],
    addendum_lock: Mapping[str, Any],
    implementation: Mapping[str, Any],
    implementation_lock: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    trajectory_addendum_lock: Mapping[str, Any],
    trajectory_addendum_manifest_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the only artifact authorized to start materialization."""

    lock = _read_json(path)
    if lock.get("schema_version") != EXPECTED_PLAN_LOCK_SCHEMA:
        raise V7FinalizationError("unexpected v7 post-plan lock schema")
    if lock.get("status") != EXPECTED_PLAN_LOCK_STATUS:
        raise V7FinalizationError("v7 post-plan lock did not authorize materialization")
    if lock.get("experiment_id") != EXPECTED_PLAN_LOCK_EXPERIMENT_ID:
        raise V7FinalizationError("v7 post-plan lock Experiment ID differs")
    if lock.get("scope") != v7_freeze.SCOPE or lock.get("gold_access") is not False:
        raise V7FinalizationError("v7 post-plan lock scope/Gold boundary differs")
    parents = lock.get("parents") or {}
    expected_parents = {
        "preregistration": preregistration_lock,
        "truncation_addendum": addendum_lock,
        "trajectory_semantics_addendum": trajectory_addendum_lock,
        "trajectory_semantics_addendum_manifest": trajectory_addendum_manifest_lock,
        "implementation_lock": implementation_lock,
    }
    for name, expected in expected_parents.items():
        observed = parents.get(name)
        if not isinstance(observed, Mapping) or not _lock_equal(observed, expected):
            raise V7FinalizationError(f"post-plan lock parent differs: {name}")
    implementation_manifest = parents.get("implementation_manifest")
    if not isinstance(implementation_manifest, Mapping):
        raise V7FinalizationError("post-plan lock lacks implementation-manifest parent")
    _assert_current_lock(
        implementation_manifest, label="plan_lock.parents.implementation_manifest"
    )
    issuer = lock.get("lock_issuer")
    if not isinstance(issuer, Mapping):
        raise V7FinalizationError("post-plan lock lacks its CPU lock-issuer hash")
    issuer_path = _assert_current_lock(issuer, label="plan_lock.lock_issuer")
    expected_issuer = Path(
        "scripts/prepare/freeze_dependent_retrieval_v7_plans.py"
    ).resolve()
    if issuer_path != expected_issuer:
        raise V7FinalizationError("post-plan lock issuer path differs")

    inputs = lock.get("inputs") or {}
    expected_input_names = {
        "development",
        "planner_cohort",
        "canonical_A_contexts",
        "planner_predictions",
        "planner_report",
        "planner_manifest",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_input_names:
        raise V7FinalizationError("post-plan lock input role set differs")
    for name, file_lock in inputs.items():
        _assert_current_lock(file_lock, label=f"plan_lock.inputs.{name}")

    runtime_code = lock.get("runtime_code")
    if not isinstance(runtime_code, Mapping) or dict(runtime_code) != dict(
        implementation.get("runtime_code") or {}
    ):
        raise V7FinalizationError("post-plan runtime code differs from implementation lock")
    if set(runtime_code) != REQUIRED_IMPLEMENTATION_FILES:
        raise V7FinalizationError("post-plan runtime-code role set differs")
    for name, file_lock in runtime_code.items():
        _assert_current_lock(file_lock, label=f"plan_lock.runtime_code.{name}")

    population = lock.get("population") or {}
    if (
        int(population.get("n", -1)) != EXPECTED_TOTAL
        or population.get("by_dataset") != {
            "hotpotqa": EXPECTED_PER_DATASET,
            "musique": EXPECTED_PER_DATASET,
        }
        or population.get("plan_executable_gate_pass") is not True
        or population.get("question_key_order_sha256")
        != (preregistration.get("population") or {}).get(
            "question_key_order_sha256"
        )
    ):
        raise V7FinalizationError("post-plan population/executable gate differs")
    materialization = lock.get("materialization_contract") or {}
    if (
        materialization.get("experiment_id")
        != v7_freeze.FUTURE_EXPERIMENT_IDS["materialization"]
        or int(materialization.get("n", -1)) != EXPECTED_TOTAL
        or materialization.get("by_dataset")
        != {"hotpotqa": EXPECTED_PER_DATASET, "musique": EXPECTED_PER_DATASET}
        or materialization.get("gold_access") is not False
        or materialization.get("network_access") is not False
        or int(materialization.get("max_plan_steps", -1)) != 4
        or not str(materialization.get("runner_version") or "")
    ):
        raise V7FinalizationError("post-plan materialization contract differs")
    if lock.get("authorization") != {
        "planner_complete": True,
        "gold_free_materialization": True,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }:
        raise V7FinalizationError("post-plan authorization differs")
    return lock, _file_lock(path)


def validate_gold_free_materialization(
    *,
    report_path: Path,
    preregistration_path: Path,
    truncation_addendum_path: Path,
    trajectory_semantics_addendum_path: Path,
    implementation_lock_path: Path,
    plan_lock_path: Path,
    enforce_canonical_trajectory_hash: bool = True,
) -> dict[str, Any]:
    """Validate every hash/safety/mechanism condition without opening Gold."""

    preregistration, prereg_lock = _validate_preregistration(preregistration_path.resolve())
    addendum, addendum_lock = _validate_addendum(
        truncation_addendum_path.resolve(), preregistration_lock=prereg_lock
    )
    (
        trajectory_addendum,
        trajectory_addendum_lock,
        trajectory_addendum_manifest_lock,
    ) = _validate_trajectory_addendum(
        trajectory_semantics_addendum_path.resolve(),
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
        enforce_canonical_hash=enforce_canonical_trajectory_hash,
    )
    implementation, implementation_lock = _validate_implementation_lock(
        implementation_lock_path.resolve(),
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
        trajectory_addendum_lock=trajectory_addendum_lock,
        trajectory_addendum_manifest_lock=trajectory_addendum_manifest_lock,
    )
    plan_lock, plan_lock_file = _validate_plan_lock(
        plan_lock_path.resolve(),
        preregistration_lock=prereg_lock,
        addendum_lock=addendum_lock,
        implementation=implementation,
        implementation_lock=implementation_lock,
        preregistration=preregistration,
        trajectory_addendum_lock=trajectory_addendum_lock,
        trajectory_addendum_manifest_lock=trajectory_addendum_manifest_lock,
    )
    report = _read_json(report_path.resolve())
    _assert_gold_free(report, location="retrieval_report")
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA or report.get(
        "status"
    ) != EXPECTED_REPORT_STATUS:
        raise V7FinalizationError("retrieval report is incomplete or has an unexpected schema")
    if report.get("development_only") is not True or report.get("gold_access") is not False:
        raise V7FinalizationError("retrieval report violates the development/Gold boundary")
    if report.get("experiment_id") != v7_freeze.FUTURE_EXPERIMENT_IDS["materialization"]:
        raise V7FinalizationError("materialization Experiment ID differs from preregistration")
    if report.get("runner_version") != plan_lock["materialization_contract"][
        "runner_version"
    ]:
        raise V7FinalizationError("materialization runner version differs from post-plan lock")
    report_refs = (
        ("preregistration", prereg_lock),
        ("truncation_addendum", addendum_lock),
        ("trajectory_semantics_addendum", trajectory_addendum_lock),
        ("implementation_lock", implementation_lock),
        ("plan_lock", plan_lock_file),
    )
    for name, expected in report_refs:
        observed = report.get(name)
        if not isinstance(observed, Mapping) or not _lock_equal(observed, expected):
            raise V7FinalizationError(f"retrieval report frozen reference differs: {name}")

    paths = {name: _artifact_from_report(report, name) for name in REPORT_OUTPUTS}
    arm_a = _read_jsonl(paths["arm_a"])
    arm_b = _read_jsonl(paths["arm_b"])
    arm_c = _read_jsonl(paths["arm_c"])
    details = _read_jsonl(paths["execution_details"])
    budget = _read_jsonl(paths["budget_ledger"])
    observed = audit_materialization(arm_a, arm_b, arm_c, details, budget)
    observed["subanswer_model_identity"] = _validate_subanswer_model_artifacts(
        details,
        preregistration=preregistration,
    )
    population = preregistration["population"]
    if observed["question_key_order_sha256"] != population["question_key_order_sha256"]:
        raise V7FinalizationError("materialized question order differs from preregistration")
    enforce_gold_free_gates(
        report,
        observed,
        materialization_gates=preregistration["decision_gates"]["materialization"],
        mechanism_gates=preregistration["decision_gates"]["gold_free_mechanism"],
    )
    evaluator_import_closure = _evaluator_import_closure_locks()
    for role, relative in (
        ("gold_finalizer", "scripts/prepare/finalize_paired_dependent_retrieval_v7.py"),
        ("evaluator", "scripts/eval/evaluate_paired_dependent_retrieval_v7.py"),
    ):
        runtime_lock = (implementation.get("runtime_code") or {}).get(role)
        if not isinstance(runtime_lock, Mapping) or not _lock_equal(
            evaluator_import_closure[relative], runtime_lock
        ):
            raise V7FinalizationError(
                f"evaluation closure {role} differs from implementation lock"
            )
    return {
        "report": report,
        "report_lock": _file_lock(report_path),
        "preregistration": preregistration,
        "preregistration_lock": prereg_lock,
        "truncation_addendum": addendum,
        "truncation_addendum_lock": addendum_lock,
        "trajectory_semantics_addendum": trajectory_addendum,
        "trajectory_semantics_addendum_lock": trajectory_addendum_lock,
        "trajectory_semantics_addendum_manifest_lock": trajectory_addendum_manifest_lock,
        "implementation": implementation,
        "implementation_lock": implementation_lock,
        "plan_lock": plan_lock,
        "plan_lock_file": plan_lock_file,
        "paths": paths,
        "arms": {"A": arm_a, "B": arm_b, "C": arm_c},
        "observed": observed,
        "evaluator_import_closure": evaluator_import_closure,
    }


def _index_raw_gold(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Open scorer Gold.  This function must only be called below GOLD BOUNDARY."""

    indexed: dict[str, dict[str, Any]] = {}
    locks: dict[str, Any] = {}
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        dataset = path.parent.name.casefold()
        if dataset not in DATASETS:
            raise V7FinalizationError(f"cannot infer allowed dataset from Gold path: {path}")
        locks[dataset] = _file_lock(path)
        for row in _read_jsonl(path):
            qid = str(row.get("id") or row.get("qid") or "").strip()
            key = question_key(dataset, qid)
            if key in indexed:
                raise V7FinalizationError(f"duplicate scorer Gold key: {key}")
            answers = row.get("golden_answers")
            if not isinstance(answers, list):
                raise V7FinalizationError(
                    f"raw scorer row lacks the frozen golden_answers list: {key}"
                )
            golds = [str(value).strip() for value in answers if str(value).strip()]
            question = str(row.get("question") or "")
            if not qid or not question.strip() or not golds:
                raise V7FinalizationError(f"invalid scorer Gold row: {key}")
            indexed[key] = {"question": question, "gold_answers": golds}
    return indexed, locks


def attach_scorer_gold(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Join one immutable Gold projection to all three already-validated arms."""

    if set(arms) != {"A", "B", "C"}:
        raise V7FinalizationError("Gold attachment requires exactly A/B/C arms")
    lengths = {arm: len(rows) for arm, rows in arms.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values()), 0) <= 0:
        raise V7FinalizationError(f"strict A/B/C population broken: {lengths}")
    result = {arm: [] for arm in arms}
    for index, triple in enumerate(zip(arms["A"], arms["B"], arms["C"])):
        rows = [dict(row) for row in triple]
        if _common_projection(rows[0]) != _common_projection(rows[1]) or _common_projection(
            rows[0]
        ) != _common_projection(rows[2]):
            raise V7FinalizationError(f"A/B/C common identity differs at Gold join row {index}")
        key = str(rows[0]["question_key"])
        if key not in gold_index:
            raise V7FinalizationError(f"scorer Gold identity join is missing {key}")
        gold = gold_index[key]
        if str(rows[0]["question"]).strip() != str(gold.get("question") or "").strip():
            raise V7FinalizationError(f"scorer Gold question mismatch for {key}")
        for arm, row in zip(("A", "B", "C"), rows):
            if any(name in row for name in ("answer", "gold_answers", "golden_answers")):
                raise V7FinalizationError(f"upstream {arm} arm already contains Gold for {key}")
            row["gold_answers"] = list(gold["gold_answers"])
            row["gold_attachment"] = "SCORER_ONLY_AFTER_ALL_GOLD_FREE_GATES"
            result[arm].append(row)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_report", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/protocol.json"
        ),
    )
    parser.add_argument(
        "--truncation_addendum",
        type=Path,
        default=Path(
            "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration_addendum_producer_truncation_v1/protocol.json"
        ),
    )
    parser.add_argument(
        "--trajectory_semantics_addendum",
        type=Path,
        default=DEFAULT_TRAJECTORY_ADDENDUM,
    )
    parser.add_argument("--implementation_lock", type=Path, required=True)
    parser.add_argument(
        "--plan_lock",
        type=Path,
        default=Path(
            "outputs/audits/subquestion_dependent_retrieval_v7_development_plans_lock_v1/protocol.json"
        ),
    )
    parser.add_argument("--hotpot_dev", type=Path, default=Path("data/hotpotqa/dev.jsonl"))
    parser.add_argument("--musique_dev", type=Path, default=Path("data/musique/dev.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--evaluation_experiment_id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, reserved_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "paired_dependent_retrieval_v7_gold_attachment",
            "finalizer_version": FINALIZER_VERSION,
            "gold_access_during_materialization": False,
        },
    )
    try:
        expected_gold_id = v7_freeze.FUTURE_EXPERIMENT_IDS["gold_attachment"]
        expected_eval_id = v7_freeze.FUTURE_EXPERIMENT_IDS["evaluation"]
        if args.experiment_id != expected_gold_id or args.evaluation_experiment_id != expected_eval_id:
            raise V7FinalizationError("Gold-attachment/evaluation Experiment IDs differ from preregistration")

        # Every function above this boundary is Gold-free.  In particular no
        # stat/hash/existence check has touched either raw dev path.
        bundle = validate_gold_free_materialization(
            report_path=args.retrieval_report,
            preregistration_path=args.preregistration,
            truncation_addendum_path=args.truncation_addendum,
            trajectory_semantics_addendum_path=args.trajectory_semantics_addendum,
            implementation_lock_path=args.implementation_lock,
            plan_lock_path=args.plan_lock,
        )

        # --------------------------- GOLD BOUNDARY ---------------------------
        gold_index, scorer_gold_locks = _index_raw_gold(
            (args.hotpot_dev, args.musique_dev)
        )
        scored = attach_scorer_gold(bundle["arms"], gold_index)

        scored_paths = {
            "A": run_dir / "arm_a.scored.jsonl",
            "B": run_dir / "arm_b.scored.jsonl",
            "C": run_dir / "arm_c.scored.jsonl",
        }
        for arm in ("A", "B", "C"):
            _write_jsonl_exclusive(scored_paths[arm], scored[arm])

        preregistration = bundle["preregistration"]
        eval_protocol = {
            "schema_version": EVAL_PROTOCOL_SCHEMA,
            "status": EVAL_PROTOCOL_STATUS,
            "experiment_id": args.evaluation_experiment_id,
            "gold_attachment_experiment_id": reserved_id,
            "materialization_experiment_id": bundle["report"]["experiment_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": v7_freeze.SCOPE,
            "development_only": True,
            "globally_fresh": False,
            "independent_confirmation": False,
            "gold_access_during_planning_retrieval_subanswer_and_merge": False,
            "gold_access_after_all_gold_free_gates": True,
            "n": EXPECTED_TOTAL,
            "by_dataset": {dataset: EXPECTED_PER_DATASET for dataset in DATASETS},
            "qid_order_sha256": bundle["observed"]["qid_order_sha256"],
            "question_key_order_sha256": bundle["observed"]["question_key_order_sha256"],
            "arms": list(ARMS),
            "inputs": {
                "arm_a": _file_lock(scored_paths["A"]),
                "arm_b": _file_lock(scored_paths["B"]),
                "arm_c": _file_lock(scored_paths["C"]),
                "retrieval_report": bundle["report_lock"],
                "retrieval_arm_a_no_gold": _file_lock(bundle["paths"]["arm_a"]),
                "retrieval_arm_b_no_gold": _file_lock(bundle["paths"]["arm_b"]),
                "retrieval_arm_c_no_gold": _file_lock(bundle["paths"]["arm_c"]),
                "execution_details_no_gold": _file_lock(bundle["paths"]["execution_details"]),
                "budget_ledger_no_gold": _file_lock(bundle["paths"]["budget_ledger"]),
                "preregistration": bundle["preregistration_lock"],
                "truncation_addendum": bundle["truncation_addendum_lock"],
                "trajectory_semantics_addendum": bundle[
                    "trajectory_semantics_addendum_lock"
                ],
                "trajectory_semantics_addendum_manifest": bundle[
                    "trajectory_semantics_addendum_manifest_lock"
                ],
                "implementation_lock": bundle["implementation_lock"],
                "plan_lock": bundle["plan_lock_file"],
                "scorer_gold": scorer_gold_locks,
            },
            "models": {
                "strong_sft": preregistration["models"]["subanswer_and_final_strong_sft"],
                "base_model": preregistration["models"]["base_model"],
                "content_locks": {
                    "strong_sft": preregistration["models"]["inherited_content_locks"]["strong_sft"],
                    "base_model": preregistration["models"]["inherited_content_locks"]["base_model"],
                },
            },
            "generation": {
                "prompt": "canonical legacy build_rl_messages",
                "kg_subgraph": [],
                "top_k_passages": 10,
                "decode": "greedy",
                "do_sample": False,
                "max_new_tokens": 512,
                "seed": 42,
                "identical_prompt_reuse": "byte-exact generation and parsed score reuse within question",
                "single_model_load_for_all_three_arms": True,
            },
            "estimand": {
                "analysis_population": "ITT_ALL_40_WITH_FALLBACKS_INCLUDED",
                "primary": "C_verified_subanswer minus B_entity_hint_top1",
                "secondary": "C_verified_subanswer minus A_canonical_one_shot",
                "subgroup_selection": False,
            },
            "decision_gates": deepcopy(preregistration["decision_gates"]),
            "gold_free_materialization_observed": bundle["observed"],
            "code": {
                "gold_finalizer": _file_lock(Path(__file__)),
                "evaluator": bundle["plan_lock"]["runtime_code"]["evaluator"],
                "evaluator_import_closure": bundle["evaluator_import_closure"],
                "required_evaluator_dependencies": sorted(
                    REQUIRED_EVALUATOR_DEPENDENCIES
                ),
            },
            "authorization": {
                "schema_version": EVAL_AUTHORIZATION_SCHEMA,
                "status": "AUTHORIZED_FOR_FROZEN_V7_ANSWER_EVALUATION_ONLY",
                "issuer": "paired-dependent-retrieval-v7-gold-finalizer",
                "issuer_code": _file_lock(Path(__file__)),
                "gold_attachment_complete": True,
                "answer_evaluation": True,
                "training": False,
                "evaluation_experiment_id": args.evaluation_experiment_id,
                "gold_attachment_experiment_id": reserved_id,
                "parent_chain": {
                    "preregistration": bundle["preregistration_lock"],
                    "truncation_addendum": bundle["truncation_addendum_lock"],
                    "trajectory_semantics_addendum": bundle[
                        "trajectory_semantics_addendum_lock"
                    ],
                    "trajectory_semantics_addendum_manifest": bundle[
                        "trajectory_semantics_addendum_manifest_lock"
                    ],
                    "implementation_lock": bundle["implementation_lock"],
                    "plan_lock": bundle["plan_lock_file"],
                    "materialization_report": bundle["report_lock"],
                },
            },
            "scientific_boundary": (
                "Globally consumed development rows only. A pass is feasibility evidence, "
                "not independent confirmation; a genuinely unseen split/test is still required."
            ),
        }
        protocol_path = run_dir / "protocol.json"
        _write_json_exclusive(protocol_path, eval_protocol)
        dump_manifest(
            run_dir,
            status="FROZEN_READY_FOR_V7_ANSWER_EVALUATION",
            extra={
                "experiment_id": reserved_id,
                "phase": "paired_dependent_retrieval_v7_gold_attachment",
                "protocol_sha256": v7_freeze.sha256_file(protocol_path),
                "gold_opened_only_after_gold_free_gates": True,
                "evaluation_experiment_id": args.evaluation_experiment_id,
                "evaluation_authorization_schema": EVAL_AUTHORIZATION_SCHEMA,
                "answer_evaluation_authorized": True,
            },
        )
        print(
            json.dumps(
                {
                    "status": "FROZEN_READY_FOR_V7_ANSWER_EVALUATION",
                    "protocol": str(protocol_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except BaseException as exc:
        dump_manifest(
            run_dir,
            status="ABORTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_RUNTIME_BEFORE_OR_DURING_GOLD_ATTACHMENT",
            extra={
                "experiment_id": reserved_id,
                "phase": "paired_dependent_retrieval_v7_gold_attachment",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise


if __name__ == "__main__":
    main()
