#!/usr/bin/env python
"""Stage the Gold-free paired dependent-retrieval v7 materialisation.

The runner is deliberately split at the GPU ownership boundary::

    roots -> external C reader depth 1 -> dependents depth 1 -> ... -> final

``--depth D`` denotes the *producer* depth.  It consumes the immutable
``c_answers.depth_D.jsonl`` artifact and executes plan steps at dependency
depth ``D + 1``.  Every stage writes a new cumulative state snapshot; no stage
overwrites an earlier artifact.

Only B's bridge value differs from C's.  B uses one deterministic passage
entity and C uses one mechanically verified extractive subanswer.  A dependent
logical hop is issued for both arms, or for neither arm.  The two arm-tagged
dependent searches are never deduplicated across arms, even when their query
strings are identical.

This module does not load a language model and never reads Gold.  The separate
reader process consumes the exact C-task contract frozen below.  Gold may be
attached only after the complete materialisation has passed its own gates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from kgproweight.retrieval.dependent import (
    dependency_refs,
    extract_deterministic_bridge_candidates,
    render_root_query,
    validate_plan_for_dependent_retrieval,
)
from kgproweight.retrieval.dependent_merge_v6 import (
    POLICY_VERSION as MERGE_POLICY_VERSION,
    merge_dependent_passages_v6,
    passage_score_key,
)
from kgproweight.retrieval.dependent_v6 import (
    QUERY_RENDERER_VERSION,
    render_question_anchored_queries_v6,
)
from kgproweight.retrieval.subanswer_v7 import (
    PARSER_VERSION as SUBANSWER_PARSER_VERSION,
    VERIFIER_VERSION as SUBANSWER_VERIFIER_VERSION,
    build_subanswer_reader_messages,
    parse_and_verify_subanswer,
)


RUNNER_VERSION = "paired-dependent-retrieval-v7-staged-gold-free-1"
ROOT_STATE_SCHEMA_VERSION = "paired-dependent-retrieval-v7-root-state-1"
STATE_SCHEMA_VERSION = "paired-dependent-retrieval-v7-state-1"
BUDGET_SCHEMA_VERSION = "paired-dependent-retrieval-v7-budget-1"
ARM_SCHEMA_VERSION = "paired-dependent-retrieval-v7-arm-1"
REPORT_SCHEMA_VERSION = "paired-dependent-retrieval-v7-report-1"
STAGE_DESCRIPTOR_SCHEMA_VERSION = "paired-dependent-retrieval-v7-stage-descriptor-2"
MODEL_ARTIFACT_SCHEMA = "dependent-retrieval-v7-strong-sft-model-content-lock-1"

DESIGN_PROTOCOL_SCHEMA = "subquestion-dependent-retrieval-v7-design-freeze-1"
DESIGN_PROTOCOL_STATUS = "RULES_AND_SELECTION_ALGORITHM_FROZEN_BEFORE_V7_GPU_OR_RETRIEVAL"
DEFAULT_DESIGN_PROTOCOL = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/protocol.json"
)
DEFAULT_PREREGISTRATION = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/protocol.json"
)
DEFAULT_TRUNCATION_ADDENDUM = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration_addendum_producer_truncation_v1/protocol.json"
)
DEFAULT_TRAJECTORY_SEMANTICS_ADDENDUM = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration_addendum_recursive_trajectory_v1/protocol.json"
)
DEFAULT_IMPLEMENTATION_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_implementation_lock_v1/protocol.json"
)
DEFAULT_EXECUTION_LOCK = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_plans_lock_v1/protocol.json"
)
PREREGISTRATION_SCHEMA = "subquestion-dependent-retrieval-v7-preregistration-1"
TRUNCATION_ADDENDUM_SCHEMA = "subquestion-dependent-retrieval-v7-effective-addendum-1"
TRAJECTORY_SEMANTICS_ADDENDUM_SCHEMA = (
    "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
)
TRAJECTORY_SEMANTICS_ADDENDUM_STATUS = "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
IMPLEMENTATION_LOCK_SCHEMA = "subquestion-dependent-retrieval-v7-implementation-lock-1"
IMPLEMENTATION_LOCK_STATUS = "AUTHORIZED_PLANNER_ONLY"
EXECUTION_LOCK_SCHEMA = "subquestion-dependent-retrieval-v7-plan-lock-1"
EXECUTION_LOCK_STATUS = "AUTHORIZED_GOLD_FREE_MATERIALIZATION"
EXECUTION_LOCK_EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-PLAN-LOCK-V1"
)
EXECUTION_SCOPE = "DEVELOPMENT_ONLY_GLOBALLY_CONSUMED_HOTPOT20_MUSIQUE20"
EXPECTED_RUNTIME_CODE_ROLES = frozenset(
    {
        "retrieval_runner",
        "subanswer_generator",
        "gold_finalizer",
        "evaluator",
    }
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_LOCK_ISSUER = PROJECT_ROOT / "scripts/prepare/freeze_dependent_retrieval_v7_plans.py"

DATASETS = ("hotpotqa", "musique")
TARGET_TYPES = {
    "hotpotqa": "relation_graph",
    "musique": "subquery_graph",
}
MAX_PLAN_STEPS = 4
TOTAL_PASSAGES = 10
PROTECTED_A_PREFIX = 8
CANDIDATES_PER_DEPENDENT_QUERY = 2
STEP_RERANK_TOPK = 10
BRIDGE_MAX_DOCS = 10
BRIDGE_MAX_BODY_CHARS = 1200
CE_MAX_CHARS = 1200
MAX_ANSWER_CHARS = 256

# This is copied from the frozen protocol rather than imported from a mutable
# finalizer.  The keys are checked recursively in cohort/context/plan inputs.
FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "decomposition",
        "evidence",
        "evidences",
        "gold_answer",
        "gold_answers",
        "gold_target",
        "golden_answers",
        "paragraph_text",
        "question_decomposition",
        "reasoning",
        "sp",
        "supporting_facts",
        "target",
    }
)

# The reader task is intentionally an exact whitelist.  In particular it does
# not carry the source cohort row, passage graph, Wikidata data, or any legacy
# arm metadata.
C_TASK_KEYS = frozenset(
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

# A generator may put model/prompt/token telemetry inside ``telemetry`` only.
# Keeping top-level identity exact makes cross-depth and cross-question swaps a
# whole-run integrity error rather than an ordinary abstention.
C_ANSWER_KEYS = frozenset(
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

_TASK_ANSWER_IDENTITY_FIELDS = (
    "task_id",
    "question_key",
    "dataset",
    "qid",
    "question_sha256",
    "target_type",
    "producer_slot",
    "step_sha256",
    "producer_passages_sha256",
)


class V7IntegrityError(RuntimeError):
    """A whole-run integrity or frozen-contract failure."""


@dataclass(frozen=True)
class StageResult:
    """In-memory result returned before append-only files are committed."""

    states: list[dict[str, Any]]
    c_tasks: list[dict[str, Any]]
    budget_rows: list[dict[str, Any]]
    arm_a_rows: list[dict[str, Any]] | None = None
    arm_b_rows: list[dict[str, Any]] | None = None
    arm_c_rows: list[dict[str, Any]] | None = None
    execution_rows: list[dict[str, Any]] | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_historical_json(value: Any) -> str:
    """Hash canonical-A passages with the repository's historical encoding.

    Frozen QPEG/SAEG context files used ``json.dumps``' default separators.
    C step/passage commitments intentionally use :func:`_sha256_json` instead;
    the two encodings must not be silently conflated.
    """

    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return _sha256_text(blob)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_lock(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or (resolved.stat().st_size <= 0 and not allow_empty):
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _assert_file_lock(
    expected: Mapping[str, Any], path: Path, *, label: str
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise V7IntegrityError(f"missing frozen file lock: {label}")
    current = _file_lock(
        path, allow_empty=int(expected.get("size_bytes", -1)) == 0
    )
    expected_path = Path(str(expected.get("path") or "")).expanduser().resolve()
    if expected_path != Path(current["path"]):
        raise V7IntegrityError(f"{label} path differs from execution lock")
    if int(expected.get("size_bytes", -1)) != current["size_bytes"]:
        raise V7IntegrityError(f"{label} size differs from execution lock")
    if str(expected.get("sha256") or "") != current["sha256"]:
        raise V7IntegrityError(f"{label} hash differs from execution lock")
    return current


def _expected_subanswer_model_artifact(
    implementation: Mapping[str, Any],
    *,
    implementation_lock: Mapping[str, Any],
    plan_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact reader-model identity already frozen before execution."""

    verified = (implementation.get("content_reverification") or {}).get("verified")
    models = verified.get("models") if isinstance(verified, Mapping) else None
    if not isinstance(models, Mapping):
        raise V7IntegrityError("implementation lock lacks verified model content")
    base_model = models.get("base_model")
    strong_sft = models.get("strong_sft")
    if not isinstance(base_model, Mapping) or not isinstance(strong_sft, Mapping):
        raise V7IntegrityError("implementation lock lacks base/strong-SFT content locks")
    for label, lock in (("base_model", base_model), ("strong_sft", strong_sft)):
        if (
            not str(lock.get("path") or "")
            or not isinstance(lock.get("tree_sha256"), str)
            or len(str(lock["tree_sha256"])) != 64
            or not isinstance(lock.get("files"), list)
        ):
            raise V7IntegrityError(f"malformed frozen model content lock: {label}")
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "base_model": dict(base_model),
        "strong_sft_adapter": dict(strong_sft),
        "load_contract": {
            "torch_dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "separate_process_required": True,
        },
        "authorization_locks": {
            "implementation_lock": dict(implementation_lock),
            "plan_lock": dict(plan_lock),
        },
    }


def _question_key(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset") or "").strip()
    qid = str(row.get("qid") or "").strip()
    if dataset not in TARGET_TYPES or not qid:
        raise V7IntegrityError(f"invalid question identity: dataset={dataset!r}, qid={qid!r}")
    expected = f"{dataset}::{qid}"
    supplied = row.get("question_key")
    if supplied is not None and str(supplied) != expected:
        raise V7IntegrityError(
            f"question_key mismatch: expected {expected!r}, observed {supplied!r}"
        )
    return expected


def _find_forbidden_fields(value: Any, *, location: str = "input") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_location = f"{location}.{key}"
            if key.casefold() in FORBIDDEN_GOLD_KEYS:
                found.append(child_location)
            if key.casefold() == "gold_access" and child is not False:
                found.append(f"{child_location}=not_false")
            found.extend(_find_forbidden_fields(child, location=child_location))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found.extend(
                _find_forbidden_fields(child, location=f"{location}[{index}]")
            )
    return found


def assert_gold_free_source(value: Any, *, label: str) -> None:
    """Reject recursively forbidden source fields before any projection."""

    found = _find_forbidden_fields(value, location=label)
    if found:
        preview = ", ".join(found[:5])
        raise V7IntegrityError(f"Gold/prohibited field observed in {label}: {preview}")


def _as_rows(value: Iterable[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, Mapping)):
        raise V7IntegrityError(f"{label} must be an iterable of rows")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            raise V7IntegrityError(f"{label} row {index} is not an object")
        copied = deepcopy(dict(row))
        assert_gold_free_source(copied, label=f"{label}[{index}]")
        rows.append(copied)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    result: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V7IntegrityError(
                    f"invalid JSONL at {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise V7IntegrityError(
                    f"JSONL row is not an object at {resolved}:{line_number}"
                )
            result.append(value)
    return result


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Commit a new JSONL artifact without overwriting an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        key = _question_key(row)
        if key in result:
            raise V7IntegrityError(f"duplicate {label} identity: {key}")
        result[key] = deepcopy(dict(row))
    return result


def _index_selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    selected_keys: set[str],
) -> dict[str, dict[str, Any]]:
    """Index only frozen cohort identities from a larger Gold-free asset."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_dataset = str(row.get("dataset") or "").strip()
        raw_qid = str(row.get("qid") or "").strip()
        candidate = f"{raw_dataset}::{raw_qid}"
        if candidate not in selected_keys:
            continue
        key = _question_key(row)
        if key in result:
            raise V7IntegrityError(f"duplicate {label} identity: {key}")
        result[key] = deepcopy(dict(row))
    return result


def _passage_text(passage: Mapping[str, Any]) -> str:
    value = passage.get("contents")
    if not isinstance(value, str) or not value.strip():
        value = passage.get("text")
    if not isinstance(value, str) or not value.strip():
        raise V7IntegrityError("passage has no non-empty contents/text")
    return value


def _document_id(passage: Mapping[str, Any], *, location: str) -> str:
    values: list[str] = []
    for key in ("id", "doc_id", "document_id"):
        if passage.get(key) is None:
            continue
        raw = passage[key]
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise V7IntegrityError(f"{location}.{key} has invalid document id type")
        value = str(raw)
        if not value or value != value.strip():
            raise V7IntegrityError(f"{location}.{key} is empty or padded")
        values.append(value)
    if not values:
        raise V7IntegrityError(f"{location} has no stable document id")
    if len(set(values)) != 1:
        raise V7IntegrityError(f"{location} has conflicting document ids")
    return values[0]


def _passage_ids(passages: Sequence[Mapping[str, Any]], *, location: str) -> list[str]:
    return [
        _document_id(passage, location=f"{location}[{index}]")
        for index, passage in enumerate(passages, start=1)
    ]


def _assert_unique_passages(
    passages: Sequence[Mapping[str, Any]], *, location: str, exact_count: int | None = None
) -> None:
    if isinstance(passages, (str, bytes)) or not isinstance(passages, Sequence):
        raise V7IntegrityError(f"{location} must be a passage sequence")
    if exact_count is not None and len(passages) != exact_count:
        raise V7IntegrityError(
            f"{location} expected {exact_count} passages, observed {len(passages)}"
        )
    ids = _passage_ids(passages, location=location)
    if len(ids) != len(set(ids)):
        raise V7IntegrityError(f"{location} contains duplicate document ids")
    for passage in passages:
        _passage_text(passage)


def _deduplicate_passages(
    passages: Sequence[Mapping[str, Any]], *, location: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    first_index: dict[str, int] = {}
    duplicates: list[dict[str, Any]] = []
    for index, passage in enumerate(passages, start=1):
        if not isinstance(passage, Mapping):
            raise V7IntegrityError(f"{location}[{index}] is not an object")
        copied = deepcopy(dict(passage))
        _passage_text(copied)
        _document_id(copied, location=f"{location}[{index}]")
        key = passage_score_key(copied)
        if key in first_index:
            duplicates.append(
                {
                    "document_key": key,
                    "kept_raw_rank": first_index[key],
                    "dropped_raw_rank": index,
                }
            )
            continue
        first_index[key] = index
        kept.append(copied)
    return kept, duplicates


def _producer_passage_projection(
    passages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Freeze the exact <=10 x 1200-char passage view used by C.

    The reader prompt builder and verifier must both receive this artifact,
    never the untruncated retrieval object.
    """

    result: list[dict[str, str]] = []
    passage_telemetry: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, passage in enumerate(list(passages)[:BRIDGE_MAX_DOCS], start=1):
        doc_id = _document_id(passage, location=f"producer_passages[{rank}]")
        if doc_id in seen:
            raise V7IntegrityError(f"duplicate producer document id: {doc_id}")
        seen.add(doc_id)
        original_text = _passage_text(passage)
        title = passage.get("title")
        if title is not None and not isinstance(title, str):
            raise V7IntegrityError(f"producer_passages[{rank}].title is not text")
        if isinstance(title, str) and title.strip():
            # Match subanswer_v7._document_title exactly, including the edge
            # case where stripping quote characters produces an empty title.
            projected_title = " ".join(
                unicodedata.normalize("NFKC", title).split()
            ).strip('"')
        else:
            projected_title = " ".join(
                unicodedata.normalize("NFKC", original_text.splitlines()[0]).split()
            ).strip('"')
        item = {
            "doc_id": doc_id,
            "title": projected_title,
            "text": original_text[:BRIDGE_MAX_BODY_CHARS],
        }
        result.append(item)
        passage_telemetry.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "original_unicode_characters": len(original_text),
                "projected_unicode_characters": len(item["text"]),
                "truncated": len(original_text) > BRIDGE_MAX_BODY_CHARS,
            }
        )
    projection_hash = _sha256_json(result)
    return result, {
        "producer_input_passage_count": len(passages),
        "projected_passage_count": len(result),
        "passages": passage_telemetry,
        "ordered_producer_projection_sha256": projection_hash,
        "projection_fields": ["doc_id", "title", "text"],
        "maximum_unicode_characters_per_passage": BRIDGE_MAX_BODY_CHARS,
        "python_slice": "text[:1200]",
        "gold_access": False,
    }


def _sanitize_task_passages(
    passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Compatibility wrapper returning only the frozen C document view."""

    projected, _ = _producer_passage_projection(passages)
    return projected


def _sanitize_plan_step(step: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "step",
        "subject",
        "relation_label",
        "relation",
        "pid",
        "subquery_template",
        "output_slot",
        "dependencies",
    )
    result = {key: deepcopy(step[key]) for key in allowed if key in step}
    if not result.get("subject") and not result.get("subquery_template"):
        raise V7IntegrityError("plan step has no reader-visible subject/subquery")
    if not result.get("output_slot"):
        raise V7IntegrityError("plan step has no output_slot")
    dependencies = result.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise V7IntegrityError("plan step dependencies are not a list")
    return result


def _normalised_dependencies(step: Mapping[str, Any]) -> list[str]:
    from kgproweight.retrieval.dependent import normalize_dependency_ref

    result: list[str] = []
    raw_dependencies = step.get("dependencies") or []
    if not isinstance(raw_dependencies, list):
        raise V7IntegrityError("step dependencies must be a list")
    for raw in raw_dependencies:
        value = normalize_dependency_ref(raw)
        if value is None:
            raise V7IntegrityError(f"invalid dependency reference: {raw!r}")
        if value not in result:
            result.append(value)
    return result


def _step_schedule(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the deterministic topological schedule used by v5/v6."""

    from kgproweight.retrieval.dependent import normalize_dependency_ref

    produced_depth: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps, start=1):
        step = deepcopy(dict(raw_step))
        slot = normalize_dependency_ref(step.get("output_slot"))
        if slot is None:
            raise V7IntegrityError(f"step_{index} has no canonical output slot")
        dependencies = _normalised_dependencies(step)
        missing = [value for value in dependencies if value not in produced_depth]
        if missing:
            raise V7IntegrityError(f"step_{index} has unresolved dependencies {missing}")
        depth = 1 + max((produced_depth[value] for value in dependencies), default=0)
        produced_depth[slot] = depth
        result.append(
            {
                "step_index": index,
                "step": step,
                "step_sha256": _sha256_json(step),
                "slot": slot,
                "dependencies": dependencies,
                "dependency_depth": depth,
                "consumers": [],
            }
        )
    for producer in result:
        producer["consumers"] = [
            candidate["slot"]
            for candidate in result
            if producer["slot"] in candidate["dependencies"]
        ]
    return result


def _extract_plan(plan_row: Mapping[str, Any]) -> dict[str, Any]:
    if "predicted_target" in plan_row:
        predicted = plan_row.get("predicted_target")
        if isinstance(predicted, Mapping):
            return deepcopy(dict(predicted))
        if predicted is None:
            # The post-plan lock permits up to 20% schema-invalid planner
            # rows.  They are ordinary predeclared A-fallback cases, not a
            # materialization-wide runtime failure.
            return {"steps": []}
        raise V7IntegrityError(
            f"plan row {_question_key(plan_row)} has an invalid predicted_target type"
        )
    if isinstance(plan_row.get("plan"), Mapping):
        return deepcopy(dict(plan_row["plan"]))
    if isinstance(plan_row.get("steps"), list):
        return deepcopy(dict(plan_row))
    raise V7IntegrityError(f"plan row {_question_key(plan_row)} has no predicted plan")


def assemble_root_rows(
    cohort_rows: Iterable[Mapping[str, Any]],
    context_rows: Iterable[Mapping[str, Any]],
    plan_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Strictly join and project frozen question-only inputs for stage roots."""

    cohort = _as_rows(cohort_rows, label="cohort")
    contexts = _as_rows(context_rows, label="contexts")
    plans = _as_rows(plan_rows, label="plans")
    selected_keys = {_question_key(row) for row in cohort}
    context_by_key = _index_selected_rows(
        contexts,
        label="context",
        selected_keys=selected_keys,
    )
    plan_by_key = _index_selected_rows(
        plans,
        label="plan",
        selected_keys=selected_keys,
    )
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in cohort:
        key = _question_key(row)
        if key in seen:
            raise V7IntegrityError(f"duplicate cohort identity: {key}")
        seen.add(key)
        if key not in context_by_key or key not in plan_by_key:
            raise V7IntegrityError(f"incomplete identity join for {key}")
        context = context_by_key[key]
        plan_row = plan_by_key[key]
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise V7IntegrityError(f"empty question for {key}")
        expected_question_sha = _sha256_text(question)
        if str(row.get("question_sha256") or "") != expected_question_sha:
            raise V7IntegrityError(f"cohort question hash drift for {key}")
        for label, joined in (("context", context), ("plan", plan_row)):
            if str(joined.get("question") or "") != question:
                raise V7IntegrityError(f"{label} question mismatch for {key}")
            if str(joined.get("question_sha256") or "") != expected_question_sha:
                raise V7IntegrityError(f"{label} question hash mismatch for {key}")
            if joined.get("family_sha256") is not None and row.get("family_sha256") is not None:
                if str(joined["family_sha256"]) != str(row["family_sha256"]):
                    raise V7IntegrityError(f"{label} family hash mismatch for {key}")
        passages = deepcopy(list(context.get("passages") or []))
        _assert_unique_passages(passages, location=f"{key}.A", exact_count=TOTAL_PASSAGES)
        if str(context.get("passages_sha256") or "") != _sha256_historical_json(passages):
            raise V7IntegrityError(f"canonical passage hash drift for {key}")
        plan = _extract_plan(plan_row)
        assert_gold_free_source(plan, label=f"{key}.plan")
        result.append(
            {
                "question_key": key,
                "dataset": str(row["dataset"]),
                "qid": str(row["qid"]),
                "question": question,
                "question_sha256": expected_question_sha,
                "family_sha256": str(row.get("family_sha256") or ""),
                "role": str(row.get("role") or "development_consumed"),
                "arm_a_passages": passages,
                "arm_a_passages_sha256": _sha256_historical_json(passages),
                "plan": plan,
                "plan_row_sha256": _sha256_json(plan_row),
                "gold_access": False,
            }
        )
    return result


def _project_arm_a(row: Mapping[str, Any]) -> dict[str, Any]:
    passages = deepcopy(list(row["arm_a_passages"]))
    _assert_unique_passages(passages, location=f"{row['question_key']}.A", exact_count=10)
    passage_hash = _sha256_historical_json(passages)
    if passage_hash != str(row["arm_a_passages_sha256"]):
        raise V7IntegrityError(f"A passage hash drift for {row['question_key']}")
    return {
        "schema_version": ARM_SCHEMA_VERSION,
        "row_id": f"dependent-retrieval-v7::{row['question_key']}",
        "question_key": str(row["question_key"]),
        "dataset": str(row["dataset"]),
        "qid": str(row["qid"]),
        "question": str(row["question"]),
        "question_sha256": str(row["question_sha256"]),
        "family_sha256": str(row.get("family_sha256") or ""),
        "role": "development_consumed",
        "arm": "A_canonical_one_shot",
        "retrieved_passages": passages,
        "passages_sha256": passage_hash,
        "kg_subgraph": [],
        "gold_access": False,
    }


def _predict_scores(cross_encoder: Any, pairs: Sequence[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    raw = cross_encoder.predict(list(pairs), show_progress_bar=False)
    try:
        values = [float(value) for value in raw]
    except TypeError:
        values = [float(raw)]
    if len(values) != len(pairs):
        raise V7IntegrityError(
            f"cross encoder returned {len(values)}/{len(pairs)} scores"
        )
    if any(not math.isfinite(value) for value in values):
        raise V7IntegrityError("cross encoder returned a missing/non-finite score")
    return values


def _rerank_one_query(
    query: str,
    raw_passages: Sequence[Mapping[str, Any]],
    *,
    cross_encoder: Any,
    topk: int = STEP_RERANK_TOPK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, duplicate_trace = _deduplicate_passages(
        raw_passages, location=f"retrieval[{_sha256_text(query)}]"
    )
    pairs = [(query, _passage_text(passage)[:CE_MAX_CHARS]) for passage in candidates]
    scores = _predict_scores(cross_encoder, pairs)
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    selected = order[:topk]
    ranked = [deepcopy(candidates[index]) for index in selected]
    trace = {
        "raw_count": len(raw_passages),
        "deduplicated_count": len(candidates),
        "duplicate_observations": duplicate_trace,
        "reranked_count": len(ranked),
        "ce_pairs": [
            {
                "question_or_query_sha256": _sha256_text(query),
                "document_key": passage_score_key(candidates[index]),
                "document_id": _document_id(
                    candidates[index], location=f"rerank_candidates[{index + 1}]"
                ),
                "score": scores[index],
                "selected_rank": selected.index(index) + 1 if index in selected else None,
            }
            for index in range(len(candidates))
        ],
        "retrieved_ids": _passage_ids(ranked, location="reranked"),
    }
    return ranked, trace


def _batch_search(
    retriever: Any, queries: Sequence[str], *, label: str
) -> list[list[Mapping[str, Any]]]:
    if not hasattr(retriever, "batch_search"):
        raise V7IntegrityError("v7 requires retriever.batch_search")
    if not queries:
        return []
    raw = retriever.batch_search(list(queries))
    batches = [list(value) for value in raw]
    if len(batches) != len(queries):
        raise V7IntegrityError(
            f"{label} batch retriever returned {len(batches)}/{len(queries)} rows"
        )
    return batches


def _logical_hop_hash(question_key: str, record: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "question_key": question_key,
            "step_index": record["step_index"],
            "slot": record["slot"],
            "step_sha256": record["step_sha256"],
            "dependency_depth": record["dependency_depth"],
        }
    )


def _record_slot(slot_values: dict[str, str], slot: str, value: str) -> None:
    clean = str(value).strip()
    if not clean:
        raise V7IntegrityError(f"attempted to record an empty value for {slot}")
    existing = slot_values.get(slot)
    if existing is not None and existing != clean:
        raise V7IntegrityError(f"slot value changed for {slot}")
    slot_values[slot] = clean


def _make_c_task(
    state: Mapping[str, Any], record: Mapping[str, Any], passages: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    safe_passages, projection_telemetry = _producer_passage_projection(passages)
    if not safe_passages:
        return None, projection_telemetry
    step = _sanitize_plan_step(record["step"])
    messages = build_subanswer_reader_messages(
        str(state["question"]),
        step,
        safe_passages,
        target_type=str(state["target_type"]),
    )
    try:
        prompt_documents = json.loads(messages[1]["content"])["retrieved_documents"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise V7IntegrityError("cannot recover C documents from reader prompt") from exc
    if prompt_documents != safe_passages:
        raise V7IntegrityError(
            "C task projection differs byte-for-byte from reader prompt projection"
        )
    prompt_projection_hash = _sha256_json(prompt_documents)
    verifier_projection_hash = _sha256_json(safe_passages)
    if prompt_projection_hash != verifier_projection_hash:
        raise V7IntegrityError("C reader/verifier projection hash mismatch")
    projection_telemetry.update(
        {
            "reader_prompt_projection_sha256": prompt_projection_hash,
            "verifier_input_projection_sha256": verifier_projection_hash,
            "prompt_verifier_projection_hash_equal": True,
        }
    )
    task_id = _sha256_json(
        {
            "question_key": state["question_key"],
            "producer_slot": record["slot"],
            "step_sha256": _sha256_json(step),
            "producer_passages_sha256": _sha256_json(safe_passages),
        }
    )
    task = {
        "task_id": task_id,
        "question_key": str(state["question_key"]),
        "dataset": str(state["dataset"]),
        "qid": str(state["qid"]),
        "question": str(state["question"]),
        "question_sha256": str(state["question_sha256"]),
        "target_type": str(state["target_type"]),
        "producer_slot": str(record["slot"]),
        "step": step,
        "step_sha256": _sha256_json(step),
        "producer_passages": safe_passages,
        "producer_passages_sha256": _sha256_json(safe_passages),
        "gold_access": False,
    }
    if frozenset(task) != C_TASK_KEYS:
        raise AssertionError("internal C task field drift")
    return task, projection_telemetry


def _hop_result(
    *,
    arm: str,
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    query: str,
    passages: Sequence[Mapping[str, Any]],
    hint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    logical_hop_id = str(record["slot"])
    return {
        "logical_hop_id": logical_hop_id,
        "logical_hop_sha256": _logical_hop_hash(str(state["question_key"]), record),
        "step_index": int(record["step_index"]),
        "dependencies": list(record["dependencies"]),
        "dependency_depth": int(record["dependency_depth"]),
        "is_dependent": bool(record["dependencies"]),
        "query_variants": [
            {
                "query_variant_id": f"{arm}::{logical_hop_id}::q1",
                "query_variant_index": 1,
                "query": query,
                "query_sha256": _sha256_text(query),
                "hint": deepcopy(dict(hint)) if hint is not None else None,
                "passages": [deepcopy(dict(value)) for value in passages],
            }
        ],
    }


def _root_budget_row(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    query: str,
    physical_id: str,
    *,
    physical_search_executed_here: bool,
) -> dict[str, Any]:
    hop_hash = _logical_hop_hash(str(state["question_key"]), record)
    logical_id = f"{state['question_key']}::{record['slot']}::root"
    return {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "question_key": str(state["question_key"]),
        "dataset": str(state["dataset"]),
        "qid": str(state["qid"]),
        "logical_hop_id": str(record["slot"]),
        "logical_hop_sha256": hop_hash,
        "dependency_depth": 1,
        "is_root": True,
        "paired_active": True,
        "paired_skip_reason": None,
        "B": {
            "logical_query_count": 1,
            "physical_slot_count": 1,
            "logical_slot_id": f"B::{logical_id}",
            "physical_slot_id": physical_id,
            "query": query,
            "query_sha256": _sha256_text(query),
        },
        "C": {
            "logical_query_count": 1,
            "physical_slot_count": 1,
            "logical_slot_id": f"C::{logical_id}",
            "physical_slot_id": physical_id,
            "query": query,
            "query_sha256": _sha256_text(query),
        },
        "actual_shared_physical_search_count": int(physical_search_executed_here),
        "actual_independent_physical_search_count": 0,
        "cross_arm_query_strings_identical": True,
        "budget_equal": True,
        "gold_access": False,
    }


def _dependent_budget_row(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    active: bool,
    reason: str | None,
    query_b: str | None = None,
    query_c: str | None = None,
) -> dict[str, Any]:
    count = int(active)
    logical_id = f"{state['question_key']}::{record['slot']}::dependent"
    hop_hash = _logical_hop_hash(str(state["question_key"]), record)
    row = {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "question_key": str(state["question_key"]),
        "dataset": str(state["dataset"]),
        "qid": str(state["qid"]),
        "logical_hop_id": str(record["slot"]),
        "logical_hop_sha256": hop_hash,
        "dependency_depth": int(record["dependency_depth"]),
        "is_root": False,
        "paired_active": bool(active),
        "paired_skip_reason": reason,
        "B": {
            "logical_query_count": count,
            "physical_slot_count": count,
            "logical_slot_id": f"B::{logical_id}",
            "physical_slot_id": f"B::{logical_id}" if active else None,
            "query": query_b,
            "query_sha256": _sha256_text(query_b) if query_b is not None else None,
        },
        "C": {
            "logical_query_count": count,
            "physical_slot_count": count,
            "logical_slot_id": f"C::{logical_id}",
            "physical_slot_id": f"C::{logical_id}" if active else None,
            "query": query_c,
            "query_sha256": _sha256_text(query_c) if query_c is not None else None,
        },
        "actual_shared_physical_search_count": 0,
        "actual_independent_physical_search_count": 2 if active else 0,
        "cross_arm_query_strings_identical": (
            query_b == query_c if active else None
        ),
        "budget_equal": True,
        "gold_access": False,
    }
    _assert_budget_row(row)
    return row


def _assert_budget_row(row: Mapping[str, Any]) -> None:
    b = int(row["B"]["logical_query_count"])
    c = int(row["C"]["logical_query_count"])
    if b != c or b not in {0, 1}:
        raise V7IntegrityError(f"B/C query budget mismatch: B={b}, C={c}")
    if bool(row["paired_active"]) != (b == 1):
        raise V7IntegrityError("paired activation differs from logical query count")
    if b == 0 and (row["B"].get("query") or row["C"].get("query")):
        raise V7IntegrityError("paired skip contains a padding query")


def _base_state(
    row: Mapping[str, Any],
    arm_a: Mapping[str, Any],
    *,
    run_locks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    target_type = TARGET_TYPES[dataset]
    plan = deepcopy(dict(row["plan"]))
    errors = validate_plan_for_dependent_retrieval(
        plan, target_type, max_steps=MAX_PLAN_STEPS
    )
    schedule: list[dict[str, Any]] = []
    if not errors:
        try:
            schedule = _step_schedule(list(plan.get("steps") or []))
        except Exception as exc:
            errors = [f"schedule_error:{type(exc).__name__}:{exc}"]
    max_depth = max(
        (int(record["dependency_depth"]) for record in schedule), default=1
    )
    return {
        "schema_version": ROOT_STATE_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "run_locks": deepcopy(dict(run_locks or {})),
        "question_key": str(row["question_key"]),
        "dataset": dataset,
        "qid": str(row["qid"]),
        "question": str(row["question"]),
        "question_sha256": str(row["question_sha256"]),
        "family_sha256": str(row.get("family_sha256") or ""),
        "target_type": target_type,
        "plan": plan,
        "plan_sha256": _sha256_json(plan),
        "plan_row_sha256": str(row.get("plan_row_sha256") or ""),
        "plan_validation_errors": list(errors),
        "plan_executable": not errors,
        "has_dependent_step": bool(
            not errors and any(record["dependencies"] for record in schedule)
        ),
        "schedule": schedule,
        "max_dependency_depth": max_depth,
        "completed_depth": 1,
        "arm_a": deepcopy(dict(arm_a)),
        "slot_values_B": {},
        "slot_values_C": {},
        "hop_results_B": [],
        "hop_results_C": [],
        "hop_telemetry": [],
        "subanswer_telemetry": [],
        "budget_ledger": [],
        "issued_dependent_queries_B": [],
        "issued_dependent_queries_C": [],
        "pending_c_tasks": [],
        "successful_paired_dependent_hops": 0,
        "execution_status": "pending" if not errors else "fallback_plan_invalid",
        "fallback_reason": None if not errors else "plan_invalid",
        "gold_access": False,
    }


def execute_root_stage(
    rows: Sequence[Mapping[str, Any]],
    retriever: Any,
    *,
    cross_encoder: Any,
    run_locks: Mapping[str, Mapping[str, Any]] | None = None,
    locked_plan_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> StageResult:
    """Execute shared physical root retrieval and materialise reader tasks."""

    states: list[dict[str, Any]] = []
    arm_a_rows: list[dict[str, Any]] = []
    query_owners: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for source in rows:
        row = deepcopy(dict(source))
        assert_gold_free_source(row, label=f"root_row[{row.get('question_key', '?')}]")
        arm_a = _project_arm_a(row)
        arm_a_rows.append(arm_a)
        state = _base_state(row, arm_a, run_locks=run_locks)
        states.append(state)
        if not state["plan_executable"]:
            continue
        for record in state["schedule"]:
            if record["dependencies"]:
                continue
            query = render_root_query(record["step"], state["target_type"])
            if not query or dependency_refs(query):
                raise V7IntegrityError("root query is empty or contains a placeholder")
            query_owners.setdefault(query, []).append((state, record))

    unique_queries = list(query_owners)
    raw_batches = _batch_search(retriever, unique_queries, label="root")
    ranked_by_query: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for query, raw in zip(unique_queries, raw_batches):
        ranked_by_query[query] = _rerank_one_query(
            query, raw, cross_encoder=cross_encoder, topk=STEP_RERANK_TOPK
        )

    c_tasks: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for query, owners in query_owners.items():
        ranked, rerank_trace = ranked_by_query[query]
        physical_id = f"root-shared::{_sha256_text(query)}"
        for owner_index, (state, record) in enumerate(owners):
            entity_candidates = (
                extract_deterministic_bridge_candidates(
                    query,
                    ranked,
                    max_docs=BRIDGE_MAX_DOCS,
                    max_candidates=1,
                    max_body_chars=BRIDGE_MAX_BODY_CHARS,
                )
                if record["consumers"]
                else []
            )
            if len(entity_candidates) > 1:
                raise V7IntegrityError("B entity selector exceeded top-1")
            if entity_candidates:
                _record_slot(
                    state["slot_values_B"],
                    record["slot"],
                    str(entity_candidates[0]["surface"]),
                )
            task, projection_telemetry = (
                _make_c_task(state, record, ranked)
                if record["consumers"] and ranked
                else (None, None)
            )
            if task is not None:
                state["pending_c_tasks"].append(deepcopy(task))
                c_tasks.append(deepcopy(task))
            for arm in ("B", "C"):
                state[f"hop_results_{arm}"].append(
                    _hop_result(
                        arm=arm,
                        state=state,
                        record=record,
                        query=query,
                        passages=ranked,
                        hint=None,
                    )
                )
            budget = _root_budget_row(
                state,
                record,
                query,
                physical_id,
                physical_search_executed_here=owner_index == 0,
            )
            _assert_budget_row(budget)
            state["budget_ledger"].append(deepcopy(budget))
            budget_rows.append(deepcopy(budget))
            state["hop_telemetry"].append(
                {
                    "logical_hop_id": record["slot"],
                    "logical_hop_sha256": _logical_hop_hash(
                        state["question_key"], record
                    ),
                    "step_index": record["step_index"],
                    "dependency_depth": 1,
                    "dependencies": [],
                    "root_query_shared": True,
                    "query": query,
                    "query_sha256": _sha256_text(query),
                    "physical_search_id": physical_id,
                    "rerank": deepcopy(rerank_trace),
                    "B_entity_candidate": deepcopy(entity_candidates[0])
                    if entity_candidates
                    else None,
                    "C_task_id": task["task_id"] if task else None,
                    "C_producer_projection": deepcopy(projection_telemetry),
                    "trajectory_semantics": {
                        "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
                        "shared_query": True,
                        "B_query_sha256": _sha256_text(query),
                        "C_query_sha256": _sha256_text(query),
                        "B_producer_passages_sha256": _sha256_json(ranked),
                        "C_producer_passages_sha256": _sha256_json(ranked),
                        "producer_passages_identical_when_query_identical": True,
                        "downstream_context_is_causal_mediator": False,
                    },
                    "gold_access": False,
                }
            )

    for state in states:
        state["pending_c_tasks"].sort(key=lambda task: task["producer_slot"])
        state["hop_results_B"].sort(key=lambda hop: int(hop["step_index"]))
        state["hop_results_C"].sort(key=lambda hop: int(hop["step_index"]))
        state["hop_telemetry"].sort(key=lambda hop: int(hop["step_index"]))
        if state["plan_executable"] and not state["has_dependent_step"]:
            state["execution_status"] = "fallback_no_dependent_step"
            state["fallback_reason"] = "no_dependent_step"
        elif state["plan_executable"]:
            state["execution_status"] = "roots_complete"
        _validate_state(
            state,
            locked_plan_rows=locked_plan_rows,
            expected_run_locks=run_locks,
        )
    c_tasks.sort(key=lambda task: (task["question_key"], task["producer_slot"]))
    budget_rows.sort(
        key=lambda row: (row["question_key"], row["dependency_depth"], row["logical_hop_id"])
    )
    return StageResult(
        states=states,
        c_tasks=c_tasks,
        budget_rows=budget_rows,
        arm_a_rows=arm_a_rows,
    )


def _validate_state(
    state: Mapping[str, Any],
    *,
    locked_plan_rows: Mapping[str, Mapping[str, Any]] | None = None,
    expected_run_locks: Mapping[str, Mapping[str, Any]] | None = None,
    expected_model_artifact: Mapping[str, Any] | None = None,
) -> None:
    key = _question_key(state)
    assert_gold_free_source(state, label=f"state[{key}]")
    if state.get("schema_version") not in {
        ROOT_STATE_SCHEMA_VERSION,
        STATE_SCHEMA_VERSION,
    }:
        raise V7IntegrityError(f"state schema version differs for {key}")
    if state.get("runner_version") != RUNNER_VERSION:
        raise V7IntegrityError(f"state runner version differs for {key}")
    if state.get("gold_access") is not False:
        raise V7IntegrityError(f"state gold_access is not false for {key}")
    question = state.get("question")
    if not isinstance(question, str) or _sha256_text(question) != state.get(
        "question_sha256"
    ):
        raise V7IntegrityError(f"state question/hash drift for {key}")
    arm_a = state.get("arm_a")
    if not isinstance(arm_a, Mapping):
        raise V7IntegrityError(f"state has no A arm for {key}")
    passages = arm_a.get("retrieved_passages")
    if not isinstance(passages, list):
        raise V7IntegrityError(f"state A passages are missing for {key}")
    _assert_unique_passages(passages, location=f"{key}.state.A", exact_count=10)
    if _sha256_historical_json(passages) != arm_a.get("passages_sha256"):
        raise V7IntegrityError(f"state A passage hash drift for {key}")
    if _sha256_json(state.get("plan")) != state.get("plan_sha256"):
        raise V7IntegrityError(f"state plan hash drift for {key}")
    if not isinstance(state.get("plan"), Mapping):
        raise V7IntegrityError(f"state plan is not an object for {key}")
    target_type = TARGET_TYPES.get(str(state.get("dataset") or ""))
    if state.get("target_type") != target_type:
        raise V7IntegrityError(f"state target type drift for {key}")
    recomputed_errors = validate_plan_for_dependent_retrieval(
        state["plan"], str(target_type), max_steps=MAX_PLAN_STEPS
    )
    recomputed_schedule: list[dict[str, Any]] = []
    if not recomputed_errors:
        try:
            recomputed_schedule = _step_schedule(list(state["plan"].get("steps") or []))
        except Exception as exc:
            recomputed_errors = [f"schedule_error:{type(exc).__name__}:{exc}"]
    if list(state.get("plan_validation_errors") or []) != list(recomputed_errors):
        raise V7IntegrityError(f"state plan validation verdict drift for {key}")
    if bool(state.get("plan_executable")) != (not recomputed_errors):
        raise V7IntegrityError(f"state plan executable flag drift for {key}")
    if list(state.get("schedule") or []) != recomputed_schedule:
        raise V7IntegrityError(f"state schedule drift for {key}")
    recomputed_maximum = max(
        (int(record["dependency_depth"]) for record in recomputed_schedule), default=1
    )
    if state.get("max_dependency_depth") != recomputed_maximum:
        raise V7IntegrityError(f"state maximum dependency depth drift for {key}")
    recomputed_dependent = bool(
        not recomputed_errors
        and any(record["dependencies"] for record in recomputed_schedule)
    )
    if state.get("has_dependent_step") is not recomputed_dependent:
        raise V7IntegrityError(f"state dependent-step flag drift for {key}")

    if locked_plan_rows is not None:
        locked_row = locked_plan_rows.get(key)
        if not isinstance(locked_row, Mapping):
            raise V7IntegrityError(f"locked planner row is missing for {key}")
        assert_gold_free_source(locked_row, label=f"locked_plan[{key}]")
        if _sha256_json(locked_row) != state.get("plan_row_sha256"):
            raise V7IntegrityError(f"state planner-row hash differs from lock for {key}")
        if _extract_plan(locked_row) != state.get("plan"):
            raise V7IntegrityError(f"state plan differs from locked planner prediction for {key}")
        for field in ("question_key", "dataset", "qid", "question", "question_sha256"):
            if locked_row.get(field) != state.get(field):
                raise V7IntegrityError(f"state/locked planner identity drift for {key}: {field}")
    completed_depth = state.get("completed_depth")
    maximum = state.get("max_dependency_depth")
    if isinstance(completed_depth, bool) or not isinstance(completed_depth, int):
        raise V7IntegrityError(f"invalid completed depth for {key}")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise V7IntegrityError(f"invalid maximum depth for {key}")
    if completed_depth < 1 or completed_depth > maximum:
        raise V7IntegrityError(f"completed depth exceeds bounds for {key}")
    if arm_a.get("question_key") != key or arm_a.get("dataset") != state.get(
        "dataset"
    ) or arm_a.get("qid") != state.get("qid"):
        raise V7IntegrityError(f"state A identity drift for {key}")
    if (
        arm_a.get("question") != question
        or arm_a.get("question_sha256") != state.get("question_sha256")
        or arm_a.get("arm") != "A_canonical_one_shot"
        or arm_a.get("kg_subgraph") != []
        or arm_a.get("gold_access") is not False
    ):
        raise V7IntegrityError(f"state A frozen contract drift for {key}")

    budget_rows = state.get("budget_ledger")
    if not isinstance(budget_rows, list):
        raise V7IntegrityError(f"state budget ledger is not a list for {key}")
    budget_identities: set[tuple[str, int]] = set()
    for row in budget_rows:
        if not isinstance(row, Mapping):
            raise V7IntegrityError(f"state budget row is not an object for {key}")
        _assert_budget_row(row)
        if row.get("question_key") != key or row.get("dataset") != state.get(
            "dataset"
        ) or row.get("qid") != state.get("qid"):
            raise V7IntegrityError(f"state budget identity drift for {key}")
        identity = (str(row.get("logical_hop_id") or ""), int(row["dependency_depth"]))
        if identity in budget_identities:
            raise V7IntegrityError(f"duplicate state budget hop for {key}: {identity}")
        budget_identities.add(identity)
    run_locks = state.get("run_locks")
    if not isinstance(run_locks, Mapping):
        raise V7IntegrityError(f"state run_locks is not an object for {key}")
    for name, lock in run_locks.items():
        if not isinstance(lock, Mapping):
            raise V7IntegrityError(f"invalid state run lock for {key}: {name}")
        if not isinstance(lock.get("sha256"), str) or len(lock["sha256"]) != 64:
            raise V7IntegrityError(f"invalid state run-lock hash for {key}: {name}")
    if expected_run_locks is not None and dict(run_locks) != dict(expected_run_locks):
        raise V7IntegrityError(f"state runtime locks differ from this execution for {key}")

    for name in (
        "slot_values_B",
        "slot_values_C",
    ):
        values = state.get(name)
        if not isinstance(values, Mapping):
            raise V7IntegrityError(f"state {name} is not an object for {key}")
        for slot, value in values.items():
            if (
                not isinstance(slot, str)
                or not slot
                or not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise V7IntegrityError(f"state {name} has an invalid slot/value for {key}")

    pending = state.get("pending_c_tasks")
    if not isinstance(pending, list):
        raise V7IntegrityError(f"state pending C tasks is not a list for {key}")
    pending_ids: set[str] = set()
    for task in pending:
        if not isinstance(task, Mapping) or frozenset(task) != C_TASK_KEYS:
            raise V7IntegrityError(f"pending C task schema drift for {key}")
        assert_gold_free_source(task, label=f"state[{key}].pending_C")
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in pending_ids:
            raise V7IntegrityError(f"duplicate/empty pending C task for {key}")
        pending_ids.add(task_id)
        if (
            task.get("question_key") != key
            or task.get("dataset") != state.get("dataset")
            or task.get("qid") != state.get("qid")
            or task.get("question") != question
            or task.get("question_sha256") != state.get("question_sha256")
            or task.get("target_type") != target_type
            or _sha256_json(task.get("step")) != task.get("step_sha256")
            or _sha256_json(task.get("producer_passages"))
            != task.get("producer_passages_sha256")
        ):
            raise V7IntegrityError(f"pending C task identity/hash drift for {key}")

    for arm in ("B", "C"):
        queries = state.get(f"issued_dependent_queries_{arm}") or []
        if not isinstance(queries, list) or any(
            not isinstance(query, str) or not query for query in queries
        ):
            raise V7IntegrityError(f"invalid issued query list in arm {arm} for {key}")
        if len(queries) != len(set(queries)):
            raise V7IntegrityError(f"duplicate issued dependent query in arm {arm} for {key}")
        hops = state.get(f"hop_results_{arm}")
        if not isinstance(hops, list):
            raise V7IntegrityError(f"state hop results are not a list in arm {arm} for {key}")
        schedule_by_slot = {
            str(record["slot"]): record for record in recomputed_schedule
        }
        for hop in hops:
            if not isinstance(hop, Mapping):
                raise V7IntegrityError(f"state hop result is not an object in arm {arm} for {key}")
            record = schedule_by_slot.get(str(hop.get("logical_hop_id") or ""))
            if record is None or hop.get("logical_hop_sha256") != _logical_hop_hash(
                key, record
            ):
                raise V7IntegrityError(f"state hop-result identity drift in arm {arm} for {key}")
            variants = hop.get("query_variants")
            if not isinstance(variants, list) or len(variants) != 1:
                raise V7IntegrityError(f"state hop-result variants drift in arm {arm} for {key}")
            variant = variants[0]
            if not isinstance(variant, Mapping) or not str(
                variant.get("query_variant_id") or ""
            ).startswith(f"{arm}::"):
                raise V7IntegrityError(f"state hop-result arm drift in arm {arm} for {key}")
            query = variant.get("query")
            if not isinstance(query, str) or variant.get("query_sha256") != _sha256_text(query):
                raise V7IntegrityError(f"state hop-result query hash drift in arm {arm} for {key}")

    hop_telemetry = state.get("hop_telemetry")
    subanswer_telemetry = state.get("subanswer_telemetry")
    if not isinstance(hop_telemetry, list) or not isinstance(subanswer_telemetry, list):
        raise V7IntegrityError(f"state telemetry collections drift for {key}")
    for hop in hop_telemetry:
        if not isinstance(hop, Mapping) or hop.get("gold_access") is not False:
            raise V7IntegrityError(f"state hop telemetry is malformed for {key}")
        semantics = hop.get("trajectory_semantics")
        if not isinstance(semantics, Mapping) or semantics.get("estimand") != (
            "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT"
        ):
            raise V7IntegrityError(f"state trajectory-semantics telemetry drift for {key}")
        if semantics.get("shared_query") is True:
            if (
                semantics.get("B_query_sha256") != semantics.get("C_query_sha256")
                or semantics.get("B_producer_passages_sha256")
                != semantics.get("C_producer_passages_sha256")
                or semantics.get("producer_passages_identical_when_query_identical")
                is not True
                or semantics.get("downstream_context_is_causal_mediator") is not False
            ):
                raise V7IntegrityError(
                    f"shared-query producer-context invariant drift for {key}"
                )
        if semantics.get("downstream_context_is_causal_mediator") is True and semantics.get(
            "shared_query"
        ) is not False:
            raise V7IntegrityError(f"arm-specific trajectory semantics drift for {key}")
    seen_subanswer_tasks: set[str] = set()
    for attempt in subanswer_telemetry:
        if not isinstance(attempt, Mapping) or attempt.get("gold_access") is not False:
            raise V7IntegrityError(f"state subanswer telemetry is malformed for {key}")
        task_id = str(attempt.get("task_id") or "")
        if not task_id or task_id in seen_subanswer_tasks:
            raise V7IntegrityError(f"duplicate/empty state subanswer task for {key}")
        seen_subanswer_tasks.add(task_id)
        telemetry = attempt.get("telemetry")
        if not isinstance(telemetry, Mapping):
            raise V7IntegrityError(f"state subanswer attempt lacks telemetry for {key}")
        raw_response = telemetry.get("raw_response")
        if not isinstance(raw_response, str) or telemetry.get(
            "raw_response_sha256"
        ) != _sha256_text(raw_response):
            raise V7IntegrityError(f"state subanswer raw-response hash drift for {key}")
        if expected_model_artifact is not None and telemetry.get(
            "model_artifact"
        ) != dict(expected_model_artifact):
            raise V7IntegrityError(f"state subanswer model content lock drift for {key}")
    active_dependent = sum(
        1
        for hop in hop_telemetry
        if isinstance(hop, Mapping)
        and hop.get("paired_active") is True
        and int(hop.get("dependency_depth", 1)) > 1
    )
    if state.get("successful_paired_dependent_hops") != active_dependent:
        raise V7IntegrityError(f"state paired-dependent-hop count drift for {key}")
    if len(state.get("issued_dependent_queries_B") or []) != active_dependent or len(
        state.get("issued_dependent_queries_C") or []
    ) != active_dependent:
        raise V7IntegrityError(f"state issued-query count drift for {key}")

    status = state.get("execution_status")
    fallback_reason = state.get("fallback_reason")
    if recomputed_errors:
        if status != "fallback_plan_invalid" or fallback_reason != "plan_invalid":
            raise V7IntegrityError(
                f"invalid plan has inconsistent execution status for {key}"
            )
    elif not recomputed_dependent:
        if (
            status != "fallback_no_dependent_step"
            or fallback_reason != "no_dependent_step"
        ):
            raise V7IntegrityError(
                f"non-dependent plan has inconsistent status for {key}"
            )
    elif completed_depth == 1:
        if status != "roots_complete" or fallback_reason is not None:
            raise V7IntegrityError(f"root-complete plan has inconsistent status for {key}")
    elif completed_depth < maximum:
        if status != "depth_complete" or fallback_reason is not None:
            raise V7IntegrityError(f"partial plan has inconsistent status for {key}")
    elif status != "dependent_retrieval_complete" or fallback_reason is not None:
        raise V7IntegrityError(f"completed plan has inconsistent status for {key}")


def _validate_c_answer(
    task: Mapping[str, Any],
    answer: Mapping[str, Any],
    *,
    expected_model_artifact: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    # Scan the complete parsed answer object before projecting or trusting any
    # whitelist.  A nested ``gold_answer``/``supporting_facts`` payload must be
    # a whole-stage error, even if it is hidden under otherwise valid telemetry.
    assert_gold_free_source(answer, label="C_answer")
    actual_keys = frozenset(answer)
    if actual_keys != C_ANSWER_KEYS:
        raise V7IntegrityError(
            "C answer fields differ from frozen contract; "
            f"missing={sorted(C_ANSWER_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - C_ANSWER_KEYS)}"
        )
    if answer.get("gold_access") is not False:
        raise V7IntegrityError("C answer gold_access is not false")
    for field in _TASK_ANSWER_IDENTITY_FIELDS:
        if answer.get(field) != task.get(field):
            raise V7IntegrityError(f"C answer identity/hash mismatch for {field}")
    verified = answer.get("verified")
    if not isinstance(verified, bool):
        raise V7IntegrityError("C answer verified flag is not boolean")
    verified_answer = answer.get("verified_answer")
    telemetry = answer.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise V7IntegrityError("C answer telemetry is not an object")
    telemetry = deepcopy(dict(telemetry))
    if telemetry.get("gold_access") is not False:
        raise V7IntegrityError("C answer telemetry gold_access is not false")
    model_artifact = telemetry.get("model_artifact")
    if expected_model_artifact is not None:
        if not isinstance(model_artifact, Mapping):
            raise V7IntegrityError("C answer telemetry lacks model content locks")
        if dict(model_artifact) != dict(expected_model_artifact):
            raise V7IntegrityError("C answer model identity/content locks differ")
    strict_parse = telemetry.get("strict_parse")
    if not isinstance(strict_parse, Mapping):
        raise V7IntegrityError("C answer telemetry lacks strict_parse")
    if not isinstance(strict_parse.get("valid"), bool):
        raise V7IntegrityError("C answer strict_parse.valid is not boolean")
    error_code = strict_parse.get("error_code")
    if strict_parse["valid"] and error_code is not None:
        raise V7IntegrityError("valid strict parse carries an error code")
    if not strict_parse["valid"] and not isinstance(error_code, str):
        raise V7IntegrityError("invalid strict parse lacks an error code")
    raw_response = telemetry.get("raw_response")
    raw_response_sha256 = telemetry.get("raw_response_sha256")
    if not isinstance(raw_response, str) or raw_response_sha256 != _sha256_text(raw_response):
        raise V7IntegrityError("C answer raw response/hash mismatch")
    for field in ("prompt_passages_sha256", "verifier_passages_sha256"):
        if telemetry.get(field) != task["producer_passages_sha256"]:
            raise V7IntegrityError(f"C answer {field} differs from its producer task")
    if telemetry.get("same_passage_bytes_for_prompt_and_verifier") is not True:
        raise V7IntegrityError("C prompt/verifier passage bytes were not identical")
    if telemetry.get("input_task_sha256") != _sha256_json(task):
        raise V7IntegrityError("C answer input-task hash differs from producer task")
    if telemetry.get("producer_passage_count") != len(task["producer_passages"]):
        raise V7IntegrityError("C answer producer passage count differs")
    prompt_hash = telemetry.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise V7IntegrityError("C answer lacks a prompt hash")
    for field, allow_zero in (("prompt_tokens", False), ("generation_tokens", True)):
        value = telemetry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < int(not allow_zero):
            raise V7IntegrityError(f"C answer has invalid {field}")
    if int(telemetry["generation_tokens"]) > 96:
        raise V7IntegrityError("C answer exceeds max_new_tokens=96")
    generation = telemetry.get("generation")
    if not isinstance(generation, Mapping) or (
        generation.get("decode") != "greedy"
        or generation.get("do_sample") is not False
        or generation.get("max_new_tokens") != 96
        or generation.get("retry_count") != 0
    ):
        raise V7IntegrityError("C answer generation settings differ from frozen contract")
    verification = telemetry.get("verification")
    if not isinstance(verification, Mapping):
        raise V7IntegrityError("C answer telemetry lacks nested verification")
    verification = deepcopy(dict(verification))
    if verification.get("gold_access") is not False:
        raise V7IntegrityError("nested C verification gold_access is not false")
    if verification.get("parser_version") != SUBANSWER_PARSER_VERSION:
        raise V7IntegrityError("nested C parser version differs")
    if verification.get("verifier_version") != SUBANSWER_VERIFIER_VERSION:
        raise V7IntegrityError("nested C verifier version differs")
    recomputed = parse_and_verify_subanswer(
        raw_response,
        task["question"],
        task["step"],
        task["producer_passages"],
        target_type=task["target_type"],
    )
    if recomputed != verification:
        raise V7IntegrityError("C verification does not match deterministic recomputation")
    if verification.get("verified") is not verified:
        raise V7IntegrityError("C answer verified flag differs from verifier telemetry")
    if verification.get("verification_scope") != "surface_locality_not_semantic_entailment":
        raise V7IntegrityError("C answer has an unexpected verification scope")
    response_hash = verification.get("response_sha256")
    if response_hash != raw_response_sha256:
        raise V7IntegrityError("nested verifier response hash differs from raw response")
    if verified and strict_parse.get("valid") is not True:
        raise V7IntegrityError("verified C answer did not pass strict parsing")
    if verified:
        if not isinstance(verified_answer, str) or not verified_answer.strip():
            raise V7IntegrityError("verified C answer is empty")
        if verified_answer != verified_answer.strip() or len(verified_answer) > MAX_ANSWER_CHARS:
            raise V7IntegrityError("verified C answer violates surface bounds")
        if verification.get("verified_answer") != verified_answer:
            raise V7IntegrityError("C answer surface differs from verifier telemetry")
        if verification.get("answer_type") not in {"entity", "number", "date"}:
            raise V7IntegrityError("verified C answer has a non-promotable answer type")
        cited = verification.get("cited_doc_ids")
        if not isinstance(cited, list) or len(cited) != 1 or not isinstance(cited[0], str):
            raise V7IntegrityError("verified C answer must contain one string citation")
        passage_ids = _passage_ids(
            task["producer_passages"], location="C_task.producer_passages"
        )
        if cited[0] not in passage_ids or verification.get("supporting_doc_id") != cited[0]:
            raise V7IntegrityError("verified C answer citation is outside its producer task")
        sentence = verification.get("supporting_sentence")
        if not isinstance(sentence, str) or not sentence:
            raise V7IntegrityError("verified C answer lacks a supporting sentence")
        if verification.get("supporting_sentence_sha256") != _sha256_text(sentence):
            raise V7IntegrityError("verified C support sentence hash drifted")
        if verification.get("support_location") not in {"text", "title"}:
            raise V7IntegrityError("verified C support location is invalid")
        if verification.get("surface_match_mode") not in {
            "nfkc_casefold_exact",
            "punctuation_normalized",
        }:
            raise V7IntegrityError("verified C surface match mode is invalid")
        return True, verified_answer, telemetry
    if verified_answer not in {None, ""}:
        raise V7IntegrityError("unverified C answer carries a promoted answer surface")
    if verification.get("verified_answer") not in {None, ""}:
        raise V7IntegrityError("unverified verifier telemetry carries an answer surface")
    return False, None, telemetry


def _consume_c_answers(
    states: Sequence[dict[str, Any]],
    c_answers: Sequence[Mapping[str, Any]],
    *,
    expected_model_artifact: Mapping[str, Any] | None = None,
) -> None:
    expected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for state in states:
        for task in state.get("pending_c_tasks") or []:
            task_id = str(task.get("task_id") or "")
            if not task_id or task_id in expected:
                raise V7IntegrityError(f"duplicate/empty pending C task id: {task_id!r}")
            if frozenset(task) != C_TASK_KEYS:
                raise V7IntegrityError("pending C task fields differ from frozen contract")
            if _sha256_json(task["producer_passages"]) != task[
                "producer_passages_sha256"
            ]:
                raise V7IntegrityError("pending C task passage hash drift")
            expected[task_id] = (state, task)
    observed: dict[str, dict[str, Any]] = {}
    for index, raw_answer in enumerate(c_answers, start=1):
        if not isinstance(raw_answer, Mapping):
            raise V7IntegrityError(f"C answer row {index} is not an object")
        answer = deepcopy(dict(raw_answer))
        assert_gold_free_source(answer, label=f"C_answers[{index}]")
        task_id = str(answer.get("task_id") or "")
        if not task_id or task_id in observed:
            raise V7IntegrityError(f"duplicate/empty C answer task id: {task_id!r}")
        observed[task_id] = answer
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise V7IntegrityError(
            f"C answer/task cardinality mismatch; missing={missing}, extra={extra}"
        )
    for task_id, (state, task) in expected.items():
        verified, value, telemetry = _validate_c_answer(
            task,
            observed[task_id],
            expected_model_artifact=expected_model_artifact,
        )
        if verified and value is not None:
            _record_slot(state["slot_values_C"], task["producer_slot"], value)
        state["subanswer_telemetry"].append(
            {
                "task_id": task_id,
                "producer_slot": task["producer_slot"],
                "step_sha256": task["step_sha256"],
                "producer_passages_sha256": task["producer_passages_sha256"],
                "verified": verified,
                "promoted_value": value,
                "telemetry": telemetry,
                "gold_access": False,
            }
        )
        state["pending_c_tasks"] = [
            current
            for current in state["pending_c_tasks"]
            if current["task_id"] != task_id
        ]


def _missing_slots(state: Mapping[str, Any], record: Mapping[str, Any], arm: str) -> list[str]:
    values = state[f"slot_values_{arm}"]
    return [dependency for dependency in record["dependencies"] if not values.get(dependency)]


def _render_one_dependent_query(
    state: Mapping[str, Any], record: Mapping[str, Any], arm: str
) -> tuple[str, dict[str, Any]]:
    queries, telemetry = render_question_anchored_queries_v6(
        question=str(state["question"]),
        step=record["step"],
        target_type=str(state["target_type"]),
        slot_values=state[f"slot_values_{arm}"],
        max_variants=1,
    )
    if len(queries) != 1 or int(telemetry.get("query_count", -1)) != 1:
        raise V7IntegrityError("dependent renderer did not return exactly one query")
    if telemetry.get("mode") != "hint_branches":
        raise V7IntegrityError("active dependent hop unexpectedly used a no-hint fallback")
    query = queries[0]
    prefix = str(state["question"]) + "\n"
    if not query.startswith(prefix) or not query[len(prefix) :].strip():
        raise V7IntegrityError("dependent query lost its exact full-question prefix")
    if dependency_refs(query):
        raise V7IntegrityError("dependent query retains an unresolved placeholder")
    return query, dict(telemetry)


def _pair_skip_reason(
    state: Mapping[str, Any], record: Mapping[str, Any]
) -> str | None:
    missing_b = _missing_slots(state, record, "B")
    missing_c = _missing_slots(state, record, "C")
    if not missing_b and not missing_c:
        return None
    parts: list[str] = []
    if missing_b:
        parts.append("B_missing=" + ",".join(missing_b))
    if missing_c:
        parts.append("C_missing=" + ",".join(missing_c))
    return "paired_missing_dependency:" + ";".join(parts)


def execute_dependent_stage(
    source_states: Sequence[Mapping[str, Any]],
    c_answers: Sequence[Mapping[str, Any]],
    *,
    producer_depth: int,
    retriever: Any,
    cross_encoder: Any,
    expected_model_artifact: Mapping[str, Any] | None = None,
    locked_plan_rows: Mapping[str, Mapping[str, Any]] | None = None,
    expected_run_locks: Mapping[str, Mapping[str, Any]] | None = None,
) -> StageResult:
    """Consume one C-answer depth and execute the next paired logical depth."""

    if isinstance(producer_depth, bool) or producer_depth < 1:
        raise V7IntegrityError("producer_depth must be a positive integer")
    states = [deepcopy(dict(state)) for state in source_states]
    seen_keys: set[str] = set()
    for state in states:
        _validate_state(
            state,
            locked_plan_rows=locked_plan_rows,
            expected_run_locks=expected_run_locks,
            expected_model_artifact=expected_model_artifact,
        )
        key = _question_key(state)
        if key in seen_keys:
            raise V7IntegrityError(f"duplicate state identity: {key}")
        seen_keys.add(key)
        if int(state["max_dependency_depth"]) > producer_depth and int(
            state["completed_depth"]
        ) != producer_depth:
            raise V7IntegrityError(
                f"state depth mismatch for {key}: expected {producer_depth}, "
                f"observed {state['completed_depth']}"
            )
    _consume_c_answers(
        states,
        c_answers,
        expected_model_artifact=expected_model_artifact,
    )

    target_depth = producer_depth + 1
    active: list[tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Any], dict[str, Any]]] = []
    budget_rows: list[dict[str, Any]] = []
    target_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    # Reserve queries while scanning the whole logical depth.  Deferring this
    # check until retrieval/results are committed lets two sibling hops in the
    # same state enqueue the same query because neither is present in the
    # state's issued-query ledger yet.  Such duplicate pairs are a preregistered
    # skip, not extra retrieval-budget padding.
    reserved_queries_b: dict[str, set[str]] = {
        _question_key(state): set(state["issued_dependent_queries_B"])
        for state in states
    }
    reserved_queries_c: dict[str, set[str]] = {
        _question_key(state): set(state["issued_dependent_queries_C"])
        for state in states
    }
    for state in states:
        if not state["plan_executable"] or int(state["max_dependency_depth"]) < target_depth:
            continue
        state_key = _question_key(state)
        for record in state["schedule"]:
            if int(record["dependency_depth"]) != target_depth:
                continue
            target_records.append((state, record))
            reason = _pair_skip_reason(state, record)
            if reason is None:
                query_b, render_b = _render_one_dependent_query(state, record, "B")
                query_c, render_c = _render_one_dependent_query(state, record, "C")
                if (
                    query_b in reserved_queries_b[state_key]
                    or query_c in reserved_queries_c[state_key]
                ):
                    reason = "paired_duplicate_or_noop_query"
                elif query_b == state["question"] or query_c == state["question"]:
                    reason = "paired_duplicate_or_noop_query"
            if reason is not None:
                budget = _dependent_budget_row(
                    state, record, active=False, reason=reason
                )
                state["budget_ledger"].append(deepcopy(budget))
                budget_rows.append(deepcopy(budget))
                state["hop_telemetry"].append(
                    {
                        "logical_hop_id": record["slot"],
                        "logical_hop_sha256": _logical_hop_hash(
                            state["question_key"], record
                        ),
                        "step_index": record["step_index"],
                        "dependency_depth": target_depth,
                        "dependencies": list(record["dependencies"]),
                        "paired_active": False,
                        "paired_skip_reason": reason,
                        "B": None,
                        "C": None,
                        "trajectory_semantics": {
                            "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
                            "paired_skipped": True,
                            "per_hop_logical_retrieval_budget_equal": True,
                        },
                        "gold_access": False,
                    }
                )
                continue
            reserved_queries_b[state_key].add(query_b)
            reserved_queries_c[state_key].add(query_c)
            active.append((state, record, query_b, query_c, render_b, render_c))

    # Separate calls are intentional.  Cross-arm-identical strings retain two
    # actual dependent retrievals and can never share candidate objects.
    queries_b = [item[2] for item in active]
    queries_c = [item[3] for item in active]
    raw_b = _batch_search(retriever, queries_b, label="dependent-B")
    raw_c = _batch_search(retriever, queries_c, label="dependent-C")
    ranked_b = [
        _rerank_one_query(query, raw, cross_encoder=cross_encoder, topk=STEP_RERANK_TOPK)
        for query, raw in zip(queries_b, raw_b)
    ]
    ranked_c = [
        _rerank_one_query(query, raw, cross_encoder=cross_encoder, topk=STEP_RERANK_TOPK)
        for query, raw in zip(queries_c, raw_c)
    ]

    next_tasks: list[dict[str, Any]] = []
    for item, b_result, c_result in zip(active, ranked_b, ranked_c):
        state, record, query_b, query_c, render_b, render_c = item
        passages_b, rerank_b = b_result
        passages_c, rerank_c = c_result
        same_query = _sha256_text(query_b) == _sha256_text(query_c)
        passages_b_hash = _sha256_json(passages_b)
        passages_c_hash = _sha256_json(passages_c)
        if same_query and (
            passages_b_hash != passages_c_hash or passages_b != passages_c
        ):
            raise V7IntegrityError(
                "identical B/C dependent queries produced non-identical producer passages"
            )
        hint_b = {
            "semantic_role": "retrieval_query_hint_not_asserted_fact",
            "dependency_values": {
                dependency: state["slot_values_B"][dependency]
                for dependency in record["dependencies"]
            },
        }
        hint_c = {
            "semantic_role": "mechanically_verified_subanswer",
            "verification_scope": "surface_locality_not_semantic_entailment",
            "dependency_values": {
                dependency: state["slot_values_C"][dependency]
                for dependency in record["dependencies"]
            },
        }
        state["hop_results_B"].append(
            _hop_result(
                arm="B",
                state=state,
                record=record,
                query=query_b,
                passages=passages_b,
                hint=hint_b,
            )
        )
        state["hop_results_C"].append(
            _hop_result(
                arm="C",
                state=state,
                record=record,
                query=query_c,
                passages=passages_c,
                hint=hint_c,
            )
        )
        state["issued_dependent_queries_B"].append(query_b)
        state["issued_dependent_queries_C"].append(query_c)
        state["successful_paired_dependent_hops"] += 1

        entity_candidates = (
            extract_deterministic_bridge_candidates(
                query_b,
                passages_b,
                max_docs=BRIDGE_MAX_DOCS,
                max_candidates=1,
                max_body_chars=BRIDGE_MAX_BODY_CHARS,
            )
            if record["consumers"]
            else []
        )
        if entity_candidates:
            _record_slot(
                state["slot_values_B"],
                record["slot"],
                str(entity_candidates[0]["surface"]),
            )
        task, projection_telemetry = (
            _make_c_task(state, record, passages_c)
            if record["consumers"] and passages_c
            else (None, None)
        )
        if task is not None:
            state["pending_c_tasks"].append(deepcopy(task))
            next_tasks.append(deepcopy(task))
        budget = _dependent_budget_row(
            state,
            record,
            active=True,
            reason=None,
            query_b=query_b,
            query_c=query_c,
        )
        state["budget_ledger"].append(deepcopy(budget))
        budget_rows.append(deepcopy(budget))
        state["hop_telemetry"].append(
            {
                "logical_hop_id": record["slot"],
                "logical_hop_sha256": _logical_hop_hash(
                    state["question_key"], record
                ),
                "step_index": record["step_index"],
                "dependency_depth": target_depth,
                "dependencies": list(record["dependencies"]),
                "paired_active": True,
                "paired_skip_reason": None,
                "B": {
                    "query": query_b,
                    "query_sha256": _sha256_text(query_b),
                    "query_renderer": render_b,
                    "rerank": rerank_b,
                    "produced_entity_candidate": deepcopy(entity_candidates[0])
                    if entity_candidates
                    else None,
                },
                "C": {
                    "query": query_c,
                    "query_sha256": _sha256_text(query_c),
                    "query_renderer": render_c,
                    "rerank": rerank_c,
                    "produced_task_id": task["task_id"] if task else None,
                    "producer_projection": deepcopy(projection_telemetry),
                },
                "trajectory_semantics": {
                    "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
                    "shared_query": same_query,
                    "B_query_sha256": _sha256_text(query_b),
                    "C_query_sha256": _sha256_text(query_c),
                    "B_producer_passages_sha256": passages_b_hash,
                    "C_producer_passages_sha256": passages_c_hash,
                    "producer_passages_identical_when_query_identical": (
                        passages_b_hash == passages_c_hash and passages_b == passages_c
                        if same_query
                        else None
                    ),
                    "downstream_context_is_causal_mediator": not same_query,
                },
                "gold_access": False,
            }
        )

    for state in states:
        if int(state["max_dependency_depth"]) >= target_depth and state[
            "plan_executable"
        ]:
            state["completed_depth"] = target_depth
        state["schema_version"] = STATE_SCHEMA_VERSION
        state["pending_c_tasks"].sort(key=lambda task: task["producer_slot"])
        state["hop_results_B"].sort(key=lambda hop: int(hop["step_index"]))
        state["hop_results_C"].sort(key=lambda hop: int(hop["step_index"]))
        state["hop_telemetry"].sort(key=lambda hop: int(hop["step_index"]))
        if not state["plan_executable"]:
            state["execution_status"] = "fallback_plan_invalid"
            state["fallback_reason"] = "plan_invalid"
        elif not state["has_dependent_step"]:
            # A max-depth=1 plan is a permanent canonical-A fallback.  It must
            # never be relabelled merely because a deeper stage is advancing
            # other rows in the same mixed-depth batch.
            state["execution_status"] = "fallback_no_dependent_step"
            state["fallback_reason"] = "no_dependent_step"
        elif int(state["completed_depth"]) >= int(state["max_dependency_depth"]):
            state["execution_status"] = "dependent_retrieval_complete"
            state["fallback_reason"] = None
        else:
            state["execution_status"] = "depth_complete"
        _validate_state(
            state,
            locked_plan_rows=locked_plan_rows,
            expected_run_locks=expected_run_locks,
            expected_model_artifact=expected_model_artifact,
        )

    all_complete = all(
        (not state["plan_executable"])
        or int(state["completed_depth"]) >= int(state["max_dependency_depth"])
        for state in states
    )
    final = _finalize_states(states, cross_encoder=cross_encoder) if all_complete else None
    next_tasks.sort(key=lambda task: (task["question_key"], task["producer_slot"]))
    budget_rows.sort(
        key=lambda row: (row["question_key"], row["dependency_depth"], row["logical_hop_id"])
    )
    return StageResult(
        states=states,
        c_tasks=next_tasks,
        budget_rows=budget_rows,
        arm_a_rows=final[0] if final else None,
        arm_b_rows=final[1] if final else None,
        arm_c_rows=final[2] if final else None,
        execution_rows=final[3] if final else None,
    )


def _full_question_scores(
    state: Mapping[str, Any], arm: str, *, cross_encoder: Any
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    original = list(state["arm_a"]["retrieved_passages"])
    unique: dict[str, dict[str, Any]] = {}
    for passage in original[PROTECTED_A_PREFIX:]:
        unique.setdefault(passage_score_key(passage), deepcopy(dict(passage)))
    for hop in state[f"hop_results_{arm}"]:
        if not hop["is_dependent"]:
            continue
        for variant in hop["query_variants"]:
            for passage in list(variant["passages"])[:CANDIDATES_PER_DEPENDENT_QUERY]:
                unique.setdefault(passage_score_key(passage), deepcopy(dict(passage)))
    question = str(state["question"])
    pairs = [
        (question, _passage_text(passage)[:CE_MAX_CHARS])
        for passage in unique.values()
    ]
    scores = _predict_scores(cross_encoder, pairs)
    result = {key: score for key, score in zip(unique, scores)}
    trace = [
        {
            "arm": arm,
            "question": question,
            "question_sha256": _sha256_text(question),
            "document_key": key,
            "document_id": _document_id(passage, location=f"final_ce.{arm}"),
            "document_text_sha256": _sha256_text(_passage_text(passage)[:CE_MAX_CHARS]),
            "score": result[key],
            "uses_exact_original_question": True,
        }
        for key, passage in unique.items()
    ]
    return result, trace


def _merge_safety(
    state: Mapping[str, Any], arm: str, output: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    original = list(state["arm_a"]["retrieved_passages"])
    original_keys = [passage_score_key(passage) for passage in original]
    output_keys = [passage_score_key(passage) for passage in output]
    root_keys = {
        passage_score_key(passage)
        for hop in state[f"hop_results_{arm}"]
        if not hop["is_dependent"]
        for variant in hop["query_variants"]
        for passage in variant["passages"]
    }
    dependent_keys = {
        passage_score_key(passage)
        for hop in state[f"hop_results_{arm}"]
        if hop["is_dependent"]
        for variant in hop["query_variants"]
        for passage in list(variant["passages"])[:CANDIDATES_PER_DEPENDENT_QUERY]
    }
    prefix_exact = list(output[:PROTECTED_A_PREFIX]) == original[:PROTECTED_A_PREFIX]
    root_only = (set(output_keys) - set(original_keys)) & (root_keys - dependent_keys)
    return {
        "output_count": len(output),
        "prefix8_exact": prefix_exact,
        "unauthorized_original_displacements": 0 if prefix_exact else 1,
        "root_passages_injected": len(root_only),
        "root_only_injected_document_keys": sorted(root_only),
        "duplicate_output_documents": len(output_keys) - len(set(output_keys)),
        "fallback_exact": list(output) == original,
    }


def _final_arm_row(
    state: Mapping[str, Any], arm: str, passages: Sequence[Mapping[str, Any]], trace: Mapping[str, Any]
) -> dict[str, Any]:
    base = state["arm_a"]
    return {
        "schema_version": ARM_SCHEMA_VERSION,
        "row_id": base["row_id"],
        "question_key": base["question_key"],
        "dataset": base["dataset"],
        "qid": base["qid"],
        "question": base["question"],
        "question_sha256": base["question_sha256"],
        "family_sha256": base["family_sha256"],
        "role": base["role"],
        "arm": arm,
        "retrieved_passages": [deepcopy(dict(value)) for value in passages],
        "passages_sha256": _sha256_historical_json(list(passages)),
        "kg_subgraph": [],
        "fallback_to_a": list(passages) == list(base["retrieved_passages"]),
        "retrieval_trace": deepcopy(dict(trace)),
        "gold_access": False,
    }


def _finalize_states(
    states: Sequence[Mapping[str, Any]], *, cross_encoder: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    arms_a: list[dict[str, Any]] = []
    arms_b: list[dict[str, Any]] = []
    arms_c: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for raw_state in states:
        state = deepcopy(dict(raw_state))
        _validate_state(state)
        original = deepcopy(list(state["arm_a"]["retrieved_passages"]))
        if int(state["successful_paired_dependent_hops"]) == 0:
            outputs = {"B": deepcopy(original), "C": deepcopy(original)}
            merge = {"B": None, "C": None}
            ce_trace = {"B": [], "C": []}
            fallback_reason = "zero_successful_paired_dependent_hops"
        else:
            outputs: dict[str, list[dict[str, Any]]] = {}
            merge: dict[str, Any] = {}
            ce_trace: dict[str, list[dict[str, Any]]] = {}
            for arm in ("B", "C"):
                scores, trace = _full_question_scores(
                    state, arm, cross_encoder=cross_encoder
                )
                merged, merge_trace = merge_dependent_passages_v6(
                    original,
                    state[f"hop_results_{arm}"],
                    scores,
                    protected_originals=PROTECTED_A_PREFIX,
                    candidates_per_query_variant=CANDIDATES_PER_DEPENDENT_QUERY,
                    total_passages=TOTAL_PASSAGES,
                )
                outputs[arm] = [deepcopy(dict(value)) for value in merged]
                merge[arm] = merge_trace
                ce_trace[arm] = trace
            fallback_reason = None
        safety = {
            arm: _merge_safety(state, arm, outputs[arm]) for arm in ("B", "C")
        }
        for arm in ("B", "C"):
            current = safety[arm]
            if (
                current["output_count"] != TOTAL_PASSAGES
                or not current["prefix8_exact"]
                or current["unauthorized_original_displacements"] != 0
                or current["root_passages_injected"] != 0
                or current["duplicate_output_documents"] != 0
            ):
                raise V7IntegrityError(
                    f"final merge safety invariant failed for {state['question_key']}::{arm}: {current}"
                )
        if int(state["successful_paired_dependent_hops"]) == 0:
            if outputs["B"] != original or outputs["C"] != original:
                raise V7IntegrityError("zero-hop fallback is not byte-exact A")
        trace_b = {
            "successful_paired_dependent_hops": state[
                "successful_paired_dependent_hops"
            ],
            "dependent_query_count": len(state["issued_dependent_queries_B"]),
            "fallback_reason": fallback_reason,
            "merge": merge["B"],
            "safety": safety["B"],
            "final_ce_trace": ce_trace["B"],
            "gold_access": False,
        }
        trace_c = {
            "successful_paired_dependent_hops": state[
                "successful_paired_dependent_hops"
            ],
            "dependent_query_count": len(state["issued_dependent_queries_C"]),
            "fallback_reason": fallback_reason,
            "merge": merge["C"],
            "safety": safety["C"],
            "final_ce_trace": ce_trace["C"],
            "gold_access": False,
        }
        arms_a.append(deepcopy(dict(state["arm_a"])))
        arms_b.append(
            _final_arm_row(state, "B_entity_hint_top1", outputs["B"], trace_b)
        )
        arms_c.append(
            _final_arm_row(state, "C_verified_subanswer", outputs["C"], trace_c)
        )
        detail = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "question_key": state["question_key"],
            "dataset": state["dataset"],
            "qid": state["qid"],
            "question": state["question"],
            "question_sha256": state["question_sha256"],
            "family_sha256": state["family_sha256"],
            "target_type": state["target_type"],
            "plan_sha256": state["plan_sha256"],
            "plan_executable": state["plan_executable"],
            "plan_validation_errors": state["plan_validation_errors"],
            "has_dependent_step": state["has_dependent_step"],
            "execution_status": state["execution_status"],
            "fallback_reason": fallback_reason or state.get("fallback_reason"),
            "successful_paired_dependent_hops": state[
                "successful_paired_dependent_hops"
            ],
            "hop_telemetry": state["hop_telemetry"],
            "subanswer_telemetry": state["subanswer_telemetry"],
            "budget_ledger": state["budget_ledger"],
            "merge": merge,
            "safety": safety,
            "arm_a_passages_sha256": state["arm_a"]["passages_sha256"],
            "arm_b_passages_sha256": _sha256_historical_json(outputs["B"]),
            "arm_c_passages_sha256": _sha256_historical_json(outputs["C"]),
            "all_dependent_queries_start_with_exact_original_question": all(
                query.startswith(state["question"] + "\n")
                for arm in ("B", "C")
                for query in state[f"issued_dependent_queries_{arm}"]
            ),
            "all_final_ce_pairs_use_exact_original_question": all(
                row["uses_exact_original_question"]
                and row["question"] == state["question"]
                for arm in ("B", "C")
                for row in ce_trace[arm]
            ),
            "gold_access": False,
        }
        details.append(detail)
    return arms_a, arms_b, arms_c, details


def validate_design_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != DESIGN_PROTOCOL_SCHEMA:
        raise V7IntegrityError("unexpected v7 design protocol schema")
    if protocol.get("status") != DESIGN_PROTOCOL_STATUS:
        raise V7IntegrityError("v7 design protocol is not frozen")
    population = protocol.get("population") or {}
    if population.get("datasets_in_order") != list(DATASETS):
        raise V7IntegrityError("v7 dataset order differs from the frozen design")
    planner = protocol.get("planner") or {}
    if int(planner.get("maximum_plan_steps", -1)) != MAX_PLAN_STEPS:
        raise V7IntegrityError("v7 plan-step cap differs from the frozen design")
    arms = protocol.get("arms") or {}
    for arm_name in ("B_entity_hint_top1", "C_verified_subanswer"):
        arm = arms.get(arm_name) or {}
        if int(arm.get("max_query_variants_per_logical_hop", -1)) != 1:
            raise V7IntegrityError(
                f"v7 query-variant cap differs from the frozen design: {arm_name}"
            )
    paired = protocol.get("paired_execution") or {}
    merge = protocol.get("retrieval_and_merge") or {}
    expected_merge = {
        "step_rerank_topk": STEP_RERANK_TOPK,
        "candidates_per_dependent_query": CANDIDATES_PER_DEPENDENT_QUERY,
        "final_passages": TOTAL_PASSAGES,
        "protected_A_prefix": PROTECTED_A_PREFIX,
        "maximum_replacements": TOTAL_PASSAGES - PROTECTED_A_PREFIX,
    }
    for key, expected in expected_merge.items():
        if int(merge.get(key, -1)) != expected:
            raise V7IntegrityError(f"v7 frozen merge setting differs: {key}")
    gold = protocol.get("gold_policy") or {}
    if set(gold.get("forbidden_recursive_keys") or []) != set(FORBIDDEN_GOLD_KEYS):
        raise V7IntegrityError("v7 frozen Gold-key set differs from runner")
    if gold.get("freeze_planner_retrieval_subanswer_and_merge_may_read_gold") is not False:
        raise V7IntegrityError("v7 protocol unexpectedly permits Gold access")


def _load_design_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise V7IntegrityError("v7 protocol is not an object")
    validate_design_protocol(protocol)
    return protocol, _file_lock(resolved)


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V7IntegrityError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V7IntegrityError(f"{label} is not an object")
    return value, _file_lock(resolved)


def load_execution_authorization(
    *,
    design_protocol_path: Path,
    preregistration_path: Path,
    truncation_addendum_path: Path,
    trajectory_semantics_addendum_path: Path,
    implementation_lock_path: Path,
    execution_lock_path: Path,
    experiment_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the post-plan authorization before any retrieval/model load."""

    _, design_lock = _load_design_protocol(design_protocol_path)
    prereg, prereg_lock = _load_json_object(
        preregistration_path, label="v7 preregistration"
    )
    addendum, addendum_lock = _load_json_object(
        truncation_addendum_path, label="v7 truncation addendum"
    )
    trajectory, trajectory_lock = _load_json_object(
        trajectory_semantics_addendum_path,
        label="v7 recursive-trajectory semantics addendum",
    )
    implementation, implementation_lock = _load_json_object(
        implementation_lock_path, label="v7 implementation lock"
    )
    execution, execution_lock = _load_json_object(
        execution_lock_path, label="v7 post-plan execution lock"
    )
    if prereg.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise V7IntegrityError("unexpected v7 preregistration schema")
    if prereg.get("execution_authorization") != "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK":
        raise V7IntegrityError("unexpected v7 preregistration authorization boundary")
    if addendum.get("schema_version") != TRUNCATION_ADDENDUM_SCHEMA:
        raise V7IntegrityError("unexpected v7 truncation addendum schema")
    if addendum.get("status") != "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL":
        raise V7IntegrityError("v7 truncation addendum is not frozen")
    if (
        trajectory.get("schema_version") != TRAJECTORY_SEMANTICS_ADDENDUM_SCHEMA
        or trajectory.get("status") != TRAJECTORY_SEMANTICS_ADDENDUM_STATUS
        or trajectory.get("gold_access") is not False
        or trajectory.get("execution_authorization")
        != "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise V7IntegrityError("v7 recursive-trajectory addendum is not frozen")
    if trajectory.get("effective_invariants") != {
        "shared_root_identical_query_requires_identical_producer_passages": True,
        "divergent_upstream_bridges_may_induce_arm_specific_producer_passages": True,
        "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
        "per_hop_logical_retrieval_budget_equal": True,
    }:
        raise V7IntegrityError("v7 recursive-trajectory estimand/invariants differ")
    trajectory_parents = trajectory.get("parents")
    if not isinstance(trajectory_parents, Mapping):
        raise V7IntegrityError("v7 recursive-trajectory addendum lacks parent locks")
    for name, lock in trajectory_parents.items():
        if not isinstance(lock, Mapping):
            raise V7IntegrityError(f"invalid recursive-trajectory parent lock: {name}")
        _assert_file_lock(
            lock,
            Path(str(lock.get("path") or "")),
            label=f"trajectory_semantics.parents.{name}",
        )
    if trajectory_parents.get("design_protocol") != design_lock:
        raise V7IntegrityError("recursive-trajectory design parent differs")
    if trajectory_parents.get("parent_preregistration") != prereg_lock:
        raise V7IntegrityError("recursive-trajectory preregistration parent differs")
    if trajectory_parents.get("producer_truncation_addendum") != addendum_lock:
        raise V7IntegrityError("recursive-trajectory truncation parent differs")
    if implementation.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA:
        raise V7IntegrityError("unexpected v7 implementation-lock schema")
    if implementation.get("status") != IMPLEMENTATION_LOCK_STATUS:
        raise V7IntegrityError("v7 implementation lock has not authorized planner completion")
    if implementation.get("experiment_id") != (
        prereg.get("future_experiment_ids") or {}
    ).get("implementation_lock"):
        raise V7IntegrityError("v7 implementation-lock Experiment ID differs")
    if execution.get("schema_version") != EXECUTION_LOCK_SCHEMA:
        raise V7IntegrityError("unexpected v7 post-plan execution-lock schema")
    if execution.get("status") != EXECUTION_LOCK_STATUS:
        raise V7IntegrityError("v7 Gold-free materialization is not authorized")
    if execution.get("experiment_id") != EXECUTION_LOCK_EXPERIMENT_ID:
        raise V7IntegrityError("v7 post-plan execution-lock Experiment ID differs")
    if implementation.get("scope") != EXECUTION_SCOPE or execution.get(
        "scope"
    ) != EXECUTION_SCOPE:
        raise V7IntegrityError("v7 implementation/post-plan scope differs")
    for label, document in (
        ("preregistration", prereg),
        ("truncation_addendum", addendum),
        ("trajectory_semantics_addendum", trajectory),
        ("implementation_lock", implementation),
        ("execution_lock", execution),
    ):
        if document.get("gold_access") is not False:
            raise V7IntegrityError(f"{label} gold_access is not false")

    parents = execution.get("parents")
    if not isinstance(parents, Mapping):
        raise V7IntegrityError("post-plan execution lock lacks parents")
    for name, current in (
        ("preregistration", prereg_lock),
        ("truncation_addendum", addendum_lock),
        ("implementation_lock", implementation_lock),
    ):
        expected = parents.get(name)
        if not isinstance(expected, Mapping) or dict(expected) != current:
            raise V7IntegrityError(f"post-plan parent lock mismatch: {name}")

    implementation_parents = implementation.get("parents")
    if not isinstance(implementation_parents, Mapping):
        raise V7IntegrityError("implementation lock lacks parents")
    for name, current in (
        ("preregistration", prereg_lock),
        ("truncation_addendum", addendum_lock),
        ("trajectory_semantics_addendum", trajectory_lock),
    ):
        expected = implementation_parents.get(name)
        if not isinstance(expected, Mapping) or dict(expected) != current:
            raise V7IntegrityError(f"implementation parent lock mismatch: {name}")

    addendum_parent = (addendum.get("parents") or {}).get("parent_preregistration")
    if not isinstance(addendum_parent, Mapping) or dict(addendum_parent) != prereg_lock:
        raise V7IntegrityError("truncation addendum parent preregistration differs")

    implementation_authorization = implementation.get("authorization")
    if implementation_authorization != {
        "planner": True,
        "gold_free_materialization": False,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }:
        raise V7IntegrityError("implementation authorization flags differ")
    if (implementation.get("content_reverification") or {}).get(
        "full_hash_verification_performed"
    ) is not True:
        raise V7IntegrityError("implementation lock did not fully re-hash content")

    authorization = execution.get("authorization")
    expected_authorization = {
        "planner_complete": True,
        "gold_free_materialization": True,
        "gold_attachment": False,
        "answer_evaluation": False,
        "training": False,
    }
    if authorization != expected_authorization:
        raise V7IntegrityError("post-plan authorization flags differ")
    execution_trajectory_parent = (execution.get("parents") or {}).get(
        "trajectory_semantics_addendum"
    )
    if execution_trajectory_parent != trajectory_lock:
        raise V7IntegrityError("post-plan recursive-trajectory parent lock differs")
    contract = execution.get("materialization_contract")
    if not isinstance(contract, Mapping):
        raise V7IntegrityError("post-plan lock lacks materialization_contract")
    expected_id = str(
        (prereg.get("future_experiment_ids") or {}).get("materialization") or ""
    )
    if not expected_id or str(contract.get("experiment_id") or "") != expected_id:
        raise V7IntegrityError("materialization Experiment ID is not preregistered")
    if str(experiment_id) != expected_id:
        raise V7IntegrityError("requested Experiment ID differs from execution lock")
    if contract.get("runner_version") != RUNNER_VERSION:
        raise V7IntegrityError("runner version differs from execution lock")
    if int(contract.get("n", -1)) != 40 or contract.get("by_dataset") != {
        "hotpotqa": 20,
        "musique": 20,
    }:
        raise V7IntegrityError("materialization population differs from frozen 20+20")
    if contract.get("gold_access") is not False or contract.get("network_access") is not False:
        raise V7IntegrityError("materialization contract permits Gold/network access")
    if int(contract.get("max_plan_steps", -1)) != MAX_PLAN_STEPS:
        raise V7IntegrityError("materialization max_plan_steps differs")

    runtime_code = execution.get("runtime_code")
    if not isinstance(runtime_code, Mapping):
        raise V7IntegrityError("post-plan lock lacks runtime_code")
    implementation_runtime = implementation.get("runtime_code")
    if (
        not isinstance(implementation_runtime, Mapping)
        or set(runtime_code) != EXPECTED_RUNTIME_CODE_ROLES
        or dict(runtime_code) != dict(implementation_runtime)
    ):
        raise V7IntegrityError(
            "post-plan runtime code differs from the implementation lock"
        )
    for name, lock in runtime_code.items():
        if not isinstance(lock, Mapping):
            raise V7IntegrityError(f"runtime-code lock is invalid: {name}")
        _assert_file_lock(
            lock,
            Path(str(lock.get("path") or "")),
            label=f"runtime_code.{name}",
        )
    if Path(str(runtime_code["retrieval_runner"]["path"])).resolve() != Path(
        __file__
    ).resolve():
        raise V7IntegrityError("runtime_code.retrieval_runner points to another file")

    _assert_file_lock(
        execution.get("lock_issuer"),
        PLAN_LOCK_ISSUER,
        label="post_plan.lock_issuer",
    )
    frozen_inputs = execution.get("inputs")
    expected_input_roles = {
        "development",
        "planner_cohort",
        "canonical_A_contexts",
        "planner_predictions",
        "planner_report",
        "planner_manifest",
    }
    if not isinstance(frozen_inputs, Mapping) or set(frozen_inputs) != expected_input_roles:
        raise V7IntegrityError("post-plan input-lock role set differs")
    for name, lock in frozen_inputs.items():
        if not isinstance(lock, Mapping):
            raise V7IntegrityError(f"post-plan input lock is invalid: {name}")
        _assert_file_lock(
            lock,
            Path(str(lock.get("path") or "")),
            label=f"post_plan.inputs.{name}",
        )

    population = execution.get("population")
    prereg_population = prereg.get("population") or {}
    if not isinstance(population, Mapping):
        raise V7IntegrityError("post-plan lock lacks population")
    if (
        int(population.get("n", -1)) != 40
        or population.get("by_dataset") != {"hotpotqa": 20, "musique": 20}
        or population.get("question_key_order_sha256")
        != prereg_population.get("question_key_order_sha256")
        or population.get("plan_executable_gate_pass") is not True
    ):
        raise V7IntegrityError("post-plan population/order/executable gate differs")
    valid_counts = population.get("schema_valid")
    valid_rates = population.get("schema_valid_rate")
    if not isinstance(valid_counts, Mapping) or not isinstance(valid_rates, Mapping):
        raise V7IntegrityError("post-plan schema-valid population telemetry is missing")
    for dataset in DATASETS:
        count = valid_counts.get(dataset)
        rate = valid_rates.get(dataset)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > 20
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isclose(float(rate), count / 20.0, rel_tol=0.0, abs_tol=1e-12)
            or float(rate) < 0.8
        ):
            raise V7IntegrityError(
                f"post-plan schema-valid rate differs for {dataset}"
            )
    return execution, {
        "design_protocol": design_lock,
        "preregistration": prereg_lock,
        "truncation_addendum": addendum_lock,
        "trajectory_semantics_addendum": trajectory_lock,
        "implementation_lock": implementation_lock,
        "post_plan_execution_lock": execution_lock,
    }


def validate_locked_root_inputs(
    execution_lock: Mapping[str, Any],
    *,
    cohort_path: Path,
    contexts_path: Path,
    plan_paths: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    inputs = execution_lock.get("inputs")
    if not isinstance(inputs, Mapping):
        raise V7IntegrityError("post-plan lock lacks input locks")
    if len(plan_paths) != 1:
        raise V7IntegrityError("formal v7 materialization requires one locked plan file")
    return {
        "development": _assert_file_lock(
            inputs.get("development"), cohort_path, label="inputs.development"
        ),
        "canonical_A_contexts": _assert_file_lock(
            inputs.get("canonical_A_contexts"),
            contexts_path,
            label="inputs.canonical_A_contexts",
        ),
        "planner_predictions": _assert_file_lock(
            inputs.get("planner_predictions"),
            plan_paths[0],
            label="inputs.planner_predictions",
        ),
    }


def _locked_plan_index(
    execution_lock: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    inputs = execution_lock.get("inputs")
    prediction_lock = inputs.get("planner_predictions") if isinstance(inputs, Mapping) else None
    if not isinstance(prediction_lock, Mapping):
        raise V7IntegrityError("post-plan lock lacks planner-prediction content lock")
    prediction_path = Path(str(prediction_lock.get("path") or ""))
    _assert_file_lock(
        prediction_lock, prediction_path, label="post_plan.inputs.planner_predictions"
    )
    rows = _read_jsonl(prediction_path)
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        assert_gold_free_source(row, label=f"locked_plans[{index}]")
        key = _question_key(row)
        if key in indexed:
            raise V7IntegrityError(f"duplicate locked planner identity: {key}")
        indexed[key] = row
    population = execution_lock.get("population") or {}
    if len(indexed) != int(population.get("n", -1)):
        raise V7IntegrityError("locked planner population differs from plan lock")
    return indexed


def validate_runtime_arguments(
    args: argparse.Namespace, preregistration: Mapping[str, Any]
) -> None:
    if int(args.expected_docs) != int(
        (preregistration.get("retrieval_assets") or {}).get("expected_documents", -1)
    ):
        raise V7IntegrityError("expected Wiki18 document count differs from preregistration")
    if int(args.rrf_candidate_k) != 100:
        raise V7IntegrityError("v7 rrf_candidate_k must remain 100")
    locked_assets = preregistration.get("retrieval_assets") or {}
    for argument_name, lock_name in (
        ("corpus_path", "corpus"),
        ("dense_index_path", "dense_index"),
        ("bm25_index_path", "bm25_index"),
    ):
        locked = locked_assets.get(lock_name) or {}
        if Path(str(getattr(args, argument_name))).expanduser().resolve() != Path(
            str(locked.get("path") or "")
        ).expanduser().resolve():
            raise V7IntegrityError(f"runtime path differs from preregistration: {argument_name}")
    expected_ce = Path(
        str((preregistration.get("models") or {}).get("cross_encoder", {}).get("path") or "")
    ).expanduser().resolve()
    if Path(args.cross_encoder_model).expanduser().resolve() != expected_ce:
        raise V7IntegrityError("cross-encoder path differs from preregistration")


def _load_retrieval_runtime(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    # Imports stay inside the real CLI boundary, so unit tests and reader stages
    # never initialize a retriever, CE model, CUDA, or the global CE cache.
    from kgproweight.retrieval.reranker import get_cross_encoder
    from scripts.pilot.audit_iterative_bridge_retrieval import (
        _build_retriever,
        _validate_full_wiki18_assets,
    )

    assets = _validate_full_wiki18_assets(
        args.corpus_path,
        args.dense_index_path,
        args.bm25_index_path,
        expected_docs=int(args.expected_docs),
    )
    retriever = _build_retriever(
        DATASETS[0],
        int(args.rrf_candidate_k),
        corpus_path=args.corpus_path,
        dense_index_path=args.dense_index_path,
        bm25_index_path=args.bm25_index_path,
    )
    cross_encoder = get_cross_encoder(str(args.cross_encoder_model))
    return retriever, cross_encoder, assets


def _state_input_for_depth(output_dir: Path, producer_depth: int) -> Path:
    return (
        output_dir / "root_state.jsonl"
        if producer_depth == 1
        else output_dir / f"state.depth_{producer_depth}.jsonl"
    )


def _stage_descriptor_for_state_depth(output_dir: Path, state_depth: int) -> Path:
    if isinstance(state_depth, bool) or not isinstance(state_depth, int) or state_depth < 1:
        raise V7IntegrityError("state depth must be a positive integer")
    return (
        output_dir / "roots_stage.json"
        if state_depth == 1
        else output_dir / f"dependents_stage.depth_{state_depth - 1}.json"
    )


def _validate_parent_stage_chain(
    output_dir: Path,
    *,
    state_depth: int,
    experiment_id: str,
    runtime_locks: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate the append-only descriptor chain and return its state lock."""

    descriptor_path = _stage_descriptor_for_state_depth(output_dir, state_depth)
    descriptor, descriptor_lock = _load_json_object(
        descriptor_path, label=f"v7 stage descriptor for state depth {state_depth}"
    )
    assert_gold_free_source(descriptor, label=f"stage_descriptor.depth_{state_depth}")
    if (
        descriptor.get("schema_version") != STAGE_DESCRIPTOR_SCHEMA_VERSION
        or descriptor.get("runner_version") != RUNNER_VERSION
        or descriptor.get("experiment_id") != experiment_id
        or descriptor.get("gold_access") is not False
        or descriptor.get("runtime_locks") != dict(runtime_locks)
    ):
        raise V7IntegrityError(f"stage descriptor contract drift at state depth {state_depth}")
    outputs = descriptor.get("outputs")
    if not isinstance(outputs, Mapping):
        raise V7IntegrityError(f"stage descriptor lacks outputs at depth {state_depth}")
    output_name = "root_state" if state_depth == 1 else "state"
    state_lock = outputs.get(output_name)
    expected_state_path = _state_input_for_depth(output_dir, state_depth)
    current_state_lock = _assert_file_lock(
        state_lock,
        expected_state_path,
        label=f"stage.depth_{state_depth}.outputs.{output_name}",
    )
    if state_depth == 1:
        if descriptor.get("stage") != "roots" or descriptor.get(
            "parent_stage_descriptor"
        ) is not None:
            raise V7IntegrityError("root stage descriptor has an invalid parent/stage")
        if descriptor.get("state_depth") != 1:
            raise V7IntegrityError("root stage descriptor depth differs")
    else:
        if (
            descriptor.get("stage") != "dependents"
            or descriptor.get("producer_depth") != state_depth - 1
            or descriptor.get("target_depth") != state_depth
            or descriptor.get("state_depth") != state_depth
        ):
            raise V7IntegrityError(f"dependent descriptor depth drift at {state_depth}")
        _, previous_descriptor_lock, previous_state_lock = _validate_parent_stage_chain(
            output_dir,
            state_depth=state_depth - 1,
            experiment_id=experiment_id,
            runtime_locks=runtime_locks,
        )
        if descriptor.get("parent_stage_descriptor") != previous_descriptor_lock:
            raise V7IntegrityError(f"parent descriptor hash drift at state depth {state_depth}")
        if descriptor.get("input_state") != previous_state_lock:
            raise V7IntegrityError(f"parent state hash drift at state depth {state_depth}")
    task_lock = outputs.get("c_tasks")
    if not isinstance(task_lock, Mapping):
        raise V7IntegrityError(f"stage descriptor lacks C-task lock at depth {state_depth}")
    _assert_file_lock(
        task_lock,
        output_dir / f"c_tasks.depth_{state_depth}.jsonl",
        label=f"stage.depth_{state_depth}.outputs.c_tasks",
    )
    return descriptor_path, descriptor_lock, current_state_lock


def _report_from_final(
    result: StageResult,
    *,
    experiment_id: str,
    runtime_locks: Mapping[str, Mapping[str, Any]],
    assets: Mapping[str, Any],
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    details = result.execution_rows or []
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        current = [row for row in details if row["dataset"] == dataset]
        active_hops = sum(
            int(row["successful_paired_dependent_hops"]) for row in current
        )
        eligible_hops = sum(
            1
            for row in current
            for hop in row["hop_telemetry"]
            if hop.get("dependencies")
        )
        reader_attempts = [
            attempt
            for row in current
            for attempt in row["subanswer_telemetry"]
        ]
        strict_parsed = sum(
            bool(
                attempt.get("telemetry", {})
                .get("strict_parse", {})
                .get("valid")
            )
            for attempt in reader_attempts
        )
        verified = sum(bool(attempt.get("verified")) for attempt in reader_attempts)
        changed_b = sum(
            row["arm_b_passages_sha256"] != row["arm_a_passages_sha256"]
            for row in current
        )
        changed_c = sum(
            row["arm_c_passages_sha256"] != row["arm_a_passages_sha256"]
            for row in current
        )
        by_dataset[dataset] = {
            "n": len(current),
            "plan_executable": sum(bool(row["plan_executable"]) for row in current),
            "plan_executable_rate": (
                sum(bool(row["plan_executable"]) for row in current) / len(current)
                if current
                else None
            ),
            "paired_dependent_hops_active": active_hops,
            "paired_dependent_hops_eligible": eligible_hops,
            "paired_dependent_hop_activation_rate": (
                active_hops / eligible_hops if eligible_hops else None
            ),
            "subanswer_reader_attempts": len(reader_attempts),
            "strict_subanswer_json_parse_count": strict_parsed,
            "strict_subanswer_json_parse_rate": (
                strict_parsed / len(reader_attempts) if reader_attempts else None
            ),
            "mechanically_verified_subanswer_count": verified,
            "mechanically_verified_subanswer_rate": (
                verified / len(reader_attempts) if reader_attempts else None
            ),
            "B_changed": changed_b,
            "C_changed": changed_c,
            "retained_new_dependent_document_question_rate_B": (
                changed_b / len(current) if current else None
            ),
            "retained_new_dependent_document_question_rate_C": (
                changed_c / len(current) if current else None
            ),
        }
    duplicate_documents = sum(
        int(row["safety"][arm]["duplicate_output_documents"])
        for row in details
        for arm in ("B", "C")
    )
    unauthorized_displacements = sum(
        int(row["safety"][arm]["unauthorized_original_displacements"])
        for row in details
        for arm in ("B", "C")
    )
    root_only_injected = sum(
        int(row["safety"][arm]["root_passages_injected"])
        for row in details
        for arm in ("B", "C")
    )
    safety_summary = {
        "runtime_errors": 0,
        "identity_join_rate": 1.0,
        "recursive_forbidden_input_fields": 0,
        "gold_access": False,
        "all_rows_and_arms_top10": (
            len(details) == len(result.states)
            and all(
                len(state["arm_a"]["retrieved_passages"]) == TOTAL_PASSAGES
                for state in result.states
            )
            and all(
                all(
                    row["safety"][arm]["output_count"] == TOTAL_PASSAGES
                    for arm in ("B", "C")
                )
                for row in details
            )
        ),
        "duplicate_output_documents": duplicate_documents,
        "unauthorized_A_prefix_displacements": unauthorized_displacements,
        "root_only_documents_injected": root_only_injected,
        "all_dependent_queries_start_with_exact_original_question": all(
            row["all_dependent_queries_start_with_exact_original_question"]
            for row in details
        ),
        "all_final_CE_pairs_use_exact_original_question": all(
            row["all_final_ce_pairs_use_exact_original_question"]
            for row in details
        ),
        "B_C_query_budget_equal_every_question_depth_and_hop": all(
            budget["budget_equal"]
            and budget["B"]["logical_query_count"]
            == budget["C"]["logical_query_count"]
            and budget["B"]["logical_query_count"] in {0, 1}
            for row in details
            for budget in row["budget_ledger"]
        ),
        "budget_padding_queries": 0,
        "unverified_subanswers_used": 0,
        "fallback_pair_and_A_byte_exact": all(
            int(row["successful_paired_dependent_hops"]) > 0
            or (
                row["arm_a_passages_sha256"] == row["arm_b_passages_sha256"]
                == row["arm_c_passages_sha256"]
            )
            for row in details
        ),
    }
    materialization_passed = (
        safety_summary["runtime_errors"] == 0
        and safety_summary["identity_join_rate"] == 1.0
        and safety_summary["recursive_forbidden_input_fields"] == 0
        and safety_summary["gold_access"] is False
        and safety_summary["all_rows_and_arms_top10"]
        and safety_summary["duplicate_output_documents"] == 0
        and safety_summary["unauthorized_A_prefix_displacements"] == 0
        and safety_summary["root_only_documents_injected"] == 0
        and safety_summary["all_dependent_queries_start_with_exact_original_question"]
        and safety_summary["all_final_CE_pairs_use_exact_original_question"]
        and safety_summary["B_C_query_budget_equal_every_question_depth_and_hop"]
        and safety_summary["budget_padding_queries"] == 0
        and safety_summary["unverified_subanswers_used"] == 0
        and safety_summary["fallback_pair_and_A_byte_exact"]
    )
    mechanism_thresholds = {
        "plan_executable_rate": 0.8,
        "strict_subanswer_json_parse_rate": 0.5,
        "mechanically_verified_subanswer_rate": 0.4,
        "paired_dependent_hop_activation_rate": 0.4,
        "retained_new_dependent_document_question_rate_B": 0.25,
        "retained_new_dependent_document_question_rate_C": 0.25,
    }
    mechanism_by_dataset: dict[str, Any] = {}
    for dataset, observed in by_dataset.items():
        checks = {}
        for metric, minimum in mechanism_thresholds.items():
            value = observed[metric]
            checks[metric] = {
                "observed": float(value) if value is not None else None,
                "minimum": minimum,
                "passed": value is not None and float(value) >= minimum,
            }
        mechanism_by_dataset[dataset] = {
            "checks": checks,
            "passed": all(check["passed"] for check in checks.values()),
        }
    mechanism_passed = all(
        value["passed"] for value in mechanism_by_dataset.values()
    )
    all_budgets = [
        budget for row in details for budget in row["budget_ledger"]
    ]
    retrieval_accounting = {
        "logical_query_slots_B": sum(
            int(row["B"]["logical_query_count"]) for row in all_budgets
        ),
        "logical_query_slots_C": sum(
            int(row["C"]["logical_query_count"]) for row in all_budgets
        ),
        "root_shared_physical_searches": sum(
            int(row["actual_shared_physical_search_count"])
            for row in all_budgets
        ),
        "dependent_independent_physical_searches": sum(
            int(row["actual_independent_physical_search_count"])
            for row in all_budgets
        ),
        "unique_root_query_strings": len(
            {
                row["B"]["query_sha256"]
                for row in all_budgets
                if row["is_root"] and row["B"]["query_sha256"]
            }
        ),
        "unique_dependent_query_strings_B": len(
            {
                row["B"]["query_sha256"]
                for row in all_budgets
                if not row["is_root"] and row["B"]["query_sha256"]
            }
        ),
        "unique_dependent_query_strings_C": len(
            {
                row["C"]["query_sha256"]
                for row in all_budgets
                if not row["is_root"] and row["C"]["query_sha256"]
            }
        ),
        "subanswer_model_calls_C_only": sum(
            int(value["subanswer_reader_attempts"]) for value in by_dataset.values()
        ),
        "claim_boundary": "retrieval-query budget matched; total compute is not matched",
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "query_renderer_version": QUERY_RENDERER_VERSION,
        "merge_policy_version": MERGE_POLICY_VERSION,
        "experiment_id": experiment_id,
        "status": "COMPLETE_GOLD_FREE_MATERIALIZATION",
        "development_only": True,
        "gold_access": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": dict(runtime_locks["preregistration"]),
        "truncation_addendum": dict(runtime_locks["truncation_addendum"]),
        "trajectory_semantics_addendum": dict(
            runtime_locks["trajectory_semantics_addendum"]
        ),
        "implementation_lock": dict(runtime_locks["implementation_lock"]),
        "plan_lock": dict(
            runtime_locks["post_plan_execution_lock"]
        ),
        "design_protocol": dict(runtime_locks["design_protocol"]),
        "retrieval_assets": dict(assets),
        "by_dataset": by_dataset,
        "retrieval_accounting": retrieval_accounting,
        "trajectory_semantics": {
            "addendum": dict(runtime_locks["trajectory_semantics_addendum"]),
            "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
            "shared_root_and_identical_query_producer_passages_identical": all(
                (
                    hop.get("trajectory_semantics", {}).get("shared_query") is not True
                    or hop.get("trajectory_semantics", {}).get(
                        "producer_passages_identical_when_query_identical"
                    )
                    is True
                )
                for row in details
                for hop in row.get("hop_telemetry", [])
            ),
            "divergent_recursive_contexts_are_causal_mediators": True,
            "per_hop_logical_retrieval_budget_equal": all(
                row.get("budget_equal") is True for row in all_budgets
            ),
            "total_compute_matched": False,
        },
        "safety_summary": safety_summary,
        "materialization_gate": {
            "passed": materialization_passed,
            "observed": safety_summary,
        },
        "gold_free_mechanism_gate": {
            "passed": mechanism_passed,
            "by_dataset": mechanism_by_dataset,
        },
        "gate_decision": (
            "PASS_READY_FOR_SEPARATE_GOLD_FINALIZER"
            if materialization_passed and mechanism_passed
            else "FAIL_STOP_BEFORE_GOLD"
        ),
        "outputs": {
            name: _file_lock(path) for name, path in output_paths.items()
        },
        "scientific_boundary": (
            "Development-only Gold-free feasibility materialisation. No answer "
            "labels, final generation, utility score, confirmation, or paper claim."
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("roots", "dependents"), required=True)
    parser.add_argument("--depth", type=int, help="producer depth for --stage dependents")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--plans", type=Path, nargs="+")
    parser.add_argument("--state_path", type=Path)
    parser.add_argument("--c_answers", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_EXECUTION_LOCK,
        help="post-plan execution lock authorizing Gold-free materialization",
    )
    parser.add_argument(
        "--design_protocol", type=Path, default=DEFAULT_DESIGN_PROTOCOL
    )
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument(
        "--truncation_addendum", type=Path, default=DEFAULT_TRUNCATION_ADDENDUM
    )
    parser.add_argument(
        "--trajectory_semantics_addendum",
        type=Path,
        default=DEFAULT_TRAJECTORY_SEMANTICS_ADDENDUM,
    )
    parser.add_argument(
        "--implementation_lock", type=Path, default=DEFAULT_IMPLEMENTATION_LOCK
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--corpus_path", default="indexes_wiki18/corpus_flashrag.jsonl")
    parser.add_argument("--dense_index_path", default="indexes_wiki18/e5_fp16.dat")
    parser.add_argument("--bm25_index_path", default="indexes_wiki18/bm25")
    parser.add_argument("--expected_docs", type=int, default=21_015_324)
    parser.add_argument("--rrf_candidate_k", type=int, default=100)
    parser.add_argument("--cross_encoder_model", default="models/bge-reranker-v2-m3")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate frozen inputs/state only; never instantiate retrieval or CE",
    )
    return parser.parse_args()


def validate_stage_cli_boundary(args: argparse.Namespace) -> None:
    """Reject alternate formal state sources before reading any execution asset."""

    if args.stage == "roots":
        if args.depth is not None or args.state_path is not None or args.c_answers is not None:
            raise SystemExit("roots stage does not accept --depth/--state_path/--c_answers")
        if args.cohort is None or args.contexts is None or not args.plans:
            raise SystemExit("roots stage requires --cohort --contexts --plans")
        return
    if args.depth is None or args.depth < 1:
        raise SystemExit("dependents stage requires a positive --depth")
    if args.cohort is not None or args.contexts is not None or args.plans:
        raise SystemExit("dependents stage does not accept cohort/context/plan inputs")
    if args.state_path is not None and not args.dry_run:
        raise SystemExit("formal dependents stage forbids --state_path; use the descriptor chain")


def main() -> None:
    args = parse_args()
    validate_stage_cli_boundary(args)
    execution_lock, runtime_locks = load_execution_authorization(
        design_protocol_path=args.design_protocol,
        preregistration_path=args.preregistration,
        truncation_addendum_path=args.truncation_addendum,
        trajectory_semantics_addendum_path=args.trajectory_semantics_addendum,
        implementation_lock_path=args.implementation_lock,
        execution_lock_path=args.protocol,
        experiment_id=args.experiment_id,
    )
    preregistration, _ = _load_json_object(
        args.preregistration, label="v7 preregistration"
    )
    implementation_document, _ = _load_json_object(
        args.implementation_lock, label="v7 implementation lock"
    )
    expected_model_artifact = _expected_subanswer_model_artifact(
        implementation_document,
        implementation_lock=runtime_locks["implementation_lock"],
        plan_lock=runtime_locks["post_plan_execution_lock"],
    )
    locked_plan_rows = _locked_plan_index(execution_lock)
    validate_runtime_arguments(args, preregistration)
    output_dir = args.output_dir.expanduser().resolve()
    if args.stage == "roots":
        if args.depth is not None or args.state_path is not None or args.c_answers is not None:
            raise SystemExit("roots stage does not accept --depth/--state_path/--c_answers")
        if args.cohort is None or args.contexts is None or not args.plans:
            raise SystemExit("roots stage requires --cohort --contexts --plans")
        locked_inputs = validate_locked_root_inputs(
            execution_lock,
            cohort_path=args.cohort,
            contexts_path=args.contexts,
            plan_paths=args.plans,
        )
        cohort = _read_jsonl(args.cohort)
        contexts = _read_jsonl(args.contexts)
        plans = [row for path in args.plans for row in _read_jsonl(path)]
        rows = assemble_root_rows(cohort, contexts, plans)
        counts = {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in DATASETS
        }
        if len(rows) != 40 or counts != {"hotpotqa": 20, "musique": 20}:
            raise V7IntegrityError(
                f"root input population differs from frozen 20+20: {counts}"
            )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "DRY_RUN_OK",
                        "stage": "roots",
                        "n": len(rows),
                        "gold_access": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if output_dir.exists():
            raise FileExistsError(
                f"append-only roots output directory already exists: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        retriever, cross_encoder, assets = _load_retrieval_runtime(args)
        result = execute_root_stage(
            rows,
            retriever,
            cross_encoder=cross_encoder,
            run_locks=runtime_locks,
            locked_plan_rows=locked_plan_rows,
        )
        _write_jsonl_new(output_dir / "arm_a.jsonl", result.arm_a_rows or [])
        _write_jsonl_new(output_dir / "root_state.jsonl", result.states)
        _write_jsonl_new(output_dir / "c_tasks.depth_1.jsonl", result.c_tasks)
        _write_jsonl_new(output_dir / "budget_ledger.roots.jsonl", result.budget_rows)
        _write_json_new(
            output_dir / "roots_stage.json",
            {
                "schema_version": STAGE_DESCRIPTOR_SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "experiment_id": args.experiment_id,
                "stage": "roots",
                "state_depth": 1,
                "parent_stage_descriptor": None,
                "gold_access": False,
                "runtime_locks": runtime_locks,
                "input_locks": locked_inputs,
                "retrieval_assets": assets,
                "trajectory_semantics": {
                    "addendum": runtime_locks["trajectory_semantics_addendum"],
                    "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
                    "shared_root_producer_passages_identical": True,
                },
                "outputs": {
                    name: _file_lock(
                        output_dir / filename,
                        allow_empty=name in {"c_tasks", "budget"},
                    )
                    for name, filename in {
                        "arm_a": "arm_a.jsonl",
                        "root_state": "root_state.jsonl",
                        "c_tasks": "c_tasks.depth_1.jsonl",
                        "budget": "budget_ledger.roots.jsonl",
                    }.items()
                },
            },
        )
        return

    if args.depth is None or args.depth < 1:
        raise SystemExit("dependents stage requires a positive --depth")
    if args.cohort is not None or args.contexts is not None or args.plans:
        raise SystemExit("dependents stage does not accept cohort/context/plan inputs")
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.state_path is not None and not args.dry_run:
        raise SystemExit("formal dependents stage forbids --state_path; use the descriptor chain")
    if args.state_path is None:
        parent_descriptor_path, parent_descriptor_lock, input_state_lock = (
            _validate_parent_stage_chain(
                output_dir,
                state_depth=args.depth,
                experiment_id=args.experiment_id,
                runtime_locks=runtime_locks,
            )
        )
        state_path = Path(str(input_state_lock["path"]))
    else:
        parent_descriptor_path = None
        parent_descriptor_lock = None
        state_path = args.state_path.expanduser().resolve()
        input_state_lock = _file_lock(state_path)
    states = _read_jsonl(state_path)
    for state in states:
        _validate_state(
            state,
            locked_plan_rows=locked_plan_rows,
            expected_run_locks=runtime_locks,
            expected_model_artifact=expected_model_artifact,
        )
    expected_tasks = sum(len(state.get("pending_c_tasks") or []) for state in states)
    answer_path: Path | None = None
    if expected_tasks:
        answer_path = (
            args.c_answers.expanduser().resolve()
            if args.c_answers is not None
            else output_dir / f"c_answers.depth_{args.depth}.jsonl"
        )
        c_answers = _read_jsonl(answer_path)
    else:
        if args.c_answers is not None:
            if not args.dry_run:
                raise SystemExit("formal stage forbids --c_answers when no C tasks are pending")
            c_answers = _read_jsonl(args.c_answers)
        else:
            c_answers = []
    if args.dry_run:
        copied = [deepcopy(dict(state)) for state in states]
        _consume_c_answers(
            copied,
            c_answers,
            expected_model_artifact=expected_model_artifact,
        )
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "stage": "dependents",
                    "producer_depth": args.depth,
                    "n": len(states),
                    "c_answers": len(c_answers),
                    "gold_access": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    retriever, cross_encoder, assets = _load_retrieval_runtime(args)
    result = execute_dependent_stage(
        states,
        c_answers,
        producer_depth=args.depth,
        retriever=retriever,
        cross_encoder=cross_encoder,
        expected_model_artifact=expected_model_artifact,
        locked_plan_rows=locked_plan_rows,
        expected_run_locks=runtime_locks,
    )
    target_depth = args.depth + 1
    state_output = output_dir / f"state.depth_{target_depth}.jsonl"
    task_output = output_dir / f"c_tasks.depth_{target_depth}.jsonl"
    budget_output = output_dir / f"budget_ledger.depth_{target_depth}.jsonl"
    _write_jsonl_new(state_output, result.states)
    _write_jsonl_new(task_output, result.c_tasks)
    _write_jsonl_new(budget_output, result.budget_rows)
    final = result.arm_b_rows is not None
    outputs: dict[str, Path] = {
        "state": state_output,
        "c_tasks": task_output,
        "budget": budget_output,
    }
    if final:
        stored_arm_a = _read_jsonl(output_dir / "arm_a.jsonl")
        if stored_arm_a != (result.arm_a_rows or []):
            raise V7IntegrityError(
                "stored root-stage Arm A differs from cumulative final state"
            )
        outputs.update(
            {
                "arm_b": output_dir / "arm_b.jsonl",
                "arm_c": output_dir / "arm_c.jsonl",
                "execution_details": output_dir / "execution_details.jsonl",
                "budget_all": output_dir / "budget_ledger.jsonl",
            }
        )
        _write_jsonl_new(outputs["arm_b"], result.arm_b_rows or [])
        _write_jsonl_new(outputs["arm_c"], result.arm_c_rows or [])
        _write_jsonl_new(outputs["execution_details"], result.execution_rows or [])
        _write_jsonl_new(
            outputs["budget_all"],
            [
                deepcopy(row)
                for state in result.states
                for row in state["budget_ledger"]
            ],
        )
        report = _report_from_final(
            result,
            experiment_id=args.experiment_id,
            runtime_locks=runtime_locks,
            assets=assets,
            output_paths={
                "arm_a": output_dir / "arm_a.jsonl",
                "arm_b": outputs["arm_b"],
                "arm_c": outputs["arm_c"],
                "execution_details": outputs["execution_details"],
                "budget_ledger": outputs["budget_all"],
            },
        )
        _write_json_new(output_dir / "report.json", report)
        outputs["report"] = output_dir / "report.json"
    _write_json_new(
        output_dir / f"dependents_stage.depth_{args.depth}.json",
        {
            "schema_version": STAGE_DESCRIPTOR_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "experiment_id": args.experiment_id,
            "stage": "dependents",
            "producer_depth": args.depth,
            "target_depth": target_depth,
            "state_depth": target_depth,
            "final": final,
            "gold_access": False,
            "runtime_locks": runtime_locks,
            "parent_stage_descriptor": parent_descriptor_lock,
            "retrieval_assets": assets,
            "input_state": input_state_lock,
            "input_c_answers": _file_lock(answer_path) if answer_path is not None else None,
            "trajectory_semantics": {
                "addendum": runtime_locks["trajectory_semantics_addendum"],
                "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
                "arm_specific_recursive_contexts_are_causal_mediators": True,
            },
            "outputs": {
                name: _file_lock(
                    path, allow_empty=name in {"c_tasks", "budget"}
                )
                for name, path in outputs.items()
            },
        },
    )


# Private aliases keep the fake-only tests terse and make the stage boundary
# discoverable to later preregistration tooling.
_execute_roots = execute_root_stage
_execute_dependents = execute_dependent_stage


if __name__ == "__main__":
    main()
