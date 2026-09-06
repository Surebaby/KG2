"""Strict schema and Gold-free diagnostics for Query Controller v1 actions.

The controller is trained to emit one JSON ``target`` object.  A complete
training record additionally binds that target to an identity, an explicit
controller state, source provenance, and a Gold-boundary declaration.  This
module deliberately contains no dataset loader, model, retriever, or answer
scorer.  It can therefore be reused by the data builder, trainer preflight,
and a later greedy-prediction scorer without opening evaluation Gold.

Validation here is mechanical.  In particular, ``state_use_valid`` means that
the verified intermediate answer is present in a dependent q2 query; it is not
a semantic-entailment judgment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval.dynamic_decomposition_v8 import (
    QueryParseError,
    parse_query_response,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


SCHEMA_VERSION = "query-controller-action-v1"
STATE_VERSION = "query-controller-state-v1"
SOURCE_ACTION = "text"
DATASETS = frozenset({"2wikimultihopqa", "musique", "hotpotqa"})
# ``confirmation`` is an action-release role, not a trainer input.  The
# trainer is required to load only train/dev; allowing the value here lets a
# held-out, train-side action set reuse the same canonical record validator.
ACTION_SPLITS = frozenset({"train", "dev", "confirmation"})
SLOTS = frozenset({"q1", "q2_dynamic"})

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "example_id",
        "dataset",
        "qid",
        "question_key",
        "question_sha256",
        "family_sha256",
        "split",
        "slot",
        "turn_index",
        "state",
        "target",
        "source_provenance",
        "gold_boundary",
    }
)
STATE_FIELDS = frozenset(
    {"state_version", "original_question", "previous_actions", "verified_observations"}
)
TARGET_FIELDS = frozenset(
    {
        "action",
        "query",
        "anchor",
        "relation_intent",
        "pid",
        "dependencies",
        "output_slot",
        "source_action",
    }
)
GOLD_BOUNDARY_FIELDS = frozenset(
    {
        "train_intermediate_annotation_used",
        "gold_final_answer_visible",
        "evaluation_gold_access",
    }
)
PREVIOUS_ACTION_FIELDS = frozenset(
    {"slot", "action", "query", "output_slot"}
)
OBSERVATION_FIELDS = frozenset(
    {
        "answer",
        "answer_sha256",
        "evidence_excerpt",
        "evidence_excerpt_sha256",
        "document_id",
        "document_title",
        "sentence_index",
        "provenance",
    }
)
OBSERVATION_PROVENANCE_FIELDS = frozenset(
    {"source", "annotation_path", "binding_method"}
)
OBSERVATION_ANNOTATION_PATHS = frozenset(
    {
        "metadata.evidences.entity[0]",
        "metadata.metadata.question_decomposition[0].answer",
    }
)
OBSERVATION_BINDING_METHODS = frozenset(
    {
        "fact_title_and_answer_surface",
        "decomposition_step_support_answer_surface",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PID_RE = re.compile(r"P[1-9][0-9]*")
_PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_])(?:\$(?:hop|step)_?[1-9]\d*|#[1-9]\d*|"
    r"(?:hop|step)_[1-9]\d*)(?![A-Za-z0-9_])"
    r"|\{\{?\s*(?:answer|entity|subject|object|hop|step)(?:_[1-9]\d*)?\s*\}?\}"
    r"|<\s*(?:answer|entity|subject|object|hop|step)(?:_[1-9]\d*)?\s*>"
    r")",
    flags=re.IGNORECASE,
)


class ActionValidationError(ValueError):
    """One Controller-v1 action record violates the frozen contract."""

    def __init__(self, codes: Sequence[str], message: str | None = None):
        values = tuple(dict.fromkeys(str(code) for code in codes if str(code)))
        if not values:
            values = ("unknown_validation_error",)
        self.codes = values
        self.code = values[0]
        super().__init__(message or ", ".join(values))


def _safe_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\n" not in value
        and "\r" not in value
        and not any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value)
    )


def _surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _contains_surface(text: Any, needle: Any) -> bool:
    """Match a normalized surface on token boundaries, never raw substrings."""

    haystack = _surface(text)
    target = _surface(needle)
    return bool(target and f" {target} " in f" {haystack} ")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def audit_action_record(record: Any, *, expected_split: str | None = None) -> dict[str, Any]:
    """Return independent mechanical checks without raising.

    The independent checks make aggregate diagnostics useful even when one
    record has more than one defect.  ``validate_action_record`` is the strict
    fail-fast facade used by builders and trainers.
    """

    errors: list[str] = []
    checks = {
        "schema_valid": False,
        "identity_valid": False,
        "query_contract_valid": False,
        "query_nonrepeat": False,
        "placeholder_free": False,
        "dependency_closed": False,
        "source_action_valid": False,
        "state_use_valid": False,
        "gold_boundary_valid": False,
    }
    if not isinstance(record, Mapping):
        errors.append("record_not_object")
        return {"valid": False, "checks": checks, "errors": errors}

    if set(record) != TOP_LEVEL_FIELDS:
        errors.append("top_level_schema")
    state = _mapping(record.get("state"))
    target = _mapping(record.get("target"))
    source = _mapping(record.get("source_provenance"))
    gold = _mapping(record.get("gold_boundary"))
    if state is None or set(state) != STATE_FIELDS:
        errors.append("state_schema")
    if target is None or set(target) != TARGET_FIELDS:
        errors.append("target_schema")
    if source is None or not source:
        errors.append("source_provenance_schema")
    if gold is None or set(gold) != GOLD_BOUNDARY_FIELDS:
        errors.append("gold_boundary_schema")

    schema_version = record.get("schema_version")
    dataset = record.get("dataset")
    qid = record.get("qid")
    question = state.get("original_question") if state is not None else None
    split = record.get("split")
    slot = record.get("slot")
    turn_index = record.get("turn_index")
    if schema_version != SCHEMA_VERSION:
        errors.append("schema_version")
    if dataset not in DATASETS:
        errors.append("dataset")
    if not _safe_text(qid):
        errors.append("qid")
    if split not in ACTION_SPLITS or (expected_split is not None and split != expected_split):
        errors.append("split")
    if slot not in SLOTS:
        errors.append("slot")
    expected_turn = 1 if slot == "q1" else 2 if slot == "q2_dynamic" else None
    if type(turn_index) is not int or turn_index != expected_turn:
        errors.append("turn_index")
    if not _safe_text(record.get("example_id")):
        errors.append("example_id")
    if not _safe_text(question):
        errors.append("original_question")

    identity_ok = False
    if dataset in DATASETS and _safe_text(qid) and _safe_text(question):
        expected_key = f"{dataset}::{qid}"
        expected_example = f"{expected_key}::{slot}"
        identity_ok = (
            record.get("question_key") == expected_key
            and record.get("example_id") == expected_example
            and record.get("question_sha256") == question_sha256(question)
            and record.get("family_sha256") == family_sha256(question)
        )
    if not identity_ok:
        errors.append("identity_binding")
    checks["identity_valid"] = identity_ok

    state_shape_ok = state is not None and set(state) == STATE_FIELDS
    previous_value = state.get("previous_actions") if state_shape_ok else None
    observations_value = state.get("verified_observations") if state_shape_ok else None
    previous = previous_value if isinstance(previous_value, list) else None
    observations = observations_value if isinstance(observations_value, list) else None
    if state_shape_ok and state.get("state_version") != STATE_VERSION:
        errors.append("state_version")
    if previous is None:
        errors.append("previous_actions_type")
        previous = ()
    if observations is None:
        errors.append("verified_observations_type")
        observations = ()

    previous_queries: list[str] = []
    previous_output_slots: list[str] = []
    previous_ok = True
    for action in previous:
        if not isinstance(action, Mapping) or set(action) != PREVIOUS_ACTION_FIELDS:
            previous_ok = False
            continue
        if action.get("slot") != "q1" or action.get("action") != "retrieve":
            previous_ok = False
        if not _safe_text(action.get("query")) or action.get("output_slot") != "q1":
            previous_ok = False
        else:
            previous_queries.append(str(action["query"]))
            previous_output_slots.append(str(action["output_slot"]))
    if not previous_ok:
        errors.append("previous_action_schema")

    observation_ok = True
    observation_provenance_ok = True
    verified_answers: list[str] = []
    for observation in observations:
        if not isinstance(observation, Mapping) or set(observation) != OBSERVATION_FIELDS:
            observation_ok = False
            continue
        answer = observation.get("answer")
        excerpt = observation.get("evidence_excerpt")
        if not _safe_text(answer) or not _safe_text(excerpt):
            observation_ok = False
            continue
        if observation.get("answer_sha256") != _sha256_text(str(answer)):
            observation_ok = False
        if observation.get("evidence_excerpt_sha256") != _sha256_text(str(excerpt)):
            observation_ok = False
        if not _safe_text(observation.get("document_id")):
            observation_ok = False
        if not _safe_text(observation.get("document_title")):
            observation_ok = False
        if type(observation.get("sentence_index")) is not int or observation["sentence_index"] < 0:
            observation_ok = False
        provenance = observation.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or set(provenance) != OBSERVATION_PROVENANCE_FIELDS
            or provenance.get("source") != "train_annotation_support"
            or provenance.get("annotation_path") not in OBSERVATION_ANNOTATION_PATHS
            or provenance.get("binding_method") not in OBSERVATION_BINDING_METHODS
        ):
            observation_ok = False
            observation_provenance_ok = False
        verified_answers.append(str(answer))
    if not observation_ok:
        errors.append("observation_schema")
    if not observation_provenance_ok:
        errors.append("observation_provenance_schema")

    target_shape_ok = target is not None and set(target) == TARGET_FIELDS
    query = target.get("query") if target_shape_ok else None
    source_action_ok = target_shape_ok and target.get("source_action") == SOURCE_ACTION
    if not source_action_ok:
        errors.append("source_action_not_text")
    checks["source_action_valid"] = bool(source_action_ok)

    target_basic_ok = target_shape_ok
    if target_shape_ok:
        if target.get("action") != "retrieve":
            target_basic_ok = False
            errors.append("target_action")
        if not _safe_text(query):
            target_basic_ok = False
            errors.append("target_query")
        anchor = target.get("anchor")
        if anchor is not None and not _safe_text(anchor):
            target_basic_ok = False
            errors.append("target_anchor")
        if not _safe_text(target.get("relation_intent")):
            target_basic_ok = False
            errors.append("relation_intent")
        pid = target.get("pid")
        if pid is not None and (not isinstance(pid, str) or _PID_RE.fullmatch(pid) is None):
            target_basic_ok = False
            errors.append("pid")
        dependencies_value = target.get("dependencies")
        dependencies = dependencies_value if isinstance(dependencies_value, list) else None
        if dependencies is None or any(not _safe_text(item) for item in dependencies):
            target_basic_ok = False
            errors.append("dependencies_type")
            dependencies = ()
    else:
        dependencies = ()

    placeholder_ok = isinstance(query, str) and _PLACEHOLDER_RE.search(query) is None
    checks["placeholder_free"] = placeholder_ok
    if not placeholder_ok:
        errors.append("unresolved_placeholder")

    query_contract_ok = False
    query_nonrepeat = False
    if _safe_text(question) and _safe_text(query):
        history = [str(question), *previous_queries]
        query_nonrepeat = _surface(query) not in {_surface(item) for item in history}
        try:
            parse_query_response(str(query), previous_queries=history)
            query_contract_ok = True
        except QueryParseError as exc:
            errors.append(f"query_contract:{exc.code}")
    if not query_nonrepeat:
        errors.append("query_repeat")
    checks["query_contract_valid"] = query_contract_ok
    checks["query_nonrepeat"] = query_nonrepeat

    dependency_ok = False
    state_use_ok = False
    if slot == "q1":
        dependency_ok = (
            previous_ok
            and observation_ok
            and len(previous) == 0
            and len(observations) == 0
            and list(dependencies) == []
            and target_shape_ok
            and target.get("output_slot") == "q1"
        )
        state_use_ok = dependency_ok
    elif slot == "q2_dynamic":
        dependency_ok = (
            previous_ok
            and observation_ok
            and len(previous) == 1
            and len(observations) == 1
            and previous_output_slots == ["q1"]
            and list(dependencies) == ["q1"]
            and target_shape_ok
            and target.get("output_slot") == "q2"
        )
        answer_surface = _surface(verified_answers[0]) if len(verified_answers) == 1 else ""
        state_use_ok = bool(
            dependency_ok
            and answer_surface
            and _contains_surface(query, answer_surface)
        )
    if not dependency_ok:
        errors.append("dependency_not_closed")
    if not state_use_ok:
        errors.append("state_not_used")
    checks["dependency_closed"] = dependency_ok
    checks["state_use_valid"] = state_use_ok

    gold_ok = False
    if gold is not None and set(gold) == GOLD_BOUNDARY_FIELDS:
        expected_intermediate = slot == "q2_dynamic"
        gold_ok = (
            gold.get("train_intermediate_annotation_used") is expected_intermediate
            and gold.get("gold_final_answer_visible") is False
            and gold.get("evaluation_gold_access") is False
        )
    if not gold_ok:
        errors.append("gold_boundary")
    checks["gold_boundary_valid"] = gold_ok

    checks["schema_valid"] = bool(
        set(record) == TOP_LEVEL_FIELDS
        and state_shape_ok
        and target_shape_ok
        and source is not None
        and bool(source)
        and gold is not None
        and set(gold) == GOLD_BOUNDARY_FIELDS
        and schema_version == SCHEMA_VERSION
        and target_basic_ok
        and previous_ok
        and observation_ok
    )
    return {
        "valid": not errors,
        "checks": checks,
        "errors": list(dict.fromkeys(errors)),
    }


def validate_action_record(
    record: Any, *, expected_split: str | None = None
) -> dict[str, Any]:
    """Validate and return a defensive copy; raise with stable error codes."""

    result = audit_action_record(record, expected_split=expected_split)
    if not result["valid"]:
        raise ActionValidationError(result["errors"])
    return deepcopy(dict(record))


def parse_target_response(response_text: str, *, reference_record: Mapping[str, Any]) -> dict[str, Any]:
    """Parse an exact JSON target and validate it against a frozen state."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise ActionValidationError(("empty_prediction",))
    try:
        target = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ActionValidationError(("prediction_not_json",), str(exc)) from exc
    if not isinstance(target, Mapping):
        raise ActionValidationError(("prediction_not_object",))
    candidate = deepcopy(dict(reference_record))
    candidate["target"] = dict(target)
    validate_action_record(candidate, expected_split=str(reference_record.get("split")))
    return deepcopy(dict(target))


def evaluate_action_records(
    records: Sequence[Mapping[str, Any]], *, expected_split: str | None = None
) -> dict[str, Any]:
    """Aggregate schema/mechanism telemetry for a data release."""

    rows = list(records)
    audits = [audit_action_record(row, expected_split=expected_split) for row in rows]
    check_names = tuple(next(iter(audits), {"checks": {}})["checks"])
    counts = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_slot: dict[str, Counter[str]] = defaultdict(Counter)
    seen_examples: set[str] = set()
    duplicate_examples = 0
    for row, audit in zip(rows, audits):
        example_id = str(row.get("example_id") or "")
        if example_id in seen_examples:
            duplicate_examples += 1
        seen_examples.add(example_id)
        dataset = str(row.get("dataset") or "UNKNOWN")
        slot = str(row.get("slot") or "UNKNOWN")
        counts["rows"] += 1
        by_dataset[dataset]["rows"] += 1
        by_slot[slot]["rows"] += 1
        for name, passed in audit["checks"].items():
            counts[name] += int(passed)
            by_dataset[dataset][name] += int(passed)
            by_slot[slot][name] += int(passed)
        for code in audit["errors"]:
            counts[f"error:{code}"] += 1

    def project(counter: Counter[str]) -> dict[str, Any]:
        n = counter["rows"]
        return {
            "n": n,
            **{
                f"{name}_rate": counter[name] / n if n else 0.0
                for name in check_names
            },
        }

    return {
        "schema_version": "query-controller-action-audit-v1",
        "n": len(rows),
        "all_valid": all(item["valid"] for item in audits) and duplicate_examples == 0,
        "duplicate_example_id_count": duplicate_examples,
        "metrics": project(counts),
        "by_dataset": {key: project(value) for key, value in sorted(by_dataset.items())},
        "by_slot": {key: project(value) for key, value in sorted(by_slot.items())},
        "error_counts": {
            key.removeprefix("error:"): value
            for key, value in sorted(counts.items())
            if key.startswith("error:")
        },
    }


__all__ = [
    "ActionValidationError",
    "ACTION_SPLITS",
    "DATASETS",
    "GOLD_BOUNDARY_FIELDS",
    "OBSERVATION_ANNOTATION_PATHS",
    "OBSERVATION_BINDING_METHODS",
    "OBSERVATION_FIELDS",
    "OBSERVATION_PROVENANCE_FIELDS",
    "PREVIOUS_ACTION_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE_ACTION",
    "STATE_VERSION",
    "TARGET_FIELDS",
    "TOP_LEVEL_FIELDS",
    "audit_action_record",
    "evaluate_action_records",
    "parse_target_response",
    "validate_action_record",
]
