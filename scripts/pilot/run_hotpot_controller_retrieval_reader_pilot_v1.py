#!/usr/bin/env python
"""Run the Gold-free Hotpot Controller retrieval/Reader pilot.

The formal entry point consumes only the answer-free runtime projection frozen
by ``freeze_hotpot_controller_retrieval_reader_pilot_v1.py``.  In particular,
it has no argument for, and never opens, ``accepted_actions.jsonl`` or the
train-annotation observation used to construct silver labels.

For every accepted query plan the state machine is fixed to::

    q1 -> canonical Wiki18 top-10 -> strong-SFT canonical sub-QA
       -> v9.1 rank-first lexical provenance binding
       -> substitute the *predicted* bound answer into q2_template
       -> canonical Wiki18 top-10
       -> [bound q1 document] + up to nine novel q2 documents
          + q1-rank backfill to ten
       -> the same strong-SFT final Reader.

If q1 cannot be parsed and bound, q2 retrieval and final generation are not
executed.  There is no Gold correction or fallback observation.  The module
offers injectable runtimes for CPU-only tests; CUDA and the real index are
loaded only behind the explicit ``--execute_runtime`` latch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.data.parsers import extract_final_answer, parse_steps  # noqa: E402
from kgproweight.data.prompts import build_inference_messages  # noqa: E402
from kgproweight.kg.question_kg import question_sha256  # noqa: E402
from kgproweight.retrieval.canonical_subqa_v9_1 import (  # noqa: E402
    PROVENANCE_BINDER_VERSION,
    build_canonical_subqa_messages,
    parse_and_bind_canonical_subanswer,
    project_rank_first_binding_for_runtime,
)
from kgproweight.retrieval.dependent import (  # noqa: E402
    DependentRetrievalError,
    dependency_refs,
    replace_dependency_refs,
)
from kgproweight.retrieval.dynamic_decomposition_v8 import (  # noqa: E402
    DynamicDecompositionV8Error,
    parse_query_response,
    project_top10_passages_for_prompt,
)
from scripts.pilot import materialize_dynamic_decomposition_v8 as v8_runtime  # noqa: E402
from scripts.pilot import run_dynamic_decomposition_v8 as v8_driver  # noqa: E402


RUNNER_VERSION = "hotpot-controller-retrieval-reader-pilot-v1-runner-1"
INPUT_SCHEMA_VERSION = "hotpot-controller-runtime-query-plan-v1"
ROW_SCHEMA_VERSION = "hotpot-controller-retrieval-reader-row-v1"
REPORT_SCHEMA_VERSION = "hotpot-controller-retrieval-reader-report-v1"
RUNNING_MANIFEST_SCHEMA_VERSION = (
    "hotpot-controller-retrieval-reader-running-manifest-v1"
)
TERMINAL_MANIFEST_SCHEMA_VERSION = (
    "hotpot-controller-retrieval-reader-terminal-manifest-v1"
)
PROTOCOL_SCHEMA_VERSION = "hotpot-controller-retrieval-reader-protocol-v1"
PROTOCOL_STATUS = "FROZEN_GOLD_FREE_RETRIEVAL_READER_PILOT_NOT_RUN_NOT_TRAINED"

EXPERIMENT_ID = (
    "QUERY-CONTROLLER-HOTPOT-RETRIEVAL-READER-PILOT-"
    "SEED20260905-ATTEMPT001"
)
DEFAULT_PROTOCOL_DIR = Path(
    "outputs/audits/query_controller_hotpot_retrieval_reader_"
    "pilot30_protocol_seed20260905_v1"
)
DEFAULT_PROTOCOL_PATH = DEFAULT_PROTOCOL_DIR / "protocol.json"
DEFAULT_RUN_DIR = Path(
    "outputs/audits/query_controller_hotpot_retrieval_reader_"
    "pilot30_seed20260905_attempt001"
)

EXPECTED_TOP_K = 10
MAX_NEW_TOKENS = 512
SEED = 42
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_Q2_SLOT_RE = re.compile(r"(?<![A-Za-z0-9_])#1(?![A-Za-z0-9_])", re.IGNORECASE)
_INPUT_FIELDS = (
    "schema_version",
    "dataset",
    "qid",
    "question",
    "question_sha256",
    "q1_query",
    "q1_query_sha256",
    "q2_template",
    "q2_template_sha256",
    "proposal_sha256",
    "source_projection_row_sha256",
)


RUNTIME_CONTRACT: dict[str, Any] = {
    "schema_version": "hotpot-controller-retrieval-reader-runtime-contract-v1",
    "runner_version": RUNNER_VERSION,
    "dataset": "hotpotqa",
    "gold_access": False,
    "training": False,
    "seed": SEED,
    "input": {
        "schema_version": INPUT_SCHEMA_VERSION,
        "fields_exact": list(_INPUT_FIELDS),
        "accepted_generation_rows_only": True,
        "accepted_actions_or_annotation_observation_read": False,
    },
    "retrieval": {
        "runtime_class": (
            "scripts.pilot.materialize_dynamic_decomposition_v8."
            "CanonicalRetrieverRuntime"
        ),
        "stages": ["q1_all", "q2_verified_only"],
        "top_k": EXPECTED_TOP_K,
        "dense_top_k": 100,
        "bm25_top_k": 100,
        "rrf_k": 60,
        "rrf_output_k": 100,
        "bge_top_k": EXPECTED_TOP_K,
        "silent_fallback_allowed": False,
    },
    "reader": {
        "runtime_class": (
            "scripts.pilot.materialize_dynamic_decomposition_v8."
            "SharedHuggingFaceRuntime"
        ),
        "base_model_path": "models/llama3-8b",
        "adapter_path": (
            "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_"
            "no_text_head/final"
        ),
        "same_physical_runtime_for_q1_and_final": True,
        "role_for_q1_and_final": "final_reader",
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "kg_triples": [],
    },
    "q1": {
        "prompt_builder": (
            "kgproweight.retrieval.canonical_subqa_v9."
            "build_canonical_subqa_messages"
        ),
        "parser_and_binder": (
            "kgproweight.retrieval.canonical_subqa_v9_1."
            "parse_and_bind_canonical_subanswer"
        ),
        "binder_version": PROVENANCE_BINDER_VERSION,
        "failure_policy": "no_q2_retrieval_no_final_reader_no_gold_correction",
    },
    "q2": {
        "template_dependency_exact": "#1",
        "substitution": (
            "kgproweight.retrieval.dependent.replace_dependency_refs"
        ),
        "slot_values_source": "verified_q1_reader_prediction_only",
        "max_variants": 1,
    },
    "final_passages": {
        "allocation": (
            "one_verified_bound_q1_document_then_up_to_nine_q2_novel_"
            "then_q1_rank_backfill_to_exactly_ten"
        ),
        "bound_q1_document_first": True,
        "q2_novel_max": 9,
        "q1_backfill": True,
        "total_exact": EXPECTED_TOP_K,
    },
    "final": {
        "prompt_builder": "kgproweight.data.prompts.build_inference_messages",
        "question_source": "answer_free_runtime_projection.question",
        "answer_scoring": False,
    },
}


class HotpotRuntimeError(ValueError):
    """A frozen input, protocol, or runtime trace violates its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required nonempty file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _assert_file_lock(lock: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(lock, Mapping):
        raise HotpotRuntimeError(f"{label} lock is not an object")
    path = Path(str(lock.get("path") or "")).resolve()
    current = _file_lock(path)
    if any(current[field] != lock.get(field) for field in current):
        raise HotpotRuntimeError(f"{label} content drift")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HotpotRuntimeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise HotpotRuntimeError(
                    f"invalid JSONL framing at {path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HotpotRuntimeError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(dict(value), newline=True))


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row), newline=True))


def _safe_error_fields(exc: BaseException) -> dict[str, str]:
    """Bind a failure without serializing paths, queries, or secret-like text."""

    return {
        "error_type": type(exc).__name__,
        "error_message_sha256": _sha256_text(str(exc)),
    }


def runtime_contract() -> dict[str, Any]:
    """Return a defensive copy of the literal runtime contract."""

    expected = v8_runtime.runtime_contract()
    if (
        expected["shared_hf_runtime"]["base_model_path"]
        != RUNTIME_CONTRACT["reader"]["base_model_path"]
        or expected["shared_hf_runtime"]["strong_sft_adapter_path"]
        != RUNTIME_CONTRACT["reader"]["adapter_path"]
        or expected["shared_hf_runtime"]["role_max_new_tokens"]["final_reader"]
        != MAX_NEW_TOKENS
        or expected["canonical_retrieval"]["bge_top_k"] != EXPECTED_TOP_K
    ):
        raise HotpotRuntimeError("v8 runtime constants drifted from Hotpot contract")
    return deepcopy(RUNTIME_CONTRACT)


def _validate_runtime_input(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(_INPUT_FIELDS):
        raise HotpotRuntimeError("runtime input row field/order drift")
    clean = deepcopy(dict(row))
    if clean.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise HotpotRuntimeError("runtime input schema version drift")
    if clean.get("dataset") != "hotpotqa":
        raise HotpotRuntimeError("runtime input dataset must be hotpotqa")
    for field in ("qid", "question", "q1_query", "q2_template"):
        value = clean.get(field)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\n" in value
            or "\r" in value
        ):
            raise HotpotRuntimeError(f"runtime input {field} is invalid")
    for field in (
        "question_sha256",
        "q1_query_sha256",
        "q2_template_sha256",
        "proposal_sha256",
        "source_projection_row_sha256",
    ):
        if not isinstance(clean.get(field), str) or _SHA256_RE.fullmatch(
            clean[field]
        ) is None:
            raise HotpotRuntimeError(f"runtime input {field} is not a SHA256")
    if clean["question_sha256"] != question_sha256(clean["question"]):
        raise HotpotRuntimeError("runtime input question hash mismatch")
    if clean["q1_query_sha256"] != _sha256_text(clean["q1_query"]):
        raise HotpotRuntimeError("runtime input q1 hash mismatch")
    if clean["q2_template_sha256"] != _sha256_text(clean["q2_template"]):
        raise HotpotRuntimeError("runtime input q2 template hash mismatch")
    proposal = {
        "schema_version": "hotpot-controller-query-proposal-v1",
        "q1": clean["q1_query"],
        "q2_template": clean["q2_template"],
    }
    if clean["proposal_sha256"] != _sha256_value(proposal):
        raise HotpotRuntimeError("runtime input proposal hash mismatch")
    # Validate both the one-hop surface contract and the unresolved dependency
    # without ever constructing q2 from a train annotation.
    parse_query_response(clean["q1_query"], previous_queries=(clean["question"],))
    if (
        len(_Q2_SLOT_RE.findall(clean["q2_template"])) != 1
        or dependency_refs(clean["q2_template"]) != ["slot_1"]
    ):
        raise HotpotRuntimeError("runtime input q2 must contain literal #1 exactly once")
    try:
        probes = replace_dependency_refs(
            clean["q2_template"], {"slot_1": "RUNTIMEPROBEVALUE"}, max_variants=1
        )
    except DependentRetrievalError as exc:
        raise HotpotRuntimeError("runtime input q2 dependency is invalid") from exc
    if len(probes) != 1:
        raise HotpotRuntimeError("runtime input q2 dependency is ambiguous")
    parse_query_response(
        probes[0], previous_queries=(clean["question"], clean["q1_query"])
    )
    return clean


def load_runtime_inputs(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    rows = [_validate_runtime_input(row) for row in _load_jsonl(path)]
    if len(rows) != expected_rows:
        raise HotpotRuntimeError(
            f"runtime input has {len(rows)} rather than {expected_rows} rows"
        )
    keys = [(row["dataset"], row["qid"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise HotpotRuntimeError("runtime input contains duplicate identities")
    return rows


def _reader_call(backend: Any, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    result = backend(deepcopy(list(messages)))
    if isinstance(result, v8_runtime.TextGenerationResult):
        response = result.response_text
        telemetry = {
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "runtime_telemetry": deepcopy(dict(result.runtime_telemetry)),
        }
    elif isinstance(result, str):
        response = result
        telemetry = {
            "prompt_tokens": None,
            "generation_tokens": None,
            "runtime_telemetry": {"mode": "injectable_cpu_test_backend"},
        }
    else:
        raise HotpotRuntimeError("Reader backend returned unsupported value")
    return {
        "response": response,
        "response_sha256": _sha256_text(response),
        "messages_sha256": _sha256_value(list(messages)),
        **telemetry,
    }


def _project_documents(
    passages: Sequence[Mapping[str, Any]], *, role: str
) -> list[dict[str, str]]:
    return project_top10_passages_for_prompt(passages, role=role)


def merge_bound_q1_with_q2(
    *,
    q1_passages: Sequence[Mapping[str, Any]],
    q2_passages: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Apply the fixed ``bound-q1 + q2-novel + q1-backfill`` allocation."""

    projection = project_rank_first_binding_for_runtime(binding)
    q1 = _project_documents(q1_passages, role="hotpot_q1_merge")
    q2 = _project_documents(q2_passages, role="hotpot_q2_merge")
    bound_id = str(projection["supporting_doc_id"])
    candidates = [document for document in q1 if document["doc_id"] == bound_id]
    if len(candidates) != 1:
        raise HotpotRuntimeError(
            "verified q1 binding does not identify exactly one q1 document"
        )
    bound = deepcopy(candidates[0])
    # Recheck the exact model-visible bytes bound by v9.1.
    bound_prompt_visible = {"id": bound["doc_id"], "contents": bound["text"]}
    if bound["title"]:
        bound_prompt_visible["title"] = bound["title"]
    bound_prompt_hash = _sha256_value(bound_prompt_visible)
    if bound_prompt_hash != projection["supporting_document_prompt_sha256"]:
        raise HotpotRuntimeError("bound q1 document prompt bytes drifted")

    selected = [bound]
    selected_ids = {bound_id}
    q1_hashes = {document["doc_id"]: _sha256_value(document) for document in q1}
    q2_added = 0
    q1_backfilled = 0
    duplicate_checks: dict[str, str] = {bound_id: _sha256_value(bound)}

    def add(document: Mapping[str, str], *, source: str) -> bool:
        nonlocal q2_added, q1_backfilled
        doc_id = str(document["doc_id"])
        prompt_hash = _sha256_value(document)
        if doc_id in selected_ids:
            if duplicate_checks[doc_id] != prompt_hash:
                raise HotpotRuntimeError(
                    "one document id maps to different model-visible bytes"
                )
            return False
        selected.append(deepcopy(dict(document)))
        selected_ids.add(doc_id)
        duplicate_checks[doc_id] = prompt_hash
        if source == "q2":
            q2_added += 1
        else:
            q1_backfilled += 1
        return True

    for document in q2:
        if len(selected) >= EXPECTED_TOP_K or q2_added >= 9:
            break
        # ``q2 novel`` is defined relative to the complete q1 top-10, not only
        # to the one already selected bound document.  Overlapping q1 evidence
        # is recovered later in original q1 rank order by the backfill stage.
        if document["doc_id"] in q1_hashes:
            if q1_hashes[document["doc_id"]] != _sha256_value(document):
                raise HotpotRuntimeError(
                    "one document id maps to different model-visible bytes"
                )
            continue
        add(document, source="q2")
    for document in q1:
        if len(selected) >= EXPECTED_TOP_K:
            break
        add(document, source="q1_backfill")
    if len(selected) != EXPECTED_TOP_K or len(selected_ids) != EXPECTED_TOP_K:
        raise HotpotRuntimeError("final passage allocation did not produce ten unique docs")
    if selected[0]["doc_id"] != bound_id:
        raise AssertionError("bound q1 document lost first position")
    return selected, {
        "policy_version": "hotpot-bound-q1-plus-q2-novel-v1",
        "gold_access": False,
        "bound_q1_document_id": bound_id,
        "bound_q1_document_first": True,
        "q2_novel_selected": q2_added,
        "q1_backfill_selected": q1_backfilled,
        "total_selected": len(selected),
        "output_doc_ids": [document["doc_id"] for document in selected],
    }


def _safe_snapshot(passages: Sequence[Mapping[str, Any]], *, role: str) -> dict[str, Any]:
    documents = _project_documents(passages, role=role)
    return {
        "documents": documents,
        "documents_sha256": _sha256_value(documents),
        "gold_access": False,
    }


def materialize_runtime(
    rows: Sequence[Mapping[str, Any]],
    *,
    hf_runtime: Any,
    retriever_runtime: Any,
) -> dict[str, Any]:
    """Execute the fixed state machine with injectable production/test runtimes."""

    clean_rows = [_validate_runtime_input(row) for row in rows]
    if not clean_rows:
        raise HotpotRuntimeError("runtime cohort is empty")
    if len({(row["dataset"], row["qid"]) for row in clean_rows}) != len(clean_rows):
        raise HotpotRuntimeError("runtime cohort contains duplicate identities")
    if not hasattr(retriever_runtime, "batch_search"):
        raise HotpotRuntimeError("retriever runtime lacks batch_search")
    if not hasattr(hf_runtime, "bind_role"):
        raise HotpotRuntimeError("HF runtime lacks bind_role")
    reader = hf_runtime.bind_role("final_reader")

    q1_batches = retriever_runtime.batch_search([row["q1_query"] for row in clean_rows])
    if not isinstance(q1_batches, Sequence) or len(q1_batches) != len(clean_rows):
        raise HotpotRuntimeError("q1 retrieval batch cardinality mismatch")

    outputs: list[dict[str, Any]] = []
    eligible: list[tuple[int, str]] = []
    for index, (row, q1_passages) in enumerate(zip(clean_rows, q1_batches)):
        base: dict[str, Any] = {
            "schema_version": ROW_SCHEMA_VERSION,
            "runner_version": RUNNER_VERSION,
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question": row["question"],
            "question_sha256": row["question_sha256"],
            "proposal_sha256": row["proposal_sha256"],
            "source_projection_row_sha256": row["source_projection_row_sha256"],
            "gold_access": False,
            "q1_query": row["q1_query"],
            "q1_query_sha256": row["q1_query_sha256"],
            "q2_template": row["q2_template"],
            "q2_template_sha256": row["q2_template_sha256"],
            "q1_retrieval": _safe_snapshot(q1_passages, role=f"q1_{index}"),
            "q2_query": None,
            "q2_query_sha256": None,
            "q2_retrieval": None,
            "final_passages": None,
            "merge_telemetry": None,
            "final_reader": None,
        }
        q1_messages = build_canonical_subqa_messages(
            subquestion=row["q1_query"], retrieved_passages=q1_passages
        )
        q1_generation = _reader_call(reader, q1_messages)
        q1_parsed = parse_and_bind_canonical_subanswer(
            q1_generation["response"],
            subquestion=row["q1_query"],
            retrieved_passages=q1_passages,
        )
        binding = q1_parsed["binding"]
        base["q1_reader"] = {
            **q1_generation,
            "parsed": q1_parsed,
        }
        if binding.get("verified") is not True:
            base["status"] = "q1_binding_failed_no_q2_no_final"
            base["failure_reason"] = str(binding.get("reason") or "unknown")
            outputs.append(base)
            continue
        try:
            q2_variants = replace_dependency_refs(
                row["q2_template"],
                {"slot_1": str(binding["verified_answer"])},
                max_variants=1,
            )
            if len(q2_variants) != 1:
                raise HotpotRuntimeError("q2 substitution produced non-singleton output")
            q2_query = q2_variants[0]
            parse_query_response(
                q2_query,
                previous_queries=(row["question"], row["q1_query"]),
            )
        except (DependentRetrievalError, DynamicDecompositionV8Error) as exc:
            base["status"] = "predicted_observation_q2_instantiation_failed_no_retrieval"
            base["failure_reason"] = type(exc).__name__
            outputs.append(base)
            continue
        base["q2_query"] = q2_query
        base["q2_query_sha256"] = _sha256_text(q2_query)
        outputs.append(base)
        eligible.append((index, q2_query))

    if eligible:
        q2_batches = retriever_runtime.batch_search([query for _, query in eligible])
        if not isinstance(q2_batches, Sequence) or len(q2_batches) != len(eligible):
            raise HotpotRuntimeError("q2 retrieval batch cardinality mismatch")
        for (row_index, _), q2_passages in zip(eligible, q2_batches):
            row = clean_rows[row_index]
            output = outputs[row_index]
            binding = output["q1_reader"]["parsed"]["binding"]
            final_passages, merge = merge_bound_q1_with_q2(
                q1_passages=q1_batches[row_index],
                q2_passages=q2_passages,
                binding=binding,
            )
            output["q2_retrieval"] = _safe_snapshot(
                q2_passages, role=f"q2_{row_index}"
            )
            output["final_passages"] = {
                "documents": final_passages,
                "documents_sha256": _sha256_value(final_passages),
                "gold_access": False,
            }
            output["merge_telemetry"] = merge
            final_messages = build_inference_messages(
                question=row["question"],
                retrieved_passages=final_passages,
                kg_triples=[],
                top_k=EXPECTED_TOP_K,
                max_kg_triples=0,
            )
            final_generation = _reader_call(reader, final_messages)
            final_answer = extract_final_answer(final_generation["response"])
            final_steps = parse_steps(final_generation["response"], known_kg=[])
            output["final_reader"] = {
                **final_generation,
                "final_answer_parsed": final_answer is not None,
                "final_answer": final_answer,
                "final_answer_sha256": (
                    _sha256_text(final_answer) if final_answer is not None else None
                ),
                "parsed_step_count": len(final_steps),
            }
            output["status"] = "complete_gold_free_runtime"
            output["failure_reason"] = None

    if len(outputs) != len(clean_rows):
        raise AssertionError("runtime row conservation failed")
    physical_model_ids = {
        value
        for output in outputs
        for block in (output.get("q1_reader"), output.get("final_reader"))
        if isinstance(block, Mapping)
        for value in [
            (block.get("runtime_telemetry") or {}).get("shared_model_object_id")
        ]
        if value is not None
    }
    status_counts = Counter(str(output["status"]) for output in outputs)
    return {
        "schema_version": "hotpot-controller-retrieval-reader-run-v1",
        "runner_version": RUNNER_VERSION,
        "gold_access": False,
        "answer_scoring_performed": False,
        "runtime_contract": runtime_contract(),
        "row_count": len(outputs),
        "q1_retrieval_requests": len(clean_rows),
        "q2_retrieval_requests": len(eligible),
        "q1_reader_requests": len(clean_rows),
        "final_reader_requests": int(status_counts["complete_gold_free_runtime"]),
        "physical_model_object_ids": sorted(physical_model_ids),
        "runtime_mode": "formal" if physical_model_ids else "injectable_test",
        "rows": outputs,
    }


def build_report(
    result: Mapping[str, Any],
    *,
    source_fixed_denominator: int | None = None,
    runtime_candidate_min: int | None = None,
) -> dict[str, Any]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HotpotRuntimeError("runtime result rows missing")
    n = len(rows)
    denominator = n if source_fixed_denominator is None else source_fixed_denominator
    candidate_min = n if runtime_candidate_min is None else runtime_candidate_min
    if denominator < n or candidate_min <= 0 or candidate_min > denominator:
        raise HotpotRuntimeError("invalid report denominator or runtime candidate gate")
    complete = [row for row in rows if row.get("status") == "complete_gold_free_runtime"]
    binding_verified = [
        row
        for row in rows
        if ((row.get("q1_reader") or {}).get("parsed") or {}).get("binding", {}).get(
            "verified"
        )
        is True
    ]
    q1_failed = [
        row for row in rows if row.get("status") == "q1_binding_failed_no_q2_no_final"
    ]
    no_forbidden_followup = all(
        row.get("q2_query") is None
        and row.get("q2_retrieval") is None
        and row.get("final_reader") is None
        for row in q1_failed
    )
    final_exact = bool(complete) and all(
        len((row.get("final_passages") or {}).get("documents") or []) == EXPECTED_TOP_K
        and len(
            {
                document["doc_id"]
                for document in (row.get("final_passages") or {}).get("documents") or []
            }
        )
        == EXPECTED_TOP_K
        and (row.get("merge_telemetry") or {}).get("bound_q1_document_first") is True
        for row in complete
    )
    final_parse = sum(
        (row.get("final_reader") or {}).get("final_answer_parsed") is True
        for row in complete
    )
    physical_ids = result.get("physical_model_object_ids") or []
    # Injectable CPU fakes intentionally have no physical CUDA model id; a
    # formal execution must expose exactly one.  ``runtime_mode`` makes this
    # distinction explicit rather than letting an empty set pass vacuously.
    runtime_mode = str(result.get("runtime_mode") or "formal")
    same_model = (
        len(physical_ids) == 1
        if runtime_mode == "formal"
        else len(physical_ids) <= 1
    )
    runtime_candidates = [
        row
        for row in complete
        if (row.get("final_reader") or {}).get("final_answer_parsed") is True
    ]
    q2_rows = [row for row in rows if row.get("q2_query") is not None]
    q2_bindings_all_verified = all(
        ((row.get("q1_reader") or {}).get("parsed") or {}).get("binding", {}).get(
            "verified"
        )
        is True
        for row in q2_rows
    )
    gates = {
        "all_inputs_accounted_for": n == int(result.get("row_count", -1)),
        "runtime_input_nonempty": n > 0,
        "q1_top10_exact": all(
            len((row.get("q1_retrieval") or {}).get("documents") or []) == EXPECTED_TOP_K
            for row in rows
        ),
        "q1_binding_failure_has_no_q2_or_final": no_forbidden_followup,
        "q2_only_for_verified_q1": int(result.get("q2_retrieval_requests", -1))
        == len(q2_rows)
        and q2_bindings_all_verified,
        "runtime_pass_candidates_min": len(runtime_candidates) >= candidate_min,
        "final_passage_budget_and_bound_first": final_exact,
        "runtime_candidate_final_answers_parse": len(runtime_candidates) == final_parse,
        "same_physical_reader_runtime": same_model,
        "gold_access_false": all(row.get("gold_access") is False for row in rows),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": _utc_now(),
        "status": "PASS_GOLD_FREE_RUNTIME_MECHANICS" if all(gates.values()) else "FAIL_STOP_RUNTIME_MECHANICS",
        "gold_access": False,
        "answer_scoring_performed": False,
        "source_fixed_denominator": denominator,
        "generation_accepted_runtime_input_rows": n,
        "source_not_in_runtime_input": denominator - n,
        "q1_binding_verified": len(binding_verified),
        "q1_binding_verified_rate": len(binding_verified) / n,
        "q1_binding_failed": len(q1_failed),
        "complete_runtime_rows": len(complete),
        "complete_runtime_rate": len(complete) / n,
        "runtime_pass_candidates": len(runtime_candidates),
        "runtime_pass_candidate_min": candidate_min,
        "runtime_pass_candidate_rate_on_fixed_denominator": (
            len(runtime_candidates) / denominator
        ),
        "final_answer_parse_rate_among_complete": (
            final_parse / len(complete) if complete else 0.0
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "scientific_boundary": (
            "Gold-free mechanism materialization only. The input q1/q2 templates are silver "
            "labels constructed earlier with train annotations, but no train-annotation field, "
            "Gold bridge observation, accepted_actions record, retrieval-recall label, EM, F1, "
            "or IHR was opened or scored by this runtime. Runtime-pass rows are only "
            "candidates; the final >=24/30 release decision remains pending the separately "
            "authorized train-annotation support/retrieval scorer. "
            "The v9.1 binder proves lexical surface locality, not semantic entailment."
        ),
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("gold_access") is not False
        or protocol.get("runtime_contract") != runtime_contract()
        or (protocol.get("authorization") or {}).get("runtime_inference") is not True
        or (protocol.get("authorization") or {}).get("training") is not False
        or (protocol.get("authorization") or {}).get("answer_scoring") is not False
    ):
        raise HotpotRuntimeError("retrieval/Reader protocol contract mismatch")
    for group in ("code_locks", "parent_locks"):
        locks = protocol.get(group)
        if not isinstance(locks, Mapping) or not locks:
            raise HotpotRuntimeError(f"protocol {group} missing")
        for label, lock in locks.items():
            _assert_file_lock(lock, label=f"{group}.{label}")
    input_lock = protocol.get("runtime_input")
    _assert_file_lock(input_lock, label="runtime_input")
    if (
        Path(str(input_lock.get("path") or "")).name
        != "runtime_inputs.answer_free.jsonl"
        or input_lock.get("schema_version") != INPUT_SCHEMA_VERSION
        or input_lock.get("fields_exact") != list(_INPUT_FIELDS)
        or input_lock.get("gold_access") is not False
    ):
        raise HotpotRuntimeError("protocol runtime input is not the minimal answer-free projection")
    decision = protocol.get("runtime_decision_gate") or {}
    if (
        decision.get("source_fixed_denominator") != 30
        or decision.get("candidate_min") != 24
        or input_lock.get("source_fixed_denominator") != 30
        or not 24 <= int(input_lock.get("row_count", -1)) <= 30
        or decision.get("generation_rejected_and_runtime_failed_rows_retained") is not True
    ):
        raise HotpotRuntimeError("formal 24-of-30 runtime decision gate drift")
    commitment = protocol.get("protocol_body_canonical_sha256")
    body = dict(protocol)
    body.pop("protocol_body_canonical_sha256", None)
    if commitment != _sha256_value(body):
        raise HotpotRuntimeError("protocol body commitment mismatch")
    return protocol


def execute(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    output_dir: Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Execute the formally frozen pilot after full asset re-verification."""

    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise HotpotRuntimeError(f"run from project root: {PROJECT_ROOT}")
    protocol_path = (
        protocol_path if protocol_path.is_absolute() else PROJECT_ROOT / protocol_path
    ).resolve()
    output_dir = (
        output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"append-only run directory exists: {output_dir}")
    protocol = _load_protocol(protocol_path)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise HotpotRuntimeError("formal experiment id drift")
    rows = load_runtime_inputs(
        Path(str(protocol["runtime_input"]["path"])),
        expected_rows=int(protocol["runtime_input"]["row_count"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    running_path = output_dir / "manifest.running.json"
    _write_json_new(
        running_path,
        {
            "schema_version": RUNNING_MANIFEST_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": _utc_now(),
            "status": "RUNNING_APPEND_ONLY_GOLD_FREE_RUNTIME",
            "protocol": _file_lock(protocol_path),
            "runtime_input": deepcopy(dict(protocol["runtime_input"])),
            "gold_access": False,
            "answer_scoring_performed": False,
            "training_started": False,
        },
    )
    terminal_written = False
    rows_path = output_dir / "rows.jsonl"
    try:
        verified = v8_driver.verify_implementation_before_cuda(
            project_root=PROJECT_ROOT
        )
        if verified["model_asset_identity"] != protocol["model_asset_identity"]:
            raise HotpotRuntimeError("model asset identity differs from protocol")
        if verified["retrieval_asset_identity"] != protocol["retrieval_asset_identity"]:
            raise HotpotRuntimeError("retrieval asset identity differs from protocol")
        hf_runtime = v8_runtime.SharedHuggingFaceRuntime(
            model_asset_identity=verified["model_asset_identity"]
        )
        retriever_runtime = v8_runtime.CanonicalRetrieverRuntime.from_local_assets(
            retrieval_asset_identity=verified["retrieval_asset_identity"]
        )
        result = materialize_runtime(
            rows, hf_runtime=hf_runtime, retriever_runtime=retriever_runtime
        )
        result["runtime_mode"] = "formal"
        _write_jsonl_new(rows_path, result["rows"])
        report = build_report(
            result,
            source_fixed_denominator=int(
                protocol["runtime_input"]["source_fixed_denominator"]
            ),
            runtime_candidate_min=int(protocol["runtime_decision_gate"]["candidate_min"]),
        )
        report["preflight"] = {
            "protocol": _file_lock(protocol_path),
            "runtime_input": deepcopy(dict(protocol["runtime_input"])),
            "parent_v8_assets_rehashed_before_cuda": True,
        }
        report_path = output_dir / "report.json"
        _write_json_new(report_path, report)
        terminal = {
            "schema_version": TERMINAL_MANIFEST_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": report["status"],
            "gold_access": False,
            "answer_scoring_performed": False,
            "training_started": False,
            "running": _file_lock(running_path),
            "rows": _file_lock(rows_path),
            "report": _file_lock(report_path),
        }
        name = "manifest.complete.json" if report["all_pass"] else "manifest.failed.json"
        _write_json_new(output_dir / name, terminal)
        terminal_written = True
        return report
    except BaseException as exc:
        if not terminal_written:
            _write_json_new(
                output_dir / "manifest.failed.json",
                {
                    "schema_version": TERMINAL_MANIFEST_SCHEMA_VERSION,
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_RETAINED_APPEND_ONLY",
                    "gold_access": False,
                    "answer_scoring_performed": False,
                    "training_started": False,
                    "running": _file_lock(running_path),
                    # The runner never reads Gold, but even answer-free query
                    # text and local paths need not be copied into a terminal
                    # failure record.  Bind the diagnostic without serializing
                    # its possibly input-derived contents.
                    **_safe_error_fields(exc),
                    "partial_rows": _file_lock(rows_path) if rows_path.is_file() else None,
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--execute_runtime",
        action="store_true",
        help="Required safety latch; without it no GPU/model/retrieval is loaded.",
    )
    args = parser.parse_args()
    if not args.execute_runtime:
        raise SystemExit(
            "No runtime call made. Re-run with --execute_runtime only after protocol review."
        )
    report = execute(protocol_path=args.protocol, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
