#!/usr/bin/env python
"""Run the Gold-free v9 canonical-subanswer diagnostic on consumed v8 rows.

This is deliberately *not* a fresh method evaluation.  It reuses the already
consumed v8 development rows and their frozen q1 retrieval output to isolate a
single interface variable:

* v8: bespoke one-line subanswer request;
* v9 phase 0: the ordinary SFT inference prompt and ``[Final Answer]`` parser.

No retrieval, Gold join, EM/F1 scoring, prospective-cohort access, training, or
reward change is implemented here.  The unchanged v8 lexical provenance binder
still decides whether the parsed answer is admissible.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kgproweight.retrieval.canonical_subqa_v9 import (  # noqa: E402
    build_canonical_subqa_messages,
    parse_and_bind_canonical_subanswer,
)
from scripts.pilot.materialize_dynamic_decomposition_v8 import (  # noqa: E402
    SharedHuggingFaceRuntime,
    TextGenerationResult,
)
from scripts.prepare.freeze_dynamic_decomposition_v8_implementation import (  # noqa: E402
    tree_lock,
)


SCHEMA_VERSION = "canonical-subqa-v9-phase0-runner-1"
EXPERIMENT_ID = "SUBQUESTION-DECOMPOSITION-V9-CANONICAL-SUBQA-PHASE0-CONSUMED-DEV90-SEED42-V1"
PROTOCOL_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v9_canonical_subqa_phase0_protocol_v1"
)
PROTOCOL_PATH = PROTOCOL_DIR / "protocol.json"
PROTOCOL_MANIFEST_PATH = PROTOCOL_DIR / "manifest.json"
RUN_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v9_canonical_subqa_phase0_consumed_dev90_seed42_v1"
)
SOURCE_DIR = Path(
    "outputs/audits/"
    "subquestion_decomposition_v8_rrf100_development90_seed42_attempt001"
)
SOURCE_ROWS_PATH = SOURCE_DIR / "rows.jsonl"
SOURCE_REPORT_PATH = SOURCE_DIR / "report.json"
SOURCE_FAILED_MANIFEST_PATH = SOURCE_DIR / "manifest.failed.json"
EXPECTED_SOURCE_ROWS_SHA256 = "54fee17e629e4beea3ba62613214588577781a1b0afceb4ddec98b48a22dc888"
EXPECTED_SOURCE_REPORT_SHA256 = "199c3f172dc224fdff2ec9e34cc4fef3c21437603862682bc959451cc6b8b6e3"
EXPECTED_SOURCE_FAILED_MANIFEST_SHA256 = "86b46429d4a54df281495134881c702bc516bac4c5dc24521ceaee26a555abd7"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPECTED_PER_DATASET = 30
EXPECTED_ROWS = 90
FINAL_PARSE_RATE_MIN = 0.95
TRACE_RATE_MIN = 0.95
ADMISSIBLE_RATE_MIN = 0.40


class V9Phase0Error(RuntimeError):
    """The frozen input, protocol, runtime, or result violated its contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V9Phase0Error(f"expected JSON object: {path}")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only output already exists: {path}")
    path.write_bytes(_canonical_json_bytes(dict(value)))


def _assert_lock(lock: Mapping[str, Any], *, label: str) -> None:
    path = Path(str(lock.get("path") or "")).resolve()
    if not path.is_file():
        raise V9Phase0Error(f"{label} path is missing: {path}")
    current = _file_lock(path)
    if any(current[field] != lock.get(field) for field in ("size_bytes", "sha256")):
        raise V9Phase0Error(f"{label} content drift")


def _load_and_verify_protocol() -> dict[str, Any]:
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_PATH)
    manifest = _read_json(PROJECT_ROOT / PROTOCOL_MANIFEST_PATH)
    if (
        protocol.get("schema_version") != "canonical-subqa-v9-phase0-protocol-1"
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("status") != "AUTHORIZED_GOLD_FREE_CONSUMED_DEV90_DIAGNOSTIC"
        or protocol.get("gold_access") is not False
        or protocol.get("prospective_opened_or_hashed") is not False
    ):
        raise V9Phase0Error("v9 phase0 protocol identity/boundary mismatch")
    if (
        manifest.get("schema_version") != "canonical-subqa-v9-phase0-protocol-manifest-1"
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("status") != protocol.get("status")
    ):
        raise V9Phase0Error("v9 phase0 protocol manifest mismatch")
    protocol_lock = _file_lock((PROJECT_ROOT / PROTOCOL_PATH).resolve())
    if manifest.get("protocol") != protocol_lock:
        raise V9Phase0Error("protocol manifest does not bind protocol bytes")
    for label, lock in (protocol.get("code_locks") or {}).items():
        _assert_lock(lock, label=f"code_locks.{label}")
    for label, lock in (protocol.get("source_locks") or {}).items():
        _assert_lock(lock, label=f"source_locks.{label}")
    return protocol


def _validate_source_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("gold_access") is not False:
        raise V9Phase0Error("source row does not assert gold_access=false")
    identity = raw.get("identity")
    shared = raw.get("shared")
    if not isinstance(identity, Mapping) or set(identity) != {"dataset", "qid", "question"}:
        raise V9Phase0Error("source identity schema mismatch")
    if not isinstance(shared, Mapping):
        raise V9Phase0Error("source shared block is missing")
    dataset = identity.get("dataset")
    qid = identity.get("qid")
    question = identity.get("question")
    if dataset not in DATASETS or any(
        not isinstance(value, str) or not value.strip()
        for value in (qid, question)
    ):
        raise V9Phase0Error("source identity value mismatch")
    action = shared.get("q1_action")
    snapshot = shared.get("q1_top10")
    old_binding = shared.get("subanswer_binding")
    if not isinstance(action, Mapping) or not isinstance(snapshot, Mapping):
        raise V9Phase0Error("source q1 action/snapshot is missing")
    if snapshot.get("gold_access") is not False:
        raise V9Phase0Error("source q1 snapshot gold_access mismatch")
    query = action.get("selected_query")
    documents = snapshot.get("documents")
    if not isinstance(query, str) or not query.strip() or not isinstance(documents, list):
        raise V9Phase0Error("source q1 query/documents are malformed")
    if len(documents) != 10 or any(not isinstance(item, Mapping) for item in documents):
        raise V9Phase0Error("source q1 snapshot must contain ten documents")
    return {
        "dataset": str(dataset),
        "qid": str(qid),
        "question": str(question),
        "q1_query": query,
        "q1_documents": [dict(item) for item in documents],
        "q1_documents_sha256": snapshot.get("documents_sha256"),
        "old_binding_verified": bool(
            isinstance(old_binding, Mapping) and old_binding.get("verified") is True
        ),
        "old_binding_reason": (
            old_binding.get("reason") if isinstance(old_binding, Mapping) else None
        ),
    }


def _load_source_rows() -> list[dict[str, Any]]:
    path = (PROJECT_ROOT / SOURCE_ROWS_PATH).resolve()
    expected = EXPECTED_SOURCE_ROWS_SHA256
    if _sha256_file(path) != expected:
        raise V9Phase0Error("frozen v8 source rows SHA mismatch")
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise V9Phase0Error(f"source row framing mismatch at line {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise V9Phase0Error(f"source row {line_number} is not an object")
            row = _validate_source_row(raw)
            key = f"{row['dataset']}::{row['qid']}"
            if key in seen:
                raise V9Phase0Error(f"duplicate source identity: {key}")
            seen.add(key)
            counts[row["dataset"]] += 1
            rows.append(row)
    if len(rows) != EXPECTED_ROWS or dict(counts) != {
        dataset: EXPECTED_PER_DATASET for dataset in DATASETS
    }:
        raise V9Phase0Error("source cohort cardinality mismatch")
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute only pre-registered, Gold-free mechanism metrics."""

    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row.get("dataset") == dataset]
        reasons = Counter(
            str((row.get("canonical_subanswer") or {}).get("binding", {}).get("reason"))
            for row in subset
        )
        parsed = sum(
            (row.get("canonical_subanswer") or {}).get("final_answer_parsed") is True
            for row in subset
        )
        trace = sum(
            (row.get("canonical_subanswer") or {}).get("has_step_trace") is True
            for row in subset
        )
        admissible = sum(
            (row.get("canonical_subanswer") or {}).get("binding", {}).get("verified")
            is True
            for row in subset
        )
        old_admissible = sum(row.get("old_binding_verified") is True for row in subset)
        by_dataset[dataset] = {
            "n": len(subset),
            "final_answer_parse_count": parsed,
            "final_answer_parse_rate": _rate(parsed, len(subset)),
            "step_trace_count": trace,
            "step_trace_rate": _rate(trace, len(subset)),
            "admissible_count": admissible,
            "admissible_rate": _rate(admissible, len(subset)),
            "v8_old_admissible_count": old_admissible,
            "v8_old_admissible_rate": _rate(old_admissible, len(subset)),
            "admissible_rate_delta_vs_v8_consumed": _rate(admissible - old_admissible, len(subset)),
            "binding_reason_counts": dict(sorted(reasons.items())),
        }
    gates = {
        "row_count_exact": len(rows) == EXPECTED_ROWS,
        "per_dataset_count_exact": all(
            by_dataset[dataset]["n"] == EXPECTED_PER_DATASET for dataset in DATASETS
        ),
        "final_answer_parse_rate_min_each_dataset": all(
            by_dataset[dataset]["final_answer_parse_rate"] >= FINAL_PARSE_RATE_MIN
            for dataset in DATASETS
        ),
        "step_trace_rate_min_each_dataset": all(
            by_dataset[dataset]["step_trace_rate"] >= TRACE_RATE_MIN
            for dataset in DATASETS
        ),
        "admissible_rate_min_each_dataset": all(
            by_dataset[dataset]["admissible_rate"] >= ADMISSIBLE_RATE_MIN
            for dataset in DATASETS
        ),
    }
    return {
        "by_dataset": by_dataset,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def run(
    *,
    generator_factory: Callable[[Mapping[str, str]], Callable[[Sequence[Mapping[str, str]]], Any]]
    | None = None,
) -> dict[str, Any]:
    """Execute the frozen diagnostic; an injectable factory supports CPU tests."""

    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise V9Phase0Error(f"run from project root: {PROJECT_ROOT}")
    protocol = _load_and_verify_protocol()
    rows = _load_source_rows()
    destination = (PROJECT_ROOT / RUN_DIR).resolve()
    if destination.exists():
        raise FileExistsError(f"append-only run directory already exists: {destination}")
    destination.mkdir(parents=True)
    running_path = destination / "manifest.running.json"
    _write_json_new(
        running_path,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "RUNNING_GOLD_FREE",
            "created_at_utc": _utc_now(),
            "gold_access": False,
            "retrieval_calls": 0,
            "prospective_opened_or_hashed": False,
            "protocol": _file_lock((PROJECT_ROOT / PROTOCOL_PATH).resolve()),
        },
    )
    rows_path = destination / "rows.jsonl"
    try:
        model_spec = protocol["model"]
        if generator_factory is None:
            for name in ("base_model", "strong_sft"):
                current = tree_lock(Path(model_spec[name]["path"]))
                expected = model_spec[name]
                for field in ("file_count", "size_bytes", "tree_sha256"):
                    if current[field] != expected[field]:
                        raise V9Phase0Error(f"model lock drift: {name}.{field}")
            runtime = SharedHuggingFaceRuntime(
                model_asset_identity={
                    "base_model_tree_sha256": model_spec["base_model"]["tree_sha256"],
                    "adapter_tree_sha256": model_spec["strong_sft"]["tree_sha256"],
                    "tokenizer_tree_sha256": model_spec["base_model"]["tree_sha256"],
                }
            )
            generator = runtime.bind_role("final_reader")
        else:
            generator = generator_factory(
                {
                    "base_model_tree_sha256": model_spec["base_model"]["tree_sha256"],
                    "adapter_tree_sha256": model_spec["strong_sft"]["tree_sha256"],
                    "tokenizer_tree_sha256": model_spec["base_model"]["tree_sha256"],
                }
            )

        outputs: list[dict[str, Any]] = []
        with rows_path.open("xb") as handle:
            for index, source in enumerate(rows):
                messages = build_canonical_subqa_messages(
                    subquestion=source["q1_query"],
                    retrieved_passages=source["q1_documents"],
                )
                raw = generator(messages)
                if isinstance(raw, TextGenerationResult):
                    generation = raw.response_text
                    runtime_telemetry = {
                        "prompt_tokens": raw.prompt_tokens,
                        "generation_tokens": raw.generation_tokens,
                        **dict(raw.runtime_telemetry),
                    }
                elif isinstance(raw, str):
                    generation = raw
                    runtime_telemetry = {"mode": "injectable_test_generator"}
                else:
                    raise V9Phase0Error("generator returned unsupported value")
                parsed = parse_and_bind_canonical_subanswer(
                    generation,
                    subquestion=source["q1_query"],
                    retrieved_passages=source["q1_documents"],
                )
                output = {
                    "schema_version": SCHEMA_VERSION,
                    "gold_access": False,
                    "dataset": source["dataset"],
                    "qid": source["qid"],
                    "question": source["question"],
                    "q1_query": source["q1_query"],
                    "q1_documents_sha256": source["q1_documents_sha256"],
                    "prompt_sha256": _sha256_bytes(_canonical_json_bytes(messages)),
                    "generation": generation,
                    "canonical_subanswer": parsed,
                    "old_binding_verified": source["old_binding_verified"],
                    "old_binding_reason": source["old_binding_reason"],
                    "runtime_telemetry": runtime_telemetry,
                }
                line = _canonical_json_bytes(output)
                handle.write(line)
                handle.flush()
                outputs.append(output)
                if (index + 1) % 10 == 0:
                    print(f"progress={index + 1}/{EXPECTED_ROWS}", flush=True)

        summary = summarize(outputs)
        report = {
            "schema_version": "canonical-subqa-v9-phase0-gold-free-report-1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "PASS_CANONICAL_SUBANSWER_INTERFACE"
                if summary["all_pass"]
                else "FAIL_STOP_CANONICAL_SUBANSWER_INTERFACE"
            ),
            "created_at_utc": _utc_now(),
            "scope": "CONSUMED_V8_DEVELOPMENT90_POSTHOC_SINGLE_VARIABLE_DIAGNOSTIC",
            "single_variable": "bespoke_one_line_subanswer_prompt_to_canonical_sft_qa_prompt",
            "unchanged": [
                "same q1 queries",
                "same q1 top-10 passages",
                "same strong SFT adapter",
                "same greedy decoding",
                "same v8 lexical provenance binder",
            ],
            "gold_access": False,
            "answer_scoring_performed": False,
            "retrieval_calls": 0,
            "prospective_opened_or_hashed": False,
            "scientific_boundary": (
                "Consumed-development Gold-free interface diagnosis only. A pass permits "
                "a separately preregistered fresh v9 pilot; it is not EM/F1 evidence."
            ),
            **summary,
            "rows": _file_lock(rows_path),
        }
        report_path = destination / "report.json"
        _write_json_new(report_path, report)
        terminal_path = destination / "manifest.complete.json"
        _write_json_new(
            terminal_path,
            {
                "schema_version": "canonical-subqa-v9-phase0-run-manifest-1",
                "experiment_id": EXPERIMENT_ID,
                "status": report["status"],
                "gold_access": False,
                "prospective_opened_or_hashed": False,
                "answer_scoring_performed": False,
                "outputs": {
                    "rows": _file_lock(rows_path),
                    "report": _file_lock(report_path),
                },
            },
        )
        return report
    except BaseException as exc:
        failed_path = destination / "manifest.failed.json"
        if not failed_path.exists():
            _write_json_new(
                failed_path,
                {
                    "schema_version": "canonical-subqa-v9-phase0-run-manifest-1",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_RETAINED_APPEND_ONLY",
                    "gold_access": False,
                    "prospective_opened_or_hashed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "partial_rows": _file_lock(rows_path) if rows_path.exists() else None,
                },
            )
        raise


def main() -> None:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
