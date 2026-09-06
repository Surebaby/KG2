#!/usr/bin/env python3
"""Append-only, aggregate-only budget audit for historical Trace artifacts.

The historical ``intermediate_data.json`` files may contain questions, Gold
answers, predictions, prompts, and generated thoughts.  This audit parses the
container but deliberately consumes only structural information under
``row["output"]``:

* the cardinality of the final cumulative ``retrieval_result`` list;
* the number and contiguity of recorded ``intermediate_output_iterN`` keys;
* structural schema checks; and
* within-record document-id uniqueness.

No row identity, question, Gold field, prediction, prompt, thought, or document
text is emitted.  The result is a historical IRCoT-style reference only.  It
does not establish per-call top-k, physical retriever calls, runtime code
identity, cache/retry behaviour, wall time, a fixed-10 matched control, or a
paper-faithful IRCoT reproduction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(
    "outputs/audits/historical_trace_ircot_budget_n300_seed42_v1"
)
DEFAULT_EXPERIMENT_ID = "HISTORICAL-TRACE-IRCOT-BUDGET-N300-SEED42-V1"

DATASET_SOURCES: dict[str, dict[str, Path]] = {
    "hotpotqa": {
        "intermediate": Path(
            "outputs/baselines_rerank/trace/hotpotqa/seed_42/"
            "hotpotqa_2026_08_21_00_50_trace/intermediate_data.json"
        ),
        "config": Path(
            "outputs/baselines_rerank/trace/hotpotqa/seed_42/"
            "hotpotqa_2026_08_21_00_50_trace/config.yaml"
        ),
        "manifest": Path(
            "outputs/baselines_rerank/trace/hotpotqa/seed_42/"
            "hotpotqa_2026_08_21_00_50_trace/manifest.json"
        ),
        "metric": Path(
            "outputs/baselines_rerank/trace/hotpotqa/seed_42/"
            "hotpotqa_2026_08_21_00_50_trace/metric_score.json"
        ),
    },
    "2wikimultihopqa": {
        "intermediate": Path(
            "outputs/baselines_rerank/trace/2wikimultihopqa/seed_42/"
            "2wikimultihopqa_2026_08_21_05_17_trace/intermediate_data.json"
        ),
        "config": Path(
            "outputs/baselines_rerank/trace/2wikimultihopqa/seed_42/"
            "2wikimultihopqa_2026_08_21_05_17_trace/config.yaml"
        ),
        "manifest": Path(
            "outputs/baselines_rerank/trace/2wikimultihopqa/seed_42/"
            "2wikimultihopqa_2026_08_21_05_17_trace/manifest.json"
        ),
        "metric": Path(
            "outputs/baselines_rerank/trace/2wikimultihopqa/seed_42/"
            "2wikimultihopqa_2026_08_21_05_17_trace/metric_score.json"
        ),
    },
    "musique": {
        "intermediate": Path(
            "outputs/baselines_rerank/trace/musique/seed_42/"
            "musique_2026_08_21_05_55_trace/intermediate_data.json"
        ),
        "config": Path(
            "outputs/baselines_rerank/trace/musique/seed_42/"
            "musique_2026_08_21_05_55_trace/config.yaml"
        ),
        "manifest": Path(
            "outputs/baselines_rerank/trace/musique/seed_42/"
            "musique_2026_08_21_05_55_trace/manifest.json"
        ),
        "metric": Path(
            "outputs/baselines_rerank/trace/musique/seed_42/"
            "musique_2026_08_21_05_55_trace/metric_score.json"
        ),
    },
}

FROZEN_SHA256: dict[str, dict[str, str]] = {
    "hotpotqa": {
        "intermediate": "6ce6737d76ef4c7b66157013c011ceed67ccd2c491c0ca322566957cca018cb7",
        "config": "64848a7610151af3f1bbc5ed552019e9f6aa2f9770ddba371e9b4088823c6acc",
        "manifest": "6e0d84e620de923a2b6a717ccba92fd6ca35fadfb47b94302efe8252c54fe3ba",
        "metric": "06331d097f5dd9a5f6f6120a3139ac17811d79fc0f501bc7ebffab8ad00e4513",
    },
    "2wikimultihopqa": {
        "intermediate": "23572d551d4c6f2f244df683a1f650112247ae93987d10e04bfa4d80be93a892",
        "config": "119a4e75bc3b80ddb9428fce2c0fdeb6ea369089606d8deefe8a52efd30ddff5",
        "manifest": "2449a5091dc97ce916d96922abb59e3645282318991cc9aba4bacfb3c1ff8d61",
        "metric": "8517b5c8d238b5009f41b91211dc2823a8728d4d741dfa3ae60babb8aa1a283a",
    },
    "musique": {
        "intermediate": "2077fd3987a88729e3c3e83bebbceacd6058597e3866c526e4743a99b5d731b7",
        "config": "04987d6ec232267164d9033696390e115bf29eccef5f77f1288a34d81c20f84b",
        "manifest": "a35b75149cac70e11184b9e39cd31e017fa3e37230872f1fa6abe853b444745c",
        "metric": "0fc5d5b6689f089f1fdbd20cb533e642b1dd9dd30cb524f45e53899255a9fec3",
    },
}

EXPECTED_CARDINALITY = 300
ITERATION_KEY = re.compile(r"^intermediate_output_iter([0-9]+)$")
REQUIRED_OUTPUT_KEYS = frozenset({"retrieval_result", "pred", "raw_pred", "metric_score"})
EXPECTED_ITERATION_VALUE_KEYS = frozenset({"input_prompt", "new_thought"})
REPORT_SCHEMA = "historical-trace-ircot-budget-audit-1"
PROTOCOL_SCHEMA = "historical-trace-ircot-budget-protocol-1"
MANIFEST_SCHEMA = "historical-trace-ircot-budget-manifest-1"


class TraceBudgetAuditError(RuntimeError):
    """The frozen source or audit contract is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise TraceBudgetAuditError("cannot summarize an empty distribution")
    counts = Counter(values)
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(math.fsum(values) / len(values), 6),
        "median": float(statistics.median(values)),
        "histogram": {str(key): counts[key] for key in sorted(counts)},
    }


def audit_records(records: Any, *, expected_cardinality: int = EXPECTED_CARDINALITY) -> dict[str, Any]:
    """Aggregate allowed structural fields without emitting row-level content."""
    if not isinstance(records, list):
        raise TraceBudgetAuditError("intermediate_data must be a JSON list")
    if len(records) != expected_cardinality:
        raise TraceBudgetAuditError(
            f"cardinality mismatch: expected {expected_cardinality}, observed {len(records)}"
        )

    doc_counts: list[int] = []
    generation_rounds: list[int] = []
    rows_output_dict = 0
    rows_retrieval_list = 0
    rows_required_output_keys = 0
    rows_iteration_keys_contiguous = 0
    rows_iteration_values_expected_schema = 0
    rows_all_doc_objects = 0
    rows_all_doc_ids_nonempty_strings = 0
    rows_unique_doc_ids = 0
    duplicate_doc_occurrences = 0
    total_documents = 0
    output_key_sets: Counter[tuple[str, ...]] = Counter()
    iteration_value_key_sets: Counter[tuple[str, ...]] = Counter()

    for row in records:
        # Deliberately do not access any top-level field except ``output``.
        if not isinstance(row, Mapping):
            raise TraceBudgetAuditError("every record must be an object")
        output = row.get("output")
        if not isinstance(output, Mapping):
            raise TraceBudgetAuditError("every record must contain an output object")
        rows_output_dict += 1
        output_key_sets[tuple(sorted(str(key) for key in output.keys()))] += 1

        if REQUIRED_OUTPUT_KEYS.issubset(output.keys()):
            rows_required_output_keys += 1

        iteration_indices: list[int] = []
        iteration_schema_ok = True
        for key, value in output.items():
            match = ITERATION_KEY.fullmatch(str(key))
            if not match:
                continue
            iteration_indices.append(int(match.group(1)))
            if not isinstance(value, Mapping):
                iteration_schema_ok = False
                iteration_value_key_sets[(f"TYPE:{type(value).__name__}",)] += 1
                continue
            key_set = tuple(sorted(str(item) for item in value.keys()))
            iteration_value_key_sets[key_set] += 1
            if frozenset(value.keys()) != EXPECTED_ITERATION_VALUE_KEYS:
                iteration_schema_ok = False
            # The prompt and generated thought values are intentionally unread.

        iteration_indices.sort()
        generation_rounds.append(len(iteration_indices))
        if iteration_indices == list(range(len(iteration_indices))):
            rows_iteration_keys_contiguous += 1
        if iteration_schema_ok:
            rows_iteration_values_expected_schema += 1

        retrieval_result = output.get("retrieval_result")
        if not isinstance(retrieval_result, list):
            raise TraceBudgetAuditError("output.retrieval_result must be a list")
        rows_retrieval_list += 1
        doc_counts.append(len(retrieval_result))
        total_documents += len(retrieval_result)

        all_doc_objects = all(isinstance(doc, Mapping) for doc in retrieval_result)
        if all_doc_objects:
            rows_all_doc_objects += 1

        # Only the opaque id is consumed, solely to count within-row duplicates.
        doc_ids: list[str] = []
        ids_ok = all_doc_objects
        if all_doc_objects:
            for doc in retrieval_result:
                doc_id = doc.get("id")
                if not isinstance(doc_id, str) or not doc_id:
                    ids_ok = False
                    continue
                doc_ids.append(doc_id)
        if ids_ok and len(doc_ids) == len(retrieval_result):
            rows_all_doc_ids_nonempty_strings += 1
            duplicates = len(doc_ids) - len(set(doc_ids))
            duplicate_doc_occurrences += duplicates
            if duplicates == 0:
                rows_unique_doc_ids += 1

    conditional_calls = [rounds + 1 for rounds in generation_rounds]
    return {
        "cardinality": {
            "expected": expected_cardinality,
            "observed": len(records),
            "exact": len(records) == expected_cardinality,
        },
        "final_cumulative_document_count": _distribution(doc_counts),
        "recorded_generation_round_count": _distribution(generation_rounds),
        "schema": {
            "rows_output_is_object": rows_output_dict,
            "rows_retrieval_result_is_list": rows_retrieval_list,
            "rows_with_required_output_keys": rows_required_output_keys,
            "rows_with_contiguous_zero_based_iteration_keys": rows_iteration_keys_contiguous,
            "rows_with_expected_iteration_value_key_schema": rows_iteration_values_expected_schema,
            "observed_output_key_sets": [
                {"keys": list(keys), "rows": count}
                for keys, count in sorted(output_key_sets.items())
            ],
            "observed_iteration_value_key_sets": [
                {"keys": list(keys), "occurrences": count}
                for keys, count in sorted(iteration_value_key_sets.items())
            ],
        },
        "document_id_uniqueness": {
            "scope": "within_each_final_cumulative_retrieval_result_only",
            "total_document_occurrences": total_documents,
            "rows_all_documents_are_objects": rows_all_doc_objects,
            "rows_all_doc_ids_are_nonempty_strings": rows_all_doc_ids_nonempty_strings,
            "rows_with_unique_doc_ids": rows_unique_doc_ids,
            "duplicate_doc_occurrences": duplicate_doc_occurrences,
            "doc_ids_emitted": False,
        },
        "conditional_retrieval_call_inference": {
            "status": "CONDITIONAL_NOT_OBSERVED",
            "assumption": (
                "Only if one initial retrieval and exactly one physical retrieval call "
                "occurred per recorded generation round"
            ),
            "formula": "conditional_calls = recorded_generation_rounds + 1",
            "distribution_under_assumption": _distribution(conditional_calls),
            "must_not_be_reported_as_observed_physical_calls": True,
        },
    }


def _source_locks(
    sources: Mapping[str, Mapping[str, Path]], root: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    locks: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, roles in sources.items():
        locks[dataset] = {}
        for role, relative_path in roles.items():
            path = _resolve(relative_path, root)
            if not path.is_file():
                raise TraceBudgetAuditError(f"missing source: {relative_path}")
            locks[dataset][role] = {
                "path": str(relative_path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return locks


def _verify_frozen_hashes(
    locks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    if set(locks) != set(expected):
        raise TraceBudgetAuditError("dataset set differs from frozen source inventory")
    for dataset, roles in expected.items():
        if set(locks[dataset]) != set(roles):
            raise TraceBudgetAuditError(f"role set differs for {dataset}")
        for role, expected_hash in roles.items():
            observed = locks[dataset][role]["sha256"]
            if observed != expected_hash:
                raise TraceBudgetAuditError(
                    f"frozen SHA256 mismatch for {dataset}/{role}: {observed}"
                )


def build_protocol(
    *, experiment_id: str, locks: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": experiment_id,
        "status": "FROZEN_BEFORE_AGGREGATION",
        "generated_at": generated_at,
        "researcher_authorization": "approved_in_thread_2026-09-04",
        "scope": "historical_trace_rerank_seed42_n300_three_datasets",
        "source_locks": locks,
        "access_contract": {
            "source_may_contain_gold": True,
            "only_consumed_container": "row.output",
            "consumed_values": [
                "len(output.retrieval_result)",
                "output.retrieval_result[*].id for within-row uniqueness counts only",
                "output key names",
                "intermediate_output_iterN nested key names",
            ],
            "unread_output_values": [
                "input_prompt",
                "new_thought",
                "pred",
                "raw_pred",
                "metric_score",
                "document contents",
            ],
            "gold_fields_used": False,
            "gold_fields_emitted": False,
            "row_identity_emitted": False,
            "question_emitted": False,
            "thought_emitted": False,
            "prediction_emitted": False,
        },
        "registered_aggregates": [
            "final cumulative retrieval_result cardinality",
            "recorded generation-round key count",
            "output/iteration structural schema and cardinality",
            "within-row document-id uniqueness counts",
        ],
        "unknowns": {
            "per_call_top_k": "UNKNOWN",
            "physical_retrieval_calls": "UNKNOWN",
            "historical_runtime_implementation_exact_hash": "UNKNOWN",
            "cache_behavior": "UNKNOWN",
            "retry_behavior": "UNKNOWN",
            "wall_time": "UNKNOWN",
        },
        "interpretation_limits": {
            "historical_ircot_style_reference_only": True,
            "fixed_10_matched_control": False,
            "confirmatory_evaluation": False,
            "paper_faithful_ircot_reproduction": False,
            "baseline_modified": False,
        },
    }


def build_report(
    *, experiment_id: str, locks: Mapping[str, Any], datasets: Mapping[str, Any], generated_at: str
) -> dict[str, Any]:
    all_doc_counts: list[int] = []
    all_round_counts: list[int] = []
    for summary in datasets.values():
        for value, count in summary["final_cumulative_document_count"]["histogram"].items():
            all_doc_counts.extend([int(value)] * int(count))
        for value, count in summary["recorded_generation_round_count"]["histogram"].items():
            all_round_counts.extend([int(value)] * int(count))
    return {
        "schema_version": REPORT_SCHEMA,
        "experiment_id": experiment_id,
        "status": "COMPLETE_HISTORICAL_REFERENCE_ONLY",
        "generated_at": generated_at,
        "source_locks": locks,
        "source_access_disclosure": {
            "source_may_contain_gold": True,
            "gold_fields_used": False,
            "gold_fields_emitted": False,
            "qid_question_thought_prediction_emitted": False,
            "aggregate_output_structure_only": True,
        },
        "datasets": datasets,
        "aggregate_three_datasets": {
            "records": len(all_doc_counts),
            "final_cumulative_document_count": _distribution(all_doc_counts),
            "recorded_generation_round_count": _distribution(all_round_counts),
        },
        "unknowns": {
            "per_call_top_k": "UNKNOWN",
            "physical_retrieval_calls": "UNKNOWN",
            "historical_runtime_implementation_exact_hash": "UNKNOWN",
            "cache_behavior": "UNKNOWN",
            "retry_behavior": "UNKNOWN",
            "wall_time": "UNKNOWN",
        },
        "conclusion": {
            "supported": (
                "The frozen historical Trace artifacts expose final cumulative document "
                "counts and recorded generation-round keys; these aggregates describe an "
                "IRCoT-style historical system reference."
            ),
            "not_supported": [
                "a fixed-10 matched-control comparison",
                "exact physical retrieval-call or per-call top-k accounting",
                "a confirmatory result",
                "a paper-faithful IRCoT reproduction",
            ],
        },
    }


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise TraceBudgetAuditError(f"append-only output already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_audit(
    *,
    output_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    root: Path = PROJECT_ROOT,
    sources: Mapping[str, Mapping[str, Path]] = DATASET_SOURCES,
    expected_hashes: Mapping[str, Mapping[str, str]] = FROZEN_SHA256,
    expected_cardinality: int = EXPECTED_CARDINALITY,
) -> dict[str, Any]:
    if output_dir.exists():
        raise TraceBudgetAuditError(f"append-only output directory already exists: {output_dir}")
    locks = _source_locks(sources, root)
    _verify_frozen_hashes(locks, expected_hashes)
    generated_at = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = build_protocol(experiment_id=experiment_id, locks=locks, generated_at=generated_at)
    _write_new_json(output_dir / "protocol.json", protocol)

    dataset_reports: dict[str, Any] = {}
    for dataset, roles in sources.items():
        intermediate_path = _resolve(roles["intermediate"], root)
        try:
            records = json.loads(intermediate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceBudgetAuditError(f"cannot parse {dataset} intermediate source") from exc
        dataset_reports[dataset] = audit_records(
            records, expected_cardinality=expected_cardinality
        )

    report = build_report(
        experiment_id=experiment_id,
        locks=locks,
        datasets=dataset_reports,
        generated_at=generated_at,
    )
    _write_new_json(output_dir / "report.json", report)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": experiment_id,
        "status": report["status"],
        "generated_at": generated_at,
        "artifacts": {
            "protocol.json": {
                "sha256": sha256_file(output_dir / "protocol.json"),
                "bytes": (output_dir / "protocol.json").stat().st_size,
            },
            "report.json": {
                "sha256": sha256_file(output_dir / "report.json"),
                "bytes": (output_dir / "report.json").stat().st_size,
            },
        },
        "script": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_locks_sha256": sha256_json(locks),
        "baseline_modified": False,
    }
    _write_new_json(output_dir / "manifest.json", manifest)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        output_dir=_resolve(args.output_dir, PROJECT_ROOT),
        experiment_id=args.experiment_id,
    )
    print(canonical_json({
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "output_dir": str(args.output_dir),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
