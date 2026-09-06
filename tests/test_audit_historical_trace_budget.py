from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.diagnose import audit_historical_trace_budget as audit


def _row(*, rounds: int, doc_ids: list[str]) -> dict:
    output = {
        "retrieval_result": [
            {
                "id": doc_id,
                "contents": "SECRET_DOCUMENT_TEXT",
                "source": "wiki18",
            }
            for doc_id in doc_ids
        ],
        "pred": "SECRET_PREDICTION",
        "raw_pred": "SECRET_RAW_PREDICTION",
        "metric_score": {"em": 1.0},
    }
    for index in range(rounds):
        output[f"intermediate_output_iter{index}"] = {
            "input_prompt": "SECRET_PROMPT",
            "new_thought": "SECRET_THOUGHT",
        }
    return {
        "id": "SECRET_QID",
        "question": "SECRET_QUESTION",
        "golden_answers": ["SECRET_GOLD"],
        "output": output,
    }


def _write_sources(root: Path) -> tuple[dict, dict]:
    sources: dict[str, dict[str, Path]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        dataset_dir = root / dataset
        dataset_dir.mkdir(parents=True)
        paths = {
            "intermediate": dataset_dir / "intermediate_data.json",
            "config": dataset_dir / "config.yaml",
            "manifest": dataset_dir / "manifest.json",
            "metric": dataset_dir / "metric_score.json",
        }
        paths["intermediate"].write_text(
            json.dumps([_row(rounds=2, doc_ids=["d1", "d2"])]),
            encoding="utf-8",
        )
        paths["config"].write_text("method: trace\n", encoding="utf-8")
        paths["manifest"].write_text("{}\n", encoding="utf-8")
        paths["metric"].write_text("{}\n", encoding="utf-8")
        sources[dataset] = {
            role: path.relative_to(root) for role, path in paths.items()
        }
        hashes[dataset] = {
            role: audit.sha256_file(path) for role, path in paths.items()
        }
    return sources, hashes


def test_audit_records_reports_only_aggregate_structure() -> None:
    records = [
        _row(rounds=2, doc_ids=["a", "b"]),
        _row(rounds=4, doc_ids=["a", "a", "c"]),
    ]
    result = audit.audit_records(records, expected_cardinality=2)

    assert result["final_cumulative_document_count"] == {
        "n": 2,
        "min": 2,
        "max": 3,
        "mean": 2.5,
        "median": 2.5,
        "histogram": {"2": 1, "3": 1},
    }
    assert result["recorded_generation_round_count"]["mean"] == 3.0
    assert result["document_id_uniqueness"]["rows_with_unique_doc_ids"] == 1
    assert result["document_id_uniqueness"]["duplicate_doc_occurrences"] == 1
    assert result["conditional_retrieval_call_inference"]["status"] == (
        "CONDITIONAL_NOT_OBSERVED"
    )

    serialized = json.dumps(result)
    for forbidden in (
        "SECRET_QID",
        "SECRET_QUESTION",
        "SECRET_GOLD",
        "SECRET_PROMPT",
        "SECRET_THOUGHT",
        "SECRET_PREDICTION",
        "SECRET_DOCUMENT_TEXT",
    ):
        assert forbidden not in serialized


def test_iteration_gap_is_counted_without_reading_thought() -> None:
    row = _row(rounds=2, doc_ids=["a"])
    row["output"]["intermediate_output_iter3"] = row["output"].pop(
        "intermediate_output_iter1"
    )
    result = audit.audit_records([row], expected_cardinality=1)
    assert result["recorded_generation_round_count"]["min"] == 2
    assert result["schema"]["rows_with_contiguous_zero_based_iteration_keys"] == 0


def test_cardinality_and_retrieval_schema_fail_closed() -> None:
    with pytest.raises(audit.TraceBudgetAuditError, match="cardinality mismatch"):
        audit.audit_records([], expected_cardinality=1)

    row = _row(rounds=1, doc_ids=["a"])
    row["output"]["retrieval_result"] = "not-a-list"
    with pytest.raises(audit.TraceBudgetAuditError, match="retrieval_result"):
        audit.audit_records([row], expected_cardinality=1)


def test_run_audit_locks_sources_and_is_append_only(tmp_path: Path) -> None:
    sources, hashes = _write_sources(tmp_path)
    output_dir = tmp_path / "audit"
    report = audit.run_audit(
        output_dir=output_dir,
        experiment_id="TEST-HISTORICAL-TRACE-BUDGET-V1",
        root=tmp_path,
        sources=sources,
        expected_hashes=hashes,
        expected_cardinality=1,
    )

    assert report["status"] == "COMPLETE_HISTORICAL_REFERENCE_ONLY"
    assert report["source_access_disclosure"]["gold_fields_used"] is False
    assert report["unknowns"]["physical_retrieval_calls"] == "UNKNOWN"
    assert (output_dir / "protocol.json").is_file()
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "manifest.json").is_file()

    serialized = (output_dir / "report.json").read_text(encoding="utf-8")
    assert "SECRET_GOLD" not in serialized
    assert "SECRET_THOUGHT" not in serialized

    with pytest.raises(audit.TraceBudgetAuditError, match="already exists"):
        audit.run_audit(
            output_dir=output_dir,
            experiment_id="TEST-HISTORICAL-TRACE-BUDGET-V1",
            root=tmp_path,
            sources=sources,
            expected_hashes=hashes,
            expected_cardinality=1,
        )


def test_hash_drift_fails_before_output_creation(tmp_path: Path) -> None:
    sources, hashes = _write_sources(tmp_path)
    hashes["hotpotqa"]["config"] = "0" * 64
    output_dir = tmp_path / "audit"
    with pytest.raises(audit.TraceBudgetAuditError, match="SHA256 mismatch"):
        audit.run_audit(
            output_dir=output_dir,
            experiment_id="TEST-HISTORICAL-TRACE-BUDGET-V1",
            root=tmp_path,
            sources=sources,
            expected_hashes=hashes,
            expected_cardinality=1,
        )
    assert not output_dir.exists()
