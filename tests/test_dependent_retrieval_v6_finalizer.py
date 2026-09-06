from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare import finalize_dependent_retrieval_v6 as finalizer
from scripts.prepare import freeze_dependent_retrieval_v6 as freeze
from tests.test_freeze_dependent_retrieval_v6 import synthetic_build_args


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _passage(document_id: str) -> dict[str, str]:
    return {"id": document_id, "title": f"Title {document_id}", "contents": f"Text {document_id}"}


def _rows() -> tuple[list[dict], list[dict], list[dict]]:
    arm_a: list[dict] = []
    arm_b: list[dict] = []
    details: list[dict] = []
    for dataset in finalizer.DATASETS:
        for index in range(30):
            qid = f"{dataset}-{index}"
            question = f"Question {dataset} {index}?"
            passages_a = [_passage(f"{qid}-a-{rank}") for rank in range(10)]
            changed = index < 15
            passages_b = list(passages_a)
            new_key = f"id:{qid}-new"
            if changed:
                passages_b[-1] = _passage(f"{qid}-new")
            common = {
                "row_id": f"dependent-retrieval-pilot::{dataset}::{qid}",
                "question_key": f"{dataset}::{qid}",
                "dataset": dataset, "qid": qid, "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "split": "dev", "gold_access": False,
                "kg_subgraph": [], "legacy_kg_sha256": hashlib.sha256(b"[]").hexdigest(),
            }
            arm_a.append({
                **common, "arm": "A_question_only", "retrieved_passages": passages_a,
                "passages_sha256": _json_hash(passages_a),
            })
            arm_b.append({
                **common, "arm": "B_question_anchored_dependent",
                "retrieved_passages": passages_b,
                "passages_sha256": _json_hash(passages_b),
                "fallback_to_a": not changed,
            })
            query = question + "\nsubject >> relation"
            merge = None
            if changed:
                merge = {
                    "selected_new": [{"document_key": new_key, "hop_id": "hop_2", "score": 2.0}],
                    "evicted_originals": [{
                        "document_key": f"id:{qid}-a-9", "original_rank": 10,
                        "score": 1.0, "replacement_score": 2.0,
                    }],
                }
            details.append({
                "dataset": dataset, "qid": qid, "gold_access": False,
                "execution_status": "executed_changed" if changed else "fallback_no_candidate_strictly_better",
                "plan_executable": True, "has_dependent_step": True,
                "dependent_query_count": 1, "second_hop_query_count": 1,
                "hops": [{
                    "hop_id": "hop_2", "dependencies": ["step_1"],
                    "query_variants": [{
                        "query": query,
                        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                        "query_variant_index": 1,
                    }],
                }],
                "final_ce_pair_count": 3,
                "all_final_ce_pairs_use_exact_original_question": True,
                "merge": merge,
                "safety": {
                    "output_count": 10, "prefix8_exact": True,
                    "unauthorized_original_displacements": 0,
                    "root_passages_injected": 0, "duplicate_output_documents": 0,
                    "fallback_exact": not changed,
                },
            })
    return arm_a, arm_b, details


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _materialization(tmp_path: Path) -> tuple[Path, Path]:
    args = synthetic_build_args(tmp_path / "freeze")
    protocol = freeze.build_protocol(**args)
    prereg_path = tmp_path / "prereg" / "protocol.json"
    prereg_path.parent.mkdir(parents=True)
    prereg_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    arm_a, arm_b, details = _rows()
    material = tmp_path / "material"
    a_path = _write_jsonl(material / "arm_a.jsonl", arm_a)
    b_path = _write_jsonl(material / "arm_b.jsonl", arm_b)
    d_path = _write_jsonl(material / "execution_details.jsonl", details)
    observed = finalizer.audit_materialization(arm_a, arm_b, details)
    report = {
        "schema_version": finalizer.EXPECTED_REPORT_SCHEMA,
        "status": finalizer.EXPECTED_REPORT_STATUS,
        "experiment_id": freeze.EXPERIMENT_IDS["materialization"],
        "development_only": True, "gold_access": False,
        "preregistration": freeze.file_lock(prereg_path),
        "runtime_locks": {
            "preregistration": freeze.file_lock(prereg_path),
            "inputs": protocol["inputs"], "code": protocol["code"],
            "models": protocol["models"], "settings": protocol["settings"],
        },
        "settings": protocol["settings"],
        "by_dataset": observed["by_dataset"],
        "safety_summary": {
            "all_top10": True, "prefix8_exact": True,
            "unauthorized_original_displacements": 0,
            "root_passages_injected": 0, "duplicate_output_documents": 0,
            "fallback_exact": True, "runtime_errors": 0,
            "all_dependent_queries_start_with_exact_question": True,
            "max_query_variants_per_logical_hop": 2,
            "all_final_ce_pairs_use_exact_original_question": True,
        },
        "outputs": {
            "arm_a": {"path": str(a_path.resolve()), "sha256": freeze.sha256_file(a_path)},
            "arm_b": {"path": str(b_path.resolve()), "sha256": freeze.sha256_file(b_path)},
            "execution_details": {"path": str(d_path.resolve()), "sha256": freeze.sha256_file(d_path)},
        },
    }
    report_path = material / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prereg_path, report_path


def test_v6_finalizer_recomputes_all_gold_free_gates() -> None:
    arm_a, arm_b, details = _rows()
    result = finalizer.audit_materialization(arm_a, arm_b, details)
    assert result["identity_join_rate"] == 1.0
    assert result["duplicate_output_documents"] == 0
    assert result["duplicate_dependent_queries"] == 0
    assert result["all_dependent_queries_start_with_exact_question"] is True
    assert result["all_final_ce_pairs_use_exact_original_question"] is True
    assert result["by_dataset"]["hotpotqa"]["retained_new_dependent_document_question_rate"] == 0.5


def test_v6_finalizer_rejects_query_without_exact_question_prefix() -> None:
    arm_a, arm_b, details = _rows()
    details[0]["hops"][0]["query_variants"][0]["query"] = "relation only"
    details[0]["hops"][0]["query_variants"][0]["query_sha256"] = hashlib.sha256(b"relation only").hexdigest()
    with pytest.raises(ValueError, match="dependent-query safety gate failed"):
        finalizer.audit_materialization(arm_a, arm_b, details)


def test_v6_finalizer_rejects_duplicate_output_document() -> None:
    arm_a, arm_b, details = _rows()
    arm_b[0]["retrieved_passages"][-1] = arm_b[0]["retrieved_passages"][-2]
    arm_b[0]["passages_sha256"] = _json_hash(arm_b[0]["retrieved_passages"])
    with pytest.raises(ValueError, match="duplicate passage gate failed"):
        finalizer.audit_materialization(arm_a, arm_b, details)


def test_v6_mechanism_failure_stops_in_gold_free_validation(tmp_path: Path) -> None:
    prereg_path, report_path = _materialization(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["by_dataset"]["musique"]["retained_new_dependent_document_question_rate"] = 0.49
    report_path.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reported/recomputed mechanism metric differs"):
        finalizer.validate_gold_free_materialization(
            report_path=report_path, prereg_path=prereg_path
        )


def test_v6_failed_gate_is_append_only_and_never_opens_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path, report_path = _materialization(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["by_dataset"]["hotpotqa"]["retained_new_dependent_document_question_rate"] = 0.49
    report_path.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "failed_finalizer"
    missing_gold = tmp_path / "must-not-be-opened.jsonl"
    monkeypatch.setattr("sys.argv", [
        "finalize_dependent_retrieval_v6.py",
        "--retrieval_report", str(report_path),
        "--preregistration", str(prereg_path),
        "--v4_frozen_protocol", str(tmp_path / "not-reached-v4.json"),
        "--hotpot_dev", str(missing_gold),
        "--musique_dev", str(missing_gold),
        "--output_dir", str(output),
        "--experiment_id", freeze.EXPERIMENT_IDS["post_materialization_freeze"],
        "--evaluation_experiment_id", freeze.EXPERIMENT_IDS["answer_evaluation"],
    ])
    with pytest.raises(ValueError, match="reported/recomputed mechanism metric differs"):
        finalizer.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_RUNTIME"
    assert not (output / "arm_a.scored.jsonl").exists()
