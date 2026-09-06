from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from kgproweight.retrieval import subanswer_v7
from scripts.diagnose import audit_v7_subanswer_contract_counterfactual as audit


def _task(
    *,
    dataset: str,
    qid: str,
    subject: str = "Ada",
    passage_text: str = "Ada studied at Georgetown University in 1999.",
) -> dict:
    target_type = "relation_graph" if dataset == "hotpotqa" else "subquery_graph"
    step = (
        {
            "step": 1,
            "subject": subject,
            "relation_label": "educated at",
            "pid": "P69",
            "output_slot": "hop_1",
            "dependencies": [],
        }
        if dataset == "hotpotqa"
        else {
            "step": 1,
            "subquery_template": f"{subject} >> educated at",
            "output_slot": "hop_1",
            "dependencies": [],
        }
    )
    passages = [{"id": f"doc-{qid}", "title": "Ada", "text": passage_text}]
    question = f"Where did {subject} study?"
    return {
        "task_id": f"task-{dataset}-{qid}",
        "question_key": f"{dataset}::{qid}",
        "dataset": dataset,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "target_type": target_type,
        "producer_slot": "slot_1",
        "step": step,
        "step_sha256": audit.sha256_json(step),
        "producer_passages": passages,
        "producer_passages_sha256": audit.sha256_json(passages),
        "gold_access": False,
    }


def _subanswer(task: dict, response: str) -> dict:
    verification = subanswer_v7.parse_and_verify_subanswer(
        response,
        task["question"],
        task["step"],
        task["producer_passages"],
        target_type=task["target_type"],
    )
    try:
        subanswer_v7.parse_subanswer_response(response)
        parse_valid = True
        parse_error = None
    except subanswer_v7.SubanswerParseError as exc:
        parse_valid = False
        parse_error = exc.code
    telemetry = {
        "input_task_sha256": audit.sha256_json(task),
        "raw_response": response,
        "raw_response_sha256": audit.sha256_text(response),
        "raw_response_utf8_bytes": len(response.encode("utf-8")),
        "strict_parse": {"valid": parse_valid, "error_code": parse_error},
        "verification": verification,
        "prompt_passages_sha256": task["producer_passages_sha256"],
        "verifier_passages_sha256": task["producer_passages_sha256"],
        "same_passage_bytes_for_prompt_and_verifier": True,
        "gold_access": False,
        "network_access": False,
    }
    return {
        key: task[key]
        for key in audit.IDENTITY_FIELDS
    } | {
        "verified": bool(verification["verified"]),
        "verified_answer": verification.get("verified_answer"),
        "telemetry": telemetry,
        "gold_access": False,
    }


def _response(
    answer: str,
    *,
    doc_id: str,
    answer_type: str = "other",
    abstain: object = False,
    citations: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "cited_doc_ids": citations if citations is not None else [doc_id],
            "answer_type": answer_type,
            "abstain": abstain,
        },
        sort_keys=True,
    )


def test_surface_class_inference_uses_frozen_date_number_entity_order() -> None:
    assert audit.infer_surface_class("1999") == "date"
    assert audit.infer_surface_class("12 million people") == "number"
    assert audit.infer_surface_class("Georgetown University") == "entity"


def test_p0_p1_p2_change_only_the_registered_contract_fields() -> None:
    task_other = _task(dataset="hotpotqa", qid="q-other")
    task_empty_abstain = _task(dataset="hotpotqa", qid="q-abstain")
    task_year = _task(dataset="musique", qid="q-year")
    task_echo = _task(dataset="musique", qid="q-echo", passage_text="Ada was a scholar.")
    responses = [
        _response(
            "Georgetown University",
            doc_id="doc-q-other",
            answer_type="other",
        ),
        _response(
            "Georgetown University",
            doc_id="doc-q-abstain",
            answer_type="other",
            abstain="",
        ),
        _response("1999", doc_id="doc-q-year", answer_type="other"),
        _response("Ada", doc_id="doc-q-echo", answer_type="other"),
    ]
    tasks = [task_other, task_empty_abstain, task_year, task_echo]
    rows = audit.audit_counterfactual_rows(
        tasks,
        [_subanswer(task, response) for task, response in zip(tasks, responses)],
    )

    assert rows[0]["p0_current"]["reason"] == "non_extractive_answer_type:other"
    assert rows[0]["p1_surface_class_inferred"]["verified"] is True
    assert rows[0]["p1_surface_class_inferred"]["effective_answer_type"] == "entity"

    assert rows[1]["p0_current"]["reason"] == "parse_error:abstain_not_boolean"
    assert rows[1]["p1_surface_class_inferred"]["verified"] is False
    assert rows[1]["p2_contract_upper_bound"]["verified"] is True
    assert rows[1]["p2_contract_upper_bound"]["abstain_empty_string_coerced"] is True

    assert rows[2]["p1_surface_class_inferred"]["effective_answer_type"] == "date"
    assert rows[2]["p1_surface_class_inferred"]["verified"] is True
    assert rows[3]["p1_surface_class_inferred"]["reason"] == "subject_echo"
    assert rows[3]["p2_contract_upper_bound"]["reason"] == "subject_echo"


def test_p2_fails_closed_for_two_citations_extra_fields_and_nonempty_boolean() -> None:
    two_citations = _response(
        "Georgetown University",
        doc_id="doc",
        abstain="",
        citations=["doc", "other"],
    )
    candidate, error, coerced, original_error = audit._parse_candidate(
        two_citations, allow_empty_string_abstain=True
    )
    assert candidate is None
    assert error == "p2_empty_abstain_coercion_precondition"
    assert coerced is False
    assert original_error == "abstain_not_boolean"

    extra = json.dumps(
        {
            "answer": "Georgetown University",
            "cited_doc_ids": ["doc"],
            "answer_type": "other",
            "abstain": "",
            "confidence": 1,
        }
    )
    candidate, error, coerced, original_error = audit._parse_candidate(
        extra, allow_empty_string_abstain=True
    )
    assert candidate is None
    assert error == "field_set"
    assert coerced is False
    assert original_error == "field_set"

    boolean = _response("yes", doc_id="doc", answer_type="other", abstain="")
    candidate, error, coerced, original_error = audit._parse_candidate(
        boolean, allow_empty_string_abstain=True
    )
    assert candidate is not None and error is None and coerced is True
    assert original_error == "abstain_not_boolean"
    assert audit.infer_surface_class(candidate["answer"]) == "entity"


def test_p0_drift_stops_before_p1_or_p2(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(dataset="hotpotqa", qid="q-drift")
    response = _response("Georgetown University", doc_id="doc-q-drift")
    row = _subanswer(task, response)
    row["telemetry"]["verification"] = dict(row["telemetry"]["verification"])
    row["telemetry"]["verification"]["reason"] = "tampered"

    def forbidden_counterfactual(*args, **kwargs):
        raise AssertionError("P1/P2 must not run after a P0 mismatch")

    monkeypatch.setattr(audit, "_evaluate_inferred_contract", forbidden_counterfactual)
    with pytest.raises(audit.CounterfactualAuditError, match="P0 verification drift"):
        audit.audit_counterfactual_rows([task], [row])


def test_task_gold_key_and_raw_data_path_fail_closed(tmp_path: Path) -> None:
    task = _task(dataset="hotpotqa", qid="q-gold")
    task["supporting_facts"] = []
    response = _response("Georgetown University", doc_id="doc-q-gold")
    with pytest.raises(audit.CounterfactualAuditError, match="exact v7 schema"):
        audit.audit_counterfactual_rows([task], [_subanswer({k: v for k, v in task.items() if k != "supporting_facts"}, response)])

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    forbidden = raw / "gold.json"
    forbidden.write_text("{}", encoding="utf-8")
    with pytest.raises(audit.CounterfactualAuditError, match="data/raw"):
        audit._safe_input(forbidden, project_root=tmp_path, label="forbidden")


def test_frozen_artifact_reproduces_p0_and_expected_counterfactual_counts() -> None:
    rows, summary, _ = audit.audit_frozen_inputs_in_memory()
    assert len(rows) == 41
    observed = {
        dataset: {
            condition: summary["by_dataset"][dataset][condition][
                "mechanically_verified"
            ]
            for condition in (
                "p0_current",
                "p1_surface_class_inferred",
                "p2_contract_upper_bound",
            )
        }
        for dataset in audit.DATASETS
    }
    assert observed == {
        "hotpotqa": {
            "p0_current": 3,
            "p1_surface_class_inferred": 11,
            "p2_contract_upper_bound": 15,
        },
        "musique": {
            "p0_current": 3,
            "p1_surface_class_inferred": 17,
            "p2_contract_upper_bound": 20,
        },
    }


def test_protocol_is_code_and_input_bound_and_outputs_are_append_only(
    tmp_path: Path,
) -> None:
    rows, summary, provenance = audit.audit_frozen_inputs_in_memory()
    experiment_id = "SUBQUESTION-DECOMPOSITION-V8-PHASE0-UNIT-TEST-V1"
    protocol = audit.build_protocol_document(
        experiment_id=experiment_id,
        provenance=provenance,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    authenticated = audit.authenticate_protocol(
        protocol_path,
        experiment_id=experiment_id,
        provenance=provenance,
        project_root=tmp_path,
    )
    output = tmp_path / "run"
    report = audit.write_audit_artifacts(
        rows=rows,
        summary=summary,
        provenance=authenticated,
        output_dir=output,
        experiment_id=experiment_id,
    )
    assert report["status"] == "PASS_P1_TYPE_ONLY_INTERFACE_DIAGNOSIS"
    assert report["gold_access"] is False
    assert report["protocol"]["sha256"] == audit.sha256_file(protocol_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == report["status"]
    assert manifest["model_calls"] == manifest["retrieval_calls"] == 0
    with pytest.raises(FileExistsError):
        audit.write_audit_artifacts(
            rows=rows,
            summary=summary,
            provenance=authenticated,
            output_dir=output,
            experiment_id=experiment_id,
        )


def test_protocol_drift_fails_before_output(tmp_path: Path) -> None:
    _, _, provenance = audit.audit_frozen_inputs_in_memory()
    experiment_id = "SUBQUESTION-DECOMPOSITION-V8-PHASE0-DRIFT-TEST-V1"
    protocol = audit.build_protocol_document(
        experiment_id=experiment_id,
        provenance=provenance,
    )
    protocol["decision_gate"]["verified_rate_min_each_dataset"] = 0.39
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(audit.CounterfactualAuditError, match="decision_gate"):
        audit.authenticate_protocol(
            protocol_path,
            experiment_id=experiment_id,
            provenance=provenance,
            project_root=tmp_path,
        )


def test_integrity_failure_record_contains_no_p1_p2_counts(tmp_path: Path) -> None:
    _, _, provenance = audit.audit_frozen_inputs_in_memory()
    experiment_id = "SUBQUESTION-DECOMPOSITION-V8-PHASE0-FAIL-RECORD-TEST-V1"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            audit.build_protocol_document(
                experiment_id=experiment_id,
                provenance=provenance,
            )
        ),
        encoding="utf-8",
    )
    authenticated = audit.authenticate_protocol(
        protocol_path,
        experiment_id=experiment_id,
        provenance=provenance,
        project_root=tmp_path,
    )
    output = tmp_path / "failed"
    audit.write_integrity_failure(
        output_dir=output,
        experiment_id=experiment_id,
        provenance=authenticated,
        error=audit.CounterfactualAuditError("P0 verification drift for synthetic task"),
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL_STOP_INTEGRITY_REPLAY_MISMATCH"
    assert report["p0_replay_complete"] is False
    assert report["p1_p2_executed"] is False
    assert "summary" not in report
    assert not (output / "per_task_counterfactual.jsonl").exists()


def test_freeze_protocol_is_exclusive_and_immediately_authenticates(
    tmp_path: Path,
) -> None:
    _, _, provenance = audit.audit_frozen_inputs_in_memory()
    experiment_id = "SUBQUESTION-DECOMPOSITION-V8-PHASE0-FREEZE-TEST-V1"
    protocol_path = tmp_path / "freeze" / "protocol.json"
    output_dir = tmp_path / "audit-output-must-not-exist"
    document = audit.freeze_protocol(
        protocol_path=protocol_path,
        experiment_id=experiment_id,
        provenance=provenance,
        project_root=tmp_path,
    )
    assert document["researcher_authorization"] == (
        "approved_in_thread_2026-09-04"
    )
    assert "created_at_utc" in document
    assert protocol_path.is_file()
    assert not output_dir.exists()
    authenticated = audit.authenticate_protocol(
        protocol_path,
        experiment_id=experiment_id,
        provenance=provenance,
        project_root=tmp_path,
    )
    assert authenticated["protocol"]["sha256"] == audit.sha256_file(protocol_path)
    with pytest.raises(FileExistsError, match="already exists"):
        audit.freeze_protocol(
            protocol_path=protocol_path,
            experiment_id=experiment_id,
            provenance=provenance,
            project_root=tmp_path,
        )
