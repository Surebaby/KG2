"""Tests for the append-only Hotpot Controller silver pilot freezer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.prepare import freeze_hotpot_controller_silver_pilot_v1 as freeze
from scripts.prepare.audit_subquestion_v8_cohort_capacity import TrainingInputSpec


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _strict_row(qid: str, question: str, *, level: str = "hard") -> dict:
    source = f"Source {qid}"
    bridge = f"Bridge {qid}"
    final = f"Final {qid}"
    return {
        "id": qid,
        "question": question,
        "golden_answers": [final],
        "metadata": {
            "type": "bridge",
            "level": level,
            "supporting_facts": {
                "title": [source, bridge],
                "sent_id": [0, 0],
            },
            "context": {
                "title": [source, bridge, f"Distractor {qid}"],
                "sentences": [
                    [f"{source} is directly associated with {bridge}."],
                    [f"{bridge} has the requested attribute {final}."],
                    ["This distractor contains no relevant entity."],
                ],
            },
        },
    }


def _sealed_parent_metadata(project: Path) -> tuple[Path, Path]:
    parent = project / "sealed-parent"
    report = {
        "status": "COMPLETE_FROZEN_SCOPE_A_DEV30_PROSPECTIVE300_NO_RESERVE",
        "checks": {
            "all_freeze_gates_pass": True,
            "raw_train_qid_overlap": 0,
            "raw_train_family_overlap": 0,
        },
        "prospective_seal": {"status": "FROZEN_UNOPENED_FOR_METHOD_DEVELOPMENT"},
    }
    report_path = parent / "report.json"
    _write_json(report_path, report)
    manifest = {
        "status": report["status"],
        "outputs": [
            {"path": "report.json", "sha256": freeze._sha256_file(report_path)},
            {
                "path": freeze.SEALED_PROSPECTIVE_FILENAME,
                "sha256": "f" * 64,
            },
        ],
    }
    manifest_path = parent / "manifest.json"
    _write_json(manifest_path, manifest)
    # Deliberately do not create prospective.identity_only.jsonl.  A successful
    # run proves that only its parent metadata, never sealed content, is read.
    return report_path, manifest_path


def _synthetic_inputs(project: Path) -> dict[str, object]:
    questions = [
        "What occupation did Source qid-0 pursue?",
        "Where did Source qid-1 study?",
        "When was Source qid-2 established?",
        "Which language did Source qid-3 use?",
        "How did Source qid-4 operate?",
        "Why did Source qid-5 dissolve?",
    ]
    levels = ("easy", "medium", "hard", "easy", "medium", "hard")
    raw_rows = [
        _strict_row(f"qid-{index}", question, level=levels[index])
        for index, question in enumerate(questions)
    ]
    raw_path = project / "raw-train.jsonl"
    _write_jsonl(raw_path, raw_rows)

    historical_path = project / "historical.jsonl"
    _write_jsonl(
        historical_path,
        [
            {
                "dataset": freeze.DATASET,
                "qid": "qid-0",
                "question": questions[0],
            },
            {
                "dataset": freeze.DATASET,
                "qid": "different-qid-same-family",
                "question": questions[1],
            },
        ],
    )
    training_path = project / "training.jsonl"
    _write_jsonl(
        training_path,
        [{"qid": "training-alias", "question": questions[2]}],
    )
    explicit_path = project / "explicit.jsonl"
    _write_jsonl(
        explicit_path,
        [
            {
                "dataset": freeze.DATASET,
                "qid": "qid-3",
                "question": questions[3],
            }
        ],
    )
    parent_report, parent_manifest = _sealed_parent_metadata(project)
    return {
        "raw_path": raw_path,
        "questions": questions,
        "historical_path": historical_path,
        "training_path": training_path,
        "explicit_path": explicit_path,
        "parent_report": parent_report,
        "parent_manifest": parent_manifest,
    }


def _run_synthetic_freeze(project: Path, inputs: dict[str, object], output: Path):
    historical_path = inputs["historical_path"]
    training_path = inputs["training_path"]
    explicit_path = inputs["explicit_path"]
    assert isinstance(historical_path, Path)
    assert isinstance(training_path, Path)
    assert isinstance(explicit_path, Path)
    return freeze.run_freeze(
        project_root=project,
        raw_train_path=inputs["raw_path"],
        output_dir=output,
        experiment_id="TEST-HOTPOT-SILVER-PILOT",
        selection_salt="test-fixed-salt",
        pilot_size=2,
        level_quotas={"easy": 0, "medium": 1, "hard": 1},
        historical_registry_paths=(historical_path.relative_to(project).as_posix(),),
        training_input_specs=(
            TrainingInputSpec(
                path=training_path.relative_to(project).as_posix(),
                evidence_path="test-evidence.json",
                ledger_state="test-only",
                dataset_hint=freeze.DATASET,
            ),
        ),
        explicit_consumed_identity_paths=(explicit_path.relative_to(project),),
        capacity_inventory_path=None,
        expected_capacity_inventory_sha256=None,
        expected_explicit_consumed_hashes=None,
        expected_raw_train_sha256=None,
        sealed_parent_report_path=inputs["parent_report"],
        sealed_parent_manifest_path=inputs["parent_manifest"],
        expected_sealed_parent_hashes=None,
        enforce_formal_locks=False,
        generated_at_utc="2026-09-04T00:00:00+00:00",
    )


def test_strict_candidate_requires_directed_nonboolean_two_support_chain() -> None:
    valid = _strict_row(
        "valid",
        "What property did Source valid possess?",
    )
    candidate = freeze._strict_candidate_from_row(valid)
    assert candidate.qid == "valid"

    invalid_cases: list[tuple[str, dict, str]] = []

    comparison = deepcopy(valid)
    comparison["metadata"]["type"] = "comparison"
    invalid_cases.append(("comparison", comparison, "not_bridge_type"))

    boolean = deepcopy(valid)
    boolean["golden_answers"] = ["yes"]
    invalid_cases.append(
        ("boolean", boolean, "unsafe_short_or_boolean_final_alias")
    )

    three_supports = deepcopy(valid)
    three_supports["metadata"]["supporting_facts"] = {
        "title": ["Source valid", "Bridge valid", "Distractor valid"],
        "sent_id": [0, 0, 0],
    }
    invalid_cases.append(
        ("three-supports", three_supports, "support_fact_count_not_two")
    )

    bidirectional = deepcopy(valid)
    bidirectional["metadata"]["context"]["sentences"][1][0] += (
        " It also names Source valid."
    )
    invalid_cases.append(
        ("bidirectional", bidirectional, "bidirectional_support_title_link")
    )

    root_missing = deepcopy(valid)
    root_missing["question"] = "What property did the documented subject possess?"
    invalid_cases.append(
        ("root-missing", root_missing, "question_root_title_not_unique")
    )

    bridge_leak = deepcopy(valid)
    bridge_leak["question"] += " Bridge valid"
    invalid_cases.append(
        ("bridge-leak", bridge_leak, "question_root_title_not_unique")
    )

    final_leak = deepcopy(valid)
    final_leak["question"] += " Final valid"
    invalid_cases.append(
        ("final-leak", final_leak, "final_alias_in_question")
    )

    final_leads_second_hop = deepcopy(valid)
    final_leads_second_hop["metadata"]["context"]["sentences"][1][0] = (
        "Final valid is the requested attribute of Bridge valid."
    )
    invalid_cases.append(
        (
            "final-leads-second-hop",
            final_leads_second_hop,
            "final_alias_leads_second_hop_support",
        )
    )

    for label, row, reason in invalid_cases:
        with pytest.raises(freeze.CandidateReject) as exc_info:
            freeze._strict_candidate_from_row(row)
        assert exc_info.value.reason == reason, label


def test_freeze_merges_consumed_union_and_writes_identity_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inputs = _synthetic_inputs(project)
    output = project / "pilot"

    result = _run_synthetic_freeze(project, inputs, output)

    rows = [json.loads(line) for line in (output / "pilot.identity_only.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert len(rows) == 2
    assert {row["qid"] for row in rows} == {"qid-4", "qid-5"}
    assert all(tuple(row) == freeze.OUTPUT_ROW_FIELDS for row in rows)
    assert all(set(row) == {"dataset", "qid", "question"} for row in rows)
    assert not any("q1" in row or "q2" in row for row in rows)

    report = result["report"]
    protocol = result["protocol"]
    manifest = result["manifest"]
    assert report["checks"]["consumed_qid_overlap"] == 0
    assert report["checks"]["consumed_family_overlap"] == 0
    assert report["consumed_union"]["stats"]["source_inventory_entries"] == 3
    assert report["consumed_union"]["stats"]["unique_source_files"] == 3
    assert report["selection"]["eligible_unique_dataset_scoped_families"] == 2
    assert report["selection"]["selected_rows_by_level"] == {
        "easy": 0,
        "medium": 1,
        "hard": 1,
    }
    assert report["checks"]["exact_level_quotas"] is True
    assert protocol["cohort"]["level_quotas"] == {
        "easy": 0,
        "medium": 1,
        "hard": 1,
    }
    assert protocol["gold_boundary"]["final_answer_gold_accessed"] is True
    assert protocol["gold_boundary"]["supporting_facts_gold_accessed"] is True
    assert protocol["gold_boundary"]["candidate_selection_is_gold_screened"] is True
    assert protocol["gold_boundary"]["label_status"].startswith("SILVER_")
    assert protocol["authorization"]["q1_q2_generation"] is False
    assert protocol["authorization"]["training"] is False
    assert protocol["freshness_scope"]["claim"] == (
        "DISJOINT_FROM_ENUMERATED_CONSUMED_UNION_ONLY"
    )
    assert protocol["freshness_scope"]["global_never_seen_claim_allowed"] is False
    assert manifest["q1_q2_generated"] is False
    assert manifest["retrieval_calls"] == 0
    assert manifest["model_calls"] == 0
    assert manifest["sealed_prospective_content_opened"] is False
    assert not (inputs["parent_report"].parent / freeze.SEALED_PROSPECTIVE_FILENAME).exists()

    # A second invocation fails before reading inputs or replacing any artifact.
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_synthetic_freeze(project, inputs, output)


def test_insufficient_capacity_fails_before_creating_output(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inputs = _synthetic_inputs(project)
    output = project / "too-large"
    with pytest.raises(ValueError, match="capacity .* below pilot size"):
        freeze.run_freeze(
            project_root=project,
            raw_train_path=inputs["raw_path"],
            output_dir=output,
            experiment_id="TEST-HOTPOT-SILVER-PILOT",
            selection_salt="test-fixed-salt",
            pilot_size=7,
            level_quotas={"easy": 3, "medium": 2, "hard": 2},
            historical_registry_paths=(),
            training_input_specs=(),
            explicit_consumed_identity_paths=(),
            capacity_inventory_path=None,
            expected_capacity_inventory_sha256=None,
            expected_explicit_consumed_hashes=None,
            expected_raw_train_sha256=None,
            sealed_parent_report_path=inputs["parent_report"],
            sealed_parent_manifest_path=inputs["parent_manifest"],
            expected_sealed_parent_hashes=None,
            enforce_formal_locks=False,
        )
    assert not output.exists()


def test_thresholds_and_future_release_gate_are_frozen_before_generation() -> None:
    assert freeze.PILOT_SIZE == 30
    assert freeze.PILOT_ACCEPTED_MIN == 24
    assert freeze.LEVEL_QUOTAS == {"easy": 10, "medium": 10, "hard": 10}
    assert freeze.FULL_RELEASE_SIZES == {"train": 600, "dev": 60, "confirmation": 30}
    gates = freeze.FUTURE_SILVER_GENERATION_GATES
    assert gates["authorization_granted_by_this_freeze"] is False
    assert gates["producer_candidates_per_identity_per_slot_exact"] == 1
    assert gates["pilot_fixed_denominator"] == 30
    assert gates["pilot_accepted_min"] == 24
    assert gates["pilot_failed_identity_replacement_allowed"] is False
    assert gates["future_release_sizes"] == {
        "train": 600,
        "dev": 60,
        "confirmation": 30,
    }
    assert gates["future_release_accepted_unique_families_min"] == 690
    assert gates["future_release_cross_role_family_overlap_max"] == 0
    assert gates["gold_bridge_injected_as_runtime_observation_allowed"] is False
    assert gates["training_q2_observation_source_exact"] == (
        "train_annotation_intermediate_bound_to_first_hop_support"
    )
    assert gates["runtime_q2_observation_source_exact"] == (
        "strong_sft_reader_prediction_bound_to_retrieved_passage"
    )
    assert gates["context_isolated_ai_review_calls_exact"] == 2
    assert gates["statistically_independent_reviewer_claim_allowed"] is False
    assert gates["ai_disagreement_or_unknown_policy"] == "reject_no_repair"


def test_sealed_prospective_filename_is_fail_closed(tmp_path: Path) -> None:
    prospective = tmp_path / freeze.SEALED_PROSPECTIVE_FILENAME
    prospective.write_text("must not be read\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="must not be opened or hashed"):
        freeze._sha256_file(prospective)
    with pytest.raises(PermissionError, match="must not be opened"):
        freeze._load_json(prospective)


def test_formal_mode_cannot_disable_a_frozen_source_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sources, hashes, thresholds"):
        freeze.run_freeze(
            project_root=tmp_path,
            expected_raw_train_sha256=None,
        )
    assert not (tmp_path / freeze.DEFAULT_OUTPUT_DIR).exists()
