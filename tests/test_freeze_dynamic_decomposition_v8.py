from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare import audit_subquestion_v8_cohort_capacity as capacity
from scripts.prepare import freeze_dynamic_decomposition_v8 as freeze


SECRET_A = "SECRET_GOLD_ALPHA"
SECRET_B = "SECRET_GOLD_BETA"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture()
def identity_sources(tmp_path):
    project_root = tmp_path / "project"
    data_root = project_root / "data"
    words = (
        "amber",
        "birch",
        "coral",
        "denim",
        "elm",
        "flint",
        "granite",
        "hazel",
    )
    dev_rows: dict[str, list[dict]] = {}
    for dataset in freeze.DATASETS:
        train_question = f"what links the {dataset} training source to its target?"
        _write_jsonl(
            data_root / dataset / "train.jsonl",
            [
                {
                    "id": f"{dataset}-train-0",
                    "question": train_question,
                    "golden_answers": [SECRET_A],
                }
            ],
        )
        rows = [
            {
                "id": f"{dataset}-dev-{index}",
                "question": f"what {word} relation identifies the {dataset} target?",
                "golden_answers": [SECRET_A],
                "metadata": {"supporting_facts": SECRET_B},
            }
            for index, word in enumerate(words)
        ]
        # This row has the raw-train lexical family and must be excluded.
        rows.append(
            {
                "id": f"{dataset}-raw-train-family",
                "question": train_question,
                "golden_answers": [SECRET_A],
            }
        )
        dev_rows[dataset] = rows
        _write_jsonl(data_root / dataset / "dev.jsonl", rows)

    history_relative = "outputs/audits/unit_history/cohort.question_only.jsonl"
    history_rows = [
        {
            "dataset": dataset,
            "qid": f"{dataset}-dev-0",
            "question": dev_rows[dataset][0]["question"],
            "answer": SECRET_A,
        }
        for dataset in freeze.DATASETS
    ]
    _write_jsonl(project_root / history_relative, history_rows)

    training_relative = "data/silver_data/unit_training/silver_train.jsonl"
    training_rows = [
        {
            "dataset": dataset,
            "qid": f"{dataset}-train-0",
            "question": f"what links the {dataset} training source to its target?",
            "answer": SECRET_B,
        }
        for dataset in freeze.DATASETS
    ]
    _write_jsonl(project_root / training_relative, training_rows)
    evidence_relative = "configs/training/unit_training.json"
    evidence_path = project_root / evidence_relative
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps({"training_input": training_relative}) + "\n", encoding="utf-8"
    )
    specs = (
        capacity.TrainingInputSpec(
            training_relative,
            evidence_relative,
            capacity.COMPLETED_TRAINING,
        ),
    )
    return {
        "project_root": project_root,
        "data_root": data_root,
        "history": (history_relative,),
        "training_specs": specs,
        "dev_rows": dev_rows,
    }


def _run_fixture(identity_sources, output_dir: Path):
    return freeze.run_freeze(
        project_root=identity_sources["project_root"],
        data_root=identity_sources["data_root"],
        capacity_audit_dir=identity_sources["project_root"] / "unused-capacity-lock",
        output_dir=output_dir,
        experiment_id="SUBQUESTION-V8-IDENTITY-FREEZE-UNIT",
        seed=123,
        development_per_dataset=1,
        prospective_per_dataset=2,
        historical_registry_paths=identity_sources["history"],
        training_input_specs=identity_sources["training_specs"],
        verify_formal_capacity_audit=False,
        generated_at_utc="2026-09-04T00:00:00+00:00",
    )


def test_freeze_is_scope_a_disjoint_and_output_is_identity_only(
    identity_sources, tmp_path
):
    output_dir = tmp_path / "freeze"
    report = _run_fixture(identity_sources, output_dir)
    development = _read_jsonl(output_dir / "development.identity_only.jsonl")
    prospective = _read_jsonl(output_dir / "prospective.identity_only.jsonl")
    combined = development + prospective

    assert len(development) == 3
    assert len(prospective) == 6
    assert all(set(row) == set(freeze.OUTPUT_ROW_FIELDS) for row in combined)
    assert SECRET_A not in (output_dir / "development.identity_only.jsonl").read_text()
    assert SECRET_B not in (output_dir / "prospective.identity_only.jsonl").read_text()
    assert all(not row["qid"].endswith("dev-0") for row in combined)
    assert all(not row["qid"].endswith("raw-train-family") for row in combined)
    assert report["checks"]["all_freeze_gates_pass"] is True
    assert report["checks"][
        "development_prospective_dataset_scoped_qid_overlap"
    ] == 0
    assert report["checks"][
        "development_prospective_dataset_scoped_family_overlap"
    ] == 0
    assert report["selection_contract"]["gold_used_for_selection"] is False
    assert report["selection_contract"]["reserve_per_dataset"] == 0


def test_selection_is_independent_of_gold_values(identity_sources, tmp_path):
    first_dir = tmp_path / "first"
    _run_fixture(identity_sources, first_dir)

    for dataset in freeze.DATASETS:
        path = identity_sources["data_root"] / dataset / "dev.jsonl"
        rows = _read_jsonl(path)
        for row in rows:
            row["golden_answers"] = [SECRET_B, "CHANGED_WITHOUT_IDENTITY_CHANGE"]
            row["metadata"] = {"decomposition": "CHANGED_GOLD_STRUCTURE"}
        _write_jsonl(path, rows)

    second_dir = tmp_path / "second"
    _run_fixture(identity_sources, second_dir)
    assert (first_dir / "development.identity_only.jsonl").read_bytes() == (
        second_dir / "development.identity_only.jsonl"
    ).read_bytes()
    assert (first_dir / "prospective.identity_only.jsonl").read_bytes() == (
        second_dir / "prospective.identity_only.jsonl"
    ).read_bytes()


def test_identity_projection_never_accesses_gold_field():
    class GuardedRow(dict):
        def get(self, key, default=None):
            if key in {"answer", "golden_answers", "decomposition", "metadata"}:
                raise AssertionError(f"forbidden field accessed: {key}")
            return super().get(key, default)

    row = GuardedRow(
        {
            "id": "safe-id",
            "question": "what relation connects the safe source?",
            "golden_answers": [SECRET_A],
            "metadata": {"decomposition": SECRET_B},
        }
    )
    candidate = freeze._candidate_from_identity_fields(row, dataset="hotpotqa")
    assert candidate is not None
    assert candidate.qid == "safe-id"


def test_selection_is_deterministic_and_one_per_family():
    candidates = {
        "a": freeze.Candidate("hotpotqa", "a", "q a", "01", "family-a"),
        "b": freeze.Candidate("hotpotqa", "b", "q b", "02", "family-b"),
        "c": freeze.Candidate("hotpotqa", "c", "q c", "03", "family-c"),
        "d": freeze.Candidate("hotpotqa", "d", "q d", "04", "family-c"),
    }
    first = freeze.select_scope_a_cohorts(
        dataset="hotpotqa",
        candidates=candidates,
        eligible_qids=set(candidates),
        development_n=1,
        prospective_n=2,
        seed=99,
    )
    second = freeze.select_scope_a_cohorts(
        dataset="hotpotqa",
        candidates=candidates,
        eligible_qids=set(reversed(tuple(candidates))),
        development_n=1,
        prospective_n=2,
        seed=99,
    )
    assert first == second
    selected = first[0] + first[1]
    assert len({item.family_sha256 for item in selected}) == 3


def test_append_only_refusal(identity_sources, tmp_path):
    output_dir = tmp_path / "freeze"
    _run_fixture(identity_sources, output_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_fixture(identity_sources, output_dir)


def test_capacity_lock_rejects_tampered_output_hash(tmp_path):
    audit = tmp_path / "capacity"
    audit.mkdir()
    report = {
        "experiment_id": capacity.EXPERIMENT_ID,
        "status": capacity.LIMITED_PASS_STATUS,
        "scope_a_strict": {"balanced_all_datasets_gate_n330": True},
        "checks": {
            "scope_a_eligible_exclusion_overlaps_zero": True,
            "training_inputs_within_corresponding_raw_train": True,
        },
    }
    inventory = {
        "historical_evaluation_protocol_registries": [
            {"path": path, "sha256": "x"}
            for path in capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
        ],
        "local_training_inputs": [
            {"path": spec.path, "sha256": "x"}
            for spec in capacity.LOCAL_TRAINING_INPUT_SPECS
        ],
        "training_input_manifest_config_evidence": [],
    }
    (audit / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (audit / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    manifest = {
        "experiment_id": capacity.EXPERIMENT_ID,
        "outputs": [
            {"path": "report.json", "sha256": "deliberately-wrong"},
            {
                "path": "inventory.json",
                "sha256": freeze._sha256_file(audit / "inventory.json"),
            },
        ],
    }
    (audit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="report.json hash mismatch"):
        freeze._capacity_artifact_lock(audit)
