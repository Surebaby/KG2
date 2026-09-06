import ast
import json
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare import audit_subquestion_v8_cohort_capacity as capacity
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


SECRET_ANSWER = "NEVER_EMIT_THIS_GOLD_ANSWER_7f39"
SECRET_SUPPORT = "NEVER_EMIT_THIS_SUPPORT_19ac"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def capacity_sources(tmp_path):
    project_root = tmp_path / "project"
    data_root = project_root / "data"

    hotpot_train = [
        {
            "id": "hotpot-train-template",
            "question": "Who directed Film Alpha in 2007?",
            "answer": SECRET_ANSWER,
            "supporting_facts": [SECRET_SUPPORT],
        },
        {
            "id": "hotpot-raw-train-qid-collision",
            "question": "When did the example treaty enter force?",
            "answer": SECRET_ANSWER,
        },
    ]
    hotpot_dev = [
        {
            "id": "hotpot-dev-train-family",
            "question": "Who directed Film Beta in 1999?",
            "answer": SECRET_ANSWER,
        },
        {
            "id": "hotpot-historical-qid-unique-82",
            "question": "How many moons orbit Mars?",
            "answer": SECRET_ANSWER,
        },
        {
            "id": "hotpot-dev-historical-family",
            "question": "Where was Grace Hopper born?",
            "answer": SECRET_ANSWER,
        },
        {
            "id": "hotpot-raw-train-qid-collision",
            "question": "Which architect designed the imaginary tower?",
            "answer": SECRET_ANSWER,
        },
        {
            "id": "hotpot-fresh-family-first",
            "question": "Which unique relation 1 connects item 1 to object 1?",
            "answer": SECRET_ANSWER,
        },
        {
            "id": "hotpot-fresh-family-second",
            "question": "Which unique relation 2 connects item 2 to object 2?",
            "answer": SECRET_ANSWER,
        },
    ]

    rows = {
        "hotpotqa": {"train": hotpot_train, "dev": hotpot_dev},
        "2wikimultihopqa": {
            "train": [
                {
                    "id": "twiki-train-unrelated",
                    "question": "Which ocean borders Testland?",
                    "answer": SECRET_ANSWER,
                }
            ],
            "dev": [
                {
                    "id": "twiki-fresh-candidate",
                    "question": "How tall is Mount Example?",
                    "answer": SECRET_ANSWER,
                }
            ],
        },
        "musique": {
            "train": [
                {
                    "id": "musique-train-unrelated",
                    "question": "Which river crosses Exampletown?",
                    "answer": SECRET_ANSWER,
                }
            ],
            # Same lexical family as a Hotpot historical row.  Dataset-scoped
            # family isolation must leave this MuSiQue identity eligible.
            "dev": [
                {
                    "id": "musique-cross-dataset-family",
                    "question": "Where was Grace Hopper born?",
                    "answer": SECRET_ANSWER,
                }
            ],
        },
    }
    for dataset, split_rows in rows.items():
        for split, values in split_rows.items():
            _write_jsonl(data_root / dataset / f"{split}.jsonl", values)

    historical_relative = "outputs/audits/unit_history/cohort.question_only.jsonl"
    historical_rows = [
        {
            "dataset": "hotpotqa",
            "qid": "hotpot-historical-qid-unique-82",
            "question": "Who wrote Hamlet?",
            "gold_answers": [SECRET_ANSWER],
        },
        {
            "dataset": "hotpotqa",
            "source_id": "hotpot-old-family-source",
            "question": "Where was Ada Lovelace born?",
            "gold_answers": [SECRET_ANSWER],
        },
    ]
    _write_jsonl(project_root / historical_relative, historical_rows)
    training_relative = "data/silver_data/unit_training/silver_train.jsonl"
    training_rows = [
        {
            "dataset": "hotpotqa",
            "qid": "hotpot-train-template",
            "question": "Who directed Film Alpha in 2007?",
            "answer": SECRET_ANSWER,
        }
    ]
    _write_jsonl(project_root / training_relative, training_rows)
    training_evidence_relative = "configs/training/unit_training.json"
    training_evidence = project_root / training_evidence_relative
    training_evidence.parent.mkdir(parents=True, exist_ok=True)
    training_evidence.write_text(
        json.dumps({"training": {"silver_path": training_relative}}) + "\n",
        encoding="utf-8",
    )
    training_specs = (
        capacity.TrainingInputSpec(
            training_relative,
            training_evidence_relative,
            capacity.COMPLETED_TRAINING,
        ),
    )
    return {
        "project_root": project_root,
        "data_root": data_root,
        "historical_paths": (historical_relative,),
        "source_rows": rows,
        "historical_rows": historical_rows,
        "training_relative": training_relative,
        "training_rows": training_rows,
        "training_specs": training_specs,
    }


def _run_fixture(capacity_sources, output_dir):
    return capacity.run_capacity_audit(
        project_root=capacity_sources["project_root"],
        data_root=capacity_sources["data_root"],
        output_dir=output_dir,
        experiment_id="SUBQUESTION-V8-CAPACITY-UNIT-TEST",
        historical_registry_paths=capacity_sources["historical_paths"],
        training_input_specs=capacity_sources["training_specs"],
    )


def test_scope_a_family_isolation_scope_b_relaxation_and_historical_exclusion(
    capacity_sources, tmp_path
):
    assert family_sha256("Who directed Film Alpha in 2007?") == family_sha256(
        "Who directed Film Beta in 1999?"
    )
    assert family_sha256(
        "Which unique relation 1 connects item 1 to object 1?"
    ) == family_sha256("Which unique relation 2 connects item 2 to object 2?")

    report = _run_fixture(capacity_sources, tmp_path / "audit")
    strict = report["scope_a_strict"]["by_dataset"]["hotpotqa"]
    relaxed = report["scope_b_relaxed_raw_train_family_isolation"]["by_dataset"][
        "hotpotqa"
    ]

    # Historical qid and historical family each exclude one row in both scopes.
    assert strict["historical_qid_hit_unique_qids"] == 1
    assert strict["historical_family_hit_unique_qids"] == 1
    assert strict["raw_train_qid_hit_unique_qids"] == 1
    assert strict["eligible_historical_registry_qid_overlap"] == 0
    assert strict["eligible_historical_registry_family_overlap"] == 0
    assert strict["eligible_raw_train_qid_overlap"] == 0
    assert relaxed["eligible_raw_train_qid_overlap"] == 0

    # Scope A additionally excludes the family represented in raw train.
    # The two remaining qids share one lexical family, so one-per-family
    # freezable capacity is one. Scope B admits the raw-train-family dev row.
    assert strict["eligible_unique_qids"] == 2
    assert strict["eligible_unique_dataset_scoped_families"] == 1
    assert strict["exact_freezable_capacity_one_per_dataset_scoped_family"] == 1
    assert relaxed["eligible_unique_qids"] == 3
    assert relaxed["eligible_unique_dataset_scoped_families"] == 2
    assert relaxed["exact_freezable_capacity_one_per_dataset_scoped_family"] == 2

    # A matching family in another dataset is not a cross-dataset exclusion.
    musique = report["scope_a_strict"]["by_dataset"]["musique"]
    assert musique["eligible_unique_qids"] == 1
    assert musique["eligible_unique_dataset_scoped_families"] == 1


def test_outputs_are_aggregate_only_and_disclose_source_access(capacity_sources, tmp_path):
    output_dir = tmp_path / "audit"
    report = _run_fixture(capacity_sources, output_dir)

    assert report["source_access_disclosure"] == {
        "source_files_opened": True,
        "source_may_contain_gold": True,
        "gold_fields_used": False,
        "gold_fields_emitted": False,
        "raw_source_fields_used": ["id", "qid", "question"],
        "historical_registry_fields_used": [
            "dataset",
            "id",
            "qid",
            "source_id",
            "question",
        ],
        "training_input_fields_used": [
            "dataset",
            "qid_or_declared_source_qid_alias",
            "question",
        ],
        "full_source_bytes_hashed_for_provenance_only": True,
        "source_file_hashes_used_for_capacity_or_selection": False,
        "stored_question_or_family_hashes_trusted": False,
    }
    assert report["source_files_opened"] is True
    assert report["source_may_contain_gold"] is True
    assert report["gold_fields_used"] is False
    assert report["gold_fields_emitted"] is False
    assert report["checks"]["individual_question_identity_rows_emitted"] is False
    assert report["checks"]["fresh_identity_rows_generated_or_frozen"] is False
    assert report["checks"]["fresh_answer_free_rows_generated_or_frozen"] is False
    evidence = report["existing_materialized_identity_source_evidence"]
    assert evidence["existing_answer_free_unused_pool"] == 0
    assert evidence["qualified_unique_dataset_scoped_qids"] == 2
    assert evidence["classified_consumed_or_protected_unique_dataset_scoped_qids"] == 2

    inventory = json.loads((output_dir / "inventory.json").read_text(encoding="utf-8"))
    assert len(inventory["historical_evaluation_protocol_registries"]) == 1
    assert len(inventory["local_training_inputs"]) == 1
    assert len(inventory["training_input_manifest_config_evidence"]) == 1
    assert all(
        set(item) == {"path", "sha256"}
        for values in inventory.values()
        for item in values
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(
        set(item) == {"path", "sha256"}
        for item in manifest["inputs"]["raw_source_inventory"]
    )
    assert all(
        set(item) == {"path", "sha256"} for item in manifest["implementation_inventory"]
    )

    serialized_outputs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    )
    assert SECRET_ANSWER not in serialized_outputs
    assert SECRET_SUPPORT not in serialized_outputs
    for split_rows in capacity_sources["source_rows"].values():
        for rows in split_rows.values():
            for row in rows:
                assert row["id"] not in serialized_outputs
                assert row["question"] not in serialized_outputs
                assert question_sha256(row["question"]) not in serialized_outputs
                assert family_sha256(row["question"]) not in serialized_outputs
    for row in capacity_sources["historical_rows"]:
        assert row["question"] not in serialized_outputs
        assert question_sha256(row["question"]) not in serialized_outputs
        assert family_sha256(row["question"]) not in serialized_outputs
    for row in capacity_sources["training_rows"]:
        assert row["qid"] not in serialized_outputs
        assert row["question"] not in serialized_outputs
        assert question_sha256(row["question"]) not in serialized_outputs
        assert family_sha256(row["question"]) not in serialized_outputs


def test_training_identity_outside_raw_train_blocks_capacity_decision(
    capacity_sources, tmp_path
):
    outside_question = "What wholly external training family has no raw source match?"
    outside_qid = "external-training-identity"
    _write_jsonl(
        capacity_sources["project_root"] / capacity_sources["training_relative"],
        [
            {
                "dataset": "hotpotqa",
                "qid": outside_qid,
                "question": outside_question,
                "answer": SECRET_ANSWER,
            }
        ],
    )
    output_dir = tmp_path / "blocked-audit"
    report = _run_fixture(capacity_sources, output_dir)
    gate = report["local_training_input_projection"]["raw_train_containment_gate"]
    assert gate["pass"] is False
    assert gate["outside_raw_train_unique_dataset_scoped_qids"] == 1
    assert gate["outside_raw_train_unique_dataset_scoped_families"] == 1
    assert report["status"] == capacity.TRAINING_LEDGER_BLOCKED_STATUS
    assert report["checks"]["training_inputs_within_corresponding_raw_train"] is False
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output_dir.iterdir()
    )
    assert outside_question not in serialized
    assert outside_qid not in serialized
    assert question_sha256(outside_question) not in serialized
    assert family_sha256(outside_question) not in serialized


def test_existing_output_directory_is_rejected_before_any_source_read(
    capacity_sources, tmp_path
):
    output_dir = tmp_path / "already-exists"
    output_dir.mkdir()
    marker = output_dir / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite append-only"):
        capacity.run_capacity_audit(
            project_root=capacity_sources["project_root"],
            data_root=capacity_sources["data_root"] / "missing-on-purpose",
            output_dir=output_dir,
            historical_registry_paths=("also-missing.jsonl",),
        )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in output_dir.iterdir()) == ["preserve.txt"]


def test_formal_inventories_are_static_and_historical_registry_has_58_paths():
    paths = capacity.HISTORICAL_EVALUATION_PROTOCOL_REGISTRY_PATHS
    assert len(paths) == 58
    assert len(set(paths)) == 58
    assert tuple(sorted(paths)) == (
        paths
    )
    assert len(capacity.LOCAL_TRAINING_INPUT_SPECS) >= 13
    assert len({spec.path for spec in capacity.LOCAL_TRAINING_INPUT_SPECS}) == len(
        capacity.LOCAL_TRAINING_INPUT_SPECS
    )

    tree = ast.parse(Path(capacity.__file__).read_text(encoding="utf-8"))
    runtime_discovery_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"glob", "rglob"}
    ]
    assert runtime_discovery_calls == []
