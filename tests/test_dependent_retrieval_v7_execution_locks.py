import json
import hashlib
from pathlib import Path

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare import freeze_dependent_retrieval_v7_implementation as impl
from scripts.prepare import freeze_dependent_retrieval_v7_plans as plans


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _synthetic_runtime(tmp_path: Path, *, add_extra_task_key: bool = False):
    task_keys = set(impl.EXPECTED_TASK_KEYS)
    if add_extra_task_key:
        task_keys.add("answer")
    answer_keys = set(impl.EXPECTED_ANSWER_KEYS)
    identity = (
        "task_id",
        "question_key",
        "dataset",
        "qid",
        "question_sha256",
        "target_type",
        "producer_slot",
        "step_sha256",
        "producer_passages_sha256",
    )
    runner = _write(
        tmp_path / "runner.py",
        "\n".join(
            [
                f"C_TASK_KEYS = frozenset({task_keys!r})",
                f"C_ANSWER_KEYS = frozenset({answer_keys!r})",
                "BRIDGE_MAX_DOCS = 10",
                "BRIDGE_MAX_BODY_CHARS = 1200",
                f"TRAJECTORY_SEMANTICS_ADDENDUM_SCHEMA = {impl.TRAJECTORY_ADDENDUM_SCHEMA!r}",
                f"TRAJECTORY_SEMANTICS_ADDENDUM_STATUS = {impl.TRAJECTORY_ADDENDUM_STATUS!r}",
                "RUNNER_VERSION = 'paired-dependent-retrieval-v7-staged-gold-free-1'",
                "def _producer_passage_projection(passages):",
                "    return list(passages)[:BRIDGE_MAX_DOCS], {}",
            ]
        )
        + "\n",
    )
    generator = _write(
        tmp_path / "generator.py",
        "\n".join(
            [
                f"TASK_KEYS = frozenset({set(impl.EXPECTED_TASK_KEYS)!r})",
                f"OUTPUT_IDENTITY_KEYS = {identity!r}",
                "MAX_NEW_TOKENS = 96",
                "MAX_PRODUCER_PASSAGES = 10",
                "MAX_PASSAGE_TEXT_CHARS = 1200",
                "def build_subanswer_reader_messages(*args, **kwargs): pass",
                "def parse_and_verify_subanswer(*args, **kwargs): pass",
                "def generate_subanswer_rows(tasks):",
                "    for task in tasks:",
                "        passages = task['producer_passages']",
                "        build_subanswer_reader_messages(task['question'], task['step'], passages)",
                "        parse_and_verify_subanswer('x', task['question'], task['step'], passages)",
            ]
        )
        + "\n",
    )
    finalizer = _write(tmp_path / "finalizer.py", "VALUE = 1\n")
    evaluator = _write(tmp_path / "evaluator.py", "VALUE = 1\n")
    return {
        "retrieval_runner": runner,
        "subanswer_generator": generator,
        "gold_finalizer": finalizer,
        "evaluator": evaluator,
    }


def _cohort_rows():
    rows = []
    for dataset, target_type in (
        ("hotpotqa", "relation_graph"),
        ("musique", "subquery_graph"),
    ):
        for index in range(20):
            question = f"{dataset} question {index}?"
            rows.append(
                {
                    "row_id": f"v7::{dataset}::{index}",
                    "question_key": f"{dataset}::{index}",
                    "dataset": dataset,
                    "qid": str(index),
                    "question": question,
                    "question_sha256": question_sha256(question),
                    "target_type": target_type,
                    "gold_access": False,
                }
            )
    return rows


def _target(target_type: str):
    if target_type == "relation_graph":
        return {
            "anchors": ["Root"],
            "steps": [
                {
                    "step": 1,
                    "subject": "Root",
                    "relation_label": "creator",
                    "pid": "P170",
                    "output_slot": "hop_1",
                    "dependencies": [],
                }
            ],
        }
    return {
        "steps": [
            {
                "step": 1,
                "subquery_template": "Root >> creator",
                "dependencies": [],
                "output_slot": "step_1",
            }
        ]
    }


def _predictions(cohort):
    result = []
    for row in cohort:
        target = _target(row["target_type"])
        result.append(
            {
                "row_id": row["row_id"],
                "question_key": row["question_key"],
                "dataset": row["dataset"],
                "qid": row["qid"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "generated_text": json.dumps(target),
                "predicted_target": target,
                "schema_valid": True,
                "validation_errors": [],
                "gold_access": False,
            }
        )
    return result


def test_runtime_contract_proves_single_projection_and_same_object(tmp_path):
    result = impl.validate_runtime_contract(_synthetic_runtime(tmp_path))
    assert result["task_schema_exact_and_equal"] is True
    assert result["generator_prompt_and_verifier_receive_same_passages_binding"] is True
    assert result["sharing_mode"] == (
        "shared_by_immutable_task_byte_commitment_and_same_object_binding"
    )


def test_runtime_contract_rejects_task_schema_drift(tmp_path):
    with pytest.raises(ValueError, match="C-task exact schema mismatch"):
        impl.validate_runtime_contract(
            _synthetic_runtime(tmp_path, add_extra_task_key=True)
        )


def test_local_import_closure_hashes_project_modules(tmp_path):
    root = tmp_path
    imported = _write(root / "kgproweight" / "demo.py", "VALUE = 1\n")
    caller = _write(root / "scripts" / "caller.py", "from kgproweight.demo import VALUE\n")
    assert impl.local_import_closure([caller], root) == sorted([caller, imported])


def test_plan_predictions_exact_join_and_reparse_pass():
    cohort = _cohort_rows()
    result = plans.validate_predictions(_predictions(cohort), cohort)
    assert result["n"] == 40
    assert result["schema_valid_rate"] == {"hotpotqa": 1.0, "musique": 1.0}
    assert result["plan_executable_gate_pass"] is True


def test_plan_predictions_reject_hidden_field_and_hash_drift():
    cohort = _cohort_rows()
    predictions = _predictions(cohort)
    predictions[0]["gold_answer"] = "leak"
    with pytest.raises(ValueError, match="exact fields drift"):
        plans.validate_predictions(predictions, cohort)

    predictions = _predictions(cohort)
    predictions[0]["question_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="cohort join drift"):
        plans.validate_predictions(predictions, cohort)


def test_plan_predictions_reject_mutated_parsed_target_and_under_gate():
    cohort = _cohort_rows()
    predictions = _predictions(cohort)
    predictions[0]["predicted_target"]["steps"][0]["pid"] = "P999"
    with pytest.raises(ValueError, match="raw-text parsed plan mismatch"):
        plans.validate_predictions(predictions, cohort)

    predictions = _predictions(cohort)
    # Five invalid Hotpot rows make the rate 15/20=.75, below the frozen .80.
    for row in predictions[:5]:
        row["generated_text"] = "not-json"
        row["predicted_target"] = None
        row["schema_valid"] = False
        row["validation_errors"] = ["json_decode:Expecting value"]
    with pytest.raises(ValueError, match="plan executable preregistered gate failed"):
        plans.validate_predictions(predictions, cohort)


def test_schema_valid_plans_can_still_fail_actual_dependent_execution_gate():
    cohort = _cohort_rows()
    predictions = _predictions(cohort)
    # The supervision schema accepts a prior dependency declaration without
    # requiring the subject to contain its slot.  The materialisation runner
    # correctly rejects that declaration/template mismatch.  Five such rows
    # put Hotpot execution at 15/20 even though schema validity remains 20/20.
    for row in predictions[:5]:
        target = {
            "anchors": ["Root"],
            "steps": [
                {
                    "step": 1,
                    "subject": "Root",
                    "relation_label": "creator",
                    "pid": "P170",
                    "output_slot": "hop_1",
                    "dependencies": [],
                },
                {
                    "step": 2,
                    "subject": "Another Root",
                    "relation_label": "birthplace",
                    "pid": "P19",
                    "output_slot": "hop_2",
                    "dependencies": ["hop_1"],
                },
            ],
        }
        row["generated_text"] = json.dumps(target)
        row["predicted_target"] = target
    assert all(row["schema_valid"] for row in predictions)
    with pytest.raises(ValueError, match="plan executable preregistered gate failed"):
        plans.validate_predictions(predictions, cohort)


def test_lock_writers_are_append_only(tmp_path):
    protocol = {
        "experiment_id": "TEST",
        "authorization": {
            "planner": True,
            "gold_free_materialization": False,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
    }
    out = tmp_path / "implementation"
    written = impl.write_protocol(protocol, out)
    assert written["protocol"]["sha256"]
    with pytest.raises(FileExistsError):
        impl.write_protocol(protocol, out)


def _jsonl(path: Path, rows):
    return _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def _artifact_file(path: Path):
    return {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "file",
        "size_bytes": path.stat().st_size,
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }


def _tiny_tree(tmp_path: Path, name: str, files: dict[str, str]) -> tuple[Path, dict]:
    root = tmp_path / name
    for filename, contents in files.items():
        _write(root / filename, contents)
    return root, impl.tree_lock(root)


def _post_plan_fixture(tmp_path: Path):
    runtime = _synthetic_runtime(tmp_path / "runtime")
    # The plan freezer is required to call the validator from the exact runner
    # committed by the implementation lock.  Other runtime roles can remain
    # tiny test doubles because this test never executes them.
    runtime["retrieval_runner"] = Path(plans.retrieval_runner.__file__).resolve()
    cohort = _cohort_rows()
    cohort_path = _jsonl(tmp_path / "cohort.jsonl", cohort)
    development_path = _jsonl(tmp_path / "development.jsonl", cohort)
    contexts_path = _jsonl(tmp_path / "contexts.jsonl", [{"x": 1}])
    predictions_path = _jsonl(tmp_path / "predictions.jsonl", _predictions(cohort))
    config_path = _write(tmp_path / "planner.yaml", "model: test\n")
    adapter_path, adapter_lock = _tiny_tree(
        tmp_path,
        "adapter",
        {
            "adapter_config.json": "{}\n",
            "adapter_model.safetensors": "synthetic adapter weights\n",
        },
    )
    base_path, base_lock = _tiny_tree(
        tmp_path, "base", {"config.json": "{}\n", "weights.bin": "base\n"}
    )
    retrieval_path, retrieval_lock = _tiny_tree(
        tmp_path, "retrieval", {"config.json": "{}\n", "weights.bin": "retrieval\n"}
    )
    cross_path, cross_lock = _tiny_tree(
        tmp_path, "cross", {"config.json": "{}\n", "weights.bin": "cross\n"}
    )
    strong_sft_path, strong_sft_lock = _tiny_tree(
        tmp_path, "strong_sft", {"config.json": "{}\n", "weights.bin": "sft\n"}
    )
    corpus_path = _write(tmp_path / "wiki18" / "corpus.jsonl", '{"id": 1}\n')
    dense_path = _write(tmp_path / "wiki18" / "dense.index", "dense\n")
    bm25_path, bm25_lock = _tiny_tree(
        tmp_path, "wiki18/bm25", {"index.bin": "bm25\n"}
    )
    model_locks = {
        "query_planner": adapter_lock,
        "retrieval_encoder": retrieval_lock,
        "cross_encoder": cross_lock,
        "strong_sft": strong_sft_lock,
        "base_model": base_lock,
    }
    wiki18_locks = {
        "corpus": impl.file_lock(corpus_path),
        "dense_index": impl.file_lock(dense_path),
        "bm25_index": bm25_lock,
    }
    prereg_path = _write(
        tmp_path / "prereg.json",
        json.dumps(
            {
                "future_experiment_ids": {"materialization": "V7-MATERIALIZE-TEST"},
                "models": {
                    "query_planner": {"content_lock": adapter_lock},
                    "inherited_content_locks": {
                        name: model_locks[name]
                        for name in (
                            "retrieval_encoder",
                            "cross_encoder",
                            "strong_sft",
                            "base_model",
                        )
                    },
                },
                "retrieval_asset_content_locks": wiki18_locks,
            }
        ),
    )
    prereg_manifest_path = _write(tmp_path / "prereg_manifest.json", "{}\n")
    addendum_path = _write(
        tmp_path / "addendum.json",
        json.dumps(
            {
                "schema_version": "subquestion-dependent-retrieval-v7-effective-addendum-1",
                "status": "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL",
                "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
                "parents": {
                    "parent_preregistration": impl.file_lock(prereg_path),
                },
                "effective_invariants": {
                    "producer_passages_max": 10,
                    "producer_text_unicode_chars_max_each": 1200,
                    "python_slice": "text[:1200]",
                    "projection_fields": ["doc_id", "title", "text"],
                    "reader_and_verifier_projection_hash_equal": True,
                    "answer_in_unseen_suffix_never_verified": True,
                },
                "gold_access": False,
            }
        ),
    )
    addendum_manifest_path = _write(tmp_path / "addendum_manifest.json", "{}\n")
    design_protocol_path = _write(tmp_path / "design.json", "{}\n")
    design_manifest_path = _write(tmp_path / "design_manifest.json", "{}\n")
    design_trajectory_path = _write(tmp_path / "design_trajectory.json", "{}\n")
    design_trajectory_manifest_path = _write(
        tmp_path / "design_trajectory_manifest.json", "{}\n"
    )
    trajectory_path = _write(
        tmp_path / "trajectory.json",
        json.dumps(
            {
                "schema_version": impl.TRAJECTORY_ADDENDUM_SCHEMA,
                "status": impl.TRAJECTORY_ADDENDUM_STATUS,
                "scope": impl.TRAJECTORY_ADDENDUM_SCOPE,
                "parents": {
                    "design_protocol": impl.file_lock(design_protocol_path),
                    "design_manifest": impl.file_lock(design_manifest_path),
                    "design_trajectory_addendum": impl.file_lock(
                        design_trajectory_path
                    ),
                    "design_trajectory_addendum_manifest": impl.file_lock(
                        design_trajectory_manifest_path
                    ),
                    "parent_preregistration": impl.file_lock(prereg_path),
                    "parent_preregistration_manifest": impl.file_lock(
                        prereg_manifest_path
                    ),
                    "producer_truncation_addendum": impl.file_lock(addendum_path),
                    "producer_truncation_addendum_manifest": impl.file_lock(
                        addendum_manifest_path
                    ),
                },
                "effective_invariants": impl.TRAJECTORY_INVARIANTS,
                "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
                "gold_access": False,
            }
        ),
    )
    trajectory_manifest_path = _write(
        tmp_path / "trajectory_manifest.json",
        json.dumps(
            {
                "status": impl.TRAJECTORY_ADDENDUM_STATUS,
                "gold_access": False,
                "protocol": impl.file_lock(trajectory_path),
            }
        ),
    )

    impl_protocol = {
        "schema_version": impl.SCHEMA_VERSION,
        "experiment_id": impl.EXPERIMENT_ID,
        "status": impl.STATUS,
        "gold_access": False,
        "authorization": {
            "planner": True,
            "gold_free_materialization": False,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "content_reverification": {
            "full_hash_verification_performed": True,
            "verified": {"models": model_locks, "wiki18": wiki18_locks},
        },
        "parents": {
            "preregistration": impl.file_lock(prereg_path),
            "preregistration_manifest": impl.file_lock(prereg_manifest_path),
            "truncation_addendum": impl.file_lock(addendum_path),
            "truncation_addendum_manifest": impl.file_lock(addendum_manifest_path),
            "trajectory_semantics_addendum": impl.file_lock(trajectory_path),
            "trajectory_semantics_addendum_manifest": impl.file_lock(
                trajectory_manifest_path
            ),
        },
        "inputs": {
            "development": impl.file_lock(development_path),
            "planner_cohort": impl.file_lock(cohort_path),
            "canonical_A_contexts": impl.file_lock(contexts_path),
        },
        "planner_contract": {
            "experiment_id": "V7-PLANNER-TEST",
            "config": impl.file_lock(config_path),
            "adapter": adapter_lock,
            "base_model": base_lock,
        },
        "runtime_code": {name: impl.file_lock(path) for name, path in runtime.items()},
        "actual_local_import_closure": {},
        "lock_issuer": impl.file_lock(Path(impl.__file__)),
    }
    impl_dir = tmp_path / "implementation"
    impl_dir.mkdir()
    impl_path = impl_dir / "protocol.json"
    impl_path.write_bytes(impl.canonical_json_bytes(impl_protocol))
    impl_manifest_path = impl_dir / "manifest.json"
    impl_manifest_path.write_bytes(
        impl.canonical_json_bytes(
            {
                "status": impl.STATUS,
                "gold_access": False,
                "protocol": impl.file_lock(impl_path),
            }
        )
    )

    prediction_artifact = _artifact_file(predictions_path)
    report = {
        "status": "RUNTIME_PLANS_FROZEN_NO_GOLD_AUDIT",
        "experiment_id": "V7-PLANNER-TEST",
        "scope": "historical non-authoritative scope",
        "generation": {"greedy": True, "max_new_tokens": 512, "batch_size": 4},
        "counts": {
            "n": 40,
            "by_dataset": {"hotpotqa": 20, "musique": 20},
            "schema_valid": 40,
        },
        "rates": {"schema_valid": 1.0},
        "inputs": {
            "cohort": _artifact_file(cohort_path),
            "adapter": {
                **plans.artifact_identity(adapter_path),
            },
            "config": _artifact_file(config_path),
            "protocol": _artifact_file(impl_path),
        },
        "outputs": {"predictions": prediction_artifact},
        "gold_access": False,
    }
    report_path = _write(tmp_path / "report.json", json.dumps(report))
    planner_manifest_path = _write(
        tmp_path / "planner_manifest.json",
        json.dumps(
            {
                "status": report["status"],
                "experiment_id": report["experiment_id"],
                "gold_access": False,
            }
        ),
    )
    return {
        "implementation_path": impl_path,
        "implementation_manifest_path": impl_manifest_path,
        "predictions_path": predictions_path,
        "planner_report_path": report_path,
        "planner_manifest_path": planner_manifest_path,
        "report": report,
        "trajectory_path": trajectory_path,
        "asset_paths": {
            "adapter": adapter_path / "adapter_model.safetensors",
            "adapter_config": adapter_path / "adapter_config.json",
            "base_model": base_path / "weights.bin",
            "retrieval_encoder": retrieval_path / "weights.bin",
            "cross_encoder": cross_path / "weights.bin",
            "strong_sft": strong_sft_path / "weights.bin",
            "wiki18_corpus": corpus_path,
            "wiki18_dense": dense_path,
            "wiki18_bm25": bm25_path / "index.bin",
        },
    }


def _build_plan_lock(fixture):
    return plans.build_plan_lock_protocol(
        implementation_path=fixture["implementation_path"],
        implementation_manifest_path=fixture["implementation_manifest_path"],
        predictions_path=fixture["predictions_path"],
        planner_report_path=fixture["planner_report_path"],
        planner_manifest_path=fixture["planner_manifest_path"],
    )


def test_post_plan_lock_end_to_end_with_synthetic_artifacts(tmp_path):
    protocol = _build_plan_lock(_post_plan_fixture(tmp_path))
    assert protocol["status"] == plans.STATUS
    assert protocol["authorization"]["gold_free_materialization"] is True
    assert protocol["population"]["plan_executable_gate_pass"] is True
    assert protocol["population"]["plan_executable"] == {
        "hotpotqa": 20,
        "musique": 20,
    }
    assert protocol["content_reverification"]["performed_after_planner_generation"] is True
    assert "trajectory_semantics_addendum" in protocol["parents"]


@pytest.mark.parametrize(
    "asset_name",
    [
        "adapter",
        "adapter_config",
        "base_model",
        "retrieval_encoder",
        "cross_encoder",
        "strong_sft",
        "wiki18_corpus",
        "wiki18_dense",
        "wiki18_bm25",
    ],
)
def test_post_plan_lock_rejects_content_toctou_drift(tmp_path, asset_name):
    fixture = _post_plan_fixture(tmp_path)
    path = fixture["asset_paths"][asset_name]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="drift"):
        _build_plan_lock(fixture)


@pytest.mark.parametrize(
    "filename", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_post_plan_lock_rejects_planner_report_adapter_hash_tampering(
    tmp_path, filename
):
    fixture = _post_plan_fixture(tmp_path)
    report = fixture["report"]
    adapter_files = report["inputs"]["adapter"]["files"]
    next(row for row in adapter_files if row["name"] == filename)["md5"] = "0" * 32
    fixture["planner_report_path"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter content identity drift"):
        _build_plan_lock(fixture)


def test_post_plan_lock_rejects_path_only_planner_adapter_record(tmp_path):
    fixture = _post_plan_fixture(tmp_path)
    report = fixture["report"]
    report["inputs"]["adapter"] = {
        "path": report["inputs"]["adapter"]["path"],
        "exists": True,
        "kind": "directory",
    }
    fixture["planner_report_path"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter content identity drift"):
        _build_plan_lock(fixture)


def test_post_plan_lock_rejects_trajectory_addendum_toctou_drift(tmp_path):
    fixture = _post_plan_fixture(tmp_path)
    fixture["trajectory_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory_semantics_addendum.*drift"):
        _build_plan_lock(fixture)


def test_post_plan_lock_accepts_dump_manifest_nested_run_layout(tmp_path):
    fixture = _post_plan_fixture(tmp_path)
    manifest = json.loads(fixture["planner_manifest_path"].read_text(encoding="utf-8"))
    manifest.pop("experiment_id")
    manifest.pop("gold_access")
    manifest["run"] = fixture["report"]
    fixture["planner_manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    protocol = _build_plan_lock(fixture)
    assert protocol["authorization"]["gold_free_materialization"] is True


def test_post_plan_lock_rejects_nested_run_payload_drift(tmp_path):
    fixture = _post_plan_fixture(tmp_path)
    manifest = json.loads(fixture["planner_manifest_path"].read_text(encoding="utf-8"))
    manifest.pop("experiment_id")
    manifest["run"] = dict(fixture["report"])
    manifest["run"]["experiment_id"] = "WRONG-PLANNER-RUN"
    fixture["planner_manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run/report payload drift"):
        _build_plan_lock(fixture)
