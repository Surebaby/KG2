"""CPU-only tests for the isolated v7 grounded-subanswer generator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.pilot import generate_grounded_subanswers_v7 as generator_v7


QUESTION = "What league does the Champions League winner play in?"
STEP = {
    "step": 1,
    "subject": "Champions League winner",
    "relation_label": "league",
    "pid": "P118",
    "output_slot": "hop_1",
    "dependencies": [],
}
PASSAGES = [
    {
        "doc_id": "doc-1",
        "title": "UEFA Champions League",
        "text": (
            "UEFA Champions League\n"
            "The winning club competed in La Liga during the relevant season."
        ),
    },
    {
        "doc_id": "doc-2",
        "title": "Premier League",
        "text": "Another finalist played in the Premier League.",
    },
]
MODEL_ARTIFACT = {
    "base_model": {"inventory_sha256": "a" * 64},
    "strong_sft_adapter": {"inventory_sha256": "b" * 64},
    "load_contract": {"torch_dtype": "bfloat16", "device": "cuda:0"},
}


def make_task(
    *,
    task_id: str = "hotpotqa::q1::hop_1",
    producer_slot: str = "hop_1",
    question: str = QUESTION,
    step: Mapping[str, Any] | None = None,
    passages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    step_value = deepcopy(dict(step or STEP))
    passage_value = deepcopy(list(passages or PASSAGES))
    return {
        "task_id": task_id,
        "question_key": "hotpotqa::q1",
        "dataset": "hotpotqa",
        "qid": "q1",
        "question": question,
        "question_sha256": question_sha256(question),
        "target_type": "relation_graph",
        "producer_slot": producer_slot,
        "step": step_value,
        "step_sha256": generator_v7.canonical_json_sha256(step_value),
        "producer_passages": passage_value,
        "producer_passages_sha256": generator_v7.canonical_json_sha256(
            passage_value
        ),
        "gold_access": False,
    }


class FakeGenerator:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> generator_v7.GenerationOutput:
        self.calls.append(
            {
                "messages": deepcopy(list(messages)),
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        response = self.responses[len(self.calls) - 1]
        prompt = json.dumps(
            list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return generator_v7.GenerationOutput(
            prompt=prompt,
            response_text=response,
            prompt_tokens=37,
            generation_tokens=12 if response else 1,
            runtime_telemetry={"backend": "fake", "peak_bytes": 123},
        )


def valid_response(answer: str = "La Liga", doc_id: str = "doc-1") -> str:
    return json.dumps(
        {
            "answer": answer,
            "cited_doc_ids": [doc_id],
            "answer_type": "entity",
            "abstain": False,
        },
        ensure_ascii=False,
    )


def write_task_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return generator_v7.sha256_file(path)


def install_lightweight_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    def prepare(output, *, experiment_id=None, extra=None):
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        (path / "manifest.json").write_text(
            json.dumps({"status": "RUNNING", "run": dict(extra or {})}),
            encoding="utf-8",
        )
        return path, str(experiment_id)

    def dump(path, extra=None, *, status="COMPLETE"):
        target = Path(path) / "manifest.json"
        target.write_text(
            json.dumps({"status": status, "run": dict(extra or {})}),
            encoding="utf-8",
        )
        return target

    monkeypatch.setattr(generator_v7, "prepare_new_run_dir", prepare)
    monkeypatch.setattr(generator_v7, "dump_manifest", dump)


def test_valid_generation_preserves_identity_and_complete_telemetry() -> None:
    task = make_task()
    fake = FakeGenerator([valid_response()])
    rows = generator_v7.generate_subanswer_rows(
        [task],
        generator=fake,
        input_file_sha256="c" * 64,
        model_artifact=MODEL_ARTIFACT,
    )

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(generator_v7.OUTPUT_IDENTITY_KEYS) | {
        "verified",
        "verified_answer",
        "telemetry",
        "gold_access",
    }
    for key in generator_v7.OUTPUT_IDENTITY_KEYS:
        assert row[key] == task[key]
    assert row["verified"] is True
    assert row["verified_answer"] == "La Liga"
    assert row["gold_access"] is False

    telemetry = row["telemetry"]
    assert telemetry["strict_parse"] == {"valid": True, "error_code": None}
    assert telemetry["verification"]["verified"] is True
    assert telemetry["verification"]["supporting_doc_id"] == "doc-1"
    assert telemetry["verification"]["verification_scope"] == (
        "surface_locality_not_semantic_entailment"
    )
    assert telemetry["raw_response_sha256"] == hashlib.sha256(
        telemetry["raw_response"].encode("utf-8")
    ).hexdigest()
    assert telemetry["prompt_sha256"] == hashlib.sha256(
        json.dumps(
            fake.calls[0]["messages"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert telemetry["prompt_tokens"] == 37
    assert telemetry["generation_tokens"] == 12
    assert telemetry["model_artifact"] == MODEL_ARTIFACT
    assert telemetry["prompt_passages_sha256"] == task["producer_passages_sha256"]
    assert telemetry["verifier_passages_sha256"] == task["producer_passages_sha256"]
    assert telemetry["same_passage_bytes_for_prompt_and_verifier"] is True
    assert telemetry["gold_access"] is False
    assert fake.calls[0]["max_new_tokens"] == 96
    assert fake.calls[0]["do_sample"] is False
    prompt_payload = json.loads(fake.calls[0]["messages"][1]["content"])
    assert prompt_payload["retrieved_documents"] == PASSAGES


def test_malformed_response_is_a_recorded_fail_closed_parse_rejection() -> None:
    fake = FakeGenerator(["```json\n{}\n```"])
    row = generator_v7.generate_subanswer_rows(
        [make_task()],
        generator=fake,
        input_file_sha256="d" * 64,
        model_artifact=MODEL_ARTIFACT,
    )[0]

    assert row["verified"] is False
    assert row["verified_answer"] is None
    assert row["telemetry"]["strict_parse"] == {
        "valid": False,
        "error_code": "invalid_json",
    }
    assert row["telemetry"]["verification"]["reason"] == (
        "parse_error:invalid_json"
    )
    assert row["telemetry"]["generation"]["retry_count"] == 0


def test_all_tasks_are_preflighted_before_fake_generator_is_called() -> None:
    first = make_task()
    second = make_task(task_id="hotpotqa::q1::hop_2", producer_slot="hop_2")
    second["producer_passages"][0]["Gold_Answer"] = "hidden"
    second["producer_passages_sha256"] = generator_v7.canonical_json_sha256(
        second["producer_passages"]
    )
    fake = FakeGenerator([valid_response(), valid_response()])

    with pytest.raises(generator_v7.TaskValidationError, match="forbidden"):
        generator_v7.generate_subanswer_rows(
            [first, second],
            generator=fake,
            input_file_sha256="e" * 64,
            model_artifact=MODEL_ARTIFACT,
        )
    assert fake.calls == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(question_key="hotpotqa::wrong"), "question_key"),
        (lambda row: row.update(question_sha256="0" * 64), "question_sha256"),
        (lambda row: row.update(step_sha256="1" * 64), "step_sha256"),
        (
            lambda row: row.update(producer_passages_sha256="2" * 64),
            "producer_passages_sha256",
        ),
        (lambda row: row.update(target_type="subquery_graph"), "target_type"),
        (lambda row: row.update(gold_access=0), "gold_access"),
        (lambda row: row.update(unexpected="side-channel"), "exact C-task schema"),
    ],
)
def test_identity_hash_and_exact_schema_drift_are_rejected(mutate, message) -> None:
    row = make_task()
    mutate(row)
    with pytest.raises(generator_v7.TaskValidationError, match=message):
        generator_v7.validate_task_rows([row])


def test_duplicate_task_and_question_slot_identities_are_rejected() -> None:
    first = make_task()
    duplicate_task = make_task()
    with pytest.raises(generator_v7.TaskValidationError, match="duplicate task_id"):
        generator_v7.validate_task_rows([first, duplicate_task])

    duplicate_slot = make_task(task_id="different-task")
    with pytest.raises(
        generator_v7.TaskValidationError, match="duplicate question_key/producer_slot"
    ):
        generator_v7.validate_task_rows([first, duplicate_slot])


def test_passage_limit_is_rejected_not_silently_truncated() -> None:
    too_long = deepcopy(PASSAGES)
    too_long[0]["text"] = "界" * 1201
    with pytest.raises(generator_v7.TaskValidationError, match="1200"):
        generator_v7.validate_task_rows([make_task(passages=too_long)])

    too_many = [
        {
            "doc_id": f"doc-{index}",
            "title": f"Document {index}",
            "text": f"visible passage {index}",
        }
        for index in range(11)
    ]
    with pytest.raises(generator_v7.TaskValidationError, match="maximum is 10"):
        generator_v7.validate_task_rows([make_task(passages=too_many)])


def test_exact_1200_unicode_chars_are_used_for_both_prompt_and_verifier() -> None:
    text = "界" * 1193 + " La Liga"
    assert len(text) == 1201  # leading space and eight answer characters
    # Make the exact frozen boundary while retaining the answer at the end.
    text = "界" * (1200 - len(" La Liga")) + " La Liga"
    assert len(text) == 1200
    passages = [{"doc_id": "edge-doc", "title": "Edge", "text": text}]
    task = make_task(passages=passages)
    fake = FakeGenerator([valid_response(doc_id="edge-doc")])

    row = generator_v7.generate_subanswer_rows(
        [task],
        generator=fake,
        input_file_sha256="f" * 64,
        model_artifact=MODEL_ARTIFACT,
    )[0]
    assert row["verified"] is True
    assert text in fake.calls[0]["messages"][1]["content"]
    assert row["telemetry"]["producer_passage_character_counts"] == [
        {"rank": 1, "text_character_counts": {"text": 1200}}
    ]


def test_noncanonical_task_projection_is_rejected_before_generation() -> None:
    """The runner must supply the exact addendum projection, not source objects."""

    passages = [
        {
            "doc_id": "doc-quoted",
            "title": '  "Quoted   Title"  ',
            "text": "The answer is La Liga.",
        }
    ]
    fake = FakeGenerator([valid_response(doc_id="doc-quoted")])
    with pytest.raises(generator_v7.TaskValidationError, match="projection differs"):
        generator_v7.generate_subanswer_rows(
            [make_task(passages=passages)],
            generator=fake,
            input_file_sha256="9" * 64,
            model_artifact=MODEL_ARTIFACT,
        )
    assert fake.calls == []


def test_generator_token_contract_is_fail_closed() -> None:
    class OverBudgetGenerator(FakeGenerator):
        def __call__(self, messages, *, max_new_tokens, do_sample):
            normal = super().__call__(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            return generator_v7.GenerationOutput(
                prompt=normal.prompt,
                response_text=normal.response_text,
                prompt_tokens=normal.prompt_tokens,
                generation_tokens=97,
            )

    with pytest.raises(ValueError, match="exceeded"):
        generator_v7.generate_subanswer_rows(
            [make_task()],
            generator=OverBudgetGenerator([valid_response()]),
            input_file_sha256="a" * 64,
            model_artifact=MODEL_ARTIFACT,
        )


def test_strict_jsonl_reader_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"task_id":"one","task_id":"two"}\n', encoding="utf-8")
    with pytest.raises(generator_v7.TaskValidationError, match="duplicate"):
        generator_v7.read_tasks_jsonl(duplicate)

    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(generator_v7.TaskValidationError, match="non-finite"):
        generator_v7.read_tasks_jsonl(nonfinite)


def test_run_generation_writes_append_only_gold_free_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_lightweight_manifest(monkeypatch)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_hash = write_task_file(tasks_path, [make_task()])
    output_dir = tmp_path / "run"
    fake = FakeGenerator([valid_response()])

    report = generator_v7.run_generation(
        tasks_path=tasks_path,
        expected_tasks_sha256=tasks_hash,
        output_dir=output_dir,
        experiment_id="V7-SUBANSWER-TEST",
        generator=fake,
        model_artifact=MODEL_ARTIFACT,
    )
    assert report["status"] == "COMPLETE_GOLD_FREE_SUBANSWERS"
    assert report["gold_access"] is False
    assert report["generation"] == {
        "decode": "greedy",
        "do_sample": False,
        "max_new_tokens": 96,
        "seed": 42,
        "retry_count": 0,
        "torch_dtype": "bfloat16",
    }
    rows = generator_v7.read_tasks_jsonl(output_dir / "subanswers.jsonl")
    assert rows[0]["verified"] is True
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE_GOLD_FREE_SUBANSWERS"
    assert manifest["run"]["gold_access"] is False

    with pytest.raises(FileExistsError, match="run"):
        generator_v7.run_generation(
            tasks_path=tasks_path,
            expected_tasks_sha256=tasks_hash,
            output_dir=output_dir,
            experiment_id="V7-SUBANSWER-TEST",
            generator=FakeGenerator([valid_response()]),
            model_artifact=MODEL_ARTIFACT,
        )


def test_input_file_hash_drift_stops_before_reserving_or_generating(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_task_file(tasks_path, [make_task()])
    output_dir = tmp_path / "must-not-exist"
    fake = FakeGenerator([valid_response()])
    with pytest.raises(generator_v7.TaskValidationError, match="caller lock"):
        generator_v7.run_generation(
            tasks_path=tasks_path,
            expected_tasks_sha256="0" * 64,
            output_dir=output_dir,
            experiment_id="V7-HASH-DRIFT",
            generator=fake,
            model_artifact=MODEL_ARTIFACT,
        )
    assert not output_dir.exists()
    assert fake.calls == []


def test_runtime_exception_preserves_failed_run_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_lightweight_manifest(monkeypatch)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_hash = write_task_file(tasks_path, [make_task()])
    output_dir = tmp_path / "failed-run"

    def exploding_generator(messages, *, max_new_tokens, do_sample):
        raise RuntimeError("synthetic generation failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        generator_v7.run_generation(
            tasks_path=tasks_path,
            expected_tasks_sha256=tasks_hash,
            output_dir=output_dir,
            experiment_id="V7-FAILED",
            generator=exploding_generator,
            model_artifact=MODEL_ARTIFACT,
        )
    assert output_dir.is_dir()
    assert not (output_dir / "subanswers.jsonl").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_RUNTIME_GOLD_FREE"
    assert manifest["run"]["failure"]["type"] == "RuntimeError"
    assert manifest["run"]["gold_access"] is False


def test_build_model_artifact_hashes_critical_files_without_loading_gpu(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")

    artifact = generator_v7.build_model_artifact(base, adapter)
    assert artifact["base_model"]["config"]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert artifact["strong_sft_adapter"]["weights"]["sha256"] == hashlib.sha256(
        b"adapter-weights"
    ).hexdigest()
    assert artifact["load_contract"] == {
        "torch_dtype": "bfloat16",
        "device": "cuda:0",
        "local_files_only": True,
        "separate_process_required": True,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _authorization_fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base"
    adapter = tmp_path / "strong_sft"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text("base", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("adapter", encoding="utf-8")
    tasks = tmp_path / "c_tasks.depth_1.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    parent = _write_json(tmp_path / "parent.json", {"gold_access": False})
    input_file = tmp_path / "predictions.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")
    generator_lock = generator_v7._file_lock(Path(generator_v7.__file__))
    implementation = {
        "schema_version": generator_v7.IMPLEMENTATION_LOCK_SCHEMA,
        "status": generator_v7.IMPLEMENTATION_LOCK_STATUS,
        "scope": generator_v7.EXECUTION_SCOPE,
        "gold_access": False,
        "authorization": {
            "planner": True,
            "gold_free_materialization": False,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "parents": {"preregistration": generator_v7._file_lock(parent)},
        "runtime_code": {"subanswer_generator": generator_lock},
        "content_reverification": {
            "verified": {
                "models": {
                    "base_model": generator_v7._tree_lock(base),
                    "strong_sft": generator_v7._tree_lock(adapter),
                }
            }
        },
    }
    implementation_path = _write_json(tmp_path / "implementation.json", implementation)
    implementation_lock = generator_v7._file_lock(implementation_path)
    plan = {
        "schema_version": generator_v7.PLAN_LOCK_SCHEMA,
        "status": generator_v7.PLAN_LOCK_STATUS,
        "scope": generator_v7.EXECUTION_SCOPE,
        "gold_access": False,
        "authorization": {
            "planner_complete": True,
            "gold_free_materialization": True,
            "gold_attachment": False,
            "answer_evaluation": False,
            "training": False,
        },
        "parents": {"implementation_lock": implementation_lock},
        "runtime_code": {"subanswer_generator": generator_lock},
        "inputs": {"planner_predictions": generator_v7._file_lock(input_file)},
    }
    plan_path = _write_json(tmp_path / "plan.json", plan)
    plan_lock = generator_v7._file_lock(plan_path)
    descriptor = {
        "schema_version": "paired-dependent-retrieval-v7-stage-descriptor-2",
        "runner_version": "paired-dependent-retrieval-v7-staged-gold-free-1",
        "experiment_id": "MATERIALIZE",
        "stage": "roots",
        "state_depth": 1,
        "runtime_locks": {
            "implementation_lock": implementation_lock,
            "post_plan_execution_lock": plan_lock,
        },
        "outputs": {"c_tasks": generator_v7._file_lock(tasks)},
        "gold_access": False,
    }
    descriptor_path = _write_json(tmp_path / "roots_stage.json", descriptor)
    return {
        "base": base,
        "adapter": adapter,
        "tasks": tasks,
        "implementation": implementation_path,
        "plan": plan_path,
        "descriptor": descriptor_path,
    }


def test_generation_authorization_binds_locks_models_and_task_descriptor(tmp_path: Path):
    fixture = _authorization_fixture(tmp_path)
    tasks_hash = generator_v7.sha256_file(fixture["tasks"])
    artifact, locks = generator_v7.load_generation_authorization(
        implementation_lock_path=fixture["implementation"],
        plan_lock_path=fixture["plan"],
        stage_descriptor_path=fixture["descriptor"],
        tasks_path=fixture["tasks"],
        tasks_sha256=tasks_hash,
        base_model=fixture["base"],
        adapter=fixture["adapter"],
    )
    assert artifact["schema_version"] == generator_v7.MODEL_ARTIFACT_SCHEMA
    assert artifact["base_model"]["tree_sha256"]
    assert artifact["strong_sft_adapter"]["tree_sha256"]
    assert locks["producer_stage_descriptor"]["sha256"]


def test_generation_authorization_rejects_model_or_stage_tampering(tmp_path: Path):
    fixture = _authorization_fixture(tmp_path)
    tasks_hash = generator_v7.sha256_file(fixture["tasks"])
    (fixture["adapter"] / "adapter_config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(generator_v7.TaskValidationError, match="model content differs"):
        generator_v7.load_generation_authorization(
            implementation_lock_path=fixture["implementation"],
            plan_lock_path=fixture["plan"],
            stage_descriptor_path=fixture["descriptor"],
            tasks_path=fixture["tasks"],
            tasks_sha256=tasks_hash,
            base_model=fixture["base"],
            adapter=fixture["adapter"],
        )

    fixture = _authorization_fixture(tmp_path / "stage")
    tasks_hash = generator_v7.sha256_file(fixture["tasks"])
    descriptor = json.loads(fixture["descriptor"].read_text(encoding="utf-8"))
    descriptor["outputs"]["c_tasks"]["sha256"] = "0" * 64
    fixture["descriptor"].write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(generator_v7.TaskValidationError, match="C tasks.*SHA256"):
        generator_v7.load_generation_authorization(
            implementation_lock_path=fixture["implementation"],
            plan_lock_path=fixture["plan"],
            stage_descriptor_path=fixture["descriptor"],
            tasks_path=fixture["tasks"],
            tasks_sha256=tasks_hash,
            base_model=fixture["base"],
            adapter=fixture["adapter"],
        )
