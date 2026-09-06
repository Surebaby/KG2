import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
import types

import pytest
import torch

import kgproweight.eval.query_controller_runner as controller_runner
import scripts.eval.evaluate_query_controller_actions as controller_scorer
from kgproweight.kg.question_kg import question_sha256
from kgproweight.eval.query_controller_runner import (
    PROTOCOL_EXPERIMENT_ID,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_STATUS,
    aggregate_predictions,
    build_prediction_record,
    evaluate_teacher_forced_mechanism_gate,
    score_response,
    verify_probe_evaluation_assets,
)
from kgproweight.training.query_controller import (
    CONTROLLER_SCHEMA_VERSION,
    QueryControllerTrainConfig,
    _adapter_config_preflight,
    _canonical_target,
    _validate_config,
    _validate_optimizer_steps,
    _validate_saved_adapter,
    _validate_training_logs,
    _runtime_config_from_cfg,
    _verify_implementation_locks,
    controller_messages,
    encode_record,
    load_config,
    preflight_records,
    run_query_controller_sft,
    select_by_quotas,
    verify_frozen_assets,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.eval.evaluate_query_controller_actions import score_prediction_rows


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    padding_side = "left"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        text = "".join(f"{row['role']}:{row['content']}\n" for row in messages)
        if add_generation_prompt:
            text += "assistant:"
        return text

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text.encode("utf-8"))}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    *,
    dataset: str = "2wikimultihopqa",
    qid: str = "q1",
    split: str = "train",
    slot: str = "q1",
    question: str = "Who directed the film starring Actor A?",
    answer: str = "Film B",
) -> dict:
    state = {
        "state_version": "query-controller-state-v1",
        "original_question": question,
        "previous_actions": [],
        "verified_observations": [],
    }
    target = {
        "action": "retrieve",
        "query": "Actor A starring film",
        "anchor": "Actor A",
        "relation_intent": "cast member inverse",
        "pid": "P161" if dataset == "2wikimultihopqa" else None,
        "dependencies": [],
        "output_slot": "q1",
        "source_action": "text",
    }
    if slot == "q2_dynamic":
        previous_query = "Actor A starring film"
        excerpt = f"Actor A appeared in {answer}."
        state["previous_actions"] = [
            {
                "slot": "q1",
                "action": "retrieve",
                "query": previous_query,
                "output_slot": "q1",
            }
        ]
        state["verified_observations"] = [
            {
                "answer": answer,
                "answer_sha256": _sha(answer),
                "evidence_excerpt": excerpt,
                "evidence_excerpt_sha256": _sha(excerpt),
                "document_id": "doc-1",
                "document_title": answer,
                "sentence_index": 0,
                "provenance": {
                    "source": "train_annotation_support",
                    "annotation_path": (
                        "metadata.evidences.entity[0]"
                        if dataset == "2wikimultihopqa"
                        else "metadata.metadata.question_decomposition[0].answer"
                    ),
                    "binding_method": (
                        "fact_title_and_answer_surface"
                        if dataset == "2wikimultihopqa"
                        else "decomposition_step_support_answer_surface"
                    ),
                },
            }
        ]
        target = {
            "action": "retrieve",
            "query": f"Who directed {answer}?",
            "anchor": answer,
            "relation_intent": "director",
            "pid": "P57" if dataset == "2wikimultihopqa" else None,
            "dependencies": ["q1"],
            "output_slot": "q2",
            "source_action": "text",
        }
    key = f"{dataset}::{qid}"
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "example_id": f"{key}::{slot}",
        "dataset": dataset,
        "qid": qid,
        "question_key": key,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "split": split,
        "slot": slot,
        "turn_index": 1 if slot == "q1" else 2,
        "state": state,
        "target": target,
        "source_provenance": {"builder": "unit-test"},
        "gold_boundary": {
            "train_intermediate_annotation_used": slot == "q2_dynamic",
            "gold_final_answer_visible": False,
            "evaluation_gold_access": False,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_asset_lock_fixture(tmp_path: Path, train_path: Path, dev_path: Path) -> dict:
    protocol_dir = tmp_path / "protocol_assets"
    protocol_dir.mkdir(exist_ok=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_report_path = protocol_dir / "report.json"
    protocol_manifest_path = protocol_dir / "manifest.json"
    protocol_experiment_id = "QUERY-CONTROLLER-UNIT-PROTOCOL"
    protocol_schema = "query-controller-v1-pilot-protocol-2"
    protocol_report_schema = "query-controller-v1-pilot-freeze-report-2"
    protocol_manifest_schema = "query-controller-v1-pilot-manifest-2"
    protocol = {
        "schema_version": protocol_schema,
        "status": "FROZEN_STRICT_ELIGIBLE_2DATASET_ACTIONS_NOT_TRAINED",
        "experiment_id": protocol_experiment_id,
        "enabled_training_datasets": ["2wikimultihopqa", "musique"],
        "implementation_locks": {
            "controller_trainer": {
                "path": "kgproweight/training/query_controller.py",
                "sha256": _file_sha(Path("kgproweight/training/query_controller.py")),
            },
            "central_action_validator": {
                "path": "kgproweight/eval/query_controller_v1.py",
                "sha256": _file_sha(Path("kgproweight/eval/query_controller_v1.py")),
            },
            "controller_train_cli": {
                "path": "scripts/train/query_controller.py",
                "sha256": _file_sha(Path("scripts/train/query_controller.py")),
            },
        },
        "action_contract": {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "source_action": "text",
            "dual_source_routing": False,
            "trainer_allowed_splits": ["train", "dev"],
        },
        "training_contract": {
            "probe_optimizer_steps": 20,
            "confirmation_read_by_trainer": False,
            "objective": "assistant target JSON tokens only; all state tokens masked",
            "initialization": "base Llama-3-8B instruct; independent Controller LoRA",
            "target_truncation": "forbidden",
            "probe_gates": {
                "adapter_save_reload_exact": True,
                "adapter_save_fidelity_exact": True,
                "adapter_clean_reload_single_adapter": True,
                "adapter_clean_reload_tensor_exact": True,
                "adapter_dtype_inventories_recorded": True,
                "adapter_saved_live_clean_reload_dtype_inventories_equal": True,
            },
            "runtime_config": {
                "phase": "probe",
                "experiment_id": "QUERY-CONTROLLER-UNIT",
                "model": {
                    "base_model": "llama3-8B-instruct",
                    "method": "qlora",
                    "initialization": "base_instruct",
                    "init_adapter_path": None,
                    "load_in_4bit": True,
                    "dtype": "bf16",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "data": {
                    "allowed_source_actions": ["text"],
                    "train_quotas": None,
                    "dev_quotas": None,
                },
                "training": {
                    "seed": 42,
                    "max_seq_length": 1024,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "learning_rate": 1.0e-4,
                    "warmup_ratio": 0.03,
                    "lr_scheduler_type": "cosine",
                    "num_train_epochs": 1.0,
                    "max_steps": 20,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "logging_steps": 1,
                    "eval_strategy": "no",
                    "eval_steps": 20,
                    "save_strategy": "no",
                    "save_steps": 20,
                    "save_total_limit": 1,
                    "verify_saved_adapter_reload": True,
                },
            },
            "forbidden_initialization": [
                "Strong-SFT Reader adapter",
                "historical query-planner adapter",
            ],
        },
        "data_release_gates": {
            "action_schema_valid_rate": 1.0,
            "dependency_closed_rate": 1.0,
            "duplicate_example_id_count": 0,
            "exact_actions_per_qid": 2,
            "gold_boundary_valid_rate": 1.0,
            "placeholder_free_rate": 1.0,
            "query_nonrepeat_rate": 1.0,
            "question_identity_join_rate": 1.0,
            "source_action_text_rate": 1.0,
            "state_use_valid_rate": 1.0,
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_report = {
        "schema_version": protocol_report_schema,
        "status": protocol["status"],
        "experiment_id": protocol_experiment_id,
    }
    protocol_report_path.write_text(json.dumps(protocol_report), encoding="utf-8")
    protocol_manifest = {
        "schema_version": protocol_manifest_schema,
        "status": protocol["status"],
        "experiment_id": protocol_experiment_id,
        "training_started": False,
        "outputs": {
            "protocol.json": _file_sha(protocol_path),
            "report.json": _file_sha(protocol_report_path),
        },
    }
    protocol_manifest_path.write_text(json.dumps(protocol_manifest), encoding="utf-8")
    outputs = {
        "train.jsonl": {"rows": sum(1 for _ in train_path.open()), "sha256": _file_sha(train_path)},
        "dev.jsonl": {"rows": sum(1 for _ in dev_path.open()), "sha256": _file_sha(dev_path)},
    }
    selection = {
        "identity_authority": "frozen_protocol_exact_join",
        "source_action": "text",
        "one_qid_per_dataset_scoped_family": True,
    }
    checks = {
        "all_records_schema_valid": True,
        "two_actions_per_qid": True,
        "cross_split_family_overlap": 0,
        "cross_split_qid_overlap": 0,
        "excluded_family_overlap": 0,
        "excluded_qid_overlap": 0,
        "gold_final_answer_visible_count": 0,
        "evaluation_gold_access_count": 0,
        "source_action_values": ["text"],
    }
    release_experiment_id = "QUERY-CONTROLLER-UNIT-RELEASE"
    release_schema = "query-controller-action-release-v1"
    release_status = "COMPLETE_NOT_TRAINED"
    release_report = {
        "schema_version": release_schema,
        "status": release_status,
        "experiment_id": release_experiment_id,
        "all_release_gates_pass": True,
        "selection": selection,
        "checks": checks,
        "outputs": outputs,
        "identity_locks": [
            {
                "role": "protocol",
                "path": str(protocol_path),
                "sha256": _file_sha(protocol_path),
            }
        ],
    }
    release_report_path = tmp_path / "report.json"
    release_manifest_path = tmp_path / "manifest.json"
    release_report_path.write_text(json.dumps(release_report), encoding="utf-8")
    release_manifest_path.write_text(
        json.dumps({"status": release_status, "run": release_report}),
        encoding="utf-8",
    )
    return {
        "protocol_path": str(protocol_path),
        "protocol_report_path": str(protocol_report_path),
        "protocol_manifest_path": str(protocol_manifest_path),
        "release_report_path": str(release_report_path),
        "release_manifest_path": str(release_manifest_path),
        "expected_protocol_sha256": _file_sha(protocol_path),
        "expected_protocol_report_sha256": _file_sha(protocol_report_path),
        "expected_protocol_manifest_sha256": _file_sha(protocol_manifest_path),
        "expected_train_sha256": _file_sha(train_path),
        "expected_dev_sha256": _file_sha(dev_path),
        "expected_release_report_sha256": _file_sha(release_report_path),
        "expected_release_manifest_sha256": _file_sha(release_manifest_path),
        "expected_protocol_schema_version": protocol_schema,
        "expected_protocol_report_schema_version": protocol_report_schema,
        "expected_protocol_manifest_schema_version": protocol_manifest_schema,
        "expected_protocol_status": protocol["status"],
        "expected_protocol_experiment_id": protocol_experiment_id,
        "expected_release_schema_version": release_schema,
        "expected_release_status": release_status,
        "expected_release_experiment_id": release_experiment_id,
    }


def _pair(**kwargs) -> list[dict]:
    return [_record(slot=slot, **kwargs) for slot in ("q1", "q2_dynamic")]


def _cfg(tmp_path: Path, train: list[dict], dev: list[dict], **kwargs) -> QueryControllerTrainConfig:
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    locks = _write_asset_lock_fixture(tmp_path, train_path, dev_path)
    return QueryControllerTrainConfig(
        experiment_id="QUERY-CONTROLLER-UNIT",
        train_path=str(train_path),
        dev_path=str(dev_path),
        output_dir=str(tmp_path / "out"),
        logging_dir=str(tmp_path / "out" / "tensorboard"),
        **locks,
        **kwargs,
    )


def test_prompt_contains_only_model_visible_state():
    record = _record(slot="q2_dynamic")
    record["source_provenance"]["SECRET_PROVENANCE"] = "MUST_NOT_APPEAR"
    record["gold_boundary"]["SECRET_GOLD"] = "MUST_NOT_APPEAR"
    record["target"]["query"] = "SECRET_TARGET Film B director"
    messages = controller_messages(record, include_target=False)
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "SECRET_PROVENANCE" not in rendered
    assert "SECRET_GOLD" not in rendered
    assert "SECRET_TARGET" not in rendered
    assert "verified_observations" in rendered
    assert "Do not answer" in messages[0]["content"]


def test_target_is_exact_canonical_json_in_fixed_key_order():
    record = _record()
    expected = (
        '{"action":"retrieve","query":"Actor A starring film","anchor":"Actor A",'
        '"relation_intent":"cast member inverse","pid":"P161","dependencies":[],'
        '"output_slot":"q1","source_action":"text"}'
    )
    assert _canonical_target(record["target"]) == expected
    assert controller_messages(record, include_target=True)[-1]["content"] == expected


def test_encode_masks_prompt_and_supervises_only_assistant_json():
    encoded = encode_record(_record(), FakeTokenizer(), max_seq_length=4096)
    boundary = encoded["prompt_length"]
    assert all(label == -100 for label in encoded["labels"][:boundary])
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert encoded["supervised_length"] > 0


def test_encode_refuses_to_truncate_target():
    with pytest.raises(ValueError, match="no truncation allowed"):
        encode_record(_record(), FakeTokenizer(), max_seq_length=20)


def test_preflight_accepts_q1_and_observation_conditioned_q2(tmp_path):
    train = _pair(qid="train")
    dev = _pair(
        qid="dev",
        split="dev",
        question="Which city contains the club where Player C competed?",
        answer="Club D",
    )
    selected_train, selected_dev, report = preflight_records(_cfg(tmp_path, train, dev))
    assert len(selected_train) == 2
    assert len(selected_dev) == 2
    assert report["status"] == "PASS"
    assert report["gates"]["source_action_text_only"] is True
    assert report["raw"]["isolation"]["family_overlap_count"] == 0


@pytest.mark.parametrize("fault", ["repeat", "placeholder", "gold", "source_action"])
def test_preflight_rejects_training_contract_faults(tmp_path, fault):
    train = _pair(qid="train")
    if fault == "repeat":
        train[0]["target"]["query"] = train[0]["state"]["original_question"]
    elif fault == "placeholder":
        train[0]["target"]["query"] = "Who directed $hop_1?"
    elif fault == "gold":
        train[0]["gold_boundary"]["gold_final_answer_visible"] = True
    else:
        train[0]["target"]["source_action"] = "graph"
    dev = _pair(
        qid="dev", split="dev", question="Where was Writer E born?"
    )
    with pytest.raises(ValueError):
        preflight_records(_cfg(tmp_path, train, dev))


@pytest.mark.parametrize("overlap", ["qid", "family"])
def test_preflight_rejects_train_dev_identity_overlap(tmp_path, overlap):
    train = _pair(qid="same")
    if overlap == "qid":
        dev = _pair(
            qid="same", split="dev", question="Where was Writer E born?"
        )
    else:
        dev = _pair(qid="different", split="dev")
    with pytest.raises(ValueError, match=f"{overlap}.*overlap"):
        preflight_records(_cfg(tmp_path, train, dev))


def test_same_family_hash_in_different_datasets_is_not_overlap(tmp_path):
    train = _pair(dataset="2wikimultihopqa", qid="train")
    dev = _pair(dataset="musique", qid="dev", split="dev")
    _, _, report = preflight_records(_cfg(tmp_path, train, dev))
    assert report["raw"]["isolation"]["family_overlap_count"] == 0


def test_dataset_slot_quotas_are_exact_and_deterministic():
    rows = [
        row
        for dataset in ("2wikimultihopqa", "musique")
        for index in range(5)
        for row in _pair(dataset=dataset, qid=f"{dataset}-{index}")
    ]
    quotas = {
        "2wikimultihopqa": {"q1": 2, "q2_dynamic": 2},
        "musique": {"q1": 3, "q2_dynamic": 3},
    }
    first = select_by_quotas(rows, quotas, seed=42, require_all_records_selected=False)
    second = select_by_quotas(rows, quotas, seed=42, require_all_records_selected=False)
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    counts = {
        (dataset, slot): sum(
            row["dataset"] == dataset and row["slot"] == slot for row in first
        )
        for dataset in quotas
        for slot in quotas[dataset]
    }
    assert counts == {
        ("2wikimultihopqa", "q1"): 2,
        ("2wikimultihopqa", "q2_dynamic"): 2,
        ("musique", "q1"): 3,
        ("musique", "q2_dynamic"): 3,
    }
    assert {
        row["qid"] for row in first if row["slot"] == "q1"
    } == {
        row["qid"] for row in first if row["slot"] == "q2_dynamic"
    }


def test_unequal_slot_quotas_are_rejected():
    rows = [row for index in range(3) for row in _pair(qid=f"q{index}")]
    with pytest.raises(ValueError, match="equal q1/q2_dynamic"):
        select_by_quotas(
            rows,
            {"2wikimultihopqa": {"q1": 2, "q2_dynamic": 1}},
            seed=42,
            require_all_records_selected=False,
        )


def test_unpaired_release_cannot_pass_preflight(tmp_path):
    train = [_record(qid="train", slot="q1")]
    dev = _pair(qid="dev", split="dev", question="Where was Writer E born?")
    with pytest.raises(ValueError, match="paired qid"):
        preflight_records(_cfg(tmp_path, train, dev))


def test_frozen_asset_lock_detects_train_byte_tamper(tmp_path):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )
    with Path(cfg.train_path).open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="hash drift for train"):
        verify_frozen_assets(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 999),
        ("base_model", "arbitrary-model"),
        ("method", "lora"),
        ("seed", 7),
        ("load_in_4bit", False),
        ("dtype", "fp32"),
        ("lora_r", 8),
        ("lora_alpha", 64),
        ("lora_dropout", 0.2),
        ("target_modules", ("q_proj",)),
        ("max_seq_length", 2048),
        ("batch_size", 2),
        ("eval_batch_size", 2),
        ("grad_accum", 4),
        ("learning_rate", 2.0e-4),
        ("warmup_ratio", 0.1),
        ("lr_scheduler_type", "linear"),
        ("epochs", 2.0),
        ("weight_decay", 0.1),
        ("max_grad_norm", 0.5),
        ("logging_steps", 2),
        ("eval_strategy", "steps"),
        ("eval_steps", 10),
        ("save_strategy", "steps"),
        ("save_steps", 10),
        ("save_total_limit", 2),
        ("verify_saved_adapter_reload", False),
        ("allowed_source_actions", ("text", "graph")),
        ("train_quotas", {"2wikimultihopqa": {"q1": 1, "q2_dynamic": 1}}),
        ("dev_quotas", {"musique": {"q1": 1, "q2_dynamic": 1}}),
    ],
)
def test_frozen_runtime_config_rejects_every_preregistered_field_tamper(
    tmp_path, field, value
):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )
    with pytest.raises(ValueError):
        verify_frozen_assets(replace(cfg, **{field: value}))


def test_probe_only_runtime_and_exact_optimizer_step_gate(tmp_path):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )
    with pytest.raises(ValueError, match="20-step probe only"):
        run_query_controller_sft(cfg, probe=False)
    assert _validate_optimizer_steps(20, cfg)["exact_optimizer_steps_pass"] is True
    with pytest.raises(RuntimeError, match="optimizer-step mismatch"):
        _validate_optimizer_steps(19, cfg)


def test_protocol_tamper_fails_before_cuda_query(tmp_path, monkeypatch):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )

    def cuda_must_not_be_queried():
        raise AssertionError("CUDA was queried before frozen-asset preflight")

    monkeypatch.setattr("torch.cuda.is_available", cuda_must_not_be_queried)
    with pytest.raises(ValueError, match="max_steps"):
        run_query_controller_sft(replace(cfg, max_steps=999), probe=True)


def test_implementation_locks_are_required_and_hash_checked():
    valid = {
        "implementation_locks": {
            "controller_trainer": {
                "path": "kgproweight/training/query_controller.py",
                "sha256": _file_sha(Path("kgproweight/training/query_controller.py")),
            },
            "central_action_validator": {
                "path": "kgproweight/eval/query_controller_v1.py",
                "sha256": _file_sha(Path("kgproweight/eval/query_controller_v1.py")),
            },
            "controller_train_cli": {
                "path": "scripts/train/query_controller.py",
                "sha256": _file_sha(Path("scripts/train/query_controller.py")),
            },
        }
    }
    assert _verify_implementation_locks(valid)["status"] == "PASS"
    missing = json.loads(json.dumps(valid))
    del missing["implementation_locks"]["controller_train_cli"]
    with pytest.raises(ValueError, match="missing required implementation"):
        _verify_implementation_locks(missing)
    tampered = json.loads(json.dumps(valid))
    tampered["implementation_locks"]["controller_trainer"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash drift"):
        _verify_implementation_locks(tampered)


def test_static_probe_config_rejects_reload_and_method_downgrades(tmp_path):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )
    with pytest.raises(ValueError, match="reload"):
        _validate_config(replace(cfg, verify_saved_adapter_reload=False))
    with pytest.raises(ValueError, match="qlora"):
        _validate_config(replace(cfg, method="lora"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("all_release_gates_pass", False, "status/all-gates"),
        ("identity_authority", "self_reported", "identity authority"),
    ],
)
def test_frozen_asset_lock_fails_closed_on_release_gate_drift(
    tmp_path, field, value, message
):
    cfg = _cfg(
        tmp_path,
        _pair(qid="train"),
        _pair(qid="dev", split="dev", question="Where was Writer E born?"),
    )
    report_path = Path(cfg.release_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if field == "identity_authority":
        report["selection"][field] = value
    else:
        report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cfg = replace(cfg, expected_release_report_sha256=_file_sha(report_path))
    with pytest.raises(ValueError, match=message):
        verify_frozen_assets(cfg)


def test_planner_adapter_initialization_preflight(tmp_path):
    adapter = tmp_path / "planner"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        ),
        encoding="utf-8",
    )
    train = _pair(qid="train")
    dev = _pair(qid="dev", split="dev", question="Where was Writer E born?")
    cfg = _cfg(
        tmp_path,
        train,
        dev,
        initialization="planner_adapter",
        init_adapter_path=str(adapter),
    )
    report = _adapter_config_preflight(cfg)
    assert report["r"] == 16
    with pytest.raises(ValueError, match="independent base_instruct"):
        preflight_records(cfg)


def test_config_forwards_tensorboard_and_keeps_formal_path_separate():
    from scripts.prepare.build_query_controller_action_supervision_v1 import (
        FORMAL_RUNTIME_CONFIG,
    )

    config_path = Path(
        "configs/training/query_controller_action_v1_probe20_seed42_v4_4.yaml"
    )
    cfg = load_config(config_path)
    assert cfg.max_steps == 20
    assert cfg.initialization == "base_instruct"
    assert cfg.report_to == ("tensorboard",)
    assert cfg.logging_dir == f"{cfg.output_dir}/tensorboard"
    assert cfg.extra["guardrails"]["formal_pilot_output_dir"] != cfg.output_dir
    assert _runtime_config_from_cfg(cfg) == FORMAL_RUNTIME_CONFIG
    overridden = load_config(config_path, output_override="outputs/probes/unique-controller")
    assert overridden.logging_dir == "outputs/probes/unique-controller/tensorboard"


def test_training_log_gate_requires_finite_loss_and_gradient():
    report = _validate_training_logs(
        [{"loss": 1.5, "grad_norm": 0.7, "step": 1}], train_loss=1.5
    )
    assert report["finite_gradient_norms"] is True
    with pytest.raises(RuntimeError, match="gradient"):
        _validate_training_logs([{"loss": 1.5}], train_loss=1.5)
    with pytest.raises(RuntimeError, match="nonzero"):
        _validate_training_logs(
            [{"loss": 1.5, "grad_norm": 0.0, "step": 1}], train_loss=1.5
        )


def test_saved_adapter_uses_clean_single_adapter_reload(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    saved_state = {"base_model.layer.lora_A.weight": torch.tensor([[1.0, 2.0]])}
    save_file(saved_state, str(final_dir / "adapter_model.safetensors"))

    calls: list[str] = []

    class FakeLiveModel:
        active_adapter = "default"
        peft_config = {"default": object()}
        states = {"default": saved_state}

        def unload(self):
            calls.append("unload")
            return object()

    class FakeReloadedModel:
        peft_config = {"default": object()}
        states = {"default": saved_state}

    class FakePeftConfig:
        peft_type = "LORA"

        @classmethod
        def from_pretrained(cls, path):
            calls.append("config")
            return cls()

    class FakePeftModel:
        @classmethod
        def from_pretrained(
            cls, base, path, *, is_trainable, autocast_adapter_dtype
        ):
            assert calls[-1] == "unload"
            assert is_trainable is False
            assert autocast_adapter_dtype is True
            calls.append("clean_from_disk")
            return FakeReloadedModel()

    def fake_get_state(model, *, adapter_name, save_embedding_layers):
        assert save_embedding_layers is False
        calls.append(f"state:{type(model).__name__}:{adapter_name}")
        return model.states[adapter_name]

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            PeftConfig=FakePeftConfig,
            PeftModel=FakePeftModel,
            get_peft_model_state_dict=fake_get_state,
        ),
    )
    report = _validate_saved_adapter(final_dir, live_model=FakeLiveModel())
    assert report["save_fidelity_exact"] is True
    assert report["clean_reload_single_adapter"] is True
    assert report["clean_reload_tensor_exact"] is True
    assert report["tensor_reload_exact"] is True
    assert report["saved_dtype_inventory"] == {"torch.float32": 1}
    assert report["live_dtype_inventory"] == {"torch.float32": 1}
    assert report["clean_reload_dtype_inventory"] == {"torch.float32": 1}
    assert report["dtype_inventories_recorded"] is True
    assert report["saved_live_clean_reload_dtype_inventories_equal"] is True
    assert calls == [
        "config",
        "state:FakeLiveModel:default",
        "unload",
        "clean_from_disk",
        "state:FakeReloadedModel:default",
    ]


def test_saved_adapter_clean_reload_detects_disk_value_drift(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    disk_state = {"base_model.layer.lora_A.weight": torch.tensor([[1.0]])}
    save_file(disk_state, str(final_dir / "adapter_model.safetensors"))

    class FakeConfig:
        peft_type = "LORA"

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class FakeLive:
        active_adapter = "default"
        peft_config = {"default": object()}
        states = {"default": disk_state}

        def unload(self):
            return object()

    class FakeReload:
        peft_config = {"default": object()}
        states = {
            "default": {"base_model.layer.lora_A.weight": torch.tensor([[2.0]])}
        }

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeReload()

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            PeftConfig=FakeConfig,
            PeftModel=FakePeftModel,
            get_peft_model_state_dict=lambda model, **kwargs: model.states["default"],
        ),
    )
    with pytest.raises(RuntimeError, match="clean_reload adapter tensors differ"):
        _validate_saved_adapter(final_dir, live_model=FakeLive())


def test_saved_adapter_clean_reload_rejects_dtype_inventory_drift(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    disk_state = {"base_model.layer.lora_A.weight": torch.tensor([[1.0]])}
    save_file(disk_state, str(final_dir / "adapter_model.safetensors"))

    class FakeConfig:
        peft_type = "LORA"

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class FakeLive:
        active_adapter = "default"
        peft_config = {"default": object()}
        states = {"default": disk_state}

        def unload(self):
            return object()

    class FakeReload:
        peft_config = {"default": object()}
        states = {
            "default": {
                "base_model.layer.lora_A.weight": torch.tensor(
                    [[1.0]], dtype=torch.bfloat16
                )
            }
        }

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["autocast_adapter_dtype"] is True
            return FakeReload()

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            PeftConfig=FakeConfig,
            PeftModel=FakePeftModel,
            get_peft_model_state_dict=lambda model, **kwargs: model.states["default"],
        ),
    )
    with pytest.raises(RuntimeError, match="dtype inventory differs"):
        _validate_saved_adapter(final_dir, live_model=FakeLive())


def test_gold_free_runner_scores_exact_valid_action_without_qa_answer():
    reference = _record()
    response = _canonical_target(reference["target"])
    scored = score_response(reference, response)
    assert scored["valid"] is True
    assert scored["structured_target_exact"] is True
    assert scored["canonical_text_exact"] is True
    prediction = build_prediction_record(
        reference, response, prompt_tokens=80, generated_tokens=20
    )
    rendered = json.dumps(prediction, ensure_ascii=False)
    assert "original_question" not in rendered
    assert "source_provenance" not in rendered
    assert "target" not in prediction
    assert "gold_boundary" not in prediction


def test_gold_free_runner_preserves_raw_text_and_rejects_invalid_state_use():
    reference = _record(slot="q2_dynamic")
    predicted = dict(reference["target"])
    predicted["query"] = "Who directed a different movie?"
    response = json.dumps(predicted, ensure_ascii=False, separators=(",", ":"))
    scored = score_response(reference, response)
    assert scored["valid"] is False
    assert scored["checks"]["state_use_valid"] is False
    assert "state_not_used" in scored["error_codes"]
    malformed = score_response(reference, "not JSON")
    assert malformed["valid"] is False
    assert "target_schema" in malformed["error_codes"]


def test_gold_free_runner_aggregate_is_mechanism_only():
    q1 = _record()
    q2 = _record(slot="q2_dynamic")
    rows = [
        build_prediction_record(
            row, _canonical_target(row["target"]), prompt_tokens=10, generated_tokens=5
        )
        for row in (q1, q2)
    ]
    report = aggregate_predictions(rows)
    assert report["scope"] == "controller_action_only_no_retrieval_no_qa_gold_no_em"
    assert report["overall"]["valid_rate"] == 1.0
    assert set(report["by_slot"]) == {"q1", "q2_dynamic"}
    assert "em" not in report["overall"]
    assert "f1" not in report["overall"]


def test_full_runner_prediction_interoperates_with_canonical_scorer():
    reference = _record()
    response = _canonical_target(reference["target"])
    prediction = build_prediction_record(
        reference, response, prompt_tokens=80, generated_tokens=20
    )
    report, details = score_prediction_rows([reference], [prediction])
    assert report["identity_join_rate"] == 1.0
    assert report["metrics"]["schema_valid_rate"] == 1.0
    assert details[0]["target_exact"] is True
    tampered = dict(prediction)
    tampered["response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="response hash mismatch"):
        score_prediction_rows([reference], [tampered])


def test_scorer_rejects_mixed_legacy_and_full_prediction_provenance():
    first = _record(qid="first")
    second = _record(qid="second", question="Where was Writer E born?")
    full = build_prediction_record(
        first, _canonical_target(first["target"]), prompt_tokens=20, generated_tokens=8
    )
    legacy = {"example_id": second["example_id"], "response_text": _canonical_target(second["target"])}
    with pytest.raises(ValueError, match="mixing legacy"):
        score_prediction_rows([first, second], [full, legacy])


def test_teacher_forced_gate_is_versioned_and_not_runtime_claim():
    references = [
        _record(
            dataset=dataset,
            qid=f"{dataset}-{index}",
            slot=slot,
            split="dev",
            question=f"Question {dataset} {index}?",
            answer=f"Entity {index}",
        )
        for dataset in ("2wikimultihopqa", "musique")
        for index in range(60)
        for slot in ("q1", "q2_dynamic")
    ]
    rows = [
        build_prediction_record(
            row, _canonical_target(row["target"]), prompt_tokens=10, generated_tokens=5
        )
        for row in references
    ]
    report = aggregate_predictions(rows)
    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "status": PROTOCOL_STATUS,
        "experiment_id": PROTOCOL_EXPERIMENT_ID,
        "probe_evaluation_contract": {
            "authorization": "post_probe_dev_teacher_forced_only",
            "cohort_role": "dev",
            "input_split": "dev",
            "datasets": ["2wikimultihopqa", "musique"],
            "slots": ["q1", "q2_dynamic"],
            "exact_qids_per_enabled_dataset": 60,
            "exact_action_rows_per_enabled_dataset": 120,
            "exact_actions": 240,
            "state_source": "annotation_derived_but_passage_bound",
            "runtime_reader_predicted": False,
            "mechanism_gates": {
                "identity_join_rate": 1.0,
                "schema_valid_rate_min_each_dataset_slot": {
                    "q1": 0.97, "q2_dynamic": 0.95,
                },
                "dependency_closed_rate_min_each_dataset_slot": {
                    "q1": 0.97, "q2_dynamic": 0.95,
                },
                "state_use_valid_rate_min_each_dataset_slot": {
                    "q1": 0.97, "q2_dynamic": 0.95,
                },
                "query_nonrepeat_rate": 1.0,
                "placeholder_free_rate": 1.0,
            },
        },
    }
    gate = evaluate_teacher_forced_mechanism_gate(report, protocol, cohort_role="dev")
    assert gate["passed"] is True
    assert gate["runtime_reader_predicted"] is False
    assert gate["schema_version"] == "query-controller-teacher-forced-mechanism-gate-v1"
    with pytest.raises(ValueError, match="240"):
        evaluate_teacher_forced_mechanism_gate(
            aggregate_predictions(rows[:4]), protocol, cohort_role="dev"
        )
    with pytest.raises(ValueError, match="dev only"):
        evaluate_teacher_forced_mechanism_gate(
            report, protocol, cohort_role="confirmation"
        )


def test_probe_evaluation_full_preflight_forwards_all_protocol_bundle_hashes(
    tmp_path, monkeypatch
):
    """Exercise the public preflight so helper signature drift fails in tests."""

    references = [
        {
            "dataset": dataset,
            "qid": f"{dataset}-{index}",
            "split": "dev",
            "slot": slot,
        }
        for dataset in ("2wikimultihopqa", "musique")
        for index in range(60)
        for slot in ("q1", "q2_dynamic")
    ]
    protocol = {
        "probe_evaluation_contract": {
            "authorization": "post_probe_dev_teacher_forced_only",
            "cohort_role": "dev",
            "input_split": "dev",
            "datasets": ["2wikimultihopqa", "musique"],
            "slots": ["q1", "q2_dynamic"],
            "confirmation_access": False,
            "prospective_access": False,
            "exact_qids_per_enabled_dataset": 60,
            "exact_action_rows_per_enabled_dataset": 120,
            "exact_actions": 240,
            "state_source": "annotation_derived_but_passage_bound",
            "runtime_reader_predicted": False,
            "outcome_metrics_authorized": {
                "em": False,
                "f1": False,
                "ihr": False,
            },
            "required_probe_artifacts": {
                "experiment_id": controller_runner.PROBE_EXPERIMENT_ID,
                "manifest_status": "COMPLETE",
                "completed_optimizer_steps": 20,
                "training_manifest_sha256": "required_external_eval_lock",
                "adapter_sha256": "required_external_eval_lock",
                "asset_lock_lineage_match": True,
            },
            "decoding": {
                "strategy": "greedy",
                "do_sample": False,
                "temperature": 0.0,
                "seed": 42,
                "batch_size": 4,
                "max_input_tokens": 1024,
                "max_new_tokens": 192,
                "base_model": "llama3-8B-instruct",
                "dtype": "bf16",
                "load_in_4bit": True,
            },
            "status_semantics": {
                "generation_complete_status": "COMPLETE_GENERATION_NOT_MECHANISM_PASS",
                "mechanism_pass_requires_separate_scorer_gate": True,
                "generation_complete_implies_mechanism_pass": False,
            },
        }
    }
    bundle_hashes = {
        "protocol_sha256": "1" * 64,
        "protocol_report_sha256": "2" * 64,
        "protocol_manifest_sha256": "3" * 64,
    }
    seen = {}

    monkeypatch.setattr(
        controller_runner,
        "_verify_protocol_bundle",
        lambda *args, **kwargs: (protocol, dict(bundle_hashes)),
    )
    monkeypatch.setattr(
        controller_runner,
        "_verify_eval_successor_bundle",
        lambda *args, **kwargs: (
            {"schema_version": controller_runner.EVAL_PROTOCOL_SCHEMA_VERSION},
            {"status": "PASS", "eval_protocol_sha256": "9" * 64},
        ),
    )
    monkeypatch.setattr(
        controller_runner, "read_reference_records", lambda path: references
    )

    def fake_release(
        input_path,
        rows,
        *,
        protocol_path,
        protocol,
        protocol_sha256,
        protocol_report_sha256,
        protocol_manifest_sha256,
    ):
        seen["release_protocol_hashes"] = (
            protocol_sha256,
            protocol_report_sha256,
            protocol_manifest_sha256,
        )
        return {
            "release_report_sha256": "4" * 64,
            "release_manifest_sha256": "5" * 64,
            "input_sha256": "6" * 64,
        }

    def fake_training(*args, **kwargs):
        seen["training_protocol_hashes"] = (
            kwargs["protocol_sha256"],
            kwargs["protocol_report_sha256"],
            kwargs["protocol_manifest_sha256"],
        )
        return {"status": "PASS"}

    monkeypatch.setattr(
        controller_runner, "_verify_dev_release_and_pairs", fake_release
    )
    monkeypatch.setattr(
        controller_runner, "_verify_probe_training_and_adapter", fake_training
    )
    monkeypatch.setattr(
        controller_runner,
        "_verify_eval_parent_asset_lineage",
        lambda *args, **kwargs: {"status": "PASS"},
    )

    rows, verified_protocol, report = verify_probe_evaluation_assets(
        input_path=tmp_path / "dev.jsonl",
        adapter_path=tmp_path / "final",
        protocol_path=tmp_path / "protocol.json",
        eval_protocol_path=tmp_path / "eval_protocol.json",
        training_manifest_path=tmp_path / "training_manifest.json",
        expected_protocol_sha256="1" * 64,
        expected_eval_protocol_sha256="9" * 64,
        expected_training_manifest_sha256="7" * 64,
        expected_adapter_sha256="8" * 64,
        generation_experiment_id=controller_runner.EVAL_GENERATION_EXPERIMENT_ID,
        generation_output_dir=controller_runner.EVAL_GENERATION_OUTPUT_DIR,
        cohort_role="dev",
        batch_size=4,
        max_input_tokens=1024,
        max_new_tokens=192,
        seed=42,
        base_model="llama3-8B-instruct",
        dtype="bf16",
        load_in_4bit=True,
    )

    expected_hashes = ("1" * 64, "2" * 64, "3" * 64)
    assert seen["release_protocol_hashes"] == expected_hashes
    assert seen["training_protocol_hashes"] == expected_hashes
    assert rows == references
    assert verified_protocol == protocol
    assert report["status"] == "PASS"


def test_formal_scorer_forwards_all_protocol_bundle_hashes(tmp_path, monkeypatch):
    references = [
        _record(
            dataset=dataset,
            qid=f"{dataset}-{index}",
            slot=slot,
            split="dev",
            question=f"Question {dataset} {index}?",
            answer=f"Entity {index}",
        )
        for dataset in ("2wikimultihopqa", "musique")
        for index in range(60)
        for slot in ("q1", "q2_dynamic")
    ]
    predictions = [
        build_prediction_record(
            row,
            _canonical_target(row["target"]),
            prompt_tokens=10,
            generated_tokens=5,
        )
        for row in references
    ]
    records_path = tmp_path / "dev.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    protocol_path = tmp_path / "protocol.json"
    eval_protocol_path = tmp_path / "eval_protocol.json"
    generation_manifest_path = tmp_path / "generation_manifest.json"
    _write_jsonl(records_path, references)
    _write_jsonl(predictions_path, predictions)
    for path in (protocol_path, eval_protocol_path, generation_manifest_path):
        path.write_text("{}\n", encoding="utf-8")

    protocol = {"probe_evaluation_contract": {}}
    parent_hashes = {
        "protocol_sha256": "1" * 64,
        "protocol_report_sha256": "2" * 64,
        "protocol_manifest_sha256": "3" * 64,
    }
    seen = {}
    monkeypatch.setattr(
        controller_scorer,
        "_verify_protocol_bundle",
        lambda *args, **kwargs: (protocol, dict(parent_hashes)),
    )
    monkeypatch.setattr(
        controller_scorer,
        "_verify_eval_successor_bundle",
        lambda *args, **kwargs: (
            {},
            {"status": "PASS", "eval_protocol_sha256": "9" * 64},
        ),
    )

    def fake_release(
        input_path,
        rows,
        *,
        protocol_path,
        protocol,
        protocol_sha256,
        protocol_report_sha256,
        protocol_manifest_sha256,
    ):
        seen["hashes"] = (
            protocol_sha256,
            protocol_report_sha256,
            protocol_manifest_sha256,
        )
        return {"input_sha256": _file_sha(records_path)}

    monkeypatch.setattr(controller_scorer, "_verify_dev_release_and_pairs", fake_release)
    monkeypatch.setattr(
        controller_scorer,
        "_verify_generation_manifest",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        controller_scorer,
        "evaluate_teacher_forced_mechanism_gate",
        lambda *args, **kwargs: {"status": "PASS", "passed": True},
    )

    report = controller_scorer.run(
        records_path=records_path,
        predictions_path=predictions_path,
        protocol_path=protocol_path,
        eval_protocol_path=eval_protocol_path,
        cohort_role="dev",
        expected_protocol_sha256="1" * 64,
        expected_eval_protocol_sha256="9" * 64,
        expected_predictions_sha256=_file_sha(predictions_path),
        generation_manifest_path=generation_manifest_path,
        expected_generation_manifest_sha256=_file_sha(generation_manifest_path),
        output_dir=tmp_path / "score",
        expected_split="dev",
        experiment_id="QUERY-CONTROLLER-E1-SCORER-TEST",
    )
    assert seen["hashes"] == ("1" * 64, "2" * 64, "3" * 64)
    assert report["status"] == "COMPLETE_SCORED"
