"""Isolation, canonical scoring, identity and conservative selection checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.eval import ppo_emf1_development_v1 as dev


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "bind_base_model", lambda base, root: {"path": "models/llama3-8b", "files": {}})
    monkeypatch.setattr(dev, "make_renderer", lambda base: lambda messages: (dev.canonical_json(messages), 100))
    development = [
        {"dataset": dataset, "qid": f"dev-{index}", "question": "Where do marmots typically live?", "role": "development"}
        for index, dataset in enumerate(dev.DATASETS)
    ]
    source = [
        {**row, "retrieved_passages": [{"id": str(i), "contents": f"Frozen evidence passage {i}"} for i in range(10)],
         "kg_subgraph": [], "gold_answers": ["France", "French Republic"]}
        for row in development
    ]
    paths = {}
    for name, rows in (("development", development), ("source", source)):
        paths[name] = tmp_path / f"{name}.jsonl"
        _rows(paths[name], rows)
    for index, name in enumerate(("confirmation", "canonical", "rollout", "replay")):
        paths[name] = tmp_path / f"{name}.jsonl"
        _rows(paths[name], [
            {"dataset": "hotpotqa", "qid": name, "question": f"How does climate influence {name} migration?"}
        ])
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_model.safetensors").write_bytes(b"sft checkpoint bytes")
    (sft / "adapter_config.json").write_text("{}")
    candidates = [{"model_id": "sft", "checkpoint_path": str(sft), "training_step": 0, "is_sft": True}]
    for step in (200, 400):
        candidates.append({"model_id": f"ppo{step}", "checkpoint_path": str(tmp_path / f"ppo{step}"),
                           "training_step": step, "is_sft": False})
    registry = tmp_path / "registry.json"
    _json(registry, {"candidates": candidates})
    builder = lambda rows, root: ({dev.key(r): [["Film Z", "director", "Person A"]] for r in rows},
                                  {"method": "synthetic test fixture", "offline": True})
    return {"root": tmp_path, "paths": paths, "registry": registry, "builder": builder,
            "candidates": candidates, "source": source, "development": development}


def _prepare(fixture):
    bank = fixture["root"] / "bank"
    dev.prepare_bank(output_dir=bank, candidate_registry=fixture["registry"], experiment_id="TEST-BANK",
                     paths=fixture["paths"], n_per_dataset=1, legacy_builder=fixture["builder"])
    return bank


def _score(fixture, bank, model_id, view, outputs):
    candidate = next(r for r in fixture["candidates"] if r["model_id"] == model_id)
    checkpoint = Path(candidate["checkpoint_path"])
    checkpoint.mkdir(exist_ok=True)
    model_file = checkpoint / "adapter_model.safetensors"
    if not model_file.exists():
        model_file.write_bytes(model_id.encode())
    config_file = checkpoint / "adapter_config.json"
    if not config_file.exists():
        config_file.write_text("{}")
    inputs = dev.read_rows(bank / f"{view}.inputs.jsonl")
    predictions = [
        {"dataset": item["dataset"], "qid": item["qid"], "view": view,
         "input_sha256": item["input_sha256"], "model_id": model_id,
         "adapter_sha256": dev.file_sha(model_file), "bank_manifest_sha256": dev.file_sha(bank / "manifest.json"),
         "adapter_config_sha256": dev.file_sha(config_file),
         "generation_contract_sha256": dev.digest(json.loads((bank / "report.json").read_text())["generation"]),
         "base_model_identity_sha256": dev.digest(json.loads((bank / "report.json").read_text())["base_model"]),
         "raw_output": raw}
        for item, raw in zip(inputs, outputs)
    ]
    prediction_path = fixture["root"] / f"{model_id}-{view}.predictions.jsonl"
    _rows(prediction_path, predictions)
    score_dir = fixture["root"] / f"{model_id}-{view}.scores"
    report = dev.score_predictions(bank_dir=bank, predictions=prediction_path, model_id=model_id,
                                   checkpoint=checkpoint, view=view, output_dir=score_dir,
                                   experiment_id=f"TEST-SCORE-{model_id}-{view}")
    return score_dir, report, prediction_path


def test_prepare_separates_labels_preserves_aliases_and_uses_fixed_step(fixture):
    before = {name: dev.file_sha(path) for name, path in fixture["paths"].items()}
    bank = _prepare(fixture)
    report, _ = dev._load_release(bank)
    assert report["by_dataset"] == {dataset: 1 for dataset in dev.DATASETS}
    assert report["unique_current_families"] == 3
    assert all(not any(overlap.values()) for overlap in report["isolation"].values())
    assert dev.read_rows(bank / "labels.jsonl")[0]["gold_answers"] == ["France", "French Republic"]
    for view in dev.VIEWS:
        for row in dev.read_rows(bank / f"{view}.inputs.jsonl"):
            assert "gold_answers" not in row
            assert "France" not in json.dumps(row["messages"])
            assert "Passage Used:" not in json.dumps(row["messages"])
            assert "[Final Answer]" in json.dumps(row["messages"])
            assert bool(row["kg_subgraph"]) == (view == "legacy")
            assert row["input_sha256"] == dev.input_hash(row)
    assert before == {name: dev.file_sha(path) for name, path in fixture["paths"].items()}
    with pytest.raises(ValueError, match="overwrite"):
        _prepare(fixture)


@pytest.mark.parametrize("boundary", ["confirmation", "canonical", "rollout", "replay"])
@pytest.mark.parametrize("collision", ["qid", "question", "family"])
def test_prepare_rejects_each_identity_overlap(fixture, boundary, collision):
    row = deepcopy(fixture["development"][0])
    if collision != "qid":
        row["qid"] = "different-id"
    if collision == "qid":
        row["question"] = "Which river traverses this continent?"
    elif collision == "family":
        row["question"] = "  WHERE do marmots typically live?  "
    _rows(fixture["paths"][boundary], [row])
    with pytest.raises(ValueError, match="isolation"):
        _prepare(fixture)
    assert not (fixture["root"] / "bank").exists()


def test_prepare_rejects_confirmation_role_or_graph_parent(fixture):
    rows = deepcopy(fixture["source"])
    rows[0]["role"] = "confirmation"
    _rows(fixture["paths"]["source"], rows)
    with pytest.raises(ValueError, match="development-role"):
        _prepare(fixture)
    rows[0]["role"] = "development"
    rows[0]["kg_subgraph"] = [["unwanted", "old", "qpeg"]]
    _rows(fixture["paths"]["source"], rows)
    with pytest.raises(ValueError, match="empty graph"):
        _prepare(fixture)


def test_scoring_preserves_aliases_uses_canonical_answer_and_empty_denominator(fixture):
    bank = _prepare(fixture)
    _, report, _ = _score(fixture, bank, "sft", "legacy", [
        "[Step 1]\nReasoning: evidence\n[Final Answer]\nFrench Republic", "", "France",
    ])
    assert report["macro"]["em"] == pytest.approx(2 / 3)
    assert report["macro"]["f1"] == pytest.approx(2 / 3)
    assert report["n"] == 3
    assert report["by_dataset"]["2wikimultihopqa"]["n"] == 1
    assert dev.canonical_token_f1("yes indeed", "yes") == 0
    assert dev.canonical_exact_match("The U.S.", "US") == 1


@pytest.mark.parametrize("mutation", ["order", "missing", "duplicate", "input", "model", "adapter", "bank", "view", "decode", "base"])
def test_score_rejects_identity_or_incomplete_predictions(fixture, mutation):
    bank = _prepare(fixture)
    _, _, prediction_path = _score(fixture, bank, "sft", "legacy", ["France"] * 3)
    rows = dev.read_rows(prediction_path)
    if mutation == "order":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    else:
        field = {"input": "input_sha256", "model": "model_id", "adapter": "adapter_sha256",
                 "bank": "bank_manifest_sha256", "view": "view", "decode": "generation_contract_sha256",
                 "base": "base_model_identity_sha256"}[mutation]
        rows[0][field] = "wrong"
    _rows(prediction_path, rows)
    with pytest.raises(ValueError):
        dev.score_predictions(bank_dir=bank, predictions=prediction_path, model_id="sft",
                              checkpoint=fixture["root"] / "sft", view="legacy",
                              output_dir=fixture["root"] / "rejected", experiment_id="BAD")
    assert not (fixture["root"] / "rejected").exists()


def test_score_rejects_changed_bank_and_frozen_sft_bytes(fixture):
    bank = _prepare(fixture)
    _, _, prediction_path = _score(fixture, bank, "sft", "legacy", ["France"] * 3)
    (fixture["root"] / "sft/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        dev.score_predictions(bank_dir=bank, predictions=prediction_path, model_id="sft",
                              checkpoint=fixture["root"] / "sft", view="legacy",
                              output_dir=fixture["root"] / "changed-model", experiment_id="BAD")
    with (bank / "labels.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact hash"):
        dev._load_release(bank)


@pytest.mark.parametrize("sft_correct", [True, False])
def test_selection_exact_ties_keep_sft_otherwise_prefer_earlier_ppo(fixture, sft_correct):
    bank = _prepare(fixture)
    score_dirs = []
    for model_id in ("sft", "ppo200", "ppo400"):
        for view in dev.VIEWS:
            answer = "Germany" if model_id == "sft" and not sft_correct else "France"
            directory, _, _ = _score(fixture, bank, model_id, view, [answer] * 3)
            score_dirs.append(directory)
    report = dev.select_checkpoint(bank_dir=bank, score_dirs=score_dirs,
                                    output_dir=fixture["root"] / "selection", experiment_id="SELECT")
    assert report["selected"]["model_id"] == ("sft" if sft_correct else "ppo200")
    assert report["selected_minus_sft"]["legacy"]["em"] == (0 if sft_correct else 1)
    with pytest.raises(ValueError, match="exactly every frozen"):
        dev.select_checkpoint(bank_dir=bank, score_dirs=score_dirs[:-1],
                              output_dir=fixture["root"] / "incomplete-selection", experiment_id="BAD")


def test_registry_portable_project_paths(tmp_path):
    checkpoint = tmp_path / "checkpoints/sft/final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"sft")
    (checkpoint / "adapter_config.json").write_text("{}")
    registry = tmp_path / "registry.json"
    _json(registry, {"candidates": [
        {"model_id": "sft", "checkpoint_path": "checkpoints/sft/final", "training_step": 0, "is_sft": True},
        {"model_id": "ppo", "checkpoint_path": "outputs/ppo/final", "training_step": 600},
    ]})
    assert dev._registry(registry, tmp_path)[0]["checkpoint_path"] == "checkpoints/sft/final"
    assert dev.logical_path(tmp_path / "outputs/ppo/final", tmp_path) == "outputs/ppo/final"


def test_generate_requires_registered_candidate_and_cuda_without_output(fixture, monkeypatch):
    import torch
    bank = _prepare(fixture)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        dev.generate_predictions(bank_dir=bank, model_id="sft", checkpoint=fixture["root"] / "sft",
                                 view="legacy", output_dir=fixture["root"] / "generation",
                                 experiment_id="TEST-NO-GPU")
    assert not (fixture["root"] / "generation").exists()
    with pytest.raises(ValueError, match="frozen candidate"):
        dev.generate_predictions(bank_dir=bank, model_id="unregistered", checkpoint=fixture["root"] / "sft",
                                 view="legacy", output_dir=fixture["root"] / "generation",
                                 experiment_id="TEST-BAD-ID")


def test_prepare_rejects_overlong_prompts_before_release(fixture, monkeypatch):
    monkeypatch.setattr(dev, "make_renderer", lambda base: lambda messages: (json.dumps(messages), 7000))
    with pytest.raises(ValueError, match="6144"):
        _prepare(fixture)
    assert not (fixture["root"] / "bank").exists()


@pytest.mark.parametrize("fail_generation", [False, True])
def test_generate_keeps_complete_outputs_attention_mask_and_failure_artifact(fixture, monkeypatch, fail_generation):
    import torch
    import peft
    import transformers
    bank = _prepare(fixture)
    calls = []

    class Tokenizer:
        pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return dev.canonical_json(messages)

        def __call__(self, prompt, **kwargs):
            assert kwargs["return_attention_mask"] is True
            return {"input_ids": torch.ones((1, 100), dtype=torch.long),
                    "attention_mask": torch.ones((1, 100), dtype=torch.long)}

        def decode(self, response, **kwargs):
            assert response.tolist() == [1001, 1002]
            return "[Step 1]\nReasoning: complete trace\n[Final Answer]\nFrance"

    class Model:
        def to(self, device):
            return self

        def eval(self):
            calls.append("eval")
            return self

        def generate(self, **kwargs):
            assert "eval" in calls
            assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 512
            calls.append("generate")
            if fail_generation:
                raise RuntimeError("synthetic generation failure")
            return torch.cat((kwargs["input_ids"], torch.tensor([[1001, 1002]])), dim=1)

    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs:
                        self if args and str(args[0]).startswith("cuda") else original_to(self, *args, **kwargs))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: Model())

    def adapter_loader(model, checkpoint, **kwargs):
        assert kwargs["is_trainable"] is False
        return model

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter_loader)
    output = fixture["root"] / "generated"
    arguments = dict(bank_dir=bank, model_id="sft", checkpoint=fixture["root"] / "sft",
                     view="legacy", output_dir=output, experiment_id="TEST-GENERATE",
                     base_model=fixture["root"] / "portable-external-model-location")
    if fail_generation:
        with pytest.raises(RuntimeError, match="synthetic"):
            dev.generate_predictions(**arguments)
        assert (output / "FAILED.json").exists()
        assert not (output / "manifest.json").exists()
    else:
        report = dev.generate_predictions(**arguments)
        assert report["n"] == 3 and calls.count("generate") == 3
        rows = dev.read_rows(output / "predictions.jsonl")
        assert all(row["raw_output"].startswith("[Step 1]") for row in rows)
        assert all(row["response_token_ids"] == [1001, 1002] for row in rows)
        _, manifest = dev._load_release(output)
        assert manifest["outputs"]["predictions.jsonl"]["path"] == "predictions.jsonl"
        scored = dev.score_predictions(bank_dir=bank, predictions=output / "predictions.jsonl", model_id="sft",
                                       checkpoint=fixture["root"] / "sft", view="legacy",
                                       output_dir=fixture["root"] / "generated-scored", experiment_id="TEST-GENERATED-SCORE")
        assert scored["macro"] == {"em": 1.0, "f1": 1.0}
    with pytest.raises(ValueError, match="overwrite"):
        dev.generate_predictions(**arguments)


def test_explicit_sft_tokenizer_is_frozen_without_changing_base_identity(fixture):
    tokenizer = fixture["root"] / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (tokenizer / name).write_text("{}")
    bank = fixture["root"] / "explicit-tokenizer-bank"
    dev.prepare_bank(output_dir=bank, candidate_registry=fixture["registry"], experiment_id="TEST-EXPLICIT-TOKENIZER",
                     paths=fixture["paths"], n_per_dataset=1, legacy_builder=fixture["builder"], tokenizer_path=tokenizer)
    report, _ = dev._load_release(bank)
    assert report["tokenizer"]["files"]["tokenizer.json"]["sha256"] == dev.file_sha(tokenizer / "tokenizer.json")
    assert "explicit tokenizer" in report["generation"]["chat_template"]
    assert report["base_model"]["path"] == "models/llama3-8b"
    (tokenizer / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="tokenizer hash differs"):
        dev.generate_predictions(bank_dir=bank, model_id="sft", checkpoint=fixture["root"] / "sft", view="legacy",
                                 output_dir=fixture["root"] / "must-fail-tokenizer", experiment_id="TEST-CHANGED-TOKENIZER",
                                 tokenizer_path=tokenizer)
