"""Synthetic CPU tests: isolation, immutable inputs and true scorer contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.kg.question_kg import make_question_kg_record
from scripts.prepare import source_quality_candidate_bank_v1 as bank


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("source-quality-fixtures")
    data = root / "data"; data.mkdir()
    values = {name: [] for name in ("silver_train", "question_kg_records", "source_gate_records", "prompt_groups")}
    for dataset in bank.DATASETS:
        for index in range(1000):
            qid = f"{dataset}-{index}"
            question = f"Where did synthetic researcher {dataset} {index} ultimately arrive?"
            graph = dataset == "2wikimultihopqa" and index < 800
            triples = [["Alpha", "links to", "Beta"], ["Beta", "links to", "Gamma"]] if graph else []
            plan = {"recognized": True, "hops": [{"subject": "Alpha", "pids": ["P1"], "output_slot": "hop_1"},
                     {"subject": "$hop_1", "pids": ["P2"], "output_slot": "hop_2"}]} if graph else {}
            record = make_question_kg_record(dataset=dataset, qid=qid, question=question, triples=triples, query_plan=plan,
                        provenance={"builder_version": "synthetic-test-only", "gold_access": False,
                                    "complete_plan_execution": graph, "historical_cutoff": "2020-12-09T23:59:59Z"})
            record["runtime_error"] = None
            record["execution"] = {"complete_plan_execution": graph, "hops": [
                {"hop_index": i + 1, "input_entities": [{"qid": f"Q{i + 1}", "score": 1.0}], "matches": [triple]}
                for i, triple in enumerate(triples)]}
            decision = bank.evaluate_graph_gate(record, dataset=dataset, qid=qid, question=question,
                                historical_cutoff="2020-12-09T23:59:59Z").to_dict()
            identity = {"dataset": dataset, "qid": qid, "question": question}
            values["silver_train"].append({**identity, "answer": "SECRET GOLD VALUE", "steps": ["SECRET GOLD TRACE"],
                "metadata": {"source_split": "train", "gold_answer": "SECRET GOLD VALUE"},
                "retrieved_passages": [{"id": str(i), "source": "synthetic", "contents": f"Frozen passage {i}"} for i in range(10)]})
            values["question_kg_records"].append(record)
            values["source_gate_records"].append({**identity, **decision})
            values["prompt_groups"].append({**identity, "evaluation_eligible": False})
    outputs = {}
    for name, rows in values.items():
        bank.write_rows(data / f"{name}.jsonl", rows)
        outputs[name] = bank.identity(data / f"{name}.jsonl")
    bank.write_json(data / "report.json", {"status": "COMPLETE_DATA_NOT_TRAINED", "gates": {"synthetic": True}, "outputs": outputs})
    ledger = root / "ledger.jsonl"
    bank.write_rows(ledger, [{"dataset": "musique", "qid": "protected", "question": "Who was the protected reserve investigator?"}])
    policy = root / "sft"; policy.mkdir()
    (policy / "adapter_model.safetensors").write_bytes(b"synthetic-policy")
    (policy / "adapter_config.json").write_text("{}")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (policy / name).write_text("{}")
    return {"root": root, "data_dir": data, "protected_ledger": ledger, "policy": policy}


@pytest.fixture
def prepared(tmp_path, monkeypatch, sources):
    monkeypatch.setattr(bank, "make_renderer", lambda model: lambda messages: (bank.canonical_json(messages), 100))
    monkeypatch.setattr(bank, "bind_base_model", lambda model, root: {"path": "models/base", "files": {}})
    monkeypatch.setattr(bank, "bind_rearag", lambda model, root: {"path": "models/rearag", "files": {}})
    model = tmp_path / "base"; model.mkdir()
    (model / "config.json").write_text('{"eos_token_id": [2, 3]}')
    (model / "generation_config.json").write_text('{"eos_token_id": [2, 3]}')
    output = tmp_path / "prepared"
    options = {name: sources[name] for name in ("data_dir", "protected_ledger", "policy")}
    bank.prepare_bank(output_dir=output, experiment_id="SYNTHETIC-PREPARE-TEST", base_model=model, **options)
    return output


def test_prepare_all800_balanced_controls_k2_no_gold_parent_immutable(prepared, sources):
    release = bank.load_release(prepared, bank.PREPARE_VERSION)
    assert release["n_questions"] == 830 and release["n_candidates"] == 1660
    assert release["graph_eligible"] == 800
    assert release["by_dataset"] == {"2wikimultihopqa": 810, "hotpotqa": 10, "musique": 10}
    assert release["protected_overlap"] == {"qid": 0, "question_sha256": 0, "family_sha256": 0}
    rows = bank.validate_inputs(prepared, release)
    assert "SECRET GOLD" not in (prepared / "inputs.jsonl").read_text()
    for row in rows:
        assert row["fullsource_record"] == row["spec"]["metadata"]["source_quality_record"] == row["source_quality_record"]
        assert row["source_record_sha256"] == bank.digest(row["source_quality_record"])
    parent = json.loads((sources["data_dir"] / "report.json").read_text())
    assert all(bank.file_sha(sources["data_dir"] / f"{name}.jsonl") == info["sha256"] for name, info in parent["outputs"].items())
    with pytest.raises(FileExistsError):
        bank.prepare_bank(output_dir=prepared, experiment_id="NO-OVERWRITE")


@pytest.mark.parametrize("kind", ["qid", "question", "family"])
def test_global_identity_isolation_even_cross_dataset(kind):
    item = {"dataset": "hotpotqa", "qid": "train", "question": "Where do marmots typically live?"}
    protected = {"dataset": "musique", "qid": "reserve", "question": "What is unrelated?"}
    if kind == "qid": protected["qid"] = item["qid"]
    elif kind == "question": protected["question"] = item["question"]
    else: protected["question"] = "  WHERE do marmots typically live?  "
    with pytest.raises(ValueError, match="isolation"):
        bank.isolation([item], [protected])


def test_token_budget_failure_is_append_only(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(bank, "make_renderer", lambda model: lambda messages: ("long", 6145))
    output = tmp_path / "overflow"
    with pytest.raises(ValueError, match="token budget"):
        bank.prepare_bank(output_dir=output, experiment_id="SYNTHETIC-OVERFLOW", **{name: sources[name] for name in ("data_dir", "protected_ledger", "policy")})
    assert (output / "FAILED.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        bank.prepare_bank(output_dir=output, experiment_id="REFUSED-RETRY")


def test_frozen_input_hash_and_nested_gold_rejected(prepared):
    manifest = bank.load_release(prepared, bank.PREPARE_VERSION)
    rows = bank.read_rows(prepared / "inputs.jsonl")
    rows[0]["source_quality_record"]["provenance"]["gold_answer"] = "secret"
    with pytest.raises(ValueError, match="gold/target"):
        bank.assert_gold_free(rows[0])
    rows[0]["source_quality_record"]["provenance"].pop("gold_answer")
    rows[0]["prompt"] = "tampered"
    (prepared / "inputs.jsonl").write_text("\n".join(bank.canonical_json(row) for row in rows))
    with pytest.raises(ValueError, match="model-input hash"):
        bank.validate_inputs(prepared, manifest)
    with pytest.raises(ValueError, match="artifact hash"):
        bank.load_release(prepared, bank.PREPARE_VERSION)


TRACE = "[Step 1]\nReasoning: Follow the first edge.\nKnowledge Used: [(Alpha, links to, Beta)]\nConclusion: Beta is reached.\n[Step 2]\nReasoning: Follow the second edge.\nKnowledge Used: [(Beta, links to, Gamma)]\nConclusion: Gamma is reached.\n[Final Answer]\nGamma"


class FakeScorer:
    max_length = 4096
    def __init__(self): self.calls = []
    def tokenizer(self, text, **kwargs): return {"input_ids": [0] * len(text.split())}
    def score_step(self, prompt, step):
        self.calls.append((prompt, step))
        return -.25 if len(self.calls) % 2 else .75


def test_scoring_real_proof_features_raw_passage_only_step_mean(prepared):
    row = next(row for row in bank.read_rows(prepared / "inputs.jsonl") if row["m_graph"])
    scorer = FakeScorer()
    result = bank.score_candidate(row, {"candidate_id": "synthetic-k0", "generation": TRACE}, scorer)
    assert result["features"]["m_graph"] == 1
    assert result["proof_result"]["scorer_version"] == bank.SCORER_VERSION
    assert result["raw_graph"] == result["proof_result"]["score"]
    assert result["raw_text"] == [-.25, .75] and result["raw_text_step_mean"] == .25
    assert result["features"]["telemetry"]["policy_entropy_used"] is False
    assert "(Alpha, links to, Beta)" not in scorer.calls[0][0]
    assert "(Alpha, links to, Beta)" in scorer.calls[1][0]  # prior generated step, not KG prompt
    assert "Frozen passage 9" in scorer.calls[0][0]
    assert "SECRET GOLD" not in bank.canonical_json(result)
    scorer.max_length = 1
    with pytest.raises(RuntimeError, match="truncation forbidden"):
        bank.score_candidate(row, {"candidate_id": "synthetic-k1", "generation": TRACE}, scorer)


def test_invalid_candidate_is_retained_and_empty_scores_abstain(prepared):
    row = bank.read_rows(prepared / "inputs.jsonl")[0]
    result = bank.score_candidate(row, {"candidate_id": "synthetic-invalid", "generation": ""}, FakeScorer())
    assert result["raw_text"] == [] and result["raw_text_step_mean"] is None
    assert result["proof_result"]["trajectory_valid"] is False
    assert result["raw_graph"] == -1


def test_model_portability_still_hashes_contents(tmp_path):
    model = tmp_path / "external"; model.mkdir()
    (model / "weights").write_bytes(b"model")
    frozen = {"path": "/different/host", "files": {"weights": bank.identity(model / "weights")}}
    bank.validate_model(model, frozen)
    (model / "weights").write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        bank.validate_model(model, frozen)


def test_gpu_unavailable_records_failure_without_generation(prepared, tmp_path, sources, monkeypatch):
    monkeypatch.setattr(bank, "require_cuda", lambda device: (_ for _ in ()).throw(RuntimeError("CUDA GPU required")))
    output = tmp_path / "gpu-failure"
    with pytest.raises(RuntimeError, match="CUDA"):
        bank.generate_bank(bank_dir=prepared, output_dir=output, experiment_id="SYNTHETIC-NO-CUDA", policy=sources["policy"])
    assert (output / "FAILED.json").exists()
    assert not (output / "generations.jsonl").exists()


def test_mock_generation_score_and_real_calibrator_contract(prepared, tmp_path, sources, monkeypatch):
    import torch
    import transformers
    import peft
    from kgproweight.reward.text_reward_model import RearagPromptScorer
    from scripts.train.calibrate_source_quality_gate_v1 import validate_bank
    calls = []
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs): return bank.canonical_json(messages)
        def __call__(self, prompt, **kwargs):
            assert kwargs["return_attention_mask"] is True
            return {"input_ids": torch.ones((1,100), dtype=torch.long), "attention_mask": torch.ones((1,100), dtype=torch.long)}
        def decode(self, ids, **kwargs): return TRACE
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[2, 3])
        def to(self, device): return self
        def eval(self): calls.append("eval"); return self
        def generate(self, **kwargs):
            assert calls and kwargs["do_sample"] is True
            assert kwargs["eos_token_id"] == [2, 3]
            assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"], kwargs["max_new_tokens"]) == (1., 1., 0, 384)
            assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
            return torch.cat([kwargs["input_ids"], torch.tensor([[4,5]])], dim=1)
    original = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self if args and str(args[0]).startswith("cuda") else original(self,*args,**kwargs))
    monkeypatch.setattr(bank, "require_cuda", lambda device: torch)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: Model())
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda model, *args, **kwargs: model)
    generated = tmp_path / "generated"
    report = bank.generate_bank(bank_dir=prepared, output_dir=generated, experiment_id="SYNTHETIC-GENERATION", policy=sources["policy"], base_model=tmp_path / "portable")
    assert report["n_candidates"] == 1660
    predictions = bank.read_rows(generated / "generations.jsonl")
    assert predictions[0]["seed"] != predictions[1]["seed"]
    assert predictions[0]["generation"] == TRACE and predictions[0]["response_token_ids"] == [4,5]
    monkeypatch.setattr(RearagPromptScorer, "from_pretrained", lambda *args, **kwargs: FakeScorer())
    scored = tmp_path / "scored"
    report = bank.score_bank(bank_dir=prepared, generation_dir=generated, output_dir=scored, experiment_id="SYNTHETIC-SCORE")
    assert report["n_candidates"] == 1660
    # The model above is a mock: explicitly mark this ephemeral fixture synthetic
    # before exercising the real calibrator; never emit a production gate here.
    manifest = json.loads((scored / "manifest.json").read_text())
    manifest.update(bank_source="synthetic", status="SYNTHETIC_TEST_ONLY")
    (scored / "manifest.json").write_text(json.dumps(manifest))
    rows, binding = validate_bank(scored / "manifest.json", scored / "isolation_proof.json", synthetic_test_only=True)
    assert len(rows) == 1660 and binding["bank_source"] == "synthetic"
    assert sum(row["features"]["m_graph"] for row in rows) == 1600


def test_ordinary_valid_format_scores_text_despite_missing_proof_graph(prepared):
    row = next(row for row in bank.read_rows(prepared / "inputs.jsonl") if row["m_graph"] == 0)
    trace = TRACE.replace("[(Alpha, links to, Beta)]", "[]").replace("[(Beta, links to, Gamma)]", "[]")
    trace = trace.replace("[Final Answer]", "[Step 3]\nReasoning: The available passages corroborate the result.\nKnowledge Used: []\nConclusion: The answer is established.\n[Final Answer]")
    result = bank.score_candidate(row, {"candidate_id": "synthetic-ordinary", "generation": trace}, FakeScorer())
    assert result["trajectory_valid"] is True
    assert result["features"]["m_graph"] == 0
    assert len(result["raw_text"]) == 3
    assert result["format_validation"]["required_steps"] == 3


@pytest.mark.parametrize("field", ["candidate_id", "input_sha256", "policy_sha256"])
def test_scoring_rejects_wrong_order_input_or_policy_before_gpu(prepared, tmp_path, field):
    release = bank.load_release(prepared, bank.PREPARE_VERSION)
    rows = bank.read_rows(prepared / "inputs.jsonl")
    bank_sha = bank.file_sha(prepared / "manifest.json")
    predictions = []
    for row in rows:
        for index in range(2):
            predictions.append({"candidate_id": f"{row['question_key']}::k{index}", "dataset": row["dataset"], "qid": row["qid"],
                "candidate_index": index, "seed": bank.candidate_seed(release["seed"], row["question_key"], index),
                "input_sha256": row["input_sha256"], "bank_manifest_sha256": bank_sha,
                "generation_contract_sha256": bank.digest(release["generation"]),
                "policy_sha256": release["source_bindings"]["policy"]["sha256"],
                "base_model_identity_sha256": bank.digest(release["base_model"]), "generation": ""})
    predictions[0][field] = "wrong"
    generated = tmp_path / "wrong-predictions"; generated.mkdir()
    bank.write_rows(generated / "generations.jsonl", predictions)
    bank.finish(generated, {"schema_version": bank.GENERATION_VERSION, "bank_manifest_sha256": bank_sha}, ["generations.jsonl"])
    output = tmp_path / "refused-scoring"
    with pytest.raises(ValueError, match="candidate qid/order/input/model/contract mismatch"):
        bank.score_bank(bank_dir=prepared, generation_dir=generated, output_dir=output, experiment_id="SYNTHETIC-WRONG-PREDICTIONS")
    assert (output / "FAILED.json").exists()
    assert not (output / "candidates.scored.jsonl").exists()
