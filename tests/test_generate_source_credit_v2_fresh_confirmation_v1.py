"""CPU-only synthetic generation contracts; never consume fresh132 outputs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.prepare import generate_source_credit_v2_fresh_confirmation_v1 as gen


@pytest.fixture
def context():
    rows = [{"dataset": "hotpotqa", "qid": f"synthetic-{i}", "question_key": f"hotpotqa::synthetic-{i}",
             "question_sha256": str(i), "family_sha256": f"family-{i}", "input_sha256": f"input-{i}",
             "messages": [{"role": "user", "content": "synthetic"}], "prompt": "fixed prompt", "prompt_tokens": 2}
            for i in range(132)]
    return {"protocol": {"schema_version": gen.PROTOCOL_SCHEMA, "status": "FROZEN", "experiment_id": "SYNTHETIC-TEST",
                         "seed": 42, "generation": {**deepcopy(gen.GENERATION_FIXED), "device": "cuda:0"},
                         "bindings": {"inputs_manifest": {"sha256": "scope-test"}}},
            "protocol_sha256": "protocol-test", "inputs": rows,
            "input_manifest": {"source_bindings": {"policy": {"sha256": "policy-test"}},
                               "models": {"base_model": {"path": "base", "files": {}}}},
            "policy_path": Path("synthetic-policy"), "base_model_path": Path("synthetic-base")}


def seal(value):
    value.pop("candidate_sha256", None)
    value["candidate_sha256"] = gen.bank.digest(value)
    return value


def prediction(context, position, raw=None):
    raw = raw or [10, 128009]
    trimmed = gen.bank._trim_response_v2(torch.tensor(raw), eos_token_ids=[128001, 128009], pad_token_id=0, max_new_tokens=384)
    ids = trimmed.tolist()
    return seal({**gen.expected_identity(context, context["inputs"][position // 5], position % 5),
                 "generation": FakeTokenizer().decode(ids, skip_special_tokens=True), "response_token_ids": ids,
                 "raw_response_token_ids": raw, "n_response_tokens": len(ids), "pad_token_id": 0,
                 "reached_max_new_tokens": gen.bank._response_is_length_capped_v2(trimmed, max_new_tokens=384,
                                                                               eos_token_ids=[128001, 128009])})


class FakeInput:
    shape = (1, 2)
    def __init__(self, values):
        self.values = values
        self.device = None
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "eos"
    def __init__(self):
        self.calls = []
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("chat", kwargs))
        return "fixed prompt"
    def __call__(self, prompt, **kwargs):
        self.calls.append(("encode", kwargs))
        return {"input_ids": FakeInput([8, 9]), "attention_mask": FakeInput([1, 1])}
    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i not in (128001, 128009, 0))


class FakeTorch:
    def __init__(self):
        self.seeds = []
    def manual_seed(self, seed):
        self.seeds.append(seed)
    inference_mode = staticmethod(torch.inference_mode)


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[128001, 128009])
    config = generation_config
    def __init__(self, raw=None):
        self.calls = []
        self.raw = raw or [10, 128009]
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[8, 9] + self.raw])


def test_exact_frozen_contract_and_slot_seed_identity(context):
    gen.validate_generation_contract(context["protocol"])
    slots = [gen.expected_identity(context, context["inputs"][0], i) for i in range(5)]
    assert [row["generation_kind"] for row in slots] == ["sampled"] * 4 + ["greedy"]
    assert [row["candidate_index"] for row in slots] == list(range(5))
    assert len({row["candidate_id"] for row in slots}) == 5
    assert [row["seed"] for row in slots] == [gen.bank.candidate_seed(42, "hotpotqa::synthetic-0", i) for i in range(5)]


@pytest.mark.parametrize("key,value", [("candidates_per_question", 2), ("max_new_tokens", 512),
    ("temperature", .9), ("top_k", 50), ("batch_size", 4), ("dtype", "float16"),
    ("eos_token_ids", [128009]), ("greedy_candidate_index", 0), ("device", "cpu"), ("do_sample", 1)])
def test_generation_mutations_fail_closed(context, key, value):
    context["protocol"]["generation"][key] = value
    with pytest.raises(ValueError):
        gen.validate_generation_contract(context["protocol"])


def test_sampling_call_matches_parent_exactly_and_greedy_separate(context):
    tokenizer, model, backend = FakeTokenizer(), FakeModel(), FakeTorch()
    source = context["inputs"][0]
    encoded, count = gen.encode_frozen_prompt(source, tokenizer)
    sample = gen.generate_one(model, tokenizer, backend, context, source, 0, encoded, count)
    greedy = gen.generate_one(model, tokenizer, backend, context, source, 4, encoded, count)
    sampling = model.calls[0]
    assert sampling == {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
        "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 384,
        "pad_token_id": 0, "eos_token_id": [128001, 128009]}
    assert model.calls[1] == {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
        "do_sample": False, "max_new_tokens": 384, "pad_token_id": 0, "eos_token_id": [128001, 128009]}
    assert all(encoded[key].device == "cuda:0" for key in encoded)
    assert backend.seeds == [gen.bank.candidate_seed(42, source["question_key"], slot) for slot in (0, 4)]
    assert sample["generation_kind"] == "sampled" and greedy["generation_kind"] == "greedy"
    gen.verify_generation_row(sample, context, 0, tokenizer=tokenizer)
    gen.verify_generation_row(greedy, context, 4, tokenizer=tokenizer)
    assert tokenizer.calls == [("chat", {"tokenize": False, "add_generation_prompt": True}),
        ("encode", {"add_special_tokens": False, "return_tensors": "pt", "truncation": False, "return_attention_mask": True})]


def test_chat_prompt_and_token_count_cannot_change(context):
    row = deepcopy(context["inputs"][0])
    row["prompt"] = "modified"
    with pytest.raises(ValueError, match="prompt/token"):
        gen.encode_frozen_prompt(row, FakeTokenizer())
    row["prompt"], row["prompt_tokens"] = "fixed prompt", 3
    with pytest.raises(ValueError, match="prompt/token"):
        gen.encode_frozen_prompt(row, FakeTokenizer())


def test_complete_population_partial_prefix_and_no_duplicate_or_reorder(context):
    rows = [prediction(context, i) for i in range(660)]
    gen.verify_generation_rows(rows, context, tokenizer=FakeTokenizer())
    gen.verify_generation_rows(rows[:13], context, allow_partial=True)
    with pytest.raises(ValueError, match="count"):
        gen.verify_generation_rows(rows[:-1], context)
    with pytest.raises(ValueError, match="identity"):
        gen.verify_generation_rows([rows[1], rows[0]] + rows[2:], context)
    with pytest.raises(ValueError, match="count"):
        gen.verify_generation_rows(rows + [rows[-1]], context, allow_partial=True)


@pytest.mark.parametrize("field,value", [("seed", 10), ("input_sha256", "changed"),
    ("policy_sha256", "changed"), ("protocol_sha256", "changed"), ("generation_kind", "greedy"),
    ("candidate_index", 4), ("candidate_id", "other::k0")])
def test_even_resealed_identity_tampering_rejected(context, field, value):
    row = prediction(context, 0)
    row[field] = value
    seal(row)
    with pytest.raises(ValueError, match="identity"):
        gen.verify_generation_rows([row], context, allow_partial=True)


def test_raw_eos_padding_cap_and_decode_are_rechecked(context):
    padded = prediction(context, 0, [10, 128009, 0, 0])
    gen.verify_generation_rows([padded], context, allow_partial=True, tokenizer=FakeTokenizer())
    assert padded["response_token_ids"] == [10, 128009]
    capped = prediction(context, 0, [10] * 384)
    assert capped["reached_max_new_tokens"] is True
    gen.verify_generation_rows([capped], context, allow_partial=True)
    last_eos = prediction(context, 0, [10] * 383 + [128009])
    assert last_eos["reached_max_new_tokens"] is False
    bad = deepcopy(padded); bad["raw_response_token_ids"][-1] = 11; seal(bad)
    with pytest.raises(ValueError, match="non-padding"):
        gen.verify_generation_rows([bad], context, allow_partial=True)
    bad = deepcopy(padded); bad["generation"] = "forged answer"; seal(bad)
    with pytest.raises(ValueError, match="decode"):
        gen.verify_generation_rows([bad], context, allow_partial=True, tokenizer=FakeTokenizer())
    bad = deepcopy(padded); bad["reached_max_new_tokens"] = True; seal(bad)
    with pytest.raises(ValueError, match="cap"):
        gen.verify_generation_rows([bad], context, allow_partial=True)


def test_candidate_hash_and_nested_gold_are_rejected(context):
    row = prediction(context, 0)
    row["generation"] = "changed"
    with pytest.raises(ValueError, match="payload hash"):
        gen.verify_generation_rows([row], context, allow_partial=True)
    row = prediction(context, 0)
    row["hidden"] = {"gold_answer": "synthetic gold"}; seal(row)
    with pytest.raises(ValueError, match="gold/target"):
        gen.verify_generation_rows([row], context, allow_partial=True)


def test_atomic_commit_retains_interrupted_attempt_and_never_overwrites(tmp_path, context):
    candidates = tmp_path / "candidates"; candidates.mkdir()
    attempts = tmp_path / "attempts"; attempts.mkdir()
    interrupted = attempts / "uncommitted.attempt"; interrupted.write_bytes(b'{"partial":')
    path = candidates / "00000000.json"
    row = prediction(context, 0)
    gen.publish_json(path, row, attempts)
    original = path.read_bytes()
    assert gen.read_committed(tmp_path, context) == [row]
    with pytest.raises(FileExistsError):
        gen.publish_json(path, prediction(context, 1), attempts)
    assert path.read_bytes() == original and interrupted.read_bytes() == b'{"partial":'
    assert len(list(attempts.glob("*.attempt"))) == 3
    gen.publish_json(candidates / "00000002.json", prediction(context, 2), attempts)
    with pytest.raises(ValueError, match="prefix"):
        gen.read_committed(tmp_path, context)


def test_resume_refuses_corrupt_prefix_before_cuda_and_preserves_bytes(tmp_path, context, monkeypatch):
    output = tmp_path / "output"; output.mkdir()
    (output / "candidates").mkdir()
    frozen = {"schema_version": gen.GENERATION_SCHEMA, "experiment_id": "SYNTHETIC-TEST",
              "protocol_sha256": "protocol-test", "expected_candidates": 660, "gold_access": False, "optimizer_updates": 0}
    (output / "started.json").write_text(json.dumps(frozen))
    row = prediction(context, 0); row["seed"] = -1
    original = json.dumps(row).encode()
    (output / "candidates/00000000.json").write_bytes(original)
    monkeypatch.setattr(gen, "verify_protocol", lambda *args, **kwargs: context)
    monkeypatch.setattr(gen.bank, "require_cuda", lambda *args: pytest.fail("must not request CUDA for corrupt resume"))
    with pytest.raises(ValueError, match="identity"):
        gen.run(protocol=tmp_path / "unused.json", out=output, resume=True)
    assert (output / "candidates/00000000.json").read_bytes() == original


def test_original_authority_replacement_fails_before_reading_any_inputs(tmp_path, context):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(context["protocol"]))
    with pytest.raises(ValueError, match="authority changed"):
        gen.verify_protocol(path)


def test_file_binding_detects_content_or_size_changes(tmp_path):
    path = tmp_path / "source"; path.write_bytes(b"retained scientific data")
    binding = gen.bank.identity(path)
    assert gen.resolve_binding(binding) == path
    with pytest.raises(ValueError, match="size"):
        gen.resolve_binding({**binding, "bytes": 1})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        gen.resolve_binding(binding)


def test_existing_scientific_directory_requires_explicit_resume(tmp_path):
    output = tmp_path / "existing"; output.mkdir()
    retained = output / "old.txt"; retained.write_text("keep")
    with pytest.raises(FileExistsError):
        with gen.output_lock(output, resume=False):
            pass
    with pytest.raises(ValueError, match="started"):
        with gen.output_lock(output, resume=True):
            pass
    assert retained.read_text() == "keep"


@pytest.fixture
def synthetic_protocol_files(tmp_path, context, monkeypatch):
    """An entirely synthetic authority, including tiny stand-in weight files."""
    policy = tmp_path / "policy"; policy.mkdir()
    base = tmp_path / "base"; base.mkdir()
    for name, data in (("adapter_model.safetensors", b"synthetic-adapter"), ("adapter_config.json", b"{}"),
                       ("tokenizer.json", b"synthetic-tokenizer")):
        (policy / name).write_bytes(data)
    for name, data in (("config.json", b'{"eos_token_id":[128001,128009]}'),
                       ("generation_config.json", b'{"eos_token_id":[128001,128009]}'),
                       ("synthetic.safetensors", b"synthetic-weight")):
        (base / name).write_bytes(data)
    rows = deepcopy(context["inputs"])
    for row in rows:
        row.update(question="Synthetic question", retrieved_passages=[{"contents": "Synthetic evidence"}] * 10,
                   kg_subgraph=[], source_quality_record={}, fullsource_record={}, source_record_sha256=gen.bank.digest({}),
                   spec={"metadata": {"source_quality_record": {}}})
        row["input_sha256"] = gen.bank.input_hash(row)
    inputs = tmp_path / "inputs.jsonl"
    gen.bank.write_rows(inputs, rows)
    manifest = {"schema_version": "source-credit-v2-fresh-confirmation-inputs-v1", "questions": 132,
                "unique_families": 132, "gold_value_access": False,
                "outputs": {"inputs.jsonl": gen.bank.identity(inputs)},
                "source_bindings": {"policy": gen.bank.identity(policy / "adapter_model.safetensors"),
                                    "policy_config": gen.bank.identity(policy / "adapter_config.json")},
                "policy_path": str(policy), "models": {
                    "policy_tokenizer": {"path": str(policy), "files": {"tokenizer.json": gen.bank.identity(policy / "tokenizer.json")}},
                    "base_model": {"path": str(base), "files": {path.name: gen.bank.identity(path) for path in base.iterdir()}}}}
    manifest_path = tmp_path / "manifest.scope_v2.json"; gen.bank.write_json(manifest_path, manifest)
    source_path = tmp_path / "source.json"
    gen.bank.write_json(source_path, {"schema_version": "source-credit-v2-fresh-confirmation-source-preparation-v1",
                        "gold_access": False, "optimizer_updates": 0, "source_bindings": {}, "outputs": {}})
    protocol = deepcopy(context["protocol"])
    protocol["bindings"] = {"inputs": gen.bank.identity(inputs), "inputs_manifest": gen.bank.identity(manifest_path),
                            "source_manifest": gen.bank.identity(source_path)}
    protocol["code_bindings"] = {}
    protocol_path = tmp_path / "protocol.json"; gen.bank.write_json(protocol_path, protocol)
    monkeypatch.setattr(gen, "INPUTS_SHA256", gen.bank.file_sha(inputs))
    monkeypatch.setattr(gen, "INPUT_MANIFEST_SHA256", gen.bank.file_sha(manifest_path))
    monkeypatch.setattr(gen, "SOURCE_MANIFEST_SHA256", gen.bank.file_sha(source_path))
    monkeypatch.setattr(gen, "GENERATION_CODE_FILES", [])
    return {"protocol": protocol_path, "base": base, "policy": policy}


def test_entire_protocol_checks_models_and_input_hashes_on_cpu(synthetic_protocol_files):
    paths = synthetic_protocol_files
    verified = gen.verify_protocol(paths["protocol"])
    assert len(verified["inputs"]) == 132
    assert verified["base_model_path"] == paths["base"]
    assert verified["policy_path"] == paths["policy"]
    (paths["base"] / "synthetic.safetensors").write_bytes(b"changed weight")
    with pytest.raises(ValueError, match="model/tokenizer hash"):
        gen.verify_protocol(paths["protocol"])
    # Scoring explicitly verifies its own model and need not rehash the 8B.
    gen.verify_protocol(paths["protocol"], verify_models=False)
    (paths["base"] / "generation_config.json").write_text('{"eos_token_id":[128009]}')
    with pytest.raises(ValueError, match="generation config"):
        gen.verify_protocol(paths["protocol"], verify_models=False)


def test_code_dependency_cannot_be_omitted_or_replaced(synthetic_protocol_files, monkeypatch):
    path = synthetic_protocol_files["protocol"]
    monkeypatch.setattr(gen, "GENERATION_CODE_FILES", ["scripts/prepare/generate_source_credit_v2_fresh_confirmation_v1.py"])
    with pytest.raises(ValueError, match="dependency code binding"):
        gen.verify_protocol(path)
    value = json.loads(path.read_text())
    name = gen.GENERATION_CODE_FILES[0]
    value["code_bindings"][name] = gen.bank.identity(gen.ROOT / name)
    path.write_text(json.dumps(value))
    gen.verify_protocol(path)
    value["code_bindings"][name] = gen.bank.identity(synthetic_protocol_files["policy"] / "tokenizer.json")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="executed repository file"):
        gen.verify_protocol(path)


def test_clean_resume_reuses_completed_prefix_and_same_next_seed(tmp_path, context, monkeypatch):
    """Stop after slot0, resume slot1, then stop again; no existing output changes."""
    import transformers
    import peft
    backend = FakeTorch()
    backend.bfloat16 = torch.bfloat16
    backend.cuda = SimpleNamespace(reset_peak_memory_stats=lambda device: None)
    tokenizer = FakeTokenizer()
    model = FakeModel()
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setattr(gen, "verify_protocol", lambda *args, **kwargs: context)
    monkeypatch.setattr(gen.bank, "require_cuda", lambda *args: backend)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *args, **kwargs: model)
    original_generate = model.generate
    calls = []
    def interrupted_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) in (2, 4):
            raise KeyboardInterrupt("synthetic interruption")
        return original_generate(**kwargs)
    model.generate = interrupted_generate
    output = tmp_path / "generated"
    with pytest.raises(KeyboardInterrupt):
        gen.run(protocol=tmp_path / "unused.json", out=output)
    first = (output / "candidates/00000000.json").read_bytes()
    assert len(gen.read_committed(output, context)) == 1
    with pytest.raises(KeyboardInterrupt):
        gen.run(protocol=tmp_path / "unused.json", out=output, resume=True)
    assert (output / "candidates/00000000.json").read_bytes() == first
    assert len(gen.read_committed(output, context)) == 2
    assert len(list(output.glob("attempt_*.json"))) == 2
    question_key = context["inputs"][0]["question_key"]
    assert backend.seeds == [gen.bank.candidate_seed(42, question_key, i) for i in (0, 1, 1, 2)]
    assert not (output / "manifest.json").exists()
