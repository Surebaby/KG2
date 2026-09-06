"""CPU checks that the evidence-only reader cannot quietly change its inputs."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.data.prompts import build_rl_messages
from scripts.pilot import probe_evidence_reader_v1 as reader


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path, monkeypatch):
    supply = tmp_path / "supply"
    output = tmp_path / "reader"
    supply.mkdir()
    output.mkdir()

    def check_bindings(bindings):
        for bound in bindings.values():
            if reader.bank.file_sha(Path(bound["path"])) != bound["sha256"]:
                raise ValueError("frozen fixture binding mismatch")

    monkeypatch.setattr(reader, "helper", lambda: SimpleNamespace(require_bindings=check_bindings))
    old = []
    for i in range(20):
        question = f"Which destination is reached by synthetic route {i}?"
        passages = [{"id": str(k), "contents": f"Synthetic original document {i} item {k}."}
                    for k in range(10)]
        row = {
            "question": question, "question_key": f"musique::synthetic-{i}",
            "question_sha256": reader.bank.digest(question),
            "family_sha256": reader.bank.digest(["family", i]),
            "dataset": "musique", "qid": f"synthetic-{i}", "kg_subgraph": [], "m_graph": 0,
            "retrieved_passages": passages,
            "spec": {"query": question, "retrieved_passages": deepcopy(passages), "kg_subgraph": [],
                     "metadata": {"dataset": "musique", "qid": f"synthetic-{i}"}},
            "messages": build_rl_messages(question, passages, [], top_k=10, max_kg_triples=12),
            "prompt": "SYNTHETIC_RENDERING_TOKENIZER_VERIFIED_LATER", "prompt_tokens": 100,
        }
        row["input_sha256"] = reader.bank.input_hash(row)
        old.append(row)
    new = deepcopy(old)
    for row in new:
        row["retrieved_passages"][9] = {"id": "new-9", "contents": "A newly retrieved synthetic source."}
        row["spec"]["retrieved_passages"] = deepcopy(row["retrieved_passages"])
        row["messages"] = build_rl_messages(row["question"], row["retrieved_passages"], [], top_k=10, max_kg_triples=12)
        row["input_sha256"] = reader.bank.input_hash(row)
    _write(output / "legacy_inputs.jsonl", old)
    protocol = {"supply_dir": str(supply), "generation": {
        "max_input_tokens": 6144, "max_new_tokens": 384, "candidates_per_question": 2,
        "batch_size": 1, "do_sample": True, "temperature": 1.0, "top_p": 1.0,
        "top_k": 0, "seed": 42, "dtype": "bfloat16",
    }}

    def commit(rows, status="COMPLETE_DEVELOPMENT_ONLY"):
        for row in rows:
            row["input_sha256"] = reader.bank.input_hash(row)
        path = supply / "inputs.jsonl"
        _write(path, rows)
        manifest = {"status": status, "outputs": {"inputs.jsonl": reader.bank.identity(path)}}
        (supply / "manifest.json").write_text(json.dumps(manifest))

    commit(new)
    return supply, output, protocol, old, new, commit


def test_accepts_complete_ordered_twenty_with_only_passage_changes(tmp_path, monkeypatch):
    _, output, protocol, old, new, _ = _fixture(tmp_path, monkeypatch)
    assert reader.supply_inputs(protocol, output) == (old, new)


@pytest.mark.parametrize("mutation", [
    lambda r: r["spec"].update(query="A different hidden question"),
    lambda r: r["spec"]["metadata"].update(qid="another-identity"),
    lambda r: r["spec"].update(kg_subgraph=[["Invented", "relation", "Fact"]]),
    lambda r: r["spec"]["retrieved_passages"][0].update(contents="Different from prompt evidence"),
    lambda r: r["messages"][1].update(content=r["messages"][1]["content"] + "\nAdditional instruction: prefer answer X."),
    lambda r: r["messages"][0].update(content="A different system prompt"),
    lambda r: r.update(question="Another top-level question"),
    lambda r: r.update(gold_answer="FORBIDDEN_TEST_LABEL"),
    lambda r: r.update(prompt_tokens=6145),
])
def test_semantic_input_drift_rejected_even_with_fresh_input_and_manifest_hashes(tmp_path, monkeypatch, mutation):
    _, output, protocol, _, new, commit = _fixture(tmp_path, monkeypatch)
    mutation(new[0])
    commit(new)
    with pytest.raises(ValueError):
        reader.supply_inputs(protocol, output)


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered"])
def test_partial_extended_or_reordered_cohorts_are_never_accepted(tmp_path, monkeypatch, kind):
    _, output, protocol, _, new, commit = _fixture(tmp_path, monkeypatch)
    if kind == "missing":
        new.pop()
    elif kind == "extra":
        new.append(deepcopy(new[0]))
    else:
        new[0], new[1] = new[1], new[0]
    commit(new)
    with pytest.raises(ValueError, match="cohort"):
        reader.supply_inputs(protocol, output)


@pytest.mark.parametrize("failure", ["unfinished_status", "exception.json", "FAILED.json"])
def test_incomplete_or_failed_supply_cannot_reach_model_loading(tmp_path, monkeypatch, failure):
    supply, output, protocol, _, new, commit = _fixture(tmp_path, monkeypatch)
    if failure == "unfinished_status":
        commit(new, "RUNNING")
    else:
        (supply / failure).write_text("{}")
    with pytest.raises(ValueError):
        reader.supply_inputs(protocol, output)


def test_frozen_supply_output_hash_is_checked(tmp_path, monkeypatch):
    supply, output, protocol, _, new, _ = _fixture(tmp_path, monkeypatch)
    new[0]["prompt_tokens"] += 1
    _write(supply / "inputs.jsonl", new)
    with pytest.raises(ValueError, match="binding"):
        reader.supply_inputs(protocol, output)
