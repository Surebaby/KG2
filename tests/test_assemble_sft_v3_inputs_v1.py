from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.prepare import assemble_sft_v3_inputs_v1 as a
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.prepare.freeze_sft_v3_candidates_v1 import family_split
from kgproweight.kg.question_kg import question_sha256


def candidate(qid="q1", question="Where was the regional museum founded?"):
    family = family_sha256(question)
    return {"schema_version": "original", "dataset": "2wikimultihopqa", "qid": qid,
            "question_key": "2wikimultihopqa::" + qid, "question": question,
            "question_sha256": question_sha256(question), "family_sha256": family,
            "family_version": FAMILY_VERSION, "split": family_split(family, 42),
            "role": "sft_candidate", "gold_access": False, "within_split_dataset_rank": 1,
            "selection_rank": "old-identity-only-sha", "source": {"path": "raw-train-never-teacher"}}


def graph(row=None):
    row = row or candidate()
    binding = {"synthetic/typed": "a" * 64}
    triple = ["Ada", "born in", "Northland"]
    entity = {"qid": "Q101", "surface": "Ada", "resolved_surface": "Ada"}
    record = {**{k: row[k] for k in a.IDENTITY_FIELDS if k != "family_sha256"}, "kg_subgraph": [triple],
              "execution": {"complete_plan_execution": True, "anchor_entities": {"Ada": entity},
                "hops": [{"input_entities": [entity], "pids": ["P19"], "matches": [triple], "match_sources": ["store"]}]}}
    evidence = {"schema_version": "qid-source-evidence-v1", "gold_used": False, "bindings": binding,
                "entities": {"Q101": {"labels": ["Ada"], "aliases": [], "demonyms": [], "bindings": binding,
                        "typed_edges": [{"head_qid": "Q101", "pid": "P19", "relation": "born in", "head_label": "Ada", "tail_qid": "Q201", "tail_value": "Northland", "source": "store", "bindings": binding}]},
                    "Q201": {"labels": ["Northland"], "aliases": [], "demonyms": [], "bindings": binding, "typed_edges": []}}}
    check = a.validate_source_integrity_v1(record, evidence)
    assert check["status"] == "PASS"
    return {**row, "fullsource_record": record, "source_record_sha256": canonical_sha256(record), "source_check": check}, evidence


def passages():
    return [{"id": str(i), "source": "e5", "contents": f"Evidence passage {i}: supported words."} for i in range(10)]


def test_combination_preserves_original_rows_and_appends_disjoint_supplement():
    original = [candidate()]; before = deepcopy(original)
    attached, _ = graph(original[0]); supplement, _ = graph(candidate("q2", "When did the bridge open to pedestrians?"))
    combined, requests = a.combine_identities(original, [attached], [supplement])
    assert original == before and combined[:1] == original
    assert len(combined) == len(requests) == 2
    assert requests[0] == a.project(original[0])
    assert combined[1]["split"] == family_split(combined[1]["family_sha256"], 42)
    assert all("source" not in r for r in requests)


def test_supplement_cannot_reuse_an_existing_global_family():
    original = [candidate()]; bad, _ = graph(candidate("different", original[0]["question"]))
    with pytest.raises(ValueError, match="family"):
        a.combine_identities(original, [], [bad])


def test_attaching_kg_to_another_question_is_rejected():
    original = [candidate()]; wrong, _ = graph(candidate("elsewhere", "What was the orchestra originally named?"))
    with pytest.raises(ValueError, match="exact"):
        a.combine_identities(original, [wrong], [])


def test_source_pass_is_recomputed_not_trusted_as_a_string():
    row, evidence = graph(); a.validate_graph_assignments([row], evidence, check_bindings=False)
    row["fullsource_record"]["kg_subgraph"][0][2] = "Other place"
    row["source_record_sha256"] = canonical_sha256(row["fullsource_record"])
    with pytest.raises(ValueError, match="source-integrity"):
        a.validate_graph_assignments([row], evidence, check_bindings=False)


def test_source_graph_over_twelve_is_rejected_without_truncation():
    row, evidence = graph(); row["fullsource_record"]["kg_subgraph"] *= 13
    row["source_record_sha256"] = canonical_sha256(row["fullsource_record"])
    with pytest.raises(ValueError, match="max12"):
        a.validate_graph_assignments([row], evidence, check_bindings=False)
    assert len(row["fullsource_record"]["kg_subgraph"]) == 13


def test_ready_join_keeps_exact_evidence_and_leaves_pending_in_original_order():
    original = [candidate(), candidate("q2", "When did the bridge open to pedestrians?")]
    row, _ = graph(original[0]); requests = [a.project(r) for r in original]
    context = {**requests[0], "passages": passages()}
    ready, pending = a.join_ready_inputs(requests, [row], [(context, {"path": "batch.json", "sha256": "a" * 64})], graph_binding={"path": "graphs.jsonl", "sha256": "b" * 64}, evidence_binding={"path": "typed.json", "sha256": "c" * 64})
    assert ready[0]["retrieved_passages"] == context["passages"]
    assert ready[0]["kg_subgraph"] == row["fullsource_record"]["kg_subgraph"]
    assert pending == requests[1:]
    assert "source" not in ready[0] and "golden_answers" not in ready[0]
    assert ready[0]["kg_source_verification"]["source_record_sha256"] == row["source_record_sha256"]


def test_conflicting_retrieval_for_same_question_fails():
    request = a.project(candidate()); first = {**request, "passages": passages()}; second = deepcopy(first)
    second["passages"][0]["contents"] = "different evidence"
    with pytest.raises(ValueError, match="conflicting"):
        a.join_ready_inputs([request], [], [(first, {}), (second, {})], graph_binding={}, evidence_binding={})


def test_test_double_retrieval_is_never_admitted(monkeypatch, tmp_path):
    monkeypatch.setattr(a.retrieval, "verify", lambda p: ({"test_double_only": True}, []))
    with pytest.raises(ValueError, match="real"):
        a.load_real_retrieval_batches(tmp_path)


def test_supplement_checker_labels_are_original_train_values(tmp_path):
    q = candidate(); raw = tmp_path / "train.jsonl"
    raw.write_text(json.dumps({"id": q["qid"], "question": q["question"], "golden_answers": ["A", "Alias, B"], "metadata": {"supporting_facts": "never copied"}}) + "\n")
    labels = a.labels_for_supplement(raw, a.bind(raw), [q])
    assert labels[0]["golden_answers"] == ["A", "Alias, B"]
    assert "metadata" not in labels[0] and "supporting_facts" not in labels[0]
    assert labels[0]["source"]["line_number"] == 1


def test_mismatched_raw_train_question_never_supplies_labels(tmp_path):
    q = candidate(); raw = tmp_path / "train.jsonl"
    raw.write_text(json.dumps({"id": q["qid"], "question": "A different question", "golden_answers": ["A"]}) + "\n")
    with pytest.raises(ValueError, match="identity"):
        a.labels_for_supplement(raw, a.bind(raw), [q])


def test_raw_outer_whitespace_is_audited_without_changing_original_answers(tmp_path):
    q = candidate(); raw = tmp_path / "train.jsonl"
    raw.write_text(json.dumps({"id": q["qid"], "question": q["question"] + " ", "golden_answers": [" Original answer "]}) + "\n")
    labels = a.labels_for_supplement(raw, a.bind(raw), [q])
    assert labels[0]["golden_answers"] == [" Original answer "]
    assert labels[0]["source_question_outer_whitespace_normalized"] is True
