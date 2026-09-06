from copy import deepcopy
import json

import pytest

from scripts.pilot import probe_evidence_supply_v1 as pilot


def doc(i, text=None):
    return {"id": str(i), "contents": text or f"Visible source number {i} states a distinct fact."}


def original():
    return [doc(i) for i in range(10)]


def test_entity_surface_offsets_are_visible_and_sentence_boundaries_respected():
    text = "The   Ada Lovelace moved to Paris. London was elsewhere. She met Charles Babbage."
    found = pilot.entity_mentions(text)
    names = {v["entity"] for v in found}
    assert {"Ada Lovelace", "Paris", "London", "Charles Babbage"} <= names
    assert "Paris. London" not in names
    for v in found:
        assert text[v["offset"]:v["offset"] + len(v["entity"])] == v["entity"]


def test_query_plan_only_original_top_three_and_novel_entities():
    passages = original()
    passages[0]["contents"] = "Ada Lovelace went to Paris and London."
    passages[1]["contents"] = "Paris had Ada Lovelace and Charles Babbage."
    passages[2]["contents"] = "Paris welcomed Mary Somerville."
    passages[3]["contents"] = "Forbidden City EvidenceOnlyInRankFour."
    plan = pilot.build_query_plan("Where did Ada Lovelace live?", passages)
    assert plan["queries"][0]["entity"] == "Paris"
    assert plan["queries"][0]["passage_frequency"] == 3
    assert len(plan["queries"]) == 3
    assert all(q["query"].startswith("Where did Ada Lovelace live? ") for q in plan["queries"])
    assert all("Ada Lovelace" != q["entity"] for q in plan["queries"])
    assert not any("Forbidden" in q["entity"] for q in plan["ranked_entity_candidates"])


def test_already_mentioned_entity_extended_surface_is_not_bridge():
    passages = original()
    passages[0]["contents"] = "Augusta Ada Lovelace worked with Charles Babbage."
    plan = pilot.build_query_plan("Where did Ada Lovelace live?", passages)
    assert "Augusta Ada Lovelace" not in [v["entity"] for v in plan["queries"]]
    assert "Charles Babbage" in [v["entity"] for v in plan["queries"]]


def test_no_entity_keeps_empty_plan_without_question_drop():
    passages = [doc(i, "all lowercase evidence with no names.") for i in range(10)]
    plan = pilot.build_query_plan("what happened?", passages)
    assert plan["queries"] == []
    assert plan["upstream_hybrid_queries"] == 0


def test_query_plans_are_deterministic_and_do_not_mutate_allowed_inputs():
    passages = original()
    before = deepcopy(passages)
    first = pilot.build_query_plan("Who was there?", passages)
    assert first == pilot.build_query_plan("Who was there?", passages)
    assert passages == before


@pytest.mark.parametrize("field", ["gold_answer", "answers", "supporting_facts", "semantic_annotation"])
def test_reject_gold_support_or_semantic_annotations_recursively(field):
    with pytest.raises(ValueError, match="forbidden"):
        pilot.assert_gold_free({"passages": [{"metadata": {field: "do not read"}}]})


def test_selection_retains_four_then_cycles_routes_and_records_origins():
    old = original()
    routes = [[doc(100 + 10*i + j) for j in range(5)] for i in range(3)]
    result, trace = pilot.select_passages(old, routes)
    assert result[:4] == old[:4]
    assert [d["id"] for d in result[4:]] == ["100", "110", "120", "101", "111", "121"]
    assert [r["origin"] for r in trace[4:]] == [f"expanded_query_{i}" for i in (0, 1, 2, 0, 1, 2)]
    assert len({d["id"] for d in result}) == 10


def test_deduplicate_equal_content_different_id_and_original_content_hits():
    old = original()
    old[1] = doc(1, old[0]["contents"].upper())
    routes = [[doc(100, old[6]["contents"]), doc(101, "New unique passage one."),
               doc(102, "NEW UNIQUE PASSAGE ONE."), *[doc(i) for i in range(103, 110)]]]
    result, trace = pilot.select_passages(old, routes)
    assert [d["id"] for d in result[:4]] == ["0", "2", "3", "4"]
    assert "100" not in {d["id"] for d in result}
    assert "102" not in {d["id"] for d in result}
    assert len({pilot.normalize(pilot.passage_text(d)) for d in result}) == 10
    assert [r["origin_rank"] for r in trace[:4]] == [1, 3, 4, 5]


def test_no_new_documents_backfill_existing_context_and_retain_question():
    old = original()
    result, trace = pilot.select_passages(old, [old, []])
    assert result == old
    assert len(result) == 10
    assert sum(r["origin"] == "legacy_backfill" for r in trace) == 6


def test_insufficient_distinct_content_fails_without_padding_or_mutation():
    old = [doc(i, "same content") for i in range(10)]
    before = deepcopy(old)
    with pytest.raises(ValueError, match="could not retain"):
        pilot.select_passages(old, [])
    assert old == before


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def __call__(self, prompt, **kwargs):
        return {"input_ids": list(range(len(prompt.split())))}


def test_rebind_changes_only_evidence_dependent_fields():
    from kgproweight.data.prompts import build_rl_messages
    old_passages = original()
    old = {"dataset": "musique", "qid": "fixture", "question": "Who was there?", "m_graph": 0, "kg_subgraph": [],
           "retrieved_passages": old_passages,
           "spec": {"query": "Who was there?", "kg_subgraph": [], "retrieved_passages": old_passages,
                    "metadata": {"qid": "fixture", "source_quality_record": {"process_reward_eligible": False}}},
           "messages": build_rl_messages("Who was there?", old_passages, [], top_k=10, max_kg_triples=12),
           "prompt": "old prompt", "prompt_tokens": 2, "input_sha256": "old", "family_sha256": "frozen-family"}
    before = deepcopy(old)
    new_passages = [doc(100+i) for i in range(10)]
    new = pilot.rebind_input(old, new_passages, Tokenizer())
    assert old == before
    assert new["messages"][0] == old["messages"][0]
    assert new["spec"]["metadata"] == old["spec"]["metadata"]
    assert new["family_sha256"] == old["family_sha256"]
    assert new["spec"]["retrieved_passages"] == new["retrieved_passages"] == new_passages
    assert new["input_sha256"] != old["input_sha256"]


def test_asset_stat_drift_rejected(tmp_path):
    path = tmp_path / "index.dat"
    path.write_bytes(b"abc")
    bound = {**pilot.identity(path), "stat_signature": pilot.stat_signature(path)}
    pilot.require_asset_stats({"index": bound})
    path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="stat drift"):
        pilot.require_asset_stats({"index": bound})
