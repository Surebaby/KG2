import json

import pytest

from kgproweight.kg.passage_sro import parse_extraction_json, validate_extracted_edges


PASSAGES = [{"id": "p1", "contents": "Ada Lovelace\nAda Lovelace was born in London in 1815."}]


def _edge(**updates):
    value = {
        "head": "Ada Lovelace",
        "relation": "birth place",
        "tail": "London",
        "passage_rank": 1,
        "evidence_quote": "Ada Lovelace was born in London in 1815.",
        "relation_trigger": "born in",
    }
    value.update(updates)
    return value


def test_parse_json_object_and_fence():
    assert parse_extraction_json(json.dumps({"edges": [_edge()]}))[0]["tail"] == "London"
    assert parse_extraction_json("```json\n{\"edges\": []}\n```") == []


def test_accepts_exact_span_with_canonical_relation():
    accepted, rejected = validate_extracted_edges([_edge()], PASSAGES)
    assert rejected == []
    assert accepted[0]["passage_id"] == "p1"
    assert accepted[0]["relation"] == "birth place"


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"relation": "friend"}, "relation_not_in_frozen_vocabulary"),
        ({"tail": "Oxford"}, "tail_not_in_quote"),
        ({"relation_trigger": "educated at"}, "relation_trigger_not_in_quote"),
        ({"evidence_quote": "Ada was a mathematician."}, "quote_not_in_passage"),
        ({"passage_rank": 2}, "invalid_passage_rank"),
    ],
)
def test_fail_closed_rejections(updates, reason):
    accepted, rejected = validate_extracted_edges([_edge(**updates)], PASSAGES)
    assert accepted == []
    assert rejected[0]["reason"] == reason


def test_title_head_allowed_for_pronoun_quote():
    accepted, _ = validate_extracted_edges(
        [_edge(evidence_quote="She was born in London in 1815.")],
        [{"id": "p1", "contents": "Ada Lovelace\nShe was born in London in 1815."}],
    )
    assert len(accepted) == 1


def test_deduplicates_and_caps_deterministically():
    edges = [_edge(), _edge(), _edge(tail="1815", relation="birth date", relation_trigger="in 1815")]
    accepted, rejected = validate_extracted_edges(edges, PASSAGES)
    assert len(accepted) == 2
    assert any(row["reason"] == "duplicate_triple" for row in rejected)
