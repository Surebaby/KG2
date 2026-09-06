from scripts.pilot.audit_query_aware_kg_coverage import (
    _chain_summary,
    _expected_explicit_anchors,
    _infer_question_relation,
    _reference_hops,
    _value_match,
)


def test_temporal_comparison_requires_date_not_place_of_birth():
    target = _infer_question_relation("Who was born earlier, Alice or Bob?")
    assert target["pids"] == ["P569"]
    hop = {"head": "Alice", "tail": "1901", "target": target}
    place_only = [("Alice", "place of birth", "Paris")]
    assert _chain_summary([hop], place_only)["all_relation_value_hit"] is False


def test_2wiki_exact_evidence_builds_multihop_reference_chain():
    row = {
        "metadata": {
            "evidences": {
                "fact": ["Film A", "Director B"],
                "relation": ["director", "country of citizenship"],
                "entity": ["Director B", "France"],
            }
        }
    }
    hops = _reference_hops("2wikimultihopqa", row)
    triples = [
        ("Film A", "director", "Director B"),
        ("Director B", "country of citizenship", "France"),
    ]
    summary = _chain_summary(hops, triples)
    assert summary["evaluable"] is True
    assert summary["all_relation_value_hit"] is True
    assert summary["all_exact_hop_hit"] is True


def test_musique_placeholder_is_resolved_to_prior_answer():
    row = {
        "metadata": {
            "metadata": {
                "question_decomposition": [
                    {"question": "Paper A >> owned by", "answer": "University B"},
                    {"question": "When was #1 founded?", "answer": "1960"},
                ]
            }
        }
    }
    hops = _reference_hops("musique", row)
    assert hops[1]["head"] == "When was University B founded?"
    assert hops[1]["target"]["pids"] == ["P571"]


def test_value_match_handles_nationality_and_date_surface_aliases():
    assert _value_match("United States", "American")
    assert _value_match("1904-11-12", "November 12, 1904")
    assert not _value_match("1904", "1935")


def test_expected_anchor_uses_only_gold_entities_explicit_in_question():
    row = {
        "metadata": {
            "evidences": {
                "fact": ["Film A", "Director B"],
                "relation": ["director", "country of citizenship"],
                "entity": ["Director B", "France"],
            }
        }
    }
    hops = _reference_hops("2wikimultihopqa", row)
    anchors = _expected_explicit_anchors(
        "2wikimultihopqa", row, "Which country is the director of Film A from?", hops
    )
    assert anchors == ["Film A"]
