from scripts.prepare.build_saeg_p_alignment_v2_candidates import (
    build_aligned_steps,
    quality_class,
    required_support_units,
)
from scripts.diagnose.audit_saeg_p_alignment_v2_near_exact import (
    canonical_sentence_tokens,
    near_exact_match,
    token_f1,
)


def test_quality_class_four_operational_cases():
    assert quality_class(2, 2, 2) == "complete"
    assert quality_class(2, 2, 1) == "partial"
    assert quality_class(2, 2, 0) == "misleading"
    assert quality_class(2, 0, 0) == "empty"
    assert quality_class(1, 1, 1) == "unresolved_gold"


def test_hotpot_required_units_use_exact_support_sentences():
    raw = {
        "metadata": {
            "context": {
                "title": ["Alpha", "Beta"],
                "sentences": [["Noise.", "Alpha supports hop one."], ["Beta supports hop two."]],
            },
            "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [1, 0]},
        }
    }
    units = required_support_units("hotpotqa", raw)
    assert [(row["title"], row["sentence"]) for row in units] == [
        ("Alpha", "Alpha supports hop one."),
        ("Beta", "Beta supports hop two."),
    ]


def test_aligned_target_cites_only_visible_matching_support():
    source = [{
        "index": 1,
        "text": (
            "Reasoning: use the support.\n"
            "Knowledge Used: [(Alpha, evidence sentence, Correct support.)]\n"
            "Conclusion: bridge"
        ),
        "cited_triples": [["Alpha", "evidence sentence", "Correct support."]],
    }, {
        "index": 2,
        "text": (
            "Reasoning: missing hop.\n"
            "Knowledge Used: [(Beta, evidence sentence, Missing support.)]\n"
            "Conclusion: answer"
        ),
        "cited_triples": [["Beta", "evidence sentence", "Missing support."]],
    }]
    steps = build_aligned_steps(source, {("alpha", "correct support"): "P2"})
    assert "Passage Used: [P2]" in steps[0]["text"]
    assert steps[0]["cited_passage_ids"] == ["P2"]
    assert "Passage Used: []" in steps[1]["text"]
    assert "cited_passage_ids" not in steps[1]
    assert all(step["cited_triples"] == [] for step in steps)


def test_near_exact_removes_only_repeated_title_prefix():
    assert canonical_sentence_tokens("Lake Managua", "Lake Managua Lake Managua is a lake.") == [
        "is", "a", "lake"
    ]
    assert canonical_sentence_tokens("Lake Managua", "Another lake is large.") == [
        "another", "lake", "is", "large"
    ]


def test_near_exact_requires_same_title_and_high_sentence_overlap():
    selected = {
        "title": "Pusher 3",
        "sentence": "Pusher 3 is a 2005 Danish independent crime tragedy film directed by Nicolas Refn.",
    }
    required = {
        "title": "Pusher 3",
        "sentence": "Pusher 3 is a 2005 Danish independent crime film directed by Nicolas Refn.",
    }
    assert near_exact_match(selected, required) >= 0.90
    assert near_exact_match({**selected, "title": "Pusher"}, required) == 0.0
    assert token_f1(["a", "b"], ["a", "c"]) == 0.5
