from __future__ import annotations

from scripts.pilot.audit_process_reward_rescoring import (
    _answer_evidence_support,
    _citation_gates,
)
from scripts.pilot.audit_rankability_hard_retrieval import (
    _literal_hit,
    _support_state,
    derive_hard_qids,
)


def test_hard_qids_require_greedy_and_every_sample_to_be_wrong():
    rows = [
        {"qid": "hard", "candidate_type": "greedy", "em": 0},
        *[{"qid": "hard", "candidate_type": "sampled", "em": 0} for _ in range(4)],
        {"qid": "oracle", "candidate_type": "greedy", "em": 0},
        {"qid": "oracle", "candidate_type": "sampled", "em": 1},
        *[{"qid": "oracle", "candidate_type": "sampled", "em": 0} for _ in range(3)],
        {"qid": "greedy", "candidate_type": "greedy", "em": 1},
        *[{"qid": "greedy", "candidate_type": "sampled", "em": 0} for _ in range(4)],
    ]
    assert derive_hard_qids(rows) == ["hard"]


def test_literal_hit_excludes_yes_no_and_uses_word_boundaries():
    passages = [{"contents": "Dante was signed by Bayern Munich."}]
    assert _literal_hit(passages, "Dante")
    assert not _literal_hit(passages, "yes")
    assert not _literal_hit(passages, "ant")
    assert _support_state(0, 2) == "none"
    assert _support_state(1, 2) == "partial"
    assert _support_state(2, 2) == "all"


def test_answer_evidence_support_is_gold_free_literal_grounding():
    passages = [{"contents": "Freaks and Geeks was created by Paul Feig."}]
    assert _answer_evidence_support("Paul Feig", passages, []) == 1.0
    assert _answer_evidence_support("Paul", passages, []) == 1.0
    assert _answer_evidence_support("yes", passages, []) == 0.0
    assert _answer_evidence_support("Dante", [], [("Dante", "instance of", "human")]) == 1.0


def test_citation_gates_distinguish_conclusion_and_question_bridge():
    class Step:
        cited_triples = [("Bayern Munich", "signed", "Dante")]
        intermediate_conclusion = "The player was Dante"

    conclusion, bridge = _citation_gates(Step(), "Which player did Bayern Munich sign?")
    assert conclusion == 1.0
    assert bridge == 1.0

    conclusion, bridge = _citation_gates(Step(), "Which footballer was signed?")
    assert conclusion == 1.0
    assert bridge == 0.0
