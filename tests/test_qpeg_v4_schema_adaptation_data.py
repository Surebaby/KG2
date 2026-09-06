from scripts.prepare.build_qpeg_v4_schema_adaptation_data import (
    _best_answer_sentence,
    _steps,
    build_trajectory,
)


def test_best_answer_sentence_prefers_answer_bearing_sentence():
    text = "The first statement is irrelevant. The institute was founded in 1960."
    assert "1960" in _best_answer_sentence(text, "1960")


def test_steps_have_exact_known_citation_and_synthesis():
    edge = ("Title", "evidence sentence", "The answer is 1960.")
    rows = _steps([edge, edge], ["1960", "1960"], "1960", cite=True)
    assert len(rows) == 3
    assert rows[0]["cited_triples"] == [list(edge)]
    assert "Knowledge Used: [(Title, evidence sentence, The answer is 1960.)]" in rows[0]["text"]
    assert rows[-1]["cited_triples"] == []


def test_no_graph_steps_remove_all_citations():
    edges = [
        ("A", "evidence sentence", "One."),
        ("B", "evidence sentence", "Two."),
    ]
    rows = _steps(edges, ["one", "two"], "answer", cite=False)
    assert all(step["cited_triples"] == [] for step in rows)
    assert all("Knowledge Used: []" in step["text"] for step in rows)


def test_build_hotpot_trajectory_uses_only_support_sentences():
    raw = {
        "id": "q1",
        "question": "Which came first?",
        "golden_answers": ["A"],
        "metadata": {
            "context": {
                "title": ["A", "B"],
                "sentences": [["A was founded in 1900."], ["B was founded in 2000."]],
            },
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
        },
    }
    row = build_trajectory("hotpotqa", raw, graph=True)
    assert len(row["kg_subgraph"]) == 2
    assert row["answer"] == "A"
    assert len(row["steps"]) == 3
    assert row["metadata"]["gold_train_only"] is True


def test_build_musique_trajectory_preserves_decomposition_order():
    raw = {
        "id": "train_1",
        "question": "When was the owner founded?",
        "golden_answers": ["1960"],
        "metadata": {"metadata": {"question_decomposition": [
            {"answer": "University X", "support_paragraph": {"idx": 1, "title": "Paper", "paragraph_text": "Paper is owned by University X."}},
            {"answer": "1960", "support_paragraph": {"idx": 2, "title": "University X", "paragraph_text": "University X was founded in 1960."}},
        ]}},
    }
    row = build_trajectory("musique", raw, graph=True)
    assert row["kg_subgraph"][0][0] == "Paper"
    assert row["kg_subgraph"][1][0] == "University X"
    assert row["steps"][1]["text"].endswith("Conclusion: 1960")
