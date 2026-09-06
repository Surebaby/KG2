from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from kgproweight.kg.qpeg_selector import edge_features, select_edges


def _edge(head, relation, tail, rank=0, score=1.0, rule="surface_pattern:born in"):
    return {
        "head_surface": head,
        "relation_surface": relation,
        "tail_surface": tail,
        "passage_id": f"p{rank}",
        "passage_rank": rank,
        "sentence_index": 0,
        "sentence_sha256": "a" * 64,
        "extraction_rule": rule,
        "relevance_score": score,
    }


def test_edge_features_are_answer_free_and_deterministic():
    edge = _edge("Ada Lovelace", "born in", "London", score=3.5)
    first = edge_features(dataset="hotpotqa", question="Where was Ada Lovelace born?", edge=edge)
    second = edge_features(dataset="hotpotqa", question="Where was Ada Lovelace born?", edge=edge)
    assert first == second
    assert "answer" not in first
    assert first["question_head_coverage"] > 0
    assert first["is_surface_relation"] == 1.0


def test_select_edges_is_thresholded_capped_and_deterministic():
    positive = _edge("Ada Lovelace", "born in", "London", score=3.5)
    negative = _edge("Unrelated", "is", "Other", rank=1, score=0.1, rule="first_sentence_copula")
    train_features = [
        edge_features(dataset="hotpotqa", question="Where was Ada Lovelace born?", edge=positive),
        edge_features(dataset="hotpotqa", question="Where was Ada Lovelace born?", edge=negative),
    ]
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(train_features)
    classifier = LogisticRegression(solver="liblinear", random_state=42).fit(matrix, [1, 0])
    record = {
        "dataset": "hotpotqa",
        "question": "Where was Ada Lovelace born?",
        "edges": [negative, positive],
    }
    first = select_edges(record=record, vectorizer=vectorizer, classifier=classifier, threshold=0.5, max_edges=1)
    second = select_edges(record=record, vectorizer=vectorizer, classifier=classifier, threshold=0.5, max_edges=1)
    assert first == second
    assert len(first[0]) <= 1
    assert all(score >= 0.5 for score in first[1])
