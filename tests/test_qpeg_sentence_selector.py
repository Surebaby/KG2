from __future__ import annotations

import numpy as np

from kgproweight.kg.qpeg import passage_sentences, validate_qpeg_record
from kgproweight.kg.qpeg_sentence_selector import (
    build_selected_sentence_record,
    select_sentence_edges,
    sentence_candidates,
    sentence_features,
)


class _Vectorizer:
    def transform(self, rows):
        return rows


class _Classifier:
    def __init__(self, values):
        self.values = values

    def predict_proba(self, rows):
        values = self.values[: len(rows)]
        return np.asarray([[1.0 - value, value] for value in values])


def _passages():
    return [
        {"id": "p0", "title": "Alpha", "contents": '"Alpha"\nAlpha was founded in 1960. It is in Paris.'},
        {"id": "p1", "title": "Beta", "contents": '"Beta"\nBeta employed Alice. Alpha acquired Beta.'},
    ]


def test_explicit_training_sentence_boundaries_are_preserved():
    passage = {"contents": '"X"\nignored', "_sentences": [" One.", "Two!"]}
    assert passage_sentences(passage) == ["One.", "Two!"]


def test_features_are_answer_free_and_deterministic():
    kwargs = dict(
        dataset="hotpotqa", question="When was Alpha founded?", title="Alpha",
        sentence="Alpha was founded in 1960.", passage_rank=0, sentence_index=0,
        all_titles=["Alpha", "Beta"],
    )
    first = sentence_features(**kwargs)
    assert first == sentence_features(**kwargs)
    assert not any("answer" in key or "gold" in key or "support" in key for key in first)


def test_candidates_retain_full_sentence_and_provenance():
    candidates = sentence_candidates(dataset="hotpotqa", question="When was Alpha founded?", passages=_passages())
    assert candidates[0]["tail_surface"] == "Alpha was founded in 1960."
    assert candidates[0]["relation_surface"] == "evidence sentence"
    assert len(candidates[0]["sentence_sha256"]) == 64


def test_selector_is_thresholded_ranked_and_fail_closed():
    candidates = sentence_candidates(dataset="hotpotqa", question="When was Alpha founded?", passages=_passages())
    selected, scores = select_sentence_edges(
        candidates=candidates,
        vectorizer=_Vectorizer(),
        classifier=_Classifier([0.2, 0.8, 0.7, 0.1]),
        threshold=0.6,
        max_edges=2,
    )
    assert scores == [0.8, 0.7]
    assert [row["sentence_index"] for row in selected] == [1, 0]
    empty, empty_scores = select_sentence_edges(
        candidates=candidates,
        vectorizer=_Vectorizer(),
        classifier=_Classifier([0.1, 0.2, 0.3, 0.4]),
        threshold=0.9,
        max_edges=2,
    )
    assert empty == [] and empty_scores == []


def test_selector_deduplicates_identical_typed_edges_before_budget():
    candidates = sentence_candidates(
        dataset="hotpotqa",
        question="When was Alpha founded?",
        passages=[
            {"id": "p0", "title": "Alpha", "contents": '"Alpha"\nAlpha was founded in 1960.'},
            {"id": "p1", "title": "Alpha", "contents": '"Alpha"\nAlpha was founded in 1960.'},
        ],
    )
    selected, _ = select_sentence_edges(
        candidates=candidates,
        vectorizer=_Vectorizer(),
        classifier=_Classifier([0.95, 0.90]),
        threshold=0.5,
        max_edges=4,
    )
    assert len(selected) == 1


def test_selected_record_validates_and_contains_no_gold_fields():
    record = build_selected_sentence_record(
        dataset="hotpotqa",
        qid="q1",
        question="When was Alpha founded?",
        passages=_passages(),
        vectorizer=_Vectorizer(),
        classifier=_Classifier([0.95, 0.1, 0.2, 0.3]),
        threshold=0.9,
        max_edges=2,
    )
    validate_qpeg_record(record, passages=_passages())
    assert record["kg_subgraph"] == [["Alpha", "evidence sentence", "Alpha was founded in 1960."]]
    assert record["gold_access"] is False
    assert "gold_answers" not in record and "supporting_facts" not in record
