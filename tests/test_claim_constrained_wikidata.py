from kgproweight.kg.claim_constrained_wikidata import (
    relation_similarity,
    select_claim_edges,
    tail_support,
)


def _edge(pid="P57", tail_qid="Q2", tail="Jane Doe", relation="director"):
    return {
        "head_qid": "Q1", "head_label": "Example Film", "pid": pid,
        "relation": relation, "tail_qid": tail_qid, "tail_value": tail,
        "tail_raw_value": None, "source_revision_id": "7",
        "source_revision_timestamp": "2020-01-01T00:00:00Z",
        "source_cutoff": "2020-12-09T23:59:59Z",
    }


def test_relation_similarity_handles_global_morphology():
    assert relation_similarity("who directed it", "director") > 0
    assert relation_similarity("place of birth", "occupation") == 0


def test_tail_support_prefers_exact_passage_title_qid():
    assert tail_support(
        _edge(), passage_title_qids={"Q2"}, passage_blob="unrelated text"
    ) == "passage_title_qid"


def test_planned_pid_requires_visible_tail():
    selected, rejected = select_claim_edges(
        [_edge()], planned_pid="P57", planned_relation="director",
        property_labels={"P57": "director"}, passage_title_qids=set(),
        passage_blob="No person is named here.",
    )
    assert selected == []
    assert any(row["reason"] == "tail_not_supported_by_frozen_passages" for row in rejected)


def test_unique_passage_entity_claim_can_correct_wrong_planned_pid():
    selected, _ = select_claim_edges(
        [_edge(pid="P144", tail_qid="Q9", tail="Nebo Zovyot", relation="based on")],
        planned_pid="P175", planned_relation="contained scenes from",
        property_labels={"P144": "based on"}, passage_title_qids={"Q9"},
        passage_blob="Nebo Zovyot is a film.",
    )
    assert [row["pid"] for row in selected] == ["P144"]
    assert selected[0]["selection_reason"] == "unique_passage_claim"


def test_ambiguous_unrelated_passage_claims_abstain():
    edges = [
        _edge(pid="P144", tail_qid="Q9", tail="A", relation="based on"),
        _edge(pid="P161", tail_qid="Q8", tail="B", relation="cast member"),
    ]
    selected, _ = select_claim_edges(
        edges, planned_pid="P175", planned_relation="friend",
        property_labels={"P144": "based on", "P161": "cast member"},
        passage_title_qids={"Q8", "Q9"}, passage_blob="A B",
    )
    assert selected == []


def test_metadata_claim_never_enters_output():
    selected, _ = select_claim_edges(
        [_edge(pid="P31", relation="instance of")], planned_pid="P31",
        planned_relation="instance of", property_labels={"P31": "instance of"},
        passage_title_qids={"Q2"}, passage_blob="Jane Doe",
    )
    assert selected == []
