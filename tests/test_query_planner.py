import pytest
from types import SimpleNamespace

from kgproweight.kg.query_planner import QueryHop, QueryPlan, plan_question
from kgproweight.kg.question_kg import (
    load_question_kg_index,
    make_question_kg_record,
    question_key,
)
from kgproweight.kg.wikidata_retriever import WikidataSubgraphRetriever
from scripts.pilot.build_query_aware_proof_kg_pilot import build_local_proof_kg


def test_direct_birth_comparison_uses_p569_only():
    plan = plan_question("Who was born earlier, Anna Seghers or Morgan Llywelyn?")
    assert plan.recognized
    assert plan.anchors == ["Anna Seghers", "Morgan Llywelyn"]
    assert [list(hop.pids) for hop in plan.hops] == [["P569"], ["P569"]]


def test_director_country_comparison_builds_two_parallel_chains():
    plan = plan_question(
        "Do both directors of films Modern Love (1990 Film) and The Candy Kid (1928 Film) "
        "share the same nationality?"
    )
    assert plan.recognized
    assert [list(hop.pids) for hop in plan.hops] == [
        ["P57"], ["P27"], ["P57"], ["P27"]
    ]


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "Which film has the director born earlier, All Of A Sudden Peggy or Appointment With Death (Film)?",
            ["All Of A Sudden Peggy", "Appointment With Death (Film)"],
        ),
        (
            "Are director of film Waiting For Caroline and director of film How To Undress In Front Of Your Husband both from the same country?",
            ["Waiting For Caroline", "How To Undress In Front Of Your Husband"],
        ),
        (
            "Do both Swordfish (Film) and New Jersey Drive films have the directors from the same country?",
            ["Swordfish (Film)", "New Jersey Drive"],
        ),
    ],
)
def test_comparison_anchor_parser_removes_relation_scaffolding(question, expected):
    plan = plan_question(question)
    assert plan.recognized
    assert plan.anchors == expected


def test_nested_owner_inception_plan():
    plan = plan_question("When was the institute that owned The Collegian founded?")
    assert plan.recognized
    assert plan.anchors == ["The Collegian"]
    assert [list(hop.pids) for hop in plan.hops] == [["P127"], ["P571"]]


def test_unknown_question_abstains():
    plan = plan_question("Explain why this unusual event mattered?")
    assert not plan.recognized
    assert plan.hops == []


@pytest.mark.parametrize(
    "question,anchors,pids",
    [
        (
            "Which film was released earlier, The Big Bad Swim or Cry For Happy?",
            ["The Big Bad Swim", "Cry For Happy"], [["P577"], ["P577"]],
        ),
        (
            "Were both Kerry Hudson and Thomas Pearsall, born in the same place?",
            ["Kerry Hudson", "Thomas Pearsall"], [["P19"], ["P19"]],
        ),
        (
            "What nationality is the director of film Return To Sender (2004 Film)?",
            ["Return To Sender (2004 Film)"], [["P57"], ["P27"]],
        ),
        (
            "What is the date of death of Archduchess Eleonora Of Austria's father?",
            ["Archduchess Eleonora Of Austria"], [["P22"], ["P570"]],
        ),
        (
            "Who is Philip Of Artois, Count Of Eu's paternal grandfather?",
            ["Philip Of Artois, Count Of Eu"], [["P22"], ["P22"]],
        ),
    ],
)
def test_generic_2wiki_relation_templates(question, anchors, pids):
    plan = plan_question(question)
    assert plan.recognized
    assert plan.anchors == anchors
    assert [list(hop.pids) for hop in plan.hops] == pids


def test_question_kg_identity_uses_dataset_qid_and_hash_guard():
    record = make_question_kg_record(
        dataset="hotpotqa",
        qid="train_1",
        question="Who directed Film A?",
        triples=[("Film A", "director", "Director B")],
    )
    assert record["question_key"] == question_key("hotpotqa", "train_1")
    assert load_question_kg_index([record])[record["question_key"]]["qid"] == "train_1"
    broken = dict(record, question="different")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_question_kg_index([broken])


def test_literal_aware_wikidata_query_is_explicit_and_cache_isolated(tmp_path):
    legacy = WikidataSubgraphRetriever(cache_dir=str(tmp_path), offline=True)
    literal = WikidataSubgraphRetriever(
        cache_dir=str(tmp_path), offline=True, include_literal_values=True
    )
    assert "isLiteral(?tail)" not in legacy._build_1hop_query("Q1")
    assert "isLiteral(?tail)" in literal._build_1hop_query("Q1")
    assert "isLiteral(?tail)" in literal._build_2hop_query("Q1")
    assert literal._raw_cache_keys("Q1") == ["Q1_2_30_literal_v1"]
    assert all("literal" not in key for key in legacy._raw_cache_keys("Q1"))


def test_query_plan_executes_iteratively_through_prior_tail():
    class Linker:
        mapping = {"The Collegian": ("Q1", "The Collegian"), "University B": ("Q2", "University B")}

        def link_single(self, surface, question=""):
            qid, label = self.mapping.get(surface, (None, ""))
            return SimpleNamespace(
                selected_qid=qid, selected_label=label, score=1.0 if qid else 0.0,
                margin=1.0 if qid else 0.0, abstained=not bool(qid),
                abstain_reason="" if qid else "missing",
            )

    class Retriever:
        values = {
            "Q1": [("The Collegian", "owned by", "University B")],
            "Q2": [("University B", "inception", "1960")],
        }

        def fetch(self, qids):
            return self.values.get(qids[0], [])

    plan = plan_question("When was the institute that owned The Collegian founded?")
    triples, diagnostics = build_local_proof_kg(
        "When was the institute that owned The Collegian founded?", plan, Linker(), Retriever()
    )
    assert triples == [
        ("The Collegian", "owned by", "University B"),
        ("University B", "inception", "1960"),
    ]
    assert diagnostics["complete_plan_execution"] is True


def test_targeted_property_tail_qid_bypasses_ambiguous_bridge_relinking():
    class Linker:
        def link_single(self, surface, question=""):
            if surface == "Film A":
                return SimpleNamespace(
                    selected_qid="Q1", selected_label="Film A", score=1.0,
                    margin=1.0, abstained=False, abstain_reason="",
                )
            raise AssertionError("the director label must not be re-linked")

    class Retriever:
        def fetch_edges(self, qid, pids):
            if qid == "Q1":
                return [{
                    "head_qid": "Q1", "head_label": "Film A", "pid": "P57",
                    "relation": "director", "tail_qid": "Q2", "tail_value": "Ambiguous Name",
                }]
            return [{
                "head_qid": "Q2", "head_label": "Ambiguous Name", "pid": "P569",
                "relation": "date of birth", "tail_qid": None, "tail_value": "1950-01-01",
            }]

    plan = QueryPlan(
        question_sha256="x", recognized=True, operation="compose_relation",
        anchors=["Film A"], abstain_reason="",
        hops=[
            QueryHop("Film A", ["P57"], "person", "bridge"),
            QueryHop("$person", ["P569"], "date", "answer"),
        ],
    )
    triples, diagnostics = build_local_proof_kg("question", plan, Linker(), Retriever())
    assert triples[-1] == ("Ambiguous Name", "date of birth", "1950-01-01")
    assert diagnostics["complete_plan_execution"] is True
