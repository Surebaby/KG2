import pytest

from kgproweight.data.silver_dataset import SilverTrajectory
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.kg.training_question_kg import apply_training_question_kg


def _traj(qid="q1", question="Question?", dataset="2wikimultihopqa"):
    return SilverTrajectory(
        qid=qid,
        question=question,
        answer="answer",
        dataset=dataset,
        steps=[],
        kg_subgraph=[("old", "relation", "value")],
    )


def _record(qid="q1", question="Question?", triples=None, query_plan=None, provenance=None):
    return make_question_kg_record(
        dataset="2wikimultihopqa",
        qid=qid,
        question=question,
        triples=triples if triples is not None else [("new", "relation", "answer")],
        query_plan=query_plan,
        provenance=provenance,
    )


def test_apply_training_question_kg_uses_dataset_qid_and_replaces_kg():
    traj = _traj()
    record = _record(
        query_plan={"hops": [{"pid": "P1"}]},
        provenance={"gold_access": False},
    )

    stats = apply_training_question_kg([traj], {record["question_key"]: record})

    assert traj.kg_subgraph == [("new", "relation", "answer")]
    assert traj.metadata["question_kg_runtime"] == {
        "question_key": "2wikimultihopqa::q1",
        "query_plan": {"hops": [{"pid": "P1"}]},
        "provenance": {"gold_access": False},
    }
    assert stats.to_dict() == {
        "trajectories": 1,
        "covered": 1,
        "absent": 0,
        "covered_empty": 0,
        "changed": 1,
        "coverage_rate": 1.0,
    }


def test_apply_training_question_kg_propagates_explicit_execution_trace():
    traj = _traj()
    record = _record(
        query_plan={"hops": [{"pid": "P1"}]},
        provenance={"gold_access": False},
    )
    record["execution"] = {"hops": [{"hop_index": 1, "matches": [["new", "relation", "answer"]]}]}
    apply_training_question_kg([traj], {record["question_key"]: record})
    assert traj.metadata["question_kg_runtime"]["execution"] == record["execution"]


def test_apply_training_question_kg_rejects_same_qid_wrong_dataset():
    traj = _traj(dataset="hotpotqa")
    record = _record()

    with pytest.raises(ValueError, match="coverage"):
        apply_training_question_kg([traj], {record["question_key"]: record})


def test_apply_training_question_kg_rejects_question_hash_mismatch():
    traj = _traj(question="Changed question?")
    record = _record()

    with pytest.raises(ValueError, match="question hash mismatch"):
        apply_training_question_kg([traj], {record["question_key"]: record})


def test_apply_training_question_kg_can_forbid_empty_records():
    traj = _traj()
    record = _record(triples=[])

    with pytest.raises(ValueError, match="covered-but-empty"):
        apply_training_question_kg(
            [traj], {record["question_key"]: record}, require_nonempty=True
        )
