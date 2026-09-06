import copy
import json

import pytest

from kgproweight.kg.qpeg import (
    QPEG_EXTRACTOR_VERSION,
    build_qpeg_record,
    validate_qpeg_record,
)
from scripts.prepare.freeze_qpeg_v1_protocol import (
    choose_disjoint_rows,
    family_sha256,
    question_family_signature,
    validate_partition,
)
from kgproweight.pipeline.kg_proweight_pipeline import KGProWeightPipeline


def _passage(pid, title, text):
    return {"id": pid, "contents": f'"{title}"\n{text}', "source": "test"}


def test_qpeg_is_deterministic_and_passage_backed():
    passages = [
        _passage("p1", "Ada Lovelace", "Ada Lovelace was born in London. She was known for mathematics."),
        _passage("p2", "London", "London is the capital of England."),
    ]
    first = build_qpeg_record(
        dataset="hotpotqa", qid="dev_x", question="Where was Ada Lovelace born?", passages=passages
    )
    second = build_qpeg_record(
        dataset="hotpotqa", qid="dev_x", question="Where was Ada Lovelace born?", passages=passages
    )
    assert first == second
    assert first["extractor_version"] == QPEG_EXTRACTOR_VERSION
    assert first["gold_access"] is False
    assert first["provenance_complete"] is True
    assert any(edge["relation_surface"] == "born in" for edge in first["edges"])
    validate_qpeg_record(first, passages=passages)


def test_qpeg_preserves_exact_question_identity_while_normalizing_for_extraction():
    passages = [_passage("p1", "Paris", "Paris is the capital of France.")]
    question = "What  is Paris?"
    record = build_qpeg_record(
        dataset="hotpotqa", qid="dev_spaces", question=question, passages=passages
    )
    assert record["question"] == question
    from kgproweight.kg.question_kg import question_sha256
    assert record["question_sha256"] == question_sha256(question)
    validate_qpeg_record(record, passages=passages)


def test_qpeg_cross_passage_title_mention_is_explicit():
    passages = [
        _passage("p1", "Film A", "Film A was directed by Jane Doe in London."),
        _passage("p2", "Jane Doe", "Jane Doe is a British director."),
    ]
    record = build_qpeg_record(
        dataset="2wikimultihopqa", qid="dev_y", question="Who directed Film A?", passages=passages
    )
    bridges = [edge for edge in record["edges"] if edge["extraction_rule"] == "cross_passage_title_mention"]
    assert bridges
    assert bridges[0]["tail_surface"] == "Jane Doe"


def test_qpeg_empty_graph_is_explicit_and_does_not_invent_sentence_edge():
    passages = [
        _passage(f"p{i}", f"Topic {i}", f"Topic {i} contains unusual wording without a recognised predicate")
        for i in range(20)
    ]
    record = build_qpeg_record(
        dataset="musique", qid="dev_z", question="What relates the topics?", passages=passages
    )
    assert record["edges"] == []
    assert record["kg_subgraph"] == []
    assert record["build_status"] == "empty"
    assert record["provenance_complete"] is False


def test_qpeg_rejects_cross_passage_self_loops_from_duplicate_titles():
    passages = [
        _passage("p1", "Same title", "Same title refers to another entry."),
        _passage("p2", "Same title", "Same title is an example."),
    ]
    record = build_qpeg_record(
        dataset="hotpotqa", qid="dev_duplicate", question="What is Same title?", passages=passages
    )
    assert all(edge["head_surface"].casefold() != edge["tail_surface"].casefold() for edge in record["edges"])


def test_qpeg_validator_rejects_hash_and_provenance_tampering():
    passages = [_passage("p1", "Paris", "Paris is the capital of France.")]
    record = build_qpeg_record(
        dataset="hotpotqa", qid="dev_t", question="What is Paris?", passages=passages
    )
    bad = copy.deepcopy(record)
    bad["question_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="question hash"):
        validate_qpeg_record(bad, passages=passages)
    bad = copy.deepcopy(record)
    bad["edges"][0]["sentence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance hash"):
        validate_qpeg_record(bad, passages=passages)


def test_answer_free_family_replaces_entity_and_number():
    first = question_family_signature("Who directed Film Alpha in 2007?")
    second = question_family_signature("Who directed Film Beta in 1999?")
    assert first == second
    assert "<entity>" in first
    assert "<num>" in first
    assert family_sha256("Who directed Film Alpha in 2007?") == family_sha256(
        "Who directed Film Beta in 1999?"
    )


def test_disjoint_selector_and_partition():
    rows = [
        {"id": f"q{i}", "question": f"Who directed Film {chr(65 + i)} in {2000 + i}?"}
        for i in range(12)
    ]
    # These questions deliberately share one template family, so only one may
    # be chosen. Add unique lexical templates to provide enough families.
    rows.extend(
        {"id": f"u{i}", "question": f"Which unique relation{i} connects item{i} to object{i}?"}
        for i in range(12)
    )
    first = choose_disjoint_rows(
        rows, excluded_qids=set(), excluded_families=set(), n=3, dataset="hotpotqa", seed=42
    )
    second = choose_disjoint_rows(
        rows,
        excluded_qids={row["qid"] for row in first},
        excluded_families={row["family_sha256"] for row in first},
        n=4,
        dataset="hotpotqa",
        seed=43,
    )
    final = [{
        "qid": "final",
        "family_sha256": family_sha256("Where is a completely separate final subject located?"),
    }]
    report = validate_partition(first, second, final)
    assert not any(report["qid_overlap"].values())
    assert not any(report["family_overlap"].values())


class _Item:
    def __init__(self, qid, question):
        self.id = qid
        self.question = question


def _qpeg_pipeline(tmp_path, record):
    path = tmp_path / "qpeg.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    pipe = object.__new__(KGProWeightPipeline)
    pipe.kg_supply_mode = "qpeg_v1"
    pipe._current_dataset_name = "hotpotqa"
    pipe._qpeg_records = {}
    pipe._kg_source_counts = {"index": 0, "fallback": 0, "empty": 0}
    pipe.inject_kg = True
    pipe.max_kg_triples = 12
    pipe._load_qpeg_records(str(path))
    return pipe


def test_pipeline_qpeg_lookup_and_fail_closed(tmp_path):
    passages = [_passage("p1", "Paris", "Paris is the capital of France.")]
    record = build_qpeg_record(
        dataset="hotpotqa", qid="dev_t", question="What is Paris?", passages=passages
    )
    pipe = _qpeg_pipeline(tmp_path, record)
    assert pipe._build_kg_context(_Item("dev_t", "What is Paris?"), passages=passages) == [
        tuple(value) for value in record["kg_subgraph"]
    ]
    with pytest.raises(ValueError, match="record missing"):
        pipe._build_kg_context(_Item("missing", "What is missing?"), passages=passages)
    with pytest.raises(ValueError, match="passages hash mismatch"):
        pipe._build_kg_context(
            _Item("dev_t", "What is Paris?"),
            passages=[_passage("p2", "Lyon", "Lyon is a city in France.")],
        )


def test_pipeline_qpeg_join_checks_question_hash(tmp_path):
    from flashrag.dataset import Dataset

    passages = [_passage("p1", "Paris", "Paris is the capital of France.")]
    record = build_qpeg_record(
        dataset="hotpotqa", qid="dev_t", question="What is Paris?", passages=passages
    )
    pipe = _qpeg_pipeline(tmp_path, record)
    dataset = Dataset(data=[{"id": "dev_t", "question": "Different question"}], config={"dataset_name": "hotpotqa"})
    with pytest.raises(ValueError, match="identity/hash join < 1.0"):
        pipe._enforce_qpeg_join(dataset)
