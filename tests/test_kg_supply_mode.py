import json

import pytest

from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.pipeline.kg_proweight_pipeline import KGProWeightPipeline


def _make_pipeline(tmp_path, records=None):
    pipe = object.__new__(KGProWeightPipeline)
    pipe.kg_supply_mode = "proofkg_v1"
    pipe._current_dataset_name = "2wikimultihopqa"
    pipe._proofkg_records = {}
    pipe._q_kg_index = {}
    pipe._kg_source_counts = {"index": 0, "fallback": 0, "empty": 0}
    pipe.inject_kg = True
    pipe.max_kg_triples = 12
    if records is not None:
        p = tmp_path / "records.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        pipe._load_proofkg_records(str(p))
    return pipe


class _Item:
    def __init__(self, id_, question):
        self.id = id_
        self.question = question


def _rec(dataset="2wikimultihopqa", qid="Q1", question="who is ...", triples=None):
    return make_question_kg_record(
        dataset=dataset, qid=qid, question=question,
        triples=triples or [["A", "mother", "B"]],
    )


def test_loads_and_looks_up_canonical_records(tmp_path):
    pipe = _make_pipeline(tmp_path, records=[_rec()])
    assert pipe._proofkg_records["2wikimultihopqa::Q1"] == [("A", "mother", "B")]
    assert pipe._build_kg_context(_Item("Q1", "who is ...")) == [("A", "mother", "B")]


def test_missing_record_is_empty_not_legacy(tmp_path):
    pipe = _make_pipeline(tmp_path, records=[_rec()])
    assert pipe._build_kg_context(_Item("Q9", "who is ...")) == []


def test_duplicate_key_raises(tmp_path):
    r = _rec()
    with pytest.raises(ValueError):
        _make_pipeline(tmp_path, records=[r, dict(r)])


def test_question_hash_mismatch_raises(tmp_path):
    r = _rec()
    r["question_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _make_pipeline(tmp_path, records=[r])


def test_dataset_qid_mismatch_raises(tmp_path):
    r = _rec()
    r["qid"] = "Q9"  # question_key still says ::Q1 -> mismatch
    with pytest.raises(ValueError):
        _make_pipeline(tmp_path, records=[r])


def test_proofkg_mode_requires_records_path(tmp_path):
    pipe = _make_pipeline(tmp_path)
    with pytest.raises(ValueError):
        pipe._load_proofkg_records(None)
    with pytest.raises(ValueError):
        pipe._load_proofkg_records(str(tmp_path / "missing.jsonl"))


def test_malformed_schema_version_raises(tmp_path):
    r = _rec()
    r["schema_version"] = "not-a-real-schema"
    with pytest.raises(ValueError, match="schema"):
        _make_pipeline(tmp_path, records=[r])


def test_malformed_triple_raises(tmp_path):
    r = _rec()
    r["kg_subgraph"] = [["A", "B"]]  # two components, not a (h, r, t) triple
    with pytest.raises(ValueError, match="invalid KG triple"):
        _make_pipeline(tmp_path, records=[r])


def _real_dataset(qids_questions):
    from flashrag.dataset import Dataset

    data = [{"id": qid, "question": q} for qid, q in qids_questions]
    return Dataset(data=data, config={"dataset_name": "2wikimultihopqa"})


def test_join_passes_when_all_questions_covered(tmp_path):
    pipe = _make_pipeline(
        tmp_path,
        records=[_rec(qid="Q1", question="who is A?"), _rec(qid="Q2", question="who is B?")],
    )
    ds = _real_dataset([("Q1", "who is A?"), ("Q2", "who is B?")])
    pipe._enforce_proofkg_join(ds)  # must not raise


def test_join_raises_on_missing_question(tmp_path):
    pipe = _make_pipeline(tmp_path, records=[_rec(qid="Q1", question="who is A?")])
    ds = _real_dataset([("Q1", "who is A?"), ("Q2", "who is B?")])
    with pytest.raises(ValueError, match="identity join < 1.0"):
        pipe._enforce_proofkg_join(ds)


def test_join_requires_known_dataset_name(tmp_path):
    pipe = _make_pipeline(tmp_path, records=[_rec()])
    pipe._current_dataset_name = ""
    ds = _real_dataset([("Q1", "who is A?")])
    with pytest.raises(ValueError, match="dataset_name"):
        pipe._enforce_proofkg_join(ds)
