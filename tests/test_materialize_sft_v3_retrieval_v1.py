import copy
import json
from pathlib import Path

import pytest

from scripts.prepare import materialize_sft_v3_retrieval_v1 as m
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256, FAMILY_VERSION
from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.materialize_qpeg_v1_retrieval import _sha_json


def request(qid="q1", question="When did this unusual archive open to visitors?"):
    return {"dataset": "musique", "qid": qid, "question": question, "question_key": f"musique::{qid}",
            "question_sha256": question_sha256(question), "family_sha256": family_sha256(question),
            "family_version": FAMILY_VERSION, "role": "sft_training", "gold_access": False,
            "split": "train", "selection_rank": 0}


def contexts(rows):
    ps = [{"id": str(i), "source": "e5", "contents": f"Document {i}\nA plain context passage."} for i in range(10)]
    return [{**r, "schema_version": m.VERSION, "passages": copy.deepcopy(ps), "passages_sha256": _sha_json(ps), "retrieval_source": m.STACK} for r in rows]


class Backend:
    attestation = {"mode": "test_double", "fallback": False, "load_succeeded": True}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, rows):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected transient failure")
        return contexts(rows), [{"question_key": r["question_key"], "gold_access": False} for r in rows]


def frozen(tmp_path, n=1):
    rows = [request()]
    if n == 2:
        rows.append(request("q2", "Who founded the coastal museum collection?"))
    p = tmp_path / "pool.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    m.prepare(p, m.sha(p), out, "SFT-V3-TEST-ONLY", 1, bind_assets=False)
    return p, out


def test_question_pool_optional_identity_fields_preserved():
    r = request()
    m.validate_requests([r]); result = contexts([r]); m.validate_contexts([r], result)
    assert result[0]["split"] == "train" and result[0]["selection_rank"] == 0


@pytest.mark.parametrize("extra", [{"gold_answer": "x"}, {"metadata": {"gold_answer": "x"}}, {"teacher_output": "x"}, {"passages": []}])
def test_gold_or_evidence_is_never_accepted_as_question_only_input(extra):
    r = {**request(), **extra}
    with pytest.raises(ValueError):
        m.validate_requests([r])


def test_duplicate_global_family_rejected_across_datasets():
    a = request(); b = {**a, "dataset": "hotpotqa", "qid": "different", "question_key": "hotpotqa::different"}
    with pytest.raises(ValueError, match="duplicate"):
        m.validate_requests([a, b])


def test_request_hash_mismatch_creates_no_release(tmp_path):
    p = tmp_path / "requests.jsonl"; p.write_text(json.dumps(request()) + "\n")
    with pytest.raises(ValueError, match="SHA"):
        m.prepare(p, "0" * 64, tmp_path / "out", "TEST", 64, bind_assets=False)
    assert not (tmp_path / "out").exists()


def test_resume_reuses_only_sealed_completed_batches(tmp_path):
    _, out = frozen(tmp_path, n=2)
    with pytest.raises(RuntimeError, match="injected"):
        m.run(out, backend=Backend(fail_at=2))
    first = (out / "batches/00000000.json").read_bytes()
    backend = Backend(); report = m.run(out, backend=backend)
    assert backend.calls == 1 and report["reused_contexts"] == 1
    assert report["contexts"] == 2 and report["status"] == "COMPLETE_TEST_DOUBLE_ONLY"
    assert (out / "batches/00000000.json").read_bytes() == first
    assert (out / "failure_0001.json").exists()


def test_resume_detects_trace_tampering_not_only_passage_tampering(tmp_path):
    _, out = frozen(tmp_path, n=2)
    with pytest.raises(RuntimeError):
        m.run(out, backend=Backend(fail_at=2))
    batch = out / "batches/00000000.json"; obj = json.loads(batch.read_text())
    obj["retrieval_trace"][0]["gold_access"] = True; batch.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="seal"):
        m.run(out, backend=Backend())
    assert not (out / "manifest.json").exists()


def test_fallback_cannot_produce_a_completed_batch(tmp_path):
    _, out = frozen(tmp_path); backend = Backend()
    backend.attestation = {"fallback": True, "load_succeeded": True}
    with pytest.raises(ValueError, match="fallback"):
        m.run(out, backend=backend)
    assert not list((out / "batches").glob("*.json"))


@pytest.mark.parametrize("change", ["short", "duplicate", "wrong_hash", "wrong_identity", "nested_gold"])
def test_malformed_contexts_fail_closed(change):
    r = request(); c = contexts([r]); ps = c[0]["passages"]
    if change == "short":
        ps.pop()
    elif change == "duplicate":
        ps[-1] = ps[0]
    elif change == "wrong_hash":
        c[0]["passages_sha256"] = "0" * 64
    elif change == "wrong_identity":
        c[0]["qid"] = "other"
    else:
        ps[0]["metadata"] = {"gold_answer": "x"}
    with pytest.raises(ValueError):
        m.validate_contexts([r], c)


def test_test_frozen_protocol_cannot_start_real_models(tmp_path):
    _, out = frozen(tmp_path)
    with pytest.raises(ValueError, match="modes"):
        m.run(out)


def test_completed_release_cannot_be_overwritten(tmp_path):
    _, out = frozen(tmp_path); m.run(out, backend=Backend())
    manifest = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="complete"):
        m.run(out, backend=Backend())
    assert (out / "manifest.json").read_bytes() == manifest


def test_batch_cap_pauses_without_truncating_the_frozen_pool(tmp_path):
    _, out = frozen(tmp_path, n=2)
    protocol = (out / "protocol.json").read_bytes()
    backend = Backend(); progress = m.run(out, backend=backend, max_batches=1)
    assert progress["status"] == "PARTIAL_CHECKPOINT_PAUSED"
    assert progress["completed_contexts"] == 1 and progress["frozen_requests"] == 2
    assert backend.calls == 1 and not (out / "manifest.json").exists()
    resumed = Backend(); report = m.run(out, backend=resumed, max_batches=1)
    assert resumed.calls == 1 and report["contexts"] == 2 and report["reused_contexts"] == 1
    assert (out / "protocol.json").read_bytes() == protocol


class Branch:
    def batch_search(self, questions, num):
        assert num == 100
        return [[{"id": str(i), "contents": f"Document {i}\nA plain context passage.", "source": "e5"} for i in range(100)] for _ in questions]


class Router:
    retriever_list = [Branch(), Branch()]

    def add_source(self, values, retriever):
        return values

    def rrf_merge(self, values, topk, k):
        assert topk == 50 and k == 60
        return [v[:50] for v in values], [[0.02] * 50 for _ in values]


class CrossEncoder:
    def __init__(self, failure=None):
        self.failure = failure

    def predict(self, pairs, show_progress_bar):
        if self.failure == "raise":
            raise RuntimeError("real BGE prediction failed")
        if self.failure == "nan":
            return [float("nan")] * len(pairs)
        return list(range(len(pairs)))


@pytest.mark.parametrize("failure", ["raise", "nan"])
def test_bge_runtime_failure_never_falls_back(monkeypatch, failure):
    from kgproweight.retrieval import reranker
    monkeypatch.setattr(reranker, "rerank_with_bm25", lambda *a, **kw: pytest.fail("BM25 fallback called"))
    backend = m.StrictCanonicalBackend.__new__(m.StrictCanonicalBackend)
    backend.router = Router(); backend.ce = CrossEncoder(failure)
    with pytest.raises((RuntimeError, ValueError), match="BGE"):
        backend([request()])


def test_direct_bge_preserves_canonical_sort_and_top_ten():
    backend = m.StrictCanonicalBackend.__new__(m.StrictCanonicalBackend)
    backend.router = Router(); backend.ce = CrossEncoder()
    rows, traces = backend([request()])
    assert [p["id"] for p in rows[0]["passages"]] == [str(i) for i in range(49, 39, -1)]
    assert traces[0]["bge_sorted_indices"] == list(range(49, -1, -1))
    m.validate_contexts([request()], rows)


def test_candidate_pool_source_and_consumption_fields_preserved():
    r = {**request(), "source_split": "train", "within_split_dataset_rank": 1, "consumption_order": 3}
    m.validate_requests([r]); m.validate_contexts([r], contexts([r]))
    for field, invalid in [("source_split", "dev"), ("within_split_dataset_rank", 0), ("consumption_order", True)]:
        with pytest.raises(ValueError):
            m.validate_requests([{**r, field: invalid}])
