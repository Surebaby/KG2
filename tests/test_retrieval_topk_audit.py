import pytest
import json

from scripts.pilot.audit_retrieval_topk import (
    _aggregate,
    _supporting_titles,
    _validate_cutoffs,
)
from scripts.pilot.audit_iterative_bridge_retrieval import (
    _apply_explicit_retrieval_assets,
    _validate_full_wiki18_assets,
)


def test_supporting_titles_handles_hotpot_and_musique_schemas():
    hotpot = {"metadata": {"supporting_facts": {"title": ["A", "B", "A"]}}}
    musique = {
        "metadata": {
            "metadata": {
                "question_decomposition": [
                    {"support_paragraph": {"title": "C"}},
                    {"support_paragraph": {"title": "D"}},
                ]
            }
        }
    }
    assert _supporting_titles(hotpot) == ["A", "B"]
    assert _supporting_titles(musique) == ["C", "D"]


def test_retrieval_aggregate_reports_any_all_and_micro_recall():
    metrics = _aggregate(
        [
            {
                "n_support_titles": 2,
                "n_support_titles_hit": 2,
                "any_support_hit": True,
                "all_support_hit": True,
                "gold_literal_hit": True,
            },
            {
                "n_support_titles": 2,
                "n_support_titles_hit": 1,
                "any_support_hit": True,
                "all_support_hit": False,
                "gold_literal_hit": False,
            },
        ]
    )
    assert metrics["any_support_recall_pct"] == 100.0
    assert metrics["all_support_recall_pct"] == 50.0
    assert metrics["support_title_micro_recall_pct"] == 75.0


def test_rerank_cutoffs_must_fit_inside_rrf_candidate_pool():
    assert _validate_cutoffs(100, [30, 10, 20, 20]) == [10, 20, 30]
    with pytest.raises(ValueError):
        _validate_cutoffs(50, [10, 100])


def test_bridge_audit_overrides_all_nested_retrieval_assets():
    config = {
        "corpus_path": "old-corpus",
        "index_path": "old-dense",
        "bm25_index_path": "old-bm25",
        "multi_retriever_setting": {
            "retriever_list": [
                {"retrieval_method": "e5", "index_path": "old", "corpus_path": "old"},
                {"retrieval_method": "bm25", "index_path": "old", "corpus_path": "old"},
            ]
        },
    }
    result = _apply_explicit_retrieval_assets(
        config,
        corpus_path="full-corpus",
        dense_index_path="full-dense",
        bm25_index_path="full-bm25",
    )
    dense, sparse = result["multi_retriever_setting"]["retriever_list"]
    assert result["corpus_path"] == dense["corpus_path"] == sparse["corpus_path"] == "full-corpus"
    assert result["index_path"] == dense["index_path"] == "full-dense"
    assert result["bm25_index_path"] == sparse["index_path"] == "full-bm25"


def test_bridge_audit_full_asset_guard_rejects_count_mismatch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"0","contents":"A"}\n', encoding="utf-8")
    dense = tmp_path / "dense.dat"
    dense.write_bytes(b"\x00" * (768 * 2))
    bm25 = tmp_path / "bm25"
    bm25.mkdir()
    (bm25 / "params.index.json").write_text(json.dumps({"num_docs": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2"):
        _validate_full_wiki18_assets(
            str(corpus),
            str(dense),
            str(bm25),
            expected_docs=2,
        )
