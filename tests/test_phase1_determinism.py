"""Phase-1 quota decisions must not depend on teacher response timing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kgproweight.training.phase1_distill import (
    Phase1Config,
    StratifiedSilverFilter,
    TeacherClient,
    _RetrievalAdapter,
    _finalize_stratified_candidates,
    _quota_selection_counts,
    phase1_candidate_path,
    run_phase1,
)
from scripts.train.phase1_generate_silver import _select_items
from kgproweight.config import ProjectConfig, load_config


def test_parallel_candidates_are_consumed_in_input_order():
    root = Path(__file__).resolve().parents[1]
    src = (root / "kgproweight" / "training" / "phase1_distill.py").read_text()
    body = src[src.index("def run_phase1"):]
    assert "for fut in futures:" in body
    assert "for fut in as_completed(" not in body


def test_random_source_subset_is_seeded_and_not_a_prefix():
    items = [{"id": str(i)} for i in range(100)]
    a = _select_items(items, 20, "random", 42)
    b = _select_items(items, 20, "random", 42)
    c = _select_items(items, 20, "random", 7)
    assert a == b
    assert a != c
    assert a != items[:20]


def test_first_strategy_remains_available_for_historical_reproduction():
    items = [{"id": str(i)} for i in range(10)]
    assert _select_items(items, 3, "first", 42) == items[:3]


def test_bridge_mode_yaml_is_a_string_and_defaults_off():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "configs/training/phase1_silver.yaml",
        "configs/training/phase1_silver_bridge_paired_pilot.yaml",
    ):
        cfg = load_config(str(root / relative), validate=ProjectConfig)
        assert cfg.training.silver_data.bridge_mode == "off"


def test_teacher_client_explicitly_disables_thinking_and_returns_usage_metadata():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="ok", reasoning_content=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            )
            return SimpleNamespace(choices=[choice], usage=usage, model="deepseek-v4-pro")

    client = object.__new__(TeacherClient)
    client.model = "deepseek-v4-pro"
    client.temperature = 0.0
    client.max_tokens = 4000
    client.max_retries = 1
    client.thinking = False
    client.reasoning_effort = None
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    content, metadata = client.chat_with_metadata([{"role": "user", "content": "q"}])
    assert content == "ok"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in calls[0]
    assert metadata["thinking"] is False
    assert metadata["total_tokens"] == 13
    assert metadata["cache_hit_tokens"] == 4


def test_retrieval_adapter_batches_before_teacher_workers():
    class Retriever:
        def __init__(self):
            self.calls = []

        def batch_search(self, queries):
            self.calls.append(list(queries))
            return [[{"id": q, "contents": q}] for q in queries]

    retriever = Retriever()
    adapter = _RetrievalAdapter(retriever, top_k=1)
    got = adapter.batch(["q1", "q2"])
    assert retriever.calls == [["q1", "q2"]]
    assert [row[0]["id"] for row in got] == ["q1", "q2"]


def test_phase1_additive_v3_preserves_originals_and_batches_bridge_queries(monkeypatch):
    class Retriever:
        def __init__(self):
            self.calls = []

        def batch_search(self, queries):
            self.calls.append(list(queries))
            if len(self.calls) == 1:
                return [
                    [
                        {"id": f"{query}-a", "contents": f"{query} Alpha Bridge\nbody"},
                        {"id": f"{query}-b", "contents": f"{query} Beta Bridge\nbody"},
                    ]
                    for query in queries
                ]
            return [
                [{"id": f"bridge-{i}", "contents": f"Bridge passage {i}"}]
                for i, _ in enumerate(queries)
            ]

    rerank_sizes = []

    def fake_rerank(questions, candidates, *, topk, **kwargs):
        rerank_sizes.append([len(row) for row in candidates])
        return [list(row)[:topk] for row in candidates]

    monkeypatch.setattr("kgproweight.retrieval.reranker.rerank_passages", fake_rerank)
    retriever = Retriever()
    adapter = _RetrievalAdapter(
        retriever,
        top_k=2,
        rerank_topk=3,
        bridge_mode="additive_v3",
        bridge_first_round_topk=2,
        bridge_max_queries=2,
        bridge_only_k=2,
    )
    got = adapter.batch(["q1", "q2"])
    assert retriever.calls[0] == ["q1", "q2"]
    assert len(retriever.calls[1]) == 4
    assert rerank_sizes == [[2, 2], [4, 4]]
    assert [doc["id"] for doc in got[0][:2]] == ["q1-a", "q1-b"]


def test_phase1_additive_v3_rejects_nonbatch_retriever():
    class SearchOnly:
        def search(self, query):
            return []

    adapter = _RetrievalAdapter(
        SearchOnly(),
        rerank_topk=10,
        bridge_mode="additive_v3",
    )
    with pytest.raises(TypeError, match="batch_search"):
        adapter("question")


def test_posthoc_quota_selection_enforces_caps_and_maximises_size():
    selected = _quota_selection_counts(10, 10, 10, medium_quota=0.35, sparse_quota=0.25)
    total = sum(selected.values())
    assert selected["kg_rich"] == 10
    assert selected["kg_medium"] / total <= 0.35
    assert selected["kg_sparse"] / total <= 0.25
    # No 25-row solution exists under the caps; 24 is maximal.
    assert total == 24


def test_posthoc_selection_preserves_candidates_and_writes_derived_output(tmp_path):
    candidate_path = tmp_path / "silver.candidates.jsonl"
    output_path = tmp_path / "silver.jsonl"
    rows = []
    for i, bucket in enumerate(["kg_rich"] * 4 + ["kg_sparse"] * 4):
        rows.append(
            {
                "qid": str(i),
                "dataset": "toy",
                "accepted": False,
                "metadata": {
                    "quality_pass": True,
                    "kg_bucket": bucket,
                    "bucket": bucket,
                },
            }
        )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    stats = _finalize_stratified_candidates(
        candidate_path, output_path, StratifiedSilverFilter(), seed=42
    )
    candidates_after = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    selected = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert all(not row["accepted"] for row in candidates_after)
    assert stats["quality_passed"] == 8
    assert stats["accepted"] == 5
    assert sum(row["accepted"] for row in selected) == 5
    assert sum(row["metadata"]["kg_bucket"] == "kg_sparse" and row["accepted"] for row in selected) == 1


def test_run_phase1_retries_bad_citation_and_preserves_candidate_sidecar(tmp_path):
    class Retriever:
        def batch_search(self, queries):
            return [[] for _ in queries]

    class Linker:
        def link_single(self, *args, **kwargs):
            return SimpleNamespace(selected_qid="Q1", abstained=False)

    class KG:
        def fetch(self, qids):
            return [
                ("Alpha", "country", "France"),
                ("Alpha", "instance of", "organization"),
                ("France", "continent", "Europe"),
            ]

    class Annotator:
        def annotate_trajectory(self, steps, kg):
            return [1.0 for _ in steps]

    invalid = """[Step 1]\nReasoning: A.\nKnowledge Used: [(Outside, country, France)]\nConclusion: A.
[Step 2]\nReasoning: B.\nKnowledge Used: []\nConclusion: B.
[Step 3]\nReasoning: C.\nKnowledge Used: []\nConclusion: C.\n[Final Answer] France"""
    valid = """[Step 1]\nReasoning: Alpha is in France.\nKnowledge Used: [(Alpha, country, France)]\nConclusion: Alpha is in France.
[Step 2]\nReasoning: Alpha is in France.\nKnowledge Used: [(Alpha, country, France)]\nConclusion: Alpha is in France.
[Step 3]\nReasoning: Alpha is in France.\nKnowledge Used: [(Alpha, country, France)]\nConclusion: Alpha is in France.\n[Final Answer] France"""

    class Teacher:
        model = "fake-teacher"

        def __init__(self):
            self.outputs = iter([invalid, valid])

        def chat(self, messages):
            return next(self.outputs)

    output = tmp_path / "silver.jsonl"
    cfg = Phase1Config(
        dataset_name="toy",
        items=[{"id": "q1", "question": "What country is Alpha in?", "golden_answers": ["France"]}],
        output_path=str(output),
        teacher_client=Teacher(),
        retriever_factory=Retriever(),
        entity_linker=Linker(),
        kg_retriever=KG(),
        prm_annotator=Annotator(),
        max_workers=1,
        accept_filter=StratifiedSilverFilter(),
    )
    stats = run_phase1(cfg)

    candidate = json.loads(phase1_candidate_path(output).read_text())
    selected = json.loads(output.read_text())
    assert candidate["accepted"] is False
    assert candidate["metadata"]["format_retried"] is True
    assert candidate["metadata"]["retry_succeeded"] is True
    assert candidate["metadata"]["citation_contract_errors"] == []
    assert selected["accepted"] is True
    assert stats["quality_passed"] == stats["accepted"] == 1
