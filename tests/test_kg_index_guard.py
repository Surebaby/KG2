"""The question-KG index guard must fire on the broken state, not the fixed one.

PPO(1) and PPO(2) both ran with an index that missed 100% of their prompts,
behind a `logger.warning` that scrolled past unread. `max_kg_index_miss_rate`
turns that into a hard stop -- but only if it measures the right quantity.

Two different things produce "no triples for this prompt":

  ABSENT   the question is not in the index at all. The index was built over a
           different question set (the shipped v2 index is built from the DEV
           splits, qids `dev_*`, while PPO rolls out on silver's `train_*`).
           This is the failure, and it is fixable by rebuilding.
  EMPTY    the question IS in the index and its subgraph is legitimately empty:
           entity linking abstained, or the subgraph cache holds nothing for the
           linked QIDs. MEASURED on a full offline rebuild of
           silver_v1_reannotated (--min_keep 5 --max_keep 12): 9.7% of the 9,839
           ACCEPTED prompts, 6.3% after the silver kg_subgraph fallback. It does
           NOT fall with a rebuild -- it is a property of the KG and the linker.

So the cap has to be checked against ABSENT alone. Checked against the sum, a
correctly rebuilt index (0.00% absent, 9.7% empty) trips a 5% cap, and the only
way to start training is to disable the guard -- which is how a guard ends up
protecting nothing.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")

from kgproweight.training import phase3_ppo as P


class _Traj:
    def __init__(self, question, kg):
        self.question = question
        self.kg_subgraph = kg
        self.retrieved_passages = []
        self.qid = "train_0"
        self.dataset = "hotpotqa"
        self.metadata = {"gold_answer": "a"}


class _Reader:
    def __init__(self, trajs):
        self._trajs = trajs

    def accepted(self):
        return self._trajs


class _Tok:
    """Only needs to be callable and decodable — no prompt-length logic here."""

    def __call__(self, text, **kw):  # noqa: ARG002
        return {"input_ids": [0] * 10}

    def decode(self, ids, **kw):  # noqa: ARG002
        return ""


_KG = [("a", "r", "b")]


def _cfg(cap=0.05):
    return types.SimpleNamespace(
        ppo_max_kg_triples=12, ppo_max_passages=5, max_input_length=6144,
        max_kg_index_miss_rate=cap, silver_path="SILVER.jsonl",
    )


def _trajs(n=100):
    return [_Traj(f"q{i}", _KG) for i in range(n)]


def test_covered_but_empty_does_not_abort():
    """~10% empty subgraphs is the measured post-rebuild state — it must train."""
    index = {f"q{i}": (_KG if i >= 10 else []) for i in range(100)}
    rows = P._prepare_prompts(_Reader(_trajs()), _Tok(), _cfg(), question_kg_index=index)
    assert len(rows) == 100


def test_absent_index_aborts():
    """The PPO(1)/PPO(2) failure: wrong question set entirely."""
    with pytest.raises(ValueError, match="ABSENT"):
        P._prepare_prompts(_Reader(_trajs()), _Tok(), _cfg(),
                           question_kg_index={"unrelated": _KG})


def test_absent_error_names_the_rebuild_command():
    """A guard that stops a 4 h run must say how to fix it, not just that it failed."""
    with pytest.raises(ValueError) as ei:
        P._prepare_prompts(_Reader(_trajs()), _Tok(), _cfg(),
                           question_kg_index={"unrelated": _KG})
    msg = str(ei.value)
    assert "06_build_question_kg_index.py" in msg
    assert "--silver SILVER.jsonl" in msg      # the file this run actually reads
    assert "--max_keep 12" in msg              # must match ppo_max_kg_triples
    assert "question_kg_index_path" in msg


def test_cap_of_one_restores_warn_only():
    """The documented escape hatch has to actually work."""
    rows = P._prepare_prompts(_Reader(_trajs()), _Tok(), _cfg(cap=1.0),
                              question_kg_index={"unrelated": _KG})
    assert len(rows) == 100


def test_empty_alone_never_aborts_even_when_large():
    """Empty is never evidence of a wrong artefact, at any rate."""
    index = {f"q{i}": [] for i in range(100)}   # 100% covered, 100% empty
    rows = P._prepare_prompts(_Reader(_trajs()), _Tok(), _cfg(), question_kg_index=index)
    assert len(rows) == 100
