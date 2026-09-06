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


class _CharTok:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kw):  # noqa: ARG002
        return {"input_ids": [ord(ch) for ch in text]}


_KG = [("a", "r", "b")]


def _cfg(cap=0.05, exact=False):
    return types.SimpleNamespace(
        ppo_min_kg_triples=5, ppo_max_kg_triples=12,
        ppo_max_passages=5, max_input_length=6144,
        max_kg_index_miss_rate=cap, silver_path="SILVER.jsonl",
        require_exact_kg_index_alignment=exact,
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
    assert "--min_keep 5" in msg
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


def test_silver_fallback_uses_the_same_min_and_max_budget(monkeypatch):
    seen = {}

    def fake_filter(triples, *, question, min_keep, max_keep):
        seen.update(question=question, min_keep=min_keep, max_keep=max_keep)
        return triples

    monkeypatch.setattr(P, "filter_and_rank_triples", fake_filter)
    rows = P._prepare_prompts(
        _Reader([_Traj("q0", _KG)]), _Tok(), _cfg(), question_kg_index={"q0": []}
    )
    assert len(rows) == 1
    assert seen == {"question": "q0", "min_keep": 5, "max_keep": 12}


def test_identity_safe_record_mode_preserves_short_proof_kg(monkeypatch):
    """A pre-applied dataset::qid KG must bypass the legacy text-index path."""
    proof = [("bridge", "relation", "answer")]

    def forbidden_filter(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("short Proof-KG was incorrectly sent through legacy filtering")

    monkeypatch.setattr(P, "filter_and_rank_triples", forbidden_filter)
    rows = P._prepare_prompts(
        _Reader([_Traj("q0", proof)]),
        _CharTok(),
        _cfg(cap=0.0),
        question_kg_index=None,
    )

    assert rows[0]["spec"].kg_subgraph == proof
    assert "(bridge, relation, answer)" in rows[0]["prompt"]


def test_exact_alignment_rejects_covered_but_different_triples():
    with pytest.raises(ValueError, match="differ from stored silver KG"):
        P._prepare_prompts(
            _Reader([_Traj("q0", _KG)]),
            _Tok(),
            _cfg(exact=True),
            question_kg_index={"q0": [("different", "r", "triple")]},
        )


def test_exact_alignment_accepts_identical_ordered_triples():
    rows = P._prepare_prompts(
        _Reader([_Traj("q0", _KG)]),
        _Tok(),
        _cfg(exact=True),
        question_kg_index={"q0": list(_KG)},
    )
    assert len(rows) == 1


def test_prompt_drops_passages_instead_of_truncating_trailing_kg():
    traj = _Traj("q0", _KG)
    traj.retrieved_passages = [{"contents": "x" * 5000}]
    cfg = _cfg(exact=True)
    cfg.max_input_length = 1600

    rows = P._prepare_prompts(
        _Reader([traj]), _CharTok(), cfg, question_kg_index={"q0": list(_KG)},
    )

    assert rows[0]["num_passages"] == 0
    assert rows[0]["spec"].retrieved_passages == []
    assert "[Knowledge Graph Context]" in rows[0]["prompt"]
    assert "(a, r, b)" in rows[0]["prompt"]
    assert rows[0]["prompt_tokens"] <= cfg.max_input_length


def test_prompt_refuses_to_truncate_kg_when_zero_passages_still_overflow():
    traj = _Traj("q0", _KG)
    cfg = _cfg(exact=True)
    cfg.max_input_length = 10

    with pytest.raises(ValueError, match="refusing to right-truncate"):
        P._prepare_prompts(
            _Reader([traj]), _CharTok(), cfg,
            question_kg_index={"q0": list(_KG)},
        )
