from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.prepare.materialize_sft_replay_legacy_strong_v4_clean_v2 as replay
from kgproweight.data.silver_dataset import SilverTrajectory
from kgproweight.kg.question_kg import question_sha256
from kgproweight.training.phase3_sft import _render_assistant_trace
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


def _row(qid: str, question: str) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer": "Answer",
        "dataset": "hotpotqa",
        "accepted": True,
        "steps": [
            {
                "index": index,
                "text": (
                    f"Reasoning: reason {index}.\n"
                    "Knowledge Used: []\n"
                    f"Conclusion: conclusion {index}."
                ),
                "label": 0.0,
                "cited_triples": [],
            }
            for index in range(1, 4)
        ],
        "kg_subgraph": [],
        "retrieved_passages": [
            {"id": "p1", "source": "unit", "contents": "evidence"}
        ],
        "metadata": {"gold_answer": "Answer"},
        "unknown_extension": {"must": "survive"},
    }


def _identity_row(qid: str, question: str) -> dict:
    return {
        "dataset": "hotpotqa",
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
        "family_version": FAMILY_VERSION,
    }


def _candidate(qid: str, question: str, *, seed: int = 42) -> replay.Candidate:
    raw = _row(qid, question)
    payload = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode()
    trajectory = SilverTrajectory.from_dict(raw)
    identity = replay._identity_from_row(
        raw, label="unit", require_stored_hashes=False
    )
    rendered = _render_assistant_trace(trajectory)
    return replay.Candidate(
        trajectory=trajectory,
        identity=identity,
        source_line_number=1,
        source_row_bytes=payload,
        source_row_sha256=hashlib.sha256(payload).hexdigest(),
        rendered_steps=3,
        rendered_trace_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        selection_rank_sha256=replay._selection_rank(identity, seed),
    )


class _CharTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        value = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            value += "<assistant>"
        return value

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


def test_identity_index_recomputes_current_family_and_rejects_stale_hash():
    question = "Where was Alice born?"
    row = _identity_row("q1", question)
    index = replay.IdentityIndex.from_rows([row], label="protected")
    probe = replay._identity_from_row(
        _row("q2", "Where was Bob born?"),
        label="probe",
        require_stored_hashes=False,
    )
    assert index.matches(probe) == {
        "qid": False,
        "question_sha256": False,
        "family_sha256": True,
    }

    row["family_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale/malformed"):
        replay.IdentityIndex.from_rows([row], label="protected")


def test_collect_candidates_blocks_qid_hash_family_and_hm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_rows = [
        _row("blocked-qid", "A unique qid-block question?"),
        _row("hash-source", "What did Exact Entity write?"),
        _row("family-source", "Where was Bob born?"),
        _row("hm-source", "Which film did HM Entity direct?"),
        _row("safe", "How tall is the uniquely named tower?"),
    ]
    source = tmp_path / "source.jsonl"
    source.write_bytes(
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
            for row in source_rows
        )
    )
    protected_rows = [
        _identity_row("blocked-qid", "Does a separate protected item exist?"),
        _identity_row("different-hash-qid", "What did Exact Entity write?"),
        _identity_row("different-family-qid", "Where was Alice born?"),
    ]
    hm_rows = [
        _identity_row("different-hm-qid", "Which film did HM Entity direct?")
    ]
    monkeypatch.setattr(replay, "assign_split", lambda trajectory, spec: "train")

    candidates, report = replay.collect_candidates(
        source=source,
        protected=replay.IdentityIndex.from_rows(
            protected_rows, label="protected"
        ),
        hm_population=replay.IdentityIndex.from_rows(hm_rows, label="hm"),
        seed=42,
    )

    assert [candidate.identity["qid"] for candidate in candidates] == ["safe"]
    assert report["source_flow"]["rejected_protected_ledger"] == 3
    assert report["source_flow"]["rejected_hm_population"] == 1
    assert report["overlap_all_accepted_train"]["protected"]["qid"] == 1
    assert report["overlap_all_accepted_train"]["protected"][
        "question_sha256"
    ] == 1
    assert report["overlap_all_accepted_train"]["protected"][
        "family_sha256"
    ] >= 2


def test_selection_is_deterministic_family_unique_and_tokenizer_safe():
    candidates = [
        _candidate("q1", "Who wrote Alpha?"),
        _candidate("q2", "Who wrote Beta?"),  # same current lexical family
        _candidate("q3", "How tall is the uniquely named tower?"),
    ]
    candidates.sort(key=lambda row: row.selection_rank_sha256)

    first, first_rows, first_report = replay.select_candidates(
        candidates,
        n_samples=2,
        tokenizer=_CharTokenizer(),
        max_input_length=100_000,
        max_new_tokens=100_000,
    )
    second, second_rows, _ = replay.select_candidates(
        candidates,
        n_samples=2,
        tokenizer=_CharTokenizer(),
        max_input_length=100_000,
        max_new_tokens=100_000,
    )

    assert [row.identity["qid"] for row in first] == [
        row.identity["qid"] for row in second
    ]
    assert first_rows == second_rows
    assert len({row.identity["family_sha256"] for row in first}) == 2
    assert first_report["tokenizer"]["rows_checked"] == 2
    assert first_report["tokenizer"]["min_assistant_tokens"] > 0


def test_selection_refuses_when_fewer_than_required():
    with pytest.raises(ValueError, match="only 1 valid isolated replay rows"):
        replay.select_candidates(
            [_candidate("q1", "What is the only safe question?")],
            n_samples=2,
            tokenizer=_CharTokenizer(),
            max_input_length=100_000,
            max_new_tokens=100_000,
        )


def test_output_preserves_source_rows_bytewise(tmp_path: Path):
    candidate = _candidate("q1", "What survives byte for byte?")
    output = tmp_path / "release"
    silver_path, selection_path = replay._write_outputs(
        output_dir=output,
        selected=[candidate],
        selection_rows=[{"qid": "q1"}],
    )

    assert silver_path.read_bytes() == candidate.source_row_bytes + b"\n"
    assert json.loads(silver_path.read_text())["unknown_extension"] == {
        "must": "survive"
    }
    assert json.loads(selection_path.read_text()) == {"qid": "q1"}
