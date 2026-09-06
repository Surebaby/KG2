import json

import pytest

from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_2wiki_proofkg_official_raw_v2 import (
    DATASET,
    DEFAULT_REPLAY,
    OUTPUT_FIELD_WHITELIST,
    _select_family_first,
    select_official_raw_candidates,
    sha256_file,
    validate_protected_ledger_release,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def test_official_raw_freezer_defaults_to_clean_replay_v2():
    assert str(DEFAULT_REPLAY).endswith(
        "sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/"
        "silver_train.jsonl"
    )


def _raw(qid: str, question: str, qtype: str = "inference") -> dict:
    return {
        "id": qid,
        "question": question,
        "golden_answers": ["must not leak"],
        "supporting_facts": {"title": ["must not leak"]},
        "evidences": {"fact": ["must not leak"]},
        "metadata": {"type": qtype, "context": {"must": "not leak"}},
    }


def _assignment(raw: dict, *, split: str = "train") -> dict:
    qid = raw["id"]
    return {
        "question_key": f"{DATASET}::{qid}",
        "dataset": DATASET,
        "qid": qid,
        "split": split,
        "family_sha256": "historical-assignment-family-" + qid,
    }


def _identity(qid: str, question: str) -> dict:
    return {
        "dataset": DATASET,
        "qid": qid,
        "question": question,
        "question_sha256": question_sha256(question),
        "family_sha256": family_sha256(question),
    }


def _select(source, *, protected=(), replay=(), old=(), quota=1, seed=42):
    return select_official_raw_candidates(
        source_rows=source,
        assignment_rows=[_assignment(row) for row in source],
        protected_rows=list(protected),
        replay_rows=list(replay),
        old_train_rows=list(old),
        quotas={"inference": quota},
        seed=seed,
    )


def test_complete_ledger_and_replay_exclude_current_family():
    protected = _identity("protected", "Who is the mother of Alice?")
    replay = _identity("replay", "Where was Charles born?")
    source = [
        _raw("same-protected-family", "Who is the mother of Bob?"),
        _raw("same-replay-family", "Where was Diana born?"),
        _raw("safe", "When was Example One founded?"),
    ]

    selected, telemetry = _select(
        source, protected=[protected], replay=[replay]
    )

    assert [row["qid"] for row in selected] == ["safe"]
    assert telemetry["exclusion_counts_first_match"]["protected_family_sha256"] == 1
    assert telemetry["exclusion_counts_first_match"]["replay_family_sha256"] == 1


def test_old_training_sources_exclude_exact_identity_but_allow_family_reuse():
    old = _identity("old", "Who is the mother of Alice?")
    exact_hash = _raw("different-qid", "Who is the mother of Alice?")
    same_family = _raw("same-family", "Who is the mother of Bob?")
    safe = _raw("safe", "When was Example Two founded?")

    selected, telemetry = _select(
        [exact_hash, same_family, safe], old=[old], quota=2
    )

    assert {row["qid"] for row in selected} == {"same-family", "safe"}
    assert telemetry["exclusion_counts_first_match"][
        "old_train_exact_question_sha256"
    ] == 1


def test_family_first_precedes_repeats_and_is_deterministic():
    source = [
        _raw("family-a-1", "Who is the mother of Alice?"),
        _raw("family-a-2", "Who is the mother of Bob?"),
        _raw("family-b", "When was Example Three founded?"),
    ]
    rows = [
        {
            "qid": row["id"],
            "family_sha256": family_sha256(row["question"]),
        }
        for row in source
    ]
    first = _select_family_first(rows, n=2, qtype="inference", seed=7)
    second = _select_family_first(
        list(reversed(rows)), n=2, qtype="inference", seed=7
    )

    assert first == second
    assert len({row["family_sha256"] for row in first}) == 2


def test_output_projection_is_strictly_gold_free_and_deterministic():
    source = [
        _raw("q1", "When was Example One founded?"),
        _raw("q2", "Where was Example Two born?"),
        _raw("q3", "Who directed Example Three?"),
    ]
    first, first_telemetry = _select(source, quota=2, seed=9)
    second, second_telemetry = select_official_raw_candidates(
        source_rows=list(reversed(source)),
        assignment_rows=list(reversed([_assignment(row) for row in source])),
        protected_rows=[],
        replay_rows=[],
        old_train_rows=[],
        quotas={"inference": 2},
        seed=9,
    )

    assert first == second
    assert first_telemetry == second_telemetry
    assert all(set(row) == OUTPUT_FIELD_WHITELIST for row in first)
    assert all(row["gold_access"] is False for row in first)
    assert not any("golden_answers" in row for row in first)


def test_non_train_rows_are_not_eligible_and_quota_fails_closed():
    raw = _raw("q1", "When was Example One founded?")
    with pytest.raises(ValueError, match="need 1"):
        select_official_raw_candidates(
            source_rows=[raw],
            assignment_rows=[_assignment(raw, split="dev")],
            protected_rows=[],
            replay_rows=[],
            old_train_rows=[],
            quotas={"inference": 1},
        )


def test_insufficient_family_repeat_capacity_fails_closed():
    rows = [{"qid": "q1", "family_sha256": "f1"}]
    with pytest.raises(ValueError, match="need 2"):
        _select_family_first(rows, n=2, qtype="inference", seed=42)


def test_protected_ledger_report_must_bind_hash_and_current_family(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(_identity("q1", "When was Example One founded?")) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "complete": True,
                "current_family_recomputed": True,
                "output": {"rows": 1, "sha256": sha256_file(ledger)},
            }
        ),
        encoding="utf-8",
    )
    loaded = validate_protected_ledger_release(
        ledger_path=ledger, report_path=report
    )
    assert loaded["complete"] is True

    bad = json.loads(report.read_text(encoding="utf-8"))
    bad["output"]["sha256"] = "0" * 64
    report.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        validate_protected_ledger_release(ledger_path=ledger, report_path=report)
