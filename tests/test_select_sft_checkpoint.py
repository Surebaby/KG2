from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.eval.select_sft_checkpoint as selector


def _write(path: Path, *, ems: list[int], hidden: list[bool], parsed: list[bool]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for index, (em, is_hidden, well_formed) in enumerate(zip(ems, hidden, parsed)):
            fh.write(json.dumps({
                "qid": f"q{index}",
                "em": em,
                "f1": float(em),
                "well_formed": well_formed,
                "gold_in_passages": not is_hidden,
            }) + "\n")


def test_selects_earliest_passing_candidate(monkeypatch, tmp_path: Path):
    first = tmp_path / "step40.jsonl"
    second = tmp_path / "step80.jsonl"
    hidden = [True, True, False, False]
    _write(first, ems=[0, 1, 1, 1], hidden=hidden, parsed=[True] * 4)
    _write(second, ems=[1, 1, 1, 1], hidden=hidden, parsed=[True] * 4)
    output = tmp_path / "selection.json"
    monkeypatch.setattr(sys, "argv", [
        "select_sft_checkpoint.py",
        "--candidate", f"step40={first}",
        "--candidate", f"step80={second}",
        "--output", str(output),
        "--expected_n", "4",
        "--min_parse_rate", "1.0",
        "--min_em", "0.75",
        "--min_hidden_em", "0.5",
    ])
    assert selector.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected"] == "step40"
    assert report["status"] == "PASS"


def test_rejects_mismatched_candidate_cohort(monkeypatch, tmp_path: Path):
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    _write(first, ems=[1], hidden=[True], parsed=[True])
    second.write_text(json.dumps({
        "qid": "different", "em": 1, "f1": 1, "well_formed": True,
        "gold_in_passages": False,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "select_sft_checkpoint.py",
        "--candidate", f"one={first}",
        "--candidate", f"two={second}",
        "--output", str(tmp_path / "out.json"),
        "--expected_n", "1",
    ])
    try:
        selector.main()
    except ValueError as exc:
        assert "qid order/cohort differs" in str(exc)
    else:
        raise AssertionError("expected mismatched qid cohort to be rejected")
