"""Independent synthetic checks; no real candidate or label file is opened."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.pilot import assess_training_format_prompt_v1 as assessment


def _cohort():
    return [
        {"question_key": f"{dataset}::{i}", "candidate_index": k,
         "dataset": dataset, "family_sha256": f"family-{dataset}-{i}",
         "question_sha256": f"question-{dataset}-{i}"}
        for dataset in ("hotpotqa", "musique", "2wikimultihopqa")
        for i in range(20) for k in (0, 1)
    ]


def test_complete_cohort_requires_only_identity_and_never_answer_fields():
    rows = _cohort()
    before = deepcopy(rows)
    assessment.validate_cohort(rows, {d: 20 for d in ("hotpotqa", "musique", "2wikimultihopqa")})
    assert rows == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate_index", "duplicate_family", "duplicate_question", "pair_metadata", "wrong_quota"])
def test_cohort_malformed_before_gold_is_rejected(mutation):
    rows = _cohort()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate_index":
        rows[1]["candidate_index"] = 0
    elif mutation in ("duplicate_family", "duplicate_question"):
        field = "family_sha256" if mutation == "duplicate_family" else "question_sha256"
        rows[2][field] = rows[3][field] = rows[0][field]
    elif mutation == "pair_metadata":
        rows[1]["question_sha256"] = "another question"
    else:
        rows[0]["dataset"] = rows[1]["dataset"] = "musique"
    with pytest.raises(ValueError):
        assessment.validate_cohort(rows, {d: 20 for d in ("hotpotqa", "musique", "2wikimultihopqa")})


def _freeze_fixture(tmp_path):
    output = tmp_path / "assessment"
    output.mkdir()
    placeholder = tmp_path / "identity_only.json"
    placeholder.write_text("{}")
    gold = tmp_path / "never_open_gold.jsonl"
    protocol = {
        "baseline_binding": assessment.bank.identity(placeholder),
        "input_binding": assessment.bank.identity(placeholder),
        "selection_binding": assessment.bank.identity(placeholder),
        "frozen_metric_code": {},
        "gold": {"binding": {"path": str(gold), "sha256": "0" * 64}},
    }
    assessment.bank.write_json(output / "protocol.json", protocol)
    assessment.bank.write_json(output / "prepared.json", {"protocol": assessment.bank.identity(output / "protocol.json")})
    assessment.freeze(output)
    return output, gold


def test_freeze_is_append_only_and_preserves_exact_executed_source(tmp_path):
    output, gold = _freeze_fixture(tmp_path)
    assert (output / "assessment.executed.py").read_bytes() == Path(assessment.__file__).read_bytes()
    assert not gold.exists()
    before = (output / "scoring_code_freeze.json").read_bytes()
    with pytest.raises(FileExistsError):
        assessment.freeze(output)
    assert (output / "scoring_code_freeze.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["running", "missing_output", "all_retained_false", "exception", "failed"])
def test_incomplete_or_failed_release_never_opens_gold(tmp_path, monkeypatch, mutation):
    output, gold = _freeze_fixture(tmp_path)
    generation = tmp_path / "generation"
    generation.mkdir()
    required = {"inputs.jsonl", "legacy_inputs.jsonl", "baseline_384.jsonl", "generations_prompt_v1.jsonl",
                "selection.question_only.jsonl", "protocol.json", "prepared.json", "execution_environment.json"}
    release = {"status": "COMPLETE_DEVELOPMENT_ONLY", "all120_candidates_retained": True,
               "outputs": {name: {} for name in required}}
    if mutation == "running":
        release["status"] = "GENERATING"
    elif mutation == "missing_output":
        release["outputs"].pop("generations_prompt_v1.jsonl")
    elif mutation == "all_retained_false":
        release["all120_candidates_retained"] = False
    else:
        (generation / ("exception.json" if mutation == "exception" else "FAILED.json")).write_text("{}")
    monkeypatch.setattr(assessment.bank, "load_release", lambda *_: release)
    real_sha = assessment.bank.file_sha
    def checked_sha(path):
        if Path(path) == gold:
            raise AssertionError("Gold was touched before successful complete release")
        return real_sha(path)
    monkeypatch.setattr(assessment.bank, "file_sha", checked_sha)
    monkeypatch.setattr(assessment.bank, "read_rows", lambda *_: pytest.fail("Rows must not be read before release admission"))
    with pytest.raises(ValueError, match="complete successful generation release"):
        assessment.assess(generation, output)
    assert not (output / "before_gold.json").exists()


def test_modified_frozen_evaluator_is_rejected_before_candidate_or_gold_read(tmp_path, monkeypatch):
    output, _ = _freeze_fixture(tmp_path)
    with (output / "assessment.executed.py").open("a") as stream:
        stream.write("\n# changed after freeze\n")
    monkeypatch.setattr(assessment.bank, "load_release", lambda *_: pytest.fail("Candidate release opened after invalid code freeze"))
    with pytest.raises(ValueError, match="frozen binding changed"):
        assessment.assess(tmp_path / "generation", output)
