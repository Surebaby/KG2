import json
from types import SimpleNamespace

import pytest

from kgproweight.data.silver_split import SplitSpec, assign_split
from scripts.prepare import freeze_sft_v3_protected_ledger_v1 as ledger


def row(qid="a", question="Who directed Silver Spring?", dataset="hotpotqa", **extra):
    return {"dataset": dataset, "qid": qid, "question": question, **extra}


def source(tmp_path, rows, name="input.jsonl", **kwargs):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return ledger.SourceSpec(path, "protected", ledger.sha_file(path), **kwargs)


def test_projection_never_copies_gold_or_evidence(tmp_path):
    spec = source(tmp_path, [row(answer="SECRET_ANSWER", metadata={"gold": "SECRET_GOLD"},
                                 supporting_facts=["SECRET_EVIDENCE"], family_sha256="historical_stale")])
    rows, inventory = ledger.merge_sources([spec])
    text = json.dumps(rows)
    assert "SECRET_" not in text
    assert rows[0]["family_sha256"] == ledger.family_sha256(rows[0]["question"])
    assert inventory[0]["counts"]["stored_family_mismatch_rows_recomputed"] == 1
    assert rows[0]["sources"][0]["line_number"] == 1


@pytest.mark.parametrize("extra", [{"id": "b"}, {"dataset_name": "musique"},
                                   {"question_sha256": "0" * 64}, {"question_key": "hotpotqa::b"}])
def test_ambiguous_or_inconsistent_identity_fails(extra):
    with pytest.raises(ValueError):
        ledger.identity(row(**extra))


def test_dataset_override_must_agree():
    with pytest.raises(ValueError):
        ledger.identity(row(), "musique")


def test_one_qid_two_questions_fails_before_success_manifest(tmp_path):
    spec = source(tmp_path, [row(), row(question="Where is the river located?")])
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="conflicting questions"):
        ledger.freeze_ledger(output_dir=out, specs=[spec], experiment_id="test")
    assert (out / "FAILED.json").is_file()
    assert (out / "protocol.json").is_file()
    assert not (out / "manifest.json").exists()


def test_cross_source_exact_question_aliases_are_all_preserved(tmp_path):
    a = source(tmp_path, [row()], name="a.jsonl")
    b = source(tmp_path, [row(qid="b")], name="b.jsonl")
    out = tmp_path / "output"
    report = ledger.freeze_ledger(output_dir=out, specs=[a, b], experiment_id="test")
    assert report["protected_dataset_qids"] == 2
    assert report["protected_exact_question_multi_qid_groups"] == 1
    aliases = json.loads((out / "exact_question_aliases.question_only.jsonl").read_text())
    assert aliases["qids"] == ["a", "b"]


def test_qid_dataset_scoped_but_question_and_family_global():
    index = ledger.make_index([row()])
    assert ledger.overlap_reasons(row(dataset="musique"), index) == ["question_sha256", "family_sha256"]
    same_template = row(qid="b", dataset="2wikimultihopqa", question="Who directed Autumn Meadow?")
    assert ledger.overlap_reasons(same_template, index) == ["family_sha256"]
    unrelated = row(dataset="musique", question="when did this happen and why?")
    assert not ledger.overlap_reasons(unrelated, index)


def test_historical_train_not_protected_and_both_holdout_strata_protected(tmp_path):
    raw = [row(qid=str(i), question=f"what unusual thing happened at site {i}?", accepted=bool(i % 2)) for i in range(100)]
    expected = [r for r in raw if assign_split(SimpleNamespace(**{k: r[k] for k in ("qid", "question", "accepted")}), SplitSpec()) != "train"]
    assert {r["accepted"] for r in expected} == {False, True}
    spec = source(tmp_path, raw, historical_holdout_only=True)
    protected, _ = ledger.merge_sources([spec])
    assert {r["qid"] for r in protected} == {r["qid"] for r in expected}
    assert all(":test" in r["source_roles"][0] or ":val" in r["source_roles"][0] for r in protected)


def test_historical_split_cannot_coerce_string_accepted(tmp_path):
    spec = source(tmp_path, [row(accepted="false")], historical_holdout_only=True)
    with pytest.raises(ValueError, match="Boolean"):
        ledger.merge_sources([spec])


def test_source_bytes_and_count_are_enforced(tmp_path):
    spec = source(tmp_path, [row()], expected_rows=2)
    with pytest.raises(ValueError, match="row count"):
        ledger.merge_sources([spec])
    spec.path.write_text(spec.path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA"):
        ledger.merge_sources([spec])


def test_capacity_refuses_duplicate_qid_in_raw_source(tmp_path):
    spec = source(tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate raw"):
        ledger.capacity_audit(spec.path, "hotpotqa", [])


def test_capacity_counts_aliases_without_silent_overwrite(tmp_path):
    spec = source(tmp_path, [row(), row(qid="b")])
    report = ledger.capacity_audit(spec.path, "hotpotqa", [])
    assert report["counts"]["safe_rows_before_internal_dedup"] == 2
    assert report["safe_unique_current_families"] == 1
    assert report["raw_exact_question_multi_qid_groups"] == 1


def test_ledger_is_append_only_and_manifest_replays(tmp_path):
    spec = source(tmp_path, [row()])
    out = tmp_path / "output"
    ledger.freeze_ledger(output_dir=out, specs=[spec], experiment_id="test")
    manifest = json.loads((out / "manifest.json").read_text())
    for name, info in manifest["outputs"].items():
        assert ledger.sha_file(out / name) == info["sha256"]
    with pytest.raises(FileExistsError):
        ledger.freeze_ledger(output_dir=out, specs=[spec], experiment_id="test")


def test_wrong_historical_fold_counts_fail_closed(tmp_path):
    spec = source(tmp_path, [row(accepted=True)], historical_holdout_only=True)
    with pytest.raises(ValueError, match="historical fold replay"):
        ledger.freeze_ledger(output_dir=tmp_path / "out", specs=[spec], experiment_id="test",
                             expected_historical_fold_counts={"train": 100, "val": 100, "test": 100})
