from copy import deepcopy
from collections import Counter
import json
import subprocess
import sys

import pytest

from scripts.pilot import probe_generation_length_384_512_v1 as probe


def population():
    return [{"dataset": d, "qid": f"{d}-{g}-{i}", "question_key": f"{d}::{d}-{g}-{i}",
             "m_graph": g, "family_sha256": f"family-{d}-{g}-{i}"}
            for d, g, n in probe.QUOTAS for i in range(n + 4)]


def test_selection_quotas_and_no_quality_dependence():
    rows = population()
    selected = probe.select_inputs(rows)
    assert Counter((r["dataset"], r["m_graph"]) for r in selected) == {(d, g): n for d, g, n in probe.QUOTAS}
    changed = deepcopy(rows)
    for i, row in enumerate(changed):
        row.update(trajectory_valid=bool(i % 2), em=1, reward=99-i)
    assert [r["qid"] for r in selected] == [r["qid"] for r in probe.select_inputs(list(reversed(changed)))]
    assert len({r["family_sha256"] for r in selected}) == 60


def test_duplicate_identity_and_missing_unique_families_rejected():
    rows = population()
    with pytest.raises(ValueError, match="duplicate"):
        probe.select_inputs(rows+[rows[0]])
    for r in rows:
        r["family_sha256"] = "one-family"
    with pytest.raises(ValueError, match="unique-family"):
        probe.select_inputs(rows)


def pred(tokens):
    return {"raw_response_token_ids": tokens}


def test_old_eos_requires_whole_sequence_equality():
    old = pred([1, 3, 128009])
    assert probe.prefix_check(old, old)["match"]
    assert not probe.prefix_check(old, pred([1, 3, 128009, 4]))["match"]
    assert not probe.prefix_check(old, pred([1, 4, 128009]))["match"]


def test_old_cap_requires_all_384_tokens_even_when_new_eos_at_384():
    old = pred([3]*384)
    assert probe.prefix_check(old, pred([3]*384+[4, 128009]))["match"]
    assert not probe.prefix_check(old, pred([3]*383+[128009]))["match"]
    assert not probe.prefix_check(old, pred([3]*383))["match"]


def test_malformed_baseline_rejected():
    with pytest.raises(ValueError, match="neither EOS"):
        probe.prefix_check(pred([3]), pred([3]))
    with pytest.raises(ValueError, match="after baseline EOS"):
        probe.prefix_check(pred([128009, 3]), pred([128009, 3]))


def test_paired_summary_counts_recovery_regression_and_token_cost():
    rows=[]
    for i,(old,new) in enumerate([(False,True),(True,False),(True,True),(False,False)]):
        rows.append({"question_key": str(i//2), "format_384": {"valid":old,"violations":[] if old else ["invalid"]},
                     "format_512": {"valid":new,"violations":[] if new else ["invalid"]}, "tokens_384":384,"tokens_512":400,
                     "prefix":{"match":True},"cap_384":True,"cap_512":False,"eos_384":False,"eos_512":True})
    r=probe.paired_summary(rows)
    assert r["candidates"]==4 and r["questions"]==2
    assert r["invalid_to_valid"]==r["valid_to_invalid"]==r["valid_to_valid"]==r["invalid_to_invalid"]==1
    assert r["token_delta"]==64 and r["prefix_matches"]==4


def test_failed_restart_keeps_existing_scientific_output(tmp_path):
    old = tmp_path / "generations_512.jsonl"
    original = b'{"candidate_id":"retained-failed-candidate"}\n'
    old.write_bytes(original)
    result = subprocess.run([sys.executable, "-m", "scripts.pilot.probe_generation_length_384_512_v1", "prepare",
                             "--parent-dir", str(tmp_path / "missing-parent"), "--output-dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert old.read_bytes() == original
    failure = json.loads((tmp_path / "exception.json").read_text())
    assert failure["status"] == "FAILED_OUTPUTS_RETAINED"
    assert failure["optimizer_updates"] == 0
