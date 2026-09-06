"""Independent identity, exact reuse and train isolation review; no outcome labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.prepare import normalization_representative_bank_v1 as supplement
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2

bank = supplement.bank


def review(directory):
    protocol = supplement.verify(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, bound in manifest["outputs"].items():
        assert bank.file_sha(directory / name) == bound["sha256"]
    rows = bank.read_rows(directory / "normalization_rows.jsonl")
    inputs = bank.read_rows(directory / "inputs.all.jsonl")
    input_index = {bank.key(row): row for row in inputs}
    assert len(rows) == 240 and len(inputs) == len(input_index) == 120
    assert len({row["candidate_id"] for row in rows}) == 240
    assert Counter(row["dataset"] for row in rows) == {name: 80 for name in bank.DATASETS}
    assert Counter((row["dataset"], row["m_graph"]) for row in inputs) == {
        ("hotpotqa", 0): 40, ("musique", 0): 40, ("2wikimultihopqa", 1): 32, ("2wikimultihopqa", 0): 8}
    expected_order = [f"{bank.key(row)}::k{k}" for row in inputs for k in range(2)]
    assert [row["candidate_id"] for row in rows] == expected_order
    assignments = bank.read_rows(supplement.OLD_FIT / "assignments.jsonl")
    consumed = {row["family_sha256"] for row in assignments if row["split"] != "train"}
    assert not consumed & {row["family_sha256"] for row in rows}
    assert len({row["family_sha256"] for row in rows}) == 120
    parent = bank.load_release(supplement.OLD_INPUT, bank.PREPARE_VERSION)
    ledger = bank.resolve(parent["source_bindings"]["protected_ledger"], supplement.OLD_INPUT)
    overlap = bank.isolation(inputs, bank.read_rows(ledger))
    old_scores = {row["candidate_id"]: row for row in bank.read_rows(supplement.OLD_SCORED / "candidates.scored.jsonl")}
    old_inputs = {bank.key(row): row for row in bank.read_rows(supplement.OLD_INPUT / "inputs.jsonl")}
    old_gens = {row["candidate_id"]: row for row in bank.read_rows(supplement.OLD_GENERATED / "generations.jsonl")}
    new_gens = {row["candidate_id"]: row for row in bank.read_rows(directory / "generated/generations.jsonl")}
    new_scores = {row["candidate_id"]: row for row in bank.read_rows(directory / "new_scored.full.jsonl")}
    new_manifest = json.loads((directory / "new_inputs/manifest.json").read_text())
    assert len(new_gens) == len(new_scores) == 146
    reused = 0
    for row in rows:
        bank.assert_gold_free(row)
        source = input_index[bank.key(row)]
        assert row["split"] == "train" and row["normalization_train_only"] is True
        assert row["family_sha256"] == source["family_sha256"]
        assert row["question_sha256"] == source["question_sha256"]
        assert row["input_sha256"] == source["input_sha256"]
        if row["origin"] == "exact_frozen_train_candidate_reuse":
            reused += 1
            original, generated = old_scores[row["candidate_id"]], old_gens[row["candidate_id"]]
            assert source == old_inputs[bank.key(row)]
        else:
            original, generated = new_scores[row["candidate_id"]], new_gens[row["candidate_id"]]
            k = int(row["candidate_id"].rsplit("::k", 1)[1])
            assert generated["seed"] == bank.candidate_seed(42, bank.key(row), k)
            assert generated["bank_manifest_sha256"] == bank.file_sha(directory / "new_inputs/manifest.json")
            assert generated["generation_contract_sha256"] == bank.digest(protocol["generation"])
            assert generated["policy_sha256"] == new_manifest["source_bindings"]["policy"]["sha256"]
            assert generated["base_model_identity_sha256"] == bank.digest(protocol["models"]["base_model"])
        assert row["raw_text"] == original["raw_text"]
        assert row["trajectory_valid"] is original["trajectory_valid"]
        assert row["format_validation"] == original["format_validation"]
        assert row["score_row_sha256"] == bank.digest(original)
        assert row["generation_sha256"] == bank.digest(generated)
        assert original["generation"] == generated["generation"]
    assert reused == 94
    stats = fit_text_normalization_v2(rows)
    result = {"schema_version": "normalization-supplement-independent-review-v1", "status": "PASS",
        "counts": {"questions": 120, "candidates": 240, "new": 146, "exact_reused": 94},
        "all_rows_retained": True, "new_generation_seed_input_model_contracts_verified": True,
        "exact_original_generation_and_score_reuse": True, "all_rows_normalization_train_only": True,
        "protected_overlap": overlap, "consumed_calibration_confirmation_family_overlap": 0,
        "gold_used": False, "policy_optimizer_updates": 0, "fixed_text_v2_stats": stats,
        "manifest": bank.identity(directory / "manifest.json"), "review_code": bank.identity(Path(__file__))}
    bank.write_json(directory / "independent_review.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=supplement.DEFAULT)
    result = review(parser.parse_args().bank_dir.resolve())
    print(json.dumps(result, indent=2))
