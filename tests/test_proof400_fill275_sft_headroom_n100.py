from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from kgproweight.reward.proofkg_process import is_automatic_proofkg
from scripts.pilot.audit_proof400_fill275_sft_headroom import (
    software_environment,
    summarize_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration"
SOURCE_DATA = ROOT / "data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"
QTYPES = {"inference", "comparison", "compositional", "bridge_comparison"}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_cohort_is_fill275_only_balanced_and_unique_family():
    rows = read_jsonl(AUDIT / "cohort.question_only.jsonl")
    assert len(rows) == len({row["qid"] for row in rows}) == 100
    assert Counter(row["question_type"] for row in rows) == Counter({q: 25 for q in QTYPES})
    assert len({row["family_sha256"] for row in rows}) == 100
    assert all(row["source_role"] == "proof400_fill275_expansion" for row in rows)
    assert all(row["train_side_development_consumed"] is True for row in rows)
    assert all(row["evaluation_eligible"] is False and row["gold_access"] is False for row in rows)

    source = {
        row["qid"]: row
        for row in read_jsonl(
            ROOT / "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/proof400.question_only.jsonl"
        )
    }
    assert all(source[row["qid"]]["proof_source"] == "automatic_proofkg_2wiki_train_k4_v1" for row in rows)
    assert not any(source[row["qid"]]["route"].startswith("2wiki_hard_") for row in rows)


def test_prompt_inputs_preserve_original_passages_and_proofkg_without_label_fields():
    prompts = read_jsonl(AUDIT / "prompt_inputs.gold_free.jsonl")
    source = {
        f"{row['dataset']}::{row['qid']}": row
        for row in read_jsonl(SOURCE_DATA / "silver_train.jsonl")
    }
    forbidden = {"answer", "gold_answer", "gold_answers", "supporting_facts", "decomposition", "steps"}
    assert len(prompts) == 100
    for row in prompts:
        assert not forbidden.intersection(row)
        original = source[f"{row['dataset']}::{row['qid']}"]
        assert row["question"] == original["question"]
        assert row["retrieved_passages"] == original["retrieved_passages"]
        assert row["kg_subgraph"] == original["kg_subgraph"]
        assert len(row["retrieved_passages"]) == 10
        assert row["kg_subgraph"]


def test_outcome_labels_are_separate_train_only_and_question_kg_is_complete():
    prompts = {row["qid"]: row for row in read_jsonl(AUDIT / "prompt_inputs.gold_free.jsonl")}
    labels = {row["qid"]: row for row in read_jsonl(AUDIT / "outcome_labels.train_only.jsonl")}
    records = {row["qid"]: row for row in read_jsonl(AUDIT / "question_kg_records.jsonl")}
    assert set(prompts) == set(labels) == set(records)
    for qid, row in prompts.items():
        assert labels[qid]["source_split"] == "train"
        assert labels[qid]["gold_use"] == "post_generation_train_outcome_scoring_only"
        assert labels[qid]["gold_answers"]
        assert records[qid]["question_sha256"] == row["question_sha256"]
        assert records[qid]["kg_subgraph"] == row["kg_subgraph"]
        assert is_automatic_proofkg(records[qid], records[qid]["kg_subgraph"])
        assert records[qid]["provenance"]["gold_access"] is False


def test_protocol_hashes_status_model_and_gates_are_frozen():
    protocol = json.loads((AUDIT / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_NOT_RUN_TRAIN_SIDE_DEVELOPMENT_CONSUMED"
    assert protocol["cohort"]["n"] == 100
    assert protocol["cohort"]["question_type_counts"] == {
        "bridge_comparison": 25,
        "comparison": 25,
        "compositional": 25,
        "inference": 25,
    }
    assert protocol["model"]["adapter_path"] == "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"
    assert protocol["generation"]["total_generations"] == 500
    assert protocol["generation"]["rollouts_per_qid"] == 4
    assert protocol["generation"]["max_new_tokens"] == 384
    assert protocol["generation"]["sampled"] == {
        "do_sample": True,
        "rollouts_per_qid": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    assert protocol["decision_gates"]["sample_valid_rate_min"] == 0.90
    assert protocol["decision_gates"]["oracle_at_4_minus_greedy_em_min"] == 0.05
    assert protocol["decision_gates"]["mixed_outcome_qid_rate_min"] == 0.20
    assert protocol["scientific_boundary"]["training_started"] is False
    assert protocol["scientific_boundary"]["gpu_generation_started"] is False
    assert protocol["scientific_boundary"]["reward_rankability_tested"] is False
    assert protocol["schema_version"] == "proof400-fill275-strong-sft-headroom-protocol-v3"
    assert protocol["software_environment"] == software_environment()
    locked_names = {Path(identity["path"]).name for identity in protocol["model"]["locked_files"].values()}
    assert "config.json" in locked_names
    assert "model.safetensors.index.json" in locked_names
    assert len({name for name in locked_names if name.startswith("model-") and name.endswith(".safetensors")}) == 4
    assert {"adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "special_tokens_map.json"} <= locked_names
    assert any(identity["path"].endswith("/manifest.json") for identity in protocol["model"]["locked_files"].values())
    for identity in protocol["model"]["locked_files"].values():
        path = Path(identity["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert sha256(path) == identity["sha256"]
    for section in ("inputs", "outputs", "code_closure"):
        for identity in protocol[section].values():
            path = Path(identity["path"])
            if not path.is_absolute():
                path = ROOT / path
            assert sha256(path) == identity["sha256"]


def test_v3_runner_reads_nested_sample_settings_and_launcher_targets_v3():
    runner = (ROOT / "scripts/pilot/audit_proof400_fill275_sft_headroom.py").read_text(encoding="utf-8")
    assert 'generation["temperature"]' not in runner
    assert runner.count('generation["sampled"]["temperature"]') == 2
    assert runner.count('generation["sampled"]["top_p"]') == 2
    launcher = (ROOT / "launch_proof400_fill275_sft_headroom_n100_local.sh").read_text(encoding="utf-8")
    assert "v3_preregistration/protocol.json" in launcher
    assert "N100-K4-SEED42-V3" in launcher


def test_failed_v2_is_preserved_and_superseded_without_candidates():
    failed = ROOT / "outputs/validation/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v2"
    manifest = json.loads((failed / "manifest.json").read_text(encoding="utf-8"))
    supersession = json.loads((failed / "supersession.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_RUNTIME"
    assert supersession["status"] == "FAILED_RUNTIME_SUPERSEDED_NO_SCIENTIFIC_RESULT"
    assert supersession["model_generations_completed"] == 0
    assert supersession["optimizer_updates"] == 0
    assert supersession["scientific_result_available"] is False
    assert not (failed / "candidates.jsonl").exists()


def _candidate(qid: str, kind: str, index: int, em: float, valid: bool = True):
    return {
        "qid": qid,
        "candidate_type": kind,
        "candidate_index": index,
        "em": em,
        "f1": em,
        "trajectory_valid": valid,
    }


def test_summary_uses_sampled_k4_for_oracle_mixed_and_valid_gates():
    rows = [_candidate("a", "greedy", 0, 0.0), _candidate("b", "greedy", 0, 1.0)]
    rows += [_candidate("a", "sampled", i, float(i == 3)) for i in range(4)]
    rows += [_candidate("b", "sampled", i, 1.0) for i in range(4)]
    result = summarize_candidates(
        rows,
        qids_expected=2,
        k=4,
        gates={
            "sample_valid_rate_min": 0.9,
            "oracle_at_4_minus_greedy_em_min": 0.05,
            "mixed_outcome_qid_rate_min": 0.2,
        },
    )
    assert result["metrics"]["greedy_em"] == pytest.approx(0.5)
    assert result["metrics"]["oracle_at_4_em"] == pytest.approx(1.0)
    assert result["metrics"]["oracle_at_4_minus_greedy_em"] == pytest.approx(0.5)
    assert result["metrics"]["mixed_outcome_qid_rate"] == pytest.approx(0.5)
    assert result["metrics"]["sample_valid_rate"] == pytest.approx(1.0)
    assert result["all_pass"] is True


def test_summary_rejects_non_k4_or_missing_greedy():
    rows = [_candidate("a", "sampled", i, 0.0) for i in range(4)]
    with pytest.raises(ValueError, match="coverage mismatch"):
        summarize_candidates(
            rows,
            qids_expected=1,
            k=4,
            gates={
                "sample_valid_rate_min": 0.9,
                "oracle_at_4_minus_greedy_em_min": 0.05,
                "mixed_outcome_qid_rate_min": 0.2,
            },
        )
