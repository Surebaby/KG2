"""CPU production replay with real generations/tokens and cached real ReaRAG.

All120 legacy outputs and the10 invalid prompt-v1 outputs are checked across
A/F/T using both the immutable old runtime and the opt-in revised runtime.
The110 valid prompt-v1 outputs have no cached ReaRAG scores and are excluded
explicitly; no scorer output is invented and no model weights are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scripts.prepare import source_quality_candidate_bank_v1 as bank
from scripts.pilot.audit_answer_format_objective_v2 import load_module, verify
from scripts.pilot.check_source_credit_runtime_cached_v1 import CachedRealReaRAG, Forbidden, token_oracle, bound


ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
OUTPUT = ROOT / "outputs/audits/answer_format_objective_v2_production_cached_20260906_v2"
COMPONENT = ROOT / "outputs/audits/answer_format_objective_v2_cached_20260906_v1"
GENERATION = ROOT / "outputs/audits/training_format_prompt_v1_paired_probe_20260906_v1"
NORMALIZATION = ROOT / "outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2"
OLD_SCORED = ROOT / "outputs/audits/source_quality_candidates_format_v2_scored_local_seed42"
CALIBRATION = ROOT / "outputs/calibration/source_credit_gate_v2_representative_local_seed42_20260906_v1"
MODES = {"A": "learned", "F": "fixed", "T": "text"}
EXPERIMENT_ID = "ANSWER-FORMAT-OBJECTIVE-V2-PRODUCTION-CACHED-20260906-V2"


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    component_protocol = json.loads((COMPONENT / "protocol.json").read_text())
    normal_protocol = json.loads((NORMALIZATION / "protocol.json").read_text())
    paths = [
        COMPONENT / "manifest.json", COMPONENT / "structure_only.jsonl", COMPONENT / "candidate_metrics.jsonl",
        COMPONENT / "protocol.json", GENERATION / "manifest.json", GENERATION / "legacy_inputs.jsonl",
        GENERATION / "inputs.jsonl", GENERATION / "baseline_384.jsonl", GENERATION / "generations_prompt_v1.jsonl",
        NORMALIZATION / "manifest.json", NORMALIZATION / "normalization_rows.jsonl", NORMALIZATION / "new_scored.full.jsonl",
        NORMALIZATION / "protocol.json", OLD_SCORED / "manifest.json", OLD_SCORED / "candidates.scored.jsonl",
        CALIBRATION / "manifest.json", CALIBRATION / "features_v2/gate.json",
        ROOT / "outputs/audits/answer_format_objective_v2_production_cached_20260906_v1/FAILED.json",
        ROOT / "outputs/audits/answer_format_objective_v2_production_cached_20260906_v1/failure_diagnosis_addendum.json",
    ]
    snapshots = {}
    for source, name in (
        (Path(__file__), "audit.executed.py"),
        (ROOT / "kgproweight/training/reward_function.py", "production_reward.frozen.py"),
        (COMPONENT / "legacy_reward_function.frozen.py", "legacy_reward.frozen.py"),
    ):
        (output / name).write_bytes(source.read_bytes())
        snapshots[name] = bank.identity(output / name)
    code_names = (
        "scripts/pilot/audit_answer_format_objective_v2.py", "scripts/pilot/check_source_credit_runtime_cached_v1.py",
        "scripts/prepare/source_quality_candidate_bank_v1.py", "scripts/eval/ppo_emf1_development_v1.py",
        "kgproweight/training/reward_function.py", "kgproweight/reward/answer_format_objective_v2.py",
        "kgproweight/data/parsers.py", "kgproweight/data/prompts.py", "kgproweight/reward/proofkg_process.py",
        "kgproweight/eval/pred_processing.py",
        "kgproweight/reward/proofkg_process_v2.py", "kgproweight/reward/proofkg_process_v2_2.py",
        "kgproweight/reward/proofkg_process_v2_3.py", "kgproweight/reward/source_quality_gate_v1.py",
        "kgproweight/reward/source_credit_gate_v1.py", "kgproweight/reward/source_credit_gate_v2.py",
        "kgproweight/reward/source_reward_normalization_v2.py", "kgproweight/reward/source_trajectory_features_v2.py",
        "kgproweight/reward/trajectory_source_gate.py",
        "kgproweight/training/ppo_tensorboard.py",
    )
    tokenizers = {}
    for name in ("policy_tokenizer", "rearag_model"):
        info = normal_protocol["models"][name]
        folder = Path(info["path"])
        if not folder.is_absolute():
            folder = ROOT / folder
        files = {}
        for filename, binding in info["files"].items():
            if "tokenizer" in filename or filename in {"special_tokens_map.json", "tokenization_chatglm.py"}:
                path = bound(binding, folder)
                files[filename] = bank.identity(path)
        tokenizers[name] = {"path": str(folder), "files": files}
    protocol = {
        "schema_version": "answer-format-objective-v2-production-cached-protocol-v1",
        "experiment_id": EXPERIMENT_ID, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_revision": "v2 fixes only generation identity comparison to hash complete prediction record; v1 failed before Gold join or any production call and is preserved",
        "selection": "all120 legacy outputs plus all10 invalid prompt-v1 outputs; based only on frozen original validity",
        "distinct_outputs": 130, "valid": 96, "invalid": 34, "shortfall": 8,
        "mode_and_runtime_replays": "A/F/T x immutable old runtime / opt-in production v2 =780 calls, not780 independent samples",
        "excluded": "110 valid prompt-v1 responses have no real cached ReaRAG; no imputed scores or GPU rescore",
        "gate": "frozen final six-feature source-credit-v2, loaded allow_unvalidated only in this offline audit",
        "oracle": "old runtime valid token tensors and raw Text/Graph unchanged; cached component audit for invalid outcome; independent linear-prefix token placement",
        "invalid_forbidden": ["Text scorer", "Graph scorers and execution trace", "learned gate predict"],
        "gold": component_protocol["gold"],
        "gold_join_after": "protocol and selected structural identities are frozen; labels only used for existing outcome formula",
        "source_bindings": {str(path): bank.identity(path) for path in paths},
        "code_snapshots": snapshots,
        "supporting_code": {name: bank.identity(ROOT / name) for name in code_names},
        "tokenizers": tokenizers,
        "optimizer_updates": 0, "model_weights_loaded": False, "fresh_confirmation_consumed": False,
        "tensorboard": "three separate A/F/T diagnostic runs, each same130 cached outputs at step130, update_index0; actual production writer and event readback; no loss/KL fabricated",
        "ppo_launch_clearance": False,
    }
    bank.write_json(output / "protocol.json", protocol)
    bank.write_json(output / "prepared.json", {"protocol": bank.identity(output / "protocol.json"), "gold_parsed_in_this_experiment": False})
    print(json.dumps({"status": "FROZEN", "experiment_id": EXPERIMENT_ID}))


def write_tensorboard(output, mode, reward_infos):
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from kgproweight.training.ppo_tensorboard import log_ppo_batch
    import torch
    directory = output / "tensorboard" / ("cached_reward_not_training_" + mode)
    assert len(reward_infos) == 130
    copies = [row["token_rewards"].clone() for row in reward_infos]
    with SummaryWriter(str(directory)) as writer:
        writer.add_text("diagnostic/contract", json.dumps({
            "kind": "cached_reward_not_training", "distinct_outputs": 130,
            "prompt_cohort": "120 legacy plus10 invalid prompt-v1; not a representative performance score",
            "mode": mode, "optimizer_updates": 0, "model_weights_loaded": False,
        }), 0)
        log_ppo_batch(writer, step=130, update_index=0,
                      stats={"diagnostic/optimizer_updates": 0, "diagnostic/cached_reward_not_training": 1},
                      reward_infos=reward_infos, histogram_every=1)
    assert all(torch.equal(row["token_rewards"], before) for row, before in zip(reward_infos, copies))
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0, "histograms": 0}).Reload()
    expected = {"diagnostic/optimizer_updates": 0.0, "diagnostic/cached_reward_not_training": 1.0,
                "reward/all/count": 130.0, "reward/all/valid_rate": 96 / 130,
                "reward/all/shortfall_salvage_rate": 8 / 130, "reward/all/severe_invalid_rate": 26 / 130,
                "reward/all/answer_signal_applied_rate": 104 / 130}
    for name, container, field in (
        ("canonical_em", "answer_format_reward", "canonical_em"), ("canonical_f1", "answer_format_reward", "canonical_f1"),
        ("answer_component", "answer_format_reward", "answer_component"), ("format_component", "answer_format_reward", "format_component"),
        ("outcome", "mixed_reward", "outcome"), ("text_component", "mixed_reward", "text"),
        ("graph_component", "mixed_reward", "process"), ("total", "mixed_reward", "total"),
    ):
        expected["reward/all/" + name + "_mean"] = sum(row[container][field] for row in reward_infos) / 130
    expected["gate/all/alpha_effective_mean"] = sum(row["source_gate"]["alpha_effective"] for row in reward_infos) / 130
    readback = {}
    for tag, value in expected.items():
        entries = events.Scalars(tag)
        assert len(entries) == 1 and entries[0].step == 130
        assert abs(entries[0].value - value) < 1e-6, (mode, tag, value, entries[0].value)
        readback[tag] = {"expected": value, "event_value": entries[0].value, "pass": True}
    assert events.Tags()["histograms"]
    return {"mode": mode, "kind": "cached_reward_not_training", "distinct_outputs": 130,
            "diagnostic_updates": 0, "checked_scalar_tags": readback,
            "scalar_tag_count": len(events.Tags()["scalars"]), "histogram_tag_count": len(events.Tags()["histograms"]),
            "token_rewards_unchanged_after_logging": True,
            "event_files": {str(path.relative_to(output)): bank.identity(path) for path in directory.glob("events.out.tfevents.*")}}


def run(output):
    if (output / "manifest.json").exists() or (output / "checks.jsonl").exists():
        raise FileExistsError("refusing to overwrite production replay")
    os.environ["HF_MODULES_CACHE"] = str(output / "tokenizer_code_cache")
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
    prepared = json.loads((output / "prepared.json").read_text())
    verify(prepared["protocol"])
    protocol = json.loads((output / "protocol.json").read_text())
    bindings = [*protocol["source_bindings"].values(), *protocol["code_snapshots"].values(), *protocol["supporting_code"].values(),
                *[binding for info in protocol["tokenizers"].values() for binding in info["files"].values()]]
    for info in bindings:
        verify(info)
    if bank.file_sha(Path(__file__)) != protocol["code_snapshots"]["audit.executed.py"]["sha256"]:
        raise ValueError("run must use frozen script bytes")
    assert not torch.cuda.is_initialized()
    bank.load_release(COMPONENT, "answer-format-objective-v2-cached-report-v1")
    bank.load_release(GENERATION, "training-format-prompt-v1-paired-development-probe-report")
    bank.load_release(NORMALIZATION, "normalization-only-train-score-bank-v1")
    old = load_module(output / "legacy_reward.frozen.py", "production_cached_old_reward")
    new = load_module(output / "production_reward.frozen.py", "production_cached_new_reward")
    gate = SourceCreditGateV2.load(CALIBRATION / "features_v2/gate.json", allow_unvalidated=True)
    assert gate.artifact["training_clearance"] is False and gate.artifact["ppo_launch_clearance"] is False
    policy_tokenizer = AutoTokenizer.from_pretrained(protocol["tokenizers"]["policy_tokenizer"]["path"], local_files_only=True)
    rearag_tokenizer = AutoTokenizer.from_pretrained(protocol["tokenizers"]["rearag_model"]["path"], trust_remote_code=True, local_files_only=True)
    inputs = {arm: {row["question_key"]: row for row in bank.read_rows(GENERATION / name)}
              for arm, name in (("legacy", "legacy_inputs.jsonl"), ("prompt_v1", "inputs.jsonl"))}
    predictions = {
        "legacy": {row["prediction"]["candidate_id"]: row["prediction"] for row in bank.read_rows(GENERATION / "baseline_384.jsonl")},
        "prompt_v1": {row["candidate_id"]: row for row in bank.read_rows(GENERATION / "generations_prompt_v1.jsonl")},
    }
    normal = {row["candidate_id"]: row for row in bank.read_rows(NORMALIZATION / "normalization_rows.jsonl")}
    scored = {row["candidate_id"]: row for row in bank.read_rows(OLD_SCORED / "candidates.scored.jsonl")}
    scored.update({row["candidate_id"]: row for row in bank.read_rows(NORMALIZATION / "new_scored.full.jsonl")})
    structures = bank.read_rows(COMPONENT / "structure_only.jsonl")
    selected = [row for row in structures if row["arm"] == "legacy" or not row["format_valid"]]
    assert len(selected) == 130 and sum(row["format_valid"] for row in selected) == 96
    assert sum(row["salvage"]["shortfall_salvage_eligible"] for row in selected) == 8
    for row in selected:
        frozen = inputs[row["arm"]][row["question_key"]]
        prediction = predictions[row["arm"]][row["candidate_id"]]
        bank.assert_gold_free(frozen)
        assert bank.input_hash(frozen) == frozen["input_sha256"] == prediction["input_sha256"]
        assert bank.digest(prediction) == row["prediction_sha256"]
        assert policy_tokenizer.decode(prediction["response_token_ids"], skip_special_tokens=True) == prediction["generation"]
        if row["format_valid"]:
            entry, full = normal[row["candidate_id"]], scored[row["candidate_id"]]
            assert entry["generation_sha256"] == bank.digest(prediction)
            assert entry["input_sha256"] == frozen["input_sha256"] == full["input_sha256"]
            assert entry["score_row_sha256"] == bank.digest(full)
            assert entry["raw_text"] == full["raw_text"] and full["generation"] == prediction["generation"]
    bank.write_rows(output / "selection.structure_only.jsonl", selected)
    bank.write_json(output / "before_gold.json", {"gold_parsed_in_this_experiment": False, "selection": bank.identity(output / "selection.structure_only.jsonl"),
                                               "protocol": prepared["protocol"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
    verify(protocol["gold"])
    labels_list = bank.read_rows(Path(protocol["gold"]["path"]))
    labels = {bank.key(row): row for row in labels_list}
    assert len(labels) == len(labels_list)
    metrics = {(row["arm"], row["candidate_id"]): row for row in bank.read_rows(COMPONENT / "candidate_metrics.jsonl")}
    checks, maximum_token_error, scorer_calls = [], 0.0, 0
    tensorboard_rows = {mode: [] for mode in MODES}
    for index, selected_row in enumerate(selected):
        arm, cid, key = selected_row["arm"], selected_row["candidate_id"], selected_row["question_key"]
        frozen, prediction, label = inputs[arm][key], predictions[arm][cid], labels[key]
        assert label["question"] == frozen["question"] and label["metadata"]["source_split"] == "train"
        ids, response = prediction["response_token_ids"], prediction["generation"]
        valid, expected = selected_row["format_valid"], metrics[(arm, cid)]
        for mode_name, mode in MODES.items():
            results = {}
            for runtime_name, module in (("legacy", old), ("v2", new)):
                spec = module.RewardSpec(**deepcopy(frozen["spec"]), gold_answer=label["metadata"]["gold_answer"],
                                        gold_answer_aliases=label["metadata"].get("gold_answer_aliases") or [])
                spec_hash = bank.digest(spec.__dict__)
                validation = module.validate_source_gate_trajectory(spec, response, format_version="v2")
                assert validation["valid"] is valid and validation["violations"] == selected_row["violations"]
                prompts, texts = module.source_gate_text_inputs_v1(spec, validation["steps"])
                raw_text = normal[cid]["raw_text"] if valid else []
                cache = CachedRealReaRAG(rearag_tokenizer, prompts, texts, raw_text)
                kwargs = {"answer_format_reward_version": "v2"} if runtime_name == "v2" else {}
                reward = module.KGProWeightRewardFunction(
                    alpha_gate=Forbidden(), prm_annotator=Forbidden(), text_reward_model=cache, tokenizer=policy_tokenizer,
                    outcome_weight=4.0, text_reward_scale=0.3, max_steps=5,
                    proofkg_process_reward=True, proofkg_process_version="v2_3", proofkg_process_weight=0.2,
                    proofkg_f1_weight=0.1, proofkg_dynamic_validity=True, mixed_outcome_reward=True, mixed_text_reward=True,
                    runtime_contract_version="v2", source_gated_reward_version="v1", source_gate_format_version="v2",
                    source_gate_credit_version="v2", source_gate_mode=mode, source_quality_gate=gate, center_text_reward=False, **kwargs,
                )
                patched = {}
                original_predict = gate.predict
                if not valid:
                    for name in ("score_grounded_process", "score_proofkg_v2", "score_proofkg_v2_2", "score_proofkg_v2_3",
                                 "build_execution_trace", "build_execution_trace_v2_2", "build_execution_trace_v2_3"):
                        patched[name] = getattr(module, name)
                        setattr(module, name, Forbidden())
                    cache.score_steps = Forbidden()
                    gate.predict = Forbidden()
                try:
                    actual = reward(frozen["prompt"], response, spec, response_ids=ids)
                finally:
                    for name, value in patched.items():
                        setattr(module, name, value)
                    gate.predict = original_predict
                assert bank.digest(spec.__dict__) == spec_hash
                assert actual["trajectory_valid"] is valid
                assert cache.calls == int(valid)
                scorer_calls += cache.calls
                expected_outcome = expected["legacy_component" if runtime_name == "legacy" else "v2_component"]
                assert actual["mixed_reward"]["outcome"] == expected_outcome
                if valid:
                    assert actual["mixed_reward"]["text_raw_step_scores"] == raw_text
                    assert bank.digest(module.source_gate_text_budget_v1(cache, prompts, texts)) == bank.digest(scored[cid]["text_token_budget"])
                else:
                    assert actual["mixed_reward"]["text"] == actual["mixed_reward"]["process"] == 0.0
                    assert actual["source_gate"]["alpha_effective"] == 0.0
                    assert actual["source_gate"]["invalid_not_scored"] is True
                    assert torch.count_nonzero(actual["token_rewards"]) == 1
                    assert not actual["proofkg_process"]["process_applied"]
                step_weights = actual["mixed_reward"]["text_weighted_step_rewards"]
                graph = actual["mixed_reward"]["process"]
                oracle = token_oracle(ids, policy_tokenizer, step_weights, expected_outcome + graph)
                tokens = actual["token_rewards"].cpu().double().numpy()
                error = float(np.max(np.abs(tokens - oracle)))
                maximum_token_error = max(maximum_token_error, error)
                assert error < 1e-6 and abs(tokens.sum() - actual["trajectory_reward"]) < 1e-6
                assert abs(sum(actual["per_step_rewards"]) - actual["trajectory_reward"]) < 1e-10
                if runtime_name == "v2":
                    assert actual["answer_format_reward"]["shortfall_salvage_eligible"] == selected_row["salvage"]["shortfall_salvage_eligible"]
                    assert actual["answer_format_reward"]["shortfall_salvage_reason"] == selected_row["salvage"]["shortfall_salvage_reason"]
                    assert actual["answer_format_reward"]["canonical_em"] == expected["em"]
                    assert actual["answer_format_reward"]["canonical_f1"] == expected["f1"]
                    tensorboard_rows[mode_name].append(actual)
                results[runtime_name] = actual
                checks.append({
                    "candidate_id": cid, "prompt_arm": arm, "mode": mode_name, "runtime": runtime_name,
                    "format_valid": valid, "shortfall_salvaged": selected_row["salvage"]["shortfall_salvage_eligible"],
                    "outcome": expected_outcome, "text": actual["mixed_reward"]["text"], "graph": graph,
                    "alpha": actual["source_gate"]["alpha_effective"], "total": actual["trajectory_reward"],
                    "token_sum": float(tokens.sum()), "token_max_abs_error": error,
                    "cached_real_text_calls": cache.calls, "cached_real_text_scores_sha256": bank.digest(raw_text),
                    "source_status": actual["source_gate"]["features"]["source_credit_mask"]["status"],
                    "process_calls_forbidden": not valid, "pass": True,
                })
            if valid:
                assert torch.equal(results["legacy"]["token_rewards"], results["v2"]["token_rewards"])
                for field in ("trajectory_reward", "per_step_rewards", "mixed_reward", "source_gate", "proofkg_process"):
                    assert results["legacy"][field] == results["v2"][field], (cid, mode, field)
        if (index + 1) % 10 == 0:
            print(json.dumps({"distinct_outputs_checked": index + 1, "runtime_calls": len(checks)}), flush=True)
    assert len(checks) == 780 and scorer_calls == 576 and not torch.cuda.is_initialized()
    bank.write_rows(output / "checks.jsonl", checks)
    tensorboard = {mode: write_tensorboard(output, mode, infos) for mode, infos in tensorboard_rows.items()}
    bank.write_json(output / "tensorboard_readback.json", tensorboard)
    report = {
        "schema_version": "answer-format-objective-v2-production-cached-report-v1", "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_REAL_CACHED_PRODUCTION_REPLAY_NO_TRAINING", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "distinct_outputs": 130, "legacy_prompt_outputs": 120, "invalid_prompt_v1_outputs": 10,
        "distinct_valid": 96, "distinct_invalid": 34, "distinct_shortfall": 8,
        "runtime_calls": 780, "valid_runtime_calls": 576, "invalid_runtime_calls": 204,
        "shortfall_v2_calls": 24, "shortfall_v2_outcome_distribution": dict(Counter(str(r["outcome"]) for r in checks if r["shortfall_salvaged"] and r["runtime"] == "v2")),
        "all_valid_tokens_and_reward_fields_exactly_equal_immutable_old_runtime": True,
        "all_invalid_text_graph_predict_forbidden_and_never_called": True,
        "actual_cached_rearag_calls": scorer_calls, "maximum_token_oracle_abs_error": maximum_token_error,
        "source_status_counts_distinct": dict(Counter(r["source_status"] for r in checks if r["mode"] == "A" and r["runtime"] == "v2")),
        "excluded_valid_prompt_v1_without_rearag_cache": 110,
        "tensorboard": tensorboard,
        "model_weights_loaded": False, "cuda_initialized": False, "optimizer_updates": 0,
        "fresh_confirmation_consumed": False, "ppo_launch_clearance": False,
        "interpretation": "130 existing outputs replayed780 times for engineering equivalence, not780 independent samples, training improvement, or α confirmation",
    }
    for info in [*bindings, protocol["gold"], prepared["protocol"]]:
        verify(info)
    bank.finish(output, report, ["protocol.json", "prepared.json", "audit.executed.py", "production_reward.frozen.py", "legacy_reward.frozen.py",
                                "selection.structure_only.jsonl", "before_gold.json", "checks.jsonl", "tensorboard_readback.json"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    freeze(args.output_dir) if args.command == "freeze" else run(args.output_dir)
