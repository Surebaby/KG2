"""Population replacement must preserve held-out isolation and all K2 rows."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.train.rebind_source_credit_v2_text_population import validate_normalization_rows
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
from scripts.train import rebind_source_credit_v2_text_population as module


def _rows():
    rows=[]
    for dataset in ("hotpotqa","2wikimultihopqa","musique"):
        for index in range(40):
            identity=f"{dataset}::{index}"
            digest=hashlib.sha256(identity.encode()).hexdigest()
            for rollout in range(2):
                rows.append({"candidate_id":f"{identity}::k{rollout}","dataset":dataset,"qid":str(index),
                    "family_sha256":digest,"question_sha256":digest,"split":"train",
                    "trajectory_valid":True,"raw_text":[.2,.8]})
    return rows


def test_balanced_bank_preserves_invalid_candidates_without_resampling():
    rows=_rows()
    rows[0].update(trajectory_valid=False,raw_text=[])
    before=deepcopy(rows)
    assert validate_normalization_rows(rows,[])["questions"]==120
    stats=fit_text_normalization_v2(rows)
    assert stats["counts"]["input_candidates"]==240
    assert stats["counts"]["valid_candidates"]==239
    assert stats["counts"]["questions_with_valid_scores"]==120
    assert rows==before


@pytest.mark.parametrize("key",["split","family_split"])
def test_consumed_split_cannot_be_relabelled_by_omitting_other_field(key):
    rows=_rows();rows[0][key]="confirmation"
    with pytest.raises(ValueError,match="train-only"):
        validate_normalization_rows(rows,[])


def test_family_overlap_is_rejected_even_when_row_claims_train():
    rows=_rows()
    with pytest.raises(ValueError,match="non-train families"):
        validate_normalization_rows(rows,[{"family_sha256":rows[0]["family_sha256"],"split":"calibration"}])


@pytest.mark.parametrize("mutation",["dropped_failure","duplicate","source_mix","question_identity","missing_family","gold"])
def test_population_binding_errors_cannot_enter_fit(mutation):
    rows=_rows()
    if mutation=="dropped_failure":rows.pop()
    elif mutation=="duplicate":rows[0]=deepcopy(rows[1])
    elif mutation=="source_mix":
        for row in rows:
            if row["dataset"]=="musique" and row["qid"]=="0":row["dataset"]="hotpotqa"
    elif mutation=="question_identity":rows[0]["question_sha256"]="f"*64
    elif mutation=="missing_family":rows[0].pop("family_sha256")
    elif mutation=="gold":rows[0]["gold_answer"]="must not be read"
    with pytest.raises(ValueError):validate_normalization_rows(rows,[])


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return module.identity(path)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return module.identity(path)


def _release(directory, payload, files):
    payload = deepcopy(payload)
    payload["outputs"] = {name: {**module.identity(directory / name), "path": name,
                                 "origin_path": str((directory / name).resolve())} for name in files}
    path = directory / "manifest.json"
    _write_json(path, payload)
    return path


def test_release_relative_path_resolves_inside_owning_bank_not_project_root(tmp_path):
    path = tmp_path / "normalization_rows.jsonl"
    path.write_text("[]\n")
    binding = {**module.identity(path), "path": path.name, "origin_path": str(path)}
    observed = {}
    assert module.bound_file(binding, tmp_path, expected=path, observed=observed) == path
    assert observed[str(path)] == module.identity(path)


def test_matching_hash_at_another_path_cannot_authorize_the_fixed_file_read(tmp_path):
    expected, other = tmp_path / "gate.json", tmp_path / "other.json"
    expected.write_text("{}")
    other.write_bytes(expected.read_bytes())
    with pytest.raises(ValueError, match="exact file"):
        module.release_outputs({"outputs": {"gate.json": module.identity(other)}}, tmp_path,
                               required={"gate.json"})


def test_missing_output_binding_or_changed_source_bytes_are_rejected(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text("{}")
    binding = module.identity(path)
    with pytest.raises(ValueError, match="exactly"):
        module.release_outputs({"outputs": {}}, tmp_path, required={"gate.json"})
    path.write_text("{} ")
    with pytest.raises(ValueError, match="bytes changed"):
        module.bound_file(binding, tmp_path)


def _source_bundle(tmp_path):
    """Complete small frozen provenance chain, without models or answer labels."""
    directory = tmp_path / "bank"
    directory.mkdir()
    policy = tmp_path / "policy"
    _write_json(policy / "adapter_config.json", {"synthetic": True})
    (policy / "adapter_model.safetensors").write_text("synthetic adapter bytes")
    ledger = tmp_path / "protected.jsonl"
    _write_rows(ledger, [])
    groups, inputs, selected, assignments = [], [], [], []
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        for index in range(40):
            question = f"where does {dataset} token{chr(97 + index // 26)}{chr(97 + index % 26)} lead?"
            common = {"dataset": dataset, "qid": str(index), "question": question,
                      "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                      "family_sha256": family_sha256(question), "family_version": FAMILY_VERSION}
            group = {**common, "gold_access": False,
                     "process_reward_eligible": dataset == "2wikimultihopqa" and index < 32}
            source = {**common, "source_split": "train"}
            source["input_sha256"] = canonical_sha256(source)
            reuse = len(inputs) < 47
            groups.append(group); inputs.append(source); selected.append({**group, "reuse": reuse})
            if reuse:
                assignments.extend({"candidate_id": f"{dataset}::{index}::k{k}", "dataset": dataset,
                    "qid": str(index), "family_sha256": common["family_sha256"], "split": "train"} for k in range(2))
    groups_file = tmp_path / "groups.jsonl"
    _write_rows(groups_file, groups)
    assignments_file = tmp_path / "assignments.jsonl"
    _write_rows(assignments_file, assignments)
    models = {"base_model": {"path": "synthetic/base", "files": {}},
              "policy_tokenizer": {"path": str(policy), "files": {}},
              "rearag_model": {"path": "synthetic/rearag", "files": {}}}
    generation = {"K": 2, "temperature": 1., "seed": 42}
    prepared = {"schema_version": "source-quality-candidate-preparation-v1", "policy_path": str(policy),
                "training_started": False, "generation": generation, **models,
                "source_bindings": {"policy": module.identity(policy / "adapter_model.safetensors"),
                                    "policy_config": module.identity(policy / "adapter_config.json"),
                                    "protected_ledger": module.identity(ledger)}}
    old_dir = tmp_path / "old_inputs"
    _write_rows(old_dir / "inputs.jsonl", inputs[:47])
    old_manifest = _release(old_dir, prepared, ["inputs.jsonl"])
    new_dir = directory / "new_inputs"
    _write_rows(new_dir / "inputs.jsonl", inputs[47:])
    new_manifest = _release(new_dir, prepared, ["inputs.jsonl"])
    predictions, scores, compact = {"reuse": [], "new": []}, {"reuse": [], "new": []}, []
    for index, source in enumerate(inputs):
        origin = "reuse" if index < 47 else "new"
        key = module.question_key(source)
        for k in range(2):
            cid = f"{key}::k{k}"
            prediction = {"candidate_id": cid, "candidate_index": k, "dataset": source["dataset"], "qid": source["qid"],
                "input_sha256": source["input_sha256"], "generation": "synthetic generation",
                "policy_sha256": prepared["source_bindings"]["policy"]["sha256"],
                "base_model_identity_sha256": canonical_sha256(models["base_model"]),
                "generation_contract_sha256": canonical_sha256(generation),
                "bank_manifest_sha256": module.identity(old_manifest if origin == "reuse" else new_manifest)["sha256"]}
            scored = {"candidate_id": cid, "dataset": source["dataset"], "qid": source["qid"],
                      "input_sha256": source["input_sha256"], "generation": prediction["generation"],
                      "raw_text": [.2, .8], "trajectory_valid": True, "format_validation": {"valid": True}}
            row = {"schema_version": "normalization-only-train-score-row-v1", "candidate_id": cid,
                   "dataset": source["dataset"], "qid": source["qid"], "split": "train",
                   "question_sha256": source["question_sha256"], "family_sha256": source["family_sha256"],
                   "input_sha256": source["input_sha256"], "generation_sha256": canonical_sha256(prediction),
                   "score_row_sha256": canonical_sha256(scored), "raw_text": scored["raw_text"],
                   "trajectory_valid": True, "format_validation": scored["format_validation"],
                   "normalization_train_only": True, "gold_used": False, "selection_uses_validity": False,
                   "origin": "exact_frozen_train_candidate_reuse" if origin == "reuse" else "new_frozen_sft_generation_original_bf16_rearag"}
            predictions[origin].append(prediction); scores[origin].append(scored); compact.append(row)
    generation_paths = {}
    for origin, where in (("reuse", tmp_path / "old_generated"), ("new", directory / "generated")):
        _write_rows(where / "generations.jsonl", predictions[origin])
        generation_paths[origin] = _release(where, {"schema_version": "source-quality-candidate-generation-v1",
            "training_started": False, "bank_manifest_sha256": module.identity(old_manifest if origin == "reuse" else new_manifest)["sha256"],
            "generation_contract_sha256": canonical_sha256(generation)}, ["generations.jsonl"])
    old_score_dir = tmp_path / "old_scored"
    _write_rows(old_score_dir / "candidates.scored.jsonl", scores["reuse"])
    old_scores_manifest = _release(old_score_dir, {"schema_version": "source-quality-candidate-bank-v1"}, ["candidates.scored.jsonl"])
    paths = {name: directory / name for name in ("protocol.json", "inputs.all.jsonl", "selection.question_only.jsonl", "new_scored.full.jsonl")}
    _write_rows(paths["inputs.all.jsonl"], inputs)
    _write_rows(paths["selection.question_only.jsonl"], selected)
    _write_rows(paths["new_scored.full.jsonl"], scores["new"])
    producer = directory / "runtime_code/producer.py"
    producer.parent.mkdir(parents=True); producer.write_text("# frozen synthetic producer\n")
    protocol = {"schema_version": "normalization-representative-bank-v1", "K": 2, "seed": 42,
        "membership": "normalization_train_only_not_independent_confirmation", "policy_optimizer_updates": 0,
        "gold_used": False, "no_failed_candidate_replacement": True,
        "quotas": module.SELECTION_QUOTAS, "selection": {"selection_uses_gold_scores_or_validity": False},
        "models": models, "generation": generation, "policy_path": str(policy),
        "code_bindings": {"producer.py": module.identity(producer)},
        "parent_bindings": {"inputs": module.identity(old_manifest), "generation": module.identity(generation_paths["reuse"]),
                            "scored": module.identity(old_scores_manifest), "assignments": module.identity(assignments_file)},
        "source_bindings": {"groups": module.identity(groups_file)},
        "input_sha256": module.identity(paths["inputs.all.jsonl"])["sha256"],
        "selection_sha256": module.identity(paths["selection.question_only.jsonl"])["sha256"],
        "new_bank_manifest_sha256": module.identity(new_manifest)["sha256"]}
    _write_json(paths["protocol.json"], protocol)
    return directory, {"protocol": module.identity(paths["protocol.json"])}, paths, compact, assignments


def test_compact_observations_replay_from_exact_train_selection_and_frozen_sft_records(tmp_path):
    args = _source_bundle(tmp_path)
    before = deepcopy(args[3])
    observed = {}
    result = module.verify_normalization_sources(*args, observed)
    assert result["replayed_candidates"] == 240
    assert result["origin_counts"] == {"reuse": 94, "new": 146}
    assert result["protected_overlap"] == 0
    assert args[3] == before
    assert len(observed) >= 15


@pytest.mark.parametrize("field,value", [
    ("raw_text", [.9]), ("trajectory_valid", False), ("input_sha256", "0" * 64),
    ("generation_sha256", "0" * 64), ("score_row_sha256", "0" * 64),
    ("family_sha256", "0" * 64), ("question_sha256", "0" * 64),
    ("candidate_id", "hotpotqa::0::k2"), ("origin", "new_frozen_sft_generation_original_bf16_rearag"),
    ("gold_used", True), ("selection_uses_validity", True),
])
def test_compact_claims_cannot_override_bound_source_observations(tmp_path, field, value):
    args = _source_bundle(tmp_path)
    args[3][0][field] = value
    with pytest.raises(ValueError):
        module.verify_normalization_sources(*args, {})


def test_changed_parent_assignments_cannot_authorize_new_train_membership(tmp_path):
    args = _source_bundle(tmp_path)
    args[4][0]["split"] = "confirmation"
    with pytest.raises(ValueError, match="family assignments"):
        module.verify_normalization_sources(*args, {})


def test_modified_policy_bytes_cannot_be_hidden_behind_compact_hashes(tmp_path):
    args = _source_bundle(tmp_path)
    (tmp_path / "policy/adapter_model.safetensors").write_text("different policy")
    with pytest.raises(ValueError, match="bytes changed"):
        module.verify_normalization_sources(*args, {})
