from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kgproweight.config import ProjectConfig, load_config
from scripts.prepare.finalize_mixed3_rearag_proof400_ppo_pair import (
    ARMS,
    BOUND_PROTOCOL_PATHS,
    EXPECTED_COUNTS,
    GPU_POSTFLIGHT_REQUIRED_STATUS,
    GPU_POSTFLIGHT_PATH,
    ROOT,
    assert_bound_protocol_contracts,
    flatten,
    inspect_data_contract,
    inspect_gpu_postflight,
)
from scripts.prepare.resolve_phase3_ppo_runtime_config import (
    resolve_phase3_ppo_runtime_config,
)


def test_proof400_pair_effective_and_real_cli_diffs_are_exact():
    control = load_config(ARMS["ppo_t"]["config"], validate=ProjectConfig)
    treatment = load_config(ARMS["ppo_tk"]["config"], validate=ProjectConfig)
    left, right = flatten(control.model_dump()), flatten(treatment.model_dump())
    assert sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    ) == ["training.output_dir", "training.ppo.proofkg_process_reward"]

    control_runtime = resolve_phase3_ppo_runtime_config(ARMS["ppo_t"]["config"])
    treatment_runtime = resolve_phase3_ppo_runtime_config(ARMS["ppo_tk"]["config"])
    left, right = flatten(control_runtime), flatten(treatment_runtime)
    assert sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    ) == ["output_dir", "proofkg_process_reward"]
    assert control_runtime["proofkg_process_reward"] is False
    assert treatment_runtime["proofkg_process_reward"] is True
    assert control_runtime["text_reward_backend"] == "rearag"
    assert treatment_runtime["text_reward_backend"] == "rearag"


def test_proof400_data_contract_is_independently_recomputed():
    result = inspect_data_contract()
    assert result["unique_population"] == EXPECTED_COUNTS["unique_population"] == 1799
    assert result["prompt_groups"] == EXPECTED_COUNTS["prompt_groups"] == 1800
    assert result["trajectories"] == EXPECTED_COUNTS["trajectories"] == 7200
    assert result["eligible_unique"] == EXPECTED_COUNTS["eligible_unique"] == 400
    assert result["eligible_prompt_groups"] == 400
    assert result["eligible_trajectories"] == 1600
    assert result["outcome_text_only_empty_kg"] == 1399
    assert result["eligible_by_question_type"] == {
        "bridge_comparison": 100,
        "comparison": 100,
        "compositional": 100,
        "inference": 100,
    }


def test_all_required_versioned_protocols_are_bound_and_semantically_valid():
    assert set(BOUND_PROTOCOL_PATHS) == {
        "v2_data_protocol",
        "v2_data_protocol_manifest",
        "v2_family_scope_addendum",
        "v2_family_scope_addendum_manifest",
        "config_comparison_v2",
        "config_comparison_v2_manifest",
        "standard_legacy_eval_protocol",
        "standard_legacy_eval_qids",
        "standard_legacy_eval_manifest",
        "v3_runtime_probe_protocol",
        "v3_runtime_probe_protocol_manifest",
        "v3_runtime_probe_local_preflight",
        "v3_runtime_probe_local_preflight_manifest",
    }
    assert all(path.is_file() for path in BOUND_PROTOCOL_PATHS.values())
    result = assert_bound_protocol_contracts()
    assert result["family_addendum_status"] == (
        "COMPLETE_APPEND_ONLY_CLARIFICATION_DATA_UNCHANGED"
    )
    assert result["config_comparison_status"] == (
        "PASS_CONFIG_ONLY_NOT_GPU_PROBED_NOT_TRAINED"
    )
    assert result["standard_eval_scope"]["n_total"] == 900


def test_gpu_postflight_gate_is_exact_and_fail_closed(tmp_path: Path):
    path = tmp_path / "postflight.json"
    assert inspect_gpu_postflight(path)["state"] == "MISSING"
    path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    near_pass = inspect_gpu_postflight(path)
    assert near_pass["state"] == "INVALID_BUNDLE"
    assert near_pass["gate_pass"] is False
    protocol = BOUND_PROTOCOL_PATHS["v3_runtime_probe_protocol"]
    protocol_identity = {
        "path": str(protocol.relative_to(ROOT)),
        "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "size_bytes": protocol.stat().st_size,
    }
    path.write_text(
        json.dumps(
            {"status": GPU_POSTFLIGHT_REQUIRED_STATUS, "protocol": protocol_identity}
        ) + "\n",
        encoding="utf-8",
    )
    postflight_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": GPU_POSTFLIGHT_REQUIRED_STATUS,
                "run": {"postflight_sha256": postflight_hash, "effect_evidence": False},
            }
        ) + "\n",
        encoding="utf-8",
    )
    passed = inspect_gpu_postflight(path)
    assert passed["state"] == "PASS"
    assert passed["gate_pass"] is True
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["run"]["postflight_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    assert inspect_gpu_postflight(path)["state"] == "NON_PASS_BUNDLE"
    manifest["run"]["postflight_sha256"] = postflight_hash
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    path.write_text("not-json\n", encoding="utf-8")
    assert inspect_gpu_postflight(path)["state"] == "INVALID_JSON"


def test_formal_launcher_binds_exact_gpu_gate_and_absent_outputs():
    launcher = ROOT / "launch_ppo_mixed3_rearag_v2_proof400_paired7200_remote.sh"
    text = launcher.read_text(encoding="utf-8")
    assert str(GPU_POSTFLIGHT_PATH.relative_to(ROOT)) in text
    assert "PASS_RUNTIME_WIRING_NOT_EFFECT_EVIDENCE" in text
    assert "postflight_sha256" in text
    assert 'p.get("status") == "PASS_CPU_PREFLIGHT_GPU_POSTFLIGHT_BOUND"' in text
    assert 'test ! -e "$OUT_T"' in text
    assert 'test ! -e "$OUT_TK"' in text
    assert 'run_arm "$CONFIG_T"' in text
    assert 'run_arm "$CONFIG_TK"' in text
    assert text.index('run_arm "$CONFIG_T"') < text.index('run_arm "$CONFIG_TK"')
    assert "scripts/train/phase3_ppo.py" in text


def test_gpu_postflight_is_currently_not_claimed_as_pass():
    # This assertion records the handoff state.  If the independently executed
    # v3 probe is later materialized, this test remains valid by accepting its
    # exact PASS state rather than pretending it existed at config-freeze time.
    state = inspect_gpu_postflight(GPU_POSTFLIGHT_PATH)
    assert state["state"] in {"MISSING", "PASS"}
