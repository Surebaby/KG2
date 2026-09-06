"""Register existing stopped-run checkpoints on unchanged development inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from scripts.eval import ppo_emf1_development_v1 as dev


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/evals/ppo_a_stopped300_development150_20260906_v1"
PARENT = ROOT / "outputs/audits/source_gated_mixed4_emf1_v1_development_a_smoke"
PPO = "outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/ppo_mixed4_answer_format_v2_a_smoke600_scoped_seed42_20260906_v1"
EXPERIMENT = "PPO-A-STOPPED300-DEVELOPMENT150-SEED42-20260906-V1"


def main():
    parent, manifest = dev._load_release(PARENT)
    assert dev.file_sha(PARENT / "manifest.json") == "139bd6c8d432b6757318e2ec6f065ab931392da04a3734eed3a6fa888dd51912"
    assert dev.file_sha(ROOT / "scripts/eval/ppo_emf1_development_v1.py") == manifest["implementation"]["sha256"]
    assert parent["by_dataset"] == {name: 50 for name in dev.DATASETS}
    assert parent["generation"]["do_sample"] is False and parent["generation"]["max_new_tokens"] == 512
    for name, frozen in parent["tokenizer"]["files"].items():
        assert dev.file_sha(ROOT / parent["tokenizer"]["path"] / name) == frozen["sha256"]
    candidates = [
        {"model_id": "strong_sft", "checkpoint_path": "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final", "training_step": 0, "is_sft": True},
        {"model_id": "ppo_a_step200", "checkpoint_path": f"{PPO}/step_200", "training_step": 200, "is_sft": False},
        {"model_id": "ppo_a_aborted300", "checkpoint_path": f"{PPO}/aborted_step_300", "training_step": 300, "is_sft": False},
    ]
    expected = ("7c0be8d6400b637c746dfe18f051cd8b518b23d8f2d960825e565172a1612f24",
                "851cd7634584d1ecb469c1a9f829d264b53fe7590cebbba9546bbdfa35bb67d7",
                "16759a1e2712c476512652d8db8836083c4e68c57f77ab542ee14adc83217f2f")
    for candidate, sha in zip(candidates, expected):
        assert dev.file_sha(ROOT / candidate["checkpoint_path"] / "adapter_model.safetensors") == sha
    OUTPUT.mkdir(parents=True, exist_ok=False)
    dev.write_json(OUTPUT / "candidate_registry_source.json", {"schema_version": dev.VERSION, "candidates": candidates})
    normalized = dev._registry(OUTPUT / "candidate_registry_source.json", ROOT)
    bank = OUTPUT / "bank"
    bank.mkdir()
    for name in ("legacy.inputs.jsonl", "no_graph.inputs.jsonl", "labels.jsonl"):
        shutil.copyfile(PARENT / name, bank / name)
        assert dev.file_sha(bank / name) == manifest["outputs"][name]["sha256"]
    dev.write_json(bank / "candidate_registry.json", {"schema_version": dev.VERSION, "candidates": normalized})
    report = deepcopy(parent)
    report.update(experiment_id=EXPERIMENT, candidate_registry_source=dev.identity(OUTPUT / "candidate_registry_source.json"))
    report["registration_successor"] = {
        "parent_manifest": dev.identity(PARENT / "manifest.json"),
        "reason": "User explicitly authorized evaluating the retained200 and stopped300 checkpoints against original Strong SFT after smoke stopped at300.",
        "input_and_label_bytes_unchanged": True, "scorer_generation_and_selection_unchanged": True,
        "original400_and_final600_do_not_exist": True,
        "aborted300_is_post_stop_diagnostic_candidate": True,
        "registered_before_any_new_predictions_or_scores": True,
        "parent_registry_and_baselines_preserved": True,
    }
    dev._finish(bank, report, ["legacy.inputs.jsonl", "no_graph.inputs.jsonl", "labels.jsonl", "candidate_registry.json"])
    code_paths = sorted({p.relative_to(ROOT).as_posix() for p in (ROOT / "kgproweight").rglob("*.py")} | {
        "scripts/eval/ppo_emf1_development_v1.py", "scripts/eval/run_stopped300_development_v1.py",
        "scripts/eval/freeze_stopped300_development_v1.py", "scripts/prepare/freeze_qpeg_v1_protocol.py"})
    protocol = {"schema_version": "ppo-a-stopped300-development-execution-v1", "experiment_id": EXPERIMENT,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "user_instruction": "行 启动评估吧", "purpose": "Stopped-run development checkpoint comparison; not canonical reporting",
                "parent_bank_manifest": dev.identity(PARENT / "manifest.json"),
                "candidates": normalized, "views": list(dev.VIEWS), "questions_per_view": 150, "predictions_total": 900,
                "base_model": parent["base_model"]["path"], "tokenizer": parent["tokenizer"]["path"],
                "generation": parent["generation"], "selection_rule": parent["selection_rule"],
                "bank_files": {p.name: dev.identity(p) for p in sorted(bank.iterdir())},
                "code_bindings": {name: dev.identity(ROOT / name) for name in code_paths},
                "training_authority": dev.identity(ROOT / "outputs/audits/ppo_a_smoke600_gpu_supervision_terminal300_20260906_v1/report.json"),
                "all_generation_before_scoring": True, "automatic_restart": False, "automatic_canonical_evaluation": False,
                "gold_values_read_at_registration": False, "new_training": False}
    dev.write_json(OUTPUT / "execution.json", protocol)
    dev.write_json(OUTPUT / "freeze_manifest.json", {"schema_version": protocol["schema_version"],
                   "experiment_id": EXPERIMENT, "status": "FROZEN_NOT_STARTED",
                   "execution": dev.identity(OUTPUT / "execution.json"), "bank_manifest": dev.identity(bank / "manifest.json")})
    print(json.dumps({"status": "FROZEN_NOT_STARTED", "experiment_id": EXPERIMENT,
                      "bank_manifest_sha256": dev.file_sha(bank / "manifest.json"), "execution_sha256": dev.file_sha(OUTPUT / "execution.json")}))


if __name__ == "__main__":
    main()
