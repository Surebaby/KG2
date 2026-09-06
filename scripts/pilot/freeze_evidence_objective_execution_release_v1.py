"""Bind this staged development execution to immutable code and audit releases."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/audits/evidence_and_objective_execution_20260906_v1"


def identity(path):
    path = path.resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(part)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def main():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("release is append-only")
    previous_path = ROOT / "outputs/audits/training_format_prompt_v1_release_20260906_v1/manifest.json"
    previous = json.loads(previous_path.read_text())["previous_98_core_code_configs_unchanged"]
    expected_changed = {"kgproweight/config/schemas.py", "kgproweight/training/phase3_ppo.py",
                        "kgproweight/training/ppo_tensorboard.py", "kgproweight/training/reward_function.py",
                        "scripts/train/phase3_ppo.py"}
    changed, preserved = {}, {}
    for name, info in previous.items():
        actual = identity(Path(info["path"]))
        if actual["sha256"] == info["sha256"]:
            preserved[name] = actual
        else:
            changed[name] = {"before": info, "after": actual}
    if set(changed) != expected_changed:
        raise ValueError("unexpected mutation of previous core code/configs")
    source_names = sorted(expected_changed | {
        "kgproweight/reward/answer_format_objective_v2.py",
        "scripts/pilot/audit_answer_format_objective_v2.py",
        "scripts/pilot/probe_evidence_supply_v1.py", "scripts/pilot/probe_dense_rank_repair_v1.py",
        "scripts/pilot/probe_evidence_reader_v1.py", "scripts/pilot/audit_evidence_reader_result_v1.py",
        "scripts/pilot/freeze_evidence_objective_execution_release_v1.py",
        "tests/test_answer_format_objective_v2.py", "tests/test_answer_format_reward_runtime_v2.py",
        "tests/test_answer_format_tensorboard_v2.py", "tests/test_evidence_reader_v1.py",
        "tests/test_probe_evidence_supply_v1.py", "tests/test_probe_dense_rank_repair_v1.py",
        "docs/evidence_and_objective_execution_20260906.md", "docs/evidence_supply_v1_consumed20_20260906.md",
        "RESEARCH_WORKFLOW.md", "docs/todo2.md",
        *[f"configs/training/phase3_ppo_mixed4_answer_format_v2_{arm}_probe_seed42.yaml" for arm in "aft"],
    })
    snapshot = OUT / "snapshot"
    snapshot.mkdir(exist_ok=False)
    snapshots = {}
    for name in source_names:
        src, dst = ROOT / name, snapshot / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = identity(src)
        shutil.copyfile(src, dst)
        after = identity(dst)
        if before["sha256"] != after["sha256"]:
            raise ValueError("snapshot mismatch")
        snapshots[name] = {"source": before, "snapshot": after}
    directories = [
        "answer_format_objective_v2_cached_20260906_v1",
        "answer_format_objective_v2_production_cached_20260906_v2",
        "evidence_supply_v1_consumed20_20260906_v1",
        "evidence_supply_v1_independent_review_20260906_v1",
        "evidence_supply_v1_reader_consumed20_20260906_v1",
        "evidence_supply_v1_reader_consumed20_20260906_v1/assessment",
        "evidence_supply_v1_reader_independent_review_20260906_v1",
        "evidence_supply_v1_posthoc_case_review_20260906_v1",
        "dense_rank_contract_consumed20_20260906_v1",
        "dense_rank_historical_lineage_20260906_v1",
    ]
    artifacts = {name: identity(ROOT / "outputs/audits" / name / "manifest.json") for name in directories}
    suites = list(ET.parse(OUT / "tests.junit.xml").getroot().iter("testsuite"))
    checks = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    if checks != {"tests": 350, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("expected complete350 tests")
    fresh = ROOT / "outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1/inputs.jsonl"
    fresh_bound = identity(fresh)
    if fresh_bound["sha256"] != "c56b68b82a7f1e0460dc2237f3765d1ded21e6eabb4583bc4ace5e6f0bac2a2f":
        raise ValueError("fresh confirmation input mutated")
    reader = json.loads((ROOT / "outputs/audits/evidence_supply_v1_reader_consumed20_20260906_v1/assessment/report.json").read_text())
    production = json.loads((ROOT / "outputs/audits/answer_format_objective_v2_production_cached_20260906_v2/report.json").read_text())
    record = {
        "schema_version": "evidence-and-answer-objective-execution-release-v1",
        "experiment_id": "EVIDENCE-AND-ANSWER-OBJECTIVE-EXECUTION-20260906-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGED_REPAIRS_VALIDATED_EVIDENCE_V1_NOT_ADOPTED_PPO_NOT_STARTED",
        "authorization": "Researcher approved executing the evidence-first staged strategy on2026-09-06",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)),
        "code_authority": "Pre-existing dirty worktree; exact code snapshots plus experiment protocols, not Git HEAD alone",
        "previous_release": identity(previous_path), "previous_core_preserved": preserved,
        "expected_opt_in_objective_core_changes": changed, "snapshots": snapshots,
        "artifacts": artifacts, "tests": checks, "junit": identity(OUT / "tests.junit.xml"),
        "config_resolution": identity(OUT / "config_resolution.json"), "fresh132_preserved": fresh_bound,
        "production_reward_validation": {k: production[k] for k in
            ("distinct_outputs", "runtime_calls", "distinct_valid", "distinct_invalid", "distinct_shortfall",
             "all_valid_tokens_and_reward_fields_exactly_equal_immutable_old_runtime",
             "all_invalid_text_graph_predict_forbidden_and_never_called", "maximum_token_oracle_abs_error")},
        "reader_metrics": reader["metrics"], "reader_surface_coverage": reader["surface_coverage"],
        "evidence_supply_v1_production_adoption": False,
        "reward_v2": "Implemented and verified independent opt-in; original configs default legacy",
        "sft_optimizer_updates": 0, "ppo_optimizer_updates": 0, "fresh_confirmation_consumed": False,
        "ppo_launch_clearance": False, "remote_synced": False,
        "next_step": "Pre-freeze evidence-conditioned process proxy contrast diagnostics and review minimum-step incentives, then conditional identity-isolated cross-domain supervision feasibility; freeze final start/prompt before fresh alpha confirmation",
    }
    with (OUT / "manifest.json").open("x") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": record["status"], "manifest": identity(OUT / "manifest.json"),
                      "snapshots": len(snapshots), "previous_core_preserved": len(preserved), "tests": checks}))


if __name__ == "__main__":
    main()
