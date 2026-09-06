#!/usr/bin/env python
"""Freeze and execute a one-condition recovery of fresh132 analysis.

The immutable parent analyzer, protocol, failed attempt, generations and scores
are preserved. ``freeze`` and ``verify`` never decode Gold values. ``run`` is an
explicit separate command and opens labels only after complete Gold-free replay.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "source-credit-v2-fresh-analysis-recovery-v1"
PARENT_ANALYZER = "scripts/pilot/analyze_source_credit_v2_fresh_confirmation_v1.py"
PARENT_SOURCE_SHA256 = "65a197387c05f790c11cf7ecd99798f18e975fb622593fd513f7bb63542480b7"
PATCHED_SOURCE_SHA256 = "22280ac082e1f74c4832fd07be5822c04c26b000bab3cb98cf01f1def27774ad"
OLD = 'finite(row["raw_graph"]) and 0 <= row["raw_graph"] <= .85 + 1e-12'
NEW = 'finite(row["raw_graph"]) and (0 if valid else -1) <= row["raw_graph"] <= .85 + 1e-12'
OUTPUTS = {"processes.jsonl", "rankings.jsonl", "report.json", "prepared.json"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(ref):
    path = Path(ref["path"])
    if not path.is_absolute(): path = ROOT / path
    require(path.is_file() and sha(path) == ref["sha256"]
            and ("bytes" not in ref or path.stat().st_size == ref["bytes"]), "recovery immutable binding changed")
    return path.resolve()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")


def fixed_patch(source):
    require(hashlib.sha256(source).hexdigest() == PARENT_SOURCE_SHA256, "unrecognized parent analyzer SHA")
    text = source.decode("utf-8")
    require(text.count(OLD) == 1 and NEW not in text, "single frozen range condition required")
    corrected = text.replace(OLD, NEW, 1)
    result = corrected.encode("utf-8")
    require(hashlib.sha256(result).hexdigest() == PATCHED_SOURCE_SHA256, "one-condition corrected SHA differs")
    old_functions = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    new_functions = {n.name: ast.get_source_segment(corrected, n) for n in ast.parse(corrected).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require(set(old_functions) == set(new_functions), "function population changed")
    changed = [name for name in old_functions if old_functions[name] != new_functions[name]]
    require(changed == ["verify_process_rows"], "recovery changes more than the invalid diagnostic range")
    unchanged = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in old_functions.items() if name not in changed}
    diff = "".join(difflib.unified_diff(text.splitlines(keepends=True), corrected.splitlines(keepends=True),
                                      fromfile="parent/" + PARENT_ANALYZER, tofile="recovery/" + PARENT_ANALYZER))
    return result, diff, unchanged


def parent_assets(parent_protocol, scoring, failed_analysis):
    parent_protocol, scoring, failed_analysis = map(lambda p: Path(p).resolve(), (parent_protocol, scoring, failed_analysis))
    parent = json.loads(parent_protocol.read_text())
    source = checked(parent["code_bindings"][PARENT_ANALYZER])
    require(source == ROOT / PARENT_ANALYZER and sha(source) == PARENT_SOURCE_SHA256, "parent executed source differs")
    require(parent.get("status") == "FROZEN" and parent.get("seed") == 42, "frozen parent protocol required")
    failed_path = failed_analysis / "failed.json"
    failed = json.loads(failed_path.read_text())
    require(failed.get("status") == "FAIL" and failed.get("error_type") == "ValueError"
            and failed.get("gold_boundary_entered") is False, "recovery requires a failure before Gold")
    require(not any((failed_analysis / name).exists() for name in ("before_gold.json", "report.json", "manifest.json")),
            "parent attempt already crossed the verified Gold boundary")
    started = failed_analysis / "started.json"
    start = json.loads(started.read_text())
    require(start.get("gold_values_opened") is False and start.get("automatic_reanalysis_allowed") is False,
            "parent failed attempt marker differs")
    manifest_path = scoring / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "COMPLETE_GOLD_FREE_SCORED_NOT_ANALYZED"
            and manifest.get("protocol_sha256") == sha(parent_protocol)
            and manifest.get("gold_access") is False and manifest.get("model_updates") == 0
            and set(manifest.get("outputs") or {}) == OUTPUTS, "complete parent scoring seal required")
    refs = {"parent_protocol": bind(parent_protocol), "parent_analyzer": bind(source),
            "parent_failed": bind(failed_path), "parent_started": bind(started), "scoring_manifest": bind(manifest_path)}
    for name, ref in manifest["outputs"].items():
        path = checked(ref)
        require(path == scoring / name, "scoring output outside parent release")
        refs["scoring:" + name] = bind(path)
    report = json.loads(checked(refs["scoring:report.json"]).read_text())
    require(report.get("n_questions") == 132 and report.get("n_candidates") == 660,
            "parent132/660 scoring population differs")
    return parent, refs


def freeze(*, parent_protocol, scoring, failed_analysis, out):
    out = Path(out).resolve()
    require(not out.exists(), "new recovery protocol directory required")
    parent, refs = parent_assets(parent_protocol, scoring, failed_analysis)
    corrected, diff, functions = fixed_patch(checked(refs["parent_analyzer"]).read_bytes())
    out.mkdir(parents=True, exist_ok=False)
    copied = out / "code_snapshot" / PARENT_ANALYZER
    copied.parent.mkdir(parents=True, exist_ok=False)
    with copied.open("xb") as handle: handle.write(corrected)
    with (out / "patch.diff").open("x", encoding="utf-8") as handle: handle.write(diff)
    patch = {"schema_version": SCHEMA, "parent_source_sha256": PARENT_SOURCE_SHA256,
             "corrected_source_sha256": PATCHED_SOURCE_SHA256, "old_condition": OLD, "new_condition": NEW,
             "changed_function": "verify_process_rows", "unchanged_function_source_sha256": functions,
             "scientific_change": False,
             "reason": "ProofKG v2.3 invalid trajectories use -1 diagnostic sentinel; invalid process terms remain zero and ineligible",
             "decision_rules_unchanged": True, "Gold_reader_unchanged": True, "rankings_unchanged": True,
             "bootstrap_unchanged": True, "no_generation_or_scoring_repeated": True}
    write_new(out / "patch.json", patch)
    refs.update({"corrected_analyzer": bind(copied), "patch_diff": bind(out / "patch.diff"), "patch_record": bind(out / "patch.json"),
                 "recovery_runner": bind(__file__)})
    protocol = {"schema_version": SCHEMA, "status": "FROZEN_BEFORE_GOLD_RECOVERY",
                "experiment_id": parent["experiment_id"] + "-ANALYSIS-RECOVERY-V1",
                "parent_protocol_sha256": refs["parent_protocol"]["sha256"], "bindings": refs,
                "scoring_directory": str(Path(scoring).resolve()), "failed_analysis_directory": str(Path(failed_analysis).resolve()),
                "parent_decision_rules": parent["analysis"]["decision_rules"],
                "runtime_path_resolution": {"executed_module_file": str(copied), "ROOT": str(ROOT),
                    "scope": "ROOT filesystem resolution only; preserve real corrected __file__; restore caller sys.path after import"},
                "analysis_execution": "unchanged copied verify_before_gold -> unchanged Gold reader -> unchanged analyze; recovery-aware report publisher",
                "gold_values_opened": False, "model_updates": 0, "parent_assets_modified": False,
                "run_requires_explicit_separate_command": True, "full_ppo_auto_launch": False}
    write_new(out / "protocol.json", protocol)
    write_new(out / "manifest.json", {"schema_version": SCHEMA, "status": protocol["status"], "protocol": bind(out / "protocol.json")})
    return {"status": protocol["status"], "protocol": bind(out / "protocol.json"), "corrected_source_sha256": PATCHED_SOURCE_SHA256,
            "gold_values_opened": False}


def load_recovery(recovery_protocol):
    path = Path(recovery_protocol).resolve()
    manifest = json.loads((path.parent / "manifest.json").read_text())
    require(checked(manifest["protocol"]) == path and manifest["status"] == "FROZEN_BEFORE_GOLD_RECOVERY", "recovery protocol not frozen")
    p = json.loads(path.read_text())
    require(p["schema_version"] == SCHEMA and p["status"] == manifest["status"], "recovery protocol schema/status mismatch")
    refs = p["bindings"]
    for ref in refs.values(): checked(ref)
    require(checked(refs["recovery_runner"]) == Path(__file__).resolve(), "executed recovery runner differs")
    parent, inherited = parent_assets(checked(refs["parent_protocol"]), p["scoring_directory"], p["failed_analysis_directory"])
    require(all(refs.get(k) == v for k,v in inherited.items()) and p["parent_protocol_sha256"] == sha(checked(refs["parent_protocol"])),
            "recovery parent/scoring/failure lineage differs")
    require(p["parent_decision_rules"] == parent["analysis"]["decision_rules"], "recovery decision rules changed")
    expected, diff, functions = fixed_patch(checked(refs["parent_analyzer"]).read_bytes())
    copied = checked(refs["corrected_analyzer"])
    require(copied == path.parent / "code_snapshot" / PARENT_ANALYZER and copied.read_bytes() == expected
            and checked(refs["patch_diff"]).read_text() == diff, "corrected code exceeds frozen single patch")
    patch = json.loads(checked(refs["patch_record"]).read_text())
    require(patch["unchanged_function_source_sha256"] == functions and patch["old_condition"] == OLD and patch["new_condition"] == NEW,
            "patch record does not describe executed change")
    require(p["runtime_path_resolution"] == {"executed_module_file": str(copied), "ROOT": str(ROOT),
        "scope": "ROOT filesystem resolution only; preserve real corrected __file__; restore caller sys.path after import"},
        "runtime path override differs")
    name = "_source_credit_fresh_recovery_" + PATCHED_SOURCE_SHA256[:16]
    spec = importlib.util.spec_from_file_location(name, copied)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    module.ROOT = ROOT
    require(Path(module.__file__).resolve() == copied and module.decision_rules() == parent["analysis"]["decision_rules"],
            "executed copy identity or unchanged decisions differ")
    return p, module, {"recovery_protocol": bind(path), "recovery_manifest": bind(path.parent / "manifest.json"), **refs}


def verify(*, recovery_protocol):
    p, module, refs = load_recovery(recovery_protocol)
    context = module.verify_before_gold(checked(p["bindings"]["parent_protocol"]), Path(p["scoring_directory"]))
    for name, ref in refs.items(): context["frozen_files"]["recovery:" + name] = ref
    context["seal"]["recovery"] = {"protocol": refs["recovery_protocol"], "executed_corrected_analyzer": refs["corrected_analyzer"],
        "parent_analyzer": refs["parent_analyzer"], "parent_failed": refs["parent_failed"], "single_condition_patch": refs["patch_record"]}
    for ref in context["frozen_files"].values(): checked(ref)
    return p, module, refs, context


def run(*, recovery_protocol, out):
    out = Path(out).resolve()
    require(not out.exists(), "new recovery analysis directory required; do not overwrite or automatically rerun")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "started.json", {"schema_version": SCHEMA, "status": "STARTED_BEFORE_GOLD", "gold_values_opened": False,
                                    "automatic_reanalysis_allowed": False, "recovery_protocol": bind(recovery_protocol)})
    entered_gold = False
    try:
        p, module, refs, context = verify(recovery_protocol=recovery_protocol)
        write_new(out / "before_gold.json", context["seal"])
        entered_gold = True
        labels = module.load_gold_after_seal(context)
        questions, candidates, summaries, decision = module.analyze(context, labels)
        del labels
        for ref in context["frozen_files"].values(): checked(ref)
        for ref in context["protocol"]["analysis"]["gold_sources"].values(): checked(ref)
        for name, rows in (("questions.jsonl", questions), ("candidate_metrics.jsonl", candidates)):
            with (out / name).open("x", encoding="utf-8") as handle:
                for row in rows: handle.write(canonical(row) + "\n")
        recovery = {"protocol": refs["recovery_protocol"], "executed_analyzer": refs["corrected_analyzer"],
                    "runner": refs["recovery_runner"], "patch": refs["patch_record"], "patch_diff": refs["patch_diff"],
                    "parent_failed": refs["parent_failed"], "original_analyzer_executed_unmodified": False,
                    "scope": "sole invalid diagnostic range correction; unchanged frozen scientific analysis functions"}
        report = {"schema_version": module.SCHEMA, "status": decision["status"], "experiment_id": p["experiment_id"],
            "parent_experiment_id": context["protocol"]["experiment_id"], "protocol_sha256": context["protocol_sha256"],
            "recovery": recovery, "overall_status": decision["overall_status"], "health_status": decision["health_status"],
            "independent_utility_status": decision["independent_utility_status"], "engineering_probe_eligibility": decision["engineering_probe_eligibility"],
            "decision": decision, "decision_rules": module.decision_rules(), "population_summaries": summaries,
            "integrity": {"status": "PASS", "before_gold": bind(out / "before_gold.json")},
            "gold_values_opened_after_rank_seal": True, "gold_values_emitted": False, "model_updates": 0,
            "gate_fitting": False, "parent_gate_modified": False, "ppo_launched": False,
            "statistical_runtime": {"numpy": module.np.__version__, "python": sys.version},
            "limitations": ["Fixed fresh132 is graph-heavy, not a balanced three-domain baseline.",
                "Ordinary36 overlaps future formal PPO training and is diagnostic only; graph96 does not.",
                "Inference type absent; source-PASS bridge stratum is small.",
                "Family bootstrap intervals are descriptive, not simultaneous multiplicity-adjusted confidence.",
                "All-invalid sampled families stay ITT with zero process top1 scores.",
                "A-F equality or degenerate intervals do not establish equivalence or alpha superiority.",
                "F-T changes Graph weight and Text coefficient; report-only.",
                "No outcome-conditioned resampling, gate choice/refit or automatic full PPO launch."]}
        write_new(out / "report.json", report)
        write_new(out / "manifest.json", {"schema_version": SCHEMA, "status": report["status"],
            "protocol_sha256": context["protocol_sha256"], "recovery_protocol": refs["recovery_protocol"],
            "executed_analyzer": refs["corrected_analyzer"],
            "outputs": {name: bind(out / name) for name in ("started.json", "before_gold.json", "questions.jsonl", "candidate_metrics.jsonl", "report.json")},
            "independent_utility_status": decision["independent_utility_status"],
            "engineering_probe_eligibility": decision["engineering_probe_eligibility"],
            "matched600_investment_clearance": decision["matched600_investment_clearance"], "full_ppo_auto_launch": False})
        return {"status": report["status"], "report": bind(out / "report.json"), "recovery_protocol": refs["recovery_protocol"],
                "independent_utility_status": decision["independent_utility_status"], "engineering_probe_eligibility": decision["engineering_probe_eligibility"],
                "matched600_investment_clearance": decision["matched600_investment_clearance"]}
    except Exception as exc:
        write_new(out / "failed.json", {"schema_version": SCHEMA, "status": "FAIL", "error_type": type(exc).__name__,
            "gold_boundary_entered": entered_gold, "matched600_investment_clearance": False,
            "full_ppo_auto_launch": False, "note": "Exception text omitted to avoid accidentally emitting labels"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    for name in ("parent-protocol", "scoring", "failed-analysis", "out"):
        freeze_parser.add_argument("--" + name, type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--recovery-protocol", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--recovery-protocol", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    try:
        if command == "freeze": result = freeze(**args)
        elif command == "verify":
            _, _, _, context = verify(**args)
            result = {"status": "PASS_BEFORE_GOLD", "checked_candidates": 660, "checked_question_rankings": 132,
                      "checked_variant_arm_rankings": 792, "gold_values_opened": False,
                      "recovery": context["seal"]["recovery"]}
        else: result = run(**args)
        print(canonical(result))
    except Exception as exc:
        print(canonical({"status": "FAIL", "error_type": type(exc).__name__, "command": command}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
