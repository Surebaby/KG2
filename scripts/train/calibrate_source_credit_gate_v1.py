"""Fit source-credit alpha under a frozen additional Graph eligibility mask.

All original candidate inputs and outputs remain intact. Source-credit clearance
applies only to Graph reward credit; it does not certify or repair
the KG that the frozen Reader saw. The original four-feature ratio fit and
family split are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from kgproweight.reward.source_quality_gate_v1 import (
    SourceQualityGateV1, assign_family_splits, canonical_sha256, heuristic_ratio_target,
)
from scripts.train import calibrate_source_quality_gate_v1 as legacy


ROOT = Path(__file__).resolve().parents[2]
ROW_SCHEMA = "source-credit-candidate-row-v1"
MANIFEST_SCHEMA = "source-credit-gate-calibration-manifest-v1"
BOUNDARY = (
    "Graph reward credit only: frozen PASS sources intersect the legacy hard gate; "
    "FAIL/UNVERIFIED/missing sources receive zero Graph credit. Original Reader "
    "prompts, passages, KG and generated trajectories are unchanged and not certified "
    "as repaired. Ratio fidelity is not process utility or PPO launch clearance."
)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_rows(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def create_parent_view(bank_manifest: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    """Relocate exact frozen bytes; never rebind parent hashes to edited code."""
    bank_manifest = bank_manifest.resolve()
    if (bank_manifest.parent / "FAILED.json").exists():
        raise ValueError("failed parent scored bank cannot be calibrated")
    original = json.loads(bank_manifest.read_text(encoding="utf-8"))
    if original.get("format_contract_version") != "source-gate-runtime-v2-format-v2":
        raise ValueError("source-credit calibration requires the original format-v2 bank")
    if original.get("schema_version") != legacy.BANK_SCHEMA:
        raise ValueError("wrong parent scored bank schema")
    # Validate the parent's declared release outputs before creating any view.
    for name, bound in original.get("outputs", {}).items():
        if Path(name).name != name:
            raise ValueError("parent output names must be plain filenames")
        target = bank_manifest.parent / name
        if legacy._identity(target)["sha256"] != bound["sha256"]:
            raise ValueError(f"parent output hash mismatch: {name}")
    view = deepcopy(original)
    relocations = []
    for name, bound in original.get("source_bindings", {}).items():
        if name.startswith("code:"):
            relative = Path(name.removeprefix("code:"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe parent code binding name")
            archived = bank_manifest.parent / "runtime_code" / relative
            identity = legacy._identity(archived)
            if identity["sha256"] != bound["sha256"]:
                raise ValueError(f"archived runtime code hash mismatch: {relative}")
            view["source_bindings"][name] = {**bound, **identity}
            relocations.append({"binding": name, "parent_path": bound["path"],
                                "archived_path": identity["path"], "sha256": identity["sha256"]})
        else:
            resolved = legacy._bound_path(bound, bank_manifest.parent)
            view["source_bindings"][name] = {**bound, "path": str(resolved)}
    for key in ("bank", "isolation_proof"):
        view[key] = {**original[key], "path": str(legacy._bound_path(original[key], bank_manifest.parent))}
    proof_path = Path(view["isolation_proof"]["path"])
    relocation = {
        "schema_version": "source-credit-parent-binding-relocation-v1",
        "parent_manifest": legacy._identity(bank_manifest),
        "source_code_relocations": relocations,
        "bank_bytes_unchanged": True, "isolation_proof_bytes_unchanged": True,
        "gold_or_candidate_data_modified": False,
        "boundary": "Only manifest binding paths change; every relocated code file retains its original SHA256.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    view["parent_view_relocation"] = relocation
    for name, bound in view.get("outputs", {}).items():
        bound["path"] = str(bank_manifest.parent / name)
    write_json(output_dir / "manifest.json", view)
    write_json(output_dir / "relocation.json", relocation)
    return output_dir / "manifest.json", proof_path, relocation


def _spec(row):
    record = row["source_quality_record"]
    return SimpleNamespace(query=row["question"], kg_subgraph=record.get("kg_subgraph") or [],
        retrieved_passages=row["retrieved_passages"], metadata={"dataset": row["dataset"],
        "qid": row["qid"], "source_quality_record": record})


def summarize_mask(parent_rows, rows, assignments) -> dict:
    by_dataset = defaultdict(Counter)
    by_operation = defaultdict(Counter)
    by_split = defaultdict(Counter)
    by_status = Counter()
    questions = {}
    split_by_candidate = {item["candidate_id"]: item["split"] for item in assignments}
    for old, row in zip(parent_rows, rows):
        if old["candidate_id"] != row["candidate_id"]:
            raise ValueError("mask changed the frozen candidate order")
        credit = row["source_credit"]
        status = credit["status"]
        by_status[status] += 1
        key = (row["dataset"], row["qid"])
        questions[key] = {"status": status, "parent_m_graph": old["features"]["m_graph"], "m_graph": row["features"]["m_graph"]}
        operation = str((row["source_quality_record"].get("query_plan") or {}).get("operation") or "ordinary")
        values = {"candidates": 1, "parent_graph_eligible": int(old["features"]["m_graph"]),
                  "graph_credit_eligible": int(row["features"]["m_graph"]),
                  "graph_credit_removed": int(bool(old["features"]["m_graph"]) and not row["features"]["m_graph"]),
                  "valid_text_candidates_retained": int(row["trajectory_valid"] and bool(row["raw_text"])),
                  "valid_text_steps_retained": len(row["raw_text"]),
                  "nonabstaining_targets": int(row["quality"]["target"] is not None)}
        for group in (by_dataset[row["dataset"]], by_operation[operation], by_split[split_by_candidate[row["candidate_id"]]]):
            group.update(values)
    return {
        "candidate_count": len(rows), "question_count": len(questions),
        "candidate_status_counts": dict(by_status),
        "question_status_counts": dict(Counter(item["status"] for item in questions.values())),
        "parent_graph_questions": sum(item["parent_m_graph"] for item in questions.values()),
        "graph_credit_questions": sum(item["m_graph"] for item in questions.values()),
        "by_dataset": dict(by_dataset), "by_query_operation": dict(by_operation), "by_family_split": dict(by_split),
        "all_candidates_retained": len(parent_rows) == len(rows),
        "all_valid_text_scores_retained": all(old["raw_text"] == row["raw_text"] for old, row in zip(parent_rows, rows)),
        "reader_inputs_and_generations_unchanged": all(
            all(old[key] == row[key] for key in ("generation", "retrieved_passages", "source_quality_record"))
            for old, row in zip(parent_rows, rows)),
        "source_integrity_clearance": False,
        "source_credit_scope": "reward_credit_only_input_unchanged",
    }


def apply_credit_view(parent_rows, mask):
    """Apply the runtime mask to validated parent rows, without resampling."""
    rows = []
    for parent_row in parent_rows:
        row = deepcopy(parent_row)
        features = mask.mask_features(_spec(row), row["features"])
        if features["values"] != parent_row["features"]["values"]:
            raise ValueError("source-credit mask must preserve all four numerical features")
        if features["m_graph"] > parent_row["features"]["m_graph"]:
            raise ValueError("source-credit mask cannot promote an ineligible parent graph")
        marker = features["source_credit_mask"]
        row["schema_version"] = ROW_SCHEMA
        row["parent_validated_row_sha256"] = canonical_sha256(parent_row)
        row["features"] = features
        row["source_credit"] = {"status": marker["status"],
                                "parent_m_graph": parent_row["features"]["m_graph"],
                                "m_graph": features["m_graph"],
                                "mask_payload_sha256": mask.payload_sha256,
                                "scope": "reward_credit_only_input_unchanged"}
        row["quality"] = heuristic_ratio_target(row["raw_graph"], row["raw_text"],
            m_graph=features["m_graph"], trajectory_valid=row["trajectory_valid"])
        rows.append(row)
    if [row["candidate_id"] for row in parent_rows] != [row["candidate_id"] for row in rows]:
        raise ValueError("source-credit view must preserve every original candidate in order")
    return rows


def fit_source_credit(rows, bindings, mask, *, experiment_id):
    """Fit with unchanged original hyperparameters, then require the new reader."""
    from kgproweight.reward.source_credit_gate_v1 import (
        ARTIFACT_SCHEMA, MASK_VERSION, SourceCreditGateV1,
    )

    fit_bindings = {**bindings, "source_integrity_clearance": False,
        "source_integrity_status": "SOURCE_LABEL_PROJECTION_UNRESOLVED_INPUT_UNCHANGED",
        "source_credit_clearance": True,
        "source_credit_scope": "reward_credit_only_input_unchanged"}
    artifact, report, assignments = legacy.fit_gate(rows, fit_bindings,
        experiment_id=experiment_id, seed=42, epochs=800)
    artifact.pop("payload_sha256")
    artifact.update(schema_version=ARTIFACT_SCHEMA, source_credit_version=MASK_VERSION,
        source_credit_clearance=True,
        source_credit_scope="reward_credit_only_input_unchanged",
        source_credit_mask={"path": str(mask.manifest_path.resolve()), "sha256": mask.manifest_sha256,
                            "payload_sha256": mask.payload_sha256},
        source_integrity_scope="original_reader_input_kg",
        reader_input_integrity_clearance=False, reader_inputs_repaired=False,
        scientific_boundary=BOUNDARY)
    artifact["normalization"].update(
        graph_fit_population="valid_credit_eligible_train_including_target_abstentions",
        fixed_alpha_population="valid_credit_eligible_train_candidate_mean_including_target_abstentions")
    artifact["payload_sha256"] = canonical_sha256(artifact)
    new_gate = SourceCreditGateV1(artifact, mask=mask, allow_synthetic=True, allow_unvalidated=True)
    for row, assignment in zip(rows, assignments):
        if new_gate.predict(row["features"]) != assignment["alpha"]:
            raise ValueError("new source-credit reader changed the fitted gate calculation")
    # A successor mask must never be accidentally consumed by the old reader.
    try:
        SourceQualityGateV1(artifact, allow_synthetic=True, allow_unvalidated=True)
    except ValueError:
        pass
    else:
        raise ValueError("legacy reader unexpectedly accepted the source-credit successor")
    for row, assignment in zip(rows, assignments):
        assignment["source_credit"] = deepcopy(row["source_credit"])
    report.update(source_integrity_clearance=False, source_credit_clearance=True,
        source_integrity_status=artifact["source_integrity_status"],
        source_credit_scope=artifact["source_credit_scope"],
        reader_input_integrity_clearance=False, reader_inputs_repaired=False,
        source_integrity_scope=artifact["source_integrity_scope"],
        source_credit_mask=artifact["source_credit_mask"],
        normalization=artifact["normalization"], scientific_boundary=BOUNDARY,
        ppo_launch_clearance=False)
    return artifact, report, assignments


def calibrate(bank_manifest: Path, mask_manifest: Path, output_dir: Path, *, experiment_id: str,
              synthetic_test_only: bool = False) -> dict:
    from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "started.json", {"experiment_id": experiment_id,
        "created_at_utc": created, "policy_optimizer_updates": 0, "boundary": BOUNDARY})
    try:
        code_paths = [Path(__file__), ROOT / "scripts/train/calibrate_source_quality_gate_v1.py",
                      ROOT / "kgproweight/reward/source_credit_gate_v1.py",
                      ROOT / "kgproweight/reward/source_quality_gate_v1.py",
                      ROOT / "kgproweight/reward/source_integrity_v1.py",
                      ROOT / "kgproweight/training/reward_function.py"]
        code_bindings = {}
        for path in code_paths:
            relative = path.resolve().relative_to(ROOT)
            identity = legacy._identity(path)
            saved = output_dir / "runtime_code" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes())
            if legacy._identity(saved)["sha256"] != identity["sha256"]:
                raise ValueError("source code changed while freezing calibration")
            code_bindings[str(relative)] = identity
        parent_view, isolation, relocation = create_parent_view(bank_manifest, output_dir / "parent_view")
        # This validator sees the original rows, not the new masked population.
        parent_rows, bindings = legacy.validate_bank(parent_view, isolation,
            synthetic_test_only=synthetic_test_only)
        mask = FrozenSourceCreditMask.load(mask_manifest)
        rows = apply_credit_view(parent_rows, mask)
        masked_path = output_dir / "candidates.credit_masked.jsonl"
        write_rows(masked_path, rows)
        parent_validation = {
            "schema_version": "source-credit-validated-parent-view-v1",
            "original_bank_manifest": legacy._identity(bank_manifest),
            "validated_parent_view_manifest": legacy._identity(parent_view),
            "parent_bank": bindings["bank"], "parent_candidates": len(parent_rows),
            "mask_applied_after_parent_validation": True,
            "masked_rows_were_not_validated_as_legacy_rows": True,
            "relocation": relocation,
        }
        write_json(output_dir / "parent_validation.json", parent_validation)
        fit_bindings = {**bindings, "parent_validation": legacy._identity(output_dir / "parent_validation.json"),
            "original_bank_manifest": legacy._identity(bank_manifest),
            "credit_masked_bank": legacy._identity(masked_path),
            "source_credit_mask": {"path": str(mask.manifest_path.resolve()),
                                   "sha256": mask.manifest_sha256, "payload_sha256": mask.payload_sha256}}
        artifact, report, assignments = fit_source_credit(rows, fit_bindings, mask, experiment_id=experiment_id)
        splits = assign_family_splits(row["family_sha256"] for row in parent_rows)
        if any(item["split"] != splits[item["family_sha256"]] for item in assignments):
            raise ValueError("original family splits changed")
        summary = summarize_mask(parent_rows, rows, assignments)
        report["source_credit_population"] = summary
        for path in code_paths:
            relative = str(path.resolve().relative_to(ROOT))
            if legacy._identity(path)["sha256"] != code_bindings[relative]["sha256"]:
                raise ValueError("calibration source changed during execution")
        if legacy._identity(bank_manifest) != parent_validation["original_bank_manifest"]:
            raise ValueError("parent bank manifest changed during calibration")
        write_json(output_dir / "gate.json", artifact)
        write_json(output_dir / "report.json", report)
        write_json(output_dir / "mask_population.json", summary)
        write_rows(output_dir / "assignments.jsonl", assignments)
        manifest = {
            "schema_version": MANIFEST_SCHEMA, "experiment_id": experiment_id,
            "created_at_utc": created, "status": report["status"],
            "training_clearance": artifact["training_clearance"],
            "source_integrity_clearance": False, "source_credit_clearance": True,
            "source_integrity_scope": artifact["source_integrity_scope"],
            "source_credit_scope": artifact["source_credit_scope"],
            "source_credit_mask": artifact["source_credit_mask"],
            "reader_input_integrity_clearance": False, "reader_inputs_repaired": False,
            "ppo_launch_clearance": False, "bank_source": bindings["bank_source"],
            "format_contract_version": artifact["format_contract_version"],
            "seed": 42, "light_gate_fit_epochs": 800,
            "policy_optimizer_updates": 0, "evaluation_protocol": legacy.TARGET_VERSION,
            "parent_validation": parent_validation,
            "outputs": {name: legacy._identity(output_dir / name) for name in (
                "gate.json", "report.json", "mask_population.json", "assignments.jsonl",
                "candidates.credit_masked.jsonl", "parent_validation.json")},
            "code_bindings": code_bindings, "git_head": legacy._optional_git_head(),
            "scientific_boundary": BOUNDARY,
        }
        write_json(output_dir / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        write_json(output_dir / "FAILED_CALIBRATION.json", {
            "experiment_id": experiment_id, "status": "FAILED_SOURCE_CREDIT_CALIBRATION",
            "error_type": type(exc).__name__, "error": str(exc),
            "policy_optimizer_updates": 0, "partial_outputs_retained": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args()
    result = calibrate(args.bank_manifest, args.mask_manifest, args.output_dir,
        experiment_id=args.experiment_id, synthetic_test_only=args.synthetic_test_only)
    print(json.dumps({"status": result["status"], "training_clearance": result["training_clearance"],
                      "source_integrity_scope": result["source_integrity_scope"],
                      "ppo_launch_clearance": False, "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
