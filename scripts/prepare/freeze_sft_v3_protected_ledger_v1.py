#!/usr/bin/env python3
"""Freeze SFT-v3 exclusions and audit remaining raw-train identity capacity.

Only identity fields and the *historical* accepted flag needed to replay the
Strong-SFT split are read semantically. Some source JSON objects contain Gold;
those values are never selected, emitted, or used to choose SFT identities.
This ledger protects consumed/held-out research identities, the entire PPO
population, and untouched alpha confirmation. It is not a training dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from kgproweight.data.silver_split import SplitSpec, assign_split
from kgproweight.kg.question_kg import question_sha256
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-protected-identity-ledger-v1"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEFAULT_OUT = ROOT / "outputs/audits/sft_v3_protected_identity_ledger_20260906_v1"
STRONG_SOURCE = "checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl"
STRONG_MANIFEST = "checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/manifest.json"


def sha_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bind(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha_file(path), "bytes": path.stat().st_size}


def read_rows(path: Path):
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            yield line_number, row, hashlib.sha256(raw).hexdigest()


def identity(row: Mapping[str, Any], dataset_override: str | None = None) -> dict[str, str]:
    candidates = {str(row[name]).strip().lower() for name in ("dataset", "dataset_name") if row.get(name)}
    if dataset_override:
        candidates.add(dataset_override)
    if len(candidates) != 1 or not candidates <= set(DATASETS):
        raise ValueError("missing, conflicting, or unknown dataset identity")
    dataset = next(iter(candidates))
    ids = {str(row[name]).strip() for name in ("qid", "id") if row.get(name)}
    if len(ids) != 1:
        raise ValueError("missing or ambiguous qid/id identity")
    qid = next(iter(ids))
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing question identity")
    question = question.strip()
    qhash = question_sha256(question)
    if row.get("question_sha256") and row["question_sha256"] != qhash:
        raise ValueError("stored question SHA mismatch")
    if row.get("question_key") and row["question_key"] != f"{dataset}::{qid}":
        raise ValueError("question_key disagrees with dataset/qid")
    return {"dataset": dataset, "qid": qid, "question": question,
            "question_sha256": qhash, "family_version": FAMILY_VERSION,
            "family_sha256": family_sha256(question)}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    role: str
    expected_sha256: str
    dataset_override: str | None = None
    historical_holdout_only: bool = False
    expected_rows: int | None = None


def _validate_bound(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha_file(path) != expected:
        raise ValueError(f"source SHA binding mismatch: {path}")


def default_sources(root: Path = ROOT) -> tuple[list[SourceSpec], list[dict[str, Any]]]:
    """Validate current authority manifests; never infer Gold-conditioned roles."""
    sources, authorities = [], []
    ledger = root / "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
    report_path, manifest_path = ledger / "report.json", ledger / "manifest.json"
    report, manifest = (json.loads(path.read_text()) for path in (report_path, manifest_path))
    expected_status = "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"
    if (report.get("status") != expected_status or manifest.get("status") != expected_status
            or report.get("complete") is not True or report.get("current_family_recomputed") is not True
            or len(report.get("source_inventory", [])) != 34):
        raise ValueError("historical protected ledger is incomplete")
    _validate_bound(report_path, manifest["run"]["report_sha256"])
    ledger_path = ledger / "protected_identities.question_only.jsonl"
    _validate_bound(ledger_path, report["output"]["sha256"])
    _validate_bound(ledger_path, manifest["run"]["protected_identities_sha256"])
    for entry in report["source_inventory"]:
        _validate_bound(root / entry["path"], entry["sha256"])
    sources.append(SourceSpec(ledger_path, "historical_evaluation_development_confirmation_and_reserves", report["output"]["sha256"], expected_rows=4690))
    authorities.extend(bind(path) for path in (report_path, manifest_path))

    release_specs = (
        ("data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42", "report.json", "prompt_groups.jsonl", "prompt_groups", "ppo_training_population", "COMPLETE_DATA_NOT_TRAINED", 3000),
        ("outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1", "manifest.json", "inputs.jsonl", "inputs.jsonl", "alpha_main_development_830", "TRAIN_ONLY_INPUTS_FROZEN_NOT_GENERATED", 830),
        ("outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2", "manifest.json", "selection.question_only.jsonl", "selection.question_only.jsonl", "alpha_representative_normalization_120", "COMPLETE_NORMALIZATION_TRAIN_ONLY_NOT_GATE_OR_PPO_CLEARANCE", 120),
        ("outputs/audits/source_credit_v2_fresh_confirmation_inputs_20260906_v1", "manifest.scope_v2.json", "inputs.jsonl", "inputs.jsonl", "alpha_fresh_confirmation_132", "INPUTS_FROZEN_SOURCE_CHECK_AND_GENERATION_PROTOCOL_PENDING", 132),
    )
    for directory, authority, filename, output_key, role, status, n_rows in release_specs:
        directory = root / directory
        manifest_path = directory / authority
        release = json.loads(manifest_path.read_text())
        if ((status and release.get("status") != status)
                or (directory / "FAILED.json").exists()
                or (release.get("gates") and not all(release["gates"].values()))):
            raise ValueError(f"failed or unexpected release: {directory}")
        expected = release["outputs"][output_key]["sha256"]
        _validate_bound(directory / filename, expected)
        sources.append(SourceSpec(directory / filename, role, expected, expected_rows=n_rows))
        authorities.append(bind(manifest_path))

    strong_manifest = root / STRONG_MANIFEST
    release = json.loads(strong_manifest.read_text())
    run = release.get("run", {})
    expected = {"split": "train", "val_ratio": .1, "test_ratio": .1, "split_seed": 42}
    if release.get("status") != "COMPLETE" or any(run.get("config", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Strong-SFT historical split protocol differs")
    source = root / STRONG_SOURCE
    source_bind = run["input_artifacts"]["silver"]
    if source.stat().st_size != source_bind["size_bytes"]:
        raise ValueError("Strong-SFT source size mismatch")
    h = hashlib.md5()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != source_bind["md5"]:
        raise ValueError("Strong-SFT historical source MD5 mismatch")
    sources.append(SourceSpec(source, "strong_sft_historical_nontrain_fold", sha_file(source), historical_holdout_only=True, expected_rows=24998))
    authorities.append(bind(strong_manifest))
    return sources, authorities


def merge_sources(specs: Iterable[SourceSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    inventory = []
    for spec in specs:
        _validate_bound(spec.path, spec.expected_sha256)
        counts: Counter[str] = Counter()
        for line_number, raw, row_sha in read_rows(spec.path):
            counts["source_rows"] += 1
            item = identity(raw, spec.dataset_override)
            role = spec.role
            if spec.historical_holdout_only:
                if type(raw.get("accepted")) is not bool:
                    raise ValueError("historical split requires explicit Boolean accepted")
                fold = assign_split(SimpleNamespace(qid=item["qid"], question=item["question"], accepted=raw["accepted"]), SplitSpec())
                counts[f"historical_{fold}_rows"] += 1
                if fold == "train":
                    continue
                role += ":" + fold
            counts["protected_source_rows"] += 1
            if raw.get("family_sha256") and raw["family_sha256"] != item["family_sha256"]:
                counts["stored_family_mismatch_rows_recomputed"] += 1
            key = (item["dataset"], item["qid"])
            origin = {"role": role, "path": str(spec.path.resolve()), "file_sha256": spec.expected_sha256,
                      "line_number": line_number, "line_bytes_sha256": row_sha}
            if key not in merged:
                merged[key] = {**item, "source_roles": set(), "source_paths": set(), "sources": []}
            prior = merged[key]
            if prior["question_sha256"] != item["question_sha256"]:
                raise ValueError(f"one dataset/qid maps to conflicting questions: {key}")
            prior["source_roles"].add(role)
            prior["source_paths"].add(str(spec.path.resolve()))
            prior["sources"].append(origin)
        # Detect concurrent mutation, including sources that might be appended.
        _validate_bound(spec.path, spec.expected_sha256)
        if spec.expected_rows is not None and counts["source_rows"] != spec.expected_rows:
            raise ValueError(f"source row count mismatch: {spec.path}")
        inventory.append({**asdict(spec), "path": str(spec.path.resolve()), "counts": dict(counts)})
    output = []
    for key in sorted(merged):
        row = merged[key]
        output.append({"schema_version": VERSION, **row,
                       "source_roles": sorted(row["source_roles"]), "source_paths": sorted(row["source_paths"]),
                       "gold_fields_emitted": False, "gold_values_used_for_identity_selection": False})
    return output, inventory


def make_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[Any]]:
    """Qids are dataset-scoped; exact questions and families are GLOBAL."""
    result: dict[str, set[Any]] = {field: set() for field in ("qid", "question_sha256", "family_sha256")}
    for row in rows:
        item = identity(row)
        for field in result:
            result[field].add((item["dataset"], item[field]) if field == "qid" else item[field])
    return result


def overlap_reasons(row: Mapping[str, Any], index: Mapping[str, set[Any]]) -> list[str]:
    item = identity(row)
    return [field for field in ("qid", "question_sha256", "family_sha256")
            if ((item["dataset"], item[field]) if field == "qid" else item[field]) in index[field]]


def capacity_audit(path: Path, dataset: str, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = bind(path)
    index = make_index(ledger)
    counts: Counter[str] = Counter()
    qid_hash = {}
    hash_qids: dict[str, set[str]] = defaultdict(set)
    safe_families, safe_qids = set(), set()
    for _, raw, _ in read_rows(path):
        item = identity(raw, dataset)
        counts["raw_rows"] += 1
        qid = item["qid"]
        if qid in qid_hash:
            raise ValueError("duplicate raw dataset/qid; no implicit overwrite/deduplication")
        qid_hash[qid] = item["question_sha256"]
        hash_qids[item["question_sha256"]].add(qid)
        reasons = overlap_reasons(item, index)
        counts.update("blocked_" + field for field in reasons)
        if reasons:
            counts["blocked_any"] += 1
        else:
            counts["safe_rows_before_internal_dedup"] += 1
            safe_families.add(item["family_sha256"])
            safe_qids.add(qid)
    _validate_bound(path, frozen["sha256"])
    return {"source": frozen, "dataset": dataset, "counts": dict(counts),
            "safe_unique_current_families": len(safe_families), "safe_unique_qids": len(safe_qids),
            "raw_exact_question_multi_qid_groups": sum(len(v) > 1 for v in hash_qids.values()),
            "capacity_is_identity_only_not_evidence_or_teacher_quality": True,
            "sft_candidates_selected": False}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def freeze_ledger(*, output_dir: Path, specs: list[SourceSpec], experiment_id: str,
                  authorities: list[dict[str, Any]] | None = None,
                  raw_train_sources: Mapping[str, Path] | None = None,
                  expected_historical_fold_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    if not experiment_id.strip() or not specs:
        raise ValueError("explicit experiment id and nonempty source set required")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": VERSION, "experiment_id": experiment_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_specs": [{**asdict(s), "path": str(s.path.resolve())} for s in specs],
                "authority_bindings": authorities or [], "code": bind(Path(__file__)),
                "family_code": bind(ROOT / "scripts/prepare/freeze_qpeg_v1_protocol.py"),
                "identity_code": bind(ROOT / "kgproweight/kg/question_kg.py"),
                "historical_split_code": bind(ROOT / "kgproweight/data/silver_split.py"),
                "historical_split": asdict(SplitSpec()), "family_version": FAMILY_VERSION,
                "identity_scope": "dataset-scoped qid; GLOBAL exact stripped-question SHA256 and GLOBAL current lexical family SHA256",
                "ambiguous_qid_policy": "fail closed on conflicting qid/id or dataset/qid -> question; keep every exact-question alias qid",
                "expected_historical_fold_counts": dict(expected_historical_fold_counts or {}),
                "raw_train_sources": {ds: bind(path) for ds, path in (raw_train_sources or {}).items()},
                "training_started": False, "teacher_api_called": False, "gpu_started": False}
    write_json(output_dir / "protocol.json", protocol)
    try:
        rows, inventory = merge_sources(specs)
        if expected_historical_fold_counts:
            observed = Counter()
            for source in inventory:
                if source["historical_holdout_only"]:
                    for fold in ("train", "val", "test"):
                        observed[fold] += source["counts"].get(f"historical_{fold}_rows", 0)
            if dict(observed) != dict(expected_historical_fold_counts):
                raise ValueError(f"Strong-SFT historical fold replay differs: {dict(observed)}")
        aliases: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in rows:
            aliases[(row["dataset"], row["question_sha256"])].append(row["qid"])
        alias_rows = [{"dataset": ds, "question_sha256": qhash, "qids": sorted(qids)}
                      for (ds, qhash), qids in sorted(aliases.items()) if len(qids) > 1]
        capacity = {ds: capacity_audit(path, ds, rows) for ds, path in (raw_train_sources or {}).items()}
        for ds, data in capacity.items():
            if data["source"] != protocol["raw_train_sources"][ds]:
                raise ValueError("raw-train source changed after protocol freeze")
        for binding in [*protocol["authority_bindings"], protocol["code"], protocol["family_code"],
                        protocol["identity_code"], protocol["historical_split_code"]]:
            _validate_bound(Path(binding["path"]), binding["sha256"])
        for name, contents in (("protected_identities.question_only.jsonl", rows), ("exact_question_aliases.question_only.jsonl", alias_rows)):
            with (output_dir / name).open("x", encoding="utf-8") as handle:
                for row in contents:
                    handle.write(canonical_json(row) + "\n")
        report = {"schema_version": VERSION, "experiment_id": experiment_id,
                  "status": "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA", "complete": True,
                  "source_inventory": inventory, "protected_dataset_qids": len(rows),
                  "protected_dataset_question_hashes": len(aliases),
                  "protected_dataset_current_families": len({(r["dataset"], r["family_sha256"]) for r in rows}),
                  "protected_global_question_hashes": len({r["question_sha256"] for r in rows}),
                  "protected_global_current_families": len({r["family_sha256"] for r in rows}),
                  "protected_exact_question_multi_qid_groups": len(alias_rows),
                  "by_dataset": {ds: {"qids": sum(r["dataset"] == ds for r in rows),
                                      "families": len({r["family_sha256"] for r in rows if r["dataset"] == ds})} for ds in DATASETS},
                  "raw_train_capacity": capacity,
                  "boundaries": {"gold_bearing_json_objects_decoded": True, "gold_values_used": False,
                                 "historical_accepted_flag_used_only_to_replay_frozen_split": True,
                                 "family_is_lexical_template_not_semantic_dependency": True,
                                 "global_family_exclusion_can_remove_many_2wiki_templates_and_change_training_population": True,
                                 "capacity_does_not_establish_evidence_sufficiency": True,
                                 "original_data_modified": False, "training_started": False},
                  "protocol": bind(output_dir / "protocol.json")}
        write_json(output_dir / "report.json", report)
        manifest = {**report, "outputs": {name: bind(output_dir / name) for name in (
            "protocol.json", "report.json", "protected_identities.question_only.jsonl", "exact_question_aliases.question_only.jsonl")}}
        write_json(output_dir / "manifest.json", manifest)
        return report
    except BaseException as exc:
        write_json(output_dir / "FAILED.json", {"status": "FAILED_NOT_TRAINED", "type": type(exc).__name__,
                                                "error": str(exc), "partial_outputs_retained": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiment-id", default="SFT-V3-PROTECTED-IDENTITY-LEDGER-20260906-V1")
    args = parser.parse_args()
    specs, authorities = default_sources()
    report = freeze_ledger(output_dir=args.output_dir, specs=specs, authorities=authorities,
                           experiment_id=args.experiment_id,
                           raw_train_sources={ds: ROOT / "data" / ds / "train.jsonl" for ds in DATASETS},
                           expected_historical_fold_counts={"train": 20049, "val": 2524, "test": 2425})
    print(json.dumps({"status": report["status"], "protected_dataset_qids": report["protected_dataset_qids"],
                      "by_dataset": report["by_dataset"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
