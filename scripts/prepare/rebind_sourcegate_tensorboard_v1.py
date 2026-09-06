"""Freeze the monitoring-only successor of the source-gate deployment v2.

This never generates candidates or runs training. It copies the frozen input
bytes, refreshes runtime source identities, and creates a new release manifest.
Model/data/checkpoint hashes are inherited from the parent after size checks;
the remote deployment verifier must subsequently verify their full SHA256s.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
PARENT_BANK = "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_gitless_fix1"
PARENT_RELEASE = "outputs/audits/source_gated_mixed4_emf1_v1_release_v2/manifest.json"
HELPERS = (
    "kgproweight/training/ppo_tensorboard.py",
    "kgproweight/training/ppo_tensorboard_runtime.py",
)
SCRIPT = "scripts/prepare/rebind_sourcegate_tensorboard_v1.py"
IMMUTABLE_PREFIXES = ("models/", "data/", "checkpoints/")
# These are the only pre-existing executable files allowed to change in this
# monitoring-only revision. New telemetry helpers are separately bound below.
ALLOWED_RUNTIME_CHANGES = {
    "kgproweight/training/phase3_ppo.py",
    "scripts/sourcegate_python.sh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def logical(path: Path) -> str:
    # Model leaves may be symlinks, so preserve their logical path here.
    absolute = path if path.is_absolute() else ROOT / path
    relative = absolute.relative_to(ROOT)
    if ".." in relative.parts:
        raise ValueError(f"path escapes project root: {path}")
    return relative.as_posix()


def project_path(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    logical(absolute)
    return absolute


def byte_identical(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            chunk = lhs.read(1024 * 1024)
            if chunk != rhs.read(1024 * 1024):
                return False
            if not chunk:
                return True


def allowed_existing_change(name: str) -> bool:
    return name in ALLOWED_RUNTIME_CHANGES or name == "RESEARCH_WORKFLOW.md" or name.startswith(("docs/", "tests/"))


def freeze(*, bank_dir: Path, release_dir: Path, remote_root: str,
           parent_bank: Path, parent_release: Path, experiment_id: str,
           include: list[Path]) -> dict:
    from scripts.prepare.source_quality_candidate_bank_v1 import (
        PREPARE_VERSION, binding, finish, load_release, resolve, stage,
        validate_code, validate_inputs,
    )

    bank_dir, release_dir, parent_bank, parent_release = (
        project_path(path) for path in (bank_dir, release_dir, parent_bank, parent_release)
    )
    for target in (bank_dir, release_dir):
        if target.exists() or target.is_symlink():
            raise ValueError(f"refusing to overwrite: {target}")
        if not target.parent.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"output parent escapes project root: {target}")
    if bank_dir == release_dir or bank_dir in release_dir.parents or release_dir in bank_dir.parents:
        raise ValueError("candidate and release output directories must be separate siblings")
    if not remote_root.startswith("/"):
        raise ValueError("remote-root must be an absolute deployment path")

    parent = json.loads(parent_release.read_text(encoding="utf-8"))
    if parent.get("training_started") is not False or not parent.get("full_model_shards_hashed"):
        raise ValueError("parent must be the untrained release with full model identities")
    old_bank_manifest_path = parent_bank / "manifest.json"
    parent_bank_sha = sha256(old_bank_manifest_path)
    parent_release_sha = sha256(parent_release)
    parent_lock = parent["files"].get(logical(old_bank_manifest_path))
    if parent_lock != identity(old_bank_manifest_path):
        raise ValueError("parent candidate bank is not exactly bound by the parent release")
    if parent.get("authoritative_candidate_input_bank") != logical(parent_bank):
        raise ValueError("candidate bank is not the parent release's authoritative input bank")
    old_bank = load_release(parent_bank, PREPARE_VERSION)
    old_rows = validate_inputs(parent_bank, old_bank)
    if old_bank.get("training_started") is not False or old_bank.get("status") != "TRAIN_ONLY_INPUTS_FROZEN_NOT_GENERATED":
        raise ValueError("only an ungenerated input bank can receive this telemetry revision")

    additions = {SCRIPT, *HELPERS, logical(parent_release)}
    for subtree in ("kgproweight/training", "scripts", "tests", "docs"):
        for path in (ROOT / subtree).rglob("*tensorboard*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".md"} and "__pycache__" not in path.parts:
                additions.add(logical(path))
    additions.update(logical(project_path(path)) for path in include)
    for name in additions:
        if not (ROOT / name).is_file():
            raise ValueError(f"required telemetry/release file missing: {name}")

    # Refresh code bindings but reject unrelated scientific changes. All
    # original non-code sources must still resolve to their frozen identities.
    new_bindings = copy.deepcopy(old_bank["source_bindings"])
    code_changes = {}
    for name, old in new_bindings.items():
        if name.startswith("code:"):
            relative = name.removeprefix("code:")
            current = binding(ROOT / relative, ROOT)
            if current["sha256"] != old["sha256"]:
                if relative not in ALLOWED_RUNTIME_CHANGES:
                    raise ValueError(f"non-telemetry candidate source changed: {relative}")
                code_changes[relative] = {"parent_sha256": old["sha256"], "sha256": current["sha256"]}
            new_bindings[name] = current
        elif name != "score_config":
            resolve(old, parent_bank, ROOT)
    for relative in HELPERS:
        new_bindings["code:" + relative] = binding(ROOT / relative, ROOT)

    # Inspect the existing inventory before creating any scientific artifact.
    # Large, unchanged assets retain their already audited hashes; all other
    # files are hashed now, including every inherited source and document.
    files = {}
    changed_files = {}
    inherited_assets = []
    for name, old in parent["files"].items():
        path = ROOT / name
        if logical(path) != name or not path.is_file():
            raise ValueError(f"missing or noncanonical parent file: {name}")
        if name.startswith(IMMUTABLE_PREFIXES):
            if path.stat().st_size != old["size_bytes"]:
                raise ValueError(f"frozen asset size changed: {name}")
            files[name] = copy.deepcopy(old)
            inherited_assets.append(name)
        else:
            current = identity(path)
            if current != old:
                if not allowed_existing_change(name):
                    raise ValueError(f"unrelated frozen file changed: {name}")
                changed_files[name] = {"parent": old, "current": current}
            files[name] = current

    revision = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "TensorBoard monitoring-only runtime revision; frozen candidate inputs and all research contracts preserved",
        "parent_release_manifest": logical(parent_release),
        "parent_deployment_manifest_sha256": parent_release_sha,
        "old_input_bank": logical(parent_bank),
        "old_bank_manifest_sha256": parent_bank_sha,
        "authoritative_input_bank": logical(bank_dir),
        "candidate_code_changes": code_changes,
        "added_candidate_code_bindings": list(HELPERS),
        "input_rows": len(old_rows),
        "inputs_byte_identical": True,
        "score_config_byte_identical": True,
        "generation_scoring_reward_data_model_evaluation_contracts_unchanged": True,
        "trained_policy_updates": 0,
    }

    with stage(bank_dir, experiment_id + "-INPUTS", "telemetry_binding_revision"):
        for filename in ("inputs.jsonl", "score_config.json"):
            shutil.copyfile(parent_bank / filename, bank_dir / filename)
            if not byte_identical(parent_bank / filename, bank_dir / filename):
                raise ValueError(f"copy was not byte identical: {filename}")
        new_bindings["score_config"] = binding(bank_dir / "score_config.json", ROOT)
        report = {key: copy.deepcopy(value) for key, value in old_bank.items() if key != "outputs"}
        report.update(experiment_id=experiment_id + "-INPUTS", source_bindings=new_bindings,
                      binding_revision=copy.deepcopy(revision))
        rows = validate_inputs(bank_dir, report)
        validate_code(report, ROOT)
        report["binding_validation"] = {"inputs_byte_identical": True, "score_config_byte_identical": True,
                                        "source_bindings_match": True, "questions": len(rows)}
        finish(bank_dir, report, ["inputs.jsonl", "score_config.json"])
        validate_code(load_release(bank_dir, PREPARE_VERSION), ROOT)

    with stage(release_dir, experiment_id + "-RELEASE", "telemetry_release_revision"):
        revision["new_bank_manifest_sha256"] = sha256(bank_dir / "manifest.json")
        revision["changed_existing_files"] = changed_files
        revision["hash_verification_contract"] = {
            "non_asset_files": "SHA256 recomputed locally",
            "model_data_checkpoint_files": "parent full SHA256 inherited after local existence and size checks",
            "inherited_asset_count": len(inherited_assets),
            "remote_full_sha256_verification": "REQUIRED_NOT_YET_RUN_BY_THIS_SCRIPT",
        }
        additions.update(logical(path) for path in bank_dir.iterdir() if path.is_file())
        revision["added_release_files"] = sorted(additions - parent["files"].keys())
        write_json(release_dir / "revision.json", revision)
        additions.update((logical(release_dir / "revision.json"), logical(release_dir / "started.json")))
        for name in sorted(additions):
            files[name] = identity(ROOT / name)
        manifest = {key: copy.deepcopy(value) for key, value in parent.items()
                    if key not in {"files", "revision", "local_verification"}}
        manifest.update(
            experiment_id=experiment_id + "-RELEASE",
            status="TELEMETRY_CODE_AND_INPUTS_FROZEN_REMOTE_VERIFICATION_PENDING",
            files=dict(sorted(files.items())), remote_root=remote_root,
            revision=revision, authoritative_candidate_input_bank=logical(bank_dir),
            parent_local_verification=copy.deepcopy(parent.get("local_verification", {})),
            local_verification={"candidate_inputs": report["binding_validation"],
                                "tests": "separate test report required; not executed by this freezing script"},
            training_started=False,
        )
        write_json(release_dir / "manifest.json", manifest)
    return {"status": manifest["status"], "candidate_bank": logical(bank_dir),
            "release_manifest": logical(release_dir / "manifest.json"),
            "release_manifest_sha256": sha256(release_dir / "manifest.json"),
            "files": len(files), "changed_existing_files": sorted(changed_files),
            "candidate_questions": len(rows), "training_started": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--parent-bank", type=Path, default=Path(PARENT_BANK))
    parser.add_argument("--parent-release", type=Path, default=Path(PARENT_RELEASE))
    parser.add_argument("--experiment-id", default="SOURCE-GATED-MIXED4-EMF1-V1-TENSORBOARD-20260905")
    parser.add_argument("--include", type=Path, action="append", default=[],
                        help="Additional exact project-relative file to bind; repeat as needed")
    result = freeze(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
