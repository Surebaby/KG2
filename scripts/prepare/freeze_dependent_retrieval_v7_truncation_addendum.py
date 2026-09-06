#!/usr/bin/env python
"""Bind the append-only v7 producer-passage truncation addendum.

The original design and cohort preregistration remain immutable.  This CPU-only
freezer verifies both parents and writes a small effective-protocol addendum.
It does not run a planner, model, retriever, verifier, finalizer, or scorer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.prepare.freeze_dependent_retrieval_v7 import (
    DEFAULT_OUT as PARENT_DIR,
    STATUS as PARENT_STATUS,
    file_lock,
    read_json,
)


EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-"
    "PRODUCER-TRUNCATION-ADDENDUM"
)
STATUS = "FROZEN_APPEND_ONLY_ADDENDUM_BEFORE_V7_GPU_OR_RETRIEVAL"
EXPECTED_PARENT_PROTOCOL_SHA256 = "c7f1674f62a191671a22844e5589c3f9b80a990aae3c0344cd4001e47a50395d"
EXPECTED_PARENT_MANIFEST_SHA256 = "01ab042c1f824dc649cddc5aa929393efe17aab9631ec07de70fcb5f8dda19cb"
EXPECTED_DESIGN_ADDENDUM_SHA256 = "3fafb83c394e16485e4bdc628f4013f064406ca88c355daeedcf720d55033cad"
EXPECTED_DESIGN_ADDENDUM_MANIFEST_SHA256 = (
    "a1f1793cebf662405c9bbce9308e396b2a2287a602c2c7d87408c27ecc7f1030"
)

DEFAULT_ADDENDUM = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/"
    "addendum_producer_passage_truncation_v1.json"
)
DEFAULT_ADDENDUM_MANIFEST = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/"
    "addendum_producer_passage_truncation_v1.manifest.json"
)
DEFAULT_OUT = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_producer_truncation_v1"
)


def _expect(lock: Mapping[str, Any], digest: str, label: str) -> None:
    if str(lock.get("sha256") or "") != digest:
        raise ValueError(f"{label} SHA256 drift")


def validate_addendum(value: Mapping[str, Any]) -> None:
    if value.get("status") != STATUS:
        raise ValueError("design addendum status drift")
    override = value.get("effective_override") or {}
    expected = {
        "maximum_passages": 10,
        "maximum_unicode_characters_per_passage": 1200,
        "same_projection_for_prompt_and_verifier": True,
    }
    for key, expected_value in expected.items():
        if override.get(key) != expected_value:
            raise ValueError(f"design addendum rule drift: {key}")
    if override.get("projected_fields") != ["doc_id", "title", "text"]:
        raise ValueError("design addendum projected field drift")
    if "text[:1200]" not in str(override.get("truncation") or ""):
        raise ValueError("design addendum Unicode truncation drift")
    if "ensure_ascii=false" not in str(override.get("projection_hash") or ""):
        raise ValueError("design addendum projection hash drift")
    if value.get("gold_access") is not False:
        raise ValueError("design addendum Gold boundary drift")


def build_protocol(
    *,
    parent_protocol_path: Path,
    parent_manifest_path: Path,
    addendum_path: Path,
    addendum_manifest_path: Path,
    expected_parent_protocol_sha256: str = EXPECTED_PARENT_PROTOCOL_SHA256,
    expected_parent_manifest_sha256: str = EXPECTED_PARENT_MANIFEST_SHA256,
    expected_addendum_sha256: str = EXPECTED_DESIGN_ADDENDUM_SHA256,
    expected_addendum_manifest_sha256: str = EXPECTED_DESIGN_ADDENDUM_MANIFEST_SHA256,
) -> dict[str, Any]:
    locks = {
        "parent_preregistration": file_lock(parent_protocol_path),
        "parent_preregistration_manifest": file_lock(parent_manifest_path),
        "design_addendum": file_lock(addendum_path),
        "design_addendum_manifest": file_lock(addendum_manifest_path),
    }
    for name, expected in (
        ("parent_preregistration", expected_parent_protocol_sha256),
        ("parent_preregistration_manifest", expected_parent_manifest_sha256),
        ("design_addendum", expected_addendum_sha256),
        ("design_addendum_manifest", expected_addendum_manifest_sha256),
    ):
        _expect(locks[name], expected, name)

    parent = read_json(parent_protocol_path)
    parent_manifest = read_json(parent_manifest_path)
    addendum = read_json(addendum_path)
    addendum_manifest = read_json(addendum_manifest_path)
    if parent.get("status") != PARENT_STATUS:
        raise ValueError("parent v7 preregistration status drift")
    if parent.get("execution_authorization") != "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK":
        raise ValueError("parent v7 execution boundary drift")
    if parent.get("gold_access") is not False:
        raise ValueError("parent v7 Gold boundary drift")
    if parent.get("gpu_calls") != 0 or parent.get("retrieval_calls") != 0:
        raise ValueError("parent v7 unexpectedly records execution")
    if parent_manifest.get("status") != PARENT_STATUS:
        raise ValueError("parent v7 manifest status drift")
    validate_addendum(addendum)
    if addendum_manifest.get("status") != STATUS:
        raise ValueError("design addendum manifest status drift")
    if str((addendum_manifest.get("addendum") or {}).get("sha256") or "") != str(
        locks["design_addendum"]["sha256"]
    ):
        raise ValueError("design addendum manifest content lock drift")

    # Revalidate every immutable cohort artifact named by the parent manifest.
    parent_artifacts = parent_manifest.get("artifacts") or {}
    verified_parent_artifacts: dict[str, Any] = {}
    for name in (
        "development",
        "planner",
        "reclassification_ledger",
        "unselected_commitment",
        "protocol",
    ):
        inherited = parent_artifacts.get(name) or {}
        path = Path(str(inherited.get("path") or ""))
        lock = file_lock(path)
        if lock["sha256"] != str(inherited.get("sha256") or ""):
            raise ValueError(f"parent artifact drift: {name}")
        verified_parent_artifacts[name] = lock

    override = dict(addendum["effective_override"])
    return {
        "schema_version": "subquestion-dependent-retrieval-v7-effective-addendum-1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": "C_PRODUCER_PROMPT_VERIFIER_PASSAGE_VIEW_IDENTITY_ONLY",
        "parents": locks,
        "verified_parent_artifacts": verified_parent_artifacts,
        "effective_override": override,
        "effective_invariants": {
            "producer_passages_max": 10,
            "producer_text_unicode_chars_max_each": 1200,
            "python_slice": "text[:1200]",
            "projection_fields": ["doc_id", "title", "text"],
            "reader_and_verifier_projection_hash_equal": True,
            "answer_in_unseen_suffix_never_verified": True,
        },
        "telemetry_additions": list(addendum["telemetry_additions"]),
        "unchanged": list(addendum["unchanged"]),
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
        "required_runtime_lock_addition": (
            "The implementation lock must hash the one shared producer-passage "
            "projection helper and prove both reader and verifier consume its exact output."
        ),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "parent_protocols_mutated": False,
        "scientific_boundary": addendum["scientific_boundary"],
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_protocol(protocol: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v7 addendum freeze: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    _write_json_exclusive(protocol_path, protocol)
    manifest = {
        "schema_version": "subquestion-dependent-retrieval-v7-addendum-manifest-1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "protocol": file_lock(protocol_path),
        "gold_access": False,
        "gpu_calls": 0,
        "retrieval_calls": 0,
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {"protocol": file_lock(protocol_path), "manifest": file_lock(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_protocol", type=Path, default=PARENT_DIR / "protocol.json")
    parser.add_argument("--parent_manifest", type=Path, default=PARENT_DIR / "manifest.json")
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--addendum_manifest", type=Path, default=DEFAULT_ADDENDUM_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = build_protocol(
        parent_protocol_path=args.parent_protocol,
        parent_manifest_path=args.parent_manifest,
        addendum_path=args.addendum,
        addendum_manifest_path=args.addendum_manifest,
    )
    result = write_protocol(protocol, args.out)
    print(json.dumps({"status": STATUS, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
